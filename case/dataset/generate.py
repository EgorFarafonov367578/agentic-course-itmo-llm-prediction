"""Сборка train / public / private из открытого датасета CASTILLO.

Источник: https://huggingface.co/datasets/danfperam/castillo (CC BY 4.0), конфигурация `sanitized`,
ревизия закреплена в REVISION. Генерировать ответы не нужно: на каждую пару «запрос × модель»
в датасете уже есть 10 длин ответов.

Шаги:
1. скачать три файла `sanitized/*.json` (родной split CASTILLO не используется);
2. склеить повторы одного sample_id (одна пара «запрос × модель» сгенерирована в нескольких файлах);
3. собрать семейства похожих запросов: кандидаты — пары с общим редким словесным шинглом
   (инвертированный индекс), точная проверка Jaccard, связные компоненты через union-find;
4. разложить семейства по train / public / private: внутри каждого источника запросов семейства
   перемешиваются по хэшу (сид + id семейства) и жадно отдаются split'у, сильнее всего отстающему
   от целевой доли строк;
5. записать JSONL и manifest.json.

python3 case/dataset/generate.py
    # скачивает в case/dataset/raw/, пишет в case/dataset/data/
python3 case/dataset/generate.py --raw-dir /path/to/castillo --out-dir /tmp/out
"""
import argparse
import collections
import hashlib
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "danfperam/castillo"
REVISION = "fb93f412e6bed3f3b834f0ded23e0865c286ed6c"
RAW_FILES = [f"sanitized/sanitized_{s}.json" for s in ("train", "validate", "test")]

SPLIT_SEED = 20261104
FRACTIONS = {"train": 0.6, "public": 0.2, "private": 0.2}
MIN_OUTPUTS = 5  # в sanitized у ~1 % строк часть из 10 ответов выброшена как вырожденная

SHINGLE = 5
JACCARD = 0.3
MAX_DF = 1000  # шингл чаще этого — общий шаблон источника, по нему кандидатов не ищем


def download(raw_dir: Path) -> None:
    for name in RAW_FILES:
        dst = raw_dir / name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
        print(f"download {url}")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(raw_dir: Path) -> list[dict]:
    rows = []
    for name in RAW_FILES:
        rows += json.load(open(raw_dir / name, encoding="utf-8"))
    return rows


REPEAT_KEYS = (
    "prompt_id", "model", "dataset", "prompt_text", "input_size",
    "category", "temp", "top_p", "top_k",
)


def merge_repeats(rows: list[dict]) -> tuple[list[dict], int]:
    """Склеивает строки с одним sample_id: в CASTILLO ~200 пар «запрос × модель» сгенерированы
    повторно в разных файлах. Запрос и параметры совпадают, поэтому длины ответов объединяются
    в одну выборку. Возвращает строки в порядке первого появления и число склеенных id."""
    merged: dict[str, dict] = {}
    repeated = set()
    for r in rows:
        first = merged.get(r["sample_id"])
        if first is None:
            merged[r["sample_id"]] = dict(r)
            continue
        if any(first[k] != r[k] for k in REPEAT_KEYS):
            raise ValueError(
                f"sample_id {r['sample_id']} repeats with a different prompt or settings"
            )
        first["output_sizes"] = first["output_sizes"] + r["output_sizes"]
        repeated.add(r["sample_id"])
    return list(merged.values()), len(repeated)


def shingles(text: str) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + SHINGLE]) for i in range(max(1, len(words) - SHINGLE + 1))}


def build_families(prompts: dict[str, str]) -> dict[str, str]:
    """prompt_id -> id семейства (наименьший prompt_id в компоненте)."""
    ids = sorted(prompts)
    parent = {i: i for i in ids}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    sh = {i: shingles(prompts[i]) for i in ids}
    index = collections.defaultdict(list)
    for i in ids:
        for g in sh[i]:
            index[g].append(i)
    for i in ids:
        cand = {o for g in sh[i] if len(index[g]) <= MAX_DF for o in index[g] if o > i}
        for o in sorted(cand):
            if len(sh[i] & sh[o]) / len(sh[i] | sh[o]) >= JACCARD:
                union(i, o)
    return {i: find(i) for i in ids}


def assign_splits(
    family: dict[str, str], source: dict[str, str], rows_per_prompt: dict[str, int]
) -> dict[str, str]:
    """id семейства -> split. Источник семейства — источник его первого запроса."""
    fam_rows = collections.Counter()
    for pid, fam in family.items():
        fam_rows[fam] += rows_per_prompt[pid]
    by_source = collections.defaultdict(list)
    for fam in sorted(fam_rows):
        by_source[source[fam]].append(fam)
    out = {}
    for _src, fams in sorted(by_source.items()):
        fams.sort(key=lambda f: hashlib.sha256(f"{SPLIT_SEED}:{f}".encode()).hexdigest())
        filled = dict.fromkeys(FRACTIONS, 0)
        for fam in fams:
            total = sum(filled.values()) + fam_rows[fam]
            s = min(FRACTIONS, key=lambda s: (filled[s] + fam_rows[fam]) / total - FRACTIONS[s])
            out[fam] = s
            filled[s] += fam_rows[fam]
    return out


def to_record(row: dict, family: str) -> dict:
    return {
        "id": row["sample_id"],
        "input": {
            "prompt": row["prompt_text"],
            "model": row["model"].replace("--", "/"),
            "input_tokens": row["input_size"],
        },
        "reference": {"output_tokens": row["output_sizes"]},
        "meta": {
            "prompt_id": row["prompt_id"],
            "family": family,
            "source": row["dataset"],
            "category": row["category"],
            "decoding": {"temperature": row["temp"], "top_p": row["top_p"], "top_k": row["top_k"]},
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=HERE / "raw")
    p.add_argument("--out-dir", type=Path, default=HERE / "data")
    a = p.parse_args()

    download(a.raw_dir)
    rows, repeated_ids = merge_repeats(load_rows(a.raw_dir))
    kept = [r for r in rows if len(r["output_sizes"]) >= MIN_OUTPUTS]
    prompts = {r["prompt_id"]: r["prompt_text"] for r in kept}
    source = {r["prompt_id"]: r["dataset"] for r in kept}
    family = build_families(prompts)
    split_of_family = assign_splits(
        family, source, collections.Counter(r["prompt_id"] for r in kept)
    )

    a.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: a.out_dir / f"castillo.{s}.jsonl" for s in FRACTIONS}
    files = {s: open(paths[s], "w", encoding="utf-8") for s in FRACTIONS}
    stats = {s: collections.Counter() for s in FRACTIONS}
    split_prompts = {s: set() for s in FRACTIONS}
    split_families = {s: set() for s in FRACTIONS}
    for r in sorted(kept, key=lambda r: r["sample_id"]):
        fam = family[r["prompt_id"]]
        s = split_of_family[fam]
        files[s].write(json.dumps(to_record(r, fam), ensure_ascii=False) + "\n")
        stats[s][r["dataset"]] += 1
        split_prompts[s].add(r["prompt_id"])
        split_families[s].add(fam)
    for f in files.values():
        f.close()

    manifest = {
        "version": "castillo-sanitized-family-split-v2",
        "source": {
            "dataset": f"https://huggingface.co/datasets/{REPO}",
            "revision": REVISION,
            "license": "CC BY 4.0",
            "files": {n: sha256(a.raw_dir / n) for n in RAW_FILES},
        },
        "params": {
            "split_seed": SPLIT_SEED, "fractions": FRACTIONS, "min_outputs": MIN_OUTPUTS,
            "shingle_words": SHINGLE, "jaccard": JACCARD, "max_df": MAX_DF,
        },
        "merged_repeated_sample_ids": repeated_ids,
        "dropped_rows_min_outputs": len(rows) - len(kept),
        "family_count": len(set(family.values())),
        "multi_prompt_families": sum(
            1 for c in collections.Counter(family.values()).values() if c > 1
        ),
        "largest_family": max(collections.Counter(family.values()).values()),
        "splits": {
            s: {
                "rows": sum(stats[s].values()),
                "prompts": len(split_prompts[s]),
                "families": len(split_families[s]),
                "rows_by_source": dict(sorted(stats[s].items())),
                "prompt_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(split_prompts[s])).encode()
                ).hexdigest(),
                "file_sha256": sha256(paths[s]),
            }
            for s in FRACTIONS
        },
        "closed_data_in_git": False,
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({s: manifest["splits"][s]["rows"] for s in FRACTIONS}))


if __name__ == "__main__":
    main()
