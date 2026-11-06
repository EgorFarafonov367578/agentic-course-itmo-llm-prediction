"""Нарезка split'ов на батчи — примеры в формате оценщика арены.

Арена запускает отдельный контейнер на каждый пример и идёт по примерам последовательно
(arena: common/runner-contract.md, evaluator/service.py). На 28 тысячах дешёвых предсказаний
это часы на одного кандидата, поэтому один пример арены — это батч строк датасета.
Обоснование — docs/adr/0001-batched-examples.md.

Формат строки на выходе (поля `input` и `reference` у арены — строки до 500 000 символов):
  {"id": "public-0000",
   "input": "{\"rows\": [{\"id\", \"prompt\", \"model\", \"input_tokens\"}, ...]}",
   "reference": "{\"output_tokens\": {\"<row id>\": [10 длин], ...}}"}

Строки перемешиваются хэшем sha256(seed:id), поэтому батч — случайная смесь источников и моделей.
`meta` в батчи не попадает.

python3 case/dataset/batch.py
    # data/castillo.{split}.jsonl -> data/castillo.{split}.batched.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "public", "private")

BATCH_SEED = 20261104
MAX_ROWS = 500
MAX_CHARS = 400_000  # запас до предела арены в 500 000 символов на input


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def input_row(record: dict) -> dict:
    return {"id": record["id"], **record["input"]}


def to_example(split: str, index: int, records: list[dict]) -> dict:
    return {
        "id": f"{split}-{index:04d}",
        "input": dumps({"rows": [input_row(r) for r in records]}),
        "reference": dumps({
            "output_tokens": {r["id"]: r["reference"]["output_tokens"] for r in records}
        }),
    }


def make_batches(
    records: list[dict], split: str, *, max_rows: int, max_chars: int, seed: int
) -> list[dict]:
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("row ids must be unique: reference is keyed by id")
    ordered = sorted(
        records,
        key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest(),
    )
    overhead = len(dumps({"rows": []}))
    batches, current, size = [], [], overhead
    for r in ordered:
        row_size = len(dumps(input_row(r))) + 1  # +1 на запятую между строками
        if overhead + row_size > max_chars:
            raise ValueError(f"row {r['id']} does not fit into {max_chars} chars")
        if current and (len(current) == max_rows or size + row_size > max_chars):
            batches.append(current)
            current, size = [], overhead
        current.append(r)
        size += row_size
    if current:
        batches.append(current)
    return [to_example(split, i, b) for i, b in enumerate(batches)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    a = p.parse_args()

    summary = {}
    for split in SPLITS:
        src = a.data_dir / f"castillo.{split}.jsonl"
        records = [json.loads(line) for line in open(src, encoding="utf-8")]
        batches = make_batches(
            records, split, max_rows=MAX_ROWS, max_chars=MAX_CHARS, seed=BATCH_SEED
        )
        dst = a.data_dir / f"castillo.{split}.batched.jsonl"
        with open(dst, "w", encoding="utf-8") as f:
            for b in batches:
                f.write(dumps(b) + "\n")
        sizes = [len(json.loads(b["input"])["rows"]) for b in batches]
        summary[split] = {
            "rows": len(records),
            "batches": len(batches),
            "rows_per_batch": {"min": min(sizes), "max": max(sizes)},
            "max_input_chars": max(len(b["input"]) for b in batches),
            "file_sha256": sha256(dst),
        }

    manifest = {
        "version": "castillo-batched-v2",
        "params": {"seed": BATCH_SEED, "max_rows": MAX_ROWS, "max_chars": MAX_CHARS},
        "splits": summary,
    }
    (HERE / "batches.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({s: summary[s]["batches"] for s in SPLITS}))


if __name__ == "__main__":
    main()
