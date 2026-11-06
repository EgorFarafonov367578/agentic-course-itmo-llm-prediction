"""Проверки фактических данных AR-09: хэши, split, батчи и baseline-квантили."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1]
SPLITS = ("train", "public", "private")
pytestmark = pytest.mark.skipif(
    not (DATA / "data" / "castillo.train.jsonl").exists(),
    reason="локальные данные не сгенерированы; запустите case/dataset/generate.py",
)


def rows(split: str) -> list[dict]:
    path = DATA / "data" / f"castillo.{split}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def batched(split: str) -> list[dict]:
    path = DATA / "data" / f"castillo.{split}.batched.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_hashes_and_row_counts_match_files():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for split in SPLITS:
        path = DATA / "data" / f"castillo.{split}.jsonl"
        assert sha256(path) == manifest["splits"][split]["file_sha256"]
        assert len(rows(split)) == manifest["splits"][split]["rows"]


def test_batches_cover_every_row_once_with_matching_reference():
    batches_manifest = json.loads((DATA / "batches.json").read_text(encoding="utf-8"))
    for split in SPLITS:
        records = rows(split)
        batches = batched(split)
        assert len(batches) == batches_manifest["splits"][split]["batches"]
        assert sha256(DATA / "data" / f"castillo.{split}.batched.jsonl") == (
            batches_manifest["splits"][split]["file_sha256"]
        )
        ids = [row["id"] for batch in batches for row in json.loads(batch["input"])["rows"]]
        assert len(ids) == len(records)
        assert len(set(ids)) == len(ids)
        assert set(ids) == {row["id"] for row in records}
        for batch in batches:
            input_ids = [row["id"] for row in json.loads(batch["input"])["rows"]]
            reference_ids = list(json.loads(batch["reference"])["output_tokens"])
            assert input_ids == reference_ids


def test_family_split_is_disjoint_and_outputs_are_valid():
    data = {split: rows(split) for split in SPLITS}
    families = {split: {row["meta"]["family"] for row in data[split]} for split in SPLITS}
    assert not (families["train"] & families["public"])
    assert not (families["train"] & families["private"])
    assert not (families["public"] & families["private"])
    for split in SPLITS:
        for row in data[split]:
            outputs = row["reference"]["output_tokens"]
            assert 5 <= len(outputs) <= 30
            assert all(0 <= value <= 100_000 for value in outputs)


def baseline_quantiles(records: list[dict]) -> dict[str, int]:
    weights: dict[int, Fraction] = {}
    for record in records:
        outputs = record["reference"]["output_tokens"]
        for value in outputs:
            weights[value] = weights.get(value, Fraction(0)) + Fraction(1, len(outputs))
    total = sum(weights.values(), Fraction(0))
    result = {}
    for name, q in (("lo", Fraction(1, 10)), ("pred", Fraction(1, 2)), ("hi", Fraction(9, 10))):
        cumulative = Fraction(0)
        for value in sorted(weights):
            cumulative += weights[value]
            if cumulative >= q * total:
                result[name] = value
                break
    return result


def test_baseline_params_are_fitted_on_train():
    expected = baseline_quantiles(rows("train"))
    params_path = DATA.parent / "ways" / "baseline" / "params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    assert {name: params[name] for name in ("lo", "pred", "hi")} == expected
