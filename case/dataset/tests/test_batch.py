import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch import make_batches  # noqa: E402


def record(i: int, prompt: str = "p") -> dict:
    return {
        "id": f"row{i:03d}",
        "input": {"prompt": prompt, "model": "m", "input_tokens": i},
        "reference": {"output_tokens": [i, i + 1]},
        "meta": {"source": "Alpaca", "family": f"f{i}"},
    }


def rows_of(batches: list[dict]) -> list[str]:
    return [r["id"] for b in batches for r in json.loads(b["input"])["rows"]]


def test_every_row_lands_in_exactly_one_batch():
    records = [record(i) for i in range(23)]
    batches = make_batches(records, "public", max_rows=5, max_chars=10_000, seed=1)
    ids = rows_of(batches)
    assert sorted(ids) == sorted(r["id"] for r in records)
    assert len(ids) == len(set(ids))


def test_batch_ids_follow_split_and_index():
    batches = make_batches(
        [record(i) for i in range(12)], "train", max_rows=5, max_chars=10_000, seed=1
    )
    assert [b["id"] for b in batches] == ["train-0000", "train-0001", "train-0002"]


def test_respects_row_limit():
    batches = make_batches(
        [record(i) for i in range(12)], "public", max_rows=5, max_chars=10_000, seed=1
    )
    assert [len(json.loads(b["input"])["rows"]) for b in batches] == [5, 5, 2]


def test_respects_char_limit():
    records = [record(i, prompt="x" * 100) for i in range(10)]
    batches = make_batches(records, "public", max_rows=100, max_chars=500, seed=1)
    assert len(batches) > 1
    assert all(len(b["input"]) <= 500 for b in batches)


def test_row_larger_than_limit_is_an_error():
    with pytest.raises(ValueError):
        make_batches([record(0, prompt="x" * 1000)], "public", max_rows=10, max_chars=500, seed=1)


def test_duplicate_row_id_is_an_error():
    with pytest.raises(ValueError):
        make_batches(
            [record(1), record(2), record(1)],
            "public", max_rows=10, max_chars=10_000, seed=1,
        )


def test_input_hides_reference_and_meta():
    batches = make_batches(
        [record(i) for i in range(3)], "public", max_rows=5, max_chars=10_000, seed=1
    )
    row = json.loads(batches[0]["input"])["rows"][0]
    assert set(row) == {"id", "prompt", "model", "input_tokens"}


def test_reference_matches_rows_by_id():
    records = [record(i) for i in range(7)]
    batches = make_batches(records, "public", max_rows=3, max_chars=10_000, seed=1)
    for b in batches:
        ids = [r["id"] for r in json.loads(b["input"])["rows"]]
        reference = json.loads(b["reference"])["output_tokens"]
        assert list(reference) == ids
        for row_id in ids:
            i = int(row_id[3:])
            assert reference[row_id] == [i, i + 1]


def test_order_is_deterministic_and_depends_on_seed():
    records = [record(i) for i in range(50)]
    first = make_batches(records, "public", max_rows=10, max_chars=10_000, seed=1)
    again = make_batches(list(reversed(records)), "public", max_rows=10, max_chars=10_000, seed=1)
    other = make_batches(records, "public", max_rows=10, max_chars=10_000, seed=2)
    assert first == again
    assert rows_of(first) != rows_of(other)
    assert rows_of(first) != sorted(rows_of(first))
