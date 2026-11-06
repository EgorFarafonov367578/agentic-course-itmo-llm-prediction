import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate import merge_repeats  # noqa: E402


def raw(sample_id: str, outputs: list[int], prompt: str = "p", gen_time: float = 1.0) -> dict:
    return {
        "sample_id": sample_id, "prompt_id": f"P{sample_id}", "model": "m",
        "dataset": "ShareGPT", "prompt_text": prompt, "input_size": 3,
        "category": "c", "temp": 0.7, "top_p": 0.8, "top_k": 20,
        "output_sizes": outputs, "gen_time": gen_time,
    }


def test_unique_rows_pass_through():
    rows = [raw("a", [1, 2]), raw("b", [3])]
    merged, repeated = merge_repeats(rows)
    assert merged == rows
    assert repeated == 0


def test_repeats_concatenate_outputs_in_file_order():
    rows = [
        raw("a", [1, 2], gen_time=1.0), raw("b", [5]),
        raw("a", [3], gen_time=2.0), raw("a", [4]),
    ]
    merged, repeated = merge_repeats(rows)
    assert [r["sample_id"] for r in merged] == ["a", "b"]
    assert merged[0]["output_sizes"] == [1, 2, 3, 4]
    assert repeated == 1


def test_merge_does_not_mutate_input():
    rows = [raw("a", [1]), raw("a", [2])]
    merge_repeats(rows)
    assert rows[0]["output_sizes"] == [1]


def test_repeat_with_different_prompt_is_an_error():
    with pytest.raises(ValueError):
        merge_repeats([raw("a", [1], prompt="x"), raw("a", [2], prompt="y")])
