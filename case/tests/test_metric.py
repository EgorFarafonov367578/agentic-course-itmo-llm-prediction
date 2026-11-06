"""Метрика ar09-quantile-v1 на синтетических батчах: точные значения, штрафы, агрегация."""
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

METRIC = Path(__file__).resolve().parents[1] / "metric"
PENALTY = 40.0
E1, E2 = math.expm1(1), math.expm1(2)  # log1p(E1) = 1, log1p(E2) = 2


def example(outputs: Mapping[str, Sequence[float]], example_id: str = "batch-0") -> dict:
    rows = [{"id": key, "prompt": "p", "model": "m", "input_tokens": 3} for key in outputs]
    return {
        "id": example_id,
        "input": json.dumps({"rows": rows}),
        "reference": json.dumps({"output_tokens": outputs}),
    }


def run(script: str, payload: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(METRIC / script)], input=json.dumps(payload),
        capture_output=True, text=True, cwd="/",
    )


def score(batch: dict, answer: object) -> dict:
    text = answer if isinstance(answer, str) else json.dumps(answer)
    process = run("score.py", {"example": batch, "answer": text})
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert set(result) == {"statistics"}
    return result["statistics"]


def row(pred: float, lo: float, hi: float) -> dict:
    return {"pred": pred, "lo": lo, "hi": hi}


def test_metric_manifest_freezes_direction_and_commands():
    manifest = json.loads((METRIC / "metric.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "ar09-quantile-v1"
    assert manifest["kind"] == "command"
    assert manifest["direction"] == "minimize"
    assert manifest["command"]["argv"] == ["python", "score.py"]
    assert manifest["aggregation"]["argv"] == ["python", "aggregate.py"]
    assert manifest["command"]["uses_model"] is False
    assert manifest["aggregation"]["uses_model"] is False


def test_exact_losses_of_point_and_interval_predictions():
    batch = example({"a": [0], "b": [E2], "c": [0, E2], "d": [0, 0]})
    stats = score(batch, {
        "a": row(E1, E1, E1),  # u = -1: 2 * (0.5 + 0.9 + 0.1)
        "b": row(0, 0, 0),  # u = 2: 2 * (1 + 0.2 + 1.8)
        "c": row(E1, 0, E2),  # 2 * mean(0.5 + 0 + 0.2, 0.5 + 0.2 + 0)
        "d": row(0, 0, 0),
    })
    assert stats["rows"] == 4
    assert stats["valid_rows"] == 4
    assert stats["invalid_rows"] == 0
    assert stats["loss_sum"] == pytest.approx(3.0 + 6.0 + 1.4 + 0.0)
    assert stats["mae_sum"] == pytest.approx(1.0 + 2.0 + 1.0 + 0.0)
    assert stats["coverage_sum"] == pytest.approx(0.0 + 0.0 + 1.0 + 1.0)
    assert stats["width_sum"] == pytest.approx(0.0 + 0.0 + 2.0 + 0.0)
    assert stats["batch_invalid"] is False


def test_true_quantiles_beat_shifted_predictions():
    outputs = [100, 150, 200, 250, 300, 350, 400, 450, 500, 900]
    batch = example({"a": outputs})
    best = score(batch, {"a": row(325, 120, 600)})["loss_sum"]
    for shifted in (row(200, 120, 600), row(325, 90, 600), row(325, 200, 600),
                    row(325, 120, 400), row(325, 120, 1000), row(325, 325, 325)):
        assert score(batch, {"a": shifted})["loss_sum"] > best


def test_merged_rows_with_20_and_30_outputs_have_exact_losses():
    batch = example({"twenty": [0] * 10 + [E2] * 10, "thirty": [0] * 10 + [E2] * 20})
    stats = score(batch, {"twenty": row(E1, E1, E1), "thirty": row(E1, 0, E1)})
    # twenty: u = +-1 поровну: 2 * (0.5 + mean(0.9, 0.1) + mean(0.1, 0.9))
    # thirty: y = 0 (1/3): 0.5 + 0 + 0.1; y = E2 (2/3): 0.5 + 0.2 + 0.9
    assert stats["loss_sum"] == pytest.approx(3.0 + 2 * (0.6 / 3 + 2 * 1.6 / 3))
    assert stats["mae_sum"] == pytest.approx(1.0 + 1.0)
    assert stats["coverage_sum"] == pytest.approx(0.0 + 1 / 3)
    assert stats["width_sum"] == pytest.approx(0.0 + 1.0)
    assert stats["invalid_rows"] == 0


@pytest.mark.parametrize("copies", [2, 3])
def test_row_weight_does_not_grow_with_number_of_outputs(copies):
    outputs = [12, 40, 75, 110, 260, 300, 420, 800, 1300, 2500]
    prediction = {"a": row(300, 60, 900)}
    single = score(example({"a": outputs}), prediction)
    merged = score(example({"a": outputs * copies}), prediction)
    assert merged["rows"] == single["rows"] == 1
    for name in ("loss_sum", "mae_sum", "coverage_sum", "width_sum"):
        assert merged[name] == pytest.approx(single[name])


def test_odd_merged_row_with_14_outputs():
    low, high = math.expm1(4), math.expm1(6)
    stats = score(example({"a": [low] * 7 + [high] * 7}), {"a": row(high, low, high)})
    # y = low: |4 - 6| + 2 * 0.1 * 2 = 2.4; y = high: 2 * 0.1 * 2 = 0.4
    assert stats["loss_sum"] == pytest.approx((2.4 + 0.4) / 2)
    assert stats["mae_sum"] == pytest.approx(1.0)
    assert stats["coverage_sum"] == pytest.approx(1.0)
    assert stats["width_sum"] == pytest.approx(2.0)


def test_bounds_and_integer_values_are_valid():
    batch = example({"a": [0, 5], "b": [100_000]})
    stats = score(batch, {"a": row(0, 0, 0), "b": {"lo": 100_000, "pred": 100_000.0, "hi": 1e5}})
    assert stats["invalid_rows"] == 0
    # y = 5 при pred = lo = hi = 0: 2 * mean(0, 0.5L + 0.1L + 0.9L), L = log1p(5)
    assert stats["loss_sum"] == pytest.approx(1.5 * math.log1p(5))


def test_worst_valid_row_stays_below_penalty():
    batch = example({"low": [0] * 10, "high": [15_000] * 10})
    stats = score(batch, {"low": row(1e5, 1e5, 1e5), "high": row(0, 0, 0)})
    assert stats["invalid_rows"] == 0
    assert stats["loss_sum"] == pytest.approx(3 * math.log1p(1e5) + 3 * math.log1p(15_000))
    assert 3 * math.log1p(1e5) < PENALTY


@pytest.mark.parametrize("value", [
    None,
    "missing",
    [1, 2, 3],
    5,
    {"pred": 1, "lo": 0},
    {"pred": 1, "lo": 0, "hi": 2, "extra": 1},
    {"pred": "1", "lo": 0, "hi": 2},
    {"pred": True, "lo": 0, "hi": 2},
    {"pred": None, "lo": 0, "hi": 2},
    {"pred": -1, "lo": -1, "hi": 2},
    {"pred": 1, "lo": 0, "hi": 100_001},
    {"pred": 10**30, "lo": 0, "hi": 10**31},
    {"pred": 10**400, "lo": 0, "hi": 10**401},
    {"pred": 1, "lo": -(10**400), "hi": 2},
    {"pred": 1, "lo": 2, "hi": 3},
    {"pred": 5, "lo": 0, "hi": 3},
])
def test_invalid_row_gets_penalty_and_does_not_touch_neighbours(value):
    batch = example({"bad": [10, 20], "good": [0]})
    answer: dict = {"good": row(0, 0, 0)}
    if value != "missing":
        answer["bad"] = value
    stats = score(batch, answer)
    assert stats["rows"] == 2
    assert stats["invalid_rows"] == 1
    assert stats["valid_rows"] == 1
    assert stats["loss_sum"] == pytest.approx(PENALTY)
    assert stats["mae_sum"] == 0
    assert stats["batch_invalid"] is False


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_nonfinite_json_numbers_get_row_penalty(literal):
    batch = example({"bad": [10], "good": [0]})
    answer = '{"good": {"pred": 0, "lo": 0, "hi": 0}, "bad": {"pred": %s, "lo": 0, "hi": 1}}'
    stats = score(batch, answer % literal)
    assert stats["invalid_rows"] == 1
    assert stats["loss_sum"] == pytest.approx(PENALTY)


def test_integer_literal_beyond_python_digit_limit_gets_row_penalty():
    batch = example({"bad": [10], "good": [0]})
    answer = ('{"good": {"pred": 0, "lo": 0, "hi": 0},'
              ' "bad": {"pred": 1%s, "lo": 0, "hi": 1}}' % ("0" * 5000))
    stats = score(batch, answer)
    assert stats["batch_invalid"] is False
    assert stats["invalid_rows"] == 1
    assert stats["loss_sum"] == pytest.approx(PENALTY)


@pytest.mark.parametrize(("answer", "reason"), [
    ({"good": {"pred": 0, "lo": 0, "hi": 0}}, "missing"),
    ({"good": {"pred": 0, "lo": 0, "hi": 0}, "bad": None}, "fields"),
    ({"good": {"pred": 0, "lo": 0, "hi": 0}, "bad": {"pred": 1, "lo": 0}}, "fields"),
    ({"good": {"pred": 0, "lo": 0, "hi": 0}, "bad": {"pred": "1", "lo": 0, "hi": 2}}, "type"),
    ({"good": {"pred": 0, "lo": 0, "hi": 0}, "bad": {"pred": 1, "lo": 0, "hi": 1e6}}, "range"),
    ({"good": {"pred": 0, "lo": 0, "hi": 0}, "bad": {"pred": 3, "lo": 0, "hi": 2}}, "order"),
    ("not json", "batch"),
])
def test_invalid_reasons_name_the_rule(answer, reason):
    stats = score(example({"bad": [10], "good": [0]}), answer)
    assert reason in stats["invalid_reasons"]


def test_duplicate_field_inside_row_gets_row_penalty():
    batch = example({"bad": [10], "good": [0]})
    answer = ('{"good": {"pred": 0, "lo": 0, "hi": 0},'
              ' "bad": {"pred": 1, "lo": 0, "hi": 2, "pred": 1}}')
    stats = score(batch, answer)
    assert stats["invalid_rows"] == 1
    assert stats["loss_sum"] == pytest.approx(PENALTY)


@pytest.mark.parametrize("answer", [
    "",
    "   ",
    "not json",
    "[]",
    '"text"',
    "42",
    "null",
    '{"a": {"pred": 0, "lo": 0, "hi": 0}, "b": {"pred": 0, "lo": 0, "hi": 0},'
    ' "c": {"pred": 0, "lo": 0, "hi": 0}}',
    '{"a": {"pred": 0, "lo": 0, "hi": 0}, "b": {"pred": 0, "lo": 0, "hi": 0},'
    ' "a": {"pred": 0, "lo": 0, "hi": 0}}',
])
def test_answer_that_does_not_match_batch_penalises_every_row(answer):
    stats = score(example({"a": [0], "b": [0]}), answer)
    assert stats["rows"] == 2
    assert stats["invalid_rows"] == 2
    assert stats["valid_rows"] == 0
    assert stats["loss_sum"] == pytest.approx(2 * PENALTY)
    assert stats["batch_invalid"] is True


def test_answer_key_order_does_not_matter():
    batch = example({"a": [0, E2], "b": [E1, 3]})
    forward = score(batch, {"a": row(E1, 0, E2), "b": row(2, 1, 3)})
    backward = score(batch, {"b": row(2, 1, 3), "a": row(E1, 0, E2)})
    assert forward == backward


@pytest.mark.parametrize("batch", [
    {"id": "x", "input": json.dumps({"rows": [{"id": "a"}]}),
     "reference": json.dumps({"output_tokens": {"b": [1]}})},
    {"id": "x", "input": json.dumps({"rows": [{"id": "a"}, {"id": "a"}]}),
     "reference": json.dumps({"output_tokens": {"a": [1]}})},
    {"id": "x", "input": json.dumps({"rows": [{"id": "a"}]}), "reference": None},
    {"id": "x", "input": json.dumps({"rows": [{"id": "a"}]}), "reference": "not json"},
    {"id": "x", "input": json.dumps({"rows": [{"id": "a"}]}),
     "reference": json.dumps({"output_tokens": {"a": []}})},
    {"id": "x", "input": json.dumps({"rows": [{"id": "a"}]}),
     "reference": json.dumps({"output_tokens": {"a": [-1]}})},
    {"id": "x", "input": json.dumps({"rows": [{"id": "a"}]}),
     "reference": json.dumps({"output_tokens": {"a": [100_001]}})},
    {"id": "x", "input": json.dumps({"rows": []}),
     "reference": json.dumps({"output_tokens": {}})},
    {"id": "x", "input": "not json", "reference": json.dumps({"output_tokens": {"a": [1]}})},
])
def test_broken_example_is_a_metric_error_not_a_score(batch):
    process = run("score.py", {"example": batch, "answer": json.dumps({"a": row(1, 1, 1)})})
    assert process.returncode != 0
    assert process.stdout == ""


def results(*batches: tuple[str, int, float]) -> list[dict]:
    return [{"example_id": key, "answer": "{}", "criteria": {},
             "statistics": {"rows": rows, "loss_sum": loss, "valid_rows": rows,
                            "invalid_rows": 0, "mae_sum": 0.0, "coverage_sum": 0.0,
                            "width_sum": 0.0, "batch_invalid": False, "invalid_reasons": {}}}
            for key, rows, loss in batches]


def test_aggregation_weights_rows_not_batches():
    process = run("aggregate.py", {"results": results(("b0", 1, 4.0), ("b1", 3, 0.0))})
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {"score": pytest.approx(1.0)}


@pytest.mark.parametrize("payload", [
    {"results": []},
    {"results": results(("b0", 1, 1.0), ("b0", 1, 1.0))},
    {"results": results(("b0", 0, 0.0))},
    {"results": [{"example_id": "b0", "answer": "{}", "criteria": {}}]},
    {"results": [{"example_id": "b0", "statistics": {"rows": 1, "loss_sum": "1"}}]},
    {"results": [{"example_id": "b0", "statistics": {"rows": 1, "loss_sum": -1.0}}]},
    {"results": [{"example_id": "b0", "statistics": {"rows": True, "loss_sum": 1.0}}]},
    {},
])
def test_incomplete_results_fail_aggregation(payload):
    process = run("aggregate.py", payload)
    assert process.returncode != 0
    assert process.stdout == ""

