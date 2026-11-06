"""Baseline по протоколу runner: валидный ответ и независимость строк от состава батча."""
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
SOLVE = CASE / "ways" / "baseline" / "solve.py"
SCORE = CASE / "metric" / "score.py"


def load_fit():
    # Без __pycache__ в каталоге кандидата: он попал бы в снимок решения.
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location("ar09_baseline_fit",
                                                      SOLVE.with_name("fit.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_fit_weights_each_row_equally_whatever_its_number_of_outputs():
    fit = load_fit()
    # Без веса 1/k тридцать единиц перевесили бы две строки по десять сотен.
    assert fit.weighted_quantiles([[1] * 30, [100] * 10, [100] * 10]) == {
        "lo": 1, "pred": 100, "hi": 100}
    assert fit.weighted_quantiles([[1] * 10, [100] * 10, [100] * 10]) == fit.weighted_quantiles(
        [[1] * 20, [100] * 30, [100] * 10])


def rows(count: int) -> list[dict]:
    return [{"id": f"r{i:03d}", "prompt": "текст " * i, "model": f"m{i % 3}",
             "input_tokens": 10 + i} for i in range(count)]


def solve(batch: list[dict]) -> dict:
    task = {"input": json.dumps({"rows": batch}, ensure_ascii=False), "assets_dir": "/assets"}
    process = subprocess.run([sys.executable, "-E", "-s", str(SOLVE)], input=json.dumps(task),
                             capture_output=True, text=True, check=True, cwd="/")
    return json.loads(json.loads(process.stdout)["answer"])


def test_answer_is_valid_for_the_metric():
    batch = rows(7)
    answer = solve(batch)
    example = {"id": "b", "input": json.dumps({"rows": batch}),
               "reference": json.dumps({"output_tokens": {r["id"]: [100, 400] for r in batch}})}
    process = subprocess.run([sys.executable, str(SCORE)], capture_output=True, text=True,
                             input=json.dumps({"example": example, "answer": json.dumps(answer)}),
                             check=True, cwd="/")
    statistics = json.loads(process.stdout)["statistics"]
    assert statistics["invalid_rows"] == 0
    assert statistics["rows"] == 7


def test_row_prediction_ignores_order_and_neighbours():
    batch = rows(20)
    full = solve(batch)
    shuffled = batch[:]
    random.Random(1).shuffle(shuffled)
    assert solve(shuffled) == full
    thinned = batch[::3]
    assert solve(thinned) == {r["id"]: full[r["id"]] for r in thinned}
