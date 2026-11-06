"""Глобальные квантили длины ответа по train для baseline.

Метрика усредняет pinball по ответам строки, а затем по строкам, поэтому оптимальная константа —
взвешенный квантиль всех записанных длин с весом 1/k у каждого из k ответов строки. Пишет
params.json рядом со скриптом; solve.py читает только его.

uv run python cases/ar09/ways/baseline/fit.py --train cases/ar09/dataset/data/castillo.train.jsonl
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUANTILES = {"lo": Fraction(1, 10), "pred": Fraction(1, 2), "hi": Fraction(9, 10)}


def weighted_quantiles(rows: list[list[int]]) -> dict[str, int]:
    weights: dict[int, Fraction] = {}
    for outputs in rows:
        for value in outputs:
            weights[value] = weights.get(value, Fraction(0)) + Fraction(1, len(outputs))
    total = sum(weights.values(), Fraction(0))
    result = {}
    for name, q in QUANTILES.items():
        cumulative = Fraction(0)
        for value in sorted(weights):
            cumulative += weights[value]
            if cumulative >= q * total:
                result[name] = value
                break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.train.open(encoding="utf-8") as source:
        rows = [json.loads(line)["reference"]["output_tokens"] for line in source if line.strip()]
    params = {"source": arguments.train.name, "rows": len(rows), **weighted_quantiles(rows)}
    (HERE / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(json.dumps(params))


if __name__ == "__main__":
    main()
