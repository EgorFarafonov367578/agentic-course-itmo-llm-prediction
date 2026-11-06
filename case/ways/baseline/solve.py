"""Baseline: одни и те же квантили 0.1 / 0.5 / 0.9 длины ответа по train для каждой строки.

Протокол немодельного runner арены: stdin — {"input": "<JSON батча>", "assets_dir", ...},
stdout — {"answer": "<JSON {id: {pred, lo, hi}}>"}. Квантили посчитаны fit.py и лежат в params.json.
"""
import json
import sys
from pathlib import Path

PARAMS = json.loads((Path(__file__).resolve().parent / "params.json").read_text())


def predict(row: dict) -> dict:
    return {"pred": PARAMS["pred"], "lo": PARAMS["lo"], "hi": PARAMS["hi"]}


def main() -> None:
    rows = json.loads(json.load(sys.stdin)["input"])["rows"]
    answer = {row["id"]: predict(row) for row in rows}
    json.dump({"answer": json.dumps(answer, separators=(",", ":"))}, sys.stdout)


if __name__ == "__main__":
    main()
