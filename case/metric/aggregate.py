"""Агрегация ar09-quantile-v1: средняя ошибка по всем строкам раздела.

stdin — JSON {"results": [...]} по контракту арены, у каждой записи statistics из score.py.
stdout — JSON {"score": сумма loss_sum / сумма rows}. Это среднее по строкам, а не среднее
средних по батчам разного размера. Неполный или противоречивый набор — сбой агрегации (exit 1).
"""
import json
import math
import sys


def is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def combine(payload: dict) -> float:
    results = payload["results"]
    if not isinstance(results, list) or not results:
        raise ValueError("cannot aggregate an empty result set")
    seen: set[str] = set()
    losses, rows = [], 0
    for result in results:
        key, statistics = result["example_id"], result["statistics"]
        count, loss = statistics["rows"], statistics["loss_sum"]
        if key in seen:
            raise ValueError("duplicate example in result set")
        if type(count) is not int or count < 1:
            raise ValueError("batch statistics must count at least one row")
        if not is_number(loss) or not math.isfinite(loss) or loss < 0:
            raise ValueError("batch loss must be a finite nonnegative number")
        seen.add(key)
        losses.append(loss)
        rows += count
    return math.fsum(losses) / rows


if __name__ == "__main__":
    json.dump({"score": combine(json.load(sys.stdin))}, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
