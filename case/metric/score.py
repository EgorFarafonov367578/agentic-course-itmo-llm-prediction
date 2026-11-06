"""Метрика ar09-quantile-v1 для одного батча.

stdin — JSON {"example": {"id", "input", "reference"}, "answer": "..."} по контракту арены,
stdout — JSON {"statistics": {...}}; итоговый score считает aggregate.py.

Всё в шкале log1p(токенов). Для строки с записанными длинами y_1..y_k:
    loss = 2 * mean_i [rho_0.5(y_i - pred) + rho_0.1(y_i - lo) + rho_0.9(y_i - hi)],
    rho_q(u) = max(q * u, (q - 1) * u),
то есть MAE(pred) + 2 * pinball(lo, 0.1) + 2 * pinball(hi, 0.9).

Невалидная строка получает штраф 40: нет ответа, поля объекта строки не равны {pred, lo, hi}
(лишнее, недостающее или повторённое поле), нечисловое (в том числе bool) или нефинитное
значение, значение вне [0, 1e5], нарушено lo <= pred <= hi. Целые литералы ответа читаются как
float, поэтому слишком большое целое становится бесконечностью и штрафует только свою строку.
Если ответ не является JSON-объектом, повторяет id строки или содержит id не из батча, штраф 40
получают все строки батча. Ошибка во входе или эталоне (в том числе длина вне [0, 1e5]) — сбой
метрики (exit 1), а не оценка.
"""
import json
import math
import sys

PENALTY = 40.0
MAX_VALUE = 1e5
QUANTILES = {"pred": 0.5, "lo": 0.1, "hi": 0.9}
MISSING = object()


class JsonObject(dict):
    """Объект JSON, который помнит, были ли в нём повторяющиеся ключи."""

    duplicate = False


def json_object(pairs: list[tuple[str, object]]) -> JsonObject:
    result = JsonObject(pairs)
    result.duplicate = len(result) != len(pairs)
    return result


def pinball(u: float, q: float) -> float:
    return max(q * u, (q - 1) * u)


def is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def load_batch(example: dict) -> tuple[list[str], dict[str, list[float]]]:
    """ID строк батча и их записанные длины в шкале log1p; ошибка данных — исключение."""
    rows = json.loads(example["input"])["rows"]
    ids = [row["id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or not all(isinstance(key, str) for key in ids):
        raise ValueError("batch input must list unique string row IDs")
    reference = example["reference"]
    if not isinstance(reference, str):
        raise ValueError("batch reference is required")
    outputs = json.loads(reference)["output_tokens"]
    if not isinstance(outputs, dict) or set(outputs) != set(ids):
        raise ValueError("reference rows differ from input rows")
    for values in outputs.values():
        if not isinstance(values, list) or not values or not all(
            is_number(value) and 0 <= value <= MAX_VALUE for value in values
        ):
            raise ValueError("reference must hold nonempty lists of lengths in [0, 1e5]")
    return ids, {key: [math.log1p(value) for value in outputs[key]] for key in ids}


def parse_answer(answer: str, ids: list[str]) -> JsonObject | None:
    """Объект ответа или None, если ответ целиком не соответствует батчу."""
    try:
        parsed = json.loads(answer, object_pairs_hook=json_object, parse_int=float)
    except (ValueError, RecursionError):
        return None
    if not isinstance(parsed, JsonObject) or parsed.duplicate or not set(parsed) <= set(ids):
        return None
    return parsed


def invalid_reason(value: object) -> str | None:
    if value is MISSING:
        return "missing"
    if not isinstance(value, JsonObject) or value.duplicate or set(value) != set(QUANTILES):
        return "fields"
    if not all(is_number(value[name]) and math.isfinite(value[name]) for name in QUANTILES):
        return "type"
    # Точное сравнение: не переполняется на любом int и ложно для NaN.
    if not all(0 <= value[name] <= MAX_VALUE for name in QUANTILES):
        return "range"
    if not value["lo"] <= value["pred"] <= value["hi"]:
        return "order"
    return None


def row_loss(prediction: dict, outputs: list[float]) -> tuple[float, float, float, float]:
    """Ошибка строки, MAE, покрытие интервала и его ширина в шкале log1p."""
    scaled = {name: math.log1p(prediction[name]) for name in QUANTILES}
    loss = 2 * math.fsum(
        pinball(y - scaled[name], q) for y in outputs for name, q in QUANTILES.items()
    ) / len(outputs)
    mae = math.fsum(abs(y - scaled["pred"]) for y in outputs) / len(outputs)
    lo, hi = scaled["lo"], scaled["hi"]
    coverage = sum(lo <= y <= hi for y in outputs) / len(outputs)
    return loss, mae, coverage, hi - lo


def evaluate(example: dict, answer: str) -> dict:
    ids, outputs = load_batch(example)
    parsed = parse_answer(answer, ids)
    losses, maes, coverages, widths = [], [], [], []
    reasons: dict[str, int] = {}
    for key in ids:
        reason = "batch" if parsed is None else invalid_reason(parsed.get(key, MISSING))
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
            losses.append(PENALTY)
            continue
        assert parsed is not None
        loss, mae, coverage, width = row_loss(parsed[key], outputs[key])
        losses.append(loss)
        maes.append(mae)
        coverages.append(coverage)
        widths.append(width)
    return {
        "rows": len(ids),
        "loss_sum": math.fsum(losses),
        "valid_rows": len(maes),
        "invalid_rows": len(ids) - len(maes),
        "mae_sum": math.fsum(maes),
        "coverage_sum": math.fsum(coverages),
        "width_sum": math.fsum(widths),
        "batch_invalid": parsed is None,
        "invalid_reasons": reasons,
    }


def main() -> None:
    payload = json.load(sys.stdin)
    answer = payload["answer"]
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    json.dump({"statistics": evaluate(payload["example"], answer)}, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
