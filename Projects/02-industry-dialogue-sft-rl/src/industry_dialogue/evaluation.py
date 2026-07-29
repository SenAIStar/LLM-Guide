from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def _binary_rate(rows: list[Mapping[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if isinstance(row.get(field), bool)]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_model_annotations(
    rows: Iterable[Mapping[str, Any]], model_field: str = "model"
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row.get(model_field, "")).strip()
        if model:
            grouped[model].append(row)

    summary: dict[str, dict[str, float | int | None]] = {}
    for model, model_rows in sorted(grouped.items()):
        summary[model] = {
            "count": len(model_rows),
            "task_success_rate": _binary_rate(model_rows, "task_success"),
            "business_correct_rate": _binary_rate(model_rows, "business_correct"),
            "policy_violation_rate": _binary_rate(model_rows, "policy_violation"),
            "over_refusal_rate": _binary_rate(model_rows, "over_refusal"),
            "asks_sensitive_info_rate": _binary_rate(model_rows, "asks_sensitive_info"),
        }
    return summary


def summarize_blind_preferences(
    rows: Iterable[Mapping[str, Any]], contender: str, baseline: str
) -> dict[str, float | int]:
    counts: Counter[str] = Counter()
    used = 0
    for row in rows:
        model_a = row.get("model_a")
        model_b = row.get("model_b")
        winner = row.get("winner")
        if {model_a, model_b} != {contender, baseline} or winner not in {"a", "b", "tie"}:
            continue
        used += 1
        if winner == "tie":
            counts["tie"] += 1
        else:
            winning_model = model_a if winner == "a" else model_b
            counts["win" if winning_model == contender else "loss"] += 1

    decisive = counts["win"] + counts["loss"]
    denominator = used or 1
    return {
        "count": used,
        "win_rate": counts["win"] / denominator,
        "loss_rate": counts["loss"] / denominator,
        "tie_rate": counts["tie"] / denominator,
        "non_tie_win_rate": counts["win"] / decisive if decisive else 0.0,
    }
