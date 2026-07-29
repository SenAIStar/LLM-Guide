from __future__ import annotations

from collections import Counter
from typing import Any


def _unique_labels(
    records: list[dict[str, Any]], id_field: str, label_field: str
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in records:
        sample_id = str(row[id_field])
        if sample_id in labels:
            raise ValueError(f"duplicate {id_field}: {sample_id}")
        labels[sample_id] = str(row[label_field])
    return labels


def classification_metrics(
    gold_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Compute accuracy and macro-F1 for label-based legal tasks."""
    gold_by_id = _unique_labels(gold_records, "id", "target_label")
    pred_by_id = _unique_labels(predictions, "id", "predicted_label")
    if not gold_by_id:
        raise ValueError("gold records are empty")

    sample_ids = sorted(gold_by_id)
    missing_ids = set(gold_by_id) - set(pred_by_id)
    unexpected_ids = set(pred_by_id) - set(gold_by_id)
    predicted_labels = {
        pred_by_id[sample_id] for sample_id in sample_ids if sample_id in pred_by_id
    }
    labels = sorted(set(gold_by_id.values()) | predicted_labels)
    per_label_f1: dict[str, float] = {}
    correct = 0

    for sample_id in sample_ids:
        correct += int(gold_by_id[sample_id] == pred_by_id.get(sample_id))

    for label in labels:
        counts = Counter()
        for sample_id in sample_ids:
            gold = gold_by_id[sample_id]
            pred = pred_by_id.get(sample_id)
            counts["tp"] += int(gold == label and pred == label)
            counts["fp"] += int(gold != label and pred == label)
            counts["fn"] += int(gold == label and pred != label)

        precision = _safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        per_label_f1[label] = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "evaluated_samples": len(sample_ids),
        "prediction_coverage": round((len(sample_ids) - len(missing_ids)) / len(sample_ids), 6),
        "missing_predictions": len(missing_ids),
        "unexpected_predictions": len(unexpected_ids),
        "accuracy": round(correct / len(sample_ids), 6),
        "macro_f1": round(sum(per_label_f1.values()) / len(per_label_f1), 6),
    }


def output_coverage(predictions: list[dict[str, Any]]) -> dict[str, float | int]:
    """Report empty generations; this is a health check, not an answer-quality score."""
    total = len(predictions)
    non_empty = sum(bool(str(item.get("answer", "")).strip()) for item in predictions)
    return {
        "predictions": total,
        "non_empty_outputs": non_empty,
        "coverage": round(non_empty / total, 6) if total else 0.0,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
