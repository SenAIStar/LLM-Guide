from __future__ import annotations


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("Recall@k is undefined without relevant evidence")
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(records: list[dict]) -> dict[str, float]:
    """Use explicit booleans produced by deterministic checks or human labels."""
    total = len(records)
    answerable = [row for row in records if row["answerable"]]
    boundary = [row for row in records if not row["answerable"]]
    return {
        "answer_accuracy": rate(
            sum(row["answer_correct"] for row in answerable), len(answerable)
        ),
        "citation_valid_rate": rate(sum(row["citation_valid"] for row in records), total),
        "boundary_correct_rate": rate(
            sum(row["boundary_correct"] for row in boundary), len(boundary)
        ),
        "unauthorized_leak_rate": rate(sum(row["unauthorized_leak"] for row in records), total),
    }
