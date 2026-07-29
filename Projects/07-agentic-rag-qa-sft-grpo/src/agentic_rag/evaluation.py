from __future__ import annotations


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("Recall@k is undefined without relevant evidence")
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


def summarize(records: list[dict]) -> dict[str, float]:
    answerable = [row for row in records if row["answerable"]]
    boundary = [row for row in records if not row["answerable"]]
    return {
        "recall_at_5": safe_rate(
            sum(
                recall_at_k(row["retrieved_ids"], set(row["gold_chunk_ids"]), 5)
                for row in answerable
            ),
            len(answerable),
        ),
        "task_success_rate": safe_rate(
            sum(row["task_success"] for row in answerable), len(answerable)
        ),
        "invalid_tool_trajectory_rate": safe_rate(
            sum(row["invalid_tool_calls"] > 0 for row in records), len(records)
        ),
        "boundary_correct_rate": safe_rate(
            sum(row["boundary_correct"] for row in boundary), len(boundary)
        ),
        "mean_retrieval_rounds": safe_rate(
            sum(row["retrieval_rounds"] for row in records), len(records)
        ),
    }
