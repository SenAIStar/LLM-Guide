from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import re


CITATION_PATTERN = re.compile(r"\[([^\[\]]+#\d{4})\]")


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("Recall@k is undefined for a query without relevant chunks")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str], relevance: Mapping[str, int | float], k: int = 10
) -> float:
    gains = [float(relevance.get(chunk_id, 0.0)) for chunk_id in retrieved[:k]]
    dcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def extract_citations(answer: str) -> list[str]:
    return CITATION_PATTERN.findall(answer)


def citation_precision(answer: str, supporting_chunk_ids: set[str]) -> float | None:
    citations = extract_citations(answer)
    if not citations:
        return None
    return sum(citation in supporting_chunk_ids for citation in citations) / len(citations)


def refusal_recall(rows: Iterable[Mapping[str, Any]]) -> float | None:
    unanswerable = [row for row in rows if row.get("answerable") is False]
    if not unanswerable:
        return None
    refused = sum(row.get("refused") is True for row in unanswerable)
    return refused / len(unanswerable)
