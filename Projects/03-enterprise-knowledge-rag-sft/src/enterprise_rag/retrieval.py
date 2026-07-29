from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


ScoredChunk = tuple[Mapping[str, Any], float]
Tokenizer = Callable[[str], list[str]]


def filter_by_acl(
    chunks: Iterable[Mapping[str, Any]], user_groups: set[str]
) -> list[Mapping[str, Any]]:
    """Apply authorization before chunks enter any retriever or model context."""
    visible: list[Mapping[str, Any]] = []
    for chunk in chunks:
        allowed = {str(group) for group in chunk.get("acl_groups", [])}
        if allowed & user_groups:
            visible.append(chunk)
    return visible


def build_bm25_index(
    chunks: Sequence[Mapping[str, Any]], tokenize: Tokenizer
) -> BM25Okapi:
    tokenized_corpus = [tokenize(str(chunk["text"])) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def search_bm25(
    query: str,
    chunks: Sequence[Mapping[str, Any]],
    index: BM25Okapi,
    tokenize: Tokenizer,
    top_k: int,
) -> list[ScoredChunk]:
    scores = index.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:top_k]
    return [(chunks[int(i)], float(scores[int(i)])) for i in order]


def build_faiss_ip_index(embeddings: np.ndarray) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """Normalize vectors so inner product is cosine similarity."""
    vectors = np.asarray(embeddings, dtype="float32").copy()
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors


def search_dense(
    query_embedding: np.ndarray,
    chunks: Sequence[Mapping[str, Any]],
    index: faiss.IndexFlatIP,
    top_k: int,
) -> list[ScoredChunk]:
    query = np.asarray(query_embedding, dtype="float32").reshape(1, -1).copy()
    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)
    return [
        (chunks[int(i)], float(score))
        for i, score in zip(indices[0], scores[0], strict=True)
        if i >= 0
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], rank_constant: int = 60
) -> list[ScoredChunk]:
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    scores: defaultdict[str, float] = defaultdict(float)
    rows: dict[str, Mapping[str, Any]] = {}
    for ranking in rankings:
        for rank, (chunk, _source_score) in enumerate(ranking, start=1):
            chunk_id = str(chunk["chunk_id"])
            rows[chunk_id] = chunk
            scores[chunk_id] += 1.0 / (rank_constant + rank)
    return sorted(((rows[key], score) for key, score in scores.items()), key=lambda x: -x[1])


def hybrid_retrieve(
    query: str,
    chunks: Sequence[Mapping[str, Any]],
    user_groups: set[str],
    bm25_search: Callable[[str, Sequence[Mapping[str, Any]], int], Sequence[ScoredChunk]],
    dense_search: Callable[[str, Sequence[Mapping[str, Any]], int], Sequence[ScoredChunk]],
    rerank: Callable[[str, Sequence[Mapping[str, Any]]], Sequence[ScoredChunk]],
    candidate_k: int = 40,
    final_k: int = 5,
) -> list[ScoredChunk]:
    visible = filter_by_acl(chunks, user_groups)
    if not visible:
        return []
    sparse = bm25_search(query, visible, candidate_k)
    dense = dense_search(query, visible, candidate_k)
    fused = reciprocal_rank_fusion([sparse, dense])[:candidate_k]
    reranked = rerank(query, [chunk for chunk, _score in fused])
    return list(reranked[:final_k])
