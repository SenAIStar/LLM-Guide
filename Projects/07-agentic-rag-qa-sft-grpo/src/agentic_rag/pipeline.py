from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence


class Stage(str, Enum):
    CLASSIFY = "classify"
    REWRITE = "rewrite"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    ANSWER = "answer"
    ABSTAIN = "abstain"


@dataclass
class AgentState:
    question: str
    stage: Stage = Stage.CLASSIFY
    rewritten_queries: list[str] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    retrieval_rounds: int = 0
    stop_reason: str = ""


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in ranked]


def run_agentic_rag(
    question: str,
    policy: Callable[[AgentState], dict],
    retrievers: dict[str, Callable[[str, int], list[str]]],
    reranker: Callable[[str, list[str], int], list[str]],
    *,
    max_retrieval_rounds: int = 2,
    max_actions: int = 12,
) -> AgentState:
    if max_retrieval_rounds <= 0 or max_actions <= 0:
        raise ValueError("retrieval and action budgets must be positive")
    state = AgentState(question=question)
    seen_queries: set[str] = set()

    while state.stage not in {Stage.ANSWER, Stage.ABSTAIN} and len(state.actions) < max_actions:
        action = policy(state)
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            state.stop_reason = "invalid_action"
            state.stage = Stage.ABSTAIN
            break
        state.actions.append(action)

        if action["type"] == "rewrite":
            query = " ".join(str(action.get("query", "")).lower().split())
            if not query or query in seen_queries:
                state.stop_reason = "empty_or_repeated_query"
                state.stage = Stage.ABSTAIN
                continue
            seen_queries.add(query)
            state.rewritten_queries.append(query)
            state.stage = Stage.RETRIEVE
        elif action["type"] == "retrieve":
            if state.retrieval_rounds >= max_retrieval_rounds:
                state.stop_reason = "retrieval_budget_exhausted"
                state.stage = Stage.ABSTAIN
                continue
            query = state.rewritten_queries[-1] if state.rewritten_queries else question
            branches = action.get("branches", ["dense"])
            if not isinstance(branches, list) or not branches or any(
                name not in retrievers for name in branches
            ):
                state.stop_reason = "invalid_retriever_branch"
                state.stage = Stage.ABSTAIN
                continue
            rankings = [retrievers[name](query, 50) for name in branches]
            state.candidate_ids = reciprocal_rank_fusion(rankings)[:50]
            state.retrieval_rounds += 1
            state.stage = Stage.RERANK
        elif action["type"] == "rerank":
            state.evidence_ids = reranker(question, state.candidate_ids, 5)
            try:
                state.stage = Stage(action.get("next", "answer"))
            except ValueError:
                state.stop_reason = "invalid_next_stage"
                state.stage = Stage.ABSTAIN
        elif action["type"] in {"answer", "abstain"}:
            state.stage = Stage(action["type"])
            state.stop_reason = action["type"]
        else:
            state.stop_reason = "unknown_action"
            state.stage = Stage.ABSTAIN
    if state.stage not in {Stage.ANSWER, Stage.ABSTAIN}:
        state.stop_reason = "action_budget_exhausted"
        state.stage = Stage.ABSTAIN
    return state
