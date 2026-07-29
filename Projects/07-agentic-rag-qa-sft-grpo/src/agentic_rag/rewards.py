from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(frozen=True)
class AgenticRagSignals:
    gold_evidence_hit: bool
    tool_calls_valid: bool
    answer_correct: bool
    citations_support_claims: bool
    boundary_correct: bool
    invalid_tool_calls: int
    repeated_queries: int
    unsupported_claims: int
    retrieval_rounds: int


def reward(signals: AgenticRagSignals) -> float:
    score = (
        0.25 * signals.gold_evidence_hit
        + 0.15 * signals.tool_calls_valid
        + 0.30 * signals.answer_correct
        + 0.20 * signals.citations_support_claims
        + 0.10 * signals.boundary_correct
    )
    score -= 0.30 * signals.invalid_tool_calls
    score -= 0.20 * signals.repeated_queries
    score -= 0.40 * signals.unsupported_claims
    score -= 0.03 * signals.retrieval_rounds
    return score


def group_relative_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two rewards from the same question")
    mean, scale = fmean(rewards), pstdev(rewards)
    if scale <= eps:
        return [0.0] * len(rewards)
    return [(reward - mean) / scale for reward in rewards]
