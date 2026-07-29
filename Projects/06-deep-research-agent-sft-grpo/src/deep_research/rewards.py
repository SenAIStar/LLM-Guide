from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(frozen=True)
class ResearchRewardSignals:
    task_coverage: float
    citation_precision: float
    citation_completeness: float
    source_coverage: float
    conflict_handling: float
    duplicate_query_ratio: float
    invalid_citation_count: int
    unsupported_claim_count: int
    step_count: int


def research_reward(signals: ResearchRewardSignals) -> float:
    score = (
        0.35 * signals.task_coverage
        + 0.25 * signals.citation_precision
        + 0.15 * signals.citation_completeness
        + 0.15 * signals.source_coverage
        + 0.10 * signals.conflict_handling
    )
    score -= 0.10 * signals.duplicate_query_ratio
    score -= 0.30 * signals.invalid_citation_count
    score -= 0.40 * signals.unsupported_claim_count
    score -= 0.01 * signals.step_count
    return score


def group_relative_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two rewards from the same question")
    mean, scale = fmean(rewards), pstdev(rewards)
    if scale <= eps:
        return [0.0] * len(rewards)
    return [(reward - mean) / scale for reward in rewards]
