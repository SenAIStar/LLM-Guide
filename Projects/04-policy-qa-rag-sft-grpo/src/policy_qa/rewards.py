from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(frozen=True)
class PolicyRewardSignals:
    answer_correct: bool
    citation_valid: bool
    evidence_supports_claims: bool
    boundary_correct: bool
    unauthorized_evidence: bool = False
    unsupported_claim_count: int = 0


def policy_reward(signals: PolicyRewardSignals) -> float:
    score = (
        0.35 * float(signals.answer_correct)
        + 0.20 * float(signals.citation_valid)
        + 0.25 * float(signals.evidence_supports_claims)
        + 0.20 * float(signals.boundary_correct)
    )
    if signals.unauthorized_evidence:
        score -= 1.0
    score -= 0.5 * signals.unsupported_claim_count
    return score


def group_relative_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """The key normalization step used by a GRPO-style objective."""
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two rewards from the same prompt")
    mean = fmean(rewards)
    scale = pstdev(rewards)
    if scale <= eps:
        return [0.0] * len(rewards)
    return [(reward - mean) / scale for reward in rewards]
