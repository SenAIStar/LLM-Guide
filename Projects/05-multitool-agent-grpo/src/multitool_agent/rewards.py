from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(frozen=True)
class AgentRewardSignals:
    goal_state_match: bool
    tool_calls_valid: bool
    policy_compliant: bool
    final_answer_consistent: bool
    unsafe_action_count: int
    repeated_action_count: int
    step_count: int


def trajectory_reward(signals: AgentRewardSignals) -> float:
    score = (
        0.50 * signals.goal_state_match
        + 0.15 * signals.tool_calls_valid
        + 0.20 * signals.policy_compliant
        + 0.15 * signals.final_answer_consistent
    )
    score -= 1.0 * signals.unsafe_action_count
    score -= 0.2 * signals.repeated_action_count
    score -= 0.02 * signals.step_count
    return score


def group_relative_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two rewards from the same task")
    mean, scale = fmean(rewards), pstdev(rewards)
    if scale <= eps:
        return [0.0] * len(rewards)
    return [(reward - mean) / scale for reward in rewards]
