from __future__ import annotations

from collections import defaultdict


def pass_pow_k(runs: list[dict], k: int) -> float:
    """Fraction of tasks that succeed in every one of their first k runs."""
    if k <= 0:
        raise ValueError("k must be positive")
    by_task: dict[str, list[bool]] = defaultdict(list)
    for run in runs:
        by_task[run["task_id"]].append(bool(run["goal_state_match"]))
    eligible = [values[:k] for values in by_task.values() if len(values) >= k]
    return sum(all(values) for values in eligible) / len(eligible) if eligible else 0.0


def summarize(runs: list[dict]) -> dict[str, float]:
    count = len(runs)
    if not count:
        return {
            "task_success_rate": 0.0,
            "invalid_tool_call_rate": 0.0,
            "unsafe_action_rate": 0.0,
            "mean_tool_calls": 0.0,
            "pass_pow_4": 0.0,
        }
    return {
        "task_success_rate": sum(row["goal_state_match"] for row in runs) / count,
        "invalid_tool_call_rate": sum(row["invalid_tool_call"] for row in runs) / count,
        "unsafe_action_rate": sum(row["unsafe_action"] for row in runs) / count,
        "mean_tool_calls": sum(row["tool_calls"] for row in runs) / count,
        "pass_pow_4": pass_pow_k(runs, 4),
    }
