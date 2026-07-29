from __future__ import annotations

import json
from dataclasses import asdict
from typing import Callable

from .tools import MockState, execute_action


def run_agent(
    instruction: str,
    state: MockState,
    allowed_tools: set[str],
    policy: Callable[[str, list[dict]], str],
    *,
    max_steps: int = 8,
) -> dict:
    trajectory: list[dict] = []
    seen_actions: set[str] = set()

    for step in range(max_steps):
        raw = policy(instruction, trajectory)
        try:
            action = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {
                "final": "Stopped: invalid JSON action",
                "trajectory": trajectory,
                "state": asdict(state),
            }
        if not isinstance(action, dict):
            return {
                "final": "Stopped: action must be a JSON object",
                "trajectory": trajectory,
                "state": asdict(state),
            }
        if action.get("type") == "final":
            return {
                "final": action.get("answer", ""),
                "trajectory": trajectory,
                "state": asdict(state),
            }

        fingerprint = json.dumps(action, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen_actions:
            return {
                "final": "Stopped: repeated action",
                "trajectory": trajectory,
                "state": asdict(state),
            }
        seen_actions.add(fingerprint)

        if action.get("type") != "tool":
            observation = {"ok": False, "error_code": "INVALID_ACTION_TYPE"}
        else:
            observation = execute_action(state, action, allowed_tools)
        trajectory.append({"step": step, "action": action, "observation": observation})

    return {
        "final": "Stopped: step budget exhausted",
        "trajectory": trajectory,
        "state": asdict(state),
    }
