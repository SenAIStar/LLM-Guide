from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any


def trajectory_segments(
    question: str,
    steps: Iterable[Mapping[str, Any]],
    report: str,
) -> list[dict[str, Any]]:
    """Mark policy text as trainable and environment text as masked."""
    segments: list[dict[str, Any]] = [
        {"role": "user", "content": question, "trainable": False}
    ]
    for step in steps:
        segments.append(
            {
                "role": "assistant",
                "content": json.dumps(step["action"], ensure_ascii=False),
                "trainable": True,
            }
        )
        segments.append(
            {
                "role": "tool",
                "content": json.dumps(step["observation"], ensure_ascii=False),
                "trainable": False,
            }
        )
    segments.append({"role": "assistant", "content": report, "trainable": True})
    return segments
