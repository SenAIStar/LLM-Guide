from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any


def build_trajectory_record(
    *,
    question_id: str,
    group_id: str,
    question: str,
    steps: Iterable[Mapping[str, Any]],
    final_answer: str,
    answerable: bool,
) -> dict[str, Any]:
    """Keep actions trainable while masking retriever and reranker observations."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": question, "trainable": False}
    ]
    for step in steps:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(step["action"], ensure_ascii=False),
                "trainable": True,
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(step["observation"], ensure_ascii=False),
                "trainable": False,
            }
        )
    messages.append({"role": "assistant", "content": final_answer, "trainable": True})
    return {
        "id": question_id,
        "group_id": group_id,
        "messages": messages,
        "metadata": {"answerable": answerable},
    }
