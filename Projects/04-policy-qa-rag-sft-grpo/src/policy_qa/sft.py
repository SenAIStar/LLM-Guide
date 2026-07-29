from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .pipeline import PolicyChunk, build_grounded_prompt


SYSTEM_PROMPT = (
    "你是政策问答助手。只根据给定证据回答，并在结论后引用 [chunk_id]。"
    "证据冲突时说明冲突，证据不足或无权限时不要推测。"
)


def build_sft_record(
    *,
    question_id: str,
    group_id: str,
    question: str,
    evidence: Sequence[PolicyChunk],
    target_answer: str,
    answerable: bool,
) -> dict[str, Any]:
    """Build one auditable SFT row from a frozen retrieval result."""
    return {
        "id": question_id,
        "group_id": group_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_grounded_prompt(question, evidence)},
            {"role": "assistant", "content": target_answer},
        ],
        "metadata": {
            "answerable": answerable,
            "evidence_ids": [chunk.chunk_id for chunk in evidence],
        },
    }
