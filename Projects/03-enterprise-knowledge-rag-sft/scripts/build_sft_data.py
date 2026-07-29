from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from enterprise_rag.prompting import SYSTEM_PROMPT, build_user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grounded RAG SFT samples")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def to_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "group_id": row["group_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(row["question"], row.get("retrieved_chunks", [])),
            },
            {"role": "assistant", "content": row["target_answer"]},
        ],
        "metadata": {
            "answerable": row["answerable"],
            "gold_chunk_ids": row.get("gold_chunk_ids", []),
            "index_version": row["index_version"],
        },
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            target.write(json.dumps(to_sft_row(json.loads(line)), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
