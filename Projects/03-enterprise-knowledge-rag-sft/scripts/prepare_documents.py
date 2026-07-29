from __future__ import annotations

import argparse
import json
from pathlib import Path

from enterprise_rag.chunking import DocumentSection, chunk_sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build versioned, ACL-aware chunks from JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--overlap-chars", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sections: list[DocumentSection] = []
    with args.input.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            sections.append(
                DocumentSection(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    section=row["section"],
                    text=row["text"],
                    version=row["version"],
                    page=row.get("page"),
                    department=row["department"],
                    acl_groups=tuple(row["acl_groups"]),
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for chunk in chunk_sections(sections, args.max_chars, args.overlap_chars):
            target.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
