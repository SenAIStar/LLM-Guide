from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dialogue prompts to VeRL parquet")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = str(row.get("prompt_id", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if not prompt_id or not prompt:
                raise ValueError(f"missing prompt_id or prompt at line {line_number}")
            if prompt_id in seen:
                raise ValueError(f"duplicate prompt_id: {prompt_id}")
            seen.add(prompt_id)
            system = str(row.get("system", "")).strip()
            prompt_messages: list[dict[str, str]] = []
            if system:
                prompt_messages.append({"role": "system", "content": system})
            prompt_messages.append({"role": "user", "content": prompt})
            rows.append(
                {
                    "prompt": prompt_messages,
                    "data_source": row.get("data_source", "industry_dialogue"),
                    "ability": row.get("group_id", "industry_dialogue"),
                    "extra_info": {
                        "prompt_id": prompt_id,
                        "group_id": row.get("group_id", prompt_id),
                    },
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"wrote {len(rows)} prompts to {args.output}")


if __name__ == "__main__":
    main()
