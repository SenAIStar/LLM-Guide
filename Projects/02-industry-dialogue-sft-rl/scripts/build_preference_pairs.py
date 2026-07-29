from __future__ import annotations

import argparse
from pathlib import Path

from industry_dialogue.data_pipeline import build_preference_pairs, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chosen/rejected pairs from ranked answers")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Use every non-tied pair instead of only adjacent ranks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = build_preference_pairs(read_jsonl(args.input), adjacent_only=not args.all_pairs)
    write_jsonl(args.output, pairs)
    print(f"wrote {len(pairs)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
