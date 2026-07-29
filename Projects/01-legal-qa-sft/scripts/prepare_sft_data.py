from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from legal_qa.data_pipeline import (
    convert_disc_pair,
    convert_disc_triplet,
    deduplicate,
    quality_report,
    split_by_group,
)
from legal_qa.pii import PiiRedactor


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(pair_path: Path, triplet_path: Path, output_dir: Path) -> dict[str, Any]:
    redactor = PiiRedactor()
    records = [convert_disc_pair(row, redactor) for row in read_jsonl(pair_path)]
    records.extend(convert_disc_triplet(row, redactor) for row in read_jsonl(triplet_path))

    records, duplicate_count = deduplicate(records)
    splits = split_by_group(records, seed="legal-sft-v1")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records)

    report = {
        **quality_report(records),
        "exact_duplicates_removed": duplicate_count,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare legal SFT data from DISC-Law-SFT files.")
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--triplet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    report = prepare(args.pair, args.triplet, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
