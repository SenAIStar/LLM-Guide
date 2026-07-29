from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


def normalize_text(value: str) -> str:
    """Normalize Unicode and repeated whitespace without changing meaning."""
    normalized = unicodedata.normalize("NFC", value)
    return " ".join(normalized.split()).strip()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield item


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def validate_conversations(conversations: object) -> list[dict[str, str]]:
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(conversations):
        if not isinstance(message, dict):
            raise ValueError(f"conversation item {index} must be an object")
        role = message.get("from")
        value = message.get("value")
        if role not in {"system", "human", "gpt"}:
            raise ValueError(f"unsupported role at item {index}: {role!r}")
        if not isinstance(value, str) or not normalize_text(value):
            raise ValueError(f"empty message at item {index}")
        normalized.append({"from": role, "value": normalize_text(value)})

    if normalized[-1]["from"] != "human":
        raise ValueError("preference prompts must end with a human message")
    return normalized


def _preference_pair(
    record: Mapping[str, Any],
    conversations: Sequence[Mapping[str, str]],
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
) -> dict[str, Any]:
    chosen_text = normalize_text(str(chosen["text"]))
    rejected_text = normalize_text(str(rejected["text"]))
    if not chosen_text or not rejected_text or chosen_text == rejected_text:
        raise ValueError("chosen and rejected answers must be non-empty and different")

    annotation = record.get("annotation") or {}
    return {
        "prompt_id": record["prompt_id"],
        "group_id": record.get("group_id", record["prompt_id"]),
        "conversations": list(conversations),
        "chosen": {"from": "gpt", "value": chosen_text},
        "rejected": {"from": "gpt", "value": rejected_text},
        "metadata": {
            "chosen_id": chosen["candidate_id"],
            "rejected_id": rejected["candidate_id"],
            "rank_gap": int(rejected["rank"]) - int(chosen["rank"]),
            "rubric_version": annotation.get("rubric_version"),
            "synthetic": bool(annotation.get("synthetic", False)),
        },
    }


def build_preference_pairs(
    records: Iterable[Mapping[str, Any]], *, adjacent_only: bool = True
) -> list[dict[str, Any]]:
    """Convert ranked candidates into pairs while skipping ties and duplicate prompts."""
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for record in records:
        prompt_id = str(record.get("prompt_id", "")).strip()
        if not prompt_id:
            raise ValueError("prompt_id is required")
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt_id: {prompt_id}")
        seen.add(prompt_id)

        conversations = validate_conversations(record.get("conversations"))
        raw_candidates = record.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise ValueError(f"{prompt_id} needs at least two candidates")

        candidates: list[dict[str, Any]] = []
        candidate_ids: set[str] = set()
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"{prompt_id} has a non-object candidate")
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError(f"{prompt_id} has a missing or duplicate candidate_id")
            candidate_ids.add(candidate_id)
            rank = candidate.get("rank")
            if not isinstance(rank, int) or rank < 1:
                raise ValueError(f"{prompt_id}/{candidate_id} needs a positive integer rank")
            if not isinstance(candidate.get("text"), str):
                raise ValueError(f"{prompt_id}/{candidate_id} needs text")
            candidates.append(candidate)

        candidates.sort(key=lambda item: (item["rank"], item["candidate_id"]))
        candidate_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        if adjacent_only:
            candidate_pairs = list(zip(candidates, candidates[1:], strict=False))
        else:
            candidate_pairs = [
                (candidates[left], candidates[right])
                for left in range(len(candidates))
                for right in range(left + 1, len(candidates))
            ]

        for chosen, rejected in candidate_pairs:
            if chosen["rank"] == rejected["rank"]:
                continue
            pairs.append(_preference_pair(record, conversations, chosen, rejected))

    return pairs


def prompt_fingerprint(conversations: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(list(conversations), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
