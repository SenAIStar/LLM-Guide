from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from typing import Any, Iterable

from .pii import PiiRedactor


SYSTEM_PROMPT = (
    "你是法律领域助手。根据问题和已提供材料给出清晰、审慎的回答；"
    "信息不足时说明缺少什么，不要虚构法条、案例或事实，也不要把回答表述为律师意见。"
)


def normalize_text(text: str) -> str:
    """Normalize characters and whitespace before hashing or deduplication."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def convert_disc_pair(item: dict[str, Any], redactor: PiiRedactor) -> dict[str, Any]:
    """Convert a DISC-Law-SFT Pair record to OpenAI-style messages."""
    question = redactor.redact(normalize_text(str(item["input"]))).text
    answer = redactor.redact(normalize_text(str(item["output"]))).text
    record_id = str(item.get("id") or _short_hash(question + answer))
    return {
        "id": f"pair-{record_id}",
        "group_id": _group_id(item, question),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "references": [],
        "metadata": {"source_dataset": "DISC-Law-SFT", "source_format": "Pair"},
    }


def convert_disc_triplet(item: dict[str, Any], redactor: PiiRedactor) -> dict[str, Any]:
    """Convert a Triplet record and keep its reference for data auditing."""
    record = convert_disc_pair(item, redactor)
    references = item.get("reference", [])
    if isinstance(references, str):
        references = [references]
    cleaned_references = []
    for reference in references:
        text = redactor.redact(normalize_text(str(reference))).text
        if text:
            cleaned_references.append(text)
    record["id"] = record["id"].replace("pair-", "triplet-", 1)
    record["metadata"]["references"] = cleaned_references
    record["metadata"]["source_format"] = "Triplet"
    return record


def deduplicate(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Exact deduplication after normalization; near-duplicate review remains a later step."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicate_count = 0
    for record in records:
        fingerprint = _content_fingerprint(record)
        if fingerprint in seen:
            duplicate_count += 1
            continue
        seen.add(fingerprint)
        kept.append(record)
    return kept, duplicate_count


def split_by_group(
    records: Iterable[dict[str, Any]],
    *,
    seed: str = "legal-qa-v1",
) -> dict[str, list[dict[str, Any]]]:
    """Stable 80/10/10 split in which one case/group cannot cross subsets."""
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    group_to_split: dict[str, str] = {}
    for record in records:
        group_id = str(record["group_id"])
        if group_id not in group_to_split:
            bucket = int(_short_hash(f"{seed}:{group_id}", length=8), 16) % 100
            group_to_split[group_id] = "train" if bucket < 80 else "dev" if bucket < 90 else "test"
        splits[group_to_split[group_id]].append(record)
    return splits


def quality_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    formats = Counter(
        record.get("metadata", {}).get("source_format", "unknown") for record in records
    )
    return {
        "records": len(records),
        "source_formats": dict(formats),
        "with_reference": sum(
            bool(record.get("metadata", {}).get("references")) for record in records
        ),
        "without_reference": sum(
            not record.get("metadata", {}).get("references") for record in records
        ),
        "groups": len({record.get("group_id") for record in records}),
    }


def _group_id(item: dict[str, Any], question: str) -> str:
    # Prefer a source case/group identifier. Falling back to normalized question
    # prevents exact paraphrases from being split independently.
    source_group = item.get("group_id") or item.get("case_id")
    return str(source_group) if source_group else f"question-{_short_hash(question)}"


def _content_fingerprint(record: dict[str, Any]) -> str:
    messages = record.get("messages", [])
    normalized = "\n".join(
        f"{message.get('role', '')}:{normalize_text(str(message.get('content', '')))}"
        for message in messages
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _short_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
