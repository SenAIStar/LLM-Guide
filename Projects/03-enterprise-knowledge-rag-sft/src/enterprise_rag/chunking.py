from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class DocumentSection:
    doc_id: str
    title: str
    section: str
    text: str
    version: str
    page: int | None
    department: str
    acl_groups: tuple[str, ...]


def _window_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            boundary = max(cleaned.rfind("。", start, end), cleaned.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(0, end - overlap_chars)
    return chunks


def chunk_sections(
    sections: Iterable[DocumentSection], max_chars: int = 600, overlap_chars: int = 80
) -> list[dict[str, object]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < max_chars")

    rows: list[dict[str, object]] = []
    next_chunk_index: dict[str, int] = {}
    for section in sections:
        content_hash = hashlib.sha256(section.text.encode("utf-8")).hexdigest()
        for text in _window_text(section.text, max_chars, overlap_chars):
            index = next_chunk_index.get(section.doc_id, 0)
            next_chunk_index[section.doc_id] = index + 1
            rows.append(
                {
                    "chunk_id": f"{section.doc_id}#{index:04d}",
                    "doc_id": section.doc_id,
                    "title": section.title,
                    "section": section.section,
                    "page": section.page,
                    "version": section.version,
                    "department": section.department,
                    "acl_groups": list(section.acl_groups),
                    "content_hash": content_hash,
                    "text": text,
                }
            )
    return rows
