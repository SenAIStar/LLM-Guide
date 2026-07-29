from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int]


class PiiRedactor:
    """Conservative regex redaction for obvious identifiers.

    This is a first pass, not a replacement for NER and human review.
    """

    PATTERNS = (
        ("email", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)),
        ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
        ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\w)")),
        ("bank_card", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    )

    def redact(self, text: str, *, sensitive_terms: list[str] | None = None) -> RedactionResult:
        redacted = text
        counts: Counter[str] = Counter()
        for name, pattern in self.PATTERNS:
            redacted, count = pattern.subn(f"<{name.upper()}>", redacted)
            counts[name] += count

        for term in sorted(set(sensitive_terms or []), key=len, reverse=True):
            term = term.strip()
            if not term:
                continue
            count = redacted.count(term)
            if count:
                redacted = redacted.replace(term, "<SENSITIVE_TERM>")
                counts["sensitive_term"] += count
        return RedactionResult(text=redacted, counts=dict(counts))
