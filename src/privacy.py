"""Mask personal identifiers before text is sent to the LLM.

Credit officers paste applicant details into chat. The agent never needs a
PAN, Aadhaar number, phone number or email to answer, so they are replaced
with placeholders before any LLM call (data minimisation, in the spirit of
India's DPDP Act). Matching is regex-based and deliberately conservative.
"""
from __future__ import annotations

import re

PATTERNS = [
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b[2-9][0-9]{3}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<![\w,])(?:\+91[\s-]?|0)?[6-9][0-9]{9}\b")),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return (masked_text, {identifier_type: count})."""
    counts: dict[str, int] = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[{label} REDACTED]", text)
        if n:
            counts[label] = n
    return text, counts
