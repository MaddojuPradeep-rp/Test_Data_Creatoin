"""PII detection utilities for conversation data.

Provides pattern-based detection of personally identifiable information
(PII) in text, including phone numbers, emails, SSNs, and credit card numbers.
Used to flag sensitive data in conversation logs before processing.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# PII detection patterns
_PATTERNS = {
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"
    ),
    "date_of_birth": re.compile(
        r"\b(?:0[1-9]|1[0-2])[/.-](?:0[1-9]|[12]\d|3[01])[/.-](?:19|20)\d{2}\b"
    ),
}


def detect_pii(text: str) -> List[Tuple[str, str]]:
    """Detect PII patterns in text.

    Returns a list of (pii_type, matched_text) tuples.
    """
    findings: List[Tuple[str, str]] = []
    for pii_type, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append((pii_type, match.group()))
    return findings


def has_pii(text: str) -> bool:
    """Return True if any PII pattern is detected in the text."""
    return bool(detect_pii(text))


def mask_pii(text: str) -> str:
    """Replace detected PII with [REDACTED] markers."""
    result = text
    for pii_type, pattern in _PATTERNS.items():
        result = pattern.sub(f"[{pii_type.upper()}_REDACTED]", result)
    return result
