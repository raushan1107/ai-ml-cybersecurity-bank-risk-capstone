"""PII detection and redaction before text reaches an LLM, a log, or a vendor.

Regex + validation (Luhn for card numbers) is transparent and fast. Production
systems add NER-based detectors (e.g. Microsoft Presidio, Azure AI Language PII)
for names and addresses; the placeholder design below works the same way.
"""

from __future__ import annotations

import re
from typing import Any


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# (type, regex, optional validator) — order matters: specific patterns first.
PATTERNS: list[tuple[str, re.Pattern[str], Any]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), _luhn_ok),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"), None),
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3} \d{4} \d{4}\b"), None),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    ("ACCOUNT_NO", re.compile(r"(?i)\b(?:a/?c|acct|account)(?:\s*(?:no|number|#))?\.?\s*[:#]?\s*(\d{6,17})\b"), None),
    ("PHONE", re.compile(r"(?<![\w.])(?:\+|\()?\d[\d ()-]{8,16}\d(?![\w.])"), lambda v: 10 <= len(re.sub(r"\D", "", v)) <= 13),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None),
    ("DATE_OF_BIRTH", re.compile(r"(?i)\b(?:dob|date of birth|born)\s*[:\-]?\s*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b"), None),
]


def _mask(value: str) -> str:
    v = value.strip()
    return v[:2] + "•" * max(len(v) - 4, 1) + v[-2:] if len(v) > 4 else "•" * len(v)


def detect_pii(text: str) -> list[dict[str, Any]]:
    """Return non-overlapping PII findings (type, masked value, span)."""
    findings: list[dict[str, Any]] = []
    taken: list[tuple[int, int]] = []
    for kind, rx, validator in PATTERNS:
        for m in rx.finditer(text or ""):
            start, end = m.span(1) if m.groups() and m.group(1) else m.span()
            value = text[start:end]
            if validator and not validator(value):
                continue
            if any(s < end and start < e for s, e in taken):
                continue
            taken.append((start, end))
            findings.append({"type": kind, "masked": _mask(value), "start": start, "end": end})
    return sorted(findings, key=lambda f: f["start"])


def redact(text: str) -> dict[str, Any]:
    """Replace each finding with a typed, numbered placeholder like [EMAIL_1]."""
    findings = detect_pii(text)
    counters: dict[str, int] = {}
    out, cursor = [], 0
    for f in findings:
        counters[f["type"]] = counters.get(f["type"], 0) + 1
        f["placeholder"] = f"[{f['type']}_{counters[f['type']]}]"
        out.append(text[cursor:f["start"]])
        out.append(f["placeholder"])
        cursor = f["end"]
    out.append(text[cursor:])
    return {"redacted_text": "".join(out), "findings": findings, "counts": counters,
            "leaked_if_unprotected": len(findings)}
