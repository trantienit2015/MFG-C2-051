"""Deterministic query-normalization helpers for MFG maintenance Q&A.

Pure functions — no agenticstar imports, no side effects. Unit-testable in isolation.
Handles noisy manufacturing-floor shorthand: equipment IDs (e.g. M-082),
fault codes (e.g. E-17 / E17), and mixed JA/EN text.
"""

from __future__ import annotations

from typing import Any
import re

# Equipment ID: a letter group, a separator, then digits (M-082, MC082, ライン-3 not matched here).
_EQUIPMENT_RE = re.compile(r"\b([A-Za-z]{1,4})[-_ ]?(\d{1,5})\b")
# Fault code: E-17, E17, ERR-203, ALM12 ...
_FAULT_RE = re.compile(r"\b([A-Za-z]{1,4})[-_ ]?(\d{1,4})\b")
_FAULT_PREFIXES = ("E", "ERR", "ALM", "AL", "F", "FLT")

# A char is "CJK" (Japanese/Chinese) if in these unicode ranges.
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")


def normalize_text(text: str) -> str:
    """Collapse whitespace and trim. Keeps original casing for IDs/codes."""
    return re.sub(r"\s+", " ", text or "").strip()


def detect_language(text: str) -> str:
    """Return 'ja' if any CJK chars present, else 'en' (best-effort heuristic)."""
    return "ja" if _CJK_RE.search(text or "") else "en"


def extract_equipment_id(text: str) -> str:
    """Extract the first equipment-id-shaped token, normalized to 'PREFIX-digits'.

    Returns "" when none found. Skips tokens whose prefix looks like a fault code.
    """
    for m in _EQUIPMENT_RE.finditer(text or ""):
        prefix, num = m.group(1).upper(), m.group(2)
        if prefix in _FAULT_PREFIXES:
            continue
        return f"{prefix}-{num}"
    return ""


def extract_fault_code(text: str) -> str:
    """Extract the first fault-code-shaped token, normalized to 'PREFIX-digits'.

    Returns "" when none found.
    """
    for m in _FAULT_RE.finditer(text or ""):
        prefix, num = m.group(1).upper(), m.group(2)
        if prefix in _FAULT_PREFIXES:
            return f"{prefix}-{num}"
    return ""


def normalize_query(text: str) -> dict[str, Any]:
    """Normalize a raw technician query into structured fields.

    Returns {normalized_query, equipment_id, fault_code, language}.
    """
    norm = normalize_text(text)
    return {
        "normalized_query": norm,
        "equipment_id": extract_equipment_id(norm),
        "fault_code": extract_fault_code(norm),
        "language": detect_language(norm),
    }
