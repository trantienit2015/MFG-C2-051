"""Deterministic grounding / citation helpers for MFG maintenance Q&A.

Pure functions — no agenticstar imports, no side effects. The S-3 grounding gate
is deterministic (NOT an LLM self-check): an answer is "grounded" only if it shares
meaningful token overlap with at least one retrieved passage AND carries a citation.
"""

from __future__ import annotations

from typing import Any
import re

# Manual phrases that mark a high-risk / safety-critical maintenance step.
_HIGH_RISK_MARKERS = (
    "high voltage",
    "高電圧",
    "lockout",
    "ロックアウト",
    "感電",
    "electric shock",
    "explos",
    "爆発",
    "高温",
    "burn",
    "やけど",
    "pressuriz",
    "加圧",
    "危険",
    "danger",
    "lethal",
    "致命",
    "do not",
    "禁止",
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[぀-ヿ㐀-鿿]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "") if len(t) >= 2}


def build_citations(passages: list[Any]) -> list[Any]:
    """Extract unique (manual, page) citations from retrieved passages, order-preserved."""
    seen: set[tuple[Any, ...]] = set()
    out: list[Any] = []
    for p in passages or []:
        manual = p.get("manual")
        page = p.get("page")
        if manual is None:
            continue
        key = (manual, page)
        if key in seen:
            continue
        seen.add(key)
        out.append({"manual": manual, "page": page})
    return out


def detect_high_risk(passages: list[Any]) -> bool:
    """True if any retrieved passage text contains a high-risk safety marker."""
    for p in passages or []:
        text = (p.get("text") or "").lower()
        if any(marker in text for marker in _HIGH_RISK_MARKERS):
            return True
    return False


def is_grounded(answer: str, passages: list[Any]) -> bool:
    """Deterministic grounding check.

    The answer is grounded when it shares >= 2 content tokens with the union of
    retrieved passage texts. Empty answer or empty passages => not grounded.
    """
    if not answer or not answer.strip() or not passages:
        return False
    answer_tokens = _tokens(answer)
    passage_tokens: set[str] = set()
    for p in passages:
        passage_tokens |= _tokens(p.get("text", ""))
    return len(answer_tokens & passage_tokens) >= 2


SAFETY_NOTE = (
    "⚠️ SAFETY ESCALATION: This procedure involves a step the manual flags as high-risk. "
    "Verify lockout/PPE requirements and consult a qualified supervisor before proceeding."
)
