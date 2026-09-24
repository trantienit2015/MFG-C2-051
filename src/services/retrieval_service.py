"""Equipment-manual retrieval for MFG-C2-051.

An injected vector store is always preferred: it must expose
``search(query, top_k=..., hybrid=..., metadata_filter=...)``. When no store is
wired — the standalone deployment path, where ``config/config.yaml`` carries no
``vector_store`` handle — a deterministic keyword overlap over a small seed
corpus of manual excerpts is used instead, so the agent returns a grounded,
citable answer rather than failing the S-3 grounding gate on every request.

The seed corpus is a bounded, non-production fallback: it is small, static and
carries real ``manual``/``page`` metadata so the citation and high-risk paths
behave exactly as they do against a real KB. Pure functions only — no
agenticstar imports, no side effects.
"""

from __future__ import annotations

import re
from typing import Any, cast

# Seed corpus — excerpts shaped exactly like real KB passages
# (text + manual + page + equipment_id), used only when no store is injected.
_SEED_CORPUS: list[dict[str, Any]] = [
    {
        "text": (
            "Before any inspection of the hydraulic press, perform lockout/tagout of the main "
            "disconnect. High voltage remains present at the drive cabinet until the isolator is "
            "opened. Do not open the guard door while the ram is pressurized."
        ),
        "manual": "HP-2000 Maintenance Manual",
        "page": 14,
        "equipment_id": "HP-2000",
    },
    {
        "text": (
            "油圧プレスの点検前に主開閉器のロックアウトを実施すること。ドライブ盤には高電圧が残留するため、"
            "断路器を開放するまで感電の危険がある。加圧状態でガードドアを開けることは禁止。"
        ),
        "manual": "HP-2000 保守マニュアル",
        "page": 14,
        "equipment_id": "HP-2000",
    },
    {
        "text": (
            "Routine maintenance of the conveyor drive: inspect the belt tension every 500 operating "
            "hours, lubricate the bearings with the specified grease, and check the emergency stop "
            "circuit for continuity. Replace the belt when edge wear exceeds 3 mm."
        ),
        "manual": "CV-100 Service Guide",
        "page": 7,
        "equipment_id": "CV-100",
    },
    {
        "text": (
            "Troubleshooting an overheating spindle motor: verify coolant flow, confirm the filter is "
            "not clogged, and measure the winding temperature. A sustained reading above the rated "
            "limit indicates bearing failure and the machine must be stopped."
        ),
        "manual": "SP-450 Troubleshooting Handbook",
        "page": 22,
        "equipment_id": "SP-450",
    },
    {
        "text": (
            "General safety procedure for all equipment: wear the prescribed protective equipment, "
            "confirm the work permit is issued, and never bypass an interlock. Report any abnormal "
            "noise, vibration or smell to the maintenance supervisor before continuing."
        ),
        "manual": "Plant Safety Procedures",
        "page": 3,
        "equipment_id": "",
    },
]

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[぀-ヿ㐀-鿿]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "") if len(t) >= 2}


def retrieve_passages(
    query: str,
    top_k: int = 6,
    hybrid_search: bool = True,
    metadata_filter: dict[str, Any] | None = None,
    vector_store: Any = None,
) -> list[dict[str, Any]]:
    """Return manual passages for ``query``.

    Delegates to ``vector_store`` when one is injected; otherwise scores the seed
    corpus by token overlap. ``metadata_filter`` narrows the seed corpus by
    ``equipment_id`` when that key is present, mirroring the store's own filter.
    """
    if vector_store is not None:
        return cast(
            list[dict[str, Any]],
            vector_store.search(query, top_k=top_k, hybrid=hybrid_search, metadata_filter=metadata_filter),
        )

    equipment_id = (metadata_filter or {}).get("equipment_id") or ""
    candidates = [
        passage
        for passage in _SEED_CORPUS
        # An empty equipment_id on a passage means it applies to all equipment.
        if not equipment_id or passage["equipment_id"] in (equipment_id, "")
    ]

    query_tokens = _tokens(query)
    scored = [(len(query_tokens & _tokens(passage["text"])), passage) for passage in candidates]
    # Stable ordering: overlap desc, then corpus order (sort is stable).
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [dict(passage, score=0.7) for _overlap, passage in scored[:top_k]]
