"""HybridRetrieveNode — inner subgraph step 1 (MFG-C2-051).

First inner node: parses the JSON payload serialized by the outer pre_process into
user_input, then performs hybrid (dense multilingual-e5 + BM25) retrieval over the
equipment-manual KB with an equipment_id metadata filter. The vector store client is
injected via the constructor (S-3: no os.environ, no secret in state).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, cast

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.retrieval_service import retrieve_passages


class HybridRetrieveNode(FunctionNode):
    """Hybrid retrieval over the manual KB."""

    # S-1: inner subgraph node — trust authenticated once at
    # the outer backbone (MaintenanceQAGraphNode); ANONYMOUS here, not a re-declaration
    # of the caller-facing requirement (INTERNAL/VERIFIED_EXTERNAL on an inner node is
    # privilege-escalation the outer GraphNode already gates).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, vector_store: Any = None, top_k: int = 6, hybrid_search: bool = True):
        super().__init__()
        self._vector_store = vector_store
        self._top_k = top_k
        self._hybrid_search = hybrid_search

    def _parse_payload(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        try:
            return cast(dict[str, Any], json.loads(raw) if raw else {})
        except (ValueError, TypeError):
            return {}

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = self._parse_payload(state.get("user_input", ""))
        query = payload.get("normalized_query") or state.get("normalized_query", "")
        equipment_id = payload.get("equipment_id") or state.get("equipment_id", "")

        if not query:
            emit_trace_event(
                "hybrid_retrieve_input_rejected",
                {
                    "equipment_id": equipment_id,
                    "correlation_id": state.get("correlation_id"),
                    "trace_id": state.get("trace_id"),
                },
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["HybridRetrieveNode: empty normalized query in inner input"],
            }

        metadata_filter = {"equipment_id": equipment_id} if equipment_id else None

        try:
            # An injected store always wins. With none wired — the standalone
            # deployment path, where config/config.yaml carries no vector_store
            # handle — retrieve_passages() falls back to the deterministic seed
            # corpus so the answer is still grounded and citable, instead of
            # returning nothing and tripping the S-3 grounding gate on every
            # request.
            passages: list[Any] = retrieve_passages(
                query,
                top_k=self._top_k,
                hybrid_search=self._hybrid_search,
                metadata_filter=metadata_filter,
                vector_store=self._vector_store,
            )
        except Exception as exc:
            # Vector-store outage must be observable + a clean ERROR, never unhandled.
            emit_trace_event(
                "kb_retrieval_failed",
                {
                    "equipment_id": equipment_id,
                    "error": str(exc),
                    "correlation_id": state.get("correlation_id"),
                    "trace_id": state.get("trace_id"),
                },
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"HybridRetrieveNode: vector store search failed: {exc}"],
            }

        emit_trace_event(
            "kb_retrieval",
            {
                "equipment_id": equipment_id,
                "hybrid": self._hybrid_search,
                "hits": len(passages),
                "correlation_id": state.get("correlation_id"),
                "trace_id": state.get("trace_id"),
            },
            state,
        )

        return {
            "normalized_query": query,
            "equipment_id": equipment_id,
            "retrieved_passages": passages,
            "status": AgentStatus.SUCCESS.value,
        }
