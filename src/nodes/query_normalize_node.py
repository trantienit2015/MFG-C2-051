"""QueryNormalizeNode — outer pre_process slot (MFG-C2-051).

Normalizes noisy floor shorthand, extracts equipment_id / fault_code, detects JA/EN,
rejects empty/invalid input, and serializes the normalized payload into
`validated_input` (JSON string) for the inner Cat 2 subgraph.
"""

from __future__ import annotations

import json

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.query_service import normalize_query


class QueryNormalizeNode(FunctionNode):
    """Validate + normalize the technician query (outer pre_process)."""

    # S-1: explicit by design. Entry of the technician Q&A pipeline —
    # verified caller required (matches agent.yaml default).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        if not user_input or not user_input.strip():
            emit_trace_event(
                "query_normalize_input_rejected",
                {"correlation_id": state.get("correlation_id"), "trace_id": state.get("trace_id")},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["QueryNormalizeNode: user_input is empty or missing"],
            }

        norm = normalize_query(user_input)
        # Serialize the structured payload for the inner subgraph (GraphNode passes a string).
        validated_input = json.dumps(norm, ensure_ascii=False)

        # S-4: audit the normalization outcome — extracted signals only, not the raw query.
        emit_trace_event(
            "query_normalized",
            {
                "language": norm["language"],
                "has_equipment_id": bool(norm["equipment_id"]),
                "has_fault_code": bool(norm["fault_code"]),
            },
            state,
        )

        return {
            "normalized_query": norm["normalized_query"],
            "equipment_id": norm["equipment_id"],
            "fault_code": norm["fault_code"],
            "language": norm["language"],
            "validated_input": validated_input,
            "status": AgentStatus.SUCCESS.value,
        }
