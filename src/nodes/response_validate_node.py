"""ResponseValidateNode — outer post_process slot (MFG-C2-051).

S-3 deterministic output gate: blocks any answer not grounded in retrieved evidence
or missing a citation (returns ERROR — does NOT emit the ungrounded answer), and
appends a safety-escalation note when a high-risk step was flagged.

This S-3 domain gate is implemented as a deterministic grounding check inside
`execute()` (business validation, not the S-2/S-3 `_extra_*` hooks — this node does
not override them) — it is deterministic, not an LLM self-check.
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.grounding_service import SAFETY_NOTE, is_grounded


class ResponseValidateNode(FunctionNode):
    """Validate grounding + citations; assemble final answer (outer post_process)."""

    # S-1: explicit by design. Final S-3 grounding gate of the Q&A
    # pipeline — verified caller required (matches agent.yaml default).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        draft = state.get("draft_answer", "")
        passages = state.get("retrieved_passages", []) or []
        citations = state.get("citations", []) or []
        high_risk = bool(state.get("high_risk", False))

        grounded = is_grounded(draft, passages) and len(citations) > 0

        if not grounded:
            emit_trace_event(
                "ungrounded_answer_blocked",
                {
                    "n_passages": len(passages),
                    "n_citations": len(citations),
                    "correlation_id": state.get("correlation_id"),
                    "trace_id": state.get("trace_id"),
                },
                state,
            )
            return {
                "grounded": False,
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "ResponseValidateNode: answer is not grounded in retrieved evidence "
                    "or missing citation — blocked by S-3 gate"
                ],
            }

        safety_note = SAFETY_NOTE if high_risk else ""
        citation_str = "; ".join(f"{c['manual']} p.{c['page']}" for c in citations)
        final = draft.strip()
        if safety_note:
            final = f"{final}\n\n{safety_note}"
        final = f"{final}\n\nSources: {citation_str}"

        emit_trace_event(
            "answer_validated",
            {
                "high_risk": high_risk,
                "correlation_id": state.get("correlation_id"),
                "trace_id": state.get("trace_id"),
            },
            state,
        )

        return {
            "grounded": True,
            "safety_note": safety_note,
            "result": final,
            "formatted_output": final,
            "status": AgentStatus.SUCCESS.value,
        }
