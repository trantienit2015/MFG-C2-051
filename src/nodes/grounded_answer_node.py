"""GroundedAnswerNode — inner subgraph step 2 (MFG-C2-051).

Generates an answer grounded strictly in the retrieved passages, emits manual+page
citations, and flags high-risk steps. The LLM client is injected via constructor
(S-3: no os.environ, no secret in state). If no passages were retrieved, the node
returns an explicit no-evidence answer (the S-3 gate downstream will mark it ungrounded).
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.grounding_service import build_citations, detect_high_risk

_PROMPT = (
    "You are an equipment-maintenance assistant. Answer the technician's question "
    "USING ONLY the retrieved manual passages below. Cite the source manual and page. "
    "If the passages do not contain the answer, say you cannot find it. Do not invent steps.\n\n"
    "Question: {query}\n\nPassages:\n{passages}\n"
)


class GroundedAnswerNode(FunctionNode):
    """Grounded answer generation with citations + risk flagging."""

    # S-1: inner subgraph node — trust authenticated once at
    # the outer backbone (MaintenanceQAGraphNode); ANONYMOUS here, not a re-declaration
    # of the caller-facing requirement (INTERNAL/VERIFIED_EXTERNAL on an inner node is
    # privilege-escalation the outer GraphNode already gates).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, llm: Any = None) -> None:
        super().__init__()
        self._llm = llm

    def _format_passages(self, passages: list[Any]) -> str:
        lines = []
        for i, p in enumerate(passages or [], 1):
            cite = f"[{p.get('manual', '?')} p.{p.get('page', '?')}]"
            lines.append(f"{i}. {cite} {p.get('text', '')}")
        return "\n".join(lines) if lines else "(no passages retrieved)"

    @staticmethod
    def _extract_text(raw: Any) -> str:
        # BaseLLM.complete() returns dict {"content": str, ...} for real clients;
        # some fakes in tests return a bare string. Normalize both, never
        # AttributeError-crash on .strip()/.splitlines() downstream (3m).
        if isinstance(raw, dict):
            content = raw.get("content", "")
            return content if isinstance(content, str) else ""
        if isinstance(raw, str):
            return raw
        return ""

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("normalized_query", "")
        passages = state.get("retrieved_passages", []) or []

        citations = build_citations(passages)
        high_risk = detect_high_risk(passages)

        if not passages:
            draft = "I could not find a grounded answer in the equipment manual KB for this query."
        elif self._llm is None:
            # No LLM wired (unit test) — produce a deterministic grounded stub from passage text.
            draft = "Based on the manual: " + " ".join(p.get("text", "") for p in passages)[:500]
        else:
            try:
                raw = self._llm.complete(_PROMPT.format(query=query, passages=self._format_passages(passages)))
                draft = self._extract_text(raw)
                if not draft:
                    # LLM returned no usable content — degrade to the deterministic
                    # grounded stub rather than let an empty answer flow downstream.
                    draft = "Based on the manual: " + " ".join(p.get("text", "") for p in passages)[:500]
            except Exception as exc:
                # LLM outage must be observable + a clean ERROR, never an unhandled exception.
                emit_trace_event(
                    "llm_call_failed",
                    {
                        "node": "GroundedAnswerNode",
                        "error": str(exc),
                        "correlation_id": state.get("correlation_id"),
                        "trace_id": state.get("trace_id"),
                    },
                    state,
                )
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": [f"GroundedAnswerNode: LLM call failed: {exc}"],
                }

        emit_trace_event(
            "answer_generated",
            {
                "high_risk": high_risk,
                "n_citations": len(citations),
                "correlation_id": state.get("correlation_id"),
                "trace_id": state.get("trace_id"),
            },
            state,
        )

        return {
            "draft_answer": draft,
            "citations": citations,
            "high_risk": high_risk,
            "status": AgentStatus.SUCCESS.value,
        }
