"""MaintenanceQAGraphNode — the `main` slot GraphNode (MFG-C2-051, Cat 2).

Wraps the inner retrieval→grounding subgraph. Forwards LLM / vector-store / retrieval
config to the inner graph via _parent_config(); the inner graph only receives the
`validated_input` string (per GraphNode contract), so the normalized payload is parsed
back inside the first inner node.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event


class MaintenanceQAGraphNode(GraphNode):
    """Cat 2 main slot — wraps the equipment-maintenance domain workflow."""

    # S-1: outer main-slot wrapper — first node in the
    # outer backbone receiving caller input; matches agent.yaml required_trust_level.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL
    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def __init__(self, llm: Any = None, vector_store: Any = None, retrieval_config: dict[str, Any] | None = None):
        super().__init__()
        self._llm = llm
        self._vector_store = vector_store
        self._retrieval_config = retrieval_config or {}

    def get_subgraph(self) -> Any:
        from src.graph.domain_workflow_graph import MaintenanceWorkflowGraph

        sg = MaintenanceWorkflowGraph(config=self._parent_config())
        sg.compile()
        return sg

    def extract_input(self, state: AgentState) -> str:
        # S-4: runs inside GraphNode.execute() — audit the dispatch into the QA subgraph.
        emit_trace_event(
            "maintenance_qa_workflow_dispatched",
            {
                "equipment_id": state.get("equipment_id", ""),
                "language": state.get("language", ""),
            },
            state,
        )
        return cast(str, state.get("validated_input", state.get("user_input", "")))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        # S-4: runs inside GraphNode.execute() — audit the subgraph outcome merged back out.
        emit_trace_event(
            "maintenance_qa_workflow_completed",
            {
                "n_passages": len(sub_result.get("retrieved_passages", [])),
                "n_citations": len(sub_result.get("citations", [])),
                "high_risk": bool(sub_result.get("high_risk", False)),
            },
            state,
        )
        return {
            "retrieved_passages": sub_result.get("retrieved_passages", []),
            "draft_answer": sub_result.get("draft_answer", ""),
            "citations": sub_result.get("citations", []),
            "high_risk": bool(sub_result.get("high_risk", False)),
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> dict[str, Any]:
        return {
            "llm": self._llm,
            "vector_store": self._vector_store,
            "retrieval_config": self._retrieval_config,
        }
