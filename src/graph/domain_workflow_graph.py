"""Inner domain workflow graph for MFG-C2-051 (Cat 2).

Topology: START → hybrid_retrieve → grounded_answer → END.
Instantiated by MaintenanceQAGraphNode.get_subgraph(). Receives only the
`user_input` string (the JSON payload serialized by the outer pre_process) plus
InvocationContext; the first node parses the payload back.
"""

from __future__ import annotations

from typing import Any
from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.nodes.grounded_answer_node import GroundedAnswerNode
from src.schemas.state import State


class MaintenanceWorkflowGraph(BaseGraph):
    """Retrieval → grounded-answer inner workflow."""

    @property
    def name(self) -> str:
        return "maintenance_qa_workflow"

    @property
    def state_schema(self) -> type:
        return State

    def _validate_config(self) -> None:
        pass

    def register_nodes(self) -> None:
        cfg = self.config if hasattr(self, "config") and self.config else {}
        llm = cfg.get("llm")
        vector_store = cfg.get("vector_store")
        rc = cfg.get("retrieval_config", {}) or {}
        self._nodes["hybrid_retrieve"] = HybridRetrieveNode(
            vector_store=vector_store,
            top_k=int(rc.get("top_k", 6)),
            hybrid_search=bool(rc.get("hybrid_search", True)),
        )
        self._nodes["grounded_answer"] = GroundedAnswerNode(llm=llm)

    def add_edges(self) -> None:
        self._sg.add_edge(START, "hybrid_retrieve")
        self._sg.add_edge("hybrid_retrieve", "grounded_answer")
        self._sg.add_edge("grounded_answer", END)

    def route(self, state: AgentState) -> str:
        # Required by ABC; linear topology means this is never invoked. Nodes set status to
        # AgentStatus.<X>.value (criterion #15), so compare against the string value.
        return END if state.get("status") == AgentStatus.ERROR.value else "grounded_answer"

    def get_output(self, state: AgentState) -> dict[str, Any]:
        return {
            "retrieved_passages": state.get("retrieved_passages", []),
            "draft_answer": state.get("draft_answer", ""),
            "citations": state.get("citations", []),
            "high_risk": bool(state.get("high_risk", False)),
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
