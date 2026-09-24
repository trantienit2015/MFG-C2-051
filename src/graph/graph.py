"""Outer graph for MFG-C2-051 — Equipment Maintenance KB & Technician Q&A Agent (Cat 2).

Backbone: initialize → pre_process (QueryNormalize) → main (MaintenanceQAGraphNode)
→ post_process (ResponseValidate) → finalize. The `main` slot wraps the inner
retrieval→grounding subgraph (Cat 2 composition).
"""

from __future__ import annotations

from framework.graph.agent_base_graph import AgentBaseGraph

from src.nodes.query_normalize_node import QueryNormalizeNode
from src.nodes.maintenance_qa_graph_node import MaintenanceQAGraphNode
from src.nodes.response_validate_node import ResponseValidateNode
from src.schemas.state import State


class EquipmentMaintenanceQAGraph(AgentBaseGraph):
    """Cat 2 outer graph for equipment maintenance Q&A."""

    @property
    def name(self) -> str:
        return "mfg-c2-051"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        super().register_nodes()  # injects initialize + finalize
        cfg = self.config if hasattr(self, "config") and self.config else {}
        llm = cfg.get("llm")
        vector_store = cfg.get("vector_store")
        retrieval_config = {
            "top_k": cfg.get("top_k", 6),
            "hybrid_search": cfg.get("hybrid_search", True),
            "score_threshold": cfg.get("score_threshold", 0.0),
        }
        self._nodes["pre_process"] = QueryNormalizeNode()
        self._nodes["main"] = MaintenanceQAGraphNode(
            llm=llm, vector_store=vector_store, retrieval_config=retrieval_config
        )
        self._nodes["post_process"] = ResponseValidateNode()


# Alias for agent.yaml `module: "src.graph"` / AgentRegistry discovery.
Graph = EquipmentMaintenanceQAGraph
