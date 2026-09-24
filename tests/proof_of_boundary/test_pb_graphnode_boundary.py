# MFG-C2-051 - GraphNode boundary test (Cat 2 outer main-slot wrapper).
#
# Why this test exists: PB-6 (test_pb_invoke_order.py) only self-discovers
# BaseNode subclasses under src/nodes/. MaintenanceQAGraphNode lives under
# src/nodes/maintenance_qa_graph_node.py and is wired into the outer graph's
# `main` slot (src/graph/graph.py) - it is a real security boundary (first
# node in the outer backbone to receive the caller's input) that no PB-6
# probe reaches, since PB-6 only checks the framework FunctionNode lifecycle,
# not GraphNode's own dispatch-into-subgraph behaviour.
#
# framework/nodes/graph_node.py: GraphNode extends BaseNode directly (not
# FunctionNode), so it has no _security_gate_input/_security_gate_output at
# all - S-2/S-3 gating is delegated entirely to the inner subgraph's own
# FunctionNode chain (HybridRetrieveNode -> GroundedAnswerNode). This test
# proves that delegation is real, not absent.

from framework.nodes.function_node import FunctionNode
from framework.schemas.trust_level import TrustLevel

from src.nodes.maintenance_qa_graph_node import MaintenanceQAGraphNode
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode


def _node():
    return MaintenanceQAGraphNode()


class TestGraphNodeS1TrustGate:
    """S-1: the outer main-slot GraphNode enforces the trust gate like any BaseNode."""

    def test_insufficient_trust_returns_error_without_invoking_subgraph(self, monkeypatch):
        node = _node()

        def _spy_get_subgraph():
            raise AssertionError("get_subgraph() must not run when the S-1 gate denies")

        monkeypatch.setattr(node, "get_subgraph", _spy_get_subgraph)

        state = {
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "user_input": "how do I reset the conveyor?",
        }
        out = node(state)

        assert out["status"] == "error"
        assert any("S-1 trust gate denied" in e for e in out["error_log"])

    def test_matches_agent_yaml_required_trust_level(self):
        # The outer main-slot wrapper must match config/agent.yaml
        # (VERIFIED_EXTERNAL), not the inner subgraph's ANONYMOUS default.
        assert MaintenanceQAGraphNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL


class TestGraphNodeBoundaryMapping:
    """Boundary mapping: extract_input()/merge_output() do not leak raw state/subgraph dicts."""

    def test_extract_input_only_forwards_the_validated_payload(self):
        node = _node()
        state = {
            "validated_input": '{"normalized_query": "reset conveyor", "equipment_id": "CNV-01"}',
            "user_input": "raw caller text should not be used when validated_input is present",
            "unrelated_secret_field": "must-not-appear",
        }
        extracted = node.extract_input(state)

        assert isinstance(extracted, str)
        assert extracted == state["validated_input"]
        assert "unrelated_secret_field" not in extracted
        assert "must-not-appear" not in extracted

    def test_merge_output_maps_fields_explicitly_no_raw_passthrough(self):
        node = _node()
        state = {}
        sub_result = {
            "retrieved_passages": [{"text": "x", "manual": "m", "page": 1}],
            "draft_answer": "Based on the manual: x",
            "citations": [{"manual": "m", "page": 1}],
            "high_risk": False,
            "status": "success",
            # A field the subgraph might carry internally that must NOT leak
            # into the outer state unless merge_output() explicitly maps it.
            "internal_debug_trace": "should-not-be-copied",
        }
        merged = node.merge_output(state, sub_result)

        assert "internal_debug_trace" not in merged
        assert merged["status"] == "success"
        assert set(merged.keys()) == {
            "retrieved_passages",
            "draft_answer",
            "citations",
            "high_risk",
            "status",
        }


class TestGraphNodeDelegatesGatingToInnerSubgraph:
    """Delegation has a real target: the inner subgraph's entry node runs S-1/S-2/S-3."""

    def test_inner_entry_node_is_a_function_node_with_security_gates(self):
        # HybridRetrieveNode is the inner subgraph's entry point
        # (domain_workflow_graph.py: START -> hybrid_retrieve). It is a
        # FunctionNode, so the framework's @final S-2/S-3 gates run on every
        # invocation of the inner subgraph - this is where the GraphNode's
        # skipped lifecycle is actually enforced, not omitted.
        assert issubclass(HybridRetrieveNode, FunctionNode)
        assert hasattr(HybridRetrieveNode, "required_trust_level")
