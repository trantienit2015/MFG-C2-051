# MFG-C2-051 - Framework Compliance Tests (TC-01..08)
#
# Adapted from sibling templates (same shape: FunctionNode outer pre/post_process +
# FunctionNode inner subgraph nodes; Cat 2 outer `main` slot is a GraphNode, which is
# NOT a FunctionNode and is covered separately by
# tests/proof_of_boundary/test_pb_graphnode_boundary.py, not this file). This repo
# does not override _extra_security_gate_input/output() on any node (optional no-op
# by default at S-2/S-3), so TC-07 only verifies the @final gate contract.

import inspect

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.query_normalize_node import QueryNormalizeNode
from src.nodes.response_validate_node import ResponseValidateNode
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.nodes.grounded_answer_node import GroundedAnswerNode
from src.schemas.state import State

_OUTER_NODES = (QueryNormalizeNode, ResponseValidateNode)
_INNER_NODES = (HybridRetrieveNode, GroundedAnswerNode)
_ALL_FUNCTION_NODES = _OUTER_NODES + _INNER_NODES


class TestTC01StateContract:
    """TC-01: State is a flat TypedDict extending AgentState (no Pydantic/dataclass)."""

    def test_state_extends_agent_state(self):
        # TypedDict class-based inheritance loses the parent from __mro__
        # (issubclass() raises TypeError for TypedDicts), so verify structurally:
        # every AgentState field must be present in State's annotations.
        assert set(AgentState.__annotations__).issubset(set(State.__annotations__))

    def test_state_fields_json_serialisable_shape(self):
        annotations = State.__annotations__
        assert "normalized_query" in annotations
        assert "retrieved_passages" in annotations
        assert "citations" in annotations


class TestTC02FailClosedValidation:
    """TC-02: invalid/empty input -> ERROR, never raise."""

    def test_empty_state_returns_error_not_raise(self):
        node = QueryNormalizeNode()
        result = node.execute({})
        assert result["status"] == AgentStatus.ERROR.value


class TestTC03NoCredentials:
    """TC-03: no os.environ / hardcoded credentials in node source (see gate-credential-scan)."""

    def test_no_os_environ_in_node_modules(self):
        for mod in _ALL_FUNCTION_NODES:
            src = inspect.getsource(mod)
            assert "os.environ" not in src


class TestTC04InvocationContextIsolation:
    """TC-04: nodes never receive InvocationContext directly in execute(state)."""

    def test_execute_signature_is_state_only(self):
        for mod in _ALL_FUNCTION_NODES:
            sig = inspect.signature(mod.execute)
            params = list(sig.parameters.keys())
            assert params == ["self", "state"]


class TestTC05AuditLogging:
    """TC-05: emit_trace_event called for domain events (no re-emit of framework lifecycle events)."""

    def test_nodes_import_emit_trace_event(self):
        for mod in _ALL_FUNCTION_NODES:
            src = inspect.getsource(inspect.getmodule(mod))
            assert "emit_trace_event" in src


class TestTC06S2GateNonBypassable:
    """TC-06: _security_gate_input() is @final — no node overrides it directly."""

    def test_no_node_overrides_final_input_gate(self):
        for mod in _ALL_FUNCTION_NODES:
            src = inspect.getsource(mod)
            assert "def _security_gate_input(" not in src


class TestTC07S3GateHook:
    """TC-07: _security_gate_output() is @final (no node overrides it directly)."""

    def test_no_node_overrides_final_output_gate(self):
        for mod in _ALL_FUNCTION_NODES:
            src = inspect.getsource(mod)
            assert "def _security_gate_output(" not in src


class TestTC08TrustGateDeclaredAndEnforced:
    """TC-08: required_trust_level declared as a valid enum on every FunctionNode."""

    def test_all_nodes_declare_valid_trust_level(self):
        for mod in _ALL_FUNCTION_NODES:
            assert issubclass(mod, FunctionNode)
            assert mod.required_trust_level in (
                TrustLevel.ANONYMOUS,
                TrustLevel.VERIFIED_EXTERNAL,
                TrustLevel.INTERNAL,
            )

    def test_outer_nodes_match_agent_yaml_inner_nodes_are_anonymous(self):
        # Outer pre/post_process nodes match config/agent.yaml
        # (required_trust_level: VERIFIED_EXTERNAL). Inner subgraph nodes are
        # ANONYMOUS — trust is authenticated once at the outer GraphNode backbone.
        assert QueryNormalizeNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL
        assert ResponseValidateNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL
        assert HybridRetrieveNode.required_trust_level == TrustLevel.ANONYMOUS
        assert GroundedAnswerNode.required_trust_level == TrustLevel.ANONYMOUS
