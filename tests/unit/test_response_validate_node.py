# MFG-C2-051 — Unit Tests: ResponseValidateNode (outer post_process + S-3 gate)

from framework.schemas.agent_status import AgentStatus
from src.nodes.response_validate_node import ResponseValidateNode


def _state(draft, passages, citations, high_risk=False):
    return {
        "draft_answer": draft,
        "retrieved_passages": passages,
        "citations": citations,
        "high_risk": high_risk,
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }


class TestResponseValidateNode:
    def setup_method(self):
        self.node = ResponseValidateNode()

    def test_grounded_answer_passes(self):
        passages = [{"text": "Reset E-17 by power cycling the controller.", "manual": "M-082", "page": 12}]
        result = self.node.execute(
            _state("Reset E-17 by power cycling the controller.", passages, [{"manual": "M-082", "page": 12}])
        )
        assert result["status"] == AgentStatus.SUCCESS
        assert result["grounded"] is True
        assert "Sources:" in result["result"]

    def test_s3_blocks_ungrounded_answer(self):
        """S-3: answer with no token overlap with passages is blocked."""
        passages = [{"text": "Reset E-17 by power cycling.", "manual": "M-082", "page": 12}]
        result = self.node.execute(
            _state("Replace the entire hydraulic pump assembly immediately.", passages,
                   [{"manual": "M-082", "page": 12}])
        )
        assert result["status"] == AgentStatus.ERROR
        assert "S-3" in result["error_log"][0]

    def test_s3_blocks_missing_citation(self):
        passages = [{"text": "Reset E-17 by power cycling.", "manual": "M-082", "page": 12}]
        result = self.node.execute(_state("Reset E-17 by power cycling.", passages, []))
        assert result["status"] == AgentStatus.ERROR
        assert result["grounded"] is False

    def test_high_risk_appends_safety_note(self):
        passages = [{"text": "Lockout high voltage before reset E-17.", "manual": "M-082", "page": 3}]
        result = self.node.execute(
            _state("Lockout high voltage before reset E-17.", passages,
                   [{"manual": "M-082", "page": 3}], high_risk=True)
        )
        assert result["status"] == AgentStatus.SUCCESS
        assert "SAFETY ESCALATION" in result["result"]
