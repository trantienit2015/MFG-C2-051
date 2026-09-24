# MFG-C2-051 — Unit Tests: QueryNormalizeNode (outer pre_process)

import json

from framework.schemas.agent_status import AgentStatus
from src.nodes.query_normalize_node import QueryNormalizeNode


class TestQueryNormalizeNode:
    def setup_method(self):
        self.node = QueryNormalizeNode()

    def _state(self, ui):
        return {"user_input": ui, "node_history": [], "error_log": [], "execution_time": {}}

    def test_success_extracts_equipment_and_fault(self):
        result = self.node.execute(
            self._state("Machine M-082 fault code E-17, what's the reset procedure?")
        )
        assert result["status"] == AgentStatus.SUCCESS
        assert result["equipment_id"] == "M-082"
        assert result["fault_code"] == "E-17"
        assert result["language"] == "en"
        # validated_input round-trips the normalized payload as JSON
        payload = json.loads(result["validated_input"])
        assert payload["equipment_id"] == "M-082"

    def test_detects_japanese(self):
        result = self.node.execute(self._state("設備 M-082 のリセット手順は？"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["language"] == "ja"
        assert result["equipment_id"] == "M-082"

    def test_empty_input_error(self):
        result = self.node.execute(self._state("   "))
        assert result["status"] == AgentStatus.ERROR
        assert "empty" in result["error_log"][0].lower()
