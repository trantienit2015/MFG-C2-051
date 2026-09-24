# MFG-C2-051 — Unit Tests: GroundedAnswerNode (inner step 2)

from framework.schemas.agent_status import AgentStatus
from src.nodes.grounded_answer_node import GroundedAnswerNode


class FakeLLM:
    def __init__(self, text="To reset E-17, power cycle the controller per the manual."):
        self._text = text

    def complete(self, prompt):
        return self._text


def _state(passages):
    return {
        "normalized_query": "reset E-17",
        "retrieved_passages": passages,
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }


class TestGroundedAnswerNode:
    def test_success_with_citations(self):
        passages = [{"text": "Reset E-17 by power cycling.", "manual": "M-082 Manual", "page": 12}]
        node = GroundedAnswerNode(llm=FakeLLM())
        result = node.execute(_state(passages))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["draft_answer"]
        assert result["citations"] == [{"manual": "M-082 Manual", "page": 12}]
        assert result["high_risk"] is False

    def test_high_risk_flagged(self):
        passages = [{"text": "WARNING high voltage — lockout before reset.", "manual": "M-082", "page": 3}]
        node = GroundedAnswerNode(llm=FakeLLM())
        result = node.execute(_state(passages))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["high_risk"] is True

    def test_no_passages_returns_no_evidence_answer(self):
        node = GroundedAnswerNode(llm=FakeLLM())
        result = node.execute(_state([]))
        assert result["status"] == AgentStatus.SUCCESS
        assert "could not find" in result["draft_answer"].lower()
        assert result["citations"] == []

    def test_llm_failure_returns_error_not_unhandled(self):
        class BoomLLM:
            def complete(self, prompt):
                raise RuntimeError("LLM provider outage")

        passages = [{"text": "Reset E-17 by power cycling.", "manual": "M-082 Manual", "page": 12}]
        node = GroundedAnswerNode(llm=BoomLLM())
        result = node.execute(_state(passages))
        assert result["status"] == AgentStatus.ERROR
        assert "LLM call failed" in result["error_log"][0]

    def test_dict_llm_response_normalized_not_crashed(self):
        # Canonical BaseLLM.complete() shape: {"content": str, ...} — must not
        # AttributeError when downstream does draft.strip() (recipe 3m).
        class DictLLM:
            def complete(self, prompt):
                return {"content": "To reset E-17, power cycle the controller.", "model": "fake"}

        passages = [{"text": "Reset E-17 by power cycling.", "manual": "M-082 Manual", "page": 12}]
        node = GroundedAnswerNode(llm=DictLLM())
        result = node.execute(_state(passages))
        assert result["status"] == AgentStatus.SUCCESS
        assert isinstance(result["draft_answer"], str)
        assert "power cycle" in result["draft_answer"]

    def test_dict_llm_missing_content_degrades_to_deterministic_fallback(self):
        # Dict response with no usable "content" must not let an empty answer
        # flow downstream — degrade to the deterministic grounded stub instead.
        class BadLLM:
            def complete(self, prompt):
                return {"model": "fake"}  # no "content" key

        passages = [{"text": "Reset E-17 by power cycling.", "manual": "M-082 Manual", "page": 12}]
        node = GroundedAnswerNode(llm=BadLLM())
        result = node.execute(_state(passages))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["draft_answer"].startswith("Based on the manual:")
