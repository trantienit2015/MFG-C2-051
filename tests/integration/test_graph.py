# MFG-C2-051 — Integration Test: full Cat 2 graph compile + invoke

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from src.graph.graph import Graph


class FakeVectorStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=6, hybrid=True, metadata_filter=None):
        return self._hits[:top_k]


class FakeLLM:
    def complete(self, prompt):
        return "Reset fault E-17 on machine M-082 by power cycling the controller per the manual."


def _invoke(query, hits):
    agent = Graph(config={
        "max_retry": 1,
        "top_k": 5,
        "hybrid_search": True,
        "llm": FakeLLM(),
        "vector_store": FakeVectorStore(hits),
    })
    agent.compile()
    ctx = InvocationContext(
        session_id="it-session",
        caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
        caller_id="it-caller",
    )
    return agent.invoke(query, ctx=ctx)


class TestEquipmentMaintenanceQAGraph:
    def test_grounded_answer_end_to_end(self):
        hits = [{
            "text": "To reset fault E-17 on machine M-082, power cycle the controller.",
            "manual": "M-082 Service Manual", "page": 12, "score": 0.95,
        }]
        result = _invoke("Machine M-082 fault E-17 reset procedure?", hits)
        assert result["status"] in (AgentStatus.SUCCESS, AgentStatus.SUCCESS.value)
        assert "Sources:" in (result.get("output") or "")

    def test_no_kb_hits_blocked_by_s3(self):
        result = _invoke("Machine M-999 unknown fault", [])
        # No grounding -> S-3 gate sets error; graph routes to finalize without a grounded answer.
        assert result["status"] in (AgentStatus.ERROR, AgentStatus.ERROR.value)

    def test_empty_input_does_not_crash(self):
        result = _invoke("   ", [])
        assert result["status"] in (AgentStatus.ERROR, AgentStatus.ERROR.value)
