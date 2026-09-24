# MFG-C2-051 — Unit Tests: HybridRetrieveNode (inner step 1)

import json

from framework.schemas.agent_status import AgentStatus
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode


class FakeVectorStore:
    def __init__(self, hits):
        self._hits = hits
        self.last_filter = None

    def search(self, query, top_k=6, hybrid=True, metadata_filter=None):
        self.last_filter = metadata_filter
        return self._hits[:top_k]


def _state(payload):
    return {
        "user_input": json.dumps(payload),
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }


class TestHybridRetrieveNode:
    def test_success_returns_passages_and_filters_by_equipment(self):
        hits = [{"text": "Reset E-17 by power cycling.", "manual": "M-082 Manual", "page": 12, "score": 0.9}]
        store = FakeVectorStore(hits)
        node = HybridRetrieveNode(vector_store=store, top_k=5, hybrid_search=True)
        result = node.execute(_state({"normalized_query": "reset E-17", "equipment_id": "M-082"}))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["retrieved_passages"] == hits
        assert store.last_filter == {"equipment_id": "M-082"}

    def test_empty_query_error(self):
        node = HybridRetrieveNode(vector_store=FakeVectorStore([]))
        result = node.execute(_state({"normalized_query": "", "equipment_id": ""}))
        assert result["status"] == AgentStatus.ERROR
        assert "empty" in result["error_log"][0].lower()

    def test_no_store_falls_back_to_seed_corpus_not_empty(self):
        # With no store wired (the standalone deployment path), retrieval falls back
        # to the deterministic seed corpus. Returning nothing here would leave every
        # answer ungrounded and trip the S-3 gate on every request.
        node = HybridRetrieveNode(vector_store=None)
        result = node.execute(_state({"normalized_query": "lockout before inspection", "equipment_id": ""}))
        assert result["status"] == AgentStatus.SUCCESS
        passages = result["retrieved_passages"]
        assert passages, "seed-corpus fallback must return citable passages"
        # Every passage carries the citation metadata the S-3 gate needs.
        for passage in passages:
            assert passage.get("manual")
            assert passage.get("page") is not None

    def test_no_store_honours_equipment_id_filter(self):
        node = HybridRetrieveNode(vector_store=None)
        result = node.execute(_state({"normalized_query": "belt tension", "equipment_id": "CV-100"}))
        assert result["status"] == AgentStatus.SUCCESS
        # Only CV-100 passages, plus the equipment-agnostic ones (empty equipment_id).
        assert {p["equipment_id"] for p in result["retrieved_passages"]} <= {"CV-100", ""}

    def test_store_failure_returns_error_not_unhandled(self):
        class BoomStore:
            def search(self, *a, **k):
                raise RuntimeError("vector store down")

        node = HybridRetrieveNode(vector_store=BoomStore())
        result = node.execute(_state({"normalized_query": "reset E-17", "equipment_id": "M-082"}))
        assert result["status"] == AgentStatus.ERROR
        assert "search failed" in result["error_log"][0]
