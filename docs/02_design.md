# 02 — Design Specification — MFG-C2-051

## Position in AgentCore Architecture

- **Agent Class**: `EquipmentMaintenanceQAGraph` (`src/graph/graph.py`)
- **L1 Base type**: `AgentBaseGraph` (L1 direct) — no L2 inheritance.
- **Inheritance**: `AgentBaseGraph` (L1 direct).
- **Category**: Cat 2 (multi-step domain workflow). Implemented as the canonical Cat 2
  pattern: outer `AgentBaseGraph` + `GraphNode` in the `main` slot + inner `BaseGraph`
  domain subgraph.
- **Pattern**: VectorRAG (hybrid dense + BM25 retrieval → grounded answer).

## Graph topology

```
OUTER (EquipmentMaintenanceQAGraph : AgentBaseGraph)
  initialize → pre_process (QueryNormalizeNode)
            → main (MaintenanceQAGraphNode : GraphNode)
            → post_process (ResponseValidateNode)
            → finalize

INNER (MaintenanceWorkflowGraph : BaseGraph)  — wrapped by the GraphNode main slot
  START → hybrid_retrieve (HybridRetrieveNode) → grounded_answer (GroundedAnswerNode) → END
```

The outer `pre_process` serializes the normalized payload into `validated_input` (JSON
string); `GraphNode.extract_input` passes that string into the inner graph, whose first node
(`HybridRetrieveNode`) `json.loads` it back. Config (LLM, vector store, retrieval knobs)
reaches the inner graph via `MaintenanceQAGraphNode._parent_config()`, never via state.

## Node responsibilities

| Slot | Node | Responsibility |
|---|---|---|
| outer pre_process | `QueryNormalizeNode` | Normalize floor shorthand, extract `equipment_id`/`fault_code`, detect JA/EN; reject empty input; serialize `validated_input`. |
| inner step 1 | `HybridRetrieveNode` | Hybrid dense(multilingual-e5)+BM25 retrieval over manual KB; metadata filter on `equipment_id`; `top_k` from config. Emits `kb_retrieval` audit event. |
| inner step 2 | `GroundedAnswerNode` | LLM answer grounded only in retrieved passages; build manual+page citations; flag high-risk steps. Emits `answer_generated` audit event. |
| outer post_process | `ResponseValidateNode` | **S-3 deterministic gate**: block ungrounded / uncited answers (return ERROR, do not emit); append safety-escalation note when high-risk. Emits `answer_validated` / `ungrounded_answer_blocked`. |

## State schema (`src/schemas/state.py`)

`class State(AgentState)` adds flat, JSON-serializable fields only:
`normalized_query`, `equipment_id`, `fault_code`, `language`, `retrieved_passages` (list of
passage dicts), `draft_answer`, `citations` (list of {manual, page}), `high_risk` (bool),
`grounded` (bool), `safety_note`. No Pydantic / InvocationContext / credentials in State.

## Services (deterministic, no agenticstar imports)

- `src/services/query_service.py` — query normalization, equipment-id / fault-code
  extraction, JA/EN language detection.
- `src/services/grounding_service.py` — citation extraction, high-risk detection, and the
  deterministic `is_grounded()` check (token-overlap; NOT an LLM self-check).

## Security (5-layer)

- **S-1**: agent default `required_trust_level: VERIFIED_EXTERNAL` in `config/agent.yaml`;
  technician queries are non-privileged.
- **S-2**: framework `_security_gate_input()` (`@final`) PII scan on standard fields;
  `QueryNormalizeNode` rejects empty/invalid input.
- **S-3**: `ResponseValidateNode` deterministic grounding+citation gate (block, never emit
  ungrounded). Secrets (LLM/vector-store) via entry-point `bound_secrets` /
  `secrets_factory` / `provision_secrets` (`src/api/server.py`); never `os.environ`, never in
  state. `requires.secrets` declared in `agent.yaml`.
- **S-4**: `emit_trace_event()` from `shared.utils.audit_logger` on retrieval, answer, and
  validation side-effects.
- **S-5**: framework credential scan; flat-TypedDict State, no secrets stored.

## Config surface (`config/agent.yaml` `agent.config`)

`max_retry`, `timeout_seconds`, `top_k`, `hybrid_search`, `score_threshold`.
