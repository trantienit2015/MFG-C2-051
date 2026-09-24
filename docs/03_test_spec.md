# 03 — Test Specification — MFG-C2-051

## Coverage map

| Test file | Scope |
|---|---|
| `tests/unit/test_query_normalize_node.py` | QueryNormalizeNode: EN extraction, JA detection, empty-input error |
| `tests/unit/test_hybrid_retrieve_node.py` | HybridRetrieveNode: hits + equipment_id filter, empty-query error, no-store explicit-empty |
| `tests/unit/test_grounded_answer_node.py` | GroundedAnswerNode: citations, high-risk flag, no-passage no-evidence |
| `tests/unit/test_response_validate_node.py` | ResponseValidateNode S-3 gate: grounded pass, ungrounded block, missing-citation block, safety note |
| `tests/integration/test_graph.py` | Full Cat 2 compile+invoke: grounded E2E, no-hits S-3 block, empty input |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4: no Level-0 imports under src/ |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2/PB-5: State has no credential fields / Pydantic / InvocationContext |

## Framework compliance tests (TC mapping)

| TC | Covered by |
|---|---|
| TC-01 State contract | `test_state_safety.py` (flat TypedDict, no Pydantic) |
| TC-02 SecurityViolationError / reject path | QueryNormalize empty-input + ResponseValidate S-3 block |
| TC-03 No JWT/credential in src | `gate-credential-scan` (CI) + no secrets in state |
| TC-04 InvocationContext via configurable | Integration invoke passes `ctx=`; state stores no context |
| TC-05 Audit logging | `emit_trace_event()` in HybridRetrieve / GroundedAnswer / ResponseValidate |
| TC-06/07 Security gates non-bypassable | FunctionNode `@final` gates inherited; nodes use `execute()` only |
| TC-08 required_trust_level | agent.yaml `VERIFIED_EXTERNAL` |

## Proof-of-Boundary (PB)

| PB | Covered by |
|---|---|
| PB-1 BaseNode→AuditLogger | `emit_trace_event` calls (verified live in integration node_history) |
| PB-2 State serialization | `test_state_safety.py` |
| PB-3 L1→external service | Hybrid retrieval via injected vector store (mock in tests; real store in deploy) |
| PB-4 Import isolation | `test_import_isolation.py` |
| PB-5 Checkpoint safety | `test_state_safety.py` (no credential fields) |
| PB-6 Invoke order | Integration: node_history shows Initialize→pre→main→post→Finalize |

## Run

```bash
PYTHONUTF8=1 python -m pytest tests/ -v
```

(`PYTHONUTF8=1` matches the CI Linux UTF-8 locale; required locally on Windows because the
KB high-risk markers + CJK regex ranges contain Japanese characters.)
