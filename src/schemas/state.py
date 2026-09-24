"""AgentCore Platform v1.0 — MFG-C2-051 State schema."""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption. Extend AgentState with agent-specific
# JSON-serializable fields only. Do NOT add credentials, secrets, or
# Pydantic models / InvocationContext.

from typing import NotRequired

from framework.schemas.agent_state import AgentState


# Type-check note: the wheel ships no py.typed, so mypy resolves AgentState to Any
# and reports every NotRequired below as valid-type. The fields are correct (the state contract
# requires NotRequired) -- the report is a packaging artifact, suppressed per field.
# Drop these ignores once the wheel ships py.typed.
class State(AgentState):
    """Equipment maintenance Q&A state.

    Shared fields (user_input, validated_input, status, result, formatted_output,
    node_history, error_log, ...) are inherited from AgentState. Only
    agent-specific fields are declared here.
    """

    # QueryNormalizeNode (outer pre_process) outputs
    normalized_query: NotRequired[str]  # type: ignore[valid-type]
    equipment_id: NotRequired[str]  # type: ignore[valid-type]
    fault_code: NotRequired[str]  # type: ignore[valid-type]
    language: NotRequired[str]  # type: ignore[valid-type]

    # HybridRetrieveNode (inner) output — list of passage dicts
    # each: {"text": str, "manual": str, "page": int, "score": float}
    retrieved_passages: NotRequired[list]  # type: ignore[valid-type]

    # GroundedAnswerNode (inner) outputs
    draft_answer: NotRequired[str]  # type: ignore[valid-type]
    citations: NotRequired[list]  # type: ignore[valid-type]  # list of {"manual": str, "page": int}
    high_risk: NotRequired[bool]  # type: ignore[valid-type]

    # ResponseValidateNode (outer post_process) outputs
    grounded: NotRequired[bool]  # type: ignore[valid-type]
    safety_note: NotRequired[str]  # type: ignore[valid-type]
