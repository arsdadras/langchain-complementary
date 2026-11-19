"""Unit tests for langcomp.core.state (State, Reduce, SmartState, reducers)."""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any, is_typeddict

import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from pydantic import AfterValidator

from langcomp import (
    AgentBuilder,
    Reduce,
    ReducerConflictError,
    SmartState,
    State,
    merge_dicts,
    node,
    overwrite,
)
from langcomp.core.state import merge_state_schemas


def _keep_max(a: int, b: int) -> int:
    return max(a or 0, b or 0)


class WorkflowState(State):
    items: list[str] = []
    metadata: dict[str, Any] = {}
    status: str = "init"
    high_water: int = Reduce(_keep_max, default=0)


def test_smart_state_create_schema_is_real_typeddict() -> None:
    schema = SmartState.create_schema(extra_fields={"counter": int})
    assert is_typeddict(schema)
    assert "messages" in schema.__annotations__
    assert "counter" in schema.__annotations__
    assert "_langcomp_done" in schema.__annotations__


def test_smart_state_create_schema_protects_reserved_keys() -> None:
    schema = SmartState.create_schema(extra_fields={"messages": str, "_langcomp_done": str})
    assert schema.__annotations__["messages"] is not str
    assert schema.__annotations__["_langcomp_done"] is not str


def test_smart_state_normalize_messages_handles_all_supported_forms() -> None:
    messages = [
        "Hello from string",
        {"role": "user", "content": "Hello from user dict"},
        {"role": "assistant", "content": "Hello from ai dict"},
        {"role": "system", "content": "Sys"},
        {"role": "developer", "content": "Dev"},
        ("user", "hello tuple"),
        ("assistant", "reply tuple"),
        {"role": "tool", "content": "42", "tool_call_id": "call_9", "name": "calc"},
        HumanMessage(content="Hello from object"),
    ]
    normalized = SmartState.normalize_messages(messages)
    assert len(normalized) == 9
    assert isinstance(normalized[0], HumanMessage)
    assert normalized[0].content == "Hello from string"
    assert isinstance(normalized[2], AIMessage)
    assert isinstance(normalized[3], SystemMessage)
    assert isinstance(normalized[4], SystemMessage)
    assert isinstance(normalized[5], HumanMessage)
    assert isinstance(normalized[6], AIMessage)
    assert isinstance(normalized[7], ToolMessage)
    assert normalized[7].tool_call_id == "call_9"
    assert normalized[7].name == "calc"


def test_smart_state_normalize_messages_rejects_invalid_items_and_roles() -> None:
    with pytest.raises(TypeError, match="Unsupported message item"):
        SmartState.normalize_messages([12345])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="Unknown message role"):
        SmartState.normalize_messages([{"role": "alien", "content": "hi"}])


def test_state_explain_reports_inferred_and_custom_reducers() -> None:
    explained = WorkflowState.explain()
    assert explained["items"] == "add"
    assert explained["metadata"] == "merge_dicts"
    assert explained["status"] == "overwrite"
    assert explained["high_water"] == "_keep_max"


def test_state_explain_handles_callable_without_dunder_name() -> None:
    custom_partial = partial(lambda base, a, b: max(base, a or 0, b or 0), 10)

    class PartialState(State):
        score: int = Reduce(custom_partial, default=0)

    explained = PartialState.explain()
    assert "functools.partial" in explained["score"]


@pytest.mark.parametrize("default_val", [1, None])
def test_reduce_rejects_default_and_default_factory_together(default_val: Any) -> None:
    with pytest.raises(ReducerConflictError):
        Reduce(overwrite, default=default_val, default_factory=list)


def test_reduce_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        Reduce("not_callable")  # type: ignore[arg-type]


def test_reduce_default_factory_and_bare_field() -> None:
    class FactoryState(State):
        tags: list[str] = Reduce(lambda a, b: list(dict.fromkeys([*(a or []), *(b or [])])), default_factory=list)
        bare: int = Reduce(_keep_max)

    defaults = FactoryState._defaults()
    assert defaults["tags"] == []
    assert "bare" not in defaults


def test_state_preserves_annotated_reducer_and_detects_conflicts() -> None:
    class MsgState(State):
        messages: Annotated[list[AnyMessage], add_messages] = []

    assert "add_messages" in MsgState.explain()["messages"]
    td = MsgState.to_typed_dict(include_internal=False)
    assert "_langcomp_done" not in td.__annotations__
    td_internal = MsgState.to_typed_dict(include_internal=True)
    assert "_langcomp_done" in td_internal.__annotations__

    class ConflictAnnotatedAndReduce(State):
        items: Annotated[list[str], lambda a, b: a + b] = Reduce(overwrite, default=[])

    with pytest.raises(ReducerConflictError, match="both an Annotated reducer and Reduce"):
        ConflictAnnotatedAndReduce.explain()

    class ConflictTwoAnnotated(State):
        items: Annotated[list[str], lambda a, b: a + b, overwrite] = []

    with pytest.raises(ReducerConflictError, match="multiple Annotated reducers"):
        ConflictTwoAnnotated.explain()


def test_state_ignores_pydantic_validators_and_unwraps_optional_containers() -> None:
    class ValidatedOptionalState(State):
        label: Annotated[str, AfterValidator(lambda v: v.strip())] = "ok"
        notes: list[str] | None = None
        meta: dict[str, int] | None = None

    explained = ValidatedOptionalState.explain()
    assert explained["label"] == "overwrite"
    assert explained["notes"] == "add"
    assert explained["meta"] == "merge_dicts"


def test_state_preserves_pydantic_defaults_in_agent_execution() -> None:
    class CountState(State):
        count: int = 42
        tags: list[str] = ["seed"]

    @node
    def bump(state: dict[str, Any]) -> dict[str, Any]:
        return {"count": state["count"] + 1, "tags": ["added"]}

    agent = AgentBuilder("DefaultHydration").with_state_schema(CountState).add_node(bump).compile()
    result = agent.invoke({})
    assert result["count"] == 43
    assert result["tags"] == ["seed", "added"]


def test_state_reducers_work_end_to_end_in_agent_builder() -> None:
    def step_one(state: dict[str, Any]) -> dict[str, Any]:
        del state
        return {
            "items": ["a"],
            "metadata": {"source": "step1"},
            "status": "running",
            "high_water": 10,
        }

    def step_two(state: dict[str, Any]) -> dict[str, Any]:
        del state
        return {
            "items": ["b"],
            "metadata": {"stage": 2},
            "status": "done",
            "high_water": 4,
        }

    agent = (
        AgentBuilder("StateReducerAgent")
        .with_state_schema(WorkflowState)
        .add_node("one", step_one)
        .add_node("two", step_two)
        .compile()
    )

    result = agent.invoke({})
    assert result["items"] == ["a", "b"]
    assert result["metadata"] == {"source": "step1", "stage": 2}
    assert result["status"] == "done"
    assert result["high_water"] == 10


def test_merge_state_schemas_and_helpers() -> None:
    assert merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}
    assert merge_dicts(None, {"x": 1}) == {"x": 1}
    assert overwrite("old", "new") == "new"

    merged = merge_state_schemas(WorkflowState, [], ensure_messages=True)
    assert "messages" in merged.__annotations__
    assert "items" in merged.__annotations__
