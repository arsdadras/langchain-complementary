"""
Tests for State, Reduce, SmartState, and reducer integration with LangGraph.
"""

from functools import partial
from typing import Any, Dict, List, is_typeddict

import pytest

from langcomp import (
    AgentBuilder,
    Reduce,
    ReducerConflictError,
    SmartState,
    State,
    merge_dicts,
    overwrite,
)


def _keep_max(a: int, b: int) -> int:
    return max(a or 0, b or 0)


class WorkflowState(State):
    items: List[str] = []
    metadata: Dict[str, Any] = {}
    status: str = "init"
    high_water: int = Reduce(_keep_max, default=0)


def test_smart_state_create_schema_is_real_typeddict():
    schema = SmartState.create_schema(extra_fields={"counter": int})
    assert is_typeddict(schema)
    assert "messages" in schema.__annotations__
    assert "counter" in schema.__annotations__
    assert "_langcomp_done" in schema.__annotations__


def test_state_explain_reports_inferred_and_custom_reducers():
    explained = WorkflowState.explain()
    assert explained["items"] == "add"
    assert explained["metadata"] == "merge_dicts"
    assert explained["status"] == "overwrite"
    assert explained["high_water"] == "_keep_max"


def test_state_explain_handles_callable_without_dunder_name():
    custom_partial = partial(lambda base, a, b: max(base, a or 0, b or 0), 10)

    class PartialState(State):
        score: int = Reduce(custom_partial, default=0)

    explained = PartialState.explain()
    assert "functools.partial" in explained["score"]


def test_reduce_rejects_default_and_default_factory_together():
    with pytest.raises(ReducerConflictError):
        Reduce(overwrite, default=1, default_factory=list)


def test_state_reducers_work_end_to_end_in_agent_builder():
    def step_one(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "items": ["a"],
            "metadata": {"source": "step1"},
            "status": "running",
            "high_water": 10,
        }

    def step_two(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "items": ["b"],
            "metadata": {"stage": 2},
            "status": "done",
            "high_water": 4,  # Lower than 10 -> _keep_max keeps 10
        }

    agent = (
        AgentBuilder("StateReducerAgent")
        .with_state_schema(WorkflowState)
        .add_node("one", step_one)
        .add_node("two", step_two)
        .compile()
    )

    result = agent.invoke({"items": [], "metadata": {}, "status": "init", "high_water": 0})
    assert result["items"] == ["a", "b"]
    assert result["metadata"] == {"source": "step1", "stage": 2}
    assert result["status"] == "done"
    assert result["high_water"] == 10


def test_merge_dicts_and_overwrite_helpers():
    assert merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}
    assert overwrite("old", "new") == "new"
