"""
Tests for persistence checkpointer factory, Team hierarchy & validation,
async ToolGuard / ApprovalGate wrappers, and GraphVisualizer edge cases.
"""

from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langcomp import (
    AgentBuilder,
    ApprovalGate,
    ApprovalRejectedError,
    GraphVisualizer,
    Team,
    ToolGuard,
    checkpointer,
)


def test_checkpointer_memory_warns_and_validation_errors():
    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        saver = checkpointer("memory")
    assert saver is not None

    with pytest.raises(ValueError, match="requires 'path'"):
        checkpointer("sqlite")

    with pytest.raises(ValueError, match="requires 'conn_string'"):
        checkpointer("postgres")

    with pytest.raises(ValueError, match="unknown checkpointer kind"):
        checkpointer("redis")


def test_team_transitive_inheritance_and_validation():
    leaf = Team(name="fact_checker", description="Checks facts")
    mid = Team(name="researcher", description="Researches topics", children=[leaf])
    root = Team(
        name="supervisor",
        description="Coordinates team",
        system_prompt="Be concise.",
        tools=["search_tool"],
        model="fake-model",
        skills=["./skills"],
        children=[mid],
    )

    root_resolved = root._resolve(parent=None)
    mid_resolved = mid._resolve_from_context(root_resolved)
    leaf_resolved = leaf._resolve_from_context(mid_resolved)

    assert leaf_resolved["system_prompt"] == "Be concise."
    assert leaf_resolved["tools"] == ["search_tool"]
    assert leaf_resolved["model"] == "fake-model"
    assert leaf_resolved["skills"] == ["./skills"]

    # Recursive compilation wiring via stubbed deepagents factories
    @dataclass
    class StubCompiledSubAgent:
        name: str
        description: str
        runnable: Any

    compiled_calls = []

    def stub_create_deep_agent(**kwargs):
        compiled_calls.append(kwargs)
        return {"compiled": kwargs}

    graph = root._compile_resolved(
        root_resolved,
        model="fallback-model",
        compiled_subagent_cls=StubCompiledSubAgent,
        create_deep_agent_fn=stub_create_deep_agent,
    )
    assert len(compiled_calls) == 2
    assert isinstance(graph["compiled"]["subagents"][0], StubCompiledSubAgent)
    assert graph["compiled"]["subagents"][0].name == "researcher"


def test_team_rejects_duplicate_siblings_and_general_purpose():
    dup = Team(
        name="root",
        description="root",
        children=[
            Team(name="worker", description="a"),
            Team(name="worker", description="b"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate child name"):
        dup._validate_children()

    shadow = Team(
        name="root",
        description="root",
        children=[Team(name="general-purpose", description="shadow")],
    )
    with pytest.raises(ValueError, match="general-purpose"):
        shadow._validate_children()


@pytest.mark.asyncio
async def test_tool_guard_async_and_approval_gate_wrappers():
    attempts = 0

    async def flaky_async(val: int) -> int:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("temp")
        return val + 1

    guard = ToolGuard(max_retries=3, retry_delay_seconds=0.0)
    wrapped_async = guard.wrap_function(flaky_async)
    assert await wrapped_async(10) == 11
    assert attempts == 2

    gate = ApprovalGate(approval_func=lambda name, params: params.get("safe") is True)

    @tool
    def delete_record(record_id: int, safe: bool = False) -> str:
        """Delete a record by ID."""
        return f"deleted:{record_id}"

    gated_tool = gate.wrap_tool(delete_record)
    with pytest.raises(ApprovalRejectedError):
        gated_tool.invoke({"record_id": 1, "safe": False})

    assert gated_tool.invoke({"record_id": 1, "safe": True}) == "deleted:1"


def test_graph_visualizer_ascii_reflects_actual_nodes_and_rejects_invalid():
    agent = (
        AgentBuilder("TopoAgent")
        .add_node("ingest", lambda s: {"messages": [AIMessage(content="ok")]})
        .add_node("publish", lambda s: {"messages": [AIMessage(content="done")]})
        .compile()
    )

    ascii_out = GraphVisualizer.to_ascii(agent)
    assert "ingest" in ascii_out
    assert "publish" in ascii_out

    with pytest.raises(TypeError, match="get_graph"):
        GraphVisualizer.to_mermaid("not-a-graph")
