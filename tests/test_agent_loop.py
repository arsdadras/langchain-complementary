"""Unit tests for langcomp's hand-built agent loop, TraceCollector interop, and AgentBuilder."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langcomp import (
    AgentBuilder,
    FakeModel,
    GraphVisualizer,
    TraceCollector,
    create_agent,
    summarize,
)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers (plain callable, auto-converted to BaseTool)."""
    return a * b


@tool
def explode() -> None:
    """Always raises to exercise error-handling path."""
    raise ValueError("kaboom")


def test_agent_with_no_tools_calls_model_once() -> None:
    model = FakeModel(script=["hello there"])
    agent = AgentBuilder("SimpleChat").with_model(model).compile()

    result = agent.invoke("hi")

    assert [m.content for m in result["messages"]] == ["hi", "hello there"]
    assert model.call_count == 1


def test_agent_executes_tool_call_and_loops_back_to_model() -> None:
    model = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1"}]),
            AIMessage("The answer is 5."),
        ]
    )
    agent = (
        AgentBuilder("Calculator")
        .with_model(model)
        .with_tools([add])
        .compile()
    )

    result = agent.invoke("what is 2 + 3?")

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "5"
    assert tool_messages[0].tool_call_id == "call_1"
    assert result["messages"][-1].content == "The answer is 5."
    assert model.call_count == 2


def test_agent_auto_wraps_plain_callable_tools() -> None:
    model = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"name": "multiply", "args": {"a": 6, "b": 7}, "id": "c1"}]),
            AIMessage("42"),
        ]
    )
    agent = create_agent(model=model, tools=[multiply])

    result = agent.invoke("6 * 7?")

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "42"
    assert result["messages"][-1].content == "42"


def test_agent_supports_multiple_parallel_tool_calls_in_one_turn() -> None:
    model = FakeModel(
        script=[
            AIMessage(
                "",
                tool_calls=[
                    {"name": "add", "args": {"a": 1, "b": 1}, "id": "call_1"},
                    {"name": "add", "args": {"a": 2, "b": 2}, "id": "call_2"},
                ],
            ),
            AIMessage("Done."),
        ]
    )
    agent = create_agent(model=model, tools=[add])

    result = agent.invoke("add two pairs")

    tool_messages = {
        m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)
    }
    assert tool_messages == {"call_1": "2", "call_2": "4"}


def test_agent_turns_tool_exception_into_error_tool_message() -> None:
    model = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"name": "explode", "args": {}, "id": "call_1"}]),
            AIMessage("It failed, as expected."),
        ]
    )
    agent = create_agent(model=model, tools=[explode])

    result = agent.invoke("try the broken tool")

    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    assert "kaboom" in tool_message.content
    assert result["messages"][-1].content == "It failed, as expected."


def test_agent_reports_unknown_tool_without_crashing() -> None:
    model = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"name": "ghost_tool", "args": {}, "id": "call_1"}]),
            AIMessage("I could not find that tool."),
        ]
    )
    agent = create_agent(model=model, tools=[add])

    result = agent.invoke("call ghost")

    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    assert "ghost_tool" in tool_message.content


def test_system_prompt_is_transient_and_not_persisted_in_state() -> None:
    model = FakeModel(script=["ok"])
    agent = (
        AgentBuilder("TerseBot")
        .with_model(model)
        .with_system_prompt("You are terse.")
        .compile()
    )

    result = agent.invoke("hi")

    assert model.calls[0][0] == SystemMessage("You are terse.")
    assert not any(isinstance(m, SystemMessage) for m in result["messages"])


def test_agent_binds_tools_to_model() -> None:
    model = FakeModel(script=["ok"])
    create_agent(model=model, tools=[add, explode])

    assert set(model.bound_tool_names) == {"add", "explode"}


@pytest.mark.asyncio
async def test_agent_supports_async_invocation_end_to_end() -> None:
    model = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"name": "add", "args": {"a": 10, "b": 5}, "id": "c1"}]),
            AIMessage("15"),
        ]
    )
    agent = create_agent(model=model, tools=[add])

    result = await agent.ainvoke("10 + 5")

    assert result["messages"][-1].content == "15"


def test_hand_built_agent_propagates_callbacks_to_trace_collector() -> None:
    model = FakeModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "add", "args": {"a": 4, "b": 5}, "id": "c1"}],
                usage_metadata={"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
            ),
            AIMessage(
                content="9",
                usage_metadata={"input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
            ),
        ]
    )
    agent = AgentBuilder("TracedLoop").with_model(model).with_tools([add]).compile()
    collector = TraceCollector(out_dir=None)

    result = agent.invoke("4 + 5?", config={"callbacks": [collector]})

    assert result["messages"][-1].content == "9"
    summary = summarize(collector)
    assert summary["total_calls"] == 2
    assert summary["total_input_tokens"] == 80
    assert summary["total_output_tokens"] == 15
    assert "model" in summary["by_node"]


def test_hand_built_agent_graph_visualizes_to_mermaid() -> None:
    model = FakeModel(script=["ok"])
    agent = create_agent(model=model, tools=[add])

    mermaid = GraphVisualizer.to_mermaid(agent)

    assert "model" in mermaid
    assert "tools" in mermaid
