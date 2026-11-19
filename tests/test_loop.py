"""Unit tests for langcomp.core.loop and langcomp.core.nodes (sync + async execution loop)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.types import Command, interrupt

from langcomp import (
    AgentBuilder,
    ApprovalGate,
    ApprovalRejectedError,
    ConfigurationError,
    FakeModel,
    ToolGuardMiddleware,
    checkpointer,
    create_agent,
    wrap_model_call,
)
from langcomp.core.loop import normalize_tools


def _multiply(a: int, b: int) -> int:
    """Multiply two integers (plain callable, auto-converted to BaseTool)."""
    return a * b


def test_agent_with_no_tools_calls_model_once() -> None:
    model = FakeModel(script=["hello there"])
    ag = AgentBuilder("SimpleChat").with_model(model).compile()

    result = ag.invoke("hi")
    assert [m.content for m in result["messages"]] == ["hi", "hello there"]
    assert model.call_count == 1


def test_agent_executes_tool_call_and_loops_back_to_model(add_tool: BaseTool) -> None:
    model = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1"}]),
            AIMessage("The answer is 5."),
        ]
    )
    ag = AgentBuilder("Calculator").with_model(model).with_tools([add_tool]).compile()

    result = ag.invoke("what is 2 + 3?")
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "5"
    assert tool_messages[0].tool_call_id == "call_1"
    assert result["messages"][-1].content == "The answer is 5."
    assert model.call_count == 2


def test_agent_auto_wraps_plain_callable_tools() -> None:
    model = FakeModel(
        script=[
            AIMessage(
                "", tool_calls=[{"name": "_multiply", "args": {"a": 6, "b": 7}, "id": "c1"}]
            ),
            AIMessage("42"),
        ]
    )
    ag = create_agent(model=model, tools=[_multiply])
    result = ag.invoke("6 * 7?")
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].content == "42"
    assert result["messages"][-1].content == "42"


def test_normalize_tools_rejects_invalid_item() -> None:
    with pytest.raises(TypeError, match="Expected BaseTool or callable"):
        normalize_tools([42])  # type: ignore[list-item]


def test_create_agent_rejects_duplicate_tool_names() -> None:
    @tool("dup")
    def t1() -> str:
        """First."""
        return "1"

    @tool("dup")
    def t2() -> str:
        """Second."""
        return "2"

    with pytest.raises(ConfigurationError, match="Duplicate tool name"):
        create_agent(FakeModel(script=["ok"]), tools=[t1, t2])


def test_agent_supports_multiple_parallel_tool_calls_in_one_turn(add_tool: BaseTool) -> None:
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
    ag = create_agent(model=model, tools=[add_tool])
    result = ag.invoke("add two pairs")
    tool_messages = {
        m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)
    }
    assert tool_messages == {"call_1": "2", "call_2": "4"}


def test_agent_turns_tool_exception_into_error_tool_message(explode_tool: BaseTool) -> None:
    model = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"name": "explode", "args": {}, "id": "call_1"}]),
            AIMessage("It failed, as expected."),
        ]
    )
    ag = create_agent(model=model, tools=[explode_tool])
    result = ag.invoke("try the broken tool")
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    assert "kaboom" in str(tool_message.content)
    assert result["messages"][-1].content == "It failed, as expected."


def test_agent_reports_unknown_tool_without_crashing(add_tool: BaseTool) -> None:
    model = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"name": "ghost_tool", "args": {}, "id": "call_1"}]),
            AIMessage("I could not find that tool."),
        ]
    )
    ag = create_agent(model=model, tools=[add_tool])
    result = ag.invoke("call ghost")
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    assert "ghost_tool" in str(tool_message.content)


def test_system_prompt_is_transient_and_not_persisted_in_state() -> None:
    model = FakeModel(script=["ok"])
    ag = (
        AgentBuilder("TerseBot")
        .with_model(model)
        .with_system_prompt("You are terse.")
        .compile()
    )
    result = ag.invoke("hi")
    assert model.calls[0][0] == SystemMessage("You are terse.")
    assert not any(isinstance(m, SystemMessage) for m in result["messages"])


def test_agent_binds_tools_to_model(add_tool: BaseTool, explode_tool: BaseTool) -> None:
    model = FakeModel(script=["ok"])
    create_agent(model=model, tools=[add_tool, explode_tool])
    assert set(model.bound_tool_names) == {"add", "explode"}


def test_graph_interrupt_inside_tool_propagates_through_loop_and_tool_guard() -> None:
    @tool
    def confirm_transfer(amount: int) -> str:
        """Pause for human confirmation."""
        decision = interrupt({"amount": amount})
        return f"transferred {amount}: {decision}"

    model = FakeModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "confirm_transfer", "args": {"amount": 500}, "id": "c1"}],
            ),
            AIMessage(content="Transfer complete."),
        ]
    )
    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        saver = checkpointer("memory")

    ag = create_agent(
        model,
        tools=[confirm_transfer],
        middleware=[ToolGuardMiddleware(max_retries=2, retry_delay_seconds=0.0)],
        checkpointer=saver,
    )
    cfg = {"configurable": {"thread_id": "hitl-tool"}}
    paused = ag.invoke("Transfer 500", config=cfg)
    assert "__interrupt__" in paused

    resumed = ag.invoke(Command(resume="approved"), config=cfg)
    assert resumed["messages"][-1].content == "Transfer complete."
    tool_msg = next(m for m in resumed["messages"] if isinstance(m, ToolMessage))
    assert "transferred 500: approved" in str(tool_msg.content)


def test_approval_rejected_error_from_wrapped_tool_propagates() -> None:
    @tool
    def danger() -> str:
        """Dangerous op."""
        return "boom"

    gated = ApprovalGate(approval_func=lambda _n, _p: False).wrap_tool(danger)
    model = FakeModel(
        script=[
            AIMessage(content="", tool_calls=[{"name": "danger", "args": {}, "id": "c1"}]),
            AIMessage(content="never reached"),
        ]
    )
    ag = create_agent(model, tools=[gated])
    with pytest.raises(ApprovalRejectedError):
        ag.invoke("do danger")


def test_model_request_override_tools_rebinds_and_clears_tools(
    add_tool: BaseTool, explode_tool: BaseTool
) -> None:
    @wrap_model_call
    def only_add(request: Any, handler: Any) -> Any:
        return handler(request.override(tools=(add_tool,)))

    model = FakeModel(script=["ok"])
    ag = create_agent(model, tools=[add_tool, explode_tool], middleware=[only_add])
    ag.invoke("hi")
    assert model.bound_tool_names == ("add",)

    @wrap_model_call
    def clear_tools(request: Any, handler: Any) -> Any:
        return handler(request.override(tools=()))

    model2 = FakeModel(script=["ok"])
    ag2 = create_agent(model2, tools=[add_tool], middleware=[clear_tools])
    ag2.invoke("hi")
    assert model2.call_count == 1


async def test_async_only_tool_and_async_loop_execution() -> None:
    async def async_add(a: int, b: int) -> int:
        return a + b

    async_tool = StructuredTool.from_function(
        coroutine=async_add,
        name="async_add",
        description="Add two numbers asynchronously.",
    )
    model = FakeModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "async_add", "args": {"a": 10, "b": 5}, "id": "c1"}],
            ),
            AIMessage(content="15"),
        ]
    )
    ag = create_agent(model=model, tools=[async_tool])
    result = await ag.ainvoke("10 + 5")
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.status != "error"
    assert tool_msg.content == "15"
    assert result["messages"][-1].content == "15"


def test_create_agent_rejects_model_kwargs_with_prebuilt_model() -> None:
    with pytest.raises(ConfigurationError, match="model_kwargs"):
        create_agent(FakeModel(script=["ok"]), temperature=0.5)
