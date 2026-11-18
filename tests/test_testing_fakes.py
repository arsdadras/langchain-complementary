"""Unit tests for langcomp.devtools.testing.ScriptedFakeChatModel."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langcomp import ScriptedFakeChatModel, ScriptExhaustedError


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_replays_responses_in_order() -> None:
    model = ScriptedFakeChatModel(responses=[AIMessage("first"), AIMessage("second")])

    first = model.invoke([HumanMessage("hi")])
    second = model.invoke([HumanMessage("hi again")])

    assert first.content == "first"
    assert second.content == "second"


def test_coerces_plain_strings_into_ai_messages() -> None:
    model = ScriptedFakeChatModel(responses=["just a string"])

    reply = model.invoke([HumanMessage("hi")])

    assert isinstance(reply, AIMessage)
    assert reply.content == "just a string"


def test_raises_script_exhausted_error_when_script_runs_out() -> None:
    model = ScriptedFakeChatModel(responses=["only one"])
    model.invoke([HumanMessage("hi")])

    with pytest.raises(ScriptExhaustedError) as excinfo:
        model.invoke([HumanMessage("hi")])
    assert excinfo.value.call_count == 1
    assert "1 response" in str(excinfo.value)


def test_records_every_call_for_assertions() -> None:
    model = ScriptedFakeChatModel(responses=["ok", "ok again"])

    model.invoke([SystemMessage("be nice"), HumanMessage("hi")])
    model.invoke([HumanMessage("hi again")])

    assert model.call_count == 2
    assert isinstance(model.calls[0][0], SystemMessage)
    assert model.calls[1][0].content == "hi again"


def test_bind_tools_records_names_without_changing_playback() -> None:
    model = ScriptedFakeChatModel(responses=["ok"])

    bound = model.bind_tools([add])
    reply = bound.invoke([HumanMessage("hi")])

    assert bound.bound_tool_names == ("add",)
    assert reply.content == "ok"


@pytest.mark.asyncio
async def test_supports_async_invocation() -> None:
    model = ScriptedFakeChatModel(responses=["async ok"])

    reply = await model.ainvoke([HumanMessage("hi")])

    assert reply.content == "async ok"
