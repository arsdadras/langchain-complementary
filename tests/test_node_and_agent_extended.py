"""
Tests for @node, idempotent=True with LangGraph interrupt(), Agent streaming/async,
and offline ReAct execution via FakeModel.
"""

from typing import Any, Dict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START
from langgraph.types import Command, interrupt

from langcomp import AgentBuilder, FakeModel, State, checkpointer, node


def test_node_decorator_and_invoke_for_test():
    @node
    def double_counter(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"counter": state["counter"] * 2}

    assert double_counter.name == "double_counter"
    assert double_counter.invoke_for_test({"counter": 7}) == {"counter": 14}
    assert double_counter({"counter": 3}) == {"counter": 6}


def test_idempotent_node_skips_reexecution_on_same_thread_with_interrupt():
    side_effect_calls = 0

    class HitlState(State):
        approved: bool = False
        charged: bool = False

    @node
    def ask_approval(state: Dict[str, Any]) -> Dict[str, Any]:
        decision = interrupt({"action": "charge_card"})
        return {"approved": bool(decision)}

    @node(idempotent=True)
    def charge_card(state: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal side_effect_calls
        side_effect_calls += 1
        return {"charged": True}

    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        saver = checkpointer("memory")

    agent = (
        AgentBuilder("PaymentAgent")
        .with_state_schema(HitlState)
        .with_checkpointer(saver)
        .add_node(ask_approval)
        .add_node(charge_card)
        .add_edge(START, ask_approval)
        .add_edge(ask_approval, charge_card)
        .add_edge(charge_card, END)
        .compile()
    )

    cfg = {"configurable": {"thread_id": "order-42"}}
    first = agent.invoke({"approved": False, "charged": False}, config=cfg)
    assert "__interrupt__" in first
    assert side_effect_calls == 0

    resumed = agent.invoke(Command(resume=True), config=cfg)
    assert resumed["approved"] is True
    assert resumed["charged"] is True
    assert side_effect_calls == 1

    # Re-running the idempotent node directly on the same thread state must be a no-op
    second_pass = agent.invoke(
        {"approved": True, "charged": True, "_langcomp_done": ["charge_card"]},
        config={"configurable": {"thread_id": "order-42-replay"}},
    )
    assert "__interrupt__" in second_pass
    resumed_again = agent.invoke(
        Command(resume=True),
        config={"configurable": {"thread_id": "order-42-replay"}},
    )
    assert resumed_again["charged"] is True
    assert side_effect_calls == 1


@pytest.mark.asyncio
async def test_agent_ainvoke_stream_and_astream():
    @node
    def step_a(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": [AIMessage(content="from_a")]}

    @node
    def step_b(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": [AIMessage(content="from_b")]}

    agent = (
        AgentBuilder("StreamAgent")
        .add_node(step_a)
        .add_node(step_b)
        .compile()
    )

    async_res = await agent.ainvoke("hello")
    assert async_res["messages"][-1].content == "from_b"

    sync_chunks = list(agent.stream("hello"))
    assert len(sync_chunks) == 2

    async_chunks = [chunk async for chunk in agent.astream("hello")]
    assert len(async_chunks) == 2


def test_agent_builder_with_fake_model_and_tools_end_to_end():
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    fake = FakeModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "multiply", "args": {"a": 6, "b": 7}, "id": "call_1"}
                ],
            ),
            AIMessage(content="The answer is 42."),
        ]
    )

    agent = (
        AgentBuilder("MathAgent")
        .with_model(fake)
        .with_tools([multiply])
        .with_system_prompt("You are a math assistant.")
        .compile()
    )

    res = agent.invoke("What is 6 * 7?")
    assert res["messages"][-1].content == "The answer is 42."
    assert fake.call_count == 2


def test_agent_builder_validation_errors():
    with pytest.raises(ValueError, match="either a model or custom nodes"):
        AgentBuilder("Empty").compile()

    builder = AgentBuilder("BadEdges").add_node("a", lambda s: s).add_edge("a", "nonexistent")
    with pytest.raises(ValueError, match="not a registered node"):
        builder.compile()
