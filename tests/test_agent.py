"""Unit tests for langcomp.core.agent (Agent, AgentBuilder, @node, @agent, interop)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START
from langgraph.types import Command, interrupt

from langcomp import (
    Agent,
    AgentBuilder,
    ConfigurationError,
    FakeModel,
    State,
    agent,
    checkpointer,
    node,
)


def test_node_decorator_custom_name_and_invoke_for_test() -> None:
    @node
    def double_counter(state: dict[str, Any]) -> dict[str, Any]:
        return {"counter": state["counter"] * 2}

    @node(name="renamed_step")
    def raw_step(state: dict[str, Any]) -> dict[str, Any]:
        return state

    assert double_counter.name == "double_counter"
    assert raw_step.name == "renamed_step"
    assert double_counter.invoke_for_test({"counter": 7}) == {"counter": 14}
    assert double_counter({"counter": 3}) == {"counter": 6}


def test_agent_builder_custom_nodes() -> None:
    def dummy_node(state: dict[str, Any]) -> dict[str, Any]:
        del state
        return {"messages": [AIMessage(content="Processed by dummy_node")]}

    agent_inst = (
        AgentBuilder("TestAgent")
        .add_node("step1", dummy_node)
        .add_edge(START, "step1")
        .add_edge("step1", END)
        .compile()
    )
    assert agent_inst.name == "TestAgent"

    result = agent_inst.invoke("User query")
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][-1].content == "Processed by dummy_node"


def test_idempotent_node_runs_once_per_checkpointed_thread_and_blocks_external_injection() -> None:
    side_effect_calls = 0

    class HitlState(State):
        approved: bool = False
        charged: bool = False

    @node(idempotent=True)
    def reserve_inventory(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal side_effect_calls
        side_effect_calls += 1
        return {"charged": True}

    @node
    def ask_approval(state: dict[str, Any]) -> dict[str, Any]:
        decision = interrupt({"action": "confirm_order"})
        return {"approved": bool(decision)}

    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        saver = checkpointer("memory")

    ag = (
        AgentBuilder("PaymentAgent")
        .with_state_schema(HitlState)
        .with_checkpointer(saver)
        .add_node(reserve_inventory)
        .add_node(ask_approval)
        .add_edge(START, reserve_inventory)
        .add_edge(reserve_inventory, ask_approval)
        .add_edge(ask_approval, END)
        .compile()
    )

    cfg = {"configurable": {"thread_id": "order-42"}}
    first = ag.invoke({}, config=cfg)
    assert "__interrupt__" in first
    assert side_effect_calls == 1

    # Resuming on the same thread completes without re-running reserve_inventory
    resumed = ag.invoke(Command(resume=True), config=cfg)
    assert resumed["approved"] is True
    assert resumed["charged"] is True
    assert side_effect_calls == 1

    # Re-invoking the same thread with a new turn still skips already-completed idempotent node
    second_turn = ag.invoke({"approved": False}, config=cfg)
    assert "__interrupt__" in second_turn
    assert side_effect_calls == 1

    # External input MUST NOT be able to inject _langcomp_done to bypass an idempotent node
    fresh_cfg = {"configurable": {"thread_id": "order-spoofed"}}
    ag.invoke({"_langcomp_done": ["reserve_inventory"]}, config=fresh_cfg)
    assert side_effect_calls == 2


async def test_async_idempotent_node_and_callable_object() -> None:
    runs = 0

    class AsyncCallableStep:
        __name__ = "async_step"

        async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
            nonlocal runs
            runs += 1
            return {"messages": [AIMessage(content="async-ok")]}

    step = node(AsyncCallableStep(), name="async_step", idempotent=True)
    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        saver = checkpointer("memory")
    ag = AgentBuilder("AsyncIdem").add_node(step).with_checkpointer(saver).compile()

    cfg = {"configurable": {"thread_id": "async-t1"}}
    res = await ag.ainvoke("go", config=cfg)
    assert res["messages"][-1].content == "async-ok"
    assert runs == 1

    # Second run on same checkpointed thread skips execution
    res2 = await ag.ainvoke("go again", config=cfg)
    assert runs == 1
    assert len(res2["messages"]) == 3


def test_idempotent_node_preserves_command_update_and_goto() -> None:
    @node(idempotent=True)
    def router(state: dict[str, Any]) -> Command[str]:
        return Command(update={"tag": "routed"}, goto="finish")

    @node
    def finish(state: dict[str, Any]) -> dict[str, Any]:
        return {"tag": f"{state['tag']}:done"}

    ag = (
        AgentBuilder("CmdIdem")
        .with_state_schema({"tag": str})
        .add_node(router)
        .add_node(finish)
        .add_edge(START, router)
        .add_edge(finish, END)
        .compile()
    )
    res = ag.invoke({"tag": ""})
    assert res["tag"] == "routed:done"


async def test_agent_ainvoke_stream_astream_and_state_methods() -> None:
    @node
    def step_a(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [AIMessage(content="from_a")]}

    @node
    def step_b(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [AIMessage(content="from_b")]}

    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        saver = checkpointer("memory")

    ag = AgentBuilder("StreamAgent").add_node(step_a).add_node(step_b).with_checkpointer(saver).compile()

    cfg = {"configurable": {"thread_id": "stream-1"}}
    async_res = await ag.ainvoke("hello", config=cfg)
    assert async_res["messages"][-1].content == "from_b"

    snap_sync = ag.get_state(cfg)
    snap_async = await ag.aget_state(cfg)
    assert snap_sync.values["messages"][-1].content == "from_b"
    assert snap_async.values["messages"][-1].content == "from_b"

    ag.update_state(cfg, {"messages": [AIMessage(content="patched")]})
    assert ag.get_state(cfg).values["messages"][-1].content == "patched"
    await ag.aupdate_state(cfg, {"messages": [AIMessage(content="apatched")]})
    assert (await ag.aget_state(cfg)).values["messages"][-1].content == "apatched"

    sync_chunks = list(ag.stream("hello", config={"configurable": {"thread_id": "stream-2"}}))
    assert len(sync_chunks) == 2

    async_chunks = [
        chunk async for chunk in ag.astream("hello", config={"configurable": {"thread_id": "stream-3"}})
    ]
    assert len(async_chunks) == 2


def test_agent_checkpointer_requires_explicit_thread_id() -> None:
    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        saver = checkpointer("memory")

    ag = AgentBuilder("Checked").add_node("n", lambda s: s).with_checkpointer(saver).compile()
    with pytest.raises(ConfigurationError, match="thread_id"):
        ag.invoke("hi")
    with pytest.raises(ConfigurationError, match="thread_id"):
        ag.invoke("hi", config={"configurable": {}})


def test_agent_stateless_generates_unique_thread_ids() -> None:
    seen_ids: list[str] = []

    class CapturingRunnable:
        def invoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
            seen_ids.append(config["configurable"]["thread_id"])
            return {"messages": [AIMessage(content="ok")]}

    ag = Agent.from_runnable(CapturingRunnable(), name="Cap")
    ag.invoke("first")
    ag.invoke("second")
    assert len(seen_ids) == 2
    assert seen_ids[0] != seen_ids[1]
    assert "default" not in seen_ids


async def test_agent_from_runnable_forwards_payload_and_config() -> None:
    captured: list[tuple[Any, Any]] = []

    class MockRunnable:
        def invoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
            captured.append((payload, config))
            return {"messages": [AIMessage(content="sync-ok")]}

        async def ainvoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
            captured.append((payload, config))
            return {"messages": [AIMessage(content="async-ok")]}

    ag = Agent.from_runnable(MockRunnable(), name="ModernLangChainAgent")
    res = ag.invoke("Test query", config={"configurable": {"thread_id": "t-99"}})
    assert res["messages"][0].content == "sync-ok"
    assert isinstance(captured[0][0]["messages"][0], HumanMessage)
    assert captured[0][1]["configurable"]["thread_id"] == "t-99"

    ares = await ag.ainvoke(HumanMessage(content="Async query"))
    assert ares["messages"][0].content == "async-ok"
    assert captured[1][0]["messages"][0].content == "Async query"


def test_agent_decorator_supports_node_and_model_modes() -> None:
    @agent(name="DecoratedNode")
    def node_fn(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [AIMessage(content="Decorated response")]}

    assert node_fn.invoke("Hi")["messages"][-1].content == "Decorated response"

    @node(name="wrapped_node")
    def my_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [AIMessage(content="Node response")]}

    node_agent = agent(name="NodeAgent")(my_node)
    assert node_agent.invoke("Hi")["messages"][-1].content == "Node response"

    fake = FakeModel(script=[AIMessage(content="LLM response")])

    @agent(name="ModelBot", model=fake)
    def bot_spec() -> None:
        """You are a terse decorated bot."""

    res = bot_spec.invoke("Hello")
    assert res["messages"][-1].content == "LLM response"
    assert fake.calls[0][0].content == "You are a terse decorated bot."

    with pytest.raises(ConfigurationError, match="requires 'model'"):

        @agent(name="Bad", system_prompt="hi")
        def _bad(state: dict[str, Any]) -> dict[str, Any]:
            return state

    with pytest.raises(ConfigurationError, match="expects a node function or Node"):

        @agent(name="BadClass")
        class _NotANode:
            pass


def test_agent_builder_validation_errors() -> None:
    with pytest.raises(ValueError, match="either a model or custom nodes"):
        AgentBuilder("Empty").compile()

    with pytest.raises(ValueError, match="requires a callable"):
        AgentBuilder("NoFn").add_node("missing_fn")

    dup = AgentBuilder("DupAgent").add_node("step", lambda s: s)
    with pytest.raises(ValueError, match="already"):
        dup.add_node("step", lambda s: s)

    bad_target = AgentBuilder("BadTarget").add_node("a", lambda s: s).add_edge("a", "nonexistent")
    with pytest.raises(ValueError, match="not a registered node"):
        bad_target.compile()

    bad_source = AgentBuilder("BadSource").add_node("a", lambda s: s).add_edge("ghost", "a")
    with pytest.raises(ValueError, match="Edge source"):
        bad_source.compile()

    with pytest.raises(ValueError, match="cannot combine custom nodes"):
        AgentBuilder("Hybrid").add_node("a", lambda s: s).with_model(FakeModel(script=["x"])).compile()

    with pytest.raises(ValueError, match="cannot combine custom nodes"):
        AgentBuilder("HybridPrompt").add_node("a", lambda s: s).with_system_prompt("hi").compile()

    class CustomState(State):
        x: int = 0

    with pytest.raises(ValueError, match="Cannot mix"):
        AgentBuilder("Mix1").with_state_schema(CustomState).with_state_schema({"y": int})

    with pytest.raises(ValueError, match="Cannot mix"):
        AgentBuilder("Mix2").with_state_schema({"y": int}).with_state_schema(CustomState)

    with pytest.raises(TypeError, match="with_state_schema expects"):
        AgentBuilder("BadSchema").with_state_schema("invalid")  # type: ignore[arg-type]
