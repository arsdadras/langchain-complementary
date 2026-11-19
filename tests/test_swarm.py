"""Unit tests for langcomp.swarm (SwarmPool, SubAgent, Team)."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from langcomp import AgentBuilder, ConfigurationError, SubAgent, SwarmPool, Team


def test_swarm_pool_crud_validation_and_execution() -> None:
    pool = SwarmPool("TestPool")

    sub = pool.register(
        name="researcher",
        description="Researches information",
        agent_or_callable=lambda q: f"Processed query: {q}",
        capabilities=["research", "search"],
    )
    assert sub.name == "researcher"
    assert "researcher" in pool
    assert len(pool) == 1
    assert pool.get("researcher") is sub
    assert "Processed query: quantum computing" in sub.run("quantum computing")

    # Duplicate registration rejected unless overwrite=True
    with pytest.raises(ConfigurationError, match="already registered"):
        pool.register("researcher", "Dup", lambda q: q)
    pool.register("researcher", "Updated", lambda q: f"v2:{q}", overwrite=True)
    assert pool.get("researcher").description == "Updated"  # type: ignore[union-attr]

    # Defensive copy in list_subagents()
    listing = pool.list_subagents()
    assert len(listing) == 1
    listing[0]["capabilities"].append("mutated")
    assert pool.get("researcher").capabilities == []  # type: ignore[union-attr]

    tools = pool.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "researcher"

    removed = pool.unregister("researcher")
    assert removed is not None
    assert len(pool) == 0


def test_subagent_agent_raw_graph_non_message_dict_and_deepagents_dict() -> None:
    sub_agent_inst = (
        AgentBuilder("SubAgentCore")
        .add_node("step", lambda _s: {"messages": [AIMessage(content="Agent output content")]})
        .compile()
    )
    pool = SwarmPool("AgentPool")
    sub = pool.register("agent_worker", "Worker agent", sub_agent_inst)

    tool_inst = sub.as_tool()
    assert tool_inst.invoke({"task_description": "do work"}) == "Agent output content"

    # Compiled runnable exports CompiledSubAgent-compatible dict (no conflicting system_prompt)
    deep_compiled = sub.to_deepagents_subagent_dict()
    assert deep_compiled["name"] == "agent_worker"
    assert deep_compiled["runnable"] is sub_agent_inst.graph
    assert "system_prompt" not in deep_compiled

    # Plain callable exports declarative dict with system_prompt
    callable_sub = SubAgent(name="reviewer", description="Security reviewer", agent_instance=lambda q: q)
    deep_decl = callable_sub.to_deepagents_subagent_dict()
    assert deep_decl["name"] == "reviewer"
    assert "reviewer" in deep_decl["system_prompt"]
    assert "runnable" not in deep_decl

    # Raw compiled graph & non-message dict output
    raw_graph = sub_agent_inst.graph
    raw_sub = SubAgent(name="raw_worker", description="Raw graph", agent_instance=raw_graph)
    assert raw_sub.run("ping") == "Agent output content"

    class NonMessageRunnable:
        def invoke(self, _payload: Any, config: Any = None) -> dict[str, Any]:
            return {"score": 99}

    dict_sub = SubAgent(name="scorer", description="Scores", agent_instance=NonMessageRunnable())
    assert '"score": 99' in dict_sub.run("eval")

    bad_sub = SubAgent(name="bad", description="Bad", agent_instance=12345)
    with pytest.raises(ValueError, match="Unsupported subagent instance type"):
        bad_sub.run("hi")


async def test_subagent_async_execution_and_sync_rejection_of_async_callable() -> None:
    async def async_worker(query: str) -> str:
        return f"async:{query}"

    sub = SubAgent(name="async_w", description="Async worker", agent_instance=async_worker)
    with pytest.raises(TypeError, match="arun"):
        sub.run("ping")

    assert await sub.arun("ping") == "async:ping"
    assert await sub.as_tool().ainvoke({"task_description": "pong"}) == "async:pong"


def test_team_public_compile_transitive_inheritance_override_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class StubCompiledSubAgent:
        name: str
        description: str
        runnable: Any

    compiled_calls: list[dict[str, Any]] = []

    def stub_create_deep_agent(**kwargs: Any) -> dict[str, Any]:
        compiled_calls.append(kwargs)
        return {"compiled": kwargs}

    stub_mod = types.ModuleType("deepagents")
    stub_mod.CompiledSubAgent = StubCompiledSubAgent  # type: ignore[attr-defined]
    stub_mod.create_deep_agent = stub_create_deep_agent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepagents", stub_mod)

    leaf = Team(name="fact_checker", description="Checks facts")
    mid = Team(name="researcher", description="Researches topics", children=[leaf])
    root = Team(
        name="supervisor",
        description="Coordinates team",
        system_prompt="Be concise.",
        tools=["search_tool"],
        model="old-model",
        skills=["./skills"],
        children=[mid],
    )

    # compile(model="override-model") MUST propagate override-model transitively to leaf!
    graph = root.compile(model="override-model")
    assert len(compiled_calls) == 2
    child_call, root_call = compiled_calls[0], compiled_calls[1]
    assert root_call["model"] == "override-model"
    assert child_call["model"] == "override-model"
    assert child_call["subagents"][0]["model"] == "override-model"
    assert child_call["subagents"][0]["system_prompt"] == "Be concise."
    assert child_call["subagents"][0]["tools"] == ["search_tool"]
    assert child_call["subagents"][0]["skills"] == ["./skills"]
    assert isinstance(graph["compiled"]["subagents"][0], StubCompiledSubAgent)

    # Empty system_prompt="" is preserved (not replaced by default via truthiness)
    quiet = Team(name="quiet", description="No prompt", system_prompt="", model="m")
    compiled_calls.clear()
    quiet.compile()
    assert compiled_calls[0]["system_prompt"] == ""

    # Missing model raises ValueError
    with pytest.raises(ValueError, match="requires a model"):
        Team(name="nomodel", description="No model").compile()

    # Duplicate sibling names & 'general-purpose' shadowing
    with pytest.raises(ValueError, match="duplicate child name"):
        Team(
            name="root",
            description="root",
            model="m",
            children=[Team(name="w", description="a"), Team(name="w", description="b")],
        ).compile()

    with pytest.raises(ValueError, match="general-purpose"):
        Team(
            name="root",
            description="root",
            model="m",
            children=[Team(name="general-purpose", description="shadow")],
        ).compile()

    # Cycle detection
    a = Team(name="a", description="A", model="m")
    b = Team(name="b", description="B")
    a.children = [b]
    b.children = [a]
    with pytest.raises(ConfigurationError, match="Cycle detected"):
        a.compile()

    # Missing deepagents package raises actionable ImportError
    monkeypatch.setitem(sys.modules, "deepagents", None)  # type: ignore[arg-type]
    with pytest.raises(ImportError, match="langcomp\\[deepagents\\]"):
        Team(name="x", description="X", model="m").compile()
