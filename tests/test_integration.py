"""Cross-module end-to-end integration tests for langcomp."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from langcomp import (
    AgentBuilder,
    ApprovalGate,
    ApprovalGateMiddleware,
    ContextBufferMiddleware,
    FakeModel,
    GraphVisualizer,
    SwarmPool,
    ToolGuardMiddleware,
    TraceCollector,
    checkpointer,
    create_agent,
    run_and_snapshot,
    summarize,
)


def test_traced_react_loop_with_callbacks_and_mermaid(add_tool: BaseTool) -> None:
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
    ag = AgentBuilder("TracedLoop").with_model(model).with_tools([add_tool]).compile()
    collector = TraceCollector(out_dir=None)

    result = ag.invoke("4 + 5?", config={"callbacks": [collector]})
    assert result["messages"][-1].content == "9"

    summary = summarize(collector)
    assert summary["total_calls"] == 2
    assert summary["total_input_tokens"] == 80
    assert summary["total_output_tokens"] == 15
    assert "model" in summary["by_node"]

    mermaid = GraphVisualizer.to_mermaid(ag)
    assert "model" in mermaid
    assert "tools" in mermaid


async def test_supervisor_with_swarm_pool_middleware_and_snapshot(tmp_path: Path) -> None:
    pool = SwarmPool("ResearchSwarm")

    async def researcher(query: str) -> str:
        return f"findings:{query}"

    pool.register("researcher", "Researches topics", researcher, capabilities=["research"])

    model = FakeModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "researcher",
                        "args": {"task_description": "langgraph"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Summary: findings:langgraph"),
        ]
    )
    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        saver = checkpointer("memory")

    supervisor = create_agent(
        model=model,
        tools=pool.get_tools(),
        middleware=[
            ContextBufferMiddleware(max_messages=10),
            ApprovalGateMiddleware(ApprovalGate(approval_func=lambda _n, _p: True)),
            ToolGuardMiddleware(max_retries=2, retry_delay_seconds=0.0),
        ],
        checkpointer=saver,
    )

    cfg = {"configurable": {"thread_id": "e2e-thread"}}
    result = await supervisor.ainvoke("Research langgraph", config=cfg)
    assert result["messages"][-1].content == "Summary: findings:langgraph"

    # Also verify golden-master snapshot works on sync Agent with explicit config
    sync_agent = create_agent(FakeModel(script=["stable"]), checkpointer=saver)
    snap_path = tmp_path / "e2e_snap.json"
    run_and_snapshot(sync_agent, "ping", snap_path, config={"configurable": {"thread_id": "snap-1"}})
    sync_agent2 = create_agent(FakeModel(script=["stable"]), checkpointer=saver)
    run_and_snapshot(sync_agent2, "ping", snap_path, config={"configurable": {"thread_id": "snap-2"}})
