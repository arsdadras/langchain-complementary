"""Unit tests for langcomp.devtools (TraceCollector, summarize, GraphVisualizer, DryRunner)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langcomp import (
    AgentBuilder,
    DryRunner,
    FakeModel,
    GraphVisualizer,
    TraceCollector,
    node,
    summarize,
)


def test_trace_collector_tokens_cost_jsonl_and_summarize(tmp_path: Path) -> None:
    fake = FakeModel(
        script=[
            AIMessage(
                content="traced reply",
                usage_metadata={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
            )
        ]
    )

    def llm_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [fake.invoke(state["messages"])]}

    ag = AgentBuilder("TracedAgent").add_node("llm_node", llm_node).compile()
    collector = TraceCollector(
        run_id="run-test-1",
        price_table={"FakeModel": (0.01, 0.03), "default": (0.001, 0.002)},
        out_dir=tmp_path,
    )
    # Out dir is not written until an event occurs
    assert not (tmp_path / "run-test-1.jsonl").exists()

    result = ag.invoke("hello tracer", config={"callbacks": [collector]})
    assert result["messages"][-1].content == "traced reply"
    assert len(collector.events) == 1

    ev = collector.events[0]
    assert ev["run_id"] == "run-test-1"
    assert ev["node"] == "llm_node"
    assert ev["model"] == "FakeModel"
    assert ev["input_tokens"] == 200
    assert ev["output_tokens"] == 100
    assert ev["cost_usd"] == pytest.approx(0.005)
    assert ev["status"] == "ok"

    jsonl_file = tmp_path / "run-test-1.jsonl"
    assert jsonl_file.exists()
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["node"] == "llm_node"

    summary = collector.summarize()
    assert summary == summarize(collector)
    assert summary["total_calls"] == 1
    assert summary["error_calls"] == 0
    assert summary["total_input_tokens"] == 200
    assert summary["total_output_tokens"] == 100
    assert summary["total_cost_usd"] == pytest.approx(0.005)
    assert "llm_node" in summary["by_node"]
    assert "FakeModel" in summary["by_model"]
    assert "latency_s" in summary["by_model"]["FakeModel"]


def test_trace_collector_hardening_run_id_real_model_name_none_tokens_and_errors(
    tmp_path: Path,
) -> None:
    # Path traversal rejected
    for bad_id in ("../../escape", "a/b", "a\\b", "", ".."):
        with pytest.raises(ValueError, match="Invalid run_id"):
            TraceCollector(run_id=bad_id, out_dir=tmp_path)

    collector = TraceCollector(
        out_dir=None,
        price_table={"gpt-4o": (0.005, 0.015)},
    )
    # Real response_metadata["model_name"] overrides serialized["name"] ("ChatOpenAI"),
    # and prefix matching handles dated IDs ("gpt-4o-2024-08-06"), and None token counts don't crash.
    collector.on_chat_model_start({"name": "ChatOpenAI"}, [[]], run_id="r1")
    fake_msg = SimpleNamespace(
        response_metadata={"model_name": "gpt-4o-2024-08-06"},
        usage_metadata={"input_tokens": None, "output_tokens": 1000},
    )
    fake_resp = SimpleNamespace(generations=[[SimpleNamespace(message=fake_msg)]], llm_output=None)
    collector.on_llm_end(fake_resp, run_id="r1")

    assert collector.events[0]["model"] == "gpt-4o-2024-08-06"
    assert collector.events[0]["input_tokens"] == 0
    assert collector.events[0]["output_tokens"] == 1000
    assert collector.events[0]["cost_usd"] == pytest.approx(0.015)

    # llm_output.token_usage fallback
    collector.on_llm_start({"name": "ChatOpenAI"}, ["hi"], run_id="r2")
    collector.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
            llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        ),
        run_id="r2",
    )
    assert collector.events[1]["input_tokens"] == 100
    assert collector.events[1]["output_tokens"] == 50

    # on_llm_error records an error event and clears inflight
    collector.on_llm_start({"name": "FakeModel"}, ["hi"], run_id="r-err")
    collector.on_llm_error(RuntimeError("boom"), run_id="r-err")
    assert collector.events[-1]["status"] == "error"
    assert "RuntimeError: boom" in collector.events[-1]["error"]
    assert collector.summarize()["error_calls"] == 1


def test_graph_visualizer_mermaid_ascii_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    ag = (
        AgentBuilder("TopoAgent")
        .add_node("ingest", lambda _s: {"messages": [AIMessage(content="ok")]})
        .add_node("publish", lambda _s: {"messages": [AIMessage(content="done")]})
        .compile()
    )

    mermaid = GraphVisualizer.to_mermaid(ag)
    assert "ingest" in mermaid
    assert "publish" in mermaid

    ascii_out = GraphVisualizer.to_ascii(ag)
    assert "ingest" in ascii_out
    assert "publish" in ascii_out

    # Test zero-dependency fallback when grandalf is missing
    orig_get_graph = ag.graph.get_graph

    def _get_graph_no_grandalf(*args: Any, **kwargs: Any) -> Any:
        drawable = orig_get_graph(*args, **kwargs)
        monkeypatch.setattr(drawable, "draw_ascii", lambda: (_ for _ in ()).throw(ImportError("no grandalf")))
        return drawable

    monkeypatch.setattr(ag.graph, "get_graph", _get_graph_no_grandalf)
    fallback_ascii = GraphVisualizer.to_ascii(ag)
    assert "[ingest]" in fallback_ascii
    assert "[publish]" in fallback_ascii

    with pytest.raises(TypeError, match="get_graph"):
        GraphVisualizer.to_mermaid("not-a-graph")


async def test_dry_runner_sync_async_raw_graph_and_errors() -> None:
    @node
    def step_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"counter": state.get("counter", 0) + 1}

    @node
    async def async_step(state: dict[str, Any]) -> dict[str, Any]:
        return {"counter": state.get("counter", 0) + 10}

    ag = (
        AgentBuilder("DryRunAgent")
        .with_state_schema({"counter": int})
        .add_node(step_node)
        .add_node(async_step)
        .compile()
    )

    runner = DryRunner(ag)
    assert runner.run_step("step_node", {"counter": 5}) == {"counter": 6}
    assert await runner.arun_step("async_step", {"counter": 5}) == {"counter": 15}

    # Works with raw compiled LangGraph too
    raw_runner = DryRunner(ag.graph)
    assert raw_runner.run_step("step_node", {"counter": 1}) == {"counter": 2}

    with pytest.raises(ValueError, match="not found"):
        runner.run_step("nonexistent", {})
