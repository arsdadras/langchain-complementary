"""
Tests for FakeModel, run_and_snapshot, and TraceCollector / summarize().
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from langcomp import (
    AgentBuilder,
    FakeModel,
    ScriptExhaustedError,
    SnapshotMismatchError,
    TraceCollector,
    run_and_snapshot,
    summarize,
)


def test_fake_model_playback_reset_and_exhaustion():
    model = FakeModel(script=["first", AIMessage(content="second")])

    r1 = model.invoke("hi")
    r2 = model.invoke("again")
    assert r1.content == "first"
    assert r2.content == "second"
    assert model.call_count == 2

    with pytest.raises(ScriptExhaustedError):
        model.invoke("overflow")

    model.reset()
    assert model.call_count == 0
    assert model.invoke("restart").content == "first"


def test_run_and_snapshot_record_replay_and_drift(tmp_path: Path):
    response_text = "deterministic output"

    def responder(state):
        return {
            "messages": [AIMessage(content=response_text)],
            "_langcomp_done": ["responder"],
        }

    agent = AgentBuilder("SnapAgent").add_node("responder", responder).compile()
    snap_file = tmp_path / "snap.json"

    # First run records snapshot and strips internal _langcomp keys
    run_and_snapshot(agent, "hello", snap_file)
    assert snap_file.exists()
    raw = json.loads(snap_file.read_text(encoding="utf-8"))
    assert "_langcomp_done" not in raw

    # Second identical run succeeds
    run_and_snapshot(agent, "hello", snap_file)

    # Drifted output raises SnapshotMismatchError
    response_text = "changed output"
    with pytest.raises(SnapshotMismatchError):
        run_and_snapshot(agent, "hello", snap_file)


def test_trace_collector_captures_tokens_cost_node_and_writes_jsonl(tmp_path: Path):
    fake = FakeModel(
        script=[
            AIMessage(
                content="traced reply",
                usage_metadata={
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "total_tokens": 300,
                },
            )
        ]
    )

    def llm_node(state):
        reply = fake.invoke(state["messages"])
        return {"messages": [reply]}

    agent = AgentBuilder("TracedAgent").add_node("llm_node", llm_node).compile()
    collector = TraceCollector(
        run_id="run-test-1",
        price_table={"FakeModel": (0.01, 0.03), "default": (0.001, 0.002)},
        out_dir=tmp_path,
    )

    result = agent.invoke("hello tracer", config={"callbacks": [collector]})
    assert result["messages"][-1].content == "traced reply"
    assert len(collector.events) == 1

    ev = collector.events[0]
    assert ev["run_id"] == "run-test-1"
    assert ev["node"] == "llm_node"
    assert ev["model"] == "FakeModel"
    assert ev["input_tokens"] == 200
    assert ev["output_tokens"] == 100
    # 200/1000 * 0.01 + 100/1000 * 0.03 = 0.002 + 0.003 = 0.005
    assert ev["cost_usd"] == pytest.approx(0.005)

    jsonl_file = tmp_path / "run-test-1.jsonl"
    assert jsonl_file.exists()
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["node"] == "llm_node"

    summary = summarize(collector)
    assert summary["total_calls"] == 1
    assert summary["total_input_tokens"] == 200
    assert summary["total_output_tokens"] == 100
    assert summary["total_cost_usd"] == pytest.approx(0.005)
    assert "llm_node" in summary["by_node"]
    assert "FakeModel" in summary["by_model"]
