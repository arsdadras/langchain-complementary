"""Unit tests for langcomp.testing (FakeModel, run_and_snapshot, arun_and_snapshot)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from langcomp import (
    AgentBuilder,
    FakeModel,
    ScriptExhaustedError,
    SnapshotMismatchError,
    arun_and_snapshot,
    run_and_snapshot,
)


async def test_fake_model_playback_reset_exhaustion_stream_and_binding() -> None:
    model = FakeModel(script=["first", AIMessage(content="second")])

    r1 = model.invoke("hi")
    r2 = await model.ainvoke("again")
    assert r1.content == "first"
    assert r2.content == "second"
    assert model.call_count == 2
    assert len(model.calls) == 2
    assert model.calls[0][0].content == "hi"
    assert model.calls[1][0].content == "again"

    # Exhaustion must NOT corrupt len(calls) == call_count invariant
    with pytest.raises(ScriptExhaustedError):
        model.invoke("overflow")
    assert len(model.calls) == model.call_count == 2

    # Mutating returned AIMessage must NOT corrupt script on replay after reset()
    r2.content = "mutated"
    model.reset()
    assert model.call_count == 0
    assert model.calls == []
    assert model.bound_tool_names == ()

    # Streaming yields AIMessageChunk
    chunks = list(model.stream("stream-1"))
    assert chunks
    assert all(isinstance(c, AIMessageChunk) for c in chunks)
    assert "".join(str(c.content) for c in chunks) == "first"

    achunks = [c async for c in model.astream("stream-2")]
    assert achunks
    assert all(isinstance(c, AIMessageChunk) for c in achunks)
    assert "".join(str(c.content) for c in achunks) == "second"

    # bind_tools is non-mutating and supports OpenAI JSON-schema dict tools
    @tool
    def alpha() -> str:
        """Alpha tool."""
        return "a"

    b1 = model.bind_tools([alpha])
    b2 = model.bind_tools([{"type": "function", "function": {"name": "dict_tool"}}])
    assert b1 is not model
    assert b2 is not b1
    assert b1.bound_tool_names == ("alpha",)
    assert b2.bound_tool_names == ("dict_tool",)


def test_run_and_snapshot_record_replay_drift_status_and_complex_types(tmp_path: Path) -> None:
    response_text = "deterministic output"

    def responder(state: dict[str, Any]) -> dict[str, Any]:
        del state
        return {
            "messages": [AIMessage(content=response_text)],
            "_langcomp_done": ["responder"],
        }

    ag = AgentBuilder("SnapAgent").add_node("responder", responder).compile()
    snap_file = tmp_path / "snap.json"

    run_and_snapshot(ag, "hello", snap_file)
    assert snap_file.exists()
    raw = json.loads(snap_file.read_text(encoding="utf-8"))
    assert "_langcomp_done" not in raw

    # Identical replay passes
    run_and_snapshot(ag, "hello", snap_file)

    # Drifted output raises SnapshotMismatchError with unified diff
    response_text = "changed output"
    with pytest.raises(SnapshotMismatchError, match="--- "):
        run_and_snapshot(ag, "hello", snap_file)

    # update_snapshot overwrites cleanly
    run_and_snapshot(ag, "hello", snap_file, update_snapshot=True)
    run_and_snapshot(ag, "hello", snap_file)

    # ToolMessage status drift MUST be detected even when content matches
    tool_snap = tmp_path / "tool_snap.json"

    class ToolStatusGraph:
        def __init__(self, status: str) -> None:
            self.status = status

        def invoke(self, _data: Any) -> dict[str, Any]:
            return {
                "messages": [
                    ToolMessage(content="x", tool_call_id="c1", status=self.status, artifact={"k": 1})
                ]
            }

    run_and_snapshot(ToolStatusGraph("success"), {}, tool_snap)
    with pytest.raises(SnapshotMismatchError):
        run_and_snapshot(ToolStatusGraph("error"), {}, tool_snap)

    # Tuple, set, BaseModel, dataclass serialize deterministically (without random message IDs)
    class PayloadModel(BaseModel):
        score: int

    @dataclass
    class PayloadDC:
        label: str

    complex_snap = tmp_path / "complex.json"

    class ComplexGraph:
        def invoke(self, _data: Any) -> dict[str, Any]:
            return {
                "messages": (HumanMessage(content="hi"),),
                "tags": {"b", "a"},
                "model": PayloadModel(score=5),
                "dc": PayloadDC(label="v1"),
            }

    run_and_snapshot(ComplexGraph(), {}, complex_snap)
    run_and_snapshot(ComplexGraph(), {}, complex_snap)


async def test_arun_and_snapshot(tmp_path: Path) -> None:
    ag = (
        AgentBuilder("AsyncSnap")
        .add_node("n", lambda _s: {"messages": [AIMessage(content="async-snap")]})
        .compile()
    )
    snap = tmp_path / "async_snap.json"
    await arun_and_snapshot(ag, "hi", snap)
    await arun_and_snapshot(ag, "hi", snap)
