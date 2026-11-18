"""
Tests verifying edge cases identified during code-simplifier review.
"""

from typing import Annotated, Any, Dict, List

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages

from langcomp import (
    AgentBuilder,
    ApprovalGate,
    ApprovalRejectedError,
    AutoSummarizer,
    Reduce,
    ReducerConflictError,
    SmartState,
    State,
    SubAgent,
    ToolGuard,
    TraceCollector,
    node,
)


class _FlakyCustomTool(BaseTool):
    name: str = "flaky_custom"
    description: str = "Fails every call to verify retry count under ainvoke."
    attempts: int = 0

    def _run(self, query: str) -> str:
        object.__setattr__(self, "attempts", self.attempts + 1)
        raise ValueError("boom")


@pytest.mark.asyncio
async def test_tool_guard_does_not_double_wrap_base_tool_arun():
    custom = _FlakyCustomTool()
    guarded = ToolGuard(max_retries=3, retry_delay_seconds=0.0).wrap_tool(custom)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        await guarded.ainvoke({"query": "test"})

    assert custom.attempts == 3


def test_tool_guard_propagates_approval_rejected_error_without_retrying():
    calls = 0

    def sensitive(x: int) -> int:
        nonlocal calls
        calls += 1
        return x

    gate = ApprovalGate(approval_func=lambda action, params: False)
    guard = ToolGuard(max_retries=3, fallback_output="fallback", retry_delay_seconds=0.0)
    wrapped = guard.wrap_function(gate.wrap_function(sensitive))

    with pytest.raises(ApprovalRejectedError):
        wrapped(1)
    assert calls == 0


@pytest.mark.asyncio
async def test_async_idempotent_node():
    runs = 0

    @node(idempotent=True)
    async def async_step(state: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal runs
        runs += 1
        return {"messages": [AIMessage(content="async-ok")]}

    agent = AgentBuilder("AsyncIdem").add_node(async_step).compile()
    res = await agent.ainvoke("go")
    assert res["messages"][-1].content == "async-ok"
    assert runs == 1

    # Already marked done -> skips execution
    res2 = await agent.ainvoke({"messages": ["go"], "_langcomp_done": ["async_step"]})
    assert runs == 1
    assert len(res2["messages"]) == 1


def test_state_preserves_annotated_reducer_and_reduce_none_conflict():
    class MsgState(State):
        messages: Annotated[List[AnyMessage], add_messages] = []

    assert MsgState._reducers()["messages"] is add_messages
    assert "add_messages" in MsgState.explain()["messages"]

    with pytest.raises(ReducerConflictError):
        Reduce(lambda a, b: b, default=None, default_factory=list)


def test_smart_state_normalizes_tuples_and_tool_dicts():
    msgs = SmartState.normalize_messages(
        [
            ("user", "hello tuple"),
            ("assistant", "reply tuple"),
            {"role": "tool", "content": "42", "tool_call_id": "call_9", "name": "calc"},
        ]
    )
    assert len(msgs) == 3
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert isinstance(msgs[2], ToolMessage)
    assert msgs[2].tool_call_id == "call_9"
    assert msgs[2].name == "calc"


def test_auto_summarizer_folds_previous_summary_on_multi_pass():
    summarizer = AutoSummarizer(trigger_threshold=3, keep_recent=1)
    pass1 = summarizer.summarize_messages(
        [HumanMessage(content=f"m{i}") for i in range(4)],
        summarizer_func=lambda txt: "SUM1",
    )
    assert len(pass1) == 2
    assert isinstance(pass1[0], SystemMessage)

    pass2 = summarizer.summarize_messages(
        pass1 + [HumanMessage(content="m4"), HumanMessage(content="m5")],
        summarizer_func=lambda txt: "SUM2",
    )
    # Must fold SUM1 into SUM2 instead of stacking two summary SystemMessages
    assert len(pass2) == 2
    assert isinstance(pass2[0], SystemMessage)
    assert "SUM2" in pass2[0].content
    assert "SUM1" not in pass2[0].content


def test_trace_collector_cleans_up_inflight_on_llm_error():
    collector = TraceCollector(out_dir=None)
    collector.on_llm_start({"name": "FakeModel"}, ["hi"], run_id="r-err")
    assert "r-err" in collector._inflight
    collector.on_llm_error(RuntimeError("boom"), run_id="r-err")
    assert "r-err" not in collector._inflight


def test_subagent_supports_raw_compiled_graph():
    raw_graph = (
        AgentBuilder("RawWorker")
        .add_node("reply", lambda s: {"messages": [AIMessage(content="raw-graph-ok")]})
        .compile()
        .graph
    )
    sub = SubAgent(name="raw_worker", description="Raw graph", agent_instance=raw_graph)
    assert sub.run("ping") == "raw-graph-ok"
