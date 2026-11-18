"""
Regression tests for bugs identified in existing langcomp modules.
"""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

import langcomp
from langcomp.core.agent import AgentBuilder
from langcomp.memory.buffer import AutoSummarizer, ContextBuffer
from langcomp.tools.guard import ToolGuard


def test_agent_decorator_exported_at_package_root():
    assert hasattr(langcomp, "agent")
    assert "agent" in langcomp.__all__


def test_auto_summarizer_keep_recent_zero_summarizes_all():
    messages = [HumanMessage(content=f"Msg {i}") for i in range(6)]
    summarizer = AutoSummarizer(trigger_threshold=3, keep_recent=0)
    result = summarizer.summarize_messages(messages)

    # Should produce ONLY the single summary SystemMessage (not all 6 messages!)
    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)
    assert "Msg 0" in result[0].content
    assert "Msg 5" in result[0].content


def test_context_buffer_rejects_non_positive_max_messages():
    with pytest.raises(ValueError):
        ContextBuffer(max_messages=0)


def test_tool_guard_preserves_function_metadata_and_validates_retries():
    with pytest.raises(ValueError):
        ToolGuard(max_retries=0)

    def sample_tool(x: int) -> int:
        """Double x."""
        return x * 2

    guard = ToolGuard(max_retries=2, retry_delay_seconds=0.0)
    wrapped = guard.wrap_function(sample_tool)
    assert wrapped.__name__ == "sample_tool"
    assert wrapped.__doc__ == "Double x."


def test_tool_guard_wraps_langchain_basetool_with_retries():
    attempts = 0

    @tool
    def flaky_search(query: str) -> str:
        """Search with transient failure."""
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("transient")
        return f"found:{query}"

    guard = ToolGuard(max_retries=3, retry_delay_seconds=0.0)
    guarded = guard.wrap_tool(flaky_search)
    assert guarded.invoke({"query": "langgraph"}) == "found:langgraph"
    assert attempts == 2


def test_agent_builder_rejects_duplicate_node_names():
    builder = AgentBuilder("DupAgent")
    builder.add_node("step", lambda s: s)
    with pytest.raises(ValueError, match="already"):
        builder.add_node("step", lambda s: s)
