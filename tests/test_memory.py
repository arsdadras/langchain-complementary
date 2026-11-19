"""Unit tests for langcomp.memory.buffer (ContextBuffer & AutoSummarizer)."""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langcomp.memory.buffer import SUMMARY_PREFIX, AutoSummarizer, ContextBuffer


def _assert_no_orphaned_tool_messages(messages: list) -> None:
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            assert i > 0, "ToolMessage cannot be the first message"
            prev = messages[i - 1]
            assert isinstance(prev, (ToolMessage, AIMessage)), (
                f"Orphaned ToolMessage after {type(prev).__name__}"
            )
            if isinstance(prev, AIMessage):
                assert prev.tool_calls, "Preceding AIMessage must have tool_calls"


def test_context_buffer_trimming_and_validation() -> None:
    with pytest.raises(ValueError, match="max_messages"):
        ContextBuffer(max_messages=0)

    sys_msg = SystemMessage(content="System instruction")
    messages = [sys_msg, *(HumanMessage(content=f"Msg {i}") for i in range(15))]

    buf = ContextBuffer(max_messages=5, preserve_system_prompt=True)
    trimmed = buf.trim(messages)
    assert len(trimmed) == 5
    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed[0].content == "System instruction"
    assert trimmed[-1].content == "Msg 14"

    # Without preserving system prompt
    buf_no_sys = ContextBuffer(max_messages=3, preserve_system_prompt=False)
    trimmed2 = buf_no_sys.trim(messages)
    assert len(trimmed2) == 3
    assert all(isinstance(m, HumanMessage) for m in trimmed2)

    # Short list returns copy unchanged
    short = [HumanMessage(content="only one")]
    assert buf.trim(short) == short


def test_context_buffer_never_orphans_tool_messages() -> None:
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "calc", "args": {"x": 1}, "id": "c1"}]),
        ToolMessage(content="42", tool_call_id="c1"),
        AIMessage(content="answer"),
    ]
    trimmed = ContextBuffer(max_messages=3).trim(msgs)
    assert len(trimmed) <= 3
    _assert_no_orphaned_tool_messages(trimmed)
    assert trimmed[0].content == "sys"
    assert trimmed[-1].content == "answer"


def test_auto_summarizer_basic_keep_recent_zero_and_validation() -> None:
    with pytest.raises(ValueError, match="trigger_threshold"):
        AutoSummarizer(trigger_threshold=0)
    with pytest.raises(ValueError, match="keep_recent"):
        AutoSummarizer(trigger_threshold=5, keep_recent=-1)

    messages = [HumanMessage(content=f"Turn {i}") for i in range(20)]
    summarizer = AutoSummarizer(trigger_threshold=10, keep_recent=3)
    summarized = summarizer.summarize_messages(messages)
    assert len(summarized) == 4
    assert isinstance(summarized[0], SystemMessage)
    assert SUMMARY_PREFIX in str(summarized[0].content)
    assert summarized[-1].content == "Turn 19"

    # keep_recent=0 summarizes all messages into 1 summary SystemMessage
    all_summarized = AutoSummarizer(trigger_threshold=3, keep_recent=0).summarize_messages(
        [HumanMessage(content=f"Msg {i}") for i in range(6)]
    )
    assert len(all_summarized) == 1
    assert isinstance(all_summarized[0], SystemMessage)
    assert "Msg 0" in str(all_summarized[0].content)
    assert "Msg 5" in str(all_summarized[0].content)
    assert not str(all_summarized[0].content).endswith("...")


def test_auto_summarizer_preserves_tool_pairs_and_single_system_message() -> None:
    msgs = [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "calc", "args": {"a": 1}, "id": "c1"}]),
        ToolMessage(content="42", tool_call_id="c1"),
        AIMessage(content="ans"),
    ]
    summarizer = AutoSummarizer(trigger_threshold=2, keep_recent=2)
    res = summarizer.summarize_messages(msgs, summarizer_func=lambda txt: f"SUM({txt})")

    sys_msgs = [m for m in res if isinstance(m, SystemMessage)]
    assert len(sys_msgs) == 1
    assert "You are helpful." in str(sys_msgs[0].content)
    assert SUMMARY_PREFIX in str(sys_msgs[0].content)
    _assert_no_orphaned_tool_messages(res)


def test_auto_summarizer_multi_pass_folding_with_and_without_system_prompt() -> None:
    summarizer = AutoSummarizer(trigger_threshold=3, keep_recent=1)
    pass1 = summarizer.summarize_messages(
        [HumanMessage(content=f"m{i}") for i in range(4)],
        summarizer_func=lambda _txt: "SUM1",
    )
    assert len(pass1) == 2

    pass2 = summarizer.summarize_messages(
        [*pass1, HumanMessage(content="m4"), HumanMessage(content="m5")],
        summarizer_func=lambda _txt: "SUM2",
    )
    assert len(pass2) == 2
    assert "SUM2" in str(pass2[0].content)
    assert "SUM1" not in str(pass2[0].content)

    # With a leading user SystemMessage, multi-pass keeps user prompt + replaces prior summary
    with_sys_1 = summarizer.summarize_messages(
        [SystemMessage(content="Core rules"), *(HumanMessage(content=f"u{i}") for i in range(4))],
        summarizer_func=lambda _txt: "S1",
    )
    assert sum(1 for m in with_sys_1 if isinstance(m, SystemMessage)) == 1
    with_sys_2 = summarizer.summarize_messages(
        [*with_sys_1, HumanMessage(content="u4"), HumanMessage(content="u5")],
        summarizer_func=lambda _txt: "S2",
    )
    assert sum(1 for m in with_sys_2 if isinstance(m, SystemMessage)) == 1
    assert "Core rules" in str(with_sys_2[0].content)
    assert "S2" in str(with_sys_2[0].content)
    assert "S1" not in str(with_sys_2[0].content)


def test_auto_summarizer_logs_warning_on_failure_and_formats_multimodal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    msgs = [
        HumanMessage(content=[{"type": "text", "text": "hello block"}]),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"k": "v"}, "id": "c1"}]),
        ToolMessage(content="found", tool_call_id="c1"),
        HumanMessage(content="next"),
    ]
    summarizer = AutoSummarizer(trigger_threshold=2, keep_recent=1)

    def broken(_txt: str) -> str:
        raise RuntimeError("summarizer offline")

    with caplog.at_level(logging.WARNING, logger="langcomp.memory"):
        res = summarizer.summarize_messages(msgs, summarizer_func=broken)

    assert "summarizer_func failed" in caplog.text
    assert "hello block" in str(res[0].content)
    assert "lookup" in str(res[0].content)


async def test_auto_summarizer_async_support() -> None:
    async def async_summarize(text: str) -> str:
        return f"ASYNC[{len(text)}]"

    summarizer = AutoSummarizer(trigger_threshold=2, keep_recent=1)
    msgs = [HumanMessage(content=f"m{i}") for i in range(4)]

    with pytest.raises(TypeError, match="asummarize_messages"):
        summarizer.summarize_messages(msgs, summarizer_func=async_summarize)  # type: ignore[arg-type]

    res = await summarizer.asummarize_messages(msgs, summarizer_func=async_summarize)
    assert len(res) == 2
    assert "ASYNC[" in str(res[0].content)
