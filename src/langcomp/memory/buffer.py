"""
Memory & Context Management for the langcomp framework.
Provides tool-call-pair-aware sliding window trimming and sync/async conversation summarization.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

logger = logging.getLogger("langcomp.memory")

SUMMARY_PREFIX = "Previous Conversation Summary:\n"
_MERGED_SUMMARY_SEP = f"\n\n{SUMMARY_PREFIX}"
_MAX_FALLBACK_CHARS = 500


def _format_message_content(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "type" in block:
                    parts.append(f"[{block['type']}]")
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        text = " ".join(p for p in parts if p)
    else:
        text = str(content)

    if isinstance(msg, AIMessage) and msg.tool_calls:
        calls_repr = ", ".join(
            f"{c.get('name', '?')}({c.get('args', {})})" for c in msg.tool_calls
        )
        text = f"{text} [tool_calls: {calls_repr}]".strip() if text else f"[tool_calls: {calls_repr}]"
    return text


def _safe_tail_start_for_trim(messages: Sequence[BaseMessage], start_idx: int) -> int:
    """Advance ``start_idx`` past orphaned ``ToolMessage``s so tail never exceeds ``max_messages``."""
    idx = max(0, start_idx)
    while idx < len(messages) and isinstance(messages[idx], ToolMessage):
        idx += 1
    return idx


def _safe_split_for_summary(messages: Sequence[BaseMessage], split_idx: int) -> int:
    """Adjust ``split_idx`` so ``messages[split_idx:]`` never starts on an orphaned ``ToolMessage``.

    Prefers pulling the preceding ``AIMessage(tool_calls=...)`` into the recent window
    when earlier messages remain to summarize; otherwise advances past the tool messages
    so the entire tool-call exchange is summarized together.
    """
    n = len(messages)
    idx = max(0, min(split_idx, n))
    if idx >= n or not isinstance(messages[idx], ToolMessage):
        return idx

    # Walk backward to find the owning AIMessage(tool_calls=...)
    back = idx - 1
    while back >= 0 and isinstance(messages[back], ToolMessage):
        back -= 1
    if back > 0:
        prev_cand = messages[back]
        if isinstance(prev_cand, AIMessage) and bool(prev_cand.tool_calls):
            return back

    # Walking backward would leave old_messages empty — advance forward past ToolMessages
    fwd = idx
    while fwd < n and isinstance(messages[fwd], ToolMessage):
        fwd += 1
    return fwd


class ContextBuffer:
    """Sliding window message buffer that trims conversation history safely.

    Never splits an ``AIMessage(tool_calls=...)`` from its following ``ToolMessage``s,
    preventing HTTP 400 errors from chat completion APIs.
    """

    def __init__(self, max_messages: int = 10, preserve_system_prompt: bool = True):
        if max_messages < 1:
            raise ValueError(f"max_messages must be >= 1, got {max_messages}")
        self.max_messages = max_messages
        self.preserve_system_prompt = preserve_system_prompt

    def trim(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        """Trim messages to at most ``max_messages`` while preserving tool-call integrity."""
        msg_list = list(messages)
        if not msg_list or len(msg_list) <= self.max_messages:
            return msg_list

        system_msg: BaseMessage | None = None
        if self.preserve_system_prompt and isinstance(msg_list[0], SystemMessage):
            system_msg = msg_list[0]
            remaining = msg_list[1:]
        else:
            remaining = msg_list

        keep_count = self.max_messages - (1 if system_msg else 0)
        if keep_count <= 0:
            return [system_msg] if system_msg else []

        raw_start = max(0, len(remaining) - keep_count)
        safe_start = _safe_tail_start_for_trim(remaining, raw_start)
        trimmed_tail = remaining[safe_start:]

        if system_msg is not None:
            return [system_msg, *trimmed_tail]
        return list(trimmed_tail)


class AutoSummarizer:
    """Automatically compresses older conversation history into a single summary SystemMessage."""

    def __init__(self, trigger_threshold: int = 15, keep_recent: int = 5):
        if trigger_threshold < 1:
            raise ValueError(f"trigger_threshold must be >= 1, got {trigger_threshold}")
        if keep_recent < 0:
            raise ValueError(f"keep_recent must be >= 0, got {keep_recent}")
        self.trigger_threshold = trigger_threshold
        self.keep_recent = keep_recent

    @staticmethod
    def _is_summary_message(msg: BaseMessage) -> bool:
        return (
            isinstance(msg, SystemMessage)
            and isinstance(msg.content, str)
            and msg.content.startswith(SUMMARY_PREFIX)
        )

    @classmethod
    def _prepare_split(
        cls,
        messages: Sequence[BaseMessage],
        trigger_threshold: int,
        keep_recent: int,
    ) -> tuple[str | None, list[BaseMessage], list[BaseMessage]] | None:
        msg_list = list(messages)
        if len(msg_list) <= trigger_threshold:
            return None

        base_system_prompt: str | None = None
        work_messages: list[BaseMessage]
        first = msg_list[0]
        if isinstance(first, SystemMessage) and not cls._is_summary_message(first):
            first_text = _format_message_content(first)
            if _MERGED_SUMMARY_SEP in first_text:
                base_part, prior_summary = first_text.split(_MERGED_SUMMARY_SEP, 1)
                base_system_prompt = base_part
                work_messages = [
                    SystemMessage(content=f"{SUMMARY_PREFIX}{prior_summary}"),
                    *msg_list[1:],
                ]
            else:
                base_system_prompt = first_text
                work_messages = msg_list[1:]
        else:
            work_messages = msg_list

        if keep_recent == 0:
            old_messages = work_messages[:]
            recent_messages: list[BaseMessage] = []
        else:
            raw_split = max(0, len(work_messages) - keep_recent)
            safe_split = _safe_split_for_summary(work_messages, raw_split)
            old_messages = work_messages[:safe_split]
            recent_messages = work_messages[safe_split:]

        if not old_messages:
            return None
        return base_system_prompt, old_messages, recent_messages

    @classmethod
    def _build_transcript(cls, old_messages: Sequence[BaseMessage]) -> str:
        lines: list[str] = []
        for msg in old_messages:
            text = _format_message_content(msg)
            if cls._is_summary_message(msg):
                body = text[len(SUMMARY_PREFIX) :]
                lines.append(f"PriorSummary: {body}")
            else:
                role = msg.__class__.__name__.replace("Message", "")
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(summary_text: str) -> str:
        if len(summary_text) > _MAX_FALLBACK_CHARS:
            return f"{summary_text[:_MAX_FALLBACK_CHARS]}..."
        return summary_text

    @staticmethod
    def _assemble_result(
        base_system_prompt: str | None,
        computed_summary: str,
        recent_messages: Sequence[BaseMessage],
    ) -> list[BaseMessage]:
        if base_system_prompt is not None:
            combined = f"{base_system_prompt}{_MERGED_SUMMARY_SEP}{computed_summary}"
        else:
            combined = f"{SUMMARY_PREFIX}{computed_summary}"
        return [SystemMessage(content=combined), *recent_messages]

    def summarize_messages(
        self,
        messages: Sequence[BaseMessage],
        summarizer_func: Callable[[str], str] | None = None,
    ) -> list[BaseMessage]:
        """Summarize older messages synchronously when conversation length exceeds ``trigger_threshold``.

        On repeated passes, prior auto-generated summaries are folded cleanly without
        stacking duplicate ``SystemMessage`` instances or orphaning ``ToolMessage``s.
        """
        prepared = self._prepare_split(messages, self.trigger_threshold, self.keep_recent)
        if prepared is None:
            return list(messages)

        base_system_prompt, old_messages, recent_messages = prepared
        summary_text = self._build_transcript(old_messages)

        if summarizer_func is not None:
            try:
                computed_summary = summarizer_func(summary_text)
                if inspect.isawaitable(computed_summary):
                    if inspect.iscoroutine(computed_summary):
                        computed_summary.close()
                    raise TypeError(
                        "Async summarizer_func cannot be used in summarize_messages(); "
                        "use asummarize_messages()."
                    )
            except TypeError:
                raise
            except Exception as exc:  # noqa: BLE001 - degrade gracefully on summarizer failure
                logger.warning(
                    "AutoSummarizer summarizer_func failed (%s: %s); falling back to truncation.",
                    type(exc).__name__,
                    exc,
                )
                computed_summary = self._fallback_summary(summary_text)
        else:
            computed_summary = self._fallback_summary(summary_text)

        return self._assemble_result(base_system_prompt, computed_summary, recent_messages)

    async def asummarize_messages(
        self,
        messages: Sequence[BaseMessage],
        summarizer_func: Callable[[str], str | Awaitable[str]] | None = None,
    ) -> list[BaseMessage]:
        """Async counterpart to :meth:`summarize_messages` supporting sync or async summarizers."""
        prepared = self._prepare_split(messages, self.trigger_threshold, self.keep_recent)
        if prepared is None:
            return list(messages)

        base_system_prompt, old_messages, recent_messages = prepared
        summary_text = self._build_transcript(old_messages)

        if summarizer_func is not None:
            try:
                raw: Any = summarizer_func(summary_text)
                computed_summary = await raw if inspect.isawaitable(raw) else str(raw)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully on summarizer failure
                logger.warning(
                    "AutoSummarizer async summarizer_func failed (%s: %s); falling back to truncation.",
                    type(exc).__name__,
                    exc,
                )
                computed_summary = self._fallback_summary(summary_text)
        else:
            computed_summary = self._fallback_summary(summary_text)

        return self._assemble_result(base_system_prompt, computed_summary, recent_messages)


__all__ = ["AutoSummarizer", "ContextBuffer", "SUMMARY_PREFIX"]
