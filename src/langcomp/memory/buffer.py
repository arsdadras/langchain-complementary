"""
Memory & Context Management for the langcomp framework.
Provides sliding window message trimming and automatic context summarization.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage

SUMMARY_PREFIX = "Previous Conversation Summary:\n"


class ContextBuffer:
    """Sliding window message buffer that trims conversation history to prevent context overflow."""

    def __init__(self, max_messages: int = 10, preserve_system_prompt: bool = True):
        if max_messages < 1:
            raise ValueError(f"max_messages must be >= 1, got {max_messages}")
        self.max_messages = max_messages
        self.preserve_system_prompt = preserve_system_prompt

    def trim(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Trim messages to at most ``max_messages`` while optionally preserving a leading SystemMessage."""
        if not messages or len(messages) <= self.max_messages:
            return list(messages)

        system_msg: Optional[BaseMessage] = None
        if self.preserve_system_prompt and isinstance(messages[0], SystemMessage):
            system_msg = messages[0]
            remaining = messages[1:]
        else:
            remaining = messages

        keep_count = self.max_messages - (1 if system_msg else 0)
        trimmed_tail = remaining[-keep_count:] if keep_count > 0 else []

        if system_msg:
            return [system_msg] + trimmed_tail
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
        return isinstance(msg, SystemMessage) and isinstance(msg.content, str) and msg.content.startswith(SUMMARY_PREFIX)

    def summarize_messages(
        self,
        messages: List[BaseMessage],
        summarizer_func: Optional[Callable[[str], str]] = None,
    ) -> List[BaseMessage]:
        """Summarize older messages when conversation length exceeds ``trigger_threshold``.

        On repeated passes, prior auto-generated summaries are folded into the new summary
        rather than stacking duplicate summary SystemMessages.
        """
        if len(messages) <= self.trigger_threshold:
            return list(messages)

        system_msg: Optional[BaseMessage] = None
        if isinstance(messages[0], SystemMessage) and not self._is_summary_message(messages[0]):
            system_msg = messages[0]
            work_messages = messages[1:]
        else:
            work_messages = messages

        if self.keep_recent == 0:
            old_messages = work_messages[:]
            recent_messages: List[BaseMessage] = []
        else:
            old_messages = work_messages[: -self.keep_recent]
            recent_messages = work_messages[-self.keep_recent :]

        if not old_messages:
            return list(messages)

        lines = []
        for msg in old_messages:
            role = msg.__class__.__name__.replace("Message", "")
            lines.append(f"{role}: {msg.content}")
        summary_text = "\n".join(lines)

        if summarizer_func is not None:
            try:
                computed_summary = summarizer_func(summary_text)
                summary_content = f"{SUMMARY_PREFIX}{computed_summary}"
            except Exception:
                summary_content = f"{SUMMARY_PREFIX}{summary_text[:500]}..."
        else:
            summary_content = f"{SUMMARY_PREFIX}{summary_text[:500]}..."

        summary_system_msg = SystemMessage(content=summary_content)

        result: List[BaseMessage] = []
        if system_msg:
            result.append(system_msg)
        result.append(summary_system_msg)
        result.extend(recent_messages)

        return result
