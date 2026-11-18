"""Built-in middleware adapters bridging existing langcomp primitives into the agent loop."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from langchain_core.messages import BaseMessage, ToolMessage

from langcomp.memory.buffer import AutoSummarizer, ContextBuffer
from langcomp.middleware.base import AgentMiddleware
from langcomp.middleware.types import (
    ModelHandler,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolHandler,
)
from langcomp.tools.guard import ApprovalGate, ToolGuard


class ContextBufferMiddleware(AgentMiddleware):
    """Applies a :class:`~langcomp.memory.buffer.ContextBuffer` transiently before each model call.

    Persisted graph state remains intact while only the trimmed message window
    is sent to the model.
    """

    def __init__(
        self,
        buffer: ContextBuffer | None = None,
        *,
        max_messages: int = 10,
        preserve_system_prompt: bool = True,
    ) -> None:
        self.buffer = buffer or ContextBuffer(
            max_messages=max_messages,
            preserve_system_prompt=preserve_system_prompt,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelHandler,
    ) -> ModelResponse:
        messages: list[BaseMessage] = [
            m for m in request.messages if isinstance(m, BaseMessage)
        ]
        trimmed = self.buffer.trim(messages)
        return handler(request.override(messages=tuple(trimmed)))


class AutoSummarizerMiddleware(AgentMiddleware):
    """Compresses older conversation turns via :class:`~langcomp.memory.buffer.AutoSummarizer`."""

    def __init__(
        self,
        summarizer: AutoSummarizer | None = None,
        *,
        summarizer_func: Callable[[str], str] | None = None,
        trigger_threshold: int = 15,
        keep_recent: int = 5,
    ) -> None:
        self.summarizer = summarizer or AutoSummarizer(
            trigger_threshold=trigger_threshold,
            keep_recent=keep_recent,
        )
        self.summarizer_func = summarizer_func

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelHandler,
    ) -> ModelResponse:
        messages: list[BaseMessage] = [
            m for m in request.messages if isinstance(m, BaseMessage)
        ]
        compressed = self.summarizer.summarize_messages(
            messages,
            summarizer_func=self.summarizer_func,
        )
        return handler(request.override(messages=tuple(compressed)))


class ToolGuardMiddleware(AgentMiddleware):
    """Retries failing tool executions using a :class:`~langcomp.tools.guard.ToolGuard` policy.

    Intercepts both raised exceptions and error :class:`~langchain_core.messages.ToolMessage`
    responses returned by the core tool invoker.
    """

    def __init__(
        self,
        guard: ToolGuard | None = None,
        *,
        max_retries: int = 3,
        fallback_output: str | None = None,
        retry_delay_seconds: float = 0.0,
    ) -> None:
        self.guard = guard or ToolGuard(
            max_retries=max_retries,
            fallback_output=fallback_output,
            retry_delay_seconds=retry_delay_seconds,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolCallResponse:
        last_response: ToolCallResponse | None = None
        last_error: Exception | None = None
        attempts = max(1, self.guard.max_retries)

        for attempt in range(1, attempts + 1):
            try:
                response = handler(request)
            except Exception as exc:  # noqa: BLE001 - retry policy intentionally catches any tool error
                last_error = exc
                last_response = None
            else:
                if response.message.status != "error":
                    return response
                last_response = response
                last_error = None

            if attempt < attempts and self.guard.retry_delay_seconds > 0:
                time.sleep(self.guard.retry_delay_seconds)

        if self.guard.fallback_output is not None:
            return ToolCallResponse(
                message=ToolMessage(
                    content=self.guard.fallback_output,
                    tool_call_id=request.tool_call["id"],
                )
            )
        if last_response is not None:
            return last_response
        return ToolCallResponse(
            message=ToolMessage(
                content=f"{type(last_error).__name__}: {last_error}" if last_error else "Tool failed",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        )


class ApprovalGateMiddleware(AgentMiddleware):
    """Enforces an :class:`~langcomp.tools.guard.ApprovalGate` before executing tools.

    Args:
        gate: Approval gate callback wrapper.
        guarded_tools: Optional allowlist of tool names requiring approval. When
            omitted, every tool call is checked against ``gate``.
        denial_message: Custom message template when a call is rejected.
    """

    def __init__(
        self,
        gate: ApprovalGate,
        *,
        guarded_tools: Sequence[str] | None = None,
        denial_message: str = "PermissionDenied: tool '{tool_name}' was rejected by approval gate.",
    ) -> None:
        self.gate = gate
        self.guarded_tools = frozenset(guarded_tools) if guarded_tools is not None else None
        self.denial_message = denial_message

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolCallResponse:
        tool_name = request.tool.name
        if self.guarded_tools is not None and tool_name not in self.guarded_tools:
            return handler(request)

        args = dict(request.tool_call.get("args", {}))
        if not self.gate.check_approval(tool_name, args):
            return ToolCallResponse(
                message=ToolMessage(
                    content=self.denial_message.format(tool_name=tool_name),
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            )
        return handler(request)
