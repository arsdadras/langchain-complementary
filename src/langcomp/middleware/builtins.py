"""Built-in middleware adapters bridging existing langcomp primitives into the agent loop."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.errors import GraphBubbleUp

from langcomp.errors import ApprovalRejectedError, ConfigurationError
from langcomp.memory.buffer import AutoSummarizer, ContextBuffer
from langcomp.middleware.base import AgentMiddleware
from langcomp.middleware.types import (
    AsyncModelHandler,
    AsyncToolHandler,
    ModelHandler,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolHandler,
)
from langcomp.tools.guard import ApprovalGate, ToolGuard

_UNSET: Any = object()
_APPROVAL_REJECTED_KEY = "langcomp_approval_rejected"


def _is_approval_rejection(response: ToolCallResponse) -> bool:
    if response.approval_rejected:
        return True
    extra = getattr(response.message, "additional_kwargs", None)
    if isinstance(extra, dict) and extra.get(_APPROVAL_REJECTED_KEY):
        return True
    return isinstance(response.message.content, str) and response.message.content.startswith(
        "ApprovalRejectedError:"
    )


def _format_denial_message(template: str, tool_name: str) -> str:
    return template.replace("{tool_name}", tool_name)


class ContextBufferMiddleware(AgentMiddleware):
    """Applies a :class:`~langcomp.memory.buffer.ContextBuffer` transiently before each model call.

    Persisted graph state remains intact while only the trimmed message window
    is sent to the model.
    """

    def __init__(
        self,
        buffer: ContextBuffer | None = None,
        *,
        max_messages: int = _UNSET,
        preserve_system_prompt: bool = _UNSET,
    ) -> None:
        if buffer is not None:
            if max_messages is not _UNSET or preserve_system_prompt is not _UNSET:
                raise ConfigurationError(
                    "ContextBufferMiddleware cannot combine an explicit 'buffer' instance with "
                    "'max_messages' or 'preserve_system_prompt' overrides."
                )
            self.buffer = buffer
        else:
            self.buffer = ContextBuffer(
                max_messages=10 if max_messages is _UNSET else max_messages,
                preserve_system_prompt=(
                    True if preserve_system_prompt is _UNSET else preserve_system_prompt
                ),
            )

    def _trim_request(self, request: ModelRequest) -> ModelRequest:
        messages: list[BaseMessage] = [
            m for m in request.messages if isinstance(m, BaseMessage)
        ]
        trimmed = self.buffer.trim(messages)
        return request.override(messages=tuple(trimmed))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelHandler,
    ) -> ModelResponse:
        return handler(self._trim_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: AsyncModelHandler,
    ) -> ModelResponse:
        return await handler(self._trim_request(request))


class AutoSummarizerMiddleware(AgentMiddleware):
    """Compresses older conversation turns via :class:`~langcomp.memory.buffer.AutoSummarizer`."""

    def __init__(
        self,
        summarizer: AutoSummarizer | None = None,
        *,
        summarizer_func: Callable[[str], str | Awaitable[str]] | None = None,
        trigger_threshold: int = _UNSET,
        keep_recent: int = _UNSET,
    ) -> None:
        if summarizer is not None:
            if trigger_threshold is not _UNSET or keep_recent is not _UNSET:
                raise ConfigurationError(
                    "AutoSummarizerMiddleware cannot combine an explicit 'summarizer' instance "
                    "with 'trigger_threshold' or 'keep_recent' overrides."
                )
            self.summarizer = summarizer
        else:
            self.summarizer = AutoSummarizer(
                trigger_threshold=15 if trigger_threshold is _UNSET else trigger_threshold,
                keep_recent=5 if keep_recent is _UNSET else keep_recent,
            )
        self.summarizer_func = summarizer_func
        self._cached_prefix_key: tuple[tuple[str, str], ...] | None = None
        self._cached_summary_output: str | None = None

    @staticmethod
    def _extract_messages(request: ModelRequest) -> list[BaseMessage]:
        return [m for m in request.messages if isinstance(m, BaseMessage)]

    def _prefix_key(self, messages: Sequence[BaseMessage]) -> tuple[tuple[str, str], ...]:
        prefix_len = max(0, len(messages) - self.summarizer.keep_recent)
        return tuple((m.__class__.__name__, str(m.content)) for m in messages[:prefix_len])

    def _cached_summary_for(self, prefix_key: tuple[tuple[str, str], ...]) -> str | None:
        if (
            self.summarizer_func is not None
            and self._cached_prefix_key == prefix_key
            and self._cached_summary_output is not None
        ):
            return self._cached_summary_output
        return None

    def _update_cache(self, prefix_key: tuple[tuple[str, str], ...], summary: str) -> None:
        self._cached_prefix_key = prefix_key
        self._cached_summary_output = summary

    def _summarize_with_cache(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        prefix_key = self._prefix_key(messages)
        cached = self._cached_summary_for(prefix_key)
        if cached is not None:
            return self.summarizer.summarize_messages(
                messages,
                summarizer_func=lambda _txt: cached,
            )

        captured: list[str] = []

        def _recording_summarizer(text: str) -> str:
            assert self.summarizer_func is not None
            res = self.summarizer_func(text)
            if not isinstance(res, str):
                raise TypeError(
                    "Async summarizer_func cannot be called synchronously; use ainvoke()/astream()."
                )
            captured.append(res)
            return res

        compressed = self.summarizer.summarize_messages(
            messages,
            summarizer_func=_recording_summarizer if self.summarizer_func is not None else None,
        )
        if captured:
            self._update_cache(prefix_key, captured[-1])
        return compressed

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelHandler,
    ) -> ModelResponse:
        compressed = self._summarize_with_cache(self._extract_messages(request))
        return handler(request.override(messages=tuple(compressed)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: AsyncModelHandler,
    ) -> ModelResponse:
        messages = self._extract_messages(request)
        prefix_key = self._prefix_key(messages)
        cached = self._cached_summary_for(prefix_key)
        if cached is not None:
            compressed = await self.summarizer.asummarize_messages(
                messages,
                summarizer_func=lambda _txt: cached,
            )
            return await handler(request.override(messages=tuple(compressed)))

        captured: list[str] = []

        async def _async_recording_summarizer(text: str) -> str:
            assert self.summarizer_func is not None
            res = self.summarizer_func(text)
            if isinstance(res, Awaitable):
                res = await res
            captured.append(res)
            return res

        compressed = await self.summarizer.asummarize_messages(
            messages,
            summarizer_func=(
                _async_recording_summarizer if self.summarizer_func is not None else None
            ),
        )
        if captured:
            self._update_cache(prefix_key, captured[-1])
        return await handler(request.override(messages=tuple(compressed)))


class ToolGuardMiddleware(AgentMiddleware):
    """Retries failing tool executions using a :class:`~langcomp.tools.guard.ToolGuard` policy.

    Intercepts both raised exceptions and error :class:`~langchain_core.messages.ToolMessage`
    responses returned by the core tool invoker. Never retries or falls back on
    :class:`~langcomp.errors.ApprovalRejectedError`, approval-rejected responses,
    or LangGraph :class:`~langgraph.errors.GraphBubbleUp` control-flow signals.
    """

    def __init__(
        self,
        guard: ToolGuard | None = None,
        *,
        max_retries: int = _UNSET,
        fallback_output: str | None = _UNSET,
        retry_delay_seconds: float = _UNSET,
    ) -> None:
        if guard is not None:
            if (
                max_retries is not _UNSET
                or fallback_output is not _UNSET
                or retry_delay_seconds is not _UNSET
            ):
                raise ConfigurationError(
                    "ToolGuardMiddleware cannot combine an explicit 'guard' instance with "
                    "'max_retries', 'fallback_output', or 'retry_delay_seconds' overrides."
                )
            self.guard = guard
        else:
            self.guard = ToolGuard(
                max_retries=3 if max_retries is _UNSET else max_retries,
                fallback_output=None if fallback_output is _UNSET else fallback_output,
                retry_delay_seconds=(
                    0.0 if retry_delay_seconds is _UNSET else retry_delay_seconds
                ),
            )

    def _terminal_failure_response(
        self,
        request: ToolCallRequest,
        last_response: ToolCallResponse | None,
        last_error: Exception | None,
    ) -> ToolCallResponse:
        call_id = request.tool_call.get("id") or ""
        if self.guard.fallback_output is not None:
            return ToolCallResponse(
                message=ToolMessage(
                    content=str(self.guard.fallback_output),
                    tool_call_id=call_id,
                    name=request.tool.name,
                )
            )
        if last_response is not None:
            return last_response
        return ToolCallResponse(
            message=ToolMessage(
                content=f"{type(last_error).__name__}: {last_error}" if last_error else "Tool failed",
                tool_call_id=call_id,
                name=request.tool.name,
                status="error",
            )
        )

    @staticmethod
    def _should_stop_retrying(response: ToolCallResponse) -> bool:
        return response.message.status != "error" or _is_approval_rejection(response)

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
            except (GraphBubbleUp, ApprovalRejectedError):
                raise
            except Exception as exc:  # noqa: BLE001 - retry policy intentionally catches any tool error
                last_error = exc
                last_response = None
            else:
                if self._should_stop_retrying(response):
                    return response
                last_response = response
                last_error = None

            if attempt < attempts and self.guard.retry_delay_seconds > 0:
                time.sleep(self.guard.retry_delay_seconds)

        return self._terminal_failure_response(request, last_response, last_error)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolCallResponse:
        last_response: ToolCallResponse | None = None
        last_error: Exception | None = None
        attempts = max(1, self.guard.max_retries)

        for attempt in range(1, attempts + 1):
            try:
                response = await handler(request)
            except (GraphBubbleUp, ApprovalRejectedError):
                raise
            except Exception as exc:  # noqa: BLE001 - retry policy intentionally catches any tool error
                last_error = exc
                last_response = None
            else:
                if self._should_stop_retrying(response):
                    return response
                last_response = response
                last_error = None

            if attempt < attempts and self.guard.retry_delay_seconds > 0:
                await asyncio.sleep(self.guard.retry_delay_seconds)

        return self._terminal_failure_response(request, last_response, last_error)


class ApprovalGateMiddleware(AgentMiddleware):
    """Enforces an :class:`~langcomp.tools.guard.ApprovalGate` before executing tools.

    Args:
        gate: Approval gate callback wrapper.
        guarded_tools: Optional allowlist of tool names requiring approval. When
            omitted, every tool call is checked against ``gate``.
        denial_message: Custom message template when a call is rejected.
        raise_on_reject: When ``True``, raises :class:`ApprovalRejectedError`
            instead of returning an error :class:`ToolMessage`.
    """

    def __init__(
        self,
        gate: ApprovalGate,
        *,
        guarded_tools: Sequence[str] | None = None,
        denial_message: str = "PermissionDenied: tool '{tool_name}' was rejected by approval gate.",
        raise_on_reject: bool = False,
    ) -> None:
        self.gate = gate
        self.guarded_tools = frozenset(guarded_tools) if guarded_tools is not None else None
        self.denial_message = denial_message
        self.raise_on_reject = raise_on_reject

    def _rejection_response(self, request: ToolCallRequest, tool_name: str) -> ToolCallResponse:
        msg = _format_denial_message(self.denial_message, tool_name)
        if self.raise_on_reject:
            raise ApprovalRejectedError(msg)
        return ToolCallResponse(
            message=ToolMessage(
                content=msg,
                tool_call_id=request.tool_call.get("id") or "",
                name=tool_name,
                status="error",
                additional_kwargs={_APPROVAL_REJECTED_KEY: True},
            ),
            approval_rejected=True,
        )

    def _is_guarded(self, tool_name: str) -> bool:
        return self.guarded_tools is None or tool_name in self.guarded_tools

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolCallResponse:
        tool_name = request.tool.name
        if not self._is_guarded(tool_name):
            return handler(request)

        args = dict(request.tool_call.get("args", {}))
        if not self.gate.check_approval(tool_name, args):
            return self._rejection_response(request, tool_name)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolCallResponse:
        tool_name = request.tool.name
        if not self._is_guarded(tool_name):
            return await handler(request)

        args = dict(request.tool_call.get("args", {}))
        if not await self.gate.acheck_approval(tool_name, args):
            return self._rejection_response(request, tool_name)
        return await handler(request)
