"""Typed immutable contracts for langcomp middleware requests, responses, and control flow."""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

ModelHandler = Callable[["ModelRequest"], "ModelResponse"]
AsyncModelHandler = Callable[["ModelRequest"], Awaitable["ModelResponse"]]
ToolHandler = Callable[["ToolCallRequest"], "ToolCallResponse"]
AsyncToolHandler = Callable[["ToolCallRequest"], Awaitable["ToolCallResponse"]]

_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class JumpTo:
    """Typed early-exit or graph jump signal returned from a lifecycle hook.

    Attributes:
        target: Destination in the agent graph (``"end"``, ``"tools"``, or ``"model"``).
        messages: Optional messages to append to state when taking the jump.
        state_update: Optional extra state fields to merge into agent state.
    """

    target: Literal["end", "tools", "model"]
    messages: Sequence[AnyMessage] = ()
    state_update: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_MAPPING)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Immutable bundle passed into ``wrap_model_call`` / ``awrap_model_call`` middleware hooks.

    Attributes:
        messages: Sequence of messages about to be sent to the model.
        system_prompt: Active transient system prompt, if any.
        model: Chat model or runnable instance to invoke.
        tools: Active tools available to the model.
        state: Read-only snapshot of current agent state.
        config: Optional LangGraph/LangChain ``RunnableConfig`` for the active invocation.
    """

    messages: Sequence[AnyMessage]
    system_prompt: str | None
    model: BaseChatModel | Runnable[Any, Any]
    tools: Sequence[BaseTool]
    state: Mapping[str, Any]
    config: Mapping[str, Any] | None = None

    def override(self, **changes: Any) -> ModelRequest:
        """Return a new :class:`ModelRequest` with the specified fields replaced."""
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Result produced by a model invocation or ``wrap_model_call`` hook.

    Attributes:
        message: The :class:`~langchain_core.messages.AIMessage` produced.
        state_update: Optional extra state fields to merge into agent state.
    """

    message: AIMessage
    state_update: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_MAPPING)


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """Immutable bundle passed into ``wrap_tool_call`` / ``awrap_tool_call`` middleware hooks.

    Attributes:
        tool: The :class:`~langchain_core.tools.BaseTool` instance to execute.
        tool_call: The raw tool-call specification emitted by the model.
        state: Read-only snapshot of current agent state.
        config: Optional LangGraph/LangChain ``RunnableConfig`` for the active invocation.
    """

    tool: BaseTool
    tool_call: ToolCall
    state: Mapping[str, Any]
    config: Mapping[str, Any] | None = None

    def override(self, **changes: Any) -> ToolCallRequest:
        """Return a new :class:`ToolCallRequest` with the specified fields replaced."""
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True, slots=True)
class ToolCallResponse:
    """Result produced by a tool execution or ``wrap_tool_call`` hook.

    Attributes:
        message: The :class:`~langchain_core.messages.ToolMessage` produced.
        state_update: Optional extra state fields to merge into agent state.
        approval_rejected: Structured flag indicating the call was denied by an
            approval policy (never retried or replaced with a fallback by guardrails).
    """

    message: ToolMessage
    state_update: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_MAPPING)
    approval_rejected: bool = False
