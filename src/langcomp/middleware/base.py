"""Base middleware abstraction for extending the langcomp agent loop."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from langcomp.middleware.types import (
    AsyncModelHandler,
    AsyncToolHandler,
    JumpTo,
    ModelHandler,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolHandler,
)

_KNOWN_HOOKS: frozenset[str] = frozenset(
    {
        "before_agent",
        "abefore_agent",
        "before_model",
        "abefore_model",
        "wrap_model_call",
        "awrap_model_call",
        "after_model",
        "aafter_model",
        "wrap_tool_call",
        "awrap_tool_call",
        "after_agent",
        "aafter_agent",
    }
)


class AgentMiddleware:
    """Base class for composable langcomp agent middleware.

    Subclasses can override synchronous lifecycle hooks (``before_agent``,
    ``before_model``, ``after_model``, ``after_agent``), onion wrap hooks
    (``wrap_model_call``, ``wrap_tool_call``), their native async counterparts
    (``abefore_agent``, ``abefore_model``, ``awrap_model_call``, ``aafter_model``,
    ``awrap_tool_call``, ``aafter_agent``), and optionally contribute ``tools``
    and a ``state_schema`` to the compiled agent graph.

    Attributes:
        tools: Sequence of tools (``BaseTool`` or plain callables) contributed by this middleware.
        state_schema: Optional TypedDict or annotated dict class whose fields are
            merged into the agent's graph state schema.
    """

    tools: Sequence[BaseTool | Callable[..., Any]] = ()
    state_schema: type[Any] | None = None

    def has_hook(self, hook_name: str) -> bool:
        """Return ``True`` if this middleware actively implements ``hook_name``."""
        if hook_name not in _KNOWN_HOOKS:
            return False
        base_method = getattr(AgentMiddleware, hook_name)
        current_method = getattr(self, hook_name)
        return bool(getattr(current_method, "__func__", current_method) != base_method)

    def before_agent(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | JumpTo | None:
        """Run once before the agent starts."""
        del state
        return None

    async def abefore_agent(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | JumpTo | None:
        """Async counterpart to :meth:`before_agent` (defaults to delegating to it)."""
        result = self.before_agent(state)
        if inspect.isawaitable(result):
            return await result
        return result

    def before_model(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | JumpTo | None:
        """Run before each model call."""
        del state
        return None

    async def abefore_model(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | JumpTo | None:
        """Async counterpart to :meth:`before_model` (defaults to delegating to it)."""
        result = self.before_model(state)
        if inspect.isawaitable(result):
            return await result
        return result

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelHandler,
    ) -> ModelResponse:
        """Wrap each model invocation (onion order: outermost registered first)."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: AsyncModelHandler,
    ) -> ModelResponse:
        """Async counterpart to :meth:`wrap_model_call`."""
        return await handler(request)

    def after_model(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | JumpTo | None:
        """Run after each model response is received (reverse registration order)."""
        del state
        return None

    async def aafter_model(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | JumpTo | None:
        """Async counterpart to :meth:`after_model` (defaults to delegating to it)."""
        result = self.after_model(state)
        if inspect.isawaitable(result):
            return await result
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolCallResponse:
        """Wrap each tool execution (onion order: outermost registered first)."""
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolCallResponse:
        """Async counterpart to :meth:`wrap_tool_call`."""
        return await handler(request)

    def after_agent(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Run once after the agent finishes (reverse registration order)."""
        del state
        return None

    async def aafter_agent(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Async counterpart to :meth:`after_agent` (defaults to delegating to it)."""
        result = self.after_agent(state)
        if inspect.isawaitable(result):
            return await result
        return result
