"""Functional decorators for lightweight single-hook middleware."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langcomp.middleware.base import AgentMiddleware
from langcomp.middleware.types import (
    JumpTo,
    ModelHandler,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolHandler,
)

_HOOK_ATTRS: dict[str, str] = {
    "before_agent": "_before_agent_fn",
    "before_model": "_before_model_fn",
    "wrap_model_call": "_wrap_model_fn",
    "after_model": "_after_model_fn",
    "wrap_tool_call": "_wrap_tool_fn",
    "after_agent": "_after_agent_fn",
}


class _FunctionalMiddleware(AgentMiddleware):
    """Adapts a single decorated hook function into an :class:`AgentMiddleware`."""

    def __init__(
        self,
        *,
        before_agent_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        before_model_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        wrap_model_fn: Callable[[ModelRequest, ModelHandler], ModelResponse] | None = None,
        after_model_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        wrap_tool_fn: Callable[[ToolCallRequest, ToolHandler], ToolCallResponse] | None = None,
        after_agent_fn: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self._before_agent_fn = before_agent_fn
        self._before_model_fn = before_model_fn
        self._wrap_model_fn = wrap_model_fn
        self._after_model_fn = after_model_fn
        self._wrap_tool_fn = wrap_tool_fn
        self._after_agent_fn = after_agent_fn

    def has_hook(self, hook_name: str) -> bool:
        """Report only the specific hook supplied to this functional wrapper."""
        attr = _HOOK_ATTRS.get(hook_name)
        return attr is not None and getattr(self, attr) is not None

    def before_agent(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._before_agent_fn is not None:
            return self._before_agent_fn(state)  # type: ignore[no-any-return]
        return None

    def before_model(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._before_model_fn is not None:
            return self._before_model_fn(state)  # type: ignore[no-any-return]
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelHandler,
    ) -> ModelResponse:
        if self._wrap_model_fn is not None:
            return self._wrap_model_fn(request, handler)
        return handler(request)

    def after_model(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._after_model_fn is not None:
            return self._after_model_fn(state)  # type: ignore[no-any-return]
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolCallResponse:
        if self._wrap_tool_fn is not None:
            return self._wrap_tool_fn(request, handler)
        return handler(request)

    def after_agent(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._after_agent_fn is not None:
            return self._after_agent_fn(state)  # type: ignore[no-any-return]
        return None


def before_agent(fn: Callable[[Mapping[str, Any]], Any]) -> AgentMiddleware:
    """Create a middleware that runs once before the agent starts."""
    return _FunctionalMiddleware(before_agent_fn=fn)


def before_model(fn: Callable[[Mapping[str, Any]], Any]) -> AgentMiddleware:
    """Create a middleware that runs before each model call."""
    return _FunctionalMiddleware(before_model_fn=fn)


def wrap_model_call(
    fn: Callable[[ModelRequest, ModelHandler], ModelResponse],
) -> AgentMiddleware:
    """Create a middleware that wraps each model invocation."""
    return _FunctionalMiddleware(wrap_model_fn=fn)


def after_model(fn: Callable[[Mapping[str, Any]], Any]) -> AgentMiddleware:
    """Create a middleware that runs after each model response."""
    return _FunctionalMiddleware(after_model_fn=fn)


def wrap_tool_call(
    fn: Callable[[ToolCallRequest, ToolHandler], ToolCallResponse],
) -> AgentMiddleware:
    """Create a middleware that wraps each tool invocation."""
    return _FunctionalMiddleware(wrap_tool_fn=fn)


def after_agent(fn: Callable[[Mapping[str, Any]], Any]) -> AgentMiddleware:
    """Create a middleware that runs once after the agent completes."""
    return _FunctionalMiddleware(after_agent_fn=fn)


def dynamic_prompt(fn: Callable[[ModelRequest], str]) -> AgentMiddleware:
    """Create a middleware that computes a dynamic system prompt per model call."""

    def _wrap(request: ModelRequest, handler: ModelHandler) -> ModelResponse:
        return handler(request.override(system_prompt=fn(request)))

    return _FunctionalMiddleware(wrap_model_fn=_wrap)
