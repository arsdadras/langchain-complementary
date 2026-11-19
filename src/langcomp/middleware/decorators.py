"""Functional decorators for lightweight sync and async single-hook middleware."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langcomp.errors import ConfigurationError
from langcomp.middleware.base import AgentMiddleware
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

_HOOK_ATTRS: dict[str, str] = {
    "before_agent": "_before_agent_fn",
    "abefore_agent": "_abefore_agent_fn",
    "before_model": "_before_model_fn",
    "abefore_model": "_abefore_model_fn",
    "wrap_model_call": "_wrap_model_fn",
    "awrap_model_call": "_awrap_model_fn",
    "after_model": "_after_model_fn",
    "aafter_model": "_aafter_model_fn",
    "wrap_tool_call": "_wrap_tool_fn",
    "awrap_tool_call": "_awrap_tool_fn",
    "after_agent": "_after_agent_fn",
    "aafter_agent": "_aafter_agent_fn",
}


def _is_async_fn(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or (
        not isinstance(fn, type) and callable(fn) and inspect.iscoroutinefunction(type(fn).__call__)
    )


class _FunctionalMiddleware(AgentMiddleware):
    """Adapts a single decorated sync or async hook function into an :class:`AgentMiddleware`."""

    def __init__(
        self,
        *,
        source_fn: Callable[..., Any] | None = None,
        before_agent_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        abefore_agent_fn: Callable[[Mapping[str, Any]], Awaitable[Any]] | None = None,
        before_model_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        abefore_model_fn: Callable[[Mapping[str, Any]], Awaitable[Any]] | None = None,
        wrap_model_fn: Callable[[ModelRequest, ModelHandler], ModelResponse] | None = None,
        awrap_model_fn: Callable[[ModelRequest, AsyncModelHandler], Awaitable[ModelResponse]]
        | None = None,
        after_model_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        aafter_model_fn: Callable[[Mapping[str, Any]], Awaitable[Any]] | None = None,
        wrap_tool_fn: Callable[[ToolCallRequest, ToolHandler], ToolCallResponse] | None = None,
        awrap_tool_fn: Callable[[ToolCallRequest, AsyncToolHandler], Awaitable[ToolCallResponse]]
        | None = None,
        after_agent_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        aafter_agent_fn: Callable[[Mapping[str, Any]], Awaitable[Any]] | None = None,
    ) -> None:
        self._source_fn = source_fn
        self.__name__ = getattr(source_fn, "__name__", self.__class__.__name__)
        self.__doc__ = getattr(source_fn, "__doc__", None)
        self._before_agent_fn = before_agent_fn
        self._abefore_agent_fn = abefore_agent_fn
        self._before_model_fn = before_model_fn
        self._abefore_model_fn = abefore_model_fn
        self._wrap_model_fn = wrap_model_fn
        self._awrap_model_fn = awrap_model_fn
        self._after_model_fn = after_model_fn
        self._aafter_model_fn = aafter_model_fn
        self._wrap_tool_fn = wrap_tool_fn
        self._awrap_tool_fn = awrap_tool_fn
        self._after_agent_fn = after_agent_fn
        self._aafter_agent_fn = aafter_agent_fn

    def __repr__(self) -> str:
        active = [hook for hook in _HOOK_ATTRS if self.has_hook(hook)]
        return f"<Middleware {self.__name__} hooks={active}>"

    def has_hook(self, hook_name: str) -> bool:
        """Report only the specific hook supplied to this functional wrapper."""
        attr = _HOOK_ATTRS.get(hook_name)
        return attr is not None and getattr(self, attr) is not None

    def before_agent(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._before_agent_fn is not None:
            return self._before_agent_fn(state)
        if self._abefore_agent_fn is not None:
            raise ConfigurationError(
                f"Async middleware hook {self.__name__!r} cannot be executed synchronously; use ainvoke()/astream()."
            )
        return None

    async def abefore_agent(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._abefore_agent_fn is not None:
            return await self._abefore_agent_fn(state)
        if self._before_agent_fn is not None:
            return self._before_agent_fn(state)
        return None

    def before_model(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._before_model_fn is not None:
            return self._before_model_fn(state)
        if self._abefore_model_fn is not None:
            raise ConfigurationError(
                f"Async middleware hook {self.__name__!r} cannot be executed synchronously; use ainvoke()/astream()."
            )
        return None

    async def abefore_model(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._abefore_model_fn is not None:
            return await self._abefore_model_fn(state)
        if self._before_model_fn is not None:
            return self._before_model_fn(state)
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelHandler,
    ) -> ModelResponse:
        if self._wrap_model_fn is not None:
            return self._wrap_model_fn(request, handler)
        if self._awrap_model_fn is not None:
            raise ConfigurationError(
                f"Async middleware hook {self.__name__!r} cannot be executed synchronously; use ainvoke()/astream()."
            )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: AsyncModelHandler,
    ) -> ModelResponse:
        if self._awrap_model_fn is not None:
            return await self._awrap_model_fn(request, handler)
        return await handler(request)

    def after_model(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._after_model_fn is not None:
            return self._after_model_fn(state)
        if self._aafter_model_fn is not None:
            raise ConfigurationError(
                f"Async middleware hook {self.__name__!r} cannot be executed synchronously; use ainvoke()/astream()."
            )
        return None

    async def aafter_model(self, state: Mapping[str, Any]) -> Mapping[str, Any] | JumpTo | None:
        if self._aafter_model_fn is not None:
            return await self._aafter_model_fn(state)
        if self._after_model_fn is not None:
            return self._after_model_fn(state)
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolCallResponse:
        if self._wrap_tool_fn is not None:
            return self._wrap_tool_fn(request, handler)
        if self._awrap_tool_fn is not None:
            raise ConfigurationError(
                f"Async middleware hook {self.__name__!r} cannot be executed synchronously; use ainvoke()/astream()."
            )
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolCallResponse:
        if self._awrap_tool_fn is not None:
            return await self._awrap_tool_fn(request, handler)
        return await handler(request)

    def after_agent(self, state: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self._after_agent_fn is not None:
            return self._after_agent_fn(state)
        if self._aafter_agent_fn is not None:
            raise ConfigurationError(
                f"Async middleware hook {self.__name__!r} cannot be executed synchronously; use ainvoke()/astream()."
            )
        return None

    async def aafter_agent(self, state: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self._aafter_agent_fn is not None:
            return await self._aafter_agent_fn(state)
        if self._after_agent_fn is not None:
            return self._after_agent_fn(state)
        return None


def before_agent(fn: Callable[[Mapping[str, Any]], Any]) -> AgentMiddleware:
    """Create a middleware that runs once before the agent starts (sync or async)."""
    if _is_async_fn(fn):
        return _FunctionalMiddleware(source_fn=fn, abefore_agent_fn=fn)
    return _FunctionalMiddleware(source_fn=fn, before_agent_fn=fn)


def before_model(fn: Callable[[Mapping[str, Any]], Any]) -> AgentMiddleware:
    """Create a middleware that runs before each model call (sync or async)."""
    if _is_async_fn(fn):
        return _FunctionalMiddleware(source_fn=fn, abefore_model_fn=fn)
    return _FunctionalMiddleware(source_fn=fn, before_model_fn=fn)


def wrap_model_call(
    fn: Callable[..., Any],
) -> AgentMiddleware:
    """Create a middleware that wraps each model invocation (sync or async)."""
    if _is_async_fn(fn):
        return _FunctionalMiddleware(source_fn=fn, awrap_model_fn=fn)
    return _FunctionalMiddleware(source_fn=fn, wrap_model_fn=fn)


def after_model(fn: Callable[[Mapping[str, Any]], Any]) -> AgentMiddleware:
    """Create a middleware that runs after each model response (sync or async)."""
    if _is_async_fn(fn):
        return _FunctionalMiddleware(source_fn=fn, aafter_model_fn=fn)
    return _FunctionalMiddleware(source_fn=fn, after_model_fn=fn)


def wrap_tool_call(
    fn: Callable[..., Any],
) -> AgentMiddleware:
    """Create a middleware that wraps each tool invocation (sync or async)."""
    if _is_async_fn(fn):
        return _FunctionalMiddleware(source_fn=fn, awrap_tool_fn=fn)
    return _FunctionalMiddleware(source_fn=fn, wrap_tool_fn=fn)


def after_agent(fn: Callable[[Mapping[str, Any]], Any]) -> AgentMiddleware:
    """Create a middleware that runs once after the agent completes (sync or async)."""
    if _is_async_fn(fn):
        return _FunctionalMiddleware(source_fn=fn, aafter_agent_fn=fn)
    return _FunctionalMiddleware(source_fn=fn, after_agent_fn=fn)


def dynamic_prompt(fn: Callable[[ModelRequest], str | Awaitable[str]]) -> AgentMiddleware:
    """Create a middleware that computes a dynamic system prompt per model call (sync or async)."""
    if _is_async_fn(fn):

        async def _awrap(request: ModelRequest, handler: AsyncModelHandler) -> ModelResponse:
            prompt = await fn(request)  # type: ignore[misc]
            return await handler(request.override(system_prompt=prompt))

        return _FunctionalMiddleware(source_fn=fn, awrap_model_fn=_awrap)

    def _wrap(request: ModelRequest, handler: ModelHandler) -> ModelResponse:
        prompt = fn(request)
        return handler(request.override(system_prompt=prompt))

    return _FunctionalMiddleware(source_fn=fn, wrap_model_fn=_wrap)
