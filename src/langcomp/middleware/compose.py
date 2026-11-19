"""Onion composition helpers for chaining sync and async middleware wrap hooks."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from langcomp.middleware.base import AgentMiddleware
from langcomp.middleware.types import (
    AsyncModelHandler,
    AsyncToolHandler,
    ModelHandler,
    ToolHandler,
)

_ReqT = TypeVar("_ReqT")
_RespT = TypeVar("_RespT")


def normalize_middleware(item: AgentMiddleware) -> AgentMiddleware:
    """Validate that an input is an :class:`AgentMiddleware` instance."""
    if isinstance(item, AgentMiddleware):
        return item
    raise TypeError(
        f"Expected AgentMiddleware instance or langcomp hook decorator, got {type(item).__name__}"
    )


def _compose_sync_wrappers(
    middlewares: Sequence[AgentMiddleware],
    hook_name: str,
    core_handler: Callable[[_ReqT], _RespT],
) -> Callable[[_ReqT], _RespT]:
    active = [mw for mw in middlewares if mw.has_hook(hook_name)]

    def _wrap(
        next_handler: Callable[[_ReqT], _RespT],
        mw: AgentMiddleware,
    ) -> Callable[[_ReqT], _RespT]:
        hook = getattr(mw, hook_name)

        def _call(request: _ReqT) -> _RespT:
            return hook(request, next_handler)

        return _call

    return functools.reduce(_wrap, reversed(active), core_handler)


def _compose_async_wrappers(
    middlewares: Sequence[AgentMiddleware],
    sync_hook_name: str,
    async_hook_name: str,
    core_handler: Callable[[_ReqT], Awaitable[_RespT]],
) -> Callable[[_ReqT], Awaitable[_RespT]]:
    active = [
        mw
        for mw in middlewares
        if mw.has_hook(async_hook_name) or mw.has_hook(sync_hook_name)
    ]

    def _wrap(
        next_handler: Callable[[_ReqT], Awaitable[_RespT]],
        mw: AgentMiddleware,
    ) -> Callable[[_ReqT], Awaitable[_RespT]]:
        if mw.has_hook(async_hook_name):
            async_hook = getattr(mw, async_hook_name)

            async def _acall(request: _ReqT) -> _RespT:
                return await async_hook(request, next_handler)

            return _acall

        sync_hook = getattr(mw, sync_hook_name)

        async def _bridged_call(request: _ReqT) -> _RespT:
            loop = asyncio.get_running_loop()

            async def _await_next(modified_request: _ReqT) -> _RespT:
                return await next_handler(modified_request)

            def _sync_next(modified_request: _ReqT) -> _RespT:
                future = asyncio.run_coroutine_threadsafe(_await_next(modified_request), loop)
                return future.result()

            return await asyncio.to_thread(sync_hook, request, _sync_next)

        return _bridged_call

    return functools.reduce(_wrap, reversed(active), core_handler)


def compose_model_wrappers(
    middlewares: Sequence[AgentMiddleware],
    core_handler: ModelHandler,
) -> ModelHandler:
    """Compose ``wrap_model_call`` hooks into a nested sync onion callable.

    The first middleware in ``middlewares`` becomes the outermost wrapper:
    ``M1(M2(...(core_handler)...))``.
    """
    return _compose_sync_wrappers(middlewares, "wrap_model_call", core_handler)


def compose_async_model_wrappers(
    middlewares: Sequence[AgentMiddleware],
    core_handler: AsyncModelHandler,
) -> AsyncModelHandler:
    """Compose ``awrap_model_call`` / ``wrap_model_call`` hooks into a nested async onion callable."""
    return _compose_async_wrappers(
        middlewares, "wrap_model_call", "awrap_model_call", core_handler
    )


def compose_tool_wrappers(
    middlewares: Sequence[AgentMiddleware],
    core_handler: ToolHandler,
) -> ToolHandler:
    """Compose ``wrap_tool_call`` hooks into a nested sync onion callable.

    The first middleware in ``middlewares`` becomes the outermost wrapper:
    ``M1(M2(...(core_handler)...))``.
    """
    return _compose_sync_wrappers(middlewares, "wrap_tool_call", core_handler)


def compose_async_tool_wrappers(
    middlewares: Sequence[AgentMiddleware],
    core_handler: AsyncToolHandler,
) -> AsyncToolHandler:
    """Compose ``awrap_tool_call`` / ``wrap_tool_call`` hooks into a nested async onion callable."""
    return _compose_async_wrappers(
        middlewares, "wrap_tool_call", "awrap_tool_call", core_handler
    )


__all__ = [
    "compose_async_model_wrappers",
    "compose_async_tool_wrappers",
    "compose_model_wrappers",
    "compose_tool_wrappers",
    "normalize_middleware",
]
