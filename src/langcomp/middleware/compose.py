"""Onion composition helpers for chaining middleware wrap hooks."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

from langcomp.middleware.base import AgentMiddleware
from langcomp.middleware.types import (
    ModelHandler,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolHandler,
)


def normalize_middleware(item: AgentMiddleware | Callable[..., Any]) -> AgentMiddleware:
    """Validate that an input is an :class:`AgentMiddleware` instance."""
    if isinstance(item, AgentMiddleware):
        return item
    raise TypeError(
        f"Expected AgentMiddleware instance or langcomp hook decorator, got {type(item).__name__}"
    )


def compose_model_wrappers(
    middlewares: Sequence[AgentMiddleware],
    core_handler: ModelHandler,
) -> ModelHandler:
    """Compose ``wrap_model_call`` hooks into a nested onion callable.

    The first middleware in ``middlewares`` becomes the outermost wrapper:
    ``M1(M2(...(core_handler)...))``.
    """
    active = [mw for mw in middlewares if mw.has_hook("wrap_model_call")]

    def _wrap(next_handler: ModelHandler, mw: AgentMiddleware) -> ModelHandler:
        def _call(request: ModelRequest) -> ModelResponse:
            return mw.wrap_model_call(request, next_handler)

        return _call

    return functools.reduce(_wrap, reversed(active), core_handler)


def compose_tool_wrappers(
    middlewares: Sequence[AgentMiddleware],
    core_handler: ToolHandler,
) -> ToolHandler:
    """Compose ``wrap_tool_call`` hooks into a nested onion callable.

    The first middleware in ``middlewares`` becomes the outermost wrapper:
    ``M1(M2(...(core_handler)...))``.
    """
    active = [mw for mw in middlewares if mw.has_hook("wrap_tool_call")]

    def _wrap(next_handler: ToolHandler, mw: AgentMiddleware) -> ToolHandler:
        def _call(request: ToolCallRequest) -> ToolCallResponse:
            return mw.wrap_tool_call(request, next_handler)

        return _call

    return functools.reduce(_wrap, reversed(active), core_handler)
