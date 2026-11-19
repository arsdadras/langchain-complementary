"""Dual-mechanism onion middleware system for langcomp.

Provides:
- :class:`AgentMiddleware` for composing capabilities (tools, state schemas, sync/async hooks).
- Immutable typed contracts (:class:`ModelRequest`, :class:`ModelResponse`,
  :class:`ToolCallRequest`, :class:`ToolCallResponse`, :class:`JumpTo`).
- Functional single-hook decorators (:func:`before_agent`, :func:`before_model`,
  :func:`wrap_model_call`, :func:`after_model`, :func:`wrap_tool_call`,
  :func:`after_agent`, :func:`dynamic_prompt`).
- Onion composition helpers (:func:`compose_model_wrappers`,
  :func:`compose_async_model_wrappers`, :func:`compose_tool_wrappers`,
  :func:`compose_async_tool_wrappers`, :func:`normalize_middleware`).
- Built-in adapters for langcomp's memory and tool guard primitives.
"""

from __future__ import annotations

from langcomp.middleware.base import AgentMiddleware
from langcomp.middleware.builtins import (
    ApprovalGateMiddleware,
    AutoSummarizerMiddleware,
    ContextBufferMiddleware,
    ToolGuardMiddleware,
)
from langcomp.middleware.compose import (
    compose_async_model_wrappers,
    compose_async_tool_wrappers,
    compose_model_wrappers,
    compose_tool_wrappers,
    normalize_middleware,
)
from langcomp.middleware.decorators import (
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
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

__all__ = [
    "AgentMiddleware",
    "ApprovalGateMiddleware",
    "AsyncModelHandler",
    "AsyncToolHandler",
    "AutoSummarizerMiddleware",
    "ContextBufferMiddleware",
    "JumpTo",
    "ModelHandler",
    "ModelRequest",
    "ModelResponse",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolGuardMiddleware",
    "ToolHandler",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "compose_async_model_wrappers",
    "compose_async_tool_wrappers",
    "compose_model_wrappers",
    "compose_tool_wrappers",
    "dynamic_prompt",
    "normalize_middleware",
    "wrap_model_call",
    "wrap_tool_call",
]
