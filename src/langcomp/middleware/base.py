"""Base middleware abstraction for extending the langcomp agent loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from langcomp.middleware.types import (
    JumpTo,
    ModelHandler,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolHandler,
)


class AgentMiddleware:
    """Base class for composable langcomp agent middleware.

    Subclasses can override lifecycle hooks (``before_agent``, ``before_model``,
    ``after_model``, ``after_agent``), onion wrap hooks (``wrap_model_call``,
    ``wrap_tool_call``), and optionally contribute ``tools`` and a ``state_schema``
    to the compiled agent graph.

    Attributes:
        tools: Sequence of tools contributed by this middleware.
        state_schema: Optional TypedDict or annotated dict class whose fields are
            merged into the agent's graph state schema.
    """

    tools: Sequence[BaseTool] = ()
    state_schema: type[Any] | None = None

    def has_hook(self, hook_name: str) -> bool:
        """Return ``True`` if this middleware actively implements ``hook_name``."""
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

    def before_model(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | JumpTo | None:
        """Run before each model call."""
        del state
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelHandler,
    ) -> ModelResponse:
        """Wrap each model invocation (onion order: outermost registered first)."""
        return handler(request)

    def after_model(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | JumpTo | None:
        """Run after each model response is received (reverse registration order)."""
        del state
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolCallResponse:
        """Wrap each tool execution (onion order: outermost registered first)."""
        return handler(request)

    def after_agent(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any] | JumpTo | None:
        """Run once after the agent finishes (reverse registration order)."""
        del state
        return None
