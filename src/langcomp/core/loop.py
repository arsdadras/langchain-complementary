"""Hand-built LangGraph agent loop with middleware hooks and lazy model resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from langcomp.core.nodes import (
    AFTER_AGENT_NODE,
    BEFORE_AGENT_NODE,
    MODEL_NODE,
    TOOLS_NODE,
    has_hook,
    make_after_agent_node,
    make_before_agent_node,
    make_call_model_node,
    make_tool_node,
)
from langcomp.core.state import SmartState, merge_state_schemas
from langcomp.middleware.base import AgentMiddleware
from langcomp.middleware.compose import (
    compose_model_wrappers,
    compose_tool_wrappers,
    normalize_middleware,
)
from langcomp.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from langcomp.models.resolver import resolve_model


def normalize_tools(tools: Sequence[BaseTool | Callable[..., Any]] | None) -> list[BaseTool]:
    """Coerce plain Python callables into LangChain ``BaseTool`` instances."""
    normalized: list[BaseTool] = []
    for item in tools or ():
        if isinstance(item, BaseTool):
            normalized.append(item)
        elif callable(item):
            normalized.append(lc_tool(item))
        else:
            raise TypeError(f"Expected BaseTool or callable, got {type(item).__name__}")
    return normalized


def create_agent_graph(
    model: str | BaseChatModel | BaseLanguageModel[Any] | None = None,
    tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    system_prompt: str | None = None,
    *,
    middleware: Sequence[AgentMiddleware | Callable[..., Any]] | None = None,
    extra_state_fields: Mapping[str, Any] | None = None,
    state_schema: type[Any] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    **model_kwargs: Any,
) -> tuple[Any, type[Any]]:
    """Build and compile a raw ``StateGraph`` model/tool loop with middleware support.

    Args:
        model: A ``BaseChatModel``/``BaseLanguageModel``, a ``"provider:model"``
            spec string, or ``None`` to resolve :data:`langcomp.DEFAULT_MODEL`.
        tools: Tools the model may call (``BaseTool`` instances or plain callables).
        system_prompt: Optional transient system prompt prepended to each model
            call without being persisted into graph state.
        middleware: Optional sequence of :class:`AgentMiddleware` instances or
            decorated hook functions.
        extra_state_fields: Optional extra state field annotations merged into
            the generated ``SmartState`` schema.
        state_schema: Optional explicit base state schema to extend with middleware schemas.
        checkpointer: Optional LangGraph checkpointer for state persistence.
        **model_kwargs: Extra kwargs forwarded to :func:`resolve_model` when
            ``model`` is a string or ``None``.

    Returns:
        A ``(compiled_graph, resolved_state_schema)`` tuple.
    """
    if isinstance(model, (BaseChatModel, BaseLanguageModel)):
        resolved_model: Any = model
    else:
        resolved_model = resolve_model(model, **model_kwargs)

    mw_list = [normalize_middleware(m) for m in (middleware or ())]

    combined_tools: list[BaseTool] = normalize_tools(tools)
    for mw in mw_list:
        combined_tools.extend(normalize_tools(mw.tools))

    middleware_schemas = [mw.state_schema for mw in mw_list if mw.state_schema is not None]
    if state_schema is not None:
        resolved_schema = merge_state_schemas(state_schema, middleware_schemas)
    else:
        resolved_schema = SmartState.create_schema(
            extra_fields=extra_state_fields,
            schemas=middleware_schemas,
        )

    has_tools = bool(combined_tools)
    if has_tools and hasattr(resolved_model, "bind_tools"):
        bound_model: Any = resolved_model.bind_tools(combined_tools)
    else:
        bound_model = resolved_model

    has_before_agent = any(has_hook(mw, "before_agent") for mw in mw_list)
    has_after_agent = any(has_hook(mw, "after_agent") for mw in mw_list)
    terminal_target = AFTER_AGENT_NODE if has_after_agent else END

    def core_tool_invoker(tr: ToolCallRequest) -> ToolCallResponse:
        try:
            output = tr.tool.invoke(tr.tool_call["args"])
            return ToolCallResponse(
                message=ToolMessage(content=str(output), tool_call_id=tr.tool_call["id"])
            )
        except Exception as exc:  # noqa: BLE001 - convert runtime tool errors into recoverable ToolMessage
            content = f"{type(exc).__name__}: {exc}"
            return ToolCallResponse(
                message=ToolMessage(
                    content=content,
                    tool_call_id=tr.tool_call["id"],
                    status="error",
                )
            )

    wrapped_tool_handler = compose_tool_wrappers(mw_list, core_tool_invoker)

    def core_model_invoker(req: ModelRequest) -> ModelResponse:
        msgs: list[AnyMessage] = list(req.messages)
        if req.system_prompt is not None:
            msgs = [SystemMessage(content=req.system_prompt), *msgs]
        model_to_call = (
            req.model.bind_tools(list(req.tools))
            if req.tools and req.model is not bound_model and isinstance(req.model, BaseChatModel)
            else req.model
        )
        resp = model_to_call.invoke(msgs)
        if not isinstance(resp, AIMessage):
            resp = AIMessage(content=str(resp))
        return ModelResponse(message=resp)

    wrapped_model_handler = compose_model_wrappers(mw_list, core_model_invoker)

    builder: StateGraph[Any, Any, Any, Any] = StateGraph(resolved_schema)

    model_destinations: tuple[str, ...] = (
        (TOOLS_NODE, MODEL_NODE, terminal_target)
        if has_tools
        else (MODEL_NODE, terminal_target)
    )
    builder.add_node(
        MODEL_NODE,
        make_call_model_node(
            bound_model,
            combined_tools,
            system_prompt,
            mw_list,
            wrapped_model_handler,
            has_tools=has_tools,
            terminal_target=terminal_target,
        ),
        destinations=model_destinations,
    )

    if has_before_agent:
        before_destinations: tuple[str, ...] = (
            (MODEL_NODE, TOOLS_NODE, terminal_target)
            if has_tools
            else (MODEL_NODE, terminal_target)
        )
        builder.add_node(
            BEFORE_AGENT_NODE,
            make_before_agent_node(
                mw_list,
                has_tools=has_tools,
                terminal_target=terminal_target,
            ),
            destinations=before_destinations,
        )
        builder.add_edge(START, BEFORE_AGENT_NODE)
    else:
        builder.add_edge(START, MODEL_NODE)

    if has_after_agent:
        builder.add_node(AFTER_AGENT_NODE, make_after_agent_node(mw_list))
        builder.add_edge(AFTER_AGENT_NODE, END)

    if has_tools:
        builder.add_node(TOOLS_NODE, make_tool_node(combined_tools, wrapped_tool_handler))
        builder.add_edge(TOOLS_NODE, MODEL_NODE)

    compiled = builder.compile(checkpointer=checkpointer)
    return compiled, resolved_schema
