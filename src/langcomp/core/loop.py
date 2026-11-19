"""Hand-built sync + async LangGraph agent loop with middleware hooks and lazy model resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langcomp.core.nodes import (
    AFTER_AGENT_NODE,
    BEFORE_AGENT_NODE,
    MODEL_NODE,
    TOOLS_NODE,
    make_after_agent_node,
    make_before_agent_node,
    make_call_model_node,
    make_tool_node,
)
from langcomp.core.state import SmartState, extract_schema_reducers, merge_state_schemas
from langcomp.errors import ApprovalRejectedError, ConfigurationError
from langcomp.middleware.base import AgentMiddleware
from langcomp.middleware.compose import (
    compose_async_model_wrappers,
    compose_async_tool_wrappers,
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


def _validate_unique_tools(tools: Sequence[BaseTool]) -> None:
    seen: set[str] = set()
    for tool_inst in tools:
        if tool_inst.name in seen:
            raise ConfigurationError(
                f"Duplicate tool name {tool_inst.name!r} registered on agent; "
                f"tool names must be unique."
            )
        seen.add(tool_inst.name)


def _tool_output_to_response(output: Any, tr: ToolCallRequest) -> ToolCallResponse:
    call_id = tr.tool_call.get("id") or ""
    if isinstance(output, ToolMessage):
        return ToolCallResponse(message=output)
    if isinstance(output, Command):
        update = dict(output.update) if isinstance(output.update, Mapping) else {}
        msgs = update.pop("messages", None)
        tool_msg: ToolMessage | None = None
        if isinstance(msgs, Sequence) and not isinstance(msgs, (str, bytes)):
            for m in msgs:
                if isinstance(m, ToolMessage):
                    tool_msg = m
                    break
        elif isinstance(msgs, ToolMessage):
            tool_msg = msgs
        if tool_msg is None:
            tool_msg = ToolMessage(
                content="ok",
                tool_call_id=call_id,
                name=tr.tool.name,
            )
        return ToolCallResponse(message=tool_msg, state_update=update)
    return ToolCallResponse(
        message=ToolMessage(
            content=str(output),
            tool_call_id=call_id,
            name=tr.tool.name,
        )
    )


def _build_tool_call_payload(tr: ToolCallRequest) -> dict[str, Any]:
    return {
        "name": tr.tool_call["name"],
        "args": tr.tool_call.get("args", {}),
        "id": tr.tool_call.get("id") or "",
        "type": "tool_call",
    }


def _tool_error_response(exc: Exception, tr: ToolCallRequest) -> ToolCallResponse:
    return ToolCallResponse(
        message=ToolMessage(
            content=f"{type(exc).__name__}: {exc}",
            tool_call_id=tr.tool_call.get("id") or "",
            name=tr.tool.name,
            status="error",
        )
    )


def _select_model_for_request(
    req: ModelRequest,
    *,
    resolved_model: Any,
    bound_model: Any,
    default_tools: tuple[BaseTool, ...],
) -> Any:
    req_tools = tuple(req.tools)
    if req.model is resolved_model and req_tools == default_tools:
        return bound_model
    if req_tools and hasattr(req.model, "bind_tools"):
        return req.model.bind_tools(list(req_tools))
    return req.model


def _prepare_model_call(
    req: ModelRequest,
    *,
    resolved_model: Any,
    bound_model: Any,
    default_tools: tuple[BaseTool, ...],
) -> tuple[Any, list[AnyMessage]]:
    msgs: list[AnyMessage] = list(req.messages)
    if req.system_prompt is not None:
        msgs = [SystemMessage(content=req.system_prompt), *msgs]
    model_to_call = _select_model_for_request(
        req,
        resolved_model=resolved_model,
        bound_model=bound_model,
        default_tools=default_tools,
    )
    return model_to_call, msgs


def _normalize_model_response(resp: Any) -> ModelResponse:
    if not isinstance(resp, AIMessage):
        resp = AIMessage(content=str(resp))
    return ModelResponse(message=resp)


def create_agent_graph(
    model: str | BaseChatModel | BaseLanguageModel[Any] | None = None,
    tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    system_prompt: str | None = None,
    *,
    middleware: Sequence[AgentMiddleware] | None = None,
    extra_state_fields: Mapping[str, Any] | None = None,
    state_schema: type[Any] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    **model_kwargs: Any,
) -> tuple[Any, type[Any]]:
    """Build and compile a raw ``StateGraph`` model/tool loop with sync/async middleware support.

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
        if model_kwargs:
            raise ConfigurationError(
                f"Cannot pass model_kwargs ({', '.join(sorted(model_kwargs))}) "
                f"when supplying a pre-built model instance."
            )
        resolved_model: Any = model
    else:
        resolved_model = resolve_model(model, **model_kwargs)

    mw_list = [normalize_middleware(m) for m in (middleware or ())]

    combined_tools: list[BaseTool] = normalize_tools(tools)
    for mw in mw_list:
        combined_tools.extend(normalize_tools(mw.tools))
    _validate_unique_tools(combined_tools)

    middleware_schemas = [mw.state_schema for mw in mw_list if mw.state_schema is not None]
    if state_schema is not None:
        resolved_schema = merge_state_schemas(
            state_schema, middleware_schemas, ensure_messages=True
        )
    else:
        resolved_schema = SmartState.create_schema(
            extra_fields=extra_state_fields,
            schemas=middleware_schemas,
        )

    reducers = extract_schema_reducers(resolved_schema)
    default_tools = tuple(combined_tools)
    has_tools = bool(combined_tools)
    if has_tools and hasattr(resolved_model, "bind_tools"):
        bound_model: Any = resolved_model.bind_tools(combined_tools)
    else:
        bound_model = resolved_model

    has_before_agent = any(
        mw.has_hook("before_agent") or mw.has_hook("abefore_agent") for mw in mw_list
    )
    has_after_agent = any(
        mw.has_hook("after_agent") or mw.has_hook("aafter_agent") for mw in mw_list
    )
    terminal_target = AFTER_AGENT_NODE if has_after_agent else END

    def core_tool_invoker(tr: ToolCallRequest) -> ToolCallResponse:
        try:
            config: RunnableConfig | None = tr.config  # type: ignore[assignment]
            output = tr.tool.invoke(_build_tool_call_payload(tr), config=config)
            return _tool_output_to_response(output, tr)
        except (GraphBubbleUp, ApprovalRejectedError):
            raise
        except Exception as exc:  # noqa: BLE001 - convert runtime tool errors into recoverable ToolMessage
            return _tool_error_response(exc, tr)

    async def acore_tool_invoker(tr: ToolCallRequest) -> ToolCallResponse:
        try:
            config: RunnableConfig | None = tr.config  # type: ignore[assignment]
            output = await tr.tool.ainvoke(_build_tool_call_payload(tr), config=config)
            return _tool_output_to_response(output, tr)
        except (GraphBubbleUp, ApprovalRejectedError):
            raise
        except Exception as exc:  # noqa: BLE001 - convert runtime tool errors into recoverable ToolMessage
            return _tool_error_response(exc, tr)

    wrapped_tool_handler = compose_tool_wrappers(mw_list, core_tool_invoker)
    wrapped_async_tool_handler = compose_async_tool_wrappers(mw_list, acore_tool_invoker)

    def core_model_invoker(req: ModelRequest) -> ModelResponse:
        model_to_call, msgs = _prepare_model_call(
            req,
            resolved_model=resolved_model,
            bound_model=bound_model,
            default_tools=default_tools,
        )
        return _normalize_model_response(model_to_call.invoke(msgs, config=req.config))

    async def acore_model_invoker(req: ModelRequest) -> ModelResponse:
        model_to_call, msgs = _prepare_model_call(
            req,
            resolved_model=resolved_model,
            bound_model=bound_model,
            default_tools=default_tools,
        )
        return _normalize_model_response(await model_to_call.ainvoke(msgs, config=req.config))

    wrapped_model_handler = compose_model_wrappers(mw_list, core_model_invoker)
    wrapped_async_model_handler = compose_async_model_wrappers(mw_list, acore_model_invoker)

    builder: StateGraph[Any, Any, Any, Any] = StateGraph(resolved_schema)

    model_destinations: tuple[str, ...] = (
        (TOOLS_NODE, MODEL_NODE, terminal_target)
        if has_tools
        else (MODEL_NODE, terminal_target)
    )
    builder.add_node(
        MODEL_NODE,
        make_call_model_node(
            resolved_model,
            combined_tools,
            system_prompt,
            mw_list,
            wrapped_model_handler,
            wrapped_async_model_handler,
            has_tools=has_tools,
            terminal_target=terminal_target,
            reducers=reducers,
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
                reducers=reducers,
            ),
            destinations=before_destinations,
        )
        builder.add_edge(START, BEFORE_AGENT_NODE)
    else:
        builder.add_edge(START, MODEL_NODE)

    if has_after_agent:
        builder.add_node(
            AFTER_AGENT_NODE,
            make_after_agent_node(mw_list, reducers=reducers),
        )
        builder.add_edge(AFTER_AGENT_NODE, END)

    if has_tools:
        builder.add_node(
            TOOLS_NODE,
            make_tool_node(
                combined_tools,
                wrapped_tool_handler,
                wrapped_async_tool_handler,
                reducers=reducers,
            ),
        )
        builder.add_edge(TOOLS_NODE, MODEL_NODE)

    compiled = builder.compile(checkpointer=checkpointer)
    return compiled, resolved_schema
