"""Sync and async graph node builders for the langcomp hand-built agent execution loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.types import Command

from langcomp.errors import ConfigurationError, UnknownToolError
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

TOOLS_NODE = "tools"
MODEL_NODE = "model"
BEFORE_AGENT_NODE = "before_agent"
AFTER_AGENT_NODE = "after_agent"

ReducerMap = Mapping[str, Callable[[Any, Any], Any]]


def has_hook(mw: AgentMiddleware, hook_name: str) -> bool:
    """Return ``True`` if ``mw`` actively implements ``hook_name``."""
    return mw.has_hook(hook_name)


def _as_message_list(value: Any) -> list[Any]:
    if isinstance(value, BaseMessage):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _merge_update(
    target: dict[str, Any],
    update: Mapping[str, Any],
    *,
    reducers: ReducerMap | None = None,
    apply_message_reducer: bool = False,
) -> None:
    """Merge a partial state update dict into ``target``, honoring schema reducers."""
    for key, value in update.items():
        if key == "messages":
            incoming = _as_message_list(value)
            existing = list(target.get("messages", []))
            if apply_message_reducer:
                target["messages"] = list(add_messages(existing, incoming))
            else:
                existing.extend(incoming)
                target["messages"] = existing
        elif reducers and key in reducers and key in target:
            target[key] = reducers[key](target[key], value)
        else:
            target[key] = value


def resolve_jump_target(
    jump: JumpTo,
    *,
    has_tools: bool,
    terminal_target: str,
) -> str:
    """Map a :class:`JumpTo` target name to a concrete graph node destination."""
    if jump.target == "end":
        return terminal_target
    if jump.target == "model":
        return MODEL_NODE
    if jump.target == "tools":
        if not has_tools:
            raise ConfigurationError(
                "JumpTo('tools') requested, but no tools are registered on this agent."
            )
        return TOOLS_NODE
    raise ConfigurationError(f"Unsupported JumpTo target: {jump.target!r}")


def _build_jump_command(
    jump: JumpTo,
    updates: dict[str, Any],
    *,
    has_tools: bool,
    terminal_target: str,
    reducers: ReducerMap | None = None,
) -> Command[str]:
    if jump.state_update:
        _merge_update(updates, jump.state_update, reducers=reducers)
    if jump.messages:
        _merge_update(updates, {"messages": list(jump.messages)}, reducers=reducers)
    goto = resolve_jump_target(
        jump,
        has_tools=has_tools,
        terminal_target=terminal_target,
    )
    return Command(update=updates, goto=goto)


def _unknown_tool_response(
    tool_by_name: Mapping[str, BaseTool],
    call: ToolCall,
) -> ToolCallResponse:
    error = UnknownToolError(call["name"], tuple(tool_by_name))
    return ToolCallResponse(
        message=ToolMessage(
            content=str(error),
            tool_call_id=call.get("id") or "",
            name=call["name"],
            status="error",
        )
    )


def run_one_tool_call(
    tool_by_name: Mapping[str, BaseTool],
    call: ToolCall,
    wrapped_handler: ToolHandler,
    state: Mapping[str, Any],
    config: RunnableConfig | None = None,
) -> ToolCallResponse:
    """Execute a single tool call synchronously through the composed middleware onion pipeline."""
    tool_inst = tool_by_name.get(call["name"])
    if tool_inst is None:
        return _unknown_tool_response(tool_by_name, call)

    req = ToolCallRequest(tool=tool_inst, tool_call=call, state=dict(state), config=config)
    return wrapped_handler(req)


async def arun_one_tool_call(
    tool_by_name: Mapping[str, BaseTool],
    call: ToolCall,
    wrapped_async_handler: AsyncToolHandler,
    state: Mapping[str, Any],
    config: RunnableConfig | None = None,
) -> ToolCallResponse:
    """Execute a single tool call asynchronously through the composed middleware onion pipeline."""
    tool_inst = tool_by_name.get(call["name"])
    if tool_inst is None:
        return _unknown_tool_response(tool_by_name, call)

    req = ToolCallRequest(tool=tool_inst, tool_call=call, state=dict(state), config=config)
    return await wrapped_async_handler(req)


def _get_last_ai_message(state: Mapping[str, Any]) -> AIMessage:
    messages = state.get("messages") or []
    if not messages:
        raise ConfigurationError("Cannot execute tools when state has no messages.")
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ConfigurationError(
            f"Expected last message to be AIMessage before tool execution, "
            f"got {type(last_message).__name__}."
        )
    return last_message


def _apply_hook_result(
    output: dict[str, Any],
    working: dict[str, Any],
    res: Mapping[str, Any],
    reducers: ReducerMap | None,
) -> None:
    _merge_update(output, res, reducers=reducers)
    _merge_update(working, res, reducers=reducers, apply_message_reducer=True)


def _collect_tool_response(
    resp: ToolCallResponse,
    messages: list[ToolMessage],
    updates: dict[str, Any],
    reducers: ReducerMap | None,
) -> None:
    messages.append(resp.message)
    if resp.state_update:
        _merge_update(updates, resp.state_update, reducers=reducers)


def _record_model_response(
    output_state: dict[str, Any],
    working_state: dict[str, Any],
    model_response: ModelResponse,
    reducers: ReducerMap | None,
) -> dict[str, Any]:
    msg_update = {"messages": [model_response.message]}
    _merge_update(output_state, msg_update, reducers=reducers)
    if model_response.state_update:
        _merge_update(output_state, model_response.state_update, reducers=reducers)

    interim_state = dict(working_state)
    _merge_update(interim_state, msg_update, reducers=reducers, apply_message_reducer=True)
    if model_response.state_update:
        _merge_update(
            interim_state,
            model_response.state_update,
            reducers=reducers,
            apply_message_reducer=True,
        )
    return interim_state


def _route_model_completion(
    message: Any,
    output_state: dict[str, Any],
    *,
    has_tools: bool,
    terminal_target: str,
) -> Command[str]:
    if has_tools and isinstance(message, AIMessage) and message.tool_calls:
        return Command(update=output_state, goto=TOOLS_NODE)
    return Command(update=output_state, goto=terminal_target)


def make_tool_node(
    tools: Sequence[BaseTool],
    wrapped_handler: ToolHandler,
    wrapped_async_handler: AsyncToolHandler | None = None,
    *,
    reducers: ReducerMap | None = None,
) -> Runnable[Any, Any]:
    """Construct the sync + async tool-execution node runnable for ``StateGraph``."""
    tool_by_name = {t.name: t for t in tools}

    def tool_node(state: Any, config: RunnableConfig | None = None) -> dict[str, Any]:
        last_message = _get_last_ai_message(state)
        messages: list[ToolMessage] = []
        updates: dict[str, Any] = {}
        for call in last_message.tool_calls:
            resp = run_one_tool_call(tool_by_name, call, wrapped_handler, state, config=config)
            _collect_tool_response(resp, messages, updates, reducers)
        _merge_update(updates, {"messages": messages}, reducers=reducers)
        return updates

    async def atool_node(state: Any, config: RunnableConfig | None = None) -> dict[str, Any]:
        if wrapped_async_handler is None:
            return tool_node(state, config=config)
        last_message = _get_last_ai_message(state)
        messages: list[ToolMessage] = []
        updates: dict[str, Any] = {}
        for call in last_message.tool_calls:
            resp = await arun_one_tool_call(
                tool_by_name, call, wrapped_async_handler, state, config=config
            )
            _collect_tool_response(resp, messages, updates, reducers)
        _merge_update(updates, {"messages": messages}, reducers=reducers)
        return updates

    return RunnableLambda(tool_node, afunc=atool_node)


def make_before_agent_node(
    mw_list: Sequence[AgentMiddleware],
    *,
    has_tools: bool,
    terminal_target: str,
    reducers: ReducerMap | None = None,
) -> Runnable[Any, Any]:
    """Construct the sync + async ``before_agent`` lifecycle node with native ``JumpTo`` routing."""

    def before_agent_node(state: Any, config: RunnableConfig | None = None) -> Command[str]:
        del config
        working_state = dict(state)
        updates: dict[str, Any] = {}

        for mw in mw_list:
            res = mw.before_agent(working_state)
            if isinstance(res, JumpTo):
                return _build_jump_command(
                    res,
                    updates,
                    has_tools=has_tools,
                    terminal_target=terminal_target,
                    reducers=reducers,
                )
            if isinstance(res, Mapping):
                _apply_hook_result(updates, working_state, res, reducers)

        return Command(update=updates, goto=MODEL_NODE)

    async def abefore_agent_node(
        state: Any, config: RunnableConfig | None = None
    ) -> Command[str]:
        del config
        working_state = dict(state)
        updates: dict[str, Any] = {}

        for mw in mw_list:
            res = await mw.abefore_agent(working_state)
            if isinstance(res, JumpTo):
                return _build_jump_command(
                    res,
                    updates,
                    has_tools=has_tools,
                    terminal_target=terminal_target,
                    reducers=reducers,
                )
            if isinstance(res, Mapping):
                _apply_hook_result(updates, working_state, res, reducers)

        return Command(update=updates, goto=MODEL_NODE)

    return RunnableLambda(before_agent_node, afunc=abefore_agent_node)


def make_after_agent_node(
    mw_list: Sequence[AgentMiddleware],
    *,
    reducers: ReducerMap | None = None,
) -> Runnable[Any, Any]:
    """Construct the sync + async ``after_agent`` lifecycle node (reverse registration order)."""

    def after_agent_node(state: Any, config: RunnableConfig | None = None) -> dict[str, Any]:
        del config
        working_state = dict(state)
        updates: dict[str, Any] = {}
        for mw in reversed(mw_list):
            res = mw.after_agent(working_state)
            if isinstance(res, JumpTo):
                raise ConfigurationError(
                    "after_agent middleware cannot return JumpTo because after_agent is terminal."
                )
            if isinstance(res, Mapping):
                _apply_hook_result(updates, working_state, res, reducers)
        return updates

    async def aafter_agent_node(
        state: Any, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        del config
        working_state = dict(state)
        updates: dict[str, Any] = {}
        for mw in reversed(mw_list):
            res = await mw.aafter_agent(working_state)
            if isinstance(res, JumpTo):
                raise ConfigurationError(
                    "after_agent middleware cannot return JumpTo because after_agent is terminal."
                )
            if isinstance(res, Mapping):
                _apply_hook_result(updates, working_state, res, reducers)
        return updates

    return RunnableLambda(after_agent_node, afunc=aafter_agent_node)


def make_call_model_node(
    model: BaseChatModel | Runnable[Any, Any],
    combined_tools: Sequence[BaseTool],
    system_prompt: str | None,
    mw_list: Sequence[AgentMiddleware],
    wrapped_model_handler: ModelHandler,
    wrapped_async_model_handler: AsyncModelHandler | None = None,
    *,
    has_tools: bool,
    terminal_target: str,
    reducers: ReducerMap | None = None,
) -> Runnable[Any, Any]:
    """Construct the sync + async model invocation node.

    Runs ``before_model`` hooks in forward order, dispatches through the
    ``wrap_model_call`` / ``awrap_model_call`` onion pipeline, runs
    ``after_model`` hooks in reverse order, and routes via
    :class:`~langgraph.types.Command`.
    """

    def call_model_node(state: Any, config: RunnableConfig | None = None) -> Command[str]:
        working_state = dict(state)
        working_state.setdefault("messages", [])
        output_state: dict[str, Any] = {}

        # 1. before_model hooks (forward order)
        for mw in mw_list:
            res = mw.before_model(working_state)
            if isinstance(res, JumpTo):
                return _build_jump_command(
                    res,
                    output_state,
                    has_tools=has_tools,
                    terminal_target=terminal_target,
                    reducers=reducers,
                )
            if isinstance(res, Mapping):
                _apply_hook_result(output_state, working_state, res, reducers)

        # 2. wrap_model_call onion pipeline
        req = ModelRequest(
            messages=tuple(working_state.get("messages", ())),
            system_prompt=system_prompt,
            model=model,
            tools=tuple(combined_tools),
            state=dict(working_state),
            config=config,
        )
        model_response = wrapped_model_handler(req)
        interim_state = _record_model_response(
            output_state, working_state, model_response, reducers
        )

        # 3. after_model hooks (reverse order)
        for mw in reversed(mw_list):
            res = mw.after_model(interim_state)
            if isinstance(res, JumpTo):
                return _build_jump_command(
                    res,
                    output_state,
                    has_tools=has_tools,
                    terminal_target=terminal_target,
                    reducers=reducers,
                )
            if isinstance(res, Mapping):
                _apply_hook_result(output_state, interim_state, res, reducers)

        # 4. Default routing based on emitted tool calls
        return _route_model_completion(
            model_response.message,
            output_state,
            has_tools=has_tools,
            terminal_target=terminal_target,
        )

    async def acall_model_node(
        state: Any, config: RunnableConfig | None = None
    ) -> Command[str]:
        if wrapped_async_model_handler is None:
            return call_model_node(state, config=config)

        working_state = dict(state)
        working_state.setdefault("messages", [])
        output_state: dict[str, Any] = {}

        # 1. abefore_model hooks (forward order)
        for mw in mw_list:
            res = await mw.abefore_model(working_state)
            if isinstance(res, JumpTo):
                return _build_jump_command(
                    res,
                    output_state,
                    has_tools=has_tools,
                    terminal_target=terminal_target,
                    reducers=reducers,
                )
            if isinstance(res, Mapping):
                _apply_hook_result(output_state, working_state, res, reducers)

        # 2. awrap_model_call onion pipeline
        req = ModelRequest(
            messages=tuple(working_state.get("messages", ())),
            system_prompt=system_prompt,
            model=model,
            tools=tuple(combined_tools),
            state=dict(working_state),
            config=config,
        )
        model_response = await wrapped_async_model_handler(req)
        interim_state = _record_model_response(
            output_state, working_state, model_response, reducers
        )

        # 3. aafter_model hooks (reverse order)
        for mw in reversed(mw_list):
            res = await mw.aafter_model(interim_state)
            if isinstance(res, JumpTo):
                return _build_jump_command(
                    res,
                    output_state,
                    has_tools=has_tools,
                    terminal_target=terminal_target,
                    reducers=reducers,
                )
            if isinstance(res, Mapping):
                _apply_hook_result(output_state, interim_state, res, reducers)

        # 4. Default routing based on emitted tool calls
        return _route_model_completion(
            model_response.message,
            output_state,
            has_tools=has_tools,
            terminal_target=terminal_target,
        )

    return RunnableLambda(call_model_node, afunc=acall_model_node)
