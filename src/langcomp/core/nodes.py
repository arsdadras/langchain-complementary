"""Graph node builders for the langcomp hand-built agent execution loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.types import Command

from langcomp.errors import ConfigurationError, UnknownToolError
from langcomp.middleware.base import AgentMiddleware
from langcomp.middleware.types import (
    JumpTo,
    ModelHandler,
    ModelRequest,
    ToolCallRequest,
    ToolCallResponse,
    ToolHandler,
)

TOOLS_NODE = "tools"
MODEL_NODE = "model"
BEFORE_AGENT_NODE = "before_agent"
AFTER_AGENT_NODE = "after_agent"


def has_hook(mw: AgentMiddleware, hook_name: str) -> bool:
    """Return ``True`` if ``mw`` actively implements ``hook_name``."""
    return mw.has_hook(hook_name)


def _merge_update(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    """Merge a partial state update dict into ``target``, concatenating ``messages``."""
    for key, value in update.items():
        if key == "messages" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            existing = list(target.get("messages", []))
            existing.extend(value)
            target["messages"] = existing
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


def run_one_tool_call(
    tool_by_name: Mapping[str, BaseTool],
    call: ToolCall,
    wrapped_handler: ToolHandler,
    state: Mapping[str, Any],
) -> ToolCallResponse:
    """Execute a single tool call through the composed middleware onion pipeline."""
    tool_inst = tool_by_name.get(call["name"])
    if tool_inst is None:
        error = UnknownToolError(call["name"], tuple(tool_by_name))
        return ToolCallResponse(
            message=ToolMessage(
                content=str(error),
                tool_call_id=call["id"],
                status="error",
            )
        )

    req = ToolCallRequest(tool=tool_inst, tool_call=call, state=dict(state))
    return wrapped_handler(req)


def make_tool_node(
    tools: Sequence[BaseTool],
    wrapped_handler: ToolHandler,
) -> Any:
    """Construct the tool-execution node function for ``StateGraph``."""
    tool_by_name = {t.name: t for t in tools}

    def tool_node(state: Any) -> dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise ConfigurationError(
                f"Expected last message to be AIMessage before tool execution, got {type(last_message).__name__}."
            )
        messages: list[ToolMessage] = []
        updates: dict[str, Any] = {}
        for call in last_message.tool_calls:
            resp = run_one_tool_call(tool_by_name, call, wrapped_handler, state)
            messages.append(resp.message)
            if resp.state_update:
                _merge_update(updates, resp.state_update)
        _merge_update(updates, {"messages": messages})
        return updates

    return tool_node


def make_before_agent_node(
    mw_list: Sequence[AgentMiddleware],
    *,
    has_tools: bool,
    terminal_target: str,
) -> Any:
    """Construct the ``before_agent`` lifecycle node with native ``JumpTo`` routing."""

    def before_agent_node(state: Any) -> Command[str]:
        working_state = dict(state)
        updates: dict[str, Any] = {}

        for mw in mw_list:
            res = mw.before_agent(working_state)
            if isinstance(res, JumpTo):
                if res.state_update:
                    _merge_update(updates, res.state_update)
                if res.messages:
                    _merge_update(updates, {"messages": list(res.messages)})
                goto = resolve_jump_target(
                    res,
                    has_tools=has_tools,
                    terminal_target=terminal_target,
                )
                return Command(update=updates, goto=goto)
            if isinstance(res, Mapping):
                _merge_update(updates, res)
                working_state = {**working_state, **res}

        return Command(update=updates, goto=MODEL_NODE)

    return before_agent_node


def make_after_agent_node(
    mw_list: Sequence[AgentMiddleware],
) -> Any:
    """Construct the ``after_agent`` lifecycle node (runs in reverse registration order)."""

    def after_agent_node(state: Any) -> dict[str, Any]:
        working_state = dict(state)
        updates: dict[str, Any] = {}
        for mw in reversed(mw_list):
            res = mw.after_agent(working_state)
            if isinstance(res, Mapping):
                _merge_update(updates, res)
                working_state = {**working_state, **res}
        return updates

    return after_agent_node


def make_call_model_node(
    bound_model: BaseChatModel | Runnable[Any, Any],
    combined_tools: Sequence[BaseTool],
    system_prompt: str | None,
    mw_list: Sequence[AgentMiddleware],
    wrapped_model_handler: ModelHandler,
    *,
    has_tools: bool,
    terminal_target: str,
) -> Any:
    """Construct the model invocation node.

    Runs ``before_model`` hooks in forward order, dispatches through the
    ``wrap_model_call`` onion pipeline, runs ``after_model`` hooks in reverse
    order, and routes via :class:`~langgraph.types.Command`.
    """

    def call_model_node(state: Any) -> Command[str]:
        working_state = dict(state)
        output_state: dict[str, Any] = {}

        # 1. before_model hooks (forward order)
        for mw in mw_list:
            res = mw.before_model(working_state)
            if isinstance(res, JumpTo):
                if res.state_update:
                    _merge_update(output_state, res.state_update)
                if res.messages:
                    _merge_update(output_state, {"messages": list(res.messages)})
                goto = resolve_jump_target(
                    res,
                    has_tools=has_tools,
                    terminal_target=terminal_target,
                )
                return Command(update=output_state, goto=goto)
            if isinstance(res, Mapping):
                _merge_update(output_state, res)
                if "messages" in res:
                    working_state["messages"] = [*working_state["messages"], *res["messages"]]
                for key, val in res.items():
                    if key != "messages":
                        working_state[key] = val

        # 2. wrap_model_call onion pipeline
        req = ModelRequest(
            messages=tuple(working_state["messages"]),
            system_prompt=system_prompt,
            model=bound_model,
            tools=tuple(combined_tools),
            state=dict(working_state),
        )
        model_response = wrapped_model_handler(req)
        _merge_update(output_state, {"messages": [model_response.message]})
        if model_response.state_update:
            _merge_update(output_state, model_response.state_update)

        # 3. after_model hooks (reverse order)
        interim_messages: list[AnyMessage] = [
            *working_state["messages"],
            model_response.message,
        ]
        interim_state: dict[str, Any] = {
            **working_state,
            **model_response.state_update,
            "messages": interim_messages,
        }
        for mw in reversed(mw_list):
            res = mw.after_model(interim_state)
            if isinstance(res, JumpTo):
                if res.state_update:
                    _merge_update(output_state, res.state_update)
                if res.messages:
                    _merge_update(output_state, {"messages": list(res.messages)})
                goto = resolve_jump_target(
                    res,
                    has_tools=has_tools,
                    terminal_target=terminal_target,
                )
                return Command(update=output_state, goto=goto)
            if isinstance(res, Mapping):
                _merge_update(output_state, res)
                if "messages" in res:
                    interim_state["messages"] = [*interim_state["messages"], *res["messages"]]
                for key, val in res.items():
                    if key != "messages":
                        interim_state[key] = val

        # 4. Default routing based on emitted tool calls
        if has_tools and isinstance(model_response.message, AIMessage) and model_response.message.tool_calls:
            return Command(update=output_state, goto=TOOLS_NODE)
        return Command(update=output_state, goto=terminal_target)

    return call_model_node
