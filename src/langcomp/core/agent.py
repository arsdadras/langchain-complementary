"""
Declarative Agent Core for langcomp.
Provides Agent, AgentBuilder, create_agent, @agent, and @node (with sync/async idempotent=True support).
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from langcomp.core.loop import create_agent_graph
from langcomp.core.state import SmartState, State
from langcomp.errors import ConfigurationError
from langcomp.middleware.base import AgentMiddleware

NodeFn = Callable[[Any], Any]


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or (
        not isinstance(fn, type) and callable(fn) and inspect.iscoroutinefunction(type(fn).__call__)
    )


@dataclass
class Node:
    """A named, callable graph node, optionally wrapped with ``idempotent=True``.

    When ``idempotent=True``, a node that has already completed successfully on the
    current checkpointed thread records its name in ``_langcomp_done`` and no-ops
    on subsequent re-entries on the same thread (e.g., after an ``interrupt()`` resume
    of a downstream node). Place side effects in their own node AFTER any node that
    calls ``interrupt()``.
    """

    fn: NodeFn
    name: str
    idempotent: bool = False

    def __call__(self, state: Any) -> Any:
        return self.fn(state)

    def invoke_for_test(self, state: Any) -> Any:
        """Execute the node function directly without compiling a graph."""
        return self.fn(state)


@overload
def node(fn: NodeFn, *, name: str | None = None, idempotent: bool = False) -> Node: ...


@overload
def node(
    fn: None = None,
    *,
    name: str | None = None,
    idempotent: bool = False,
) -> Callable[[NodeFn], Node]: ...


def node(
    fn: NodeFn | None = None,
    *,
    name: str | None = None,
    idempotent: bool = False,
) -> Node | Callable[[NodeFn], Node]:
    """Wrap a plain node function into a :class:`Node`.

    Usable as ``@node``, ``@node(idempotent=True)``, or ``@node(name="custom")``.
    """

    def decorator(func: NodeFn) -> Node:
        return Node(fn=func, name=name or func.__name__, idempotent=idempotent)

    if fn is not None:
        return decorator(fn)
    return decorator


def _append_done_marker(update: dict[str, Any], node_name: str) -> dict[str, Any]:
    existing = update.get("_langcomp_done")
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        update["_langcomp_done"] = [*existing, node_name]
    else:
        update["_langcomp_done"] = [node_name]
    return update


def _attach_done_marker(raw_result: Any, node_name: str) -> Any:
    if isinstance(raw_result, Command):
        raw_update = raw_result.update
        if isinstance(raw_update, Mapping):
            new_update: Any = _append_done_marker(dict(raw_update), node_name)
        elif isinstance(raw_update, list):
            new_update = [*raw_update, ("_langcomp_done", [node_name])]
        elif isinstance(raw_update, BaseModel):
            new_update = _append_done_marker(raw_update.model_dump(), node_name)
        else:
            new_update = {"_langcomp_done": [node_name]}
        return Command(
            graph=raw_result.graph,
            update=new_update,
            resume=raw_result.resume,
            goto=raw_result.goto,
        )
    result = dict(raw_result or {})
    return _append_done_marker(result, node_name)


def _node_runner(n: Node) -> NodeFn:
    if not n.idempotent:
        return n.fn

    if _is_async_callable(n.fn):

        async def async_runner(state: dict[str, Any]) -> Any:
            done = state.get("_langcomp_done") or ()
            if n.name in done:
                return {}
            return _attach_done_marker(await n.fn(state), n.name)

        async_runner.__name__ = n.name
        return async_runner

    def sync_runner(state: dict[str, Any]) -> Any:
        done = state.get("_langcomp_done") or ()
        if n.name in done:
            return {}
        return _attach_done_marker(n.fn(state), n.name)

    sync_runner.__name__ = n.name
    return sync_runner


class Agent:
    """Wrapper around a compiled LangGraph / LangChain runnable.

    Provides normalized ``.invoke()``, ``.ainvoke()``, ``.stream()``, ``.astream()``,
    ``.get_state()``, ``.aget_state()``, ``.update_state()``, and ``.aupdate_state()``
    execution interfaces.
    """

    def __init__(
        self,
        name: str,
        compiled_graph: Any,
        state_schema: Any = None,
        *,
        state_model: type[State] | None = None,
    ):
        self.name = name
        self.graph = compiled_graph
        self.state_schema = state_schema
        self._state_model = state_model

    @classmethod
    def from_runnable(cls, runnable: Any, name: str = "LangChainAgent") -> Agent:
        """Wrap any compiled runnable from LangChain ``create_agent()`` or DeepAgents."""
        return cls(name=name, compiled_graph=runnable, state_schema=None)

    @staticmethod
    def _prepare_payload(input_data: Any) -> Any:
        if isinstance(input_data, (str, BaseMessage)):
            return {"messages": SmartState.normalize_messages([input_data])}
        if isinstance(input_data, list):
            return {"messages": SmartState.normalize_messages(input_data)}
        if isinstance(input_data, dict):
            payload = {
                k: v for k, v in input_data.items() if not str(k).startswith("_langcomp")
            }
            if "messages" in payload:
                msgs = payload["messages"]
                if isinstance(msgs, (str, BaseMessage)):
                    payload["messages"] = SmartState.normalize_messages([msgs])
                elif isinstance(msgs, list):
                    payload["messages"] = SmartState.normalize_messages(msgs)
            return payload
        return input_data

    def _apply_state_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._state_model is not None
        defaults = self._state_model._defaults()
        return {**defaults, **payload} if defaults else payload

    def _hydrate_state_defaults_sync(
        self,
        payload: Any,
        config: dict[str, Any],
    ) -> Any:
        if self._state_model is None or not isinstance(payload, dict):
            return payload
        if getattr(self.graph, "checkpointer", None) is not None and hasattr(
            self.graph, "get_state"
        ):
            snapshot = self.graph.get_state(config)
            if getattr(snapshot, "values", None):
                return payload
        return self._apply_state_defaults(payload)

    async def _hydrate_state_defaults_async(
        self,
        payload: Any,
        config: dict[str, Any],
    ) -> Any:
        if self._state_model is None or not isinstance(payload, dict):
            return payload
        if getattr(self.graph, "checkpointer", None) is not None and hasattr(
            self.graph, "aget_state"
        ):
            snapshot = await self.graph.aget_state(config)
            if getattr(snapshot, "values", None):
                return payload
        return self._apply_state_defaults(payload)

    def _default_config(self, config: Mapping[str, Any] | None) -> dict[str, Any]:
        merged: dict[str, Any] = dict(config or {})
        configurable: dict[str, Any] = dict(merged.get("configurable") or {})
        if "thread_id" not in configurable or not configurable["thread_id"]:
            if getattr(self.graph, "checkpointer", None) is not None:
                raise ConfigurationError(
                    f"Agent {self.name!r} has a checkpointer enabled and requires an explicit "
                    f"config={{'configurable': {{'thread_id': '...'}}}} to prevent cross-session state leakage."
                )
            configurable["thread_id"] = uuid.uuid4().hex
        merged["configurable"] = configurable
        return merged

    def invoke(
        self,
        input_data: Any,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronously invoke the compiled agent."""
        resolved_config = self._default_config(config)
        payload = self._hydrate_state_defaults_sync(
            self._prepare_payload(input_data), resolved_config
        )
        result: dict[str, Any] = self.graph.invoke(payload, config=resolved_config)
        return result

    async def ainvoke(
        self,
        input_data: Any,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Asynchronously invoke the compiled agent."""
        resolved_config = self._default_config(config)
        payload = await self._hydrate_state_defaults_async(
            self._prepare_payload(input_data), resolved_config
        )
        result: dict[str, Any] = await self.graph.ainvoke(payload, config=resolved_config)
        return result

    def stream(
        self,
        input_data: Any,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream intermediate updates from the compiled agent."""
        resolved_config = self._default_config(config)
        payload = self._hydrate_state_defaults_sync(
            self._prepare_payload(input_data), resolved_config
        )
        stream_iter: Iterator[Any] = self.graph.stream(
            payload, config=resolved_config, **kwargs
        )
        return stream_iter

    async def astream(
        self,
        input_data: Any,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Asynchronously stream intermediate updates from the compiled agent."""
        resolved_config = self._default_config(config)
        payload = await self._hydrate_state_defaults_async(
            self._prepare_payload(input_data), resolved_config
        )
        async for chunk in self.graph.astream(
            payload, config=resolved_config, **kwargs
        ):
            yield chunk

    def get_state(self, config: Mapping[str, Any]) -> Any:
        """Retrieve current agent state snapshot for a given ``thread_id`` config."""
        return self.graph.get_state(self._default_config(config))

    async def aget_state(self, config: Mapping[str, Any]) -> Any:
        """Asynchronously retrieve current agent state snapshot for a ``thread_id`` config."""
        return await self.graph.aget_state(self._default_config(config))

    def update_state(
        self,
        config: Mapping[str, Any],
        values: Mapping[str, Any],
        as_node: str | None = None,
    ) -> Any:
        """Update checkpointed agent state synchronously."""
        return self.graph.update_state(
            self._default_config(config), values, as_node=as_node
        )

    async def aupdate_state(
        self,
        config: Mapping[str, Any],
        values: Mapping[str, Any],
        as_node: str | None = None,
    ) -> Any:
        """Update checkpointed agent state asynchronously."""
        return await self.graph.aupdate_state(
            self._default_config(config), values, as_node=as_node
        )


class AgentBuilder:
    """Fluent builder for constructing LangGraph agents without boilerplate."""

    def __init__(self, name: str = "LangCompAgent"):
        self.name = name
        self._model: BaseChatModel | BaseLanguageModel[Any] | str | None = None
        self._model_kwargs: dict[str, Any] = {}
        self._tools: list[BaseTool | Callable[..., Any]] = []
        self._middleware: list[AgentMiddleware] = []
        self._system_prompt: str | None = None
        self._state_fields: dict[str, Any] = {}
        self._state_model: type[State] | None = None
        self._custom_nodes: dict[str, Node | NodeFn] = {}
        self._edges: list[tuple[str, str]] = []
        self._checkpointer: BaseCheckpointSaver[Any] | None = None

    def with_model(
        self,
        model: BaseChatModel | BaseLanguageModel[Any] | str,
        **model_kwargs: Any,
    ) -> AgentBuilder:
        """Set the chat model instance or ``"provider:model"`` spec string."""
        self._model = model
        self._model_kwargs = dict(model_kwargs)
        return self

    def with_model_kwargs(self, **kwargs: Any) -> AgentBuilder:
        """Configure additional keyword arguments forwarded to :func:`resolve_model`."""
        self._model_kwargs.update(kwargs)
        return self

    def with_tools(self, tools: Sequence[BaseTool | Callable[..., Any]]) -> AgentBuilder:
        self._tools.extend(tools)
        return self

    def with_middleware(
        self,
        middleware: Sequence[AgentMiddleware],
    ) -> AgentBuilder:
        """Attach a sequence of middleware instances or decorated hook functions."""
        self._middleware.extend(middleware)
        return self

    def add_middleware(
        self,
        middleware: AgentMiddleware,
    ) -> AgentBuilder:
        """Attach a single middleware instance or decorated hook function."""
        self._middleware.append(middleware)
        return self

    def with_system_prompt(self, prompt: str) -> AgentBuilder:
        self._system_prompt = prompt
        return self

    def with_state_schema(
        self,
        extra_fields_or_state: dict[str, Any] | type[State],
    ) -> AgentBuilder:
        """Configure custom state fields via a dict or a :class:`State` subclass."""
        if isinstance(extra_fields_or_state, type) and issubclass(
            extra_fields_or_state, State
        ):
            if self._state_fields:
                raise ValueError(
                    "Cannot mix a State subclass with dict extra_fields in with_state_schema()."
                )
            self._state_model = extra_fields_or_state
        elif isinstance(extra_fields_or_state, dict):
            if self._state_model is not None:
                raise ValueError(
                    "Cannot mix dict extra_fields with a State subclass in with_state_schema()."
                )
            self._state_fields.update(extra_fields_or_state)
        else:
            raise TypeError(
                "with_state_schema expects a dict of field types or a langcomp.State subclass."
            )
        return self

    def with_checkpointer(self, checkpointer: BaseCheckpointSaver[Any]) -> AgentBuilder:
        self._checkpointer = checkpointer
        return self

    def add_node(
        self,
        name_or_node: str | Node,
        func: NodeFn | None = None,
    ) -> AgentBuilder:
        """Register a graph node by ``(name, func)`` or by passing a :class:`Node`."""
        if isinstance(name_or_node, Node):
            node_name = name_or_node.name
            node_obj: Node | NodeFn = name_or_node
        else:
            if func is None:
                raise ValueError("add_node(name, func) requires a callable 'func'.")
            node_name = name_or_node
            node_obj = func

        if node_name in self._custom_nodes:
            raise ValueError(
                f"A node named {node_name!r} was already added to AgentBuilder {self.name!r}; "
                f"node names must be unique."
            )
        self._custom_nodes[node_name] = node_obj
        return self

    def add_edge(
        self,
        start_node: str | Node,
        end_node: str | Node,
    ) -> AgentBuilder:
        start_name = start_node.name if isinstance(start_node, Node) else start_node
        end_name = end_node.name if isinstance(end_node, Node) else end_node
        self._edges.append((start_name, end_name))
        return self

    def _build_state_schema(self) -> Any:
        if self._state_model is not None:
            return self._state_model.to_typed_dict(include_internal=True)
        return SmartState.create_schema(extra_fields=self._state_fields)

    def _validate_edges(self) -> None:
        valid_endpoints = set(self._custom_nodes.keys()) | {START, END}
        for start, end in self._edges:
            if start not in valid_endpoints:
                raise ValueError(
                    f"Edge source {start!r} is not a registered node in {self.name!r}."
                )
            if end not in valid_endpoints:
                raise ValueError(
                    f"Edge target {end!r} is not a registered node in {self.name!r}."
                )

    def _compile_model_agent(self, base_schema: Any) -> tuple[Any, type[Any]]:
        explicit_schema = base_schema if self._state_model is not None else None
        return create_agent_graph(
            model=self._model,
            tools=self._tools,
            system_prompt=self._system_prompt,
            middleware=self._middleware,
            extra_state_fields=self._state_fields if explicit_schema is None else None,
            state_schema=explicit_schema,
            checkpointer=self._checkpointer,
            **self._model_kwargs,
        )

    def compile(self) -> Agent:
        """Compile builder configuration into a ready-to-run :class:`Agent`."""
        state_schema = self._build_state_schema()

        if self._custom_nodes:
            if (
                self._model is not None
                or self._tools
                or self._middleware
                or self._system_prompt is not None
                or self._model_kwargs
            ):
                raise ValueError(
                    f"AgentBuilder {self.name!r} cannot combine custom nodes with "
                    f"with_model()/with_tools()/with_middleware()/with_system_prompt(); "
                    f"use either custom graph nodes or a model-backed agent."
                )
            self._validate_edges()
            builder: StateGraph[Any, Any, Any, Any] = StateGraph(state_schema)
            for node_name, node_obj in self._custom_nodes.items():
                runner: Any = _node_runner(node_obj) if isinstance(node_obj, Node) else node_obj
                builder.add_node(node_name, runner)

            if not self._edges:
                node_names = list(self._custom_nodes.keys())
                builder.add_edge(START, node_names[0])
                for i in range(len(node_names) - 1):
                    builder.add_edge(node_names[i], node_names[i + 1])
                builder.add_edge(node_names[-1], END)
            else:
                for start, end in self._edges:
                    builder.add_edge(start, end)

            compiled = builder.compile(checkpointer=self._checkpointer)
            return Agent(
                name=self.name,
                compiled_graph=compiled,
                state_schema=state_schema,
                state_model=self._state_model,
            )

        if self._model is not None:
            compiled, resolved_schema = self._compile_model_agent(state_schema)
            return Agent(
                name=self.name,
                compiled_graph=compiled,
                state_schema=resolved_schema,
                state_model=self._state_model,
            )

        raise ValueError("AgentBuilder requires either a model or custom nodes to compile.")


def create_agent(
    model: str | BaseChatModel | BaseLanguageModel[Any] | None = None,
    tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    system_prompt: str | None = None,
    *,
    name: str = "LangCompAgent",
    middleware: Sequence[AgentMiddleware] | None = None,
    extra_state_fields: Mapping[str, Any] | None = None,
    state_schema: type[Any] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    **model_kwargs: Any,
) -> Agent:
    """Build and compile a tool-calling :class:`Agent` on a hand-built ``StateGraph``."""
    state_model: type[State] | None = None
    explicit_schema = state_schema
    if isinstance(state_schema, type) and issubclass(state_schema, State):
        state_model = state_schema
        explicit_schema = state_schema.to_typed_dict(include_internal=True)

    compiled, resolved_schema = create_agent_graph(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        extra_state_fields=extra_state_fields,
        state_schema=explicit_schema,
        checkpointer=checkpointer,
        **model_kwargs,
    )
    return Agent(
        name=name,
        compiled_graph=compiled,
        state_schema=resolved_schema,
        state_model=state_model,
    )


def agent(
    name: str = "DecoratedAgent",
    model: Any = None,
    tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Callable[[Any], Agent]:
    """Decorator for quick agent creation from a node function or LLM configuration.

    - When ``model`` is supplied, builds a model-backed tool-calling :class:`Agent`
      (using the decorated function or class docstring as a fallback ``system_prompt``).
    - When ``model`` is omitted, wraps a single-node function or :class:`Node` into
      a compiled single-node :class:`Agent`.
    """

    def decorator(func_or_class: Any) -> Agent:
        builder = AgentBuilder(name=name)
        if checkpointer is not None:
            builder.with_checkpointer(checkpointer)

        if model is not None:
            builder.with_model(model, **(dict(model_kwargs) if model_kwargs else {}))
            if tools:
                builder.with_tools(tools)
            if middleware:
                builder.with_middleware(middleware)
            effective_prompt = system_prompt
            if effective_prompt is None:
                doc = inspect.getdoc(func_or_class)
                if doc:
                    effective_prompt = doc
            if effective_prompt is not None:
                builder.with_system_prompt(effective_prompt)
            return builder.compile()

        if tools or middleware or system_prompt is not None or model_kwargs:
            raise ConfigurationError(
                "@agent with tools/middleware/system_prompt/model_kwargs requires 'model'."
            )

        if isinstance(func_or_class, Node):
            builder.add_node(func_or_class)
            builder.add_edge(START, func_or_class.name)
            builder.add_edge(func_or_class.name, END)
            return builder.compile()

        if callable(func_or_class) and not isinstance(func_or_class, type):
            builder.add_node(name, func_or_class)
            builder.add_edge(START, name)
            builder.add_edge(name, END)
            return builder.compile()

        raise ConfigurationError(
            f"@agent without 'model' expects a node function or Node, got {type(func_or_class).__name__}."
        )

    return decorator
