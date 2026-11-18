"""
Declarative Agent Core for langcomp.
Provides Agent, AgentBuilder, create_agent, @agent, and @node (with sync/async idempotent=True support).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Type, Union

from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langcomp.core.loop import create_agent_graph
from langcomp.core.state import SmartState, State
from langcomp.middleware.base import AgentMiddleware

NodeFn = Callable[[Any], Any]


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


def node(
    fn: Optional[NodeFn] = None,
    *,
    name: Optional[str] = None,
    idempotent: bool = False,
) -> Any:
    """Wrap a plain node function into a :class:`Node`.

    Usable as ``@node``, ``@node(idempotent=True)``, or ``@node(name="custom")``.
    """

    def decorator(func: NodeFn) -> Node:
        return Node(fn=func, name=name or func.__name__, idempotent=idempotent)

    if fn is not None:
        return decorator(fn)
    return decorator


def _attach_done_marker(raw_result: Any, node_name: str) -> Any:
    if isinstance(raw_result, Command):
        update = dict(raw_result.update or {}) if isinstance(raw_result.update, dict) else {}
        update["_langcomp_done"] = [node_name]
        return Command(
            graph=raw_result.graph,
            update=update,
            resume=raw_result.resume,
            goto=raw_result.goto,
        )
    result = dict(raw_result or {})
    result["_langcomp_done"] = [node_name]
    return result


def _node_runner(n: Node) -> NodeFn:
    if not n.idempotent:
        return n.fn

    if inspect.iscoroutinefunction(n.fn):

        async def async_runner(state: Dict[str, Any]) -> Any:
            if n.name in state.get("_langcomp_done", []):
                return {}
            return _attach_done_marker(await n.fn(state), n.name)

        async_runner.__name__ = n.name
        return async_runner

    def sync_runner(state: Dict[str, Any]) -> Any:
        if n.name in state.get("_langcomp_done", []):
            return {}
        return _attach_done_marker(n.fn(state), n.name)

    sync_runner.__name__ = n.name
    return sync_runner


class Agent:
    """Wrapper around a compiled LangGraph / LangChain runnable.

    Provides normalized ``.invoke()``, ``.ainvoke()``, ``.stream()``, ``.astream()``,
    and ``.get_state()`` execution interfaces.
    """

    def __init__(self, name: str, compiled_graph: Any, state_schema: Any = None):
        self.name = name
        self.graph = compiled_graph
        self.state_schema = state_schema or SmartState.create_schema()

    @classmethod
    def from_runnable(cls, runnable: Any, name: str = "LangChainAgent") -> "Agent":
        """Wrap any compiled runnable from LangChain ``create_agent()`` or DeepAgents."""
        return cls(name=name, compiled_graph=runnable)

    @staticmethod
    def _prepare_payload(input_data: Any) -> Any:
        if isinstance(input_data, str):
            return {"messages": SmartState.normalize_messages([input_data])}
        if isinstance(input_data, list):
            return {"messages": SmartState.normalize_messages(input_data)}
        if isinstance(input_data, dict):
            payload = input_data.copy()
            if "messages" in payload and isinstance(payload["messages"], list):
                payload["messages"] = SmartState.normalize_messages(payload["messages"])
            return payload
        return input_data

    @staticmethod
    def _default_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if config is None:
            return {"configurable": {"thread_id": "default"}}
        merged = dict(config)
        configurable = dict(merged.get("configurable") or {})
        configurable.setdefault("thread_id", "default")
        merged["configurable"] = configurable
        return merged

    def invoke(
        self,
        input_data: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synchronously invoke the compiled agent."""
        payload = self._prepare_payload(input_data)
        result: Dict[str, Any] = self.graph.invoke(payload, config=self._default_config(config))
        return result

    async def ainvoke(
        self,
        input_data: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously invoke the compiled agent."""
        payload = self._prepare_payload(input_data)
        result: Dict[str, Any] = await self.graph.ainvoke(
            payload, config=self._default_config(config)
        )
        return result

    def stream(
        self,
        input_data: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream intermediate updates from the compiled agent."""
        payload = self._prepare_payload(input_data)
        stream_iter: Iterator[Any] = self.graph.stream(
            payload, config=self._default_config(config), **kwargs
        )
        return stream_iter

    async def astream(
        self,
        input_data: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Asynchronously stream intermediate updates from the compiled agent."""
        payload = self._prepare_payload(input_data)
        async for chunk in self.graph.astream(
            payload, config=self._default_config(config), **kwargs
        ):
            yield chunk

    def get_state(self, config: Dict[str, Any]) -> Any:
        """Retrieve current agent state snapshot for a given ``thread_id`` config."""
        return self.graph.get_state(config)


class AgentBuilder:
    """Fluent builder for constructing LangGraph agents without boilerplate."""

    def __init__(self, name: str = "LangCompAgent"):
        self.name = name
        self._model: Optional[Union[BaseChatModel, BaseLanguageModel[Any], str]] = None
        self._model_kwargs: Dict[str, Any] = {}
        self._tools: List[Union[BaseTool, Callable[..., Any]]] = []
        self._middleware: List[Union[AgentMiddleware, Callable[..., Any]]] = []
        self._system_prompt: Optional[str] = None
        self._state_fields: Dict[str, Any] = {}
        self._state_model: Optional[Type[State]] = None
        self._custom_nodes: Dict[str, Union[Node, NodeFn]] = {}
        self._edges: List[tuple[str, str]] = []
        self._checkpointer: Optional[BaseCheckpointSaver[Any]] = None

    def with_model(
        self,
        model: Union[BaseChatModel, BaseLanguageModel[Any], str],
        **model_kwargs: Any,
    ) -> "AgentBuilder":
        """Set the chat model instance or ``"provider:model"`` spec string."""
        self._model = model
        if model_kwargs:
            self._model_kwargs.update(model_kwargs)
        return self

    def with_model_kwargs(self, **kwargs: Any) -> "AgentBuilder":
        """Configure additional keyword arguments forwarded to :func:`resolve_model`."""
        self._model_kwargs.update(kwargs)
        return self

    def with_tools(self, tools: Sequence[Union[BaseTool, Callable[..., Any]]]) -> "AgentBuilder":
        self._tools.extend(tools)
        return self

    def with_middleware(
        self,
        middleware: Sequence[Union[AgentMiddleware, Callable[..., Any]]],
    ) -> "AgentBuilder":
        """Attach a sequence of middleware instances or decorated hook functions."""
        self._middleware.extend(middleware)
        return self

    def add_middleware(
        self,
        middleware: Union[AgentMiddleware, Callable[..., Any]],
    ) -> "AgentBuilder":
        """Attach a single middleware instance or decorated hook function."""
        self._middleware.append(middleware)
        return self

    def with_system_prompt(self, prompt: str) -> "AgentBuilder":
        self._system_prompt = prompt
        return self

    def with_state_schema(
        self,
        extra_fields_or_state: Union[Dict[str, Any], Type[State]],
    ) -> "AgentBuilder":
        """Configure custom state fields via a dict or a :class:`State` subclass."""
        if isinstance(extra_fields_or_state, type) and issubclass(extra_fields_or_state, State):
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

    def with_checkpointer(self, checkpointer: BaseCheckpointSaver[Any]) -> "AgentBuilder":
        self._checkpointer = checkpointer
        return self

    def add_node(
        self,
        name_or_node: Union[str, Node],
        func: Optional[NodeFn] = None,
    ) -> "AgentBuilder":
        """Register a graph node by ``(name, func)`` or by passing a :class:`Node`."""
        if isinstance(name_or_node, Node):
            node_name = name_or_node.name
            node_obj: Union[Node, NodeFn] = name_or_node
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
        start_node: Union[str, Node],
        end_node: Union[str, Node],
    ) -> "AgentBuilder":
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
            if self._model is not None or self._tools or self._middleware:
                raise ValueError(
                    f"AgentBuilder {self.name!r} cannot combine custom nodes with "
                    f"with_model()/with_tools()/with_middleware(); use either custom graph nodes "
                    f"or a model-backed agent."
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
            return Agent(name=self.name, compiled_graph=compiled, state_schema=state_schema)

        if self._model is not None:
            compiled, resolved_schema = self._compile_model_agent(state_schema)
            return Agent(name=self.name, compiled_graph=compiled, state_schema=resolved_schema)

        raise ValueError("AgentBuilder requires either a model or custom nodes to compile.")


def create_agent(
    model: Union[str, BaseChatModel, BaseLanguageModel[Any], None] = None,
    tools: Optional[Sequence[Union[BaseTool, Callable[..., Any]]]] = None,
    system_prompt: Optional[str] = None,
    *,
    name: str = "LangCompAgent",
    middleware: Optional[Sequence[Union[AgentMiddleware, Callable[..., Any]]]] = None,
    extra_state_fields: Optional[Mapping[str, Any]] = None,
    checkpointer: Optional[BaseCheckpointSaver[Any]] = None,
    **model_kwargs: Any,
) -> Agent:
    """Build and compile a tool-calling :class:`Agent` on a hand-built ``StateGraph``."""
    compiled, resolved_schema = create_agent_graph(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        extra_state_fields=extra_state_fields,
        checkpointer=checkpointer,
        **model_kwargs,
    )
    return Agent(name=name, compiled_graph=compiled, state_schema=resolved_schema)


def agent(
    name: str = "DecoratedAgent",
    model: Optional[Any] = None,
    tools: Optional[Sequence[Union[BaseTool, Callable[..., Any]]]] = None,
    system_prompt: Optional[str] = None,
    middleware: Optional[Sequence[Union[AgentMiddleware, Callable[..., Any]]]] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> Callable[[Any], Agent]:
    """Decorator for quick agent creation from a node function or LLM config."""

    def decorator(func_or_class: Any) -> Agent:
        builder = AgentBuilder(name=name)
        if model is not None:
            builder.with_model(model, **(dict(model_kwargs) if model_kwargs else {}))
        if tools:
            builder.with_tools(tools)
        if middleware:
            builder.with_middleware(middleware)
        if system_prompt:
            builder.with_system_prompt(system_prompt)

        if isinstance(func_or_class, Node):
            builder.add_node(func_or_class)
            builder.add_edge(START, func_or_class.name)
            builder.add_edge(func_or_class.name, END)
        elif callable(func_or_class) and not isinstance(func_or_class, type):
            builder.add_node(name, func_or_class)
            builder.add_edge(START, name)
            builder.add_edge(name, END)

        return builder.compile()

    return decorator
