"""
Declarative Agent Core for langcomp.
Provides Agent, AgentBuilder, @agent, and @node (with sync/async idempotent=True support).
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Type, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langcomp.core.state import SmartState, State

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
        return self.graph.invoke(payload, config=self._default_config(config))

    async def ainvoke(
        self,
        input_data: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously invoke the compiled agent."""
        payload = self._prepare_payload(input_data)
        return await self.graph.ainvoke(payload, config=self._default_config(config))

    def stream(
        self,
        input_data: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream intermediate updates from the compiled agent."""
        payload = self._prepare_payload(input_data)
        return self.graph.stream(payload, config=self._default_config(config), **kwargs)

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
        self._model: Optional[Union[BaseLanguageModel, str]] = None
        self._tools: List[Union[BaseTool, Callable[..., Any]]] = []
        self._system_prompt: Optional[str] = None
        self._state_fields: Dict[str, Any] = {}
        self._state_model: Optional[Type[State]] = None
        self._custom_nodes: Dict[str, Union[Node, NodeFn]] = {}
        self._edges: List[tuple[str, str]] = []
        self._checkpointer: Optional[BaseCheckpointSaver] = None

    def with_model(self, model: Union[BaseLanguageModel, str]) -> "AgentBuilder":
        self._model = model
        return self

    def with_tools(self, tools: List[Union[BaseTool, Callable[..., Any]]]) -> "AgentBuilder":
        self._tools.extend(tools)
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

    def with_checkpointer(self, checkpointer: BaseCheckpointSaver) -> "AgentBuilder":
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

    def _compile_model_agent(self, state_schema: Any) -> Any:
        # Prefer LangChain v1 create_agent when `langchain` is installed.
        try:
            from langchain.agents import create_agent  # type: ignore[import-not-found]
        except ImportError:
            create_agent = None

        if create_agent is not None:
            kwargs: Dict[str, Any] = {
                "model": self._model,
                "tools": self._tools,
                "system_prompt": self._system_prompt,
                "checkpointer": self._checkpointer,
            }
            if self._state_model is not None or self._state_fields:
                kwargs["state_schema"] = state_schema
            return create_agent(**kwargs)

        # Fallback cleanly to langgraph.prebuilt.create_react_agent when only langgraph is installed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from langgraph.prebuilt import create_react_agent

            react_kwargs: Dict[str, Any] = {
                "model": self._model,
                "tools": self._tools,
                "prompt": self._system_prompt,
                "checkpointer": self._checkpointer,
            }
            if self._state_model is not None or self._state_fields:
                react_kwargs["state_schema"] = state_schema
            return create_react_agent(**react_kwargs)

    def compile(self) -> Agent:
        """Compile builder configuration into a ready-to-run :class:`Agent`."""
        state_schema = self._build_state_schema()

        if self._custom_nodes:
            if self._model is not None or self._tools:
                raise ValueError(
                    f"AgentBuilder {self.name!r} cannot combine custom nodes with "
                    f"with_model()/with_tools(); use either custom graph nodes or a model-backed agent."
                )
            self._validate_edges()
            builder = StateGraph(state_schema)
            for node_name, node_obj in self._custom_nodes.items():
                runner = _node_runner(node_obj) if isinstance(node_obj, Node) else node_obj
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
            compiled = self._compile_model_agent(state_schema)
            return Agent(name=self.name, compiled_graph=compiled, state_schema=state_schema)

        raise ValueError("AgentBuilder requires either a model or custom nodes to compile.")


def agent(
    name: str = "DecoratedAgent",
    model: Optional[Any] = None,
    tools: Optional[List[Any]] = None,
    system_prompt: Optional[str] = None,
) -> Callable[[Any], Agent]:
    """Decorator for quick agent creation from a node function or LLM config."""

    def decorator(func_or_class: Any) -> Agent:
        builder = AgentBuilder(name=name)
        if model is not None:
            builder.with_model(model)
        if tools:
            builder.with_tools(tools)
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
