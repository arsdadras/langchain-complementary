"""Declarative Agent Core for langcomp.

Provides :class:`Agent`, fluent :class:`AgentBuilder`, :func:`create_agent`,
and the :func:`agent` decorator built directly on LangGraph's ``StateGraph``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from langcomp.core.loop import create_agent_graph
from langcomp.core.state import SmartState
from langcomp.errors import ConfigurationError
from langcomp.middleware.base import AgentMiddleware


class Agent:
    """Wrapper around a compiled LangGraph runnable.

    Provides ergonomic execution (:meth:`invoke`, :meth:`ainvoke`) with automatic
    message normalization and state inspection (:meth:`get_state`).
    """

    def __init__(self, name: str, compiled_graph: Any, state_schema: Any = None) -> None:
        self.name = name
        self.graph = compiled_graph
        self.state_schema = state_schema or SmartState.create_schema()

    @classmethod
    def from_runnable(cls, runnable: Any, name: str = "LangChainAgent") -> Agent:
        """Wrap any compiled runnable from LangChain ``create_agent`` or DeepAgents."""
        return cls(name=name, compiled_graph=runnable)

    @staticmethod
    def _build_payload(input_data: str | dict[str, Any] | list[Any]) -> dict[str, Any]:
        if isinstance(input_data, str):
            return {"messages": SmartState.normalize_messages([input_data])}
        if isinstance(input_data, list):
            return {"messages": SmartState.normalize_messages(input_data)}
        if isinstance(input_data, dict):
            payload = input_data.copy()
            if "messages" in payload and isinstance(payload["messages"], list):
                payload["messages"] = SmartState.normalize_messages(payload["messages"])
            return payload
        return {"messages": []}

    def invoke(
        self,
        input_data: str | dict[str, Any] | list[Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the compiled agent with a string prompt, dict state, or message list."""
        resolved_config = config or {"configurable": {"thread_id": "default"}}
        payload = self._build_payload(input_data)
        result: dict[str, Any] = self.graph.invoke(payload, config=resolved_config)
        return result

    async def ainvoke(
        self,
        input_data: str | dict[str, Any] | list[Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Asynchronously invoke the compiled agent."""
        resolved_config = config or {"configurable": {"thread_id": "default"}}
        payload = self._build_payload(input_data)
        result: dict[str, Any] = await self.graph.ainvoke(payload, config=resolved_config)
        return result

    def get_state(self, config: dict[str, Any]) -> Any:
        """Retrieve current agent state snapshot for a given ``thread_id`` config."""
        return self.graph.get_state(config)


class AgentBuilder:
    """Fluent builder for constructing LangGraph agents without graph ceremony."""

    def __init__(self, name: str = "LangCompAgent") -> None:
        self.name = name
        self._model: BaseChatModel | BaseLanguageModel[Any] | str | None = None
        self._model_kwargs: dict[str, Any] = {}
        self._tools: list[BaseTool | Callable[..., Any]] = []
        self._middleware: list[AgentMiddleware | Callable[..., Any]] = []
        self._system_prompt: str | None = None
        self._state_fields: dict[str, Any] = {}
        self._custom_nodes: dict[str, Callable[..., Any]] = {}
        self._edges: list[tuple[str, str]] = []
        self._checkpointer: BaseCheckpointSaver[Any] | None = None

    def with_model(
        self,
        model: BaseChatModel | BaseLanguageModel[Any] | str,
        **model_kwargs: Any,
    ) -> AgentBuilder:
        """Set the chat model instance or ``"provider:model"`` spec string."""
        self._model = model
        if model_kwargs:
            self._model_kwargs.update(model_kwargs)
        return self

    def with_model_kwargs(self, **kwargs: Any) -> AgentBuilder:
        """Provide additional keyword arguments forwarded to :func:`resolve_model`."""
        self._model_kwargs.update(kwargs)
        return self

    def with_tools(self, tools: Sequence[BaseTool | Callable[..., Any]]) -> AgentBuilder:
        """Register one or more tools available to the agent."""
        self._tools.extend(tools)
        return self

    def with_middleware(
        self,
        middleware: Sequence[AgentMiddleware | Callable[..., Any]],
    ) -> AgentBuilder:
        """Attach a sequence of middleware instances or decorated hook functions."""
        self._middleware.extend(middleware)
        return self

    def add_middleware(self, middleware: AgentMiddleware | Callable[..., Any]) -> AgentBuilder:
        """Attach a single middleware instance or decorated hook function."""
        self._middleware.append(middleware)
        return self

    def with_system_prompt(self, prompt: str) -> AgentBuilder:
        """Configure a transient system prompt prepended on each model call."""
        self._system_prompt = prompt
        return self

    def with_state_schema(self, extra_fields: Mapping[str, Any]) -> AgentBuilder:
        """Merge extra typed fields into the agent's ``SmartState`` schema."""
        self._state_fields.update(extra_fields)
        return self

    def with_checkpointer(self, checkpointer: BaseCheckpointSaver[Any]) -> AgentBuilder:
        """Attach a LangGraph checkpointer for persistent threads."""
        self._checkpointer = checkpointer
        return self

    def add_node(self, name: str, func: Callable[..., Any]) -> AgentBuilder:
        """Register a custom LangGraph node function."""
        self._custom_nodes[name] = func
        return self

    def add_edge(self, start_node: str, end_node: str) -> AgentBuilder:
        """Connect two nodes in a custom-node graph."""
        self._edges.append((start_node, end_node))
        return self

    def compile(self) -> Agent:
        """Compile the configured nodes or model/tool loop into a ready-to-run :class:`Agent`."""
        if self._custom_nodes:
            state_schema = SmartState.create_schema(extra_fields=self._state_fields)
            builder: StateGraph[Any, Any, Any, Any] = StateGraph(state_schema)
            for node_name, node_func in self._custom_nodes.items():
                builder.add_node(node_name, node_func)

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
            compiled, state_schema = create_agent_graph(
                model=self._model,
                tools=self._tools,
                system_prompt=self._system_prompt,
                middleware=self._middleware,
                extra_state_fields=self._state_fields,
                checkpointer=self._checkpointer,
                **self._model_kwargs,
            )
            return Agent(name=self.name, compiled_graph=compiled, state_schema=state_schema)

        raise ConfigurationError(
            "AgentBuilder requires either a model or custom nodes to compile."
        )


def create_agent(
    model: str | BaseChatModel | BaseLanguageModel[Any] | None = None,
    tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    system_prompt: str | None = None,
    *,
    name: str = "LangCompAgent",
    middleware: Sequence[AgentMiddleware | Callable[..., Any]] | None = None,
    extra_state_fields: Mapping[str, Any] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    **model_kwargs: Any,
) -> Agent:
    """Build and compile a tool-calling :class:`Agent` on a hand-built ``StateGraph``.

    Args:
        model: A chat model instance, ``"provider:model"`` spec string, or ``None``
            to resolve :data:`langcomp.DEFAULT_MODEL`.
        tools: Tools the model may call.
        system_prompt: Optional transient system prompt prepended to every model call.
        name: Human-readable agent name.
        middleware: Optional middleware instances or decorated hook functions.
        extra_state_fields: Optional extra state fields added to ``SmartState``.
        checkpointer: Optional checkpointer for persistence across turns.
        **model_kwargs: Extra kwargs forwarded to :func:`resolve_model`.

    Returns:
        A compiled :class:`Agent` ready for ``.invoke(...)`` and ``.ainvoke(...)``.
    """
    compiled, state_schema = create_agent_graph(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        extra_state_fields=extra_state_fields,
        checkpointer=checkpointer,
        **model_kwargs,
    )
    return Agent(name=name, compiled_graph=compiled, state_schema=state_schema)


def agent(
    name: str = "DecoratedAgent",
    model: BaseChatModel | BaseLanguageModel[Any] | str | None = None,
    tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware | Callable[..., Any]] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> Callable[[Any], Agent]:
    """Decorator for quick agent creation from a node function or model config."""

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

        if callable(func_or_class) and not isinstance(func_or_class, type):
            builder.add_node(name, func_or_class)
            builder.add_edge(START, name)
            builder.add_edge(name, END)

        return builder.compile()

    return decorator
