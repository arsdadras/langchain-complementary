"""
Local DevTools, Visualization, and Dry-Runner for the langcomp framework.
Allows zero-cloud graph inspection, Mermaid export, ASCII visualization, and sync/async step execution.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from langcomp.core.agent import Agent


class GraphVisualizer:
    """Renders visual diagrams (Mermaid, ASCII) for any LangGraph or langcomp Agent."""

    @staticmethod
    def _resolve_graph(agent_or_graph: Any, *, xray: bool | int = False) -> Any:
        graph_obj = agent_or_graph.graph if isinstance(agent_or_graph, Agent) else agent_or_graph
        if not hasattr(graph_obj, "get_graph"):
            raise TypeError(
                f"Expected a langcomp Agent or compiled LangGraph with get_graph(), "
                f"got {type(agent_or_graph).__name__}."
            )
        return graph_obj.get_graph(xray=xray)

    @classmethod
    def to_mermaid(cls, agent_or_graph: Any, *, xray: bool | int = False) -> str:
        """Generate Mermaid diagram markup for the compiled graph."""
        drawable = cls._resolve_graph(agent_or_graph, xray=xray)
        return str(drawable.draw_mermaid())

    @classmethod
    def to_ascii(cls, agent_or_graph: Any, *, xray: bool | int = False) -> str:
        """Generate an ASCII diagram of the graph structure.

        Uses ``grandalf`` when installed, and falls back cleanly to a zero-dependency
        edge-list rendering of the actual compiled graph topology otherwise.
        """
        drawable = cls._resolve_graph(agent_or_graph, xray=xray)
        try:
            return str(drawable.draw_ascii())
        except ImportError:
            edges = getattr(drawable, "edges", [])
            if not edges:
                nodes = list(getattr(drawable, "nodes", {}).keys())
                return " -> ".join(f"[{n}]" for n in nodes) if nodes else "[EMPTY GRAPH]"
            lines: list[str] = []
            for edge in edges:
                src = getattr(edge, "source", "?")
                tgt = getattr(edge, "target", "?")
                conditional = getattr(edge, "conditional", False)
                label = getattr(edge, "data", None)
                arrow = "-.->" if conditional else "--->"
                if label:
                    lines.append(f"[{src}] {arrow} ({label}) [{tgt}]")
                else:
                    lines.append(f"[{src}] {arrow} [{tgt}]")
            return "\n".join(lines)


class DryRunner:
    """Development/testing utility for executing individual compiled graph nodes step-by-step.

    Note: Invokes the target node's runnable directly (if the node calls a live model or
    external tool, those side effects will still execute unless mocked with :class:`FakeModel`).
    """

    def __init__(self, agent_or_graph: Agent | Any):
        self.agent = agent_or_graph
        self._graph = agent_or_graph.graph if isinstance(agent_or_graph, Agent) else agent_or_graph

    def _get_node_runnable(self, node_name: str) -> Any:
        nodes = getattr(self._graph, "nodes", {})
        if node_name not in nodes:
            raise ValueError(f"Node '{node_name}' not found in agent graph.")
        return nodes[node_name]

    def run_step(
        self,
        node_name: str,
        current_state: dict[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        """Synchronously execute a single compiled graph node against a state dictionary."""
        node_runnable = self._get_node_runnable(node_name)
        if hasattr(node_runnable, "invoke") and callable(node_runnable.invoke):
            if config is not None:
                return node_runnable.invoke(current_state, config=config)
            return node_runnable.invoke(current_state)
        if callable(node_runnable):
            return node_runnable(current_state)
        raise TypeError(
            f"Node '{node_name}' runnable of type {type(node_runnable).__name__} is not invokable."
        )

    async def arun_step(
        self,
        node_name: str,
        current_state: dict[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        """Asynchronously execute a single compiled graph node against a state dictionary."""
        node_runnable = self._get_node_runnable(node_name)
        if hasattr(node_runnable, "ainvoke") and callable(node_runnable.ainvoke):
            if config is not None:
                return await node_runnable.ainvoke(current_state, config=config)
            return await node_runnable.ainvoke(current_state)
        if callable(node_runnable):
            res = node_runnable(current_state)
            return await res if inspect.isawaitable(res) else res
        raise TypeError(
            f"Node '{node_name}' runnable of type {type(node_runnable).__name__} is not invokable."
        )


__all__ = ["DryRunner", "GraphVisualizer"]
