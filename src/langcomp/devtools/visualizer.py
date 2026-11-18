"""
Local DevTools, Visualization, and Dry-Runner for the langcomp framework.
Allows zero-cloud graph inspection, Mermaid export, ASCII visualization, and step dry-running.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langcomp.core.agent import Agent


class GraphVisualizer:
    """Renders visual diagrams (Mermaid, ASCII) for any LangGraph or langcomp Agent."""

    @staticmethod
    def _resolve_graph(agent_or_graph: Any) -> Any:
        graph_obj = agent_or_graph.graph if isinstance(agent_or_graph, Agent) else agent_or_graph
        if not hasattr(graph_obj, "get_graph"):
            raise TypeError(
                f"Expected a langcomp Agent or compiled LangGraph with get_graph(), "
                f"got {type(agent_or_graph).__name__}."
            )
        return graph_obj.get_graph()

    @classmethod
    def to_mermaid(cls, agent_or_graph: Any) -> str:
        """Generate Mermaid diagram markup for the compiled graph."""
        drawable = cls._resolve_graph(agent_or_graph)
        return drawable.draw_mermaid()

    @classmethod
    def to_ascii(cls, agent_or_graph: Any) -> str:
        """Generate an ASCII diagram of the graph structure.

        Uses ``grandalf`` when installed, and falls back cleanly to a zero-dependency
        edge-list rendering of the actual compiled graph topology otherwise.
        """
        drawable = cls._resolve_graph(agent_or_graph)
        try:
            return drawable.draw_ascii()
        except ImportError:
            edges = getattr(drawable, "edges", [])
            if not edges:
                nodes = list(getattr(drawable, "nodes", {}).keys())
                return " -> ".join(f"[{n}]" for n in nodes) if nodes else "[EMPTY GRAPH]"
            lines: List[str] = []
            for edge in edges:
                src = getattr(edge, "source", "?")
                tgt = getattr(edge, "target", "?")
                conditional = getattr(edge, "conditional", False)
                arrow = "-.->" if conditional else "--->"
                lines.append(f"[{src}] {arrow} [{tgt}]")
            return "\n".join(lines)


class DryRunner:
    """Dry-runner engine for testing node state mutations step-by-step without remote API calls."""

    def __init__(self, agent_inst: Agent):
        self.agent = agent_inst

    def run_step(self, node_name: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node logic function directly against a state dictionary."""
        graph_obj = self.agent.graph
        nodes = getattr(graph_obj, "nodes", {})
        if node_name in nodes:
            node_runnable = nodes[node_name]
            if hasattr(node_runnable, "invoke"):
                return node_runnable.invoke(current_state)
            if callable(node_runnable):
                return node_runnable(current_state)

        raise ValueError(f"Node '{node_name}' not found in agent graph.")
