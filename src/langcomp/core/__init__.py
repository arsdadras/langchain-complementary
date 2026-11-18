"""Core agent, state, loop, and node primitives for langcomp."""

from langcomp.core.agent import Agent, AgentBuilder, Node, agent, create_agent, node
from langcomp.core.loop import create_agent_graph, normalize_tools
from langcomp.core.nodes import (
    AFTER_AGENT_NODE,
    BEFORE_AGENT_NODE,
    MODEL_NODE,
    TOOLS_NODE,
    has_hook,
)
from langcomp.core.state import (
    Reduce,
    SmartState,
    State,
    merge_dicts,
    merge_state_schemas,
    overwrite,
)

__all__ = [
    "AFTER_AGENT_NODE",
    "BEFORE_AGENT_NODE",
    "MODEL_NODE",
    "TOOLS_NODE",
    "Agent",
    "AgentBuilder",
    "Node",
    "Reduce",
    "SmartState",
    "State",
    "agent",
    "create_agent",
    "create_agent_graph",
    "has_hook",
    "merge_dicts",
    "merge_state_schemas",
    "node",
    "normalize_tools",
    "overwrite",
]
