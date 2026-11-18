"""
langcomp: LangChain-Complementary Framework
High-level, developer-friendly abstractions on top of LangChain, LangGraph, and DeepAgents.
"""

__version__ = "0.2.0"

from langcomp.core.agent import Agent, AgentBuilder, Node, agent, node
from langcomp.core.state import Reduce, SmartState, State, merge_dicts, overwrite
from langcomp.devtools.trace import DEFAULT_PRICE_TABLE, TraceCollector, summarize
from langcomp.devtools.visualizer import DryRunner, GraphVisualizer
from langcomp.errors import (
    ApprovalRejectedError,
    LangcompError,
    ReducerConflictError,
    ScriptExhaustedError,
    SnapshotMismatchError,
)
from langcomp.memory.buffer import AutoSummarizer, ContextBuffer
from langcomp.persistence import checkpointer
from langcomp.swarm.pool import SubAgent, SwarmPool
from langcomp.swarm.team import Team
from langcomp.testing import FakeModel, run_and_snapshot
from langcomp.tools.guard import ApprovalGate, ToolGuard

__all__ = [
    "__version__",
    # Core Agent & Graph
    "Agent",
    "AgentBuilder",
    "agent",
    "node",
    "Node",
    # State & Reducers
    "State",
    "SmartState",
    "Reduce",
    "overwrite",
    "merge_dicts",
    # Swarm & Hierarchical Teams
    "SwarmPool",
    "SubAgent",
    "Team",
    # Memory & Context
    "ContextBuffer",
    "AutoSummarizer",
    # Tools & Guardrails
    "ToolGuard",
    "ApprovalGate",
    # Persistence
    "checkpointer",
    # DevTools, Tracing & Testing
    "GraphVisualizer",
    "DryRunner",
    "TraceCollector",
    "DEFAULT_PRICE_TABLE",
    "summarize",
    "FakeModel",
    "run_and_snapshot",
    # Errors
    "LangcompError",
    "ReducerConflictError",
    "SnapshotMismatchError",
    "ScriptExhaustedError",
    "ApprovalRejectedError",
]
