"""
langcomp: LangChain-Complementary Framework
High-level, developer-friendly abstractions on top of LangChain, LangGraph, and DeepAgents.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("langcomp")
except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
    __version__ = "0.2.0"

from langcomp.core.agent import Agent, AgentBuilder, Node, agent, create_agent, node
from langcomp.core.state import Reduce, SmartState, State, merge_dicts, overwrite
from langcomp.devtools.trace import DEFAULT_PRICE_TABLE, TraceCollector, summarize
from langcomp.devtools.visualizer import DryRunner, GraphVisualizer
from langcomp.errors import (
    ApprovalRejectedError,
    ConfigurationError,
    InvalidModelSpecError,
    LangcompError,
    MissingApiKeyError,
    ProviderNotInstalledError,
    ReducerConflictError,
    ScriptExhaustedError,
    SnapshotMismatchError,
    ToolExecutionError,
    UnknownProviderError,
    UnknownToolError,
)
from langcomp.memory.buffer import SUMMARY_PREFIX, AutoSummarizer, ContextBuffer
from langcomp.middleware import (
    AgentMiddleware,
    ApprovalGateMiddleware,
    AsyncModelHandler,
    AsyncToolHandler,
    AutoSummarizerMiddleware,
    ContextBufferMiddleware,
    JumpTo,
    ModelHandler,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolGuardMiddleware,
    ToolHandler,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langcomp.models import (
    CAPABLE_MODEL,
    DEFAULT_MODEL,
    ProviderSpec,
    get_provider,
    known_providers,
    parse_model_spec,
    register_provider,
    resolve_model,
    unregister_provider,
)
from langcomp.persistence import async_checkpointer, checkpointer, managed_checkpointer
from langcomp.swarm.pool import SubAgent, SwarmPool
from langcomp.swarm.team import Team
from langcomp.testing import FakeModel, ScriptTurn, arun_and_snapshot, run_and_snapshot
from langcomp.tools.guard import ApprovalGate, ToolGuard

__all__ = [
    "CAPABLE_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_PRICE_TABLE",
    "SUMMARY_PREFIX",
    "Agent",
    "AgentBuilder",
    "AgentMiddleware",
    "ApprovalGate",
    "ApprovalGateMiddleware",
    "ApprovalRejectedError",
    "AsyncModelHandler",
    "AsyncToolHandler",
    "AutoSummarizer",
    "AutoSummarizerMiddleware",
    "ConfigurationError",
    "ContextBuffer",
    "ContextBufferMiddleware",
    "DryRunner",
    "FakeModel",
    "GraphVisualizer",
    "InvalidModelSpecError",
    "JumpTo",
    "LangcompError",
    "MissingApiKeyError",
    "ModelHandler",
    "ModelRequest",
    "ModelResponse",
    "Node",
    "ProviderNotInstalledError",
    "ProviderSpec",
    "Reduce",
    "ReducerConflictError",
    "ScriptExhaustedError",
    "ScriptTurn",
    "SmartState",
    "SnapshotMismatchError",
    "State",
    "SubAgent",
    "SwarmPool",
    "Team",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolExecutionError",
    "ToolGuard",
    "ToolGuardMiddleware",
    "ToolHandler",
    "TraceCollector",
    "UnknownProviderError",
    "UnknownToolError",
    "__version__",
    "after_agent",
    "after_model",
    "agent",
    "arun_and_snapshot",
    "async_checkpointer",
    "before_agent",
    "before_model",
    "checkpointer",
    "create_agent",
    "dynamic_prompt",
    "get_provider",
    "known_providers",
    "managed_checkpointer",
    "merge_dicts",
    "node",
    "overwrite",
    "parse_model_spec",
    "register_provider",
    "resolve_model",
    "run_and_snapshot",
    "summarize",
    "unregister_provider",
    "wrap_model_call",
    "wrap_tool_call",
]
