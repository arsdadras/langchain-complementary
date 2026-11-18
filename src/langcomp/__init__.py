"""
langcomp: LangChain-Complementary Framework
High-level, developer-friendly abstractions on top of LangChain, LangGraph, and DeepAgents.
"""

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
    UnknownProviderError,
    UnknownToolError,
)
from langcomp.memory.buffer import AutoSummarizer, ContextBuffer
from langcomp.middleware import (
    AgentMiddleware,
    ApprovalGateMiddleware,
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
    known_providers,
    parse_model_spec,
    register_provider,
    resolve_model,
)
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
    "create_agent",
    "agent",
    "node",
    "Node",
    # State & Reducers
    "State",
    "SmartState",
    "Reduce",
    "overwrite",
    "merge_dicts",
    # Models & Lazy Provider Registry
    "DEFAULT_MODEL",
    "CAPABLE_MODEL",
    "ProviderSpec",
    "known_providers",
    "parse_model_spec",
    "register_provider",
    "resolve_model",
    # Onion Middleware & Built-in Adapters
    "AgentMiddleware",
    "JumpTo",
    "ModelHandler",
    "ModelRequest",
    "ModelResponse",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolHandler",
    "before_agent",
    "before_model",
    "wrap_model_call",
    "after_model",
    "wrap_tool_call",
    "after_agent",
    "dynamic_prompt",
    "ContextBufferMiddleware",
    "AutoSummarizerMiddleware",
    "ToolGuardMiddleware",
    "ApprovalGateMiddleware",
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
    "ConfigurationError",
    "InvalidModelSpecError",
    "UnknownProviderError",
    "ProviderNotInstalledError",
    "MissingApiKeyError",
    "ReducerConflictError",
    "SnapshotMismatchError",
    "ScriptExhaustedError",
    "ApprovalRejectedError",
    "UnknownToolError",
]
