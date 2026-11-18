"""langcomp: LangChain-Complementary Framework.

High-level, developer-friendly abstractions on top of LangChain, LangGraph,
and DeepAgents.
"""

from __future__ import annotations

__version__ = "0.1.0"

from langcomp.core.agent import Agent, AgentBuilder, agent, create_agent
from langcomp.core.state import SmartState
from langcomp.devtools.testing import ScriptedFakeChatModel
from langcomp.devtools.visualizer import DryRunner, GraphVisualizer
from langcomp.errors import (
    ConfigurationError,
    InvalidModelSpecError,
    LangCompError,
    MissingApiKeyError,
    ProviderNotInstalledError,
    ScriptExhaustedError,
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
from langcomp.swarm.pool import SubAgent, SwarmPool
from langcomp.tools.guard import ApprovalGate, ToolGuard

__all__ = [
    "CAPABLE_MODEL",
    "DEFAULT_MODEL",
    "Agent",
    "AgentBuilder",
    "AgentMiddleware",
    "ApprovalGate",
    "ApprovalGateMiddleware",
    "AutoSummarizer",
    "AutoSummarizerMiddleware",
    "ConfigurationError",
    "ContextBuffer",
    "ContextBufferMiddleware",
    "DryRunner",
    "GraphVisualizer",
    "InvalidModelSpecError",
    "JumpTo",
    "LangCompError",
    "MissingApiKeyError",
    "ModelHandler",
    "ModelRequest",
    "ModelResponse",
    "ProviderNotInstalledError",
    "ProviderSpec",
    "ScriptExhaustedError",
    "ScriptedFakeChatModel",
    "SmartState",
    "SubAgent",
    "SwarmPool",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolGuard",
    "ToolGuardMiddleware",
    "ToolHandler",
    "UnknownProviderError",
    "UnknownToolError",
    "__version__",
    "after_agent",
    "after_model",
    "agent",
    "before_agent",
    "before_model",
    "create_agent",
    "dynamic_prompt",
    "known_providers",
    "parse_model_spec",
    "register_provider",
    "resolve_model",
    "wrap_model_call",
    "wrap_tool_call",
]
