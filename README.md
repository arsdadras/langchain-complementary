# `langcomp` (LangChain-Complementary Framework)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph Compatible](https://img.shields.io/badge/LangGraph-100%25%20Compatible-green.svg)](https://docs.langchain.com/oss/python/langgraph/overview)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`langcomp`** is a lightweight, developer-friendly complementary framework built on top of [LangChain](https://docs.langchain.com), [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), and [DeepAgents](https://docs.langchain.com/oss/python/deepagents/overview).

It eliminates common developer friction points across the LangChain/LangGraph ecosystem without replacing the underlying runtimes: declarative graph wiring, automatic state reducers, composable onion middleware, lazy provider resolution, deterministic offline testing with `FakeModel` and golden-master snapshots, zero-SaaS local token/cost tracing, resilient sync/async tool guardrails, durability-aware checkpointers, and hierarchical subagent teams.

---

## Key Features

1. **Declarative Agent Blueprint (`AgentBuilder`, `create_agent`, `@agent`, `@node`)**
   - Fluent graph construction with automatic linear edge chaining, duplicate node checks, and edge validation.
   - Hand-built `StateGraph` tool-calling loop with transient system prompts, parallel tool execution, and recoverable error `ToolMessage`s.
   - Sync/async invocation and streaming (`invoke`, `ainvoke`, `stream`, `astream`, `get_state`).
   - `@node(idempotent=True)` for at-most-once-per-thread execution across LangGraph `interrupt()` resumes, plus `Node.invoke_for_test()`.
   - Seamless interop via `Agent.from_runnable(...)` for any LangChain `create_agent()` or DeepAgents runnable.

2. **Smart State & Automatic Reducers (`State`, `Reduce`, `SmartState`)**
   - `State(BaseModel)` automatically infers reducers (`list` -> `operator.add`, `dict` -> `merge_dicts`, scalars -> `overwrite`), supports custom `Reduce(fn)` overrides, and exposes `State.explain()` + `State.to_typed_dict()`.
   - `SmartState` dynamically generates `TypedDict` schemas with `add_messages`, merges middleware `state_schema` definitions, and normalizes raw strings/dicts/tuples into LangChain messages.

3. **Dual-Mechanism Onion Middleware (`AgentMiddleware`, `@before_model`, `@wrap_model_call`, `@dynamic_prompt`)**
   - Composable lifecycle hooks (`before_agent`, `before_model`, `after_model`, `after_agent`) with typed `JumpTo` routing via LangGraph `Command`.
   - Immutable onion wrappers (`ModelRequest`, `ToolCallRequest`) for retries, caching, dynamic prompts, and policy enforcement.
   - Built-in adapters plugging langcomp's memory and guardrails directly into the loop: `ContextBufferMiddleware`, `AutoSummarizerMiddleware`, `ToolGuardMiddleware`, and `ApprovalGateMiddleware`.

4. **Lazy Provider Registry (`resolve_model`, `register_provider`)**
   - Resolve `"provider:model"` specs (`openai:...`, `google_genai:...`, `openrouter:...`, or custom providers) lazily without forcing unused provider SDK dependencies.
   - Actionable typed exceptions under `LangcompError` (`InvalidModelSpecError`, `UnknownProviderError`, `ProviderNotInstalledError`, `MissingApiKeyError`, `UnknownToolError`).

5. **Deterministic Offline Testing (`FakeModel`, `run_and_snapshot`, `DryRunner`)**
   - `FakeModel(BaseChatModel)`: scripted multi-turn chat double supporting `tool_calls`, `usage_metadata`, `.calls` prompt inspection, and `.bound_tool_names` with zero network calls.
   - `run_and_snapshot()`: records clean JSON snapshots on first run and raises `SnapshotMismatchError` on output drift.
   - `DryRunner`: executes individual compiled graph nodes directly against state dictionaries.

6. **Zero-Cloud Local Observability (`TraceCollector`, `summarize`, `GraphVisualizer`)**
   - `TraceCollector(BaseCallbackHandler)` logs per-node latency, input/output tokens, estimated USD cost, and `langgraph_checkpoint_ns` subagent depth to local `.jsonl` files.
   - `GraphVisualizer.to_mermaid()` and zero-dependency `GraphVisualizer.to_ascii()` diagram exporters.

7. **Swarm & Hierarchical Subagent Teams (`SwarmPool`, `SubAgent`, `Team`)**
   - `SwarmPool` wraps agents or callables as context-isolated LangChain tools or DeepAgents subagent dicts.
   - `Team` composes arbitrarily nested DeepAgents subagent trees with uniform transitive field inheritance (`system_prompt`, `tools`, `model`, `skills`, `middleware`).

8. **Memory, Guardrails & Durability (`ContextBuffer`, `AutoSummarizer`, `ToolGuard`, `ApprovalGate`, `checkpointer`)**
   - Sliding-window message trimming and automatic conversation summarization.
   - Sync/async `ToolGuard` retries + fallbacks and `ApprovalGate` enforcement (Pydantic v2 `BaseTool` safe).
   - `checkpointer("memory" | "sqlite" | "postgres")` preset factory with explicit durability warning for in-memory state.

---

## Installation

```bash
# Core framework (langchain-core, langgraph, pydantic)
pip install -e .

# Optional extras
pip install -e ".[openai]"       # OpenAI & OpenRouter chat models
pip install -e ".[google]"       # Google Gemini chat models
pip install -e ".[sqlite]"       # SQLite checkpointer
pip install -e ".[postgres]"     # PostgreSQL checkpointer
pip install -e ".[deepagents]"   # Hierarchical Team compilation
pip install -e ".[visualizer]"   # Enhanced ASCII graphs via grandalf
pip install -e ".[dev]"          # Testing & linting tools
```

---

## Quickstart

### 1. Declarative Agent with Typed Reducers & `@node`

```python
from langcomp import AgentBuilder, State, Reduce, node

class ResearchState(State):
    notes: list[str] = []
    metadata: dict[str, str] = {}
    score: int = Reduce(lambda a, b: max(a or 0, b or 0), default=0)

@node
def gather(state):
    return {"notes": ["collected data"], "metadata": {"stage": "gather"}, "score": 8}

@node
def refine(state):
    return {"notes": ["verified facts"], "metadata": {"reviewed": "true"}, "score": 5}

agent = (
    AgentBuilder("ResearchAgent")
    .with_state_schema(ResearchState)
    .add_node(gather)
    .add_node(refine)
    .compile()
)

result = agent.invoke({"notes": [], "metadata": {}, "score": 0})
print(result["notes"])     # ['collected data', 'verified facts']
print(result["metadata"])  # {'stage': 'gather', 'reviewed': 'true'}
print(result["score"])     # 8
```

### 2. Offline Agent Testing, Middleware & Local Tracing

```python
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langcomp import (
    AgentBuilder,
    ContextBufferMiddleware,
    FakeModel,
    ToolGuardMiddleware,
    TraceCollector,
    summarize,
)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

model = FakeModel(script=[
    AIMessage(
        content="",
        tool_calls=[{"name": "multiply", "args": {"a": 6, "b": 7}, "id": "call_1"}],
        usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
    ),
    AIMessage(
        content="6 * 7 = 42.",
        usage_metadata={"input_tokens": 80, "output_tokens": 15, "total_tokens": 95},
    ),
])

agent = (
    AgentBuilder("MathAgent")
    .with_model(model)
    .with_tools([multiply])
    .with_middleware([
        ContextBufferMiddleware(max_messages=10),
        ToolGuardMiddleware(max_retries=2, retry_delay_seconds=0.0),
    ])
    .compile()
)
collector = TraceCollector(out_dir=None)  # in-memory tracing

result = agent.invoke("What is 6 * 7?", config={"callbacks": [collector]})
print(result["messages"][-1].content)
print(summarize(collector))
```

---

## Examples & Docs

- Runnable offline examples live in [`examples/`](./examples/)
- Full step-by-step tutorial chapters live in [`docs/tutorial/`](./docs/tutorial/INDEX.md)

---

## Development

```bash
uv run pytest -v
uv run ruff check src tests examples
```

---

## License

MIT License. See [LICENSE](./LICENSE). Developed by AR_S.DADRAS.
