# Chapter 01: Getting Started & Core Agents

In this chapter, you will install `langcomp`, define typed state schemas with `State`/`Reduce` and `SmartState`, build custom graphs with `@node` and `AgentBuilder`, create model-backed tool-calling agents with `create_agent`, configure lazy model providers and onion middleware, attach durable checkpointers, and run agents synchronously or asynchronously.

---

## Installation & Optional Extras

Core `langcomp` only installs `langchain-core`, `langgraph`, and `pydantic`:

```bash
pip install -e .
```

Install optional extras for the specific integrations you use:

| Extra | Command | What it provides |
| :--- | :--- | :--- |
| `openai` | `pip install -e ".[openai]"` | `langchain-openai` (`openai:*` and `openrouter:*` providers) |
| `google` | `pip install -e ".[google]"` | `langchain-google-genai` (`google_genai:*` provider) |
| `sqlite` | `pip install -e ".[sqlite]"` | `langgraph-checkpoint-sqlite` (sync & async SQLite persistence) |
| `postgres` | `pip install -e ".[postgres]"` | `langgraph-checkpoint-postgres` + `psycopg[binary]` |
| `deepagents` | `pip install -e ".[deepagents]"` | `deepagents` for `Team.compile()` (Python >= 3.11) |
| `visualizer` | `pip install -e ".[visualizer]"` | `grandalf` for box-drawing ASCII diagrams in `GraphVisualizer` |
| `all` | `pip install -e ".[all]"` | All optional runtime extras above |
| `dev` | `pip install -e ".[dev]"` | `pytest`, `pytest-asyncio`, `pytest-cov`, `mypy`, `ruff` |

---

## 1. State Management: `State`/`Reduce` vs `SmartState`

### Typed State with `State` and `Reduce`

Subclass `langcomp.State` (a Pydantic `BaseModel`) to declare graph state with **automatic type-inferred reducers**:

- `list[...]` (or `list[...] | None`) -> `operator.add` (append)
- `dict[...]` (or `dict[...] | None`) -> `merge_dicts` (shallow merge, newer keys win)
- Scalars / other types -> `overwrite` (last write wins)
- Explicit `Annotated[T, reducer_fn]` -> preserved as-is (e.g., `add_messages`)

Override any field's inferred reducer with `Reduce(fn, default=...)` or `Reduce(fn, default_factory=...)`, and inspect the resolved reducers with `.explain()`:

```python
from typing import Annotated, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langcomp import Reduce, State, overwrite

class ResearchState(State):
    messages: Annotated[list[AnyMessage], add_messages] = []
    notes: list[str] = []                                       # inferred: add (append)
    metadata: dict[str, Any] = {}                               # inferred: merge_dicts
    summary: str = ""                                           # inferred: overwrite
    active_tags: list[str] = Reduce(overwrite, default_factory=list)  # explicit override

print(ResearchState.explain())
# {'messages': '_add_messages', 'notes': 'add', 'metadata': 'merge_dicts',
#  'summary': 'overwrite', 'active_tags': 'overwrite'}
```

> [!NOTE]
> Declared defaults and `default_factory` values on a `State` subclass are automatically hydrated on a fresh thread's first invocation (without clobbering existing checkpointed state on later turns).

### Dynamic Schemas & Payload Normalization with `SmartState`

When you do not need a full `State` class, `SmartState` generates a LangGraph `TypedDict` with `messages` (using `add_messages`) and normalizes strings, `(role, content)` tuples, dicts, and `BaseMessage` objects:

```python
from langcomp import SmartState

schema = SmartState.create_schema(extra_fields={"topic": str})

messages = SmartState.normalize_messages([
    "User query string",
    ("assistant", "AI reply"),
    {"role": "user", "content": "Follow-up"},
])
```

`Agent.invoke()` / `ainvoke()` automatically normalize plain strings, `BaseMessage`s, message lists, and dicts containing `"messages"`.

---

## 2. Custom Graphs with `@node` and `AgentBuilder`

Use `@node` to wrap plain sync or async functions into `Node` objects. Pass `idempotent=True` on nodes with external side effects so they record completion in `_langcomp_done` and no-op if the same checkpointed thread re-enters them (for example, after resuming a downstream `interrupt()`).

> [!TIP]
> Always place side-effecting `@node(idempotent=True)` steps in their own node **after** any node that calls `interrupt()`.

```python
from typing import Annotated
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from langcomp import AgentBuilder, State, node

class PipelineState(State):
    messages: Annotated[list[AnyMessage], add_messages] = []
    keywords: list[str] = []

@node
def extract_keywords(state):
    query = state["messages"][-1].content
    kws = [w for w in query.split() if len(w) > 4]
    return {"keywords": kws}

@node(idempotent=True)
def emit_report(state):
    kws = state["keywords"]
    return {"messages": [AIMessage(content=f"Keywords: {', '.join(kws)}")]}

pipeline = (
    AgentBuilder("KeywordPipeline")
    .with_state_schema(PipelineState)
    .add_node(extract_keywords)
    .add_node(emit_report)
    .compile()
)

result = pipeline.invoke("Analyze machine learning deployment architecture")
print(result["messages"][-1].content)
# Output: Keywords: Analyze, machine, learning, deployment, architecture
```

> [!TIP]
> When you add custom nodes without calling `.add_edge(...)`, `AgentBuilder` wires them sequentially: `START -> node_1 -> node_2 -> ... -> END`. Note that `AgentBuilder` operates in **either** custom-node mode (`add_node`/`add_edge`) **or** model-backed mode (`with_model`/`with_tools`/`with_middleware`)—combining both on the same builder raises `ValueError`.

---

## 3. Quick Single-Node & Model-Backed `@agent` Decorator

For single-node graphs or concise model-backed agents, use `@agent`:

```python
from langchain_core.messages import AIMessage
from langcomp import agent

@agent(name="QuickAnalyzer")
def quick_analyzer_node(state):
    text = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Processed: {text.upper()}")]}

# The decorated function is now a compiled Agent instance!
result = quick_analyzer_node.invoke("hello world")
print(result["messages"][-1].content)  # Output: Processed: HELLO WORLD
```

When `model=...` is passed to `@agent`, it compiles a model-backed tool-calling `Agent` and uses the decorated function or class docstring as a fallback `system_prompt`.

---

## 4. Model-Backed Agents: `create_agent` & `AgentBuilder`

`langcomp.create_agent()` builds a tool-calling `Agent` on a hand-built sync + async `StateGraph` with middleware support and lazy model resolution. `AgentBuilder` exposes the exact same capabilities via a fluent API:

```python
from langchain_core.tools import tool
from langcomp import AgentBuilder, FakeModel, create_agent

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# 1. Concise factory (pass a "provider:model" string or any BaseChatModel / FakeModel)
calc_agent = create_agent(
    model=FakeModel(script=["The product is 42."]),
    tools=[multiply],
    system_prompt="You are a concise math assistant.",
    name="CalcAgent",
)

# 2. Equivalent fluent AgentBuilder API
builder_agent = (
    AgentBuilder("CalcAgent")
    .with_model("openai:gpt-4o-mini", temperature=0.0)   # resolved lazily at compile()
    .with_tools([multiply])
    .with_system_prompt("You are a concise math assistant.")
    .compile()
)
```

Plain Python functions passed in `tools=[...]` are automatically coerced into LangChain `BaseTool` instances.

---

## 5. Lazy Provider Registry (`resolve_model` & `register_provider`)

Instead of importing provider SDKs at module import time, `langcomp` resolves `"provider:model"` strings lazily via `resolve_model()`.

Built-in providers registered out of the box (`known_providers()`):

- `"openai"` -> `langchain_openai.ChatOpenAI` (extra: `openai`)
- `"google_genai"` -> `langchain_google_genai.ChatGoogleGenerativeAI` (extra: `google`)
- `"openrouter"` -> `langchain_openai.ChatOpenAI` pointed at `https://openrouter.ai/api/v1` using `OPENROUTER_API_KEY` with a default `max_tokens=4096` cap (extra: `openai`)

Tiered model constants `DEFAULT_MODEL` (`"openrouter:nvidia/nemotron-3.5-lightning"`) and `CAPABLE_MODEL` (`"openrouter:openai/gpt-5.6-terra"`) are exported from `langcomp`. Calling `resolve_model()` with `None` resolves `DEFAULT_MODEL`.

```python
from langcomp import ProviderSpec, known_providers, register_provider, resolve_model

print(known_providers())  # ('google_genai', 'openai', 'openrouter')

# Register a custom provider (e.g., Anthropic) lazily
register_provider(
    "anthropic",
    ProviderSpec(
        module="langchain_anthropic",
        class_name="ChatAnthropic",
        extra="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
    ),
)

# Instantiated only when called (raises ProviderNotInstalledError or MissingApiKeyError if misconfigured)
# llm = resolve_model("openai:gpt-4o-mini", temperature=0.2)
```

---

## 6. Onion Middleware Overview

Both `create_agent(..., middleware=[...])` and `AgentBuilder.with_middleware([...])` / `.add_middleware(...)` accept `AgentMiddleware` subclasses or lightweight single-hook decorators (`@before_agent`, `@before_model`, `@wrap_model_call`, `@after_model`, `@wrap_tool_call`, `@after_agent`, `@dynamic_prompt`).

- **Lifecycle hooks** (`before_agent`, `before_model`) run in **forward** registration order and may return a partial state update dict or `JumpTo("end" | "model" | "tools", messages=..., state_update=...)`.
- **Wrap hooks** (`wrap_model_call`, `wrap_tool_call`) compose in **onion order**: the first registered middleware is the outermost wrapper (`M1(M2(core))`).
- **Post hooks** (`after_model`, `after_agent`) run in **reverse** registration order.

```python
from langcomp import FakeModel, create_agent, dynamic_prompt

@dynamic_prompt
def role_prompt(request):
    role = request.state.get("user_role", "developer")
    return f"You are assisting a {role}. Keep answers brief."

agent = create_agent(
    model=FakeModel(script=["Ready to help!"]),
    extra_state_fields={"user_role": str},
    middleware=[role_prompt],
)

out = agent.invoke({"messages": ["Hi"], "user_role": "SRE"})
print(out["messages"][-1].content)  # Output: Ready to help!
```

Built-in middleware adapters (`ContextBufferMiddleware`, `AutoSummarizerMiddleware`, `ApprovalGateMiddleware`, `ToolGuardMiddleware`) are covered in Chapters 03 and 04.

---

## 7. Persistence & the Explicit `thread_id` Requirement

`langcomp` provides three checkpointer helpers supporting `"memory"`, `"sqlite"`, and `"postgres"`:

- `checkpointer(kind, *, path=..., conn_string=...)` — quick factory (attaches a `.close()` helper and emits a `UserWarning` for `"memory"` durability)
- `managed_checkpointer(kind, *, path=..., conn_string=...)` — sync context manager that deterministically closes DB connections/pools on exit (recommended for SQLite/Postgres)
- `async_checkpointer(kind, *, path=..., conn_string=...)` — async context manager yielding `AsyncSqliteSaver` / `AsyncPostgresSaver` / `MemorySaver`

> [!IMPORTANT]
> Whenever an `Agent` has a checkpointer enabled, you **must** pass an explicit `config={"configurable": {"thread_id": "..."}}` to `invoke()`, `ainvoke()`, `stream()`, `astream()`, `get_state()`, and `update_state()`. Omitting `thread_id` raises `ConfigurationError` to prevent accidental cross-session state leakage.

```python
from langcomp import FakeModel, checkpointer, create_agent

saver = checkpointer("memory")  # warns that in-memory state is lost on restart
chat_agent = create_agent(
    model=FakeModel(script=["Hello Alice!", "Your name is Alice."]),
    checkpointer=saver,
)

cfg = {"configurable": {"thread_id": "session-alice-1"}}
chat_agent.invoke("Hi, I'm Alice.", config=cfg)
reply = chat_agent.invoke("What is my name?", config=cfg)
print(reply["messages"][-1].content)  # Output: Your name is Alice.

# Inspect persisted checkpoint state
snapshot = chat_agent.get_state(cfg)
print(len(snapshot.values["messages"]))  # 4 messages persisted on this thread
```

For durable disk-backed SQLite persistence with automatic cleanup:

```python
from langcomp import FakeModel, create_agent, managed_checkpointer

# Requires: pip install 'langcomp[sqlite]'
with managed_checkpointer("sqlite", path=".langcomp/state.db") as saver:
    durable_agent = create_agent(
        model=FakeModel(script=["Saved to SQLite."]),
        checkpointer=saver,
    )
    durable_agent.invoke("Persist this", config={"configurable": {"thread_id": "job-1"}})
```

---

## 8. Async Execution (`ainvoke` & `astream`)

Every `Agent` supports first-class async invocation (`ainvoke`, `astream`, `aget_state`, `aupdate_state`), matching LangGraph's async execution model:

```python
import asyncio
from langcomp import FakeModel, async_checkpointer, create_agent

async def main() -> None:
    async with async_checkpointer("memory") as saver:
        agent = create_agent(
            model=FakeModel(script=["Async reply 1", "Async reply 2"]),
            checkpointer=saver,
        )
        cfg = {"configurable": {"thread_id": "async-thread-1"}}

        res = await agent.ainvoke("First question", config=cfg)
        print("ainvoke:", res["messages"][-1].content)

        async for chunk in agent.astream("Second question", config=cfg):
            print("astream chunk:", list(chunk.keys()))

asyncio.run(main())
```

---

## Summary Checklist

- [x] Use `State` + `Reduce` for Pydantic typed state with inferred reducers (or `SmartState` for dynamic schemas).
- [x] Use `@node` and `@node(idempotent=True)` for reusable, replay-safe custom graph nodes.
- [x] Use `create_agent()` or `AgentBuilder.with_model("provider:model")` for model-backed tool-calling loops with lazy provider resolution.
- [x] Attach `checkpointer` / `managed_checkpointer` / `async_checkpointer` and always pass an explicit `thread_id`.
- [x] Run synchronously (`invoke`/`stream`) or asynchronously (`ainvoke`/`astream`).
