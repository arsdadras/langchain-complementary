# Chapter 05: DevTools, Tracing & Offline Testing

In this chapter, you will test agents deterministically offline with `FakeModel`, lock behavior with golden snapshots (`run_and_snapshot` / `arun_and_snapshot`), collect local latency/token/cost traces with `TraceCollector` and `summarize()`, unit-test nodes via `Node.invoke_for_test()` and `DryRunner`, and visualize graph topologies with `GraphVisualizer`.

---

## 1. Deterministic Offline Testing with `FakeModel`

`FakeModel` is a thread-safe scripted `BaseChatModel` double that runs completely offline with zero network calls. It supports:

- **Scripted turns:** `script: list[str | AIMessage]` (`ScriptTurn`), including scripted `AIMessage(tool_calls=...)` to test full multi-turn tool loops.
- **Call inspection:** `.calls` (list of input message lists per invocation, oldest first), `.call_count`, and `.bound_tool_names` (recorded non-mutatingly via `bind_tools`).
- **Resetting:** `.reset()` clears the playback cursor, recorded calls, and bound tool names.
- **Sync, async & streaming:** supports `invoke`, `ainvoke`, `stream`, and `astream`.
- **Strict bounds:** invoking beyond `len(script)` raises `ScriptExhaustedError` so unexpected extra model calls fail tests immediately.

```python
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langcomp import FakeModel, ScriptExhaustedError, create_agent

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

model = FakeModel(
    script=[
        AIMessage(
            content="",
            tool_calls=[
                {"name": "add", "args": {"a": 2, "b": 3}, "id": "call_add", "type": "tool_call"}
            ],
        ),
        "2 + 3 = 5",
    ]
)

agent = create_agent(model=model, tools=[add], system_prompt="Be concise.")
result = agent.invoke("What is 2 + 3?")

assert result["messages"][-1].content == "2 + 3 = 5"
assert model.call_count == 2
assert model.bound_tool_names == ("add",)
assert len(model.calls) == 2

# Streaming also works deterministically
stream_model = FakeModel(script=["Hello stream!"])
chunks = list(stream_model.stream("hi"))
assert "".join(str(c.content) for c in chunks) == "Hello stream!"

# Extra calls raise ScriptExhaustedError
try:
    stream_model.invoke("one too many")
except ScriptExhaustedError as exc:
    print("Caught expected exhaustion:", exc)
```

---

## 2. Golden Snapshot Testing (`run_and_snapshot` & `arun_and_snapshot`)

`run_and_snapshot()` (sync) and `arun_and_snapshot()` (async) invoke any `langcomp` `Agent` or LangGraph runnable and compare its normalized JSON output against a recorded golden file:

- On the **first run** (when `snapshot_path` does not yet exist) or when `update_snapshot=True`, the snapshot is written atomically.
- On **subsequent runs**, any drift raises `SnapshotMismatchError` with a unified diff showing exactly what changed.
- Internal `_langcomp*` state keys are automatically stripped from snapshots.

```python
import asyncio
from pathlib import Path
from langcomp import (
    FakeModel,
    SnapshotMismatchError,
    arun_and_snapshot,
    create_agent,
    run_and_snapshot,
)

snap_file = Path(".langcomp/snapshots/greeting.json")

agent_v1 = create_agent(model=FakeModel(script=["Hello, world!"]))
# 1st run records the snapshot; 2nd run asserts an exact match
run_and_snapshot(agent_v1, "Say hi", snap_file, update_snapshot=True)

agent_same = create_agent(model=FakeModel(script=["Hello, world!"]))
run_and_snapshot(agent_same, "Say hi", snap_file)  # passes!

# Output drift raises SnapshotMismatchError with unified diff
agent_drifted = create_agent(model=FakeModel(script=["Goodbye!"]))
try:
    run_and_snapshot(agent_drifted, "Say hi", snap_file)
except SnapshotMismatchError as exc:
    print("Detected regression:\n", exc)

# Async equivalent
async def check_async_snapshot() -> None:
    async_agent = create_agent(model=FakeModel(script=["Hello, world!"]))
    await arun_and_snapshot(async_agent, "Say hi", snap_file)

asyncio.run(check_async_snapshot())
```

---

## 3. Local Observability with `TraceCollector` & `summarize()`

`TraceCollector` is a thread-safe LangChain `BaseCallbackHandler` that captures per-call latency, token usage, estimated USD cost, errors, LangGraph node names, and subagent checkpoint namespaces—without needing a cloud tracing service.

- Attach via `config={"callbacks": [collector]}`.
- Writes JSONL events to `out_dir/<run_id>.jsonl` by default (`".langcomp/traces"`), or pass `out_dir=None` for in-memory collection only.
- Supply `price_table` mapping model names (or `"default"`) to `(input_cost_per_1k_tokens, output_cost_per_1k_tokens)` with longest-prefix matching for dated model IDs (e.g., `"gpt-4o-2024-08-06"` matches `"gpt-4o"`).
- Optionally bound in-memory history with `max_events=...`.
- Call `collector.summarize()` or `summarize(events)` for totals (`total_calls`, `error_calls`, `total_latency_s`, `total_input_tokens`, `total_output_tokens`, `total_cost_usd`) plus `by_node` and `by_model` breakdowns.

```python
from langchain_core.messages import AIMessage
from langcomp import FakeModel, TraceCollector, create_agent, summarize

traced_model = FakeModel(
    script=[
        AIMessage(
            content="All systems nominal.",
            response_metadata={"model_name": "gpt-4o-mini"},
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        )
    ]
)

collector = TraceCollector(
    run_id="health-check-01",
    price_table={"gpt-4o-mini": (0.00015, 0.0006)},
    out_dir=None,  # set a directory path to persist JSONL traces to disk
)

agent = create_agent(model=traced_model, name="HealthAgent")
agent.invoke("Status?", config={"callbacks": [collector]})

report = collector.summarize()  # equivalent to summarize(collector.events)
print("Total calls:", report["total_calls"])
print("Input/Output tokens:", report["total_input_tokens"], report["total_output_tokens"])
print("Estimated cost (USD):", report["total_cost_usd"])
print("Per-node breakdown:", report["by_node"])
print("Per-model breakdown:", report["by_model"])
```

---

## 4. Direct Node Unit Testing with `Node.invoke_for_test()`

Functions decorated with `@node` return a `Node` object with `.invoke_for_test(state)`, letting you test pure node logic directly without compiling a `StateGraph`:

```python
from langcomp import node

@node(idempotent=True)
def score_items(state):
    items = state.get("items", [])
    return {"score": sum(len(item) for item in items)}

# Execute directly without compiling a graph
out = score_items.invoke_for_test({"items": ["apple", "banana", "cherry"]})
assert out == {"score": 17}
```

---

## 5. Graph Inspection with `GraphVisualizer` (`xray` Support)

`GraphVisualizer` exports any `langcomp.Agent` or compiled LangGraph runnable to **Mermaid** or **ASCII** diagrams offline:

- `GraphVisualizer.to_mermaid(agent_or_graph, xray=False)`
- `GraphVisualizer.to_ascii(agent_or_graph, xray=False)` — uses `grandalf` when `langcomp[visualizer]` is installed, and falls back cleanly to a zero-dependency edge-list rendering otherwise.
- Pass `xray=True` (or an integer depth like `xray=2`) to expand nested subgraphs.

```python
from langchain_core.messages import AIMessage
from langcomp import AgentBuilder, GraphVisualizer

def step_one(state):
    return {"messages": [AIMessage(content="Step 1 complete")]}

def step_two(state):
    return {"messages": [AIMessage(content="Step 2 complete")]}

pipeline = (
    AgentBuilder("PipelineAgent")
    .add_node("step_one", step_one)
    .add_node("step_two", step_two)
    .compile()
)

print("=== Mermaid ===")
print(GraphVisualizer.to_mermaid(pipeline, xray=True))

print("\n=== ASCII ===")
print(GraphVisualizer.to_ascii(pipeline, xray=True))
```

---

## 6. Stepping Compiled Nodes with `DryRunner` (`run_step` & `arun_step`)

`DryRunner` executes a single compiled node inside an `Agent` or `CompiledStateGraph` against a state dictionary, supporting both synchronous (`run_step`) and asynchronous (`arun_step`) nodes with optional `config`:

> [!NOTE]
> `DryRunner` invokes the target node's runnable directly. If a node calls a chat model or external API, pass a `FakeModel` or mock tool when constructing the agent to keep tests fully offline.

```python
import asyncio
from langcomp import AgentBuilder, DryRunner

def calculate_score(state):
    items = state.get("items", [])
    return {"score": sum(len(item) for item in items)}

async def double_score(state):
    return {"score": state["score"] * 2}

scoring_agent = (
    AgentBuilder("ScoringAgent")
    .with_state_schema({"items": list, "score": int})
    .add_node("calculate_score", calculate_score)
    .add_node("double_score", double_score)
    .compile()
)

runner = DryRunner(scoring_agent)

# Sync step
step1 = runner.run_step("calculate_score", {"items": ["apple", "banana", "cherry"]})
assert step1["score"] == 17

# Async step
async def run_async_step() -> None:
    step2 = await runner.arun_step("double_score", step1)
    assert step2["score"] == 34

asyncio.run(run_async_step())
```

---

## 7. Complete `pytest` Template

Combining `FakeModel`, `Node.invoke_for_test()`, `DryRunner`, `run_and_snapshot`, and `TraceCollector` gives you fast, zero-cost, CI-friendly agent tests:

```python
from pathlib import Path
import pytest
from langchain_core.messages import AIMessage
from langcomp import (
    AgentBuilder,
    DryRunner,
    FakeModel,
    TraceCollector,
    create_agent,
    node,
    run_and_snapshot,
)

@node
def normalize_node(state):
    text = state["messages"][-1].content
    return {"messages": [AIMessage(content=text.strip().upper())]}

def test_node_directly():
    res = normalize_node.invoke_for_test({"messages": [AIMessage(content="  hello  ")]})
    assert res["messages"][-1].content == "HELLO"

def test_dry_runner_step():
    agent = AgentBuilder("NormAgent").add_node(normalize_node).compile()
    res = DryRunner(agent).run_step("normalize_node", {"messages": [AIMessage(content="unit test")]})
    assert res["messages"][-1].content == "UNIT TEST"

def test_model_agent_with_snapshot_and_trace(tmp_path: Path):
    model = FakeModel(
        script=[
            AIMessage(
                content="Checked!",
                response_metadata={"model_name": "gpt-4o-mini"},
                usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
            )
        ]
    )
    collector = TraceCollector(out_dir=None)
    agent = create_agent(model=model, system_prompt="Reply briefly.")

    snap = tmp_path / "checked.json"
    result = run_and_snapshot(agent, "Ping", snap, config={"callbacks": [collector]})
    assert result["messages"][-1].content == "Checked!"
    assert collector.summarize()["total_calls"] == 1

    # Second run verifies zero drift against the recorded snapshot
    model.reset()
    run_and_snapshot(agent, "Ping", snap)
```

---

## Tutorial Suite Wrap-Up

You have completed the `langcomp` v0.2.0 tutorial suite. You can now:

- Define typed state with `State`/`Reduce` and `SmartState`, and build replay-safe graphs with `@node(idempotent=True)`, `AgentBuilder`, `create_agent`, and `@agent`.
- Resolve models lazily with `"provider:model"` specs (`resolve_model`, `register_provider`) and compose sync/async onion middleware.
- Persist threads safely with `checkpointer`, `managed_checkpointer`, and `async_checkpointer` using explicit `thread_id`s.
- Orchestrate flat tool-isolated swarms with `SwarmPool`/`SubAgent` and nested DeepAgents hierarchies with `Team.compile()`.
- Protect context windows with `ContextBuffer`/`AutoSummarizer` and their middleware adapters, and guard tools with fail-closed `ApprovalGate` + `ToolGuard` middleware.
- Test and inspect everything offline with `FakeModel`, `run_and_snapshot`/`arun_and_snapshot`, `TraceCollector`/`summarize()`, `Node.invoke_for_test()`, `GraphVisualizer`, and `DryRunner`.
