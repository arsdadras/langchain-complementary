# Chapter 03: Memory & Context Management

In this chapter, you will prevent context-window overflow using `langcomp`'s **standalone memory primitives** (`ContextBuffer`, `AutoSummarizer`) and wire them cleanly into agent loops via their **built-in middleware adapters** (`ContextBufferMiddleware`, `AutoSummarizerMiddleware`).

---

## Primitives vs. Middleware Adapters

`langcomp` cleanly separates pure message-processing primitives from agent-loop middleware:

| Layer | Classes | Responsibility |
| :--- | :--- | :--- |
| **Standalone Memory Primitives** (`langcomp.memory`) | `ContextBuffer`, `AutoSummarizer` | Pure functions/objects over `Sequence[BaseMessage]` that you can call inside any custom graph node, script, or pipeline. |
| **Middleware Adapters** (`langcomp.middleware`) | `ContextBufferMiddleware`, `AutoSummarizerMiddleware` | `AgentMiddleware` wrappers that apply trimming or summarization **transiently** on `ModelRequest.messages` inside `wrap_model_call` / `awrap_model_call`, keeping persisted checkpoint state intact. |

---

## 1. Tool-Call-Pair-Safe Trimming with `ContextBuffer`

`ContextBuffer` caps a message list to `max_messages` (default `10`) while optionally preserving a leading `SystemMessage` (`preserve_system_prompt=True`).

Crucially, `ContextBuffer.trim()` is **tool-call-pair-safe**: if the sliding window cut lands in the middle of a tool exchange, it advances past orphaned `ToolMessage`s so an `AIMessage(tool_calls=...)` is never separated from its matching `ToolMessage`s (which would otherwise cause HTTP 400 errors from chat completion APIs).

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langcomp import ContextBuffer

messages = [
    SystemMessage(content="You are a support assistant."),
    HumanMessage(content="Check order #123"),
    AIMessage(
        content="",
        tool_calls=[{"name": "lookup_order", "args": {"id": "123"}, "id": "tc_1", "type": "tool_call"}],
    ),
    ToolMessage(content="Order #123 shipped yesterday.", tool_call_id="tc_1", name="lookup_order"),
    AIMessage(content="Order #123 shipped yesterday."),
    HumanMessage(content="Thanks!"),
]

# Cap at 4 messages while keeping index 0 SystemMessage and avoiding orphaned ToolMessages
buffer = ContextBuffer(max_messages=4, preserve_system_prompt=True)
trimmed = buffer.trim(messages)

for msg in trimmed:
    print(f"- [{msg.__class__.__name__}]: {msg.content}")
# Output:
# - [SystemMessage]: You are a support assistant.
# - [AIMessage]: Order #123 shipped yesterday.
# - [HumanMessage]: Thanks!
```

Notice that `trimmed` contains 3 messages (not 4) because keeping `ToolMessage` without its preceding `AIMessage(tool_calls=...)` would break tool-call pairing—`ContextBuffer` safely skips the orphan.

---

## 2. Conversation Compression with `AutoSummarizer`

Instead of dropping older context completely, `AutoSummarizer` compresses earlier turns into a single `SystemMessage` prefixed with `SUMMARY_PREFIX` (`"Previous Conversation Summary:\n"`) once `len(messages) > trigger_threshold` (default `15`), keeping the most recent `keep_recent` messages (default `5`).

### Safety Guarantees

- **Tool-call-pair-safe split:** Never leaves an orphaned `ToolMessage` at the start of the recent window (pulls the owning `AIMessage(tool_calls=...)` into the recent window when earlier messages exist, or folds the entire exchange into the summary).
- **Clean folding on repeated passes:** Prior auto-generated summaries are folded into the new summary instead of stacking multiple `SystemMessage`s—even when a base system prompt is present at index `0`.
- **Graceful fallback:** If `summarizer_func` is omitted or raises a runtime exception, `AutoSummarizer` logs a warning and falls back to a truncated transcript (up to 500 chars) instead of crashing the agent.

### Synchronous Summarization (`summarize_messages`)

```python
from langchain_core.messages import AIMessage, HumanMessage
from langcomp import AutoSummarizer

history = []
for i in range(1, 8):
    history.append(HumanMessage(content=f"User question {i}"))
    history.append(AIMessage(content=f"Assistant answer {i}"))

summarizer = AutoSummarizer(trigger_threshold=10, keep_recent=4)

def custom_summarizer(transcript: str) -> str:
    line_count = len(transcript.splitlines())
    return f"Condensed {line_count} earlier turns about support questions."

compact = summarizer.summarize_messages(history, summarizer_func=custom_summarizer)
print(f"Before: {len(history)} -> After: {len(compact)}")
print(compact[0].content)
# Output:
# Before: 14 -> After: 5
# Previous Conversation Summary:
# Condensed 10 earlier turns about support questions.
```

### Asynchronous Summarization (`asummarize_messages`)

In async workflows (`ainvoke`/`astream`), use `await summarizer.asummarize_messages(...)`, which accepts either a sync or an `async def` `summarizer_func`:

```python
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langcomp import AutoSummarizer

async def async_llm_summarizer(transcript: str) -> str:
    # In production, await your fast/cheap chat model here
    return f"Async summary ({len(transcript)} chars compressed)."

async def main() -> None:
    history = [
        msg
        for i in range(1, 8)
        for msg in (HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
    ]
    summarizer = AutoSummarizer(trigger_threshold=10, keep_recent=4)
    compact = await summarizer.asummarize_messages(
        history,
        summarizer_func=async_llm_summarizer,
    )
    print(compact[0].content)

asyncio.run(main())
```

---

## 3. Wiring Middleware into `create_agent` & `AgentBuilder`

To automatically trim or summarize before every model call without altering persisted checkpoint history, attach `ContextBufferMiddleware` or `AutoSummarizerMiddleware`.

### `ContextBufferMiddleware`

```python
from langcomp import (
    ContextBufferMiddleware,
    FakeModel,
    checkpointer,
    create_agent,
)

model = FakeModel(script=["Reply 1", "Reply 2", "Reply 3"])
agent = create_agent(
    model=model,
    middleware=[ContextBufferMiddleware(max_messages=2)],
    checkpointer=checkpointer("memory"),
)

cfg = {"configurable": {"thread_id": "trimmed-session"}}
agent.invoke("Turn 1", config=cfg)
agent.invoke("Turn 2", config=cfg)
agent.invoke("Turn 3", config=cfg)

# Only 2 messages were sent to the model on Turn 3...
print("Messages sent to model on Turn 3:", len(model.calls[-1]))  # 2
# ...while the checkpoint still retains the complete 6-message transcript!
print("Messages persisted in state:", len(agent.get_state(cfg).values["messages"]))  # 6
```

### `AutoSummarizerMiddleware` with `AgentBuilder`

`AutoSummarizerMiddleware` supports both sync (`invoke`) and async (`ainvoke`) execution and caches the computed summary for unchanged message prefixes so repeated model turns within a tool loop do not re-summarize the same prefix.

```python
from langcomp import AgentBuilder, AutoSummarizerMiddleware, FakeModel

def fast_summary(transcript: str) -> str:
    return f"Summary of {len(transcript.splitlines())} prior turns."

agent = (
    AgentBuilder("LongHorizonAgent")
    .with_model(FakeModel(script=["Got it!"]))
    .with_middleware([
        AutoSummarizerMiddleware(
            trigger_threshold=8,
            keep_recent=4,
            summarizer_func=fast_summary,
        )
    ])
    .compile()
)
```

> [!NOTE]
> Pass **either** an existing instance (`ContextBufferMiddleware(buffer=my_buf)` / `AutoSummarizerMiddleware(summarizer=my_sum)`) **or** keyword parameters (`max_messages=...`, `trigger_threshold=...`), not both—combining them raises `ConfigurationError`.

---

## Memory Strategy Summary

| Component | Layer | Primary Purpose | Best Used When |
| :--- | :--- | :--- | :--- |
| `ContextBuffer` | Primitive | Tool-call-safe hard sliding window | Trimming message lists manually in custom nodes |
| `AutoSummarizer` | Primitive | Sync & async (`asummarize_messages`) prefix compression | Compressing transcripts manually in custom workflows |
| `ContextBufferMiddleware` | Middleware | Transient sliding window per model call | Keeping API latency & token spend strictly bounded in `create_agent` / `AgentBuilder` |
| `AutoSummarizerMiddleware` | Middleware | Cached transient prefix summarization per model call | Long-running multi-turn conversations that must retain earlier facts |
