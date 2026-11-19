# `langcomp` Repository Guidelines

## Project Context

This is **`langcomp`** — a lightweight, developer-friendly complementary framework on top of LangChain, LangGraph, and DeepAgents (Python 3.10+).

- **Language / framework:** Python 3.10+, `langchain-core`, `langgraph`, `pydantic` v2 (optional extras: `google`, `openai`, `sqlite`, `postgres`, `deepagents`, `visualizer`)
- **Package layout:** `src/langcomp/` (`core/`, `middleware/`, `models/`, `memory/`, `tools/`, `swarm/`, `devtools/`, `errors.py`, `persistence.py`, `testing.py`)
- **Package manager:** `uv` (`pyproject.toml`, `uv.lock`)
- **Test runner:** `uv run pytest -v` (coverage: `uv run pytest --cov=langcomp --cov-report=term-missing`)
- **Lint / typecheck tools:** `uv run ruff check src tests examples` and `mypy src`

## Coding Standards

- Type hints (PEP 585 / PEP 604: `list[...]`, `dict[...]`, `X | None`) are required on all public functions and classes (`py.typed` is shipped).
- All framework exceptions should inherit from `langcomp.errors.LangcompError`.
- Follow the existing module layout — do not introduce a new top-level package without asking first.
- Keep all optional provider/backend SDK imports (`langchain-openai`, `langchain-google-genai`, `deepagents`, `langgraph-checkpoint-sqlite`, `langgraph-checkpoint-postgres`, `grandalf`) lazy inside functions/methods.
- Never mutate caller-owned Pydantic v2 `BaseTool` instances in place; wrap via non-mutating clones/subclasses.

## LangGraph / Agent-Specific

- State schemas and reducers live in `src/langcomp/core/state.py` (`State`, `Reduce`, `SmartState`). Read it before adding new state keys or internal `_langcomp*` keys.
- Node builders live in `src/langcomp/core/nodes.py` and the agent loop lives in `src/langcomp/core/loop.py`. Maintain sync + async parity (`RunnableLambda(sync_fn, afunc=async_fn)`).
- Never catch `langgraph.errors.GraphBubbleUp` or `ApprovalRejectedError` in tool retry/error wrappers.
- Strip external `_langcomp*` keys in `Agent._prepare_payload()`; require explicit `thread_id` whenever a checkpointer is attached.
- Before adding or changing a LangGraph/LangChain/DeepAgents call, verify the current signature via the `docs-langchain` MCP server.

## Testing Requirements

- Tests live in `tests/` with a 1:1 module mapping (`test_state.py`, `test_agent.py`, `test_loop.py`, `test_middleware.py`, `test_models.py`, `test_memory.py`, `test_tools_guard.py`, `test_swarm.py`, `test_persistence.py`, `test_testing.py`, `test_devtools.py`, `test_package_api.py`, `test_integration.py`) plus shared fixtures in `tests/conftest.py`.
- Every new node, middleware hook, or service function gets both sync and async tests where applicable, running 100% offline using `FakeModel`.
- Run `uv run ruff check src tests examples` and `uv run pytest --cov=langcomp` before declaring a task done, and report real command outputs.
- Never weaken or skip an existing test to make a change pass.

## Things That Require Explicit Confirmation First

- Any breaking public API or security-default change.
- Any `git push --force`, history rewrite, or branch/file deletion.
- Any change to production config, secrets, or deployment files.

## Do Not

- Do not hardcode API keys/tokens/secrets in source files — use environment variables (`.env`, gitignored).
- Do not invent library functions/parameters you have not verified.
- Do not silently change public function signatures other code depends on.
