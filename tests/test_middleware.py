"""Unit tests for langcomp's dual-mechanism onion middleware system and built-in adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph.message import add_messages

from langcomp import (
    AgentBuilder,
    AgentMiddleware,
    ApprovalGate,
    ApprovalGateMiddleware,
    ApprovalRejectedError,
    AutoSummarizer,
    AutoSummarizerMiddleware,
    ConfigurationError,
    ContextBuffer,
    ContextBufferMiddleware,
    FakeModel,
    JumpTo,
    ModelHandler,
    ModelRequest,
    ModelResponse,
    State,
    ToolCallRequest,
    ToolCallResponse,
    ToolGuard,
    ToolGuardMiddleware,
    ToolHandler,
    after_agent,
    after_model,
    before_agent,
    before_model,
    create_agent,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langcomp.core.nodes import AFTER_AGENT_NODE, BEFORE_AGENT_NODE


@tool
def echo_tool(text: str) -> str:
    """Echo input text back."""
    return f"echo: {text}"


@tool
def dangerous_tool(cmd: str) -> str:
    """Simulated dangerous tool."""
    return f"executed: {cmd}"


def test_empty_middleware_behaves_normally() -> None:
    fake = FakeModel(script=[AIMessage(content="Hello world")])
    agent = create_agent(model=fake, middleware=[])

    result = agent.invoke("Hi")

    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == "Hello world"


def test_wrap_model_call_retry_onion() -> None:
    attempts = 0

    class FlakyFakeModel(FakeModel):
        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            del input, config, kwargs
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError(f"Transient error on attempt {attempts}")
            return AIMessage(content="Success on attempt 3")

    class RetryMiddleware(AgentMiddleware):
        def __init__(self, max_retries: int = 3) -> None:
            self.max_retries = max_retries
            self.retry_count = 0

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            last_err: Exception | None = None
            for _ in range(self.max_retries):
                try:
                    return handler(request)
                except RuntimeError as err:
                    self.retry_count += 1
                    last_err = err
            assert last_err is not None
            raise last_err

    retry_mw = RetryMiddleware(max_retries=3)
    agent = (
        AgentBuilder("RetryAgent")
        .with_model(FlakyFakeModel(script=[]))
        .add_middleware(retry_mw)
        .compile()
    )

    result = agent.invoke("Go")

    assert result["messages"][-1].content == "Success on attempt 3"
    assert retry_mw.retry_count == 2
    assert attempts == 3


def test_wrap_model_call_short_circuit_cache() -> None:
    fake = FakeModel(script=[AIMessage(content="Live response")])

    class CachingMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            if request.messages and "cached" in str(request.messages[0].content):
                return ModelResponse(message=AIMessage(content="Cached response"))
            return handler(request)

    agent = create_agent(model=fake, middleware=[CachingMiddleware()])

    result = agent.invoke("Please return cached info")

    assert result["messages"][-1].content == "Cached response"
    assert fake.call_count == 0


def test_wrap_model_call_request_override() -> None:
    fake = FakeModel(script=["Done"])

    class AugmentPromptMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            new_prompt = f"{request.system_prompt or ''} [injected]".strip()
            return handler(request.override(system_prompt=new_prompt))

    agent = create_agent(
        model=fake,
        system_prompt="Base prompt.",
        middleware=[AugmentPromptMiddleware()],
    )

    agent.invoke("Hello")

    assert fake.calls[0][0] == SystemMessage(content="Base prompt. [injected]")


def test_wrap_tool_call_guardrail_and_state_update() -> None:
    class AuditState(TypedDict):
        blocked_count: int

    fake = FakeModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "dangerous_tool", "args": {"cmd": "rm -rf /"}}
                ],
            ),
            AIMessage(content="Understood, command blocked."),
        ]
    )

    class GuardrailMiddleware(AgentMiddleware):
        state_schema = AuditState

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolCallResponse],
        ) -> ToolCallResponse:
            if request.tool.name == "dangerous_tool":
                prev = int(request.state.get("blocked_count", 0))
                return ToolCallResponse(
                    message=ToolMessage(
                        content="PermissionDenied: blocked by policy",
                        tool_call_id=request.tool_call["id"],
                        status="error",
                    ),
                    state_update={"blocked_count": prev + 1},
                )
            return handler(request)

    agent = create_agent(
        model=fake,
        tools=[dangerous_tool],
        middleware=[GuardrailMiddleware()],
    )

    result = agent.invoke("Please delete files")

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "PermissionDenied" in tool_messages[0].content
    assert tool_messages[0].status == "error"
    assert result["blocked_count"] == 1


def test_state_subclass_merges_with_middleware_state_schema() -> None:
    class AppState(State):
        messages: Annotated[list[AnyMessage], add_messages] = []
        tags: list[str] = []

    class MiddlewareExtra(TypedDict):
        mw_runs: int

    class TaggingMiddleware(AgentMiddleware):
        state_schema = MiddlewareExtra

        def before_model(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
            return {
                "tags": ["checked"],
                "mw_runs": int(state.get("mw_runs", 0)) + 1,
            }

    fake = FakeModel(script=["all good"])
    agent = (
        AgentBuilder("TypedMiddlewareAgent")
        .with_state_schema(AppState)
        .with_model(fake)
        .add_middleware(TaggingMiddleware())
        .compile()
    )

    result = agent.invoke({"messages": ["hi"], "tags": ["init"]})

    assert result["tags"] == ["init", "checked"]
    assert result["mw_runs"] == 1
    assert result["messages"][-1].content == "all good"


def test_lifecycle_hooks_execution_order() -> None:
    trace: list[str] = []

    class TracingMiddleware(AgentMiddleware):
        def __init__(self, name: str) -> None:
            self.name = name

        def before_agent(self, state: Mapping[str, Any]) -> None:
            del state
            trace.append(f"{self.name}.before_agent")

        def before_model(self, state: Mapping[str, Any]) -> None:
            del state
            trace.append(f"{self.name}.before_model")

        def after_model(self, state: Mapping[str, Any]) -> None:
            del state
            trace.append(f"{self.name}.after_model")

        def after_agent(self, state: Mapping[str, Any]) -> None:
            del state
            trace.append(f"{self.name}.after_agent")

    fake = FakeModel(script=["Final response"])
    agent = create_agent(model=fake, middleware=[TracingMiddleware("M1"), TracingMiddleware("M2")])

    agent.invoke("Hi")

    assert trace == [
        "M1.before_agent",
        "M2.before_agent",
        "M1.before_model",
        "M2.before_model",
        "M2.after_model",
        "M1.after_model",
        "M2.after_agent",
        "M1.after_agent",
    ]


def test_before_agent_jump_to_end_skips_model() -> None:
    fake = FakeModel(script=["Should never run"])
    after_agent_called = False

    class EarlyGateMiddleware(AgentMiddleware):
        def before_agent(self, state: Mapping[str, Any]) -> JumpTo | None:
            if "forbidden" in str(state["messages"][-1].content):
                return JumpTo("end", messages=[AIMessage(content="Blocked before model.")])
            return None

        def after_agent(self, state: Mapping[str, Any]) -> None:
            del state
            nonlocal after_agent_called
            after_agent_called = True

    agent = create_agent(model=fake, middleware=[EarlyGateMiddleware()])

    result = agent.invoke("This is forbidden input")

    assert result["messages"][-1].content == "Blocked before model."
    assert fake.call_count == 0
    assert after_agent_called is True


def test_before_model_jump_to_end_short_circuits() -> None:
    fake = FakeModel(script=["Should not be called"])

    class LimitMiddleware(AgentMiddleware):
        def before_model(self, state: Mapping[str, Any]) -> JumpTo | None:
            if len(state["messages"]) >= 2:
                return JumpTo("end", messages=[AIMessage(content="Max turns reached.")])
            return None

    agent = create_agent(model=fake, middleware=[LimitMiddleware()])

    result = agent.invoke([HumanMessage("Msg 1"), HumanMessage("Msg 2")])

    assert result["messages"][-1].content == "Max turns reached."
    assert fake.call_count == 0


def test_after_model_jump_to_end_cancels_tool_execution() -> None:
    tool_executed = False

    @tool
    def side_effect_tool() -> str:
        """Should not execute when after_model jumps to end."""
        nonlocal tool_executed
        tool_executed = True
        return "executed"

    fake = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"id": "c1", "name": "side_effect_tool", "args": {}}]),
        ]
    )

    @after_model
    def halt_on_tool_call(state: Mapping[str, Any]) -> JumpTo | None:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return JumpTo("end", messages=[AIMessage(content="Tool call halted by after_model.")])
        return None

    agent = create_agent(model=fake, tools=[side_effect_tool], middleware=[halt_on_tool_call])

    result = agent.invoke("Run side effect")

    assert tool_executed is False
    assert result["messages"][-1].content == "Tool call halted by after_model."


def test_middleware_contributes_tools_and_state_schema() -> None:
    class CounterState(TypedDict):
        echo_calls: int

    fake = FakeModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "echo_tool", "args": {"text": "langcomp"}}],
            ),
            AIMessage(content="Finished"),
        ]
    )

    class EchoCapability(AgentMiddleware):
        tools: tuple[BaseTool, ...] = (echo_tool,)
        state_schema = CounterState

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolCallResponse],
        ) -> ToolCallResponse:
            resp = handler(request)
            count = int(request.state.get("echo_calls", 0)) + 1
            return ToolCallResponse(message=resp.message, state_update={"echo_calls": count})

    agent = create_agent(model=fake, middleware=[EchoCapability()])

    result = agent.invoke("Call echo")

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "echo: langcomp"
    assert result["echo_calls"] == 1


def test_functional_decorators_and_no_spurious_lifecycle_nodes() -> None:
    trace: list[str] = []

    @before_model
    def log_before(state: Mapping[str, Any]) -> None:
        del state
        trace.append("before_model")

    @after_model
    def log_after(state: Mapping[str, Any]) -> None:
        del state
        trace.append("after_model")

    @wrap_model_call
    def wrap_model(
        req: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        trace.append("wrap_before")
        res = handler(req)
        trace.append("wrap_after")
        return res

    @dynamic_prompt
    def make_prompt(req: ModelRequest) -> str:
        return f"Dynamic prompt for {len(req.messages)} msg(s)"

    fake = FakeModel(script=["sniffed"])
    agent = create_agent(
        model=fake,
        middleware=[log_before, log_after, wrap_model, make_prompt],
    )

    node_names = set(agent.graph.nodes)
    assert BEFORE_AGENT_NODE not in node_names
    assert AFTER_AGENT_NODE not in node_names

    res = agent.invoke("Hello")

    assert res["messages"][-1].content == "sniffed"
    assert trace == ["before_model", "wrap_before", "wrap_after", "after_model"]
    assert fake.calls[0][0] == SystemMessage(content="Dynamic prompt for 1 msg(s)")


def test_before_and_after_agent_decorators_and_wrap_tool_decorator() -> None:
    trace: list[str] = []

    @before_agent
    def on_start(state: Mapping[str, Any]) -> None:
        del state
        trace.append("start")

    @wrap_tool_call
    def audit_tool(
        req: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResponse],
    ) -> ToolCallResponse:
        trace.append(f"tool:{req.tool.name}")
        return handler(req)

    @after_agent
    def on_end(state: Mapping[str, Any]) -> None:
        del state
        trace.append("end")

    fake = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"id": "c1", "name": "echo_tool", "args": {"text": "hi"}}]),
            AIMessage("done"),
        ]
    )
    agent = create_agent(
        model=fake,
        tools=[echo_tool],
        middleware=[on_start, audit_tool, on_end],
    )

    agent.invoke("go")

    assert trace == ["start", "tool:echo_tool", "end"]


def test_context_buffer_middleware_trims_model_input_only() -> None:
    fake = FakeModel(script=["trimmed reply"])
    agent = create_agent(
        model=fake,
        middleware=[ContextBufferMiddleware(ContextBuffer(max_messages=3))],
    )

    history = [HumanMessage(f"msg {i}") for i in range(6)]
    result = agent.invoke(history)

    assert len(fake.calls[0]) == 3
    assert [m.content for m in fake.calls[0]] == ["msg 3", "msg 4", "msg 5"]
    assert len(result["messages"]) == 7


def test_auto_summarizer_middleware_compresses_long_history() -> None:
    fake = FakeModel(script=["summarized reply"])
    summarizer = AutoSummarizer(trigger_threshold=4, keep_recent=2)
    agent = create_agent(
        model=fake,
        middleware=[
            AutoSummarizerMiddleware(
                summarizer,
                summarizer_func=lambda text: f"SHORT({len(text)})",
            )
        ],
    )

    history = [HumanMessage(f"Turn {i}") for i in range(6)]
    agent.invoke(history)

    sent = fake.calls[0]
    assert len(sent) == 3  # 1 summary SystemMessage + 2 recent
    assert isinstance(sent[0], SystemMessage)
    assert "SHORT(" in str(sent[0].content)


def test_tool_guard_middleware_retries_and_recovers() -> None:
    attempts = 0

    @tool
    def flaky_tool(x: int) -> int:
        """Fails twice then succeeds."""
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary failure")
        return x * 10

    fake = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"id": "c1", "name": "flaky_tool", "args": {"x": 4}}]),
            AIMessage("Result is 40."),
        ]
    )
    agent = create_agent(
        model=fake,
        tools=[flaky_tool],
        middleware=[
            ToolGuardMiddleware(ToolGuard(max_retries=3, retry_delay_seconds=0.0)),
        ],
    )

    result = agent.invoke("compute")

    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert attempts == 3
    assert tool_msg.content == "40"
    assert tool_msg.status != "error"


def test_tool_guard_middleware_returns_fallback_after_exhausting_retries() -> None:
    @tool
    def always_broken() -> str:
        """Always fails."""
        raise RuntimeError("permanent failure")

    fake = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"id": "c1", "name": "always_broken", "args": {}}]),
            AIMessage("Used fallback."),
        ]
    )
    agent = create_agent(
        model=fake,
        tools=[always_broken],
        middleware=[
            ToolGuardMiddleware(
                ToolGuard(
                    max_retries=2,
                    fallback_output="cached-safe-default",
                    retry_delay_seconds=0.0,
                )
            )
        ],
    )

    result = agent.invoke("try broken")

    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "cached-safe-default"


def test_tool_guard_middleware_never_retries_approval_rejection() -> None:
    checks = 0

    def deny(_name: str, _args: dict[str, Any]) -> bool:
        nonlocal checks
        checks += 1
        return False

    fake = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"id": "c1", "name": "dangerous_tool", "args": {"cmd": "x"}}]),
            AIMessage("Stopped."),
        ]
    )
    agent = create_agent(
        model=fake,
        tools=[dangerous_tool],
        middleware=[
            ToolGuardMiddleware(max_retries=3, fallback_output="should-not-use", retry_delay_seconds=0.0),
            ApprovalGateMiddleware(ApprovalGate(approval_func=deny), raise_on_reject=True),
        ],
    )

    with pytest.raises(ApprovalRejectedError):
        agent.invoke("run")
    assert checks == 1


def test_approval_gate_middleware_blocks_unapproved_and_allows_approved() -> None:
    gate = ApprovalGate(approval_func=lambda name, args: args.get("cmd") != "rm -rf /")
    fake = FakeModel(
        script=[
            AIMessage(
                "",
                tool_calls=[
                    {"id": "c1", "name": "dangerous_tool", "args": {"cmd": "rm -rf /"}},
                    {"id": "c2", "name": "echo_tool", "args": {"text": "safe"}},
                ],
            ),
            AIMessage("Done."),
        ]
    )
    agent = create_agent(
        model=fake,
        tools=[dangerous_tool, echo_tool],
        middleware=[
            ApprovalGateMiddleware(gate, guarded_tools=["dangerous_tool"]),
        ],
    )

    result = agent.invoke("run commands")

    by_id = {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}
    assert by_id["c1"].status == "error"
    assert "PermissionDenied" in str(by_id["c1"].content)
    assert by_id["c2"].content == "echo: safe"


def test_tool_guard_does_not_retry_or_fallback_on_non_raising_approval_rejection() -> None:
    checks = 0

    def deny(_name: str, _args: dict[str, Any]) -> bool:
        nonlocal checks
        checks += 1
        return False

    fake = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"id": "c1", "name": "dangerous_tool", "args": {"cmd": "x"}}]),
            AIMessage("Blocked safely."),
        ]
    )
    ag = create_agent(
        model=fake,
        tools=[dangerous_tool],
        middleware=[
            ToolGuardMiddleware(max_retries=3, fallback_output="BYPASSED", retry_delay_seconds=0.0),
            ApprovalGateMiddleware(ApprovalGate(approval_func=deny), guarded_tools=["dangerous_tool"]),
        ],
    )
    res = ag.invoke("run")
    tool_msg = next(m for m in res["messages"] if isinstance(m, ToolMessage))
    assert checks == 1
    assert tool_msg.status == "error"
    assert "BYPASSED" not in str(tool_msg.content)


def test_before_agent_and_after_agent_preserve_messages_and_reject_after_agent_jump() -> None:
    seen: list[list[str]] = []

    @before_agent
    def mw1(state: Mapping[str, Any]) -> dict[str, Any]:
        del state
        return {"messages": [HumanMessage(content="injected")]}

    @before_agent
    def mw2(state: Mapping[str, Any]) -> None:
        seen.append([str(m.content) for m in state["messages"]])
        return None

    ag = create_agent(FakeModel(script=["ok"]), middleware=[mw1, mw2])
    ag.invoke("original")
    assert seen == [["original", "injected"]]

    @after_agent
    def bad_jump(state: Mapping[str, Any]) -> JumpTo:
        del state
        return JumpTo("model")

    ag_bad = create_agent(FakeModel(script=["ok"]), middleware=[bad_jump])
    with pytest.raises(ConfigurationError, match="after_agent middleware cannot return JumpTo"):
        ag_bad.invoke("hi")


def test_builtin_middleware_rejects_conflicting_instance_and_kwargs() -> None:
    from langcomp import AutoSummarizer, ContextBuffer, ToolGuard
    from langcomp.middleware.compose import normalize_middleware

    with pytest.raises(ConfigurationError, match="ContextBufferMiddleware"):
        ContextBufferMiddleware(buffer=ContextBuffer(max_messages=5), max_messages=10)
    with pytest.raises(ConfigurationError, match="AutoSummarizerMiddleware"):
        AutoSummarizerMiddleware(summarizer=AutoSummarizer(trigger_threshold=5), keep_recent=2)
    with pytest.raises(ConfigurationError, match="ToolGuardMiddleware"):
        ToolGuardMiddleware(guard=ToolGuard(max_retries=2), max_retries=5)
    with pytest.raises(TypeError, match="Expected AgentMiddleware"):
        normalize_middleware("not-middleware")  # type: ignore[arg-type]


async def test_async_middleware_hooks_and_bridged_sync_middleware() -> None:
    events: list[str] = []

    @before_agent
    async def a_before_agent(state: Mapping[str, Any]) -> None:
        del state
        events.append("abefore_agent")

    @before_model
    async def a_before_model(state: Mapping[str, Any]) -> None:
        del state
        events.append("abefore_model")

    @dynamic_prompt
    async def a_prompt(req: ModelRequest) -> str:
        del req
        events.append("aprompt")
        return "async-system"

    @wrap_model_call
    def sync_outer(req: ModelRequest, handler: ModelHandler) -> ModelResponse:
        events.append("sync_outer_in")
        resp = handler(req)
        events.append("sync_outer_out")
        return resp

    @after_model
    async def a_after_model(state: Mapping[str, Any]) -> None:
        del state
        events.append("aafter_model")

    @after_agent
    async def a_after_agent(state: Mapping[str, Any]) -> None:
        del state
        events.append("aafter_agent")

    async def async_policy(_name: str, args: dict[str, Any]) -> bool:
        return args.get("text") != "forbidden"

    fake = FakeModel(
        script=[
            AIMessage("", tool_calls=[{"id": "c1", "name": "echo_tool", "args": {"text": "ok"}}]),
            AIMessage("final-async"),
        ]
    )
    ag = create_agent(
        model=fake,
        tools=[echo_tool],
        middleware=[
            a_before_agent,
            a_before_model,
            a_prompt,
            sync_outer,
            a_after_model,
            a_after_agent,
            ContextBufferMiddleware(max_messages=10),
            ToolGuardMiddleware(max_retries=2, retry_delay_seconds=0.0),
            ApprovalGateMiddleware(ApprovalGate(approval_func=async_policy)),
        ],
    )
    result = await ag.ainvoke("hi async")
    assert result["messages"][-1].content == "final-async"
    assert fake.calls[0][0] == SystemMessage("async-system")
    assert events[0] == "abefore_agent"
    assert events[-1] == "aafter_agent"
    assert "aprompt" in events
    assert "sync_outer_in" in events
    assert "sync_outer_out" in events

    # Async hook called synchronously fails cleanly with ConfigurationError
    ag_sync_fail = create_agent(FakeModel(script=["x"]), middleware=[a_before_model])
    with pytest.raises(ConfigurationError, match="cannot be executed synchronously"):
        ag_sync_fail.invoke("hi")


def test_parallel_tool_calls_respect_state_reducers() -> None:
    import operator

    class LogState(State):
        logs: list[str] = []

    class TaggingToolMW(AgentMiddleware):
        def wrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolCallResponse:
            resp = handler(request)
            return ToolCallResponse(
                message=resp.message,
                state_update={"logs": [request.tool_call["id"]]},
            )

    fake = FakeModel(
        script=[
            AIMessage(
                "",
                tool_calls=[
                    {"id": "c1", "name": "echo_tool", "args": {"text": "a"}},
                    {"id": "c2", "name": "echo_tool", "args": {"text": "b"}},
                ],
            ),
            AIMessage("done"),
        ]
    )
    ag = create_agent(
        model=fake,
        tools=[echo_tool],
        state_schema=LogState,
        middleware=[TaggingToolMW()],
    )
    assert operator.add is not None
    res = ag.invoke("go")
    assert res["logs"] == ["c1", "c2"]
