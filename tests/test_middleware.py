"""Unit tests for langcomp's dual-mechanism onion middleware system and built-in adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langcomp.core.nodes import AFTER_AGENT_NODE, BEFORE_AGENT_NODE

from langcomp import (
    AgentBuilder,
    AgentMiddleware,
    ApprovalGate,
    ApprovalGateMiddleware,
    AutoSummarizer,
    AutoSummarizerMiddleware,
    ContextBuffer,
    ContextBufferMiddleware,
    JumpTo,
    ModelRequest,
    ModelResponse,
    ScriptedFakeChatModel,
    ToolCallRequest,
    ToolCallResponse,
    ToolGuard,
    ToolGuardMiddleware,
    after_agent,
    after_model,
    before_agent,
    before_model,
    create_agent,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)


@tool
def echo_tool(text: str) -> str:
    """Echo input text back."""
    return f"echo: {text}"


@tool
def dangerous_tool(cmd: str) -> str:
    """Simulated dangerous tool."""
    return f"executed: {cmd}"


def test_empty_middleware_behaves_normally() -> None:
    fake = ScriptedFakeChatModel(responses=[AIMessage(content="Hello world")])
    agent = create_agent(model=fake, middleware=[])

    result = agent.invoke("Hi")

    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == "Hello world"


def test_wrap_model_call_retry_onion() -> None:
    attempts = 0

    class FlakyFakeModel(ScriptedFakeChatModel):
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
    agent = AgentBuilder("RetryAgent").with_model(FlakyFakeModel(responses=[])).add_middleware(retry_mw).compile()

    result = agent.invoke("Go")

    assert result["messages"][-1].content == "Success on attempt 3"
    assert retry_mw.retry_count == 2
    assert attempts == 3


def test_wrap_model_call_short_circuit_cache() -> None:
    fake = ScriptedFakeChatModel(responses=[AIMessage(content="Live response")])

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
    fake = ScriptedFakeChatModel(responses=["Done"])

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

    fake = ScriptedFakeChatModel(
        responses=[
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

    fake = ScriptedFakeChatModel(responses=["Final response"])
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
    fake = ScriptedFakeChatModel(responses=["Should never run"])
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
    fake = ScriptedFakeChatModel(responses=["Should not be called"])

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

    fake = ScriptedFakeChatModel(
        responses=[
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

    fake = ScriptedFakeChatModel(
        responses=[
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

    fake = ScriptedFakeChatModel(responses=["sniffed"])
    agent = create_agent(
        model=fake,
        middleware=[log_before, log_after, wrap_model, make_prompt],
    )

    # None of the decorators above declared before_agent or after_agent, so those
    # graph nodes must not be added unnecessarily.
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

    fake = ScriptedFakeChatModel(
        responses=[
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
    fake = ScriptedFakeChatModel(responses=["trimmed reply"])
    agent = create_agent(
        model=fake,
        middleware=[ContextBufferMiddleware(ContextBuffer(max_messages=3))],
    )

    history = [HumanMessage(f"msg {i}") for i in range(6)]
    result = agent.invoke(history)

    assert len(fake.calls[0]) == 3
    assert [m.content for m in fake.calls[0]] == ["msg 3", "msg 4", "msg 5"]
    # Full conversation history remains intact in persisted graph state.
    assert len(result["messages"]) == 7


def test_auto_summarizer_middleware_compresses_long_history() -> None:
    fake = ScriptedFakeChatModel(responses=["summarized reply"])
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

    fake = ScriptedFakeChatModel(
        responses=[
            AIMessage("", tool_calls=[{"id": "c1", "name": "flaky_tool", "args": {"x": 4}}]),
            AIMessage("Result is 40."),
        ]
    )
    agent = create_agent(
        model=fake,
        tools=[flaky_tool],
        middleware=[
            ToolGuardMiddleware(
                ToolGuard(max_retries=3, retry_delay_seconds=0.0)
            )
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

    fake = ScriptedFakeChatModel(
        responses=[
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


def test_approval_gate_middleware_blocks_unapproved_and_allows_approved() -> None:
    gate = ApprovalGate(approval_func=lambda name, args: args.get("cmd") != "rm -rf /")
    fake = ScriptedFakeChatModel(
        responses=[
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
