"""Example 06: Onion middleware, built-in adapters, FakeModel, and local TraceCollector.

Runs completely offline with zero API keys:
    python examples/06_middleware_and_lazy_models.py
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langcomp import (
    AgentBuilder,
    ApprovalGate,
    ApprovalGateMiddleware,
    ContextBufferMiddleware,
    FakeModel,
    GraphVisualizer,
    ModelRequest,
    ToolGuardMiddleware,
    TraceCollector,
    dynamic_prompt,
    summarize,
)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def delete_records(table: str) -> str:
    """Delete all rows from a table (guarded by ApprovalGateMiddleware)."""
    return f"Deleted {table}"


@dynamic_prompt
def contextual_prompt(req: ModelRequest) -> str:
    return f"You are a safe assistant. Active message count: {len(req.messages)}."


def main() -> None:
    fake_model = FakeModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "add", "args": {"a": 17, "b": 25}},
                    {"id": "call_2", "name": "delete_records", "args": {"table": "users"}},
                ],
                usage_metadata={"input_tokens": 60, "output_tokens": 25, "total_tokens": 85},
            ),
            AIMessage(
                content="17 + 25 = 42, and delete_records was safely blocked.",
                usage_metadata={"input_tokens": 90, "output_tokens": 20, "total_tokens": 110},
            ),
        ]
    )

    gate = ApprovalGate(approval_func=lambda tool_name, _args: tool_name != "delete_records")

    agent = (
        AgentBuilder("GuardedCalculator")
        .with_model(fake_model)
        .with_tools([add, delete_records])
        .with_middleware(
            [
                contextual_prompt,
                ContextBufferMiddleware(max_messages=8),
                ToolGuardMiddleware(max_retries=2, retry_delay_seconds=0.0),
                ApprovalGateMiddleware(gate, guarded_tools=["delete_records"]),
            ]
        )
        .compile()
    )

    collector = TraceCollector(out_dir=None)
    result = agent.invoke(
        "Add 17 and 25, and also clear the users table.",
        config={"callbacks": [collector]},
    )
    print("Final answer:", result["messages"][-1].content)
    print("Trace summary:", summarize(collector))
    print("\nGraph Mermaid:\n")
    print(GraphVisualizer.to_mermaid(agent))


if __name__ == "__main__":
    main()
