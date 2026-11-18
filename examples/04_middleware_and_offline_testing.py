"""Offline example: AgentBuilder with onion middleware, built-in adapters, and ScriptedFakeChatModel.

Run without any API keys:
    python examples/04_middleware_and_offline_testing.py
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langcomp import (
    AgentBuilder,
    ApprovalGate,
    ApprovalGateMiddleware,
    ContextBufferMiddleware,
    GraphVisualizer,
    ModelRequest,
    ScriptedFakeChatModel,
    ToolGuardMiddleware,
    dynamic_prompt,
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
    fake_model = ScriptedFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "add", "args": {"a": 17, "b": 25}},
                    {"id": "call_2", "name": "delete_records", "args": {"table": "users"}},
                ],
            ),
            AIMessage(content="17 + 25 = 42, and delete_records was safely blocked."),
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

    result = agent.invoke("Add 17 and 25, and also clear the users table.")
    print("Final answer:", result["messages"][-1].content)
    print("\nGraph Mermaid:\n")
    print(GraphVisualizer.to_mermaid(agent))


if __name__ == "__main__":
    main()
