"""
Example 02: Deterministic offline testing with FakeModel, TraceCollector, and snapshots.
Runs completely offline with zero API keys.
"""

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langcomp import AgentBuilder, FakeModel, TraceCollector, summarize


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def main() -> None:
    model = FakeModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "multiply", "args": {"a": 12, "b": 8}, "id": "call_1"}
                ],
                usage_metadata={
                    "input_tokens": 45,
                    "output_tokens": 18,
                    "total_tokens": 63,
                },
            ),
            AIMessage(
                content="12 multiplied by 8 equals 96.",
                usage_metadata={
                    "input_tokens": 75,
                    "output_tokens": 12,
                    "total_tokens": 87,
                },
            ),
        ]
    )

    agent = (
        AgentBuilder("OfflineMathAgent")
        .with_model(model)
        .with_tools([multiply])
        .with_system_prompt("You solve arithmetic accurately.")
        .compile()
    )

    collector = TraceCollector(
        price_table={"FakeModel": (0.005, 0.015)},
        out_dir=None,  # keep trace in memory for this demo
    )
    result = agent.invoke("What is 12 * 8?", config={"callbacks": [collector]})
    print("Final answer:", result["messages"][-1].content)
    print("Trace summary:", summarize(collector))


if __name__ == "__main__":
    main()
