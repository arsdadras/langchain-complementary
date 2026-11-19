"""
Example 05: Deterministic offline testing with FakeModel, TraceCollector, and snapshots.
Runs completely offline with zero API keys.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langcomp import AgentBuilder, FakeModel, TraceCollector, run_and_snapshot, summarize


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def _build_math_agent() -> tuple[FakeModel, object]:
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
    return model, agent


def main() -> None:
    model, agent = _build_math_agent()
    collector = TraceCollector(
        price_table={"FakeModel": (0.005, 0.015)},
        out_dir=None,  # keep trace in memory for this demo
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        snap_path = Path(tmp_dir) / "math_agent_snapshot.json"
        # First run records golden-master snapshot; second run verifies zero drift
        result = run_and_snapshot(
            agent,
            "What is 12 * 8?",
            snap_path,
            config={"callbacks": [collector]},
        )
        model.reset()
        run_and_snapshot(agent, "What is 12 * 8?", snap_path)

    print("Final answer:", result["messages"][-1].content)
    print("Trace summary:", summarize(collector))


if __name__ == "__main__":
    main()
