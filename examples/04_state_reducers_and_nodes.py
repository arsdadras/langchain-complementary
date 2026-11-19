"""
Example 04: Declarative agents, automatic State reducers, and @node decorators.
Runs completely offline with zero API keys.
"""

from __future__ import annotations

from typing import Any

from langcomp import AgentBuilder, Reduce, State, node


def _keep_max(a: int, b: int) -> int:
    return max(a or 0, b or 0)


class AnalysisState(State):
    findings: list[str] = []
    tags: dict[str, str] = {}
    confidence: int = Reduce(_keep_max, default=0)


@node
def collect_metrics(state: dict[str, Any]) -> dict[str, Any]:
    del state
    return {
        "findings": ["Q3 revenue up 18%"],
        "tags": {"region": "EMEA"},
        "confidence": 85,
    }


@node
def verify_metrics(state: dict[str, Any]) -> dict[str, Any]:
    del state
    return {
        "findings": ["Churn decreased 2.1%"],
        "tags": {"audited": "true"},
        "confidence": 72,  # _keep_max preserves 85
    }


def main() -> None:
    print("State reducers:", AnalysisState.explain())

    agent = (
        AgentBuilder("MetricsPipeline")
        .with_state_schema(AnalysisState)
        .add_node(collect_metrics)
        .add_node(verify_metrics)
        .compile()
    )

    result = agent.invoke({})
    print("Findings:", result["findings"])
    print("Tags:", result["tags"])
    print("Max confidence:", result["confidence"])


if __name__ == "__main__":
    main()
