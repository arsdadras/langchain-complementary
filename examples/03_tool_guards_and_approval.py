"""
Example 03: Tool Guardrails & Human Approval Gates.
Demonstrates non-mutating ToolGuard retries and fail-closed ApprovalGate wrapping.
Runs completely offline with zero API keys.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from langcomp import ApprovalGate, ApprovalRejectedError, ToolGuard


def main() -> None:
    print("=== langcomp Example 03: Tool Guards & Approval Gates ===")

    # 1. ToolGuard Demonstration
    print("\n--- 1. Guarding Flaky Tools with Auto-Retries ---")
    attempts = 0

    def unstable_external_api(endpoint: str) -> str:
        nonlocal attempts
        attempts += 1
        print(f"  [API Call Attempt {attempts}] Calling endpoint: {endpoint}")
        if attempts < 3:
            raise ConnectionError("Network timeout connecting to external server.")
        return f"Successfully fetched data from {endpoint}"

    guard = ToolGuard(max_retries=3, retry_delay_seconds=0.0)
    guarded_api = guard.wrap_function(unstable_external_api)

    result = guarded_api("https://api.example.com/v1/data")
    print(f"Guarded Call Result: {result}")

    # 2. Fail-Closed ApprovalGate with wrap_tool()
    print("\n--- 2. Human-In-The-Loop Approval Gate ---")

    def user_approval_callback(action_name: str, params: dict[str, Any]) -> bool:
        print(f"  [APPROVAL REQUEST] Action: '{action_name}', Params: {params}")
        if "test" in str(params.get("database_name", "")):
            print("  [DECISION] APPROVED (Test database)")
            return True
        print("  [DECISION] DENIED (Production database protected!)")
        return False

    @tool
    def drop_database(database_name: str) -> str:
        """Drop a target database by name."""
        return f"Database '{database_name}' dropped successfully."

    gate = ApprovalGate(approval_func=user_approval_callback)
    gated_drop = gate.wrap_tool(drop_database)

    print(f"Action Execution Output: {gated_drop.invoke({'database_name': 'test_db_v1'})}")

    try:
        gated_drop.invoke({"database_name": "prod_users_primary"})
    except ApprovalRejectedError as exc:
        print(f"Action blocked by Approval Gate: {exc}")


if __name__ == "__main__":
    main()
