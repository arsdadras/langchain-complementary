"""Unit tests for langcomp.tools.guard (ToolGuard & ApprovalGate, sync + async)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool, tool

from langcomp import ApprovalGate, ApprovalRejectedError, ToolExecutionError, ToolGuard


class _FlakyCustomTool(BaseTool):
    name: str = "flaky_custom"
    description: str = "Fails every call to verify retry count and non-mutating cloning."
    tag: str = "orig"
    attempts: int = 0

    def _run(self, query: str) -> str:
        object.__setattr__(self, "attempts", self.attempts + 1)
        raise ValueError(f"boom:{self.tag}:{query}")


class _TagTool(BaseTool):
    name: str = "tag_tool"
    description: str = "Returns its own tag."
    tag: str = "orig"

    def _run(self, x: str) -> str:
        return f"{self.tag}:{x}"


def test_tool_guard_validation_metadata_retry_and_fallback() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        ToolGuard(max_retries=0)
    with pytest.raises(ValueError, match="retry_delay_seconds"):
        ToolGuard(retry_delay_seconds=-1.0)

    attempts = 0

    def flaky_fn(x: int) -> int:
        """Double x."""
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Temporary glitch")
        return x * 2

    guard = ToolGuard(max_retries=3, retry_delay_seconds=0.0)
    guarded_fn = guard.wrap_function(flaky_fn)
    assert guarded_fn.__name__ == "flaky_fn"
    assert guarded_fn.__doc__ == "Double x."
    assert guarded_fn(5) == 10
    assert attempts == 3

    def always_fails() -> str:
        raise RuntimeError("Fatal error")

    fallback_fn = ToolGuard(
        max_retries=2, fallback_output="Fallback Result", retry_delay_seconds=0.0
    ).wrap_function(always_fails)
    assert fallback_fn() == "Fallback Result"

    raising_fn = ToolGuard(max_retries=2, retry_delay_seconds=0.0).wrap_function(always_fails)
    with pytest.raises(ToolExecutionError, match="failed after 2 attempts"):
        raising_fn()


async def test_tool_guard_async_function_retry_fallback_and_exhaustion() -> None:
    attempts = 0

    async def flaky_async(val: int) -> int:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("temp")
        return val + 1

    guard = ToolGuard(max_retries=3, retry_delay_seconds=0.0)
    wrapped = guard.wrap_function(flaky_async)
    assert await wrapped(10) == 11
    assert attempts == 2

    async def broken_async() -> str:
        raise ValueError("always broken")

    with pytest.raises(ToolExecutionError, match="failed after 2 attempts"):
        await ToolGuard(max_retries=2, retry_delay_seconds=0.0).wrap_function(broken_async)()

    assert (
        await ToolGuard(
            max_retries=2, fallback_output="safe-default", retry_delay_seconds=0.0
        ).wrap_function(broken_async)()
        == "safe-default"
    )


async def test_tool_guard_wrap_tool_non_mutating_no_double_wrap_and_model_copy_safe() -> None:
    calls = 0

    @tool
    def flaky_sync(x: str) -> str:
        """Flaky sync tool."""
        nonlocal calls
        calls += 1
        raise ValueError("boom")

    guard = ToolGuard(max_retries=3, retry_delay_seconds=0.0)
    guarded = guard.wrap_tool(flaky_sync)
    assert guarded is not flaky_sync

    # Sync @tool under arun() MUST execute 3 times (never 3*3 = 9!)
    with pytest.raises(ToolExecutionError, match="failed after 3 attempts"):
        await guarded.arun({"x": "hi"})
    assert calls == 3

    # Custom BaseTool subclass: non-mutating clone + preserves model_copy(update=...)
    custom = _FlakyCustomTool()
    guarded_custom = guard.wrap_tool(custom)
    assert guarded_custom is not custom
    with pytest.raises(ToolExecutionError, match="failed after 3 attempts"):
        await guarded_custom.ainvoke({"query": "test"})
    assert custom.attempts == 0
    assert guarded_custom.attempts == 3

    tag_orig = _TagTool(tag="orig")
    tag_wrapped = guard.wrap_tool(tag_orig)
    tag_cloned = tag_wrapped.model_copy(update={"tag": "CLONED"})
    assert tag_wrapped.invoke({"x": "1"}) == "orig:1"
    assert tag_cloned.invoke({"x": "1"}) == "CLONED:1"


def test_tool_guard_propagates_approval_rejected_error_without_retrying() -> None:
    calls = 0

    def sensitive(x: int) -> int:
        nonlocal calls
        calls += 1
        return x

    gate = ApprovalGate(approval_func=lambda _action, _params: False)
    guard = ToolGuard(max_retries=3, fallback_output="fallback", retry_delay_seconds=0.0)
    wrapped = guard.wrap_function(gate.wrap_function(sensitive))

    with pytest.raises(ApprovalRejectedError):
        wrapped(1)
    assert calls == 0


async def test_approval_gate_fail_closed_async_and_param_extraction() -> None:
    # Default fail-closed
    assert ApprovalGate().check_approval("drop_db", {}) is False
    assert await ApprovalGate().acheck_approval("drop_db", {}) is False

    # Sync gate rejects async approval_func instead of failing open (bool(coro) == True)
    async def async_policy(action: str, params: dict[str, Any]) -> bool:
        return bool(params.get("safe", False))

    async_gate = ApprovalGate(approval_func=async_policy)
    with pytest.raises(TypeError, match="acheck_approval"):
        async_gate.check_approval("drop_db", {"safe": False})
    assert await async_gate.acheck_approval("drop_db", {"safe": False}) is False
    assert await async_gate.acheck_approval("drop_db", {"safe": True}) is True

    # Legitimate user 'config' parameter is preserved (not blindly stripped!)
    seen_params: list[dict[str, Any]] = []

    def capture_gate(action: str, params: dict[str, Any]) -> bool:
        seen_params.append(dict(params))
        return True

    def deploy(config: str, env: str) -> str:
        return f"{env}:{config}"

    gated_deploy = ApprovalGate(approval_func=capture_gate).wrap_function(deploy)
    assert gated_deploy(config="ALLOW_ALL", env="prod") == "prod:ALLOW_ALL"
    assert seen_params == [{"config": "ALLOW_ALL", "env": "prod"}]


async def test_approval_gate_wrap_tool_sync_and_async_non_mutating() -> None:
    approval_calls = 0

    def policy(action: str, params: dict[str, Any]) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return params.get("safe") is True

    @tool
    def delete_record(record_id: int, safe: bool = False) -> str:
        """Delete a record by ID."""
        return f"deleted:{record_id}"

    gate = ApprovalGate(approval_func=policy)
    gated = gate.wrap_tool(delete_record)
    assert gated is not delete_record

    with pytest.raises(ApprovalRejectedError):
        gated.invoke({"record_id": 1, "safe": False})
    assert gated.invoke({"record_id": 1, "safe": True}) == "deleted:1"

    # arun() on sync @tool checks approval once (never twice)
    approval_calls = 0
    assert await gated.arun({"record_id": 2, "safe": True}) == "deleted:2"
    assert approval_calls == 1

    # Async-only StructuredTool with async approval policy
    async def async_del(record_id: int, safe: bool = False) -> str:
        return f"async-deleted:{record_id}"

    async_tool = StructuredTool.from_function(
        coroutine=async_del, name="async_del", description="Async delete."
    )

    async def async_deny(_a: str, params: dict[str, Any]) -> bool:
        return bool(params.get("safe"))

    gated_async = ApprovalGate(approval_func=async_deny).wrap_tool(async_tool)
    with pytest.raises(ApprovalRejectedError):
        await gated_async.ainvoke({"record_id": 7, "safe": False})
    assert await gated_async.ainvoke({"record_id": 7, "safe": True}) == "async-deleted:7"
