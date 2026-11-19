"""
Tool Guardrails & Human Approval Gates for the langcomp framework.
Provides non-mutating, Pydantic v2 safe sync/async execution wrappers and fail-closed approval gates.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphBubbleUp

from langcomp.errors import ApprovalRejectedError, ToolExecutionError

logger = logging.getLogger("langcomp.tools")


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or (
        not isinstance(fn, type) and callable(fn) and inspect.iscoroutinefunction(type(fn).__call__)
    )


def _is_framework_injected_param(name: str, value: Any) -> bool:
    if name == "run_manager":
        return value is None or isinstance(
            value, (CallbackManagerForToolRun, AsyncCallbackManagerForToolRun)
        )
    if name == "config":
        return (
            isinstance(value, dict)
            and bool(value)
            and set(value.keys()).issubset(
                {"tags", "metadata", "callbacks", "run_name", "max_concurrency", "recursion_limit", "configurable", "run_id"}
            )
            and any(k in value for k in ("configurable", "callbacks", "run_id", "recursion_limit"))
        )
    return False


class ToolGuard:
    """Resilient wrapper for sync and async tools with retries and fallback outputs."""

    def __init__(
        self,
        max_retries: int = 3,
        fallback_output: Any = None,
        retry_delay_seconds: float = 0.5,
    ):
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if retry_delay_seconds < 0:
            raise ValueError(
                f"retry_delay_seconds must be >= 0, got {retry_delay_seconds}"
            )
        self.max_retries = max_retries
        self.fallback_output = fallback_output
        self.retry_delay_seconds = retry_delay_seconds

    def _log_attempt_failure(self, name: str, attempt: int, exc: Exception) -> None:
        logger.warning(
            "Tool '%s' failed attempt %d/%d (%s)",
            name,
            attempt,
            self.max_retries,
            type(exc).__name__,
        )

    def _handle_exhausted_retries(self, name: str, last_err: Exception | None) -> Any:
        if self.fallback_output is not None:
            return self.fallback_output
        raise ToolExecutionError(
            f"Tool '{name}' failed after {self.max_retries} attempts: {last_err}"
        ) from last_err

    def wrap_function(
        self,
        func: Callable[..., Any],
        tool_name: str | None = None,
    ) -> Callable[..., Any]:
        """Wrap a sync or async Python callable with retry and fallback guardrails."""
        name = str(tool_name or getattr(func, "__name__", "ToolFunction"))

        if _is_async_callable(func):

            @functools.wraps(func)
            async def guarded_async(*args: Any, **kwargs: Any) -> Any:
                last_err: Exception | None = None
                for attempt in range(1, self.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except (GraphBubbleUp, ApprovalRejectedError):
                        raise
                    except Exception as exc:  # noqa: BLE001 - retry policy catches runtime tool errors
                        last_err = exc
                        self._log_attempt_failure(name, attempt, exc)
                        if attempt < self.max_retries and self.retry_delay_seconds > 0:
                            await asyncio.sleep(self.retry_delay_seconds)

                return self._handle_exhausted_retries(name, last_err)

            return guarded_async

        @functools.wraps(func)
        def guarded_sync(*args: Any, **kwargs: Any) -> Any:
            last_err: Exception | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (GraphBubbleUp, ApprovalRejectedError):
                    raise
                except Exception as exc:  # noqa: BLE001 - retry policy catches runtime tool errors
                    last_err = exc
                    self._log_attempt_failure(name, attempt, exc)
                    if attempt < self.max_retries and self.retry_delay_seconds > 0:
                        time.sleep(self.retry_delay_seconds)

            return self._handle_exhausted_retries(name, last_err)

        return guarded_sync

    def wrap_tool(self, tool_inst: BaseTool) -> BaseTool:
        """Return a non-mutating, Pydantic v2 safe wrapped clone of a LangChain ``BaseTool``.

        For ``StructuredTool``, wraps ``func`` / ``coroutine`` directly (avoiding ``arun()``
        double-wrapping and closing direct ``.func`` bypasses). For custom ``BaseTool``
        subclasses, creates a lightweight wrapper subclass clone that preserves ``model_copy()``.
        """
        cloned = tool_inst.model_copy(deep=False)
        if isinstance(cloned, StructuredTool):
            if cloned.func is not None:
                object.__setattr__(
                    cloned, "func", self.wrap_function(cloned.func, tool_name=cloned.name)
                )
            if cloned.coroutine is not None:
                object.__setattr__(
                    cloned,
                    "coroutine",
                    self.wrap_function(cloned.coroutine, tool_name=cloned.name),
                )
            return cloned

        guard = self
        base_cls = type(tool_inst)
        has_custom_arun = base_cls._arun is not BaseTool._arun

        class _GuardedBaseTool(base_cls):  # type: ignore[valid-type, misc]
            _langcomp_guard: ClassVar[ToolGuard] = guard

            def _run(self, *args: Any, **kwargs: Any) -> Any:
                def _call(*a: Any, **kw: Any) -> Any:
                    return base_cls._run(self, *a, **kw)

                return self._langcomp_guard.wrap_function(_call, tool_name=self.name)(
                    *args, **kwargs
                )

            if has_custom_arun:

                async def _arun(self, *args: Any, **kwargs: Any) -> Any:
                    async def _acall(*a: Any, **kw: Any) -> Any:
                        res = base_cls._arun(self, *a, **kw)
                        return await res if inspect.isawaitable(res) else res

                    return await self._langcomp_guard.wrap_function(_acall, tool_name=self.name)(
                        *args, **kwargs
                    )

        _GuardedBaseTool.__name__ = f"Guarded{base_cls.__name__}"
        cloned.__class__ = _GuardedBaseTool
        return cloned


class ApprovalGate:
    """Fail-closed Human-In-The-Loop gate for verifying sensitive tool actions before execution."""

    def __init__(
        self,
        approval_func: Callable[[str, dict[str, Any]], bool | Awaitable[bool]] | None = None,
    ):
        self.approval_func = approval_func

    def check_approval(self, action_name: str, parameters: dict[str, Any]) -> bool:
        """Synchronously determine whether an action is approved (fails closed if unset)."""
        if self.approval_func is None:
            return False
        if _is_async_callable(self.approval_func):
            raise TypeError(
                "Async approval_func cannot be called in sync check_approval(); "
                "use await gate.acheck_approval(...)."
            )
        result = self.approval_func(action_name, parameters)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError(
                "Async approval_func returned an awaitable in sync check_approval(); "
                "use await gate.acheck_approval(...)."
            )
        return bool(result)

    async def acheck_approval(self, action_name: str, parameters: dict[str, Any]) -> bool:
        """Asynchronously determine whether an action is approved (fails closed if unset)."""
        if self.approval_func is None:
            return False
        result = self.approval_func(action_name, parameters)
        if inspect.isawaitable(result):
            return bool(await result)
        return bool(result)

    @staticmethod
    def _extract_params(
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        strip_framework_args: bool = False,
    ) -> dict[str, Any]:
        try:
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            params: dict[str, Any] = {}
            for param_name, param_value in bound.arguments.items():
                param = sig.parameters.get(param_name)
                if param is not None and param.kind == inspect.Parameter.VAR_KEYWORD:
                    if isinstance(param_value, dict):
                        params.update(param_value)
                else:
                    params[param_name] = param_value
        except (TypeError, ValueError):
            params = dict(kwargs)

        if strip_framework_args:
            return {
                k: v for k, v in params.items() if not _is_framework_injected_param(k, v)
            }
        return params

    def wrap_function(
        self,
        func: Callable[..., Any],
        action_name: str | None = None,
        *,
        _strip_framework_args: bool = False,
    ) -> Callable[..., Any]:
        """Wrap a sync or async callable with an approval check before execution."""
        name = str(action_name or getattr(func, "__name__", "Action"))

        if _is_async_callable(func):

            @functools.wraps(func)
            async def gated_async(*args: Any, **kwargs: Any) -> Any:
                params = self._extract_params(
                    func, args, kwargs, strip_framework_args=_strip_framework_args
                )
                if not await self.acheck_approval(name, params):
                    raise ApprovalRejectedError(f"Action '{name}' was not approved.")
                return await func(*args, **kwargs)

            return gated_async

        @functools.wraps(func)
        def gated_sync(*args: Any, **kwargs: Any) -> Any:
            params = self._extract_params(
                func, args, kwargs, strip_framework_args=_strip_framework_args
            )
            if not self.check_approval(name, params):
                raise ApprovalRejectedError(f"Action '{name}' was not approved.")
            return func(*args, **kwargs)

        return gated_sync

    @classmethod
    def _extract_tool_params(
        cls,
        func: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        params = cls._extract_params(
            func, (instance, *args), kwargs, strip_framework_args=True
        )
        params.pop("self", None)
        return params

    def wrap_tool(
        self,
        tool_inst: BaseTool,
        action_name: str | None = None,
    ) -> BaseTool:
        """Return a non-mutating wrapped clone of a LangChain ``BaseTool`` requiring approval."""
        name = action_name or tool_inst.name
        cloned = tool_inst.model_copy(deep=False)

        if isinstance(cloned, StructuredTool):
            if cloned.func is not None:
                object.__setattr__(
                    cloned,
                    "func",
                    self.wrap_function(cloned.func, action_name=name, _strip_framework_args=True),
                )
            if cloned.coroutine is not None:
                object.__setattr__(
                    cloned,
                    "coroutine",
                    self.wrap_function(
                        cloned.coroutine, action_name=name, _strip_framework_args=True
                    ),
                )
            return cloned

        gate = self
        base_cls = type(tool_inst)
        has_custom_arun = base_cls._arun is not BaseTool._arun

        class _GatedBaseTool(base_cls):  # type: ignore[valid-type, misc]
            _langcomp_gate: ClassVar[ApprovalGate] = gate
            _langcomp_action_name: ClassVar[str] = name

            def _run(self, *args: Any, **kwargs: Any) -> Any:
                params = self._langcomp_gate._extract_tool_params(
                    base_cls._run, self, args, kwargs
                )
                if not self._langcomp_gate.check_approval(self._langcomp_action_name, params):
                    raise ApprovalRejectedError(
                        f"Action '{self._langcomp_action_name}' was not approved."
                    )
                return base_cls._run(self, *args, **kwargs)

            if has_custom_arun:

                async def _arun(self, *args: Any, **kwargs: Any) -> Any:
                    params = self._langcomp_gate._extract_tool_params(
                        base_cls._arun, self, args, kwargs
                    )
                    if not await self._langcomp_gate.acheck_approval(
                        self._langcomp_action_name, params
                    ):
                        raise ApprovalRejectedError(
                            f"Action '{self._langcomp_action_name}' was not approved."
                        )
                    res = base_cls._arun(self, *args, **kwargs)
                    return await res if inspect.isawaitable(res) else res

        _GatedBaseTool.__name__ = f"Gated{base_cls.__name__}"
        cloned.__class__ = _GatedBaseTool
        return cloned
