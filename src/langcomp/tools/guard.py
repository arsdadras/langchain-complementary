"""
Tool Guardrails & Human Approval Gates for the langcomp framework.
Provides resilient sync/async execution wrappers and approval enforcement for tools.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, Optional

from langchain_core.tools import BaseTool

from langcomp.errors import ApprovalRejectedError

logger = logging.getLogger("langcomp.tools")


class ToolGuard:
    """Resilient wrapper for sync and async tools with retries and fallback outputs."""

    def __init__(
        self,
        max_retries: int = 3,
        fallback_output: Optional[Any] = None,
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

    def wrap_function(
        self,
        func: Callable[..., Any],
        tool_name: Optional[str] = None,
    ) -> Callable[..., Any]:
        """Wrap a sync or async Python callable with retry and fallback guardrails."""
        name = tool_name or getattr(func, "__name__", "ToolFunction")

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def guarded_async(*args: Any, **kwargs: Any) -> Any:
                last_err: Optional[Exception] = None
                for attempt in range(1, self.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except ApprovalRejectedError:
                        raise
                    except Exception as exc:
                        last_err = exc
                        logger.warning(
                            "Tool '%s' failed attempt %d/%d: %s",
                            name,
                            attempt,
                            self.max_retries,
                            exc,
                        )
                        if attempt < self.max_retries and self.retry_delay_seconds > 0:
                            await asyncio.sleep(self.retry_delay_seconds)

                if self.fallback_output is not None:
                    return self.fallback_output
                raise RuntimeError(
                    f"Tool '{name}' failed after {self.max_retries} attempts: {last_err}"
                ) from last_err

            return guarded_async

        @functools.wraps(func)
        def guarded_sync(*args: Any, **kwargs: Any) -> Any:
            last_err: Optional[Exception] = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ApprovalRejectedError:
                    raise
                except Exception as exc:
                    last_err = exc
                    logger.warning(
                        "Tool '%s' failed attempt %d/%d: %s",
                        name,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    if attempt < self.max_retries and self.retry_delay_seconds > 0:
                        time.sleep(self.retry_delay_seconds)

            if self.fallback_output is not None:
                return self.fallback_output
            raise RuntimeError(
                f"Tool '{name}' failed after {self.max_retries} attempts: {last_err}"
            ) from last_err

        return guarded_sync

    def wrap_tool(self, tool_inst: BaseTool) -> BaseTool:
        """Wrap a LangChain ``BaseTool`` (Pydantic v2 safe) with retry/fallback guardrails.

        Only wraps ``_arun`` when the tool subclass overrides ``BaseTool._arun``, avoiding
        double-wrapping when ``BaseTool._arun`` delegates to ``self._run``.
        """
        original_run = tool_inst._run
        guarded_run = self.wrap_function(original_run, tool_name=tool_inst.name)
        object.__setattr__(tool_inst, "_run", guarded_run)

        if (
            type(tool_inst)._arun is not BaseTool._arun
            and inspect.iscoroutinefunction(tool_inst._arun)
        ):
            guarded_arun = self.wrap_function(tool_inst._arun, tool_name=tool_inst.name)
            object.__setattr__(tool_inst, "_arun", guarded_arun)

        return tool_inst


class ApprovalGate:
    """Human-In-The-Loop gate for verifying sensitive tool actions before execution."""

    def __init__(
        self,
        approval_func: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self.approval_func = approval_func

    def check_approval(self, action_name: str, parameters: Dict[str, Any]) -> bool:
        """Determine whether an action is approved."""
        if self.approval_func is not None:
            return bool(self.approval_func(action_name, parameters))
        return True

    @staticmethod
    def _extract_params(
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            params: Dict[str, Any] = {}
            for param_name, param_value in bound.arguments.items():
                param = sig.parameters.get(param_name)
                if param is not None and param.kind == inspect.Parameter.VAR_KEYWORD:
                    if isinstance(param_value, dict):
                        params.update(param_value)
                else:
                    params[param_name] = param_value
            params.pop("config", None)
            params.pop("run_manager", None)
            return params
        except (TypeError, ValueError):
            filtered = dict(kwargs)
            filtered.pop("config", None)
            filtered.pop("run_manager", None)
            return filtered

    def wrap_function(
        self,
        func: Callable[..., Any],
        action_name: Optional[str] = None,
    ) -> Callable[..., Any]:
        """Wrap a sync or async callable with an approval check before execution."""
        name = action_name or getattr(func, "__name__", "Action")

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def gated_async(*args: Any, **kwargs: Any) -> Any:
                params = self._extract_params(func, args, kwargs)
                if not self.check_approval(name, params):
                    raise ApprovalRejectedError(f"Action '{name}' was not approved.")
                return await func(*args, **kwargs)

            return gated_async

        @functools.wraps(func)
        def gated_sync(*args: Any, **kwargs: Any) -> Any:
            params = self._extract_params(func, args, kwargs)
            if not self.check_approval(name, params):
                raise ApprovalRejectedError(f"Action '{name}' was not approved.")
            return func(*args, **kwargs)

        return gated_sync

    def wrap_tool(
        self,
        tool_inst: BaseTool,
        action_name: Optional[str] = None,
    ) -> BaseTool:
        """Wrap a LangChain ``BaseTool`` so execution requires approval."""
        name = action_name or tool_inst.name
        gated_run = self.wrap_function(tool_inst._run, action_name=name)
        object.__setattr__(tool_inst, "_run", gated_run)

        if (
            type(tool_inst)._arun is not BaseTool._arun
            and inspect.iscoroutinefunction(tool_inst._arun)
        ):
            gated_arun = self.wrap_function(tool_inst._arun, action_name=name)
            object.__setattr__(tool_inst, "_arun", gated_arun)

        return tool_inst
