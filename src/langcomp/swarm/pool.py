"""
Swarm & Subagent Router for the langcomp framework.
Enables clean sync/async multi-agent orchestration, context isolation, and tool-based delegation.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import uuid
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from langcomp.core.agent import Agent
from langcomp.errors import ConfigurationError

_VALID_SUBAGENT_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _is_async_callable(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn) or (
        not isinstance(fn, type) and callable(fn) and inspect.iscoroutinefunction(type(fn).__call__)
    )


def _merge_subagent_config(
    name: str,
    thread_id: str | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(config or {})
    configurable = dict(merged.get("configurable") or {})
    configurable["thread_id"] = thread_id or f"subagent_{name}_{uuid.uuid4().hex}"
    merged["configurable"] = configurable
    return merged


class SubAgent(BaseModel):
    """Subagent specification wrapping an :class:`Agent`, compiled runnable, or plain callable."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    agent_instance: Any

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not _VALID_SUBAGENT_NAME.fullmatch(cleaned):
            raise ValueError(
                f"Invalid subagent name {value!r}: must match ^[A-Za-z0-9_-]{{1,64}}$."
            )
        return cleaned

    @staticmethod
    def _extract_output(res: Any) -> str:
        if isinstance(res, dict):
            if "messages" in res:
                msgs = res.get("messages") or []
                if msgs:
                    last_msg = msgs[-1]
                    if isinstance(last_msg, BaseMessage):
                        return (
                            last_msg.content
                            if isinstance(last_msg.content, str)
                            else str(last_msg.content)
                        )
                    return str(last_msg)
                return "Subagent finished with no output messages."
            return json.dumps(res, default=str, sort_keys=True)
        return str(res)

    def run(
        self,
        query: str,
        thread_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> str:
        """Synchronously execute the subagent with an isolated context query."""
        run_config = _merge_subagent_config(self.name, thread_id, config)
        if isinstance(self.agent_instance, Agent):
            return self._extract_output(self.agent_instance.invoke(query, config=run_config))
        if hasattr(self.agent_instance, "invoke") and callable(self.agent_instance.invoke):
            res = self.agent_instance.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config=run_config,
            )
            return self._extract_output(res)
        if callable(self.agent_instance):
            if _is_async_callable(self.agent_instance):
                raise TypeError(
                    f"SubAgent {self.name!r} wraps an async callable; use await subagent.arun(...)."
                )
            out = self.agent_instance(query)
            if inspect.isawaitable(out):
                if inspect.iscoroutine(out):
                    out.close()
                raise TypeError(
                    f"SubAgent {self.name!r} returned an awaitable; use await subagent.arun(...)."
                )
            return str(out)
        raise ValueError(
            f"Unsupported subagent instance type: {type(self.agent_instance)}"
        )

    async def arun(
        self,
        query: str,
        thread_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> str:
        """Asynchronously execute the subagent with an isolated context query."""
        run_config = _merge_subagent_config(self.name, thread_id, config)
        if isinstance(self.agent_instance, Agent):
            return self._extract_output(
                await self.agent_instance.ainvoke(query, config=run_config)
            )
        if hasattr(self.agent_instance, "ainvoke") and callable(self.agent_instance.ainvoke):
            res = await self.agent_instance.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config=run_config,
            )
            return self._extract_output(res)
        if hasattr(self.agent_instance, "invoke") and callable(self.agent_instance.invoke):
            res = self.agent_instance.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config=run_config,
            )
            return self._extract_output(res)
        if callable(self.agent_instance):
            out = self.agent_instance(query)
            if inspect.isawaitable(out):
                out = await out
            return str(out)
        raise ValueError(
            f"Unsupported subagent instance type: {type(self.agent_instance)}"
        )

    def as_tool(self) -> BaseTool:
        """Convert this subagent into a sync + async LangChain tool for supervisor delegation."""
        sub_name = self.name
        sub_desc = self.description
        sub_run = self.run
        sub_arun = self.arun

        def _sync_delegate(task_description: str) -> str:
            return sub_run(task_description)

        async def _async_delegate(task_description: str) -> str:
            return await sub_arun(task_description)

        return StructuredTool.from_function(
            func=_sync_delegate,
            coroutine=_async_delegate,
            name=sub_name,
            description=f"Delegate task to subagent '{sub_name}': {sub_desc}",
        )

    def to_deepagents_subagent_dict(self) -> dict[str, Any]:
        """Convert this SubAgent into a dictionary compatible with DeepAgents.

        Returns a ``CompiledSubAgent``-shaped dict (``name``, ``description``, ``runnable``)
        when wrapping a compiled runnable/:class:`Agent`, or a declarative dict
        (``name``, ``description``, ``system_prompt``) when wrapping a plain callable.
        """
        if isinstance(self.agent_instance, Agent):
            return {
                "name": self.name,
                "description": self.description,
                "runnable": self.agent_instance.graph,
            }
        if hasattr(self.agent_instance, "invoke") and callable(self.agent_instance.invoke):
            return {
                "name": self.name,
                "description": self.description,
                "runnable": self.agent_instance,
            }
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": f"You are subagent '{self.name}'. {self.description}",
        }


class SwarmPool:
    """Thread-safe manager pool for multi-agent swarms."""

    def __init__(self, name: str = "DefaultSwarmPool"):
        self.name = name
        self.subagents: dict[str, SubAgent] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        description: str,
        agent_or_callable: Any,
        capabilities: list[str] | None = None,
        *,
        overwrite: bool = False,
    ) -> SubAgent:
        """Register a subagent in the swarm pool."""
        subagent = SubAgent(
            name=name,
            description=description,
            capabilities=list(capabilities) if capabilities else [],
            agent_instance=agent_or_callable,
        )
        with self._lock:
            if subagent.name in self.subagents and not overwrite:
                raise ConfigurationError(
                    f"Subagent {subagent.name!r} is already registered in SwarmPool {self.name!r}. "
                    f"Pass overwrite=True to replace it."
                )
            self.subagents[subagent.name] = subagent
        return subagent

    def get(self, name: str) -> SubAgent | None:
        """Retrieve a registered subagent by name."""
        with self._lock:
            return self.subagents.get(name)

    def unregister(self, name: str) -> SubAgent | None:
        """Remove and return a registered subagent by name if present."""
        with self._lock:
            return self.subagents.pop(name, None)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self.subagents

    def __len__(self) -> int:
        with self._lock:
            return len(self.subagents)

    def get_tools(self) -> list[BaseTool]:
        """Get all registered subagents converted as LangChain tools."""
        with self._lock:
            return [sub.as_tool() for sub in self.subagents.values()]

    def list_subagents(self) -> list[dict[str, Any]]:
        """Return defensive metadata list of registered subagents."""
        with self._lock:
            return [
                {
                    "name": sub.name,
                    "description": sub.description,
                    "capabilities": list(sub.capabilities),
                }
                for sub in self.subagents.values()
            ]
