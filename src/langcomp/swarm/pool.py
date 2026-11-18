"""
Swarm & Subagent Router for the langcomp framework.
Enables clean multi-agent orchestration, context isolation, and tool-based subagent delegation.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

from langcomp.core.agent import Agent


class SubAgent(BaseModel):
    """Subagent specification wrapping an :class:`Agent`, compiled runnable, or plain callable."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    capabilities: List[str] = Field(default_factory=list)
    agent_instance: Any

    @staticmethod
    def _extract_output(res: Any) -> str:
        if isinstance(res, dict):
            msgs = res.get("messages", [])
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
        return str(res)

    def run(self, query: str, thread_id: Optional[str] = None) -> str:
        """Execute the subagent with an isolated context query."""
        isolated_id = thread_id or f"subagent_{self.name}_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": isolated_id}}
        if isinstance(self.agent_instance, Agent):
            return self._extract_output(self.agent_instance.invoke(query, config=config))
        if hasattr(self.agent_instance, "invoke") and callable(self.agent_instance.invoke):
            res = self.agent_instance.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config=config,
            )
            return self._extract_output(res)
        if callable(self.agent_instance):
            return str(self.agent_instance(query))
        raise ValueError(
            f"Unsupported subagent instance type: {type(self.agent_instance)}"
        )

    def as_tool(self) -> BaseTool:
        """Convert this subagent into a LangChain tool for supervisor delegation."""
        sub_name = self.name
        sub_desc = self.description
        sub_run = self.run

        @tool(sub_name, description=f"Delegate task to subagent '{sub_name}': {sub_desc}")
        def subagent_tool(task_description: str) -> str:
            """Subagent tool wrapper."""
            return sub_run(task_description)

        return subagent_tool

    def to_deepagents_subagent_dict(self) -> Dict[str, Any]:
        """Convert this SubAgent into a dictionary compatible with DeepAgents SubAgentMiddleware."""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": f"You are subagent '{self.name}'. {self.description}",
            "runnable": (
                self.agent_instance.graph
                if isinstance(self.agent_instance, Agent)
                else self.agent_instance
            ),
        }


class SwarmPool:
    """Manager pool for multi-agent swarms."""

    def __init__(self, name: str = "DefaultSwarmPool"):
        self.name = name
        self.subagents: Dict[str, SubAgent] = {}

    def register(
        self,
        name: str,
        description: str,
        agent_or_callable: Any,
        capabilities: Optional[List[str]] = None,
    ) -> SubAgent:
        """Register a subagent in the swarm pool."""
        subagent = SubAgent(
            name=name,
            description=description,
            capabilities=capabilities or [],
            agent_instance=agent_or_callable,
        )
        self.subagents[name] = subagent
        return subagent

    def get_tools(self) -> List[BaseTool]:
        """Get all registered subagents converted as LangChain tools."""
        return [sub.as_tool() for sub in self.subagents.values()]

    def list_subagents(self) -> List[Dict[str, Any]]:
        """Return metadata list of registered subagents."""
        return [
            {
                "name": sub.name,
                "description": sub.description,
                "capabilities": sub.capabilities,
            }
            for sub in self.subagents.values()
        ]
