"""
Hierarchical subagent Team abstraction over DeepAgents.
Supports arbitrary nesting and uniform transitive inheritance across team trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class Team:
    """A subagent (or root agent) that may recursively own child subagents.

    Every optional field (``system_prompt``, ``tools``, ``model``, ``skills``,
    ``middleware``) inherits transitively from its parent Team unless explicitly
    overridden. Requires the optional ``deepagents`` package only when ``compile()``
    is called.
    """

    name: str
    description: str
    system_prompt: Optional[str] = None
    tools: Optional[List[Any]] = None
    model: Optional[Union[str, Any]] = None
    skills: Optional[List[str]] = None
    middleware: Optional[List[Any]] = None
    children: Sequence["Team"] = field(default_factory=tuple)

    def _resolve_from_context(self, inherited: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        parent_ctx = inherited or {}

        def own_or_inherited(own: Any, key: str) -> Any:
            return own if own is not None else parent_ctx.get(key)

        tools_val = own_or_inherited(self.tools, "tools")
        skills_val = own_or_inherited(self.skills, "skills")
        middleware_val = own_or_inherited(self.middleware, "middleware")

        return {
            "system_prompt": (
                own_or_inherited(self.system_prompt, "system_prompt")
                or "You are a helpful assistant."
            ),
            "tools": list(tools_val) if tools_val is not None else [],
            "model": own_or_inherited(self.model, "model"),
            "skills": list(skills_val) if skills_val is not None else None,
            "middleware": list(middleware_val) if middleware_val is not None else [],
        }

    def _resolve(self, parent: Optional["Team"]) -> Dict[str, Any]:
        inherited = None
        if parent is not None:
            inherited = {
                "system_prompt": parent.system_prompt,
                "tools": parent.tools,
                "model": parent.model,
                "skills": parent.skills,
                "middleware": parent.middleware,
            }
        return self._resolve_from_context(inherited)

    def _validate_children(self) -> None:
        """Raise ValueError on duplicate sibling names or 'general-purpose' shadowing."""
        seen: set[str] = set()
        for child in self.children:
            if child.name in seen:
                raise ValueError(
                    f"Team {self.name!r} has duplicate child name {child.name!r}; "
                    f"child names must be unique among siblings."
                )
            seen.add(child.name)
            if child.name == "general-purpose":
                raise ValueError(
                    f"Team {self.name!r} has a child named 'general-purpose', which "
                    f"silently shadows DeepAgents' built-in general-purpose subagent; "
                    f"rename this child."
                )

    def _as_subagent_entry(
        self,
        inherited: Optional[Dict[str, Any]],
        fallback_model: Any,
        compiled_subagent_cls: Any,
        create_deep_agent_fn: Any,
    ) -> Any:
        resolved = self._resolve_from_context(inherited)
        model = resolved["model"] or fallback_model

        if self.children:
            child_graph = self._compile_resolved(
                resolved,
                model=model,
                compiled_subagent_cls=compiled_subagent_cls,
                create_deep_agent_fn=create_deep_agent_fn,
            )
            return compiled_subagent_cls(
                name=self.name,
                description=self.description,
                runnable=child_graph,
            )

        entry: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "system_prompt": resolved["system_prompt"],
            "tools": resolved["tools"],
            "model": model,
            "middleware": resolved["middleware"],
        }
        if resolved["skills"] is not None:
            entry["skills"] = resolved["skills"]
        return entry

    def _compile_resolved(
        self,
        resolved: Dict[str, Any],
        *,
        model: Optional[Any],
        compiled_subagent_cls: Any,
        create_deep_agent_fn: Any,
        checkpointer: Any = None,
        **root_kwargs: Any,
    ) -> Any:
        self._validate_children()
        own_model = model if model is not None else self.model
        if own_model is None:
            raise ValueError(
                f"Team {self.name!r} requires a model (either on Team.model or passed to compile(model=...))."
            )
        subagents = [
            child._as_subagent_entry(
                inherited=resolved,
                fallback_model=own_model,
                compiled_subagent_cls=compiled_subagent_cls,
                create_deep_agent_fn=create_deep_agent_fn,
            )
            for child in self.children
        ]

        kwargs: Dict[str, Any] = {
            "model": own_model,
            "system_prompt": resolved["system_prompt"],
            "tools": resolved["tools"],
            "subagents": subagents,
            "middleware": resolved["middleware"],
            "checkpointer": checkpointer,
        }
        if resolved["skills"] is not None:
            kwargs["skills"] = resolved["skills"]
        kwargs.update(root_kwargs)

        return create_deep_agent_fn(**kwargs)

    def compile(
        self,
        *,
        model: Optional[Any] = None,
        checkpointer: Any = None,
        **root_kwargs: Any,
    ) -> Any:
        """Compile this Team into a runnable DeepAgents graph.

        Raises:
            ImportError: If the optional ``deepagents`` package is not installed.
        """
        try:
            from deepagents import CompiledSubAgent, create_deep_agent
        except ImportError as exc:
            raise ImportError(
                "deepagents is required to compile a Team into a deep agent graph. "
                "Install it with: pip install 'langcomp[deepagents]'"
            ) from exc

        resolved = self._resolve(parent=None)
        return self._compile_resolved(
            resolved,
            model=model,
            compiled_subagent_cls=CompiledSubAgent,
            create_deep_agent_fn=create_deep_agent,
            checkpointer=checkpointer,
            **root_kwargs,
        )
