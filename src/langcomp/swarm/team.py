"""
Hierarchical subagent Team abstraction over DeepAgents.
Supports arbitrary nesting, cycle protection, and uniform transitive inheritance across team trees.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langcomp.errors import ConfigurationError

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_MAX_TEAM_DEPTH = 32


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
    system_prompt: str | None = None
    tools: list[Any] | None = None
    model: str | Any | None = None
    skills: list[str] | None = None
    middleware: list[Any] | None = None
    children: Sequence[Team] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Team.name must not be empty.")

    def _resolve_from_context(self, inherited: dict[str, Any] | None) -> dict[str, Any]:
        parent_ctx = inherited or {}

        def own_or_inherited(own: Any, key: str) -> Any:
            return own if own is not None else parent_ctx.get(key)

        prompt_val = own_or_inherited(self.system_prompt, "system_prompt")
        tools_val = own_or_inherited(self.tools, "tools")
        skills_val = own_or_inherited(self.skills, "skills")
        middleware_val = own_or_inherited(self.middleware, "middleware")

        return {
            "system_prompt": (
                prompt_val if prompt_val is not None else _DEFAULT_SYSTEM_PROMPT
            ),
            "tools": list(tools_val) if tools_val is not None else [],
            "model": own_or_inherited(self.model, "model"),
            "skills": list(skills_val) if skills_val is not None else None,
            "middleware": list(middleware_val) if middleware_val is not None else [],
        }

    def _resolve(self, parent: Team | None = None) -> dict[str, Any]:
        """Resolve this Team's effective configuration (optionally inheriting from ``parent``)."""
        inherited = parent._resolve(None) if parent is not None else None
        return self._resolve_from_context(inherited)

    def _validate_children(self) -> None:
        """Raise ValueError on duplicate sibling names or 'general-purpose' shadowing."""
        seen: set[str] = set()
        for child in self.children:
            if not child.name or not child.name.strip():
                raise ValueError(f"Team {self.name!r} has a child with an empty name.")
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
        inherited: dict[str, Any] | None,
        fallback_model: Any,
        compiled_subagent_cls: Any,
        create_deep_agent_fn: Any,
        *,
        _visited: frozenset[int],
        _depth: int,
    ) -> Any:
        if id(self) in _visited:
            raise ConfigurationError(
                f"Cycle detected in Team hierarchy involving {self.name!r}."
            )
        if _depth > _MAX_TEAM_DEPTH:
            raise ConfigurationError(
                f"Team hierarchy exceeds maximum nesting depth of {_MAX_TEAM_DEPTH}."
            )

        resolved = self._resolve_from_context(inherited)
        effective_model = resolved["model"] if resolved["model"] is not None else fallback_model
        resolved["model"] = effective_model

        if self.children:
            child_graph = self._compile_resolved(
                resolved,
                model=effective_model,
                compiled_subagent_cls=compiled_subagent_cls,
                create_deep_agent_fn=create_deep_agent_fn,
                _visited=_visited,
                _depth=_depth,
            )
            return compiled_subagent_cls(
                name=self.name,
                description=self.description,
                runnable=child_graph,
            )

        entry: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "system_prompt": resolved["system_prompt"],
            "tools": resolved["tools"],
            "model": effective_model,
            "middleware": resolved["middleware"],
        }
        if resolved["skills"] is not None:
            entry["skills"] = resolved["skills"]
        return entry

    def _compile_resolved(
        self,
        resolved: dict[str, Any],
        *,
        model: Any | None,
        compiled_subagent_cls: Any,
        create_deep_agent_fn: Any,
        checkpointer: Any = None,
        _visited: frozenset[int] | None = None,
        _depth: int = 0,
        **root_kwargs: Any,
    ) -> Any:
        visited = _visited or frozenset()
        if id(self) in visited:
            raise ConfigurationError(
                f"Cycle detected in Team hierarchy involving {self.name!r}."
            )
        next_visited = visited | {id(self)}

        self._validate_children()
        own_model = model if model is not None else resolved.get("model")
        if own_model is None:
            own_model = self.model
        if own_model is None:
            raise ValueError(
                f"Team {self.name!r} requires a model (either on Team.model or passed to compile(model=...))."
            )

        effective_inherited = {**resolved, "model": own_model}

        subagents = [
            child._as_subagent_entry(
                inherited=effective_inherited,
                fallback_model=own_model,
                compiled_subagent_cls=compiled_subagent_cls,
                create_deep_agent_fn=create_deep_agent_fn,
                _visited=next_visited,
                _depth=_depth + 1,
            )
            for child in self.children
        ]

        kwargs: dict[str, Any] = {
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
        model: Any | None = None,
        checkpointer: Any = None,
        **root_kwargs: Any,
    ) -> Any:
        """Compile this Team into a runnable DeepAgents graph.

        Raises:
            ImportError: If the optional ``deepagents`` package is not installed.
            ConfigurationError: If a cycle is detected in ``children``.
            ValueError: If no model is configured or sibling names conflict.
        """
        try:
            from deepagents import CompiledSubAgent, create_deep_agent
        except ImportError as exc:
            raise ImportError(
                "deepagents is required to compile a Team into a deep agent graph. "
                "Install it with: pip install 'langcomp[deepagents]'"
            ) from exc

        resolved = self._resolve_from_context(None)
        return self._compile_resolved(
            resolved,
            model=model,
            compiled_subagent_cls=CompiledSubAgent,
            create_deep_agent_fn=create_deep_agent,
            checkpointer=checkpointer,
            **root_kwargs,
        )
