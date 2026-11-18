"""SmartState management for langcomp framework.

Eliminates boilerplate TypedDict and reducer definitions for LangGraph while
supporting schema composition from middleware capabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, get_type_hints

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages


def _extract_schema_annotations(schema: type[Any]) -> dict[str, Any]:
    """Extract field annotations from a TypedDict or annotated dict subclass."""
    try:
        return dict(get_type_hints(schema, include_extras=True))
    except Exception:  # noqa: BLE001 - fall back to raw annotations if forward refs fail
        return dict(getattr(schema, "__annotations__", {}))


class SmartState:
    """Helper for dynamically building LangGraph state schemas without boilerplate.

    Provides standard ``add_messages`` accumulation, middleware schema merging,
    and message normalization.
    """

    @classmethod
    def create_schema(
        cls,
        extra_fields: Mapping[str, Any] | None = None,
        messages_key: str = "messages",
        schemas: Sequence[type[Any]] | None = None,
    ) -> type[Any]:
        """Dynamically create a TypedDict-compatible state schema for LangGraph.

        Args:
            extra_fields: Optional mapping of extra state field names to type hints.
            messages_key: Key name used for conversation history messages.
            schemas: Optional sequence of TypedDict or annotated dict classes
                (e.g. contributed by middleware) whose fields are merged in order.

        Returns:
            A dynamically generated dict subclass with LangGraph-compatible annotations.
        """
        annotations: dict[str, Any] = {
            messages_key: Annotated[list[BaseMessage], add_messages],
        }

        if schemas:
            for schema in schemas:
                for key, val_type in _extract_schema_annotations(schema).items():
                    if key != messages_key:
                        annotations[key] = val_type

        if extra_fields:
            for key, val_type in extra_fields.items():
                annotations[key] = val_type

        namespace = {"__annotations__": annotations}
        return type("GeneratedSmartState", (dict,), namespace)

    @staticmethod
    def normalize_messages(
        messages: Sequence[BaseMessage | dict[str, str] | str],
    ) -> list[BaseMessage]:
        """Convert dicts or raw strings into standard LangChain ``BaseMessage`` instances."""
        normalized: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                normalized.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user").lower()
                content = msg.get("content", "")
                if role in ("user", "human"):
                    normalized.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    normalized.append(AIMessage(content=content))
                elif role == "system":
                    normalized.append(SystemMessage(content=content))
                else:
                    normalized.append(HumanMessage(content=content))
            elif isinstance(msg, str):
                normalized.append(HumanMessage(content=msg))
        return normalized
