"""
State management for the langcomp framework.
Provides both SmartState (zero-boilerplate dynamic schemas + message normalization)
and State / Reduce (Pydantic-based schemas with type-inferred reducers & explain()).
"""

from __future__ import annotations

import operator
import typing
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, Type, Union

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from langcomp.errors import ReducerConflictError

_REDUCER_KEY = "langcomp_reducer"
_UNSET = object()


def overwrite(a: Any, b: Any) -> Any:
    """Default scalar reducer: latest update wins."""
    return b


def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge two dictionaries, with keys in ``b`` taking precedence."""
    return {**(a or {}), **(b or {})}


def Reduce(
    fn: Callable[[Any, Any], Any],
    *,
    default: Any = _UNSET,
    default_factory: Any = _UNSET,
) -> FieldInfo:
    """Declare an explicit reducer on a ``State`` field, overriding the type default."""
    if default is not _UNSET and default_factory is not _UNSET:
        raise ReducerConflictError(
            "Cannot specify both 'default' and 'default_factory' in Reduce()"
        )
    extra: Dict[str, Any] = {_REDUCER_KEY: fn}
    if default_factory is not _UNSET:
        return Field(default_factory=default_factory, json_schema_extra=extra)
    if default is not _UNSET:
        return Field(default=default, json_schema_extra=extra)
    return Field(json_schema_extra=extra)


def _default_reducer_for(annotation: Any) -> Callable[[Any, Any], Any]:
    origin = typing.get_origin(annotation)
    if origin is list or annotation is list:
        return operator.add
    if origin is dict or annotation is dict:
        return merge_dicts
    return overwrite


class State(BaseModel):
    """Pydantic base class for typed LangGraph state with automatic reducer inference.

    Default reducers by field type:
    - ``Annotated[T, reducer_fn]`` -> preserves ``reducer_fn``
    - ``list[...]`` -> ``operator.add`` (append)
    - ``dict[...]`` -> ``merge_dicts`` (shallow merge)
    - scalars / other types -> ``overwrite`` (last write wins)

    Override any field's reducer explicitly using :func:`Reduce`.
    """

    @classmethod
    def _reducers(cls) -> Dict[str, Callable[[Any, Any], Any]]:
        reducers: Dict[str, Callable[[Any, Any], Any]] = {}
        for name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and _REDUCER_KEY in extra:
                reducers[name] = extra[_REDUCER_KEY]  # type: ignore[assignment]
                continue

            # Preserve callable reducers declared via typing.Annotated[..., reducer]
            annotated_reducer = next(
                (meta for meta in (field_info.metadata or ()) if callable(meta)),
                None,
            )
            if annotated_reducer is not None:
                reducers[name] = annotated_reducer
            else:
                reducers[name] = _default_reducer_for(field_info.annotation)
        return reducers

    @classmethod
    def explain(cls) -> Dict[str, str]:
        """Return a mapping of ``{field_name: reducer_name}`` for introspection."""
        return {
            name: getattr(fn, "__name__", repr(fn))
            for name, fn in cls._reducers().items()
        }

    @classmethod
    def _annotated_fields(cls) -> Dict[str, Any]:
        reducers = cls._reducers()
        return {
            name: Annotated[field_info.annotation, reducers[name]]
            for name, field_info in cls.model_fields.items()
        }

    @classmethod
    def to_typed_dict(cls, *, include_internal: bool = False) -> Type[Any]:
        """Compile this Pydantic ``State`` into a LangGraph-compatible ``TypedDict``."""
        annotations = dict(cls._annotated_fields())
        if include_internal and "_langcomp_done" not in annotations:
            annotations["_langcomp_done"] = Annotated[List[str], operator.add]
        return typing.TypedDict(cls.__name__ + "Dict", annotations, total=False)  # type: ignore[misc]


def _message_from_role(
    role: str,
    content: Any,
    *,
    tool_call_id: Optional[str] = None,
    name: Optional[str] = None,
) -> BaseMessage:
    normalized_role = role.lower()
    if normalized_role in ("user", "human"):
        return HumanMessage(content=content)
    if normalized_role in ("assistant", "ai"):
        return AIMessage(content=content)
    if normalized_role == "system":
        return SystemMessage(content=content)
    if normalized_role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id or "",
            name=name,
        )
    return HumanMessage(content=content)


class SmartState:
    """Helper for dynamically building LangGraph state schemas without boilerplate.

    Provides standard message accumulation (``add_messages``) and message normalization.
    """

    @classmethod
    def create_schema(
        cls,
        extra_fields: Optional[Dict[str, Any]] = None,
        messages_key: str = "messages",
    ) -> Type[Any]:
        """Dynamically create a ``TypedDict`` class compatible with LangGraph."""
        annotations: Dict[str, Any] = {
            messages_key: Annotated[Sequence[AnyMessage], add_messages],
            "_langcomp_done": Annotated[List[str], operator.add],
        }

        if extra_fields:
            for key, val_type in extra_fields.items():
                annotations[key] = val_type

        return typing.TypedDict("GeneratedSmartState", annotations, total=False)  # type: ignore[misc]

    @staticmethod
    def normalize_messages(
        messages: Sequence[Union[BaseMessage, Dict[str, Any], tuple[str, Any], str]],
    ) -> List[BaseMessage]:
        """Convert strings, ``(role, content)`` tuples, or dicts into LangChain messages."""
        normalized: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(
                    _message_from_role(
                        str(msg.get("role", "user")),
                        msg.get("content", ""),
                        tool_call_id=msg.get("tool_call_id"),
                        name=msg.get("name"),
                    )
                )
            elif isinstance(msg, tuple) and len(msg) == 2:
                normalized.append(_message_from_role(str(msg[0]), msg[1]))
            elif isinstance(msg, str):
                normalized.append(HumanMessage(content=msg))
        return normalized
