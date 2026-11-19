"""
State management for the langcomp framework.
Provides both SmartState (zero-boilerplate dynamic schemas + message normalization)
and State / Reduce (Pydantic-based schemas with type-inferred reducers & explain()).
"""

from __future__ import annotations

import inspect
import operator
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, get_type_hints

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
from pydantic_core import PydanticUndefined

from langcomp.errors import ReducerConflictError

_REDUCER_KEY = "langcomp_reducer"
_UNSET = object()


def overwrite(a: Any, b: Any) -> Any:
    """Default scalar reducer: latest update wins."""
    del a
    return b


def merge_dicts(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow-merge two dictionaries, with keys in ``b`` taking precedence."""
    return {**(a or {}), **(b or {})}


def Reduce(
    fn: Callable[[Any, Any], Any],
    *,
    default: Any = _UNSET,
    default_factory: Any = _UNSET,
) -> FieldInfo:
    """Declare an explicit reducer on a ``State`` field, overriding the type default."""
    if not callable(fn):
        raise TypeError(f"Reduce() expects a callable reducer function, got {type(fn).__name__}")
    if default is not _UNSET and default_factory is not _UNSET:
        raise ReducerConflictError(
            "Cannot specify both 'default' and 'default_factory' in Reduce()"
        )
    extra: dict[str, Any] = {_REDUCER_KEY: fn}
    if default_factory is not _UNSET:
        return Field(default_factory=default_factory, json_schema_extra=extra)
    if default is not _UNSET:
        return Field(default=default, json_schema_extra=extra)
    return Field(json_schema_extra=extra)


def _unwrap_optional(annotation: Any) -> Any:
    """Unwrap ``Optional[T]`` / ``T | None`` to ``T`` for reducer inference."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        non_none = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _default_reducer_for(annotation: Any) -> Callable[[Any, Any], Any]:
    unwrapped = _unwrap_optional(annotation)
    origin = typing.get_origin(unwrapped)
    if origin is list or unwrapped is list:
        return operator.add
    if origin is dict or unwrapped is dict:
        return merge_dicts
    return overwrite


def _is_reducer_metadata(meta: Any) -> bool:
    """Return True if an ``Annotated`` metadata entry is a 2-arg reducer (not a Pydantic validator)."""
    if not callable(meta):
        return False
    mod = getattr(type(meta), "__module__", "")
    if mod.startswith("pydantic"):
        return False
    try:
        sig = inspect.signature(meta)
    except (TypeError, ValueError):
        return True
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    return has_varargs or len(positional) >= 2


def _extract_schema_annotations(schema: type[Any]) -> dict[str, Any]:
    """Extract field annotations from a TypedDict, Pydantic State, or annotated dict class."""
    if isinstance(schema, type) and issubclass(schema, State):
        return dict(schema._annotated_fields())
    try:
        return dict(get_type_hints(schema, include_extras=True))
    except Exception:  # noqa: BLE001 - fall back to raw annotations if forward refs fail
        return dict(getattr(schema, "__annotations__", {}))


def extract_schema_reducers(schema: type[Any]) -> dict[str, Callable[[Any, Any], Any]]:
    """Extract callable reducers from an ``Annotated[T, reducer]`` schema (TypedDict or State)."""
    if isinstance(schema, type) and issubclass(schema, State):
        return schema._reducers()
    reducers: dict[str, Callable[[Any, Any], Any]] = {}
    for key, ann in _extract_schema_annotations(schema).items():
        if typing.get_origin(ann) is Annotated:
            args = typing.get_args(ann)
            for meta in args[1:]:
                if _is_reducer_metadata(meta):
                    reducers[key] = meta
                    break
    return reducers


def merge_state_schemas(
    base_schema: type[Any],
    extra_schemas: Sequence[type[Any]],
    *,
    messages_key: str = "messages",
    ensure_messages: bool = False,
) -> type[Any]:
    """Merge extra middleware state schemas onto an existing LangGraph ``TypedDict`` schema."""
    if not extra_schemas and not ensure_messages:
        return base_schema
    annotations = _extract_schema_annotations(base_schema)
    if ensure_messages and messages_key not in annotations:
        annotations[messages_key] = Annotated[Sequence[AnyMessage], add_messages]
    for schema in extra_schemas:
        for key, val_type in _extract_schema_annotations(schema).items():
            if key != messages_key:
                annotations[key] = val_type
    return typing.TypedDict(f"{base_schema.__name__}Merged", annotations, total=False)  # type: ignore[operator]


class State(BaseModel):
    """Pydantic base class for typed LangGraph state with automatic reducer inference.

    Default reducers by field type:
    - ``Annotated[T, reducer_fn]`` -> preserves ``reducer_fn``
    - ``list[...]`` / ``list[...] | None`` -> ``operator.add`` (append)
    - ``dict[...]`` / ``dict[...] | None`` -> ``merge_dicts`` (shallow merge)
    - scalars / other types -> ``overwrite`` (last write wins)

    Override any field's reducer explicitly using :func:`Reduce`.
    """

    @classmethod
    def _reducers(cls) -> dict[str, Callable[[Any, Any], Any]]:
        reducers: dict[str, Callable[[Any, Any], Any]] = {}
        for name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra
            explicit_reduce = (
                extra[_REDUCER_KEY]
                if isinstance(extra, dict) and _REDUCER_KEY in extra
                else None
            )
            annotated_reducers = [
                meta for meta in (field_info.metadata or ()) if _is_reducer_metadata(meta)
            ]

            if len(annotated_reducers) > 1:
                raise ReducerConflictError(
                    f"Field {name!r} on {cls.__name__} declares multiple Annotated reducers."
                )
            if explicit_reduce is not None and annotated_reducers:
                raise ReducerConflictError(
                    f"Field {name!r} on {cls.__name__} specifies both an Annotated reducer and Reduce()."
                )

            if explicit_reduce is not None:
                reducers[name] = explicit_reduce  # type: ignore[assignment]
            elif annotated_reducers:
                reducers[name] = annotated_reducers[0]
            else:
                reducers[name] = _default_reducer_for(field_info.annotation)
        return reducers

    @classmethod
    def _defaults(cls) -> dict[str, Any]:
        """Materialize declared field defaults and default factories for initial state hydration."""
        defaults: dict[str, Any] = {}
        for name, field_info in cls.model_fields.items():
            if field_info.default_factory is not None:
                defaults[name] = field_info.default_factory()  # type: ignore[call-arg]
            elif field_info.default is not PydanticUndefined:
                val = field_info.default
                if isinstance(val, (list, dict, set)):
                    defaults[name] = val.copy()
                else:
                    defaults[name] = val
        return defaults

    @classmethod
    def explain(cls) -> dict[str, str]:
        """Return a mapping of ``{field_name: reducer_name}`` for introspection."""
        return {
            name: getattr(fn, "__name__", repr(fn))
            for name, fn in cls._reducers().items()
        }

    @classmethod
    def _annotated_fields(cls) -> dict[str, Any]:
        reducers = cls._reducers()
        return {
            name: Annotated[field_info.annotation, reducers[name]]
            for name, field_info in cls.model_fields.items()
        }

    @classmethod
    def to_typed_dict(cls, *, include_internal: bool = False) -> type[Any]:
        """Compile this Pydantic ``State`` into a LangGraph-compatible ``TypedDict``."""
        annotations = dict(cls._annotated_fields())
        if include_internal and "_langcomp_done" not in annotations:
            annotations["_langcomp_done"] = Annotated[list[str], operator.add]
        return typing.TypedDict(cls.__name__ + "Dict", annotations, total=False)  # type: ignore[operator]


def _message_from_role(
    role: str,
    content: Any,
    *,
    tool_call_id: str | None = None,
    name: str | None = None,
    tool_calls: list[Any] | None = None,
    msg_id: str | None = None,
    additional_kwargs: dict[str, Any] | None = None,
) -> BaseMessage:
    normalized_role = role.lower()
    extra: dict[str, Any] = {}
    if name is not None:
        extra["name"] = name
    if msg_id is not None:
        extra["id"] = msg_id
    if additional_kwargs:
        extra["additional_kwargs"] = dict(additional_kwargs)

    if normalized_role in ("user", "human"):
        return HumanMessage(content=content, **extra)
    if normalized_role in ("assistant", "ai"):
        if tool_calls is not None:
            extra["tool_calls"] = tool_calls
        return AIMessage(content=content, **extra)
    if normalized_role in ("system", "developer"):
        return SystemMessage(content=content, **extra)
    if normalized_role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id or "",
            **extra,
        )
    raise ValueError(
        f"Unknown message role {role!r}; expected one of 'user', 'human', "
        f"'assistant', 'ai', 'system', 'developer', or 'tool'."
    )


class SmartState:
    """Helper for dynamically building LangGraph state schemas without boilerplate.

    Provides standard message accumulation (``add_messages``), middleware schema
    composition, and message normalization.
    """

    @classmethod
    def create_schema(
        cls,
        extra_fields: Mapping[str, Any] | None = None,
        messages_key: str = "messages",
        schemas: Sequence[type[Any]] | None = None,
    ) -> type[Any]:
        """Dynamically create a ``TypedDict`` class compatible with LangGraph."""
        annotations: dict[str, Any] = {
            messages_key: Annotated[Sequence[AnyMessage], add_messages],
            "_langcomp_done": Annotated[list[str], operator.add],
        }

        if schemas:
            for schema in schemas:
                for key, val_type in _extract_schema_annotations(schema).items():
                    if key not in (messages_key, "_langcomp_done"):
                        annotations[key] = val_type

        if extra_fields:
            for key, val_type in extra_fields.items():
                if key not in (messages_key, "_langcomp_done"):
                    annotations[key] = val_type

        return typing.TypedDict("GeneratedSmartState", annotations, total=False)  # type: ignore[operator]

    @staticmethod
    def normalize_messages(
        messages: Sequence[BaseMessage | dict[str, Any] | tuple[str, Any] | str],
    ) -> list[BaseMessage]:
        """Convert strings, ``(role, content)`` tuples, or dicts into LangChain messages."""
        normalized: list[BaseMessage] = []
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
                        tool_calls=msg.get("tool_calls"),
                        msg_id=msg.get("id"),
                        additional_kwargs=msg.get("additional_kwargs"),
                    )
                )
            elif isinstance(msg, tuple) and len(msg) == 2:
                normalized.append(_message_from_role(str(msg[0]), msg[1]))
            elif isinstance(msg, str):
                normalized.append(HumanMessage(content=msg))
            else:
                raise TypeError(
                    f"Unsupported message item of type {type(msg).__name__}; expected "
                    f"BaseMessage, dict, (role, content) tuple, or str."
                )
        return normalized
