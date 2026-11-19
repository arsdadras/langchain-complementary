"""
Deterministic offline testing utilities for langcomp & LangGraph agents.
Provides FakeModel (thread-safe scripted chat double) and run_and_snapshot / arun_and_snapshot.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import os
import tempfile
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from langcomp.errors import ScriptExhaustedError, SnapshotMismatchError

ScriptTurn = str | AIMessage


class _SharedPlaybackState:
    """Thread-safe playback cursor shared between a FakeModel and its bound clones."""

    __slots__ = ("bound_tool_names", "calls", "lock", "recorded_calls")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.calls: int = 0
        self.recorded_calls: list[list[BaseMessage]] = []
        self.bound_tool_names: tuple[str, ...] = ()


def _extract_tool_name(tool_obj: Any) -> str:
    if isinstance(tool_obj, BaseTool):
        return tool_obj.name
    if isinstance(tool_obj, dict):
        fn = tool_obj.get("function")
        if isinstance(fn, dict) and "name" in fn:
            return str(fn["name"])
        if "name" in tool_obj:
            return str(tool_obj["name"])
    return getattr(tool_obj, "__name__", str(tool_obj))


class FakeModel(BaseChatModel):
    """Deterministic, thread-safe scripted chat model for offline testing without network calls.

    Records every call's input messages in :attr:`calls` and bound tool names
    in :attr:`bound_tool_names` so tests can verify exact prompts and tool bindings.
    Supports ``invoke``, ``ainvoke``, ``stream``, ``astream``, and non-mutating ``bind_tools``.
    """

    script: list[ScriptTurn]
    _playback: _SharedPlaybackState = PrivateAttr(default_factory=_SharedPlaybackState)
    _local_bound_tool_names: tuple[str, ...] | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "langcomp-fake"

    @property
    def call_count(self) -> int:
        """Number of times this scripted model has been invoked."""
        with self._playback.lock:
            return self._playback.calls

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """Copy of input message lists passed to each call so far, oldest first."""
        with self._playback.lock:
            return [list(msgs) for msgs in self._playback.recorded_calls]

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        """Names of tools bound on this clone, or most recently bound on the base model."""
        if self._local_bound_tool_names is not None:
            return self._local_bound_tool_names
        with self._playback.lock:
            return self._playback.bound_tool_names

    def reset(self) -> None:
        """Reset playback cursor, recorded call history, and bound tool tracking."""
        with self._playback.lock:
            self._playback.calls = 0
            self._playback.recorded_calls = []
            self._playback.bound_tool_names = ()
        self._local_bound_tool_names = None

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeModel:
        """Return a non-mutating bound clone recording the bound tool names."""
        del kwargs
        names = tuple(_extract_tool_name(t) for t in tools)
        with self._playback.lock:
            self._playback.bound_tool_names = names
        clone = self.model_copy(deep=False)
        clone._playback = self._playback
        clone._local_bound_tool_names = names
        return clone

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        with self._playback.lock:
            if self._playback.calls >= len(self.script):
                raise ScriptExhaustedError(
                    f"FakeModel script exhausted after {self._playback.calls} call(s); "
                    f"add more turns to `script` or check for an unexpected extra call."
                )
            turn = self.script[self._playback.calls]
            self._playback.calls += 1
            self._playback.recorded_calls.append(list(messages))
        if isinstance(turn, AIMessage):
            return turn.model_copy(deep=True)
        return AIMessage(content=turn)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        message = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del stop, run_manager, kwargs
        message = self._next_message(messages)
        chunk = AIMessageChunk(
            content=message.content,
            additional_kwargs=dict(message.additional_kwargs),
            response_metadata=dict(message.response_metadata),
            usage_metadata=message.usage_metadata,
        )
        yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk


def _to_serializable(value: Any, _seen: frozenset[int] = frozenset()) -> Any:
    obj_id = id(value)
    if isinstance(value, (dict, list, tuple, set, BaseModel)) or dataclasses.is_dataclass(value):
        if obj_id in _seen:
            return "<circular-ref>"
        _seen = _seen | {obj_id}

    if isinstance(value, BaseMessage):
        serialized: dict[str, Any] = {
            "type": value.__class__.__name__,
            "content": _to_serializable(value.content, _seen),
        }
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            serialized["tool_calls"] = _to_serializable(tool_calls, _seen)
        name = getattr(value, "name", None)
        if name:
            serialized["name"] = name
        if isinstance(value, ToolMessage):
            serialized["tool_call_id"] = value.tool_call_id
            serialized["status"] = value.status
            if value.artifact is not None:
                serialized["artifact"] = _to_serializable(value.artifact, _seen)
        else:
            tool_call_id = getattr(value, "tool_call_id", None)
            if tool_call_id:
                serialized["tool_call_id"] = tool_call_id
        additional_kwargs = getattr(value, "additional_kwargs", None)
        if additional_kwargs:
            serialized["additional_kwargs"] = _to_serializable(additional_kwargs, _seen)
        return serialized

    if isinstance(value, dict):
        return {
            str(k): _to_serializable(v, _seen)
            for k, v in value.items()
            if not str(k).startswith("_langcomp")
        }
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v, _seen) for v in value]
    if isinstance(value, set):
        return sorted((_to_serializable(v, _seen) for v in value), key=lambda x: str(x))
    if isinstance(value, BaseModel):
        return _to_serializable(value.model_dump(), _seen)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_serializable(dataclasses.asdict(value), _seen)
    return value


def _write_snapshot_atomically(path: Path, payload_text: str) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to write snapshot to symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".snap_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload_text)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _compare_or_write_snapshot(
    result: Any,
    snapshot_path: str | Path,
    *,
    update_snapshot: bool = False,
) -> Any:
    serializable = _to_serializable(result)
    formatted_actual = json.dumps(serializable, indent=2, sort_keys=True, default=str)
    path = Path(snapshot_path)

    if update_snapshot or not path.exists():
        _write_snapshot_atomically(path, formatted_actual)
        return result

    if path.is_symlink():
        raise ValueError(f"Refusing to read snapshot from symlink: {path}")

    recorded_text = path.read_text(encoding="utf-8")
    recorded = json.loads(recorded_text)
    normalized = json.loads(formatted_actual)
    if recorded != normalized:
        formatted_recorded = json.dumps(recorded, indent=2, sort_keys=True, default=str)
        diff = "\n".join(
            difflib.unified_diff(
                formatted_recorded.splitlines(),
                formatted_actual.splitlines(),
                fromfile=f"{path} (recorded)",
                tofile="actual",
                lineterm="",
            )
        )
        raise SnapshotMismatchError(
            f"Run output differs from recorded snapshot at {path}:\n{diff}"
        )
    return result


def run_and_snapshot(
    graph: Any,
    input_data: Any,
    snapshot_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    update_snapshot: bool = False,
) -> Any:
    """Synchronously invoke a graph or Agent, recording or asserting against a JSON snapshot.

    Raises:
        SnapshotMismatchError: When actual output drifts from the recorded snapshot.
    """
    result = graph.invoke(input_data, config=config) if config is not None else graph.invoke(input_data)
    return _compare_or_write_snapshot(result, snapshot_path, update_snapshot=update_snapshot)


async def arun_and_snapshot(
    graph: Any,
    input_data: Any,
    snapshot_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    update_snapshot: bool = False,
) -> Any:
    """Asynchronously invoke a graph or Agent, recording or asserting against a JSON snapshot.

    Raises:
        SnapshotMismatchError: When actual output drifts from the recorded snapshot.
    """
    result = (
        await graph.ainvoke(input_data, config=config)
        if config is not None
        else await graph.ainvoke(input_data)
    )
    return _compare_or_write_snapshot(result, snapshot_path, update_snapshot=update_snapshot)


__all__ = ["FakeModel", "ScriptTurn", "arun_and_snapshot", "run_and_snapshot"]
