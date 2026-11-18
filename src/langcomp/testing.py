"""
Deterministic offline testing utilities for langcomp & LangGraph agents.
Provides FakeModel (scripted chat test double) and run_and_snapshot (golden-master diffing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from langcomp.errors import ScriptExhaustedError, SnapshotMismatchError

ScriptTurn = Union[str, AIMessage]


class FakeModel(BaseChatModel):
    """Deterministic, scripted chat model for offline testing without network calls."""

    script: List[ScriptTurn]
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "langcomp-fake"

    @property
    def call_count(self) -> int:
        """Number of times this scripted model has been invoked."""
        return self._calls

    def reset(self) -> None:
        """Reset playback cursor to the start of the script."""
        self._calls = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeModel":
        """No-op tool binder; tool calls are supplied directly via AIMessage turns."""
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._calls >= len(self.script):
            raise ScriptExhaustedError(
                f"FakeModel script exhausted after {self._calls} call(s); "
                f"add more turns to `script` or check for an unexpected extra call."
            )
        turn = self.script[self._calls]
        self._calls += 1
        message = turn if isinstance(turn, AIMessage) else AIMessage(content=turn)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _to_serializable(v)
            for k, v in value.items()
            if not str(k).startswith("_langcomp")
        }
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "content") and hasattr(value, "type"):
        serialized: dict[str, Any] = {
            "type": value.__class__.__name__,
            "content": value.content,
        }
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            serialized["tool_calls"] = tool_calls
        name = getattr(value, "name", None)
        if name:
            serialized["name"] = name
        tool_call_id = getattr(value, "tool_call_id", None)
        if tool_call_id:
            serialized["tool_call_id"] = tool_call_id
        additional_kwargs = getattr(value, "additional_kwargs", None)
        if additional_kwargs:
            serialized["additional_kwargs"] = additional_kwargs
        return serialized
    return value


def run_and_snapshot(
    graph: Any,
    input_data: Any,
    snapshot_path: Union[str, Path],
    config: Optional[dict[str, Any]] = None,
) -> Any:
    """Invoke a graph or Agent, recording a JSON snapshot on first run and asserting
    exact equivalence on subsequent runs.

    Raises:
        SnapshotMismatchError: When actual output drifts from the recorded snapshot.
    """
    if config is not None:
        result = graph.invoke(input_data, config=config)
    else:
        result = graph.invoke(input_data)

    serializable = _to_serializable(result)
    path = Path(snapshot_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(serializable, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return result

    recorded = json.loads(path.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(serializable, default=str))
    if recorded != normalized:
        raise SnapshotMismatchError(
            f"Run output differs from recorded snapshot at {path}.\n"
            f"Recorded: {recorded}\nActual:   {normalized}"
        )
    return result
