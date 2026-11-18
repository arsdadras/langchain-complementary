"""Deterministic test doubles for exercising langcomp agents offline.

:class:`ScriptedFakeChatModel` replays a fixed sequence of messages (including
tool-calling ``AIMessage`` turns) without making any network calls, recording
every invocation so tests can assert on exact prompts and tool bindings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr, field_validator

from langcomp.errors import ScriptExhaustedError


class ScriptedFakeChatModel(BaseChatModel):
    """Replays a fixed list of responses, one per call, in order.

    Each call to :meth:`invoke` or :meth:`ainvoke` returns the next scripted
    message and advances an internal cursor. Calling past the end of the script
    raises :class:`~langcomp.errors.ScriptExhaustedError` instead of silently
    looping.

    Plain strings in ``responses`` are coerced to ``AIMessage(content=...)``.
    To script a tool call, pass an ``AIMessage`` with ``tool_calls`` populated.
    Every call's input messages are recorded in :attr:`calls`.
    """

    responses: Sequence[BaseMessage | str] = Field(default_factory=list)

    _index: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tool_names: tuple[str, ...] = PrivateAttr(default=())

    @field_validator("responses", mode="before")
    @classmethod
    def _coerce_strings_to_ai_messages(
        cls,
        value: Sequence[BaseMessage | str],
    ) -> list[Any]:
        return [AIMessage(content=item) if isinstance(item, str) else item for item in value]

    @property
    def _llm_type(self) -> str:
        return "langcomp-scripted-fake"

    @property
    def call_count(self) -> int:
        """How many times this model has been invoked."""
        return self._index

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """Input message lists passed to each call so far, oldest first."""
        return self._calls

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        """Names of tools most recently passed to :meth:`bind_tools`."""
        return self._bound_tool_names

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **_kwargs: Any,
    ) -> ScriptedFakeChatModel:
        """Record bound tool names for test assertions without altering playback."""
        del tool_choice
        self._bound_tool_names = tuple(
            t.name if isinstance(t, BaseTool) else getattr(t, "__name__", str(t)) for t in tools
        )
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **_kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        self._calls.append(list(messages))
        if self._index >= len(self.responses):
            raise ScriptExhaustedError(self._index)
        scripted = self.responses[self._index]
        message = AIMessage(content=scripted) if isinstance(scripted, str) else scripted
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])
