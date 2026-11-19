"""Shared pytest fixtures and test doubles for the langcomp test suite."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool

from langcomp.models import PROVIDERS


class DummyProviderChatModel(BaseChatModel):
    """Minimal BaseChatModel used to verify lazy provider instantiation offline."""

    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None

    @property
    def _llm_type(self) -> str:
        return "dummy-provider"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


@pytest.fixture
def dummy_chat_model_cls() -> type[DummyProviderChatModel]:
    """Return the offline dummy BaseChatModel subclass for provider tests."""
    return DummyProviderChatModel


@pytest.fixture
def isolated_providers() -> Iterator[dict[str, Any]]:
    """Snapshot and restore the global provider registry around a test."""
    snapshot = dict(PROVIDERS)
    try:
        yield PROVIDERS
    finally:
        PROVIDERS.clear()
        PROVIDERS.update(snapshot)


@pytest.fixture
def add_tool() -> BaseTool:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return add


@pytest.fixture
def explode_tool() -> BaseTool:
    @tool
    def explode() -> None:
        """Always raises to exercise error-handling paths."""
        raise ValueError("kaboom")

    return explode
