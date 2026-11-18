"""Unit tests for langcomp.models lazy provider registry and resolution."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langcomp import (
    CAPABLE_MODEL,
    DEFAULT_MODEL,
    ConfigurationError,
    FakeModel,
    InvalidModelSpecError,
    MissingApiKeyError,
    ProviderNotInstalledError,
    ProviderSpec,
    UnknownProviderError,
    known_providers,
    parse_model_spec,
    register_provider,
    resolve_model,
)
from langcomp.models import PROVIDERS


class _DummyProviderChatModel(BaseChatModel):
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


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("openrouter:google/gemini-3.5-flash-lite", ("openrouter", "google/gemini-3.5-flash-lite")),
        ("openai:gpt-4o-mini", ("openai", "gpt-4o-mini")),
        ("  openai : gpt-4o-mini  ", ("openai", "gpt-4o-mini")),
        ("openai:ft:my-model:abc", ("openai", "ft:my-model:abc")),
    ],
)
def test_parse_model_spec_splits_on_first_colon(spec: str, expected: tuple[str, str]) -> None:
    assert parse_model_spec(spec) == expected


@pytest.mark.parametrize(
    "spec",
    ["", "   ", "no-separator", ":gpt-4o-mini", "openai:", "openai:   "],
)
def test_parse_model_spec_rejects_malformed_input(spec: str) -> None:
    with pytest.raises(InvalidModelSpecError):
        parse_model_spec(spec)


def test_default_and_capable_models_are_known() -> None:
    for spec in (DEFAULT_MODEL, CAPABLE_MODEL):
        provider, model_id = parse_model_spec(spec)
        assert provider in known_providers()
        assert model_id


def test_resolve_model_passes_through_existing_chat_model() -> None:
    fake = FakeModel(script=["hi"])
    assert resolve_model(fake) is fake


def test_resolve_model_rejects_unknown_provider() -> None:
    with pytest.raises(UnknownProviderError) as excinfo:
        resolve_model("acme:super-model")
    assert excinfo.value.provider == "acme"
    assert "acme" in str(excinfo.value)


def test_resolve_model_reports_missing_integration_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        PROVIDERS,
        "ghost",
        ProviderSpec(module="langcomp_nonexistent_provider", class_name="ChatGhost", extra="ghost"),
    )
    with pytest.raises(ProviderNotInstalledError) as excinfo:
        resolve_model("ghost:any-model")
    assert "langcomp[ghost]" in str(excinfo.value)


def test_resolve_model_demands_gateway_api_key_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_mod = types.ModuleType("dummy_openai_mod")
    dummy_mod.ChatOpenAI = _DummyProviderChatModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dummy_openai_mod", dummy_mod)
    monkeypatch.setitem(
        PROVIDERS,
        "dummy_gateway",
        ProviderSpec(
            module="dummy_openai_mod",
            class_name="ChatOpenAI",
            extra="openai",
            api_key_env="DUMMY_GATEWAY_API_KEY",
            defaults={"base_url": "https://gateway.example/v1", "max_tokens": 4096},
        ),
    )
    monkeypatch.delenv("DUMMY_GATEWAY_API_KEY", raising=False)

    with pytest.raises(MissingApiKeyError) as excinfo:
        resolve_model("dummy_gateway:some/model")
    assert excinfo.value.env_var == "DUMMY_GATEWAY_API_KEY"
    assert "DUMMY_GATEWAY_API_KEY" in str(excinfo.value)


def test_resolve_model_applies_provider_defaults_and_caller_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_mod = types.ModuleType("dummy_openai_mod")
    dummy_mod.ChatOpenAI = _DummyProviderChatModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dummy_openai_mod", dummy_mod)
    monkeypatch.setitem(
        PROVIDERS,
        "dummy_gateway",
        ProviderSpec(
            module="dummy_openai_mod",
            class_name="ChatOpenAI",
            extra="openai",
            api_key_env="DUMMY_GATEWAY_API_KEY",
            defaults={"base_url": "https://gateway.example/v1", "max_tokens": 4096},
        ),
    )
    monkeypatch.setenv("DUMMY_GATEWAY_API_KEY", "env-secret-key")

    default_model = resolve_model("dummy_gateway:vendor/fast")
    assert isinstance(default_model, _DummyProviderChatModel)
    assert default_model.model == "vendor/fast"
    assert default_model.api_key == "env-secret-key"
    assert default_model.base_url == "https://gateway.example/v1"
    assert default_model.max_tokens == 4096

    overridden = resolve_model(
        "dummy_gateway:vendor/fast",
        api_key="explicit-key",
        base_url="http://localhost:8000",
        max_tokens=128,
        temperature=0.0,
    )
    assert isinstance(overridden, _DummyProviderChatModel)
    assert overridden.api_key == "explicit-key"
    assert overridden.base_url == "http://localhost:8000"
    assert overridden.max_tokens == 128
    assert overridden.temperature == 0.0


def test_register_provider_adds_custom_provider_and_prevents_accidental_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(PROVIDERS, "custom_local", raising=False)
    spec = ProviderSpec(module="custom_mod", class_name="ChatLocal", extra="local")

    register_provider("custom_local", spec)
    try:
        assert "custom_local" in known_providers()
        with pytest.raises(ConfigurationError):
            register_provider("custom_local", spec)
        replacement = ProviderSpec(module="custom_mod_v2", class_name="ChatLocalV2", extra="local")
        register_provider("custom_local", replacement, overwrite=True)
        assert PROVIDERS["custom_local"] == replacement
    finally:
        PROVIDERS.pop("custom_local", None)
