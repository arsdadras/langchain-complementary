"""Unit tests for langcomp.models lazy provider registry and resolution."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

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
    get_provider,
    known_providers,
    parse_model_spec,
    register_provider,
    resolve_model,
    unregister_provider,
)


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
    ["", "   ", "no-separator", ":gpt-4o-mini", "openai:", "openai:   ", 123],
)
def test_parse_model_spec_rejects_malformed_input(spec: Any) -> None:
    with pytest.raises(InvalidModelSpecError):
        parse_model_spec(spec)


def test_default_and_capable_models_are_known() -> None:
    for spec in (DEFAULT_MODEL, CAPABLE_MODEL):
        provider, model_id = parse_model_spec(spec)
        assert provider in known_providers()
        assert model_id


def test_resolve_model_passes_through_existing_chat_model_and_rejects_kwargs() -> None:
    fake = FakeModel(script=["hi"])
    assert resolve_model(fake) is fake
    with pytest.raises(ConfigurationError, match="constructor kwargs"):
        resolve_model(fake, temperature=0.2)


def test_resolve_model_rejects_empty_string_and_unknown_provider() -> None:
    with pytest.raises(InvalidModelSpecError):
        resolve_model("")

    with pytest.raises(UnknownProviderError) as excinfo:
        resolve_model("acme:super-model")
    assert excinfo.value.provider == "acme"
    assert "acme" in str(excinfo.value)


def test_resolve_model_reports_missing_integration_package(
    isolated_providers: dict[str, Any],
) -> None:
    register_provider(
        "ghost",
        ProviderSpec(module="langcomp_nonexistent_provider", class_name="ChatGhost", extra="ghost"),
    )
    with pytest.raises(ProviderNotInstalledError) as excinfo:
        resolve_model("ghost:any-model")
    assert "langcomp[ghost]" in str(excinfo.value)


def test_resolve_model_validates_target_class_is_base_chat_model(
    monkeypatch: pytest.MonkeyPatch,
    isolated_providers: dict[str, Any],
) -> None:
    bad_mod = types.ModuleType("dummy_bad_mod")
    bad_mod.NotAChatModel = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dummy_bad_mod", bad_mod)
    register_provider(
        "bad_cls",
        ProviderSpec(module="dummy_bad_mod", class_name="NotAChatModel", extra="bad"),
    )
    with pytest.raises(ConfigurationError, match="not a BaseChatModel"):
        resolve_model("bad_cls:m1")


def test_resolve_model_gateway_api_key_isolation_and_base_url_guard(
    monkeypatch: pytest.MonkeyPatch,
    isolated_providers: dict[str, Any],
    dummy_chat_model_cls: type[Any],
) -> None:
    dummy_mod = types.ModuleType("dummy_openai_mod")
    dummy_mod.ChatOpenAI = dummy_chat_model_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dummy_openai_mod", dummy_mod)
    register_provider(
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
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-do-not-leak")

    with pytest.raises(MissingApiKeyError) as excinfo:
        resolve_model("dummy_gateway:some/model")
    assert excinfo.value.env_var == "DUMMY_GATEWAY_API_KEY"

    # api_key=None or whitespace MUST NOT bypass MissingApiKeyError
    with pytest.raises(MissingApiKeyError):
        resolve_model("dummy_gateway:some/model", api_key=None)
    with pytest.raises(MissingApiKeyError):
        resolve_model("dummy_gateway:some/model", api_key="   ")

    monkeypatch.setenv("DUMMY_GATEWAY_API_KEY", "   ")
    with pytest.raises(MissingApiKeyError):
        resolve_model("dummy_gateway:some/model")

    monkeypatch.setenv("DUMMY_GATEWAY_API_KEY", "env-secret-key")
    # Overriding base_url without explicit api_key MUST be blocked to prevent key exfiltration
    with pytest.raises(ConfigurationError, match="Refusing to auto-inject"):
        resolve_model("dummy_gateway:vendor/fast", base_url="https://attacker.example/v1")

    default_model = resolve_model("dummy_gateway:vendor/fast")
    assert isinstance(default_model, dummy_chat_model_cls)
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
    assert isinstance(overridden, dummy_chat_model_cls)
    assert overridden.api_key == "explicit-key"
    assert overridden.base_url == "http://localhost:8000"
    assert overridden.max_tokens == 128
    assert overridden.temperature == 0.0


def test_register_and_unregister_provider_validation(
    isolated_providers: dict[str, Any],
) -> None:
    spec = ProviderSpec(module="custom_mod", class_name="ChatLocal", extra="local", defaults={"a": 1})
    with pytest.raises(TypeError):
        spec.defaults["a"] = 2  # type: ignore[index]

    with pytest.raises(ConfigurationError, match="must not be empty"):
        register_provider("   ", spec)
    with pytest.raises(ConfigurationError, match="must not contain separator"):
        register_provider("a:b", spec)

    register_provider("custom_local", spec)
    assert "custom_local" in known_providers()
    assert get_provider("custom_local") == spec

    with pytest.raises(ConfigurationError):
        register_provider("custom_local", spec)

    replacement = ProviderSpec(module="custom_mod_v2", class_name="ChatLocalV2", extra="local")
    register_provider("custom_local", replacement, overwrite=True)
    assert get_provider("custom_local") == replacement

    removed = unregister_provider("custom_local")
    assert removed == replacement
    assert get_provider("custom_local") is None
