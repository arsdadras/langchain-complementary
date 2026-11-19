"""Thread-safe lazy provider registry and registration helpers for langcomp."""

from __future__ import annotations

import threading
from types import MappingProxyType
from typing import Final

from langcomp.errors import ConfigurationError
from langcomp.models.constants import SPEC_SEPARATOR
from langcomp.models.spec import ProviderSpec

_REGISTRY_LOCK = threading.RLock()

PROVIDERS: Final[dict[str, ProviderSpec]] = {
    "google_genai": ProviderSpec(
        module="langchain_google_genai",
        class_name="ChatGoogleGenerativeAI",
        extra="google",
    ),
    "openai": ProviderSpec(
        module="langchain_openai",
        class_name="ChatOpenAI",
        extra="openai",
    ),
    "openrouter": ProviderSpec(
        module="langchain_openai",
        class_name="ChatOpenAI",
        extra="openai",
        api_key_env="OPENROUTER_API_KEY",
        defaults=MappingProxyType(
            {
                "base_url": "https://openrouter.ai/api/v1",
                # Explicit completion cap prevents OpenRouter from reserving the
                # entire context window up front on low-balance accounts.
                "max_tokens": 4096,
            }
        ),
    ),
}


def known_providers() -> tuple[str, ...]:
    """Return the provider names currently registered with langcomp."""
    with _REGISTRY_LOCK:
        return tuple(PROVIDERS)


def get_provider(name: str) -> ProviderSpec | None:
    """Thread-safely retrieve a registered provider specification by name."""
    with _REGISTRY_LOCK:
        return PROVIDERS.get(name)


def register_provider(
    name: str,
    spec: ProviderSpec,
    *,
    overwrite: bool = False,
) -> None:
    """Register a custom model provider in the lazy provider registry.

    Args:
        name: Provider prefix used in ``"<name>:<model-id>"`` specs.
        spec: Lazy import and constructor metadata for the provider.
        overwrite: Allow replacing an existing provider entry when ``True``.

    Raises:
        ConfigurationError: If ``name`` is blank, contains ``":"``, or is already
            registered without ``overwrite=True``.
    """
    normalized = name.strip()
    if not normalized:
        raise ConfigurationError("Provider name must not be empty.")
    if SPEC_SEPARATOR in normalized:
        raise ConfigurationError(
            f"Provider name {normalized!r} must not contain separator {SPEC_SEPARATOR!r}."
        )
    with _REGISTRY_LOCK:
        if normalized in PROVIDERS and not overwrite:
            raise ConfigurationError(
                f"Provider {normalized!r} is already registered. Pass overwrite=True to replace it."
            )
        PROVIDERS[normalized] = spec


def unregister_provider(name: str) -> ProviderSpec | None:
    """Remove a registered provider by name and return its specification if present."""
    with _REGISTRY_LOCK:
        return PROVIDERS.pop(name.strip(), None)
