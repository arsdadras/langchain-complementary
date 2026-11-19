"""Lazy model resolution from ``"provider:model"`` spec strings."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from langchain_core.language_models import BaseChatModel

from langcomp.errors import (
    ConfigurationError,
    MissingApiKeyError,
    ProviderNotInstalledError,
    UnknownProviderError,
)
from langcomp.models.constants import DEFAULT_MODEL
from langcomp.models.registry import get_provider, known_providers
from langcomp.models.spec import parse_model_spec


def resolve_model(
    model: str | BaseChatModel | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Resolve a chat model from a ``"provider:model"`` spec string or instance.

    Args:
        model: A ``"provider:model"`` specification string, an already-built
            :class:`~langchain_core.language_models.BaseChatModel` (returned
            unchanged), or ``None`` to use :data:`DEFAULT_MODEL`.
        **kwargs: Extra keyword arguments forwarded to the provider constructor
            (always overriding provider defaults).

    Returns:
        A ready-to-use :class:`~langchain_core.language_models.BaseChatModel`.

    Raises:
        InvalidModelSpecError: The specification string was malformed.
        UnknownProviderError: The provider prefix is not registered.
        ProviderNotInstalledError: The provider SDK package is not installed.
        MissingApiKeyError: A required gateway API key environment variable is unset.
        ConfigurationError: Misconfigured provider class, conflicting kwargs with a
            pre-built model, or unsafe ``base_url`` override without an explicit key.
    """
    if isinstance(model, BaseChatModel):
        if kwargs:
            raise ConfigurationError(
                f"Cannot pass constructor kwargs ({', '.join(sorted(kwargs))}) "
                f"when resolve_model() receives a pre-built BaseChatModel instance."
            )
        return model

    spec_str = DEFAULT_MODEL if model is None else model
    provider, model_id = parse_model_spec(spec_str)

    provider_spec = get_provider(provider)
    if provider_spec is None:
        raise UnknownProviderError(provider, known_providers())

    try:
        module = import_module(provider_spec.module)
    except ModuleNotFoundError as exc:
        top_pkg = provider_spec.module.split(".")[0]
        if exc.name in (provider_spec.module, top_pkg):
            raise ProviderNotInstalledError(
                provider, provider_spec.module, provider_spec.extra
            ) from exc
        raise

    model_cls = getattr(module, provider_spec.class_name, None)
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseChatModel):
        raise ConfigurationError(
            f"Provider {provider!r} target {provider_spec.module}.{provider_spec.class_name} "
            f"is not a BaseChatModel subclass."
        )

    init_kwargs: dict[str, Any] = {
        provider_spec.model_kwarg: model_id,
        **provider_spec.defaults,
        **kwargs,
    }

    if provider_spec.api_key_env is not None:
        key_kwarg = provider_spec.api_key_kwarg
        explicit_key = kwargs.get(key_kwarg)
        has_explicit_key = isinstance(explicit_key, str) and bool(explicit_key.strip())

        if (
            "base_url" in kwargs
            and "base_url" in provider_spec.defaults
            and kwargs["base_url"] != provider_spec.defaults["base_url"]
            and not has_explicit_key
        ):
            raise ConfigurationError(
                f"Refusing to auto-inject {provider_spec.api_key_env} when base_url is overridden "
                f"for provider {provider!r}; pass {key_kwarg}=... explicitly."
            )

        raw_key = init_kwargs.get(key_kwarg)
        if raw_key is None or (isinstance(raw_key, str) and not raw_key.strip()):
            env_key = (os.getenv(provider_spec.api_key_env) or "").strip()
            if not env_key:
                raise MissingApiKeyError(provider, provider_spec.api_key_env)
            init_kwargs[key_kwarg] = env_key
        elif isinstance(raw_key, str):
            init_kwargs[key_kwarg] = raw_key.strip()

    return model_cls(**init_kwargs)
