"""Lazy model resolution from ``"provider:model"`` spec strings."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from langchain_core.language_models import BaseChatModel

from langcomp.errors import (
    MissingApiKeyError,
    ProviderNotInstalledError,
    UnknownProviderError,
)
from langcomp.models.constants import DEFAULT_MODEL
from langcomp.models.registry import PROVIDERS, known_providers
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
    """
    if isinstance(model, BaseChatModel):
        return model

    provider, model_id = parse_model_spec(model or DEFAULT_MODEL)
    try:
        provider_spec = PROVIDERS[provider]
    except KeyError:
        raise UnknownProviderError(provider, known_providers()) from None

    try:
        module = import_module(provider_spec.module)
    except ImportError as exc:
        raise ProviderNotInstalledError(
            provider, provider_spec.module, provider_spec.extra
        ) from exc

    model_cls: type[BaseChatModel] = getattr(module, provider_spec.class_name)
    init_kwargs: dict[str, Any] = {
        provider_spec.model_kwarg: model_id,
        **provider_spec.defaults,
        **kwargs,
    }

    if provider_spec.api_key_env is not None and "api_key" not in init_kwargs:
        api_key = os.getenv(provider_spec.api_key_env)
        if not api_key:
            raise MissingApiKeyError(provider, provider_spec.api_key_env)
        init_kwargs["api_key"] = api_key

    return model_cls(**init_kwargs)
