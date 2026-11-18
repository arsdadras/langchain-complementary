"""Lazy provider registry and model resolution for langcomp."""

from __future__ import annotations

from langcomp.models.constants import CAPABLE_MODEL, DEFAULT_MODEL, SPEC_SEPARATOR
from langcomp.models.registry import PROVIDERS, known_providers, register_provider
from langcomp.models.resolver import resolve_model
from langcomp.models.spec import ProviderSpec, parse_model_spec

__all__ = [
    "CAPABLE_MODEL",
    "DEFAULT_MODEL",
    "PROVIDERS",
    "SPEC_SEPARATOR",
    "ProviderSpec",
    "known_providers",
    "parse_model_spec",
    "register_provider",
    "resolve_model",
]
