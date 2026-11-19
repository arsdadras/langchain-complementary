"""Provider specification structures and ``provider:model`` string parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from langcomp.errors import ConfigurationError, InvalidModelSpecError
from langcomp.models.constants import SPEC_SEPARATOR


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Metadata describing how to lazily instantiate a provider's chat model.

    Attributes:
        module: Import path of the integration package (e.g. ``"langchain_openai"``).
        class_name: Name of the ``BaseChatModel`` subclass inside ``module``.
        extra: The langcomp optional-dependency extra that installs ``module``.
        model_kwarg: Constructor argument name that receives the model id.
        api_key_env: Optional environment variable holding this provider's API key.
            When set, langcomp reads and passes it explicitly so gateway providers
            never silently pick up another provider's key.
        api_key_kwarg: Constructor argument name that receives the API key.
        defaults: Default constructor arguments applied unless overridden by caller.
    """

    module: str
    class_name: str
    extra: str
    model_kwarg: str = "model"
    api_key_env: str | None = None
    api_key_kwarg: str = "api_key"
    defaults: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.module or not self.module.strip():
            raise ConfigurationError("ProviderSpec.module must not be empty.")
        if not self.class_name or not self.class_name.strip():
            raise ConfigurationError("ProviderSpec.class_name must not be empty.")
        if not self.model_kwarg or not self.model_kwarg.strip():
            raise ConfigurationError("ProviderSpec.model_kwarg must not be empty.")
        if not self.api_key_kwarg or not self.api_key_kwarg.strip():
            raise ConfigurationError("ProviderSpec.api_key_kwarg must not be empty.")
        if not isinstance(self.defaults, MappingProxyType):
            object.__setattr__(self, "defaults", MappingProxyType(dict(self.defaults)))


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split a ``"provider:model"`` string on the first separator.

    Args:
        spec: The specification string, for example ``"openai:gpt-4o-mini"``
            or ``"openai:ft:my-model:abc"``.

    Returns:
        A ``(provider, model_id)`` pair with surrounding whitespace stripped.

    Raises:
        InvalidModelSpecError: If ``spec`` is not a string, is empty, missing the
            separator, or has an empty provider or model segment.
    """
    if not isinstance(spec, str):
        raise InvalidModelSpecError(
            repr(spec), f"expected a string, got {type(spec).__name__}"
        )
    text = spec.strip()
    if not text:
        raise InvalidModelSpecError(spec, "the spec is empty")
    if SPEC_SEPARATOR not in text:
        raise InvalidModelSpecError(spec, f"expected the form 'provider{SPEC_SEPARATOR}model'")

    provider, _, model_id = text.partition(SPEC_SEPARATOR)
    provider, model_id = provider.strip(), model_id.strip()
    if not provider:
        raise InvalidModelSpecError(spec, "the provider half is empty")
    if not model_id:
        raise InvalidModelSpecError(spec, "the model half is empty")
    return provider, model_id
