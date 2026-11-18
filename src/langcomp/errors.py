"""Typed exception hierarchy for the langcomp framework.

Every error raised by langcomp inherits from :class:`LangcompError`, allowing
callers to catch all framework errors cleanly while still handling specific
failure modes individually.
"""

from __future__ import annotations


class LangcompError(Exception):
    """Base class for all langcomp framework errors."""


class ConfigurationError(LangcompError):
    """Raised when langcomp is configured in a way it cannot act on."""


class InvalidModelSpecError(ConfigurationError):
    """Raised when a ``"provider:model"`` specification string cannot be parsed."""

    def __init__(self, spec: str, reason: str) -> None:
        self.spec = spec
        self.reason = reason
        super().__init__(f"Invalid model spec {spec!r}: {reason}")


class UnknownProviderError(ConfigurationError):
    """Raised when a model spec names a provider not registered with langcomp."""

    def __init__(self, provider: str, known: tuple[str, ...]) -> None:
        self.provider = provider
        self.known = known
        super().__init__(
            f"Unknown model provider {provider!r}. Known providers: {', '.join(sorted(known))}."
        )


class ProviderNotInstalledError(ConfigurationError):
    """Raised when a known provider's integration SDK package is not installed."""

    def __init__(self, provider: str, module: str, extra: str) -> None:
        self.provider = provider
        self.module = module
        self.extra = extra
        super().__init__(
            f"Provider {provider!r} needs the {module!r} package, which is not installed. "
            f"Install it with: pip install 'langcomp[{extra}]'"
        )


class MissingApiKeyError(ConfigurationError):
    """Raised when a gateway provider's API key environment variable is unset."""

    def __init__(self, provider: str, env_var: str) -> None:
        self.provider = provider
        self.env_var = env_var
        super().__init__(
            f"Provider {provider!r} needs an API key. Set {env_var} in your environment "
            f"or pass api_key=... to resolve_model()."
        )


class ReducerConflictError(LangcompError):
    """Raised when a State field's reducer configuration is contradictory."""


class SnapshotMismatchError(LangcompError):
    """Raised when a replayed run's output differs from its recorded snapshot."""


class ScriptExhaustedError(LangcompError):
    """Raised when FakeModel is invoked more times than its script provides."""


class ApprovalRejectedError(LangcompError):
    """Raised when an ApprovalGate rejects a tool or function execution."""


class UnknownToolError(LangcompError):
    """Raised when a model requests a tool that is not registered on the agent."""

    def __init__(self, tool_name: str, known: tuple[str, ...]) -> None:
        self.tool_name = tool_name
        self.known = known
        super().__init__(
            f"Model called unknown tool {tool_name!r}. Registered tools: "
            f"{', '.join(sorted(known)) or '(none)'}."
        )
