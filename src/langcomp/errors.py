"""Exception hierarchy for langcomp.

Every error the framework raises inherits from :class:`LangCompError`, so callers
can catch everything langcomp raises with a single ``except`` clause while still
handling specific failure modes when desired.
"""

from __future__ import annotations


class LangCompError(Exception):
    """Base class for every error raised by langcomp."""


class ConfigurationError(LangCompError):
    """The framework was configured in a way it cannot act on."""


class InvalidModelSpecError(ConfigurationError):
    """A model specification string could not be parsed.

    Args:
        spec: The specification string that failed to parse.
        reason: Human-readable explanation of what was wrong with it.
    """

    def __init__(self, spec: str, reason: str) -> None:
        self.spec = spec
        self.reason = reason
        super().__init__(f"Invalid model spec {spec!r}: {reason}")


class UnknownProviderError(ConfigurationError):
    """A model spec named a provider langcomp does not know about.

    Args:
        provider: The provider name taken from the spec.
        known: The provider names langcomp currently supports.
    """

    def __init__(self, provider: str, known: tuple[str, ...]) -> None:
        self.provider = provider
        self.known = known
        super().__init__(
            f"Unknown model provider {provider!r}. Known providers: {', '.join(sorted(known))}."
        )


class ProviderNotInstalledError(ConfigurationError):
    """A known provider was requested but its integration package is missing.

    Args:
        provider: The provider name taken from the spec.
        module: The Python module that failed to import.
        extra: The langcomp optional extra that installs the missing package.
    """

    def __init__(self, provider: str, module: str, extra: str) -> None:
        self.provider = provider
        self.module = module
        self.extra = extra
        super().__init__(
            f"Provider {provider!r} needs the {module!r} package, which is not installed. "
            f"Install it with: pip install 'langcomp[{extra}]'"
        )


class MissingApiKeyError(ConfigurationError):
    """A provider needs an API key and the environment variable holding it is unset.

    Args:
        provider: The provider name taken from the spec.
        env_var: The environment variable langcomp looked in.
    """

    def __init__(self, provider: str, env_var: str) -> None:
        self.provider = provider
        self.env_var = env_var
        super().__init__(
            f"Provider {provider!r} needs an API key. Set {env_var} in your environment "
            f"or pass api_key=... to resolve_model()."
        )


class ScriptExhaustedError(LangCompError):
    """A :class:`~langcomp.devtools.testing.ScriptedFakeChatModel` ran out of responses.

    Args:
        call_count: How many responses were scripted (and already consumed).
    """

    def __init__(self, call_count: int) -> None:
        self.call_count = call_count
        super().__init__(
            f"ScriptedFakeChatModel received its {call_count + 1}th call but only "
            f"{call_count} response(s) were scripted. Add more responses to the script."
        )


class UnknownToolError(LangCompError):
    """The model called a tool that was never registered with the agent.

    Args:
        tool_name: The name the model tried to call.
        known: The tool names actually registered.
    """

    def __init__(self, tool_name: str, known: tuple[str, ...]) -> None:
        self.tool_name = tool_name
        self.known = known
        super().__init__(
            f"Model called unknown tool {tool_name!r}. Registered tools: "
            f"{', '.join(sorted(known)) or '(none)'}."
        )
