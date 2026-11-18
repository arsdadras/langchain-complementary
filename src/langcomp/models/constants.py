"""Model naming constants and defaults for langcomp.

Defines standard tiered model identifiers and the provider/model separator used
across the framework.
"""

from __future__ import annotations

from typing import Final

#: Fast, economical default model for routine calls via OpenRouter.
DEFAULT_MODEL: Final[str] = "openrouter:nvidia/nemotron-3.5-lightning"

#: Stronger model tier for complex reasoning calls via OpenRouter.
CAPABLE_MODEL: Final[str] = "openrouter:openai/gpt-5.6-terra"

#: Separator between provider prefix and model id in ``"provider:model"`` specs.
SPEC_SEPARATOR: Final[str] = ":"
