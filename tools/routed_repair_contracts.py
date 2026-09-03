"""Import-inert public constants and errors for routed repair prompts."""

from __future__ import annotations


ROUTED_REPAIR_PROMPT_FORMAT = "memory-condense-routed-repair-prompt-v1"
MAX_ROUTED_PROMPT_TOKENS = 8_000


class RoutedRepairPromptError(ValueError):
    """Raised when a routed request loses identity or budget integrity."""


__all__ = [
    "MAX_ROUTED_PROMPT_TOKENS",
    "ROUTED_REPAIR_PROMPT_FORMAT",
    "RoutedRepairPromptError",
]
