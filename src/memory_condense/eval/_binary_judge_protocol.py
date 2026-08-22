"""Provider-free constants and parsing for the binary semantic judge."""

from __future__ import annotations

import re


# The official judge needs enough room for provider-side reasoning plus the
# visible verdict.  Keep this value in a provider-free leaf so replay and
# preflight imports cannot trigger LiteLLM initialization or network egress.
JUDGE_MAX_TOKENS = 1024

_BINARY_JUDGE_VERDICT = re.compile(
    r"^\s*(CORRECT|INCORRECT)\b(?P<remainder>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_binary_judge_verdict(text: str) -> bool:
    """Parse one unambiguous judge label and reject protocol noise."""

    match = _BINARY_JUDGE_VERDICT.match(text or "")
    if match is None:
        raise RuntimeError("judge returned an empty or malformed verdict")
    remainder = match.group("remainder").lstrip(" \t\r\n,.:;-—")
    if remainder.casefold().startswith("or ") or remainder.startswith("/"):
        raise RuntimeError("judge returned an ambiguous verdict")
    return match.group(1).casefold() == "correct"


__all__ = ["JUDGE_MAX_TOKENS", "parse_binary_judge_verdict"]
