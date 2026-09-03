"""Pure extraction helpers for sealed confirmation provider prompts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_stage_question(stage: Mapping[str, Any]) -> str:
    """Recover the dated question from a sealed provider-ready prompt."""

    messages = stage.get("provider_messages")
    if not isinstance(messages, list):
        raise ValueError("stage provider messages are missing")
    user_contents = [
        item.get("content")
        for item in messages
        if isinstance(item, Mapping) and item.get("role") == "user"
    ]
    if not user_contents or not isinstance(user_contents[-1], str):
        raise ValueError("stage provider prompt has no user question")
    content = user_contents[-1]
    marker = "\n\nQuestion: "
    suffix = "\nShort answer:"
    if marker not in content or suffix not in content:
        raise ValueError("cannot recover question from sealed provider prompt")
    question = content.rsplit(marker, 1)[1].rsplit(suffix, 1)[0].strip()
    if not question:
        raise ValueError("sealed provider prompt contains an empty question")
    return question


__all__ = ["extract_stage_question"]
