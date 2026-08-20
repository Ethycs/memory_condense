"""Shared litellm request construction and response reading for eval calls.

Every eval call site assembles its ``litellm.completion`` kwargs through
:func:`build_completion_request` so the codex-route sampling guard is applied
uniformly, and reads assistant text through :func:`_content` so a ``None``
content (refusal, content filter, ``max_tokens`` stop before visible text)
cannot raise ``AttributeError`` deep in a paid run.

This module is a leaf on purpose: it imports nothing from the eval package,
so ``judge``, ``responder``, and ``provider_runtime`` can all use it without
an import cycle. It also imports no provider SDK — each call site keeps its
own ``import litellm`` so tests can patch the module actually making the call.
"""

from __future__ import annotations

from typing import Any


def build_completion_request(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    num_retries: int,
    temperature: float | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Assemble ``litellm.completion`` kwargs with the codex-route guard.

    Codex GPT-5 routes (``codex_sdk/`` models on the central-dev gateway)
    reject non-default sampling parameters with a 400, so ``temperature`` is
    omitted for those models even when the caller supplies one; historical
    model routes still receive it. Pass ``temperature=None`` for callers
    (judges) that must never send sampling parameters at all.
    """
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "num_retries": num_retries,
    }
    if temperature is not None and "codex_sdk/" not in model:
        request["temperature"] = temperature
    if client is not None:
        request["client"] = client
    return request


def _content(response: Any) -> str:
    """The assistant text, or "" if the provider returned none.

    A refusal, a content filter, or a ``max_tokens`` stop before any visible
    text all yield ``content=None``. Reaching ``.strip()`` on that raises
    ``AttributeError`` deep in a paid run, after every preceding call has
    already been billed.
    """
    try:
        return (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        return ""
