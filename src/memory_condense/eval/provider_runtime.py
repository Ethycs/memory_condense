"""Stateful provider adapters for evaluation answerers and judges."""

from __future__ import annotations

import os
import re
import time

from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    build_judge_prompt,
)
from memory_condense.eval.judge import JUDGE_MAX_TOKENS
from memory_condense.eval.schemas import UsageStats

def _content(response) -> str:
    """The assistant text, or "" if the provider returned none.

    A refusal, a content filter, or a `max_tokens` stop before any visible text
    all yield ``content=None``. Reaching ``.strip()`` on that raises
    ``AttributeError`` deep in a paid run, after every preceding call has
    already been billed.
    """
    try:
        return (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        return ""


_BINARY_JUDGE_VERDICT = re.compile(
    r"^\s*(CORRECT|INCORRECT)\b(?P<remainder>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _parse_binary_judge_verdict(text: str) -> bool:
    """Parse one unambiguous judge label and reject provider/protocol noise."""

    match = _BINARY_JUDGE_VERDICT.match(text or "")
    if match is None:
        raise RuntimeError("judge returned an empty or malformed verdict")
    remainder = match.group("remainder").lstrip(" \t\r\n,.:;-—")
    if remainder.casefold().startswith("or ") or remainder.startswith("/"):
        raise RuntimeError("judge returned an ambiguous verdict")
    return match.group(1).casefold() == "correct"


def _make_central_dev_client(model: str):
    """Return the trusted OpenAI-compatible client for central-dev routes.

    LiteLLM otherwise constructs its own certifi-backed transport, which does
    not see the internal Caddy CA installed in the Windows trust store.  Keep
    this in one place so answer, judge, and sufficiency calls cannot silently
    diverge onto different transports.
    """
    api_base = os.environ.get("OPENAI_API_BASE", "") or os.environ.get(
        "LITELLM_API_BASE", ""
    )
    api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get(
        "LITELLM_KEY", ""
    )
    # The codex_sdk namespace is served by the central-dev v1 gateway. A
    # checked-in command should work with the gateway-native LITELLM_KEY name;
    # requiring callers to duplicate it into OPENAI_API_KEY made normal pixi
    # runs fall through to LiteLLM's unconfigured generic OpenAI transport.
    if not api_base and model.startswith("openai/codex_sdk/") and api_key:
        api_base = "https://central-dev.zt:4000/v1"
    if not model.startswith("openai/") or "central-dev.zt" not in api_base:
        return None
    if not api_key:
        return None

    import ssl

    import httpx
    import truststore
    from openai import OpenAI

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return OpenAI(
        api_key=api_key,
        base_url=api_base,
        http_client=httpx.Client(verify=ssl_context),
        # ``num_retries`` is controlled and budgeted by this harness. Letting
        # the nested SDK retry too would exceed --max-provider-calls silently.
        max_retries=0,
    )


def _make_answer_fn(
    model: str,
    *,
    retries: int = 0,
    client_factory=_make_central_dev_client,
):
    """Answer a benchmark question. Short, deterministic answers — F1/EM depend on it."""
    import litellm

    central_dev_client = client_factory(model)

    def answer_fn(
        messages: list[dict[str, str]],
    ) -> tuple[str, UsageStats]:
        started = time.perf_counter()
        request = {
            "model": model,
            "messages": messages,
            "max_tokens": BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
            "num_retries": retries,
        }
        # Codex GPT-5 routes reject non-default temperature values. Omitting
        # the field keeps the central-dev codex_sdk gateway compatible while
        # preserving deterministic temperature=0 for historical model routes.
        if "codex_sdk/" not in model:
            request["temperature"] = 0.0
        if central_dev_client is not None:
            request["client"] = central_dev_client
        response = litellm.completion(**request)
        content = _content(response)
        if not content:
            raise RuntimeError("responder returned no answer text")
        return content, UsageStats.from_litellm(
            response,
            time.perf_counter() - started,
        )

    return answer_fn


def _make_judge_fn(
    model: str,
    *,
    retries: int = 0,
    client_factory=_make_central_dev_client,
):
    """Semantic-equivalence grading, for answers that F1 scores unfairly.

    ``max_tokens`` is JUDGE_MAX_TOKENS for the reason spelled out in
    ``judge.py``: the default judge is Sonnet 5, which runs adaptive thinking,
    and ``max_tokens`` caps thinking + visible text together. A tight 256 spends
    the whole budget on thinking and returns an empty verdict — which this path
    then scored as INCORRECT for every answer. The replay judge got this fix;
    this one did not, so it is deliberately expressed as the same constant.
    """
    import litellm

    central_dev_client = client_factory(model)

    def judge_fn(
        question: str,
        gold: str,
        prediction: str,
    ) -> tuple[bool, str, UsageStats]:
        started = time.perf_counter()
        request = {
            "model": model,
            "messages": build_judge_prompt(question, gold, prediction),
            "max_tokens": JUDGE_MAX_TOKENS,
            "num_retries": retries,
        }
        if central_dev_client is not None:
            request["client"] = central_dev_client
        response = litellm.completion(**request)
        text = _content(response)
        return (
            _parse_binary_judge_verdict(text),
            text,
            UsageStats.from_litellm(response, time.perf_counter() - started),
        )

    return judge_fn


def _make_sufficiency_fn(
    model: str,
    *,
    retries: int = 0,
    client_factory=_make_central_dev_client,
):
    """Judge whether excerpts can derive the gold answer, not an answer string."""

    import litellm

    from memory_condense.eval.sufficiency import build_sufficiency_prompt

    central_dev_client = client_factory(model)

    def sufficiency_fn(
        question: str,
        gold: str,
        context: list[str],
    ) -> tuple[bool, str, UsageStats]:
        started = time.perf_counter()
        request = {
            "model": model,
            "messages": build_sufficiency_prompt(question, gold, context),
            "max_tokens": JUDGE_MAX_TOKENS,
            "num_retries": retries,
        }
        if central_dev_client is not None:
            request["client"] = central_dev_client
        response = litellm.completion(**request)
        verdict = _content(response)
        return (
            verdict.upper().startswith("SUFFICIENT"),
            verdict,
            UsageStats.from_litellm(response, time.perf_counter() - started),
        )

    return sufficiency_fn
