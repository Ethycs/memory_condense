"""Optional provider binding for LLM-proposed memory.

:class:`~memory_condense.ingest.extractor.LLMExtractor` takes an injected
``complete(system_prompt, user_prompt) -> str`` so the core package stays
provider-agnostic. This module is the one place that supplies such a callable,
and it is the *only* core module allowed to touch an LLM SDK — ``import
litellm`` happens inside the function body, never at module scope, so importing
``memory_condense`` still costs nothing and needs no credentials.

Selection is environment-driven::

    MEMORY_CONDENSE_EXTRACTOR = rules | llm | auto     (default: rules)
    MEMORY_CONDENSE_LLM_MODEL = <litellm model string> (default: Haiku 4.5)

**The default is deliberately ``rules``.** Auto-extraction fires on every
``ingest``, so making the LLM the default would spend money on every MCP tool
call without the user ever asking for it. ``auto`` opts in when a key is
present; ``llm`` asks for it explicitly and still falls back rather than
failing.

**Nothing here raises.** A missing key, an unknown mode, an SDK that will not
import — every path returns the rule-based extractor plus a one-line reason for
the caller to log. This matters more than it looks: ``docs/02 - Implementation/02``
records that the MCP stdio client does not forward the parent environment to
the server, so "no key present" is the *normal* case, not the edge case.

Why bind it at all: ``08 - Analysis/01`` measured what the rule-based extractor
produces on real prose — 65% ``Constraint`` from ordinary technical modality
("the map must be smooth"), 93% sourced from assistant turns, and 8 ``Decision``
or ``Preference`` items out of 4,463. That is the evidence the roadmap wanted
before promoting ``LLMExtractor`` out of gated status.
"""

from __future__ import annotations

import os
from typing import Callable, Literal

from memory_condense.ingest.extractor import Extractor, LLMExtractor, RuleBasedExtractor

#: Model used when nothing overrides it. Matches the eval harness's responder
#: default so a project only has to think about one model tier.
DEFAULT_MODEL = "anthropic/claude-haiku-4-5"

#: Upper bound on a memory_ops response. Generous because the schema is a JSON
#: object with one entry per extracted fact, and truncation loses memories
#: silently — `parse_memory_ops` returns empty on unparsable output.
DEFAULT_MAX_TOKENS = 4096

ExtractorMode = Literal["rules", "llm", "auto"]

#: Environment variable holding the API key for each litellm provider prefix.
#: Only the providers this project has actually been exercised against are
#: listed; an unknown prefix is treated as "cannot verify a key", which routes
#: to the rule-based extractor rather than a runtime failure on first ingest.
_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
    "bedrock": "AWS_ACCESS_KEY_ID",
}


def _load_env() -> None:
    """Best-effort ``.env`` load. Never fatal, never at import time."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv is a hard dep, but be safe
        pass


def api_key_present(model: str = DEFAULT_MODEL) -> bool:
    """Whether the environment carries a usable key for ``model``'s provider."""
    provider = model.split("/", 1)[0] if "/" in model else "anthropic"
    env_var = _PROVIDER_KEYS.get(provider)
    if env_var is None:
        return False
    return bool(os.environ.get(env_var))


def make_completer(
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    completion_fn: Callable | None = None,
) -> Callable[[str, str], str]:
    """Build the ``(system_prompt, user_prompt) -> str`` callable.

    ``completion_fn`` overrides the litellm call, so tests can exercise the
    binding without a network or a key.

    No ``temperature`` is passed, for the reason spelled out in
    ``eval/judge.py``: several current Claude models reject non-default
    sampling parameters with a 400. Steer the extractor with its prompt.
    """

    def complete(system_prompt: str, user_prompt: str) -> str:
        fn = completion_fn
        if fn is None:
            import litellm  # local import: keeps the SDK out of module scope

            fn = litellm.completion

        response = fn(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            num_retries=3,
        )
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return ""

    return complete


def resolve_extractor(
    mode: ExtractorMode | None = None,
    model: str | None = None,
    completion_fn: Callable | None = None,
) -> tuple[Extractor, str]:
    """Pick an extractor from the environment. Returns ``(extractor, reason)``.

    ``reason`` is a short line for the caller to log once at startup, so the
    choice — and any silent fallback — is visible rather than inferred from
    behaviour.
    """
    _load_env()

    mode = (mode or os.environ.get("MEMORY_CONDENSE_EXTRACTOR", "rules")).strip().lower()
    model = model or os.environ.get("MEMORY_CONDENSE_LLM_MODEL", DEFAULT_MODEL)

    if mode == "rules":
        return RuleBasedExtractor(), "rule-based extraction (default)"

    if mode not in ("llm", "auto"):
        return (
            RuleBasedExtractor(),
            f"rule-based extraction (unknown MEMORY_CONDENSE_EXTRACTOR={mode!r})",
        )

    if completion_fn is None and not api_key_present(model):
        provider = model.split("/", 1)[0] if "/" in model else "anthropic"
        env_var = _PROVIDER_KEYS.get(provider, f"<{provider} key>")
        return (
            RuleBasedExtractor(),
            f"rule-based extraction ({mode!r} requested but {env_var} is not set)",
        )

    try:
        completer = make_completer(model=model, completion_fn=completion_fn)
        return LLMExtractor(complete=completer), f"LLM extraction via {model}"
    except Exception as exc:  # pragma: no cover - defensive
        return (
            RuleBasedExtractor(),
            f"rule-based extraction (could not bind {model}: {exc})",
        )
