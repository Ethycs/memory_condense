from __future__ import annotations

import hashlib
import importlib.metadata
from functools import lru_cache
from typing import Mapping, Sequence

import tiktoken

_encoders: dict[str, tiktoken.Encoding] = {}

DEFAULT_ENCODING = "cl100k_base"

# This is deliberately a provider-independent *proxy*.  It gives local
# packing a deterministic hard ceiling, but it is not an assertion about any
# provider's private tokenizer or chat template.  The fixed framing allowance
# is intentionally larger than the common OpenAI-style 3--4 tokens/message
# heuristic and is reported separately from message-content tokens.
PROMPT_TOKEN_PROXY_SCHEMA = "memory-condense-prompt-token-proxy-v1"
CHAT_FRAMING_TOKENS_PER_MESSAGE = 8
CHAT_FRAMING_TOKENS_FIXED = 8


def _get_encoder(encoding: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    encoder = _encoders.get(encoding)
    if encoder is None:
        encoder = tiktoken.get_encoding(encoding)
        _encoders[encoding] = encoder
    return encoder


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    """Count BPE tokens in text using tiktoken.

    Uses cl100k_base (GPT-4 family) as a reasonable proxy
    for token budgets across modern LLMs.
    """
    # Corpus text is untrusted data. Strings such as ``<|endoftext|>`` may
    # occur literally in web/chat exports and must be budgeted as ordinary
    # text, never interpreted as tokenizer control input.
    return len(_get_encoder(encoding).encode(text, disallowed_special=()))


def count_chat_prompt_token_proxy(
    messages: Sequence[Mapping[str, str]],
    encoding: str = DEFAULT_ENCODING,
) -> int:
    """Return the deterministic local proxy for a chat request's input.

    The count is message-content BPE tokens plus a fixed, explicit framing
    reserve.  It is suitable for a hard *local proxy* cap.  Provider-reported
    input usage, when available, remains the authoritative post-request count.
    """

    content_tokens = sum(
        count_tokens(str(message.get("content", "")), encoding=encoding)
        for message in messages
    )
    return (
        content_tokens
        + CHAT_FRAMING_TOKENS_PER_MESSAGE * len(messages)
        + CHAT_FRAMING_TOKENS_FIXED
    )


@lru_cache(maxsize=None)
def _tokenizer_proxy_identity_items(
    encoding: str = DEFAULT_ENCODING,
) -> tuple[tuple[str, str | int], ...]:
    """Identify the exact local vocabulary used for token-budget proxies.

    Encoding name alone is not a sufficient frozen identity.  The digest walks
    the encoding's mergeable ranks and special-token table, so a vocabulary
    change cannot silently retain the same benchmark policy identity.  The
    environment lock separately binds the installed tiktoken implementation.
    """

    encoder = _get_encoder(encoding)
    digest = hashlib.sha256()
    digest.update(PROMPT_TOKEN_PROXY_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(encoding.encode("utf-8"))
    digest.update(b"\0")
    # tiktoken exposes no public vocabulary export.  These two immutable
    # tables are the data from which Encoding performs tokenization.
    mergeable_ranks = getattr(encoder, "_mergeable_ranks", None)
    special_tokens = getattr(encoder, "_special_tokens", None)
    if not isinstance(mergeable_ranks, dict) or not isinstance(
        special_tokens, dict
    ):
        raise RuntimeError("tiktoken Encoding does not expose identity tables")
    for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1]):
        digest.update(int(rank).to_bytes(8, "big", signed=False))
        digest.update(len(token).to_bytes(8, "big", signed=False))
        digest.update(token)
    for token, rank in sorted(special_tokens.items()):
        encoded = token.encode("utf-8")
        digest.update(int(rank).to_bytes(8, "big", signed=False))
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return tuple({
        "schema": PROMPT_TOKEN_PROXY_SCHEMA,
        "implementation": "tiktoken",
        "implementation_version": importlib.metadata.version("tiktoken"),
        "encoding": encoding,
        "vocabulary_sha256": digest.hexdigest(),
        "chat_framing_tokens_per_message": CHAT_FRAMING_TOKENS_PER_MESSAGE,
        "chat_framing_tokens_fixed": CHAT_FRAMING_TOKENS_FIXED,
    }.items())


def tokenizer_proxy_identity(
    encoding: str = DEFAULT_ENCODING,
) -> dict[str, str | int]:
    """Return a fresh mapping for the cached immutable vocabulary identity.

    Callers may safely modify their copy without corrupting identities emitted
    by later requests.
    """

    return dict(_tokenizer_proxy_identity_items(encoding))


def truncate_to_tokens(
    text: str, max_tokens: int, encoding: str = DEFAULT_ENCODING
) -> str:
    """Cut text down to at most ``max_tokens`` tokens.

    Truncation happens on token boundaries, so the result decodes cleanly even
    mid-word. Returns the text unchanged when it already fits.
    """
    if max_tokens <= 0:
        return ""
    enc = _get_encoder(encoding)
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
