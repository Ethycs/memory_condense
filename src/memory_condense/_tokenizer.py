from __future__ import annotations

import tiktoken

_encoder: tiktoken.Encoding | None = None

DEFAULT_ENCODING = "cl100k_base"


def _get_encoder(encoding: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(encoding)
    return _encoder


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    """Count BPE tokens in text using tiktoken.

    Uses cl100k_base (GPT-4 family) as a reasonable proxy
    for token budgets across modern LLMs.
    """
    return len(_get_encoder(encoding).encode(text))


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
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
