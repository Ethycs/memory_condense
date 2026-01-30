from __future__ import annotations

import tiktoken

_encoder: tiktoken.Encoding | None = None


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count BPE tokens in text using tiktoken.

    Uses cl100k_base (GPT-4 family) as a reasonable proxy
    for token budgets across modern LLMs.
    """
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(encoding)
    return len(_encoder.encode(text))
