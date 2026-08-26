"""Exact downstream synthesis-scaffold projection used by Hebbian H2."""

from __future__ import annotations

import json
from typing import Sequence

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._fast_hebbian_h2_io import (
    FastHebbianH2ValidationError,
)
from memory_condense.eval.fast_cav_link_synthesis import (
    _GUIDE_SLOT_SENTINEL,
    _messages as _canonical_synthesis_messages,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastEvidence,
)


def _catalog(evidence: Sequence[FastEvidence]) -> str:
    rows: list[str] = []
    for ordinal, row in enumerate(evidence, start=1):
        source = json.dumps(
            row.source_id,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        rows.append(f"[E{ordinal:03d}] source_id={source}\n{row.text}")
    return "Canonical S3 evidence catalog:\n\n" + "\n\n".join(rows)


def build_fast_hebbian_h2_scaffold(
    evidence: Sequence[FastEvidence],
    dated_question: str,
) -> tuple[str, str, int]:
    """Return the exact catalog, message, and token-proxy identities."""

    catalog = _catalog(evidence)
    messages = _canonical_synthesis_messages(
        dated_question=dated_question,
        catalog=catalog,
        guide=_GUIDE_SLOT_SENTINEL,
    )
    tokens = count_chat_prompt_token_proxy(messages)
    if type(tokens) is not int or tokens < 1:
        raise FastHebbianH2ValidationError("downstream token proxy is invalid")
    return quote_sha256(catalog), identity_sha256(list(messages)), tokens


__all__ = ["build_fast_hebbian_h2_scaffold"]
