"""Admit routing summaries against complete authenticated raw source spans.

Summary text has no factual authority. Model-generated quote strings are
diagnostics, not entailment proofs. Speaker, time, coordinates and hydration
identity always come from the input span, never from model output.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_batch_summary import RawSummaryFragment, batch_messages


@dataclass(frozen=True, slots=True)
class SourceBoundSummaryBatch(SealedIdentity):
    atoms: tuple[SectionSummary, ...]
    quote_diagnostics: tuple[dict, ...]
    summary_entailment_verified: bool = False
    factual_authority: str = "none; route then hydrate the complete exact raw section"
    receipt_sha256: str = ""

    def __post_init__(self):
        if self.summary_entailment_verified or self.factual_authority != "none; route then hydrate the complete exact raw section":
            raise ValueError("source binding cannot certify summary entailment")
        self._seal()


def admit_source_bound_summaries(response: str, fragments: Sequence[RawSummaryFragment], *,
                                 compiler_identity: str) -> SourceBoundSummaryBatch:
    """Require full coverage, attribution, bounded summaries, and exact raw input.

    Retain original generated quote failures explicitly. No fuzzy matching,
    support deletion, summary rewriting, raw truncation, or provider calls occur.
    Invalid summary/schema/attribution still rejects the complete batch.
    """
    batch_messages(fragments)
    body = json.loads(response)
    if type(body) is not dict or set(body) != {"atoms"} or type(body["atoms"]) is not list or len(body["atoms"]) != len(fragments):
        raise ValueError("one summary is required for every raw fragment")
    atoms, audits = [], []
    for i, (row, fragment) in enumerate(zip(body["atoms"], fragments, strict=True)):
        if type(row) is not dict or set(row) != {"label", "summary", "support"} or row["label"] != f"T{i}":
            raise ValueError("summary attribution or schema changed")
        summary = row["summary"]
        if type(summary) is not str or not summary.strip() or count_tokens(summary) > 128:
            raise ValueError("routing summary exceeds its output budget or is empty")
        support = row["support"]
        failures = []
        if type(support) is not list or not 1 <= len(support) <= 4:
            failures.append({"quote_index": None, "reason": "quote_list_shape"})
        if type(support) is list:
            for j, quote in enumerate(support):
                if type(quote) is not str or not quote.strip():
                    failures.append({"quote_index": j, "reason": "quote_type_or_empty"})
                else:
                    if quote not in fragment.text:
                        failures.append({"quote_index": j, "reason": "quote_not_exact"})
                    if count_tokens(quote) > 32:
                        failures.append({"quote_index": j, "reason": "quote_token_budget"})
        atoms.append(SectionSummary("spine-atom-" + fragment.span.receipt_sha256,
            fragment.span.source_id, summary, (fragment.span,), compiler_identity))
        audits.append({"atom_index": i, "raw_span_receipt_sha256": fragment.span.receipt_sha256,
            "original_reported_support": support, "failures": failures,
            "authoritative_hydration_scope": "entire_input_fragment", "summary_text_unchanged": True})
    return SourceBoundSummaryBatch(tuple(atoms), tuple(audits))
