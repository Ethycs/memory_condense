"""One local Qwen attention pass over a bounded, summary-only shortlist.

The hierarchy is compiled at ingest time. A resident summary index supplies the
shortlist; this optional reranker cannot start a recursive model tournament or
hydrate raw text. It fails on partial workspace coverage so callers can retain
their conventional route instead of silently dropping uninspected candidates.
"""
from __future__ import annotations

import math

from memory_condense.associations.head_memory_models import AssociativeMemoryCandidate
from memory_condense.domain._discourse_identity import canonical_json, quote_sha256
from memory_condense.search.episodes.qwen_episode_signal import qwen_linker_identity
from memory_condense.search.section_routing import (
    SectionRoute, SectionRoutePlan, SectionSummaryIndex, SummaryAttentionPass, SummaryAttentionReceipt,
)
from memory_condense.search.section_summary import bound_int, exact_text


def rerank_summary_shortlist(query: str, index: SectionSummaryIndex, shortlist: SectionRoutePlan,
                             *, linker, max_sections: int = 4) -> SectionRoutePlan:
    """Reorder exact summary candidates with one complete QK/OV readout batch.

    shortlist carries the summary-index/query/scope binding. Candidate aliases
    and raw span locators remain outside the Qwen prompt. QK is the primary
    score, OV the tie-breaker, matching inspect_coverage's ordering. Positive
    route weights encode rank; signed QK scores remain in the attention receipt.
    This mechanism is experimental and does not certify an answer or frontier.
    """
    exact_text(query, "query")
    bound_int(max_sections, "max_sections", 1)
    if type(shortlist) is not SectionRoutePlan or shortlist.index_sha256 != index.receipt_sha256 or shortlist.query_sha256 != quote_sha256(query):
        raise ValueError("shortlist index or query binding changed")
    if shortlist.routing_backend not in {"summary_bm25", "summary_dense", "summary_hybrid"}:
        raise ValueError("bounded attention requires a conventional summary shortlist")
    by_id = {section.section_id: section for section in index.sections}
    if any(by_id.get(route.section.section_id) != route.section for route in shortlist.routes):
        raise ValueError("shortlist contains foreign or changed summary descriptors")
    if len(shortlist.routes) > linker.max_candidates:
        raise ValueError("shortlist exceeds one Qwen workspace; reduce it before attention")
    if not callable(getattr(linker, "inspect_coverage", None)):
        raise TypeError("Qwen shortlist attention requires inspect_coverage")
    identity = canonical_json(qwen_linker_identity(linker, strict=True))
    if not shortlist.routes:
        return SectionRoutePlan(index.receipt_sha256, quote_sha256(query), (), 0,
            shortlist.eligible_source_ids, max_sections, routing_backend="qwen_hierarchical_summaries",
            attention_receipt=SummaryAttentionReceipt(identity, (), 1))
    candidates = tuple(AssociativeMemoryCandidate(episode_id=f"summary-{i}", text=route.section.summary,
                       route="section_summary") for i, route in enumerate(shortlist.routes))
    aliases = {c.episode_id: route for c, route in zip(candidates, shortlist.routes, strict=True)}
    inspection = linker.inspect_coverage(query, candidates)
    if (inspection.passes != 1 or inspection.total_candidate_inspections != len(candidates) or
        inspection.workspace_candidates != len(candidates) or inspection.workspace_tokens > linker.max_workspace_tokens):
        raise ValueError("one-pass attention did not inspect the entire bounded shortlist")
    hits = tuple(inspection.hits)
    if len(hits) != len(candidates) or {h.episode_id for h in hits} != set(aliases):
        raise ValueError("attention omitted, duplicated, or introduced a summary")
    if any(not math.isfinite(h.qk_score) or not math.isfinite(h.ov_transport) or h.ov_transport < 0 for h in hits):
        raise ValueError("invalid QK/OV summary scores")
    if canonical_json(qwen_linker_identity(linker, strict=True)) != identity:
        raise ValueError("Qwen identity changed during attention")
    selected = sorted(hits, key=lambda h: (-h.qk_score, -h.ov_transport, aliases[h.episode_id].section.section_id))[:max_sections]
    rounds = (SummaryAttentionPass(
        tuple(route.section.section_id for route in shortlist.routes),
        tuple(aliases[h.episode_id].section.section_id for h in selected),
        tuple(float(h.qk_score) for h in selected), tuple(float(h.ov_transport) for h in selected),
        1, inspection.workspace_candidates, inspection.workspace_tokens, inspection.total_candidate_inspections,
    ),)
    routes = tuple(SectionRoute(aliases[h.episode_id].section, 1.0 / (rank + 1),
                               aliases[h.episode_id].matched_terms) for rank, h in enumerate(selected))
    # Return only scalar scores and authenticated pointers, never the transient
    # transport signatures or model tensors present in an inspection result.
    return SectionRoutePlan(index.receipt_sha256, quote_sha256(query), routes,
        len(candidates), shortlist.eligible_source_ids, max_sections,
        routing_backend="qwen_hierarchical_summaries",
        attention_receipt=SummaryAttentionReceipt(identity, rounds, 1))
