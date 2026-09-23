"""Qwen attention over hierarchical summaries; no transcript access capability."""

from __future__ import annotations

import math
from collections.abc import Sequence

from memory_condense.associations.head_memory_models import AssociativeMemoryCandidate
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.domain._discourse_identity import canonical_json, quote_sha256
from memory_condense.search.episodes.qwen_episode_signal import qwen_linker_identity
from memory_condense.search.section_routing import (
    SectionRoute, SectionRoutePlan, SectionSummaryIndex,
    SummaryAttentionPass, SummaryAttentionReceipt,
)
from memory_condense.search.section_summary import bound_int, exact_text


def route_summary_hierarchy(
    query: str, index: SectionSummaryIndex, *, linker: QwenMemoryLinker,
    max_sections: int = 4, max_depth: int = 32,
    eligible_source_ids: Sequence[str] | None = None,
) -> SectionRoutePlan:
    """Attend to roots, descend selected branches, then return exact raw pointers.

    Every Qwen candidate consists solely of a stored summary. Local opaque
    aliases identify candidates outside the prompt. Only the question is used
    as the attention probe; raw text, IDs, hashes and coordinates are never
    model inputs. Each round uses bounded Qwen workspaces. Leaves compete with
    newly expanded children so mixed-depth forests respect the same beam cap.
    Reaching max_depth selects whole internal sections; hydration may reject an
    oversized section atomically. This approximate route never claims closure.
    """
    exact_text(query, "query")
    bound_int(max_sections, "max_sections", 1)
    bound_int(max_depth, "max_depth", 1)
    scope = None if eligible_source_ids is None else tuple(eligible_source_ids)
    if scope is not None and (len(set(scope)) != len(scope) or any(type(s) is not str or not s for s in scope)):
        raise ValueError("source scope must contain unique nonempty exact IDs")
    identity = canonical_json(qwen_linker_identity(linker, strict=True))
    width = bound_int(linker.max_candidates, "Qwen max_candidates", 2)
    if not callable(getattr(linker, "inspect_nested", None)):
        raise TypeError("Qwen routing requires inspect_nested")
    by_id = {section.section_id: section for section in index.sections}
    children = {child for section in index.sections for child in section.child_section_ids}
    frontier = [section for section in index.sections if section.section_id not in children
                and (scope is None or section.source_id in scope)]
    rounds, selected, selected_scores = [], [], []
    for _ in range(max_depth):
        if not frontier:
            break
        candidates = [AssociativeMemoryCandidate(
            episode_id=f"summary-{i}", text=section.summary, route="section_summary",
        ) for i, section in enumerate(frontier)]
        aliases = {candidate.episode_id: section for candidate, section in zip(candidates, frontier)}
        inspection = linker.inspect_nested(
            query, [candidates[i:i + width] for i in range(0, len(candidates), width)],
            beam_per_group=min(max_sections, width - 1), top_k=max_sections, score_mode="qk_ov",
        )
        hits = tuple(inspection.hits)
        if not hits or len(hits) > max_sections or len({hit.episode_id for hit in hits}) != len(hits):
            raise ValueError("Qwen returned an invalid summary selection")
        if any(hit.episode_id not in aliases or not math.isfinite(hit.qk_score)
               or not math.isfinite(hit.ov_transport) for hit in hits):
            raise ValueError("Qwen returned foreign summaries or nonfinite attention scores")
        if inspection.max_workspace_candidates > width or inspection.max_workspace_tokens > linker.max_workspace_tokens:
            raise ValueError("Qwen summary routing exceeded its workspace bounds")
        selected = [aliases[hit.episode_id] for hit in hits]
        selected_scores = [max(1e-12, max(0.0, float(hit.qk_score)) + math.log1p(max(0.0, float(hit.ov_transport))))
                           for hit in hits]
        rounds.append(SummaryAttentionPass(
            tuple(section.section_id for section in frontier),
            tuple(section.section_id for section in selected),
            tuple(float(hit.qk_score) for hit in hits), tuple(float(hit.ov_transport) for hit in hits),
            inspection.passes, inspection.max_workspace_candidates, inspection.max_workspace_tokens,
            inspection.total_candidate_inspections,
        ))
        del inspection, hits  # Keep IDs and scalars, never transient transports.
        if all(not section.child_section_ids for section in selected):
            break
        frontier = [child for section in selected for child in (
            [by_id[child_id] for child_id in section.child_section_ids]
            if section.child_section_ids else [section]
        )]
    if canonical_json(qwen_linker_identity(linker, strict=True)) != identity:
        raise ValueError("Qwen identity changed during summary routing")
    return SectionRoutePlan(
        index.receipt_sha256, quote_sha256(query),
        tuple(SectionRoute(section, score, ()) for section, score in zip(selected, selected_scores)),
        len(rounds[-1].candidate_section_ids) if rounds else 0, scope, max_sections,
        routing_backend="qwen_hierarchical_summaries",
        attention_receipt=SummaryAttentionReceipt(identity, tuple(rounds), max_depth),
    )
