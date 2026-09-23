"""Expand selected summary sources to exact user turns before attached context.

Compilation partitions the original raw coordinates by role without reading raw
text. Query expansion uses only selected sources, transcript order and token
metadata. The existing hydrator remains the byte and final budget authority.
"""
from collections import Counter
import json

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.section_routing import SectionSummaryIndex, SectionRoute, SectionRoutePlan


class SourceSpineHydrationIndex:
    def __init__(self, hierarchy, atoms):
        self.parent_sha256 = hierarchy.receipt_sha256
        leaves = tuple(s for s in hierarchy.sections if not s.child_section_ids)
        expected = Counter(p.receipt_sha256 for s in leaves for p in s.spans)
        if any(len(a.spans) != 1 or a.child_section_ids for a in atoms):
            raise ValueError("source expansion requires single-fragment summary atoms")
        if Counter(p.receipt_sha256 for a in atoms for p in a.spans) != expected or any(v != 1 for v in expected.values()):
            raise ValueError("source expansion atoms do not partition the bound memory")
        users = {}
        for atom in atoms:
            span = atom.spans[0]
            if span.role == "user":
                users.setdefault(span.turn_id, []).append(atom)
        self.by_source = {}
        sections = []
        for group in users.values():
            spans = tuple(a.spans[0] for a in group)
            section = SectionSummary("source-user-" + identity_sha256([p.receipt_sha256 for p in spans]),
                spans[0].source_id, "\n".join(a.summary for a in group), spans, "source-spine-hydration-v1")
            self.by_source.setdefault(section.source_id, []).append(section)
            sections.append(section)
        self.attachments = {}
        for leaf in leaves:
            spans = tuple(p for p in leaf.spans if p.role != "user")
            if spans:
                body = json.loads(leaf.summary)
                section = SectionSummary("source-context-" + identity_sha256([p.receipt_sha256 for p in spans]),
                    leaf.source_id, body["attached_context_not_user_assertions"] or "Attached transcript context.",
                    spans, "source-spine-hydration-v1")
                self.attachments[leaf.section_id] = section
                sections.append(section)
        self.index = SectionSummaryIndex(sections)
        if Counter(p.receipt_sha256 for s in sections for p in s.spans) != expected:
            raise ValueError("role projection changed the raw partition")
        self.originals = {s.section_id: s for s in leaves}

    def expand(self, plan):
        if plan.index_sha256 != self.parent_sha256 or any(
                self.originals.get(r.section.section_id) != r.section for r in plan.routes):
            raise ValueError("source expansion route belongs to another memory")
        sources = tuple(dict.fromkeys(r.section.source_id for r in plan.routes))[:6]
        selected_turns = tuple(dict.fromkeys(p.turn_id for r in plan.routes for p in r.section.spans
                                            if p.role == "user"))
        priority = {turn_id: i for i, turn_id in enumerate(selected_turns)}
        candidates = {source: sorted(self.by_source.get(source, ()),
            key=lambda s: priority.get(s.spans[0].turn_id, len(priority))) for source in sources}
        selected = []
        reserved = 0
        skipped = []
        # Whole user turns enter in source rounds. One long conversation cannot
        # consume the entire reservation before other selected sources enter.
        for position in range(24):
            for source in sources:
                rows = candidates[source]
                if position >= len(rows):
                    continue
                section = rows[position]
                estimate = sum(p.token_count for p in section.spans) + 64 * len(section.spans) + 32
                if reserved + estimate <= 2048 and len(selected) < 24:
                    selected.append(section)
                    reserved += estimate
                else:
                    skipped.append(section.section_id)
        selected.extend(self.attachments[r.section.section_id] for r in plan.routes
                        if r.section.section_id in self.attachments)
        unique = {s.section_id: s for s in selected}
        routes = tuple(SectionRoute(s, 1 / (i + 1), ()) for i, s in enumerate(unique.values()))
        expanded = SectionRoutePlan(self.index.receipt_sha256, plan.query_sha256, routes,
            len(self.index.sections), None, 38, routing_backend="summary_hybrid")
        return expanded, {"parent_route_sha256": plan.receipt_sha256, "selected_sources": sources,
            "selected_user_turns_prioritized_before_source_backfill": True,
            "user_reservation_token_estimate": reserved, "user_sections_skipped_by_metadata_budget": skipped,
            "whole_user_turns": True, "raw_reads_during_expansion": 0,
            "final_context_budget_enforced_by_hydrator": True}
