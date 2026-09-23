"""A dense root shortlist followed by bounded Qwen summary-tree traversal."""
from __future__ import annotations

from datetime import date, datetime

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.search.section_routing import (
    SectionRoute, SectionRoutePlan, SectionSummaryIndex, SummaryAttentionReceipt,
)
from memory_condense.search.section_summary import SectionSummary, bound_int, exact_text
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from memory_condense.search.summary_shortlist_attention import rerank_summary_shortlist


class BoundedSpineHierarchyRouter:
    """Qwen sees parents and their descendants, one complete workspace per level.

    Dense leaf addresses nominate roots without hiding leaf matches behind lossy
    parent summaries. Qwen then makes each branch decision using stored summaries.
    Scopes and mention-day eligibility apply before root admission and descent.
    A route must reach leaves; a depth limit never substitutes oversized parents.
    """

    def __init__(self, semantic):
        if type(semantic) is not SemanticSectionIndex:
            raise TypeError("hierarchical routing requires authenticated summary addresses")
        self.semantic = semantic
        self.index = semantic.hierarchy
        self.by_id = {s.section_id: s for s in self.index.sections}
        if not any(s.child_section_ids for s in self.index.sections):
            raise ValueError("hierarchical routing requires populated parent summaries")
        if any(len(s.child_section_ids) not in (0, 2) for s in self.index.sections):
            raise ValueError("bounded spine routing requires the binary attention tree")
        parented = {child for s in self.index.sections for child in s.child_section_ids}
        self.roots = tuple(s.section_id for s in self.index.sections if s.section_id not in parented)
        self.leaf_positions = {s.section_id: i for i, s in enumerate(semantic.sections)}
        self.minimum_day = {s.section_id: min(datetime.fromisoformat(p.created_at).date() for p in s.spans)
                            for s in semantic.sections}
        self.members, self.depth = {}, {}

        def visit(sid):
            children = self.by_id[sid].child_section_ids
            if children:
                for child in children:
                    visit(child)
                self.members[sid] = tuple(i for child in children for i in self.members[child])
                self.depth[sid] = 1 + max(self.depth[child] for child in children)
            else:
                self.members[sid] = (self.leaf_positions[sid],)
                self.depth[sid] = 1
        for sid in self.roots:
            visit(sid)

    def route_vector(self, query, vector, *, embedding_identity, linker,
                     asked_day=None, eligible_source_ids=None, root_shortlist=8,
                     beam=4, max_depth=16):
        exact_text(query, "query")
        for name, value in (("root_shortlist", root_shortlist), ("beam", beam), ("max_depth", max_depth)):
            bound_int(value, name, 1)
        if embedding_identity != self.semantic.embedding_identity:
            raise ValueError("query vector encoder identity changed")
        scope = SemanticSectionIndex._scope(eligible_source_ids)
        if asked_day is not None and type(asked_day) is not date:
            raise TypeError("asked_day must be an explicit calendar date")
        if root_shortlist > linker.max_candidates or 2 * beam > linker.max_candidates:
            raise ValueError("root shortlist or binary beam exceeds one Qwen workspace")
        scores = self.semantic._dense.score_all(vector)
        eligible = {i for i, s in enumerate(self.semantic.sections)
                    if (scope is None or s.source_id in scope)
                    and (asked_day is None or self.minimum_day[s.section_id] <= asked_day)}
        maxima = {sid: max((float(scores[i]) for i in members if i in eligible), default=float("-inf"))
                  for sid, members in self.members.items()}
        roots = [sid for sid in self.roots if maxima[sid] != float("-inf")]
        frontier = sorted(roots, key=lambda sid: (-maxima[sid], sid))[:root_shortlist]
        if any(self.depth[sid] > max_depth for sid in frontier):
            raise ValueError("depth budget cannot reach every admitted branch's leaves")
        rounds, identity = [], None
        selected = ()
        for _ in range(max_depth):
            if not frontier:
                break
            shortlist = SectionRoutePlan(self.index.receipt_sha256, quote_sha256(query),
                tuple(SectionRoute(self.by_id[sid], 1 / (i + 1), ()) for i, sid in enumerate(frontier)),
                len(frontier), scope, len(frontier), routing_backend="summary_dense")
            selection = rerank_summary_shortlist(query, self.index, shortlist,
                                                 linker=linker, max_sections=beam)
            receipt = selection.attention_receipt
            if identity is not None and identity != receipt.linker_identity_json:
                raise ValueError("Qwen identity changed between hierarchy levels")
            identity = receipt.linker_identity_json
            rounds.extend(receipt.rounds)
            selected = selection.routes
            if all(not route.section.child_section_ids for route in selected):
                break
            frontier = [sid for route in selected
                        for sid in (route.section.child_section_ids or (route.section.section_id,))
                        if maxima[sid] != float("-inf")]
        if any(route.section.child_section_ids for route in selected):
            raise ValueError("hierarchical route ended before reaching raw-addressable leaves")
        if identity is None:
            # The shared adapter supplies the authenticated empty-route identity.
            empty = SectionRoutePlan(self.index.receipt_sha256, quote_sha256(query), (), 0,
                                     scope, beam, routing_backend="summary_dense")
            identity = rerank_summary_shortlist(query, self.index, empty, linker=linker,
                                               max_sections=beam).attention_receipt.linker_identity_json
        return SectionRoutePlan(self.index.receipt_sha256, quote_sha256(query), selected,
            len(frontier), scope, beam, routing_backend="qwen_hierarchical_summaries",
            attention_receipt=SummaryAttentionReceipt(identity, tuple(rounds), max_depth))


def project_hierarchy_leaves(plan, index, asked_day):
    """Keep the original flat renderer and exact, question-day-eligible spans."""
    if type(asked_day) is not date or plan.index_sha256 != index.receipt_sha256:
        raise ValueError("hierarchy projection date or index binding changed")
    by_id = {s.section_id: s for s in index.sections}
    sections, changes = [], []
    for route in plan.routes:
        original = route.section
        if original.child_section_ids or by_id.get(original.section_id) != original:
            raise ValueError("hierarchy projection requires unchanged bound leaves")
        spans = tuple(p for p in original.spans if datetime.fromisoformat(p.created_at).date() <= asked_day)
        if not spans:
            raise ValueError("hierarchy admitted an entirely future leaf")
        section = original
        if spans != original.spans:
            binding = {"parent_section_sha256": original.receipt_sha256,
                       "asked_day": asked_day.isoformat(), "span_sha256s": [p.receipt_sha256 for p in spans]}
            section = SectionSummary("as-of-" + identity_sha256(binding), original.source_id,
                                     original.summary, spans, "as-of-hierarchy-leaf-v1")
            changes.append({**binding, "projected_section_sha256": section.receipt_sha256})
        sections.append(section)
    projected = SectionSummaryIndex(sections)
    result = SectionRoutePlan(projected.receipt_sha256, plan.query_sha256,
        tuple(SectionRoute(s, 1 / (i + 1), ()) for i, s in enumerate(sections)), len(sections),
        plan.eligible_source_ids, plan.max_sections, routing_backend="summary_dense")
    return result, {"hierarchical_plan_sha256": plan.receipt_sha256,
                   "projected_plan_sha256": result.receipt_sha256, "asked_day": asked_day.isoformat(),
                   "partial_projections": changes, "raw_reads_during_projection": 0}
