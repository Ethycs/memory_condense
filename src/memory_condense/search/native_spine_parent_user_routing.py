"""Append bounded user atoms selected through complete parent user summaries."""
from dataclasses import dataclass, fields

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.native_spine_additive_lexical import (
    NativeSpineAdditiveLexicalRoute, NativeSpineAdditiveLexicalRouter,
    route_from_payload as prior_route_from_payload,
)
from memory_condense.search.native_spine_context_routing import NativeSpineContextRoute
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.summary_semantic_index import SemanticSectionIndex


FORMAT = 'native-spine-additive-parent-users-v1'
PARENT_ATOM_LIMIT = 2


def source_pool(base):
    return tuple(sorted({r.section.source_id for r in base.baseline.routes}))


def expanded_index(base, parent):
    return identity_sha256({'format': FORMAT, 'base_plan': base.receipt_sha256,
                            'parent_plan': parent.receipt_sha256, 'atom_limit': PARENT_ATOM_LIMIT})


@dataclass(frozen=True, slots=True)
class NativeSpineParentUserRoute(NativeSpineContextRoute):
    parent_base: NativeSpineContextRoute | None = None
    parent_plan: SectionRoutePlan | None = None
    parent_added_atomic_ids: tuple[str, ...] = ()

    def __post_init__(self):
        base, parent = self.parent_base, self.parent_plan
        if type(base) not in (NativeSpineContextRoute, NativeSpineAdditiveLexicalRoute) or type(parent) is not SectionRoutePlan:
            raise ValueError('parent supplement requires sealed base and parent plans')
        for name in ('consulted_chunk_ids', 'context_atomic_ids', 'added_atomic_ids', 'parent_added_atomic_ids'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        preserved = ('baseline', 'hierarchy_sha256', 'consulted_chunk_ids', 'context_atomic_ids',
                     'protected_direct', 'ancestor_hops', 'raw_reads_during_routing', 'query_qwen_passes')
        tail = self.expanded.routes[len(base.expanded.routes):]
        parent_spans = {s for r in parent.routes for s in r.section.spans}
        if (any(getattr(self, k) != getattr(base, k) for k in preserved)
                or parent.max_sections != 1 or len(parent.routes) != 1
                or parent.routing_backend != 'summary_dense'
                or parent.query_sha256 != base.baseline.query_sha256
                or parent.eligible_source_ids != source_pool(base)
                or any(s.role != 'user' for s in parent_spans)
                or not 1 <= len(tail) <= PARENT_ATOM_LIMIT
                or self.expanded.routes[:len(base.expanded.routes)] != base.expanded.routes
                or self.parent_added_atomic_ids != tuple(r.section.section_id for r in tail)
                or any(r.section.child_section_ids or len(r.section.spans) != 1
                       or r.section.spans[0] not in parent_spans for r in tail)
                or self.expanded.index_sha256 != expanded_index(base.expanded, parent)
                or self.expanded.routing_backend != 'summary_hybrid'):
            raise ValueError('parent supplement changed prior evidence, source scope or its bounded user atoms')
        NativeSpineContextRoute.__post_init__(self)


class NativeSpineParentUserRouter(NativeSpineAdditiveLexicalRouter):
    def __init__(self, atomic_semantic, hierarchy, parent_semantic):
        super().__init__(atomic_semantic, hierarchy)
        if (type(parent_semantic) is not SemanticSectionIndex
                or parent_semantic.hierarchy.receipt_sha256 != project_parent_users(hierarchy).receipt_sha256
                or parent_semantic.embedding_identity != atomic_semantic.embedding_identity):
            raise ValueError('parent vectors must match the stored hierarchy and atomic query encoder')
        self.parent_semantic = parent_semantic
        self.atomic_positions = {s.section_id: i for i, s in enumerate(self.semantic.sections)}

    def route_vector(self, query, dated_question, vector, *, embedding_identity, **options):
        base = super().route_vector(query, dated_question, vector,
                                    embedding_identity=embedding_identity, **options)
        parent = self.parent_semantic.route_vector(query, vector, embedding_identity=embedding_identity,
            max_sections=1, lexical_reserve=0, eligible_source_ids=source_pool(base))
        if not parent.routes:
            return base
        selected = {r.section.section_id for r in base.expanded.routes}
        candidates = [self.by_span[s.receipt_sha256] for s in parent.routes[0].section.spans
                      if self.by_span[s.receipt_sha256].section_id not in selected]
        if not candidates:
            return base
        scores = self.semantic._dense.score_all(vector)
        candidates.sort(key=lambda s: (-float(scores[self.atomic_positions[s.section_id]]), s.section_id))
        additions = tuple(SectionRoute(s, 1.0 / (len(base.expanded.routes) + i + 1), ())
                          for i, s in enumerate(candidates[:PARENT_ATOM_LIMIT]))
        routes = (*base.expanded.routes, *additions)
        expanded = SectionRoutePlan(expanded_index(base.expanded, parent), base.expanded.query_sha256,
            routes, max(base.expanded.matched_section_count, len(routes)), base.expanded.eligible_source_ids,
            len(routes), routing_backend='summary_hybrid')
        payload = {f.name: getattr(base, f.name) for f in fields(NativeSpineContextRoute)}
        direct = {r.section.section_id for r in base.baseline.routes}
        payload.update(expanded=expanded, receipt_sha256='',
                       added_atomic_ids=tuple(r.section.section_id for r in routes if r.section.section_id not in direct))
        return NativeSpineParentUserRoute(**payload, parent_base=base, parent_plan=parent,
            parent_added_atomic_ids=tuple(r.section.section_id for r in additions))


def route_from_payload(payload):
    if 'parent_base' not in payload:
        return prior_route_from_payload(payload)
    def plan(value):
        p = dict(value)
        p['routes'] = tuple(SectionRoute(SectionSummary.from_dict(r['section']), r['score'],
            tuple(r['matched_terms']), r['receipt_sha256']) for r in p['routes'])
        return SectionRoutePlan(**p)
    p = dict(payload)
    p['baseline'], p['expanded'], p['parent_plan'] = (plan(p[k]) for k in ('baseline', 'expanded', 'parent_plan'))
    p['parent_base'] = prior_route_from_payload(p['parent_base'])
    result = NativeSpineParentUserRoute(**p)
    if result.identity_payload() != payload:
        raise ValueError('stored parent supplement changed during reconstruction')
    return result
