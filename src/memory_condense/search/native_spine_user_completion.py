"""Hydrate routed user statements before assistant context, then complete them.

Two measured losses motivate this stage (Analysis 35, priority 1). First, the
sealed order appended late user additions after long assistant replies, so the
exact hydrator's shared token budget dropped a decisive user turn while it kept
assistant sections that the user-evidence projection then discarded. Second, a
conversation that routing had already selected still lacked short later user
turns (a subclass choice, a code error) that no summary stage reached.

This stage keeps every prior route object and only (a) moves user-role routes
ahead of assistant-only routes after the protected prefix and (b) appends the
remaining user atoms of already routed conversations, round-robin in route
rank order and transcript order within each conversation, up to a bounded
count. No raw text, question text or new vector is read; the existing exact
hydrator still owns the token and span caps.
"""
from collections import deque
from dataclasses import dataclass, fields

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.native_spine_additive_lexical import NativeSpineAdditiveLexicalRoute
from memory_condense.search.native_spine_context_routing import NativeSpineContextRoute
from memory_condense.search.native_spine_parent_user_routing import (
    NativeSpineParentUserRoute, NativeSpineParentUserRouter,
    route_from_payload as prior_route_from_payload,
)
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.section_summary import SectionSummary, bound_int


FORMAT = 'native-spine-user-completion-v1'
BASE_TYPES = (NativeSpineContextRoute, NativeSpineAdditiveLexicalRoute, NativeSpineParentUserRoute)


def is_user(route):
    return route.section.spans[0].role == 'user'


def source_rank(base):
    """Routed conversations in rank order: direct matches first, then additions."""
    return tuple(dict.fromkeys(r.section.source_id
                               for plan in (base.baseline, base.expanded) for r in plan.routes))


def user_first(routes, protected):
    prefix, rest = tuple(routes[:protected]), tuple(routes[protected:])
    return (*prefix, *(r for r in rest if is_user(r)), *(r for r in rest if not is_user(r)))


def expanded_index(base_plan, limit):
    return identity_sha256({'format': FORMAT, 'base_plan': base_plan.receipt_sha256,
                            'user_completion_atoms': limit})


@dataclass(frozen=True, slots=True)
class NativeSpineUserCompletionRoute(NativeSpineContextRoute):
    completion_base: NativeSpineContextRoute | None = None
    user_completion_atoms: int = 0
    completion_added_atomic_ids: tuple[str, ...] = ()

    def __post_init__(self):
        base = self.completion_base
        if type(base) not in BASE_TYPES:
            raise ValueError('user completion requires a sealed prior native route')
        for name in ('consulted_chunk_ids', 'context_atomic_ids', 'added_atomic_ids',
                     'completion_added_atomic_ids'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        bound_int(self.user_completion_atoms, 'user_completion_atoms')
        preserved = ('baseline', 'hierarchy_sha256', 'consulted_chunk_ids', 'context_atomic_ids',
                     'protected_direct', 'ancestor_hops', 'raw_reads_during_routing', 'query_qwen_passes')
        prior = {r.section.section_id for r in base.expanded.routes}
        added = tuple(r for r in self.expanded.routes if r.section.section_id not in prior)
        sources = set(source_rank(base))
        if (any(getattr(self, k) != getattr(base, k) for k in preserved)
                or len(added) > self.user_completion_atoms
                or self.completion_added_atomic_ids != tuple(r.section.section_id for r in added)
                or any(r.section.child_section_ids or len(r.section.spans) != 1 or not is_user(r)
                       or r.section.source_id not in sources or r.matched_terms for r in added)
                or self.expanded.routes != user_first((*base.expanded.routes, *added), self.protected_direct)
                or self.expanded.query_sha256 != base.expanded.query_sha256
                or self.expanded.eligible_source_ids != base.expanded.eligible_source_ids
                or self.expanded.index_sha256 != expanded_index(base.expanded, self.user_completion_atoms)
                or self.expanded.routing_backend != 'summary_hybrid'):
            raise ValueError('user completion changed prior evidence, its order rule or its bounded user atoms')
        NativeSpineContextRoute.__post_init__(self)


class NativeSpineUserCompletionRouter(NativeSpineParentUserRouter):
    """Apply the bounded user completion after the unchanged parent supplement."""

    def __init__(self, atomic_semantic, hierarchy, parent_semantic):
        super().__init__(atomic_semantic, hierarchy, parent_semantic)
        children = {c for s in hierarchy.sections for c in s.child_section_ids}
        self.user_atoms_by_source = {}
        # Each stored root lists its exact spans in transcript order. Sections are
        # id-sorted, so a conversation split across roots keeps a fixed order.
        for root in hierarchy.sections:
            if root.section_id in children:
                continue
            for span in root.spans:
                if span.role == 'user':
                    atom = self.by_span[span.receipt_sha256]
                    self.user_atoms_by_source.setdefault(root.source_id, []).append(atom)

    def route_vector(self, query, dated_question, vector, *, embedding_identity,
                     user_completion_atoms=0, **options):
        base = super().route_vector(query, dated_question, vector,
                                    embedding_identity=embedding_identity, **options)
        return self.complete(base, user_completion_atoms)

    def complete(self, base, user_completion_atoms):
        """Reorder and complete one sealed prior route without new scoring."""
        if type(base) not in BASE_TYPES:
            raise TypeError('user completion requires a sealed prior native route')
        bound_int(user_completion_atoms, 'user_completion_atoms')
        selected = {r.section.section_id for r in base.expanded.routes}
        queues = [deque(a for a in self.user_atoms_by_source.get(sid, ()) if a.section_id not in selected)
                  for sid in source_rank(base)]
        additions = []
        while len(additions) < user_completion_atoms and any(queues):
            for queue in queues:
                if queue and len(additions) < user_completion_atoms:
                    additions.append(queue.popleft())
        offset = len(base.expanded.routes)
        routes = user_first((*base.expanded.routes,
                             *(SectionRoute(a, 1.0 / (offset + i + 1), ()) for i, a in enumerate(additions))),
                            base.protected_direct)
        expanded = SectionRoutePlan(expanded_index(base.expanded, user_completion_atoms),
            base.expanded.query_sha256, routes, max(base.expanded.matched_section_count, len(routes)),
            base.expanded.eligible_source_ids, len(routes), routing_backend='summary_hybrid')
        direct = {r.section.section_id for r in base.baseline.routes}
        payload = {f.name: getattr(base, f.name) for f in fields(NativeSpineContextRoute)}
        payload.update(expanded=expanded, receipt_sha256='',
                       added_atomic_ids=tuple(r.section.section_id for r in routes
                                              if r.section.section_id not in direct))
        return NativeSpineUserCompletionRoute(**payload, completion_base=base,
            user_completion_atoms=user_completion_atoms,
            completion_added_atomic_ids=tuple(a.section_id for a in additions))


def route_from_payload(payload):
    if 'completion_base' not in payload:
        return prior_route_from_payload(payload)
    def plan(value):
        p = dict(value)
        p['routes'] = tuple(SectionRoute(SectionSummary.from_dict(r['section']), r['score'],
            tuple(r['matched_terms']), r['receipt_sha256']) for r in p['routes'])
        return SectionRoutePlan(**p)
    p = dict(payload)
    p['baseline'], p['expanded'] = plan(p['baseline']), plan(p['expanded'])
    p['completion_base'] = prior_route_from_payload(p['completion_base'])
    result = NativeSpineUserCompletionRoute(**p)
    if result.identity_payload() != payload:
        raise ValueError('stored user completion changed during reconstruction')
    return result
