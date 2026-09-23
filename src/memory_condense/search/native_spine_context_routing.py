"""Bounded parent-chunk context, scheduled before low-priority raw evidence."""
from collections import deque
from dataclasses import dataclass

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.native_spine_routing import NativeSpineRouter
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.section_summary import bound_int


@dataclass(frozen=True, slots=True)
class NativeSpineContextRoute(SealedIdentity):
    baseline: SectionRoutePlan
    expanded: SectionRoutePlan
    hierarchy_sha256: str
    consulted_chunk_ids: tuple[str, ...]
    context_atomic_ids: tuple[str, ...]
    added_atomic_ids: tuple[str, ...]
    protected_direct: int
    ancestor_hops: int
    raw_reads_during_routing: int = 0
    query_qwen_passes: int = 0
    receipt_sha256: str = ""

    def __post_init__(self):
        for name in ("consulted_chunk_ids", "context_atomic_ids", "added_atomic_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        bound_int(self.protected_direct, "protected_direct")
        bound_int(self.ancestor_hops, "ancestor_hops")
        b, e = self.baseline, self.expanded
        before = {r.section.section_id: r for r in b.routes}
        after = {r.section.section_id: r for r in e.routes}
        protected = min(self.protected_direct, len(b.routes))
        if (e.routes[:protected] != b.routes[:protected]
                or any(after.get(sid) != row for sid, row in before.items())
                or e.query_sha256 != b.query_sha256 or e.eligible_source_ids != b.eligible_source_ids
                or set(self.context_atomic_ids) - after.keys()
                or tuple(sid for sid in after if sid not in before) != self.added_atomic_ids
                or self.raw_reads_during_routing != 0 or self.query_qwen_passes != 0):
            raise ValueError("parent context changed direct addresses, scope or protected routes")
        self._seal()


class NativeSpineContextRouter(NativeSpineRouter):
    """Use stored attention topology; no raw-text scoring or live Qwen pass.

    Direct candidates are retained, but only their strongest prefix is protected
    in packet order. User evidence from a bounded parent neighborhood precedes
    assistant context. The existing exact hydrator owns all token/read limits.
    Unlike append-only expansion, this can promote a direct tail candidate that
    otherwise loses its budget to an earlier, long assistant reply.
    """

    def __init__(self, atomic_semantic, hierarchy):
        super().__init__(atomic_semantic, hierarchy)
        self.chunks = {s.section_id: s for s in hierarchy.sections}
        self.parents = {c: s.section_id for s in hierarchy.sections for c in s.child_section_ids}
        self.by_span = {a.spans[0].receipt_sha256: a for a in self.atoms.values()}

    def route_vector(self, query, dated_question, vector, *, embedding_identity,
                     max_direct=32, lexical_reserve=2, context_seed_limit=4,
                     max_additions=8, protected_direct=4, ancestor_hops=1):
        bound_int(protected_direct, "protected_direct")
        bound_int(ancestor_hops, "ancestor_hops")
        bound_int(max_additions, "max_additions")
        if protected_direct > max_direct or ancestor_hops > 2:
            raise ValueError("parent context requires a bounded prefix and at most two ancestor hops")
        direct = super().route_vector(query, dated_question, vector,
            embedding_identity=embedding_identity, max_direct=max_direct,
            lexical_reserve=lexical_reserve, context_seed_limit=context_seed_limit, max_additions=0).baseline
        prefix = direct.routes[:protected_direct]
        seen = {r.section.section_id for r in prefix}
        chunks, queues = [], []
        if max_additions:
            for seed in direct.routes[:context_seed_limit]:
                span = seed.section.spans[0]
                chunk_id = self.owners.get(span.receipt_sha256)
                if chunk_id is None:
                    continue
                for _ in range(ancestor_hops):
                    chunk_id = self.parents.get(chunk_id, chunk_id)
                if chunk_id in chunks:
                    continue
                chunks.append(chunk_id)
                atoms = [self.by_span[s.receipt_sha256] for s in self.chunks[chunk_id].spans]
                origin = atoms.index(seed.section)
                ordered = sorted(enumerate(atoms), key=lambda item:
                    (item[1].spans[0].role != "user", abs(item[0] - origin), item[0]))
                queues.append(deque(a for _, a in ordered if a.section_id not in seen))
        context = []
        # Exhaust user candidates across neighborhoods before assistant context.
        for user_phase in (True, False):
            while len(context) < max_additions:
                progressed = False
                for queue in queues:
                    while queue and queue[0].section_id in seen:
                        queue.popleft()
                    if not queue or (queue[0].spans[0].role == "user") != user_phase:
                        continue
                    atom = queue.popleft()
                    context.append(atom)
                    seen.add(atom.section_id)
                    progressed = True
                    if len(context) == max_additions:
                        break
                if not progressed:
                    break
        original = {r.section.section_id: r for r in direct.routes}
        contextual = [original[a.section_id] if a.section_id in original else
            SectionRoute(a, 1.0 / (len(direct.routes) + i + 1), ()) for i, a in enumerate(context)]
        tail = [r for r in direct.routes[protected_direct:] if r.section.section_id not in seen]
        is_user = lambda r: r.section.spans[0].role == "user"
        routes = (*prefix, *(r for r in contextual if is_user(r)), *(r for r in tail if is_user(r)),
                  *(r for r in contextual if not is_user(r)), *(r for r in tail if not is_user(r)))
        expanded = SectionRoutePlan(identity_sha256({"format": "native-spine-parent-context-v1",
            "indexes": self.expanded_index_sha256, "protected_direct": protected_direct,
            "ancestor_hops": ancestor_hops, "context_seed_limit": context_seed_limit,
            "max_context_atoms": max_additions}), direct.query_sha256, routes,
            direct.matched_section_count, direct.eligible_source_ids, max(1, len(routes)),
            routing_backend="summary_hybrid")
        return NativeSpineContextRoute(direct, expanded, self.hierarchy.receipt_sha256, tuple(chunks),
            tuple(a.section_id for a in context), tuple(r.section.section_id for r in routes
                if r.section.section_id not in original), protected_direct, ancestor_hops)
