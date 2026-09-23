"""Direct atomic summary retrieval with additive attention-chunk context."""
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan, SectionSummaryIndex
from memory_condense.search.section_summary import bound_int
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from memory_condense.search.summary_time_prior_v2 import question_day


@dataclass(frozen=True, slots=True)
class NativeSpineRoute(SealedIdentity):
    baseline: SectionRoutePlan
    expanded: SectionRoutePlan
    hierarchy_sha256: str
    consulted_leaf_ids: tuple[str, ...]
    added_atomic_ids: tuple[str, ...]
    raw_reads_during_routing: int = 0
    query_qwen_passes: int = 0
    receipt_sha256: str = ""

    def __post_init__(self):
        for name in ("consulted_leaf_ids", "added_atomic_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        b, e = self.baseline, self.expanded
        if (e.routes[:len(b.routes)] != b.routes or e.query_sha256 != b.query_sha256
                or e.eligible_source_ids != b.eligible_source_ids
                or tuple(r.section.section_id for r in e.routes[len(b.routes):]) != self.added_atomic_ids
                or self.raw_reads_during_routing != 0 or self.query_qwen_passes != 0):
            raise ValueError("native context expansion changed the direct routes or query")
        self._seal()


class NativeSpineRouter:
    """Parent scores cannot exclude atoms; cold topology supplies extra context.

    The baseline and expanded arms share every direct route in the same order.
    The existing sequential hydrator therefore preserves baseline evidence and
    charges additions against the same raw-read and rendered-context budgets.
    Qwen attention determines the stored chunk boundaries at ingest only.
    """

    def __init__(self, atomic_semantic, hierarchy):
        if type(atomic_semantic) is not SemanticSectionIndex or type(hierarchy) is not SectionSummaryIndex:
            raise TypeError("native routing requires authenticated atomic vectors and a hierarchy")
        self.semantic, self.hierarchy = atomic_semantic, hierarchy
        self.atoms = {s.section_id: s for s in atomic_semantic.sections}
        by_span = {}
        sources = {}
        for atom in atomic_semantic.hierarchy.sections:
            if atom.child_section_ids or len(atom.spans) != 1:
                raise ValueError("direct native addresses must be original single-span atoms")
            span = atom.spans[0]
            if span.receipt_sha256 in by_span:
                raise ValueError("native atomic addresses overlap")
            by_span[span.receipt_sha256] = atom
            if sources.setdefault(atom.source_id, span.created_at) != span.created_at:
                raise ValueError("an occurrence must have one actual source timestamp")
        self.owners, self.leaf_atoms = {}, {}
        for leaf in hierarchy.sections:
            if leaf.child_section_ids:
                continue
            atoms = []
            for span in leaf.spans:
                atom = by_span.get(span.receipt_sha256)
                if atom is None or atom.spans[0] != span or span.receipt_sha256 in self.owners:
                    raise ValueError("attention chunks must partition original atoms without foreign evidence")
                self.owners[span.receipt_sha256] = leaf.section_id
                atoms.append(atom)
            self.leaf_atoms[leaf.section_id] = tuple(atoms)
        self.dated_sources = sorted((datetime.fromisoformat(day).date(), sid) for sid, day in sources.items())
        self.source_days = [day for day, _ in self.dated_sources]
        self.expanded_index_sha256 = identity_sha256({
            "atomic_index_sha256": atomic_semantic.hierarchy.receipt_sha256,
            "attention_hierarchy_sha256": hierarchy.receipt_sha256,
        })

    def route_vector(self, query, dated_question, vector, *, embedding_identity,
                     max_direct=32, lexical_reserve=2, context_seed_limit=8, max_additions=8):
        bound_int(context_seed_limit, "context_seed_limit", 1)
        bound_int(max_additions, "max_additions")
        asked = question_day(query, dated_question)
        scope = tuple(sid for _, sid in self.dated_sources[:bisect_right(self.source_days, asked)])
        baseline = self.semantic.route_vector(query, vector, embedding_identity=embedding_identity,
            max_sections=max_direct, lexical_reserve=lexical_reserve, eligible_source_ids=scope)
        seen = {r.section.section_id for r in baseline.routes}
        leaves = tuple(dict.fromkeys(self.owners[r.section.spans[0].receipt_sha256]
            for r in baseline.routes[:context_seed_limit] if r.section.spans[0].receipt_sha256 in self.owners))
        # Round-robin prevents one oversized exchange from consuming every
        # context slot. All candidates are unchanged original atomic addresses.
        queues = [[a for a in self.leaf_atoms[leaf] if a.section_id not in seen] for leaf in leaves]
        additions = []
        while len(additions) < max_additions and any(queues):
            for queue in queues:
                if queue and len(additions) < max_additions:
                    atom = queue.pop(0)
                    if atom.section_id not in seen:
                        additions.append(SectionRoute(atom, 1 / (len(baseline.routes) + len(additions) + 1), ()))
                        seen.add(atom.section_id)
        expanded = baseline
        if additions:
            routes = (*baseline.routes, *additions)
            expanded = SectionRoutePlan(self.expanded_index_sha256, baseline.query_sha256, routes,
                baseline.matched_section_count, scope, len(routes), routing_backend="summary_hybrid")
        return NativeSpineRoute(baseline, expanded, self.hierarchy.receipt_sha256, leaves,
                                tuple(r.section.section_id for r in additions))
