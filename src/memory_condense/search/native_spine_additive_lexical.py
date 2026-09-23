"""Append one summary lexical match without competing for parent-context slots."""
from dataclasses import dataclass, fields

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.native_spine_context_routing import (
    NativeSpineContextRoute, NativeSpineContextRouter,
)
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.section_summary import SectionSummary


FORMAT = "native-spine-additive-lexical-v1"


def expanded_index(context, lexical):
    return identity_sha256({"format": FORMAT, "context_index": context.index_sha256,
                            "lexical_index": lexical.index_sha256})


@dataclass(frozen=True, slots=True)
class NativeSpineAdditiveLexicalRoute(NativeSpineContextRoute):
    context_route: NativeSpineContextRoute | None = None
    lexical_plan: SectionRoutePlan | None = None
    lexical_atomic_ids: tuple[str, ...] = ()

    def __post_init__(self):
        base, lexical = self.context_route, self.lexical_plan
        if type(base) is not NativeSpineContextRoute or type(lexical) is not SectionRoutePlan:
            raise ValueError("additive lexical routing requires sealed context and lexical plans")
        for name in ("lexical_atomic_ids", "consulted_chunk_ids", "context_atomic_ids", "added_atomic_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        existing = {r.section.section_id for r in base.expanded.routes}
        additions = tuple(r for r in lexical.routes if r.section.section_id not in existing)
        preserved = ("baseline", "hierarchy_sha256", "consulted_chunk_ids", "context_atomic_ids",
                     "protected_direct", "ancestor_hops", "raw_reads_during_routing", "query_qwen_passes")
        if (any(getattr(self, key) != getattr(base, key) for key in preserved)
                or lexical.max_sections != 1 or len(additions) != 1
                or lexical.routing_backend != "summary_bm25"
                or lexical.query_sha256 != base.expanded.query_sha256
                or lexical.eligible_source_ids != base.expanded.eligible_source_ids
                or any(r.section.child_section_ids or len(r.section.spans) != 1 for r in additions)
                or self.lexical_atomic_ids != tuple(r.section.section_id for r in additions)
                or self.expanded.routes != (*base.expanded.routes, *additions)
                or self.expanded.index_sha256 != expanded_index(base.expanded, lexical)
                or self.expanded.routing_backend != "summary_hybrid"):
            raise ValueError("additive lexical routing changed prior selections, scope or its one-match limit")
        NativeSpineContextRoute.__post_init__(self)


class NativeSpineAdditiveLexicalRouter(NativeSpineContextRouter):
    """Keep the existing context route intact, then offer one additional atom.

    Lexical scoring sees the atomic summaries only. The extra match never enters
    the parent seed pool. The existing hydrator enforces the unchanged raw caps
    after it has considered every original route in its original order.
    """

    def route_vector(self, query, dated_question, vector, *, embedding_identity, **options):
        base = super().route_vector(query, dated_question, vector,
                                    embedding_identity=embedding_identity, **options)
        lexical = self.semantic.hierarchy.route(query, max_sections=1,
                                                eligible_source_ids=base.baseline.eligible_source_ids)
        selected = {r.section.section_id for r in base.expanded.routes}
        additions = tuple(r for r in lexical.routes if r.section.section_id not in selected)
        if not additions:
            return base
        routes = (*base.expanded.routes, *additions)
        expanded = SectionRoutePlan(expanded_index(base.expanded, lexical), base.expanded.query_sha256,
            routes, max(base.expanded.matched_section_count, len(routes)),
            base.expanded.eligible_source_ids, len(routes), routing_backend="summary_hybrid")
        direct = {r.section.section_id for r in base.baseline.routes}
        payload = {f.name: getattr(base, f.name) for f in fields(base)}
        payload.update(expanded=expanded, receipt_sha256="",
                       added_atomic_ids=tuple(r.section.section_id for r in routes
                                              if r.section.section_id not in direct))
        return NativeSpineAdditiveLexicalRoute(**payload, context_route=base, lexical_plan=lexical,
            lexical_atomic_ids=tuple(r.section.section_id for r in additions))


def route_from_payload(payload):
    """Reconstruct and verify both unchanged and additive serialized receipts."""
    def plan(value):
        p = dict(value)
        p["routes"] = tuple(SectionRoute(SectionSummary.from_dict(r["section"]), r["score"],
            tuple(r["matched_terms"]), r["receipt_sha256"]) for r in p["routes"])
        return SectionRoutePlan(**p)
    p = dict(payload)
    p["baseline"], p["expanded"] = plan(p["baseline"]), plan(p["expanded"])
    if "context_route" in p:
        p["context_route"] = route_from_payload(p["context_route"])
        p["lexical_plan"] = plan(p["lexical_plan"])
        result = NativeSpineAdditiveLexicalRoute(**p)
    else:
        result = NativeSpineContextRoute(**p)
    if result.identity_payload() != payload:
        raise ValueError("saved native route changed during reconstruction")
    return result
