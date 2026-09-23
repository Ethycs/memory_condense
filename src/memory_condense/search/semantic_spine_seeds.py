"""Experimental semantic seed priority with unchanged user and calendar routes.

Only the initial six whole-summary seeds differ from SpineUnionRouter. This
does not establish better evidence or answers: changing seeds can remove raw
evidence when the source expansion and hydration budgets are applied.
"""
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.spine_union_routing import SpineUnionRouter


class SemanticSpineSeedRouter(SpineUnionRouter):
    def route_vectors(self, query, dated_question, vector, *, embedding_identity, content_vector=None):
        prior, audit = super().route_vectors(query, dated_question, vector,
            embedding_identity=embedding_identity, content_vector=content_vector)
        dense = self.semantic.route_vector(query, vector, embedding_identity=embedding_identity,
            max_sections=6, lexical_reserve=0)
        originals = {route.section.section_id: route.section for route in prior.routes}
        ordered = [originals[sid] for sid in audit["preferred_section_ids"]]
        ordered.extend(route.section for route in dense.routes)
        ordered.extend(originals[sid] for sid in audit["user_supplement_section_ids"])
        sections = {section.section_id: section for section in ordered}
        plan = SectionRoutePlan(prior.index_sha256, prior.query_sha256,
            tuple(SectionRoute(section, 1 / (i + 1), ()) for i, section in enumerate(sections.values())),
            prior.matched_section_count, None, prior.max_sections, routing_backend="summary_hybrid")
        return plan, {**audit, "previous_baseline_section_ids": audit["baseline_section_ids"],
            "baseline_section_ids": [route.section.section_id for route in dense.routes],
            "baseline_routes_retained": False, "lexical_reserve": 0,
            "seed_policy": "six dense whole-summary leaves; unchanged calendar and user supplements",
            "raw_reads_during_routing": 0}
