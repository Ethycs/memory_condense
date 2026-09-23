"""Preserve admitted user turns, then offer omitted routed users to hydration.

This experimental expansion uses only existing summary descriptors and raw
coordinates. Exact hydration retains its original token/span budgets. Extra
users can displace attached context, so answer quality needs a matched test.
"""
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan


class SourceSpineOverflow:
    def __init__(self, source_spine):
        self.base = source_spine
        self.by_turn = {section.spans[0].turn_id: section
            for sections in source_spine.by_source.values() for section in sections}

    def expand(self, selected):
        prior, audit = self.base.expand(selected)
        protected = [r.section for r in prior.routes if all(p.role == "user" for p in r.section.spans)]
        attached = [r.section for r in prior.routes if not all(p.role == "user" for p in r.section.spans)]
        seen = {s.section_id for s in protected}
        additions = []
        for route in selected.routes:
            for span in route.section.spans:
                if span.role != "user":
                    continue
                section = self.by_turn[span.turn_id]
                if section.section_id not in seen:
                    additions.append(section)
                    seen.add(section.section_id)
        sections = (*protected, *additions, *attached)
        routes = tuple(SectionRoute(section, 1 / (i + 1), ()) for i, section in enumerate(sections))
        plan = SectionRoutePlan(prior.index_sha256, prior.query_sha256, routes,
            len(self.base.index.sections), None, max(1, len(routes)), routing_backend="summary_hybrid")
        return plan, {**audit, "protected_user_section_ids": [s.section_id for s in protected],
            "additional_routed_user_section_ids": [s.section_id for s in additions],
            "prior_user_order_preserved": True, "extra_sources_require_an_existing_summary_route": True,
            "raw_reads_during_expansion": 0, "attached_context_may_be_displaced": True,
            "final_context_budget_enforced_by_hydrator": True}
