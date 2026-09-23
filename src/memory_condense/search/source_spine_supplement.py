"""Offer additional summary-selected users after the baseline user evidence."""
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.source_spine_overflow import SourceSpineOverflow


class SourceSpineSupplement:
    def __init__(self, source_spine):
        self.overflow = SourceSpineOverflow(source_spine)
        self.base = source_spine

    def expand(self, selected, supplement):
        if (supplement.index_sha256 != self.base.parent_sha256 or
                supplement.query_sha256 != selected.query_sha256 or any(
                    self.base.originals.get(r.section.section_id) != r.section for r in supplement.routes)):
            raise ValueError("supplement must bind the same query and unchanged leaf descriptors")
        original, audit = self.overflow.expand(selected)
        users = [r.section for r in original.routes if all(p.role == "user" for p in r.section.spans)]
        attached = [r.section for r in original.routes if not all(p.role == "user" for p in r.section.spans)]
        seen = {s.section_id for s in users}
        additions = []
        for route in supplement.routes:
            for span in route.section.spans:
                if span.role == "user":
                    section = self.overflow.by_turn[span.turn_id]
                    if section.section_id not in seen:
                        additions.append(section)
                        seen.add(section.section_id)
        sections = (*users, *additions, *attached)
        plan = SectionRoutePlan(original.index_sha256, original.query_sha256,
            tuple(SectionRoute(s, 1 / (i + 1), ()) for i, s in enumerate(sections)),
            len(self.base.index.sections), None, max(1, len(sections)), routing_backend="summary_hybrid")
        return plan, {**audit, "baseline_user_section_ids": [s.section_id for s in users],
            "supplemental_user_section_ids": [s.section_id for s in additions],
            "baseline_user_order_preserved": True, "raw_reads_during_expansion": 0,
            "attached_context_may_be_displaced": True}
