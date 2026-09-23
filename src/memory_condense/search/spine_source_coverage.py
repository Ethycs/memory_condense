"""Offer source-diverse summary matches after previously selected user turns.

Eight slots in each summary channel represent distinct conversations. Selection
examines at most 32 already ranked leaf descriptors per channel and never reads
raw text. Previous user evidence stays first; added users may displace context.
"""
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.source_spine_overflow import SourceSpineOverflow


class SpineSourceCoverage:
    FRONTIER = 32
    SOURCES_PER_CHANNEL = 8

    def __init__(self, source_spine):
        self.base = source_spine
        self.by_turn = SourceSpineOverflow(source_spine).by_turn
        self.sections = {s.section_id: s for s in source_spine.index.sections}

    def expand(self, prior, user_frontier, facet_frontier):
        if (prior.index_sha256 != self.base.index.receipt_sha256 or
                prior.eligible_source_ids is not None or any(
                    self.sections.get(r.section.section_id) != r.section for r in prior.routes)):
            raise ValueError("prior must bind the unchanged unscoped source partition")
        channels, audit_channels = [], []
        for frontier in (user_frontier, facet_frontier):
            if (frontier.index_sha256 != self.base.parent_sha256 or
                    frontier.query_sha256 != prior.query_sha256 or frontier.eligible_source_ids is not None or
                    len(frontier.routes) > self.FRONTIER or any(
                        self.base.originals.get(r.section.section_id) != r.section for r in frontier.routes)):
                raise ValueError("frontier must bind the same query and unchanged unscoped summary leaves")
            sources, selected, audited = set(), [], []
            for rank, route in enumerate(frontier.routes, 1):
                section = route.section
                if section.source_id in sources or not any(p.role == "user" for p in section.spans):
                    continue
                sources.add(section.source_id)
                selected.append(section)
                audited.append({"rank": rank, "source_id": section.source_id, "section_id": section.section_id})
                if len(sources) == self.SOURCES_PER_CHANNEL:
                    break
            channels.append(selected)
            audit_channels.append(audited)
        users = [r.section for r in prior.routes if all(p.role == "user" for p in r.section.spans)]
        attached = [r.section for r in prior.routes if not all(p.role == "user" for p in r.section.spans)]
        seen = {s.section_id for s in users}
        additions = []
        for i in range(max(map(len, channels))):
            for channel in channels:
                if i >= len(channel):
                    continue
                for span in channel[i].spans:
                    if span.role != "user":
                        continue
                    section = self.by_turn[span.turn_id]
                    if section.section_id not in seen:
                        additions.append(section)
                        seen.add(section.section_id)
        sections = (*users, *additions, *attached)
        plan = SectionRoutePlan(prior.index_sha256, prior.query_sha256,
            tuple(SectionRoute(s, 1 / (i + 1), ()) for i, s in enumerate(sections)),
            len(self.base.index.sections), None, max(1, len(sections)), routing_backend="summary_hybrid")
        return plan, {"user_channel": audit_channels[0], "facet_channel": audit_channels[1],
            "frontier": self.FRONTIER, "sources_per_channel": self.SOURCES_PER_CHANNEL,
            "baseline_user_section_ids": [s.section_id for s in users],
            "added_user_section_ids": [s.section_id for s in additions],
            "added_source_ids": sorted({s.source_id for s in additions} - {r.section.source_id for r in prior.routes}),
            "prior_user_order_preserved": True, "raw_reads_during_expansion": 0,
            "attached_context_may_be_displaced": True}
