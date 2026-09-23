"""Add user-summary and calendar addresses to the existing summary route.

All candidates remain exact leaf descriptors. Calendar dates are mention-time
priors, never eligibility restrictions or certified event dates. No raw text or
reference labels are available to the router.
"""
from datetime import datetime

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_time_prior import _ASKED, mention_window


class SpineUnionRouter:
    def __init__(self, semantic, user_addresses):
        if (semantic.hierarchy.receipt_sha256 != user_addresses.hierarchy.receipt_sha256 or
                semantic.embedding_identity != user_addresses.embedding_identity):
            raise ValueError("summary channels belong to different memories or encoders")
        self.semantic = semantic
        self.user_addresses = user_addresses
        self.source_dates = {}
        for section in semantic.sections:
            self.source_dates.setdefault(section.source_id, set()).update(
                datetime.fromisoformat(span.created_at).date() for span in section.spans)

    def route_vectors(self, query, dated_question, vector, *, embedding_identity, content_vector=None):
        if _ASKED.sub("", dated_question).strip() != query.strip():
            raise ValueError("dated question does not bind the retrieval query")
        view = ordered_content_query(query)
        if (view != query) != (content_vector is not None):
            raise ValueError("ordering content view requires its own live query vector")
        baseline = self.semantic.route_vector(query, vector, embedding_identity=embedding_identity,
                                               max_sections=6, lexical_reserve=2)
        user = self.user_addresses.route_vector(query, vector, embedding_identity=embedding_identity,
                                                max_sections=4, user_weight=1.0)
        preferred = []
        window = mention_window(dated_question)
        sources = ()
        if window is not None:
            start, end = window
            sources = tuple(sorted(source for source, dates in self.source_dates.items()
                                   if any(start <= day < end for day in dates)))
            if sources:
                dated = self.user_addresses.route_vector(view, vector if content_vector is None else content_vector,
                    embedding_identity=embedding_identity, max_sections=32, user_weight=1.0,
                    eligible_source_ids=sources)
                seen_sources = set()
                for route in dated.routes:
                    if route.section.source_id not in seen_sources:
                        preferred.append(route)
                        seen_sources.add(route.section.source_id)
                    if len(preferred) == 4:
                        break
        selected = {}
        for route in (*preferred, *baseline.routes, *user.routes):
            selected.setdefault(route.section.section_id, route.section)
        routes = tuple(SectionRoute(section, 1 / (i + 1), ()) for i, section in enumerate(selected.values()))
        plan = SectionRoutePlan(self.semantic.hierarchy.receipt_sha256, quote_sha256(query), routes,
            len(self.semantic.sections), None, 14, routing_backend="summary_hybrid")
        return plan, {"baseline_section_ids": [r.section.section_id for r in baseline.routes],
            "user_supplement_section_ids": [r.section.section_id for r in user.routes],
            "preferred_section_ids": [r.section.section_id for r in preferred],
            "preferred_source_count": len(sources), "content_query": view,
            "mention_window": [d.isoformat() for d in window] if window else None,
            "hard_time_filter": False, "baseline_routes_retained": True,
            "hydration_budget_may_reject_sections": True}
