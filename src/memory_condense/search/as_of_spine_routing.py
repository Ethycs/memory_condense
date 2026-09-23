"""Date-eligible summary routing and authenticated projection of exact raw spans.

Summary vectors stay unchanged. A mixed-date leaf remains addressable if it has
any eligible span, so its summary may still describe later mentions. Only raw
spans dated on or before the question day can reach the answer model. Event
dates inside text do not determine eligibility. No raw text is read here.
"""
from copy import copy
from datetime import datetime

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan, SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_time_prior import mention_window
from memory_condense.search.summary_time_prior_v2 import question_day, relative_mention_window


def _plan(index_sha, query_sha, sections, *, population=None):
    sections = tuple({section.section_id: section for section in sections}.values())
    return SectionRoutePlan(index_sha, query_sha,
        tuple(SectionRoute(section, 1 / (i + 1), ()) for i, section in enumerate(sections)),
        len(sections) if population is None else population, None, max(1, len(sections)),
        routing_backend="summary_dense")


class AsOfSpineRouter:
    def __init__(self, semantic, users, facets):
        if (len({channel.hierarchy.receipt_sha256 for channel in (semantic, users, facets)}) != 1 or
                len({channel.embedding_identity for channel in (semantic, users, facets)}) != 1 or
                semantic.sections != users.sections or semantic.sections != facets.sections):
            raise ValueError("as-of summary channels must bind the same hierarchy and encoder")
        self.semantic, self.users, self.facets = semantic, users, facets
        self.sections = semantic.sections
        self.by_id = {section.section_id: section for section in self.sections}
        self.days = {s.section_id: tuple(datetime.fromisoformat(p.created_at).date() for p in s.spans)
                     for s in self.sections}
        self.minimum_day = {sid: min(days) for sid, days in self.days.items()}

    def route_vectors(self, query, dated_question, vector, *, embedding_identity,
                      content_vector=None, extended_relative_prior=False):
        asked = question_day(query, dated_question)
        if embedding_identity != self.semantic.embedding_identity:
            raise ValueError("query vector encoder identity changed")
        view = ordered_content_query(query)
        if (view != query) != (content_vector is not None):
            raise ValueError("ordering content view requires its own live query vector")
        if type(extended_relative_prior) is not bool:
            raise ValueError("relative-prior option must be boolean")
        eligible = [i for i, s in enumerate(self.sections) if self.minimum_day[s.section_id] <= asked]
        dense_scores = self.semantic._dense.score_all(vector)
        user_scores = self.users._spine.score_all(vector)
        facet_scores = self.facets._dense.score_all(vector)
        best = {}
        for facet, score in zip(self.facets.facets, facet_scores, strict=True):
            best[facet.section_id] = max(best.get(facet.section_id, float("-inf")), float(score))

        def ranked(values, candidates=eligible):
            return [self.sections[i] for i in sorted(candidates,
                    key=lambda i: (-float(values[i]), self.sections[i].section_id))]

        dense = ranked(dense_scores)[:6]
        user_rank = ranked(user_scores)[:32]
        facet_rank = sorted((self.sections[i] for i in eligible),
                            key=lambda s: (-best[s.section_id], s.section_id))[:32]
        window = relative_mention_window(dated_question) if extended_relative_prior else mention_window(dated_question)
        preferred, sources = [], set()
        if window:
            start, end = window
            sources = {self.sections[i].source_id for i in eligible
                       if any(start <= day < end and day <= asked for day in self.days[self.sections[i].section_id])}
            if sources:
                values = user_scores if content_vector is None else self.users._spine.score_all(content_vector)
                dated = ranked(values, [i for i in eligible if self.sections[i].source_id in sources])[:32]
                seen = set()
                for section in dated:
                    if section.source_id not in seen:
                        preferred.append(section)
                        seen.add(section.source_id)
                    if len(preferred) == 4:
                        break
        index_sha, query_sha = self.semantic.hierarchy.receipt_sha256, quote_sha256(query)

        def plan(sections):
            return _plan(index_sha, query_sha, sections, population=len(eligible))

        return {"selected": plan((*preferred, *dense, *user_rank[:4])),
            "users": plan(user_rank[:8]), "facets": plan(facet_rank[:8]),
            "user_frontier": plan(user_rank), "facet_frontier": plan(facet_rank)}, {
                "asked_day": asked.isoformat(), "eligible_leaf_count": len(eligible),
                "excluded_future_leaf_count": len(self.sections) - len(eligible),
                "mixed_date_eligible_leaf_count": sum(max(self.days[self.sections[i].section_id]) > asked for i in eligible),
                "baseline_section_ids": [s.section_id for s in dense],
                "preferred_section_ids": [s.section_id for s in preferred],
                "user_supplement_section_ids": [s.section_id for s in user_rank[:4]],
                "mention_window": [day.isoformat() for day in window] if window else None,
                "extended_relative_prior": extended_relative_prior, "preferred_source_count": len(sources),
                "filter_before_top_k": True, "same_day_retained": True,
                "mixed_leaf_summary_can_describe_later_mentions": True,
                "event_dates_used_for_eligibility": False, "raw_reads_during_routing": 0}


class AsOfSpineExpansion:
    def __init__(self, source_spine, supplement, source_coverage, term_coverage):
        if (supplement.base is not source_spine or source_coverage.base is not source_spine or
                term_coverage.source_spine is not source_spine):
            raise ValueError("as-of expansion requires the same authenticated source partition")
        self.base, self.supplement = source_spine, supplement
        self.coverage, self.terms = source_coverage, term_coverage
        self.sections = {s.section_id: s for s in source_spine.index.sections}
        self.days = {p.created_at: datetime.fromisoformat(p.created_at).date()
                     for s in self.sections.values() for p in s.spans}
        self.user_days = {}
        for rows in source_spine.by_source.values():
            for section in rows:
                days = {self.days[p.created_at] for p in section.spans}
                if len(days) != 1:
                    raise ValueError("fragments of one user turn must share their mention day")
                self.user_days[section.section_id] = days.pop()

    def project(self, prior, asked):
        if prior.index_sha256 != self.base.index.receipt_sha256 or prior.eligible_source_ids is not None or any(
                self.sections.get(r.section.section_id) != r.section for r in prior.routes):
            raise ValueError("as-of projection requires unchanged bound source descriptors")
        sections, projected, excluded = [], [], []
        for route in prior.routes:
            original = route.section
            spans = tuple(p for p in original.spans if self.days[p.created_at] <= asked)
            if not spans:
                excluded.append(original.section_id)
                continue
            section = original
            if spans != original.spans:
                binding = {"parent_section_sha256": original.receipt_sha256,
                           "asked_day": asked.isoformat(), "span_sha256s": [p.receipt_sha256 for p in spans]}
                section = SectionSummary("as-of-" + identity_sha256(binding), original.source_id,
                    original.summary, spans, "as-of-source-projection-v1")
                projected.append({**binding, "projected_section_sha256": section.receipt_sha256})
            sections.append(section)
        index = SectionSummaryIndex(sections)
        plan = _plan(index.receipt_sha256, prior.query_sha256, sections)
        return plan, {"parent_index_sha256": prior.index_sha256, "parent_route_sha256": prior.receipt_sha256,
            "projected_index_sha256": index.receipt_sha256, "asked_day": asked.isoformat(),
            "excluded_section_ids": excluded, "partial_projections": projected,
            "unchanged_span_descriptors": True, "raw_reads_during_projection": 0,
            "selected_projection_is_not_complete_corpus": True}

    def expand(self, query, dated_question, plans, supplement_plan):
        asked = question_day(query, dated_question)
        if any(plan.query_sha256 != quote_sha256(query) for plan in (*plans.values(), supplement_plan)):
            raise ValueError("as-of expansion query binding changed")
        # Filter before the first 2,048-token metadata reservation. Keep the full
        # by-turn lookup for mixed leaves; later additions have no token budget
        # until projection removes their future spans and hydration runs once.
        source_view = copy(self.base)
        source_view.by_source = {source: [s for s in rows if self.user_days[s.section_id] <= asked]
                                for source, rows in self.base.by_source.items()}
        supplement = copy(self.supplement)
        supplement.overflow = copy(self.supplement.overflow)
        supplement.overflow.base = source_view
        prior, supplement_audit = supplement.expand(plans["selected"], supplement_plan)
        diverse, coverage_audit = self.coverage.expand(prior, plans["user_frontier"], plans["facet_frontier"])
        terms = copy(self.terms)
        terms.leaves = tuple(s for s in self.terms.leaves if any(self.days[p.created_at] <= asked for p in s.spans))
        expanded, term_audit = terms.expand(query, diverse)
        final, projection_audit = self.project(expanded, asked)
        return final, {"supplement": supplement_audit, "source_coverage": coverage_audit,
            "term_coverage": term_audit, "projection": projection_audit,
            "user_backfill_filtered_before_reservation": True,
            "old_turns_in_mixed_date_sources_retained_as_candidates": True}
