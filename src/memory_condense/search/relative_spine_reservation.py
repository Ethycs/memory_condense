"""Reserve source-diverse user turns for one explicit relative day hint.

The day is a retrieval prior, never proof of an event date. This selection reads
summary scores and authenticated coordinates only. Exact raw hydration and its
unchanged 3,072-token limit remain the final budget authority.
"""
from datetime import datetime, timedelta

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.as_of_spine_routing import _plan
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_time_prior_v2 import (
    _AGO, question_day, relative_mention_window,
)


class RelativeSpineReservation:
    SOURCE_CAP = 12
    TOKEN_RESERVATION = 1536

    def __init__(self, router, source_spine):
        if router.semantic.hierarchy.receipt_sha256 != source_spine.parent_sha256:
            raise ValueError("relative reservation requires the bound source partition")
        self.router = router
        self.turns = {s.spans[0].turn_id: s
                      for rows in source_spine.by_source.values() for s in rows}
        self.allowed = {p.receipt_sha256 for s in source_spine.index.sections for p in s.spans}

    def reserve(self, query, dated_question, prior, user_scores):
        asked = question_day(query, dated_question)
        if prior.query_sha256 != quote_sha256(query) or any(
                p.receipt_sha256 not in self.allowed for r in prior.routes for p in r.section.spans):
            raise ValueError("relative reservation changed the query or source coordinates")
        if len(user_scores) != len(self.router.sections):
            raise ValueError("relative reservation scores do not cover the summary index")
        window = relative_mention_window(dated_question) if _AGO.search(query) else None
        if window is None or window[1] - window[0] != timedelta(days=1):
            return prior, {"active": False, "raw_reads_during_reservation": 0}

        candidates = []
        seen_turns = set()
        # Rank only eligible user mentions. An assistant's date cannot qualify
        # another user turn, and a source can have several mention dates.
        for i in sorted(range(len(user_scores)),
                        key=lambda i: (-float(user_scores[i]), self.router.sections[i].section_id)):
            for span in self.router.sections[i].spans:
                if span.role != "user" or span.turn_id in seen_turns:
                    continue
                seen_turns.add(span.turn_id)
                section = self.turns[span.turn_id]
                days = [datetime.fromisoformat(p.created_at).date() for p in section.spans]
                if all(window[0] <= day < window[1] and day <= asked for day in days):
                    candidates.append(section)

        selected, sources, reserved = [], set(), 0
        for section in candidates:
            estimate = sum(p.token_count for p in section.spans) + 48 * len(section.spans) + 16
            if section.source_id in sources or reserved + estimate > self.TOKEN_RESERVATION:
                continue
            selected.append(section)
            sources.add(section.source_id)
            reserved += estimate
            if len(sources) == self.SOURCE_CAP:
                break
        audit = {"active": True, "mention_window": [day.isoformat() for day in window],
                 "candidate_user_turn_count": len(candidates), "reserved_source_count": len(sources),
                 "reserved_section_ids": [s.section_id for s in selected],
                 "reservation_token_estimate": reserved, "source_cap": self.SOURCE_CAP,
                 "reservation_limit": self.TOKEN_RESERVATION, "raw_reads_during_reservation": 0,
                 "event_date_inferred_from_mention_date": False}
        if not selected:
            return prior, audit
        sections = tuple({s.section_id: s for s in (
            *selected, *(r.section for r in prior.routes))}.values())
        index = SectionSummaryIndex(sections)
        return _plan(index.receipt_sha256, prior.query_sha256, sections), audit
