"""Retrieve exact user turns through their individual fragment summaries.

Coarse summaries still provide date reservations and attached conversation
context. Fine addresses replace unconditional conversation backfill. No raw
text is read while scoring or selecting coordinates.
"""
from datetime import datetime

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.as_of_spine_routing import _plan
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_time_prior_v2 import question_day


class FineSpineRouter:
    def __init__(self, fine_index, source_spine):
        self.fine = fine_index
        self.turns = {s.spans[0].turn_id: s for rows in source_spine.by_source.values() for s in rows}
        expected = {p.receipt_sha256 for s in self.turns.values() for p in s.spans}
        observed = [p.receipt_sha256 for s in fine_index.sections for p in s.spans]
        if (len(observed) != len(set(observed)) or set(observed) != expected or
                any(len(s.spans) != 1 or s.spans[0].role != 'user' for s in fine_index.sections)):
            raise ValueError('fine summaries must partition all authenticated user fragments')
        self.allowed = {p.receipt_sha256 for s in source_spine.index.sections for p in s.spans}

    def route_vector(self, query, dated_question, vector, *, embedding_identity, prior, reserved_ids=()):
        asked = question_day(query, dated_question)
        if embedding_identity != self.fine.embedding_identity:
            raise ValueError('fine query encoder changed')
        if (prior.query_sha256 != quote_sha256(query) or any(
                p.receipt_sha256 not in self.allowed or datetime.fromisoformat(p.created_at).date() > asked
                for r in prior.routes for p in r.section.spans)):
            raise ValueError('fine routing prior changed source or query')
        by_id = {r.section.section_id: r.section for r in prior.routes}
        if len(set(reserved_ids)) != len(reserved_ids) or any(sid not in by_id for sid in reserved_ids):
            raise ValueError('date reservation must belong to the prior plan')
        reserved = [by_id[sid] for sid in reserved_ids]
        if any(p.role != 'user' or datetime.fromisoformat(p.created_at).date() > asked
               for s in reserved for p in s.spans):
            raise ValueError('date reservation must contain eligible user evidence')
        scores = self.fine._dense.score_all(vector)
        ranked, seen = [], set()
        for i in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.fine.sections[i].section_id)):
            span = self.fine.sections[i].spans[0]
            if span.turn_id in seen:
                continue
            seen.add(span.turn_id)
            section = self.turns[span.turn_id]
            if any(datetime.fromisoformat(p.created_at).date() > asked for p in section.spans):
                continue
            ranked.append(section)
            if len(ranked) == 32:
                break
        contexts = [r.section for r in prior.routes if all(p.role != 'user' for p in r.section.spans)]
        # The first eight direct user matches precede two attached contexts;
        # remaining user matches then precede the old attached-context tail.
        selected = tuple({s.section_id: s for s in (
            *reserved, *ranked[:8], *contexts[:2], *ranked[8:], *contexts[2:])}.values())
        index = SectionSummaryIndex(selected)
        plan = _plan(index.receipt_sha256, quote_sha256(query), selected)
        return plan, {'fine_user_sections': [s.section_id for s in ranked],
            'date_reserved_sections': list(reserved_ids), 'direct_user_limit': 32,
            'primary_user_limit': 8, 'early_context_limit': 2, 'raw_reads_during_routing': 0,
            'fine_index_sha256': self.fine.receipt_sha256,
            'prior_plan_sha256': prior.receipt_sha256, 'asked_day': asked.isoformat()}
