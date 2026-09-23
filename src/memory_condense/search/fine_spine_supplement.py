"""Rank additional whole user turns using existing fine summary addresses."""
from datetime import datetime

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.as_of_spine_routing import _plan
from memory_condense.search.summary_time_prior_v2 import question_day


class FineSpineSupplement:
    RANKED_TURN_LIMIT = 32

    def __init__(self, authenticated_fine_router):
        self.base = authenticated_fine_router

    def route_vector(self, query, dated_question, vector, *, embedding_identity, selected_turn_ids):
        fine = self.base.fine
        if embedding_identity != fine.embedding_identity:
            raise ValueError('supplement query encoder changed')
        asked = question_day(query, dated_question)
        scores = fine._dense.score_all(vector)
        ranked, seen = [], set()
        for i in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), fine.sections[i].section_id)):
            turn_id = fine.sections[i].spans[0].turn_id
            if turn_id in seen:
                continue
            seen.add(turn_id)
            section = self.base.turns[turn_id]
            if any(p.role != 'user' or datetime.fromisoformat(p.created_at).date() > asked for p in section.spans):
                continue
            ranked.append(section)
            if len(ranked) == self.RANKED_TURN_LIMIT:
                break
        # Give each source one additional turn before taking another from it.
        # Existing evidence is excluded using coordinates, never raw text.
        groups = {}
        for section in ranked:
            if not any(p.turn_id in selected_turn_ids for p in section.spans):
                groups.setdefault(section.source_id, []).append(section)
        ordered = []
        while any(groups.values()):
            for rows in groups.values():
                if rows:
                    ordered.append(rows.pop(0))
        return _plan(fine.receipt_sha256, quote_sha256(query), ordered), {
            'fine_index_sha256': fine.receipt_sha256, 'ranked_turn_limit': self.RANKED_TURN_LIMIT,
            'ranked_user_turn_count': len(ranked), 'candidate_count': len(ordered),
            'asked_day': asked.isoformat(), 'raw_reads_during_routing': 0,
            'ranking_input': 'stored generated user-fragment summaries', 'source_round_robin': True}
