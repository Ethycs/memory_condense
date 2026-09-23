from dataclasses import replace
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.as_of_spine_routing import _plan
from memory_condense.search.relative_spine_reservation import RelativeSpineReservation
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.source_spine_hydration import SourceSpineHydrationIndex


def fixture():
    leaves, atoms, turns = [], [], {}
    for i in range(15):
        day = 10 if i < 13 else 20
        role = 'user' if i != 12 else 'assistant'
        turn = Turn(turn_id=f't{i}', source_id=f'opaque-{i}', role=role,
            text=f'Exact original observation {i}.', created_at=datetime(2026, 9, day, tzinfo=timezone.utc))
        turns[turn.turn_id] = turn
        span = RawSectionSpan.from_turn(turn)
        atoms.append(SectionSummary(f'a{i}', turn.source_id, 'Summary.', (span,), 'fixture'))
        leaves.append(SectionSummary(f'l{i:02}', turn.source_id,
            json.dumps({'attached_context_not_user_assertions': 'Context.'}), (span,), 'fixture'))
    hierarchy = SectionSummaryIndex(leaves)
    source = SourceSpineHydrationIndex(hierarchy, atoms)
    router = SimpleNamespace(semantic=SimpleNamespace(hierarchy=hierarchy), sections=tuple(leaves))
    reserve = RelativeSpineReservation(router, source)
    query = 'What did I acquire nine days ago?'
    dated = '[Question asked at 2026/09/19 (Sat) 12:00]\n' + query
    prior = _plan(source.index.receipt_sha256, quote_sha256(query), ())
    return reserve, turns, query, dated, prior


def test_expands_beyond_four_sources_without_reading_raw_or_admitting_future_and_assistant_turns():
    reserve, turns, query, dated, prior = fixture()
    plan, audit = reserve.reserve(query, dated, prior, list(range(15)))
    assert audit['reserved_source_count'] == 12
    assert audit['raw_reads_during_reservation'] == 0
    loaded = []
    def load(turn_id):
        loaded.append(turn_id)
        return turns[turn_id]
    hydrated = hydrate_section_plan(plan, load_turn=load, max_context_tokens=3072, max_raw_spans=128)
    assert set(loaded) == {f't{i}' for i in range(12)}
    assert all(e.text == turns[e.span.turn_id].text for s in hydrated.sections for e in s.evidence)


@pytest.mark.parametrize('query', ['What did I acquire?', 'What did I acquire last month?',
    'Compare things acquired nine days ago and two weeks ago.'])
def test_no_new_plan_for_queries_without_one_explicit_relative_day(query):
    reserve, _, _, _, prior = fixture()
    prior = replace(prior, query_sha256=quote_sha256(query), receipt_sha256='')
    plan, audit = reserve.reserve(query, '[Question asked at 2026/09/19 (Sat) 12:00]\n' + query,
                                  prior, list(range(15)))
    assert plan is prior
    assert not audit['active']


def test_reservation_budget_leaves_room_for_original_packet_and_deduplicates_selected_turns():
    reserve, _, query, dated, prior = fixture()
    reserve.TOKEN_RESERVATION = 150
    selected = reserve.turns['t11']
    prior = _plan(prior.index_sha256, prior.query_sha256, (selected, reserve.turns['t0']))
    plan, audit = reserve.reserve(query, dated, prior, list(range(15)))
    assert audit['reservation_token_estimate'] <= 150
    assert len(plan.routes) == 3
    assert plan.routes[-1].section == reserve.turns['t0']
    assert len({p.receipt_sha256 for r in plan.routes for p in r.section.spans}) == 3


def test_foreign_coordinates_cannot_enter_reservation():
    reserve, _, query, dated, prior = fixture()
    foreign_turn = Turn(turn_id='foreign', source_id='foreign', role='user', text='Foreign.',
                        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc))
    section = SectionSummary('foreign', 'foreign', 'Summary.', (RawSectionSpan.from_turn(foreign_turn),), 'fixture')
    prior = _plan(prior.index_sha256, prior.query_sha256, (section,))
    with pytest.raises(ValueError, match='coordinates'):
        reserve.reserve(query, dated, prior, list(range(15)))


def test_question_date_must_be_bound_to_exact_query():
    reserve, _, query, dated, prior = fixture()
    with pytest.raises(ValueError, match='bound'):
        reserve.reserve(query + ' changed', dated, prior, list(range(15)))
