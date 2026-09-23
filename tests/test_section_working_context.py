from datetime import datetime, timezone

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.section_working_context import prioritize_unseen_direct_routes


def fixture_atom(name, text):
    turn = Turn(turn_id=name, source_id='source', role='system', text=text,
                created_at=datetime(2026, 9, 22, tzinfo=timezone.utc))
    atom = SectionSummary(name, 'source', name, (RawSectionSpan.from_turn(turn),), quote_sha256(name))
    return turn, SectionRoute(atom, 1., ())


def plan(routes):
    return SectionRoutePlan(quote_sha256('index'), quote_sha256('query'), tuple(routes),
                            len(routes), ('source',), max(1, len(routes)), routing_backend='summary_hybrid')


def test_direct_evidence_survives_budget_consumed_by_neighborhood():
    context, c = fixture_atom('context', 'neighborhood ' * 250)
    target, t = fixture_atom('target', 'useful ' * 200)
    lookup = {context.turn_id: context, target.turn_id: target}
    baseline, expanded = plan([t]), plan([c, t])
    before = hydrate_section_plan(expanded, load_turn=lookup.get, max_context_tokens=500)
    after = hydrate_section_plan(prioritize_unseen_direct_routes(baseline, expanded),
                                load_turn=lookup.get, max_context_tokens=500)
    assert [section.section.section_id for section in before.sections] == ['context']
    assert [section.section.section_id for section in after.sections] == ['target']
    assert after.sections[0].evidence[0].text == target.text


def test_visible_and_metadata_excluded_live_deferred_without_raw_reads():
    pairs = [fixture_atom(name, name) for name in ('live', 'visible', 'metadata', 'direct', 'neighbor')]
    live, visible, metadata, direct, neighbor = [r for _, r in pairs]
    result = prioritize_unseen_direct_routes(plan([live, visible, metadata, direct]),
        plan([neighbor, live, visible, metadata, direct]), visible_section_ids=['visible'],
        excluded_turn_ids=['metadata'], deferred_turn_ids=['live'])
    assert [r.section.section_id for r in result.routes] == ['direct', 'neighbor', 'live']
    assert result.routes[0] is direct
    assert result.routes[-1] is live  # An incomplete live preview does not erase its evidence.
    loaded = []
    lookup = {t.turn_id: t for t, _ in pairs}
    def load(turn_id):
        loaded.append(turn_id)
        return lookup[turn_id]
    hydrate_section_plan(result, load_turn=load, max_context_tokens=2048)
    assert set(loaded) == {'direct', 'neighbor', 'live'}


def test_cannot_add_a_direct_address_absent_from_expansion():
    _, direct = fixture_atom('direct', 'direct')
    _, foreign = fixture_atom('foreign', 'foreign')
    with pytest.raises(ValueError, match='preserve every direct'):
        prioritize_unseen_direct_routes(plan([foreign]), plan([direct]))
