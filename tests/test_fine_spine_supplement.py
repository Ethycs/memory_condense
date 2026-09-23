from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memory_condense.domain.schemas import Turn
from memory_condense.search.fine_spine_supplement import FineSpineSupplement
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary


def fixture(count=6):
    sections = []
    for i in range(count):
        t = Turn(turn_id=f't{i}', source_id='A' if i < 3 else f'B{i}', role='user',
                 text='RAW TEXT NEVER USED TO RANK',
                 created_at=datetime(2026, 9, 20 if i == count-1 else 10, tzinfo=timezone.utc))
        sections.append(SectionSummary(f's{i:02}', t.source_id, f'Generated summary {i}',
                                      (RawSectionSpan.from_turn(t),), 'fixture'))
    fine = SimpleNamespace(sections=sections, embedding_identity='encoder', receipt_sha256='a' * 64,
                           _dense=SimpleNamespace(score_all=lambda _: list(reversed(range(count)))))
    return FineSpineSupplement(SimpleNamespace(fine=fine, turns={s.spans[0].turn_id: s for s in sections}))


def test_routing_excludes_existing_and_future_turns_and_spreads_sources_without_raw_access():
    router = fixture()
    plan, audit = router.route_vector('question', '[Question asked at 2026/09/10 (Thu) 12:00]\nquestion',
                                      [1, 0], embedding_identity='encoder', selected_turn_ids={'t0'})
    assert [r.section.spans[0].turn_id for r in plan.routes] == ['t1', 't3', 't4', 't2']
    assert audit['raw_reads_during_routing'] == 0
    assert audit['ranked_user_turn_count'] == 5


def test_ranked_population_remains_bounded_even_when_all_top_turns_are_already_present():
    router = fixture(40)
    plan, audit = router.route_vector('question', '[Question asked at 2026/09/10 (Thu) 12:00]\nquestion',
                                      [1, 0], embedding_identity='encoder', selected_turn_ids={f't{i}' for i in range(32)})
    assert not plan.routes and audit['ranked_user_turn_count'] == 32


def test_changed_encoder_is_rejected_before_scoring():
    with pytest.raises(ValueError, match='encoder'):
        fixture().route_vector('question', '', [1, 0], embedding_identity='other', selected_turn_ids=set())
