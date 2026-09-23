from datetime import date, datetime, timezone

import pytest

from memory_condense.application.additive_threaded_context import supplement_threaded_context, attempted_spans
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.application.threaded_section_context import TranscriptOrder
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.as_of_spine_routing import _plan
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary


def turn(i, *, words=8, role='user', future=False):
    return Turn(turn_id=f't{i}', source_id=f'source{i // 3}', role=role,
                text=f'Exact Café 雪 {i}. ' + 'word ' * words,
                created_at=datetime(2026, 9, 20 if future else 10, tzinfo=timezone.utc))


def section(t):
    return SectionSummary(f'section-{t.turn_id}', t.source_id, 'PRIVATE GENERATED SUMMARY',
                          (RawSectionSpan.from_turn(t),), 'fixture')


def prepared(base_turns, extras):
    lookup = {t.turn_id: t for t in (*base_turns, *extras)}
    base_plan = _plan('a' * 64, quote_sha256('question'), [section(t) for t in base_turns])
    base = hydrate_section_plan(base_plan, load_turn=lookup.get, max_context_tokens=3072, max_raw_spans=128)
    candidates = _plan('b' * 64, base_plan.query_sha256, [section(t) for t in extras])
    return lookup, base, candidates, TranscriptOrder(lookup.values())


def test_addition_preserves_all_original_exact_spans_and_enforces_eight_turn_limit():
    lookup, base, candidates, order = prepared([turn(0)], [turn(i) for i in range(1, 12)])
    result = supplement_threaded_context(base, order, candidates, load_turn=lookup.get, asked_day=date(2026, 9, 10))
    assert len(result.added_section_ids) == 8
    assert result.extra_raw_read_count == 8
    assert result.attempted_raw_spans == 9
    assert result.rendered.token_count <= 3072
    assert 'PRIVATE GENERATED SUMMARY' not in result.rendered.text
    expected = {r.span.receipt_sha256: r.text for s in result.hydration.sections for r in s.evidence}
    assert all(result.rendered.text[start:end] == expected[sha] for sha, start, end in result.rendered.placements)
    assert lookup['t0'].text in result.rendered.text
    assert result.base_hydration_sha256 == base.receipt_sha256


def test_over_budget_addition_is_skipped_without_evicting_old_evidence_and_later_small_one_fits():
    lookup, base, candidates, order = prepared([turn(i, words=55) for i in range(20)],
                                              [turn(30, words=2900), turn(31, words=2)])
    original = {r.span.receipt_sha256 for s in base.sections for r in s.evidence}
    result = supplement_threaded_context(base, order, candidates, load_turn=lookup.get, asked_day=date(2026, 9, 10))
    assert result.added_section_ids == ('section-t31',)
    assert ('section-t30', 'reader_context_budget') in result.diagnostics
    assert original <= {sha for sha, _, _ in result.rendered.placements}
    assert result.rendered.token_count <= 3072
    assert result.extra_raw_read_count == 2


def test_future_duplicate_and_exhausted_raw_span_budget_do_not_read_additional_text():
    lookup, base, candidates, order = prepared([turn(0)], [turn(1, future=True)])
    candidates = _plan(candidates.index_sha256, candidates.query_sha256,
                       [section(lookup['t0']), *[r.section for r in candidates.routes]])
    def forbidden(_):
        pytest.fail('ineligible turn was read')
    result = supplement_threaded_context(base, order, candidates, load_turn=forbidden, asked_day=date(2026, 9, 10))
    assert result.hydration is base and not result.added_section_ids
    assert {reason for _, reason in result.diagnostics} == {'existing_turn', 'future_turn'}
    lookup, base, candidates, order = prepared([turn(i, words=1) for i in range(128)], [turn(130)])
    assert attempted_spans(base) == 128
    result = supplement_threaded_context(base, order, candidates, load_turn=forbidden, asked_day=date(2026, 9, 10))
    assert result.attempted_raw_spans == 128 and result.extra_raw_read_count == 0


def test_missing_candidate_does_not_prevent_later_valid_evidence():
    lookup, base, candidates, order = prepared([turn(0)], [turn(1), turn(2)])
    result = supplement_threaded_context(base, order, candidates,
        load_turn=lambda tid: None if tid == 't1' else lookup[tid], asked_day=date(2026, 9, 10))
    assert result.added_section_ids == ('section-t2',)
    assert ('section-t1', 'raw_turn_missing') in result.diagnostics


@pytest.mark.parametrize('corruption', ['query', 'assistant', 'partial', 'candidate_cap'])
def test_invalid_supplement_cannot_change_the_reader_packet(corruption):
    lookup, base, candidates, order = prepared([turn(0)], [turn(i) for i in range(1, 35)])
    sections = [r.section for r in candidates.routes[:1]]
    query_sha = base.plan.query_sha256
    if corruption == 'query':
        query_sha = 'c' * 64
    elif corruption == 'assistant':
        sections = [section(turn(1, role='assistant'))]
    elif corruption == 'partial':
        t = lookup['t1']
        sections = [SectionSummary('partial', t.source_id, 'Summary',
                    (RawSectionSpan.from_turn(t, start_char=0, end_char=10),), 'fixture')]
    else:
        sections = [r.section for r in candidates.routes]
    candidates = _plan(candidates.index_sha256, query_sha, sections)
    with pytest.raises(ValueError):
        supplement_threaded_context(base, order, candidates, load_turn=lookup.get, asked_day=date(2026, 9, 10))
