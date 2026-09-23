from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.application.threaded_section_context import TranscriptOrder, render_threaded_sections
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.as_of_spine_routing import _plan
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary


def fixture():
    turns = [Turn(turn_id=f't{i}', source_id='PRIVATE_SOURCE_A' if i < 3 else 'PRIVATE_SOURCE_B',
        role='user' if i % 2 == 0 else 'assistant',
        text=f'Exact source {i}: Café 雪. Second clause {i}.',
        created_at=datetime(2026, 9, 9 if i != 2 else 10, 12, tzinfo=timezone.utc)) for i in range(5)]
    sections = []
    for i, turn in enumerate(turns):
        if i == 0:
            cut = turn.text.index('Second')
            spans = (RawSectionSpan.from_turn(turn, start_char=0, end_char=cut),
                     RawSectionSpan.from_turn(turn, start_char=cut, end_char=len(turn.text)))
        else:
            spans = (RawSectionSpan.from_turn(turn),)
        sections.append(SectionSummary(f's{i}', turn.source_id, 'PRIVATE GENERATED SUMMARY', spans, 'fixture'))
    index = SectionSummaryIndex(sections)
    plan = _plan(index.receipt_sha256, quote_sha256('query'), [sections[i] for i in (2, 4, 0, 3, 1)])
    lookup = {t.turn_id: t for t in turns}
    hydrated = hydrate_section_plan(plan, load_turn=lookup.get, max_context_tokens=3072, max_raw_spans=128)
    return turns, hydrated


def test_groups_sources_restores_transcript_order_and_preserves_every_span_byte():
    turns, hydrated = fixture()
    result = render_threaded_sections(hydrated, TranscriptOrder(turns))
    assert result.conversation_count == 2
    assert [result.text.index(t.text) for t in turns] == sorted(result.text.index(t.text) for t in turns)
    expected = {row.span.receipt_sha256: row.text for s in hydrated.sections for row in s.evidence}
    assert set(expected) == {sha for sha, _, _ in result.placements}
    assert all(result.text[start:end] == expected[sha] for sha, start, end in result.placements)
    assert 'PRIVATE' not in result.text
    assert result.token_count < hydrated.context_token_count
    assert '<AT 2026-09-10T12:00:00+00:00>' in result.text


def test_transcript_order_and_rendering_metadata_are_immutable():
    turns, hydrated = fixture()
    order = TranscriptOrder(turns)
    with pytest.raises(TypeError):
        order.positions['t0'] = 9
    with pytest.raises(AttributeError):
        order.positions = {}
    result = render_threaded_sections(hydrated, order)
    payload = result.identity_payload()
    payload['placements'][0]['start_char'] = 99999
    assert result.identity_payload()['placements'][0]['start_char'] != 99999


@pytest.mark.parametrize('field', ['text', 'source_id', 'role', 'created_at'])
def test_changed_transcript_identity_is_rejected(field):
    turns, hydrated = fixture()
    values = {'text': 'Replacement bytes', 'source_id': 'foreign', 'role': 'assistant',
              'created_at': datetime(2026, 9, 8, tzinfo=timezone.utc)}
    turns[0] = turns[0].model_copy(update={field: values[field]})
    with pytest.raises(ValueError, match='bound transcript'):
        render_threaded_sections(hydrated, TranscriptOrder(turns))


def test_gaps_in_one_turn_are_not_silently_concatenated():
    turns, _ = fixture()
    turn = turns[0]
    spans = (RawSectionSpan.from_turn(turn, start_char=0, end_char=10),
             RawSectionSpan.from_turn(turn, start_char=20, end_char=len(turn.text)))
    sections = [SectionSummary(f'part{i}', turn.source_id, 'Summary', (p,), 'fixture') for i, p in enumerate(spans)]
    index = SectionSummaryIndex(sections)
    plan = _plan(index.receipt_sha256, quote_sha256('query'), sections)
    hydrated = hydrate_section_plan(plan, load_turn={turn.turn_id: turn}.get, max_context_tokens=3072, max_raw_spans=128)
    rendered = render_threaded_sections(hydrated, TranscriptOrder(turns))
    assert turn.text[:10] + turn.text[20:] not in rendered.text
    assert len(rendered.placements) == 2


def test_duplicate_transcript_turns_are_rejected():
    turns, _ = fixture()
    with pytest.raises(ValueError, match='unique'):
        TranscriptOrder([*turns, turns[0]])
