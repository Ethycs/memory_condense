from dataclasses import replace

import pytest

from memory_condense.application.user_spine_section_context import (
    FORMAT, TranscriptOrder, render_user_spine_sections,
)
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.as_of_spine_routing import _plan
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from tests.test_threaded_section_context import fixture


def test_user_statements_grouped_per_source_with_original_turn_labels_and_all_bytes():
    turns, hydrated = fixture()
    result = render_user_spine_sections(hydrated, TranscriptOrder(turns))
    assert result.identity_payload()['format'] == FORMAT
    # User turns 0 and 2 precede assistant 1, but T labels preserve chronology.
    assert result.text.index(turns[0].text) < result.text.index(turns[2].text) < result.text.index(turns[1].text)
    assert '<T3 user>' in result.text and '<T2 assistant>' in result.text
    assert result.text.index(turns[4].text) < result.text.index(turns[3].text)
    assert '<AT 2026-09-10T12:00:00+00:00>' in result.text
    assert 'PRIVATE' not in result.text
    assert result.token_count <= hydrated.max_context_tokens
    expected = {r.span.receipt_sha256: r.text for s in hydrated.sections for r in s.evidence}
    assert len(result.placements) == len(expected)
    assert {sha for sha, _, _ in result.placements} == set(expected)
    assert all(result.text[start:end] == expected[sha] for sha, start, end in result.placements)
    assert result.hydration_sha256 == hydrated.receipt_sha256


def test_gap_is_not_concatenated_and_raw_markup_is_never_parsed():
    turns, _ = fixture()
    turn = turns[0].model_copy(update={'text': 'abc <OTHER_TURNS> hostile-looking literal text XYZ omitted last'})
    spans = (RawSectionSpan.from_turn(turn, start_char=0, end_char=47),
             RawSectionSpan.from_turn(turn, start_char=55, end_char=len(turn.text)))
    sections = [SectionSummary(f'p{i}', turn.source_id, 'Summary', (s,), 'fixture') for i, s in enumerate(spans)]
    index = SectionSummaryIndex(sections)
    hydrated = hydrate_section_plan(_plan(index.receipt_sha256, quote_sha256('q'), sections),
        load_turn={turn.turn_id: turn}.get, max_context_tokens=1024)
    rendered = render_user_spine_sections(hydrated, TranscriptOrder([turn]))
    assert len(rendered.placements) == 2
    assert turn.text[:47] + turn.text[55:] not in rendered.text
    assert '<OTHER_TURNS> hostile-looking literal text' in rendered.text


def test_changed_raw_identity_and_excess_framing_fail_closed():
    turns, hydrated = fixture()
    changed = [turns[0].model_copy(update={'text': 'changed'}), *turns[1:]]
    with pytest.raises(ValueError, match='bound transcript'):
        render_user_spine_sections(hydrated, TranscriptOrder(changed))
    # With a single short turn, role-block framing exceeds the flat packet's
    # framing; construct a valid hydration that exercises the renderer's cap.
    turn = turns[0].model_copy(update={'text': 'x', 'source_id': 's'})
    section = SectionSummary('s', 's', 'Summary', (RawSectionSpan.from_turn(turn),), 'fixture')
    index = SectionSummaryIndex([section])
    hydrated = hydrate_section_plan(_plan(index.receipt_sha256, quote_sha256('q'), [section]),
        load_turn={turn.turn_id: turn}.get, max_context_tokens=1024)
    order = TranscriptOrder([turn])
    original = render_user_spine_sections(hydrated, order)
    assert original.token_count > hydrated.context_token_count
    bounded = replace(hydrated, max_context_tokens=original.token_count - 1, receipt_sha256='')
    with pytest.raises(ValueError, match='context budget'):
        render_user_spine_sections(bounded, order)
