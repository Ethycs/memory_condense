from datetime import datetime, timezone

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.application.user_evidence_projection import TranscriptOrder, render_user_spine_sections
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.as_of_spine_routing import _plan
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from tests.test_threaded_section_context import fixture


def test_projection_retains_every_exact_user_span_without_changing_input():
    turns, hydration = fixture()
    before = hydration.identity_payload()
    result = render_user_spine_sections(hydration, TranscriptOrder(turns))
    expected = {r.span.receipt_sha256: r.text for s in hydration.sections for r in s.evidence
                if r.span.role == 'user'}
    payload = result.identity_payload()
    assert {p['span_sha256'] for p in payload['placements']} == set(expected)
    assert all(result.text[p['start_char']:p['end_char']] == expected[p['span_sha256']]
               for p in payload['placements'])
    assert turns[0].text in result.text  # Adjacent original raw slices rejoin exactly.
    assert all(t.text not in result.text for t in turns if t.role == 'assistant')
    assert set(payload['omitted_section_ids']) == {'s1', 's3'}
    assert payload['source_hydration_sha256'] == hydration.receipt_sha256
    assert payload['hydration_sha256'] != hydration.receipt_sha256
    assert hydration.identity_payload() == before


def packet(roles, groups):
    turns = [Turn(turn_id=f't{i}', source_id='s', role=role,
                  text=f'Exact {role} text {i}.', created_at=datetime(2026, 9, 10, tzinfo=timezone.utc))
             for i, role in enumerate(roles)]
    sections = [SectionSummary(f's{i}', 's', 'Routing summary',
                tuple(RawSectionSpan.from_turn(turns[j]) for j in group), 'fixture')
                for i, group in enumerate(groups)]
    index = SectionSummaryIndex(sections)
    plan = _plan(index.receipt_sha256, quote_sha256('query'), sections)
    return turns, hydrate_section_plan(plan, load_turn={t.turn_id: t for t in turns}.get,
                                      max_context_tokens=3072, max_raw_spans=128)


def test_mixed_role_section_is_never_split_and_system_section_is_retained():
    turns, hydration = packet(['user', 'assistant', 'assistant', 'system'], [(0, 1), (2,), (3,)])
    result = render_user_spine_sections(hydration, TranscriptOrder(turns))
    assert result.omitted_section_ids == ('s1',)
    assert all(turns[i].text in result.text for i in (0, 1, 3))
    assert turns[2].text not in result.text


def test_assistant_only_packet_is_preserved_when_no_user_evidence_exists():
    turns, hydration = packet(['assistant', 'assistant'], [(0,), (1,)])
    result = render_user_spine_sections(hydration, TranscriptOrder(turns))
    assert result.omitted_section_ids == ()
    assert all(t.text in result.text for t in turns)
    assert result.identity_payload()['hydration_sha256'] == hydration.receipt_sha256


@pytest.mark.parametrize('field,value', [('text', 'Altered'), ('role', 'user'), ('source_id', 'foreign')])
def test_excluded_assistant_evidence_still_requires_original_transcript_identity(field, value):
    turns, hydration = fixture()
    turns[1] = turns[1].model_copy(update={field: value})
    with pytest.raises(ValueError, match='bound transcript'):
        render_user_spine_sections(hydration, TranscriptOrder(turns))
