from memory_condense.application.user_spine_section_context_v2 import (
    TranscriptOrder, render_user_spine_sections,
)
from tests.test_threaded_section_context import fixture


def test_other_role_block_restores_timestamp_after_later_user_turn():
    turns, hydrated = fixture()
    rendered = render_user_spine_sections(hydrated, TranscriptOrder(turns))
    first_user = rendered.text.index('<T1 user>')
    later_user_time = rendered.text.index('<AT 2026-09-10T12:00:00+00:00>')
    other_block = rendered.text.index('<OTHER_TURNS>')
    restored_time = rendered.text.index('<AT 2026-09-09T12:00:00+00:00>')
    assistant_turn = rendered.text.index('<T2 assistant>')
    assert first_user < later_user_time < other_block < restored_time < assistant_turn
    expected = {r.span.receipt_sha256: r.text for s in hydrated.sections for r in s.evidence}
    assert all(rendered.text[a:b] == expected[sha] for sha, a, b in rendered.placements)
    assert len(rendered.placements) == len(expected)
