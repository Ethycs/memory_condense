from datetime import date

import pytest

from memory_condense.search.summary_time_prior import mention_window


@pytest.mark.parametrize("asked,phrase,expected", [
    ("2026/09/09 (Wed) 19:25", "last Friday", (date(2026,9,4),date(2026,9,5))),
    ("2026/09/04 (Fri) 19:25", "last Friday", (date(2026,8,28),date(2026,8,29))),
    ("2024/03/31 (Sun) 19:25", "past month", (date(2024,2,29),date(2024,3,31))),
    ("2024/02/29 (Thu) 19:25", "last year", (date(2023,2,28),date(2024,2,29))),
    ("2026/01/03 (Sat) 19:25", "past week", (date(2025,12,27),date(2026,1,3))),
])
def test_explicit_calendar_windows(asked, phrase, expected):
    assert mention_window(f"[Question asked at {asked}]\nWhat happened {phrase}?") == expected


@pytest.mark.parametrize("query", ["What happened last Friday?", "What is my current camera?",
    "[Question asked at 2026/09/09 (Wed) 19:25]\nCompare last Friday and last Monday.",
    "[Question asked at 2026/09/09 (Wed) 19:25]\nHow many days before the party did I buy the gift?"])
def test_absent_or_compound_hints_do_not_create_a_time_scope(query):
    assert mention_window(query) is None


def test_mention_priority_retains_retrospective_evidence_outside_window():
    from datetime import datetime, timezone
    import numpy as np
    from memory_condense.domain.schemas import Turn
    from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
    from memory_condense.search.section_routing import SectionSummaryIndex
    from memory_condense.search.summary_semantic_index import SemanticSectionIndex
    from memory_condense.search.summary_time_prior import route_with_time_prior
    sections = []
    for name, day in (("inside", 4), ("outside", 6)):
        turn = Turn(turn_id=name, source_id=name, role="user", text="exact raw event",
                    created_at=datetime(2026,9,day,tzinfo=timezone.utc))
        sections.append(SectionSummary(name,name,"event",(RawSectionSpan.from_turn(turn),),"fixture"))
    semantic = SemanticSectionIndex(SectionSummaryIndex(sections),np.array([[0,1],[1,0]],dtype=np.float32),
                                    embedding_identity="fixture")
    query = "What happened last Friday?"
    dated = "[Question asked at 2026/09/09 (Wed) 19:25]\n" + query
    plan, hint = route_with_time_prior(semantic,query,dated,np.array([1,0]),embedding_identity="fixture",preferred_sections=1)
    assert [r.section.source_id for r in plan.routes] == ["inside","outside"]
    assert plan.eligible_source_ids is None and not plan.frontier_closed
    assert hint["event_time_certified"] is False
    with pytest.raises(ValueError,match="bind"):
        route_with_time_prior(semantic,"another query",dated,np.array([1,0]),embedding_identity="fixture")
    with pytest.raises(ValueError,match="fallback"):
        route_with_time_prior(semantic,query,dated,np.array([1,0]),embedding_identity="fixture",preferred_sections=6)
