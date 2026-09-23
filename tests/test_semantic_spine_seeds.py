from datetime import datetime, timezone
import json

import numpy as np
import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.semantic_spine_seeds import SemanticSpineSeedRouter
from memory_condense.search.spine_union_routing import SpineUnionRouter
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from memory_condense.search.user_spine_addresses import UserSpineAddressIndex


def fixture():
    turns, sections = {}, []
    for i in range(10):
        turn = Turn(turn_id=f"turn-{i}", source_id=f"opaque-source-{i}", role="user",
            text=f"Exact original user statement {i}.",
            created_at=datetime(2026, 9, 4 if i >= 6 else 8, tzinfo=timezone.utc))
        turns[turn.turn_id] = turn
        summary = json.dumps({"user_spine": "accessories" if i >= 8 else "compatible equipment",
            "attached_context_not_user_assertions": None,
            "transcript_date_range": [turn.created_at.isoformat()] * 2})
        sections.append(SectionSummary(f"section-{i}", turn.source_id, summary,
            (RawSectionSpan.from_turn(turn),), "fixture"))
    hierarchy = SectionSummaryIndex(sections)
    combined = np.array([[10-i, i+1] for i in range(10)], dtype=np.float32)
    combined /= np.linalg.norm(combined, axis=1, keepdims=True)
    semantic = SemanticSectionIndex(hierarchy, combined, embedding_identity="fixture")
    users = UserSpineAddressIndex(hierarchy, combined, combined[::-1].copy(), embedding_identity="fixture")
    return turns, SpineUnionRouter(semantic, users), SemanticSpineSeedRouter(semantic, users)


def test_lexical_matches_no_longer_precede_semantic_seeds_and_hydration_stays_exact():
    turns, control, candidate = fixture()
    vector = np.array([1, 0])
    prior, old = control.route_vectors("accessories", "accessories", vector, embedding_identity="fixture")
    plan, audit = candidate.route_vectors("accessories", "accessories", vector, embedding_identity="fixture")
    assert old["baseline_section_ids"][:2] == ["section-8", "section-9"]
    assert audit["baseline_section_ids"] == [f"section-{i}" for i in range(6)]
    assert audit["user_supplement_section_ids"] == old["user_supplement_section_ids"]
    assert [r.section.section_id for r in plan.routes[:6]] == audit["baseline_section_ids"]
    assert len(plan.routes) <= 14 and len({r.section.section_id for r in plan.routes}) == len(plan.routes)
    hydrated = hydrate_section_plan(plan, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    assert all(e.text == turns[e.span.turn_id].text for section in hydrated.sections for e in section.evidence)
    assert "compatible equipment" not in hydrated.render_context()
    assert control.route_vectors("accessories", "accessories", vector, embedding_identity="fixture")[0] == prior


def test_calendar_priority_is_preserved_without_a_hard_time_filter():
    _, control, candidate = fixture()
    query = "What accessories did I buy last Friday?"
    dated = "[Question asked at 2026/09/09 (Wed) 19:25]\n" + query
    _, old = control.route_vectors(query, dated, np.array([1, 0]), embedding_identity="fixture")
    plan, audit = candidate.route_vectors(query, dated, np.array([1, 0]), embedding_identity="fixture")
    assert audit["preferred_section_ids"] == old["preferred_section_ids"]
    assert [r.section.section_id for r in plan.routes[:4]] == audit["preferred_section_ids"]
    assert any(r.section.spans[0].created_at.startswith("2026-09-08") for r in plan.routes)
    assert plan.eligible_source_ids is None and audit["hard_time_filter"] is False


@pytest.mark.parametrize("query,dated,identity,message", [
    ("accessories", "unrelated", "fixture", "bind"),
    ("accessories", "accessories", "foreign", "identity"),
    ("What is the order of the three workshops I attended, from earliest to latest?",
     "What is the order of the three workshops I attended, from earliest to latest?", "fixture", "own live query vector"),
])
def test_query_date_encoder_and_ordering_bindings_are_still_enforced(query, dated, identity, message):
    _, _, candidate = fixture()
    with pytest.raises(ValueError, match=message):
        candidate.route_vectors(query, dated, np.array([1, 0]), embedding_identity=identity)
