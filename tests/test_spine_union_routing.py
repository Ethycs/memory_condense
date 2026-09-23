from datetime import datetime, timezone
import json

import numpy as np
import pytest

from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from memory_condense.search.user_spine_addresses import UserSpineAddressIndex
from memory_condense.search.spine_union_routing import SpineUnionRouter


def fixture():
    sections = []
    for i in range(10):
        source = f"source-{i // 2}"
        turn = Turn(turn_id=f"turn-{i}", source_id=source, role="user", text=f"RAW_ONLY_{i}",
            created_at=datetime(2026, 9, 4 if i >= 6 else 8, tzinfo=timezone.utc))
        summary = json.dumps({"user_spine": "recorded event", "attached_context_not_user_assertions": "context",
                              "transcript_date_range": [turn.created_at.isoformat()] * 2})
        sections.append(SectionSummary(f"section-{i}", source, summary, (RawSectionSpan.from_turn(turn),), "fixture"))
    hierarchy = SectionSummaryIndex(sections)
    combined = np.array([[10-i, i+1] for i in range(10)], dtype=np.float32)
    combined /= np.linalg.norm(combined, axis=1, keepdims=True)
    user = combined[::-1].copy()
    semantic = SemanticSectionIndex(hierarchy, combined, embedding_identity="fixture")
    addresses = UserSpineAddressIndex(hierarchy, combined, user, embedding_identity="fixture")
    return SpineUnionRouter(semantic, addresses)


def test_user_channel_adds_missing_evidence_without_replacing_baseline_descriptors():
    router = fixture()
    query = "Which purchase?"
    plan, audit = router.route_vectors(query, query, np.array([1, 0]), embedding_identity="fixture")
    assert len(plan.routes) == 10
    assert [r.section.section_id for r in plan.routes[:6]] == audit["baseline_section_ids"]
    originals = {s.section_id: s for s in router.semantic.sections}
    assert all(r.section is originals[r.section.section_id] for r in plan.routes)
    assert set(audit["user_supplement_section_ids"]) <= {r.section.section_id for r in plan.routes}
    assert plan.eligible_source_ids is None and not plan.frontier_closed


def test_calendar_prior_diversifies_sources_and_retains_outside_window_evidence():
    router = fixture()
    query = "What happened last Friday?"
    dated = "[Question asked at 2026/09/09 (Wed) 19:25]\n" + query
    plan, audit = router.route_vectors(query, dated, np.array([1, 0]), embedding_identity="fixture")
    assert [r.section.source_id for r in plan.routes[:2]] == ["source-4", "source-3"]
    assert len(audit["preferred_section_ids"]) == 2
    assert set(audit["baseline_section_ids"]) <= {r.section.section_id for r in plan.routes}
    assert audit["hard_time_filter"] is False


def test_ordering_query_cannot_reuse_original_vector_for_content_view():
    router = fixture()
    query = "What is the order of the three workshops I attended, from earliest to latest?"
    with pytest.raises(ValueError, match="own live query vector"):
        router.route_vectors(query, query, np.array([1, 0]), embedding_identity="fixture")
    with pytest.raises(ValueError, match="bind"):
        router.route_vectors("unrelated", query, np.array([1, 0]), embedding_identity="fixture")
