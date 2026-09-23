from datetime import datetime, timezone
import json

import numpy as np
import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_summary_facets import SpineFacetAddressIndex, summary_facets
from memory_condense.search.user_spine_addresses import user_spine_text


def section(text, name="leaf"):
    turn = Turn(turn_id=name, source_id="source", role="user", text="EXACT_RAW_CANARY",
                created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
    summary = json.dumps({"user_spine": text, "attached_context_not_user_assertions": "RAW_NOT_AN_INPUT",
                          "transcript_date_range": [turn.created_at.isoformat()] * 2})
    return turn, SectionSummary(name, "source", summary, (RawSectionSpan.from_turn(turn),), "fixture")


def test_fact_and_followup_facets_are_exact_summary_passages():
    text = "User paid 19.95 euros and asks about the receipt. They plan a refund."
    _, leaf = section(text)
    facets = summary_facets(leaf)
    assert [f.text for f in facets] == ["User paid 19.95 euros", "asks about the receipt.", "They plan a refund."]
    assert all(f.text == text[f.start_char:f.end_char] for f in facets)
    assert "RAW" not in " ".join(f.text for f in facets)


def test_overlapping_windows_bound_long_summary_passages_without_truncating_tail():
    text = " ".join(f"word{i}" for i in range(110))
    _, leaf = section(text)
    facets = summary_facets(leaf)
    assert all(len(f.text.split()) <= 48 for f in facets)
    assert facets[-1].end_char == len(text)
    assert all(a.end_char >= b.start_char for a, b in zip(facets, facets[1:]))
    for maximum, stride in [(0, 1), (5, 6), (True, 1)]:
        with pytest.raises(ValueError):
            summary_facets(leaf, max_words=maximum, stride_words=stride)


def test_facet_hits_deduplicate_to_unchanged_leaves_and_hydrate_raw_exactly():
    turn, leaf = section("User bought a pass. User asks about a trip.")
    other_turn, other = section("User discusses a schedule.", "other")
    hierarchy = SectionSummaryIndex((leaf, other))
    facets = [f for s in hierarchy.sections for f in summary_facets(s)]
    matrix = np.array([[1, 0] if f.section_id == "leaf" else [0, 1] for f in facets], dtype=np.float32)
    index = SpineFacetAddressIndex(hierarchy, matrix, embedding_identity="fixture")
    matrix[:] = 0
    plan, audit = index.route_vector("purchased pass", np.array([1, 0]), embedding_identity="fixture", max_sections=2)
    assert [r.section for r in plan.routes] == [leaf, other]
    assert audit[0]["winning_facet"]["section_id"] == leaf.section_id
    result = hydrate_section_plan(plan, load_turn={turn.turn_id: turn, other_turn.turn_id: other_turn}.get,
                                 max_context_tokens=3072, max_raw_spans=128)
    assert all(e.text == "EXACT_RAW_CANARY" for s in result.sections for e in s.evidence)
    assert "User bought" not in result.render_context()
    with pytest.raises(ValueError, match="identities"):
        index.route_vector("pass", np.array([1, 0]), embedding_identity="foreign")
    with pytest.raises(AttributeError, match="immutable"):
        index.embedding_identity = "foreign"
