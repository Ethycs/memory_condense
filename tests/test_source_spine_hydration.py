from datetime import datetime, timezone
import json

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.source_spine_hydration import SourceSpineHydrationIndex


def fixture():
    turns, atoms, leaves = {}, [], []
    for source in ("alpha", "beta"):
        for i, text in enumerate(("I completed the event today.", "Follow-up training question.")):
            group = []
            for role, raw in (("user", text), ("assistant", "Long training advice. " * 20)):
                key = f"{source}-{i}-{role}"
                turn = Turn(turn_id=key, source_id=source, role=role, text=raw,
                            created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
                turns[key] = turn
                atom = SectionSummary(key, source, text if role == "user" else "Training suggestions.",
                                      (RawSectionSpan.from_turn(turn),), "fixture")
                atoms.append(atom)
                group.append(atom)
            summary = json.dumps({"user_spine": text, "attached_context_not_user_assertions": "Training suggestions.",
                                  "transcript_date_range": [turn.created_at.isoformat()] * 2})
            leaves.append(SectionSummary(f"{source}-{i}", source, summary, tuple(a.spans[0] for a in group), "fixture"))
    return turns, atoms, SectionSummaryIndex(leaves)


def test_later_topic_match_expands_to_exact_earlier_user_evidence_and_keeps_attached_context():
    turns, atoms, hierarchy = fixture()
    route = hierarchy.route("Follow-up", max_sections=2)
    assert all(r.section.section_id.endswith("-1") for r in route.routes)
    projection = SourceSpineHydrationIndex(hierarchy, atoms)
    plan, audit = projection.expand(route)
    result = hydrate_section_plan(plan, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    assert not result.diagnostics
    assert [r.section.source_id for r in plan.routes[:4]] == ["alpha", "beta", "alpha", "beta"]
    assert all(r.section.spans[0].turn_id.endswith("-1-user") for r in plan.routes[:2])
    assert "I completed the event today." in result.render_context()
    assert "Long training advice." in result.render_context()
    assert "Training suggestions." not in result.render_context()
    assert all(e.text == turns[e.span.turn_id].text for s in result.sections for e in s.evidence)
    assert audit["raw_reads_during_expansion"] == 0


def test_source_expansion_rejects_incomplete_or_foreign_raw_coordinates():
    _, atoms, hierarchy = fixture()
    with pytest.raises(ValueError, match="partition"):
        SourceSpineHydrationIndex(hierarchy, atoms[:-1])
    projection = SourceSpineHydrationIndex(hierarchy, atoms)
    foreign = SectionSummaryIndex(hierarchy.sections[:-1]).route("Follow-up")
    with pytest.raises(ValueError, match="another memory"):
        projection.expand(foreign)
