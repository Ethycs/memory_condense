from datetime import datetime, timezone
import json

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.source_spine_hydration import SourceSpineHydrationIndex
from memory_condense.search.source_spine_overflow import SourceSpineOverflow


def fixture(*, oversized=False):
    turns, atoms, leaves = {}, [], []
    for i in range(8):
        source = str(i)
        group = []
        for role in ("user", "assistant"):
            key = source + role
            text = f"I attended event {i}." if role == "user" else "Detailed unrelated suggestions. " * 200
            if oversized and i == 7 and role == "user":
                text = "User detail " * 3000
            turn = Turn(turn_id=key, source_id=source, role=role, text=text,
                created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
            turns[key] = turn
            atom = SectionSummary(key, source, "event summary", (RawSectionSpan.from_turn(turn),), "fixture")
            atoms.append(atom)
            group.append(atom)
        summary = json.dumps({"user_spine": "event summary", "attached_context_not_user_assertions": "suggestions"})
        leaves.append(SectionSummary(source, source, summary, tuple(a.spans[0] for a in group), "fixture"))
    hierarchy = SectionSummaryIndex(leaves)
    base = SourceSpineHydrationIndex(hierarchy, atoms)
    return turns, hierarchy, base


def user_evidence(result):
    return {(e.span.turn_id, e.span.receipt_sha256, e.text) for s in result.sections for e in s.evidence if e.span.role == "user"}


def test_recovers_routed_users_outside_source_cap_and_preserves_prior_user_bytes():
    turns, hierarchy, base = fixture()
    selected = hierarchy.route("event", max_sections=8)
    prior, _ = base.expand(selected)
    candidate, audit = SourceSpineOverflow(base).expand(selected)
    hydrate = lambda p: hydrate_section_plan(p, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    before, after = hydrate(prior), hydrate(candidate)
    assert len(user_evidence(before)) == 6
    assert len(user_evidence(after)) == 8
    assert user_evidence(before) < user_evidence(after)
    assert "event summary" not in after.render_context()
    assert after.context_token_count <= 3072
    assert len(audit["additional_routed_user_section_ids"]) == 2
    assert audit["raw_reads_during_expansion"] == 0
    assert not candidate.frontier_closed


def test_oversized_extra_user_is_rejected_without_displacing_protected_users():
    turns, hierarchy, base = fixture(oversized=True)
    selected = hierarchy.route("event", max_sections=8)
    prior, _ = base.expand(selected)
    candidate, _ = SourceSpineOverflow(base).expand(selected)
    hydrate = lambda p: hydrate_section_plan(p, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    before, after = hydrate(prior), hydrate(candidate)
    assert user_evidence(before) <= user_evidence(after)
    assert len(user_evidence(after)) == 7
    assert all(e.span.turn_id != "7user" for s in after.sections for e in s.evidence)
    assert after.diagnostics and after.context_token_count <= 3072


def test_foreign_route_cannot_supply_overflow_users():
    _, hierarchy, base = fixture()
    foreign = SectionSummaryIndex(hierarchy.sections[:-1]).route("event")
    with pytest.raises(ValueError, match="another memory"):
        SourceSpineOverflow(base).expand(foreign)
