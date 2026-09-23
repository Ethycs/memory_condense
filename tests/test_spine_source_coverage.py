from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan, SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.source_spine_hydration import SourceSpineHydrationIndex
from memory_condense.search.spine_source_coverage import SpineSourceCoverage


def fixture(count=12):
    turns, atoms, leaves = {}, [], []
    for i in range(count):
        source = "frequent" if i < 8 else f"other-{i}"
        turn = Turn(turn_id=f"turn-{i}", source_id=source, role="user", text=f"Exact evidence {i}.",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        turns[turn.turn_id] = turn
        span = RawSectionSpan.from_turn(turn)
        atoms.append(SectionSummary(f"atom-{i}", source, f"Summary {i}.", (span,), "fixture"))
        summary = json.dumps({"user_spine": f"Summary {i}.", "attached_context_not_user_assertions": None,
            "transcript_date_range": [turn.created_at.isoformat()] * 2})
        leaves.append(SectionSummary(f"leaf-{i:02d}", source, summary, (span,), "fixture"))
    hierarchy = SectionSummaryIndex(leaves)
    source_spine = SourceSpineHydrationIndex(hierarchy, atoms)
    def plan(sections, index=hierarchy):
        return SectionRoutePlan(index.receipt_sha256, quote_sha256("query"),
            tuple(SectionRoute(s, 1 / (i + 1), ()) for i, s in enumerate(sections)),
            len(index.sections), None, max(1, len(sections)), routing_backend="summary_dense")
    prior, _ = source_spine.expand(plan(leaves[:1]))
    return turns, leaves, source_spine, plan, prior


def test_repeated_conversation_does_not_hide_later_distinct_source_and_raw_is_exact():
    turns, leaves, base, plan, prior = fixture()
    expanded, audit = SpineSourceCoverage(base).expand(prior, plan(leaves), plan(leaves))
    assert [s["rank"] for s in audit["user_channel"]] == [1, 9, 10, 11, 12]
    before = hydrate_section_plan(prior, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    after = hydrate_section_plan(expanded, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    assert after.sections[:len(before.sections)] == before.sections
    assert "Exact evidence 8." in after.render_context()
    assert "Summary" not in after.render_context()
    assert all(e.text == turns[e.span.turn_id].text for s in after.sections for e in s.evidence)
    assert len({r.section.section_id for r in expanded.routes}) == len(expanded.routes)
    assert audit["raw_reads_during_expansion"] == 0


def test_foreign_query_scope_or_descriptor_cannot_supply_a_source():
    _, leaves, base, plan, prior = fixture()
    coverage = SpineSourceCoverage(base)
    valid = plan(leaves)
    altered = replace(leaves[-1], summary="Altered summary", receipt_sha256="")
    bad = [replace(valid, query_sha256=quote_sha256("foreign"), receipt_sha256=""),
        replace(plan(leaves[:8]), eligible_source_ids=("frequent",), receipt_sha256=""), plan([*leaves[:-1], altered])]
    for frontier in bad:
        with pytest.raises(ValueError, match="frontier"):
            coverage.expand(prior, valid, frontier)
    with pytest.raises(ValueError, match="prior"):
        coverage.expand(replace(prior, index_sha256=quote_sha256("foreign"), receipt_sha256=""), valid, valid)


def test_empty_frontiers_preserve_the_prior():
    _, _, base, plan, prior = fixture()
    expanded, audit = SpineSourceCoverage(base).expand(prior, plan([]), plan([]))
    assert [r.section for r in expanded.routes] == [r.section for r in prior.routes]
    assert audit["added_user_section_ids"] == []


def test_source_coverage_is_bounded_even_with_many_distinct_sources():
    _, leaves, base, plan, prior = fixture(40)
    coverage = SpineSourceCoverage(base)
    _, audit = coverage.expand(prior, plan(leaves[:32]), plan(leaves[:32]))
    assert len(audit["user_channel"]) == len(audit["facet_channel"]) == 8
    assert len(audit["added_source_ids"]) == 7
    with pytest.raises(ValueError, match="frontier"):
        coverage.expand(prior, plan(leaves), plan(leaves[:32]))
