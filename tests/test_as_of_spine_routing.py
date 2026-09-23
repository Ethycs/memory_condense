from dataclasses import replace
from datetime import date, datetime, timezone
import json

import numpy as np
import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.as_of_spine_routing import AsOfSpineRouter, AsOfSpineExpansion
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.semantic_spine_seeds import SemanticSpineSeedRouter
from memory_condense.search.source_spine_hydration import SourceSpineHydrationIndex
from memory_condense.search.source_spine_supplement import SourceSpineSupplement
from memory_condense.search.spine_source_coverage import SpineSourceCoverage
from memory_condense.search.spine_summary_facets import SpineFacetAddressIndex, summary_facets
from memory_condense.search.spine_term_coverage_v2 import ScopedSpineTermCoverage
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from memory_condense.search.summary_time_prior_v2 import question_day, relative_mention_window
from tools.spine_facet_memory import supplemental_plan
from memory_condense.search.user_spine_addresses import UserSpineAddressIndex


def fixture():
    turns, atoms, leaves = {}, [], []
    for i in range(12):
        spans = []
        source = "mixed" if i in (0, 8) else f"source-{i:02d}"
        for j, role in enumerate(("user", "assistant", "assistant")):
            day = 20 if i < 8 or (i == 8 and j == 2) else 6 if i == 11 else 19 if i == 10 else 10
            turn = Turn(turn_id=f"turn-{i:02d}-{j}", source_id=source, role=role,
                text=f"Exact {role} observation {i}-{j}, with an event planned for December.",
                created_at=datetime(2026, 9, day, j, tzinfo=timezone.utc))
            turns[turn.turn_id] = turn
            span = RawSectionSpan.from_turn(turn)
            spans.append(span)
            atoms.append(SectionSummary(turn.turn_id, source, "A stored topic summary.", (span,), "fixture"))
        summary = json.dumps({"user_spine": "A stored topic summary.",
            "attached_context_not_user_assertions": "Context about the same topic.",
            "transcript_date_range": [spans[0].created_at, spans[-1].created_at]})
        leaves.append(SectionSummary(f"section-{i:02d}", source, summary, tuple(spans), "fixture"))
    hierarchy = SectionSummaryIndex(leaves)
    matrix = np.array([[12-i, i+1] for i in range(12)], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    semantic = SemanticSectionIndex(hierarchy, matrix, embedding_identity="fixture")
    users = UserSpineAddressIndex(hierarchy, matrix, matrix, embedding_identity="fixture")
    facet_matrix = np.array([matrix[i] for i, s in enumerate(hierarchy.sections) for _ in summary_facets(s)])
    facets = SpineFacetAddressIndex(hierarchy, facet_matrix, embedding_identity="fixture")
    source = SourceSpineHydrationIndex(hierarchy, atoms)
    supplement = SourceSpineSupplement(source)
    coverage = SpineSourceCoverage(source)
    terms = ScopedSpineTermCoverage(hierarchy, source)
    expansion = AsOfSpineExpansion(source, supplement, coverage, terms)
    return turns, AsOfSpineRouter(semantic, users, facets), expansion


def dated(query, day="2026/09/19"):
    return f"[Question asked at {day} (Sat) 00:00]\n{query}"


def execute(router, expansion, query="topic", day="2026/09/19", **options):
    prompt = dated(query, day)
    plans, audit = router.route_vectors(query, prompt, np.array([1, 0]), embedding_identity="fixture", **options)
    plan, expanded = expansion.expand(query, prompt, plans, supplemental_plan(plans["users"], plans["facets"]))
    return plans, audit, plan, expanded


def test_future_high_scores_are_removed_before_every_channel_top_k_and_same_day_is_kept():
    _, router, expansion = fixture()
    plans, audit, _, _ = execute(router, expansion)
    expected = {f"section-{i:02d}" for i in range(8, 12)}
    for name in ("selected", "users", "facets", "user_frontier", "facet_frontier"):
        assert {r.section.section_id for r in plans[name].routes} == expected
    assert audit["excluded_future_leaf_count"] == 8
    assert audit["mixed_date_eligible_leaf_count"] == 1


def test_mixed_leaf_and_mixed_source_retain_older_exact_raw_with_no_future_loads():
    turns, router, expansion = fixture()
    before = {source: tuple(rows) for source, rows in expansion.base.by_source.items()}
    _, _, plan, audit = execute(router, expansion)
    calls = []

    def load(key):
        calls.append(key)
        assert turns[key].created_at.date() <= date(2026, 9, 19)
        return turns[key]

    hydrated = hydrate_section_plan(plan, load_turn=load, max_context_tokens=3072, max_raw_spans=128)
    evidence = [e for s in hydrated.sections for e in s.evidence]
    assert "turn-08-0" in calls and "turn-08-1" in calls
    assert "turn-08-2" not in calls and not any(key.startswith("turn-00-") for key in calls)
    assert all(e.text == turns[e.span.turn_id].text for e in evidence)
    assert any("planned for December" in e.text for e in evidence)
    assert audit["projection"]["partial_projections"]
    assert {source: tuple(rows) for source, rows in expansion.base.by_source.items()} == before
    assert hydrated.context_token_count <= 3072


def test_no_future_dates_reproduces_semantic_seed_routes_and_complete_raw_packet():
    turns, router, expansion = fixture()
    query, prompt = "topic", dated("topic", "2026/09/25")
    old, _ = SemanticSpineSeedRouter(router.semantic, router.users).route_vectors(
        query, prompt, np.array([1, 0]), embedding_identity="fixture")
    plans, _, candidate, _ = execute(router, expansion, day="2026/09/25")
    assert [r.section for r in old.routes] == [r.section for r in plans["selected"].routes]
    prior, _ = expansion.supplement.expand(old, supplemental_plan(plans["users"], plans["facets"]))
    diverse, _ = expansion.coverage.expand(prior, plans["user_frontier"], plans["facet_frontier"])
    prior, _ = expansion.terms.expand(query, diverse)
    a = hydrate_section_plan(prior, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    b = hydrate_section_plan(candidate, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    assert a.render_context() == b.render_context()


def test_date_backfill_reservation_never_spends_budget_on_future_user_turns():
    _, router, expansion = fixture()
    _, _, _, audit = execute(router, expansion)
    mixed_old = next(s for s in expansion.base.by_source["mixed"] if s.spans[0].turn_id == "turn-08-0")
    future = next(s for s in expansion.base.by_source["mixed"] if s.spans[0].turn_id == "turn-00-0")
    selected = audit["supplement"]["protected_user_section_ids"]
    assert mixed_old.section_id in selected and future.section_id not in selected


def test_all_future_memory_returns_empty_evidence_without_reading_raw():
    _, router, expansion = fixture()
    _, audit, plan, _ = execute(router, expansion, day="2026/09/01")
    assert audit["eligible_leaf_count"] == 0 and not plan.routes
    def forbidden(_):
        raise AssertionError("empty eligible memory cannot read raw")
    result = hydrate_section_plan(plan, load_turn=forbidden, max_context_tokens=3072, max_raw_spans=128)
    assert not result.sections


def test_relative_hint_prioritizes_an_older_source_but_retains_global_fallback():
    _, router, expansion = fixture()
    query = "What topic did I discuss two weeks ago?"
    plans, audit, _, _ = execute(router, expansion, query, "2026/09/20", extended_relative_prior=True)
    assert audit["mention_window"] == ["2026-09-06", "2026-09-07"]
    assert plans["selected"].routes[0].section.section_id == "section-11"
    assert any(r.section.section_id == "section-00" for r in plans["selected"].routes)


@pytest.mark.parametrize("body,day,expected", [
    ("two weeks ago", "2026/09/20", (date(2026, 9, 6), date(2026, 9, 7))),
    ("14 days ago", "2026/09/20", (date(2026, 9, 6), date(2026, 9, 7))),
    ("one month ago", "2024/03/31", (date(2024, 2, 29), date(2024, 3, 1))),
    ("one year ago", "2024/02/29", (date(2023, 2, 28), date(2023, 3, 1))),
    ("last week", "2026/09/20", (date(2026, 9, 13), date(2026, 9, 20))),
    ("last Monday", "2026/09/20", (date(2026, 9, 14), date(2026, 9, 15))),
    ("two weeks ago and last month", "2026/09/20", None),
    ("two weeks ago or three weeks ago", "2026/09/20", None),
    ("0 days ago", "2026/09/20", None),
    ("9999999999999999 years ago", "2026/09/20", None),
])
def test_relative_mentions_calendar_boundaries_and_ambiguity(body, day, expected):
    assert relative_mention_window(dated(body, day)) == expected


def test_foreign_descriptors_cannot_be_resealed_as_date_projections():
    _, router, expansion = fixture()
    plans, _ = router.route_vectors("topic", dated("topic"), np.array([1, 0]), embedding_identity="fixture")
    prior, _ = expansion.base.expand(plans["selected"])
    forged = replace(prior.routes[0].section, summary="forged", receipt_sha256="")
    route = replace(prior.routes[0], section=forged, receipt_sha256="")
    forged_plan = replace(prior, routes=(route, *prior.routes[1:]), receipt_sha256="")
    with pytest.raises(ValueError, match="unchanged bound"):
        expansion.project(forged_plan, date(2026, 9, 19))


@pytest.mark.parametrize("query,prompt,identity,vector", [
    ("topic", "topic", "fixture", np.array([1, 0])),
    ("different", dated("topic"), "fixture", np.array([1, 0])),
    ("topic", dated("topic"), "foreign", np.array([1, 0])),
    ("topic", dated("topic"), "fixture", np.array([float("nan"), 0])),
])
def test_date_query_and_live_vector_bindings(query, prompt, identity, vector):
    _, router, _ = fixture()
    with pytest.raises(ValueError):
        router.route_vectors(query, prompt, vector, embedding_identity=identity)


def test_day_cutoff_uses_timestamp_day_and_ignores_unknown_question_timezone():
    assert question_day("topic", dated("topic")) == date(2026, 9, 19)
