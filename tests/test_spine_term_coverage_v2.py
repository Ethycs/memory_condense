from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.spine_term_coverage_v2 import ScopedSpineTermCoverage
from tests.test_spine_term_coverage import fixture, prior_plan


def test_a_rare_term_cannot_introduce_an_unselected_conversation():
    _, projection, hierarchy = fixture()
    query = "Orion"
    prior = prior_plan(hierarchy, projection, query)
    result, audit = ScopedSpineTermCoverage(hierarchy, projection).expand(query, prior)
    assert result is prior
    assert audit["source_scope"] == ["alpha"]
    assert not audit["selected_exchange_ids"] and audit["new_sources_admitted"] is False


def test_selected_source_retains_its_missing_named_exchange_without_mutating_index():
    turns, projection, hierarchy = fixture(other_source="beta")
    query = "Orion"
    selected = SectionRoutePlan(hierarchy.receipt_sha256, quote_sha256(query),
        (SectionRoute(hierarchy.sections[0], 1.0, ()), SectionRoute(hierarchy.sections[3], 0.5, ())), 2, None, 2)
    prior = projection.expand(selected)[0]
    coverage = ScopedSpineTermCoverage(hierarchy, projection)
    before = (coverage.leaves, dict(coverage.frequencies))
    plan, audit = coverage.expand(query, prior)
    result = hydrate_section_plan(plan, load_turn=turns.get, max_context_tokens=512, max_raw_spans=128)
    assert "Sunday | Orion | 08:00-16:00" in result.render_context()
    assert all(r.section.source_id in audit["source_scope"] for r in plan.routes)
    assert (coverage.leaves, coverage.frequencies) == before
    assert coverage.expand(query, prior)[0] == plan
    assert audit["raw_reads"] == 0 and audit["rarity_population"] == "complete memory"
