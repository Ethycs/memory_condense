import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.source_spine_hydration import SourceSpineHydrationIndex
from memory_condense.search.source_spine_supplement import SourceSpineSupplement
from tests.test_source_spine_hydration import fixture


def test_supplement_preserves_existing_users_before_adding_other_source():
    turns, atoms, hierarchy = fixture()
    selected = hierarchy.route("Follow-up", max_sections=1, eligible_source_ids=("alpha",))
    supplement = hierarchy.route("Follow-up", max_sections=2)
    expansion = SourceSpineSupplement(SourceSpineHydrationIndex(hierarchy, atoms))
    before, _ = expansion.overflow.expand(selected)
    plan, audit = expansion.expand(selected, supplement)
    users = [r.section for r in before.routes if r.section.spans[0].role == "user"]
    assert [r.section for r in plan.routes[:len(users)]] == users
    assert audit["supplemental_user_section_ids"]
    assert audit["raw_reads_during_expansion"] == 0
    old = hydrate_section_plan(before, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    new = hydrate_section_plan(plan, load_turn=turns.get, max_context_tokens=3072, max_raw_spans=128)
    old_users = {e.span.turn_id for s in old.sections for e in s.evidence if e.span.role == "user"}
    new_users = {e.span.turn_id for s in new.sections for e in s.evidence if e.span.role == "user"}
    assert old_users < new_users
    assert all(e.text == turns[e.span.turn_id].text for s in new.sections for e in s.evidence)


def test_foreign_query_or_leaf_population_cannot_supplement_evidence():
    _, atoms, hierarchy = fixture()
    expansion = SourceSpineSupplement(SourceSpineHydrationIndex(hierarchy, atoms))
    selected = hierarchy.route("Follow-up")
    foreign = SectionSummaryIndex(hierarchy.sections[:-1]).route("Follow-up")
    for supplement in (foreign, hierarchy.route("completed event")):
        with pytest.raises(ValueError, match="same query"):
            expansion.expand(selected, supplement)
