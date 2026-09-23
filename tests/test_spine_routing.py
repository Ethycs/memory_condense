import json

import pytest

from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_routing import route_user_spine_hierarchy
from tests.test_summary_reasoning import Reasoner, section, tree


def test_broad_parent_is_explored_before_final_one_section_choice():
    _, garden = section(0, "garden")
    _, target = section(1, "target telescope reservation")
    _, neighbor = section(2, "observatory stargazing hours")
    parent = SectionSummary("broad", "source", "garden and telescope topics", garden.spans + target.spans,
                            "test", child_section_ids=(garden.section_id, target.section_id))
    index = SectionSummaryIndex([garden, target, neighbor, parent])
    reasoner = Reasoner()
    result = route_user_spine_hierarchy("target", index, reasoner=reasoner, max_sections=1, beam_sections=3, group_size=4)
    assert len(reasoner.requests) == 1
    assert set(reasoner.requests[0].summaries) == {garden.summary, target.summary, neighbor.summary}
    assert result.plan.routes[0].section == target
    assert result.pruned_frontier_count == 0
    assert result.plan.reasoning_receipt.raw_content_inspections == 0
    assert "RAW_SECRET" not in json.dumps(reasoner.requests[0].messages)


def test_over_budget_beam_prunes_in_bounded_summary_groups():
    leaves = [section(i, "target" if i == 19 else "other")[1] for i in range(20)]
    parents = [SectionSummary(f"parent-{i}", "source", leaf.summary, leaf.spans, "test",
                              child_section_ids=(leaf.section_id,)) for i, leaf in enumerate(leaves)]
    reasoner = Reasoner()
    result = route_user_spine_hierarchy("target", SectionSummaryIndex([*leaves, *parents]), reasoner=reasoner,
                                        max_sections=1, beam_sections=2, group_size=4)
    assert result.plan.routes[0].section == leaves[-1]
    assert result.pruned_frontier_count == 1
    assert all(len(r.summaries) <= 4 for r in reasoner.requests)
    assert len(result.plan.routes) == 1


def test_depth_exhaustion_cannot_hydrate_a_broad_parent():
    _, index = tree()
    reasoner = Reasoner()
    with pytest.raises(ValueError, match="depth budget"):
        route_user_spine_hierarchy("target", index, reasoner=reasoner, max_depth=1)
    assert reasoner.requests == []


def test_scope_is_applied_before_any_model_work():
    _, index = tree()
    reasoner = Reasoner()
    result = route_user_spine_hierarchy("target", index, reasoner=reasoner, eligible_source_ids=["foreign"])
    assert not result.plan.routes and not reasoner.requests and not result.plan.frontier_closed


@pytest.mark.parametrize("response", ['{"selected_labels":[0,0]}', '{"selected_labels":[999]}', 'bad JSON'])
def test_invalid_final_selection_fails_before_raw_hydration(response):
    _, index = tree()
    with pytest.raises(ValueError, match="Qwen summary choice"):
        route_user_spine_hierarchy("target", index, reasoner=Reasoner(response))


def test_call_limit_and_group_limit_fail_closed():
    leaves = [section(i, "target")[1] for i in range(8)]
    with pytest.raises(ValueError, match="call budget"):
        route_user_spine_hierarchy("target", SectionSummaryIndex(leaves), reasoner=Reasoner(),
                                  max_sections=1, beam_sections=2, group_size=4, max_calls=1)
    with pytest.raises(ValueError, match="group_size"):
        route_user_spine_hierarchy("target", SectionSummaryIndex(leaves), reasoner=Reasoner(),
                                  max_sections=1, beam_sections=4, group_size=4)
