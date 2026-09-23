from dataclasses import replace

import pytest

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan, SectionSummaryIndex
from tests.test_spine_summary_facets import section
from tools.spine_facet_memory import supplemental_plan


def test_summary_lanes_interleave_and_deduplicate_without_changing_descriptors():
    leaves = [section("User states a fact.", name)[1] for name in ("a", "b", "c", "d")]
    index = SectionSummaryIndex(leaves)
    def plan(order):
        return SectionRoutePlan(index.receipt_sha256, quote_sha256("query"),
            tuple(SectionRoute(leaves[i], 1 / (j + 1), ()) for j, i in enumerate(order)),
            len(leaves), None, 8, routing_backend="summary_dense")
    merged = supplemental_plan(plan([0, 1, 2]), plan([1, 3, 0]))
    assert [r.section for r in merged.routes] == [leaves[i] for i in (0, 1, 3, 2)]
    assert merged.query_sha256 == quote_sha256("query")
    assert not merged.frontier_closed


def test_supplement_rejects_changed_query_scope_and_same_id_foreign_descriptor():
    _, leaf = section("User states a fact.")
    index = SectionSummaryIndex((leaf,))
    plan = SectionRoutePlan(index.receipt_sha256, quote_sha256("query"),
        (SectionRoute(leaf, 1, ()),), 1, None, 8, routing_backend="summary_dense")
    changed_leaf = replace(leaf, summary=leaf.summary.replace("fact", "plan"), receipt_sha256="")
    variants = [replace(plan, query_sha256=quote_sha256("different"), receipt_sha256=""),
        replace(plan, eligible_source_ids=(leaf.source_id,), receipt_sha256=""),
        replace(plan, routes=(SectionRoute(changed_leaf, 1, ()),), receipt_sha256="")]
    for other in variants:
        with pytest.raises(ValueError):
            supplemental_plan(plan, other)
