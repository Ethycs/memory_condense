from datetime import date, datetime, timezone

import numpy as np
import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.bounded_spine_hierarchy import BoundedSpineHierarchyRouter, project_hierarchy_leaves
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from tests.test_summary_shortlist_attention import Linker


def fixture():
    turns, sections = [], []
    for i in range(4):
        turn = Turn(turn_id=f"private-turn-{i}", source_id=f"private-source-{i // 2}", role="user",
                    created_at=datetime(2026, 9, 8 + i % 2, tzinfo=timezone.utc),
                    text=f"RAW_CANARY exact café {i}\r\n")
        turns.append(turn)
        sections.append(SectionSummary(f"leaf-{i}", turn.source_id,
            "travel completed" if i == 1 else "travel considered", (RawSectionSpan.from_turn(turn),), "fixture"))
    for i in range(2):
        children = sections[2*i:2*i+2]
        sections.append(SectionSummary(f"root-{i}", children[0].source_id,
            "travel completed overview" if i == 0 else "travel plans overview",
            tuple(p for child in children for p in child.spans), "fixture",
            child_section_ids=tuple(s.section_id for s in children)))
    index = SectionSummaryIndex(sections)
    semantic = SemanticSectionIndex(index, np.array([[1., 0.], [.9, .4358899], [0., 1.], [0., 1.]], dtype=np.float32),
                                    embedding_identity="fixture-embedding")
    return turns, BoundedSpineHierarchyRouter(semantic)


def route(router, linker, **kwargs):
    return router.route_vector("travel completed", np.array([1., 0.], dtype=np.float32),
        embedding_identity="fixture-embedding", linker=linker, root_shortlist=2, beam=1, **kwargs)


def test_qwen_visits_parent_then_children_and_hydrates_exact_raw_leaf():
    turns, router = fixture()
    linker = Linker()
    plan = route(router, linker)
    assert len(linker.inputs) == 2
    assert "overview" in repr(linker.inputs[0]) and "overview" not in repr(linker.inputs[1])
    assert "RAW_CANARY" not in repr(linker.inputs) and "private-" not in repr(linker.inputs)
    assert plan.routes[0].section.section_id == "leaf-1"
    assert len(plan.attention_receipt.rounds) == 2
    assert all(r.model_passes == 1 for r in plan.attention_receipt.rounds)
    projected, audit = project_hierarchy_leaves(plan, router.index, date(2026, 9, 9))
    result = hydrate_section_plan(projected, load_turn={t.turn_id: t for t in turns}.get,
                                  max_context_tokens=3072, max_raw_spans=128)
    assert result.sections[0].evidence[0].text == turns[1].text
    assert audit["hierarchical_plan_sha256"] == plan.receipt_sha256


def test_date_and_source_scope_apply_before_attention_and_descent():
    _, router = fixture()
    linker = Linker()
    plan = route(router, linker, asked_day=date(2026, 9, 8), eligible_source_ids=["private-source-0"])
    assert plan.routes[0].section.section_id == "leaf-0"
    assert all("completed" not in text or "overview" in text for text in linker.inputs[1][1])
    assert len(linker.inputs[0][1]) == len(linker.inputs[1][1]) == 1


def test_empty_scope_has_no_qwen_calls():
    _, router = fixture()
    linker = Linker()
    assert not route(router, linker, eligible_source_ids=[]).routes
    assert not linker.inputs


@pytest.mark.parametrize("kwargs", [{"max_depth": 1}, {"asked_day": "2026-09-08"}])
def test_invalid_limits_fail_before_attention(kwargs):
    _, router = fixture()
    linker = Linker()
    with pytest.raises((ValueError, TypeError)):
        route(router, linker, **kwargs)
    assert not linker.inputs


def test_leaf_only_index_is_rejected_instead_of_mislabeled_as_hierarchical():
    _, router = fixture()
    leaves = router.semantic.sections
    semantic = SemanticSectionIndex(SectionSummaryIndex(leaves),
        np.array([[1., 0.]] * len(leaves), dtype=np.float32), embedding_identity="fixture-embedding")
    with pytest.raises(ValueError, match="populated parent"):
        BoundedSpineHierarchyRouter(semantic)


def test_partial_model_workspace_cannot_become_a_route():
    _, router = fixture()
    with pytest.raises(ValueError, match="entire bounded shortlist"):
        route(router, Linker("partial"))


def test_mixed_date_leaf_projection_keeps_exact_eligible_span_and_rejects_stale_binding():
    from memory_condense.domain._discourse_identity import quote_sha256
    from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
    turns, router = fixture()
    root = router.by_id["root-0"]
    mixed = SectionSummary("mixed-leaf", root.source_id, root.summary, root.spans, "fixture")
    index = SectionSummaryIndex([mixed])
    plan = SectionRoutePlan(index.receipt_sha256, quote_sha256("travel"),
        (SectionRoute(mixed, 1., ()),), 1, None, 1, routing_backend="summary_dense")
    projected, audit = project_hierarchy_leaves(plan, index, date(2026, 9, 8))
    assert projected.routes[0].section.spans == (mixed.spans[0],)
    hydrated = hydrate_section_plan(projected, load_turn={t.turn_id:t for t in turns}.get)
    assert hydrated.sections[0].evidence[0].text == turns[0].text
    assert len(audit["partial_projections"]) == 1
    with pytest.raises(ValueError, match="binding"):
        project_hierarchy_leaves(plan, router.index, date(2026, 9, 8))
