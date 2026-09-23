from dataclasses import replace

import numpy as np
import pytest

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.native_spine_routing import NativeSpineRouter
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from tests.test_native_spine_routing import DATED, QUERY, fixture


def parent_fixture(*, future=False):
    history, semantic, _, encoder = fixture(future=future)
    sections = []
    for i in range(0, len(history.atoms), 3):
        atoms = history.atoms[i:i+3]
        spans = tuple(a.spans[0] for a in atoms)
        left = SectionSummary(f"exchange-{i}", atoms[0].source_id, "A user exchange.", spans[:2], "test")
        right = SectionSummary(f"exchange-{i+1}", atoms[0].source_id, "A related user fact.", spans[2:], "test")
        parent = SectionSummary(f"parent-{i}", atoms[0].source_id, "Stored attention parent summary.",
            spans, "test", (left.section_id, right.section_id))
        sections.extend((left, right, parent))
    return history, semantic, SectionSummaryIndex(sections), encoder


def test_parent_context_reaches_user_sibling_that_exchange_only_expansion_misses():
    h, s, hierarchy, encoder = parent_fixture()
    old = NativeSpineRouter(s, hierarchy).route_vector(QUERY, DATED, np.array([1, 0]),
        embedding_identity=s.embedding_identity, max_direct=1, lexical_reserve=0)
    assert h.atoms[2].section_id not in {r.section.section_id for r in old.expanded.routes}
    memory = ResidentNativeSpineContextMemory(s, hierarchy, encoder=encoder, load_turn=h.get_turn)
    result = memory.retrieve(QUERY, DATED, max_direct=1, protected_direct=1, lexical_reserve=0,
        context_seed_limit=1, max_raw_spans=2)
    assert result.routing.consulted_chunk_ids == ("parent-0",)
    assert [x.section for x in result.hydration.sections] == [h.atoms[0], h.atoms[2]]
    assert result.hydration.sections[1].evidence[0].text == h.get_turn(h.atoms[2].spans[0].turn_id).text
    assert result.hydration.raw_turn_read_count == 2
    assert result.routing.raw_reads_during_routing == result.routing.query_qwen_passes == 0
    assert encoder.calls == [QUERY]


def test_user_tail_is_promoted_even_when_already_in_direct_candidate_pool():
    h, s, hierarchy, encoder = parent_fixture()
    memory = ResidentNativeSpineContextMemory(s, hierarchy, encoder=encoder, load_turn=h.get_turn)
    result = memory.retrieve(QUERY, DATED, max_direct=3, protected_direct=1, lexical_reserve=0,
        context_seed_limit=1, max_raw_spans=2)
    assert result.routing.added_atomic_ids == ()
    assert h.atoms[2].section_id in result.routing.context_atomic_ids
    assert result.routing.expanded.routes[0] == result.routing.baseline.routes[0]
    assert {r.section for r in result.routing.expanded.routes} == {r.section for r in result.routing.baseline.routes}
    assert [x.section.spans[0].role for x in result.hydration.sections] == ["user", "user"]


def test_role_only_control_uses_no_hierarchy_context_and_keeps_exact_budget():
    h, s, hierarchy, encoder = parent_fixture()
    memory = ResidentNativeSpineContextMemory(s, hierarchy, encoder=encoder, load_turn=h.get_turn)
    result = memory.retrieve(QUERY, DATED, max_direct=3, protected_direct=1, lexical_reserve=0,
        max_additions=0, max_context_tokens=80)
    assert result.routing.consulted_chunk_ids == result.routing.context_atomic_ids == ()
    assert result.hydration.context_token_count <= 80
    assert all(x.section in h.atoms for x in result.hydration.sections)


def test_assistant_at_rank_one_cannot_starve_user_facts_with_zero_protected_prefix():
    h, s, hierarchy, encoder = parent_fixture()
    matrix = np.array([[1, 0] if a.spans[0].role == "assistant" else [0, 1]
                       for a in s.sections], dtype=np.float32)
    semantic = SemanticSectionIndex(s.hierarchy, matrix, embedding_identity=s.embedding_identity)
    memory = ResidentNativeSpineContextMemory(semantic, hierarchy, encoder=encoder, load_turn=h.get_turn)
    options = dict(max_direct=3, lexical_reserve=0, max_additions=0, max_raw_spans=2)
    old = memory.retrieve(QUERY, DATED, protected_direct=1, **options)
    fixed = memory.retrieve(QUERY, DATED, protected_direct=0, **options)
    assert old.hydration.sections[0].section.spans[0].role == "assistant"
    assert [s.section.spans[0].role for s in fixed.hydration.sections] == ["user", "user"]
    assert h.atoms[2] in [s.section for s in fixed.hydration.sections]
    assert {r.section for r in fixed.routing.expanded.routes} == {r.section for r in old.routing.expanded.routes}
    assert fixed.hydration.raw_turn_read_count == 2


def test_context_stays_in_dated_source_and_bad_raw_still_fails_closed():
    h, s, hierarchy, encoder = parent_fixture(future=True)
    memory = ResidentNativeSpineContextMemory(s, hierarchy, encoder=encoder, load_turn=h.get_turn)
    result = memory.retrieve(QUERY, DATED, max_direct=1, protected_direct=1, lexical_reserve=0)
    assert len(result.hydration.sections) == 3
    assert all(x.section.source_id == h.atoms[0].source_id for x in result.hydration.sections)
    rejected = hydrate_section_plan(result.routing.expanded,
        load_turn=lambda tid: h.get_turn(tid).model_copy(update={"text": "Changed content."}))
    assert not rejected.sections
    assert {d.reason for d in rejected.diagnostics} == {"raw_turn_identity_changed"}
    with pytest.raises(ValueError, match="protected routes"):
        replace(result.routing, expanded=result.routing.baseline, receipt_sha256="")


@pytest.mark.parametrize("options", [{"ancestor_hops": 3}, {"protected_direct": 33}])
def test_unbounded_context_rejected_before_raw_reads(options):
    h, s, hierarchy, encoder = parent_fixture()
    memory = ResidentNativeSpineContextMemory(s, hierarchy, encoder=encoder,
        load_turn=lambda _: pytest.fail("must not read raw"))
    with pytest.raises(ValueError, match="bounded prefix"):
        memory.retrieve(QUERY, DATED, **options)
