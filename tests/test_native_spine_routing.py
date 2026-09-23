from dataclasses import replace

import numpy as np
import pytest

from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.native_spine_memory import materialize_history
from memory_condense.search.native_spine_routing import NativeSpineRouter
from memory_condense.search.native_spine_summary import body_identity, fragment_body
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.summary_semantic_index import SemanticSectionIndex, summary_embedding_identity


QUERY = "Where did I go?"
DATED = "[Question asked at 2026/09/12 (Saturday) 12:00] " + QUERY


class Encoder:
    model_name = "deterministic test vectors, not a model measurement"
    model_revision = "test"
    checkpoint_sha256 = "test"
    execution_identity = {"test_double": True}

    def __init__(self):
        self.calls = []

    def embed_query(self, query):
        self.calls.append(query)
        return np.array([1, 0], dtype=np.float32)


def fixture(*, future=False):
    body = {"turns": [
        {"role": "user", "text": "I visited the botanical garden."},
        {"role": "assistant", "text": "You said the botanical garden in Kyoto."},
        {"role": "user", "text": "I also enjoy watching birds."},
    ]}
    cached = [{"pointer": f.pointer(), "summary": f"Stored summary {i}."}
              for i, f in enumerate(fragment_body(body, token_cap=128))]
    def source(day):
        s = {"original_session_ordinal": 0, "session_id": "fixture", "created_at": day + "T00:00:00+00:00",
             "metadata_text": "source", "body_sha256": body_identity(body), "dataset_origin": "M"}
        s["occurrence_id"] = identity_sha256(s)
        return s
    sessions = [source("2026-09-01")]
    if future:
        sessions.append(source("2026-09-13"))
    history = materialize_history(sessions, load_body=lambda _: body, load_summaries=lambda _: cached,
                                  compiler_identity="test")
    atoms = history.atoms
    leaves = [SectionSummary(f"leaf-{i}", atoms[i].source_id,
        "An intentionally unhelpful parent summary about unrelated material.",
        tuple(a.spans[0] for a in atoms[i:i+3]), "test") for i in range(0, len(atoms), 3)]
    index, hierarchy = SectionSummaryIndex(atoms), SectionSummaryIndex(leaves)
    encoder = Encoder()
    vectors = np.array([[1, 0] if a == atoms[0] or (future and a == atoms[3]) else [0, 1]
                        for a in index.sections], dtype=np.float32)
    semantic = SemanticSectionIndex(index, vectors, embedding_identity=summary_embedding_identity(encoder))
    return history, semantic, hierarchy, encoder


def test_unhelpful_parent_cannot_remove_direct_match_and_context_adds_exact_atoms():
    history, semantic, hierarchy, encoder = fixture()
    router = NativeSpineRouter(semantic, hierarchy)
    route = router.route_vector(QUERY, DATED, encoder.embed_query(QUERY),
        embedding_identity=semantic.embedding_identity, max_direct=1, lexical_reserve=0)
    assert route.baseline.routes[0].section == history.atoms[0]
    assert route.expanded.routes[:1] == route.baseline.routes
    assert set(route.added_atomic_ids) == {a.section_id for a in history.atoms[1:]}
    baseline = hydrate_section_plan(route.baseline, load_turn=history.get_turn)
    expanded = hydrate_section_plan(route.expanded, load_turn=history.get_turn)
    assert expanded.sections[:len(baseline.sections)] == baseline.sections
    assert [e.text for s in expanded.sections for e in s.evidence] == [t.text for t in history.turns.values()]
    assert route.raw_reads_during_routing == route.query_qwen_passes == 0


@pytest.mark.parametrize("span_budget,context_budget", [(1, 3072), (128, 80), (128, 3072)])
def test_additions_share_budgets_and_preserve_all_accepted_baseline_evidence(span_budget, context_budget):
    history, semantic, hierarchy, encoder = fixture()
    memory = ResidentNativeSpineMemory(semantic, hierarchy, encoder=encoder, load_turn=history.get_turn)
    options = dict(max_direct=1, lexical_reserve=0, max_raw_spans=span_budget, max_context_tokens=context_budget)
    b = memory.retrieve(QUERY, DATED, augment=False, **options)
    e = memory.retrieve(QUERY, DATED, **options)
    assert e.hydration.sections[:len(b.hydration.sections)] == b.hydration.sections
    assert e.hydration.raw_turn_read_count <= span_budget
    assert e.hydration.context_token_count <= context_budget
    assert encoder.calls == [QUERY, QUERY]  # Fresh query encoding in each arm.


def test_future_occurrences_excluded_from_direct_and_context_routes():
    history, semantic, hierarchy, encoder = fixture(future=True)
    memory = ResidentNativeSpineMemory(semantic, hierarchy, encoder=encoder, load_turn=history.get_turn)
    result = memory.retrieve(QUERY, DATED, max_direct=1, lexical_reserve=0)
    assert len(result.hydration.sections) == 3
    assert all(s.section.source_id == history.atoms[0].source_id for s in result.hydration.sections)


def test_incomplete_hierarchy_still_routes_all_original_atoms():
    history, semantic, _, encoder = fixture()
    router = NativeSpineRouter(semantic, SectionSummaryIndex(()))
    result = router.route_vector(QUERY, DATED, encoder.embed_query(QUERY),
        embedding_identity=semantic.embedding_identity, lexical_reserve=0)
    assert len(result.baseline.routes) == len(history.atoms)
    assert result.expanded == result.baseline


def test_foreign_hierarchy_is_rejected_and_invalid_date_does_not_encode_or_read():
    history, semantic, hierarchy, encoder = fixture()
    foreign, _, _, _ = fixture(future=True)
    bad = SectionSummary("foreign", foreign.atoms[-1].source_id, "Other occurrence.",
                         (foreign.atoms[-1].spans[0],), "test")
    with pytest.raises(ValueError, match="foreign evidence"):
        NativeSpineRouter(semantic, SectionSummaryIndex((bad,)))
    def forbidden(_):
        raise AssertionError("invalid request must not hydrate")
    memory = ResidentNativeSpineMemory(semantic, hierarchy, encoder=encoder, load_turn=forbidden)
    with pytest.raises(ValueError, match="date bound"):
        memory.retrieve(QUERY, DATED + " changed")
    assert encoder.calls == []


def test_stale_raw_identity_is_rejected_without_new_section_ids_or_clipping():
    history, semantic, hierarchy, encoder = fixture()
    def stale(turn_id):
        return history.get_turn(turn_id).model_copy(update={"text": "Different raw text."})
    memory = ResidentNativeSpineMemory(semantic, hierarchy, encoder=encoder, load_turn=stale)
    result = memory.retrieve(QUERY, DATED, max_direct=1, lexical_reserve=0)
    assert not result.hydration.sections
    assert {d.reason for d in result.hydration.diagnostics} == {"raw_turn_identity_changed"}
    assert {r.section.section_id for r in result.routing.expanded.routes} == {a.section_id for a in history.atoms}
