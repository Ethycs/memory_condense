from datetime import datetime, timezone

import numpy as np
import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from memory_condense.search.summary_shortlist_attention import rerank_summary_shortlist


class Encoder:
    model_name, model_revision, checkpoint_sha256 = "test-embedder", "v1", "a" * 64
    execution_identity = {"dtype": "float32", "device": "cpu"}

    def __init__(self):
        self.inputs = []

    def embed_queries(self, texts):
        self.inputs.append(list(texts))
        return np.array([[0, 1] if "pine" in s else [1, 0] for s in texts], dtype=np.float32)

    def embed_query(self, text):
        self.inputs.append(text)
        return np.array([1, 0], dtype=np.float32)


def hierarchy():
    turns = [Turn(turn_id=str(i), source_id=source, role="user", text=f"raw secret {i} τ\r\n",
                  created_at=datetime(2026, 9, 9, tzinfo=timezone.utc)) for i, source in enumerate(("a", "a", "b"))]
    leaves = [SectionSummary(str(i), turn.source_id, summary, (RawSectionSpan.from_turn(turn),), "terra")
              for i, (turn, summary) in enumerate(zip(turns, ("pine forest walk", "coastal drive", "coastal drive")))]
    parent = SectionSummary("parent", "a", "a lossy overview", leaves[0].spans + leaves[1].spans,
                            "qwen", child_section_ids=("0", "1"))
    return turns, SectionSummaryIndex([parent, *leaves])


def test_semantic_leaf_can_enter_without_a_lexical_or_parent_match_and_hydrate_exactly():
    turns, index = hierarchy()
    encoder = Encoder()
    dense = SemanticSectionIndex.compile(index, encoder=encoder)
    assert not index.route("road trip").routes
    plan = dense.route("road trip", encoder=encoder, max_sections=1)
    assert plan.routing_backend == "summary_dense" and plan.routes[0].section.section_id == "1"
    assert plan.index_sha256 == index.receipt_sha256 and not plan.frontier_closed
    assert "raw secret" not in repr(encoder.inputs) and "lossy overview" not in repr(encoder.inputs)
    hydrated = hydrate_section_plan(plan, load_turn={t.turn_id: t for t in turns}.get)
    assert hydrated.sections[0].evidence[0].text == turns[1].text


def test_hybrid_reserve_is_deduplicated_and_exact_scope_applies_before_ranking():
    _, index = hierarchy()
    encoder = Encoder()
    dense = SemanticSectionIndex.compile(index, encoder=encoder)
    hybrid = dense.route("pine", encoder=encoder, lexical_reserve=1, max_sections=2, eligible_source_ids=["a"])
    assert hybrid.routing_backend == "summary_hybrid"
    assert [r.section.section_id for r in hybrid.routes] == ["0", "1"]
    assert all(r.section.source_id == "a" for r in hybrid.routes)
    calls = len(encoder.inputs)
    assert not dense.route("road trip", encoder=encoder, eligible_source_ids=[]).routes
    assert len(encoder.inputs) == calls


def test_embedding_identity_invalid_vectors_and_source_mutation_are_rejected():
    _, index = hierarchy()
    encoder = Encoder()
    dense = SemanticSectionIndex.compile(index, encoder=encoder)
    with pytest.raises(ValueError, match="encoder identity"):
        dense.route_vector("query", np.array([1, 0]), embedding_identity="other")
    with pytest.raises(ValueError):
        dense.route_vector("query", np.array([0, 0]), embedding_identity=dense.embedding_identity)
    encoder.model_revision = "changed"
    with pytest.raises(ValueError, match="does not match"):
        dense.route("query", encoder=encoder)
    with pytest.raises(AttributeError):
        dense.sections = ()


def test_dense_shortlist_can_feed_the_existing_one_pass_attention_contract():
    from tests.test_summary_shortlist_attention import Linker
    _, index = hierarchy()
    encoder = Encoder()
    dense = SemanticSectionIndex.compile(index, encoder=encoder)
    shortlist = dense.route("road trip", encoder=encoder, max_sections=2)
    linker = Linker()
    plan = rerank_summary_shortlist("road trip", index, shortlist, linker=linker, max_sections=1)
    assert len(linker.inputs) == 1 and plan.attention_receipt.rounds[0].model_passes == 1
    assert plan.routes[0].section.section_id in {r.section.section_id for r in shortlist.routes}
