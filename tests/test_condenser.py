"""Facade wiring tests.

These use a deterministic fake embedder so the whole pipeline — chunk, embed,
index, extract, validate, rank, pack — is exercised without downloading bge-m3.
The real-model path is covered by tests/test_integration.py behind `slow`.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

from memory_condense.association_store import AssociationArtifact
from memory_condense.condenser import MemoryCondenser
from memory_condense.consolidation import ConsolidationNode
from memory_condense.context_packer import ContextBudget
from memory_condense.head_memory import MemoryLinkHit, MemoryLinkResult
from memory_condense.schemas import (
    Chunk,
    MemoryOps,
    MemoryStatus,
    MemoryType,
    PinState,
    Provenance,
    CreateOp,
    PinOp,
)


class FakeEmbedder:
    """Bag-of-words hashing embedder — deterministic across processes."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(token.encode()) % self._dim] += 1.0
        if not v.any():
            v[0] = 1.0
        return v

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            c.model_copy(update={"embedding": self._vec(c.text).tolist()})
            for c in chunks
        ]


class FakeQwenTurnLinker:
    """Records the bounded workspace and emits deterministic QK/OV hits."""

    def __init__(self) -> None:
        self.candidate_ids: tuple[str, ...] = ()

    def link(self, source_text, candidates, *, top_k=None):
        assert source_text
        bounded = list(candidates[:top_k])
        self.candidate_ids = tuple(candidate.episode_id for candidate in bounded)
        return MemoryLinkResult(
            hits=tuple(
                MemoryLinkHit(
                    episode_id=candidate.episode_id,
                    qk_score=float(len(bounded) - index),
                    ov_transport=1.0,
                    head_weights=(1.0,),
                )
                for index, candidate in enumerate(bounded)
            ),
            source_cav_signature=(0.25, 0.75),
            workspace_candidates=len(bounded),
            workspace_tokens=32,
        )


CONVERSATION = [
    ("user", "I prefer Python for this project. We decided to use SQLite for storage."),
    ("assistant", "Good call, SQLite in WAL mode never blocks readers."),
    ("user", "The index must never exceed one gigabyte on disk."),
]


def association_artifact() -> AssociationArtifact:
    return AssociationArtifact.create(
        model_id="Qwen/Qwen3-8B",
        checkpoint_id="bf16:first-shard:test",
        prefix_layers=7,
        head_layer=1,
        cav_layer=5,
        concept_names=("context_dependency", "binding_constraint"),
        head_count=4,
    )


@pytest.fixture
def mc(tmp_path):
    condenser = MemoryCondenser(
        data_dir=tmp_path / "data",
        embedder=FakeEmbedder(),
    )
    yield condenser
    condenser.close()


@pytest.fixture
def populated(mc):
    for role, text in CONVERSATION:
        mc.ingest(role, text)
    return mc


class TestIngest:
    def test_returns_turn_and_chunks(self, mc):
        turn, chunks = mc.ingest("user", "I prefer dark mode in all my apps.")
        assert turn.role == "user"
        assert chunks
        assert all(c.turn_id == turn.turn_id for c in chunks)

    def test_chunks_are_embedded_and_indexed(self, mc):
        _, chunks = mc.ingest("user", "SQLite is the storage layer.")
        assert all(c.embedding is not None for c in chunks)
        assert mc.search("storage layer", k=3)

    def test_source_identity_reaches_retrieval(self, mc):
        mc.ingest("user", "Project alpha uses SQLite.", source_id="session-alpha")
        mc.ingest("assistant", "Alpha keeps WAL enabled.", source_id="session-alpha")
        mc.ingest("user", "Project beta uses Redis.", source_id="session-beta")

        results = mc.search_sources("alpha SQLite", k_sources=1)
        assert results
        assert {result.turn.source_id for result in results} == {"session-alpha"}

    def test_hybrid_anchors_expand_complete_sources_fairly(self, mc):
        mc.ingest("user", "Alpha uses SQLite.", source_id="session-alpha")
        mc.ingest("assistant", "Alpha keeps WAL enabled.", source_id="session-alpha")
        mc.ingest("user", "Beta uses Redis.", source_id="session-beta")

        results = mc.search_anchored_sources("Alpha SQLite", k=2)

        assert results
        assert all(result.route == "anchored_source" for result in results)
        assert {result.turn.source_id for result in results} >= {"session-alpha"}
        assert {result.chunk.text for result in results} >= {
            "Alpha uses SQLite.",
            "Alpha keeps WAL enabled.",
        }

    def test_hybrid_source_rerank_only_admits_activated_sources(self, mc):
        mc.ingest("user", "Alpha uses SQLite.", source_id="session-alpha")
        mc.ingest("assistant", "Alpha keeps WAL enabled.", source_id="session-alpha")
        mc.ingest("user", "Beta uses Redis.", source_id="session-beta")

        results = mc.search_hybrid_sources(
            "Alpha SQLite", k=1, source_slots=3, source_candidate_pool=3
        )

        assert results[0].chunk.text == "Alpha uses SQLite."
        assert {result.turn.source_id for result in results} == {"session-alpha"}
        assert results[1].route == "hybrid_source"
        assert results[1].anchor_chunk_id == results[0].chunk.chunk_id

    def test_hybrid_source_validates_bounded_pool(self, populated):
        with pytest.raises(ValueError, match="at least k"):
            populated.search_hybrid_sources(
                "SQLite", k=3, source_candidate_pool=2
            )
        with pytest.raises(ValueError, match="between k and the pool"):
            populated.search_hybrid_sources(
                "SQLite", k=1, source_activation_k=4, source_candidate_pool=3
            )

    def test_hybrid_source_keeps_normal_candidate_pool_for_anchors(self, mc):
        mc.ingest("user", "Alpha uses SQLite.", source_id="session-alpha")
        calls = []
        original = mc.search_hybrid_from_embedding

        def recording_search(*args, **kwargs):
            calls.append((kwargs["k"], kwargs["candidates"]))
            return original(*args, **kwargs)

        mc.search_hybrid_from_embedding = recording_search
        mc.search_hybrid_sources(
            "Alpha SQLite",
            k=1,
            candidates=17,
            source_slots=2,
            source_candidate_pool=31,
        )

        assert calls == [(1, 17), (31, 31)]

    def test_hybrid_graph_unions_directional_neighbors_and_source_rerank(self, mc):
        mc.ingest("user", "Alpha begins here.", source_id="session-alpha")
        mc.ingest("assistant", "Alpha uses SQLite.", source_id="session-alpha")
        mc.ingest("user", "Alpha keeps WAL enabled.", source_id="session-alpha")
        mc.ingest("assistant", "Alpha ends here.", source_id="session-alpha")
        mc.ingest("user", "Beta uses Redis.", source_id="session-beta")

        results = mc.search_hybrid_graph(
            "Alpha SQLite",
            k=1,
            neighbor_radius=1,
            neighbor_slots=1,
            neighbor_direction="next",
            source_slots=2,
            source_candidate_pool=5,
            source_activation_k=2,
        )

        assert results[0].chunk.text == "Alpha uses SQLite."
        assert results[1].chunk.text == "Alpha keeps WAL enabled."
        assert results[1].transition_direction == "next"
        assert len({result.chunk.chunk_id for result in results}) == len(results)
        assert {result.turn.source_id for result in results} == {"session-alpha"}

    def test_hybrid_from_precomputed_embedding_matches_normal_path(self, mc):
        mc.ingest("user", "Alpha uses SQLite.", source_id="session-alpha")
        mc.ingest("assistant", "Beta uses Redis.", source_id="session-beta")
        query = "Alpha SQLite"
        embedding = mc._embedder.embed_query(query)

        normal = mc.search_hybrid(query, k=2)
        precomputed = mc.search_hybrid_from_embedding(query, embedding, k=2)

        assert [result.chunk.chunk_id for result in precomputed] == [
            result.chunk.chunk_id for result in normal
        ]
        assert [result.score for result in precomputed] == pytest.approx(
            [result.score for result in normal]
        )

    def test_hybrid_anchors_expand_only_bounded_source_neighbors(self, mc):
        mc.ingest("user", "Alpha begins here.", source_id="session-alpha")
        mc.ingest("assistant", "Alpha uses SQLite.", source_id="session-alpha")
        mc.ingest("user", "Alpha keeps WAL enabled.", source_id="session-alpha")
        mc.ingest("assistant", "Alpha ends here.", source_id="session-alpha")
        mc.ingest("user", "Beta uses Redis.", source_id="session-beta")

        results = mc.search_hybrid_neighbors("Alpha SQLite", k=1, radius=1)

        assert 1 <= len(results) <= 3
        assert "Alpha uses SQLite." in {result.chunk.text for result in results}
        assert all(result.turn.source_id == "session-alpha" for result in results)
        assert all(
            result.route == "hybrid_neighbor" for result in results[1:]
        )

    def test_hybrid_neighbors_can_replace_weak_anchors_at_fixed_count(self, mc):
        mc.ingest("user", "Alpha begins here.", source_id="session-alpha")
        mc.ingest("assistant", "Alpha uses SQLite.", source_id="session-alpha")
        mc.ingest("user", "Alpha keeps WAL enabled.", source_id="session-alpha")
        mc.ingest("assistant", "Beta uses Redis.", source_id="session-beta")

        results = mc.search_hybrid_neighbors(
            "Alpha SQLite",
            k=2,
            radius=1,
            max_neighbors=1,
            replacement_slots=1,
        )

        assert len(results) == 2
        assert results[-1].route == "hybrid_neighbor"

    def test_ingest_many_batches_embeddings_and_preserves_sources(self, tmp_path):
        class CountingEmbedder(FakeEmbedder):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def embed_chunks(self, chunks):
                self.calls += 1
                return super().embed_chunks(chunks)

        embedder = CountingEmbedder()
        with MemoryCondenser(
            data_dir=tmp_path / "batch",
            embedder=embedder,
            auto_extract=False,
            chunker_min_tokens=1,
            chunker_max_tokens=20,
        ) as batched:
            ingested = batched.ingest_many(
                [
                    ("user", "Alpha uses SQLite.", "session-alpha"),
                    ("assistant", "Alpha enables WAL.", "session-alpha"),
                    ("user", "Beta uses Redis.", "session-beta"),
                ]
            )

            assert embedder.calls == 1
            assert len(ingested) == 3
            assert [turn.source_id for turn, _ in ingested] == [
                "session-alpha",
                "session-alpha",
                "session-beta",
            ]
            assert all(chunks for _, chunks in ingested)
            source_results = batched.search_sources("Alpha SQLite", k_sources=1)
            assert {result.turn.source_id for result in source_results} == {
                "session-alpha"
            }

    def test_ingest_many_keeps_auto_extraction_turn_causal(self, tmp_path):
        with MemoryCondenser(
            data_dir=tmp_path / "causal-batch",
            embedder=FakeEmbedder(),
            auto_extract=True,
        ) as batched:
            batched.ingest_many(
                [
                    ("user", "I prefer dark mode.", "session-1"),
                    ("user", "We decided to use SQLite.", "session-2"),
                ]
            )

            assert batched.transcript.count() == 2
            assert len(batched.memory.list_items()) == 2

    def test_transcript_is_appended(self, populated):
        assert populated.transcript.count() == len(CONVERSATION)

    def test_memory_extracted_automatically(self, populated):
        items = populated.memory.list_items()
        assert items, "rule-based extractor should have produced memory items"

    def test_every_memory_carries_provenance(self, populated):
        for item in populated.memory.list_items():
            assert item.provenance, f"{item.mem_id} has no provenance"
            for prov in item.provenance:
                turn = populated.transcript.get_turn(prov.turn_id)
                assert turn is not None
                assert prov.quote.strip() in turn.text

    def test_auto_extract_can_be_disabled(self, tmp_path):
        with MemoryCondenser(
            data_dir=tmp_path / "d", embedder=FakeEmbedder(), auto_extract=False
        ) as quiet:
            quiet.ingest("user", "We decided to use SQLite for storage.")
            assert quiet.memory.list_items() == []

    def test_empty_text_is_safe(self, mc):
        turn, chunks = mc.ingest("user", "   ")
        assert chunks == []


class TestExtractionSafety:
    def test_fabricated_provenance_is_rejected(self, populated):
        fake = CreateOp(
            type=MemoryType.DECISION,
            content="We decided to rewrite everything in Rust.",
            provenance=[Provenance(turn_id="nope", quote="rewrite in Rust")],
        )
        report = populated.validator.validate(MemoryOps(create=[fake]))
        assert not report.ok
        assert report.accepted.is_empty()

    def test_misquoted_provenance_is_rejected(self, populated):
        real_turn = populated.transcript.get_all()[0]
        fake = CreateOp(
            type=MemoryType.DECISION,
            content="We decided to use Postgres.",
            provenance=[
                Provenance(turn_id=real_turn.turn_id, quote="we decided to use Postgres")
            ],
        )
        report = populated.validator.validate(MemoryOps(create=[fake]))
        assert not report.ok


class TestRetrieval:
    def test_dense_search_returns_scored_chunks(self, populated):
        results = populated.search("storage", k=3)
        assert results
        assert all(r.score is not None for r in results)

    def test_hybrid_search_populates_both_components(self, populated):
        results = populated.search_hybrid("SQLite storage", k=3)
        assert results
        assert all(r.dense_score is not None for r in results)
        assert all(r.lexical_score is not None for r in results)

    def test_hybrid_is_sorted_descending(self, populated):
        scores = [r.score for r in populated.search_hybrid("SQLite", k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_recall_memories_returns_ranked_items(self, populated):
        results = populated.recall_memories("what storage did we pick?", k=5)
        assert results
        assert all(r.item.status is MemoryStatus.ACTIVE for r in results)
        assert [r.score for r in results] == sorted(
            [r.score for r in results], reverse=True
        )

    def test_associative_search_preserves_cap_and_requires_explicit_slots(
        self, populated
    ):
        baseline = populated.search_hybrid("SQLite storage", k=3)
        assert len(baseline) == 3
        first, displaced, linked = baseline
        artifact = populated.associations.register_artifact(association_artifact())
        populated.associations.upsert_edge(
            first.chunk.chunk_id,
            linked.chunk.chunk_id,
            artifact.artifact_id,
            [0.9, 0.1, 0.3, 0.2],
            qk_score=0.9,
            ov_transport=0.4,
        )

        conservative = populated.search_associative(
            "SQLite storage",
            artifact.artifact_id,
            k=2,
            association_slots=0,
            touch=False,
        )
        assert [result.chunk.chunk_id for result in conservative] == [
            first.chunk.chunk_id,
            displaced.chunk.chunk_id,
        ]

        expanded = populated.search_associative(
            "SQLite storage",
            artifact.artifact_id,
            k=2,
            association_slots=1,
        )
        assert len(expanded) == 2
        assert [result.route for result in expanded] == ["hybrid", "qk"]
        assert expanded[1].chunk.chunk_id == linked.chunk.chunk_id
        assert expanded[1].anchor_chunk_id == first.chunk.chunk_id
        stored_edge = populated.associations.neighbors(
            first.chunk.chunk_id, artifact.artifact_id, top_k=1
        )[0]
        assert stored_edge.traversal_count == 1

    def test_associative_search_can_fill_a_reserved_slot_from_cavs(self, populated):
        baseline = populated.search_hybrid("SQLite storage", k=3)
        first, _, linked = baseline
        artifact = populated.associations.register_artifact(association_artifact())
        populated.associations.put_signature(
            first.chunk.chunk_id, artifact.artifact_id, [1.0, -0.5]
        )
        populated.associations.put_signature(
            linked.chunk.chunk_id, artifact.artifact_id, [0.8, -0.2]
        )

        expanded = populated.search_associative(
            "SQLite storage",
            artifact.artifact_id,
            k=2,
            association_slots=1,
        )
        assert len(expanded) == 2
        assert expanded[1].route == "cav"
        assert expanded[1].chunk.chunk_id == linked.chunk.chunk_id
        signature = populated.associations.get_signature(
            linked.chunk.chunk_id, artifact.artifact_id
        )
        assert signature is not None
        assert signature.access_count == 1

    def test_associative_expansion_can_follow_bounded_external_hops(self, populated):
        baseline = populated.search_hybrid("SQLite storage", k=3)
        source, gold, middle = baseline
        artifact = populated.associations.register_artifact(association_artifact())
        populated.associations.upsert_edge(
            source.chunk.chunk_id,
            middle.chunk.chunk_id,
            artifact.artifact_id,
            [0.9] * 4,
            qk_score=0.9,
        )
        populated.associations.upsert_edge(
            middle.chunk.chunk_id,
            gold.chunk.chunk_id,
            artifact.artifact_id,
            [0.8] * 4,
            qk_score=0.8,
        )
        anchors = [source]

        one_hop = populated.expand_associative(
            anchors,
            artifact.artifact_id,
            k=1,
            association_slots=1,
            association_hops=1,
            cav_candidates=0,
            lexical_protection_threshold=None,
            max_prompt_token_increase=None,
            touch=False,
        )
        two_hops = populated.expand_associative(
            anchors,
            artifact.artifact_id,
            k=1,
            association_slots=1,
            association_hops=2,
            max_association_candidates=4,
            cav_candidates=0,
            lexical_protection_threshold=None,
            max_prompt_token_increase=None,
            touch=False,
        )
        assert one_hop[0].chunk.chunk_id == middle.chunk.chunk_id
        assert two_hops[0].chunk.chunk_id == gold.chunk.chunk_id
        assert two_hops[0].anchor_chunk_id == source.chunk.chunk_id
        assert two_hops[0].association_hop == 2
        assert two_hops[0].edge_source_chunk_id == middle.chunk.chunk_id
        assert two_hops[0].association_path == (
            source.chunk.chunk_id,
            middle.chunk.chunk_id,
            gold.chunk.chunk_id,
        )

    def test_association_cannot_displace_protected_lexical_tail(self, populated):
        source, _, linked = populated.search_hybrid("SQLite storage", k=3)
        artifact = populated.associations.register_artifact(association_artifact())
        populated.associations.upsert_edge(
            source.chunk.chunk_id,
            linked.chunk.chunk_id,
            artifact.artifact_id,
            [0.9] * 4,
            qk_score=0.9,
        )
        protected = source.model_copy(update={"lexical_score": 0.95})

        expanded = populated.expand_associative(
            [protected],
            artifact.artifact_id,
            k=1,
            association_slots=1,
            lexical_protection_threshold=0.9,
            cav_candidates=0,
            touch=False,
        )

        assert expanded == [protected]

    def test_association_rolls_back_before_touch_when_prompt_cost_grows(
        self, populated
    ):
        source, _, linked = populated.search_hybrid("SQLite storage", k=3)
        artifact = populated.associations.register_artifact(association_artifact())
        populated.associations.upsert_edge(
            source.chunk.chunk_id,
            linked.chunk.chunk_id,
            artifact.artifact_id,
            [0.9] * 4,
            qk_score=0.9,
        )
        cheap_source = source.model_copy(
            update={
                "chunk": source.chunk.model_copy(update={"token_count": 1}),
                "lexical_score": 0.0,
            }
        )

        expanded = populated.expand_associative(
            [cheap_source],
            artifact.artifact_id,
            k=1,
            association_slots=1,
            cav_candidates=0,
            touch=True,
        )

        assert expanded == [cheap_source]
        edge = populated.associations.neighbors(
            source.chunk.chunk_id,
            artifact.artifact_id,
            top_k=1,
        )[0]
        assert edge.traversal_count == 0

    def test_live_hebbian_access_learns_and_recalls_co_retrieved_chunk(
        self, populated
    ):
        first, displaced, linked = populated.search_hybrid("SQLite storage", k=3)
        artifact = populated.associations.register_artifact(association_artifact())
        update = populated.observe_retrieval_access(
            [first, linked],
            artifact.artifact_id,
            access_event_id="generation-1",
        )
        assert update.edges_reinforced == 1

        expanded = populated.expand_hebbian(
            [first, displaced],
            artifact.artifact_id,
            k=2,
            hebbian_slots=1,
            lexical_protection_threshold=None,
            max_prompt_token_increase=None,
        )

        assert [result.chunk.chunk_id for result in expanded] == [
            first.chunk.chunk_id,
            linked.chunk.chunk_id,
        ]
        assert expanded[-1].route == "hebbian_coaccess"

    def test_retriever_deletion_also_removes_live_associations(self, populated):
        baseline = populated.search_hybrid("SQLite storage", k=3)
        first, _, linked = baseline
        artifact = populated.associations.register_artifact(association_artifact())
        populated.associations.put_signature(
            linked.chunk.chunk_id, artifact.artifact_id, [1.0, 0.0]
        )
        populated.associations.upsert_edge(
            first.chunk.chunk_id,
            linked.chunk.chunk_id,
            artifact.artifact_id,
            [0.5] * 4,
            qk_score=0.5,
            reverse=True,
        )
        populated.associations.reinforce_retrieval_coaccess(
            artifact.artifact_id,
            "delete-test",
            {
                first.chunk.chunk_id: 1.0,
                linked.chunk.chunk_id: 1.0,
            },
        )

        assert populated.retriever.delete_chunk(linked.chunk.chunk_id) is True
        assert populated.associations.stats(artifact.artifact_id)["signatures"] == 0
        assert populated.associations.stats(artifact.artifact_id)["edges"] == 0
        assert populated.associations.hebbian_stats(artifact.artifact_id)["edges"] == 0


class TestContextAssembly:
    def test_returns_packed_context(self, populated):
        ctx = populated.build_context("What storage are we using?")
        assert ctx.messages
        assert ctx.messages[-1]["role"] == "user"
        assert ctx.messages[-1]["content"] == "What storage are we using?"

    def test_system_prompt_comes_first(self, populated):
        ctx = populated.build_context("q", system_prompt="You are helpful.")
        assert ctx.messages[0] == {"role": "system", "content": "You are helpful."}

    def test_memory_header_present_when_memories_exist(self, populated):
        ctx = populated.build_context("What did we decide?")
        assert ctx.memory_header.startswith("Relevant memory:")

    def test_budget_is_respected(self, tmp_path):
        budget = ContextBudget(
            recent_window_tokens=30, memory_header_tokens=30, expansion_tokens=30
        )
        with MemoryCondenser(
            data_dir=tmp_path / "d", embedder=FakeEmbedder(), budget=budget
        ) as small:
            for role, text in CONVERSATION * 4:
                small.ingest(role, text)
            ctx = small.build_context("what did we decide?")
            assert ctx.token_counts["memory_header"] <= 30
            assert ctx.token_counts["recent_turns"] <= 30
            assert ctx.token_counts["expansions"] <= 30

    def test_zero_k_produces_no_memory_or_expansions(self, populated):
        ctx = populated.build_context("q", k_memories=0, k_expansions=0)
        assert ctx.memory_header == ""
        assert ctx.expansions == []

    def test_only_memories_that_reach_the_header_are_reheated(self, tmp_path):
        budget = ContextBudget(memory_header_tokens=24)
        with MemoryCondenser(
            data_dir=tmp_path / "selective-reheat",
            embedder=FakeEmbedder(),
            budget=budget,
        ) as small:
            for role, text in CONVERSATION:
                small.ingest(role, text)
            small.ingest("assistant", "A filler turn advances the decay coordinate.")
            before = {i.mem_id: i for i in small.memory.list_items()}

            ctx = small.build_context(
                "What constraints and decisions matter?",
                recent_turns=0,
                k_memories=10,
                k_expansions=0,
            )

            assert ctx.memory_ids
            assert ctx.dropped["memories"] > 0
            now_turn = small.transcript.current_turn()
            for mem_id, old in before.items():
                current = small.memory.get(mem_id)
                if mem_id in ctx.memory_ids:
                    assert current.last_access_turn == now_turn
                else:
                    assert current.last_access_turn == old.last_access_turn

    def test_measurement_can_pack_without_reheating(self, populated):
        before = {
            i.mem_id: (i.energy, i.last_access_turn)
            for i in populated.memory.list_items()
        }

        populated.build_context(
            "What did we decide?",
            k_expansions=0,
            reheat_memories=False,
        )

        after = {
            i.mem_id: (i.energy, i.last_access_turn)
            for i in populated.memory.list_items()
        }
        assert after == before

    def test_context_build_learns_only_the_items_that_reach_the_prompt(
        self, populated
    ):
        packed = populated.build_context(
            "What storage constraints and decisions matter?",
            recent_turns=0,
        )
        stats = populated.consolidation.stats()
        assert stats["nodes"] == len(packed.memory_ids) + len(
            packed.expansion_chunk_ids
        )
        assert stats["edges"] > 0
        assert stats["retained_prompt_state_bytes"] == 0

    def test_repeating_same_context_build_is_idempotent(self, populated):
        kwargs = dict(recent_turns=0, use_consolidation=False)
        populated.build_context("What did we decide?", **kwargs)
        before = populated.consolidation.stats()
        populated.build_context("What did we decide?", **kwargs)
        assert populated.consolidation.stats() == before

    def test_measurement_can_disable_consolidation_learning(self, populated):
        populated.build_context(
            "What did we decide?",
            recent_turns=0,
            learn_consolidation=False,
        )
        assert populated.consolidation.stats() == {
            "nodes": 0,
            "edges": 0,
            "event_receipts": 0,
            "retained_prompt_state_bytes": 0,
        }

    def test_explicit_qwen_weighted_context_observation(self, populated):
        packed = populated.build_context(
            "What did we decide?",
            recent_turns=0,
            use_consolidation=False,
            learn_consolidation=False,
        )
        left = ConsolidationNode.memory(packed.memory_ids[0])
        right = ConsolidationNode.chunk(packed.expansion_chunk_ids[0])
        update = populated.observe_context_access(
            packed.memory_ids,
            packed.expansion_chunk_ids,
            access_event_id="qwen-turn-1",
            node_activations={left: 1.0, right: 0.8},
            pair_affinities={(left, right): 0.4},
        )
        assert update.nodes_observed == 2
        neighbor = populated.consolidation.neighbors(
            {left: 1.0},
            top_k=1,
            min_coactivation_count=1,
        )[0]
        assert neighbor.node == right
        assert neighbor.score == pytest.approx(0.4)

    def test_qwen_consolidates_packed_direct_members_without_retaining_workspace(
        self, populated
    ):
        packed = populated.build_context(
            "What storage constraints and decisions matter?",
            recent_turns=0,
            use_consolidation=False,
            learn_consolidation=False,
        )
        linker = FakeQwenTurnLinker()

        result, update = populated.consolidate_context_with_qwen(
            "What storage constraints and decisions matter?",
            packed,
            linker,
        )

        expected_ids = {
            *(f"m:{mem_id}" for mem_id in packed.direct_memory_ids),
            *(f"c:{chunk_id}" for chunk_id in packed.direct_expansion_chunk_ids),
        }
        assert set(linker.candidate_ids) == expected_ids
        assert result.workspace_candidates == len(expected_ids)
        assert update.created is True
        assert update.nodes_observed == len(expected_ids)
        assert populated.consolidation.stats()["retained_prompt_state_bytes"] == 0

    def test_qwen_consolidation_rejects_double_learning(self, populated):
        packed = populated.build_context(
            "What did we decide?",
            recent_turns=0,
        )
        with pytest.raises(ValueError, match="already used rank-based"):
            populated.consolidate_context_with_qwen(
                "What did we decide?",
                packed,
                FakeQwenTurnLinker(),
            )


class TestLifecycle:
    def test_heat_counts_reported(self, populated):
        counts = populated.heat_counts()
        assert set(counts) <= {"HOT", "WARM", "COLD"}
        assert sum(counts.values()) == len(populated.memory.list_items())

    def test_pinning_survives_reopen(self, tmp_path):
        data_dir = tmp_path / "persist"
        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as first:
            first.ingest("user", "We decided to use SQLite for storage.")
            item = first.memory.list_items()[0]
            first.memory.pin(PinOp(mem_id=item.mem_id, pin=PinState.USER))
            mem_id = item.mem_id

        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as second:
            reloaded = second.memory.get(mem_id)
            assert reloaded is not None
            assert reloaded.pin is PinState.USER

    def test_transcript_and_index_persist(self, tmp_path):
        data_dir = tmp_path / "persist2"
        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as first:
            for role, text in CONVERSATION:
                first.ingest(role, text)

        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as second:
            assert second.transcript.count() == len(CONVERSATION)
            assert second.search("storage", k=3)

    def test_associative_retrieval_survives_a_full_facade_restart(self, tmp_path):
        data_dir = tmp_path / "association-restart"
        artifact = association_artifact()
        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as first:
            for role, text in CONVERSATION:
                first.ingest(role, text)
            baseline = first.search_hybrid("SQLite storage", k=3)
            source_id = baseline[0].chunk.chunk_id
            linked_id = baseline[2].chunk.chunk_id
            first.associations.register_artifact(artifact)
            first.associations.upsert_edge(
                source_id,
                linked_id,
                artifact.artifact_id,
                [0.9, 0.2, 0.1, 0.1],
                qk_score=0.9,
            )

        with MemoryCondenser(data_dir=data_dir, embedder=FakeEmbedder()) as second:
            results = second.search_associative(
                "SQLite storage",
                artifact.artifact_id,
                k=2,
                association_slots=1,
                touch=False,
            )
            assert len(results) == 2
            assert results[1].route == "qk"
            assert results[1].chunk.chunk_id == linked_id

    def test_context_manager_closes_cleanly(self, tmp_path):
        with MemoryCondenser(data_dir=tmp_path / "d", embedder=FakeEmbedder()) as c:
            c.ingest("user", "hello there")
        assert (tmp_path / "d" / "memory.db").exists()
