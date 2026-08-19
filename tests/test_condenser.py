"""Facade wiring tests.

These use a deterministic fake embedder so the whole pipeline — chunk, embed,
index, extract, validate, rank, pack — is exercised without downloading bge-m3.
The real-model path is covered by tests/test_integration.py behind `slow`.
"""

from __future__ import annotations

import re
import zlib
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from memory_condense.associations.association_store import AssociationArtifact
from memory_condense.application.condenser import (
    MemoryCondenser,
    is_multi_fact_query,
    query_facets,
    rank_concept_members,
    role_aware_results,
    source_diverse_results,
)
from memory_condense.associations.consolidation import ConsolidationNode
from memory_condense.search.packing.context_packer import ContextBudget
from memory_condense.associations.head_memory_models import (
    MemoryLinkHit,
    MemoryLinkResult,
)
from memory_condense.domain.schemas import (
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


class FakeCAVCompiler:
    def __init__(self) -> None:
        self.cav_bank = SimpleNamespace(
            names=("context_dependency", "binding_constraint"),
            layer=5,
        )
        self.calls: list[tuple[str, ...]] = []

    def signatures(self, texts, *, batch_size=8):
        self.calls.append(tuple(texts))
        return tuple((float(index), -float(index)) for index, _ in enumerate(texts, 1))


class FakeAttentionFeedback:
    candidate_pool = 8

    def __init__(self) -> None:
        self.last_report = None
        self.seen_routes = []

    def select(self, query, candidates, *, top_k):
        self.seen_routes = [candidate.route for candidate in candidates]
        self.last_report = SimpleNamespace(
            model_dump=lambda: {
                "passes": 1,
                "max_workspace_candidates": len(candidates),
                "max_workspace_tokens": 64,
                "total_candidate_inspections": len(candidates),
                "qwen_candidates_added": min(top_k, len(candidates)),
            }
        )
        return list(candidates[:top_k])


class FakeSourceReranker:
    candidate_pool = 6

    def __init__(self) -> None:
        self.last_report = None
        self.candidate_sources: list[str] = []
        self.unique_sources = False

    def rerank(self, query, candidates, *, top_k, unique_sources=False):
        self.candidate_sources = [
            str(candidate.turn.source_id) for candidate in candidates
        ]
        self.unique_sources = unique_sources
        self.last_report = SimpleNamespace(model_dump=lambda: {})
        return list(candidates[:top_k])


def test_query_facets_extracts_explicit_multi_event_list():
    query = (
        "[Question asked at 2023/06/01]\n"
        "Which happened first: the day I prepared the nursery, "
        "the day I picked baby-shower gifts, and the day I ordered a phone case?"
    )

    assert query_facets(query) == [
        "the day I prepared the nursery",
        "the day I picked baby-shower gifts",
        "the day I ordered a phone case",
    ]


def test_query_facets_does_not_guess_for_singleton_question():
    assert query_facets("Where do I take yoga classes?") == []


def test_concept_member_ranking_binds_event_membership_to_query_object(populated):
    first, second = populated.search_hybrid("SQLite", k=2)
    unrelated = first.model_copy(
        update={
            "chunk": first.chunk.model_copy(
                update={"text": "I completed a cooking class yesterday."}
            ),
            "score": 1.0,
        }
    )
    museum = second.model_copy(
        update={
            "chunk": second.chunk.model_copy(
                update={"text": "I visited the modern art museum yesterday."}
            ),
            "score": 0.1,
        }
    )

    ranked = rank_concept_members(
        "Which museums did I visit?", [unrelated, museum]
    )

    assert ranked[0].chunk.text == museum.chunk.text


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

    def test_cav_compilation_persists_only_fixed_width_coordinates(self, mc):
        _, first = mc.ingest("user", "I visited the modern art museum yesterday.")
        _, second = mc.ingest("user", "I plan to visit another museum tomorrow.")
        linker = FakeCAVCompiler()
        artifact = association_artifact()

        report = mc.compile_cav_signatures(
            linker,
            artifact,
            [*first, *second],
            batch_size=2,
        )
        repeated = mc.compile_cav_signatures(
            linker,
            artifact,
            [*first, *second],
            batch_size=2,
        )

        assert report == {
            "requested": 2,
            "compiled": 2,
            "reused": 0,
            "compiled_spans": 2,
            "signature_width": 2,
            "retained_request_token_state_bytes": 0,
            "retained_token_state_bytes": 0,
        }
        assert repeated["compiled"] == 0
        assert repeated["reused"] == 2
        assert len(linker.calls) == 2
        assert linker.calls[-1] == ()
        assert mc.associations.stats(artifact.artifact_id) == {
            "signatures": 2,
            "edges": 0,
            "cav_payload_bytes": 16,
            "head_payload_bytes": 0,
            "retained_request_token_state_bytes": 0,
            "retained_token_state_bytes": 0,
        }

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
        assert results[1].chunk.text == "Alpha keeps WAL enabled."
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

    def test_partition_local_search_recovers_chunk_absent_from_global_pool(self, mc):
        _, anchor_chunks = mc.ingest(
            "user",
            "Alpha project overview.",
            source_id="session-alpha",
        )
        _, target_chunks = mc.ingest(
            "assistant",
            "The launch code is cerulean.",
            source_id="session-alpha",
        )
        anchor = mc.retriever.hydrate_chunk(
            anchor_chunks[0].chunk_id,
            score=1.0,
            route="hybrid",
        )
        assert anchor is not None

        def truncated_global_pool(*args, **kwargs):
            return [anchor]

        mc.search_hybrid_from_embedding = truncated_global_pool
        legacy = mc.search_hybrid_sources(
            "What is the launch code?",
            k=1,
            source_slots=1,
            source_candidate_pool=1,
            source_activation_k=1,
            source_local_search=False,
        )
        local = mc.search_hybrid_sources(
            "What is the launch code?",
            k=1,
            source_slots=1,
            source_candidate_pool=1,
            source_activation_k=1,
            source_local_search=True,
        )

        assert [result.chunk.chunk_id for result in legacy] == [
            anchor_chunks[0].chunk_id
        ]
        assert [result.chunk.chunk_id for result in local] == [
            anchor_chunks[0].chunk_id,
            target_chunks[0].chunk_id,
        ]
        assert local[1].route == "hybrid_source_local"
        assert local[1].anchor_chunk_id == anchor_chunks[0].chunk_id

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

    def test_role_prior_prefers_user_fact_for_first_person_question(self, mc):
        _, user_chunks = mc.ingest(
            "user", "I went to Hawaii with my family.", source_id="trip"
        )
        _, assistant_chunks = mc.ingest(
            "assistant", "You could visit Cancun with family.", source_id="advice"
        )
        user = mc.retriever.hydrate_chunk(
            user_chunks[0].chunk_id, score=0.70, route="hybrid"
        )
        assistant = mc.retriever.hydrate_chunk(
            assistant_chunks[0].chunk_id, score=1.0, route="hybrid"
        )
        assert user is not None and assistant is not None

        ranked = role_aware_results(
            "Where did I go with my family?", [assistant, user]
        )

        assert ranked[0].turn.role == "user"
        assert role_aware_results("What destinations exist?", [assistant, user]) == [
            assistant,
            user,
        ]

    def test_multi_fact_source_diversity_round_robins_sources(self, mc):
        _, alpha_chunks = mc.ingest(
            "user", "Alpha museum one. Alpha museum two.", source_id="alpha"
        )
        _, beta_chunks = mc.ingest(
            "user", "Beta museum.", source_id="beta"
        )
        alpha = [
            mc.retriever.hydrate_chunk(chunk.chunk_id, score=1.0, route="hybrid")
            for chunk in alpha_chunks
        ]
        beta = mc.retriever.hydrate_chunk(
            beta_chunks[0].chunk_id, score=0.8, route="hybrid"
        )
        assert all(alpha) and beta is not None

        ranked = source_diverse_results([alpha[0], alpha[-1], beta])

        assert [result.turn.source_id for result in ranked] == [
            "alpha",
            "beta",
            "alpha",
        ]
        assert is_multi_fact_query("Put every museum in order") is True
        assert is_multi_fact_query("Where did I go?") is False

    def test_hybrid_graph_reserves_round_robin_slots_for_query_facets(self, mc):
        _, anchor_chunks = mc.ingest(
            "assistant", "General event planning advice.", source_id="general"
        )
        _, nursery_chunks = mc.ingest(
            "user", "I prepared my friend's nursery today.", source_id="nursery"
        )
        _, shower_chunks = mc.ingest(
            "user", "I picked gifts for the baby shower today.", source_id="shower"
        )
        _, phone_chunks = mc.ingest(
            "user", "I ordered the customized phone case today.", source_id="phone"
        )
        hydrated = {
            "main": mc.retriever.hydrate_chunk(
                anchor_chunks[0].chunk_id, score=1.0, route="hybrid"
            ),
            "nursery": mc.retriever.hydrate_chunk(
                nursery_chunks[0].chunk_id, score=1.0, route="hybrid"
            ),
            "shower": mc.retriever.hydrate_chunk(
                shower_chunks[0].chunk_id, score=1.0, route="hybrid"
            ),
            "phone": mc.retriever.hydrate_chunk(
                phone_chunks[0].chunk_id, score=1.0, route="hybrid"
            ),
        }
        assert all(hydrated.values())

        def facet_pool(query, _embedding, **_kwargs):
            if query.startswith("Which happened first"):
                return [hydrated["main"]]
            if "nursery" in query:
                return [hydrated["nursery"]]
            if "baby shower" in query:
                return [hydrated["shower"]]
            if "phone case" in query:
                return [hydrated["phone"]]
            return [hydrated["main"]]

        mc.search_hybrid_from_embedding = facet_pool
        results = mc.search_hybrid_graph(
            "Which happened first: I prepared the nursery, "
            "I picked baby shower gifts, and I ordered a phone case?",
            k=1,
            neighbor_radius=0,
            neighbor_slots=0,
            source_slots=3,
            source_candidate_pool=4,
            source_activation_k=1,
            query_facet_retrieval=True,
            query_facet_slots=3,
            query_facet_max=3,
        )

        assert [result.route for result in results[1:]] == [
            "query_facet",
            "query_facet",
            "query_facet",
        ]
        assert {result.turn.source_id for result in results[1:]} == {
            "nursery",
            "shower",
            "phone",
        }
        assert mc.last_query_facet_report == {
            "enabled": True,
            "facets": 3,
            "reserved_slots": 3,
            "candidates_added": 3,
        }

    def test_hybrid_graph_tfisf_can_activate_a_source_below_chunk_prefix(self, mc):
        _, alpha_chunks = mc.ingest(
            "user", "Alpha overview.", source_id="session-alpha"
        )
        _, beta_chunks = mc.ingest(
            "user", "The cerulean launch code.", source_id="session-beta"
        )
        alpha = mc.retriever.hydrate_chunk(
            alpha_chunks[0].chunk_id, score=1.0, route="hybrid"
        )
        beta = mc.retriever.hydrate_chunk(
            beta_chunks[0].chunk_id, score=0.5, route="hybrid"
        )
        assert alpha is not None and beta is not None

        def fixed_pool(*args, **kwargs):
            return [alpha] if kwargs["k"] == 1 else [alpha, beta]

        mc.search_hybrid_from_embedding = fixed_pool
        mc.retriever.source_tfisf_query = lambda *args, **kwargs: [
            ("session-beta", 2.0)
        ]

        results = mc.search_hybrid_graph(
            "What is the launch code?",
            k=1,
            neighbor_radius=0,
            neighbor_slots=0,
            source_slots=1,
            source_candidate_pool=2,
            source_activation_k=1,
            source_tfisf_activation=True,
            source_tfisf_slots=1,
        )

        assert [result.chunk.chunk_id for result in results] == [
            alpha_chunks[0].chunk_id,
            beta_chunks[0].chunk_id,
        ]
        assert mc.last_source_tfisf_report["admitted_sources"] == [
            "session-beta"
        ]

    def test_hybrid_graph_hsc_reserve_hydrates_original_sibling_source(self, mc):
        _, alpha_chunks = mc.ingest(
            "user", "Alpha project overview.", source_id="project::alpha"
        )
        _, beta_chunks = mc.ingest(
            "user", "The cerulean launch code.", source_id="project::beta"
        )
        alpha = mc.retriever.hydrate_chunk(
            alpha_chunks[0].chunk_id, score=1.0, route="hybrid"
        )
        assert alpha is not None

        mc.search_hybrid_from_embedding = lambda *args, **kwargs: [alpha]
        mc.retriever.source_hsc_expand = lambda *args, **kwargs: [
            ("project::beta", 0.9)
        ]

        results = mc.search_hybrid_graph(
            "What is the launch code?",
            k=1,
            neighbor_radius=0,
            neighbor_slots=0,
            source_slots=1,
            source_candidate_pool=1,
            source_activation_k=1,
            source_hsc_activation=True,
            source_hsc_slots=1,
            source_hsc_hops=1,
            source_hsc_chunk_slots=1,
        )

        assert [result.chunk.chunk_id for result in results] == [
            alpha_chunks[0].chunk_id,
            beta_chunks[0].chunk_id,
        ]
        assert results[1].route == "hsc_contraction"
        assert results[1].memory_source_id == "project::beta"

    def test_hybrid_graph_routes_inside_selected_hierarchical_partition(self, mc):
        for text in [
            "Alpha project overview.",
            "Alpha uses SQLite.",
            "Alpha keeps WAL enabled.",
        ]:
            mc.ingest("user", text, source_id="project-alpha::session-main")
        mc.ingest(
            "user",
            "Unrelated garden notes.",
            source_id="project-beta::session-main",
        )

        results = mc.search_hybrid_graph(
            "How does Alpha store data?",
            k=1,
            neighbor_radius=0,
            neighbor_slots=0,
            source_slots=2,
            source_candidate_pool=4,
            source_activation_k=2,
            source_partition_routing=True,
            source_partition_slots=1,
        )

        assert results
        assert {
            result.turn.source_id.split("::", 1)[0] for result in results
        } == {"project-alpha"}
        assert mc.last_partition_routing_report["selected_partitions"] == [
            "project-alpha"
        ]
        assert mc.last_partition_routing_report["routed_sources"] == 1

    def test_typed_partition_scan_forces_source_below_global_pool_at_fixed_count(
        self, tmp_path, monkeypatch
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-scan",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            exact_chunks = []
            filler_chunks = []
            for index, venue in enumerate(
                ("Science Museum", "Museum of History"),
                start=1,
            ):
                source_id = f"history::visit-{index}"
                small.ingest(
                    "system",
                    f"[{source_id} took place at 2024/03/0{index}]",
                    source_id=source_id,
                )
                _, exact = small.ingest(
                    "user",
                    f"I visited the {venue} today.",
                    source_id=source_id,
                )
                _, filler = small.ingest(
                    "assistant",
                    "Here is a generic museum recap with no answer value.",
                    source_id=source_id,
                )
                exact_chunks.append(exact[0])
                filler_chunks.append(filler[0])
            _, decoy = small.ingest(
                "user",
                "Museum catalog notes without a completed visit.",
                source_id="history::decoy",
            )
            baseline = []
            for index, chunk in enumerate(
                [exact_chunks[0], filler_chunks[0], decoy[0]]
            ):
                result = small.retriever.hydrate_chunk(
                    chunk.chunk_id,
                    score=1.0 - index * 0.1,
                    route="hybrid_source_local",
                )
                assert result is not None
                baseline.append(result)
            small.search_hybrid_from_embedding = lambda *args, **kwargs: list(
                baseline
            )
            small.retriever.hybrid_query_sources = (
                lambda *args, **kwargs: list(baseline)
            )

            results = small.search_hybrid_graph(
                "Put the two museums I visited in chronological order.",
                k=1,
                neighbor_radius=0,
                neighbor_slots=0,
                source_slots=2,
                source_candidate_pool=3,
                source_activation_k=3,
                source_partition_routing=True,
                source_partition_slots=1,
            )

            assert len(results) == len(baseline) == 3
            assert {
                result.chunk.chunk_id for result in results[:2]
            } == {chunk.chunk_id for chunk in exact_chunks}
            recovered = next(
                result
                for result in results
                if result.chunk.chunk_id == exact_chunks[1].chunk_id
            )
            assert recovered.chunk.text == exact_chunks[1].text
            assert recovered.turn is not None
            assert recovered.turn.source_id == "history::visit-2"
            report = small.last_partition_routing_report
            assert report["active_partition_scan_status"] == "applied"
            assert report["active_partition_exhaustive"] is True
            assert report["active_partition_structural_hypotheses"] == 2
            assert report["active_partition_candidates_already_present"] == 1
            assert report["active_partition_candidates_admitted"] == 1
            assert report["active_partition_candidates_replaced"] == 1
            assert report["active_partition_candidate_count_before"] == 3
            assert report["active_partition_candidate_count_after"] == 3

            class PartitionAuditSelector:
                last_report = None
                last_candidate_trace = []

                def __init__(self):
                    self.calls = []
                    self.scans = []

                def select(
                    self,
                    _query,
                    values,
                    *,
                    source_timestamps=None,
                    active_partition_total=None,
                    active_partition_inspected=None,
                    active_partition_scan=None,
                ):
                    del source_timestamps
                    self.calls.append(
                        (active_partition_total, active_partition_inspected)
                    )
                    self.scans.append(active_partition_scan)
                    return list(values)

            selector = PartitionAuditSelector()
            small.set_context_candidate_selector(selector)
            query = "Put the two museums I visited in chronological order."
            small.build_context(
                query,
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=False,
                learn_consolidation=False,
                expansion_results=results,
            )
            assert selector.calls[-1] == (
                report["active_partition_total"],
                report["active_partition_inspected"],
            )
            assert selector.scans[-1] == {
                "active_partition_total": report["active_partition_total"],
                "active_partition_inspected": report[
                    "active_partition_inspected"
                ],
                "active_partition_exhaustive": True,
                "active_partition_sources_total": report[
                    "active_partition_sources_total"
                ],
                "active_partition_structural_rows": 2,
                "active_partition_structural_hypotheses": 2,
                "active_partition_candidates_admitted": 1,
                "active_partition_candidates_already_present": 1,
                "active_partition_candidates_replaced": 1,
                "active_partition_candidates_truncated": 0,
                "active_partition_structural_overflow": 0,
                "active_partition_scan_contract": (
                    "canonical_venue_episode_aligned_v1"
                ),
                "active_partition_semantically_complete": True,
                "partition_scope_kind": "global",
                "partition_inventory_total": 1,
                "selected_partition_count": 1,
                "partition_scope_exhaustive": True,
                "selected_scope_structurally_complete": True,
                "global_semantic_complete": True,
            }
            assert small.last_partition_routing_report[
                "active_partition_snapshot_validated"
            ] is True

            small.build_context(
                query,
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=False,
                learn_consolidation=False,
                expansion_results=list(reversed(results)),
            )
            assert selector.calls[-1] == (
                report["active_partition_total"],
                report["active_partition_inspected"],
            )
            assert selector.scans[-1] is not None
            assert small.last_partition_routing_report[
                "active_partition_snapshot_validated"
            ] is True

            def reorder_after_prevalidation(memories, expansions, **_kwargs):
                return list(memories), list(reversed(expansions))

            monkeypatch.setattr(
                "memory_condense.application.condenser.expand_context_associations",
                reorder_after_prevalidation,
            )
            small.build_context(
                query,
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=True,
                learn_consolidation=False,
                expansion_results=results,
            )
            assert selector.calls[-1] == (
                report["active_partition_total"],
                report["active_partition_inspected"],
            )
            assert selector.scans[-1] is not None
            assert small.last_partition_routing_report[
                "active_partition_snapshot_prevalidated"
            ] is True
            assert small.last_partition_routing_report[
                "active_partition_snapshot_validated"
            ] is True

            def mutate_audited_route(memories, expansions, **_kwargs):
                changed = list(expansions)
                changed[0] = changed[0].model_copy(update={"route": "hybrid"})
                return list(memories), changed

            monkeypatch.setattr(
                "memory_condense.application.condenser.expand_context_associations",
                mutate_audited_route,
            )
            small.build_context(
                query,
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=True,
                learn_consolidation=False,
                expansion_results=results,
            )
            assert selector.calls[-1] == (None, None)
            assert selector.scans[-1] is None
            assert small.last_partition_routing_report[
                "active_partition_snapshot_prevalidated"
            ] is True
            assert small.last_partition_routing_report[
                "active_partition_snapshot_validated"
            ] is False
            assert small.last_partition_routing_report[
                "active_partition_snapshot_invalidated_reason"
            ] == "audited_frontier_changed_before_pack"

    def test_active_partition_snapshot_invalidates_when_transcript_advances(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-route-race",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            source_id = "history::visit-1"
            small.ingest(
                "system",
                f"[{source_id} took place at 2023/02/15]",
                source_id=source_id,
            )
            _, chunks = small.ingest(
                "user",
                "I visited the Science Museum today.",
                source_id=source_id,
            )
            baseline = small.retriever.hydrate_chunk(
                chunks[0].chunk_id,
                score=1.0,
                route="hybrid_source_local",
            )
            assert baseline is not None
            small.search_hybrid_from_embedding = lambda *args, **kwargs: [
                baseline
            ]
            advanced = False

            def advancing_local_search(*args, **kwargs):
                nonlocal advanced
                if not advanced:
                    advanced = True
                    small.ingest(
                        "user",
                        "A concurrent turn arrived after source routing.",
                        source_id="history::late-source",
                    )
                return [baseline]

            small.retriever.hybrid_query_sources = advancing_local_search

            results = small.search_hybrid_graph(
                "Name one museum I visited.",
                k=1,
                neighbor_radius=0,
                neighbor_slots=0,
                source_slots=0,
                source_candidate_pool=1,
                source_partition_routing=True,
                source_partition_slots=1,
            )

            assert [result.chunk.chunk_id for result in results] == [
                chunks[0].chunk_id
            ]
            report = small.last_partition_routing_report
            assert report["active_partition_scan_status"] == "invalidated"
            assert report["active_partition_exhaustive"] is False
            assert report["active_partition_semantically_complete"] is False
            assert report["active_partition_snapshot_invalidated_reason"] == (
                "transcript_advanced_during_route"
            )
            assert report["active_partition_snapshot_validated"] is False
            assert small._active_partition_routing_snapshot is None

    def test_active_partition_snapshot_invalidates_when_chunks_advance_without_turn(
        self, tmp_path, monkeypatch
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-content-race",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            source_id = "history::visit-1"
            small.ingest(
                "system",
                f"[{source_id} took place at 2023/02/15]",
                source_id=source_id,
            )
            _, chunks = small.ingest(
                "user",
                "I visited the Science Museum today.",
                source_id=source_id,
            )
            baseline = small.retriever.hydrate_chunk(
                chunks[0].chunk_id,
                score=1.0,
                route="hybrid_source_local",
            )
            assert baseline is not None
            small.search_hybrid_from_embedding = lambda *args, **kwargs: [
                baseline
            ]
            small.retriever.hybrid_query_sources = lambda *args, **kwargs: [
                baseline
            ]
            generations = iter((100, 101))
            monkeypatch.setattr(
                small,
                "_content_high_watermark",
                lambda: next(generations),
            )

            results = small.search_hybrid_graph(
                "Name one museum I visited.",
                k=1,
                neighbor_radius=0,
                neighbor_slots=0,
                source_slots=0,
                source_candidate_pool=1,
                source_partition_routing=True,
                source_partition_slots=1,
            )

            assert [result.chunk.chunk_id for result in results] == [
                chunks[0].chunk_id
            ]
            report = small.last_partition_routing_report
            assert report["active_partition_scan_status"] == "invalidated"
            assert report["active_partition_exhaustive"] is False
            assert report["global_semantic_complete"] is False
            assert report["active_partition_snapshot_invalidated_reason"] == (
                "content_changed_during_route"
            )
            assert small._active_partition_routing_snapshot is None

    def test_source_companion_cannot_replace_or_launder_structural_scan_row(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-companion-protection",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            source_id = "history::visit-1"
            small.ingest(
                "system",
                f"[{source_id} took place at 2023/02/15]",
                source_id=source_id,
            )
            _, plan = small.ingest(
                "user",
                "I am planning to visit the Science Museum tomorrow.",
                source_id=source_id,
            )
            _, visit = small.ingest(
                "user",
                "I visited the Science Museum today.",
                source_id=source_id,
            )
            structural = small.retriever.hydrate_chunk(
                visit[0].chunk_id,
                score=1.0,
                route="active_partition_structural",
                anchor_chunk_id=visit[0].chunk_id,
            )
            assert structural is not None

            class PlanFirstSelector:
                last_report = None
                last_candidate_trace = []
                strict = False

                def select_source_companions(self, _query, candidates_by_source):
                    selected = {
                        key: candidates[0]
                        for key, candidates in candidates_by_source.items()
                    }
                    self.last_source_companion_report = {
                        "input_sources": len(selected),
                        "input_candidates": len(selected),
                        "inspected_candidates": len(selected),
                        "selected_chunk_ids": {
                            key: result.chunk.chunk_id
                            for key, result in selected.items()
                        },
                        "selected_membership_scores": {
                            key: 0.9 for key in selected
                        },
                        "fallback_reason": "",
                    }
                    return selected

                def select(self, _query, candidates, **_kwargs):
                    return list(candidates)

            small.set_context_candidate_selector(PlanFirstSelector())
            output, orphans = small._hydrate_source_metadata_companions(
                "Name one museum I visited.",
                [structural],
                small._embedder.embed_query("Name one museum I visited."),
            )

            assert orphans == set()
            assert len(output) == 1
            assert output[0].chunk.chunk_id == visit[0].chunk_id
            assert output[0].chunk.chunk_id != plan[0].chunk_id
            assert output[0].route == "active_partition_structural"
            report = small.last_source_companion_report
            assert report["selected_chunk_ids"][source_id] == plan[0].chunk_id
            assert report["hydrated_sources"] == []
            assert report["refreshed_sources"] == []
            assert report["active_partition_protected_sources"] == [source_id]

    def test_active_partition_admission_protects_typed_prefix_from_hsc_variance(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-hsc-invariance",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            hydrated = []
            for index, text in enumerate(
                (
                    "I visited the Science Museum today.",
                    "I visited the Museum of History today.",
                    "volatile HSC row alpha",
                    "volatile HSC row beta",
                    "protected scalar anchor",
                )
            ):
                _, chunks = small.ingest(
                    "user",
                    text,
                    source_id=f"history::source-{index}",
                )
                result = small.retriever.hydrate_chunk(
                    chunks[0].chunk_id,
                    score=1.0 - index * 0.1,
                    route=("hsc_contraction" if index in {2, 3} else "hybrid"),
                )
                assert result is not None
                hydrated.append(result)
            typed = hydrated[:2]
            anchor = hydrated[4]
            first, first_report = small._admit_active_partition_candidates(
                [anchor, hydrated[2], hydrated[3]],
                typed,
                anchor_chunk_ids={anchor.chunk.chunk_id},
            )
            second, second_report = small._admit_active_partition_candidates(
                [anchor, hydrated[3], hydrated[2]],
                typed,
                anchor_chunk_ids={anchor.chunk.chunk_id},
            )

            expected = [result.chunk.chunk_id for result in typed]
            assert [result.chunk.chunk_id for result in first[:2]] == expected
            assert [result.chunk.chunk_id for result in second[:2]] == expected
            assert len(first) == len(second) == 3
            assert first[-1].chunk.chunk_id == anchor.chunk.chunk_id
            assert second[-1].chunk.chunk_id == anchor.chunk.chunk_id
            assert first_report["active_partition_candidates_replaced"] == 2
            assert second_report["active_partition_candidates_replaced"] == 2

    def test_incomplete_partition_scan_preserves_anchors_and_frontier_reserve(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-incomplete-admission",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            hydrated = []
            for index in range(16):
                _, chunks = small.ingest(
                    "user",
                    f"candidate row {index}",
                    source_id=f"history::source-{index}",
                )
                result = small.retriever.hydrate_chunk(
                    chunks[0].chunk_id,
                    score=1.0 - index / 100.0,
                    route=(
                        "hybrid"
                        if index < 8
                        else "active_partition_structural"
                    ),
                )
                assert result is not None
                hydrated.append(result)
            baseline = hydrated[:8]
            typed = hydrated[8:]

            output, report = small._admit_active_partition_candidates(
                baseline,
                typed,
                anchor_chunk_ids={baseline[0].chunk.chunk_id},
                semantic_complete=False,
            )

            assert len(output) == len(baseline) == 8
            assert [result.chunk.chunk_id for result in output[:6]] == [
                result.chunk.chunk_id for result in typed[:6]
            ]
            assert {result.chunk.chunk_id for result in output[6:]} == {
                baseline[0].chunk.chunk_id,
                baseline[1].chunk.chunk_id,
            }
            assert report["active_partition_baseline_protected"] == 2
            assert report["active_partition_candidates_admitted"] == 6
            assert report["active_partition_candidates_replaced"] == 6
            assert report["active_partition_candidates_truncated"] == 2

    def test_venue_scan_separates_episode_aligned_primary_from_dated_recap(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-recap",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            venues = (
                "Science Museum",
                "Museum of Contemporary Art",
                "Metropolitan Museum of Art",
                "Museum of History",
                "Modern Art Museum",
                "Natural History Museum",
            )
            primary_ids = set()
            for index, venue in enumerate(venues, start=1):
                source_id = f"history::answer-{index}"
                small.ingest(
                    "system",
                    f"[{source_id} took place at 2023/02/{index + 10:02d}]",
                    source_id=source_id,
                )
                _, chunks = small.ingest(
                    "user",
                    f"I visited the {venue} today.",
                    source_id=source_id,
                )
                primary_ids.add(chunks[0].chunk_id)
            recap_source = "history::dated-recap"
            small.ingest(
                "system",
                f"[{recap_source} took place at 2023/02/20]",
                source_id=recap_source,
            )
            _, recap_chunks = small.ingest(
                "user",
                "I'm planning to visit the Modern Art Gallery again soon. "
                "I participated in a guided tour there on February 17th.",
                source_id=recap_source,
            )
            future_source = "history::future"
            small.ingest(
                "system",
                f"[{future_source} took place at 2023/03/20]",
                source_id=future_source,
            )
            small.ingest(
                "user",
                "I visited the Maritime Museum today.",
                source_id=future_source,
            )
            assistant_source = "history::assistant"
            small.ingest(
                "system",
                f"[{assistant_source} took place at 2023/02/21]",
                source_id=assistant_source,
            )
            small.ingest(
                "assistant",
                "I visited the Aviation Museum today.",
                source_id=assistant_source,
            )
            query = (
                "[Question asked at 2023/03/10] "
                "Put the six museums I visited in order from earliest to latest."
            )
            source_ids = small.retriever.source_ids_in_partitions(["history"])

            candidates, report = small._scan_active_partition_frontier(
                query,
                small._embedder.embed_query(query),
                ["history"],
                source_ids,
                separator="::",
            )

            assert {result.chunk.chunk_id for result in candidates[:6]} == primary_ids
            assert candidates[-1].chunk.chunk_id == recap_chunks[0].chunk_id
            assert candidates[-1].route == "active_partition_alternative"
            assert candidates[-1].turn is not None
            assert candidates[-1].turn.source_id == recap_source
            assert report["active_partition_structural_hypotheses"] == 6
            assert report["active_partition_alternative_hypotheses"] == 1
            assert report["active_partition_recap_conflict_rows"] == 1
            assert report["active_partition_time_rejected_rows"] == 1
            assert report["active_partition_role_rejected_rows"] == 1
            assert report["active_partition_structural_overflow"] == 0
            assert report["active_partition_semantically_complete"] is True

    def test_venue_scan_retains_two_distinct_venues_from_one_source(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-two-venues-one-source",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            source_id = "history::combined-visit"
            small.ingest(
                "system",
                f"[{source_id} took place at 2023/02/15]",
                source_id=source_id,
            )
            _, science = small.ingest(
                "user",
                "I visited the Science Museum today.",
                source_id=source_id,
            )
            _, history = small.ingest(
                "user",
                "I visited the Museum of History today.",
                source_id=source_id,
            )
            query = "Put the two museums I visited in chronological order."

            candidates, report = small._scan_active_partition_frontier(
                query,
                small._embedder.embed_query(query),
                ["history"],
                [source_id],
                separator="::",
            )

            assert {result.chunk.chunk_id for result in candidates} == {
                science[0].chunk_id,
                history[0].chunk_id,
            }
            assert all(
                result.route == "active_partition_structural"
                for result in candidates
            )
            assert all(
                result.turn is not None and result.turn.source_id == source_id
                for result in candidates
            )
            assert report["active_partition_sources_total"] == 1
            assert report["active_partition_structural_rows"] == 2
            assert report["active_partition_structural_hypotheses"] == 2
            assert report["active_partition_semantically_complete"] is True

    def test_venue_scan_deduplicates_identity_across_sources_not_within_source(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-cross-source-identity",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            expected_ids = set()
            for index, (suffix, text) in enumerate(
                (
                    ("science-first", "I visited the Science Museum today."),
                    ("science-repeat", "I visited the Science Museum today."),
                    ("history", "I visited the Museum of History today."),
                    (
                        "science-ambiguous",
                        "I visited the Science Museum during a family trip.",
                    ),
                ),
                start=1,
            ):
                source_id = f"history::{suffix}"
                small.ingest(
                    "system",
                    f"[{source_id} took place at 2023/02/{index:02d}]",
                    source_id=source_id,
                )
                _, chunks = small.ingest(
                    "user",
                    text,
                    source_id=source_id,
                )
                if suffix in {"science-first", "history"}:
                    expected_ids.add(chunks[0].chunk_id)
            query = "Put the two museums I visited in chronological order."
            source_ids = small.retriever.source_ids_in_partitions(["history"])

            candidates, report = small._scan_active_partition_frontier(
                query,
                small._embedder.embed_query(query),
                ["history"],
                source_ids,
                separator="::",
            )

            assert {result.chunk.chunk_id for result in candidates} == expected_ids
            assert report["active_partition_structural_rows"] == 3
            assert report["active_partition_structural_hypotheses"] == 2
            assert report["active_partition_ambiguous_structural_rows"] == 1
            assert report["active_partition_alternative_hypotheses"] == 0
            assert report["active_partition_semantically_complete"] is True

    def test_venue_scan_retains_unaligned_occurrence_as_fail_open_alternative(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-ambiguous-venue",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            source_id = "history::ambiguous-visit"
            small.ingest(
                "system",
                f"[{source_id} took place at 2023/02/15]",
                source_id=source_id,
            )
            _, chunks = small.ingest(
                "user",
                "I visited the Aviation Museum during a memorable family trip.",
                source_id=source_id,
            )
            query = "Name one museum I visited."

            candidates, report = small._scan_active_partition_frontier(
                query,
                small._embedder.embed_query(query),
                ["history"],
                [source_id],
                separator="::",
            )

            assert [result.chunk.chunk_id for result in candidates] == [
                chunks[0].chunk_id
            ]
            assert candidates[0].route == "active_partition_alternative"
            assert report["active_partition_structural_hypotheses"] == 0
            assert report["active_partition_alternative_hypotheses"] == 1
            assert report["active_partition_ambiguous_structural_rows"] == 1
            assert report["active_partition_semantically_complete"] is False

    def test_seven_aligned_venue_occurrences_report_fixed_six_overflow(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-overflow",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            venues = (
                "Science Museum",
                "Museum of Contemporary Art",
                "Metropolitan Museum of Art",
                "Museum of History",
                "Modern Art Museum",
                "Natural History Museum",
                "Maritime Museum",
            )
            for index, venue in enumerate(venues, start=1):
                source_id = f"history::visit-{index}"
                small.ingest(
                    "system",
                    f"[{source_id} took place at 2023/02/{index + 10:02d}]",
                    source_id=source_id,
                )
                small.ingest(
                    "user",
                    f"I visited the {venue} today.",
                    source_id=source_id,
                )
            query = (
                "[Question asked at 2023/03/10] "
                "Put the six museums I visited in order from earliest to latest."
            )
            source_ids = small.retriever.source_ids_in_partitions(["history"])

            candidates, report = small._scan_active_partition_frontier(
                query,
                small._embedder.embed_query(query),
                ["history"],
                source_ids,
                separator="::",
            )

            assert len(candidates) == 7
            assert report["active_partition_structural_hypotheses"] == 7
            assert report["active_partition_structural_overflow"] == 1
            assert report["active_partition_semantically_complete"] is False

    def test_performance_scan_retains_distinct_keyed_events_in_one_source(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-multi-event-source",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            source_id = "music::two-events"
            small.ingest(
                "system",
                f"[{source_id} took place at 2023/04/01]",
                source_id=source_id,
            )
            _, concert = small.ingest(
                "user",
                "I attended the Alpha concert at Harbor Hall.",
                source_id=source_id,
            )
            _, festival = small.ingest(
                "user",
                "I attended the Beta music festival at River Park.",
                source_id=source_id,
            )
            query = (
                "[Question asked at 2023/04/22] List all concerts and musical "
                "events I attended in the past two months in chronological order."
            )

            candidates, report = small._scan_active_partition_frontier(
                query,
                small._embedder.embed_query(query),
                ["music"],
                [source_id],
                separator="::",
            )

            assert {result.chunk.chunk_id for result in candidates} == {
                concert[0].chunk_id,
                festival[0].chunk_id,
            }
            assert all(
                result.route == "active_partition_structural"
                for result in candidates
            )
            assert report["active_partition_structural_rows"] == 2
            assert report["active_partition_structural_hypotheses"] == 2
            assert report["active_partition_performance_multirow_sources"] == 1
            assert report["active_partition_semantically_complete"] is True

    def test_performance_scan_contracts_same_and_cross_source_keyed_recaps(
        self, tmp_path
    ):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-performance-recaps",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            brooklyn_source = "music::brooklyn"
            queen_source = "music::queen"
            small.ingest(
                "system",
                f"[{brooklyn_source} took place at 2023/04/01]",
                source_id=brooklyn_source,
            )
            _, brooklyn = small.ingest(
                "user",
                "I just got back from a music festival in Brooklyn with friends.",
                source_id=brooklyn_source,
            )
            _, same_source_recap = small.ingest(
                "user",
                "I attended the music festival in Brooklyn featuring Glass "
                "Animals and other indie bands.",
                source_id=brooklyn_source,
            )
            small.ingest(
                "system",
                f"[{queen_source} took place at 2023/04/15]",
                source_id=queen_source,
            )
            _, queen = small.ingest(
                "user",
                "I just saw Queen live with Adam Lambert at the Prudential "
                "Center in Newark.",
                source_id=queen_source,
            )
            _, cross_source_recap = small.ingest(
                "user",
                "I recently attended a music festival in Brooklyn that "
                "featured my favorite indie bands.",
                source_id=queen_source,
            )
            query = (
                "[Question asked at 2023/04/22] List all concerts and musical "
                "events I attended in the past two months in chronological order."
            )

            candidates, report = small._scan_active_partition_frontier(
                query,
                small._embedder.embed_query(query),
                ["music"],
                [brooklyn_source, queen_source],
                separator="::",
            )

            assert {result.chunk.chunk_id for result in candidates} == {
                brooklyn[0].chunk_id,
                queen[0].chunk_id,
            }
            assert same_source_recap[0].chunk_id not in {
                result.chunk.chunk_id for result in candidates
            }
            assert cross_source_recap[0].chunk_id not in {
                result.chunk.chunk_id for result in candidates
            }
            assert report["active_partition_structural_rows"] == 4
            assert report["active_partition_structural_hypotheses"] == 2
            assert report["active_partition_alternative_hypotheses"] == 0
            assert report["active_partition_performance_multirow_sources"] == 2
            assert report["active_partition_semantically_complete"] is True

    def test_performance_scan_keeps_keyless_direct_row_fail_open(self, tmp_path):
        with MemoryCondenser(
            data_dir=tmp_path / "typed-partition-performance-abstention",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as small:
            source_id = "music::ambiguous"
            small.ingest(
                "system",
                f"[{source_id} took place at 2023/04/01]",
                source_id=source_id,
            )
            _, ambiguous = small.ingest(
                "user",
                "I attended a concert yesterday.",
                source_id=source_id,
            )
            small.ingest(
                "user",
                "I plan to attend a concert at Future Hall next month.",
                source_id=source_id,
            )
            small.ingest(
                "user",
                "I watched a concert livestream on YouTube from home.",
                source_id=source_id,
            )
            query = (
                "[Question asked at 2023/04/22] List all concerts and musical "
                "events I attended in the past two months in chronological order."
            )

            candidates, report = small._scan_active_partition_frontier(
                query,
                small._embedder.embed_query(query),
                ["music"],
                [source_id],
                separator="::",
            )

            assert [result.chunk.chunk_id for result in candidates] == [
                ambiguous[0].chunk_id
            ]
            assert candidates[0].route == "active_partition_alternative"
            assert report["active_partition_structural_rows"] == 0
            assert report["active_partition_structural_hypotheses"] == 0
            assert report["active_partition_alternative_hypotheses"] == 1
            assert report["active_partition_ambiguous_structural_rows"] == 1
            assert report["active_partition_semantically_complete"] is False

    def test_multi_fact_qwen_uses_protected_scalar_then_broad_sources(self, mc):
        hydrated = []
        for index in range(8):
            _, chunks = mc.ingest(
                "user",
                f"I visited museum number {index}.",
                source_id=f"history::event-{index}",
            )
            result = mc.retriever.hydrate_chunk(
                chunks[0].chunk_id,
                score=1.0 - index / 20.0,
                route="hybrid",
            )
            assert result is not None
            hydrated.append(result)
        mc.search_hybrid_from_embedding = lambda *args, **kwargs: hydrated

        def local_search(*args, **kwargs):
            selected = set(args[2])
            return [
                result
                for result in hydrated
                if result.turn.source_id in selected
            ][: kwargs["k"]]

        mc.retriever.hybrid_query_sources = local_search
        reranker = FakeSourceReranker()
        mc.set_source_candidate_reranker(reranker)

        mc.search_hybrid_graph(
            "What is the order of all museums I visited?",
            k=1,
            neighbor_radius=0,
            neighbor_slots=0,
            source_slots=4,
            source_candidate_pool=8,
            source_activation_k=8,
            multi_fact_source_diversity=True,
            source_local_search=True,
            use_source_reranker=True,
        )

        assert reranker.unique_sources is True
        assert reranker.candidate_sources == [
            "history::event-0",
            "history::event-1",
            "history::event-2",
            "history::event-3",
            "history::event-4",
            "history::event-5",
        ]
        assert mc.last_source_diversity_report["protected_activation_k"] == 5
        assert mc.last_source_diversity_report["protected_candidates"] == 4
        assert (
            mc.last_source_diversity_report["attention_exploration_candidates"]
            == 2
        )

    def test_role_prior_is_applied_before_hierarchical_partition_routing(self, mc):
        _, assistant_chunks = mc.ingest(
            "assistant",
            "I might prefer the echoed answer.",
            source_id="echo-history::session-main",
        )
        _, user_chunks = mc.ingest(
            "user",
            "My actual preference is the user-authored answer.",
            source_id="user-history::session-main",
        )
        assistant = mc.retriever.hydrate_chunk(
            assistant_chunks[0].chunk_id, score=1.0, route="hybrid"
        )
        user = mc.retriever.hydrate_chunk(
            user_chunks[0].chunk_id, score=0.7, route="hybrid"
        )
        assert assistant is not None
        assert user is not None
        mc.search_hybrid_from_embedding = lambda *args, **kwargs: [assistant, user]

        results = mc.search_hybrid_graph(
            "What is my actual preference?",
            k=1,
            neighbor_radius=0,
            neighbor_slots=0,
            source_slots=1,
            source_candidate_pool=2,
            source_activation_k=1,
            source_partition_routing=True,
            source_partition_slots=1,
            role_aware_retrieval=True,
            role_user_weight=2.0,
            role_assistant_weight=0.5,
        )

        assert results
        assert mc.last_partition_routing_report["selected_partitions"] == [
            "user-history"
        ]
        assert all(
            result.turn.source_id.startswith("user-history::")
            for result in results
            if result.turn is not None
        )

    def test_attention_feedback_runs_a_bounded_second_source_retrieval(self, mc):
        for text in [
            "Alpha project overview.",
            "Alpha joined three weeks ago.",
            "Alpha discussed another book.",
            "Alpha attended the meetup last week.",
            "Alpha planned a future event.",
        ]:
            mc.ingest("user", text, source_id="session-alpha")
        controller = FakeAttentionFeedback()
        mc.set_source_candidate_reranker(controller)

        results = mc.search_hybrid_graph(
            "How long before the meetup?",
            k=1,
            neighbor_radius=0,
            neighbor_slots=0,
            source_slots=2,
            source_candidate_pool=5,
            source_activation_k=1,
            source_local_search=True,
            use_attention_feedback=True,
            feedback_slots=1,
            feedback_seed_slots=1,
            feedback_evidence_tokens=12,
            feedback_query_tokens=64,
        )

        assert len(results) == 3
        assert sum(
            result.route == "qwen_activation_feedback" for result in results
        ) == 1
        assert mc.last_source_rerank_report["feedback_rounds"] == 1
        assert mc.last_source_rerank_report["feedback_candidates_added"] == 1
        assert mc.last_source_rerank_report["feedback_activation_candidates"] == 1
        assert mc.last_source_rerank_report["feedback_query_tokens"] <= 64
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

    def test_ingest_many_preserves_authoritative_source_times(self, tmp_path):
        source_time = datetime(2024, 5, 1, 12, 30, tzinfo=timezone.utc)
        with MemoryCondenser(
            data_dir=tmp_path / "dated-batch",
            embedder=FakeEmbedder(),
            auto_extract=False,
        ) as condenser:
            rows = condenser.ingest_many(
                [("user", "The deployment completed.", "session-1", source_time)]
            )

            assert rows[0][0].created_at == source_time
            stored = condenser.transcript.get_turn(rows[0][0].turn_id)
            assert stored is not None
            assert stored.created_at == source_time

    def test_explicit_batch_turn_ids_make_fresh_chunk_ids_replayable(self, tmp_path):
        source_time = datetime(2024, 5, 1, 12, 30, tzinfo=timezone.utc)
        records = [
            (
                "user",
                "The deployment completed with the amber badge.",
                "session-1",
                source_time,
                "stable-turn-0001",
            )
        ]

        def ingest_at(path, rows):
            with MemoryCondenser(
                data_dir=path,
                embedder=FakeEmbedder(),
                auto_extract=False,
                chunker_min_tokens=1,
                chunker_max_tokens=20,
            ) as condenser:
                return condenser.ingest_many(rows)

        first = ingest_at(tmp_path / "stable-a", records)
        replay = ingest_at(tmp_path / "stable-b", records)
        changed = ingest_at(
            tmp_path / "stable-c",
            [
                (
                    "user",
                    "The deployment completed with the green badge.",
                    "session-1",
                    source_time,
                    "stable-turn-0001",
                )
            ],
        )

        assert first[0][0].turn_id == replay[0][0].turn_id == "stable-turn-0001"
        assert [chunk.chunk_id for chunk in first[0][1]] == [
            chunk.chunk_id for chunk in replay[0][1]
        ]
        assert [chunk.chunk_id for chunk in first[0][1]] != [
            chunk.chunk_id for chunk in changed[0][1]
        ]

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

    def test_context_selector_sees_complete_bounded_expansion_set(self, mc):
        hydrated = []
        for index in range(3):
            _, chunks = mc.ingest(
                "user",
                f"I visited museum {index}.",
                source_id=f"event-{index}",
            )
            result = mc.retriever.hydrate_chunk(
                chunks[0].chunk_id,
                score=1.0 - index / 10.0,
                route="hybrid",
            )
            assert result is not None
            hydrated.append(result)

        class ReversingSelector:
            def __init__(self):
                self.seen = []
                self.last_report = None

            def select(
                self,
                query,
                candidates,
                *,
                max_results=None,
                source_timestamps=None,
            ):
                self.seen = list(candidates)
                self.last_report = SimpleNamespace(
                    model_dump=lambda: {
                        "input_candidates": len(candidates),
                        "output_candidates": len(candidates),
                        "retained_transformer_state_bytes": 0,
                    }
                )
                return list(reversed(candidates))

        selector = ReversingSelector()
        mc.set_context_candidate_selector(selector)

        packed = mc.build_context(
            "List all museums I visited",
            recent_turns=0,
            k_memories=0,
            k_expansions=0,
            use_consolidation=False,
            learn_consolidation=False,
            expansion_results=hydrated,
        )

        assert selector.seen == hydrated
        assert packed.expansion_chunk_ids == [
            result.chunk.chunk_id for result in reversed(hydrated)
        ]
        assert mc.last_coverage_selection_report == {
            "input_candidates": 3,
            "output_candidates": 3,
            "retained_transformer_state_bytes": 0,
        }
        trace = mc.last_coverage_candidate_trace
        assert [row["original_rank"] for row in trace] == [1, 2, 3]
        assert [row["post_selector_rank"] for row in trace] == [3, 2, 1]
        assert [row["packed_rank"] for row in trace] == [3, 2, 1]
        assert all(row["cutoff_reason"] == "packed" for row in trace)
        assert all("text" not in row for row in trace)

    def test_metadata_only_activation_hydrates_real_companion_in_place(
        self, tmp_path
    ):
        budget = ContextBudget(
            expansion_tokens=200,
            source_metadata_expansions=True,
        )
        with MemoryCondenser(
            data_dir=tmp_path / "metadata-companion",
            embedder=FakeEmbedder(),
            budget=budget,
            auto_extract=False,
        ) as small:
            _, first_metadata = small.ingest(
                "system",
                "[session-a took place at 2024/01/02]",
                source_id="session-a",
            )
            _, duplicate_metadata = small.ingest(
                "system",
                "[session-a took place at 2024/01/02]",
                source_id="session-a",
            )
            _, content = small.ingest(
                "user",
                "I attended the cerulean concert at the Harbor Theater.",
                source_id="session-a",
            )
            activated = [
                small.retriever.hydrate_chunk(
                    first_metadata[0].chunk_id,
                    score=0.9,
                    route="hsc_contraction",
                ),
                small.retriever.hydrate_chunk(
                    duplicate_metadata[0].chunk_id,
                    score=0.8,
                    route="hsc_contraction",
                ),
            ]
            assert all(result is not None for result in activated)

            class RecordingSelector:
                last_report = None
                last_candidate_trace = []

                def select(self, _query, candidates, **_kwargs):
                    self.seen = list(candidates)
                    return list(candidates)

            selector = RecordingSelector()
            small.set_context_candidate_selector(selector)
            packed = small.build_context(
                "Which concerts did I attend?",
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=False,
                learn_consolidation=False,
                expansion_results=activated,
            )

            assert [result.chunk.chunk_id for result in selector.seen] == [
                content[0].chunk_id
            ]
            companion = selector.seen[0]
            assert companion.turn.source_id == "session-a"
            assert companion.route == "hsc_contraction"
            assert companion.anchor_chunk_id == first_metadata[0].chunk_id
            assert packed.expansion_chunk_ids == [content[0].chunk_id]
            assert "@ 2024/01/02" in packed.expansions[0]
            companion_trace = next(
                row
                for row in small.last_coverage_candidate_trace
                if row["chunk_id"] == content[0].chunk_id
            )
            assert companion_trace["anchor_chunk_id"] == first_metadata[0].chunk_id
            assert small.last_source_companion_report == {
                "requested_sources": ["session-a"],
                "hydrated_sources": ["session-a"],
                "refreshed_sources": [],
                "already_present_sources": [],
                "orphan_sources": [],
                "orphan_count": 0,
                "direct_date_retained": 0,
                "candidate_count_before": 2,
                "candidate_count_after": 2,
                "max_candidates_per_source": 1,
                "companion_candidate_count": 1,
                "selector_used": False,
                "selector_fallback_sources": [],
                "selector_fallback_reason": "",
                "semantic_selector_report": {},
                "selected_chunk_ids": {
                    "session-a": content[0].chunk_id,
                },
                "refresh_all_activated_sources": False,
                "choice_diagnostics": [
                    {
                        "source_id": "session-a",
                        "candidate_count": 1,
                        "candidate_chunk_ids": [content[0].chunk_id],
                        "selected_chunk_id": content[0].chunk_id,
                        "selected_local_rank": 1,
                        "selected_by": "retrieval",
                    }
                ],
            }

    def test_metadata_companion_uses_optional_bounded_semantic_choice(
        self, tmp_path
    ):
        budget = ContextBudget(
            expansion_tokens=200,
            source_metadata_expansions=True,
        )
        with MemoryCondenser(
            data_dir=tmp_path / "semantic-metadata-companion",
            embedder=FakeEmbedder(),
            budget=budget,
            auto_extract=False,
        ) as small:
            _, metadata = small.ingest(
                "system",
                "[session-a took place at 2024/01/02]",
                source_id="session-a",
            )
            _, generic = small.ingest(
                "user",
                "I framed the concert ticket after today's show.",
                source_id="session-a",
            )
            _, specific = small.ingest(
                "user",
                "I attended the Billie Eilish concert at Wells Fargo Center.",
                source_id="session-a",
            )
            activated = small.retriever.hydrate_chunk(
                metadata[0].chunk_id,
                score=0.9,
                route="hsc_contraction",
            )
            assert activated is not None

            class SemanticSelector:
                last_report = None
                last_candidate_trace = []

                def select_source_companions(
                    self, query, candidates_by_source
                ):
                    self.query = query
                    self.groups = {
                        source_id: list(candidates)
                        for source_id, candidates in candidates_by_source.items()
                    }
                    selected = next(
                        result
                        for result in self.groups["session-a"]
                        if result.chunk.chunk_id == specific[0].chunk_id
                    )
                    return {"session-a": selected}

                def select(self, _query, candidates, **_kwargs):
                    self.seen = list(candidates)
                    return list(candidates)

            selector = SemanticSelector()
            small.set_context_candidate_selector(selector)
            packed = small.build_context(
                "Which concerts did I attend?",
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=False,
                learn_consolidation=False,
                expansion_results=[activated],
            )

            assert selector.query == "Which concerts did I attend?"
            assert {
                result.chunk.chunk_id for result in selector.groups["session-a"]
            } == {generic[0].chunk_id, specific[0].chunk_id}
            assert [result.chunk.chunk_id for result in selector.seen] == [
                specific[0].chunk_id
            ]
            selected = selector.seen[0]
            assert selected.route == "hsc_contraction"
            assert selected.anchor_chunk_id == metadata[0].chunk_id
            assert packed.expansion_chunk_ids == [specific[0].chunk_id]
            report = small.last_source_companion_report
            assert report["candidate_count_before"] == 1
            assert report["candidate_count_after"] == 1
            assert report["max_candidates_per_source"] == 4
            assert report["companion_candidate_count"] == 2
            assert report["selector_used"] is True
            assert report["selector_fallback_sources"] == []
            assert report["selector_fallback_reason"] == ""
            assert report["choice_diagnostics"] == [
                {
                    "source_id": "session-a",
                    "candidate_count": 2,
                    "candidate_chunk_ids": [
                        result.chunk.chunk_id
                        for result in selector.groups["session-a"]
                    ],
                    "selected_chunk_id": specific[0].chunk_id,
                    "selected_local_rank": next(
                        rank
                        for rank, result in enumerate(
                            selector.groups["session-a"], start=1
                        )
                        if result.chunk.chunk_id == specific[0].chunk_id
                    ),
                    "selected_by": "semantic",
                }
            ]

    def test_metadata_companion_selector_failure_falls_back_to_local_rank_one(
        self, tmp_path
    ):
        budget = ContextBudget(
            expansion_tokens=200,
            source_metadata_expansions=True,
        )
        with MemoryCondenser(
            data_dir=tmp_path / "metadata-companion-fallback",
            embedder=FakeEmbedder(),
            budget=budget,
            auto_extract=False,
        ) as small:
            _, metadata = small.ingest(
                "system",
                "[session-a took place at 2024/01/02]",
                source_id="session-a",
            )
            small.ingest(
                "user",
                "The first concert was at Harbor Theater.",
                source_id="session-a",
            )
            small.ingest(
                "user",
                "The second concert was in Central Park.",
                source_id="session-a",
            )
            activated = small.retriever.hydrate_chunk(
                metadata[0].chunk_id,
                score=0.9,
                route="source_tfisf",
            )
            assert activated is not None

            class FailingSelector:
                last_report = None
                last_candidate_trace = []

                def select_source_companions(
                    self, _query, candidates_by_source
                ):
                    self.groups = {
                        source_id: list(candidates)
                        for source_id, candidates in candidates_by_source.items()
                    }
                    raise RuntimeError("synthetic backend failure")

                def select(self, _query, candidates, **_kwargs):
                    self.seen = list(candidates)
                    return list(candidates)

            selector = FailingSelector()
            small.set_context_candidate_selector(selector)
            packed = small.build_context(
                "Which concerts did I attend?",
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=False,
                learn_consolidation=False,
                expansion_results=[activated],
            )

            fallback = selector.groups["session-a"][0]
            assert packed.expansion_chunk_ids == [fallback.chunk.chunk_id]
            report = small.last_source_companion_report
            assert report["selector_used"] is True
            assert report["selector_fallback_sources"] == ["session-a"]
            assert report["selector_fallback_reason"] == "RuntimeError"
            assert report["choice_diagnostics"][0]["selected_by"] == "retrieval"

    def test_complete_set_refresh_replaces_existing_assistant_rows_for_two_sources(
        self, tmp_path
    ):
        budget = ContextBudget(
            expansion_tokens=400,
            source_metadata_expansions=True,
        )
        with MemoryCondenser(
            data_dir=tmp_path / "complete-set-source-refresh",
            embedder=FakeEmbedder(),
            budget=budget,
            auto_extract=False,
        ) as small:
            sources = {
                "museum-session-2": (
                    "Natural History Museum",
                    "You later mentioned art blogs and online exhibits.",
                ),
                "museum-session-5": (
                    "Metropolitan Museum of Art",
                    "You later recapped adhesives and a permanent collection.",
                ),
            }
            direct_chunks = {}
            activated = []
            for index, (source_id, (venue, recap)) in enumerate(
                sources.items(), start=1
            ):
                small.ingest(
                    "system",
                    f"[{source_id} took place at 2024/0{index}/02]",
                    source_id=source_id,
                )
                _, direct = small.ingest(
                    "user",
                    f"I visited the {venue} during my trip.",
                    source_id=source_id,
                )
                # Query-overlap filler makes the answer-bearing first-person
                # row depend on the canonical shortlist, not a gold ID.
                for filler_index in range(5):
                    small.ingest(
                        "user",
                        f"I organized museum notes section {filler_index}.",
                        source_id=source_id,
                    )
                _, assistant = small.ingest(
                    "assistant",
                    recap,
                    source_id=source_id,
                )
                direct_chunks[source_id] = direct[0]
                routed = small.retriever.hydrate_chunk(
                    assistant[0].chunk_id,
                    score=0.9 - index * 0.1,
                    route=(
                        "hsc_contraction"
                        if index == 1
                        else "live_consolidation"
                    ),
                    anchor_chunk_id=f"route-anchor-{index}",
                )
                assert routed is not None
                activated.append(
                    routed.model_copy(
                        update={
                            "memory_source_id": source_id,
                            "source_heat": 0.5,
                            "source_token_budget": 80,
                        }
                    )
                )

            class InspectedCompanionSelector:
                last_report = None
                last_candidate_trace = []
                strict = False

                def select_source_companions(
                    self, _query, candidates_by_source
                ):
                    self.groups = {
                        source_id: list(candidates)
                        for source_id, candidates in candidates_by_source.items()
                    }
                    # This synthetic neural chooser always prefers the last
                    # supplied row.  The query-head constraint must therefore
                    # expose only the earliest direct canonical venue row,
                    # never the later generic recap it would otherwise favor.
                    selected = {
                        source_id: candidates[-1]
                        for source_id, candidates in self.groups.items()
                    }
                    # Both are inspected relative winners. One deliberately
                    # sits below 0.5: these scores are uncalibrated and must
                    # not be treated as an absolute acceptance threshold.
                    scores = {
                        "museum-session-2": 0.44,
                        "museum-session-5": 0.37,
                    }
                    self.last_source_companion_report = {
                        "input_sources": len(self.groups),
                        "input_candidates": sum(
                            len(candidates)
                            for candidates in self.groups.values()
                        ),
                        "inspected_candidates": sum(
                            len(candidates)
                            for candidates in self.groups.values()
                        ),
                        "selected_chunk_ids": {
                            source_id: result.chunk.chunk_id
                            for source_id, result in selected.items()
                        },
                        "selected_membership_scores": scores,
                        "retained_transformer_state_bytes": 0,
                        "fallback_reason": "",
                    }
                    return selected

                def select(self, _query, candidates, **_kwargs):
                    self.seen = list(candidates)
                    return list(candidates)

            selector = InspectedCompanionSelector()
            small.set_context_candidate_selector(selector)
            packed = small.build_context(
                "Put the six museums I visited in chronological order.",
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=False,
                learn_consolidation=False,
                expansion_results=activated,
            )

            assert len(selector.seen) == len(activated) == 2
            by_source = {
                result.turn.source_id: result for result in selector.seen
            }
            assert set(by_source) == set(sources)
            for index, (source_id, (venue, _recap)) in enumerate(
                sources.items(), start=1
            ):
                selected = by_source[source_id]
                assert selected.turn.role == "user"
                assert "I visited" in selected.chunk.text
                assert venue in selected.chunk.text
                assert selected.chunk.chunk_id == direct_chunks[source_id].chunk_id
                assert selected.route == (
                    "hsc_contraction"
                    if index == 1
                    else "live_consolidation"
                )
                assert selected.anchor_chunk_id == f"route-anchor-{index}"
                assert selected.memory_source_id == source_id
                assert selected.source_heat == 0.5
                assert selected.source_token_budget == 80
                assert selector.groups[source_id][0].chunk.chunk_id == (
                    direct_chunks[source_id].chunk_id
                )
                assert len(selector.groups[source_id]) == 1

            assert set(packed.expansion_chunk_ids) == {
                chunk.chunk_id for chunk in direct_chunks.values()
            }
            report = small.last_source_companion_report
            assert report["refresh_all_activated_sources"] is True
            assert report["requested_sources"] == list(sources)
            assert report["refreshed_sources"] == list(sources)
            assert report["candidate_count_before"] == 2
            assert report["candidate_count_after"] == 2
            assert report["selector_fallback_sources"] == []
            assert report["selected_chunk_ids"] == {
                source_id: chunk.chunk_id
                for source_id, chunk in direct_chunks.items()
            }

    def test_complete_performance_set_refresh_uses_first_direct_event_per_source(
        self, tmp_path
    ):
        budget = ContextBudget(
            expansion_tokens=500,
            source_metadata_expansions=True,
        )
        with MemoryCondenser(
            data_dir=tmp_path / "complete-performance-source-refresh",
            embedder=FakeEmbedder(),
            budget=budget,
            auto_extract=False,
        ) as small:
            rows = {
                "concert-session-queen": (
                    "I'm looking for rock music playlists on streaming services. "
                    "I've been listening to Queen lately and actually just saw "
                    "them live with Adam Lambert at the Prudential Center in "
                    "Newark with my parents.",
                    "You asked for classic-rock playlists after seeing Queen.",
                ),
                "concert-session-radiohead": (
                    "I saw Radiohead at Madison Square Garden with my sister.",
                    "You wanted more music recommendations for your commute.",
                ),
            }
            direct_chunks = {}
            activated = []
            for index, (source_id, (direct_text, recap)) in enumerate(
                rows.items(), start=1
            ):
                small.ingest(
                    "system",
                    f"[{source_id} took place at 2024/0{index}/02]",
                    source_id=source_id,
                )
                small.ingest(
                    "user",
                    "I'm planning to attend an upcoming concert next month.",
                    source_id=source_id,
                )
                small.ingest(
                    "user",
                    "I watched the band live on YouTube from my apartment.",
                    source_id=source_id,
                )
                small.ingest(
                    "user",
                    "I was at home organizing a concert playlist after "
                    "streaming the album.",
                    source_id=source_id,
                )
                small.ingest(
                    "assistant",
                    "You saw the orchestra live at Harbor Theater last week.",
                    source_id=source_id,
                )
                _, direct = small.ingest(
                    "user",
                    direct_text,
                    source_id=source_id,
                )
                # A second genuine occurrence must not replace the primary
                # source event merely because it is shorter or ranks better.
                small.ingest(
                    "user",
                    "I attended a music festival in Brooklyn last weekend.",
                    source_id=source_id,
                )
                small.ingest(
                    "user",
                    "I later recapped that I attended the music festival in "
                    "Brooklyn with several indie bands.",
                    source_id=source_id,
                )
                _, assistant = small.ingest(
                    "assistant",
                    recap,
                    source_id=source_id,
                )
                direct_chunks[source_id] = direct[0]
                routed = small.retriever.hydrate_chunk(
                    assistant[0].chunk_id,
                    score=0.9 - index * 0.1,
                    route=(
                        "hsc_contraction"
                        if index == 1
                        else "live_consolidation"
                    ),
                    anchor_chunk_id=f"performance-anchor-{index}",
                )
                assert routed is not None
                activated.append(
                    routed.model_copy(
                        update={
                            "memory_source_id": source_id,
                            "source_heat": 0.6,
                            "source_token_budget": 96,
                        }
                    )
                )

            class InspectedCompanionSelector:
                last_report = None
                last_candidate_trace = []
                strict = False

                def select_source_companions(
                    self, _query, candidates_by_source
                ):
                    self.groups = {
                        source_id: list(candidates)
                        for source_id, candidates in candidates_by_source.items()
                    }
                    selected = {
                        source_id: candidates[-1]
                        for source_id, candidates in self.groups.items()
                    }
                    self.last_source_companion_report = {
                        "input_sources": len(self.groups),
                        "input_candidates": sum(
                            len(candidates)
                            for candidates in self.groups.values()
                        ),
                        "inspected_candidates": sum(
                            len(candidates)
                            for candidates in self.groups.values()
                        ),
                        "selected_chunk_ids": {
                            source_id: result.chunk.chunk_id
                            for source_id, result in selected.items()
                        },
                        "selected_membership_scores": {
                            source_id: 0.3 for source_id in selected
                        },
                        "retained_transformer_state_bytes": 0,
                        "fallback_reason": "",
                    }
                    return selected

                def select(self, _query, candidates, **_kwargs):
                    self.seen = list(candidates)
                    return list(candidates)

            selector = InspectedCompanionSelector()
            small.set_context_candidate_selector(selector)
            packed = small.build_context(
                "[Question asked at 2024/04/22] List all concerts and musical "
                "events I attended in the past two months in chronological order.",
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=False,
                learn_consolidation=False,
                expansion_results=activated,
            )

            assert len(selector.seen) == len(activated) == 2
            by_source = {
                result.turn.source_id: result for result in selector.seen
            }
            assert set(by_source) == set(rows)
            for index, source_id in enumerate(rows, start=1):
                selected = by_source[source_id]
                assert selected.turn.role == "user"
                assert selected.chunk.chunk_id == direct_chunks[source_id].chunk_id
                assert "upcoming concert" not in selected.chunk.text
                assert "YouTube" not in selected.chunk.text
                assert "Brooklyn" not in selected.chunk.text
                assert len(selector.groups[source_id]) == 1
                assert selected.route == (
                    "hsc_contraction"
                    if index == 1
                    else "live_consolidation"
                )
                assert selected.anchor_chunk_id == f"performance-anchor-{index}"
                assert selected.memory_source_id == source_id
                assert selected.source_heat == 0.6
                assert selected.source_token_budget == 96

            assert set(packed.expansion_chunk_ids) == {
                chunk.chunk_id for chunk in direct_chunks.values()
            }
            report = small.last_source_companion_report
            assert report["refresh_all_activated_sources"] is True
            assert report["requested_sources"] == list(rows)
            assert report["refreshed_sources"] == list(rows)
            assert report["candidate_count_before"] == 2
            assert report["candidate_count_after"] == 2
            assert report["selector_fallback_sources"] == []
            assert report["selected_chunk_ids"] == {
                source_id: chunk.chunk_id
                for source_id, chunk in direct_chunks.items()
            }

    def test_metadata_orphan_is_kept_only_for_direct_date_query(self, tmp_path):
        budget = ContextBudget(source_metadata_expansions=True)
        with MemoryCondenser(
            data_dir=tmp_path / "metadata-orphan",
            embedder=FakeEmbedder(),
            budget=budget,
            auto_extract=False,
        ) as small:
            timestamp = "[orphan took place at 2024/03/04]"
            _, chunks = small.ingest(
                "system",
                timestamp,
                source_id="orphan",
            )
            activated = small.retriever.hydrate_chunk(
                chunks[0].chunk_id,
                score=0.9,
                route="hybrid_source_local",
            )
            assert activated is not None

            ordinary = small.build_context(
                "Tell me about the museum visit.",
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=False,
                learn_consolidation=False,
                expansion_results=[activated],
            )
            assert ordinary.expansions == []
            assert small.last_source_companion_report["orphan_count"] == 1
            assert small.last_source_companion_report["direct_date_retained"] == 0

            direct_date = small.build_context(
                "When did the orphan source take place?",
                recent_turns=0,
                k_memories=0,
                k_expansions=0,
                use_consolidation=False,
                learn_consolidation=False,
                expansion_results=[activated],
            )
            assert timestamp in direct_date.expansions[0]
            assert small.last_source_companion_report["orphan_count"] == 1
            assert small.last_source_companion_report["direct_date_retained"] == 1

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
