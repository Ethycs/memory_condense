from __future__ import annotations

import re
import zlib
from dataclasses import replace

import numpy as np
import pytest

from memory_condense.associations.head_memory import (
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import DiscourseArtifact, identity_sha256
from memory_condense.domain.schemas import Chunk, RetrievalResult
from memory_condense.eval.diffuse_longmemeval import (
    LongMemEvalDiffuseQueryReceipt,
    measure_longmemeval_diffuse_packet,
    retrieve_longmemeval_diffuse_packet,
)
from memory_condense.search.episodes import (
    EpisodeBuilder,
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRetrievalPolicy,
    FixedIntervalBoundaryDetector,
)


class _DeterministicEmbedder:
    def __init__(self, dimension: int = 48) -> None:
        self._dimension = dimension

    @property
    def dim(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            vector[zlib.crc32(token.encode("utf-8")) % self._dimension] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]


class _NoSearchFacade:
    """Expose only the bridge contract, proving it cannot rerun retrieval."""

    def __init__(self, condenser: MemoryCondenser) -> None:
        self._condenser = condenser

    @property
    def discourse(self):
        return self._condenser.discourse

    def expand_discourse_episode_seeds(self, *args, **kwargs):
        return self._condenser.expand_discourse_episode_seeds(*args, **kwargs)

    def close_discourse_evidence(self, *args, **kwargs):
        return self._condenser.close_discourse_evidence(*args, **kwargs)

    def retrieve_discourse_episode_representatives(self, *args, **kwargs):
        return self._condenser.retrieve_discourse_episode_representatives(
            *args,
            **kwargs,
        )

    def pack_discourse_evidence(self, *args, **kwargs):
        return self._condenser.pack_discourse_evidence(*args, **kwargs)


class _SelectEpisodeLinker:
    max_candidates = 8

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id

    def inspect_nested(
        self,
        _query,
        _groups,
        *,
        beam_per_group,
        top_k,
        score_mode,
    ):
        assert beam_per_group == 2
        assert score_mode == "qk_ov"
        hit = MemoryLinkHit(
            episode_id=self.episode_id,
            qk_score=0.7,
            ov_transport=0.4,
            head_weights=(1.0,),
        )
        return NestedMemoryInspection(
            hits=(hit,)[:top_k],
            passes=1,
            max_workspace_candidates=1,
            max_workspace_tokens=64,
            total_candidate_inspections=1,
        )


def _artifact() -> DiscourseArtifact:
    return DiscourseArtifact.create(
        kind="longmemeval-provider-free-diffuse-test",
        implementation_sha256="d" * 64,
        policy={
            "episode_boundary": "fixed_interval_16",
            "linker": "rules-v1",
            "coverage": "all_current_chunks",
        },
        metadata={
            "boundary_policy_id": "fixed_interval_16",
            "scorer_id": "none",
        },
    )


def _condenser(path) -> MemoryCondenser:
    return MemoryCondenser(
        data_dir=path,
        embedder=_DeterministicEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
        persist_index_on_close=False,
    )


def _publish_fixture(condenser: MemoryCondenser):
    facts = (
        "Atlas retrieval objective target is 95 percent factual accuracy.",
        "Atlas retrieval currently reaches 82 percent judged accuracy.",
        "Atlas retrieval must keep the prompt below 8000 tokens.",
        "Atlas retrieval decision selected dense top-k only.",
        "Atlas retrieval failure: dense top-k missed diffuse dependencies.",
        "Atlas retrieval depends on exact source-span provenance.",
        "Atlas retrieval measured result showed source coverage reached 100 percent.",
        "Atlas retrieval instead revised the decision to hybrid evidence closure.",
        "Atlas retrieval unresolved open issue is calibrating semantic linking.",
    )
    ingested = [
        condenser.ingest("user", text, source_id="engineering-thread")
        for text in facts
    ]
    artifact = _artifact()
    chunks = tuple(row[1][0] for row in ingested)
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    condenser.build_and_publish_discourse_episodes(
        artifact,
        chunk_ids,
        builder=EpisodeBuilder(
            min_size=1,
            max_size=16,
            detector=FixedIntervalBoundaryDetector(interval=16),
        ),
        embeddings={chunk.chunk_id: chunk.embedding for chunk in chunks},
        representative_limit=1,
    )
    condenser.link_and_publish_discourse(artifact, chunk_ids=chunk_ids)
    condenser.discourse.finalize_artifact_coverage(
        artifact.artifact_id,
        coverage_kind="episode",
    )
    condenser.finalize_discourse_coverage(artifact.artifact_id)
    return artifact, ingested, chunks


def test_gold_blind_bridge_reuses_anchors_and_packs_exact_final_prompt(tmp_path):
    with _condenser(tmp_path / "diffuse") as condenser:
        artifact, ingested, chunks = _publish_fixture(condenser)
        anchor = RetrievalResult(
            chunk=chunks[4],
            turn=ingested[4][0],
            score=0.91,
            dense_score=0.87,
            lexical_score=0.44,
            route="frozen_hybrid_anchor",
        )
        query = "How should we improve the Atlas retrieval system?"

        # This facade has no search method. The bridge must consume the exact
        # already-ranked row rather than issue a gold-influenced second query.
        retrieval = retrieve_longmemeval_diffuse_packet(
            _NoSearchFacade(condenser),
            query=query,
            prompt_question=(
                "[Question asked at 2026/08/19 (Wed) 09:00]\n" + query
            ),
            anchors=(anchor,),
            artifact_id=artifact.artifact_id,
            episode_policy=EpisodeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                previous_episodes=1,
                next_episodes=1,
            ),
            max_context_tokens=4096,
            max_prompt_tokens=8000,
            responder_output_token_reserve=128,
        )

        assert retrieval.receipt.input_anchor_chunk_ids == (anchor.chunk.chunk_id,)
        assert retrieval.plan.expansion_receipt_sha256 == (
            retrieval.receipt.combined_expansion_sha256
        )
        assert retrieval.receipt.expansion_receipt_sha256 == (
            retrieval.expansion.receipt_sha256
        )
        assert retrieval.representative_expansion is None
        assert retrieval.packet.receipt.plan_sha256 == retrieval.plan.plan_sha256
        assert retrieval.packet.receipt.receipt_sha256 == (
            retrieval.receipt.packet_receipt_sha256
        )
        assert retrieval.receipt.prompt_messages_sha256 == (
            retrieval.packet.receipt.prompt_messages_sha256
        )
        assert retrieval.receipt.prompt_token_proxy == (
            count_chat_prompt_token_proxy(retrieval.provider_messages())
        )
        assert retrieval.receipt.prompt_token_proxy <= 8000
        assert retrieval.receipt.max_prompt_workspace_token_proxy == 8128
        assert retrieval.receipt.prompt_workspace_token_proxy == (
            retrieval.receipt.prompt_token_proxy + 128
        )
        assert retrieval.receipt.packet_retained_request_token_state_bytes == 0
        assert retrieval.receipt.store_retained_request_token_state_bytes == 0
        assert "2026" not in retrieval.plan.query_program.subject_terms
        assert retrieval.receipt.retrieval_query_sha256 == identity_sha256(
            {"query": query}
        )
        assert retrieval.receipt.prompt_question_sha256 == identity_sha256(
            {
                "prompt_question": (
                    "[Question asked at 2026/08/19 (Wed) 09:00]\n" + query
                )
            }
        )
        assert retrieval.packet.context == retrieval.messages[1]["content"].split(
            "[1] ", 1
        )[1].split("\n\nQuestion:", 1)[0]

        # Gold labels enter only after the packet and its receipt are frozen.
        metrics = measure_longmemeval_diffuse_packet(
            retrieval,
            question_id="longmemeval-q1",
            gold_answer="95 percent",
            evidence_source_ids=("engineering-thread",),
            hydrate_span=condenser.discourse.hydrate_span,
        )
        assert metrics.answer_present is True
        assert metrics.evidence_source_recall == 1.0
        assert metrics.any_evidence_source is True
        assert metrics.all_evidence_sources is True
        assert metrics.source_span_hash_valid is True
        assert metrics.hard_budget_compliant is True
        assert metrics.retrieval_receipt_sha256 == retrieval.receipt.receipt_sha256


def test_unannotated_anchor_fails_open_and_receipts_are_deterministic(tmp_path):
    with _condenser(tmp_path / "fail-open") as condenser:
        artifact, _ingested, _chunks = _publish_fixture(condenser)
        fallback_turn, fallback_chunks = condenser.ingest(
            "user",
            "The fallback answer is cobalt blue.",
            source_id="unannotated-source",
        )
        anchor = RetrievalResult(
            chunk=fallback_chunks[0],
            turn=fallback_turn,
            score=0.75,
            route="frozen_dense_anchor",
        )
        kwargs = {
            "query": "What is the fallback answer?",
            "anchors": (anchor,),
            "artifact_id": artifact.artifact_id,
            "episode_policy": EpisodeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                previous_episodes=0,
                next_episodes=0,
            ),
            "max_context_tokens": 512,
            "max_prompt_tokens": 1000,
            "responder_output_token_reserve": 64,
        }

        first = retrieve_longmemeval_diffuse_packet(
            _NoSearchFacade(condenser),
            **kwargs,
        )
        second = retrieve_longmemeval_diffuse_packet(
            _NoSearchFacade(condenser),
            **kwargs,
        )

        assert first.expansion.direct_fallbacks[0].failure_code == "not_annotated"
        assert fallback_chunks[0].chunk_id in first.plan.direct_chunk_ids
        assert "The fallback answer is cobalt blue." in first.packet.context
        assert first.receipt.receipt_sha256 == second.receipt.receipt_sha256
        assert first.plan.plan_sha256 == second.plan.plan_sha256
        assert first.packet.receipt.receipt_sha256 == (
            second.packet.receipt.receipt_sha256
        )

        metrics = measure_longmemeval_diffuse_packet(
            first,
            question_id="longmemeval-fallback",
            gold_answer="cobalt blue",
            evidence_source_ids=("unannotated-source",),
            hydrate_span=condenser.discourse.hydrate_span,
        )
        assert metrics.answer_present is True
        assert metrics.evidence_source_recall == 1.0
        assert metrics.closure_complete_claimed is False
        assert metrics.closure_scope_exhaustive is False

        with pytest.raises(ValueError, match="does not match"):
            replace(
                first.receipt,
                context_sha256="0" * 64,
            )


def test_metrics_make_failed_source_hydration_explicit(tmp_path):
    with _condenser(tmp_path / "tamper") as condenser:
        artifact, ingested, chunks = _publish_fixture(condenser)
        retrieval = retrieve_longmemeval_diffuse_packet(
            _NoSearchFacade(condenser),
            query="How should we improve Atlas retrieval?",
            anchors=(
                RetrievalResult(
                    chunk=chunks[0],
                    turn=ingested[0][0],
                    score=1.0,
                    route="frozen_anchor",
                ),
            ),
            artifact_id=artifact.artifact_id,
            max_context_tokens=4096,
            max_prompt_tokens=8000,
        )

        metrics = measure_longmemeval_diffuse_packet(
            retrieval,
            question_id="longmemeval-provenance",
            gold_answer="95 percent",
            evidence_source_ids=("engineering-thread",),
            hydrate_span=lambda _span: "tampered evidence",
        )

        assert metrics.source_span_hash_valid is False


def test_representative_discovery_is_bound_into_final_packet_receipt(tmp_path):
    with _condenser(tmp_path / "representative") as condenser:
        artifact, ingested, chunks = _publish_fixture(condenser)
        episode_id = condenser.discourse.episode_ids_for_chunks(
            (chunks[0].chunk_id,),
            artifact_id=artifact.artifact_id,
        )[chunks[0].chunk_id]
        query = "What is the Atlas retrieval accuracy target?"
        anchors = (
            RetrievalResult(
                chunk=chunks[4],
                turn=ingested[4][0],
                score=0.5,
                route="frozen_anchor",
            ),
        )
        source_scope = condenser.route_discourse_episode_sources(
            query,
            anchors,
            artifact_id=artifact.artifact_id,
            max_sources=1,
        )
        retrieval = retrieve_longmemeval_diffuse_packet(
            _NoSearchFacade(condenser),
            query=query,
            anchors=anchors,
            artifact_id=artifact.artifact_id,
            max_context_tokens=1024,
            max_prompt_tokens=8000,
            source_candidate_scope=source_scope,
            representative_linker=_SelectEpisodeLinker(episode_id),
            representative_policy=EpisodeRepresentativeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                max_source_groups=1,
                max_episodes_per_source=4,
                max_total_episodes=4,
                top_k=1,
                group_size=4,
            ),
        )

        assert retrieval.representative_expansion is not None
        assert retrieval.receipt.representative_receipt_sha256 == (
            retrieval.representative_expansion.receipt_sha256
        )
        assert retrieval.receipt.representative_seed_episode_ids == (episode_id,)
        assert retrieval.receipt.representative_scope_exhaustive is True
        assert (
            retrieval.representative_expansion.source_scope_receipt_sha256
            == source_scope.receipt_sha256
        )
        assert retrieval.plan.expansion_receipt_sha256 == (
            retrieval.receipt.combined_expansion_sha256
        )
