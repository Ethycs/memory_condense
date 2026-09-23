"""Opt-in application composition for source-grounded discourse closure."""

from __future__ import annotations

import re
import zlib
from dataclasses import replace

import numpy as np
import pytest

from memory_condense.associations.head_memory_models import (
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.discourse import (
    DiscourseArtifact,
    EvidenceAtom,
    EvidenceObligation,
    QueryProgram,
    make_atom_id,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult
from memory_condense.ingest.discourse_linker import (
    LinkerInput,
    LinkerOutput,
    RuleBasedDiscourseLinker,
)
from memory_condense.persistence.discourse_store import (
    ArtifactCoverageMark,
    DiscourseStore,
)
from memory_condense.search.episodes import (
    EpisodeBuilder,
    EpisodeRepresentativePrefilterPolicy,
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRetrievalPolicy,
    EpisodeSourceCandidate,
)


class _FakeEmbedder:
    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        value = np.zeros(self._dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            value[zlib.crc32(token.encode()) % self._dim] += 1.0
        if not value.any():
            value[0] = 1.0
        return value

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(
                update={"embedding": self._vector(chunk.text).tolist()}
            )
            for chunk in chunks
        ]


class _BombDiscourseStore:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"legacy path consulted discourse store attribute {name}")


class _RecordingLinker:
    def __init__(self) -> None:
        self.chunk_ids: tuple[str, ...] = ()

    def link(self, artifact_id, inputs):
        self.chunk_ids = tuple(item.atom.span.chunk_id for item in inputs)
        return RuleBasedDiscourseLinker().link(artifact_id, inputs)


class _NoOutputLinker:
    def link(self, artifact_id, inputs):
        return LinkerOutput((), ())


class _BombNestedLinker:
    max_candidates = 8

    def inspect_nested(self, *_args, **_kwargs):
        raise AssertionError("invalid source scope reached the nested linker")


class _SelectNestedEpisode:
    max_candidates = 8

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id

    def inspect_nested(self, _query, groups, **_kwargs):
        return NestedMemoryInspection(
            hits=(
                MemoryLinkHit(
                    episode_id=self.episode_id,
                    qk_score=1.0,
                    ov_transport=1.0,
                    head_weights=(1.0,),
                ),
            ),
            passes=1,
            max_workspace_candidates=max(len(group) for group in groups),
            max_workspace_tokens=64,
            total_candidate_inspections=sum(len(group) for group in groups),
        )


def _artifact() -> DiscourseArtifact:
    return DiscourseArtifact.create(
        kind="rule-based-discourse-test",
        implementation_sha256="a" * 64,
        policy={"episode_boundary": "fixed-one", "linker": "rules-v1"},
    )


def _condenser(tmp_path) -> MemoryCondenser:
    return MemoryCondenser(
        data_dir=tmp_path,
        embedder=_FakeEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
        persist_index_on_close=False,
    )


def test_discourse_source_streams_separate_metadata_without_emitting_text(
    tmp_path,
) -> None:
    with _condenser(tmp_path / "source-streams") as condenser:
        marker = condenser.ingest(
            "system",
            "[session-a took place at 2024/05/01 (Wed) 12:00]",
            source_id="session-a",
        )[1][0]
        content = condenser.ingest(
            "user",
            "The launch badge was amber.",
            source_id="session-a",
        )[1][0]
        other = condenser.ingest(
            "assistant",
            "The deployment completed.",
            source_id="session-b",
        )[1][0]

        streams = condenser.discourse_source_streams()

        assert [stream.source_id for stream in streams] == [
            "session-a",
            "session-b",
        ]
        assert streams[0].content_chunk_ids == (content.chunk_id,)
        assert streams[0].metadata_chunk_ids == (marker.chunk_id,)
        assert streams[1].content_chunk_ids == (other.chunk_id,)
        assert "amber" not in repr(streams)


def test_episode_source_router_attests_the_complete_source_universe(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = _artifact()
    with _condenser(tmp_path / "source-router") as condenser:
        direct = condenser.ingest(
            "user",
            "The directly retrieved alpha note.",
            source_id="source-a",
        )
        condenser.ingest(
            "assistant",
            "The independently routed beta note.",
            source_id="source-b",
        )
        condenser.discourse.publish(artifact)
        anchor = condenser.retriever.hydrate_chunk(
            direct[1][0].chunk_id,
            score=0.8,
            route="hybrid",
        )
        assert anchor is not None
        monkeypatch.setattr(
            condenser._retriever,
            "source_tfisf_query",
            lambda *_args, **_kwargs: [("source-b", 9.0)],
        )

        scope = condenser.route_discourse_episode_sources(
            "Where is beta?",
            (anchor,),
            artifact_id=artifact.artifact_id,
            max_sources=2,
        )

        assert scope.universe_source_ids == ("source-a", "source-b")
        assert {item.source_id for item in scope.candidates} == {
            "source-a",
            "source-b",
        }
        assert scope.selected_scope_exhaustive
        assert scope.truncated_source_ids == ()


def test_representative_retrieval_rejects_self_hashed_forged_source_universes(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = _artifact()
    query = "Where is beta?"
    with _condenser(tmp_path / "forged-source-scope") as condenser:
        direct = condenser.ingest(
            "user",
            "The directly retrieved alpha note.",
            source_id="source-a",
        )
        condenser.ingest(
            "assistant",
            "The independently routed beta note.",
            source_id="source-b",
        )
        condenser.discourse.publish(artifact)
        anchor = condenser.retriever.hydrate_chunk(
            direct[1][0].chunk_id,
            score=0.8,
            route="hybrid",
        )
        assert anchor is not None
        monkeypatch.setattr(
            condenser._retriever,
            "source_tfisf_query",
            lambda *_args, **_kwargs: [("source-b", 9.0)],
        )
        scope = condenser.route_discourse_episode_sources(
            query,
            (anchor,),
            artifact_id=artifact.artifact_id,
            max_sources=2,
        )
        source_a = tuple(
            candidate
            for candidate in scope.candidates
            if candidate.source_id == "source-a"
        )
        forged_scopes = (
            replace(
                scope,
                universe_source_ids=("source-a",),
                candidates=source_a,
                truncated_source_ids=(),
                receipt_sha256="",
            ),
            replace(
                scope,
                universe_source_ids=("source-a", "source-b", "source-c"),
                candidates=(
                    *scope.candidates,
                    EpisodeSourceCandidate(
                        "source-c",
                        0.0,
                        "forged-unscored-source",
                    ),
                ),
                truncated_source_ids=(),
                receipt_sha256="",
            ),
            replace(
                scope,
                universe_source_ids=tuple(reversed(scope.universe_source_ids)),
                receipt_sha256="",
            ),
        )
        policy = EpisodeRepresentativeRetrievalPolicy(
            artifact_id=artifact.artifact_id,
        )

        for forged in forged_scopes:
            with pytest.raises(ValueError, match="scope universe"):
                condenser.retrieve_discourse_episode_representatives(
                    query,
                    forged.candidates,
                    _BombNestedLinker(),
                    policy=policy,
                    source_scope=forged,
                )


def test_representative_retrieval_rejects_scope_after_source_universe_changes(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = _artifact()
    query = "Where is beta?"
    with _condenser(tmp_path / "stale-source-scope") as condenser:
        direct = condenser.ingest(
            "user",
            "The directly retrieved alpha note.",
            source_id="source-a",
        )
        condenser.ingest(
            "assistant",
            "The independently routed beta note.",
            source_id="source-b",
        )
        condenser.discourse.publish(artifact)
        anchor = condenser.retriever.hydrate_chunk(
            direct[1][0].chunk_id,
            score=0.8,
            route="hybrid",
        )
        assert anchor is not None
        monkeypatch.setattr(
            condenser._retriever,
            "source_tfisf_query",
            lambda *_args, **_kwargs: [("source-b", 9.0)],
        )
        scope = condenser.route_discourse_episode_sources(
            query,
            (anchor,),
            artifact_id=artifact.artifact_id,
            max_sources=2,
        )
        condenser.ingest(
            "user",
            "A new source appeared after routing.",
            source_id="source-c",
        )

        with pytest.raises(ValueError, match="stale or foreign"):
            condenser.retrieve_discourse_episode_representatives(
                query,
                scope.candidates,
                _BombNestedLinker(),
                policy=EpisodeRepresentativeRetrievalPolicy(
                    artifact_id=artifact.artifact_id,
                ),
                source_scope=scope,
            )


def test_prepared_episode_descriptors_feed_shadow_without_narrowing_qwen(
    tmp_path,
) -> None:
    artifact = _artifact()
    embedding_identity = {
        "backend": "fixture",
        "model_id": "fixture-embedding",
        "dimension": 32,
        "space": "cosine",
    }
    with _condenser(tmp_path / "descriptor-shadow") as condenser:
        chunks = tuple(
            condenser.ingest(
                "user",
                text,
                source_id="source-a",
            )[1][0]
            for text in (
                "alpha launch badge evidence",
                "beta restaurant evidence",
                "gamma travel evidence",
                "delta meeting evidence",
            )
        )
        chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
        publication = condenser.build_and_publish_discourse_episodes(
            artifact,
            chunk_ids,
            builder=EpisodeBuilder(min_size=1, max_size=1),
            embeddings=condenser.discourse_chunk_embeddings(chunk_ids),
            representative_limit=1,
        )
        prepared = condenser.prepare_discourse_episode_descriptors(
            artifact.artifact_id,
            embedding_identity=embedding_identity,
        )

        target = publication.build.episodes[0]
        plan = condenser.retrieve_discourse_episode_representatives(
            "alpha launch badge evidence",
            (EpisodeSourceCandidate("source-a", 1.0),),
            _SelectNestedEpisode(target.episode_id),
            policy=EpisodeRepresentativeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                max_episodes_per_source=4,
                max_total_episodes=4,
                group_size=2,
                top_k=1,
                max_representatives_per_episode=1,
            ),
            prefilter_policy=EpisodeRepresentativePrefilterPolicy(
                mode="shadow",
                cap=2,
                top_k=1,
                cutoff_margin=0.0,
            ),
            prefilter_embedding_identity=embedding_identity,
        )

        assert prepared.receipt.episode_count == 4
        assert plan.seeds[0].episode_id == target.episode_id
        report = condenser.last_episode_prefilter_report
        assert report["descriptor_source"] == "resident_index"
        assert report["input_count"] == 4
        assert report["proposal_count"] == 2
        assert report["output_count"] == 4
        assert report["proposal_is_narrower"] is True
        assert len(report["proposal_episode_ids"]) == 2
        assert report["full_qwen_hit_count"] == 1
        assert report["full_qwen_hit_recall"] == 1.0


def test_descriptor_preparation_rejects_a_concurrent_snapshot_change(
    tmp_path,
) -> None:
    artifact = _artifact()
    with _condenser(tmp_path / "descriptor-snapshot-race") as condenser:
        chunk = condenser.ingest(
            "user",
            "initial episode content",
            source_id="initial-source",
        )[1][0]
        condenser.build_and_publish_discourse_episodes(
            artifact,
            (chunk.chunk_id,),
            builder=EpisodeBuilder(min_size=1, max_size=1),
            embeddings=condenser.discourse_chunk_embeddings((chunk.chunk_id,)),
            representative_limit=1,
        )

        class _ChangingIndex:
            invalidated = False

            def ensure(self, *_args, **_kwargs):
                condenser.ingest(
                    "user",
                    "content arriving during descriptor preparation",
                    source_id="late-source",
                )
                return object()

            def invalidate(self):
                self.invalidated = True

            def release(self):
                pass

        changing = _ChangingIndex()
        condenser._episode_descriptor_index = changing

        with pytest.raises(RuntimeError, match="snapshot changed"):
            condenser.prepare_discourse_episode_descriptors(
                artifact.artifact_id,
                embedding_identity={"backend": "fixture", "dimension": 32},
            )

        assert changing.invalidated is True


def test_discourse_workflow_builds_links_closes_and_packs_exact_evidence(
    tmp_path,
) -> None:
    artifact = _artifact()
    with _condenser(tmp_path / "workflow") as condenser:
        annotated = [
            condenser.ingest(
                "user",
                "The objective is improve factual retrieval accuracy.",
                source_id="engineering-thread",
            ),
            condenser.ingest(
                "assistant",
                "We decided to use exact source spans.",
                source_id="engineering-thread",
            ),
            condenser.ingest(
                "user",
                "The measured accuracy result passed the target.",
                source_id="engineering-thread",
            ),
        ]
        fallback_turn, fallback_chunks = condenser.ingest(
            "user",
            "This unannotated raw chunk must survive episode lookup failure.",
            source_id="another-thread",
        )
        chunks = [row[1][0] for row in annotated]
        chunk_ids = tuple(chunk.chunk_id for chunk in reversed(chunks))

        first = condenser.build_and_publish_discourse_episodes(
            artifact,
            chunk_ids,
            builder=EpisodeBuilder(min_size=1, max_size=1),
            representative_limit=1,
        )
        assert len(first.build.episodes) == 3
        assert [episode.sequence_no for episode in first.build.episodes] == [0, 1, 2]
        assert [
            episode.evidence[0].chunk_id for episode in first.build.episodes
        ] == [chunk.chunk_id for chunk in chunks]
        assert len(first.representatives) == 3
        assert first.returned_signal_transformer_state_bytes == 0

        replay = condenser.build_and_publish_discourse_episodes(
            artifact,
            chunk_ids,
            builder=EpisodeBuilder(min_size=1, max_size=1),
            representative_limit=1,
        )
        assert replay == first

        linker = _RecordingLinker()
        publication = condenser.link_and_publish_discourse(
            artifact,
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            linker=linker,
        )
        assert linker.chunk_ids == tuple(chunk.chunk_id for chunk in chunks)
        assert len(publication.output.units) == 3
        assert publication.output.retained_request_token_state_bytes == 0
        assert publication.output.relations
        assert condenser.link_and_publish_discourse(
            artifact,
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        ) == publication
        assert condenser.publish_discourse_graph(artifact) == publication.snapshot

        # This chunk was deliberately processed by both annotation stages with
        # no output.  Explicit zero-output marks make whole-corpus completeness
        # distinguishable from a chunk that was never processed.
        condenser.discourse.publish(
            artifact,
            coverage=(
                ArtifactCoverageMark(
                    fallback_chunks[0].chunk_id,
                    "episode",
                    "no_output",
                ),
            ),
        )
        no_output = condenser.link_and_publish_discourse(
            artifact,
            chunk_ids=(fallback_chunks[0].chunk_id,),
            linker=_NoOutputLinker(),
        )
        assert no_output.output.units == ()
        coverage_receipt = condenser.finalize_discourse_coverage(
            artifact.artifact_id
        )
        assert coverage_receipt.chunk_count == 4

        direct = RetrievalResult(
            chunk=chunks[0],
            turn=annotated[0][0],
            score=0.9,
            route="hybrid",
        )
        missing = RetrievalResult(
            chunk=fallback_chunks[0],
            turn=fallback_turn,
            score=0.8,
            route="hybrid",
        )
        expansion = condenser.expand_discourse_episode_seeds(
            (direct, missing),
            policy=EpisodeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                previous_episodes=0,
                next_episodes=0,
            ),
        )
        assert len(expansion.seeds) == 1
        assert expansion.direct_chunk_ids == (
            chunks[0].chunk_id,
            fallback_chunks[0].chunk_id,
        )

        program = QueryProgram(
            query="What objective did the team set?",
            intent="lookup",
            subject_terms=("factual retrieval",),
            obligations=(
                EvidenceObligation(
                    obligation_id="objective",
                    kind="objective",
                    required=True,
                    weight=1.0,
                    unit_kinds=("goal",),
                    subject_terms=("factual retrieval",),
                ),
            ),
        )
        plan = condenser.retrieve_close_discourse_evidence(
            (direct, missing),
            artifact_id=artifact.artifact_id,
            query_program=program,
            episode_policy=EpisodeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                previous_episodes=0,
                next_episodes=0,
            ),
        )
        assert plan.complete_claimed
        assert plan.expansion_receipt_sha256 == expansion.receipt_sha256
        assert plan.direct_chunk_ids == tuple(sorted(expansion.direct_chunk_ids))
        assert fallback_chunks[0].chunk_id in {
            atom.span.chunk_id for atom in plan.atoms
        }

        packet = condenser.pack_discourse_evidence(
            plan,
            max_context_tokens=1024,
            base_messages=(
                {"role": "system", "content": "Answer only from evidence."},
            ),
            evidence_prefix="Evidence follows:\n",
            max_prompt_tokens=2048,
            output_token_reserve=64,
        )
        assert "The objective is improve factual retrieval accuracy." in packet.context
        assert packet.receipt.prompt_workspace_token_proxy is not None
        assert packet.receipt.prompt_workspace_token_proxy <= 2048
        assert packet.receipt.retained_request_token_state_bytes == 0
        assert isinstance(condenser.discourse, DiscourseStore)


def test_legacy_retrieval_and_context_paths_never_consult_discourse(tmp_path) -> None:
    with _condenser(tmp_path / "legacy") as condenser:
        condenser.ingest(
            "user",
            "Factual accuracy improves when every answer cites raw evidence.",
            source_id="legacy-thread",
        )
        condenser.ingest(
            "assistant",
            "The existing hybrid retrieval output remains unchanged.",
            source_id="legacy-thread",
        )
        query = "How does factual accuracy improve?"
        before_results = condenser.search_hybrid(query, k=2)
        before_result_bytes = tuple(
            result.model_dump_json() for result in before_results
        )
        before_context = condenser.build_context(
            query,
            recent_turns=2,
            k_memories=0,
            expansion_results=before_results,
            reheat_memories=False,
            use_consolidation=False,
            learn_consolidation=False,
        ).model_dump_json()

        condenser._discourse = _BombDiscourseStore()
        after_results = condenser.search_hybrid(query, k=2)
        after_result_bytes = tuple(
            result.model_dump_json() for result in after_results
        )
        after_context = condenser.build_context(
            query,
            recent_turns=2,
            k_memories=0,
            expansion_results=after_results,
            reheat_memories=False,
            use_consolidation=False,
            learn_consolidation=False,
        ).model_dump_json()

        assert after_result_bytes == before_result_bytes
        assert after_context == before_context


def test_injected_linker_inputs_require_complete_authoritative_source(tmp_path):
    with _condenser(tmp_path / "missing-source") as condenser:
        _, chunks_a = condenser.ingest("user", "A claim.", source_id="source-a")
        _, chunks_b = condenser.ingest("user", "A contradiction.", source_id="source-b")
        inputs = []
        for span in condenser.discourse.evidence_for_chunks(
            (chunks_a[0].chunk_id, chunks_b[0].chunk_id)
        ):
            incomplete = replace(span, source_id=None)
            inputs.append(
                LinkerInput(
                    EvidenceAtom(
                        make_atom_id(incomplete),
                        incomplete,
                        condenser.discourse.hydrate_span(incomplete),
                        "injected",
                    )
                )
            )
        with pytest.raises(ValueError, match="complete authoritative provenance"):
            condenser.link_and_publish_discourse(_artifact(), inputs=tuple(inputs))


def test_multiple_artifacts_require_explicit_scope_and_never_mix_units(tmp_path):
    first = _artifact()
    second = DiscourseArtifact.create(
        kind="rule-based-discourse-test",
        implementation_sha256="a" * 64,
        policy={"episode_boundary": "fixed-one", "linker": "rules-v2"},
    )
    with _condenser(tmp_path / "artifact-scope") as condenser:
        _, chunks = condenser.ingest(
            "user",
            "The objective is exact artifact isolation.",
            source_id="thread",
        )
        chunk_ids = (chunks[0].chunk_id,)
        condenser.link_and_publish_discourse(first, chunk_ids=chunk_ids)
        condenser.link_and_publish_discourse(second, chunk_ids=chunk_ids)
        with pytest.raises(ValueError, match="artifact_id is required"):
            condenser.close_discourse_evidence(
                "What is the objective?",
                direct_chunk_ids=chunk_ids,
            )
        plan = condenser.close_discourse_evidence(
            "What is the objective?",
            direct_chunk_ids=chunk_ids,
            artifact_id=first.artifact_id,
        )
        assert plan.artifact_id == first.artifact_id
        assert all(
            condenser.discourse.get_unit(unit_id).artifact_id == first.artifact_id
            for unit_id in plan.visited_unit_ids
        )
