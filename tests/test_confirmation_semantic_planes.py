from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import DiscourseArtifact, Episode, EvidenceSpan, quote_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    CombinedCumulativeStoreReceipt,
)
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.episodes.retrieval import EpisodeRetrievalPolicy
from memory_condense.search.indexes.lexical import LexicalIndex
from tests.test_confirmation_terminal_policy_boundary import _build_fixture
from tools import confirmation_cumulative_retrieval as cumulative
from tools import confirmation_semantic_planes as semantic
from tools import confirmation_terminal_policy_boundary as terminal
from tools.confirmation_namespace_store_adapter import SealedPayload
from tools.materialize_confirmation_numeric_v5_overlay import (
    VerifiedNamespaceStore,
    VerifiedNamespaceStoreSet,
)
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.semantic_global_completion import SemanticGlobalCompletionResult
from tools.matched_eval.semantic_global_terminal_adapter import (
    SemanticGlobalTerminalCompilation,
)
from tools.matched_eval.semantic_residual_search import SemanticResidualSearchResult
from tools.matched_eval.source_group_reinjection import SourceGroupReinjectionResult
from tools.v4_population_firebreak.canonical import canonical_sha256


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _policy_bindings() -> dict[str, object]:
    return {
        "eligibility_policy": semantic.ELIGIBILITY_POLICY.projection(),
        "global_policy": semantic.GLOBAL_POLICY.projection(),
        "local_policy": semantic.LOCAL_POLICY.projection(),
        "residual_search_policy": semantic.RESIDUAL_POLICY.projection(),
        "terminal_compilation_format": terminal.V5_COMPILATION_FORMAT,
        "terminal_policy": semantic.TERMINAL_POLICY.projection(),
    }


def _build_index_and_episode_store(
    tmp_path: Path,
    namespace_id: str,
    values: tuple[int, ...],
) -> tuple[object, DiscourseStore, EpisodeRetrievalPolicy, Database]:
    database = Database(tmp_path / "semantic.db")
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    spans: list[tuple[str, EvidenceSpan]] = []
    for ordinal, value in enumerate(values, start=1):
        text = f"Fact {value} is cobalt-{value}."
        source_id = f"session-{value}"
        created_at = datetime(2026, 1, ordinal, tzinfo=timezone.utc)
        turn = transcript.append(
            "user",
            text,
            source_id=source_id,
            created_at=created_at,
        )
        chunk = Chunk(
            chunk_id=f"chunk-{value}",
            turn_id=turn.turn_id,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=count_tokens(text),
        )
        lexical.add_chunks([chunk])
        spans.append(
            (
                source_id,
                EvidenceSpan(
                    chunk_id=chunk.chunk_id,
                    start_char=0,
                    end_char=len(text),
                    quote_sha256=quote_sha256(text),
                    ordinal=ordinal,
                    source_id=source_id,
                    turn_id=turn.turn_id,
                    role="user",
                    created_at=created_at.isoformat(),
                ),
            )
        )
    artifact = DiscourseArtifact.create(
        kind="fixed_interval",
        implementation_sha256=_sha("episode-implementation"),
        policy={"fixture": "semantic-plane"},
    )
    episodes = tuple(
        Episode(
            episode_id=f"episode-{ordinal}",
            artifact_id=artifact.artifact_id,
            source_id=source_id,
            sequence_no=0,
            first_ordinal=span.ordinal,
            last_ordinal=span.ordinal,
            evidence=(span,),
            boundary_method="fixed_interval",
        )
        for ordinal, (source_id, span) in enumerate(spans)
    )
    DiscourseStore(database).publish(artifact, episodes=episodes)
    database.close()
    database = Database(tmp_path / "semantic.db", read_only=True)
    streams = scan_discourse_source_chunks(database)
    frozen = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("snapshot"),
        combined_store_receipt_sha256=_sha("combined-store"),
        source_streams=streams,
    )
    cache = cache_namespace_partitions(
        database,
        frozen,
        source_database_sha256=_sha("database"),
        source_store_receipt_sha256=_sha("combined-store"),
    )
    window = build_full_store_window_index(cache)
    index = semantic.residual.build_semantic_residual_index(
        window,
        {row.source_id: None for row in window.rows},
        policy=semantic.RESIDUAL_POLICY,
    )
    index = replace(index, namespace_id=namespace_id, receipt_sha256="")
    episode_store = DiscourseStore(database)
    return (
        index,
        episode_store,
        EpisodeRetrievalPolicy(
            artifact_id=artifact.artifact_id,
            max_anchor_episodes=8,
            previous_episodes=1,
            next_episodes=1,
            max_episode_seeds=24,
            max_direct_fallbacks=16,
        ),
        database,
    )


class _Backend:
    def __init__(self, resources: dict[str, semantic.SemanticNamespaceResources]) -> None:
        self.identity_sha256 = _sha("semantic-backend")
        self.resources = resources
        self.opened: list[str] = []

    @contextmanager
    def open_namespace(self, store: VerifiedNamespaceStore):
        self.opened.append(store.namespace_id)
        yield self.resources[store.namespace_id]


class _Protected:
    identity_sha256 = _sha("protected-adapter")
    protected_owner_artifact_sha256 = _sha("protected-owner")

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def protected_evidence(self, parent: terminal.TerminalParentRow):
        self.calls.append((parent.row_receipt_sha256, parent.question))
        return ()


def _sealed_payload(path: Path, payload: dict[str, object]) -> SealedPayload:
    sealed, _ = cumulative._publish_sealed(path, payload, label="semantic fixture")  # noqa: SLF001
    return sealed


def _stores_and_vectors(
    tmp_path: Path,
    inputs: terminal.ConfirmationTerminalInputs,
) -> tuple[VerifiedNamespaceStoreSet, semantic.ConfirmationSemanticVectorRelease]:
    stores: dict[str, VerifiedNamespaceStore] = {}
    descriptors: dict[str, semantic.SemanticFacetVectorDescriptor] = {}
    for namespace_id, namespace_receipt, _question_ids in inputs.namespaces:
        store_id = _sha(f"store:{namespace_id}")
        receipt = CombinedCumulativeStoreReceipt(
            source_store_identity_sha256=_sha(f"identity:{namespace_id}"),
            target_store_identity_sha256=_sha(f"identity:{namespace_id}"),
            source_database_sha256=_sha(f"source-db:{namespace_id}"),
            target_database_sha256=_sha(f"target-db:{namespace_id}"),
            target_index_sha256=_sha(f"target-index:{namespace_id}"),
            retrieval_policy_sha256=_sha(f"retrieval:{namespace_id}"),
            context_budget_sha256=_sha(f"budget:{namespace_id}"),
            training_query_batch_sha256=_sha(f"training:{namespace_id}"),
            held_out_query_batch_sha256=_sha(f"held-out:{namespace_id}"),
            compilation_receipt_sha256=_sha(f"compilation:{namespace_id}"),
            artifact_id=f"artifact-{namespace_id}",
            snapshot_sha256=_sha(f"snapshot:{namespace_id}"),
            turn_count=1,
            chunk_count=1,
            causal_events=0,
            causal_graph_edges=0,
        )
        prep_sha = _sha(f"preparation:{namespace_id}")
        store = VerifiedNamespaceStore(
            namespace_id=namespace_id,
            namespace_receipt_sha256=namespace_receipt,
            namespace_store_id=store_id,
            store_dir=tmp_path,
            preparation_checkpoint_sha256=prep_sha,
            combined_store_receipt=receipt,
            store_identity_sha256=_sha(f"verified-store:{namespace_id}"),
        )
        stores[namespace_id] = store
        parents = tuple(row for row in inputs.rows if row.namespace_id == namespace_id)
        batch = semantic.semantic_facet_query_batch(
            tuple(row.dated_question for row in parents)
        )
        rows = [
            {
                "query": query,
                "query_sha256": quote_sha256(query),
                "vector": [1.0, 0.0],
                "vector_sha256": canonical_sha256([1.0, 0.0]),
            }
            for query in batch
        ]
        vector_body = {
            "dimension": 2,
            "embedding_identity_sha256": _sha("embedding"),
            "format": semantic.FACET_VECTOR_FORMAT,
            "namespace_id": namespace_id,
            "namespace_store_id": store_id,
            "preparation_checkpoint_sha256": prep_sha,
            "query_batch_sha256": identity_sha256(
                [{"query_sha256": quote_sha256(query)} for query in batch]
            ),
            "retrieval_query_vector_artifact_sha256": _sha(
                f"retrieval-vectors:{namespace_id}"
            ),
            "rows": rows,
            "vector_values_sha256": canonical_sha256(
                [
                    {
                        "query_sha256": row["query_sha256"],
                        "vector_sha256": row["vector_sha256"],
                    }
                    for row in rows
                ]
            ),
            "work_receipt_sha256": _sha(f"work:{namespace_id}"),
        }
        payload = {
            **vector_body,
            "artifact_receipt_sha256": identity_sha256(vector_body),
        }
        artifact = _sealed_payload(tmp_path / f"vectors-{store_id}.json", payload)
        descriptors[namespace_id] = semantic.SemanticFacetVectorDescriptor(
            namespace_id=namespace_id,
            namespace_store_id=store_id,
            preparation_checkpoint_sha256=prep_sha,
            retrieval_vector_artifact_sha256=vector_body[
                "retrieval_query_vector_artifact_sha256"
            ],
            artifact_path=artifact.path.resolve(),
            artifact_sha256=artifact.sha256,
            artifact_receipt_sha256=payload["artifact_receipt_sha256"],
            query_batch=batch,
            query_batch_sha256=vector_body["query_batch_sha256"],
            vector_values_sha256=vector_body["vector_values_sha256"],
            dimension=2,
        )
    barrier = _sealed_payload(tmp_path / "barrier.json", {"fixture": "barrier"})
    preparation = _sealed_payload(
        tmp_path / "facet-preparation.json", {"fixture": "facet-preparation"}
    )
    release = _sealed_payload(
        tmp_path / "facet-release.json", {"fixture": "facet-release"}
    )
    store_set = VerifiedNamespaceStoreSet(
        policy_manifest_sha256=inputs.policy.sha256,
        treatment_preflight_sha256=inputs.treatment_preflight.sha256,
        barrier_sha256=barrier.sha256,
        barrier_receipt_sha256=_sha("barrier-receipt"),
        stores_by_namespace=MappingProxyType(stores),
        identity_sha256=_sha("store-set"),
    )
    vector_release = semantic.ConfirmationSemanticVectorRelease(
        preparation=preparation,
        release=release,
        barrier=barrier,
        descriptors_by_namespace=MappingProxyType(descriptors),
    )
    return store_set, vector_release


def test_exact_frozen_semantic_policy_receipts_and_no_population_router() -> None:
    assert {
        "eligibility_policy": semantic.ELIGIBILITY_POLICY.receipt_sha256,
        "residual_search_policy": semantic.RESIDUAL_POLICY.receipt_sha256,
        "local_policy": semantic.LOCAL_POLICY.receipt_sha256,
        "global_policy": semantic.GLOBAL_POLICY.receipt_sha256,
        "terminal_policy": semantic.TERMINAL_POLICY.receipt_sha256,
    } == dict(semantic.EXPECTED_POLICY_RECEIPTS)
    assert semantic.RESIDUAL_POLICY.max_cell_tokens == 2_048
    assert semantic.RESIDUAL_POLICY.payload_token_cap == 2_400
    assert semantic.RESIDUAL_POLICY.dual_gate_enabled is True
    with pytest.raises(
        semantic.ConfirmationSemanticPlanesError,
        match="population/label routing",
    ):
        semantic._assert_routing_neutral(  # noqa: SLF001
            {"validation_ordinals": [7]}, "fixture"
        )


def test_eligible_ineligible_materialization_resume_replay_and_tamper(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=(0, 1),
        eligible_semantics=frozenset({0}),
        id_prefix="semantic",
        namespace_sizes=(2,),
    )
    fixture.inputs.policy.payload["treatment_policy"][
        "full100_policy_bindings"
    ] = _policy_bindings()
    namespace_id = fixture.inputs.rows[0].namespace_id
    index, episode_store, episode_policy, database = _build_index_and_episode_store(
        tmp_path, namespace_id, fixture.semantics
    )
    stores, vectors = _stores_and_vectors(tmp_path, fixture.inputs)
    resource = semantic.SemanticNamespaceResources(
        residual_index=index,
        episode_lookup=episode_store,
        episode_policy=episode_policy,
        episode_artifact_binding_receipt_sha256=_sha("episode-binding"),
    )
    protected = _Protected()
    output = tmp_path / "output"
    try:
        first_backend = _Backend({namespace_id: resource})
        first = semantic.materialize_confirmation_semantic_planes(
            fixture.inputs,
            stores,
            vectors,
            protected,
            output_root=output,
            backend=first_backend,
        )
        assert first.physical_provider_calls == 0
        assert first.created_checkpoint_count == 1
        assert first_backend.opened == [namespace_id]
        assert protected.calls == [
            (
                fixture.inputs.rows[0].row_receipt_sha256,
                fixture.inputs.rows[0].question,
            )
        ]
        eligible, ineligible = first.rows
        assert type(eligible.residual_result) is SemanticResidualSearchResult
        assert type(eligible.local_result) is SourceGroupReinjectionResult
        assert type(eligible.global_result) is SemanticGlobalCompletionResult
        assert type(eligible.terminal_compilation) is SemanticGlobalTerminalCompilation
        assert ineligible.eligibility.eligible is False
        assert ineligible.query is ineligible.residual_result is None
        assert len(first.terminal_plan_export.rows_by_parent_receipt) == 1

        second_backend = _Backend({namespace_id: resource})
        second = semantic.materialize_confirmation_semantic_planes(
            fixture.inputs,
            stores,
            vectors,
            protected,
            output_root=output,
            backend=second_backend,
        )
        assert second.reused_checkpoint_count == 1
        assert second.artifact.sha256 == first.artifact.sha256
        assert [row.receipt_sha256 for row in second.rows] == [
            row.receipt_sha256 for row in first.rows
        ]

        replay_backend = _Backend({namespace_id: resource})
        replay = semantic.replay_confirmation_semantic_planes(
            fixture.inputs,
            stores,
            vectors,
            protected,
            output_root=output,
            expected_materialization_sha256=first.artifact.sha256,
            expected_checkpoint_sha256_by_namespace_receipt=(
                first.checkpoint_sha256_by_namespace_receipt
            ),
            backend=replay_backend,
        )
        assert replay.artifact.sha256 == first.artifact.sha256
        assert (output / semantic.REPLAY_NAME).read_bytes() == first.artifact.path.read_bytes()

        checkpoint = first.checkpoint_paths[0]
        checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
        with pytest.raises(semantic.ConfirmationSemanticPlanesError):
            semantic.materialize_confirmation_semantic_planes(
                fixture.inputs,
                stores,
                vectors,
                protected,
                output_root=output,
                backend=_Backend({namespace_id: resource}),
            )
    finally:
        database.close()
