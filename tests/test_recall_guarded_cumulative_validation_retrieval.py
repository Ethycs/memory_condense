from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval import _recall_guarded_cumulative_validation_campaign as merge_impl
from memory_condense.eval import _recall_guarded_cumulative_validation_shard as shard_impl
from memory_condense.eval._recall_guarded_cumulative_contracts import (
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    RecallGuardedCumulativeReceipt,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    gold_blind_from_treatment_sample,
)
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_100Q_OFFSETS,
    LOCKED_CONTEXT_TARGET_TOKENS,
    LOCKED_LONGMEMEVAL_DATASET_SHA256,
    LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
    LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    POPULATION_IDENTITY_FORMAT,
    QUESTION_PROBE_FORMAT,
    SHARD_IDENTITY_FORMAT,
)
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    CombinedCumulativeStoreReceipt,
)
from memory_condense.eval.recall_guarded_cumulative_validation_retrieval import (
    LOCKED_VALIDATION_POLICY_MANIFEST_SHA256,
    VALIDATION_EXTERNAL_RECONSTRUCTION_FORMAT,
    VALIDATION_MERGED_RETRIEVAL_FORMAT,
    VALIDATION_SHARD_QUESTION_FORMAT,
    VALIDATION_SHARD_RETRIEVAL_FORMAT,
    ValidationShardPreflight,
    load_frozen_validation_policy,
    main,
    merge_locked_validation_retrievals,
    merged_question_store_receipts,
    preflight_locked_validation_shard,
    validate_merged_validation_retrieval,
    validate_validation_shard_retrieval,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)


POLICY = Path(
    "docs/10 - Research Log/data/"
    "longmemeval-qwen-choice-coverage-operational-validation-v3.json"
)
_H = "a" * 64


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _samples() -> tuple[BenchmarkSample, ...]:
    rows: list[BenchmarkSample] = []
    for shard_index, offset in enumerate(LOCKED_100Q_OFFSETS):
        questions = [
            BenchmarkQuestion(
                question_id=f"question-{offset + local:03d}",
                question=f"Which code belongs to item {offset + local:03d}?",
                answer=f"secret-{offset + local:03d}",
                evidence_sources=[f"gold-{offset + local:03d}"],
            )
            for local in range(10)
        ]
        rows.append(
            BenchmarkSample(
                sample_id=f"shard-{shard_index}",
                turns=[("user", f"fixture corpus {shard_index}")],
                turn_source_ids=[f"source-{shard_index}"],
                turn_created_at=[datetime(2026, 8, 22, tzinfo=timezone.utc)],
                questions=questions,
            )
        )
    return tuple(rows)


def _population(
    samples: tuple[BenchmarkSample, ...],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    shards: list[dict[str, object]] = []
    ordered_question_ids: list[str] = []
    ordered_probe_ids: list[str] = []
    for offset, sample in zip(LOCKED_100Q_OFFSETS, samples, strict=True):
        probes: list[dict[str, object]] = []
        for local, question in enumerate(sample.questions):
            body: dict[str, object] = {
                "format": QUESTION_PROBE_FORMAT,
                "ordinal": local,
                "question_id_sha256": identity_sha256(
                    {"question_id": question.question_id}
                ),
                "retrieval_query_sha256": quote_sha256(question.question),
                "prompt_question_sha256": quote_sha256(question.dated_question),
            }
            body["probe_identity_sha256"] = identity_sha256(body)
            probes.append(body)
            ordered_question_ids.append(str(body["question_id_sha256"]))
            ordered_probe_ids.append(str(body["probe_identity_sha256"]))
        shard: dict[str, object] = {
            "format": SHARD_IDENTITY_FORMAT,
            "benchmark_format": "longmemeval",
            "dataset_sha256": LOCKED_LONGMEMEVAL_DATASET_SHA256,
            "split_manifest_sha256": (
                LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256
            ),
            "split": "validation",
            "construction": {
                "target_tokens": LOCKED_CONTEXT_TARGET_TOKENS,
                "questions_per_shard": 10,
                "sample_offset": offset,
            },
            "sample_id_sha256": identity_sha256(
                {"sample_id": sample.sample_id}
            ),
            "gold_blind_corpus_sha256": _digest(f"corpus-{offset}"),
            "transcript_tokens": 10,
            "turn_count": 1,
            "source_count": 1,
            "question_count": 10,
            "ordered_question_probes": probes,
            "gold_fields_present": False,
        }
        shard["shard_identity_sha256"] = identity_sha256(shard)
        shards.append(shard)
    population: dict[str, object] = {
        "format": POPULATION_IDENTITY_FORMAT,
        "benchmark_format": "longmemeval",
        "dataset_sha256": LOCKED_LONGMEMEVAL_DATASET_SHA256,
        "split_manifest_sha256": LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
        "split": "validation",
        "construction": {
            "target_tokens": LOCKED_CONTEXT_TARGET_TOKENS,
            "questions_per_shard": 10,
            "shard_offsets": list(LOCKED_100Q_OFFSETS),
        },
        "shard_count": 10,
        "question_count": 100,
        "total_transcript_tokens": 100,
        "total_turn_count": 10,
        "ordered_shard_identity_sha256s": [
            shard["shard_identity_sha256"] for shard in shards
        ],
        "ordered_question_id_sha256s": ordered_question_ids,
        "ordered_question_probe_sha256s": ordered_probe_ids,
        "gold_fields_present": False,
    }
    population["population_identity_sha256"] = identity_sha256(population)
    return tuple(shards), population


def _source_receipt(
    sample: BenchmarkSample,
    *,
    shard_index: int,
) -> dict[str, object]:
    key = _digest(f"store-key-{shard_index}")
    query_key = _digest(f"query-key-{shard_index}")
    body: dict[str, object] = {
        "format": shard_impl.CURRENT_SOURCE_FORMAT,
        "source_scope": shard_impl.CURRENT_SOURCE_SCOPE,
        "timestamp_semantics": shard_impl.CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
        "base_store_key": key,
        "selected_store_entry": f"stores/{key}",
        "store_manifest_sha256": _digest(f"manifest-{shard_index}"),
        "store_artifact_sha256": _digest(f"artifact-{shard_index}"),
        "database_sha256": _digest(f"database-{shard_index}"),
        "index_sha256": _digest(f"index-{shard_index}"),
        "corpus_sha256": gold_blind_from_treatment_sample(sample).corpus_sha256,
        "turn_count": len(sample.turns),
        "chunk_count": 1,
        "deterministic_turn_ids_sha256": _digest(f"turns-{shard_index}"),
        "turn_sequence_sha256": _digest(f"turn-seq-{shard_index}"),
        "chunk_sequence_sha256": _digest(f"chunks-{shard_index}"),
        "source_streams_sha256": _digest(f"streams-{shard_index}"),
        "embedding_identity": {
            "backend": "sentence-transformers.encode-v1",
            "model_id": DEFAULT_MODEL_NAME,
            "model_revision": DEFAULT_MODEL_REVISION,
            "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
            "dimension": DEFAULT_MODEL_DIM,
            "device": "cpu",
            "batch_size": 32,
            "normalize_embeddings": False,
            "output_dtype": "float32",
        },
        "embedding_identity_sha256": _digest(f"embedding-{shard_index}"),
        "build_runtime_identity_sha256": _digest(f"runtime-{shard_index}"),
        "implementation_sha256": _digest(f"source-impl-{shard_index}"),
        "environment_lock_sha256": _digest(f"source-env-{shard_index}"),
        "query_input_key": query_key,
        "selected_query_entry": f"query-inputs/{query_key}",
        "query_manifest_sha256": _digest(f"query-manifest-{shard_index}"),
        "query_artifact_sha256": _digest(f"query-artifact-{shard_index}"),
    }
    body["receipt_sha256"] = identity_sha256(body)
    return body


def _combined_receipt(
    source: dict[str, object],
    *,
    retrieval_policy_sha256: str,
    shard_index: int,
) -> CombinedCumulativeStoreReceipt:
    store_identity = _digest(f"store-identity-{shard_index}")
    return CombinedCumulativeStoreReceipt(
        source_store_identity_sha256=store_identity,
        target_store_identity_sha256=store_identity,
        source_database_sha256=str(source["database_sha256"]),
        target_database_sha256=_digest(f"target-db-{shard_index}"),
        target_index_sha256=_digest(f"target-index-{shard_index}"),
        retrieval_policy_sha256=retrieval_policy_sha256,
        context_budget_sha256=_digest(f"budget-{shard_index}"),
        training_query_batch_sha256=_digest(f"training-{shard_index}"),
        held_out_query_batch_sha256=_digest(f"held-out-{shard_index}"),
        compilation_receipt_sha256=_digest(f"compilation-{shard_index}"),
        artifact_id=f"artifact-{shard_index}",
        snapshot_sha256=_digest(f"snapshot-{shard_index}"),
        turn_count=1,
        chunk_count=1,
        causal_events=1,
        causal_graph_edges=1,
    )


def _sealed_question(
    question: BenchmarkQuestion,
    *,
    context: ValidationShardPreflight,
    local_ordinal: int,
    source_sha: str,
    combined: CombinedCumulativeStoreReceipt,
) -> dict[str, object]:
    messages = [
        {"role": "system", "content": "Answer only from the excerpts."},
        {
            "role": "user",
            "content": (
                "Retrieved excerpts:\nA code was recorded.\n\nQuestion: "
                f"{question.dated_question}\nShort answer:"
            ),
        },
    ]
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    evidence = [
        {"evidence_id": "e0", "source_id": "s0", "text": "A record."},
        {"evidence_id": "e1", "source_id": "s1", "text": "The code is blue."},
    ]
    typed_stages: list[CumulativeRetrievalStageReceipt] = []
    stages: list[dict[str, object]] = []
    for index, stage_id in enumerate(shard_impl.STAGE_IDS):
        selected = ("e0",) if index == 0 else ("e0", "e1")
        parent = () if index == 0 else typed_stages[-1].selected_evidence_ids
        added = selected if index == 0 else selected[len(parent) :]
        status = "root" if index == 0 else "added" if added else "no_novel_evidence"
        receipt = CumulativeRetrievalStageReceipt(
            stage_id=stage_id,
            matched_controls_sha256=_H,
            method_evidence_sha256=_digest(f"method-{stage_id}"),
            parent_stage_receipt_sha256=(
                None if index == 0 else typed_stages[-1].receipt_sha256
            ),
            parent_evidence_ids=parent,
            selected_evidence_ids=selected,
            added_evidence_ids=added,
            admission_status=status,
            evidence_projection_sha256=_digest(f"projection-{stage_id}"),
            context_sha256=_digest(f"context-{stage_id}"),
            prompt_messages_sha256=identity_sha256(messages),
            context_token_proxy=2,
            max_context_token_proxy=7_000,
            prompt_token_proxy=prompt_tokens,
            max_prompt_token_proxy=8_000,
            responder_output_token_reserve=256,
        )
        typed_stages.append(receipt)
        stages.append(
            {
                "stage_id": stage_id,
                "stage_receipt": asdict(receipt),
                "provider_messages": copy.deepcopy(messages),
                "evidence": copy.deepcopy(evidence[: len(selected)]),
            }
        )
    ladder = CumulativeRetrievalLadder(stages=tuple(typed_stages))
    predecessor = CausalCoveragePredecessorReceipt(
        matched_controls_sha256=_H,
        retrieval_query_sha256=quote_sha256(question.question),
        prompt_question_sha256=quote_sha256(question.dated_question),
        retrieval_policy_sha256=context.policy.retrieval_policy_sha256,
        context_budget_sha256=_H,
        raw_graph_anchor_sequence_sha256=_H,
        raw_graph_chunk_ids=("c0",),
        packed_chunk_ids=("c0",),
        protected_chunk_ids=("c0",),
        direct_protected_chunk_ids=("c0",),
        protected_excerpt_projection_sha256=_H,
        protected_context_sha256=_H,
        selected_anchor_sequence_sha256=_H,
        coverage_selector_report_sha256=_H,
        coverage_candidate_trace_sha256=_H,
        coverage_runtime_certified=True,
        packed_token_counts=(),
        packed_dropped_counts=(),
        prompt_messages_sha256=typed_stages[0].prompt_messages_sha256,
        prompt_token_proxy=prompt_tokens,
        max_prompt_token_proxy=8_000,
        responder_output_token_reserve=256,
    )
    final = RecallGuardedCumulativeReceipt(
        matched_controls_sha256=_H,
        predecessor_receipt_sha256=predecessor.receipt_sha256,
        direct_expansion_receipt_sha256=_H,
        representative_expansion_receipt_sha256=_H,
        closure_plan_sha256s=(_H, _H, _H),
        novel_projection_receipt_sha256s=(_H, _H, _H),
        addition_packet_receipt_sha256s=(_H, None, None),
        stage_admission_statuses=("added", "no_novel_evidence", "no_novel_evidence"),
        ladder_receipt_sha256=ladder.receipt_sha256,
        representative_runtime_certified=True,
        protected_chunk_ids=("c0",),
        protected_evidence_ids=("e0",),
        added_atom_ids=("e1",),
        added_chunk_ids=("c1",),
        final_chunk_ids=("c0", "c1"),
        final_evidence_ids=("e0", "e1"),
        protected_excerpt_projection_sha256=_H,
        addition_evidence_projection_sha256=_H,
        final_context_sha256=_H,
        prompt_messages_sha256=typed_stages[-1].prompt_messages_sha256,
        context_token_proxy=2,
        max_context_token_proxy=7_000,
        prompt_token_proxy=prompt_tokens,
        max_prompt_token_proxy=8_000,
        responder_output_token_reserve=256,
        prompt_workspace_token_proxy=prompt_tokens + 256,
    )
    probe = context.shard_identity["ordered_question_probes"][local_ordinal]
    return {
        "format": VALIDATION_SHARD_QUESTION_FORMAT,
        "population_identity_sha256": context.population_identity[
            "population_identity_sha256"
        ],
        "shard_identity_sha256": context.shard_identity[
            "shard_identity_sha256"
        ],
        "shard_offset": context.sample_offset,
        "local_ordinal": local_ordinal,
        "ordinal": context.sample_offset + local_ordinal,
        "question_id": question.question_id,
        "question_id_sha256": identity_sha256(
            {"question_id": question.question_id}
        ),
        "question_sha256": quote_sha256(question.question),
        "dated_question_sha256": quote_sha256(question.dated_question),
        "probe_identity_sha256": probe["probe_identity_sha256"],
        "validation_policy_manifest_sha256": (
            LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
        ),
        "validation_policy_attestation_sha256": context.policy.attestation_sha256,
        "validation_execution_policy_sha256": context.policy.execution_policy_sha256,
        "retrieval_policy_sha256": context.policy.retrieval_policy_sha256,
        "retrieval_implementation_sha256": context.retrieval_implementation_sha256,
        "environment_lock_sha256": context.environment_lock_sha256,
        "source_store_receipt_sha256": source_sha,
        "combined_store_receipt_sha256": combined.receipt_sha256,
        "compilation_receipt_sha256": combined.compilation_receipt_sha256,
        "retrieval_receipt": asdict(final),
        "predecessor_receipt": asdict(predecessor),
        "stage_ids": list(shard_impl.STAGE_IDS),
        "stages": stages,
        "elapsed_seconds": 0.25,
        "provider_calls": 0,
    }


def _contexts_and_artifacts(
    tmp_path: Path,
    *,
    implementation: str = "d" * 64,
    environment: str = "e" * 64,
):
    samples = _samples()
    shards, population = _population(samples)
    policy = load_frozen_validation_policy(POLICY, device="cpu")
    contexts: list[ValidationShardPreflight] = []
    artifacts: list[dict[str, object]] = []
    for index, (offset, sample, shard) in enumerate(
        zip(LOCKED_100Q_OFFSETS, samples, shards, strict=True)
    ):
        context = ValidationShardPreflight(
            sample=sample,
            shard_identity=shard,
            population_identity=population,
            policy=policy,
            sample_offset=offset,
            shard_root=tmp_path / f"offset-{offset:03d}",
            qwen_prefix_model_dir=tmp_path,
            qwen_choice_model_dir=tmp_path,
            retrieval_implementation_sha256=implementation,
            environment_lock_sha256=environment,
            source_embedding_device="cpu",
        )
        source = _source_receipt(sample, shard_index=index)
        combined = _combined_receipt(
            source,
            retrieval_policy_sha256=policy.retrieval_policy_sha256,
            shard_index=index,
        )
        questions = [
            _sealed_question(
                question,
                context=context,
                local_ordinal=local,
                source_sha=str(source["receipt_sha256"]),
                combined=combined,
            )
            for local, question in enumerate(sample.questions)
        ]
        artifact: dict[str, object] = {
            "format": VALIDATION_SHARD_RETRIEVAL_FORMAT,
            "campaign_format": shard_impl.VALIDATION_CAMPAIGN_FORMAT,
            "population_identity": population,
            "population_identity_sha256": population["population_identity_sha256"],
            "shard_identity": shard,
            "shard_identity_sha256": shard["shard_identity_sha256"],
            "shard_offset": offset,
            "validation_policy_attestation": dict(policy.attestation),
            "validation_policy_attestation_sha256": policy.attestation_sha256,
            "validation_policy_manifest_sha256": (
                LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
            ),
            "validation_execution_policy": dict(policy.execution_policy),
            "validation_execution_policy_sha256": policy.execution_policy_sha256,
            "retrieval_policy_sha256": policy.retrieval_policy_sha256,
            "retrieval_implementation_sha256": implementation,
            "environment_lock_sha256": environment,
            "source_embedding_device": "cpu",
            "source_timestamp_semantics": (
                shard_impl.CURRENT_SOURCE_TIMESTAMP_SEMANTICS
            ),
            "source_store_mode": "verified_cache_hit",
            "source_store_receipt": source,
            "source_store_receipt_sha256": source["receipt_sha256"],
            "combined_store_mode": "verified_cache_hit",
            "combined_store_receipt": asdict(combined),
            "combined_store_receipt_sha256": combined.receipt_sha256,
            "compilation_receipt_sha256": combined.compilation_receipt_sha256,
            "transcript_tokens": shard["transcript_tokens"],
            "turn_count": shard["turn_count"],
            "question_count": 10,
            "stage_ids": list(shard_impl.STAGE_IDS),
            "question_part_sha256s": [
                hashlib.sha256(shard_impl._canonical_json_bytes(row)).hexdigest()
                for row in questions
            ],
            "questions": questions,
            "provider_calls": 0,
            "gold_fields_present": False,
        }
        contexts.append(context)
        artifacts.append(artifact)
    return samples, shards, population, policy, contexts, artifacts


def test_loads_only_the_exact_frozen_validation_policy(tmp_path: Path) -> None:
    policy = load_frozen_validation_policy(POLICY, device="cpu")

    assert policy.attestation["manifest_sha256"] == (
        LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
    )
    assert policy.config.min_target_questions == 100
    assert policy.config.max_prompt_tokens == 8_000
    assert policy.execution_policy["population_plan"]["ordered_shard_offsets"] == list(
        LOCKED_100Q_OFFSETS
    )

    changed = tmp_path / "changed-policy.json"
    changed.write_bytes(POLICY.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="policy SHA-256 mismatch"):
        load_frozen_validation_policy(changed, device="cpu")


def test_preflight_reconstructs_identity_without_touching_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples()
    shards, population = _population(samples)
    monkeypatch.setattr(
        shard_impl,
        "build_locked_cumulative_population_identity",
        lambda *_args, **_kwargs: (samples, shards, population),
    )
    monkeypatch.setattr(
        shard_impl,
        "current_source_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight must not initialize model runtime")
        ),
    )

    result = preflight_locked_validation_shard(
        dataset_path=tmp_path / "not-read-because-reconstruction-is-injected.json",
        split_manifest_path=tmp_path / "not-read.json",
        policy_path=POLICY,
        output_root=tmp_path / "output",
        sample_offset=30,
        qwen_prefix_model_dir=tmp_path,
        qwen_choice_model_dir=tmp_path,
        device="cpu",
    )

    assert result.sample is samples[3]
    assert result.public_report()["sample_offset"] == 30
    assert result.public_report()["question_count"] == 10
    assert not (tmp_path / "output").exists()


def test_shard_validator_crossbinds_policy_population_store_and_parts(
    tmp_path: Path,
) -> None:
    _samples_value, _shards, _population_value, _policy, contexts, artifacts = (
        _contexts_and_artifacts(tmp_path)
    )
    validate_validation_shard_retrieval(artifacts[0], preflight=contexts[0])
    encoded = json.dumps(artifacts[0], sort_keys=True)
    assert "secret-000" not in encoded
    assert "gold-000" not in encoded

    changed = copy.deepcopy(artifacts[0])
    changed["questions"][0]["combined_store_receipt_sha256"] = "f" * 64
    changed["question_part_sha256s"][0] = hashlib.sha256(
        shard_impl._canonical_json_bytes(changed["questions"][0])
    ).hexdigest()
    with pytest.raises(ValueError, match="another shard/campaign"):
        validate_validation_shard_retrieval(changed, preflight=contexts[0])


def test_strict_ten_shard_merge_publishes_self_contained_100q_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation, environment = "d" * 64, "e" * 64
    samples, shards, population, _policy, _contexts, artifacts = (
        _contexts_and_artifacts(
            tmp_path,
            implementation=implementation,
            environment=environment,
        )
    )
    paths: list[Path] = []
    for offset, artifact in zip(LOCKED_100Q_OFFSETS, artifacts, strict=True):
        path = tmp_path / f"offset-{offset:03d}" / "retrieval.json"
        shard_impl._atomic_write_json(path, artifact)
        paths.append(path)
    monkeypatch.setattr(
        merge_impl,
        "build_locked_cumulative_population_identity",
        lambda *_args, **_kwargs: (samples, shards, population),
    )
    monkeypatch.setattr(
        merge_impl,
        "merge_locked_cumulative_shard_identities",
        lambda *_args, **_kwargs: population,
    )
    monkeypatch.setattr(merge_impl, "implementation_sha256", lambda: implementation)
    monkeypatch.setattr(merge_impl, "environment_lock_sha256", lambda: environment)

    merged, digest = merge_locked_validation_retrievals(
        dataset_path=tmp_path / "dataset.json",
        split_manifest_path=tmp_path / "split.json",
        policy_path=POLICY,
        output_root=tmp_path,
        output_path=tmp_path / "retrieval.json",
        shard_retrieval_paths=paths,
        device="cpu",
    )

    assert merged["format"] == VALIDATION_MERGED_RETRIEVAL_FORMAT
    assert merged["question_count"] == 100
    assert merged["external_reconstruction_receipt"]["format"] == (
        VALIDATION_EXTERNAL_RECONSTRUCTION_FORMAT
    )
    assert len(merged_question_store_receipts(merged)) == 100
    assert (tmp_path / "retrieval.json.sha256").read_text(
        encoding="ascii"
    ).startswith(digest)
    validate_merged_validation_retrieval(merged)

    changed = copy.deepcopy(merged)
    changed["questions"][0]["source_shard_retrieval_sha256"] = "f" * 64
    changed["question_part_sha256s"][0] = hashlib.sha256(
        shard_impl._canonical_json_bytes(changed["questions"][0])
    ).hexdigest()
    with pytest.raises(ValueError, match="cross-binding"):
        validate_merged_validation_retrieval(changed)

    with pytest.raises(ValueError, match="another campaign"):
        merge_locked_validation_retrievals(
            dataset_path=tmp_path / "dataset.json",
            split_manifest_path=tmp_path / "split.json",
            policy_path=POLICY,
            output_root=tmp_path,
            output_path=tmp_path / "swapped.json",
            shard_retrieval_paths=[paths[1], paths[0], *paths[2:]],
            device="cpu",
        )


def test_merge_phase_does_not_enter_gpu_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "memory_condense.eval.recall_guarded_cumulative_validation_retrieval."
        "merge_locked_validation_retrievals",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "memory_condense.eval.recall_guarded_cumulative_validation_retrieval."
        "preflight_locked_validation_shard",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("merge must not enter shard/model preflight")
        ),
    )

    assert main(
        [
            "--phase",
            "merge",
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--output-root",
            str(tmp_path / "output"),
        ]
    ) == 0
    assert len(calls) == 1
