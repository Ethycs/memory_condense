from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from memory_condense.eval.fast_hebbian_h2 import (
    FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS,
    FAST_HEBBIAN_H2_STAGE_ID,
    FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES,
    FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE,
    FastHebbianH2HistorySource,
    FastHebbianH2Policy,
    FastHebbianH2ValidationError,
    build_fast_hebbian_h2_consumer_source_manifest,
    build_fast_hebbian_h2_population,
    load_fast_hebbian_h2_history,
    load_fast_hebbian_h2_retrieval_source,
)
from memory_condense.eval.fast_hebbian_prompts import (
    ARM_IDS,
    MAX_PROMPT_TOKEN_INCREASE,
    S0_STAGE_ID,
)
from memory_condense.eval.hebbian_derived_store import (
    apply_hebbian_history_to_staged_store,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import STAGE_IDS
from memory_condense.eval.reproducibility import environment_lock_sha256
from memory_condense.persistence.db import Database
from tests.test_fast_hebbian_prompts import (
    _build as _build_h1,
    _derived_store as _h1_derived_store,
    _fast_artifact,
)
from tests.test_hebbian_derived_store import (
    SOURCE_RECEIPT_SHA256,
    _build_store,
    _copy_stage,
    _history,
    _remove_zero_wal_sidecars,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _publish(path: Path, value: object) -> str:
    raw = _canonical_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return digest


def _events():
    from memory_condense.eval.consolidation_replay import RetrievalAccessEvent

    return [
        RetrievalAccessEvent(
            "causal-user:3",
            2,
            ("chunk-0", "chunk-1"),
        ),
        RetrievalAccessEvent(
            "causal-user:5",
            4,
            ("chunk-0", "chunk-1", "chunk-2", "chunk-3"),
        ),
        RetrievalAccessEvent(
            "causal-user:14",
            13,
            ("chunk-0", "chunk-1", "chunk-2", "chunk-3"),
        ),
    ]


def _fixture(
    tmp_path: Path,
    *,
    text_suffix: str = "",
):
    source = _build_store(tmp_path / "source")
    if text_suffix:
        oversized = f"SECRET source candidate{text_suffix}"
        with Database(source / "memory.db") as database:
            database.execute(
                "UPDATE chunks SET text = ?, end_char = ?, token_count = ? "
                "WHERE chunk_id IN ('chunk-2', 'chunk-3')",
                (oversized, len(oversized), len(oversized.split())),
            )
            database.commit()
            checkpoint = database.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            assert checkpoint is not None and int(checkpoint[0]) == 0
        _remove_zero_wal_sidecars(source / "memory.db")
    history = _history(source, events=_events())
    derived_root = _copy_stage(source, tmp_path / "derived")
    derived = apply_hebbian_history_to_staged_store(
        derived_root,
        source_database_path=source / "memory.db",
        source_index_path=source / "hnsw_index.bin",
        history=history,
    )
    history_path = tmp_path / "history.json"
    history_sha = _publish(history_path, history.payload())
    history_source = load_fast_hebbian_h2_history(
        history_path,
        expected_sha256=history_sha,
    )

    artifact = _fast_artifact(
        protected_chunk_ids=("chunk-0", "chunk-1"),
    )
    artifact = replace(
        artifact,
        combined_store_receipt_sha256=SOURCE_RECEIPT_SHA256,
        turn_count=14,
    )
    question = artifact.questions[0]
    raw_retrieval = {
        "questions": [
            {
                "question_id": question.question_id,
                "retrieval_receipt": {
                    "final_evidence_ids": list(
                        question.stage(STAGE_IDS[-1]).evidence_ids
                    ),
                    "final_chunk_ids": ["chunk-0", "chunk-1"],
                    "receipt_sha256": question.retrieval_receipt_sha256,
                },
            }
        ]
    }
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_sha = _publish(retrieval_path, raw_retrieval)
    artifact = replace(artifact, raw_sha256=retrieval_sha)
    retrieval_source = load_fast_hebbian_h2_retrieval_source(
        retrieval_path,
        artifact=artifact,
    )
    return artifact, retrieval_source, history_source, derived_root, derived


def _build(tmp_path: Path, *, text_suffix: str = "", policy=None):
    artifact, retrieval, history, derived_root, derived = _fixture(
        tmp_path,
        text_suffix=text_suffix,
    )
    population = build_fast_hebbian_h2_population(
        artifact,
        retrieval,
        history,
        derived_root,
        policy=policy,
    )
    return artifact, history, derived, population


def test_h2_is_monotonic_s3_append_with_robust_separate_provenance(
    tmp_path: Path,
) -> None:
    artifact, history, derived, population = _build(tmp_path)
    receipt = population.question_receipts[0]
    base = artifact.questions[0].stage(STAGE_IDS[-1]).evidence
    final = population.final_evidence[0]

    assert FAST_HEBBIAN_H2_STAGE_ID == STAGE_IDS[-1]
    assert final[: len(base)] == base
    assert len(final) == len(base) + 1
    assert receipt.appended_source_chunk_ids == ("chunk-2",)
    assert receipt.ranked_candidates[0].support == 2
    assert receipt.ranked_candidates[0].coaccess_count == 4
    assert receipt.ranked_candidates[0].admission_status == "appended"
    assert receipt.ranked_candidates[1].admission_status == (
        "addition_cap_rejected"
    )
    assert receipt.final_prompt_token_proxy <= FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS
    assert population.appended_question_count == 1
    assert population.appended_evidence_count == 1
    assert population.gold_fields_consumed is False
    assert population.provider_calls == 0
    assert population.cav_links_computed is False
    assert population.history_producer_implementation_sha256 == (
        history.artifact.receipt.implementation_sha256
    )
    assert population.derived_store_producer_implementation_sha256 == (
        derived.implementation_sha256
    )
    assert population.h2_consumer_source_sha256 == (
        population.h2_consumer_source_manifest.source_sha256
    )
    assert population.h2_consumer_environment_lock_sha256 == (
        environment_lock_sha256()
    )
    assert receipt.h2_consumer_environment_lock_sha256 == (
        population.h2_consumer_environment_lock_sha256
    )
    manifest = population.h2_consumer_source_manifest
    paths = tuple(row.path for row in manifest.files)
    assert manifest.root_modules == FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES
    assert manifest.scope == FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE
    assert paths == tuple(sorted(paths))
    assert {
        "src/memory_condense/domain/_discourse_identity.py",
        "src/memory_condense/domain/schemas.py",
        "src/memory_condense/associations/association_models.py",
        "src/memory_condense/associations/association_repository.py",
        "src/memory_condense/domain/decay.py",
        "src/memory_condense/eval/fast_cav_feature_session.py",
    }.issubset(paths)
    assert population.history_producer_implementation_sha256 != (
        population.h2_consumer_source_sha256
    )


def test_h2_ranking_and_population_are_deterministic(tmp_path: Path) -> None:
    artifact, retrieval, history, derived_root, _derived = _fixture(tmp_path)
    kwargs = {
        "policy": FastHebbianH2Policy(),
    }

    first = build_fast_hebbian_h2_population(
        artifact, retrieval, history, derived_root, **kwargs
    )
    second = build_fast_hebbian_h2_population(
        artifact, retrieval, history, derived_root, **kwargs
    )

    first_candidates = first.question_receipts[0].ranked_candidates
    second_candidates = second.question_receipts[0].ranked_candidates
    assert tuple(row.source_chunk_id for row in first_candidates) == (
        "chunk-2",
        "chunk-3",
    )
    assert first_candidates == second_candidates
    assert first.population_sha256 == second.population_sha256


def test_h2_rejects_candidate_that_breaks_downstream_8k_budget(
    tmp_path: Path,
) -> None:
    _artifact, _history_source, _derived, population = _build(
        tmp_path,
        text_suffix=" oversized" * 9_000,
        policy=FastHebbianH2Policy(max_neighbor_candidates=2),
    )
    receipt = population.question_receipts[0]

    assert receipt.outcome == "no_budget_admissible_candidate"
    assert receipt.appended_evidence_ids == ()
    assert all(
        row.admission_status == "budget_rejected"
        for row in receipt.ranked_candidates
    )
    assert receipt.final_coordinates == receipt.base_s3_coordinates
    assert receipt.final_scaffold_sha256 == receipt.base_scaffold_sha256
    assert receipt.final_prompt_token_proxy == receipt.base_prompt_token_proxy
    assert all(
        row.proposed_prompt_token_proxy > FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS
        for row in receipt.ranked_candidates
    )


def test_h2_rejects_resealed_receipt_and_history_tampering(tmp_path: Path) -> None:
    _artifact, history, _derived, population = _build(tmp_path)
    receipt = population.question_receipts[0]

    with pytest.raises(ValueError, match="appended tail|seal"):
        replace(receipt, appended_evidence_ids=("forged",))
    with pytest.raises(ValueError, match="receipt|contents"):
        replace(receipt, h2_consumer_environment_lock_sha256="0" * 64)
    changed_history = replace(
        history.artifact,
        artifact_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="artifact seal"):
        FastHebbianH2HistorySource(
            source_path=history.source_path,
            raw_sha256=history.raw_sha256,
            artifact=changed_history,
        )


def test_legacy_h1_contract_and_builder_are_unchanged(tmp_path: Path) -> None:
    artifact = _fast_artifact()
    store, association_id, receipt_sha = _h1_derived_store(tmp_path, artifact)

    population = _build_h1(artifact, store, association_id, receipt_sha)

    assert ARM_IDS == ("base", "h1")
    assert S0_STAGE_ID == STAGE_IDS[0]
    assert MAX_PROMPT_TOKEN_INCREASE == 0
    assert population.stage_id == S0_STAGE_ID
    assert tuple(row.arm_id for row in population.logical_prompts) == ARM_IDS
    assert population.question_receipts[0].effective_status == "replaced"
    assert population.logical_prompts[1].chunk_ids[-1] == "chunk-neighbor"


def test_h2_policy_fails_closed_on_weaker_than_robust_thresholds() -> None:
    with pytest.raises(FastHebbianH2ValidationError, match="min_support"):
        FastHebbianH2Policy(min_support=1)
    with pytest.raises(FastHebbianH2ValidationError, match="min_coaccess_count"):
        FastHebbianH2Policy(min_coaccess_count=1)


def test_h2_s3_atom_and_source_chunk_cardinalities_are_independent(
    tmp_path: Path,
) -> None:
    artifact = _fast_artifact(protected_chunk_ids=("chunk-0", "chunk-1"))
    question = artifact.questions[0]
    evidence_ids = question.stage(STAGE_IDS[-1]).evidence_ids
    source_chunk_ids = ["chunk-0", "chunk-1", "chunk-extra"]
    assert len(evidence_ids) != len(source_chunk_ids)
    raw_retrieval = {
        "questions": [
            {
                "question_id": question.question_id,
                "retrieval_receipt": {
                    "final_evidence_ids": list(evidence_ids),
                    "final_chunk_ids": source_chunk_ids,
                    "receipt_sha256": question.retrieval_receipt_sha256,
                },
            }
        ]
    }
    retrieval_path = tmp_path / "unequal-cardinality-retrieval.json"
    retrieval_sha = _publish(retrieval_path, raw_retrieval)
    artifact = replace(artifact, raw_sha256=retrieval_sha)

    source = load_fast_hebbian_h2_retrieval_source(
        retrieval_path,
        artifact=artifact,
    )

    assert source.questions[0].s3_evidence_ids == evidence_ids
    assert source.questions[0].s3_source_chunk_ids == tuple(source_chunk_ids)


def test_h2_consumer_identity_ignores_unrelated_files_but_tracks_scope(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_root = tmp_path / "consumer-source"
    repository_manifest = build_fast_hebbian_h2_consumer_source_manifest()
    for row in repository_manifest.files:
        destination = source_root.joinpath(*row.path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / row.path, destination)

    baseline = build_fast_hebbian_h2_consumer_source_manifest(source_root)
    assert baseline == repository_manifest
    unrelated = source_root / "src/memory_condense/eval/unrelated_future_runner.py"
    unrelated.write_text("UNRELATED = True\n", encoding="utf-8")
    after_unrelated = build_fast_hebbian_h2_consumer_source_manifest(source_root)

    assert after_unrelated == baseline
    relevant = source_root.joinpath(*baseline.files[0].path.split("/"))
    relevant.write_bytes(relevant.read_bytes() + b"\n# relevant test change\n")
    after_relevant = build_fast_hebbian_h2_consumer_source_manifest(source_root)

    assert after_relevant.source_sha256 != baseline.source_sha256
    assert after_relevant.manifest_sha256 != baseline.manifest_sha256
    assert after_relevant.files[0].file_sha256 != baseline.files[0].file_sha256
    assert after_relevant.files[1:] == baseline.files[1:]

    shutil.copy2(repository / baseline.files[0].path, relevant)
    synthetic_relative = "src/memory_condense/eval/synthetic_h2_dependency.py"
    synthetic = source_root.joinpath(*synthetic_relative.split("/"))
    synthetic.write_text("SYNTHETIC = True\n", encoding="utf-8")
    root_relative = (
        "src/" + FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES[0].replace(".", "/") + ".py"
    )
    root_module = source_root.joinpath(*root_relative.split("/"))
    root_module.write_bytes(
        root_module.read_bytes()
        + b"\nimport memory_condense.eval.synthetic_h2_dependency\n"
    )
    expanded = build_fast_hebbian_h2_consumer_source_manifest(source_root)

    assert synthetic_relative in {row.path for row in expanded.files}
    assert expanded.source_sha256 != baseline.source_sha256
