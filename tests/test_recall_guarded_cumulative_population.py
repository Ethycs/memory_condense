from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from memory_condense.domain.discourse import identity_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.context_stress import transcript_tokens
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_100Q_OFFSETS,
    LockedCumulativePopulationPlan,
    build_locked_cumulative_population_identity,
    merge_locked_cumulative_shard_identities,
    reconstruct_locked_cumulative_shard,
    validate_locked_cumulative_population_identity,
    validate_locked_cumulative_shard_identity,
)
from memory_condense.ingest.loader import load_benchmark


def _locked_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, LockedCumulativePopulationPlan]:
    records = []
    for ordinal in range(100):
        records.append(
            {
                "question_id": f"question-{ordinal:03d}",
                "question_type": "single-session-user",
                "question": f"What is private value {ordinal:03d}?",
                "answer": f"secret-answer-{ordinal:03d}",
                "question_date": "2026/01/02 03:04",
                "haystack_dates": ["2026/01/01 01:02"],
                "haystack_session_ids": ["session"],
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": (
                                "alpha beta gamma delta epsilon zeta eta theta"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": "iota kappa lambda mu nu xi omicron pi",
                        },
                    ]
                ],
                "answer_session_ids": ["session"],
            }
        )
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps(records), encoding="utf-8")
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "memory-condense-locked-benchmark-split-v1",
                "dataset_sha256": file_sha256(dataset),
                "salt": "cumulative-population-test",
                "algorithm": "stratified-largest-remainder-v1",
                "splits": {"validation": 100},
            }
        ),
        encoding="utf-8",
    )
    one_sample_tokens = transcript_tokens(load_benchmark(dataset, "longmemeval")[0])
    plan = LockedCumulativePopulationPlan(
        dataset_sha256=file_sha256(dataset),
        split_manifest_sha256=file_sha256(manifest),
        split="validation",
        target_tokens=one_sample_tokens * 10,
        shard_offsets=LOCKED_100Q_OFFSETS,
    )
    return dataset, manifest, plan


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in _all_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def test_reconstructs_arbitrary_hash_locked_shard_with_gold_blind_identity(
    tmp_path: Path,
):
    dataset, manifest, plan = _locked_fixture(tmp_path)

    sample, identity = reconstruct_locked_cumulative_shard(
        dataset,
        manifest,
        dataset_sha256=plan.dataset_sha256,
        split_manifest_sha256=plan.split_manifest_sha256,
        split_name=plan.split,
        sample_offset=30,
        target_tokens=plan.target_tokens,
    )

    assert len(sample.questions) == 10
    assert identity["construction"] == {
        "target_tokens": plan.target_tokens,
        "questions_per_shard": 10,
        "sample_offset": 30,
    }
    assert identity["question_count"] == 10
    assert identity["gold_fields_present"] is False
    assert validate_locked_cumulative_shard_identity(identity) == identity
    encoded = json.dumps(identity, sort_keys=True)
    assert "secret-answer" not in encoded
    assert "What is private value" not in encoded
    for question in sample.questions:
        assert question.question_id not in encoded
        assert question.question not in encoded
        assert question.answer not in encoded
    assert not (
        _all_keys(identity)
        & {"answer", "category", "evidence", "evidence_sources", "question"}
    )


def test_reconstruction_rejects_dataset_and_split_digest_drift(tmp_path: Path):
    dataset, manifest, plan = _locked_fixture(tmp_path)
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        reconstruct_locked_cumulative_shard(
            dataset,
            manifest,
            dataset_sha256=plan.dataset_sha256,
            split_manifest_sha256=plan.split_manifest_sha256,
            split_name=plan.split,
            sample_offset=0,
            target_tokens=plan.target_tokens,
        )

    dataset, manifest, plan = _locked_fixture(tmp_path)
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split-manifest SHA-256 mismatch"):
        reconstruct_locked_cumulative_shard(
            dataset,
            manifest,
            dataset_sha256=plan.dataset_sha256,
            split_manifest_sha256=plan.split_manifest_sha256,
            split_name=plan.split,
            sample_offset=0,
            target_tokens=plan.target_tokens,
        )


def test_builds_exact_ordered_ten_shard_hundred_question_population(tmp_path: Path):
    dataset, manifest, plan = _locked_fixture(tmp_path)

    samples, shards, population = build_locked_cumulative_population_identity(
        dataset,
        manifest,
        plan=plan,
    )

    assert len(samples) == len(shards) == 10
    assert [row["construction"]["sample_offset"] for row in shards] == list(
        LOCKED_100Q_OFFSETS
    )
    assert population["shard_count"] == 10
    assert population["question_count"] == 100
    assert population["total_turn_count"] == sum(
        len(sample.turns) for sample in samples
    )
    assert population["total_transcript_tokens"] == sum(
        transcript_tokens(sample) for sample in samples
    )
    assert len(set(population["ordered_question_id_sha256s"])) == 100
    assert len(set(population["ordered_question_probe_sha256s"])) == 100
    assert validate_locked_cumulative_population_identity(
        population,
        plan=plan,
    ) == population
    assert merge_locked_cumulative_shard_identities(
        shards,
        dataset_path=dataset,
        split_manifest_path=manifest,
        plan=plan,
    ) == population


def test_merge_rejects_reorder_omission_and_self_consistent_forgery(tmp_path: Path):
    dataset, manifest, plan = _locked_fixture(tmp_path)
    _samples, shards, _population = build_locked_cumulative_population_identity(
        dataset,
        manifest,
        plan=plan,
    )

    reordered = list(shards)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="differs from locked reconstruction"):
        merge_locked_cumulative_shard_identities(
            reordered,
            dataset_path=dataset,
            split_manifest_path=manifest,
            plan=plan,
        )

    with pytest.raises(ValueError, match="exactly ten"):
        merge_locked_cumulative_shard_identities(
            shards[:-1],
            dataset_path=dataset,
            split_manifest_path=manifest,
            plan=plan,
        )

    forged = [copy.deepcopy(row) for row in shards]
    forged[0]["turn_count"] += 1
    body = {
        key: child
        for key, child in forged[0].items()
        if key != "shard_identity_sha256"
    }
    forged[0]["shard_identity_sha256"] = identity_sha256(body)
    assert validate_locked_cumulative_shard_identity(forged[0]) == forged[0]
    with pytest.raises(ValueError, match="differs from locked reconstruction"):
        merge_locked_cumulative_shard_identities(
            forged,
            dataset_path=dataset,
            split_manifest_path=manifest,
            plan=plan,
        )


def test_closed_schemas_reject_unbound_fields_and_invalid_plan(tmp_path: Path):
    dataset, manifest, plan = _locked_fixture(tmp_path)
    _sample, shard = reconstruct_locked_cumulative_shard(
        dataset,
        manifest,
        dataset_sha256=plan.dataset_sha256,
        split_manifest_sha256=plan.split_manifest_sha256,
        split_name=plan.split,
        sample_offset=0,
        target_tokens=plan.target_tokens,
    )
    shard["answer"] = "secret"
    with pytest.raises(ValueError, match="unexpected schema"):
        validate_locked_cumulative_shard_identity(shard)

    with pytest.raises(ValueError, match="exact offsets"):
        LockedCumulativePopulationPlan(
            dataset_sha256=plan.dataset_sha256,
            split_manifest_sha256=plan.split_manifest_sha256,
            split="validation",
            target_tokens=plan.target_tokens,
            shard_offsets=(0,) * 10,
        )
