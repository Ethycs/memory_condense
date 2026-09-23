from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools import assay_qwen_minilm_distillation as assay


@pytest.mark.parametrize("key", ["answer", "gold", "has_answer", "is_answer"])
def test_recursive_firewall_rejects_nested_truth_keys(key: str) -> None:
    with pytest.raises(assay.AssayError, match="forbidden gold-bearing key"):
        assay._assert_gold_free({"safe": [{"deeper": {key: True}}]})


def test_projection_is_deterministic_disjoint_and_parent_hash_honest(
    tmp_path: Path,
) -> None:
    records = []
    for index in range(500):
        records.append(
            {
                "question_id": f"q-{index:03d}",
                "question_type": f"type-{index % 6}",
                "question": f"Which event concerns topic {index % 23}?",
                "answer": f"hidden-{index}",
                "question_date": f"2026/01/{index % 28 + 1:02d}",
                "answer_session_ids": [f"raw-source-{index}"],
                "haystack_session_ids": [f"raw-source-{index}"],
                "haystack_dates": [f"2025/12/{index % 28 + 1:02d}"],
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": f"memory event {index} topic {index % 23}",
                            "has_answer": True,
                        }
                    ]
                ],
            }
        )
    dataset = tmp_path / "oracle.json"
    dataset.write_text(json.dumps(records), encoding="utf-8")
    manifest = tmp_path / "split.json"
    parent_sha = "a" * 64
    manifest.write_text(
        json.dumps(
            {
                "format": assay.LOCKED_SPLIT_FORMAT,
                "dataset_sha256": parent_sha,
                "salt": "locked-test-salt",
                "splits": {
                    "development": 200,
                    "validation": 100,
                    "confirmation": 200,
                },
                "algorithm": assay.LOCKED_SPLIT_ALGORITHM,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        dataset=dataset,
        split_manifest=manifest,
        candidates=16,
        lexical=12,
        distractors=4,
        candidate_tokens=48,
        query_tokens=48,
    )
    first = assay._project(args)
    second = assay._project(args)
    assert assay._canonical_json(first) == assay._canonical_json(second)
    assert first["counts"]["questions_by_partition"] == assay.PARTITION_COUNTS
    assert first["projection_dataset_sha256"] != parent_sha
    assert first["locked_split"]["parent_dataset_sha256"] == parent_sha
    assert first["locked_split"]["parent_dataset_bytes_unavailable"] is True
    source_partitions: dict[str, set[str]] = {}
    for memory in first["memories"]:
        source_partitions.setdefault(memory["source_id"], set()).add(memory["partition"])
    assert all(len(partitions) == 1 for partitions in source_partitions.values())
    assert "raw-source-" not in assay._canonical_json(first)
    assay._assert_gold_free(first)


def test_seal_rejects_payload_and_sidecar_tampering(tmp_path: Path) -> None:
    path = tmp_path / "sealed.json"
    payload = {"format": "safe-test", "nested": {"value": 1}}
    digest, sidecar = assay._publish_json(path, payload)
    replay, replay_digest = assay._read_sealed(path, label="test")
    assert replay == payload
    assert replay_digest == digest
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(assay.AssayError):
        assay._read_sealed(path, label="test")

    second = tmp_path / "sidecar.json"
    assay._publish_json(second, payload)
    second.with_name(second.name + ".sha256").write_text("0" * 64, encoding="ascii")
    with pytest.raises(assay.AssayError, match="sidecar"):
        assay._read_sealed(second, label="test")
    assert sidecar.is_file()


def test_calibration_and_decision_use_strict_margin_boundary() -> None:
    student = {
        "q0": {"a": 0.5, "b": 0.0},
        "q1": {"a": 0.5, "b": 0.0},
        "q2": {"a": 1.0, "b": 0.0},
    }
    teacher = {
        "q0": {"a": 1.0, "b": 0.0},
        "q1": {"a": 0.0, "b": 1.0},
        "q2": {"a": 1.0, "b": 0.0},
    }
    calibrated = assay._calibrate_margin(
        student, teacher, precision_target=1.0, minimum_fast=1
    )
    threshold = float.fromhex(calibrated["threshold_hex"])
    assert threshold == 0.5
    assert calibrated["fast_count"] == 1
    assert assay._uses_fast_path(0.5, threshold) is False
    assert assay._uses_fast_path(1.0, threshold) is True


def test_pairwise_agreement_covers_all_pairs_across_two_teacher_groups() -> None:
    ids = [f"m{index:02d}" for index in range(16)]
    scores = {memory_id: float(index) for index, memory_id in enumerate(ids)}
    metrics = assay._agreement_metrics(
        {"q": scores}, {"q": scores}, {"q": [ids[:8], ids[8:]]}
    )
    assert metrics["pairs"] == 120
    assert metrics["pairwise_agree"] == 120
    assert metrics["top1_agree"] == 1
