from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval import run_fast_1m_em_facts as em_runner
from tools import score_locked_retrieval_target_ownership as scorer


POPULATION_SHA = "a" * 64


def _publish(path: Path, value: dict[str, Any]) -> str:
    return em_runner._publish(path, value)


def _answer_run(root: Path, method: str, arm: str) -> tuple[str, str]:
    questions = [
        {
            "ordinal": ordinal,
            "question_id": f"q{ordinal}",
            "question_sha256": str(ordinal + 1) * 64,
            "dated_question_sha256": str(ordinal + 3) * 64,
        }
        for ordinal in range(2)
    ]
    value = {
        "format": "memory-condense-locked-retrieval-mechanism-arm-run-v1",
        "arm_label": arm,
        "population_identity_sha256": POPULATION_SHA,
        "question_count": 2,
        "questions": questions,
        "gold_loaded": False,
    }
    path = root / method / "run.json"
    digest = _publish(path, value)
    _publish(path.with_name("run-replay.json"), value)
    return str(path), digest


def _event(
    target_id: str,
    discovering: str,
    sources: list[str],
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "target_kind": "evidence",
        "discovering_method": discovering,
        "disposition": "selected_and_admitted",
        "route_local_receipt_sha256": "d" * 64,
        "source_target_ids": sources,
    }


def _ledger(
    root: Path,
    *,
    method: str,
    arm: str,
    run_sha: str,
    rows: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> tuple[str, str]:
    questions = []
    for ordinal, (before, after) in enumerate(rows):
        questions.append(
            row := {
                "ordinal": ordinal,
                "question_id": f"q{ordinal}",
                "candidate_targets_pre_dedup": before,
                "admitted_targets_post_dedup": after,
            }
        )
        row["ledger_row_sha256"] = scorer._self_sha(row, "ledger_row_sha256")
    value = {
        "format": scorer.LEDGER_FORMAT,
        "arm_label": arm,
        "source_run_sha256": run_sha,
        "population_identity_sha256": POPULATION_SHA,
        "question_count": 2,
        "ownership_policy": "join-primary-owner-from-posthoc-desired-target-registry",
        "discovery_projection": "candidate_targets_pre_dedup",
        "admission_projection": "admitted_targets_post_dedup",
        "questions": questions,
    }
    value["ledger_sha256"] = scorer._self_sha(value, "ledger_sha256")
    path = root / method / "targets.json"
    return str(path), _publish(path, value)


def _target(ordinal: int, source_id: str, owner: str) -> dict[str, Any]:
    value = {
        "ordinal": ordinal,
        "question_id": f"q{ordinal}",
        "target_kind": "source_id",
        "target_id": source_id,
        "primary_owner": owner,
    }
    return value | {"target_sha256": identity_sha256(value)}


def _fixture(tmp_path: Path) -> dict[str, Any]:
    a_path, a_sha = _answer_run(tmp_path, "root", "S0_CONTROL")
    b_path, b_sha = _answer_run(tmp_path, "episodic", "S0_PLUS_EM_FACTS")
    a_q0 = _event("e0", "root", ["s0"])
    a_q1_s1 = _event("e1", "root", ["s1"])
    a_q1_s3 = _event("e3", "root", ["s3"])
    b_q1 = _event("fact1", "episodic", ["s1"])
    a_ledger, a_ledger_sha = _ledger(
        tmp_path,
        method="root",
        arm="S0_CONTROL",
        run_sha=a_sha,
        rows=[([a_q0], [a_q0]), ([a_q1_s1, a_q1_s3], [])],
    )
    b_ledger, b_ledger_sha = _ledger(
        tmp_path,
        method="episodic",
        arm="S0_PLUS_EM_FACTS",
        run_sha=b_sha,
        rows=[([], []), ([b_q1], [b_q1])],
    )
    questions = [
        {
            "ordinal": ordinal,
            "question_id": f"q{ordinal}",
            "question_sha256": str(ordinal + 1) * 64,
            "dated_question_sha256": str(ordinal + 3) * 64,
        }
        for ordinal in range(2)
    ]
    answer_specs = [f"root={a_path}={a_sha}", f"episodic={b_path}={b_sha}"]
    registry = {
        "format": scorer.REGISTRY_FORMAT,
        "population_identity_sha256": POPULATION_SHA,
        "answer_run_bindings": [
            {
                "discovering_method": "root",
                "arm_label": "S0_CONTROL",
                "run_sha256": a_sha,
                "run_replay_sha256": a_sha,
            },
            {
                "discovering_method": "episodic",
                "arm_label": "S0_PLUS_EM_FACTS",
                "run_sha256": b_sha,
                "run_replay_sha256": b_sha,
            },
        ],
        "question_count": 2,
        "ordered_questions": questions,
        "desired_target_count": 4,
        "desired_targets": [
            _target(0, "s0", "root"),
            _target(1, "s1", "episodic"),
            _target(1, "s2", "episodic"),
            _target(1, "s3", "episodic"),
        ],
        "constructed_after_all_answer_run_seals": True,
        "gold_target_tags_posthoc_only": True,
    }
    registry["registry_sha256"] = scorer._self_sha(registry, "registry_sha256")
    registry_path = tmp_path / "registry.json"
    registry_sha = _publish(registry_path, registry)
    return {
        "answer_specs": answer_specs,
        "registry": registry,
        "registry_path": registry_path,
        "registry_sha": registry_sha,
        "ledger_specs": [
            f"root={a_ledger}={a_ledger_sha}",
            f"episodic={b_ledger}={b_ledger_sha}",
        ],
    }


def test_scores_primary_alternate_union_and_pre_dedup_credit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result, digest = scorer.score(
        answer_run_specs=fixture["answer_specs"],
        registry_path=fixture["registry_path"],
        registry_sha256=fixture["registry_sha"],
        ledger_specs=fixture["ledger_specs"],
        output_path=tmp_path / "target-score.json",
    )
    assert digest
    assert result["aggregate"] == {
        "desired_targets": 4,
        "assigned_primary_owner_targets": 4,
        "unassigned_primary_owner_count": 0,
        "union_discovered": 3,
        "union_discovery_recall": 0.75,
        "union_admitted": 2,
        "union_admission_recall": 0.5,
        "union_unreached_count": 1,
        "discovered_then_deduped_count": 1,
    }
    episodic = next(
        row for row in result["per_owner"] if row["primary_owner"] == "episodic"
    )
    assert episodic["desired_targets"] == 3
    assert episodic["primary_discovered"] == 1
    assert episodic["alternate_reachable"] == 2
    assert episodic["alternate_only_reachable"] == 1
    s3 = next(row for row in result["targets"] if row["target_sha256"] == _target(1, "s3", "episodic")["target_sha256"])
    assert s3["union_discovered"] is True
    assert s3["union_admitted"] is False
    assert result["provider_calls"] == 0


def test_registry_rejects_duplicate_target_owner_rows(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    answer_runs = scorer._verify_answer_runs(fixture["answer_specs"])
    registry = copy.deepcopy(fixture["registry"])
    duplicate = dict(registry["desired_targets"][0])
    duplicate["primary_owner"] = "episodic"
    body = {key: value for key, value in duplicate.items() if key != "target_sha256"}
    duplicate["target_sha256"] = identity_sha256(body)
    registry["desired_targets"].append(duplicate)
    registry["desired_target_count"] += 1
    registry["registry_sha256"] = scorer._self_sha(registry, "registry_sha256")
    with pytest.raises(scorer.TargetCoverageError, match="more than one"):
        scorer._validate_registry(registry, answer_runs)


def test_target_report_failure_is_isolated_from_semantic_judge(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bad = list(fixture["ledger_specs"])
    bad.pop()
    with pytest.raises(scorer.TargetCoverageError, match="every registered method"):
        scorer.score(
            answer_run_specs=fixture["answer_specs"],
            registry_path=fixture["registry_path"],
            registry_sha256=fixture["registry_sha"],
            ledger_specs=bad,
            output_path=tmp_path / "must-not-exist.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_structural_event_cannot_claim_primary_ownership() -> None:
    event = _event("e0", "root", ["s0"])
    event["primary_owner"] = "root"
    with pytest.raises(scorer.TargetCoverageError, match="assign target ownership"):
        scorer._event(event, "root")


def test_strict_score_requires_one_loader_per_ledger(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(scorer.TargetCoverageError, match="one loader"):
        scorer.score(
            answer_run_specs=fixture["answer_specs"],
            registry_path=fixture["registry_path"],
            registry_sha256=fixture["registry_sha"],
            ledger_specs=fixture["ledger_specs"],
            output_path=tmp_path / "must-not-exist.json",
            require_ledger_loaders=True,
        )
