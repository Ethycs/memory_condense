from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from memory_condense.eval import run_fast_1m_em_facts as em_runner
from tools import build_locked_retrieval_target_registry as builder


POPULATION_SHA = "a" * 64


def _publish(path: Path, value: dict[str, Any]) -> str:
    return em_runner._publish(path, value)


def _answer_runs(root: Path) -> list[str]:
    result = []
    questions = [
        {
            "ordinal": ordinal,
            "question_id": f"q{ordinal}",
            "question_sha256": str(ordinal + 1) * 64,
            "dated_question_sha256": str(ordinal + 3) * 64,
        }
        for ordinal in range(3)
    ]
    for method in builder.REQUIRED_METHODS:
        run = {
            "format": "memory-condense-locked-retrieval-mechanism-arm-run-v1",
            "arm_label": method.upper(),
            "population_identity_sha256": POPULATION_SHA,
            "question_count": len(questions),
            "questions": questions,
            "gold_loaded": False,
        }
        path = root / method / "run.json"
        digest = _publish(path, run)
        _publish(path.with_name("run-replay.json"), run)
        result.append(f"{method}={path}={digest}")
    return result


def _style(root: Path) -> tuple[Path, str, dict[str, Any]]:
    rows = [
        {
            "ordinal": 0,
            "question_id": "q0",
            "benchmark_category": "single-session-user",
            "retrieval_topology": "point",
            "answer_operator": "direct_lookup",
            "expected_source_count": 1,
            "expected_source_ids": ["s0"],
            "expected_source_positions": [2],
        },
        {
            "ordinal": 1,
            "question_id": "q1",
            "benchmark_category": "multi-session",
            "retrieval_topology": "dispersed_join",
            "answer_operator": "numeric_aggregate_compare",
            "expected_source_count": 2,
            "expected_source_ids": ["s1", "s2"],
            "expected_source_positions": [1, 8],
        },
        {
            "ordinal": 2,
            "question_id": "q2",
            "benchmark_category": "temporal-reasoning",
            "retrieval_topology": "dispersed_join",
            "answer_operator": "temporal_order_select",
            "expected_source_count": 2,
            "expected_source_ids": ["s3", "s4"],
            "expected_source_positions": [3, 19],
        },
    ]
    value = {
        "format": builder.STYLE_FORMAT,
        "provider_calls": 0,
        "bindings": {"population_identity_sha256": POPULATION_SHA},
        "questions": rows,
    }
    path = root / "style.json"
    return path, _publish(path, value), value


def test_builds_external_universe_and_disjoint_owner_union(tmp_path: Path) -> None:
    runs = _answer_runs(tmp_path)
    style_path, style_sha, _ = _style(tmp_path)
    plan_path = tmp_path / "target-plan.json"
    plan, plan_sha = builder.build_target_plan(
        style_ledger_path=style_path,
        style_ledger_sha256=style_sha,
        output_path=plan_path,
    )
    registry, digest = builder.build_registry(
        answer_run_specs=runs,
        target_plan_path=plan_path,
        target_plan_sha256=plan_sha,
        output_path=tmp_path / "registry.json",
    )
    assert digest
    assert registry["desired_source_target_count"] == 5
    assert registry["desired_relation_target_count"] == 2
    assert registry["desired_coverage_check_target_count"] == 0
    assert registry["desired_target_count"] == 7
    assert registry["primary_owner_counts"] == {
        "s0": 1,
        "em": 2,
        "representative": 0,
        "global": 2,
        "hebbian": 0,
        "cav": 2,
    }
    targets = registry["desired_targets"]
    assert len({row["target_sha256"] for row in targets}) == len(targets)
    assert registry["unassigned_primary_owner_count"] == 0
    assert registry["runtime_use_forbidden"] is True
    assert registry["answer_runs_verified_before_gold_target_plan_load"] is True
    assert registry["target_plan"] == plan
    assert registry["target_plan_file_sha256"] == plan_sha
    assert registry["immutable_target_plan_reproduced_byte_for_byte"] is True


def test_insufficient_evidence_adds_owned_coverage_check(tmp_path: Path) -> None:
    runs = _answer_runs(tmp_path)
    _style_path, _style_sha, style = _style(tmp_path)
    style["questions"][0]["answer_operator"] = "insufficient_evidence"
    style_path = tmp_path / "insufficient-style.json"
    style_sha = _publish(style_path, style)
    plan_path = tmp_path / "target-plan.json"
    _plan, plan_sha = builder.build_target_plan(
        style_ledger_path=style_path,
        style_ledger_sha256=style_sha,
        output_path=plan_path,
    )
    registry, _ = builder.build_registry(
        answer_run_specs=runs,
        target_plan_path=plan_path,
        target_plan_sha256=plan_sha,
        output_path=tmp_path / "registry.json",
    )
    coverage = [
        row for row in registry["desired_targets"] if row["target_kind"] == "coverage_check"
    ]
    assert len(coverage) == 1
    assert coverage[0]["primary_owner"] == "s0"


def test_rejects_missing_method_before_opening_target_plan(tmp_path: Path) -> None:
    runs = _answer_runs(tmp_path)[:-1]
    missing_plan = tmp_path / "missing-plan.json"
    with pytest.raises(builder.RegistryBuildError, match="methods/order"):
        builder.build_registry(
            answer_run_specs=runs,
            target_plan_path=missing_plan,
            target_plan_sha256="f" * 64,
            output_path=tmp_path / "must-not-exist.json",
        )
    assert not missing_plan.exists()


def test_rejects_duplicate_source_target(tmp_path: Path) -> None:
    style_path, _style_sha, style = _style(tmp_path)
    bad = copy.deepcopy(style)
    bad["questions"][1]["expected_source_ids"] = ["s1", "s1"]
    style_path = tmp_path / "duplicate-style.json"
    bad_sha = _publish(style_path, bad)
    with pytest.raises(builder.RegistryBuildError, match="desired source universe"):
        builder.build_target_plan(
            style_ledger_path=style_path,
            style_ledger_sha256=bad_sha,
            output_path=tmp_path / "must-not-exist.json",
        )


def test_registry_cannot_diverge_from_immutable_plan(tmp_path: Path) -> None:
    runs = _answer_runs(tmp_path)
    style_path, style_sha, _ = _style(tmp_path)
    plan_path = tmp_path / "target-plan.json"
    _plan, plan_sha = builder.build_target_plan(
        style_ledger_path=style_path,
        style_ledger_sha256=style_sha,
        output_path=plan_path,
    )
    registry, _ = builder.build_registry(
        answer_run_specs=runs,
        target_plan_path=plan_path,
        target_plan_sha256=plan_sha,
        output_path=tmp_path / "registry.json",
    )
    registry["desired_targets"] = registry["desired_targets"][:-1]
    registry["desired_target_count"] -= 1
    registry["registry_sha256"] = builder.scorer._self_sha(
        registry, "registry_sha256"
    )
    answer_runs = builder.scorer._verify_answer_runs(runs)
    with pytest.raises(
        builder.scorer.TargetCoverageError, match="does not reproduce"
    ):
        builder.scorer._validate_registry(registry, answer_runs)
