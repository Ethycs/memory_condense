from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.matched_eval import live, query_answer_judging
from tools.matched_eval.adaptive_evidence_solver_judge_adapter import (
    DIRECT_PROFILE,
    VerifiedAdaptiveEvidenceSolverDirectJudgePlane,
    adapt_verified_adaptive_evidence_solver,
    expected_source_bindings,
    validate_adaptive_judge_plane,
)
from tools.matched_eval.adaptive_evidence_solver_live import (
    capture_adaptive_solver_completions,
    materialize_adaptive_evidence_solver,
    replay_adaptive_evidence_solver,
)
from tools.matched_eval.contracts import MatchedEvalContractError
from tools.matched_eval.ledger import _validated_runtime_ledger
from tools.matched_eval.source_history_fact_union import FactLane
from tests.test_run_locked_adaptive_evidence_solver_v3 import _loaded


def _sha(character: str) -> str:
    return character * 64


def _judge_plane(tmp_path: Path):
    loaded = _loaded(tmp_path / "inputs", with_source_fact=True)
    planned = loaded.plan.submitted_rows[0]
    completion = json.dumps(
        {
            "decision": "replace",
            "prediction": "The source-backed answer is blue.",
            "used_evidence_ids": [planned.allowed_source_fact_ids[0]],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    completion_plane = capture_adaptive_solver_completions(
        loaded.plan,
        loaded.preflight,
        {planned.question_id: completion},
    )
    run = materialize_adaptive_evidence_solver(
        loaded.plan,
        loaded.preflight,
        completion_plane,
    )
    verified = replay_adaptive_evidence_solver(
        loaded.plan,
        loaded.preflight,
        completion_plane,
        run,
    )
    plane = adapt_verified_adaptive_evidence_solver(
        lanes=(FactLane.DIRECT,),
        plan=loaded.plan,
        preflight=loaded.preflight,
        completion_plane=completion_plane,
        run=run,
        verified_plane=verified,
        terminal_run_sha256=_sha("1"),
        solver_preflight_artifact_sha256=_sha("2"),
        source_preflight_sha256=_sha("3"),
        source_work_manifest_sha256=_sha("4"),
        source_materialization_sha256=_sha("5"),
        lane_filter_receipt_sha256=_sha("6"),
    )
    return loaded, plane


def test_native_adaptive_run_projects_into_shared_changed_only_judge_surface(
    tmp_path: Path,
) -> None:
    loaded, plane = _judge_plane(tmp_path)

    assert type(plane) is VerifiedAdaptiveEvidenceSolverDirectJudgePlane
    assert validate_adaptive_judge_plane(plane) is DIRECT_PROFILE
    adapter = query_answer_judging._adapter_for_plane(plane)
    _identity, runtime_row_ids = _validated_runtime_ledger(
        live._thaw_json(plane.runtime_ledger)
    )
    assert adapter.kind == "adaptive_evidence_solver_v3_d"
    assert plane.run_sha256 == _sha("1")
    assert len(plane.changed_rows) == 1
    assert plane.rows[0].changed_from_parent is True
    assert runtime_row_ids == (plane.rows[0].runtime_row_id,)
    assert expected_source_bindings(plane)["answer_run"] == _sha("1")
    assert plane.parent_plane is loaded.map_plan.direct_plane
    assert plane.runtime_ledger["total_provider_calls"] == 1


def test_adaptive_judge_tamper_fails_before_shared_judging(
    tmp_path: Path,
) -> None:
    _loaded_plan, plane = _judge_plane(tmp_path)
    tampered = replace(plane, source_materialization_sha256=_sha("9"))

    with pytest.raises(MatchedEvalContractError, match="projection changed"):
        query_answer_judging._validate_answer_plane(
            tampered,
            expected_question_count=1,
        )


def test_adaptive_judge_rejects_a_lane_label_that_does_not_match_source_facts(
    tmp_path: Path,
) -> None:
    loaded, plane = _judge_plane(tmp_path)

    with pytest.raises(MatchedEvalContractError, match="escaped lane profile"):
        adapt_verified_adaptive_evidence_solver(
            lanes=(FactLane.GUIDED,),
            plan=loaded.plan,
            preflight=plane.native_preflight,
            completion_plane=plane.native_completion_plane,
            run=plane.native_run,
            verified_plane=plane.native_verified_plane,
            terminal_run_sha256=plane.run_sha256,
            solver_preflight_artifact_sha256=(
                plane.solver_preflight_artifact_sha256
            ),
            source_preflight_sha256=plane.source_preflight_sha256,
            source_work_manifest_sha256=plane.source_work_manifest_sha256,
            source_materialization_sha256=plane.source_materialization_sha256,
            lane_filter_receipt_sha256=plane.lane_filter_receipt_sha256,
        )
