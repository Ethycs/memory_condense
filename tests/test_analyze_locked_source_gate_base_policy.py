from __future__ import annotations

from hashlib import sha256

import pytest

from tools.analyze_locked_source_gate_base_policy import (
    BASE_CAP_POLICIES,
    SourceGatePolicySweepError,
    SweepQuestion,
    attach_posthoc_source_coverage,
    sweep_base_policies,
)
from tools.matched_eval.contracts import identity_sha256


def _sha(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _rows() -> tuple[SweepQuestion, ...]:
    return (
        SweepQuestion(
            0,
            "q0",
            "temporal_timeline",
            _sha("n0"),
            True,
            ("base",),
            ("shared", "d2", "d3", "d4", "d5"),
            ("p1",),
            ("shared", "g2"),
            (("shared", 1), ("d2", 2), ("d3", 1), ("d4", 1), ("d5", 1), ("p1", 2), ("g2", 3)),
        ),
        SweepQuestion(
            1,
            "q1",
            "direct_extract",
            _sha("n1"),
            True,
            ("base2",),
            ("x",),
            ("p2",),
            ("y",),
            (("x", 1), ("p2", 1), ("y", 1)),
        ),
        SweepQuestion(
            2,
            "q2",
            "direct_extract",
            None,
            False,
            ("z",),
            (),
            (),
            (),
            (),
        ),
    )


def _target_plan(rows: tuple[SweepQuestion, ...], targets: tuple[tuple[int, str], ...]) -> dict:
    body = {
        "desired_source_target_count": len(targets),
        "desired_targets": [
            {
                "ordinal": ordinal,
                "question_id": rows[ordinal].question_id,
                "target_id": source_id,
                "target_kind": "source_id",
                "target_sha256": _sha(f"{ordinal}:{source_id}"),
            }
            for ordinal, source_id in targets
        ],
        "format": "memory-condense-retrieval-target-owner-plan-v1",
        "gold_target_tags_posthoc_only": True,
        "ordered_question_keys": [
            {"ordinal": row.ordinal, "question_id": row.question_id} for row in rows
        ],
        "provider_calls": 0,
        "question_count": len(rows),
        "runtime_use_forbidden": True,
    }
    return {**body, "plan_sha256": identity_sha256(body)}


def test_sweep_preserves_lane_credit_and_deduplicates_physical_work() -> None:
    structural = sweep_base_policies(
        _rows(),
        source_parent_receipt_sha256=_sha("sources"),
        map_adapter_receipt_sha256=_sha("map"),
    )
    policies = {row["policy_id"]: row for row in structural["policies"]}
    d1 = policies["D1/G1"]
    assert structural["activated_question_count"] == 2
    assert structural["no_op_question_count"] == 1
    assert d1["logical_selection_count"] == 4
    assert d1["physical_unique_source_call_count"] == 3
    assert d1["physical_window_call_count"] == 3
    assert d1["per_question"][0]["logical_selection_count"] == 2
    assert d1["per_question"][0]["physical_unique_source_call_count"] == 1
    assert d1["per_activated_question_distributions"] == {
        "logical_selection_count": {"2": 2},
        "physical_unique_source_call_count": {"1": 1, "2": 1},
        "physical_window_call_count": {"1": 1, "2": 1},
    }
    assert d1["route_totals"]["temporal_timeline"] == {
        "activated_question_count": 1,
        "logical_selection_count": 2,
        "physical_unique_source_call_count": 1,
        "physical_window_call_count": 1,
    }
    assert policies["D5/G2"]["logical_selection_count"] == 9
    assert policies["D5/G2"]["physical_unique_source_call_count"] == 8
    assert policies["D5/G2"]["physical_window_call_count"] == 11
    assert structural["target_plan_loaded"] is False


def test_posthoc_targets_annotate_but_cannot_change_selection_or_routing() -> None:
    rows = _rows()
    structural = sweep_base_policies(
        rows,
        source_parent_receipt_sha256=_sha("sources"),
        map_adapter_receipt_sha256=_sha("map"),
    )
    targets = ((0, "base"), (0, "d2"), (0, "g2"), (0, "missing"), (1, "y"), (2, "z"))
    plan = _target_plan(rows, targets)
    result = attach_posthoc_source_coverage(
        structural,
        rows,
        plan,
        target_plan_artifact_sha256=_sha("target-artifact"),
    )
    coverage = {row["policy_id"]: row for row in result["coverage_call_pareto"]}
    assert result["selection_and_routing_frozen_before_target_plan_load"] is True
    assert result["structural_selection_sha256"] == structural["structural_selection_sha256"]
    assert result["structural_selection"] == structural
    assert coverage["D1/G1"]["covered_source_target_count"] == 3
    assert coverage["D2/G1"]["covered_source_target_count"] == 4
    assert coverage["D3/G2"]["covered_source_target_count"] == 5
    assert coverage["D5/G2"]["covered_source_target_count"] == 5
    assert coverage["D5/G2"]["pareto_on_physical_window_calls"] is False
    assert coverage["D3/G1"]["pareto_on_unique_source_calls"] is False

    alternate = _target_plan(rows, ((0, "shared"),))
    alternate_result = attach_posthoc_source_coverage(
        structural,
        rows,
        alternate,
        target_plan_artifact_sha256=_sha("alternate-artifact"),
    )
    assert alternate_result["structural_selection"] == result["structural_selection"]
    assert alternate_result["structural_selection_sha256"] == result["structural_selection_sha256"]


def test_posthoc_target_plan_must_be_explicitly_runtime_forbidden() -> None:
    rows = _rows()
    structural = sweep_base_policies(
        rows,
        source_parent_receipt_sha256=_sha("sources"),
        map_adapter_receipt_sha256=_sha("map"),
    )
    plan = _target_plan(rows, ((0, "base"),))
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    unsigned["runtime_use_forbidden"] = False
    bad = {**unsigned, "plan_sha256": identity_sha256(unsigned)}
    with pytest.raises(SourceGatePolicySweepError, match="posthoc-only"):
        attach_posthoc_source_coverage(
            structural,
            rows,
            bad,
            target_plan_artifact_sha256=_sha("target-artifact"),
        )


def test_requested_policy_ladder_is_fixed() -> None:
    assert tuple(row.policy_id for row in BASE_CAP_POLICIES) == (
        "D1/G0",
        "D0/P1/G0",
        "D0/G1",
        "D1/G1",
        "D1/P1/G1",
        "D2/G1",
        "D3/G1",
        "D3/G2",
        "D5/G2",
    )
    assert tuple(
        row.policy_id for row in BASE_CAP_POLICIES if row.partition_base_source_cap
    ) == ("D0/P1/G0", "D1/P1/G1")
