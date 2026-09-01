from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import analyze_locked_query_answer_joint_failures as analysis
from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import identity_sha256


def test_namespaced_source_matching_is_exact() -> None:
    assert analysis._matches_target(
        "question-a::answer-1", "question-a", "answer-1"
    )
    assert analysis._matches_target("answer-1", "question-a", "answer-1")
    assert not analysis._matches_target(
        "question-b::answer-1", "question-a", "answer-1"
    )
    assert not analysis._matches_target(
        "question-a::prefix-answer-1", "question-a", "answer-1"
    )


def test_posthoc_labels_have_locked_aggregate() -> None:
    counts = {
        cause: sum(value == cause for value in analysis.CAUSE_BY_ORDINAL.values())
        for cause in analysis.CAUSE_ORDER
    }

    assert counts == {
        "source_missing": 2,
        "candidate_reached_but_packing_dropped": 2,
        "partial_multi_source_coverage": 5,
        "operator_failure_despite_source_coverage": 16,
        "answer_shape_or_judge_ambiguity": 2,
        "other": 1,
    }
    assert tuple(sorted(analysis.CAUSE_BY_ORDINAL)) == tuple(
        sorted(analysis.JOINT_FAILURE_ORDINALS)
    )


def test_cause_guards_reject_changed_evidence_lifecycle() -> None:
    with pytest.raises(
        analysis.JointFailureAnalysisError, match="no longer supports source_missing"
    ):
        analysis._validate_cause(
            ordinal=36,
            cause="source_missing",
            actual_count=0,
            target_count=1,
            query_candidate_count=0,
            guided_candidate_count=1,
        )
    with pytest.raises(
        analysis.JointFailureAnalysisError,
        match="no longer has partial multi-source coverage",
    ):
        analysis._validate_cause(
            ordinal=31,
            cause="partial_multi_source_coverage",
            actual_count=2,
            target_count=2,
            query_candidate_count=0,
            guided_candidate_count=0,
        )


def test_verification_failure_prevents_reference_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_verification(**kwargs: Any) -> Any:
        calls.append("verify")
        raise RuntimeError("tampered score artifact")

    def forbidden_references(*args: Any, **kwargs: Any) -> Any:
        calls.append("references")
        raise AssertionError("references were opened")

    monkeypatch.setattr(analysis, "verify_sealed_inputs", fail_verification)
    monkeypatch.setattr(analysis, "_load_references", forbidden_references)

    with pytest.raises(RuntimeError, match="tampered score artifact"):
        analysis.analyze_paths()

    assert calls == ["verify"]


def test_success_path_opens_references_after_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    verified = SimpleNamespace(
        target_plan_path=Path("target.json"),
        guided_audit_path=Path("guided.json"),
    )

    def verify(**kwargs: Any) -> Any:
        calls.append("verify")
        return verified

    def references(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        calls.append("references")
        return ("gold",)

    def read(path: Path) -> Any:
        calls.append(path.stem)
        if path == verified.target_plan_path:
            return SimpleNamespace(
                sha256=analysis.EXPECTED_TARGET_PLAN_SHA256,
                payload={"target": True},
            )
        return SimpleNamespace(
            sha256=analysis.EXPECTED_GUIDED_AUDIT_SHA256,
            payload={"guided": True},
        )

    def build(**kwargs: Any) -> dict[str, Any]:
        calls.append("build")
        assert kwargs["inputs"] is verified
        assert kwargs["references"] == ("gold",)
        return {"ok": True}

    monkeypatch.setattr(analysis, "verify_sealed_inputs", verify)
    monkeypatch.setattr(analysis, "_load_references", references)
    monkeypatch.setattr(analysis, "read_sealed_json", read)
    monkeypatch.setattr(analysis, "build_analysis_payload", build)

    assert analysis.analyze_paths() == {"ok": True}
    assert calls == ["verify", "references", "target", "guided", "build"]


def test_deployable_policy_is_route_keyed_and_separate_from_posthoc_labels() -> None:
    assert set(analysis.QUESTION_ONLY_POLICY) == {
        "numeric_reduce",
        "temporal_timeline",
        "set_join",
        "synthesize",
        "direct_extract",
    }
    assert not set(analysis.QUESTION_ONLY_POLICY).intersection(analysis.CAUSE_ORDER)
    assert "numeric_executor" in analysis.QUESTION_ONLY_POLICY["numeric_reduce"][
        "mechanisms"
    ]
    assert "timeline_event_table" in analysis.QUESTION_ONLY_POLICY[
        "temporal_timeline"
    ]["mechanisms"]


def test_published_taxonomy_is_sealed_and_row_self_sealed() -> None:
    artifact = read_sealed_json(analysis.DEFAULT_OUTPUT)
    payload = artifact.payload

    assert payload["format"] == analysis.ANALYSIS_FORMAT
    assert payload["question_count"] == 28
    assert payload["provider_calls"] == 0
    assert payload["aggregate"]["prospective_union_question_coverage"] == {
        "full": 24,
        "none": 3,
        "partial": 1,
    }
    unsigned = dict(payload)
    declared = unsigned.pop("analysis_sha256")
    assert identity_sha256(unsigned) == declared
    for row in payload["rows"]:
        unsigned_row = dict(row)
        row_sha256 = unsigned_row.pop("row_sha256")
        assert identity_sha256(unsigned_row) == row_sha256


def test_checkpoint_hashes_are_not_cli_configurable() -> None:
    destinations = {action.dest for action in analysis.build_parser()._actions}

    assert "expected_payload_judge_sha256" not in destinations
    assert "expected_fact_score_sha256" not in destinations
    assert "expected_guided_audit_sha256" not in destinations
    assert "expected_target_plan_sha256" not in destinations
