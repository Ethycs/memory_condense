from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_matched_eval_closure import _sealed_campaign
from tools import analyze_locked_independent_closure_targets as analysis
from tools.matched_eval import closure
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.contracts import EvidenceItem
from memory_condense.domain._tokenizer import count_tokens


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _target(
    *,
    target_id: str,
    target_kind: str,
    owner: str,
    expected_sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    body = {
        "ordinal": 0,
        "question_id": "q000",
        "target_kind": target_kind,
        "target_id": target_id,
        "primary_owner": owner,
    }
    basis: dict[str, Any] = {"rule": "synthetic"}
    if expected_sources:
        basis["expected_source_ids"] = list(expected_sources)
    return body | {
        "target_sha256": identity_sha256(body),
        "assignment_basis": basis,
        "assignment_basis_sha256": identity_sha256(basis),
    }


def _event(source_id: str, route: str) -> dict[str, Any]:
    return {
        "target_id": f"atom-{source_id}",
        "target_kind": "evidence_atom",
        "discovering_method": route,
        "_source_target_ids": [source_id],
    }


def _one_question_events() -> dict[
    str, dict[str, dict[int, tuple[dict[str, Any], ...]]]
]:
    representative = (_event("global-source", closure.REPRESENTATIVE_ARM),)
    global_events = (_event("rep-source", closure.GLOBAL_ARM),)
    return {
        analysis.REPRESENTATIVE_ROUTE: {
            stage: {0: representative} for stage in analysis.STAGES
        },
        analysis.GLOBAL_ROUTE: {
            stage: {0: global_events} for stage in analysis.STAGES
        },
    }


def test_scoring_separates_formal_relations_operands_and_cross_arm_rescues() -> None:
    plan = {
        "ordered_question_keys": [{"ordinal": 0, "question_id": "q000"}],
        "desired_targets": [
            _target(
                target_id="rep-source",
                target_kind="source_id",
                owner="representative",
            ),
            _target(
                target_id="global-source",
                target_kind="source_id",
                owner="global",
            ),
            _target(
                target_id="relation-join",
                target_kind="relation",
                owner="cav",
                expected_sources=("rep-source", "global-source"),
            ),
            _target(
                target_id="coverage",
                target_kind="coverage_check",
                owner="s0",
                expected_sources=("rep-source", "global-source"),
            ),
        ],
    }

    scored = analysis.score_target_reach(plan, _one_question_events())
    raw_union = scored["routes"][analysis.UNION_ROUTE]["stages"][
        "raw_candidate_reach"
    ]

    assert raw_union["source_targets"] == {
        "target_count": 2,
        "hit_count": 2,
        "recall": 1.0,
    }
    assert raw_union["formal_relation_targets"] == {
        "target_count": 1,
        "hit_count": 0,
        "recall": 0.0,
    }
    assert raw_union["relation_operand_completeness"] == {
        "target_count": 1,
        "hit_count": 1,
        "recall": 1.0,
    }
    assert raw_union["coverage_check_targets"]["hit_count"] == 1
    rescues = scored["alternate_method_rescues_by_stage"][
        "raw_candidate_reach"
    ]
    assert rescues["representative_owner_rescued_by_global"]["count"] == 1
    assert rescues["global_owner_rescued_by_representative"]["count"] == 1
    assert rescues["total_count"] == 2
    assert (
        scored["closure_primary_owner_outcomes_by_stage"]["raw_candidate_reach"]
        ["representative"]["primary_hit_count"]
        == 0
    )
    relation = scored["targets"][2]["stages"]["raw_candidate_reach"]
    assert relation["formal_hit_by_route"]["union"] is False
    assert relation["relation_operand_complete_by_route"]["union"] is True


def test_runtime_projection_keeps_candidate_selection_and_admission_distinct() -> None:
    population, eligibility, eligibility_sha, raw_generation, generation_sha = (
        _sealed_campaign()
    )
    generation = closure.project_independent_closure_generation(
        raw_generation,
        generation_sha256=generation_sha,
        eligibility_manifest=eligibility,
        eligibility_manifest_sha256=eligibility_sha,
        population=population,
    )

    projections, events = analysis._project_runtime_stages(generation)
    representative = events[analysis.REPRESENTATIVE_ROUTE]
    global_events = events[analysis.GLOBAL_ROUTE]

    def sources(rows: tuple[dict[str, Any], ...]) -> set[str]:
        return {str(row["_source_target_ids"][0]) for row in rows}

    assert set(projections) == set(closure.ARM_LABELS)
    assert sources(representative["raw_candidate_reach"][6]) == {
        "q006::answer_overlap",
        "q006::answer_novel",
        "q006::answer_projection",
        "q006::answer_dropped",
        "q006::answer_unselected",
    }
    assert sources(representative["selected_before_dedup"][6]) == {
        "q006::answer_overlap",
        "q006::answer_novel",
        "q006::answer_projection",
        "q006::answer_dropped",
    }
    assert sources(representative["post_s0_admission"][6]) == {
        "q006::answer_novel"
    }
    assert sources(global_events["raw_candidate_reach"][6]) == sources(
        representative["raw_candidate_reach"][6]
    )
    assert sources(global_events["selected_before_dedup"][6]) == {
        "q006::answer_novel"
    }
    assert sources(global_events["post_s0_admission"][6]) == {
        "q006::answer_novel"
    }


def test_eligible_incremental_funnel_separates_s0_overlap_and_ineligible() -> None:
    def evidence(evidence_id: str, source_id: str) -> EvidenceItem:
        text = f"evidence for {source_id}"
        return EvidenceItem(evidence_id, source_id, text, count_tokens(text))

    targets = [
        _target(target_id="s0-hit", target_kind="source_id", owner="global"),
        _target(target_id="novel", target_kind="source_id", owner="global"),
        _target(target_id="ineligible", target_kind="source_id", owner="global"),
    ]
    targets[2]["ordinal"] = 1
    targets[2]["question_id"] = "q001"
    generation = SimpleNamespace(
        questions=(
            SimpleNamespace(
                ordinal=0,
                eligible=True,
                root_protected_evidence=(evidence("e0", "q000::s0-hit"),),
            ),
            SimpleNamespace(
                ordinal=1,
                eligible=False,
                root_protected_evidence=(evidence("e1", "q001::ineligible"),),
            ),
        )
    )
    empty = {0: (), 1: ()}
    global_raw = {
        0: (_event("q000::s0-hit", closure.GLOBAL_ARM),),
        1: (_event("q001::ineligible", closure.GLOBAL_ARM),),
    }
    events = {
        analysis.REPRESENTATIVE_ROUTE: {
            stage: dict(empty) for stage in analysis.STAGES
        },
        analysis.GLOBAL_ROUTE: {
            stage: (global_raw if stage == "raw_candidate_reach" else dict(empty))
            for stage in analysis.STAGES
        },
    }

    result = analysis.eligible_incremental_source_funnel(
        {"desired_targets": targets}, generation, events
    )

    assert result["eligible_source_target_count"] == 2
    assert result["ineligible_source_target_count"] == 1
    assert result["s0"]["hit_count"] == 1
    assert result["s0_missing"]["target_count"] == 1
    global_result = result["raw_closure_routes"][analysis.GLOBAL_ROUTE]
    assert global_result["all_eligible_source_targets"]["hit_count"] == 1
    assert global_result["novel_over_s0_missing_sources"]["hit_count"] == 0


def test_generation_failure_prevents_target_plan_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def load_population(*args: Any, **kwargs: Any) -> object:
        calls.append("population")
        return object()

    def fail_generation(*args: Any, **kwargs: Any) -> object:
        calls.append("generation")
        raise RuntimeError("tampered generation")

    def open_plan(*args: Any, **kwargs: Any) -> object:
        calls.append("plan")
        raise AssertionError("gold-bearing plan was opened")

    monkeypatch.setattr(analysis, "load_s0_population", load_population)
    monkeypatch.setattr(
        analysis, "load_independent_closure_generation", fail_generation
    )
    monkeypatch.setattr(analysis, "_load_pinned_target_plan", open_plan)

    with pytest.raises(RuntimeError, match="tampered generation"):
        analysis.analyze_paths(
            retrieval_path=Path("retrieval.json"),
            expected_retrieval_sha256=SHA_A,
            generation_path=Path("generation.json"),
            expected_generation_sha256=SHA_B,
            eligibility_manifest_path=Path("eligibility.json"),
            expected_eligibility_manifest_sha256=SHA_C,
            target_plan_path=Path("target-plan.json"),
        )

    assert calls == ["population", "generation"]


def test_successful_path_projects_runtime_before_opening_target_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    population = object()
    generation = object()
    projections = object()
    events = object()
    plan = object()

    def load_population(*args: Any, **kwargs: Any) -> object:
        calls.append("population")
        return population

    def load_generation(*args: Any, **kwargs: Any) -> object:
        assert kwargs["population"] is population
        calls.append("generation")
        return generation

    def project_runtime(value: object) -> tuple[object, object]:
        assert value is generation
        calls.append("projection")
        return projections, events

    def load_plan(path: Path) -> tuple[object, str]:
        calls.append("plan")
        return plan, analysis.PINNED_TARGET_PLAN_SHA256

    def build_payload(**kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "generation": generation,
            "projections": projections,
            "events": events,
            "plan": plan,
            "target_plan_file_sha256": analysis.PINNED_TARGET_PLAN_SHA256,
        }
        calls.append("analysis")
        return {"ok": True}

    monkeypatch.setattr(analysis, "load_s0_population", load_population)
    monkeypatch.setattr(
        analysis, "load_independent_closure_generation", load_generation
    )
    monkeypatch.setattr(analysis, "_project_runtime_stages", project_runtime)
    monkeypatch.setattr(analysis, "_load_pinned_target_plan", load_plan)
    monkeypatch.setattr(analysis, "build_analysis_payload", build_payload)

    result = analysis.analyze_paths(
        retrieval_path=Path("retrieval.json"),
        expected_retrieval_sha256=SHA_A,
        generation_path=Path("generation.json"),
        expected_generation_sha256=SHA_B,
        eligibility_manifest_path=Path("eligibility.json"),
        expected_eligibility_manifest_sha256=SHA_C,
        target_plan_path=Path("target-plan.json"),
    )

    assert result == {"ok": True}
    assert calls == ["population", "generation", "projection", "plan", "analysis"]


def test_target_plan_digest_is_not_configurable(tmp_path: Path) -> None:
    path = tmp_path / "target-plan.json"
    artifact, _created = publish_sealed_json(path, {"format": "not-the-plan"})
    assert artifact.sha256 != analysis.PINNED_TARGET_PLAN_SHA256

    with pytest.raises(
        analysis.ClosureTargetAnalysisError,
        match="immutable pinned checkpoint",
    ):
        analysis._load_pinned_target_plan(path)

    parser_destinations = {action.dest for action in analysis.build_parser()._actions}
    assert "expected_target_plan_sha256" not in parser_destinations
