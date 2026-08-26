from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from tools import compose_routed_mechanisms as composer


def _baseline(root: Path) -> tuple[str, str]:
    answers: dict[str, Any] = {
        "format": "memory-condense-recall-guarded-fixed-stage-final-answers-v1",
        "gold_fields_present": False,
        "question_count": 2,
        "retrieval_sha256": "a" * 64,
        "population_identity_sha256": "b" * 64,
        "questions": [
            {
                "ordinal": ordinal,
                "question_id": f"q{ordinal}",
                "answer": {
                    "text": text,
                    "sha256": composer._text_digest(text),
                },
            }
            for ordinal, text in enumerate(("base-zero", "base-one"))
        ],
    }
    answers_sha = composer._publish(root / "final-answers.json", answers)
    judge: dict[str, Any] = {
        "format": "memory-condense-fixed-stage-final-answer-semantic-judge-score-v1",
        "question_count": 2,
        "retrieval_sha256": "a" * 64,
        "population_identity_sha256": "b" * 64,
        "final_answer_artifact_sha256": answers_sha,
        "aggregate": {"correct": 1},
        "questions": [
            {
                "ordinal": ordinal,
                "question_id": f"q{ordinal}",
                "prediction_sha256": answers["questions"][ordinal]["answer"][
                    "sha256"
                ],
                "correct": ordinal == 1,
            }
            for ordinal in range(2)
        ],
    }
    judge_sha = composer._publish(
        root / "final-answer-semantic-judge-sol.json", judge
    )
    return answers_sha, judge_sha


def _arm(
    root: Path,
    *,
    directory: str,
    style: str,
    baseline_answers_sha: str,
    baseline_judge_sha: str,
    bad_baseline_binding: bool = False,
) -> composer.ArmSpec:
    route_styles = ("numeric_reduce", "direct_extract")
    eligible_ordinal = route_styles.index(style)
    baseline_text = ("base-zero", "base-one")
    baseline_correct = (False, True)
    plan_questions = [
        {
            "ordinal": ordinal,
            "question_id": f"q{ordinal}",
            "question_sha256": str(ordinal + 1) * 64,
            "dated_question_sha256": str(ordinal + 3) * 64,
            "adapter_row_binding_sha256": str(ordinal + 5) * 64,
            "baseline_prediction_sha256": composer._text_digest(
                baseline_text[ordinal]
            ),
            "eligible": ordinal == eligible_ordinal,
            "route": {
                "format": "test-route-v1",
                "style": route_styles[ordinal],
                "receipt_sha256": str(ordinal + 7) * 64,
            },
        }
        for ordinal in range(2)
    ]
    plan = {
        "format": "memory-condense-routed-full-source-route-plan-v1",
        "question_count": 2,
        "treatment_style": style,
        "eligible_question_count": 1,
        "required_authorized_compression_calls": 1,
        "retrieval_sha256": "a" * 64,
        "population_identity_sha256": "b" * 64,
        "baseline_final_answers_sha256": (
            "f" * 64 if bad_baseline_binding else baseline_answers_sha
        ),
        "provider_calls": 0,
        "gold_loaded": False,
        "questions": plan_questions,
    }
    arm_root = root / directory
    plan_sha = composer._publish(arm_root / "route-plan.json", plan)
    candidate = "numeric-answer" if style == "numeric_reduce" else "direct-answer"
    run_questions = []
    verdict_questions = []
    candidate_correct = []
    for ordinal, route_style in enumerate(route_styles):
        eligible = route_style == style
        prediction = candidate if eligible else baseline_text[ordinal]
        prediction_sha = composer._text_digest(prediction)
        correct = (
            True
            if style == "numeric_reduce" and eligible
            else False
            if style == "direct_extract" and eligible
            else baseline_correct[ordinal]
        )
        candidate_correct.append(correct)
        run_questions.append(
            {
                "ordinal": ordinal,
                "question_id": f"q{ordinal}",
                "eligible": eligible,
                "route_style": route_style,
                "prediction": prediction,
                "prediction_sha256": prediction_sha,
                "baseline_prediction_sha256": composer._text_digest(
                    baseline_text[ordinal]
                ),
                "changed_from_baseline": eligible,
            }
        )
        verdict_questions.append(
            {
                "ordinal": ordinal,
                "question_id": f"q{ordinal}",
                "eligible": eligible,
                "route_style": route_style,
                "prediction_sha256": prediction_sha,
                "baseline_prediction_sha256": composer._text_digest(
                    baseline_text[ordinal]
                ),
                "changed_from_baseline": eligible,
                "baseline_correct": baseline_correct[ordinal],
                "correct": correct,
                "rescued": correct and not baseline_correct[ordinal],
                "regressed": baseline_correct[ordinal] and not correct,
            }
        )
    rescued = sum(
        row["eligible"] and row["correct"] and not row["baseline_correct"]
        for row in verdict_questions
    )
    regressed = sum(
        row["eligible"] and row["baseline_correct"] and not row["correct"]
        for row in verdict_questions
    )
    run = {
        "format": "memory-condense-routed-full-source-run-v1",
        "question_count": 2,
        "treatment_style": style,
        "eligible_question_count": 1,
        "valid_compression_count": 1,
        "baseline_fallback_count": 0,
        "required_authorized_answer_calls": 1,
        "total_sealed_terra_calls": 2,
        "retrieval_sha256": "a" * 64,
        "population_identity_sha256": "b" * 64,
        "baseline_final_answers_sha256": baseline_answers_sha,
        "route_plan_sha256": plan_sha,
        "gold_loaded": False,
        "settings": {"max_prompt_tokens": 8000},
        "budget": {
            "compression_prompt_tokens": {"total": 100},
            "answer_prompt_tokens": {"total": 80},
        },
        "questions": run_questions,
    }
    run_sha = composer._publish(arm_root / "run.json", run)
    composer._publish(arm_root / "run-replay.json", run)
    judge = {
        "format": "memory-condense-routed-full-source-sol-judge-v1",
        "question_count": 2,
        "changed_eligible_prediction_count": 1,
        "unique_sol_completion_count": 1,
        "explicit_gold_answer_field_persisted": False,
        "campaign_binding": {
            "treatment_run_sha256": run_sha,
            "route_plan_sha256": plan_sha,
            "baseline_judge_sha256": baseline_judge_sha,
            "retrieval_sha256": "a" * 64,
            "population_identity_sha256": "b" * 64,
            "question_count": 2,
        },
        "aggregate": {
            "baseline_correct": 1,
            "candidate_correct": sum(candidate_correct),
            "eligible_rescued": rescued,
            "eligible_regressed": regressed,
            "eligible_net_marginal": rescued - regressed,
        },
        "questions": verdict_questions,
    }
    judge_sha = composer._publish(arm_root / "semantic-judge-sol.json", judge)
    composer._publish(arm_root / "semantic-judge-sol-replay.json", judge)
    return composer.ArmSpec(directory, style, plan_sha, run_sha, judge_sha)


def _fixture(
    tmp_path: Path, *, bad_baseline_binding: bool = False
) -> tuple[Path, Path, tuple[composer.ArmSpec, ...], str, str]:
    baseline_root = tmp_path / "baseline"
    answers_sha, judge_sha = _baseline(baseline_root)
    input_root = tmp_path / "arms"
    specs = (
        _arm(
            input_root,
            directory="numeric-v1",
            style="numeric_reduce",
            baseline_answers_sha=answers_sha,
            baseline_judge_sha=judge_sha,
            bad_baseline_binding=bad_baseline_binding,
        ),
        _arm(
            input_root,
            directory="direct-v1",
            style="direct_extract",
            baseline_answers_sha=answers_sha,
            baseline_judge_sha=judge_sha,
        ),
    )
    return input_root, baseline_root, specs, answers_sha, judge_sha


def test_positive_only_composition_is_numeric_projection_and_provider_free(
    tmp_path: Path,
) -> None:
    input_root, baseline_root, specs, answers_sha, judge_sha = _fixture(tmp_path)
    output = tmp_path / "combined"
    result = composer.compose(
        input_root=input_root,
        baseline_root=baseline_root,
        output_root=output,
        arms=specs,
        expected_count=2,
        baseline_answers_sha256=answers_sha,
        baseline_judge_sha256=judge_sha,
    )

    assert result["ledger"]["accepted_styles"] == ["numeric_reduce"]
    assert result["ledger"]["rejected_styles"] == ["direct_extract"]
    assert result["score"]["aggregate"] == {
        "baseline_correct": 1,
        "candidate_correct": 2,
        "rescued": 1,
        "regressed": 0,
        "net_marginal": 1,
    }
    assert [row["prediction"] for row in result["run"]["questions"]] == [
        "numeric-answer",
        "base-one",
    ]
    assert result["run"]["prediction_projection_sha256"] == result["run"][
        "numeric_v1_prediction_projection_sha256"
    ]
    assert all(result[name]["provider_calls"] == 0 for name in ("ledger", "run", "score"))
    assert all(
        (output / name).is_file() and (output / f"{name}.sha256").is_file()
        for name in (
            "route-budget-ledger.json",
            "run.json",
            "semantic-judge-sol.json",
        )
    )
    replayed = composer.compose(
        input_root=input_root,
        baseline_root=baseline_root,
        output_root=output,
        arms=specs,
        expected_count=2,
        baseline_answers_sha256=answers_sha,
        baseline_judge_sha256=judge_sha,
    )
    assert replayed["sha256"] == result["sha256"]


def test_rejects_nonidentical_run_replay(tmp_path: Path) -> None:
    input_root, baseline_root, specs, answers_sha, judge_sha = _fixture(tmp_path)
    replay_path = input_root / "numeric-v1" / "run-replay.json"
    changed = copy.deepcopy(composer._read(replay_path, specs[0].run_sha256)[0])
    changed["questions"][0]["prediction"] = "tampered"
    replay_path.unlink()
    replay_path.with_name(replay_path.name + ".sha256").unlink()
    composer._publish(replay_path, changed)

    with pytest.raises(composer.CompositionError, match="artifact hash changed"):
        composer.compose(
            input_root=input_root,
            baseline_root=baseline_root,
            output_root=tmp_path / "combined",
            arms=specs,
            expected_count=2,
            baseline_answers_sha256=answers_sha,
            baseline_judge_sha256=judge_sha,
        )


def test_rejects_arm_with_changed_baseline_binding(tmp_path: Path) -> None:
    input_root, baseline_root, specs, answers_sha, judge_sha = _fixture(
        tmp_path, bad_baseline_binding=True
    )
    with pytest.raises(composer.CompositionError, match="baseline or campaign binding"):
        composer.compose(
            input_root=input_root,
            baseline_root=baseline_root,
            output_root=tmp_path / "combined",
            arms=specs,
            expected_count=2,
            baseline_answers_sha256=answers_sha,
            baseline_judge_sha256=judge_sha,
        )
