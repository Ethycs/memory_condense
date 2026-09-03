from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.benchmark import build_judge_prompt
from tools import plan_provider_free_differential_judge as differential
from tools.matched_eval.artifacts import SealedArtifact, read_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_arm import judge_row_projection


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _receipt(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: identity_sha256(body)}


def _question(ordinal: int) -> str:
    return f"What is remembered fact {ordinal}?"


def _reference(ordinal: int) -> str:
    return f"reference-{ordinal}"


def _parent(ordinal: int) -> str:
    return f"parent-{ordinal}"


def _policy_source(
    prediction_overrides: dict[int, str] | None = None,
) -> tuple[SealedArtifact, SealedArtifact]:
    prediction_overrides = prediction_overrides or {}
    questions: list[dict[str, Any]] = []
    projected: list[dict[str, Any]] = []
    for ordinal in range(differential.QUESTION_COUNT):
        parent = _parent(ordinal)
        prediction = prediction_overrides.get(ordinal, parent)
        body = {
            "changed_from_parent": prediction != parent,
            "dated_question_sha256": _sha(f"dated-{ordinal}"),
            "format": "synthetic-policy-result-v1",
            "gold_loaded": False,
            "ordinal": ordinal,
            "parent_prediction_sha256": quote_sha256(parent),
            "physical_provider_calls": 0,
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "prediction_source": "synthetic-policy",
            "question_id": f"q-{ordinal}",
            "question_sha256": quote_sha256(_question(ordinal)),
            "route_id": "synthetic-route",
        }
        row = _receipt(body, "source_row_sha256")
        questions.append(row)
        projected.append(judge_row_projection(row))
    run = SealedArtifact(
        Path("policy-run.json"),
        _sha("policy-run-" + repr(sorted(prediction_overrides.items()))),
        {
            "format": differential.policy_cli.RUN_FORMAT,
            "gold_loaded": False,
            "judge_rows": projected,
            "physical_provider_calls_during_revalidation": 0,
            "question_count": differential.QUESTION_COUNT,
            "questions": questions,
        },
    )
    replay = SealedArtifact(
        Path("policy-replay.json"),
        _sha("policy-replay-" + run.sha256),
        {
            "byte_identical": True,
            "expected_run_sha256": run.sha256,
            "format": differential.policy_cli.REPLAY_FORMAT,
            "gold_loaded": False,
            "physical_provider_calls": 0,
            "replayed_run_sha256": run.sha256,
        },
    )
    return run, replay


def _prior(
    name: str,
    predictions: dict[int, str],
    *,
    correct: dict[int, bool] | None = None,
    ordinals: tuple[int, ...] | None = None,
    model: str | None = differential.DEFAULT_JUDGE_MODEL,
    legacy_message_question: bool = False,
) -> differential.AuthenticatedJudgeRun:
    correct = correct or {}
    selected = ordinals or tuple(range(differential.QUESTION_COUNT))
    prompts: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    for ordinal in selected:
        question = _question(ordinal)
        reference = _reference(ordinal)
        prediction = predictions.get(ordinal, _parent(ordinal))
        messages = build_judge_prompt(question, reference, prediction)
        prompt_body = {
            "messages": messages,
            "messages_sha256": identity_sha256(messages),
            "ordinal": ordinal,
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "question": question,
            "question_id": f"q-{ordinal}",
            "question_sha256": quote_sha256(question),
            "reference": reference,
            "reference_sha256": quote_sha256(reference),
        }
        if legacy_message_question:
            del prompt_body["question"]
        prompt = _receipt(prompt_body, "prompt_row_receipt_sha256")
        prompts.append(prompt)
        verdict = correct.get(ordinal, ordinal % 3 != 0)
        output = "CORRECT synthetic" if verdict else "INCORRECT synthetic"
        judgment_body = {
            "correct": verdict,
            "judge_output": output,
            "judge_output_sha256": quote_sha256(output),
            "messages_sha256": prompt["messages_sha256"],
            "ordinal": ordinal,
            "prediction_sha256": prompt["prediction_sha256"],
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "reference_sha256": prompt["reference_sha256"],
        }
        judgments.append(_receipt(judgment_body, "judge_row_sha256"))
    preflight_payload: dict[str, Any] = {
        "gold_loaded": True,
        "physical_provider_calls": 0,
        "prompt_rows": prompts,
    }
    if legacy_message_question:
        preflight_payload["format"] = (
            "memory-condense-locked-specialist-final-reconciliation-"
            "sol-judge-preflight-v3"
        )
    if model is not None:
        preflight_payload["model"] = model
    preflight = SealedArtifact(
        Path(f"{name}-preflight.json"), _sha(f"{name}-preflight"), preflight_payload
    )
    judge_payload = {
        "gold_loaded": True,
        "preflight_artifact_sha256": preflight.sha256,
        "questions": judgments,
    }
    judge_sha = _sha(f"{name}-judge")
    judge = SealedArtifact(Path(f"{name}-judge.json"), judge_sha, judge_payload)
    replay = SealedArtifact(
        Path(f"{name}-judge-replay.json"), judge_sha, judge_payload
    )
    return differential.authenticate_prior_judge_run(preflight, judge, replay)


def test_exact_legacy_v3_message_recovers_omitted_plaintext_question() -> None:
    prior = _prior(
        "legacy-v3",
        {},
        ordinals=(5,),
        legacy_message_question=True,
    )

    assert len(prior.entries) == 1
    assert prior.entries[0]["ordinal"] == 5
    assert prior.entries[0]["question"] == _question(5)
    assert prior.entries[0]["question_sha256"] == quote_sha256(_question(5))


def test_exact_reuse_across_multiple_priors_including_parent_reversion() -> None:
    policy_run, policy_replay = _policy_source()
    v3 = _prior("v3", {})
    v5_predictions = {5: "terra-rewrite", 36: "another-rewrite"}
    v5 = _prior("v5", v5_predictions)

    plan = differential.build_differential_judge_plan(
        policy_run, policy_replay, (v5, v3)
    )

    assert plan["reused_judgment_count"] == 100
    assert plan["novel_prompt_count"] == 0
    assert plan["novel_prompt_rows"] == []
    reused_five = next(row for row in plan["reused_judgments"] if row["ordinal"] == 5)
    assert reused_five["prediction_sha256"] == quote_sha256(_parent(5))
    assert {source["judge_artifact_sha256"] for source in reused_five["source_judgments"]} == {
        v3.judge.sha256
    }


def test_prediction_mismatch_emits_only_the_novel_prompt() -> None:
    policy_run, policy_replay = _policy_source({5: "brand-new-answer"})
    prior = _prior("baseline", {})

    plan = differential.build_differential_judge_plan(
        policy_run, policy_replay, (prior,)
    )

    assert plan["reused_judgment_count"] == 99
    assert plan["novel_prompt_count"] == 1
    prompt = plan["novel_prompt_rows"][0]
    assert prompt["ordinal"] == 5
    assert prompt["prediction"] == "brand-new-answer"
    assert prompt["prediction_sha256"] == quote_sha256("brand-new-answer")
    assert prompt["messages"] == build_judge_prompt(
        _question(5), _reference(5), "brand-new-answer"
    )
    assert plan["score_emitted"] is False
    assert plan["merge_ready"] is False


def test_conflicting_exact_prior_judgments_fail_closed() -> None:
    policy_run, policy_replay = _policy_source()
    first = _prior("first", {}, correct={5: True})
    second = _prior("second", {}, correct={5: False})

    with pytest.raises(
        differential.DifferentialJudgePlannerError,
        match="conflicting authenticated prior judgments",
    ):
        differential.build_differential_judge_plan(
            policy_run, policy_replay, (first, second)
        )


def test_merge_requires_novel_judgment_then_reconstructs_all_100() -> None:
    policy_run, policy_replay = _policy_source({5: "brand-new-answer"})
    prior = _prior("baseline-merge", {})
    plan_payload = differential.build_differential_judge_plan(
        policy_run, policy_replay, (prior,)
    )
    plan = SealedArtifact(Path("plan.json"), _sha("plan"), plan_payload)

    with pytest.raises(
        differential.DifferentialJudgeIncompleteError,
        match="novel judgments are required before scoring: 5",
    ):
        differential.merge_differential_judgments(plan)

    novel = _prior(
        "novel-five",
        {5: "brand-new-answer"},
        correct={5: True},
        ordinals=(5,),
    )
    merged = differential.merge_differential_judgments(plan, (novel,))

    assert merged["score_complete"] is True
    assert merged["question_count"] == 100
    assert len(merged["questions"]) == 100
    assert [row["ordinal"] for row in merged["questions"]] == list(range(100))
    row_five = merged["questions"][5]
    assert row_five["judgment_source"] == "authenticated_novel_judgment"
    assert row_five["correct"] is True
    assert merged["correct"] == sum(row["correct"] for row in merged["questions"])
    assert merged["accuracy"] == merged["correct"] / 100
    assert merged["physical_provider_calls_during_merge"] == 0


def test_model_mismatch_is_not_reused() -> None:
    policy_run, policy_replay = _policy_source()
    prior = _prior("other-model", {}, model="different-sol-model")

    plan = differential.build_differential_judge_plan(
        policy_run,
        policy_replay,
        (prior,),
        judge_model=differential.DEFAULT_JUDGE_MODEL,
    )

    assert plan["reused_judgment_count"] == 0
    assert plan["novel_prompt_count"] == 100


def test_plan_and_complete_merge_publish_as_distinct_sealed_artifacts(
    tmp_path: Path,
) -> None:
    policy_run, policy_replay = _policy_source({5: "brand-new-answer"})
    prior = _prior("sealed-baseline", {})
    plan, plan_created = differential.publish_differential_judge_plan(
        tmp_path, policy_run, policy_replay, (prior,)
    )
    novel = _prior(
        "sealed-novel",
        {5: "brand-new-answer"},
        correct={5: True},
        ordinals=(5,),
    )
    merge, merge_created = differential.publish_differential_judge_merge(
        tmp_path, plan, (novel,)
    )

    assert plan_created is merge_created is True
    assert plan.path != merge.path
    assert read_sealed_json(tmp_path / differential.PLAN_NAME).sha256 == plan.sha256
    assert read_sealed_json(tmp_path / differential.MERGE_NAME).sha256 == merge.sha256
    assert plan.payload["novel_prompt_count"] == 1
    assert merge.payload["score_complete"] is True
    assert len(merge.payload["questions"]) == 100


def test_cli_has_no_provider_or_ordinal_execution_path() -> None:
    parser = differential.build_parser()
    choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    assert set(choices) == {"plan", "merge"}
    for command in choices.values():
        options = {
            option
            for action in command._actions  # noqa: SLF001
            for option in action.option_strings
        }
        assert "--ordinal" not in options
        assert "--enable-provider" not in options
        assert "--authorized-provider-calls" not in options
    with pytest.raises(SystemExit):
        parser.parse_args(["provider-run"])
