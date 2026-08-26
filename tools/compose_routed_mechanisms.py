#!/usr/bin/env python3
"""Compose independently judged retrieval mechanisms without provider access.

Every source arm is first verified against its canonical artifact, byte-identical
run/judge replay, sealed baseline, and shared 100-question routing partition.
Only arms with a strictly positive semantic net marginal may replace baseline
predictions.  No benchmark corpus or raw gold answer is loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # support ``python tools/compose_...py``
    _REPOSITORY = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_REPOSITORY / "src"), str(_REPOSITORY)]

from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval.run_fast_1m_em_facts import _publish


LEDGER_FORMAT = "memory-condense-routed-mechanism-budget-ledger-v1"
RUN_FORMAT = "memory-condense-routed-mechanism-combined-run-v1"
SCORE_FORMAT = "memory-condense-routed-mechanism-combined-semantic-score-v1"

DEFAULT_ROOT = Path(
    "eval_results/longmemeval-1m-routed-full-source-repair-20260826"
)
DEFAULT_BASELINE_ROOT = Path(
    "eval_results/longmemeval-1m-fixed-s1-validation-20260826"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "combined-v1"
EXPECTED_BASELINE_ANSWERS_SHA256 = (
    "d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd"
)
EXPECTED_BASELINE_JUDGE_SHA256 = (
    "5dc56a240315c5577d1032d40429df7e39adad0f40a098abc371ee2ea2ec77df"
)


class CompositionError(ValueError):
    """Raised when a sealed source or composition invariant changes."""


@dataclass(frozen=True, slots=True)
class ArmSpec:
    directory: str
    style: str
    route_plan_sha256: str
    run_sha256: str
    judge_sha256: str


DEFAULT_ARMS = (
    ArmSpec(
        "numeric-v1",
        "numeric_reduce",
        "11ff958c4a9a4c46fc67671775c20de86be04021e619a87157d0a5bb39a07972",
        "793a487b7a16b5ce3c6acc072abcee15001ca6a2c30d58fcb2a0cc159582ad8f",
        "84cc3d0ccd69fc6b690faa639217b0648451d16367a3ccdf9fea2b7f52a49962",
    ),
    ArmSpec(
        "direct-extract-v1",
        "direct_extract",
        "1c4fa32bc6f13c9dda92848bb1303da169ca366cc6a30e99bc51ef487525ed1a",
        "91ab8473efed3787f7e0f9d1bfdf326e0eb5b72cfa3b3397da8283615496fc48",
        "66ad80b140143607e5beb578c91180714d8c628f8aaeb99d5e97215fcf2b993b",
    ),
    ArmSpec(
        "synthesize-v1",
        "synthesize",
        "0a1005e74c85b61023164e676ab988af997f6d7e5ecfd1c5681778b43bfee5fd",
        "6cadd1f837ff250df3921304f620ff8fb0ec03de6c8a1c6c9377e75d8e8e37af",
        "1ffb2d6cabe4169f46825ba4929f6ac0a45bb9c1b13e5cbd9eb0f6b02771cae3",
    ),
    ArmSpec(
        "temporal-timeline-v1",
        "temporal_timeline",
        "50896d5efa73b6e5aba0eb6053cf98cd4c3868b5d248608bfc289b5a222ab6c0",
        "f94293146989686c09331fbb3ee5c8614077303a67015b537b9280a4f00a0ddd",
        "0a493ea3ce6e9e463ef8be420e78f587e2364d17d546809cce2bc77cd5aa89a2",
    ),
    ArmSpec(
        "set-join-v1",
        "set_join",
        "f349339faf01ebfaa0de61dd2d53d46049acbde7c3692762a140e1a85846655e",
        "109767f7542f127ca1992fd17286fbbb0fb9d37d8ad355190620e0b2de885fc6",
        "30124462fc48377bf82f3abf9f7b2e99608bcfe4e827d4ea03ba1eb51a5c9c54",
    ),
    ArmSpec(
        "state-chain-v1",
        "state_chain",
        "9c4756d6e1993c22071a18f1ec395e9b2d512e9cfc6007c06424a97534784942",
        "c655a581ba0cfcd9845cc625618f6997ab6c0790370fc8151d7d7b4959e59b4c",
        "10c41ede5aee22be219761fa7a1ae0a25b75c90268398bba02af27b392f6edf0",
    ),
)


@dataclass(frozen=True, slots=True)
class _Baseline:
    answers: Mapping[str, Any]
    answers_sha256: str
    judge: Mapping[str, Any]
    judge_sha256: str


@dataclass(frozen=True, slots=True)
class _Arm:
    spec: ArmSpec
    plan: Mapping[str, Any]
    run: Mapping[str, Any]
    judge: Mapping[str, Any]
    routing_projection: tuple[dict[str, Any], ...]
    budget: dict[str, Any]
    accepted: bool


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompositionError(message)


def _read(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"not a regular artifact: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompositionError(f"invalid JSON artifact: {path}") from exc
    _require(type(value) is dict, f"artifact root is not an object: {path}")
    _require(raw == canonical_json_bytes(value), f"artifact is not canonical: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    _require(digest == expected_sha256, f"artifact hash changed: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{digest}  {path.name}\n".encode()
    _require(
        sidecar.is_file()
        and not sidecar.is_symlink()
        and sidecar.read_bytes() == expected_sidecar,
        f"artifact sidecar changed: {path}",
    )
    return value, digest


def _questions(artifact: Mapping[str, Any], count: int, label: str) -> list[Any]:
    rows = artifact.get("questions")
    _require(
        artifact.get("question_count") == count
        and isinstance(rows, list)
        and len(rows) == count,
        f"{label} is not the sealed {count}-question population",
    )
    return rows


def _load_baseline(
    root: Path,
    *,
    expected_count: int,
    answers_sha256: str,
    judge_sha256: str,
) -> _Baseline:
    answers, actual_answers_sha = _read(root / "final-answers.json", answers_sha256)
    judge, actual_judge_sha = _read(
        root / "final-answer-semantic-judge-sol.json", judge_sha256
    )
    answer_rows = _questions(answers, expected_count, "baseline answers")
    judge_rows = _questions(judge, expected_count, "baseline judge")
    _require(
        answers.get("format")
        == "memory-condense-recall-guarded-fixed-stage-final-answers-v1"
        and answers.get("gold_fields_present") is False,
        "baseline answer format or gold firewall changed",
    )
    _require(
        judge.get("format")
        == "memory-condense-fixed-stage-final-answer-semantic-judge-score-v1"
        and judge.get("final_answer_artifact_sha256") == actual_answers_sha
        and judge.get("retrieval_sha256") == answers.get("retrieval_sha256")
        and judge.get("population_identity_sha256")
        == answers.get("population_identity_sha256"),
        "baseline judge binding changed",
    )
    correct = 0
    for ordinal, (answer_row, judge_row) in enumerate(
        zip(answer_rows, judge_rows, strict=True)
    ):
        answer = answer_row.get("answer") if isinstance(answer_row, Mapping) else None
        text = answer.get("text") if isinstance(answer, Mapping) else None
        prediction_sha = answer.get("sha256") if isinstance(answer, Mapping) else None
        _require(
            isinstance(text, str)
            and prediction_sha == _text_digest(text)
            and answer_row.get("ordinal") == ordinal
            and judge_row.get("ordinal") == ordinal
            and answer_row.get("question_id") == judge_row.get("question_id")
            and judge_row.get("prediction_sha256") == prediction_sha
            and type(judge_row.get("correct")) is bool,
            f"baseline question binding changed at ordinal {ordinal}",
        )
        correct += int(judge_row["correct"])
    _require(
        judge.get("aggregate", {}).get("correct") == correct,
        "baseline semantic aggregate changed",
    )
    return _Baseline(answers, actual_answers_sha, judge, actual_judge_sha)


def _routing_projection(rows: list[Any]) -> tuple[dict[str, Any], ...]:
    keys = (
        "ordinal",
        "question_id",
        "question_sha256",
        "dated_question_sha256",
        "adapter_row_binding_sha256",
        "baseline_prediction_sha256",
        "route",
    )
    return tuple({key: row.get(key) for key in keys} for row in rows)


def _load_arm(
    root: Path,
    spec: ArmSpec,
    baseline: _Baseline,
    expected_count: int,
) -> _Arm:
    arm_root = root / spec.directory
    plan, plan_sha = _read(arm_root / "route-plan.json", spec.route_plan_sha256)
    run, run_sha = _read(arm_root / "run.json", spec.run_sha256)
    replay, replay_sha = _read(arm_root / "run-replay.json", spec.run_sha256)
    judge, judge_sha = _read(
        arm_root / "semantic-judge-sol.json", spec.judge_sha256
    )
    judge_replay, judge_replay_sha = _read(
        arm_root / "semantic-judge-sol-replay.json", spec.judge_sha256
    )
    _require(
        run == replay and run_sha == replay_sha,
        f"{spec.style} run replay is not byte-identical",
    )
    _require(
        judge == judge_replay and judge_sha == judge_replay_sha,
        f"{spec.style} judge replay is not byte-identical",
    )
    plan_rows = _questions(plan, expected_count, f"{spec.style} route plan")
    run_rows = _questions(run, expected_count, f"{spec.style} run")
    judge_rows = _questions(judge, expected_count, f"{spec.style} judge")
    common = (
        plan.get("format") == "memory-condense-routed-full-source-route-plan-v1"
        and run.get("format") == "memory-condense-routed-full-source-run-v1"
        and judge.get("format")
        == "memory-condense-routed-full-source-sol-judge-v1"
        and plan.get("treatment_style") == spec.style
        and run.get("treatment_style") == spec.style
        and plan.get("retrieval_sha256") == baseline.answers.get("retrieval_sha256")
        and run.get("retrieval_sha256") == baseline.answers.get("retrieval_sha256")
        and plan.get("population_identity_sha256")
        == baseline.answers.get("population_identity_sha256")
        and run.get("population_identity_sha256")
        == baseline.answers.get("population_identity_sha256")
        and plan.get("baseline_final_answers_sha256") == baseline.answers_sha256
        and run.get("baseline_final_answers_sha256") == baseline.answers_sha256
        and run.get("route_plan_sha256") == plan_sha
        and plan.get("provider_calls") == 0
        and plan.get("gold_loaded") is False
        and run.get("gold_loaded") is False
        and judge.get("explicit_gold_answer_field_persisted") is False
    )
    binding = judge.get("campaign_binding")
    _require(
        common
        and isinstance(binding, Mapping)
        and binding.get("treatment_run_sha256") == run_sha
        and binding.get("route_plan_sha256") == plan_sha
        and binding.get("baseline_judge_sha256") == baseline.judge_sha256
        and binding.get("retrieval_sha256") == baseline.answers.get("retrieval_sha256")
        and binding.get("population_identity_sha256")
        == baseline.answers.get("population_identity_sha256")
        and binding.get("question_count") == expected_count,
        f"{spec.style} baseline or campaign binding changed",
    )

    baseline_answers = baseline.answers["questions"]
    baseline_judgments = baseline.judge["questions"]
    eligible = changed = correct = rescued = regressed = 0
    for ordinal, (planned, source, verdict, base_answer, base_verdict) in enumerate(
        zip(
            plan_rows,
            run_rows,
            judge_rows,
            baseline_answers,
            baseline_judgments,
            strict=True,
        )
    ):
        qid = base_answer["question_id"]
        route = planned.get("route")
        route_style = route.get("style") if isinstance(route, Mapping) else None
        is_eligible = route_style == spec.style
        prediction = source.get("prediction")
        baseline_text = base_answer["answer"]["text"]
        baseline_sha = base_answer["answer"]["sha256"]
        is_changed = source.get("prediction_sha256") != baseline_sha
        base_correct = base_verdict["correct"]
        candidate_correct = verdict.get("correct")
        row_ok = (
            planned.get("ordinal") == ordinal
            and source.get("ordinal") == ordinal
            and verdict.get("ordinal") == ordinal
            and planned.get("question_id") == qid
            and source.get("question_id") == qid
            and verdict.get("question_id") == qid
            and planned.get("baseline_prediction_sha256") == baseline_sha
            and source.get("baseline_prediction_sha256") == baseline_sha
            and verdict.get("baseline_prediction_sha256") == baseline_sha
            and planned.get("eligible") is is_eligible
            and source.get("eligible") is is_eligible
            and verdict.get("eligible") is is_eligible
            and source.get("route_style") == route_style
            and verdict.get("route_style") == route_style
            and isinstance(prediction, str)
            and source.get("prediction_sha256") == _text_digest(prediction)
            and verdict.get("prediction_sha256") == source.get("prediction_sha256")
            and source.get("changed_from_baseline") is is_changed
            and verdict.get("changed_from_baseline") is is_changed
            and verdict.get("baseline_correct") is base_correct
            and type(candidate_correct) is bool
            and verdict.get("rescued") is (candidate_correct and not base_correct)
            and verdict.get("regressed") is (base_correct and not candidate_correct)
        )
        if not is_eligible:
            row_ok = row_ok and prediction == baseline_text and not is_changed
            row_ok = row_ok and candidate_correct is base_correct
        _require(row_ok, f"{spec.style} question binding changed at ordinal {ordinal}")
        eligible += int(is_eligible)
        changed += int(is_eligible and is_changed)
        correct += int(candidate_correct)
        rescued += int(is_eligible and candidate_correct and not base_correct)
        regressed += int(is_eligible and base_correct and not candidate_correct)

    aggregate = judge.get("aggregate")
    baseline_correct = baseline.judge["aggregate"]["correct"]
    valid = run.get("valid_compression_count")
    fallback = run.get("baseline_fallback_count")
    answer_calls = run.get("required_authorized_answer_calls")
    compression_calls = plan.get("required_authorized_compression_calls")
    _require(
        isinstance(aggregate, Mapping)
        and aggregate.get("baseline_correct") == baseline_correct
        and aggregate.get("candidate_correct") == correct
        and aggregate.get("eligible_rescued") == rescued
        and aggregate.get("eligible_regressed") == regressed
        and aggregate.get("eligible_net_marginal") == rescued - regressed
        and plan.get("eligible_question_count") == eligible
        and run.get("eligible_question_count") == eligible
        and compression_calls == eligible
        and isinstance(valid, int)
        and isinstance(fallback, int)
        and valid + fallback == eligible
        and answer_calls == valid
        and run.get("total_sealed_terra_calls") == compression_calls + answer_calls
        and judge.get("changed_eligible_prediction_count") == changed
        and judge.get("unique_sol_completion_count") == changed,
        f"{spec.style} aggregate or independent budget changed",
    )
    budget = {
        "directory": spec.directory,
        "style": spec.style,
        "accepted": rescued - regressed > 0,
        "acceptance_reason": (
            "positive_semantic_net_marginal"
            if rescued - regressed > 0
            else "nonpositive_semantic_net_marginal"
        ),
        "input_sha256": {
            "route_plan": plan_sha,
            "run_and_replay": run_sha,
            "semantic_judge_and_replay": judge_sha,
        },
        "question_budget": {
            "eligible": eligible,
            "valid_compressions": valid,
            "baseline_fallbacks": fallback,
            "changed_candidates_judged": changed,
        },
        "provider_call_budget_measured": {
            "terra_compression": compression_calls,
            "terra_answer": answer_calls,
            "terra_total": compression_calls + answer_calls,
            "sol_judge": changed,
        },
        "prompt_token_budget": {
            "workspace_cap_per_prompt": run.get("settings", {}).get(
                "max_prompt_tokens"
            ),
            "compression": run.get("budget", {}).get("compression_prompt_tokens"),
            "answer": run.get("budget", {}).get("answer_prompt_tokens"),
        },
        "semantic_result": {
            "baseline_correct": baseline_correct,
            "candidate_correct": correct,
            "rescued": rescued,
            "regressed": regressed,
            "net_marginal": rescued - regressed,
        },
    }
    return _Arm(
        spec,
        plan,
        run,
        judge,
        _routing_projection(plan_rows),
        budget,
        rescued - regressed > 0,
    )


def compose(
    *,
    input_root: Path = DEFAULT_ROOT,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    output_root: Path = DEFAULT_OUTPUT,
    arms: tuple[ArmSpec, ...] = DEFAULT_ARMS,
    expected_count: int = 100,
    baseline_answers_sha256: str = EXPECTED_BASELINE_ANSWERS_SHA256,
    baseline_judge_sha256: str = EXPECTED_BASELINE_JUDGE_SHA256,
) -> dict[str, Any]:
    """Verify, gate, compose, and canonically publish one provider-free result."""

    _require(expected_count > 0 and arms, "composition population must be nonempty")
    _require(
        len({arm.style for arm in arms}) == len(arms),
        "arm styles must be unique",
    )
    baseline = _load_baseline(
        baseline_root,
        expected_count=expected_count,
        answers_sha256=baseline_answers_sha256,
        judge_sha256=baseline_judge_sha256,
    )
    loaded = tuple(
        _load_arm(input_root, spec, baseline, expected_count) for spec in arms
    )
    shared_routing = loaded[0].routing_projection
    _require(
        all(arm.routing_projection == shared_routing for arm in loaded),
        "independent arms do not share one routing population",
    )
    arm_by_style = {arm.spec.style: arm for arm in loaded}
    route_styles = [row["route"]["style"] for row in shared_routing]
    _require(
        len(route_styles) == expected_count
        and all(style in arm_by_style for style in route_styles),
        "route partition is incomplete",
    )
    for arm in loaded:
        _require(
            sum(style == arm.spec.style for style in route_styles)
            == arm.budget["question_budget"]["eligible"],
            f"{arm.spec.style} eligibility does not match the shared partition",
        )

    base_answers = baseline.answers["questions"]
    base_judge = baseline.judge["questions"]
    run_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for ordinal, (route_row, answer_row, judge_row) in enumerate(
        zip(shared_routing, base_answers, base_judge, strict=True)
    ):
        style = route_styles[ordinal]
        arm = arm_by_style[style]
        selected = arm.accepted
        source_run = arm.run["questions"][ordinal] if selected else None
        source_judge = arm.judge["questions"][ordinal] if selected else None
        baseline_prediction = answer_row["answer"]["text"]
        baseline_prediction_sha = answer_row["answer"]["sha256"]
        prediction = (
            source_run["prediction"] if source_run is not None else baseline_prediction
        )
        prediction_sha = _text_digest(prediction)
        correct = (
            source_judge["correct"] if source_judge is not None else judge_row["correct"]
        )
        baseline_correct = judge_row["correct"]
        provenance = (
            {
                "kind": "accepted_route",
                "style": style,
                "source_artifact_sha256": arm.spec.run_sha256,
                "source_question_row_sha256": _digest(source_run),
            }
            if selected
            else {
                "kind": "sealed_baseline",
                "style": style,
                "source_artifact_sha256": baseline.answers_sha256,
                "source_question_row_sha256": _digest(answer_row),
            }
        )
        run_rows.append(
            {
                "ordinal": ordinal,
                "question_id": answer_row["question_id"],
                "routed_style": style,
                "selection": provenance,
                "prediction": prediction,
                "prediction_sha256": prediction_sha,
                "baseline_prediction_sha256": baseline_prediction_sha,
                "changed_from_baseline": prediction_sha != baseline_prediction_sha,
            }
        )
        score_rows.append(
            {
                "ordinal": ordinal,
                "question_id": answer_row["question_id"],
                "routed_style": style,
                "verdict_source": (
                    "accepted_route_semantic_verdict"
                    if selected
                    else "sealed_baseline_semantic_verdict"
                ),
                "source_artifact_sha256": (
                    arm.spec.judge_sha256 if selected else baseline.judge_sha256
                ),
                "prediction_sha256": prediction_sha,
                "baseline_correct": baseline_correct,
                "correct": correct,
                "rescued": correct and not baseline_correct,
                "regressed": baseline_correct and not correct,
            }
        )

    projection = [
        {
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "prediction": row["prediction"],
            "prediction_sha256": row["prediction_sha256"],
        }
        for row in run_rows
    ]
    accepted = [arm for arm in loaded if arm.accepted]
    rejected = [arm for arm in loaded if not arm.accepted]
    baseline_correct = baseline.judge["aggregate"]["correct"]
    candidate_correct = sum(row["correct"] for row in score_rows)
    ledger = {
        "format": LEDGER_FORMAT,
        "admission_policy": {
            "metric": "independent_sealed_semantic_net_marginal",
            "operator": ">",
            "threshold": 0,
            "zero_marginal_is_accepted": False,
        },
        "question_count": expected_count,
        "routing_projection_sha256": _digest(list(shared_routing)),
        "baseline_binding": {
            "final_answers_sha256": baseline.answers_sha256,
            "semantic_judge_sha256": baseline.judge_sha256,
            "retrieval_sha256": baseline.answers["retrieval_sha256"],
            "population_identity_sha256": baseline.answers[
                "population_identity_sha256"
            ],
            "semantic_correct": baseline_correct,
        },
        "method_budgets": [arm.budget for arm in loaded],
        "accepted_styles": [arm.spec.style for arm in accepted],
        "rejected_styles": [arm.spec.style for arm in rejected],
        "composition_result": {
            "accepted_route_questions": sum(
                arm.budget["question_budget"]["eligible"] for arm in accepted
            ),
            "baseline_route_questions": expected_count
            - sum(arm.budget["question_budget"]["eligible"] for arm in accepted),
            "baseline_correct": baseline_correct,
            "candidate_correct": candidate_correct,
            "net_marginal": candidate_correct - baseline_correct,
        },
        "provider_calls": 0,
        "raw_gold_loaded": False,
    }
    ledger_sha = _publish(output_root / "route-budget-ledger.json", ledger)
    combined_run = {
        "format": RUN_FORMAT,
        "route_budget_ledger_sha256": ledger_sha,
        "question_count": expected_count,
        "baseline_final_answers_sha256": baseline.answers_sha256,
        "retrieval_sha256": baseline.answers["retrieval_sha256"],
        "population_identity_sha256": baseline.answers["population_identity_sha256"],
        "accepted_styles": ledger["accepted_styles"],
        "prediction_projection_sha256": _digest(projection),
        "questions": run_rows,
        "provider_calls": 0,
        "raw_gold_loaded": False,
    }
    numeric = arm_by_style.get("numeric_reduce")
    if numeric is not None and [arm.spec.style for arm in accepted] == [
        "numeric_reduce"
    ]:
        numeric_projection = [
            {
                "ordinal": row["ordinal"],
                "question_id": row["question_id"],
                "prediction": row["prediction"],
                "prediction_sha256": row["prediction_sha256"],
            }
            for row in numeric.run["questions"]
        ]
        _require(
            canonical_json_bytes(projection) == canonical_json_bytes(numeric_projection),
            "numeric-only composition is not byte-identical to numeric-v1",
        )
        combined_run["numeric_v1_prediction_projection_sha256"] = _digest(
            numeric_projection
        )
    run_sha = _publish(output_root / "run.json", combined_run)
    combined_score = {
        "format": SCORE_FORMAT,
        "route_budget_ledger_sha256": ledger_sha,
        "combined_run_sha256": run_sha,
        "baseline_semantic_judge_sha256": baseline.judge_sha256,
        "question_count": expected_count,
        "accepted_styles": ledger["accepted_styles"],
        "aggregate": {
            "baseline_correct": baseline_correct,
            "candidate_correct": candidate_correct,
            "rescued": sum(row["rescued"] for row in score_rows),
            "regressed": sum(row["regressed"] for row in score_rows),
            "net_marginal": candidate_correct - baseline_correct,
        },
        "questions": score_rows,
        "provider_calls": 0,
        "raw_gold_loaded": False,
        "semantic_verdicts_reused_from_sealed_artifacts": True,
    }
    score_sha = _publish(output_root / "semantic-judge-sol.json", combined_score)
    return {
        "ledger": ledger,
        "run": combined_run,
        "score": combined_score,
        "sha256": {
            "route_budget_ledger": ledger_sha,
            "run": run_sha,
            "semantic_judge_sol": score_sha,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = compose(
        input_root=args.input_root,
        baseline_root=args.baseline_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "accepted_styles": result["ledger"]["accepted_styles"],
                "candidate_correct": result["score"]["aggregate"][
                    "candidate_correct"
                ],
                "provider_calls": 0,
                "sha256": result["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ArmSpec", "CompositionError", "compose", "main"]
