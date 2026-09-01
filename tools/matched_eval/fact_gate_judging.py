"""Changed-only Sol judging for the routed fixed-S1 fact-gate arm.

The verified fact-gate answer plane fixes the provider-call subset before any
gold is loaded.  Only predictions whose digest differs from the sealed S0-v2
parent are judged again; every identical prediction inherits its fully
verified parent verdict.  The resulting score ledger still covers all 100
runtime answer rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import exact_match, f1_score
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
)

from . import closure_judging, judging, live
from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .fact_gate_live import (
    ANSWER_PLAN_ID,
    ANSWER_STAGE_ID,
    ARM_LABEL,
    ARM_PLAN_ID,
    FACT_GATE_STAGE_ID,
    PARENT_ARM_LABEL,
    RENDERER_ID,
    VerifiedFactGateAnswerPlane,
    VerifiedFactGateAnswerRow,
)
from .ledger import (
    ScoreLedgerEntry,
    _validated_runtime_ledger,
    build_score_ledger,
)
from .population import EXPECTED_QUESTION_COUNT


FACT_GATE_JUDGE_FORMAT = (
    "memory-condense-matched-fact-gate-changed-only-sol-judge-v1"
)
FACT_GATE_JUDGE_PREFLIGHT_FORMAT = (
    "memory-condense-matched-fact-gate-changed-only-sol-preflight-v1"
)
EMPTY_PROMPT_POPULATION_FORMAT = (
    "memory-condense-matched-fact-gate-changed-only-empty-prompts-v1"
)
JUDGE_PLAN_ID = "matched_s0_plus_routed_em_fact_gate_changed_only_sol_judge_v1"

JUDGE_PREFLIGHT_NAME = judging.JUDGE_PREFLIGHT_NAME
JUDGE_NAME = judging.JUDGE_NAME
JUDGE_REPLAY_NAME = judging.JUDGE_REPLAY_NAME
SCORE_LEDGER_NAME = judging.SCORE_LEDGER_NAME
SCORE_LEDGER_REPLAY_NAME = judging.SCORE_LEDGER_REPLAY_NAME
JUDGE_CHECKPOINT_DIR_NAME = judging.JUDGE_CHECKPOINT_DIR_NAME


def _require(condition: object, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


@dataclass(frozen=True, slots=True)
class _FactGateJudgePlan:
    answer_plane: VerifiedFactGateAnswerPlane
    gold_rows: tuple[judging._GoldRow, ...]
    gold_population_sha256: str
    change_projection: Mapping[str, Any]
    prompt_rows: tuple[judging._JudgePromptRow, ...]
    prompt_population: FastPromptPopulation | None
    parent_judge: closure_judging._VerifiedParentJudge

    @property
    def required_calls(self) -> int:
        return 0 if self.prompt_population is None else (
            self.prompt_population.unique_prompt_count
        )

    @property
    def changed_count(self) -> int:
        return len(self.prompt_rows)


@dataclass(frozen=True, slots=True)
class FactGateJudgeRunResult:
    judge_artifact: SealedArtifact
    score_ledger_artifact: SealedArtifact
    correct: int
    physical_provider_calls: int
    checkpoint_hits: int


def _validate_answer_plane(
    answer_plane: VerifiedFactGateAnswerPlane,
    *,
    expected_question_count: int,
) -> dict[str, Any]:
    """Fix the changed-only population before gold or verdicts are opened."""

    _require(
        type(answer_plane) is VerifiedFactGateAnswerPlane,
        "fact-gate judge requires an exact verified answer plane",
    )
    _require(
        type(expected_question_count) is int and expected_question_count > 0,
        "expected question count must be a positive exact integer",
    )
    _require(
        answer_plane.arm_label == ARM_LABEL
        and answer_plane.parent_arm_label == PARENT_ARM_LABEL
        and answer_plane.renderer_id == RENDERER_ID
        and answer_plane.arm_plan_id == ARM_PLAN_ID
        and answer_plane.answer_plan_id == ANSWER_PLAN_ID,
        "fact-gate answer identity changed",
    )
    for value, label in (
        (answer_plane.run_sha256, "fact-gate answer run SHA-256"),
        (answer_plane.replay_sha256, "fact-gate answer replay SHA-256"),
        (answer_plane.runtime_ledger_sha256, "fact-gate runtime-ledger SHA-256"),
        (answer_plane.matched_population_id, "fact-gate population ID"),
        (answer_plane.population_identity_sha256, "fact-gate population SHA-256"),
        (answer_plane.retrieval_sha256, "fact-gate retrieval SHA-256"),
        (answer_plane.source_em_run_sha256, "fact-gate EM run SHA-256"),
        (
            answer_plane.source_em_compression_sha256,
            "fact-gate EM compression SHA-256",
        ),
        (answer_plane.source_preflight_sha256, "fact-gate preflight SHA-256"),
        (answer_plane.route_policy_sha256, "fact-gate route-policy SHA-256"),
        (answer_plane.snapshot_id, "fact-gate snapshot ID"),
    ):
        require_sha256(value, label)
    _require(
        answer_plane.run_sha256 == answer_plane.replay_sha256,
        "fact-gate answer run and replay differ",
    )

    parent = answer_plane.parent_plane
    _require(
        type(parent) is live.VerifiedS0V2AnswerPlane,
        "fact-gate answer plane lost its verified S0-v2 parent",
    )
    judging._validate_preverified_answer_plane(
        parent,
        profile=judging._V2_JUDGE_PROFILE,
    )
    _require(
        answer_plane.parent_answer_run_sha256 == parent.run_sha256
        and parent.run_sha256 == parent.replay_sha256,
        "fact-gate parent answer binding changed",
    )
    _require(
        answer_plane.matched_population_id == parent.matched_population_id
        and answer_plane.population_identity_sha256
        == parent.population_identity_sha256,
        "fact-gate and parent populations differ",
    )
    _require(
        type(answer_plane.rows) is tuple
        and len(answer_plane.rows) == len(parent.rows) == expected_question_count,
        "fact-gate judge population size changed",
    )

    runtime = answer_plane.runtime_ledger_json()
    ledger_identity, answer_row_ids = _validated_runtime_ledger(runtime)
    require_sha256(ledger_identity, "fact-gate runtime-ledger identity")
    _require(
        runtime.get("snapshot_id") == answer_plane.snapshot_id
        and runtime.get("plan_id") == ANSWER_PLAN_ID,
        "fact-gate runtime envelope changed",
    )
    _require(
        answer_row_ids == tuple(row.runtime_row_id for row in answer_plane.rows),
        "fact-gate runtime answer-row order changed",
    )
    source_map = {
        row["role"]: row["sha256"] for row in runtime["source_artifacts"]
    }
    expected_sources = {
        f"{ARM_LABEL}:sealed_retrieval": answer_plane.retrieval_sha256,
        f"{ARM_LABEL}:em_run": answer_plane.source_em_run_sha256,
        f"{ARM_LABEL}:em_compression": (
            answer_plane.source_em_compression_sha256
        ),
        f"{ARM_LABEL}:route_policy": answer_plane.route_policy_sha256,
        f"{ARM_LABEL}:parent_answer_run": parent.run_sha256,
        f"{ARM_LABEL}:answer_preflight": answer_plane.source_preflight_sha256,
        f"{ARM_LABEL}:answer_run": answer_plane.run_sha256,
    }
    _require(
        source_map == expected_sources,
        "fact-gate runtime source-artifact envelope changed",
    )

    raw_answers = tuple(
        row for row in runtime["rows"] if row["event_type"] == "answer_observation"
    )
    _require(
        len(raw_answers) == len(answer_plane.rows),
        "fact-gate runtime answer population changed",
    )
    projection_rows: list[dict[str, Any]] = []
    ordinals: list[int] = []
    question_ids: list[str] = []
    for child, parent_row, raw in zip(
        answer_plane.rows,
        parent.rows,
        raw_answers,
        strict=True,
    ):
        _require(
            type(child) is VerifiedFactGateAnswerRow,
            "fact-gate answer rows must have the exact verified type",
        )
        _require(
            type(child.ordinal) is int and child.ordinal >= 0,
            "fact-gate answer ordinal is invalid",
        )
        require_text(child.question_id, "fact-gate question ID")
        require_text(child.route_id, "fact-gate route ID")
        require_text(child.gate_disposition, "fact-gate disposition")
        require_text(child.gate_reason, "fact-gate reason")
        for value, label in (
            (child.question_sha256, "fact-gate question SHA-256"),
            (child.dated_question_sha256, "fact-gate dated-question SHA-256"),
            (child.prediction_sha256, "fact-gate prediction SHA-256"),
            (child.parent_prediction_sha256, "fact-gate parent prediction SHA-256"),
            (child.fact_gate_receipt_sha256, "fact-gate receipt SHA-256"),
            (child.final_packet_id, "fact-gate final packet ID"),
            (child.final_prompt_id, "fact-gate final prompt ID"),
            (
                child.final_prompt_messages_sha256,
                "fact-gate final prompt SHA-256",
            ),
            (child.source_row_sha256, "fact-gate source row SHA-256"),
            (child.runtime_row_id, "fact-gate runtime row ID"),
        ):
            require_sha256(value, label)
        _require(
            type(child.prediction) is str
            and bool(child.prediction)
            and quote_sha256(child.prediction) == child.prediction_sha256,
            f"fact-gate prediction changed at ordinal {child.ordinal}",
        )
        _require(
            child.ordinal == parent_row.ordinal
            and child.question_id == parent_row.question_id
            and child.question_sha256 == parent_row.question_sha256
            and child.dated_question_sha256 == parent_row.dated_question_sha256
            and child.parent_prediction_sha256 == parent_row.prediction_sha256,
            f"fact-gate parent row binding changed at ordinal {child.ordinal}",
        )
        changed = child.prediction_sha256 != parent_row.prediction_sha256
        _require(
            type(child.changed_from_parent) is bool
            and child.changed_from_parent == changed,
            f"fact-gate change flag changed at ordinal {child.ordinal}",
        )
        _require(
            raw.get("row_id") == child.runtime_row_id
            and raw.get("ordinal") == child.ordinal
            and raw.get("question_id") == child.question_id
            and raw.get("question_sha256") == child.question_sha256
            and raw.get("arm_label") == ARM_LABEL
            and raw.get("parent_arm_label") == PARENT_ARM_LABEL
            and raw.get("stage_id") == ANSWER_STAGE_ID
            and raw.get("parent_stage_id") == FACT_GATE_STAGE_ID
            and raw.get("renderer_id") == RENDERER_ID
            and raw.get("prompt_messages_sha256")
            == child.final_prompt_messages_sha256
            and raw.get("prediction") == child.prediction
            and raw.get("prediction_sha256") == child.prediction_sha256
            and raw.get("changed_from_parent") == changed
            and raw.get("source_row_sha256") == child.source_row_sha256,
            f"fact-gate answer/runtime binding changed at ordinal {child.ordinal}",
        )
        if child.prediction_source == "sealed_parent_fallback":
            _require(
                child.gate_disposition == "parent_fallback"
                and not changed
                and child.prediction == parent_row.prediction
                and child.call_key_sha256 is None
                and child.request_journal_sha256 is None
                and child.response_journal_sha256 is None
                and raw.get("provider_calls") == 0
                and raw.get("mechanism_id") == "sealed_parent_prediction_reuse",
                f"fact-gate parent fallback changed at ordinal {child.ordinal}",
            )
        elif child.prediction_source == "terra_fact_gate":
            _require(
                child.gate_disposition == "compiled"
                and child.route_id in ("numeric_reduce", "state_chain")
                and raw.get("provider_calls") == 1
                and raw.get("mechanism_id")
                == "routed_em_fact_gate_terra_responder",
                f"fact-gate Terra admission changed at ordinal {child.ordinal}",
            )
            for value, label in (
                (child.call_key_sha256, "fact-gate Terra call key"),
                (child.request_journal_sha256, "fact-gate Terra request journal"),
                (child.response_journal_sha256, "fact-gate Terra response journal"),
            ):
                require_sha256(str(value), label)
        else:
            raise MatchedEvalContractError(
                f"unknown fact-gate prediction source at ordinal {child.ordinal}"
            )
        ordinals.append(child.ordinal)
        question_ids.append(child.question_id)
        projection_rows.append(
            {
                "changed_from_parent": changed,
                "dated_question_sha256": child.dated_question_sha256,
                "fact_gate_receipt_sha256": child.fact_gate_receipt_sha256,
                "gate_disposition": child.gate_disposition,
                "ordinal": child.ordinal,
                "parent_prediction_sha256": parent_row.prediction_sha256,
                "prediction_sha256": child.prediction_sha256,
                "question_id": child.question_id,
                "question_sha256": child.question_sha256,
                "route_id": child.route_id,
                "runtime_row_id": child.runtime_row_id,
            }
        )
    _require(
        tuple(ordinals) == tuple(sorted(set(ordinals))),
        "fact-gate answer row order changed",
    )
    _require(
        len(set(question_ids)) == len(question_ids),
        "fact-gate question IDs must be unique",
    )
    _require(
        sum(bool(row["changed_from_parent"]) for row in projection_rows)
        == len(answer_plane.changed_rows),
        "fact-gate changed-row projection changed",
    )
    body: dict[str, Any] = {
        "answer_run_sha256": answer_plane.run_sha256,
        "arm_label": ARM_LABEL,
        "format": "memory-condense-matched-fact-gate-change-projection-v1",
        "parent_answer_run_sha256": parent.run_sha256,
        "population_identity_sha256": answer_plane.population_identity_sha256,
        "rows": projection_rows,
        "runtime_ledger_identity_sha256": ledger_identity,
    }
    body["change_projection_sha256"] = identity_sha256(body)
    return body


def _load_gold(
    *,
    dataset_path: str | Path,
    split_path: str | Path,
    answer_plane: VerifiedFactGateAnswerPlane,
) -> tuple[tuple[judging._GoldRow, ...], str]:
    return judging._load_gold(
        dataset_path=dataset_path,
        split_path=split_path,
        answer_plane=answer_plane,  # type: ignore[arg-type]
    )


def _build_plan(
    *,
    answer_plane: VerifiedFactGateAnswerPlane,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    expected_question_count: int,
) -> _FactGateJudgePlan:
    # This structural projection fixes the 14-call subset before gold or any
    # parent outcome is loaded.
    change_projection = _validate_answer_plane(
        answer_plane,
        expected_question_count=expected_question_count,
    )
    gold_rows, gold_population_sha256 = _load_gold(
        dataset_path=dataset_path,
        split_path=split_path,
        answer_plane=answer_plane,
    )
    _require(
        len(gold_rows) == len(answer_plane.rows),
        "fact-gate gold population changed",
    )
    prompt_rows, prompt_population = closure_judging._prompt_plan(
        answer_plane,  # type: ignore[arg-type]
        gold_rows,
    )
    parent_judge = closure_judging._load_parent_judge(
        parent=answer_plane.parent_plane,
        gold_rows=gold_rows,
        gold_population_sha256=gold_population_sha256,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=(
            expected_parent_score_ledger_sha256
        ),
    )
    return _FactGateJudgePlan(
        answer_plane=answer_plane,
        gold_rows=gold_rows,
        gold_population_sha256=gold_population_sha256,
        change_projection=change_projection,
        prompt_rows=prompt_rows,
        prompt_population=prompt_population,
        parent_judge=parent_judge,
    )


def _empty_prompt_population() -> dict[str, Any]:
    body: dict[str, Any] = {
        "format": EMPTY_PROMPT_POPULATION_FORMAT,
        "logical_prompt_count": 0,
        "max_prompt_token_proxy": judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS,
        "ordered_rows": [],
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "unique_prompt_count": 0,
    }
    body["prompt_population_sha256"] = identity_sha256(body)
    return body


def _prompt_population_projection(plan: _FactGateJudgePlan) -> dict[str, Any]:
    if plan.prompt_population is None:
        return _empty_prompt_population()
    return plan.prompt_population.model_dump()


def _preflight_artifact(plan: _FactGateJudgePlan) -> dict[str, Any]:
    prompt_population = _prompt_population_projection(plan)
    return {
        "answer_plan_id": ANSWER_PLAN_ID,
        "answer_run_sha256": plan.answer_plane.run_sha256,
        "arm_plan_id": ARM_PLAN_ID,
        "arm_label": ARM_LABEL,
        "change_projection": dict(plan.change_projection),
        "changed_prediction_count": plan.changed_count,
        "format": FACT_GATE_JUDGE_PREFLIGHT_FORMAT,
        "gold_loaded_posthoc": True,
        "gold_population_sha256": plan.gold_population_sha256,
        "inherited_prediction_count": len(plan.answer_plane.rows)
        - plan.changed_count,
        "judge_model": judging.DEFAULT_SOL_CALLER_MODEL,
        "judge_plan_id": JUDGE_PLAN_ID,
        "logical_prompt_count": plan.changed_count,
        "matched_population_id": plan.answer_plane.matched_population_id,
        "maximum_prompt_token_proxy": max(
            (row.prompt_token_proxy for row in plan.prompt_rows),
            default=0,
        ),
        "new_provider_calls": 0,
        "parent_answer_run_sha256": plan.answer_plane.parent_plane.run_sha256,
        "parent_runtime_ledger_sha256": (
            plan.answer_plane.parent_plane.runtime_ledger_sha256
        ),
        "parent_judge_preflight_sha256": plan.parent_judge.preflight_sha256,
        "parent_judge_sha256": plan.parent_judge.judge_sha256,
        "parent_score_ledger_sha256": plan.parent_judge.score_ledger_sha256,
        "population_identity_sha256": (
            plan.answer_plane.population_identity_sha256
        ),
        "prompt_population": prompt_population,
        "renderer_id": RENDERER_ID,
        "required_authorized_provider_calls": plan.required_calls,
        "retrieval_sha256": plan.answer_plane.retrieval_sha256,
        "route_policy_sha256": plan.answer_plane.route_policy_sha256,
        "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
        "snapshot_id": plan.answer_plane.snapshot_id,
        "source_em_compression_sha256": (
            plan.answer_plane.source_em_compression_sha256
        ),
        "source_em_run_sha256": plan.answer_plane.source_em_run_sha256,
        "source_preflight_sha256": plan.answer_plane.source_preflight_sha256,
        "unique_prompt_count": plan.required_calls,
    }


def _runtime(
    plan: _FactGateJudgePlan,
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    max_concurrency: int,
    preflight_artifact_sha256: str,
) -> FastCompletionRuntime:
    require_sha256(preflight_artifact_sha256, "fact-gate judge preflight")
    _require(bool(plan.prompt_rows), "empty fact-gate judge plan has no runtime")
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[row.messages for row in plan.prompt_rows],
        model=judging.DEFAULT_SOL_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_run_sha256": plan.answer_plane.run_sha256,
            "arm_label": ARM_LABEL,
            "authorized_unique_calls": plan.required_calls,
            "change_projection_sha256": plan.change_projection[
                "change_projection_sha256"
            ],
            "gateway_url": live.DEFAULT_GATEWAY_URL,
            "gold_population_sha256": plan.gold_population_sha256,
            "judge_plan_id": JUDGE_PLAN_ID,
            "parent_answer_run_sha256": plan.answer_plane.parent_plane.run_sha256,
            "parent_judge_sha256": plan.parent_judge.judge_sha256,
            "parent_score_ledger_sha256": plan.parent_judge.score_ledger_sha256,
            "preflight_artifact_sha256": preflight_artifact_sha256,
            "route_policy_sha256": plan.answer_plane.route_policy_sha256,
            "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
            "source_em_compression_sha256": (
                plan.answer_plane.source_em_compression_sha256
            ),
            "source_em_run_sha256": plan.answer_plane.source_em_run_sha256,
        },
    )


def _judge_artifact(
    plan: _FactGateJudgePlan,
    batch: FastCompletionBatch | None,
    *,
    preflight_artifact_sha256: str,
) -> dict[str, Any]:
    if plan.required_calls:
        _require(batch is not None, "changed fact-gate predictions require verdicts")
        assert batch is not None
        _require(
            len(batch.logical_completions) == plan.changed_count,
            "fact-gate Sol verdict population changed",
        )
        records = {row.messages_sha256: row for row in batch.unique_records}
        changed_outputs = {
            prompt.answer_ordinal: (prompt, verdict, records[prompt.messages_sha256])
            for prompt, verdict in zip(
                plan.prompt_rows,
                batch.logical_completions,
                strict=True,
            )
        }
    else:
        _require(batch is None, "unchanged fact-gate plan cannot carry verdicts")
        changed_outputs = {}
    parent_outcomes = {row.ordinal: row for row in plan.parent_judge.outcomes}
    rows: list[dict[str, Any]] = []
    for answer, gold in zip(plan.answer_plane.rows, plan.gold_rows, strict=True):
        parent = parent_outcomes[answer.ordinal]
        changed = answer.changed_from_parent
        if changed:
            prompt, output, record = changed_outputs[answer.ordinal]
            correct = parse_binary_judge_verdict(output)
            verdict_sha = quote_sha256(output)
            verdict_source = "new_sol_judge"
            call_key = record.call_key_sha256
            request_journal = record.request_journal_sha256
            response_journal = record.response_journal_sha256
            judge_messages = prompt.messages_sha256
            prompt_tokens: int | None = prompt.prompt_token_proxy
            persisted_output: str | None = output
            output_sha: str | None = verdict_sha
            demand_class = prompt.demand_class
        else:
            correct = parent.correct
            verdict_sha = parent.judge_verdict_sha256
            verdict_source = "sealed_parent_s0_v2_judge"
            call_key = request_journal = response_journal = None
            judge_messages = None
            prompt_tokens = None
            persisted_output = None
            output_sha = None
            demand_class = parent.demand_class
        rescued = not parent.correct and correct
        regressed = parent.correct and not correct
        body: dict[str, Any] = {
            "baseline_correct": parent.correct,
            "call_key_sha256": call_key,
            "category": gold.category,
            "changed_from_parent": changed,
            "correct": correct,
            "dated_question_sha256": gold.dated_question_sha256,
            "demand_class": demand_class,
            "judge_messages_sha256": judge_messages,
            "judge_output": persisted_output,
            "judge_output_sha256": output_sha,
            "judge_prompt_token_proxy": prompt_tokens,
            "normalized_exact_match": exact_match(
                answer.prediction,
                gold.reference,
            ),
            "normalized_f1": f1_score(answer.prediction, gold.reference),
            "ordinal": answer.ordinal,
            "parent_judge_row_sha256": parent.judge_row_sha256,
            "parent_judge_verdict_sha256": parent.judge_verdict_sha256,
            "parent_prediction_sha256": answer.parent_prediction_sha256,
            "prediction_sha256": answer.prediction_sha256,
            "question_id": answer.question_id,
            "question_sha256": answer.question_sha256,
            "reference_sha256": gold.reference_sha256,
            "regressed": regressed,
            "request_journal_sha256": request_journal,
            "rescued": rescued,
            "response_journal_sha256": response_journal,
            "runtime_row_id": answer.runtime_row_id,
            "verdict_sha256": verdict_sha,
            "verdict_source": verdict_source,
        }
        body["judge_row_sha256"] = identity_sha256(body)
        rows.append(body)
    correct_count = sum(bool(row["correct"]) for row in rows)
    baseline_correct = sum(bool(row["baseline_correct"]) for row in rows)
    rescued = sum(bool(row["rescued"]) for row in rows)
    regressed = sum(bool(row["regressed"]) for row in rows)
    prompt_population = _prompt_population_projection(plan)
    return {
        "aggregate": {
            "accuracy": correct_count / len(rows),
            "baseline_correct": baseline_correct,
            "changed_predictions": plan.changed_count,
            "correct": correct_count,
            "fresh_judgments": plan.changed_count,
            "fresh_unique_provider_prompts": plan.required_calls,
            "gate_passed": correct_count / len(rows) >= judging.TARGET_ACCURACY,
            "incorrect": len(rows) - correct_count,
            "inherited_judgments": len(rows) - plan.changed_count,
            "mean_f1": sum(float(row["normalized_f1"]) for row in rows)
            / len(rows),
            "net_marginal": rescued - regressed,
            "normalized_exact_match": sum(
                bool(row["normalized_exact_match"]) for row in rows
            ),
            "questions": len(rows),
            "regressed": regressed,
            "rescued": rescued,
            "target_accuracy": judging.TARGET_ACCURACY,
        },
        "answer_plan_id": ANSWER_PLAN_ID,
        "answer_run_sha256": plan.answer_plane.run_sha256,
        "arm_plan_id": ARM_PLAN_ID,
        "arm_label": ARM_LABEL,
        "category_aggregates": judging._category_aggregates(rows),
        "change_projection_sha256": plan.change_projection[
            "change_projection_sha256"
        ],
        "completion_batch": None if batch is None else judging._stable_batch(batch),
        "format": FACT_GATE_JUDGE_FORMAT,
        "gold_loaded_posthoc": True,
        "gold_population_sha256": plan.gold_population_sha256,
        "judge_completions_may_echo_gold": True,
        "judge_model": judging.DEFAULT_SOL_CALLER_MODEL,
        "judge_plan_id": JUDGE_PLAN_ID,
        "matched_population_id": plan.answer_plane.matched_population_id,
        "parent_answer_run_sha256": plan.answer_plane.parent_plane.run_sha256,
        "parent_runtime_ledger_sha256": (
            plan.answer_plane.parent_plane.runtime_ledger_sha256
        ),
        "parent_judge_preflight_sha256": plan.parent_judge.preflight_sha256,
        "parent_judge_sha256": plan.parent_judge.judge_sha256,
        "parent_score_ledger_sha256": plan.parent_judge.score_ledger_sha256,
        "population_identity_sha256": (
            plan.answer_plane.population_identity_sha256
        ),
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "prompt_population_sha256": prompt_population[
            "prompt_population_sha256"
        ],
        "question_count": len(rows),
        "questions": rows,
        "renderer_id": RENDERER_ID,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": plan.answer_plane.retrieval_sha256,
        "route_policy_sha256": plan.answer_plane.route_policy_sha256,
        "runtime_ledger_identity_sha256": plan.answer_plane.runtime_ledger.get(
            "ledger_identity_sha256"
        ),
        "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
        "snapshot_id": plan.answer_plane.snapshot_id,
        "source_em_compression_sha256": (
            plan.answer_plane.source_em_compression_sha256
        ),
        "source_em_run_sha256": plan.answer_plane.source_em_run_sha256,
        "source_preflight_sha256": plan.answer_plane.source_preflight_sha256,
        "unique_provider_prompt_count": plan.required_calls,
    }


def _score_ledger(
    plan: _FactGateJudgePlan,
    judge_payload: Mapping[str, Any],
    *,
    judge_artifact_sha256: str,
) -> dict[str, Any]:
    raw_rows = judge_payload.get("questions")
    _require(type(raw_rows) is list, "fact-gate judge questions must be an array")
    entries: list[ScoreLedgerEntry] = []
    for answer, raw in zip(plan.answer_plane.rows, raw_rows, strict=True):
        _require(
            type(raw) is dict
            and raw.get("runtime_row_id") == answer.runtime_row_id
            and type(raw.get("correct")) is bool
            and type(raw.get("baseline_correct")) is bool,
            f"fact-gate judge/runtime binding changed at ordinal {answer.ordinal}",
        )
        correct = bool(raw["correct"])
        baseline = bool(raw["baseline_correct"])
        entries.append(
            ScoreLedgerEntry(
                runtime_row_id=answer.runtime_row_id,
                correct=correct,
                baseline_correct=baseline,
                changed_from_baseline=answer.changed_from_parent,
                rescued=not baseline and correct,
                regressed=baseline and not correct,
                question_only_demand_class=str(raw["demand_class"]),
                judge_row_sha256=str(raw["judge_row_sha256"]),
                judge_verdict_sha256=str(raw["verdict_sha256"]),
                baseline_judge_row_sha256=str(raw["parent_judge_row_sha256"]),
                historical_provider_calls=int(not answer.changed_from_parent),
            )
        )
    return build_score_ledger(
        runtime_ledger=plan.answer_plane.runtime_ledger_json(),
        entries=entries,
        source_artifacts=(
            {"role": f"{ARM_LABEL}:judge", "sha256": judge_artifact_sha256},
            {
                "role": f"{PARENT_ARM_LABEL}:parent_judge",
                "sha256": plan.parent_judge.judge_sha256,
            },
            {
                "role": f"{PARENT_ARM_LABEL}:parent_score_ledger",
                "sha256": plan.parent_judge.score_ledger_sha256,
            },
        ),
    )


def _authorize(
    plan: _FactGateJudgePlan,
    *,
    enable_provider: bool,
    authorized_provider_calls: int,
) -> None:
    _require(type(enable_provider) is bool, "provider enablement must be an exact bool")
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == plan.required_calls,
        f"authorized provider calls must exactly equal {plan.required_calls}",
    )
    if plan.required_calls:
        _require(enable_provider, "changed fact-gate judge requires provider enablement")
    else:
        _require(
            not enable_provider,
            "unchanged fact-gate judge forbids provider enablement",
        )


def preflight_fact_gate_changed_only_judge(
    *,
    answer_plane: VerifiedFactGateAnswerPlane,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    output_root: str | Path,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> SealedArtifact:
    plan = _build_plan(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=expected_parent_score_ledger_sha256,
        expected_question_count=expected_question_count,
    )
    artifact, _created = publish_sealed_json(
        Path(output_root) / JUDGE_PREFLIGHT_NAME,
        _preflight_artifact(plan),
    )
    return artifact


def run_fact_gate_changed_only_judge(
    *,
    answer_plane: VerifiedFactGateAnswerPlane,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = live.DEFAULT_API_KEY_ENV,
    max_concurrency: int = 4,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> FactGateJudgeRunResult:
    plan = _build_plan(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=expected_parent_score_ledger_sha256,
        expected_question_count=expected_question_count,
    )
    _authorize(
        plan,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
    )
    output = Path(output_root)
    preflight, _created = publish_sealed_json(
        output / JUDGE_PREFLIGHT_NAME,
        _preflight_artifact(plan),
    )
    existing = output / JUDGE_NAME
    if existing.exists():
        source = read_sealed_json(existing)
        return _replay_plan(
            plan,
            expected_judge_sha256=source.sha256,
            output_root=output,
            max_concurrency=max_concurrency,
        )
    batch: FastCompletionBatch | None = None
    if plan.required_calls:
        load_dotenv()
        api_key = os.environ.get(api_key_env, "").strip()
        _require(bool(api_key), f"provider API key is empty: {api_key_env}")
        client = judging._make_provider_client(api_key, live.DEFAULT_GATEWAY_URL)
        try:
            batch = _runtime(
                plan,
                checkpoint_dir=output / JUDGE_CHECKPOINT_DIR_NAME,
                client=client,
                max_concurrency=max_concurrency,
                preflight_artifact_sha256=preflight.sha256,
            ).run()
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        _require(
            batch.usage.physical_calls + batch.usage.checkpoint_hits
            == plan.required_calls,
            "fact-gate Sol journal population changed",
        )
    payload = _judge_artifact(
        plan,
        batch,
        preflight_artifact_sha256=preflight.sha256,
    )
    judge, _created = publish_sealed_json(output / JUDGE_NAME, payload)
    score_payload = _score_ledger(
        plan,
        payload,
        judge_artifact_sha256=judge.sha256,
    )
    score, _created = publish_sealed_json(
        output / SCORE_LEDGER_NAME,
        score_payload,
    )
    return FactGateJudgeRunResult(
        judge_artifact=judge,
        score_ledger_artifact=score,
        correct=int(payload["aggregate"]["correct"]),
        physical_provider_calls=0 if batch is None else batch.usage.physical_calls,
        checkpoint_hits=0 if batch is None else batch.usage.checkpoint_hits,
    )


def _replay_plan(
    plan: _FactGateJudgePlan,
    *,
    expected_judge_sha256: str,
    output_root: str | Path,
    max_concurrency: int,
) -> FactGateJudgeRunResult:
    require_sha256(expected_judge_sha256, "expected fact-gate judge SHA-256")
    output = Path(output_root)
    preflight = read_sealed_json(output / JUDGE_PREFLIGHT_NAME)
    _require(
        preflight.payload == _preflight_artifact(plan),
        "fact-gate judge preflight changed during replay",
    )
    source = read_sealed_json(output / JUDGE_NAME)
    _require(
        source.sha256 == expected_judge_sha256,
        "fact-gate judge SHA-256 changed",
    )
    batch = (
        _runtime(
            plan,
            checkpoint_dir=output / JUDGE_CHECKPOINT_DIR_NAME,
            client=None,
            max_concurrency=max_concurrency,
            preflight_artifact_sha256=preflight.sha256,
        ).run()
        if plan.required_calls
        else None
    )
    if batch is not None:
        _require(batch.usage.physical_calls == 0, "fact-gate judge replay made calls")
        _require(
            batch.usage.checkpoint_hits == plan.required_calls,
            "fact-gate judge replay checkpoint population changed",
        )
    expected = _judge_artifact(
        plan,
        batch,
        preflight_artifact_sha256=preflight.sha256,
    )
    _require(
        canonical_json_bytes(expected) == canonical_json_bytes(source.payload),
        "fact-gate judge differs from sealed changed-only journals",
    )
    replay, _created = publish_sealed_json(output / JUDGE_REPLAY_NAME, expected)
    score_expected = _score_ledger(
        plan,
        expected,
        judge_artifact_sha256=source.sha256,
    )
    score = read_sealed_json(output / SCORE_LEDGER_NAME)
    _require(
        score.payload == score_expected,
        "fact-gate score ledger differs from replayed verdicts",
    )
    score_replay, _created = publish_sealed_json(
        output / SCORE_LEDGER_REPLAY_NAME,
        score_expected,
    )
    return FactGateJudgeRunResult(
        judge_artifact=replay,
        score_ledger_artifact=score_replay,
        correct=int(expected["aggregate"]["correct"]),
        physical_provider_calls=0,
        checkpoint_hits=0 if batch is None else batch.usage.checkpoint_hits,
    )


def replay_fact_gate_changed_only_judge(
    *,
    answer_plane: VerifiedFactGateAnswerPlane,
    expected_judge_sha256: str,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    output_root: str | Path,
    max_concurrency: int = 4,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> FactGateJudgeRunResult:
    plan = _build_plan(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=expected_parent_score_ledger_sha256,
        expected_question_count=expected_question_count,
    )
    return _replay_plan(
        plan,
        expected_judge_sha256=expected_judge_sha256,
        output_root=output_root,
        max_concurrency=max_concurrency,
    )


__all__ = [
    "EMPTY_PROMPT_POPULATION_FORMAT",
    "FACT_GATE_JUDGE_FORMAT",
    "FACT_GATE_JUDGE_PREFLIGHT_FORMAT",
    "FactGateJudgeRunResult",
    "JUDGE_CHECKPOINT_DIR_NAME",
    "JUDGE_NAME",
    "JUDGE_PLAN_ID",
    "JUDGE_PREFLIGHT_NAME",
    "JUDGE_REPLAY_NAME",
    "SCORE_LEDGER_NAME",
    "SCORE_LEDGER_REPLAY_NAME",
    "preflight_fact_gate_changed_only_judge",
    "replay_fact_gate_changed_only_judge",
    "run_fact_gate_changed_only_judge",
]
