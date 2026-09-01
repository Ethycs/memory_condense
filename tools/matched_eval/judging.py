"""Post-hoc Sol judging for sealed matched S0-v2 predictions.

The answer plane is fully replayed and verified before this module reads the
locked LongMemEval dataset or split.  Gold is used only to build judge prompts
and score artifacts; it never enters the responder journals or runtime ledger.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import (
    build_judge_prompt,
    exact_match,
    f1_score,
)
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.locked_split import (
    load_split_manifest,
    select_locked_split,
)
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_LONGMEMEVAL_DATASET_SHA256,
    LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
)
from memory_condense.ingest.loader import load_benchmark
from tools._routed_repair_routing import route_question

from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .ledger import (
    ScoreLedgerEntry,
    _validated_runtime_ledger,
    build_score_ledger,
)
from .live import (
    ARM_LABEL,
    DEFAULT_API_KEY_ENV,
    DEFAULT_GATEWAY_URL,
    VerifiedS0V2AnswerPlane,
    VerifiedS0V2AnswerRow,
    V3_ARM_LABEL,
    V4_ARM_LABEL,
    load_verified_s0_v2_answer_plane,
)
from .population import (
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
)
from .renderer import RENDERER_ID, V3_RENDERER_ID, V4_RENDERER_ID


JUDGE_FORMAT = "memory-condense-matched-s0-v2-sol-judge-v1"
JUDGE_PREFLIGHT_FORMAT = "memory-condense-matched-s0-v2-sol-preflight-v1"
JUDGE_PLAN_ID = "matched_s0_control_v2_sol_judge_v1"
V3_JUDGE_FORMAT = "memory-condense-matched-s0-v3-sol-judge-v1"
V3_JUDGE_PREFLIGHT_FORMAT = "memory-condense-matched-s0-v3-sol-preflight-v1"
V3_JUDGE_PLAN_ID = "matched_s0_control_v3_sol_judge_v1"
V4_JUDGE_FORMAT = "memory-condense-matched-s0-v4-sol-judge-v1"
V4_JUDGE_PREFLIGHT_FORMAT = "memory-condense-matched-s0-v4-sol-preflight-v1"
V4_JUDGE_PLAN_ID = "matched_s0_control_v4_sol_judge_v1"
DUAL_ANSWER_SYNTHESIS_RENDERER_ID = "matched_s0_dual_answer_synthesis_v1"
DUAL_ANSWER_SYNTHESIS_ARM_LABEL = "S0_DUAL_ANSWER_SYNTHESIS_V1"
DUAL_ANSWER_SYNTHESIS_JUDGE_FORMAT = (
    "memory-condense-matched-s0-dual-answer-synthesis-sol-judge-v1"
)
DUAL_ANSWER_SYNTHESIS_JUDGE_PREFLIGHT_FORMAT = (
    "memory-condense-matched-s0-dual-answer-synthesis-sol-preflight-v1"
)
DUAL_ANSWER_SYNTHESIS_JUDGE_PLAN_ID = (
    "matched_s0_dual_answer_synthesis_sol_judge_v1"
)
DEFAULT_SOL_GATEWAY_MODEL = "codex_sdk/gpt-5.6-sol"
DEFAULT_SOL_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-sol"
DEFAULT_MAX_JUDGE_PROMPT_TOKENS = 8_000
TARGET_ACCURACY = 0.95

JUDGE_PREFLIGHT_NAME = "judge-preflight.json"
JUDGE_NAME = "semantic-judge-sol.json"
JUDGE_REPLAY_NAME = "semantic-judge-sol-replay.json"
SCORE_LEDGER_NAME = "score-ledger.json"
SCORE_LEDGER_REPLAY_NAME = "score-ledger-replay.json"
JUDGE_CHECKPOINT_DIR_NAME = "sol-judge-calls"


@dataclass(frozen=True, slots=True)
class _JudgeExecutionProfile:
    renderer_id: str
    arm_label: str
    judge_format: str
    preflight_format: str
    plan_id: str


_V2_JUDGE_PROFILE = _JudgeExecutionProfile(
    renderer_id=RENDERER_ID,
    arm_label=ARM_LABEL,
    judge_format=JUDGE_FORMAT,
    preflight_format=JUDGE_PREFLIGHT_FORMAT,
    plan_id=JUDGE_PLAN_ID,
)
_V3_JUDGE_PROFILE = _JudgeExecutionProfile(
    renderer_id=V3_RENDERER_ID,
    arm_label=V3_ARM_LABEL,
    judge_format=V3_JUDGE_FORMAT,
    preflight_format=V3_JUDGE_PREFLIGHT_FORMAT,
    plan_id=V3_JUDGE_PLAN_ID,
)
_V4_JUDGE_PROFILE = _JudgeExecutionProfile(
    renderer_id=V4_RENDERER_ID,
    arm_label=V4_ARM_LABEL,
    judge_format=V4_JUDGE_FORMAT,
    preflight_format=V4_JUDGE_PREFLIGHT_FORMAT,
    plan_id=V4_JUDGE_PLAN_ID,
)
_DUAL_ANSWER_SYNTHESIS_JUDGE_PROFILE = _JudgeExecutionProfile(
    renderer_id=DUAL_ANSWER_SYNTHESIS_RENDERER_ID,
    arm_label=DUAL_ANSWER_SYNTHESIS_ARM_LABEL,
    judge_format=DUAL_ANSWER_SYNTHESIS_JUDGE_FORMAT,
    preflight_format=DUAL_ANSWER_SYNTHESIS_JUDGE_PREFLIGHT_FORMAT,
    plan_id=DUAL_ANSWER_SYNTHESIS_JUDGE_PLAN_ID,
)


def _judge_profile(renderer_id: str) -> _JudgeExecutionProfile:
    if renderer_id == RENDERER_ID:
        return _V2_JUDGE_PROFILE
    if renderer_id == V3_RENDERER_ID:
        return _V3_JUDGE_PROFILE
    if renderer_id == V4_RENDERER_ID:
        return _V4_JUDGE_PROFILE
    raise MatchedEvalContractError(
        f"unsupported S0 judge renderer: {renderer_id!r}"
    )

_RECORD_DISPOSITION_FIELDS = frozenset({"checkpoint_hit", "physical_call"})
_USAGE_DISPOSITION_FIELDS = frozenset({"checkpoint_hits", "physical_calls"})


def _require(condition: object, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    from memory_condense.eval.run_fast_1m_em_facts import _make_provider_client

    return _make_provider_client(api_key, gateway_url)


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    value = batch.model_dump()
    return {
        "logical_completions": value["logical_completions"],
        "unique_records": [
            {
                key: child
                for key, child in row.items()
                if key not in _RECORD_DISPOSITION_FIELDS
            }
            for row in value["unique_records"]
        ],
        "usage": {
            key: child
            for key, child in value["usage"].items()
            if key not in _USAGE_DISPOSITION_FIELDS
        },
        "provenance": value["provenance"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "prompt_population": value["prompt_population"],
    }


@dataclass(frozen=True, slots=True)
class _GoldRow:
    ordinal: int
    question_id: str
    question: str
    question_sha256: str
    dated_question: str
    dated_question_sha256: str
    reference: str
    reference_sha256: str
    category: str


@dataclass(frozen=True, slots=True)
class _JudgePromptRow:
    answer_ordinal: int
    messages: tuple[dict[str, str], ...]
    messages_sha256: str
    prompt_token_proxy: int
    demand_class: str


@dataclass(frozen=True, slots=True)
class _JudgePlan:
    answer_plane: VerifiedS0V2AnswerPlane
    gold_rows: tuple[_GoldRow, ...]
    prompt_rows: tuple[_JudgePromptRow, ...]
    prompt_population: FastPromptPopulation
    gold_population_sha256: str
    profile: _JudgeExecutionProfile

    @property
    def required_calls(self) -> int:
        return self.prompt_population.unique_prompt_count


@dataclass(frozen=True, slots=True)
class S0V2JudgeRunResult:
    judge_artifact: SealedArtifact
    score_ledger_artifact: SealedArtifact
    correct: int
    physical_provider_calls: int
    checkpoint_hits: int


def _load_gold(
    *,
    dataset_path: str | Path,
    split_path: str | Path,
    answer_plane: VerifiedS0V2AnswerPlane,
) -> tuple[tuple[_GoldRow, ...], str]:
    dataset = Path(dataset_path)
    split = Path(split_path)
    _require(
        file_sha256(dataset) == LOCKED_LONGMEMEVAL_DATASET_SHA256,
        "locked LongMemEval dataset SHA-256 changed",
    )
    _require(
        file_sha256(split) == LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
        "locked split-manifest SHA-256 changed",
    )
    samples = load_benchmark(dataset, "longmemeval")
    selected = select_locked_split(
        samples,
        dataset_path=dataset,
        manifest=load_split_manifest(split),
        split="validation",
    )
    questions = tuple(
        question for sample in selected for question in sample.questions
    )
    _require(
        len(selected) == len(questions),
        "locked validation question population changed",
    )
    result: list[_GoldRow] = []
    projection: list[dict[str, Any]] = []
    for source in answer_plane.rows:
        _require(
            source.ordinal < len(questions),
            f"gold ordinal is outside the locked population: {source.ordinal}",
        )
        question = questions[source.ordinal]
        question_sha256 = quote_sha256(question.question)
        dated_sha256 = quote_sha256(question.dated_question)
        _require(
            question.question_id == source.question_id
            and question_sha256 == source.question_sha256
            and dated_sha256 == source.dated_question_sha256,
            f"gold question binding changed at ordinal {source.ordinal}",
        )
        reference_sha256 = quote_sha256(question.answer)
        category = str(question.category or "uncategorized")
        result.append(
            _GoldRow(
                ordinal=source.ordinal,
                question_id=source.question_id,
                question=question.question,
                question_sha256=question_sha256,
                dated_question=question.dated_question,
                dated_question_sha256=dated_sha256,
                reference=question.answer,
                reference_sha256=reference_sha256,
                category=category,
            )
        )
        projection.append(
            {
                "category": category,
                "dated_question_sha256": dated_sha256,
                "ordinal": source.ordinal,
                "question_id": source.question_id,
                "question_sha256": question_sha256,
                "reference_sha256": reference_sha256,
            }
        )
    return tuple(result), identity_sha256(projection)


def _validate_preverified_answer_plane(
    answer_plane: VerifiedS0V2AnswerPlane,
    *,
    profile: _JudgeExecutionProfile,
) -> None:
    """Validate the immutable score boundary before any gold is loaded."""

    _require(
        type(answer_plane) is VerifiedS0V2AnswerPlane,
        "judge requires an exact pre-verified answer plane",
    )
    _require(
        answer_plane.renderer_id == profile.renderer_id,
        "pre-verified answer-plane renderer changed",
    )
    require_text(answer_plane.renderer_id, "answer-plane renderer ID")
    for value, label in (
        (answer_plane.run_sha256, "answer-plane run SHA-256"),
        (answer_plane.replay_sha256, "answer-plane replay SHA-256"),
        (answer_plane.matched_population_id, "answer-plane population ID"),
        (
            answer_plane.population_identity_sha256,
            "answer-plane source-population SHA-256",
        ),
        (answer_plane.snapshot_id, "answer-plane snapshot ID"),
        (answer_plane.runtime_ledger_sha256, "answer-plane runtime-ledger SHA-256"),
    ):
        require_sha256(value, label)
    _require(
        answer_plane.run_sha256 == answer_plane.replay_sha256,
        "pre-verified answer run and replay differ",
    )
    _require(
        type(answer_plane.rows) is tuple and bool(answer_plane.rows),
        "pre-verified answer plane requires immutable rows",
    )

    ordinals: list[int] = []
    question_ids: list[str] = []
    for row in answer_plane.rows:
        _require(
            type(row) is VerifiedS0V2AnswerRow,
            "pre-verified answer rows must have the exact verified type",
        )
        _require(
            type(row.ordinal) is int and row.ordinal >= 0,
            "pre-verified answer ordinal is invalid",
        )
        require_text(row.question_id, "pre-verified answer question ID")
        for value, label in (
            (row.question_sha256, "pre-verified question SHA-256"),
            (row.dated_question_sha256, "pre-verified dated-question SHA-256"),
            (row.messages_sha256, "pre-verified messages SHA-256"),
            (row.prediction_sha256, "pre-verified prediction SHA-256"),
            (row.call_key_sha256, "pre-verified call-key SHA-256"),
            (row.request_journal_sha256, "pre-verified request-journal SHA-256"),
            (row.response_journal_sha256, "pre-verified response-journal SHA-256"),
            (row.source_row_sha256, "pre-verified source-row SHA-256"),
            (row.runtime_row_id, "pre-verified runtime-row ID"),
        ):
            require_sha256(value, label)
        if row.alias_receipt_sha256 is not None:
            require_sha256(
                row.alias_receipt_sha256,
                "pre-verified alias-receipt SHA-256",
            )
        _require(
            type(row.prediction) is str
            and bool(row.prediction)
            and quote_sha256(row.prediction) == row.prediction_sha256,
            f"pre-verified prediction changed at ordinal {row.ordinal}",
        )
        ordinals.append(row.ordinal)
        question_ids.append(row.question_id)
    _require(
        tuple(ordinals) == tuple(sorted(set(ordinals))),
        "pre-verified answer row order changed",
    )
    _require(
        len(set(question_ids)) == len(question_ids),
        "pre-verified answer question IDs must be unique",
    )

    runtime_ledger = answer_plane.runtime_ledger_json()
    _ledger_identity, answer_row_ids = _validated_runtime_ledger(runtime_ledger)
    _require(
        runtime_ledger.get("snapshot_id") == answer_plane.snapshot_id,
        "pre-verified answer snapshot/runtime binding changed",
    )
    _require(
        answer_row_ids == tuple(row.runtime_row_id for row in answer_plane.rows),
        "pre-verified answer/runtime row order changed",
    )
    raw_answer_rows = tuple(
        row
        for row in runtime_ledger["rows"]
        if row.get("event_type") == "answer_observation"
    )
    for answer, raw in zip(answer_plane.rows, raw_answer_rows, strict=True):
        _require(
            raw.get("row_id") == answer.runtime_row_id
            and raw.get("ordinal") == answer.ordinal
            and raw.get("question_id") == answer.question_id
            and raw.get("question_sha256") == answer.question_sha256
            and raw.get("arm_label") == profile.arm_label
            and raw.get("renderer_id") == profile.renderer_id
            and raw.get("prompt_messages_sha256") == answer.messages_sha256
            and raw.get("prediction") == answer.prediction
            and raw.get("prediction_sha256") == answer.prediction_sha256
            and raw.get("source_row_sha256") == answer.source_row_sha256,
            f"pre-verified answer/runtime binding changed at ordinal {answer.ordinal}",
        )


def _build_plan_from_answer_plane(
    *,
    answer_plane: VerifiedS0V2AnswerPlane,
    dataset_path: str | Path,
    split_path: str | Path,
    profile: _JudgeExecutionProfile,
) -> _JudgePlan:
    # The caller must finish and verify the responder plane before this point.
    # Keep both structural validation and every future answer-plane check above
    # the first dataset/split access below.
    _validate_preverified_answer_plane(answer_plane, profile=profile)
    gold_rows, gold_population_sha256 = _load_gold(
        dataset_path=dataset_path,
        split_path=split_path,
        answer_plane=answer_plane,
    )
    prompt_rows: list[_JudgePromptRow] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for source, gold in zip(answer_plane.rows, gold_rows, strict=True):
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                gold.question, gold.reference, source.prediction
            )
        )
        route = route_question(gold.dated_question)
        messages_sha256 = identity_sha256(list(messages))
        prompt_rows.append(
            _JudgePromptRow(
                answer_ordinal=source.ordinal,
                messages=messages,
                messages_sha256=messages_sha256,
                prompt_token_proxy=0,
                demand_class=route.style.value,
            )
        )
        prompts.append(messages)
    preflight = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=DEFAULT_MAX_JUDGE_PROMPT_TOKENS
    )
    _require(
        preflight.logical_prompt_count == preflight.unique_prompt_count
        == len(answer_plane.rows),
        "judge prompt population must be one unique prompt per prediction",
    )
    normalized_rows = tuple(
        _JudgePromptRow(
            answer_ordinal=row.answer_ordinal,
            messages=row.messages,
            messages_sha256=row.messages_sha256,
            prompt_token_proxy=receipt.prompt_token_proxy,
            demand_class=row.demand_class,
        )
        for row, receipt in zip(prompt_rows, preflight.ordered_rows, strict=True)
    )
    _require(
        tuple(row.messages_sha256 for row in normalized_rows)
        == tuple(row.messages_sha256 for row in preflight.ordered_rows),
        "judge prompt order changed after preflight",
    )
    return _JudgePlan(
        answer_plane=answer_plane,
        gold_rows=gold_rows,
        prompt_rows=normalized_rows,
        prompt_population=preflight,
        gold_population_sha256=gold_population_sha256,
        profile=profile,
    )


def _build_plan(
    *,
    answer_run_path: str | Path,
    answer_replay_path: str | Path,
    expected_answer_run_sha256: str,
    retrieval_path: str | Path,
    dataset_path: str | Path,
    split_path: str | Path,
    answer_checkpoint_dir: str | Path | None,
    max_concurrency: int,
    expected_retrieval_sha256: str | None,
    expected_question_count: int,
    renderer_id: str = RENDERER_ID,
    selected_ordinals: Sequence[int] | None = None,
) -> _JudgePlan:
    # This complete replay is intentionally first.  Do not move dataset or
    # split access above it.
    answer_plane = load_verified_s0_v2_answer_plane(
        answer_run_path,
        answer_replay_path,
        expected_run_sha256=expected_answer_run_sha256,
        retrieval_path=retrieval_path,
        checkpoint_dir=answer_checkpoint_dir,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=renderer_id,
        selected_ordinals=selected_ordinals,
    )
    return _build_plan_from_answer_plane(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        profile=_judge_profile(renderer_id),
    )


def _preflight_artifact(plan: _JudgePlan) -> dict[str, Any]:
    return {
        "answer_run_sha256": plan.answer_plane.run_sha256,
        "format": plan.profile.preflight_format,
        "gold_loaded_posthoc": True,
        "gold_population_sha256": plan.gold_population_sha256,
        "judge_model": DEFAULT_SOL_CALLER_MODEL,
        "logical_prompt_count": plan.prompt_population.logical_prompt_count,
        "maximum_prompt_token_proxy": max(
            row.prompt_token_proxy for row in plan.prompt_population.ordered_rows
        ),
        "new_provider_calls": 0,
        "prompt_population": plan.prompt_population.model_dump(),
        "required_authorized_provider_calls": plan.required_calls,
        "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
        "unique_prompt_count": plan.prompt_population.unique_prompt_count,
    }


def _runtime(
    plan: _JudgePlan,
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    max_concurrency: int,
    preflight_artifact_sha256: str,
) -> FastCompletionRuntime:
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[row.messages for row in plan.prompt_rows],
        model=DEFAULT_SOL_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=DEFAULT_MAX_JUDGE_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_run_sha256": plan.answer_plane.run_sha256,
            "arm_label": plan.profile.arm_label,
            "authorized_unique_calls": plan.required_calls,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_population_sha256": plan.gold_population_sha256,
            "judge_plan_id": plan.profile.plan_id,
            "preflight_artifact_sha256": preflight_artifact_sha256,
            "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
        },
    )


def _category_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
    result: list[dict[str, Any]] = []
    for category, selected in sorted(groups.items()):
        correct = sum(bool(row["correct"]) for row in selected)
        result.append(
            {
                "accuracy": correct / len(selected),
                "category": category,
                "correct": correct,
                "mean_f1": sum(float(row["normalized_f1"]) for row in selected)
                / len(selected),
                "normalized_exact_match": sum(
                    bool(row["normalized_exact_match"]) for row in selected
                ),
                "questions": len(selected),
            }
        )
    return result


def _judge_artifact(
    plan: _JudgePlan,
    batch: FastCompletionBatch,
    *,
    preflight_artifact_sha256: str,
) -> dict[str, Any]:
    records = {row.messages_sha256: row for row in batch.unique_records}
    rows: list[dict[str, Any]] = []
    for answer, gold, prompt, verdict in zip(
        plan.answer_plane.rows,
        plan.gold_rows,
        plan.prompt_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = records[prompt.messages_sha256]
        correct = parse_binary_judge_verdict(verdict)
        body: dict[str, Any] = {
            "call_key_sha256": record.call_key_sha256,
            "category": gold.category,
            "correct": correct,
            "dated_question_sha256": gold.dated_question_sha256,
            "demand_class": prompt.demand_class,
            "judge_messages_sha256": prompt.messages_sha256,
            "judge_output": verdict,
            "judge_output_sha256": quote_sha256(verdict),
            "judge_prompt_token_proxy": prompt.prompt_token_proxy,
            "normalized_exact_match": exact_match(
                answer.prediction, gold.reference
            ),
            "normalized_f1": f1_score(answer.prediction, gold.reference),
            "ordinal": answer.ordinal,
            "prediction_sha256": answer.prediction_sha256,
            "question_id": answer.question_id,
            "question_sha256": answer.question_sha256,
            "reference_sha256": gold.reference_sha256,
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "runtime_row_id": answer.runtime_row_id,
        }
        body["judge_row_sha256"] = identity_sha256(body)
        rows.append(body)
    correct = sum(bool(row["correct"]) for row in rows)
    exact = sum(bool(row["normalized_exact_match"]) for row in rows)
    result = {
        "aggregate": {
            "accuracy": correct / len(rows),
            "correct": correct,
            "gate_passed": correct / len(rows) >= TARGET_ACCURACY,
            "incorrect": len(rows) - correct,
            "mean_f1": sum(float(row["normalized_f1"]) for row in rows)
            / len(rows),
            "normalized_exact_match": exact,
            "questions": len(rows),
            "target_accuracy": TARGET_ACCURACY,
        },
        "answer_run_sha256": plan.answer_plane.run_sha256,
        "category_aggregates": _category_aggregates(rows),
        "completion_batch": _stable_batch(batch),
        "format": plan.profile.judge_format,
        "gold_loaded_posthoc": True,
        "gold_population_sha256": plan.gold_population_sha256,
        "judge_completions_may_echo_gold": True,
        "judge_model": DEFAULT_SOL_CALLER_MODEL,
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "prompt_population_sha256": (
            plan.prompt_population.prompt_population_sha256
        ),
        "question_count": len(rows),
        "questions": rows,
        "retained_request_token_state_bytes": 0,
        "runtime_ledger_identity_sha256": plan.answer_plane.runtime_ledger.get(
            "ledger_identity_sha256"
        ),
        "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
        "unique_provider_prompt_count": plan.required_calls,
    }
    return result


def _score_ledger(
    plan: _JudgePlan,
    judge_payload: Mapping[str, Any],
    *,
    judge_artifact_sha256: str,
) -> dict[str, Any]:
    raw_rows = judge_payload.get("questions")
    _require(type(raw_rows) is list, "judge questions must be an array")
    entries: list[ScoreLedgerEntry] = []
    for answer, raw in zip(plan.answer_plane.rows, raw_rows, strict=True):
        _require(
            type(raw) is dict
            and raw.get("runtime_row_id") == answer.runtime_row_id
            and type(raw.get("correct")) is bool,
            f"judge/runtime binding changed at ordinal {answer.ordinal}",
        )
        entries.append(
            ScoreLedgerEntry(
                runtime_row_id=answer.runtime_row_id,
                correct=raw["correct"],
                question_only_demand_class=str(raw["demand_class"]),
                judge_row_sha256=str(raw["judge_row_sha256"]),
                judge_verdict_sha256=str(raw["judge_output_sha256"]),
            )
        )
    return build_score_ledger(
        runtime_ledger=plan.answer_plane.runtime_ledger_json(),
        entries=entries,
        source_artifacts=(
            {
                "role": f"{plan.profile.arm_label}:judge",
                "sha256": judge_artifact_sha256,
            },
        ),
    )


def _run_prebuilt_judge_plan(
    plan: _JudgePlan,
    *,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str,
    max_concurrency: int,
) -> S0V2JudgeRunResult:
    _require(
        enable_provider,
        f"{plan.profile.arm_label} judge run requires provider enablement",
    )
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == plan.required_calls,
        f"authorized provider calls must exactly equal {plan.required_calls}",
    )
    output = Path(output_root)
    preflight, _created = publish_sealed_json(
        output / JUDGE_PREFLIGHT_NAME, _preflight_artifact(plan)
    )
    existing_judge = output / JUDGE_NAME
    if existing_judge.exists():
        source = read_sealed_json(existing_judge)
        return _replay_prebuilt_judge_plan(
            plan,
            expected_judge_sha256=source.sha256,
            output_root=output,
            max_concurrency=max_concurrency,
        )
    load_dotenv()
    api_key = os.environ.get(api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {api_key_env}")
    client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
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
        "Sol completion journal population changed",
    )
    payload = _judge_artifact(
        plan, batch, preflight_artifact_sha256=preflight.sha256
    )
    judge, _created = publish_sealed_json(output / JUDGE_NAME, payload)
    score_payload = _score_ledger(
        plan, payload, judge_artifact_sha256=judge.sha256
    )
    score, _created = publish_sealed_json(output / SCORE_LEDGER_NAME, score_payload)
    return S0V2JudgeRunResult(
        judge_artifact=judge,
        score_ledger_artifact=score,
        correct=int(payload["aggregate"]["correct"]),
        physical_provider_calls=batch.usage.physical_calls,
        checkpoint_hits=batch.usage.checkpoint_hits,
    )


def _replay_prebuilt_judge_plan(
    plan: _JudgePlan,
    *,
    expected_judge_sha256: str,
    output_root: str | Path,
    max_concurrency: int,
) -> S0V2JudgeRunResult:
    require_sha256(expected_judge_sha256, "expected judge SHA-256")
    output = Path(output_root)
    preflight = read_sealed_json(output / JUDGE_PREFLIGHT_NAME)
    _require(
        preflight.payload == _preflight_artifact(plan),
        "judge preflight changed during replay",
    )
    source = read_sealed_json(output / JUDGE_NAME)
    _require(source.sha256 == expected_judge_sha256, "judge SHA-256 changed")
    batch = _runtime(
        plan,
        checkpoint_dir=output / JUDGE_CHECKPOINT_DIR_NAME,
        client=None,
        max_concurrency=max_concurrency,
        preflight_artifact_sha256=preflight.sha256,
    ).run()
    _require(batch.usage.physical_calls == 0, "judge replay made provider calls")
    _require(
        batch.usage.checkpoint_hits == plan.required_calls,
        "judge replay checkpoint population changed",
    )
    expected = _judge_artifact(
        plan, batch, preflight_artifact_sha256=preflight.sha256
    )
    _require(
        canonical_json_bytes(expected) == canonical_json_bytes(source.payload),
        "judge differs from immutable Sol journals",
    )
    replay, _created = publish_sealed_json(output / JUDGE_REPLAY_NAME, expected)
    score_expected = _score_ledger(
        plan, expected, judge_artifact_sha256=source.sha256
    )
    score = read_sealed_json(output / SCORE_LEDGER_NAME)
    _require(
        canonical_json_bytes(score.payload) == canonical_json_bytes(score_expected),
        "score ledger differs from replayed Sol verdicts",
    )
    score_replay, _created = publish_sealed_json(
        output / SCORE_LEDGER_REPLAY_NAME, score_expected
    )
    return S0V2JudgeRunResult(
        judge_artifact=replay,
        score_ledger_artifact=score_replay,
        correct=int(expected["aggregate"]["correct"]),
        physical_provider_calls=0,
        checkpoint_hits=batch.usage.checkpoint_hits,
    )


def preflight_dual_answer_synthesis_judge(
    *,
    answer_plane: VerifiedS0V2AnswerPlane,
    dataset_path: str | Path,
    split_path: str | Path,
    output_root: str | Path,
) -> SealedArtifact:
    """Seal the Sol prompt population for a pre-verified synthesis plane."""

    plan = _build_plan_from_answer_plane(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        profile=_DUAL_ANSWER_SYNTHESIS_JUDGE_PROFILE,
    )
    artifact, _created = publish_sealed_json(
        Path(output_root) / JUDGE_PREFLIGHT_NAME, _preflight_artifact(plan)
    )
    return artifact


def run_dual_answer_synthesis_judge(
    *,
    answer_plane: VerifiedS0V2AnswerPlane,
    dataset_path: str | Path,
    split_path: str | Path,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    max_concurrency: int = 4,
) -> S0V2JudgeRunResult:
    """Judge a supplied, verified synthesis plane without reloading S0."""

    plan = _build_plan_from_answer_plane(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        profile=_DUAL_ANSWER_SYNTHESIS_JUDGE_PROFILE,
    )
    return _run_prebuilt_judge_plan(
        plan,
        output_root=output_root,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
        api_key_env=api_key_env,
        max_concurrency=max_concurrency,
    )


def replay_dual_answer_synthesis_judge(
    *,
    answer_plane: VerifiedS0V2AnswerPlane,
    expected_judge_sha256: str,
    dataset_path: str | Path,
    split_path: str | Path,
    output_root: str | Path,
    max_concurrency: int = 4,
) -> S0V2JudgeRunResult:
    """Replay Sol journals for a supplied, verified synthesis plane."""

    plan = _build_plan_from_answer_plane(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        profile=_DUAL_ANSWER_SYNTHESIS_JUDGE_PROFILE,
    )
    return _replay_prebuilt_judge_plan(
        plan,
        expected_judge_sha256=expected_judge_sha256,
        output_root=output_root,
        max_concurrency=max_concurrency,
    )


# Generic seam names retained for callers that operate on the verified-plane
# protocol without owning the synthesis arm's public naming convention.
preflight_verified_answer_plane_judge = preflight_dual_answer_synthesis_judge
run_verified_answer_plane_judge = run_dual_answer_synthesis_judge
replay_verified_answer_plane_judge = replay_dual_answer_synthesis_judge


def preflight_s0_v2_judge(
    *,
    answer_run_path: str | Path,
    answer_replay_path: str | Path,
    expected_answer_run_sha256: str,
    retrieval_path: str | Path,
    dataset_path: str | Path,
    split_path: str | Path,
    output_root: str | Path,
    answer_checkpoint_dir: str | Path | None = None,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    renderer_id: str = RENDERER_ID,
    selected_ordinals: Sequence[int] | None = None,
) -> SealedArtifact:
    plan = _build_plan(
        answer_run_path=answer_run_path,
        answer_replay_path=answer_replay_path,
        expected_answer_run_sha256=expected_answer_run_sha256,
        retrieval_path=retrieval_path,
        dataset_path=dataset_path,
        split_path=split_path,
        answer_checkpoint_dir=answer_checkpoint_dir,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=renderer_id,
        selected_ordinals=selected_ordinals,
    )
    artifact, _created = publish_sealed_json(
        Path(output_root) / JUDGE_PREFLIGHT_NAME, _preflight_artifact(plan)
    )
    return artifact


def run_s0_v2_judge(
    *,
    answer_run_path: str | Path,
    answer_replay_path: str | Path,
    expected_answer_run_sha256: str,
    retrieval_path: str | Path,
    dataset_path: str | Path,
    split_path: str | Path,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    answer_checkpoint_dir: str | Path | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    renderer_id: str = RENDERER_ID,
    selected_ordinals: Sequence[int] | None = None,
) -> S0V2JudgeRunResult:
    plan = _build_plan(
        answer_run_path=answer_run_path,
        answer_replay_path=answer_replay_path,
        expected_answer_run_sha256=expected_answer_run_sha256,
        retrieval_path=retrieval_path,
        dataset_path=dataset_path,
        split_path=split_path,
        answer_checkpoint_dir=answer_checkpoint_dir,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=renderer_id,
        selected_ordinals=selected_ordinals,
    )
    return _run_prebuilt_judge_plan(
        plan,
        output_root=output_root,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
        api_key_env=api_key_env,
        max_concurrency=max_concurrency,
    )


def _v3_arguments(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(kwargs)
    if "renderer_id" in values:
        raise TypeError("v3 wrappers own the renderer identity")
    values["renderer_id"] = V3_RENDERER_ID
    return values


def preflight_s0_v3_judge(**kwargs: Any) -> SealedArtifact:
    return preflight_s0_v2_judge(**_v3_arguments(kwargs))


def run_s0_v3_judge(**kwargs: Any) -> S0V2JudgeRunResult:
    return run_s0_v2_judge(**_v3_arguments(kwargs))


def replay_s0_v3_judge(**kwargs: Any) -> S0V2JudgeRunResult:
    return replay_s0_v2_judge(**_v3_arguments(kwargs))


def _v4_arguments(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(kwargs)
    if "renderer_id" in values:
        raise TypeError("v4 wrappers own the renderer identity")
    values["renderer_id"] = V4_RENDERER_ID
    return values


def preflight_s0_v4_judge(**kwargs: Any) -> SealedArtifact:
    return preflight_s0_v2_judge(**_v4_arguments(kwargs))


def run_s0_v4_judge(**kwargs: Any) -> S0V2JudgeRunResult:
    return run_s0_v2_judge(**_v4_arguments(kwargs))


def replay_s0_v4_judge(**kwargs: Any) -> S0V2JudgeRunResult:
    return replay_s0_v2_judge(**_v4_arguments(kwargs))


def replay_s0_v2_judge(
    *,
    answer_run_path: str | Path,
    answer_replay_path: str | Path,
    expected_answer_run_sha256: str,
    expected_judge_sha256: str,
    retrieval_path: str | Path,
    dataset_path: str | Path,
    split_path: str | Path,
    output_root: str | Path,
    answer_checkpoint_dir: str | Path | None = None,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    renderer_id: str = RENDERER_ID,
    selected_ordinals: Sequence[int] | None = None,
) -> S0V2JudgeRunResult:
    plan = _build_plan(
        answer_run_path=answer_run_path,
        answer_replay_path=answer_replay_path,
        expected_answer_run_sha256=expected_answer_run_sha256,
        retrieval_path=retrieval_path,
        dataset_path=dataset_path,
        split_path=split_path,
        answer_checkpoint_dir=answer_checkpoint_dir,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=renderer_id,
        selected_ordinals=selected_ordinals,
    )
    return _replay_prebuilt_judge_plan(
        plan,
        expected_judge_sha256=expected_judge_sha256,
        output_root=output_root,
        max_concurrency=max_concurrency,
    )


__all__ = [
    "DEFAULT_MAX_JUDGE_PROMPT_TOKENS",
    "DEFAULT_SOL_CALLER_MODEL",
    "DEFAULT_SOL_GATEWAY_MODEL",
    "DUAL_ANSWER_SYNTHESIS_ARM_LABEL",
    "DUAL_ANSWER_SYNTHESIS_JUDGE_FORMAT",
    "DUAL_ANSWER_SYNTHESIS_JUDGE_PLAN_ID",
    "DUAL_ANSWER_SYNTHESIS_JUDGE_PREFLIGHT_FORMAT",
    "DUAL_ANSWER_SYNTHESIS_RENDERER_ID",
    "JUDGE_CHECKPOINT_DIR_NAME",
    "JUDGE_FORMAT",
    "JUDGE_NAME",
    "JUDGE_PREFLIGHT_FORMAT",
    "JUDGE_PREFLIGHT_NAME",
    "JUDGE_REPLAY_NAME",
    "SCORE_LEDGER_NAME",
    "SCORE_LEDGER_REPLAY_NAME",
    "S0V2JudgeRunResult",
    "V3_JUDGE_FORMAT",
    "V3_JUDGE_PLAN_ID",
    "V3_JUDGE_PREFLIGHT_FORMAT",
    "V4_JUDGE_FORMAT",
    "V4_JUDGE_PLAN_ID",
    "V4_JUDGE_PREFLIGHT_FORMAT",
    "preflight_s0_v2_judge",
    "preflight_s0_v3_judge",
    "preflight_s0_v4_judge",
    "preflight_dual_answer_synthesis_judge",
    "preflight_verified_answer_plane_judge",
    "replay_s0_v2_judge",
    "replay_s0_v3_judge",
    "replay_s0_v4_judge",
    "replay_dual_answer_synthesis_judge",
    "replay_verified_answer_plane_judge",
    "run_s0_v2_judge",
    "run_s0_v3_judge",
    "run_s0_v4_judge",
    "run_dual_answer_synthesis_judge",
    "run_verified_answer_plane_judge",
]
