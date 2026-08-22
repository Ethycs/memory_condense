"""Strict schema-v3 comparison for fixed-stage semantic-judge scores.

This module is loaded lazily by ``tools.mem0_eval.compare`` so the legacy
schema-v2 comparator remains independent.  The treatment artifact intentionally
contains hashes rather than answer/gold text; only binary-judge metrics are
therefore comparable here.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    build_responder_prompt_policy_identity,
)

from .compare import (
    COMPARISON_REPORT_TYPE,
    FIXED_STAGE_COMPARISON_SCHEMA_VERSION,
    FIXED_STAGE_ID,
    FIXED_STAGE_JUDGE_CAMPAIGN_FORMAT,
    FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
    FIXED_STAGE_JUDGE_REQUEST_FORMAT,
    FIXED_STAGE_JUDGE_RESPONSE_FORMAT,
    FIXED_STAGE_JUDGE_RUNTIME_FORMAT,
    FIXED_STAGE_PROMPT_CAP,
    FIXED_STAGE_RESPONDER_MAX_OUTPUT_TOKENS,
    FIXED_STAGE_RESPONDER_RUNTIME_FORMAT,
    FIXED_STAGE_SOL_GATEWAY_MODEL,
    FIXED_STAGE_SOL_MODEL,
    FIXED_STAGE_TERRA_GATEWAY_MODEL,
    FIXED_STAGE_TERRA_MODEL,
    FIXED_STAGE_TREATMENT_FORMAT,
    FROZEN_QUESTION_COUNT,
    MEM0_ATTRIBUTION_KIND,
    MEM0_SOURCE_COVERAGE_REASON,
    PairedComparisonError,
    _FIXED_STAGE_COMPLETION_FIELDS,
    _FIXED_STAGE_JUDGE_RUNTIME_FIELDS,
    _FIXED_STAGE_QUESTION_FIELDS,
    _FIXED_STAGE_TREATMENT_TOP_FIELDS,
    _ValidatedArm,
    _boolean,
    _exact_keys,
    _integer,
    _judge_verdict,
    _list,
    _mapping,
    _must_close,
    _must_equal,
    _number,
    _sha256,
    _text,
    _walk_json,
    canonical_sha256,
)
from .compare_fixed_stage_derivation import verify_treatment_prompt_derivation


@dataclass(frozen=True, slots=True)
class _ValidatedFixedStageScore:
    report: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    canonical_sha256: str
    judge_accuracy: float
    correct: int


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_fixed_accuracy_summary(
    value: Any,
    *,
    label: str,
    questions: int,
    correct: int,
    gate: bool = False,
) -> dict[str, Any]:
    summary = _mapping(value, label)
    fields = {
        "questions",
        "correct",
        "incorrect",
        "binary_accuracy",
        "target_accuracy",
        "minimum_questions",
        "minimum_correct_at_observed_population",
        "accuracy_threshold_met",
        "minimum_population_met",
        "gate_passed",
        "status",
    }
    if gate:
        fields.update({"gate_unit", "fixed_stage_id"})
    _exact_keys(summary, fields, label)
    accuracy = correct / questions if questions else 0.0
    accuracy_met = accuracy >= 0.95
    population_met = questions >= FROZEN_QUESTION_COUNT
    passed = accuracy_met and population_met
    expected = {
        "questions": questions,
        "correct": correct,
        "incorrect": questions - correct,
        "binary_accuracy": accuracy,
        "target_accuracy": 0.95,
        "minimum_questions": FROZEN_QUESTION_COUNT,
        "minimum_correct_at_observed_population": math.ceil(0.95 * questions),
        "accuracy_threshold_met": accuracy_met,
        "minimum_population_met": population_met,
        "gate_passed": passed,
        "status": (
            "pass"
            if passed
            else "insufficient_population"
            if not population_met
            else "below_accuracy_target"
        ),
    }
    if gate:
        expected.update(
            {
                "gate_unit": "one preregistered fixed retrieval stage",
                "fixed_stage_id": FIXED_STAGE_ID,
            }
        )
    _must_equal(summary, expected, label)
    return summary


def _validate_fixed_stage_policy(value: Any) -> dict[str, Any]:
    label = "fixed treatment.semantic_judge_policy"
    policy = _mapping(value, label)
    fields = {
        "format",
        "answer_artifact_validator",
        "judge_prompt_builder",
        "verdict_parser",
        "question_form",
        "fixed_stage_id",
        "gate_unit",
        "target_accuracy",
        "minimum_questions",
        "responder_model",
        "responder_gateway_model",
        "responder_runtime_format",
        "responder_prompt_policy_binding",
        "responder_max_prompt_token_proxy",
        "responder_max_new_tokens",
        "judge_model",
        "judge_gateway_model",
        "judge_runtime_format",
        "judge_max_new_tokens",
        "gateway_url",
        "provider_retries",
        "responder_temperature",
        "judge_temperature",
        "deduplication",
        "gold_access",
    }
    _exact_keys(policy, fields, label)
    exact = {
        "format": "memory-condense-fixed-stage-semantic-judge-policy-v1",
        "answer_artifact_validator": (
            "memory_condense.eval.recall_guarded_cumulative_final_answer."
            "validate_final_answer_artifact"
        ),
        "judge_prompt_builder": "memory_condense.eval.benchmark.build_judge_prompt",
        "verdict_parser": (
            "memory_condense.eval._binary_judge_protocol."
            "parse_binary_judge_verdict"
        ),
        "question_form": "undated benchmark question",
        "fixed_stage_id": FIXED_STAGE_ID,
        "gate_unit": "one preregistered fixed retrieval stage",
        "target_accuracy": 0.95,
        "minimum_questions": FROZEN_QUESTION_COUNT,
        "responder_model": FIXED_STAGE_TERRA_MODEL,
        "responder_gateway_model": FIXED_STAGE_TERRA_GATEWAY_MODEL,
        "responder_runtime_format": FIXED_STAGE_RESPONDER_RUNTIME_FORMAT,
        "responder_prompt_policy_binding": (
            "derived from sealed system messages and verified QA framing"
        ),
        "responder_max_prompt_token_proxy": FIXED_STAGE_PROMPT_CAP,
        "responder_max_new_tokens": FIXED_STAGE_RESPONDER_MAX_OUTPUT_TOKENS,
        "judge_model": FIXED_STAGE_SOL_MODEL,
        "judge_gateway_model": FIXED_STAGE_SOL_GATEWAY_MODEL,
        "judge_runtime_format": FIXED_STAGE_JUDGE_RUNTIME_FORMAT,
        "judge_max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        "provider_retries": 0,
        "responder_temperature": None,
        "judge_temperature": None,
        "deduplication": (
            "identical canonical question+gold+prediction messages"
        ),
        "gold_access": "after complete final-answer artifact validation",
    }
    for field, expected in exact.items():
        _must_equal(policy[field], expected, f"{label}.{field}")
    _text(policy["gateway_url"], f"{label}.gateway_url")
    return policy


def _validate_fixed_stage_campaign(
    value: Any,
    *,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    label = "fixed treatment.campaign_binding"
    campaign = _mapping(value, label)
    fields = {
        "format",
        "final_answer_artifact_sha256",
        "responder_runtime_identity_sha256",
        "responder_prompt_policy",
        "responder_prompt_policy_sha256",
        "retrieval_sha256",
        "population_identity_sha256",
        "gold_scoring_population_sha256",
        "question_count",
        "fixed_stage_id",
        "ordered_judgment_population_sha256",
        "logical_judgment_count",
        "unique_judge_prompt_count",
        "judge_prompt_population_sha256",
        "maximum_judge_prompt_token_proxy",
        "authorized_unique_judge_calls",
        "semantic_judge_policy_sha256",
        "semantic_judge_implementation_sha256",
        "target_accuracy",
        "minimum_questions",
        "responder_model",
        "responder_max_new_tokens",
        "responder_prompt_cap",
        "judge_model",
        "judge_max_new_tokens",
        "provider_retries",
    }
    _exact_keys(campaign, fields, label)
    exact = {
        "format": FIXED_STAGE_JUDGE_CAMPAIGN_FORMAT,
        "final_answer_artifact_sha256": report[
            "final_answer_artifact_sha256"
        ],
        "responder_runtime_identity_sha256": report[
            "responder_runtime_identity_sha256"
        ],
        "responder_prompt_policy": report["responder_prompt_policy"],
        "responder_prompt_policy_sha256": report[
            "responder_prompt_policy_sha256"
        ],
        "retrieval_sha256": report["retrieval_sha256"],
        "population_identity_sha256": report[
            "population_identity_sha256"
        ],
        "gold_scoring_population_sha256": report[
            "gold_scoring_population_sha256"
        ],
        "question_count": FROZEN_QUESTION_COUNT,
        "fixed_stage_id": FIXED_STAGE_ID,
        "logical_judgment_count": FROZEN_QUESTION_COUNT,
        "semantic_judge_policy_sha256": report[
            "semantic_judge_policy_sha256"
        ],
        "semantic_judge_implementation_sha256": report[
            "semantic_judge_implementation_sha256"
        ],
        "target_accuracy": 0.95,
        "minimum_questions": FROZEN_QUESTION_COUNT,
        "responder_model": FIXED_STAGE_TERRA_MODEL,
        "responder_max_new_tokens": FIXED_STAGE_RESPONDER_MAX_OUTPUT_TOKENS,
        "responder_prompt_cap": FIXED_STAGE_PROMPT_CAP,
        "judge_model": FIXED_STAGE_SOL_MODEL,
        "judge_max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        "provider_retries": 0,
    }
    for field, expected in exact.items():
        _must_equal(campaign[field], expected, f"{label}.{field}")
    for field in (
        "ordered_judgment_population_sha256",
        "judge_prompt_population_sha256",
    ):
        _sha256(campaign[field], f"{label}.{field}")
    unique = _integer(
        campaign["unique_judge_prompt_count"],
        f"{label}.unique_judge_prompt_count",
        minimum=1,
    )
    _must_equal(
        campaign["authorized_unique_judge_calls"],
        unique,
        f"{label}.authorized_unique_judge_calls",
    )
    _integer(
        campaign["maximum_judge_prompt_token_proxy"],
        f"{label}.maximum_judge_prompt_token_proxy",
    )
    return campaign


def _validate_fixed_stage_runtime(
    value: Any,
    *,
    campaign: Mapping[str, Any],
    campaign_sha256: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    label = "fixed treatment.judge_runtime_identity"
    runtime = _mapping(value, label)
    _exact_keys(runtime, _FIXED_STAGE_JUDGE_RUNTIME_FIELDS, label)
    exact = {
        "format": FIXED_STAGE_JUDGE_RUNTIME_FORMAT,
        "gateway_url": policy["gateway_url"],
        "caller_model": FIXED_STAGE_SOL_MODEL,
        "gateway_model": FIXED_STAGE_SOL_GATEWAY_MODEL,
        "default_max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        "retries": 0,
        "temperature": None,
        "campaign_binding": campaign,
        "campaign_binding_sha256": campaign_sha256,
        "request_journal_format": FIXED_STAGE_JUDGE_REQUEST_FORMAT,
        "response_journal_format": FIXED_STAGE_JUDGE_RESPONSE_FORMAT,
    }
    for field, expected in exact.items():
        _must_equal(runtime[field], expected, f"{label}.{field}")
    authorized = _integer(
        runtime["authorized_unique_calls"],
        f"{label}.authorized_unique_calls",
        minimum=1,
    )
    _must_equal(
        authorized,
        campaign["authorized_unique_judge_calls"],
        f"{label}.authorized_unique_calls",
    )
    return runtime


def _validate_fixed_stage_completion(
    value: Any,
    *,
    label: str,
    row: Mapping[str, Any],
    runtime: Mapping[str, Any],
    runtime_sha256: str,
    campaign_sha256: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    completion = _mapping(value, label)
    _exact_keys(completion, _FIXED_STAGE_COMPLETION_FIELDS, label)
    exact = {
        "gateway_url": policy["gateway_url"],
        "caller_model": FIXED_STAGE_SOL_MODEL,
        "gateway_model": FIXED_STAGE_SOL_GATEWAY_MODEL,
        "call_key_sha256": row["call_key_sha256"],
        "runtime_identity_sha256": runtime_sha256,
        "campaign_binding_sha256": campaign_sha256,
        "request_journal_sha256": row["request_journal_sha256"],
        "messages_sha256": row["judge_messages_sha256"],
        "completion_sha256": row["judge_output_sha256"],
        "max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        "input_token_proxy": row["judge_prompt_token_proxy"],
        "retries": 0,
    }
    for field, expected in exact.items():
        _must_equal(completion[field], expected, f"{label}.{field}")
    for field in ("response_id", "response_model", "finish_reason"):
        _text(completion[field], f"{label}.{field}", allow_empty=True)
    available_fields = (
        "reported_input_tokens_available",
        "reported_output_tokens_available",
        "reported_total_tokens_available",
    )
    token_fields = (
        "reported_input_tokens",
        "reported_output_tokens",
        "reported_total_tokens",
    )
    availability = tuple(
        _boolean(completion[field], f"{label}.{field}")
        for field in available_fields
    )
    tokens = tuple(
        _integer(completion[field], f"{label}.{field}")
        for field in token_fields
    )
    _must_equal(
        availability,
        tuple(value > 0 for value in tokens),
        f"{label}.reported token availability",
    )
    _must_equal(
        completion["reported_usage_available"],
        any(availability),
        f"{label}.reported_usage_available",
    )
    output_token_proxy = _integer(
        completion["output_token_proxy"], f"{label}.output_token_proxy"
    )
    _must_equal(
        output_token_proxy,
        count_tokens(str(row["judge_output"])),
        f"{label}.output_token_proxy",
    )
    _number(completion["elapsed_s"], f"{label}.elapsed_s", minimum=0.0)
    cache_hit = _boolean(completion["cache_hit"], f"{label}.cache_hit")
    physical_call = _boolean(
        completion["physical_call"], f"{label}.physical_call"
    )
    _must_equal(cache_hit, False, f"{label}.cache_hit")
    _must_equal(physical_call, True, f"{label}.physical_call")
    logical_calls = _integer(
        completion["cumulative_logical_calls"],
        f"{label}.cumulative_logical_calls",
        minimum=1,
    )
    unique_calls = _integer(
        completion["cumulative_unique_calls"],
        f"{label}.cumulative_unique_calls",
        minimum=1,
    )
    physical_calls = _integer(
        completion["cumulative_physical_calls"],
        f"{label}.cumulative_physical_calls",
        minimum=1,
    )
    checkpoint_hits = _integer(
        completion["cumulative_checkpoint_hits"],
        f"{label}.cumulative_checkpoint_hits",
    )
    authorized_calls = int(runtime["authorized_unique_calls"])
    if (
        unique_calls != logical_calls
        or physical_calls > unique_calls
        or unique_calls > authorized_calls
        or checkpoint_hits > logical_calls
        or physical_calls + checkpoint_hits != logical_calls
    ):
        raise PairedComparisonError(
            f"{label} has producer-impossible cumulative call counters"
        )
    call_key_payload = {
        "messages_sha256": row["judge_messages_sha256"],
        "runtime_identity_sha256": runtime_sha256,
        "max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        "campaign_binding_sha256": campaign_sha256,
    }
    _must_equal(
        row["call_key_sha256"],
        canonical_sha256(call_key_payload),
        f"{label}.call_key_sha256",
    )
    return completion


def _validate_fixed_stage_question(
    value: Any,
    *,
    index: int,
    runtime: Mapping[str, Any],
    runtime_sha256: str,
    campaign_sha256: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    label = f"fixed treatment.questions[{index}]"
    row = _mapping(value, label)
    _exact_keys(row, _FIXED_STAGE_QUESTION_FIELDS, label)
    _must_equal(row["ordinal"], index, f"{label}.ordinal")
    question_id = _text(row["question_id"], f"{label}.question_id")
    category = _text(row["category"], f"{label}.category")
    for field in (
        "question_sha256",
        "dated_question_sha256",
        "gold_answer_sha256",
        "prediction_sha256",
        "answer_call_key_sha256",
        "answer_response_journal_sha256",
        "judge_messages_sha256",
        "judge_output_sha256",
        "call_key_sha256",
        "request_journal_sha256",
        "response_journal_sha256",
    ):
        _sha256(row[field], f"{label}.{field}")
    _must_equal(row["fixed_stage_id"], FIXED_STAGE_ID, f"{label}.fixed_stage_id")
    prompt_tokens = _integer(
        row["judge_prompt_token_proxy"],
        f"{label}.judge_prompt_token_proxy",
    )
    judge_output = _text(row["judge_output"], f"{label}.judge_output")
    _must_equal(
        row["judge_output_sha256"],
        _text_sha256(judge_output),
        f"{label}.judge_output_sha256",
    )
    verdict = _judge_verdict(judge_output, f"{label}.judge_output")
    _must_equal(row["correct"], verdict, f"{label}.correct")
    completion = _validate_fixed_stage_completion(
        row["completion_report"],
        label=f"{label}.completion_report",
        row=row,
        runtime=runtime,
        runtime_sha256=runtime_sha256,
        campaign_sha256=campaign_sha256,
        policy=policy,
    )
    return {
        **row,
        "question_id": question_id,
        "category": category,
        "judge_prompt_token_proxy": prompt_tokens,
        "correct": verdict,
        "completion_report": completion,
    }


def _validate_fixed_stage_judge_usage(
    value: Any,
    *,
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    label = "fixed treatment.judge_usage"
    usage = _mapping(value, label)
    fields = {
        "unique_journaled_calls",
        "reported_input_tokens_available_calls",
        "reported_input_tokens",
        "reported_output_tokens_available_calls",
        "reported_output_tokens",
        "reported_total_tokens_available_calls",
        "reported_total_tokens",
        "input_token_proxy",
        "output_token_proxy",
        "elapsed_s",
        "retries",
    }
    _exact_keys(usage, fields, label)

    def available(name: str) -> list[int]:
        return [
            int(report[name])
            for report in reports
            if report[f"{name}_available"] is True
        ]

    inputs = available("reported_input_tokens")
    outputs = available("reported_output_tokens")
    totals = available("reported_total_tokens")
    expected = {
        "unique_journaled_calls": len(reports),
        "reported_input_tokens_available_calls": len(inputs),
        "reported_input_tokens": sum(inputs),
        "reported_output_tokens_available_calls": len(outputs),
        "reported_output_tokens": sum(outputs),
        "reported_total_tokens_available_calls": len(totals),
        "reported_total_tokens": sum(totals),
        "input_token_proxy": sum(
            int(report["input_token_proxy"]) for report in reports
        ),
        "output_token_proxy": sum(
            int(report["output_token_proxy"]) for report in reports
        ),
        "elapsed_s": math.fsum(float(report["elapsed_s"]) for report in reports),
        "retries": 0,
    }
    for field in fields - {"elapsed_s"}:
        _must_equal(usage[field], expected[field], f"{label}.{field}")
    _must_close(usage["elapsed_s"], float(expected["elapsed_s"]), f"{label}.elapsed_s")
    return usage


def _validate_fixed_stage_score(value: Any) -> _ValidatedFixedStageScore:
    label = "fixed treatment score"
    report = _mapping(value, label)
    _walk_json(report, label)
    _exact_keys(report, _FIXED_STAGE_TREATMENT_TOP_FIELDS, label)
    _must_equal(report["format"], FIXED_STAGE_TREATMENT_FORMAT, f"{label}.format")
    for field in (
        "final_answer_artifact_sha256",
        "responder_runtime_identity_sha256",
        "responder_prompt_policy_sha256",
        "retrieval_sha256",
        "population_identity_sha256",
        "gold_scoring_population_sha256",
        "semantic_judge_policy_sha256",
        "semantic_judge_implementation_sha256",
        "campaign_binding_sha256",
        "judge_runtime_identity_sha256",
    ):
        _sha256(report[field], f"{label}.{field}")
    _must_equal(report["question_count"], FROZEN_QUESTION_COUNT, f"{label}.question_count")
    _must_equal(report["gold_loaded_posthoc"], True, f"{label}.gold_loaded_posthoc")
    _must_equal(report["independent_llm_judge"], True, f"{label}.independent_llm_judge")
    _must_equal(report["fixed_stage_id"], FIXED_STAGE_ID, f"{label}.fixed_stage_id")
    _must_equal(report["responder_model"], FIXED_STAGE_TERRA_MODEL, f"{label}.responder_model")
    _must_equal(report["judge_model"], FIXED_STAGE_SOL_MODEL, f"{label}.judge_model")

    responder_prompt_policy = _mapping(
        report["responder_prompt_policy"],
        f"{label}.responder_prompt_policy",
    )
    _must_equal(
        report["responder_prompt_policy_sha256"],
        canonical_sha256(responder_prompt_policy),
        f"{label}.responder_prompt_policy_sha256",
    )

    policy = _validate_fixed_stage_policy(report["semantic_judge_policy"])
    _must_equal(
        report["semantic_judge_policy_sha256"],
        canonical_sha256(policy),
        f"{label}.semantic_judge_policy_sha256",
    )
    campaign = _validate_fixed_stage_campaign(
        report["campaign_binding"], report=report
    )
    campaign_sha = canonical_sha256(campaign)
    _must_equal(
        report["campaign_binding_sha256"],
        campaign_sha,
        f"{label}.campaign_binding_sha256",
    )
    runtime = _validate_fixed_stage_runtime(
        report["judge_runtime_identity"],
        campaign=campaign,
        campaign_sha256=campaign_sha,
        policy=policy,
    )
    runtime_sha = canonical_sha256(runtime)
    _must_equal(
        report["judge_runtime_identity_sha256"],
        runtime_sha,
        f"{label}.judge_runtime_identity_sha256",
    )

    raw_rows = _list(report["questions"], f"{label}.questions")
    rows = tuple(
        _validate_fixed_stage_question(
            row,
            index=index,
            runtime=runtime,
            runtime_sha256=runtime_sha,
            campaign_sha256=campaign_sha,
            policy=policy,
        )
        for index, row in enumerate(raw_rows)
    )
    ids = tuple(str(row["question_id"]) for row in rows)
    if len(ids) != FROZEN_QUESTION_COUNT:
        raise PairedComparisonError(
            "fixed treatment questions must contain exactly 100 rows"
        )
    if len(set(ids)) != len(ids):
        raise PairedComparisonError(
            "fixed treatment question IDs must be unique"
        )

    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        message_sha = str(row["judge_messages_sha256"])
        existing = unique.setdefault(message_sha, row)
        if any(
            existing[field] != row[field]
            for field in (
                "judge_output",
                "judge_output_sha256",
                "call_key_sha256",
                "request_journal_sha256",
                "response_journal_sha256",
                "completion_report",
            )
        ):
            raise PairedComparisonError(
                "fixed treatment deduplicated judge prompt has conflicting outcomes"
            )
    _must_equal(
        report["logical_judgment_count"],
        FROZEN_QUESTION_COUNT,
        f"{label}.logical_judgment_count",
    )
    _must_equal(
        report["unique_judge_prompt_count"],
        len(unique),
        f"{label}.unique_judge_prompt_count",
    )
    _must_equal(
        report["deduplicated_logical_judgment_count"],
        FROZEN_QUESTION_COUNT - len(unique),
        f"{label}.deduplicated_logical_judgment_count",
    )
    _must_equal(
        campaign["unique_judge_prompt_count"],
        len(unique),
        "fixed treatment.campaign_binding.unique_judge_prompt_count",
    )
    _must_equal(
        runtime["authorized_unique_calls"],
        len(unique),
        "fixed treatment.judge_runtime_identity.authorized_unique_calls",
    )
    for field in (
        "call_key_sha256",
        "request_journal_sha256",
        "response_journal_sha256",
    ):
        receipts = [str(row[field]) for row in unique.values()]
        if len(set(receipts)) != len(receipts):
            raise PairedComparisonError(
                "distinct fixed treatment judge prompts alias the same "
                f"{field}"
            )
    call_keys = {str(row["call_key_sha256"]) for row in unique.values()}
    requests = {
        str(row["request_journal_sha256"]) for row in unique.values()
    }
    responses = {
        str(row["response_journal_sha256"]) for row in unique.values()
    }
    if call_keys & requests or call_keys & responses or requests & responses:
        raise PairedComparisonError(
            "fixed treatment call-key/request/response receipt namespaces "
            "must be disjoint"
        )

    gold_population = [
        {
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
            "dated_question_sha256": row["dated_question_sha256"],
            "gold_answer_sha256": row["gold_answer_sha256"],
        }
        for row in rows
    ]
    _must_equal(
        report["gold_scoring_population_sha256"],
        canonical_sha256(gold_population),
        f"{label}.gold_scoring_population_sha256",
    )
    ordered_judgments = [
        {
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
            "gold_answer_sha256": row["gold_answer_sha256"],
            "prediction_sha256": row["prediction_sha256"],
            "judge_messages_sha256": row["judge_messages_sha256"],
            "answer_call_key_sha256": row["answer_call_key_sha256"],
            "answer_response_journal_sha256": row[
                "answer_response_journal_sha256"
            ],
        }
        for row in rows
    ]
    _must_equal(
        campaign["ordered_judgment_population_sha256"],
        canonical_sha256(ordered_judgments),
        "fixed treatment.campaign_binding.ordered_judgment_population_sha256",
    )
    prompt_population = [
        {
            "messages_sha256": digest,
            "logical_references": sum(
                row["judge_messages_sha256"] == digest for row in rows
            ),
        }
        for digest in unique
    ]
    _must_equal(
        campaign["judge_prompt_population_sha256"],
        canonical_sha256(prompt_population),
        "fixed treatment.campaign_binding.judge_prompt_population_sha256",
    )
    max_prompt = max(int(row["judge_prompt_token_proxy"]) for row in rows)
    _must_equal(
        campaign["maximum_judge_prompt_token_proxy"],
        max_prompt,
        "fixed treatment.campaign_binding.maximum_judge_prompt_token_proxy",
    )
    preflight = _mapping(
        report["judge_prompt_preflight"],
        f"{label}.judge_prompt_preflight",
    )
    expected_preflight = {
        "completed_before_provider_calls": True,
        "logical_prompt_count": FROZEN_QUESTION_COUNT,
        "unique_prompt_count": len(unique),
        "maximum_prompt_token_proxy": max_prompt,
    }
    _exact_keys(preflight, expected_preflight, f"{label}.judge_prompt_preflight")
    _must_equal(preflight, expected_preflight, f"{label}.judge_prompt_preflight")
    reports = [row["completion_report"] for row in unique.values()]
    _validate_fixed_stage_judge_usage(report["judge_usage"], reports=reports)

    category_counts: dict[str, int] = {}
    category_correct: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
        category_correct[category] = category_correct.get(category, 0) + int(
            bool(row["correct"])
        )
    _must_equal(
        report["category_counts"],
        dict(sorted(category_counts.items())),
        f"{label}.category_counts",
    )
    category_aggregates = _list(
        report["category_aggregates"], f"{label}.category_aggregates"
    )
    if len(category_aggregates) != len(category_counts):
        raise PairedComparisonError(
            "fixed treatment.category_aggregates category count mismatch"
        )
    for index, category in enumerate(sorted(category_counts)):
        category_label = f"{label}.category_aggregates[{index}]"
        child = _mapping(category_aggregates[index], category_label)
        _must_equal(child.get("category"), category, f"{category_label}.category")
        _validate_fixed_accuracy_summary(
            {key: value for key, value in child.items() if key != "category"},
            label=category_label,
            questions=category_counts[category],
            correct=category_correct[category],
        )

    correct = sum(int(bool(row["correct"])) for row in rows)
    _validate_fixed_accuracy_summary(
        report["aggregate"],
        label=f"{label}.aggregate",
        questions=FROZEN_QUESTION_COUNT,
        correct=correct,
    )
    _validate_fixed_accuracy_summary(
        report["target_gate"],
        label=f"{label}.target_gate",
        questions=FROZEN_QUESTION_COUNT,
        correct=correct,
        gate=True,
    )
    return _ValidatedFixedStageScore(
        report=report,
        rows=rows,
        canonical_sha256=canonical_sha256(report),
        judge_accuracy=correct / FROZEN_QUESTION_COUNT,
        correct=correct,
    )


def _compare_fixed_stage_score(
    treatment: _ValidatedFixedStageScore,
    mem0: _ValidatedArm,
    *,
    treatment_prompt_derivation_verified: bool,
) -> dict[str, Any]:
    """Compare hash-only treatment judgments without overstating fairness."""

    report = treatment.report
    mem0_report = mem0.report
    mem0_config = _mapping(mem0_report["config"], "mem0.config")
    mem0_models = _mapping(mem0_report["model_identity"], "mem0.model_identity")
    exact_shared = {
        "responder_model": FIXED_STAGE_TERRA_MODEL,
        "judge_model": FIXED_STAGE_SOL_MODEL,
        "max_prompt_tokens": FIXED_STAGE_PROMPT_CAP,
        "responder_output_token_reserve": (
            FIXED_STAGE_RESPONDER_MAX_OUTPUT_TOKENS
        ),
    }
    for field, expected in exact_shared.items():
        _must_equal(mem0_report[field], expected, f"paired fixed-stage {field}")
    _must_equal(
        mem0_config["responder_max_output_tokens"],
        FIXED_STAGE_RESPONDER_MAX_OUTPUT_TOKENS,
        "paired fixed-stage responder_max_output_tokens",
    )
    _must_equal(
        mem0_config["judge_max_output_tokens"],
        FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        "paired fixed-stage judge_max_output_tokens",
    )
    _must_equal(
        mem0_config["authorized_local_wrapper_retries"],
        0,
        "paired fixed-stage provider retries",
    )
    _must_equal(
        mem0_models["responder_model"],
        FIXED_STAGE_TERRA_MODEL,
        "paired fixed-stage responder model identity alias",
    )
    _must_equal(
        mem0_models["judge_model"],
        FIXED_STAGE_SOL_MODEL,
        "paired fixed-stage judge model identity alias",
    )
    try:
        mem0_prompt_policy = build_responder_prompt_policy_identity(
            [row["messages"] for row in mem0.rows]
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PairedComparisonError(
            f"Mem0 responder prompt policy cannot be proven: {exc}"
        ) from exc
    mem0_prompt_policy_sha256 = canonical_sha256(mem0_prompt_policy)
    _must_equal(
        mem0_prompt_policy,
        report["responder_prompt_policy"],
        "paired fixed-stage responder prompt policy object",
    )
    _must_equal(
        mem0_prompt_policy_sha256,
        report["responder_prompt_policy_sha256"],
        "paired fixed-stage responder prompt policy identity",
    )
    treatment_ids = tuple(str(row["question_id"]) for row in treatment.rows)
    mem0_ids = tuple(str(row["question_id"]) for row in mem0.rows)
    if set(treatment_ids) != set(mem0_ids):
        missing = sorted(set(treatment_ids) - set(mem0_ids))
        extra = sorted(set(mem0_ids) - set(treatment_ids))
        raise PairedComparisonError(
            "campaign question populations differ: "
            f"missing_from_mem0={missing!r}, extra_in_mem0={extra!r}"
        )
    paired_rows: list[dict[str, Any]] = []
    matched_gold_population: list[dict[str, Any]] = []
    wins = ties = losses = 0
    mem0_by_id = {str(row["question_id"]): row for row in mem0.rows}
    for treatment_row in treatment.rows:
        question_id = str(treatment_row["question_id"])
        mem0_row = mem0_by_id[question_id]
        semantic_hashes = {
            "question_sha256": _text_sha256(str(mem0_row["question"])),
            "dated_question_sha256": _text_sha256(
                str(mem0_row["dated_question"])
            ),
            "gold_answer_sha256": _text_sha256(str(mem0_row["gold_answer"])),
        }
        for field, expected in semantic_hashes.items():
            _must_equal(
                treatment_row[field],
                expected,
                f"paired question {question_id} {field}",
            )
        _must_equal(
            treatment_row["category"],
            mem0_row["category"],
            f"paired question {question_id} category",
        )
        treatment_correct = bool(treatment_row["correct"])
        mem0_correct = bool(mem0_row["judge_correct"])
        if treatment_correct and not mem0_correct:
            outcome = "treatment_win"
            wins += 1
        elif mem0_correct and not treatment_correct:
            outcome = "treatment_loss"
            losses += 1
        else:
            outcome = "tie"
            ties += 1
        matched_gold_population.append(
            {
                "ordinal": treatment_row["ordinal"],
                "question_id": question_id,
                **semantic_hashes,
            }
        )
        paired_rows.append(
            {
                "ordinal": treatment_row["ordinal"],
                "question_id": question_id,
                "category": treatment_row["category"],
                **semantic_hashes,
                "outcome": outcome,
                "treatment": {
                    "prediction_sha256": treatment_row[
                        "prediction_sha256"
                    ],
                    "judge_correct": treatment_correct,
                    "answer_call_key_sha256": treatment_row[
                        "answer_call_key_sha256"
                    ],
                    "answer_response_journal_sha256": treatment_row[
                        "answer_response_journal_sha256"
                    ],
                    "judge_call_key_sha256": treatment_row[
                        "call_key_sha256"
                    ],
                    "judge_response_journal_sha256": treatment_row[
                        "response_journal_sha256"
                    ],
                },
                "mem0": {
                    "prediction_sha256": _text_sha256(
                        str(mem0_row["prediction"])
                    ),
                    "judge_correct": mem0_correct,
                    "retrieval_row_sha256": mem0_row[
                        "retrieval_row_sha256"
                    ],
                    "messages_sha256": mem0_row["messages_sha256"],
                },
                "treatment_minus_mem0": {
                    "judge_correct": int(treatment_correct) - int(mem0_correct)
                },
            }
        )
    question_ids_sha256 = canonical_sha256(sorted(treatment_ids))
    mem0_population = _mapping(
        mem0_report["population_identity"], "mem0.population_identity"
    )
    _must_equal(
        mem0_population["question_ids_sha256"],
        question_ids_sha256,
        "paired question population hash",
    )
    matched_gold_population_sha256 = canonical_sha256(matched_gold_population)
    _must_equal(
        matched_gold_population_sha256,
        report["gold_scoring_population_sha256"],
        "paired gold scoring population hash",
    )

    shared_scoring_configuration = {
        "responder": {
            "model": FIXED_STAGE_TERRA_MODEL,
            "max_prompt_tokens": FIXED_STAGE_PROMPT_CAP,
            "max_output_tokens": FIXED_STAGE_RESPONDER_MAX_OUTPUT_TOKENS,
        },
        "judge": {
            "model": FIXED_STAGE_SOL_MODEL,
            "max_output_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        },
        "provider_retries": 0,
        "treatment_sampling_parameters": "omitted (temperature=null)",
        "mem0_sampling_parameters": "not_published_by_schema_v2",
    }
    required_schema_updates = {
        "mem0_campaign": {
            "minimum_schema_version": 3,
            "required_fields": {
                "population_identity_sha256": (
                    "must be transitively bound to the Mem0 retrieval corpus "
                    "and equal treatment.population_identity_sha256"
                ),
                "scoring_request_policy.sampling_parameters": (
                    "must prove responder and judge sampling parameters were "
                    "omitted, not merely state provider_retries=0"
                ),
                "scoring_request_policy.provider_retry_attempts": (
                    "must prove zero provider retry attempts for both responder "
                    "and judge; schema-v2 only limits local wrapper retries"
                ),
            },
        },
        "treatment_semantic_score": {
            "required_fields": {
                "responder_model_identity_sha256": (
                    "must identify the deployment independently of the "
                    "treatment-specific runtime/campaign hash"
                ),
                "judge_model_identity_sha256": (
                    "must identify the deployment independently of the "
                    "treatment-specific judge-runtime hash"
                ),
            }
        },
        "paired_comparison_inputs": {
            "required_for_treatment_prompt_derivation": [
                "final_answer_artifact",
                "retrieval",
            ]
        },
    }
    blockers = [
        "paired_source_population_identity_unverified",
        "shared_sampling_policy_identity_unverified",
        "shared_zero_retry_policy_identity_unverified",
        "shared_model_deployment_identity_unverified",
    ]
    if not treatment_prompt_derivation_verified:
        blockers.append(
            "treatment_final_answer_artifact_derivation_unverified"
        )
    mem0_integrity = bool(
        mem0_report["locked_population_verified"]
        and mem0_report["local_comparison_protocol_verified"]
    )
    if not mem0_integrity:
        blockers.append("mem0_locked_comparison_protocol_not_verified")
    if not bool(mem0_report["production_binding_certified"]):
        blockers.append("mem0_production_binding_certified_false")

    mem0_accuracy = float(mem0.metrics["judge_accuracy"])
    result = {
        "schema_version": FIXED_STAGE_COMPARISON_SCHEMA_VERSION,
        "report_type": COMPARISON_REPORT_TYPE,
        "comparison_scope": (
            "matched_100q_hash_bound_binary_judge_metric_only_v3"
        ),
        "input_hashes": {
            "treatment_semantic_score_canonical_sha256": (
                treatment.canonical_sha256
            ),
            "treatment_final_answer_artifact_sha256": report[
                "final_answer_artifact_sha256"
            ],
            "treatment_responder_runtime_identity_sha256": report[
                "responder_runtime_identity_sha256"
            ],
            "treatment_responder_prompt_policy_sha256": report[
                "responder_prompt_policy_sha256"
            ],
            "treatment_retrieval_sha256": report["retrieval_sha256"],
            "treatment_population_identity_sha256": report[
                "population_identity_sha256"
            ],
            "treatment_judge_runtime_identity_sha256": report[
                "judge_runtime_identity_sha256"
            ],
            "treatment_judge_campaign_binding_sha256": report[
                "campaign_binding_sha256"
            ],
            "mem0_campaign_canonical_sha256": mem0.canonical_sha256,
            "mem0_campaign_input_set_sha256": mem0_report[
                "input_set_sha256"
            ],
            "question_ids_sha256": question_ids_sha256,
            "matched_gold_scoring_population_sha256": (
                matched_gold_population_sha256
            ),
        },
        "shared_identity": {
            "mem0_dataset_sha256": mem0_report["dataset_sha256"],
            "mem0_split_manifest_sha256": mem0_report[
                "split_manifest_sha256"
            ],
            "mem0_benchmark_split": "validation",
            "fixed_stage_id": FIXED_STAGE_ID,
            "scoring_configuration": shared_scoring_configuration,
            "scoring_configuration_sha256": canonical_sha256(
                shared_scoring_configuration
            ),
            "identity_hash_bindings": {
                "treatment_responder_runtime_identity_sha256": report[
                    "responder_runtime_identity_sha256"
                ],
                "treatment_responder_prompt_policy_sha256": report[
                    "responder_prompt_policy_sha256"
                ],
                "mem0_responder_prompt_policy_sha256": (
                    mem0_prompt_policy_sha256
                ),
                "treatment_judge_runtime_identity_sha256": report[
                    "judge_runtime_identity_sha256"
                ],
                "mem0_responder_model_identity_sha256": mem0_models[
                    "responder_model_identity_sha256"
                ],
                "mem0_judge_model_identity_sha256": mem0_models[
                    "judge_model_identity_sha256"
                ],
                "cross_arm_model_deployment_identity_verified": False,
            },
            "responder_prompt_policy_identity_object_equal": True,
            "responder_prompt_policy_derivation_verified": (
                treatment_prompt_derivation_verified
            ),
            "responder_prompt_policy_identity_verified": (
                treatment_prompt_derivation_verified
            ),
            "judge_prompt_derivation_verified": (
                treatment_prompt_derivation_verified
            ),
            "sampling_policy_identity_verified": False,
            "zero_retry_policy_identity_verified": False,
        },
        "paired_population_identity": {
            "question_count": FROZEN_QUESTION_COUNT,
            "question_ids_sha256": question_ids_sha256,
            "gold_scoring_population_sha256": (
                matched_gold_population_sha256
            ),
            "question_and_gold_identity_verified": True,
            "treatment_source_population_identity_sha256": report[
                "population_identity_sha256"
            ],
            "mem0_source_population_identity_sha256": None,
            "same_source_population_certified": False,
        },
        "metric_comparison": {
            "valid": True,
            "status": (
                "paired_binary_judge_metrics_recomputed_from_hash_bound_rows"
            ),
            "num_questions": FROZEN_QUESTION_COUNT,
            "supported_metrics": ["binary_judge_accuracy"],
            "unsupported_metrics": [
                "f1",
                "exact_match",
                "context_tokens",
                "prompt_tokens",
            ],
        },
        "certification": {
            "certified": False,
            "status": "metric_only_noncertified",
            "blocking_reasons": blockers,
            "required_schema_updates": required_schema_updates,
            "treatment_semantic_score_internal_structure_verified": True,
            "treatment_semantic_score_internal_contract_verified": (
                treatment_prompt_derivation_verified
            ),
            "treatment_semantic_score_prompt_accounting_verified": (
                treatment_prompt_derivation_verified
            ),
            "treatment_semantic_score_contract_verified": (
                treatment_prompt_derivation_verified
            ),
            "mem0_local_comparison_protocol_verified": mem0_integrity,
            "mem0_production_binding_certified": bool(
                mem0_report["production_binding_certified"]
            ),
        },
        "arm_metrics": {
            "treatment": {
                "num_questions": FROZEN_QUESTION_COUNT,
                "correct": treatment.correct,
                "judge_accuracy": treatment.judge_accuracy,
            },
            "mem0": {
                "num_questions": FROZEN_QUESTION_COUNT,
                "correct": sum(
                    int(bool(row["judge_correct"])) for row in mem0.rows
                ),
                "judge_accuracy": mem0_accuracy,
            },
        },
        "paired_judge_outcomes": {
            "treatment_wins": wins,
            "ties": ties,
            "treatment_losses": losses,
        },
        "treatment_minus_mem0": {
            "judge_accuracy": treatment.judge_accuracy - mem0_accuracy
        },
        "provenance_comparison": {
            "comparable": False,
            "status": "not_applicable_to_mem0",
            "treatment": {
                "status": (
                    "answer_and_judge_journal_hashes_bound"
                    if treatment_prompt_derivation_verified
                    else "judge_hashes_internally_bound_answer_derivation_unverified"
                ),
                "value": None,
            },
            "mem0": {
                "status": "not_applicable",
                "value": None,
                "supports_exact_source_provenance": False,
                "attribution_kind": MEM0_ATTRIBUTION_KIND,
                "reason": MEM0_SOURCE_COVERAGE_REASON,
            },
        },
        "question_results": paired_rows,
    }
    _walk_json(result, "fixed-stage paired comparison result")
    return result


def compare_fixed_stage_score(
    treatment_campaign: Mapping[str, Any],
    mem0: _ValidatedArm,
    *,
    final_answer_artifact: Mapping[str, Any] | None = None,
    retrieval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and compare one schema-v3 semantic score and Mem0 arm."""

    treatment = _validate_fixed_stage_score(treatment_campaign)
    derivation_verified = verify_treatment_prompt_derivation(
        treatment.report,
        final_answer_artifact=final_answer_artifact,
        retrieval=retrieval,
        scoring_rows=mem0.rows,
    )
    return _compare_fixed_stage_score(
        treatment,
        mem0,
        treatment_prompt_derivation_verified=derivation_verified,
    )


__all__ = ["compare_fixed_stage_score"]
