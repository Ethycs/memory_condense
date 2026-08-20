"""Pure campaign merging, metric reduction, and distribution summaries."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from memory_condense.eval.cache_receipts import (
    cache_receipts_sha256,
    validated_cache_receipts,
)
from memory_condense.eval.campaign_models import (
    CampaignMergeError,
    ExpectedStressShard,
    LockedValidationPlan,
)
from memory_condense.eval.campaign_plan import (
    _assert_locked_plan_unchanged,
    _revalidate_locked_claim_profile,
)
from memory_condense.eval.campaign_validation import (
    _HASH_FIELDS,
    _assert_policy_retrieval_identity,
    _canonical_json,
    _ensure_same_identity,
    _identity,
    _load_report,
    _locked_judge_verdict,
    _require_bool,
    _require_float,
    _require_int,
    _require_list,
    _require_mapping,
    _require_nonempty_string,
    _require_sha256,
    _validate_question,
)

def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def _distribution(values: Iterable[int]) -> dict[str, int | float | list[int]]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "values": [],
        }
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": math.fsum(float(value) for value in ordered) / len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
        "values": ordered,
    }


def _sum_usage(rows: Iterable[dict[str, int | float]]) -> dict[str, int | float]:
    items = list(rows)
    return {
        "input_tokens": sum(int(row["input_tokens"]) for row in items),
        "output_tokens": sum(int(row["output_tokens"]) for row in items),
        "cache_read_input_tokens": sum(
            int(row["cache_read_input_tokens"]) for row in items
        ),
        "elapsed_s": math.fsum(float(row["elapsed_s"]) for row in items),
        "calls": sum(int(row["calls"]) for row in items),
    }


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return math.fsum(materialized) / len(materialized) if materialized else 0.0


def _category_metrics(questions: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        raw_category = question.get("category")
        category = (
            raw_category
            if isinstance(raw_category, str) and raw_category.strip()
            else "uncategorized"
        )
        grouped[category].append(question)
    return {
        category: {
            "category": category,
            "num_questions": len(rows),
            "mean_f1": _mean(float(row["f1"]) for row in rows),
            "exact_match_rate": _mean(
                1.0 if row["exact_match"] else 0.0 for row in rows
            ),
            "judge_accuracy": _mean(
                1.0 if row["judge_correct"] else 0.0 for row in rows
            ),
        }
        for category, rows in sorted(grouped.items())
    }


@dataclass
class _MergeAccumulator:
    """State accumulated across shard reports during a campaign merge.

    Every cross-report invariant (identity pinning, question deduplication,
    cache-entry reuse, frozen-offset coverage) lives here so each validation
    stage can be exercised in isolation.
    """

    expected_identity: dict[str, Any] | None = None
    input_rows: list[dict[str, Any]] = field(default_factory=list)
    questions_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    question_sources: dict[str, dict[str, str]] = field(default_factory=dict)
    responder_usage_by_question: dict[str, dict[str, int | float]] = field(
        default_factory=dict
    )
    judge_usage_by_question: dict[str, dict[str, int | float]] = field(
        default_factory=dict
    )
    cache_receipts_by_sample: dict[str, dict[str, list[dict[str, object]]]] = field(
        default_factory=dict
    )
    observed_compiled_cache_keys: set[str] = field(default_factory=set)
    observed_causal_cache_keys: set[str] = field(default_factory=set)
    sample_count: int = 0
    observed_offsets: set[int] = field(default_factory=set)


def _validated_campaign_policy(
    min_questions: int,
    accuracy_target: float,
    locked_plan: LockedValidationPlan | None,
) -> tuple[float, bool]:
    """Validate merge arguments against the frozen validation policy."""

    if (
        isinstance(min_questions, bool)
        or not isinstance(min_questions, int)
        or min_questions < 1
    ):
        raise CampaignMergeError("min_questions must be an integer >= 1")
    accuracy_target = _require_float(
        accuracy_target, "accuracy_target", minimum=0.0, maximum=1.0
    )
    claim_profile_verified = False
    if locked_plan is not None:
        if locked_plan.evaluation.get("min_target_questions") != min_questions:
            raise CampaignMergeError(
                "campaign min_questions does not match the frozen validation policy"
            )
        if locked_plan.evaluation.get("accuracy_target") != accuracy_target:
            raise CampaignMergeError(
                "campaign accuracy_target does not match the frozen validation policy"
            )
        claim_profile_verified = _revalidate_locked_claim_profile(locked_plan)
    return accuracy_target, claim_profile_verified


def _assert_identity_matches_plan(
    identity: dict[str, Any],
    label: str,
    locked_plan: LockedValidationPlan,
) -> None:
    """Check the first report's identity against the frozen campaign plan."""

    for field_name in _HASH_FIELDS:
        expected = getattr(locked_plan, field_name)
        if identity[field_name] != expected:
            raise CampaignMergeError(
                f"{label}.{field_name} does not match the independently "
                "verified campaign plan"
            )
    if identity["evaluation_protocol"] != locked_plan.evaluation:
        raise CampaignMergeError(
            f"{label}.evaluation_protocol does not match the frozen "
            "validation policy"
        )
    if _canonical_json(
        identity["prompt_token_proxy_identity"]
    ) != _canonical_json(
        locked_plan.evaluation.get("prompt_token_proxy_identity")
    ):
        raise CampaignMergeError(
            f"{label}.prompt_token_proxy_identity disagrees with "
            "the frozen evaluation protocol"
        )
    if identity["responder_output_token_reserve"] != (
        locked_plan.evaluation.get("responder_output_token_reserve")
    ):
        raise CampaignMergeError(
            f"{label}.responder_output_token_reserve disagrees "
            "with the frozen evaluation protocol"
        )
    for identity_field, evaluation_field in (
        ("responder_model", "responder_model"),
        ("judge_model", "judge_model"),
        ("embedding_device", "embedding_device"),
        ("max_prompt_tokens", "max_prompt_tokens"),
        ("recent_window", "recent_window"),
    ):
        if identity[identity_field] != locked_plan.evaluation.get(
            evaluation_field
        ):
            raise CampaignMergeError(
                f"{label}.{identity_field} disagrees with the frozen "
                "evaluation protocol"
            )
    if identity["benchmark"] != locked_plan.dataset_path.stem:
        raise CampaignMergeError(
            f"{label}.benchmark does not identify the locked dataset"
        )
    _assert_policy_retrieval_identity(identity, locked_plan, label)


def _record_report_identity(
    accumulator: _MergeAccumulator,
    report: dict[str, Any],
    label: str,
    locked_plan: LockedValidationPlan | None,
) -> dict[str, Any]:
    """Pin the campaign identity on the first report; compare the rest."""

    identity = _identity(report, label)
    if accumulator.expected_identity is None:
        accumulator.expected_identity = identity
        if locked_plan is not None:
            _assert_identity_matches_plan(identity, label, locked_plan)
    else:
        _ensure_same_identity(accumulator.expected_identity, identity, label)
    return identity


def _validate_shard_policy(
    report: dict[str, Any],
    label: str,
    *,
    min_questions: int,
    accuracy_target: float,
) -> str:
    """Validate one shard's evaluation policy; return its target status."""

    config = _require_mapping(report.get("config"), f"{label}.config")
    config_accuracy_target = _require_float(
        config.get("accuracy_target"),
        f"{label}.config.accuracy_target",
        minimum=0.0,
        maximum=1.0,
    )
    report_accuracy_target = _require_float(
        report.get("accuracy_target"),
        f"{label}.accuracy_target",
        minimum=0.0,
        maximum=1.0,
    )
    if (
        config_accuracy_target != accuracy_target
        or report_accuracy_target != accuracy_target
    ):
        raise CampaignMergeError(
            f"{label} accuracy_target drift: config="
            f"{config_accuracy_target}, report={report_accuracy_target}, "
            f"campaign={accuracy_target}"
        )
    config_min_questions = _require_int(
        config.get("min_target_questions"),
        f"{label}.config.min_target_questions",
        minimum=1,
    )
    report_min_questions = _require_int(
        report.get("min_target_questions"),
        f"{label}.min_target_questions",
        minimum=1,
    )
    if (
        config_min_questions != min_questions
        or report_min_questions != min_questions
    ):
        raise CampaignMergeError(
            f"{label} min_target_questions drift: config="
            f"{config_min_questions}, report={report_min_questions}, "
            f"campaign={min_questions}"
        )

    if report.get("prompt_budget_compliance") is not True:
        raise CampaignMergeError(
            f"{label}.prompt_budget_compliance must be true"
        )
    shard_target_status = _require_nonempty_string(
        report.get("target_status"), f"{label}.target_status"
    )
    if shard_target_status not in {
        "insufficient_questions",
        "passed",
        "failed",
    }:
        raise CampaignMergeError(
            f"{label}.target_status is not a completed graded status: "
            f"{shard_target_status!r}"
        )
    return shard_target_status


def _validated_samples(report: dict[str, Any], label: str) -> list[Any]:
    """Validate one shard's sample list shape; return the sample rows."""

    samples = _require_list(report.get("samples"), f"{label}.samples")
    declared_samples = _require_int(
        report.get("num_samples"), f"{label}.num_samples"
    )
    if declared_samples != len(samples):
        raise CampaignMergeError(
            f"{label}.num_samples={declared_samples} but contains "
            f"{len(samples)} sample rows"
        )
    return samples


def _resolve_expected_shard(
    accumulator: _MergeAccumulator,
    report: dict[str, Any],
    label: str,
    num_samples: int,
    locked_plan: LockedValidationPlan | None,
) -> ExpectedStressShard | None:
    """Match a locked report to its frozen shard by sample offset."""

    if locked_plan is None:
        return None
    protocol = _require_mapping(
        report.get("evaluation_protocol"),
        f"{label}.evaluation_protocol",
    )
    sample_offset = _require_int(
        protocol.get("sample_offset"),
        f"{label}.evaluation_protocol.sample_offset",
    )
    if sample_offset in accumulator.observed_offsets:
        raise CampaignMergeError(
            f"duplicate validation sample_offset {sample_offset}"
        )
    try:
        expected_shard = locked_plan.shards[sample_offset]
    except KeyError as exc:
        raise CampaignMergeError(
            f"validation sample_offset {sample_offset} is not in the "
            "frozen campaign plan"
        ) from exc
    accumulator.observed_offsets.add(sample_offset)
    if num_samples != 1:
        raise CampaignMergeError(
            f"{label} must contain exactly one reconstructed stress sample"
        )
    return expected_shard


def _validate_locked_sample(
    accumulator: _MergeAccumulator,
    sample: dict[str, Any],
    sample_label: str,
    sample_id: str,
    expected_shard: ExpectedStressShard,
    locked_plan: LockedValidationPlan,
) -> None:
    """Check a locked sample against its reconstructed frozen shard."""

    if sample_id != expected_shard.sample_id:
        raise CampaignMergeError(
            f"{sample_label}.sample_id does not match reconstructed shard"
        )
    reported_sample_sha256 = _require_sha256(
        sample.get("sample_sha256"),
        f"{sample_label}.sample_sha256",
    )
    if reported_sample_sha256 != expected_shard.sample_sha256:
        raise CampaignMergeError(
            f"{sample_label}.sample_sha256 does not match the exact "
            "reconstructed stress sample"
        )
    reported_turns = _require_int(
        sample.get("num_turns"), f"{sample_label}.num_turns"
    )
    if reported_turns != expected_shard.num_turns:
        raise CampaignMergeError(
            f"{sample_label}.num_turns does not match reconstructed shard"
        )
    try:
        sample_cache_receipts = validated_cache_receipts(
            sample.get("cache_receipts"),
            expected_sample_sha256=expected_shard.sample_sha256,
            expected_implementation_sha256=locked_plan.implementation_sha256,
            expected_environment_lock_sha256=locked_plan.environment_lock_sha256,
        )
    except ValueError as exc:
        raise CampaignMergeError(
            f"{sample_label}.cache_receipts: {exc}"
        ) from exc
    if (
        sample_cache_receipts["compiled"][0]["turn_count"]
        != expected_shard.num_turns
    ):
        raise CampaignMergeError(
            f"{sample_label}.cache_receipts compiled turn_count does "
            "not match the reconstructed stress sample"
        )
    reported_receipt_sha256 = _require_sha256(
        sample.get("cache_receipts_sha256"),
        f"{sample_label}.cache_receipts_sha256",
    )
    actual_receipt_sha256 = cache_receipts_sha256(sample_cache_receipts)
    if reported_receipt_sha256 != actual_receipt_sha256:
        raise CampaignMergeError(
            f"{sample_label}.cache_receipts_sha256 does not match "
            "the exact cache receipt pair"
        )
    compiled_key = str(sample_cache_receipts["compiled"][0]["cache_key"])
    causal_key = str(sample_cache_receipts["causal"][0]["cache_key"])
    if compiled_key in accumulator.observed_compiled_cache_keys:
        raise CampaignMergeError(
            "locked validation shards reuse a compiled cache entry"
        )
    if causal_key in accumulator.observed_causal_cache_keys:
        raise CampaignMergeError(
            "locked validation shards reuse a causal cache entry"
        )
    accumulator.observed_compiled_cache_keys.add(compiled_key)
    accumulator.observed_causal_cache_keys.add(causal_key)
    accumulator.cache_receipts_by_sample[
        expected_shard.sample_sha256
    ] = sample_cache_receipts


def _validated_question_rows(
    sample: dict[str, Any],
    sample_label: str,
    expected_shard: ExpectedStressShard | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Validate a sample's question-row shape and locked ID coverage."""

    rows = _require_list(
        sample.get("question_results"),
        f"{sample_label}.question_results",
    )
    declared_questions = _require_int(
        sample.get("num_questions"), f"{sample_label}.num_questions"
    )
    if declared_questions != len(rows):
        raise CampaignMergeError(
            f"{sample_label}.num_questions={declared_questions} but "
            f"contains {len(rows)} question rows"
        )
    expected_questions_by_id = (
        {
            str(question["question_id"]): question
            for question in expected_shard.questions
        }
        if expected_shard is not None
        else {}
    )
    if expected_shard is not None:
        reported_question_ids: set[str] = set()
        for question_index, raw_question in enumerate(rows):
            raw_mapping = _require_mapping(
                raw_question,
                f"{sample_label}.question_results[{question_index}]",
            )
            reported_question_ids.add(
                _require_nonempty_string(
                    raw_mapping.get("question_id"),
                    f"{sample_label}.question_results[{question_index}]"
                    ".question_id",
                )
            )
        if reported_question_ids != set(expected_questions_by_id):
            raise CampaignMergeError(
                f"{sample_label} question IDs do not match reconstructed shard"
            )
    return rows, expected_questions_by_id


def _ingest_question(
    accumulator: _MergeAccumulator,
    raw_question: Any,
    question_label: str,
    *,
    identity: dict[str, Any],
    locked_plan: LockedValidationPlan | None,
    expected_shard: ExpectedStressShard | None,
    expected_questions_by_id: dict[str, Any],
    path_label: str,
    portable_name: str,
    report_sha256: str,
    sample_id: str,
    sample_sha256: str,
) -> dict[str, Any]:
    """Validate one question row and record it in the accumulator."""

    question, responder_usage, judge_usage = _validate_question(
        raw_question,
        question_label,
        prompt_cap=int(identity["max_prompt_tokens"]),
        output_token_reserve=int(
            identity["responder_output_token_reserve"]
        ),
        require_proxy_fields=locked_plan is not None,
    )
    if locked_plan is not None:
        if responder_usage["calls"] != 1 or judge_usage["calls"] != 1:
            raise CampaignMergeError(
                f"{question_label} must contain exactly one completed "
                "responder call and one completed judge call"
            )
        parsed_verdict = _locked_judge_verdict(
            question.get("judge_reasoning"),
            f"{question_label}.judge_reasoning",
        )
        if parsed_verdict != question["judge_correct"]:
            raise CampaignMergeError(
                f"{question_label}.judge_correct disagrees with the "
                "provider verdict"
            )
        _require_nonempty_string(
            question.get("predicted_answer"),
            f"{question_label}.predicted_answer",
        )
    question_id = str(question["question_id"])
    if expected_shard is not None:
        expected_question = expected_questions_by_id[question_id]
        for field_name in ("question", "gold_answer", "category"):
            if question.get(field_name) != expected_question[field_name]:
                raise CampaignMergeError(
                    f"{question_label}.{field_name} does not match the "
                    "locked validation dataset"
                )
        question_transcript_tokens = _require_int(
            question.get("transcript_tokens"),
            f"{question_label}.transcript_tokens",
        )
        if question_transcript_tokens != expected_shard.transcript_tokens:
            raise CampaignMergeError(
                f"{question_label}.transcript_tokens does not match "
                "the reconstructed stress sample"
            )
    if question_id in accumulator.questions_by_id:
        prior = accumulator.question_sources[question_id]
        raise CampaignMergeError(
            f"duplicate question_id {question_id!r} in {path_label}; "
            f"already present in {prior['report_name']}"
        )
    accumulator.questions_by_id[question_id] = question
    accumulator.question_sources[question_id] = {
        "report_name": portable_name,
        "report_sha256": report_sha256,
        "sample_id": sample_id,
        "sample_sha256": sample_sha256,
    }
    accumulator.responder_usage_by_question[question_id] = responder_usage
    accumulator.judge_usage_by_question[question_id] = judge_usage
    return question


def _validate_shard_totals(
    report: dict[str, Any],
    label: str,
    *,
    question_count: int,
    prompt_counts: Sequence[int],
    provider_compliances: Sequence[bool | None],
    locked_plan: LockedValidationPlan | None,
) -> None:
    """Check shard-level totals against the recomputed question rows."""

    declared_report_questions = _require_int(
        report.get("num_questions"), f"{label}.num_questions"
    )
    if declared_report_questions != question_count:
        raise CampaignMergeError(
            f"{label}.num_questions={declared_report_questions} but "
            f"contains {question_count} question rows"
        )
    observed_max = _require_int(
        report.get("max_prompt_tokens_observed"),
        f"{label}.max_prompt_tokens_observed",
    )
    recomputed_max = max(prompt_counts, default=0)
    if observed_max != recomputed_max:
        raise CampaignMergeError(
            f"{label}.max_prompt_tokens_observed={observed_max} but "
            f"the question rows have maximum {recomputed_max}"
        )
    raw_proxy_max = report.get("max_prompt_token_proxy_observed")
    if raw_proxy_max is None:
        if locked_plan is not None:
            raise CampaignMergeError(
                f"{label}.max_prompt_token_proxy_observed is required"
            )
    elif _require_int(
        raw_proxy_max,
        f"{label}.max_prompt_token_proxy_observed",
    ) != recomputed_max:
        raise CampaignMergeError(
            f"{label}.max_prompt_token_proxy_observed does not match "
            "the question rows"
        )
    raw_proxy_compliance = report.get(
        "prompt_token_proxy_budget_compliance"
    )
    if raw_proxy_compliance is None:
        if locked_plan is not None:
            raise CampaignMergeError(
                f"{label}.prompt_token_proxy_budget_compliance is required"
            )
    elif _require_bool(
        raw_proxy_compliance,
        f"{label}.prompt_token_proxy_budget_compliance",
    ) is not True:
        raise CampaignMergeError(
            f"{label}.prompt_token_proxy_budget_compliance must be true"
        )
    available_provider_rows = [
        value
        for value in provider_compliances
        if value is not None
    ]
    expected_provider_compliance = (
        all(available_provider_rows) if available_provider_rows else None
    )
    expected_provider_status = (
        "unavailable"
        if not available_provider_rows
        else "complete"
        if len(available_provider_rows) == len(provider_compliances)
        else "partial"
    )
    if locked_plan is not None or (
        "provider_prompt_budget_compliance" in report
    ):
        if (
            report.get("provider_prompt_budget_compliance")
            != expected_provider_compliance
        ):
            raise CampaignMergeError(
                f"{label}.provider_prompt_budget_compliance disagrees with "
                "per-question provider usage"
            )
    if locked_plan is not None or "provider_input_usage_status" in report:
        if (
            report.get("provider_input_usage_status")
            != expected_provider_status
        ):
            raise CampaignMergeError(
                f"{label}.provider_input_usage_status disagrees with "
                "per-question provider usage availability"
            )


def _finalize_population(
    accumulator: _MergeAccumulator,
    locked_plan: LockedValidationPlan | None,
    min_questions: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Certify the merged question population against the frozen plan."""

    question_ids = sorted(accumulator.questions_by_id)
    questions = [
        accumulator.questions_by_id[question_id] for question_id in question_ids
    ]
    num_questions = len(questions)
    if locked_plan is not None:
        if accumulator.observed_offsets != set(locked_plan.sample_offsets):
            missing_offsets = (
                set(locked_plan.sample_offsets) - accumulator.observed_offsets
            )
            raise CampaignMergeError(
                "campaign is missing frozen validation shards at offsets: "
                + ", ".join(str(value) for value in sorted(missing_offsets))
            )
        if set(question_ids) != set(locked_plan.question_ids):
            raise CampaignMergeError(
                "campaign question IDs do not equal the locked validation population"
            )
    if num_questions < min_questions:
        raise CampaignMergeError(
            f"campaign has {num_questions} unique questions; "
            f"at least {min_questions} are required"
        )
    return question_ids, questions


def _campaign_summary(
    accumulator: _MergeAccumulator,
    questions: Sequence[dict[str, Any]],
    question_ids: Sequence[str],
    *,
    accuracy_target: float,
    locked_plan: LockedValidationPlan | None,
    claim_profile_verified: bool,
) -> dict[str, Any]:
    """Recompute campaign metrics from the flattened question rows."""

    expected_identity = accumulator.expected_identity
    assert expected_identity is not None
    num_questions = len(questions)
    prompt_proxy_distribution = _distribution(
        int(question["prompt_token_proxy"]) for question in questions
    )
    provider_input_counts = [
        int(accumulator.responder_usage_by_question[question_id]["input_tokens"])
        for question_id in question_ids
        if int(
            accumulator.responder_usage_by_question[question_id]["input_tokens"]
        )
        > 0
    ]
    metric_target_met = (
        _mean(1.0 if question["judge_correct"] else 0.0 for question in questions)
        >= accuracy_target
    )
    target_met = bool(
        locked_plan is not None
        and claim_profile_verified
        and metric_target_met
    )
    ordered_inputs = sorted(
        accumulator.input_rows, key=lambda row: (row["sha256"], row["name"])
    )
    return {
        "mean_f1": _mean(float(question["f1"]) for question in questions),
        "exact_match_rate": _mean(
            1.0 if question["exact_match"] else 0.0 for question in questions
        ),
        "judge_accuracy": _mean(
            1.0 if question["judge_correct"] else 0.0 for question in questions
        ),
        "context_distribution": _distribution(
            int(question["context_tokens"]) for question in questions
        ),
        "prompt_proxy_distribution": prompt_proxy_distribution,
        "request_proxy_distribution": _distribution(
            int(question["prompt_token_proxy"])
            + int(expected_identity["responder_output_token_reserve"])
            for question in questions
        ),
        "provider_input_distribution": _distribution(provider_input_counts),
        "provider_prompt_budget_compliance": (
            all(
                count <= int(expected_identity["max_prompt_tokens"])
                for count in provider_input_counts
            )
            if provider_input_counts
            else None
        ),
        "provider_input_usage_status": (
            "unavailable"
            if not provider_input_counts
            else "complete"
            if len(provider_input_counts) == num_questions
            else "partial"
        ),
        "transcript_distribution": _distribution(
            int(question.get("transcript_tokens", 0)) for question in questions
        ),
        "metric_target_met": metric_target_met,
        "target_met": target_met,
        "target_status": (
            "passed"
            if target_met
            else "unverified_claim_profile"
            if locked_plan is not None and not claim_profile_verified
            else "failed"
            if locked_plan is not None
            else "unverified_population"
        ),
        "ordered_inputs": ordered_inputs,
        "input_set_sha256": hashlib.sha256(
            _canonical_json(
                sorted(row["sha256"] for row in ordered_inputs)
            ).encode("utf-8")
        ).hexdigest(),
    }


def _campaign_report(
    accumulator: _MergeAccumulator,
    summary: dict[str, Any],
    questions: list[dict[str, Any]],
    question_ids: Sequence[str],
    *,
    accuracy_target: float,
    min_questions: int,
    locked_plan: LockedValidationPlan | None,
    claim_profile_verified: bool,
) -> dict[str, Any]:
    """Assemble the certified campaign report document."""

    expected_identity = accumulator.expected_identity
    assert expected_identity is not None
    prompt_proxy_distribution = summary["prompt_proxy_distribution"]
    context_distribution = summary["context_distribution"]
    request_proxy_distribution = summary["request_proxy_distribution"]
    ordered_inputs = summary["ordered_inputs"]
    return {
        "schema_version": 1,
        "report_type": "benchmark_campaign",
        "inputs": ordered_inputs,
        "input_count": len(ordered_inputs),
        "input_set_sha256": summary["input_set_sha256"],
        "benchmark": expected_identity["benchmark"],
        "dataset_sha256": expected_identity["dataset_sha256"],
        "split_manifest_sha256": expected_identity["split_manifest_sha256"],
        "benchmark_split": expected_identity["benchmark_split"],
        "implementation_sha256": expected_identity["implementation_sha256"],
        "environment_lock_sha256": expected_identity["environment_lock_sha256"],
        "policy_manifest_sha256": expected_identity["policy_manifest_sha256"],
        "chunker_config": expected_identity["chunker_config"],
        "retrieval_config": expected_identity["retrieval_config"],
        "responder_model": expected_identity["responder_model"],
        "judge_model": expected_identity["judge_model"],
        "embedding_device": expected_identity["embedding_device"],
        "recent_window": expected_identity["recent_window"],
        "max_prompt_tokens": expected_identity["max_prompt_tokens"],
        "prompt_token_proxy_identity": expected_identity[
            "prompt_token_proxy_identity"
        ],
        "responder_output_token_reserve": expected_identity[
            "responder_output_token_reserve"
        ],
        "evaluation_protocol": expected_identity["evaluation_protocol"],
        "claim_profile": locked_plan.claim_profile if locked_plan is not None else "",
        "claim_profile_verified": bool(
            locked_plan is not None and claim_profile_verified
        ),
        "cache_receipts_by_sample": {
            digest: accumulator.cache_receipts_by_sample[digest]
            for digest in sorted(accumulator.cache_receipts_by_sample)
        },
        "num_samples": accumulator.sample_count,
        "num_questions": len(questions),
        "question_results": questions,
        "question_sources": {
            question_id: accumulator.question_sources[question_id]
            for question_id in question_ids
        },
        "mean_f1": summary["mean_f1"],
        "exact_match_rate": summary["exact_match_rate"],
        "judge_accuracy": summary["judge_accuracy"],
        "mean_context_tokens": context_distribution["mean"],
        "mean_prompt_token_proxy": prompt_proxy_distribution["mean"],
        "p95_prompt_token_proxy": prompt_proxy_distribution["p95"],
        "max_prompt_token_proxy_observed": prompt_proxy_distribution["max"],
        "mean_request_token_proxy": request_proxy_distribution["mean"],
        # Compatibility aliases for historical report consumers.
        "mean_prompt_tokens": prompt_proxy_distribution["mean"],
        "p95_prompt_tokens": prompt_proxy_distribution["p95"],
        "max_prompt_tokens_observed": prompt_proxy_distribution["max"],
        "context_token_distribution": context_distribution,
        "prompt_token_proxy_distribution": prompt_proxy_distribution,
        "request_token_proxy_distribution": request_proxy_distribution,
        "provider_input_token_distribution": summary[
            "provider_input_distribution"
        ],
        "prompt_token_distribution": prompt_proxy_distribution,
        "transcript_token_distribution": summary["transcript_distribution"],
        "prompt_token_proxy_budget_compliance": True,
        "provider_prompt_budget_compliance": (
            summary["provider_prompt_budget_compliance"]
        ),
        "provider_input_usage_status": summary["provider_input_usage_status"],
        "prompt_budget_compliance": True,
        "responder_usage": _sum_usage(
            accumulator.responder_usage_by_question[question_id]
            for question_id in question_ids
        ),
        "judge_usage": _sum_usage(
            accumulator.judge_usage_by_question[question_id]
            for question_id in question_ids
        ),
        "by_category": _category_metrics(questions),
        "accuracy_target": accuracy_target,
        "min_target_questions": min_questions,
        "accuracy_target_met": summary["target_met"],
        "metric_accuracy_target_met": summary["metric_target_met"],
        "locked_population_verified": locked_plan is not None,
        "target_status": summary["target_status"],
    }


def merge_benchmark_reports(
    report_paths: Iterable[str | Path],
    *,
    min_questions: int = 100,
    accuracy_target: float = 0.95,
    locked_plan: LockedValidationPlan | None = None,
) -> dict[str, Any]:
    """Validate and merge locked validation shards.

    Metrics are recomputed from the flattened question rows.  Shard-level
    means and target decisions are never averaged or trusted.
    """

    accuracy_target, claim_profile_verified = _validated_campaign_policy(
        min_questions, accuracy_target, locked_plan
    )
    paths = [Path(path) for path in report_paths]
    if not paths:
        raise CampaignMergeError("at least one report is required")

    loaded = [_load_report(path) for path in paths]
    # Argument order and artifact location cannot change floating-point
    # reduction order or campaign identity.
    loaded.sort(key=lambda row: (row[1], row[3]))

    accumulator = _MergeAccumulator()
    for report, digest, path_label, portable_name in loaded:
        label = f"report[{path_label}]"
        identity = _record_report_identity(accumulator, report, label, locked_plan)
        shard_target_status = _validate_shard_policy(
            report,
            label,
            min_questions=min_questions,
            accuracy_target=accuracy_target,
        )
        samples = _validated_samples(report, label)
        expected_shard = _resolve_expected_shard(
            accumulator, report, label, len(samples), locked_plan
        )

        report_question_count = 0
        report_prompt_counts: list[int] = []
        report_provider_compliances: list[bool | None] = []
        for sample_index, raw_sample in enumerate(samples):
            sample_label = f"{label}.samples[{sample_index}]"
            sample = _require_mapping(raw_sample, sample_label)
            sample_id = _require_nonempty_string(
                sample.get("sample_id"), f"{sample_label}.sample_id"
            )
            if expected_shard is not None:
                assert locked_plan is not None
                _validate_locked_sample(
                    accumulator,
                    sample,
                    sample_label,
                    sample_id,
                    expected_shard,
                    locked_plan,
                )
            rows, expected_questions_by_id = _validated_question_rows(
                sample, sample_label, expected_shard
            )
            report_question_count += len(rows)
            accumulator.sample_count += 1
            for question_index, raw_question in enumerate(rows):
                question_label = (
                    f"{sample_label}.question_results[{question_index}]"
                )
                question = _ingest_question(
                    accumulator,
                    raw_question,
                    question_label,
                    identity=identity,
                    locked_plan=locked_plan,
                    expected_shard=expected_shard,
                    expected_questions_by_id=expected_questions_by_id,
                    path_label=path_label,
                    portable_name=portable_name,
                    report_sha256=digest,
                    sample_id=sample_id,
                    sample_sha256=str(sample.get("sample_sha256") or ""),
                )
                report_prompt_counts.append(
                    int(question["prompt_token_proxy"])
                )
                report_provider_compliances.append(
                    question["provider_prompt_budget_compliant"]
                )

        _validate_shard_totals(
            report,
            label,
            question_count=report_question_count,
            prompt_counts=report_prompt_counts,
            provider_compliances=report_provider_compliances,
            locked_plan=locked_plan,
        )
        accumulator.input_rows.append(
            {
                "name": portable_name,
                "sha256": digest,
                "num_samples": len(samples),
                "num_questions": report_question_count,
                "target_status": shard_target_status,
            }
        )

    assert accumulator.expected_identity is not None
    question_ids, questions = _finalize_population(
        accumulator, locked_plan, min_questions
    )
    summary = _campaign_summary(
        accumulator,
        questions,
        question_ids,
        accuracy_target=accuracy_target,
        locked_plan=locked_plan,
        claim_profile_verified=claim_profile_verified,
    )

    # This is deliberately the final operation before emitting a certified
    # result. A source, policy, environment, or implementation that changed
    # while shard reports were being inspected cannot retain claim status.
    if locked_plan is not None:
        _assert_locked_plan_unchanged(locked_plan)

    return _campaign_report(
        accumulator,
        summary,
        questions,
        question_ids,
        accuracy_target=accuracy_target,
        min_questions=min_questions,
        locked_plan=locked_plan,
        claim_profile_verified=claim_profile_verified,
    )
