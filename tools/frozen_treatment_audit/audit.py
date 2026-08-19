"""End-to-end frozen-treatment audit orchestration."""

from __future__ import annotations

import re
import math
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .cache_artifacts import (
    build_cache_pairs,
    cache_receipts_sha256,
    immutable_database,
    load_source_records,
    source_surface_sha256,
    validate_cache_receipts,
)
from .campaign_validation import _validate_campaign_inputs
from .canonical import (
    AuditError,
    FileSnapshot,
    assert_file_snapshot_unchanged,
    bytes_sha256,
    canonical_sha256,
    package_sha256,
    parse_json_object,
    read_file_snapshot,
    require_int,
    require_list,
    require_mapping,
    require_number,
    require_sha256,
    require_text,
    snapshot_receipt,
    tree_snapshot,
    validate_output_location,
)
from .frozen_source import load_frozen_source
from .population import PopulationPlan, Question, build_population_plan
from .prompt import FrozenPromptRuntime
from .provenance import ExcerptResolver


_JUDGE_VERDICT = re.compile(r"^\s*(CORRECT|INCORRECT)\b(?P<rest>[\s\S]*)$", re.I)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.I)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

_CAMPAIGN_FIELDS = {
    "schema_version",
    "report_type",
    "inputs",
    "input_count",
    "input_set_sha256",
    "benchmark",
    "dataset_sha256",
    "split_manifest_sha256",
    "benchmark_split",
    "implementation_sha256",
    "environment_lock_sha256",
    "policy_manifest_sha256",
    "chunker_config",
    "retrieval_config",
    "responder_model",
    "judge_model",
    "embedding_device",
    "recent_window",
    "max_prompt_tokens",
    "prompt_token_proxy_identity",
    "responder_output_token_reserve",
    "evaluation_protocol",
    "claim_profile",
    "claim_profile_verified",
    "cache_receipts_by_sample",
    "num_samples",
    "num_questions",
    "question_results",
    "question_sources",
    "mean_f1",
    "exact_match_rate",
    "judge_accuracy",
    "mean_context_tokens",
    "mean_prompt_token_proxy",
    "p95_prompt_token_proxy",
    "max_prompt_token_proxy_observed",
    "mean_request_token_proxy",
    "mean_prompt_tokens",
    "p95_prompt_tokens",
    "max_prompt_tokens_observed",
    "context_token_distribution",
    "prompt_token_proxy_distribution",
    "request_token_proxy_distribution",
    "provider_input_token_distribution",
    "prompt_token_distribution",
    "transcript_token_distribution",
    "prompt_token_proxy_budget_compliance",
    "provider_prompt_budget_compliance",
    "provider_input_usage_status",
    "prompt_budget_compliance",
    "responder_usage",
    "judge_usage",
    "by_category",
    "accuracy_target",
    "min_target_questions",
    "accuracy_target_met",
    "metric_accuracy_target_met",
    "locked_population_verified",
    "target_status",
}

_QUESTION_FIELDS = {
    "question_id",
    "question",
    "gold_answer",
    "predicted_answer",
    "category",
    "retrieved_chunks",
    "f1",
    "exact_match",
    "judge_correct",
    "judge_reasoning",
    "context_tokens",
    "prompt_token_proxy",
    "prompt_tokens",
    "responder_output_token_reserve",
    "request_token_proxy",
    "provider_prompt_budget_compliant",
    "transcript_tokens",
    "context_fraction",
    "transcript_token_savings",
    "responder_usage",
    "judge_usage",
}

def _policy_identity(
    policy_snapshot: FileSnapshot,
    dataset_snapshot: FileSnapshot,
    split_snapshot: FileSnapshot,
    repository_root: Path,
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], Any, FileSnapshot]:
    source = load_frozen_source(repository_root, source_commit)
    policy = parse_json_object(policy_snapshot.payload, "policy manifest")
    split = parse_json_object(split_snapshot.payload, "split manifest")
    if policy.get("format") != "memory-condense-retrieval-policy-v1":
        raise AuditError("policy format mismatch")
    if policy.get("status") != "validation_frozen":
        raise AuditError("policy is not frozen for validation")
    if policy.get("split") != "validation":
        raise AuditError("policy does not bind the validation split")
    if policy.get("claim_profile") != "longmemeval-s-1m-100q-95-v1":
        raise AuditError("policy does not bind the frozen 1M/100q/95% claim profile")
    if policy.get("split_manifest") != split_snapshot.path.name:
        raise AuditError("policy split filename mismatch")
    dataset_hash = dataset_snapshot.sha256
    split_hash = split_snapshot.sha256
    expected = {
        "dataset_sha256": dataset_hash,
        "split_manifest_sha256": split_hash,
        "implementation_sha256": source.implementation_sha256,
        "environment_lock_sha256": source.environment_lock_sha256,
    }
    for field, actual in expected.items():
        if policy.get(field) != actual:
            raise AuditError(f"policy {field} mismatch")
    if split.get("dataset_sha256") != dataset_hash:
        raise AuditError("split manifest dataset hash mismatch")
    if policy.get("selection_artifact_required") is not True:
        raise AuditError("policy does not require its selection artifact")
    selection_value = require_text(
        policy.get("selection_artifact"), "policy.selection_artifact"
    )
    selection_relative = Path(selection_value)
    if selection_relative.is_absolute() or ".." in selection_relative.parts:
        raise AuditError("selection artifact must be a safe repository-relative path")
    selection_path = (repository_root / selection_relative).resolve()
    try:
        selection_path.relative_to(repository_root)
    except ValueError as exc:
        raise AuditError("selection artifact escapes the repository") from exc
    selection_snapshot = read_file_snapshot(selection_path, "selection artifact")
    if policy.get("selection_artifact_sha256") != selection_snapshot.sha256:
        raise AuditError("selection artifact hash mismatch")
    for snapshot, label in (
        (policy_snapshot, "policy manifest"),
        (split_snapshot, "split manifest"),
        (selection_snapshot, "selection artifact"),
    ):
        try:
            relative = snapshot.path.relative_to(source.repository_root).as_posix()
        except ValueError as exc:
            raise AuditError(f"{label} must be inside the repository root") from exc
        if snapshot.payload != source.blob(relative):
            raise AuditError(
                f"{label} differs from {source.source_commit}:{relative}"
            )
    return policy, split, source, selection_snapshot


def _validate_report_identity(
    report: dict[str, Any],
    policy: dict[str, Any],
    *,
    benchmark_name: str,
    dataset_hash: str,
    split_hash: str,
    policy_hash: str,
    implementation_hash: str,
    environment_hash: str,
) -> None:
    if set(report) != _CAMPAIGN_FIELDS:
        raise AuditError("campaign report has an unexpected top-level shape")
    if report.get("schema_version") != 1 or report.get("report_type") != "benchmark_campaign":
        raise AuditError("only a merged v3 benchmark_campaign report is auditable")
    if report.get("benchmark") != benchmark_name:
        raise AuditError("campaign benchmark name differs from the dataset filename")
    expected_hashes = {
        "dataset_sha256": dataset_hash,
        "split_manifest_sha256": split_hash,
        "policy_manifest_sha256": policy_hash,
        "implementation_sha256": implementation_hash,
        "environment_lock_sha256": environment_hash,
    }
    for field, expected in expected_hashes.items():
        if report.get(field) != expected:
            raise AuditError(f"campaign report {field} mismatch")
    if report.get("benchmark_split") != "validation":
        raise AuditError("campaign report is not the validation partition")
    if report.get("claim_profile") != policy.get("claim_profile"):
        raise AuditError("campaign claim profile mismatch")
    if report.get("claim_profile_verified") is not True:
        raise AuditError("campaign claim profile was not verified")
    if report.get("locked_population_verified") is not True:
        raise AuditError("campaign population was not verified")
    retrieval = require_mapping(policy.get("retrieval"), "policy.retrieval")
    report_retrieval = require_mapping(report.get("retrieval_config"), "report.retrieval_config")
    chunker = require_mapping(report.get("chunker_config"), "report.chunker_config")
    maximum = require_int(report.get("max_prompt_tokens"), "report.max_prompt_tokens", minimum=1)
    report_expected = {
        field: value
        for field, value in retrieval.items()
        if field not in {"chunker_min_tokens", "chunker_max_tokens"}
    }
    if canonical_sha256(report_retrieval) != canonical_sha256(report_expected):
        raise AuditError("campaign retrieval_config has extra, missing, or changed fields")
    if set(chunker) != {"min_tokens", "max_tokens"}:
        raise AuditError("campaign chunker_config has an unexpected shape")
    actual: dict[str, Any] = dict(report_retrieval)
    actual["chunker_min_tokens"] = chunker.get("min_tokens")
    actual["chunker_max_tokens"] = chunker.get("max_tokens")
    actual["max_prompt_tokens"] = maximum
    if canonical_sha256(actual) != canonical_sha256(retrieval):
        raise AuditError("campaign retrieval configuration differs from policy")
    evaluation = dict(require_mapping(policy.get("evaluation"), "policy.evaluation"))
    evaluation.pop("sample_offsets", None)
    if canonical_sha256(report.get("evaluation_protocol")) != canonical_sha256(evaluation):
        raise AuditError("campaign evaluation protocol differs from policy")
    for report_field, policy_field in (
        ("responder_model", "responder_model"),
        ("judge_model", "judge_model"),
        ("embedding_device", "embedding_device"),
        ("recent_window", "recent_window"),
        ("max_prompt_tokens", "max_prompt_tokens"),
        ("responder_output_token_reserve", "responder_output_token_reserve"),
    ):
        if report.get(report_field) != evaluation.get(policy_field):
            raise AuditError(f"campaign {report_field} differs from policy")


def _usage(value: Any, label: str) -> dict[str, int | float]:
    row = require_mapping(value, label)
    if set(row) != {
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "elapsed_s",
        "calls",
    }:
        raise AuditError(f"{label} has an unexpected shape")
    return {
        "input_tokens": require_int(row["input_tokens"], f"{label}.input_tokens"),
        "output_tokens": require_int(row["output_tokens"], f"{label}.output_tokens"),
        "cache_read_input_tokens": require_int(
            row["cache_read_input_tokens"], f"{label}.cache_read_input_tokens"
        ),
        "elapsed_s": require_number(row["elapsed_s"], f"{label}.elapsed_s", minimum=0),
        "calls": require_int(row["calls"], f"{label}.calls"),
    }


def _normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = lowered.translate(_PUNCT_TABLE)
    without_articles = _ARTICLES_RE.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def _f1(prediction: str, gold: str) -> float:
    predicted = _normalize_answer(prediction).split()
    expected = _normalize_answer(gold).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    common = Counter(predicted) & Counter(expected)
    matches = sum(common.values())
    if not matches:
        return 0.0
    precision = matches / len(predicted)
    recall = matches / len(expected)
    return 2 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _nearest_rank(values: list[int], quantile: float) -> int:
    return values[max(0, math.ceil(quantile * len(values)) - 1)] if values else 0


def _distribution(values: list[int]) -> dict[str, int | float | list[int]]:
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
        "mean": _mean([float(value) for value in ordered]),
        "p50": _nearest_rank(ordered, 0.50),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
        "values": ordered,
    }


def _sum_usage(rows: list[dict[str, int | float]]) -> dict[str, int | float]:
    return {
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "cache_read_input_tokens": sum(
            int(row["cache_read_input_tokens"]) for row in rows
        ),
        "elapsed_s": math.fsum(float(row["elapsed_s"]) for row in rows),
        "calls": sum(int(row["calls"]) for row in rows),
    }


def _category_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw = row.get("category")
        category = raw if isinstance(raw, str) and raw.strip() else "uncategorized"
        grouped[category].append(row)
    return {
        category: {
            "category": category,
            "num_questions": len(items),
            "mean_f1": _mean([float(item["f1"]) for item in items]),
            "exact_match_rate": _mean(
                [1.0 if item["exact_match"] else 0.0 for item in items]
            ),
            "judge_accuracy": _mean(
                [1.0 if item["judge_correct"] else 0.0 for item in items]
            ),
        }
        for category, items in sorted(grouped.items())
    }


def _must_equal(actual: Any, expected: Any, label: str) -> None:
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise AuditError(f"campaign aggregate mismatch: {label}")


def _validate_question_rows(
    report: dict[str, Any],
    policy: dict[str, Any],
    plan: PopulationPlan,
    runtime: FrozenPromptRuntime,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    rows = require_list(report.get("question_results"), "report.question_results")
    declared = require_int(report.get("num_questions"), "report.num_questions", minimum=1)
    if declared != len(rows):
        raise AuditError("campaign question count does not match its rows")
    evaluation = require_mapping(policy.get("evaluation"), "policy.evaluation")
    minimum = require_int(
        evaluation.get("min_target_questions"),
        "policy.evaluation.min_target_questions",
        minimum=1,
    )
    if minimum != 100 or len(rows) != 100:
        raise AuditError("frozen treatment receipt requires exactly 100 locked questions")
    prompt_cap = require_int(
        evaluation.get("max_prompt_tokens"),
        "policy.evaluation.max_prompt_tokens",
        minimum=1,
    )
    reserve = require_int(
        evaluation.get("responder_output_token_reserve"),
        "policy.evaluation.responder_output_token_reserve",
        minimum=1,
    )
    sources = require_mapping(report.get("question_sources"), "report.question_sources")
    if set(sources) != set(plan.question_to_sample):
        raise AuditError("campaign question_sources does not cover the locked population")
    expected_questions: dict[str, Question] = {
        question.question_id: question
        for sample in plan.samples.values()
        for question in sample.questions
    }
    audits: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {digest: [] for digest in plan.samples}
    judged = 0
    provider_counts: list[int] = []
    prompt_counts: list[int] = []
    context_counts: list[int] = []
    request_counts: list[int] = []
    transcript_counts: list[int] = []
    responder_usages: list[dict[str, int | float]] = []
    judge_usages: list[dict[str, int | float]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    observed_order: list[str] = []
    for index, raw in enumerate(rows):
        row = require_mapping(raw, f"report.question_results[{index}]")
        if set(row) != _QUESTION_FIELDS:
            raise AuditError(f"question row {index} has an unexpected shape")
        question_id = require_text(row.get("question_id"), f"question[{index}].question_id")
        if question_id in audits:
            raise AuditError(f"duplicate campaign question ID: {question_id}")
        expected = expected_questions.get(question_id)
        if expected is None:
            raise AuditError(f"campaign contains an unlocked question: {question_id}")
        sample_sha = plan.question_to_sample[question_id]
        source = require_mapping(sources[question_id], f"question_sources.{question_id}")
        if set(source) != {"report_name", "report_sha256", "sample_id", "sample_sha256"}:
            raise AuditError(f"question source has an unexpected shape: {question_id}")
        if source.get("sample_sha256") != sample_sha:
            raise AuditError(f"question source sample mismatch: {question_id}")
        if source.get("sample_id") != plan.samples[sample_sha].sample_id:
            raise AuditError(f"question source sample ID mismatch: {question_id}")
        require_text(source.get("report_name"), f"question_sources.{question_id}.report_name")
        require_sha256(source.get("report_sha256"), f"question_sources.{question_id}.report_sha256")
        for field, expected_value in (
            ("question", expected.question),
            ("gold_answer", expected.answer),
            ("category", expected.category),
        ):
            if row.get(field) != expected_value:
                raise AuditError(f"question {question_id} field mismatch: {field}")
        predicted = require_text(row.get("predicted_answer"), f"{question_id}.predicted_answer")
        expected_f1 = _f1(predicted, expected.answer)
        if require_number(row.get("f1"), f"{question_id}.f1", minimum=0, maximum=1) != expected_f1:
            raise AuditError(f"{question_id} F1 differs from frozen normalization")
        expected_exact = _normalize_answer(predicted) == _normalize_answer(expected.answer)
        if row.get("exact_match") is not expected_exact:
            raise AuditError(f"{question_id} exact-match flag differs from frozen normalization")
        if not isinstance(row.get("judge_correct"), bool):
            raise AuditError(f"{question_id}.judge_correct must be boolean")
        verdict = _JUDGE_VERDICT.fullmatch(
            require_text(row.get("judge_reasoning"), f"{question_id}.judge_reasoning")
        )
        if verdict is None:
            raise AuditError(f"{question_id} has no exact binary judge verdict")
        remainder = verdict.group("rest").lstrip(" \t\r\n,.:;-—")
        if remainder.casefold().startswith("or ") or remainder.startswith("/"):
            raise AuditError(f"{question_id} has an ambiguous judge verdict")
        verdict_value = verdict.group(1).casefold() == "correct"
        if verdict_value is not row["judge_correct"]:
            raise AuditError(f"{question_id} judge boolean disagrees with verdict")
        judged += int(verdict_value)
        excerpts_raw = require_list(row.get("retrieved_chunks"), f"{question_id}.retrieved_chunks")
        if any(not isinstance(excerpt, str) or not excerpt for excerpt in excerpts_raw):
            raise AuditError(f"{question_id} has a non-string or empty retrieved excerpt")
        excerpts = [str(excerpt) for excerpt in excerpts_raw]
        context, messages = runtime.prompt_messages(expected.dated_question, excerpts)
        context_tokens = sum(runtime.count_tokens(excerpt) for excerpt in excerpts)
        if row.get("context_tokens") != context_tokens:
            raise AuditError(f"{question_id} context token count mismatch")
        if row.get("transcript_tokens") != plan.transcript_tokens[sample_sha]:
            raise AuditError(f"{question_id} transcript token count mismatch")
        prompt_proxy = runtime.prompt_token_proxy(messages)
        if row.get("prompt_token_proxy") != prompt_proxy or row.get("prompt_tokens") != prompt_proxy:
            raise AuditError(f"{question_id} prompt-token proxy mismatch")
        if prompt_proxy > prompt_cap:
            raise AuditError(f"{question_id} exceeds the frozen local prompt cap")
        if row.get("responder_output_token_reserve") != reserve:
            raise AuditError(f"{question_id} output reserve mismatch")
        if row.get("request_token_proxy") != prompt_proxy + reserve:
            raise AuditError(f"{question_id} request-token proxy mismatch")
        responder = _usage(row.get("responder_usage"), f"{question_id}.responder_usage")
        judge = _usage(row.get("judge_usage"), f"{question_id}.judge_usage")
        if responder["calls"] != 1 or judge["calls"] != 1:
            raise AuditError(f"{question_id} does not have exactly one responder and judge call")
        provider_input = int(responder["input_tokens"])
        expected_compliance = None if provider_input <= 0 else provider_input <= prompt_cap
        if row.get("provider_prompt_budget_compliant") != expected_compliance:
            raise AuditError(f"{question_id} provider budget flag mismatch")
        if expected_compliance is False:
            raise AuditError(f"{question_id} provider input exceeded the frozen cap")
        if provider_input > 0:
            provider_counts.append(provider_input)
        prompt_counts.append(prompt_proxy)
        context_counts.append(context_tokens)
        request_counts.append(prompt_proxy + reserve)
        transcript_counts.append(plan.transcript_tokens[sample_sha])
        responder_usages.append(responder)
        judge_usages.append(judge)
        transcript_tokens = plan.transcript_tokens[sample_sha]
        expected_fraction = context_tokens / transcript_tokens if transcript_tokens else 0.0
        if require_number(
            row.get("context_fraction"),
            f"{question_id}.context_fraction",
            minimum=0,
            maximum=1,
        ) != expected_fraction:
            raise AuditError(f"{question_id} context fraction mismatch")
        if require_number(
            row.get("transcript_token_savings"),
            f"{question_id}.transcript_token_savings",
            minimum=0,
            maximum=1,
        ) != 1.0 - expected_fraction:
            raise AuditError(f"{question_id} transcript savings mismatch")
        judge_messages = runtime.judge_messages(
            expected.question,
            expected.answer,
            predicted,
        )
        audits[question_id] = {
            "sample_sha256": sample_sha,
            "question_sha256": bytes_sha256(expected.question.encode("utf-8")),
            "gold_answer_sha256": bytes_sha256(expected.answer.encode("utf-8")),
            "predicted_answer_sha256": bytes_sha256(predicted.encode("utf-8")),
            "dated_question_sha256": bytes_sha256(expected.dated_question.encode("utf-8")),
            "rendered_context_sha256": bytes_sha256(context.encode("utf-8")),
            "prompt_messages_sha256": canonical_sha256(messages),
            "prompt_message_hash_encoding": "canonical-json-utf8-v1",
            "prompt_token_proxy": prompt_proxy,
            "judge_prompt_messages_sha256": canonical_sha256(judge_messages),
            "reported_judge_reasoning_sha256": bytes_sha256(
                str(row["judge_reasoning"]).encode("utf-8")
            ),
            "reported_judge_correct": verdict_value,
            "judge_verdict_structurally_consistent": True,
            "provider_input_tokens": provider_input if provider_input > 0 else None,
            "retrieved_excerpt_count": len(excerpts),
            "excerpts": excerpts,
            "provenance": [],
        }
        grouped[sample_sha].append(row)
        rows_by_id[question_id] = row
        observed_order.append(question_id)
    if set(audits) != set(plan.question_to_sample):
        raise AuditError("campaign rows do not equal the locked question population")
    if observed_order != sorted(observed_order):
        raise AuditError("campaign question rows are not in canonical question-ID order")
    target = require_number(
        evaluation.get("accuracy_target"),
        "policy.evaluation.accuracy_target",
        minimum=0,
        maximum=1,
    )
    accuracy = judged / len(rows)
    if target != 0.95 or accuracy < target:
        raise AuditError("campaign report does not claim the frozen >=95% judge target")
    if report.get("accuracy_target") != target or report.get("min_target_questions") != minimum:
        raise AuditError("campaign target parameters differ from policy")
    if require_number(report.get("judge_accuracy"), "report.judge_accuracy") != accuracy:
        raise AuditError("campaign judge accuracy does not match question verdicts")
    if report.get("accuracy_target_met") is not True or report.get("target_status") != "passed":
        raise AuditError("campaign does not carry a passed target decision")
    if report.get("metric_accuracy_target_met") is not True:
        raise AuditError("campaign metric target is not met")
    if report.get("prompt_token_proxy_budget_compliance") is not True:
        raise AuditError("campaign local prompt-cap decision is not true")
    if report.get("prompt_budget_compliance") is not True:
        raise AuditError("campaign combined prompt-cap decision is not true")
    if report.get("max_prompt_token_proxy_observed") != max(prompt_counts):
        raise AuditError("campaign maximum prompt proxy is inconsistent")
    expected_status = (
        "unavailable"
        if not provider_counts
        else "complete"
        if len(provider_counts) == len(rows)
        else "partial"
    )
    if report.get("provider_input_usage_status") != expected_status:
        raise AuditError("campaign provider input-usage status is inconsistent")
    expected_provider = all(value <= prompt_cap for value in provider_counts) if provider_counts else None
    if report.get("provider_prompt_budget_compliance") != expected_provider:
        raise AuditError("campaign provider prompt-cap decision is inconsistent")
    expected_distributions = {
        "context_token_distribution": _distribution(context_counts),
        "prompt_token_proxy_distribution": _distribution(prompt_counts),
        "request_token_proxy_distribution": _distribution(request_counts),
        "provider_input_token_distribution": _distribution(provider_counts),
        "prompt_token_distribution": _distribution(prompt_counts),
        "transcript_token_distribution": _distribution(transcript_counts),
    }
    for field, expected_value in expected_distributions.items():
        _must_equal(report.get(field), expected_value, field)
    expected_scalars = {
        "mean_f1": _mean([float(row["f1"]) for row in rows_by_id.values()]),
        "exact_match_rate": _mean(
            [1.0 if row["exact_match"] else 0.0 for row in rows_by_id.values()]
        ),
        "mean_context_tokens": expected_distributions["context_token_distribution"]["mean"],
        "mean_prompt_token_proxy": expected_distributions["prompt_token_proxy_distribution"]["mean"],
        "p95_prompt_token_proxy": expected_distributions["prompt_token_proxy_distribution"]["p95"],
        "max_prompt_token_proxy_observed": expected_distributions["prompt_token_proxy_distribution"]["max"],
        "mean_request_token_proxy": expected_distributions["request_token_proxy_distribution"]["mean"],
        "mean_prompt_tokens": expected_distributions["prompt_token_proxy_distribution"]["mean"],
        "p95_prompt_tokens": expected_distributions["prompt_token_proxy_distribution"]["p95"],
        "max_prompt_tokens_observed": expected_distributions["prompt_token_proxy_distribution"]["max"],
    }
    for field, expected_value in expected_scalars.items():
        if report.get(field) != expected_value:
            raise AuditError(f"campaign aggregate mismatch: {field}")
    _must_equal(report.get("responder_usage"), _sum_usage(responder_usages), "responder_usage")
    _must_equal(report.get("judge_usage"), _sum_usage(judge_usages), "judge_usage")
    _must_equal(
        report.get("by_category"),
        _category_metrics(list(rows_by_id.values())),
        "by_category",
    )
    return (
        audits,
        grouped,
        rows_by_id,
        {
            "reported_correct_questions": judged,
            "reported_total_questions": len(rows),
            "reported_judge_accuracy": accuracy,
            "reported_accuracy_target": target,
            "report_claim_meets_target": accuracy >= target,
            "provider_execution_authenticated": False,
            "judge_execution_authenticated": False,
            "factual_accuracy_independently_verified": False,
        },
    )


def _sample_receipts(
    report: dict[str, Any],
    plan: PopulationPlan,
    *,
    implementation_hash: str,
    environment_hash: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    raw = require_mapping(
        report.get("cache_receipts_by_sample"), "report.cache_receipts_by_sample"
    )
    if set(raw) != set(plan.samples):
        raise AuditError("campaign cache receipt samples differ from locked shards")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for sample_sha in sorted(plan.samples):
        result[sample_sha] = validate_cache_receipts(
            raw[sample_sha],
            sample_sha256=sample_sha,
            implementation_sha256=implementation_hash,
            environment_sha256=environment_hash,
        )
    return result




def _receipt_outcome_sections(reported_outcome: dict[str, Any]) -> dict[str, Any]:
    """Keep report claims visibly separate from unauthenticated outcomes."""

    return {
        "reported_outcome_consistency": dict(reported_outcome),
        "independent_verification": {
            "operational_claim_status": "not_authenticated",
            "provider_execution_authenticated": False,
            "judge_execution_authenticated": False,
            "factual_accuracy_independently_verified": False,
            "reason": (
                "provider and judge outputs are report assertions without "
                "authenticated execution evidence"
            ),
        },
    }


def _audit_tool_receipt(
    source_sha256: str,
    expected_source_sha256: str | None,
) -> dict[str, Any]:
    return {
        "python_source_sha256": source_sha256,
        "expected_python_source_sha256": expected_source_sha256,
        "pre_post_equal": True,
        "audit_tool_source_externally_pinned": expected_source_sha256 is not None,
        "loaded_execution_authenticated": False,
        "source_pin_scope": (
            "caller-supplied expected Python source package digest; not a "
            "signature or loaded-bytecode attestation"
        ),
    }


def audit_frozen_treatment(
    *,
    report_path: str | Path,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    policy_path: str | Path,
    repository_root: str | Path,
    source_commit: str,
    compiled_cache_root: str | Path,
    causal_cache_root: str | Path,
    shard_report_root: str | Path | None = None,
    output_path: str | Path | None = None,
    expected_audit_tool_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify frozen lineage and structural consistency.

    Provider and judge execution remain unauthenticated, so the returned
    receipt never independently certifies factual accuracy.  Supplying
    ``expected_audit_tool_sha256`` externally pins the Python source package
    used for the audit; omitting it leaves that source identity self-reported.
    """

    package_root = Path(__file__).resolve().parent
    audit_tool_before = package_sha256(package_root)
    expected_tool_digest = (
        None
        if expected_audit_tool_sha256 is None
        else require_sha256(
            expected_audit_tool_sha256,
            "expected audit-tool SHA-256",
        )
    )
    if expected_tool_digest is not None and audit_tool_before != expected_tool_digest:
        raise AuditError("audit-tool Python source differs from the externally expected digest")
    report_snapshot = read_file_snapshot(report_path, "benchmark campaign report")
    dataset_snapshot = read_file_snapshot(dataset_path, "LongMemEval dataset")
    split_snapshot = read_file_snapshot(split_manifest_path, "split manifest")
    policy_snapshot = read_file_snapshot(policy_path, "policy manifest")
    report_file = report_snapshot.path
    dataset_file = dataset_snapshot.path
    split_file = split_snapshot.path
    policy_file = policy_snapshot.path
    repository = Path(repository_root).resolve()
    policy, split, source, selection_snapshot = _policy_identity(
        policy_snapshot,
        dataset_snapshot,
        split_snapshot,
        repository,
        source_commit,
    )
    immutable_inputs: dict[str, FileSnapshot] = {
        "report": report_snapshot,
        "dataset": dataset_snapshot,
        "split_manifest": split_snapshot,
        "policy_manifest": policy_snapshot,
        "selection_artifact": selection_snapshot,
    }
    shard_root = (
        report_file.parent
        if shard_report_root is None
        else Path(shard_report_root).resolve()
    )
    if output_path is not None:
        validate_output_location(
            output_path,
            protected_roots=(repository, compiled_cache_root, causal_cache_root, shard_root),
            protected_files=(snapshot.path for snapshot in immutable_inputs.values()),
        )
    report = parse_json_object(report_snapshot.payload, "benchmark campaign report")
    policy_hash = policy_snapshot.sha256
    _validate_report_identity(
        report,
        policy,
        benchmark_name=dataset_file.stem,
        dataset_hash=dataset_snapshot.sha256,
        split_hash=split_snapshot.sha256,
        policy_hash=policy_hash,
        implementation_hash=source.implementation_sha256,
        environment_hash=source.environment_lock_sha256,
    )
    evaluation = require_mapping(policy.get("evaluation"), "policy.evaluation")
    runtime = FrozenPromptRuntime(
        source,
        require_mapping(
            evaluation.get("prompt_token_proxy_identity"),
            "policy.evaluation.prompt_token_proxy_identity",
        ),
    )
    if canonical_sha256(report.get("prompt_token_proxy_identity")) != canonical_sha256(
        runtime.tokenizer_identity()
    ):
        raise AuditError("campaign tokenizer identity differs from reconstructed runtime")
    plan = build_population_plan(dataset_snapshot.payload, split, policy, runtime)
    if report.get("num_samples") != len(plan.samples):
        raise AuditError("campaign sample count differs from reconstructed shards")
    question_audits, grouped_rows, campaign_rows, reported_outcome = (
        _validate_question_rows(report, policy, plan, runtime)
    )
    shard_snapshots = _validate_campaign_inputs(
        report,
        shard_root=shard_root,
        campaign_rows=campaign_rows,
        plan=plan,
    )
    immutable_inputs.update(
        {f"shard_report:{name}": snapshot for name, snapshot in shard_snapshots.items()}
    )
    if output_path is not None:
        validate_output_location(
            output_path,
            protected_roots=(repository, compiled_cache_root, causal_cache_root, shard_root),
            protected_files=(snapshot.path for snapshot in immutable_inputs.values()),
        )
    receipts = _sample_receipts(
        report,
        plan,
        implementation_hash=source.implementation_sha256,
        environment_hash=source.environment_lock_sha256,
    )

    compiled_before = tree_snapshot(compiled_cache_root)
    causal_before = tree_snapshot(causal_cache_root)
    pairs = build_cache_pairs(
        compiled_cache_root,
        causal_cache_root,
        samples=plan.samples,
        sample_receipts=receipts,
        implementation_sha256=source.implementation_sha256,
        environment_sha256=source.environment_lock_sha256,
        database_schema_sql=source.database_schema_sql,
    )
    retrieval = require_mapping(policy.get("retrieval"), "policy.retrieval")
    query_aware = retrieval.get("consolidation_query_aware_sentence_packing") is True
    source_metadata = retrieval.get("consolidation_source_metadata_packing") is True
    max_sentences = require_int(
        retrieval.get("consolidation_max_sentences_per_expansion"),
        "policy.retrieval.consolidation_max_sentences_per_expansion",
        minimum=1,
    )
    cache_audits: dict[str, dict[str, Any]] = {}
    for sample_sha in sorted(plan.samples):
        pair = pairs[sample_sha]
        compiled_surface = source_surface_sha256(pair.compiled.database)
        causal_surface = source_surface_sha256(pair.causal.database)
        if compiled_surface != causal_surface:
            raise AuditError("compiled and causal provenance surfaces differ")
        _compiled_turns, compiled_chunks = load_source_records(
            pair.compiled.database,
            plan.samples[sample_sha],
        )
        _causal_turns, chunks = load_source_records(
            pair.causal.database,
            plan.samples[sample_sha],
        )
        compiled_by_chunk = {chunk.chunk_id: chunk for chunk in compiled_chunks}
        if set(compiled_by_chunk) != {chunk.chunk_id for chunk in chunks}:
            raise AuditError("compiled and causal chunk ID sets differ")
        for chunk in chunks:
            compiled_chunk = compiled_by_chunk[chunk.chunk_id]
            if (
                compiled_chunk.text,
                compiled_chunk.start_char,
                compiled_chunk.end_char,
                compiled_chunk.token_count,
                compiled_chunk.role,
                compiled_chunk.turn_text,
                compiled_chunk.source_id,
                compiled_chunk.source_timestamp,
            ) != (
                chunk.text,
                chunk.start_char,
                chunk.end_char,
                chunk.token_count,
                chunk.role,
                chunk.turn_text,
                chunk.source_id,
                chunk.source_timestamp,
            ):
                raise AuditError("compiled and causal chunk provenance differs")
        with immutable_database(pair.causal.database) as connection:
            resolver = ExcerptResolver(
                connection,
                chunks,
                runtime,
                source_metadata=source_metadata,
                query_aware=query_aware,
                max_sentences=max_sentences,
            )
            expected_by_id = {
                question.question_id: question
                for question in plan.samples[sample_sha].questions
            }
            for row in grouped_rows[sample_sha]:
                question_id = str(row["question_id"])
                expected = expected_by_id[question_id]
                excerpts = question_audits[question_id].pop("excerpts")
                provenance = resolver.resolve_question(
                    expected.dated_question,
                    excerpts,
                )
                for excerpt_receipt in provenance:
                    compiled_chunk = compiled_by_chunk[str(excerpt_receipt["chunk_id"])]
                    excerpt_receipt["compiled_turn_id"] = compiled_chunk.turn_id
                    excerpt_receipt["causal_turn_id"] = excerpt_receipt.pop("turn_id")
                question_audits[question_id]["provenance"] = provenance
        cache_audits[sample_sha] = {
            "sample_offset": plan.offsets[sample_sha],
            "compiled_entry": pair.compiled.directory.name,
            "causal_entry": pair.causal.directory.name,
            "source_surface_sha256": compiled_surface,
            "cache_receipts_sha256": cache_receipts_sha256(receipts[sample_sha]),
            "compiled_receipt": receipts[sample_sha]["compiled"],
            "causal_receipt": receipts[sample_sha]["causal"],
        }

    compiled_after = tree_snapshot(compiled_cache_root)
    causal_after = tree_snapshot(causal_cache_root)
    if compiled_before != compiled_after or causal_before != causal_after:
        raise AuditError("cache roots changed while the audit was reading them")
    for name, snapshot in immutable_inputs.items():
        assert_file_snapshot_unchanged(snapshot, name)
    audit_tool_after = package_sha256(package_root)
    if audit_tool_before != audit_tool_after:
        raise AuditError("audit-tool Python sources changed during execution")
    if expected_tool_digest is not None and audit_tool_after != expected_tool_digest:
        raise AuditError("audit-tool Python source no longer matches its external digest")
    if output_path is not None:
        validate_output_location(
            output_path,
            protected_roots=(repository, compiled_cache_root, causal_cache_root, shard_root),
            protected_files=(snapshot.path for snapshot in immutable_inputs.values()),
        )

    prompt_cap = int(evaluation["max_prompt_tokens"])
    body: dict[str, Any] = {
        "schema_version": 2,
        "receipt_type": "memory-condense-frozen-treatment-structural-audit-v2",
        "audit_semantics": {
            "provider_free": True,
            "structural_audit_only": True,
            "provider_execution_authenticated": False,
            "judge_execution_authenticated": False,
            "factual_accuracy_independently_verified": False,
            "retrieval_execution_replayed": False,
            "report_verdicts_are_untrusted_claims": True,
            "cache_access": "sqlite-mode-ro-immutable-and-file-hash-guarded",
            "excerpt_resolution": "unique-exact-frozen-transform-or-fail",
            "prompt_message_hash": "sha256(canonical-json-utf8(messages))",
            "self_hash": "sha256(canonical-json-utf8(receipt-without-receipt_sha256))",
        },
        "source": {
            "source_commit": source.source_commit,
            "implementation_sha256": source.implementation_sha256,
            "environment_lock_sha256": source.environment_lock_sha256,
            "benchmark_source_sha256": source.benchmark_source_sha256,
            "tokenizer_source_sha256": source.tokenizer_source_sha256,
            "context_packer_source_sha256": source.context_packer_source_sha256,
            "lexical_source_sha256": source.lexical_source_sha256,
            "database_source_sha256": source.database_source_sha256,
        },
        "inputs": {
            name: {"path": str(snapshot.path), "sha256": snapshot.sha256}
            for name, snapshot in immutable_inputs.items()
        },
        "population": {
            "benchmark_split": "validation",
            "locked_question_count": len(question_audits),
            "stress_sample_count": len(plan.samples),
            "sample_sha256": sorted(plan.samples),
            "question_ids_sha256": canonical_sha256(sorted(question_audits)),
        },
        **_receipt_outcome_sections(reported_outcome),
        "prompt_cap": {
            "semantics": evaluation.get("prompt_cap_semantics"),
            "maximum_prompt_token_proxy": prompt_cap,
            "all_reconstructed_prompts_within_local_cap": True,
            "provider_usage_status": report.get("provider_input_usage_status"),
            "provider_reported_compliance": report.get(
                "provider_prompt_budget_compliance"
            ),
            "provider_usage_authenticated": False,
            "tokenizer_identity": runtime.tokenizer_identity(),
        },
        "cache_artifacts": cache_audits,
        "cache_immutability": {
            "compiled": snapshot_receipt(compiled_cache_root, compiled_before),
            "causal": snapshot_receipt(causal_cache_root, causal_before),
            "pre_post_hashes_equal": True,
        },
        "persisted_request_state_evidence": {
            "scope": [
                str(Path(compiled_cache_root).resolve()),
                str(Path(causal_cache_root).resolve()),
            ],
            "closed_world_cache_files": [
                "compiled-store.json",
                "causal-store.json",
                "memory.db",
                "hnsw_index.bin",
            ],
            "sqlite_schema_version": 9,
            "exact_schema_objects_and_columns_validated": True,
            "runtime_storage_classes_validated": True,
            "database_embeddings_fixed_width_and_finite": True,
            "filesystem_xattrs_and_ntfs_ads_rejected_where_supported": True,
            "memory_cav_head_and_hebbian_partitions_empty": True,
            "manifest_stats_are_scalar_only": True,
            "reported_retained_prompt_state_bytes": 0,
            "forbidden_request_state_observed_in_validated_partitions": False,
            "absolute_zero_transformer_state_certified": False,
            "retained_request_token_state_bytes": None,
            "limitation": (
                "permitted vector, ANN, and scalar graph payload semantics cannot "
                "prove that arbitrary values were not intentionally used to encode "
                "transformer request state"
            ),
        },
        "questions": {key: question_audits[key] for key in sorted(question_audits)},
        "audit_tool": _audit_tool_receipt(
            audit_tool_before,
            expected_tool_digest,
        ),
    }
    receipt = dict(body)
    receipt["receipt_sha256"] = canonical_sha256(body)
    return receipt


__all__ = ["AuditError", "audit_frozen_treatment"]
