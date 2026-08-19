"""Content-addressed validation of frozen campaign input shards."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .canonical import (
    AuditError,
    FileSnapshot,
    canonical_sha256,
    parse_json_object,
    read_file_snapshot,
    require_int,
    require_list,
    require_mapping,
    require_number,
    require_sha256,
    require_text,
)
from .population import PopulationPlan


_SHARD_FIELDS = {
    "config",
    "benchmark",
    "samples",
    "num_samples",
    "num_questions",
    "mean_f1",
    "exact_match_rate",
    "judge_accuracy",
    "mean_context_tokens",
    "mean_prompt_token_proxy",
    "p95_prompt_token_proxy",
    "max_prompt_token_proxy_observed",
    "mean_request_token_proxy",
    "responder_output_token_reserve",
    "prompt_token_proxy_identity",
    "prompt_token_proxy_budget_compliance",
    "provider_prompt_budget_compliance",
    "provider_input_usage_status",
    "mean_prompt_tokens",
    "p95_prompt_tokens",
    "mean_transcript_tokens",
    "mean_context_fraction",
    "mean_transcript_token_savings",
    "max_prompt_tokens_observed",
    "prompt_budget_compliance",
    "accuracy_target",
    "min_target_questions",
    "accuracy_target_met",
    "target_status",
    "responder_usage",
    "judge_usage",
    "dataset_sha256",
    "split_manifest_sha256",
    "benchmark_split",
    "implementation_sha256",
    "environment_lock_sha256",
    "policy_manifest_sha256",
    "evaluation_protocol",
    "by_category",
    "run_timestamp",
}

_SHARD_SAMPLE_FIELDS = {
    "sample_id",
    "sample_sha256",
    "cache_receipts",
    "cache_receipts_sha256",
    "num_turns",
    "num_questions",
    "question_results",
    "mean_f1",
    "exact_match_rate",
    "judge_accuracy",
    "mean_context_tokens",
    "mean_prompt_token_proxy",
    "mean_request_token_proxy",
    "mean_prompt_tokens",
    "transcript_tokens",
    "mean_context_fraction",
    "mean_transcript_token_savings",
}

_SHARD_CONFIG_FIELDS = {
    "chunker",
    "retrieval",
    "judge_model",
    "responder_model",
    "embedding_device",
    "conversation_dir",
    "results_dir",
    "max_conversations",
    "recent_window",
    "accuracy_target",
    "min_target_questions",
    "max_prompt_tokens",
}


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


def _nearest_rank(values: list[int], quantile: float) -> int:
    return values[max(0, math.ceil(quantile * len(values)) - 1)] if values else 0


def _ordinary_mean(values: list[float]) -> float:
    """Match the frozen per-shard reducer, which used built-in ``sum``."""

    return sum(values) / len(values) if values else 0.0


def _ordinary_sum_usage(rows: list[dict[str, int | float]]) -> dict[str, int | float]:
    elapsed = 0.0
    for row in rows:
        elapsed += float(row["elapsed_s"])
    return {
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "cache_read_input_tokens": sum(
            int(row["cache_read_input_tokens"]) for row in rows
        ),
        "elapsed_s": elapsed,
        "calls": sum(int(row["calls"]) for row in rows),
    }


def _ordinary_category_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw = row.get("category")
        category = raw if isinstance(raw, str) and raw else "uncategorized"
        grouped[category].append(row)
    return {
        category: {
            "category": category,
            "num_questions": len(items),
            "mean_f1": _ordinary_mean([float(item["f1"]) for item in items]),
            "exact_match_rate": _ordinary_mean(
                [1.0 if item["exact_match"] else 0.0 for item in items]
            ),
            "judge_accuracy": _ordinary_mean(
                [1.0 if item["judge_correct"] else 0.0 for item in items]
            ),
        }
        for category, items in sorted(grouped.items())
    }


def _validate_shard_identity_and_aggregates(
    shard: dict[str, Any],
    *,
    campaign: dict[str, Any],
    rows: list[dict[str, Any]],
    sample_sha: str,
    sample_row: dict[str, Any],
    plan: PopulationPlan,
    label: str,
) -> None:
    """Recompute the frozen shard/sample reductions from bound question rows."""

    if set(shard) != _SHARD_FIELDS:
        raise AuditError(f"{label} has an unexpected top-level shape")
    if set(sample_row) != _SHARD_SAMPLE_FIELDS:
        raise AuditError(f"{label} sample has an unexpected shape")
    for field in (
        "benchmark",
        "dataset_sha256",
        "split_manifest_sha256",
        "benchmark_split",
        "implementation_sha256",
        "environment_lock_sha256",
        "policy_manifest_sha256",
        "responder_output_token_reserve",
        "prompt_token_proxy_identity",
        "accuracy_target",
        "min_target_questions",
    ):
        if canonical_sha256(shard.get(field)) != canonical_sha256(campaign.get(field)):
            raise AuditError(f"{label} identity differs from campaign field {field}")
    config = require_mapping(shard.get("config"), f"{label}.config")
    if set(config) != _SHARD_CONFIG_FIELDS:
        raise AuditError(f"{label}.config has an unexpected shape")
    config_expected = {
        "chunker": campaign.get("chunker_config"),
        "retrieval": campaign.get("retrieval_config"),
        "responder_model": campaign.get("responder_model"),
        "judge_model": campaign.get("judge_model"),
        "embedding_device": campaign.get("embedding_device"),
        "recent_window": campaign.get("recent_window"),
        "max_prompt_tokens": campaign.get("max_prompt_tokens"),
        "accuracy_target": campaign.get("accuracy_target"),
        "min_target_questions": campaign.get("min_target_questions"),
    }
    for field, expected in config_expected.items():
        if canonical_sha256(config.get(field)) != canonical_sha256(expected):
            raise AuditError(f"{label}.config differs from campaign field {field}")
    protocol = dict(
        require_mapping(
            shard.get("evaluation_protocol"),
            f"{label}.evaluation_protocol",
        )
    )
    offset = require_int(
        protocol.pop("sample_offset", None),
        f"{label}.evaluation_protocol.sample_offset",
    )
    if offset != plan.offsets[sample_sha]:
        raise AuditError(f"{label} has the wrong frozen sample offset")
    if canonical_sha256(protocol) != canonical_sha256(
        campaign.get("evaluation_protocol")
    ):
        raise AuditError(f"{label} evaluation protocol differs from campaign")
    run_timestamp = require_text(shard.get("run_timestamp"), f"{label}.run_timestamp")
    try:
        parsed_timestamp = datetime.fromisoformat(run_timestamp)
    except ValueError as exc:
        raise AuditError(f"{label}.run_timestamp is not ISO-8601") from exc
    if parsed_timestamp.tzinfo is None or len(run_timestamp) > 48:
        raise AuditError(f"{label}.run_timestamp lacks a bounded timezone")

    sample = plan.samples[sample_sha]
    if sample_row.get("sample_id") != sample.sample_id:
        raise AuditError(f"{label} sample ID differs from the frozen population")
    if sample_row.get("sample_sha256") != sample_sha:
        raise AuditError(f"{label} sample digest differs from the frozen population")
    if sample_row.get("num_turns") != len(sample.turns):
        raise AuditError(f"{label} sample turn count mismatch")
    if sample_row.get("transcript_tokens") != plan.transcript_tokens[sample_sha]:
        raise AuditError(f"{label} sample transcript-token count mismatch")
    campaign_receipts = require_mapping(
        campaign.get("cache_receipts_by_sample"),
        "campaign cache_receipts_by_sample",
    )
    if canonical_sha256(sample_row.get("cache_receipts")) != canonical_sha256(
        campaign_receipts.get(sample_sha)
    ):
        raise AuditError(f"{label} sample cache receipts differ from campaign")
    if sample_row.get("cache_receipts_sha256") != canonical_sha256(
        sample_row.get("cache_receipts")
    ):
        raise AuditError(f"{label} sample cache-receipt digest mismatch")

    if sample_row.get("num_questions") != len(rows):
        raise AuditError(f"{label} sample question count mismatch")
    sample_expected = {
        "mean_f1": _ordinary_mean([float(row["f1"]) for row in rows]),
        "exact_match_rate": _ordinary_mean(
            [1.0 if row["exact_match"] else 0.0 for row in rows]
        ),
        "judge_accuracy": _ordinary_mean(
            [1.0 if row["judge_correct"] else 0.0 for row in rows]
        ),
        "mean_context_tokens": _ordinary_mean(
            [float(row["context_tokens"]) for row in rows]
        ),
        "mean_prompt_token_proxy": _ordinary_mean(
            [float(row["prompt_token_proxy"]) for row in rows]
        ),
        "mean_request_token_proxy": _ordinary_mean(
            [float(row["request_token_proxy"]) for row in rows]
        ),
        "mean_prompt_tokens": _ordinary_mean(
            [float(row["prompt_token_proxy"]) for row in rows]
        ),
        "mean_context_fraction": _ordinary_mean(
            [float(row["context_fraction"]) for row in rows]
        ),
        "mean_transcript_token_savings": _ordinary_mean(
            [float(row["transcript_token_savings"]) for row in rows]
        ),
    }
    for field, expected in sample_expected.items():
        if sample_row.get(field) != expected:
            raise AuditError(f"{label} sample aggregate mismatch: {field}")

    prompt_counts = [int(row["prompt_token_proxy"]) for row in rows]
    provider_counts: list[int] = []
    responder_rows: list[dict[str, int | float]] = []
    judge_rows: list[dict[str, int | float]] = []
    for row in rows:
        responder = _usage(row["responder_usage"], f"{label} responder usage")
        judge = _usage(row["judge_usage"], f"{label} judge usage")
        responder_rows.append(responder)
        judge_rows.append(judge)
        provider_input = int(responder["input_tokens"])
        if provider_input > 0:
            provider_counts.append(provider_input)
    accuracy = sample_expected["judge_accuracy"]
    minimum = int(campaign["min_target_questions"])
    target = float(campaign["accuracy_target"])
    expected_status = (
        "insufficient_questions"
        if len(rows) < minimum
        else "passed"
        if accuracy >= target
        else "failed"
    )
    expected_provider_status = (
        "unavailable"
        if not provider_counts
        else "complete"
        if len(provider_counts) == len(rows)
        else "partial"
    )
    expected_provider_compliance = (
        all(value <= int(campaign["max_prompt_tokens"]) for value in provider_counts)
        if provider_counts
        else None
    )
    report_expected = {
        "num_samples": 1,
        "num_questions": len(rows),
        "mean_f1": sample_expected["mean_f1"],
        "exact_match_rate": sample_expected["exact_match_rate"],
        "judge_accuracy": accuracy,
        "mean_context_tokens": sample_expected["mean_context_tokens"],
        "mean_prompt_token_proxy": sample_expected["mean_prompt_token_proxy"],
        "p95_prompt_token_proxy": _nearest_rank(sorted(prompt_counts), 0.95),
        "max_prompt_token_proxy_observed": max(prompt_counts, default=0),
        "mean_request_token_proxy": sample_expected["mean_request_token_proxy"],
        "mean_prompt_tokens": sample_expected["mean_prompt_tokens"],
        "p95_prompt_tokens": _nearest_rank(sorted(prompt_counts), 0.95),
        "mean_transcript_tokens": _ordinary_mean(
            [float(row["transcript_tokens"]) for row in rows]
        ),
        "mean_context_fraction": sample_expected["mean_context_fraction"],
        "mean_transcript_token_savings": sample_expected[
            "mean_transcript_token_savings"
        ],
        "max_prompt_tokens_observed": max(prompt_counts, default=0),
        "prompt_token_proxy_budget_compliance": True,
        "provider_prompt_budget_compliance": expected_provider_compliance,
        "provider_input_usage_status": expected_provider_status,
        "prompt_budget_compliance": True,
        "accuracy_target_met": expected_status == "passed",
        "target_status": expected_status,
        "responder_usage": _ordinary_sum_usage(responder_rows),
        "judge_usage": _ordinary_sum_usage(judge_rows),
        "by_category": _ordinary_category_metrics(rows),
    }
    for field, expected in report_expected.items():
        if canonical_sha256(shard.get(field)) != canonical_sha256(expected):
            raise AuditError(f"{label} aggregate mismatch: {field}")


def _validate_campaign_inputs(
    report: dict[str, Any],
    *,
    shard_root: Path,
    campaign_rows: dict[str, dict[str, Any]],
    plan: PopulationPlan,
) -> dict[str, FileSnapshot]:
    raw_inputs = require_list(report.get("inputs"), "report.inputs")
    if report.get("input_count") != len(raw_inputs) or len(raw_inputs) != len(
        plan.samples
    ):
        raise AuditError("campaign input count differs from the frozen shard population")
    root = shard_root.resolve()
    if not root.is_dir():
        raise AuditError(f"campaign shard-report root is not a directory: {root}")
    parsed_rows: list[dict[str, Any]] = []
    snapshots: dict[str, FileSnapshot] = {}
    digest_by_name: dict[str, str] = {}
    shard_questions: dict[str, tuple[str, str, str, dict[str, Any]]] = {}
    observed_samples: set[str] = set()
    for index, raw in enumerate(raw_inputs):
        row = require_mapping(raw, f"report.inputs[{index}]")
        if set(row) != {
            "name",
            "sha256",
            "num_samples",
            "num_questions",
            "target_status",
        }:
            raise AuditError("campaign input row has an unexpected shape")
        name = require_text(row.get("name"), f"report.inputs[{index}].name")
        if (
            Path(name).name != name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or ":" in name
        ):
            raise AuditError("campaign input names must be safe basenames")
        if name in snapshots:
            raise AuditError(f"campaign repeats shard report name {name}")
        expected_digest = require_sha256(
            row.get("sha256"), f"report.inputs[{index}].sha256"
        )
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AuditError("campaign shard report escapes its input root") from exc
        snapshot = read_file_snapshot(path, f"campaign shard report {name}")
        if snapshot.sha256 != expected_digest:
            raise AuditError(f"campaign shard report hash mismatch: {name}")
        shard = parse_json_object(snapshot.payload, f"campaign shard report {name}")
        samples = require_list(shard.get("samples"), f"campaign shard {name}.samples")
        if len(samples) != 1:
            raise AuditError(f"campaign shard {name} must contain exactly one sample")
        if require_int(
            row.get("num_samples"), f"campaign input {name}.num_samples"
        ) != len(samples):
            raise AuditError(f"campaign input sample count mismatch: {name}")
        if shard.get("num_samples") != len(samples):
            raise AuditError(f"campaign shard sample count mismatch: {name}")
        question_count = 0
        shard_sample_row: dict[str, Any] | None = None
        shard_sample_sha: str | None = None
        shard_question_rows: list[dict[str, Any]] = []
        for sample_index, raw_sample in enumerate(samples):
            sample = require_mapping(
                raw_sample,
                f"campaign shard {name}.samples[{sample_index}]",
            )
            sample_id = require_text(
                sample.get("sample_id"), f"campaign shard {name} sample_id"
            )
            sample_sha = require_sha256(
                sample.get("sample_sha256"),
                f"campaign shard {name} sample_sha256",
            )
            if (
                sample_sha not in plan.samples
                or plan.samples[sample_sha].sample_id != sample_id
            ):
                raise AuditError(f"campaign shard {name} contains an unlocked sample")
            if sample_sha in observed_samples:
                raise AuditError(f"campaign shard reports repeat sample {sample_sha}")
            observed_samples.add(sample_sha)
            shard_sample_row = sample
            shard_sample_sha = sample_sha
            questions = require_list(
                sample.get("question_results"),
                f"campaign shard {name} question_results",
            )
            if sample.get("num_questions") != len(questions):
                raise AuditError(f"campaign shard {name} question count mismatch")
            question_count += len(questions)
            for raw_question in questions:
                question = require_mapping(
                    raw_question, f"campaign shard {name} question"
                )
                question_id = require_text(
                    question.get("question_id"),
                    f"campaign shard {name} question_id",
                )
                if question_id in shard_questions:
                    raise AuditError(
                        f"question appears in multiple shard reports: {question_id}"
                    )
                if plan.question_to_sample.get(question_id) != sample_sha:
                    raise AuditError(
                        f"campaign shard {name} places a question in the wrong sample"
                    )
                shard_question_rows.append(question)
                shard_questions[question_id] = (
                    name,
                    expected_digest,
                    sample_id,
                    question,
                )
                if canonical_sha256(question) != canonical_sha256(
                    campaign_rows.get(question_id)
                ):
                    raise AuditError(
                        "campaign question row differs from its hashed shard: "
                        f"{question_id}"
                    )
        if require_int(
            row.get("num_questions"), f"campaign input {name}.num_questions"
        ) != question_count:
            raise AuditError(f"campaign input question count mismatch: {name}")
        if shard.get("num_questions") != question_count:
            raise AuditError(f"campaign shard total question count mismatch: {name}")
        if row.get("target_status") != shard.get("target_status"):
            raise AuditError(f"campaign input target status mismatch: {name}")
        if shard_sample_row is None or shard_sample_sha is None:
            raise AuditError(f"campaign shard {name} has no sample")
        _validate_shard_identity_and_aggregates(
            shard,
            campaign=report,
            rows=shard_question_rows,
            sample_sha=shard_sample_sha,
            sample_row=shard_sample_row,
            plan=plan,
            label=f"campaign shard {name}",
        )
        snapshots[name] = snapshot
        digest_by_name[name] = expected_digest
        parsed_rows.append(dict(row))
    if parsed_rows != sorted(
        parsed_rows, key=lambda row: (row["sha256"], row["name"])
    ):
        raise AuditError("campaign inputs are not in canonical digest/name order")
    if len(set(digest_by_name.values())) != len(digest_by_name):
        raise AuditError("campaign shard reports reuse a content digest")
    if report.get("input_set_sha256") != canonical_sha256(
        sorted(digest_by_name.values())
    ):
        raise AuditError("campaign input-set digest mismatch")
    if set(shard_questions) != set(campaign_rows):
        raise AuditError(
            "campaign shard reports do not cover the exact campaign questions"
        )
    if observed_samples != set(plan.samples):
        raise AuditError("campaign shard reports do not cover the exact campaign samples")
    sources = require_mapping(report.get("question_sources"), "report.question_sources")
    for question_id, (name, digest, sample_id, _question) in shard_questions.items():
        source = require_mapping(
            sources.get(question_id), f"question source {question_id}"
        )
        expected_sample_sha = plan.question_to_sample[question_id]
        expected_sample_id = plan.samples[expected_sample_sha].sample_id
        if sample_id != expected_sample_id:
            raise AuditError(
                f"campaign question lineage has the wrong sample: {question_id}"
            )
        if source != {
            "report_name": name,
            "report_sha256": digest,
            "sample_id": expected_sample_id,
            "sample_sha256": expected_sample_sha,
        }:
            raise AuditError(
                "campaign question lineage differs from its hashed shard: "
                f"{question_id}"
            )
    return snapshots


__all__ = ["_validate_campaign_inputs"]
