"""Strict CSV row parsing, validation, and aggregate reduction."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

if __package__:
    from .merge_locked_v3_recall_json import _strict_json, canonical_sha256
    from .merge_locked_v3_recall_schema import (
        CSV_SCHEMA,
        MAX_CSV_FIELD_CHARS,
        ExpectedRecallQuestion,
        ExpectedRecallShard,
        RecallCampaignError,
        RecallCsvShard,
        _FIXED_FLOAT_RE,
        _JSON_FIELDS,
        _NONNEGATIVE_INT_RE,
        _OPTIONAL_BINARY_FIELDS,
        _OPTIONAL_FIXED_FLOAT_FIELDS,
        _OPTIONAL_INT_FIELDS,
        _PIPE_FIELDS,
        _REQUIRED_BINARY_FIELDS,
        _REQUIRED_FIXED_FLOAT_FIELDS,
        _REQUIRED_INT_FIELDS,
        _TEXT_FIELDS,
    )
else:  # Support direct execution of the facade script.
    from merge_locked_v3_recall_json import _strict_json, canonical_sha256
    from merge_locked_v3_recall_schema import (
        CSV_SCHEMA,
        MAX_CSV_FIELD_CHARS,
        ExpectedRecallQuestion,
        ExpectedRecallShard,
        RecallCampaignError,
        RecallCsvShard,
        _FIXED_FLOAT_RE,
        _JSON_FIELDS,
        _NONNEGATIVE_INT_RE,
        _OPTIONAL_BINARY_FIELDS,
        _OPTIONAL_FIXED_FLOAT_FIELDS,
        _OPTIONAL_INT_FIELDS,
        _PIPE_FIELDS,
        _REQUIRED_BINARY_FIELDS,
        _REQUIRED_FIXED_FLOAT_FIELDS,
        _REQUIRED_INT_FIELDS,
        _TEXT_FIELDS,
    )

def _required_integer(value: str, label: str) -> int:
    if _NONNEGATIVE_INT_RE.fullmatch(value) is None:
        raise RecallCampaignError(f"{label} must be a canonical non-negative integer")
    return int(value)


def _optional_integer(value: str, label: str) -> int | None:
    return None if value == "" else _required_integer(value, label)


def _required_binary(value: str, label: str) -> bool:
    if value not in {"0", "1"}:
        raise RecallCampaignError(f"{label} must be 0 or 1")
    return value == "1"


def _optional_binary(value: str, label: str) -> bool | None:
    return None if value == "" else _required_binary(value, label)


def _required_fixed_float(
    value: str,
    label: str,
    *,
    maximum: float | None = None,
) -> float:
    if _FIXED_FLOAT_RE.fullmatch(value) is None:
        raise RecallCampaignError(f"{label} must be a finite four-decimal number")
    parsed = float(value)
    if not math.isfinite(parsed) or (maximum is not None and parsed > maximum):
        raise RecallCampaignError(f"{label} is outside its valid range")
    return parsed


def _optional_fixed_float(
    value: str,
    label: str,
    *,
    maximum: float | None = None,
) -> float | None:
    if value == "":
        return None
    return _required_fixed_float(value, label, maximum=maximum)


def _pipe_values(value: str, label: str) -> tuple[str, ...]:
    if value == "":
        return ()
    parts = tuple(value.split("|"))
    if any(not part for part in parts):
        raise RecallCampaignError(f"{label} contains an empty pipe-delimited value")
    return parts


def _parse_csv_row(cells: Sequence[str], label: str) -> dict[str, Any]:
    if len(cells) != len(CSV_SCHEMA):
        raise RecallCampaignError(
            f"{label} has {len(cells)} cells; expected {len(CSV_SCHEMA)}"
        )
    row: dict[str, Any] = dict(zip(CSV_SCHEMA, cells, strict=True))
    for field_name in _REQUIRED_BINARY_FIELDS:
        row[field_name] = _required_binary(
            row[field_name], f"{label}.{field_name}"
        )
    for field_name in _OPTIONAL_BINARY_FIELDS:
        row[field_name] = _optional_binary(
            row[field_name], f"{label}.{field_name}"
        )
    for field_name in _REQUIRED_INT_FIELDS:
        row[field_name] = _required_integer(
            row[field_name], f"{label}.{field_name}"
        )
    for field_name in _OPTIONAL_INT_FIELDS:
        row[field_name] = _optional_integer(
            row[field_name], f"{label}.{field_name}"
        )
    for field_name in _REQUIRED_FIXED_FLOAT_FIELDS:
        row[field_name] = _required_fixed_float(
            row[field_name],
            f"{label}.{field_name}",
            maximum=1.0 if field_name == "best_f1" else None,
        )
    for field_name in _OPTIONAL_FIXED_FLOAT_FIELDS:
        row[field_name] = _optional_fixed_float(
            row[field_name], f"{label}.{field_name}", maximum=1.0
        )
    for field_name in _PIPE_FIELDS:
        row[field_name] = _pipe_values(row[field_name], f"{label}.{field_name}")
    for field_name in _JSON_FIELDS:
        parsed = _strict_json(row[field_name], f"{label}.{field_name}")
        if not isinstance(parsed, list):
            raise RecallCampaignError(f"{label}.{field_name} must be a JSON list")
        row[field_name] = parsed
    for field_name in _TEXT_FIELDS:
        value = row[field_name]
        if value.strip().casefold() in {
            "nan",
            "+nan",
            "-nan",
            "inf",
            "+inf",
            "-inf",
            "infinity",
            "+infinity",
            "-infinity",
        }:
            raise RecallCampaignError(f"{label}.{field_name} is non-finite")
    return row


def _same_four_decimals(actual: float, expected: float) -> bool:
    return f"{actual:.4f}" == f"{expected:.4f}"


def _validate_answer_value(row: dict[str, Any], label: str) -> None:
    expected = row["answer_value_components_expected"]
    found = row["answer_value_components_found"]
    recall = row["answer_value_component_recall"]
    all_components = row["all_answer_value_components"]
    mask = row["answer_value_component_hit_mask"]
    metric = row["answer_value_metric_kind"]
    values = (expected, found, recall, all_components)
    if all(value is None for value in values):
        if mask or metric:
            raise RecallCampaignError(
                f"{label} has answer-value metadata without a scored value set"
            )
        return
    if any(value is None for value in values):
        raise RecallCampaignError(f"{label} has a partial answer-value record")
    assert isinstance(expected, int)
    assert isinstance(found, int)
    assert isinstance(recall, float)
    assert isinstance(all_components, bool)
    if expected < 1 or found > expected:
        raise RecallCampaignError(f"{label} has invalid answer-value counts")
    if len(mask) != expected or any(value not in {"0", "1"} for value in mask):
        raise RecallCampaignError(f"{label} has an invalid answer-value hit mask")
    if sum(value == "1" for value in mask) != found:
        raise RecallCampaignError(f"{label} answer-value mask disagrees with found")
    if not _same_four_decimals(recall, found / expected):
        raise RecallCampaignError(f"{label} answer-value recall is not recomputable")
    if all_components != (found == expected):
        raise RecallCampaignError(f"{label} answer-value completion is inconsistent")
    if not metric.strip():
        raise RecallCampaignError(f"{label} scored answer values lack a metric kind")


def _validate_retrieval_identity(
    row: dict[str, Any], retrieval: Mapping[str, Any], label: str
) -> None:
    prefix_fields = {
        "coverage_selector_prefix_model_id": "coverage_selector_prefix_model_id",
        "coverage_selector_prefix_model_revision": "coverage_selector_prefix_revision",
        "coverage_selector_prefix_checkpoint_sha256": (
            "coverage_selector_prefix_checkpoint_sha256"
        ),
        "coverage_selector_prefix_device": "coverage_selector_prefix_device",
        "coverage_selector_prefix_dtype": "coverage_selector_prefix_dtype",
        "coverage_selector_prefix_layers": "coverage_selector_prefix_layers",
        "coverage_selector_prefix_attention_layer": (
            "coverage_selector_attention_layer"
        ),
    }
    for csv_field, policy_field in prefix_fields.items():
        expected = retrieval.get(policy_field)
        if row[csv_field] != expected:
            raise RecallCampaignError(
                f"{label}.{csv_field} does not match frozen retrieval identity"
            )
    expected_closure = retrieval.get("allow_selected_scope_fixed_k_closure")
    if not isinstance(expected_closure, bool):
        raise RecallCampaignError(
            "frozen retrieval identity lacks a boolean closure policy"
        )
    if row["coverage_selector_allow_selected_scope_fixed_k_closure"] != (
        expected_closure
    ):
        raise RecallCampaignError(f"{label} closure policy identity mismatch")

    choice_fields = {
        "coverage_selector_score_provider_model_id": (
            "coverage_selector_choice_model_id"
        ),
        "coverage_selector_score_provider_model_revision": (
            "coverage_selector_choice_revision"
        ),
        "coverage_selector_score_provider_checkpoint_sha256": (
            "coverage_selector_choice_checkpoint_sha256"
        ),
        "coverage_selector_score_provider_device": (
            "coverage_selector_choice_device"
        ),
        "coverage_selector_score_provider_dtype": "coverage_selector_choice_dtype",
    }
    active_choice = row["coverage_selector_score_provider_forward_passes"] > 0 or any(
        bool(row[field_name]) for field_name in choice_fields
    )
    if active_choice:
        for csv_field, policy_field in choice_fields.items():
            if row[csv_field] != retrieval.get(policy_field):
                raise RecallCampaignError(
                    f"{label}.{csv_field} does not match frozen choice identity"
                )


def _validate_question_row(
    row: dict[str, Any],
    expected: ExpectedRecallQuestion,
    retrieval: Mapping[str, Any],
    label: str,
) -> None:
    if row["question_id"] != expected.question_id:
        raise RecallCampaignError(f"{label} question ID order/substitution mismatch")
    if row["category"] != expected.category:
        raise RecallCampaignError(f"{label} question category mismatch")
    for field_name in (
        "retrieved_source_ids",
        "raw_retrieved_source_ids",
    ):
        values = row[field_name]
        if len(values) != len(set(values)):
            raise RecallCampaignError(f"{label}.{field_name} contains duplicates")

    expected_sources = set(expected.evidence_sources)
    retrieved_sources = set(row["retrieved_source_ids"])
    coverage = len(expected_sources & retrieved_sources) / len(expected_sources)
    reported_coverage = row["evidence_source_recall"]
    if reported_coverage is None or not _same_four_decimals(
        reported_coverage, coverage
    ):
        raise RecallCampaignError(f"{label} evidence-source recall is not recomputable")
    if row["all_evidence_sources"] != (coverage == 1.0):
        raise RecallCampaignError(f"{label} all-evidence-source flag is inconsistent")
    row["evidence_any_source"] = bool(expected_sources & retrieved_sources)

    raw_sources = set(row["raw_retrieved_source_ids"])
    raw_coverage = row["raw_evidence_source_recall"]
    raw_all = row["raw_all_evidence_sources"]
    if raw_sources:
        expected_raw = len(expected_sources & raw_sources) / len(expected_sources)
        if raw_coverage is None or not _same_four_decimals(raw_coverage, expected_raw):
            raise RecallCampaignError(
                f"{label} raw evidence-source recall is not recomputable"
            )
        if raw_all != (expected_raw == 1.0):
            raise RecallCampaignError(
                f"{label} raw all-evidence-source flag is inconsistent"
            )
    elif raw_coverage is not None or raw_all is not None:
        raise RecallCampaignError(
            f"{label} reports raw source coverage without raw sources"
        )

    _validate_answer_value(row, label)
    _validate_retrieval_identity(row, retrieval, label)
    if row["coverage_selector_score_provider_retained_state_bytes"] != 0:
        raise RecallCampaignError(f"{label} retained score-provider token state")
    if row["coverage_selector_retained_state_bytes"] != 0:
        raise RecallCampaignError(f"{label} retained selector token state")
    if row["closure_applied"] and not row["closure_scope"].strip():
        raise RecallCampaignError(f"{label} applied closure lacks a scope")
    if not row["closure_applied"] and row["closure_scope"].strip():
        raise RecallCampaignError(f"{label} reports closure scope without closure")
    if row["closure_applied"] and row["closure_global_recall_guaranteed"] is None:
        raise RecallCampaignError(f"{label} applied closure lacks a recall guarantee")
    if (
        not row["closure_applied"]
        and row["closure_global_recall_guaranteed"] is not None
    ):
        raise RecallCampaignError(
            f"{label} reports a closure guarantee without closure"
        )
    if (
        row["closure_scope"] == "selected_scope_policy"
        and row["closure_global_recall_guaranteed"] is not False
    ):
        raise RecallCampaignError(
            f"{label} selected-scope closure must be explicitly non-global"
        )


def _parse_shard(
    shard: RecallCsvShard,
    expected: ExpectedRecallShard,
    retrieval: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, str]:
    try:
        text = shard.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecallCampaignError(
            f"recall CSV at offset {shard.sample_offset} is not UTF-8"
        ) from exc
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(MAX_CSV_FIELD_CHARS)
        raw_rows = list(
            csv.reader(io.StringIO(text, newline=""), strict=True)
        )
    except csv.Error as exc:
        raise RecallCampaignError(
            f"cannot parse recall CSV at offset {shard.sample_offset}: {exc}"
        ) from exc
    finally:
        csv.field_size_limit(previous_limit)
    if not raw_rows:
        raise RecallCampaignError(f"recall CSV at offset {shard.sample_offset} is empty")
    if tuple(raw_rows[0]) != CSV_SCHEMA:
        raise RecallCampaignError(
            f"recall CSV at offset {shard.sample_offset} has a non-canonical schema"
        )
    body = raw_rows[1:]
    if len(body) != len(expected.questions):
        raise RecallCampaignError(
            f"recall CSV at offset {shard.sample_offset} has {len(body)} rows; "
            f"expected {len(expected.questions)}"
        )
    parsed: list[dict[str, Any]] = []
    for index, (cells, question) in enumerate(
        zip(body, expected.questions, strict=True), start=1
    ):
        label = f"offset[{shard.sample_offset}].row[{index}]"
        row = _parse_csv_row(cells, label)
        _validate_question_row(row, question, retrieval, label)
        parsed.append(row)
    digest = hashlib.sha256(shard.payload).hexdigest()
    question_ids_digest = canonical_sha256([row["question_id"] for row in parsed])
    return parsed, digest, question_ids_digest


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _rate(values: Sequence[bool]) -> float:
    return math.fsum(1.0 if value else 0.0 for value in values) / len(values) if values else 0.0


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evidence_rows = [row for row in rows if row["evidence_source_recall"] is not None]
    raw_rows = [row for row in rows if row["raw_evidence_source_recall"] is not None]
    answer_rows = [row for row in rows if row["answer_value_component_recall"] is not None]
    calls = [
        row
        for row in rows
        if row["coverage_selector_inspected"] > 0
        or row["coverage_selector_frontier_candidates"] > 0
        or bool(row["coverage_selector_operator"])
        or bool(row["coverage_selector_status"])
        or bool(row["coverage_selector_bypass_reason"])
        or bool(row["coverage_selector_fallback_reason"])
        or bool(row["coverage_selector_score_provider_fallback"])
    ]

    def bypassed(row: dict[str, Any]) -> bool:
        return row["coverage_selector_status"] == "bypassed" or bool(
            row["coverage_selector_bypass_reason"]
        )

    routed = [
        row
        for row in calls
        if not bypassed(row)
        and row["coverage_selector_routed_frontier_exhaustive"] is not None
    ]
    active = [
        row
        for row in calls
        if not bypassed(row)
        and row["coverage_selector_active_partition_exhaustive"] is not None
    ]
    answer_expected = sum(
        int(row["answer_value_components_expected"]) for row in answer_rows
    )
    answer_found = sum(
        int(row["answer_value_components_found"]) for row in answer_rows
    )
    score_provider_retained = [
        row["coverage_selector_score_provider_retained_state_bytes"]
        for row in rows
    ]
    selector_retained = [
        row["coverage_selector_retained_state_bytes"] for row in rows
    ]
    return {
        "questions": len(rows),
        "literal": {
            "haystack_hits": sum(row["in_haystack"] for row in rows),
            "haystack_recall": _rate([row["in_haystack"] for row in rows]),
            "context_hits": sum(row["in_context"] for row in rows),
            "context_recall": _rate([row["in_context"] for row in rows]),
            "header_hits": sum(row["in_header"] for row in rows),
            "header_recall": _rate([row["in_header"] for row in rows]),
            "expansion_hits": sum(row["in_expansions"] for row in rows),
            "expansion_recall": _rate([row["in_expansions"] for row in rows]),
            "mean_best_f1": _mean([row["best_f1"] for row in rows]),
            "mean_context_tokens": _mean(
                [float(row["context_tokens"]) for row in rows]
            ),
        },
        "evidence": {
            "scored_questions": len(evidence_rows),
            "any_source_hits": sum(row["evidence_any_source"] for row in evidence_rows),
            "any_source_recall": _rate(
                [row["evidence_any_source"] for row in evidence_rows]
            ),
            "mean_source_recall": _mean(
                [row["evidence_source_recall"] for row in evidence_rows]
            ),
            "all_source_hits": sum(
                row["all_evidence_sources"] for row in evidence_rows
            ),
            "all_source_recall": _rate(
                [row["all_evidence_sources"] for row in evidence_rows]
            ),
        },
        "raw_evidence": {
            "scored_questions": len(raw_rows),
            "mean_source_recall": (
                _mean([row["raw_evidence_source_recall"] for row in raw_rows])
                if raw_rows
                else None
            ),
            "all_source_hits": sum(
                row["raw_all_evidence_sources"] for row in raw_rows
            ),
            "all_source_recall": (
                _rate([row["raw_all_evidence_sources"] for row in raw_rows])
                if raw_rows
                else None
            ),
        },
        "answer_value": {
            "scored_questions": len(answer_rows),
            "components_expected": answer_expected,
            "components_found": answer_found,
            "component_weighted_recall": (
                answer_found / answer_expected if answer_expected else None
            ),
            "macro_component_recall": (
                _mean([row["answer_value_component_recall"] for row in answer_rows])
                if answer_rows
                else None
            ),
            "all_component_questions": sum(
                row["all_answer_value_components"] for row in answer_rows
            ),
            "all_component_question_recall": (
                _rate([row["all_answer_value_components"] for row in answer_rows])
                if answer_rows
                else None
            ),
        },
        "selector_diagnostics": {
            "calls": len(calls),
            "bypasses": sum(bypassed(row) for row in calls),
            "selector_fallbacks": sum(
                bool(row["coverage_selector_fallback_reason"]) and not bypassed(row)
                for row in calls
            ),
            "score_provider_fallbacks": sum(
                bool(row["coverage_selector_score_provider_fallback"])
                for row in calls
            ),
            "degraded_calls": sum(
                (
                    bool(row["coverage_selector_fallback_reason"])
                    and not bypassed(row)
                )
                or bool(row["coverage_selector_score_provider_fallback"])
                for row in calls
            ),
            "inspected_total": sum(
                row["coverage_selector_inspected"] for row in calls
            ),
            "classified_total": sum(
                row["coverage_selector_classified"] for row in calls
            ),
            "score_provider_forward_passes": sum(
                row["coverage_selector_score_provider_forward_passes"]
                for row in calls
            ),
            "routed_frontier_audited_calls": len(routed),
            "routed_frontier_exhaustive_calls": sum(
                row["coverage_selector_routed_frontier_exhaustive"] is True
                for row in routed
            ),
            "routed_frontier_non_exhaustive_calls": sum(
                row["coverage_selector_routed_frontier_exhaustive"] is False
                for row in routed
            ),
            "active_partition_audited_calls": len(active),
            "active_partition_exhaustive_calls": sum(
                row["coverage_selector_active_partition_exhaustive"] is True
                for row in active
            ),
            "active_partition_non_exhaustive_calls": sum(
                row["coverage_selector_active_partition_exhaustive"] is False
                for row in active
            ),
            "active_partition_semantically_complete_calls": sum(
                row["coverage_selector_active_partition_semantically_complete"] is True
                for row in active
            ),
            "active_partition_semantically_incomplete_calls": sum(
                row["coverage_selector_active_partition_semantically_complete"] is False
                for row in active
            ),
            "active_partition_candidates_admitted_total": sum(
                row["coverage_selector_active_partition_candidates_admitted"]
                for row in active
            ),
            "active_partition_structural_overflow_total": sum(
                row["coverage_selector_active_partition_structural_overflow"]
                for row in active
            ),
            "selected_scope_structurally_complete_calls": sum(
                row["coverage_selector_selected_scope_structurally_complete"] is True
                for row in calls
            ),
            "global_semantic_complete_calls": sum(
                row["coverage_selector_global_semantic_complete"] is True
                for row in calls
            ),
            "closure_calls": sum(row["closure_applied"] for row in calls),
            "selected_scope_policy_closure_calls": sum(
                row["closure_scope"] == "selected_scope_policy" for row in calls
            ),
            "globally_recall_guaranteed_closure_calls": sum(
                row["closure_applied"]
                and row["closure_global_recall_guaranteed"] is True
                for row in calls
            ),
            "cardinality_deficit_calls": sum(
                row["coverage_selector_cardinality_deficit"] > 0 for row in calls
            ),
            "cardinality_deficit_total": sum(
                row["coverage_selector_cardinality_deficit"] for row in calls
            ),
            "max_retained_state_bytes": max(
                (
                    max(
                        row["coverage_selector_score_provider_retained_state_bytes"],
                        row["coverage_selector_retained_state_bytes"],
                    )
                    for row in rows
                ),
                default=0,
            ),
        },
        "reported_zero_state_consistency": {
            "evidence_scope": "self_reported_csv_fields_only",
            "independently_verified": False,
            "questions_checked": len(rows),
            "score_provider_zero_state_questions": sum(
                value == 0 for value in score_provider_retained
            ),
            "selector_zero_state_questions": sum(
                value == 0 for value in selector_retained
            ),
            "retained_state_violation_questions": sum(
                score_value != 0 or selector_value != 0
                for score_value, selector_value in zip(
                    score_provider_retained,
                    selector_retained,
                    strict=True,
                )
            ),
            "score_provider_retained_state_bytes_total": sum(
                score_provider_retained
            ),
            "selector_retained_state_bytes_total": sum(selector_retained),
            "max_retained_state_bytes": max(
                (*score_provider_retained, *selector_retained),
                default=0,
            ),
        },
    }


def _category_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"] or "uncategorized"].append(row)
    return {
        category: _aggregate_rows(grouped[category])
        for category in sorted(grouped)
    }
