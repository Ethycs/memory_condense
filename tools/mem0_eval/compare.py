"""Strict paired comparison of frozen treatment and Mem0 campaign reports.

This module intentionally accepts only the public dictionaries emitted by
``merge_benchmark_reports`` and ``merge_mem0_shard_reports``.  It does not
load unvalidated shard reports and it does not provide a permissive file CLI.
Both campaign objects are nevertheless treated as hostile input here: their
public schemas, identities, primitive question rows, aggregates, and hashes
are checked again before a paired result is emitted.

The Mem0 arm has no exact evidence-provenance primitive.  Its request-window
attribution is therefore preserved as not-applicable for provenance metric
comparison; it is never converted to a zero score.  A valid metric comparison
is also distinct from a certified comparison.  In particular, the latter is
impossible while ``production_binding_certified`` is false.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .source_compat import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.eval.benchmark import build_qa_prompt, exact_match, f1_score
from memory_condense.eval.validation_profile import LONGMEMEVAL_1M_95_PROFILE

from .policy import (
    MEM0_EMBEDDER_CHECKPOINT_SHA256,
    MEM0_EMBEDDER_DIMENSION,
    MEM0_EMBEDDER_DTYPE,
    MEM0_EMBEDDER_MODEL,
    MEM0_EMBEDDER_PROVIDER,
    MEM0_EMBEDDER_REVISION,
)
from .prompt_pack import (
    MEM0_CONFIGURED_RECENT_WINDOW,
    MEM0_EFFECTIVE_RECENT_WINDOW,
    MEM0_PROMPT_PACK_PROTOCOL,
    MEM0_RECENT_WINDOW_SEMANTICS,
)


COMPARISON_SCHEMA_VERSION = 2
COMPARISON_REPORT_TYPE = "treatment_mem0_paired_campaign"
TREATMENT_REPORT_TYPE = "benchmark_campaign"
MEM0_REPORT_TYPE = "mem0_longmemeval_campaign"
MEM0_ARM_ID = "mem0_oss_2_0_18_direct_1m_v1"
COMMON_QUESTION_SCHEMA = "memory-condense-common-qa-result-v1"
FROZEN_QUESTION_COUNT = 100
FROZEN_SHARD_COUNT = 10
FROZEN_OFFSETS = tuple(range(0, 100, 10))
FROZEN_SAMPLE_SHA256_BY_OFFSET = MappingProxyType({
    "0": "41e52404d4f323c7add44a59a2faf8a58a95125d8e291cd9d118560833c5e14d",
    "10": "155ed6672dd876f633c1a2498fd6c503da10e65710011e7aae1408a140dca62c",
    "20": "60962645b4ebd93772e9fbda0d9c4d260403d4aa5551a95b79b85d4f741912b0",
    "30": "de1c162ff6f07e0bd3b35559d1755dc246400c14b7ba2f4f11300995dd688cd3",
    "40": "b49099d223163bab29633b9700b9891a8a34c1ea049e62a1cae6306e185575bd",
    "50": "f466bee5866031f3dd07164cbbfe9726f45540394c7bd9ce5452d1ece8b0b0ea",
    "60": "5583c05a9e877a75a4e6def0fa74ab00be2a7c7b0a3c0b5bc59a7cc53ba30fd1",
    "70": "558bc25d0a8851bbad529445ab624a17e7b814404b8d8c4ad99235611aeade0f",
    "80": "be4d976199fd3e08551bf106a2bf1c9d9e67bffeb8fec1cad2c29e0a19231e74",
    "90": "a9c707cedfbdb22bd11e40d2747e85b940dbc6850d3dc1524b97645b7b096576",
})
PROMPT_CAP_SEMANTICS = (
    "local_prompt_token_proxy_with_provider_usage_postcheck_v1"
)
MEM0_ATTRIBUTION_KIND = "request_window_non_evidence"
MEM0_SOURCE_COVERAGE_STATUS = "unavailable_exact_source_provenance"
MEM0_SOURCE_COVERAGE_REASON = (
    "mem0_request_window_attribution_is_not_exact_evidence_provenance"
)
FROZEN_DATASET_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
FROZEN_SPLIT_MANIFEST_SHA256 = (
    "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
)
FROZEN_SOURCE_IMPLEMENTATION_SHA256 = (
    "452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83"
)
FROZEN_SOURCE_ENVIRONMENT_SHA256 = (
    "058083871240979257ada7ca4c71dd816fee64792b275ef11e4857c9f5ebba33"
)
FROZEN_SOURCE_POLICY_SHA256 = (
    "5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883"
)
FROZEN_RETRIEVAL_IDENTITY_SHA256 = (
    "08ffd89a8b30803a0d8121445c1d54171120b1f1e51c866d4015f2d36b87cbaf"
)
FROZEN_RETRIEVAL_FIELDS = (
    "mode", "k", "ef_search", "alpha", "candidates", "neighbor_radius",
    "neighbor_slots", "neighbor_replacement_slots", "max_prompt_tokens",
    "chunker_min_tokens", "chunker_max_tokens", "source_slots",
    "source_activation_k", "source_candidate_pool", "source_local_search",
    "source_tfisf_activation", "source_tfisf_slots", "source_hsc_activation",
    "source_hsc_slots", "source_hsc_hops", "source_hsc_chunk_slots",
    "source_partition_routing", "source_partition_slots",
    "source_partition_separator", "neighbor_direction",
    "role_aware_retrieval", "role_user_weight", "role_assistant_weight",
    "role_system_weight", "consolidation_chunk_slots", "consolidation_hops",
    "consolidation_candidates", "consolidation_diffusion_width",
    "consolidation_min_count", "consolidation_expansion_tokens",
    "consolidation_training_expansion_tokens",
    "consolidation_budget_aware_packing", "consolidation_training_k",
    "consolidation_max_event_nodes", "consolidation_new_event_nodes",
    "consolidation_max_training_prompt_tokens",
    "consolidation_query_aware_sentence_packing",
    "consolidation_max_sentences_per_expansion",
    "consolidation_information_gain_packing",
    "consolidation_min_information_gain_per_token",
    "consolidation_source_metadata_packing", "coverage_selection",
    "coverage_selector_backend", "coverage_selector_model",
    "coverage_selector_dtype", "coverage_selector_candidate_pool",
    "coverage_selector_candidate_tokens", "coverage_selector_query_tokens",
    "coverage_selector_max_workspace_tokens",
    "coverage_selector_max_new_tokens", "coverage_selector_null_threshold",
    "coverage_selector_uncertainty_entropy",
    "coverage_selector_prefix_layers", "coverage_selector_attention_layer",
    "coverage_selector_merge_similarity",
    "coverage_selector_same_source_merge_similarity",
    "coverage_selector_strict", "allow_selected_scope_fixed_k_closure",
    "coverage_selector_prefix_model_id", "coverage_selector_prefix_revision",
    "coverage_selector_prefix_checkpoint_sha256",
    "coverage_selector_prefix_device", "coverage_selector_prefix_dtype",
    "coverage_selector_choice_model_id", "coverage_selector_choice_revision",
    "coverage_selector_choice_checkpoint_sha256",
    "coverage_selector_choice_device", "coverage_selector_choice_dtype",
    "coverage_selector_choice_batch_size",
    "coverage_selector_choice_max_candidates",
    "coverage_selector_choice_query_tokens",
    "coverage_selector_choice_candidate_tokens",
    "coverage_selector_choice_max_prompt_tokens",
    "coverage_selector_choice_max_workspace_tokens",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JUDGE_RE = re.compile(
    r"^\s*(CORRECT|INCORRECT)\b(?P<remainder>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_SECRET_KEYS = {
    "api_key",
    "api-key",
    "x_api_key",
    "x-api-key",
    "authorization",
    "proxy_authorization",
    "proxy-authorization",
    "auth",
    "cookie",
    "set_cookie",
    "set-cookie",
    "password",
    "passwd",
    "secret",
    "secret_key",
    "token",
    "access_token",
    "refresh_token",
    "sas_token",
    "client_secret",
    "client_key",
    "credentials",
    "private_key",
    "signing_key",
    "connection_string",
}
_FORBIDDEN_SECRET_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_auth_token",
    "_bearer_token",
    "_id_token",
    "_oauth_token",
    "_sas_token",
    "_security_token",
    "_session_token",
    "_password",
    "_secret",
    "_secret_key",
    "_access_token",
    "_refresh_token",
    "_private_key",
    "_signing_key",
    "_connection_string",
)
_SECRET_VALUE_RE = re.compile(
    r"(?:^\s*(?:bearer|basic)\s+\S+|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk|ghp|github_pat|xox[baprs]|AIza)[-_][A-Za-z0-9_-]{8,}|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)


class PairedComparisonError(ValueError):
    """A campaign pair cannot support the frozen fair comparison."""


_USAGE_FIELDS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "elapsed_s",
        "calls",
    }
)

_DISTRIBUTION_FIELDS = frozenset(
    {"count", "min", "mean", "p50", "p90", "p95", "p99", "max", "values"}
)

_TREATMENT_TOP_FIELDS = frozenset(
    {
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
)

_MEM0_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "report_type",
        "arm_id",
        "run_status",
        "inputs",
        "input_count",
        "input_set_sha256",
        "identity",
        "model_identity",
        "runtime_model_identity_probe",
        "config",
        "benchmark",
        "dataset_sha256",
        "split_manifest_sha256",
        "benchmark_split",
        "implementation_sha256",
        "environment_lock_sha256",
        "policy_manifest_sha256",
        "responder_model",
        "judge_model",
        "recent_window",
        "max_prompt_tokens",
        "prompt_token_proxy_identity",
        "responder_output_token_reserve",
        "evaluation_protocol",
        "population_identity",
        "prompt_identity",
        "sample_offsets",
        "num_samples",
        "num_questions",
        "question_results",
        "common_question_result_schema",
        "common_question_results",
        "question_sources",
        "raw_input_totals",
        "operation_totals",
        "mean_f1",
        "exact_match_rate",
        "judge_accuracy",
        "mean_context_tokens",
        "mean_prompt_token_proxy",
        "p95_prompt_token_proxy",
        "context_token_distribution",
        "prompt_token_proxy_distribution",
        "request_token_proxy_distribution",
        "max_prompt_token_proxy_observed",
        "prompt_token_proxy_budget_compliance",
        "provider_prompt_budget_compliance",
        "provider_input_usage_status",
        "external_provider_usage_certified",
        "responder_usage",
        "judge_usage",
        "provenance",
        "source_coverage_status",
        "source_coverage",
        "exact_provenance_requirement_met",
        "local_request_token_state_contract_satisfied",
        "zero_persisted_transformer_token_state_verified",
        "external_provider_persistence_certified",
        "production_binding_certified",
        "certification_status",
        "locked_population_verified",
        "local_comparison_protocol_verified",
        "accuracy_target",
        "min_target_questions",
        "metric_accuracy_target_met",
        "accuracy_target_met",
        "target_status",
    }
)

_TREATMENT_QUESTION_FIELDS = frozenset(
    {
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
)

_MEM0_QUESTION_FIELDS = frozenset(
    {
        "question_index",
        "question_id",
        "question",
        "dated_question",
        "gold_answer",
        "prediction",
        "category",
        "retrieval_row_sha256",
        "query_sha256",
        "prompt_pack_protocol",
        "context",
        "context_sha256",
        "context_tokens",
        "messages",
        "messages_sha256",
        "prompt_token_proxy",
        "max_prompt_tokens",
        "residual_prompt_tokens",
        "prompt_token_proxy_identity",
        "raw_pool_count",
        "raw_pool_sha256",
        "raw_memory_tokens",
        "packed_count",
        "packed_memory_tokens",
        "packed_pool_sha256",
        "search_latency_s",
        "attribution_kind",
        "supports_exact_source_provenance",
        "exact_match",
        "f1",
        "judge_correct",
        "judge_reasoning",
        "provider_prompt_budget_compliant",
        "configured_recent_window",
        "effective_recent_window",
        "recent_window_semantics",
        "responder_usage",
        "judge_usage",
    }
)

_COMMON_QUESTION_FIELDS = frozenset(
    {
        "question_id",
        "predicted_answer",
        "judge_correct",
        "f1",
        "exact_match",
        "context_tokens",
        "prompt_token_proxy",
        "responder_usage",
        "judge_usage",
    }
)

_TREATMENT_PROTOCOL_FIELDS = frozenset(
    {
        "responder_model",
        "judge_model",
        "embedding_device",
        "benchmark_format",
        "use_judge",
        "provider_retries",
        "max_provider_calls",
        "max_prompt_tokens",
        "prompt_cap_semantics",
        "prompt_token_proxy_identity",
        "responder_output_token_reserve",
        "recent_window",
        "accuracy_target",
        "min_target_questions",
        "stress_context_tokens",
        "stress_questions",
        "stress_question_offset",
        "max_samples",
    }
)

_MEM0_PROTOCOL_FIELDS = frozenset(
    {
        "responder_model",
        "judge_model",
        "use_judge",
        "provider_retries",
        "max_provider_calls_per_shard",
        "max_prompt_tokens",
        "prompt_cap_semantics",
        "prompt_token_proxy_identity",
        "responder_output_token_reserve",
        "recent_window",
        "accuracy_target",
        "min_target_questions",
        "stress_context_tokens",
        "stress_questions",
        "stress_question_offset",
        "max_samples",
        "sample_offsets",
    }
)

_SHARED_PROTOCOL_FIELDS = (
    "responder_model",
    "judge_model",
    "use_judge",
    "provider_retries",
    "max_prompt_tokens",
    "prompt_cap_semantics",
    "prompt_token_proxy_identity",
    "responder_output_token_reserve",
    "recent_window",
    "accuracy_target",
    "min_target_questions",
    "stress_context_tokens",
    "stress_questions",
    "stress_question_offset",
    "max_samples",
)


@dataclass(frozen=True, slots=True)
class _ValidatedArm:
    report: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    protocol: dict[str, Any]
    canonical_sha256: str


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PairedComparisonError(f"value is not finite canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    """Return the content hash used for comparison inputs and identities."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PairedComparisonError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise PairedComparisonError(f"{label} keys must be strings")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PairedComparisonError(f"{label} must be a JSON array")
    return list(value)


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise PairedComparisonError(
            f"{label} fields mismatch: missing={sorted(wanted - actual)!r}, "
            f"extra={sorted(actual - wanted)!r}"
        )


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PairedComparisonError(f"{label} must be a string")
    if not allow_empty and (not value or value != value.strip()):
        raise PairedComparisonError(f"{label} must be a normalized non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise PairedComparisonError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PairedComparisonError(f"{label} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedComparisonError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PairedComparisonError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise PairedComparisonError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise PairedComparisonError(f"{label} must be <= {maximum}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PairedComparisonError(f"{label} must be a boolean")
    return value


def _optional_boolean(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, label)


def _must_equal(actual: Any, expected: Any, label: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise PairedComparisonError(f"{label} mismatch")


def _must_close(actual: Any, expected: float, label: str) -> None:
    observed = _number(actual, label)
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise PairedComparisonError(
            f"{label} drift: reported={observed!r}, recomputed={expected!r}"
        )


def _walk_json(value: Any, label: str) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if value != "<redacted>" and _SECRET_VALUE_RE.search(value):
            raise PairedComparisonError(f"{label} contains credential-shaped text")
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PairedComparisonError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk_json(child, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise PairedComparisonError(f"{label} contains a non-string key")
            lowered = raw_key.casefold()
            if lowered in _FORBIDDEN_SECRET_KEYS or lowered.endswith(
                _FORBIDDEN_SECRET_SUFFIXES
            ):
                if child not in (None, "", "<redacted>"):
                    raise PairedComparisonError(
                        f"{label} contains unredacted secret field {raw_key!r}"
                    )
            _walk_json(child, f"{label}.{raw_key}")
        return
    raise PairedComparisonError(f"{label} contains non-JSON value {type(value).__name__}")


def _usage(value: Any, label: str) -> dict[str, int | float]:
    row = _mapping(value, label)
    _exact_keys(row, _USAGE_FIELDS, label)
    return {
        "input_tokens": _integer(row["input_tokens"], f"{label}.input_tokens"),
        "output_tokens": _integer(row["output_tokens"], f"{label}.output_tokens"),
        "cache_read_input_tokens": _integer(
            row["cache_read_input_tokens"], f"{label}.cache_read_input_tokens"
        ),
        "elapsed_s": _number(row["elapsed_s"], f"{label}.elapsed_s", minimum=0.0),
        "calls": _integer(row["calls"], f"{label}.calls"),
    }


def _sum_usage(rows: Iterable[Mapping[str, int | float]]) -> dict[str, int | float]:
    materialized = list(rows)
    return {
        "input_tokens": sum(int(row["input_tokens"]) for row in materialized),
        "output_tokens": sum(int(row["output_tokens"]) for row in materialized),
        "cache_read_input_tokens": sum(
            int(row["cache_read_input_tokens"]) for row in materialized
        ),
        "elapsed_s": math.fsum(float(row["elapsed_s"]) for row in materialized),
        "calls": sum(int(row["calls"]) for row in materialized),
    }


def _validate_usage_total(
    value: Any, expected: Mapping[str, int | float], label: str
) -> None:
    observed = _usage(value, label)
    for field in _USAGE_FIELDS - {"elapsed_s"}:
        if observed[field] != expected[field]:
            raise PairedComparisonError(f"{label}.{field} disagrees with question rows")
    _must_close(observed["elapsed_s"], float(expected["elapsed_s"]), f"{label}.elapsed_s")


def _nearest(values: Sequence[int], quantile: float) -> int:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


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
        "p50": _nearest(ordered, 0.50),
        "p90": _nearest(ordered, 0.90),
        "p95": _nearest(ordered, 0.95),
        "p99": _nearest(ordered, 0.99),
        "max": ordered[-1],
        "values": ordered,
    }


def _validate_distribution(value: Any, expected: Mapping[str, Any], label: str) -> None:
    observed = _mapping(value, label)
    _exact_keys(observed, _DISTRIBUTION_FIELDS, label)
    for field in _DISTRIBUTION_FIELDS - {"mean"}:
        _must_equal(observed[field], expected[field], f"{label}.{field}")
    _must_close(observed["mean"], float(expected["mean"]), f"{label}.mean")


def _judge_verdict(value: Any, label: str) -> bool:
    reasoning = _text(value, label)
    match = _JUDGE_RE.match(reasoning)
    if match is None:
        raise PairedComparisonError(f"{label} has no exact binary judge verdict")
    remainder = match.group("remainder").lstrip(" \t\r\n,.:;-—")
    if remainder.casefold().startswith("or ") or remainder.startswith("/"):
        raise PairedComparisonError(f"{label} has an ambiguous judge verdict")
    return match.group(1).casefold() == "correct"


def _validate_tokenizer_identity(value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, label)
    expected = tokenizer_proxy_identity()
    _exact_keys(row, expected, label)
    _must_equal(row, expected, label)
    return row


def _validate_inputs(value: Any, *, arm: str) -> tuple[list[dict[str, Any]], str]:
    rows = _list(value, f"{arm}.inputs")
    if len(rows) != FROZEN_SHARD_COUNT:
        raise PairedComparisonError(f"{arm}.inputs must contain exactly ten shards")
    normalized: list[dict[str, Any]] = []
    digests: list[str] = []
    if arm == "treatment":
        fields = {"name", "sha256", "num_samples", "num_questions", "target_status"}
    else:
        fields = {
            "sample_offset",
            "sample_sha256",
            "name",
            "sha256",
            "retrieval_artifact_sha256",
            "retrieval_trace_sha256",
            "scoring_trace_sha256",
        }
    for index, value_row in enumerate(rows):
        label = f"{arm}.inputs[{index}]"
        row = _mapping(value_row, label)
        _exact_keys(row, fields, label)
        _text(row["name"], f"{label}.name")
        if "/" in row["name"] or "\\" in row["name"]:
            raise PairedComparisonError(f"{label}.name must be a portable basename")
        digest = _sha256(row["sha256"], f"{label}.sha256")
        digests.append(digest)
        if arm == "treatment":
            if _integer(row["num_samples"], f"{label}.num_samples") != 1:
                raise PairedComparisonError(f"{label} must describe one sample")
            if _integer(row["num_questions"], f"{label}.num_questions") != 10:
                raise PairedComparisonError(f"{label} must describe ten questions")
            if row["target_status"] != "insufficient_questions":
                raise PairedComparisonError(f"{label}.target_status mismatch")
        else:
            if _integer(row["sample_offset"], f"{label}.sample_offset") != FROZEN_OFFSETS[index]:
                raise PairedComparisonError(f"{label}.sample_offset order mismatch")
            _sha256(row["sample_sha256"], f"{label}.sample_sha256")
            for field in (
                "retrieval_artifact_sha256",
                "retrieval_trace_sha256",
                "scoring_trace_sha256",
            ):
                _sha256(row[field], f"{label}.{field}")
        normalized.append(row)
    if len(set(digests)) != len(digests):
        raise PairedComparisonError(f"{arm}.inputs repeats a shard digest")
    expected_set_hash = canonical_sha256(
        sorted(digests) if arm == "treatment" else digests
    )
    return normalized, expected_set_hash


def _metrics(rows: Sequence[Mapping[str, Any]], *, output_reserve: int) -> dict[str, Any]:
    contexts = [int(row["context_tokens"]) for row in rows]
    prompts = [int(row["prompt_token_proxy"]) for row in rows]
    requests = [value + output_reserve for value in prompts]
    count = len(rows)
    mean_context = math.fsum(float(value) for value in contexts) / count
    mean_prompt = math.fsum(float(value) for value in prompts) / count
    return {
        "num_questions": count,
        "judge_accuracy": math.fsum(
            1.0 if bool(row["judge_correct"]) else 0.0 for row in rows
        )
        / count,
        "mean_f1": math.fsum(float(row["f1"]) for row in rows) / count,
        "exact_match_rate": math.fsum(
            1.0 if bool(row["exact_match"]) else 0.0 for row in rows
        )
        / count,
        "mean_context_tokens": mean_context,
        "mean_prompt_token_proxy": mean_prompt,
        "mean_request_token_proxy": math.fsum(float(value) for value in requests)
        / count,
        "p95_prompt_token_proxy": _distribution(prompts)["p95"],
        "max_prompt_token_proxy_observed": max(prompts),
        "context_to_prompt_ratio": _safe_ratio(mean_context, mean_prompt),
        "context_token_distribution": _distribution(contexts),
        "prompt_token_proxy_distribution": _distribution(prompts),
        "request_token_proxy_distribution": _distribution(requests),
    }


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _validate_reported_core_metrics(
    report: Mapping[str, Any], metrics: Mapping[str, Any], label: str
) -> None:
    for field in (
        "judge_accuracy",
        "mean_f1",
        "exact_match_rate",
        "mean_context_tokens",
        "mean_prompt_token_proxy",
    ):
        _must_close(report[field], float(metrics[field]), f"{label}.{field}")
    if report["p95_prompt_token_proxy"] != metrics["p95_prompt_token_proxy"]:
        raise PairedComparisonError(f"{label}.p95_prompt_token_proxy drift")
    if report["max_prompt_token_proxy_observed"] != metrics["max_prompt_token_proxy_observed"]:
        raise PairedComparisonError(f"{label}.max_prompt_token_proxy_observed drift")
    for field in (
        "context_token_distribution",
        "prompt_token_proxy_distribution",
        "request_token_proxy_distribution",
    ):
        _validate_distribution(report[field], metrics[field], f"{label}.{field}")


def _validate_treatment_protocol(report: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _mapping(report["evaluation_protocol"], "treatment.evaluation_protocol")
    _exact_keys(protocol, _TREATMENT_PROTOCOL_FIELDS, "treatment.evaluation_protocol")
    exact = {
        "benchmark_format": "longmemeval",
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls": 20,
        "max_prompt_tokens": 8000,
        "prompt_cap_semantics": PROMPT_CAP_SEMANTICS,
        "responder_output_token_reserve": 256,
        "recent_window": 4,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "stress_context_tokens": 1_000_000,
        "stress_questions": 10,
        "stress_question_offset": 0,
        "max_samples": 1,
    }
    for field, expected in exact.items():
        _must_equal(protocol[field], expected, f"treatment.evaluation_protocol.{field}")
    for field in ("responder_model", "judge_model", "embedding_device"):
        _text(protocol[field], f"treatment.evaluation_protocol.{field}")
    _validate_tokenizer_identity(
        protocol["prompt_token_proxy_identity"],
        "treatment.evaluation_protocol.prompt_token_proxy_identity",
    )
    return protocol


def _validate_mem0_protocol(report: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _mapping(report["evaluation_protocol"], "mem0.evaluation_protocol")
    _exact_keys(protocol, _MEM0_PROTOCOL_FIELDS, "mem0.evaluation_protocol")
    exact = {
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls_per_shard": 20,
        "max_prompt_tokens": 8000,
        "prompt_cap_semantics": PROMPT_CAP_SEMANTICS,
        "responder_output_token_reserve": 256,
        "recent_window": 4,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "stress_context_tokens": 1_000_000,
        "stress_questions": 10,
        "stress_question_offset": 0,
        "max_samples": 1,
        "sample_offsets": list(FROZEN_OFFSETS),
    }
    for field, expected in exact.items():
        _must_equal(protocol[field], expected, f"mem0.evaluation_protocol.{field}")
    for field in ("responder_model", "judge_model"):
        _text(protocol[field], f"mem0.evaluation_protocol.{field}")
    _validate_tokenizer_identity(
        protocol["prompt_token_proxy_identity"],
        "mem0.evaluation_protocol.prompt_token_proxy_identity",
    )
    return protocol


def _validate_treatment_question(
    value: Any, *, index: int, prompt_cap: int, output_reserve: int
) -> dict[str, Any]:
    label = f"treatment.question_results[{index}]"
    row = _mapping(value, label)
    _exact_keys(row, _TREATMENT_QUESTION_FIELDS, label)
    question_id = _text(row["question_id"], f"{label}.question_id")
    question = _text(row["question"], f"{label}.question")
    gold = _text(row["gold_answer"], f"{label}.gold_answer", allow_empty=True)
    prediction = _text(row["predicted_answer"], f"{label}.predicted_answer")
    category = row["category"]
    if category is not None:
        _text(category, f"{label}.category")
    chunks = _list(row["retrieved_chunks"], f"{label}.retrieved_chunks")
    if any(not isinstance(chunk, str) for chunk in chunks):
        raise PairedComparisonError(f"{label}.retrieved_chunks must contain strings")
    recomputed_f1 = f1_score(prediction, gold)
    _must_close(row["f1"], recomputed_f1, f"{label}.f1")
    _must_equal(row["exact_match"], exact_match(prediction, gold), f"{label}.exact_match")
    verdict = _judge_verdict(row["judge_reasoning"], f"{label}.judge_reasoning")
    _must_equal(row["judge_correct"], verdict, f"{label}.judge_correct")
    context_tokens = _integer(row["context_tokens"], f"{label}.context_tokens")
    recomputed_context = sum(count_tokens(chunk) for chunk in chunks)
    if context_tokens != recomputed_context:
        raise PairedComparisonError(f"{label}.context_tokens drift")
    prompt = _integer(row["prompt_token_proxy"], f"{label}.prompt_token_proxy")
    if prompt > prompt_cap:
        raise PairedComparisonError(f"{label}.prompt_token_proxy exceeds the cap")
    _must_equal(row["prompt_tokens"], prompt, f"{label}.prompt_tokens")
    _must_equal(
        row["responder_output_token_reserve"],
        output_reserve,
        f"{label}.responder_output_token_reserve",
    )
    _must_equal(row["request_token_proxy"], prompt + output_reserve, f"{label}.request_token_proxy")
    transcript = _integer(row["transcript_tokens"], f"{label}.transcript_tokens", minimum=1)
    fraction = context_tokens / transcript
    _must_close(row["context_fraction"], fraction, f"{label}.context_fraction")
    _must_close(
        row["transcript_token_savings"],
        1.0 - fraction,
        f"{label}.transcript_token_savings",
    )
    responder = _usage(row["responder_usage"], f"{label}.responder_usage")
    judge = _usage(row["judge_usage"], f"{label}.judge_usage")
    if responder["calls"] != 1 or judge["calls"] != 1:
        raise PairedComparisonError(f"{label} must bind one responder and one judge call")
    expected_provider_compliance = (
        None
        if int(responder["input_tokens"]) == 0
        else int(responder["input_tokens"]) <= prompt_cap
    )
    if expected_provider_compliance is False:
        raise PairedComparisonError(f"{label} provider input exceeds the prompt cap")
    _must_equal(
        row["provider_prompt_budget_compliant"],
        expected_provider_compliance,
        f"{label}.provider_prompt_budget_compliant",
    )
    return {
        **row,
        "question_id": question_id,
        "question": question,
        "gold_answer": gold,
        "predicted_answer": prediction,
        "f1": recomputed_f1,
        "exact_match": exact_match(prediction, gold),
        "judge_correct": verdict,
        "context_tokens": context_tokens,
        "prompt_token_proxy": prompt,
        "responder_usage": responder,
        "judge_usage": judge,
    }


def _validate_messages(value: Any, label: str) -> list[dict[str, str]]:
    rows = _list(value, label)
    normalized: list[dict[str, str]] = []
    for index, child in enumerate(rows):
        row_label = f"{label}[{index}]"
        row = _mapping(child, row_label)
        _exact_keys(row, {"role", "content"}, row_label)
        normalized.append(
            {
                "role": _text(row["role"], f"{row_label}.role"),
                "content": _text(row["content"], f"{row_label}.content", allow_empty=True),
            }
        )
    return normalized


def _validate_mem0_question(
    value: Any, *, index: int, prompt_cap: int, proxy_identity: Mapping[str, Any]
) -> dict[str, Any]:
    label = f"mem0.question_results[{index}]"
    row = _mapping(value, label)
    _exact_keys(row, _MEM0_QUESTION_FIELDS, label)
    question_index = _integer(row["question_index"], f"{label}.question_index", minimum=1)
    if question_index > 10:
        raise PairedComparisonError(f"{label}.question_index must be within its shard")
    question_id = _text(row["question_id"], f"{label}.question_id")
    _text(row["question"], f"{label}.question")
    dated_question = _text(row["dated_question"], f"{label}.dated_question")
    gold = _text(row["gold_answer"], f"{label}.gold_answer", allow_empty=True)
    prediction = _text(row["prediction"], f"{label}.prediction")
    if row["category"] is not None:
        _text(row["category"], f"{label}.category")
    for field in (
        "retrieval_row_sha256",
        "query_sha256",
        "context_sha256",
        "messages_sha256",
        "raw_pool_sha256",
        "packed_pool_sha256",
    ):
        _sha256(row[field], f"{label}.{field}")
    _must_equal(row["query_sha256"], hashlib.sha256(dated_question.encode("utf-8")).hexdigest(), f"{label}.query_sha256")
    context = _text(row["context"], f"{label}.context", allow_empty=True)
    _must_equal(row["context_sha256"], hashlib.sha256(context.encode("utf-8")).hexdigest(), f"{label}.context_sha256")
    context_tokens = _integer(row["context_tokens"], f"{label}.context_tokens")
    if context_tokens != count_tokens(context):
        raise PairedComparisonError(f"{label}.context_tokens drift")
    messages = _validate_messages(row["messages"], f"{label}.messages")
    expected_messages = build_qa_prompt(dated_question, [context] if context else [])
    _must_equal(messages, expected_messages, f"{label}.messages prompt contract")
    _must_equal(row["messages_sha256"], canonical_sha256(messages), f"{label}.messages_sha256")
    prompt = _integer(row["prompt_token_proxy"], f"{label}.prompt_token_proxy")
    if prompt != count_chat_prompt_token_proxy(messages):
        raise PairedComparisonError(f"{label}.prompt_token_proxy drift")
    if prompt > prompt_cap:
        raise PairedComparisonError(f"{label}.prompt_token_proxy exceeds the cap")
    _must_equal(row["max_prompt_tokens"], prompt_cap, f"{label}.max_prompt_tokens")
    _must_equal(
        row["prompt_pack_protocol"],
        MEM0_PROMPT_PACK_PROTOCOL,
        f"{label}.prompt_pack_protocol",
    )
    _must_equal(row["residual_prompt_tokens"], prompt_cap - prompt, f"{label}.residual_prompt_tokens")
    _must_equal(row["prompt_token_proxy_identity"], proxy_identity, f"{label}.prompt_token_proxy_identity")
    _must_equal(
        row["configured_recent_window"],
        MEM0_CONFIGURED_RECENT_WINDOW,
        f"{label}.configured_recent_window",
    )
    _must_equal(
        row["effective_recent_window"],
        MEM0_EFFECTIVE_RECENT_WINDOW,
        f"{label}.effective_recent_window",
    )
    _must_equal(
        row["recent_window_semantics"],
        MEM0_RECENT_WINDOW_SEMANTICS,
        f"{label}.recent_window_semantics",
    )
    raw_count = _integer(row["raw_pool_count"], f"{label}.raw_pool_count")
    packed_count = _integer(row["packed_count"], f"{label}.packed_count")
    if packed_count > raw_count:
        raise PairedComparisonError(f"{label}.packed_count exceeds raw_pool_count")
    raw_tokens = _integer(row["raw_memory_tokens"], f"{label}.raw_memory_tokens")
    packed_tokens = _integer(row["packed_memory_tokens"], f"{label}.packed_memory_tokens")
    if packed_tokens > raw_tokens:
        raise PairedComparisonError(f"{label}.packed_memory_tokens exceeds raw_memory_tokens")
    _number(row["search_latency_s"], f"{label}.search_latency_s", minimum=0.0)
    _must_equal(row["attribution_kind"], MEM0_ATTRIBUTION_KIND, f"{label}.attribution_kind")
    _must_equal(row["supports_exact_source_provenance"], False, f"{label}.supports_exact_source_provenance")
    recomputed_f1 = f1_score(prediction, gold)
    _must_close(row["f1"], recomputed_f1, f"{label}.f1")
    _must_equal(row["exact_match"], exact_match(prediction, gold), f"{label}.exact_match")
    verdict = _judge_verdict(row["judge_reasoning"], f"{label}.judge_reasoning")
    _must_equal(row["judge_correct"], verdict, f"{label}.judge_correct")
    responder = _usage(row["responder_usage"], f"{label}.responder_usage")
    judge = _usage(row["judge_usage"], f"{label}.judge_usage")
    if responder["calls"] != 1 or judge["calls"] != 1:
        raise PairedComparisonError(f"{label} must bind one responder and one judge call")
    provider_input = int(responder["input_tokens"])
    expected_provider_compliance: bool | None = (
        None if provider_input == 0 else provider_input <= prompt_cap
    )
    _must_equal(
        row["provider_prompt_budget_compliant"],
        expected_provider_compliance,
        f"{label}.provider_prompt_budget_compliant",
    )
    if expected_provider_compliance is False:
        raise PairedComparisonError(f"{label} responder input exceeds the prompt cap")
    return {
        **row,
        "question_id": question_id,
        "prediction": prediction,
        "f1": recomputed_f1,
        "exact_match": exact_match(prediction, gold),
        "judge_correct": verdict,
        "context_tokens": context_tokens,
        "prompt_token_proxy": prompt,
        "responder_usage": responder,
        "judge_usage": judge,
    }


def _validate_question_order(rows: Sequence[Mapping[str, Any]], label: str) -> tuple[str, ...]:
    ids = tuple(str(row["question_id"]) for row in rows)
    if len(ids) != FROZEN_QUESTION_COUNT:
        raise PairedComparisonError(f"{label} must contain exactly 100 questions")
    if len(set(ids)) != len(ids):
        raise PairedComparisonError(f"{label} contains duplicate question IDs")
    if ids != tuple(sorted(ids)):
        raise PairedComparisonError(f"{label} question IDs must be in canonical order")
    return ids


def _category_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        raw = row["category"]
        category = raw if isinstance(raw, str) and raw.strip() else "uncategorized"
        grouped.setdefault(category, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for category in sorted(grouped):
        items = grouped[category]
        result[category] = {
            "category": category,
            "num_questions": len(items),
            "mean_f1": math.fsum(float(row["f1"]) for row in items) / len(items),
            "exact_match_rate": math.fsum(1.0 if row["exact_match"] else 0.0 for row in items) / len(items),
            "judge_accuracy": math.fsum(1.0 if row["judge_correct"] else 0.0 for row in items) / len(items),
        }
    return result


def _validate_category_metrics(value: Any, expected: Mapping[str, Mapping[str, Any]]) -> None:
    observed = _mapping(value, "treatment.by_category")
    if set(observed) != set(expected):
        raise PairedComparisonError("treatment.by_category category set drift")
    fields = {"category", "num_questions", "mean_f1", "exact_match_rate", "judge_accuracy"}
    for category, wanted in expected.items():
        row = _mapping(observed[category], f"treatment.by_category[{category!r}]")
        _exact_keys(row, fields, f"treatment.by_category[{category!r}]")
        _must_equal(row["category"], category, f"treatment.by_category[{category!r}].category")
        _must_equal(row["num_questions"], wanted["num_questions"], f"treatment.by_category[{category!r}].num_questions")
        for field in ("mean_f1", "exact_match_rate", "judge_accuracy"):
            _must_close(row[field], float(wanted[field]), f"treatment.by_category[{category!r}].{field}")


def _validate_treatment(value: Any) -> _ValidatedArm:
    report = _mapping(value, "treatment campaign")
    _walk_json(report, "treatment campaign")
    _exact_keys(report, _TREATMENT_TOP_FIELDS, "treatment campaign")
    _must_equal(report["schema_version"], 1, "treatment.schema_version")
    _must_equal(report["report_type"], TREATMENT_REPORT_TYPE, "treatment.report_type")
    _text(report["benchmark"], "treatment.benchmark")
    _must_equal(report["benchmark_split"], "validation", "treatment.benchmark_split")
    for field in (
        "dataset_sha256",
        "split_manifest_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
        "policy_manifest_sha256",
    ):
        _sha256(report[field], f"treatment.{field}")
    frozen_hashes = {
        "dataset_sha256": FROZEN_DATASET_SHA256,
        "split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
        "implementation_sha256": FROZEN_SOURCE_IMPLEMENTATION_SHA256,
        "environment_lock_sha256": FROZEN_SOURCE_ENVIRONMENT_SHA256,
        "policy_manifest_sha256": FROZEN_SOURCE_POLICY_SHA256,
    }
    for field, expected in frozen_hashes.items():
        _must_equal(report[field], expected, f"treatment frozen {field}")
    chunker = _mapping(report["chunker_config"], "treatment.chunker_config")
    _exact_keys(chunker, {"min_tokens", "max_tokens"}, "treatment.chunker_config")
    retrieval = _mapping(report["retrieval_config"], "treatment.retrieval_config")
    if not retrieval:
        raise PairedComparisonError("treatment.retrieval_config must be non-empty")
    frozen_retrieval: dict[str, Any] = {}
    for field in FROZEN_RETRIEVAL_FIELDS:
        if field == "chunker_min_tokens":
            value = chunker["min_tokens"]
        elif field == "chunker_max_tokens":
            value = chunker["max_tokens"]
        elif field == "max_prompt_tokens":
            value = report["max_prompt_tokens"]
        else:
            if field not in retrieval:
                raise PairedComparisonError(
                    f"treatment.retrieval_config lacks frozen field {field!r}"
                )
            value = retrieval[field]
        frozen_retrieval[field] = value
    _must_equal(
        canonical_sha256(frozen_retrieval),
        FROZEN_RETRIEVAL_IDENTITY_SHA256,
        "treatment frozen retrieval identity",
    )
    protocol = _validate_treatment_protocol(report)
    for field in ("responder_model", "judge_model", "recent_window", "max_prompt_tokens", "responder_output_token_reserve"):
        _must_equal(report[field], protocol[field], f"treatment.{field}")
    _must_equal(report["embedding_device"], protocol["embedding_device"], "treatment.embedding_device")
    proxy = _validate_tokenizer_identity(report["prompt_token_proxy_identity"], "treatment.prompt_token_proxy_identity")
    _must_equal(proxy, protocol["prompt_token_proxy_identity"], "treatment tokenizer identity binding")
    _must_equal(report["claim_profile"], LONGMEMEVAL_1M_95_PROFILE, "treatment.claim_profile")
    _must_equal(report["claim_profile_verified"], True, "treatment.claim_profile_verified")
    _must_equal(report["locked_population_verified"], True, "treatment.locked_population_verified")
    inputs, input_set_hash = _validate_inputs(report["inputs"], arm="treatment")
    _must_equal(report["input_count"], len(inputs), "treatment.input_count")
    _must_equal(report["input_set_sha256"], input_set_hash, "treatment.input_set_sha256")
    _must_equal(report["num_samples"], FROZEN_SHARD_COUNT, "treatment.num_samples")
    _must_equal(report["num_questions"], FROZEN_QUESTION_COUNT, "treatment.num_questions")
    raw_rows = _list(report["question_results"], "treatment.question_results")
    rows = tuple(
        _validate_treatment_question(
            row,
            index=index,
            prompt_cap=int(report["max_prompt_tokens"]),
            output_reserve=int(report["responder_output_token_reserve"]),
        )
        for index, row in enumerate(raw_rows)
    )
    ids = _validate_question_order(rows, "treatment.question_results")
    metrics = _metrics(rows, output_reserve=int(report["responder_output_token_reserve"]))
    _validate_reported_core_metrics(report, metrics, "treatment")
    _must_close(report["mean_request_token_proxy"], float(metrics["mean_request_token_proxy"]), "treatment.mean_request_token_proxy")
    _must_close(report["mean_prompt_tokens"], float(metrics["mean_prompt_token_proxy"]), "treatment.mean_prompt_tokens")
    _must_equal(report["p95_prompt_tokens"], metrics["p95_prompt_token_proxy"], "treatment.p95_prompt_tokens")
    _must_equal(report["max_prompt_tokens_observed"], metrics["max_prompt_token_proxy_observed"], "treatment.max_prompt_tokens_observed")
    _validate_distribution(report["prompt_token_distribution"], metrics["prompt_token_proxy_distribution"], "treatment.prompt_token_distribution")
    transcripts = _distribution(int(row["transcript_tokens"]) for row in rows)
    _validate_distribution(report["transcript_token_distribution"], transcripts, "treatment.transcript_token_distribution")
    provider_inputs = [
        int(row["responder_usage"]["input_tokens"])
        for row in rows
        if int(row["responder_usage"]["input_tokens"]) > 0
    ]
    provider_distribution = _distribution(provider_inputs)
    _validate_distribution(report["provider_input_token_distribution"], provider_distribution, "treatment.provider_input_token_distribution")
    provider_status = "unavailable" if not provider_inputs else "complete" if len(provider_inputs) == len(rows) else "partial"
    _must_equal(report["provider_input_usage_status"], provider_status, "treatment.provider_input_usage_status")
    provider_compliance = all(value <= int(report["max_prompt_tokens"]) for value in provider_inputs) if provider_inputs else None
    _must_equal(report["provider_prompt_budget_compliance"], provider_compliance, "treatment.provider_prompt_budget_compliance")
    _must_equal(report["prompt_token_proxy_budget_compliance"], True, "treatment.prompt_token_proxy_budget_compliance")
    _must_equal(report["prompt_budget_compliance"], True, "treatment.prompt_budget_compliance")
    responders = [row["responder_usage"] for row in rows]
    judges = [row["judge_usage"] for row in rows]
    _validate_usage_total(report["responder_usage"], _sum_usage(responders), "treatment.responder_usage")
    _validate_usage_total(report["judge_usage"], _sum_usage(judges), "treatment.judge_usage")
    _validate_category_metrics(report["by_category"], _category_metrics(rows))
    sources = _mapping(report["question_sources"], "treatment.question_sources")
    if set(sources) != set(ids):
        raise PairedComparisonError("treatment.question_sources does not match question IDs")
    input_by_hash = {row["sha256"]: row for row in inputs}
    sample_hashes: set[str] = set()
    source_counts = {digest: 0 for digest in input_by_hash}
    sample_by_input: dict[str, tuple[str, str]] = {}
    for question_id in ids:
        label = f"treatment.question_sources[{question_id!r}]"
        source = _mapping(sources[question_id], label)
        _exact_keys(source, {"report_name", "report_sha256", "sample_id", "sample_sha256"}, label)
        _text(source["report_name"], f"{label}.report_name")
        report_hash = _sha256(source["report_sha256"], f"{label}.report_sha256")
        if report_hash not in input_by_hash or input_by_hash[report_hash]["name"] != source["report_name"]:
            raise PairedComparisonError(f"{label} does not bind a campaign input")
        sample_id = _text(source["sample_id"], f"{label}.sample_id")
        sample_hash = _sha256(source["sample_sha256"], f"{label}.sample_sha256")
        source_counts[report_hash] += 1
        binding = (sample_id, sample_hash)
        previous = sample_by_input.setdefault(report_hash, binding)
        if previous != binding:
            raise PairedComparisonError(
                f"{label} disagrees with its input's sample binding"
            )
        sample_hashes.add(sample_hash)
    if set(source_counts.values()) != {10}:
        raise PairedComparisonError(
            "treatment question sources must bind exactly ten questions per input"
        )
    if (
        len(sample_by_input) != FROZEN_SHARD_COUNT
        or len(sample_hashes) != FROZEN_SHARD_COUNT
    ):
        raise PairedComparisonError(
            "treatment inputs must bind ten distinct one-sample shards"
        )
    receipts = _mapping(report["cache_receipts_by_sample"], "treatment.cache_receipts_by_sample")
    if set(receipts) != sample_hashes:
        raise PairedComparisonError("treatment cache receipts do not match source sample hashes")
    accuracy_target = _number(report["accuracy_target"], "treatment.accuracy_target", minimum=0.0, maximum=1.0)
    _must_equal(accuracy_target, protocol["accuracy_target"], "treatment accuracy target binding")
    _must_equal(report["min_target_questions"], 100, "treatment.min_target_questions")
    target_met = float(metrics["judge_accuracy"]) >= accuracy_target
    _must_equal(report["metric_accuracy_target_met"], target_met, "treatment.metric_accuracy_target_met")
    _must_equal(report["accuracy_target_met"], target_met, "treatment.accuracy_target_met")
    _must_equal(report["target_status"], "passed" if target_met else "failed", "treatment.target_status")
    return _ValidatedArm(report, rows, metrics, protocol, canonical_sha256(report))


def _validate_mem0_identity(report: Mapping[str, Any]) -> None:
    identity = _mapping(report["identity"], "mem0.identity")
    fields = {
        "source_validation_policy_sha256",
        "source_implementation_sha256",
        "source_environment_lock_sha256",
        "mem0_policy_sha256",
        "mem0_environment_lock_sha256",
        "mem0_tool_implementation_sha256",
        "mem0_stable_config_sha256",
        "extraction_model_identity",
        "extraction_model_identity_sha256",
        "embedder_model_identity",
        "embedder_model_identity_sha256",
        "scoring_policy_sha256",
        "source_evaluation_identity_sha256",
    }
    _exact_keys(identity, fields, "mem0.identity")
    for field in fields - {"extraction_model_identity", "embedder_model_identity"}:
        _sha256(identity[field], f"mem0.identity.{field}")
    _must_equal(identity["source_validation_policy_sha256"], report["policy_manifest_sha256"], "mem0 source policy identity")
    _must_equal(identity["source_implementation_sha256"], report["implementation_sha256"], "mem0 source implementation identity")
    _must_equal(identity["source_environment_lock_sha256"], report["environment_lock_sha256"], "mem0 source environment identity")
    extraction = _mapping(identity["extraction_model_identity"], "mem0.identity.extraction_model_identity")
    extraction_fields = {
        "provider", "model", "revision", "provider_retries", "logical_call_boundary",
        "logical_calls_per_add", "http_attempts_certified", "model_identity_sha256",
    }
    _exact_keys(extraction, extraction_fields, "mem0.identity.extraction_model_identity")
    for field in ("provider", "model", "revision"):
        _text(extraction[field], f"mem0.identity.extraction_model_identity.{field}")
    _must_equal(extraction["provider_retries"], 0, "mem0 extraction retries")
    _must_equal(extraction["logical_call_boundary"], "Memory.llm.generate_response", "mem0 extraction boundary")
    _must_equal(extraction["logical_calls_per_add"], 1, "mem0 extraction calls per add")
    _must_equal(extraction["http_attempts_certified"], False, "mem0 extraction HTTP certification")
    extraction_without_hash = {key: extraction[key] for key in extraction_fields - {"model_identity_sha256"}}
    _must_equal(extraction["model_identity_sha256"], canonical_sha256(extraction_without_hash), "mem0 extraction model identity hash")
    _must_equal(identity["extraction_model_identity_sha256"], canonical_sha256(extraction), "mem0 extraction identity hash")
    embedder = _mapping(identity["embedder_model_identity"], "mem0.identity.embedder_model_identity")
    embedder_fields = {
        "provider",
        "model",
        "revision",
        "checkpoint_sha256",
        "dimension",
        "device",
        "dtype",
        "execution",
        "network_calls_authorized",
        "runtime_probe_required",
        "model_identity_sha256",
    }
    _exact_keys(embedder, embedder_fields, "mem0.identity.embedder_model_identity")
    for field in ("provider", "model", "revision", "dtype"):
        _text(embedder[field], f"mem0.identity.embedder_model_identity.{field}")
    _sha256(embedder["checkpoint_sha256"], "mem0.identity.embedder_model_identity.checkpoint_sha256")
    _integer(embedder["dimension"], "mem0.identity.embedder_model_identity.dimension", minimum=1)
    if embedder["device"] not in {"cpu", "cuda"}:
        raise PairedComparisonError("mem0 embedder device must be cpu or cuda")
    _must_equal(
        embedder["execution"],
        "local_offline",
        "mem0 embedder execution",
    )
    _must_equal(
        embedder["network_calls_authorized"],
        0,
        "mem0 embedder network call authorization",
    )
    _must_equal(
        embedder["runtime_probe_required"],
        True,
        "mem0 embedder runtime probe requirement",
    )
    expected_embedder = {
        "provider": MEM0_EMBEDDER_PROVIDER,
        "model": MEM0_EMBEDDER_MODEL,
        "revision": MEM0_EMBEDDER_REVISION,
        "checkpoint_sha256": MEM0_EMBEDDER_CHECKPOINT_SHA256,
        "dimension": MEM0_EMBEDDER_DIMENSION,
        "dtype": MEM0_EMBEDDER_DTYPE,
    }
    for field, expected in expected_embedder.items():
        _must_equal(
            embedder[field],
            expected,
            f"mem0 embedder frozen {field}",
        )
    embedder_without_hash = {key: embedder[key] for key in embedder_fields - {"model_identity_sha256"}}
    _must_equal(embedder["model_identity_sha256"], canonical_sha256(embedder_without_hash), "mem0 embedder model identity hash")
    _must_equal(identity["embedder_model_identity_sha256"], canonical_sha256(embedder), "mem0 embedder identity hash")
    model_identity = _mapping(report["model_identity"], "mem0.model_identity")
    model_fields = {"responder_model", "responder_model_identity_sha256", "judge_model", "judge_model_identity_sha256"}
    _exact_keys(model_identity, model_fields, "mem0.model_identity")
    for field in ("responder_model", "judge_model"):
        _text(model_identity[field], f"mem0.model_identity.{field}")
        _must_equal(model_identity[field], report[field], f"mem0.model_identity.{field} binding")
    for field in ("responder_model_identity_sha256", "judge_model_identity_sha256"):
        _sha256(model_identity[field], f"mem0.model_identity.{field}")
    runtime_probe = _mapping(
        report["runtime_model_identity_probe"],
        "mem0.runtime_model_identity_probe",
    )
    runtime_probe_fields = {
        "kind",
        "extraction_model_identity_sha256",
        "embedder_model_identity_sha256",
        "before_match",
        "after_match",
        "comparison_certified",
    }
    _exact_keys(
        runtime_probe,
        runtime_probe_fields,
        "mem0.runtime_model_identity_probe",
    )
    expected_runtime_probe = {
        "kind": "unavailable_injected_nonproduction",
        "extraction_model_identity_sha256": identity[
            "extraction_model_identity_sha256"
        ],
        "embedder_model_identity_sha256": identity[
            "embedder_model_identity_sha256"
        ],
        "before_match": False,
        "after_match": False,
        "comparison_certified": False,
    }
    _must_equal(
        runtime_probe,
        expected_runtime_probe,
        "mem0.runtime_model_identity_probe",
    )
    config = _mapping(report["config"], "mem0.config")
    config_fields = {
        "max_prompt_tokens", "responder_max_output_tokens", "judge_max_output_tokens",
        "authorized_local_wrapper_retries", "external_retry_attempts_certified",
        "mem0_top_k", "mem0_threshold", "rendering_mode",
    }
    _exact_keys(config, config_fields, "mem0.config")
    _must_equal(config["max_prompt_tokens"], report["max_prompt_tokens"], "mem0.config.max_prompt_tokens")
    _must_equal(config["responder_max_output_tokens"], report["responder_output_token_reserve"], "mem0.config.responder_max_output_tokens")
    _integer(config["judge_max_output_tokens"], "mem0.config.judge_max_output_tokens", minimum=1)
    _must_equal(config["authorized_local_wrapper_retries"], 0, "mem0.config.authorized_local_wrapper_retries")
    _must_equal(config["external_retry_attempts_certified"], False, "mem0.config.external_retry_attempts_certified")
    _must_equal(config["mem0_top_k"], 200, "mem0.config.mem0_top_k")
    _must_close(config["mem0_threshold"], 0.1, "mem0.config.mem0_threshold")
    _must_equal(config["rendering_mode"], "official-memory-text-created-at", "mem0.config.rendering_mode")
    _must_equal(
        identity["source_evaluation_identity_sha256"],
        canonical_sha256(report["evaluation_protocol"]),
        "mem0 source evaluation identity hash",
    )


def _validate_common_mem0_rows(
    value: Any, full_rows: Sequence[Mapping[str, Any]]
) -> None:
    rows = _list(value, "mem0.common_question_results")
    if len(rows) != len(full_rows):
        raise PairedComparisonError("mem0 common question row count drift")
    for index, (raw_common, full) in enumerate(zip(rows, full_rows, strict=True)):
        label = f"mem0.common_question_results[{index}]"
        common = _mapping(raw_common, label)
        _exact_keys(common, _COMMON_QUESTION_FIELDS, label)
        expected = {
            "question_id": full["question_id"],
            "predicted_answer": full["prediction"],
            "judge_correct": full["judge_correct"],
            "f1": full["f1"],
            "exact_match": full["exact_match"],
            "context_tokens": full["context_tokens"],
            "prompt_token_proxy": full["prompt_token_proxy"],
            "responder_usage": full["responder_usage"],
            "judge_usage": full["judge_usage"],
        }
        _must_equal(common, expected, label)


def _validate_mem0(value: Any) -> _ValidatedArm:
    report = _mapping(value, "mem0 campaign")
    _walk_json(report, "mem0 campaign")
    _exact_keys(report, _MEM0_TOP_FIELDS, "mem0 campaign")
    _must_equal(report["schema_version"], 2, "mem0.schema_version")
    _must_equal(report["report_type"], MEM0_REPORT_TYPE, "mem0.report_type")
    _must_equal(report["arm_id"], MEM0_ARM_ID, "mem0.arm_id")
    _must_equal(report["run_status"], "complete", "mem0.run_status")
    _must_equal(report["benchmark"], "longmemeval", "mem0.benchmark")
    _must_equal(report["benchmark_split"], "validation", "mem0.benchmark_split")
    for field in (
        "dataset_sha256", "split_manifest_sha256", "implementation_sha256",
        "environment_lock_sha256", "policy_manifest_sha256",
    ):
        _sha256(report[field], f"mem0.{field}")
    protocol = _validate_mem0_protocol(report)
    for field in ("responder_model", "judge_model", "recent_window", "max_prompt_tokens", "responder_output_token_reserve"):
        _must_equal(report[field], protocol[field], f"mem0.{field}")
    proxy = _validate_tokenizer_identity(report["prompt_token_proxy_identity"], "mem0.prompt_token_proxy_identity")
    _must_equal(proxy, protocol["prompt_token_proxy_identity"], "mem0 tokenizer identity binding")
    _validate_mem0_identity(report)
    inputs, input_set_hash = _validate_inputs(report["inputs"], arm="mem0")
    _must_equal(report["input_count"], len(inputs), "mem0.input_count")
    _must_equal(report["input_set_sha256"], input_set_hash, "mem0.input_set_sha256")
    _must_equal(report["sample_offsets"], list(FROZEN_OFFSETS), "mem0.sample_offsets")
    _must_equal(report["num_samples"], FROZEN_SHARD_COUNT, "mem0.num_samples")
    _must_equal(report["num_questions"], FROZEN_QUESTION_COUNT, "mem0.num_questions")
    population = _mapping(report["population_identity"], "mem0.population_identity")
    _exact_keys(population, {"question_ids_sha256", "sample_offsets", "sample_sha256_by_offset"}, "mem0.population_identity")
    _must_equal(population["sample_offsets"], list(FROZEN_OFFSETS), "mem0.population_identity.sample_offsets")
    sample_hashes = _mapping(population["sample_sha256_by_offset"], "mem0.population_identity.sample_sha256_by_offset")
    _exact_keys(sample_hashes, {str(offset) for offset in FROZEN_OFFSETS}, "mem0.population_identity.sample_sha256_by_offset")
    for offset, digest in sample_hashes.items():
        _sha256(digest, f"mem0.population_identity.sample_sha256_by_offset[{offset}]")
    _must_equal(
        sample_hashes,
        dict(FROZEN_SAMPLE_SHA256_BY_OFFSET),
        "mem0 frozen composed-sample population",
    )
    prompt_identity = _mapping(report["prompt_identity"], "mem0.prompt_identity")
    _exact_keys(
        prompt_identity,
        {
            "prompt_pack_protocol",
            "max_prompt_tokens",
            "prompt_cap_semantics",
            "prompt_token_proxy_identity",
            "responder_output_token_reserve",
            "configured_recent_window",
            "effective_recent_window",
            "recent_window_semantics",
        },
        "mem0.prompt_identity",
    )
    expected_prompt_identity = {
        "prompt_pack_protocol": MEM0_PROMPT_PACK_PROTOCOL,
        "max_prompt_tokens": report["max_prompt_tokens"],
        "prompt_cap_semantics": protocol["prompt_cap_semantics"],
        "prompt_token_proxy_identity": proxy,
        "responder_output_token_reserve": report["responder_output_token_reserve"],
        "configured_recent_window": protocol["recent_window"],
        "effective_recent_window": MEM0_EFFECTIVE_RECENT_WINDOW,
        "recent_window_semantics": MEM0_RECENT_WINDOW_SEMANTICS,
    }
    _must_equal(prompt_identity, expected_prompt_identity, "mem0.prompt_identity")
    raw_rows = _list(report["question_results"], "mem0.question_results")
    rows = tuple(
        _validate_mem0_question(
            row, index=index, prompt_cap=int(report["max_prompt_tokens"]), proxy_identity=proxy
        )
        for index, row in enumerate(raw_rows)
    )
    ids = _validate_question_order(rows, "mem0.question_results")
    _must_equal(population["question_ids_sha256"], canonical_sha256(list(ids)), "mem0.population_identity.question_ids_sha256")
    _must_equal(report["common_question_result_schema"], COMMON_QUESTION_SCHEMA, "mem0.common_question_result_schema")
    _validate_common_mem0_rows(report["common_question_results"], rows)
    metrics = _metrics(rows, output_reserve=int(report["responder_output_token_reserve"]))
    _validate_reported_core_metrics(report, metrics, "mem0")
    _must_equal(report["prompt_token_proxy_budget_compliance"], True, "mem0.prompt_token_proxy_budget_compliance")
    responders = [row["responder_usage"] for row in rows]
    judges = [row["judge_usage"] for row in rows]
    _validate_usage_total(report["responder_usage"], _sum_usage(responders), "mem0.responder_usage")
    _validate_usage_total(report["judge_usage"], _sum_usage(judges), "mem0.judge_usage")
    provider_inputs = [
        int(row["responder_usage"]["input_tokens"])
        for row in rows
        if int(row["responder_usage"]["input_tokens"]) > 0
    ]
    provider_compliance = (
        all(value <= int(report["max_prompt_tokens"]) for value in provider_inputs)
        if provider_inputs
        else None
    )
    provider_status = (
        "unavailable"
        if not provider_inputs
        else "complete"
        if len(provider_inputs) == len(rows)
        else "partial"
    )
    _must_equal(
        report["provider_prompt_budget_compliance"],
        provider_compliance,
        "mem0.provider_prompt_budget_compliance",
    )
    _must_equal(
        report["provider_input_usage_status"],
        "local_injected_receipts_" + provider_status,
        "mem0.provider_input_usage_status",
    )
    _must_equal(report["external_provider_usage_certified"], False, "mem0.external_provider_usage_certified")
    sources = _mapping(report["question_sources"], "mem0.question_sources")
    if set(sources) != set(ids):
        raise PairedComparisonError("mem0.question_sources does not match question IDs")
    input_by_hash = {row["sha256"]: row for row in inputs}
    source_counts = {digest: 0 for digest in input_by_hash}
    question_indices_by_offset = {offset: set() for offset in FROZEN_OFFSETS}
    row_by_id = {str(row["question_id"]): row for row in rows}
    for input_row in inputs:
        offset = int(input_row["sample_offset"])
        _must_equal(
            input_row["sample_sha256"],
            sample_hashes[str(offset)],
            f"mem0 input sample hash at offset {offset}",
        )
    for question_id in ids:
        label = f"mem0.question_sources[{question_id!r}]"
        source = _mapping(sources[question_id], label)
        _exact_keys(source, {"sample_offset", "report_name", "report_sha256", "retrieval_artifact_sha256"}, label)
        offset = _integer(source["sample_offset"], f"{label}.sample_offset")
        if offset not in FROZEN_OFFSETS:
            raise PairedComparisonError(f"{label}.sample_offset is not frozen")
        digest = _sha256(source["report_sha256"], f"{label}.report_sha256")
        if digest not in input_by_hash:
            raise PairedComparisonError(f"{label} does not bind a campaign input")
        bound_input = input_by_hash[digest]
        _must_equal(offset, bound_input["sample_offset"], f"{label}.sample_offset")
        _must_equal(source["report_name"], input_by_hash[digest]["name"], f"{label}.report_name")
        _must_equal(source["retrieval_artifact_sha256"], input_by_hash[digest]["retrieval_artifact_sha256"], f"{label}.retrieval_artifact_sha256")
        source_counts[digest] += 1
        question_index = _integer(
            row_by_id[question_id]["question_index"],
            f"mem0.question_results[{question_id!r}].question_index",
            minimum=1,
        )
        if question_index in question_indices_by_offset[offset]:
            raise PairedComparisonError(
                f"mem0 offset {offset} repeats question_index {question_index}"
            )
        question_indices_by_offset[offset].add(question_index)
    if set(source_counts.values()) != {10}:
        raise PairedComparisonError(
            "mem0 question sources must bind exactly ten questions per input"
        )
    if any(
        indices != set(range(1, 11))
        for indices in question_indices_by_offset.values()
    ):
        raise PairedComparisonError(
            "mem0 question_index values must be exactly 1..10 within every shard"
        )
    raw_totals = _mapping(report["raw_input_totals"], "mem0.raw_input_totals")
    _exact_keys(raw_totals, {"raw_pairs", "skipped_empty_pairs"}, "mem0.raw_input_totals")
    _must_equal(raw_totals, {"raw_pairs": 24_928, "skipped_empty_pairs": 5}, "mem0.raw_input_totals")
    operations = _mapping(report["operation_totals"], "mem0.operation_totals")
    operation_fields = {
        "mem0_adds", "mem0_searches", "responder_logical_wrapper_calls",
        "judge_logical_wrapper_calls", "answer_judge_logical_wrapper_calls",
        "mem0_local_logical_wrapper_calls", "mem0_logical_extraction_call_boundary",
        "external_http_attempts_certified", "underlying_mem0_provider_calls",
        "underlying_mem0_provider_usage_status",
    }
    _exact_keys(operations, operation_fields, "mem0.operation_totals")
    exact_operations = {
        "mem0_adds": 24_923,
        "mem0_searches": 100,
        "responder_logical_wrapper_calls": 100,
        "judge_logical_wrapper_calls": 100,
        "answer_judge_logical_wrapper_calls": 200,
        "mem0_local_logical_wrapper_calls": 24_923,
        "mem0_logical_extraction_call_boundary": "Memory.llm.generate_response",
        "external_http_attempts_certified": False,
        "underlying_mem0_provider_calls": None,
        "underlying_mem0_provider_usage_status": (
            "unavailable_from_mem0_oss_public_api"
        ),
    }
    _must_equal(operations, exact_operations, "mem0.operation_totals")
    provenance = _mapping(report["provenance"], "mem0.provenance")
    provenance_fields = {
        "attribution_kind",
        "supports_exact_source_provenance",
        "source_session_date_exposure",
        "retrieved_created_at_exposure",
        "source_coverage_status",
        "source_coverage",
        "request_window_diagnostic_only",
        "source_coverage_reason",
    }
    _exact_keys(provenance, provenance_fields, "mem0.provenance")
    expected_provenance = {
        "attribution_kind": MEM0_ATTRIBUTION_KIND,
        "supports_exact_source_provenance": False,
        "source_session_date_exposure": "diagnostics_only_not_model_input",
        "retrieved_created_at_exposure": "answer_prompt_date_headings",
        "source_coverage_status": MEM0_SOURCE_COVERAGE_STATUS,
        "source_coverage": None,
        "request_window_diagnostic_only": True,
        "source_coverage_reason": MEM0_SOURCE_COVERAGE_REASON,
    }
    _must_equal(provenance, expected_provenance, "mem0.provenance")
    _must_equal(report["source_coverage_status"], MEM0_SOURCE_COVERAGE_STATUS, "mem0.source_coverage_status")
    _must_equal(report["source_coverage"], None, "mem0.source_coverage")
    _must_equal(report["exact_provenance_requirement_met"], False, "mem0.exact_provenance_requirement_met")
    _must_equal(report["local_request_token_state_contract_satisfied"], True, "mem0.local_request_token_state_contract_satisfied")
    for field in (
        "zero_persisted_transformer_token_state_verified",
        "external_provider_persistence_certified",
        "production_binding_certified",
    ):
        _boolean(report[field], f"mem0.{field}")
    _must_equal(
        report["production_binding_certified"],
        False,
        "mem0.production_binding_certified",
    )
    _text(report["certification_status"], "mem0.certification_status")
    _must_equal(report["locked_population_verified"], True, "mem0.locked_population_verified")
    _must_equal(report["local_comparison_protocol_verified"], True, "mem0.local_comparison_protocol_verified")
    accuracy_target = _number(report["accuracy_target"], "mem0.accuracy_target", minimum=0.0, maximum=1.0)
    _must_equal(accuracy_target, protocol["accuracy_target"], "mem0 accuracy target binding")
    _must_equal(report["min_target_questions"], 100, "mem0.min_target_questions")
    target_met = float(metrics["judge_accuracy"]) >= accuracy_target
    _must_equal(report["metric_accuracy_target_met"], target_met, "mem0.metric_accuracy_target_met")
    production = bool(report["production_binding_certified"])
    if not production:
        _must_equal(report["accuracy_target_met"], False, "mem0.accuracy_target_met")
        _must_equal(report["target_status"], "metric_passed_noncertified" if target_met else "metric_failed_noncertified", "mem0.target_status")
    else:
        _must_equal(report["accuracy_target_met"], target_met, "mem0.accuracy_target_met")
        _must_equal(report["target_status"], "passed" if target_met else "failed", "mem0.target_status")
    return _ValidatedArm(report, rows, metrics, protocol, canonical_sha256(report))


def _validate_shared_identity(treatment: _ValidatedArm, mem0: _ValidatedArm) -> dict[str, Any]:
    left = treatment.report
    right = mem0.report
    common_hashes = (
        "dataset_sha256",
        "split_manifest_sha256",
        "policy_manifest_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
    )
    for field in common_hashes:
        _must_equal(right[field], left[field], f"paired identity {field}")
    for field in (
        "responder_model", "judge_model", "recent_window", "max_prompt_tokens",
        "prompt_token_proxy_identity", "responder_output_token_reserve",
    ):
        _must_equal(right[field], left[field], f"paired identity {field}")
    for field in _SHARED_PROTOCOL_FIELDS:
        _must_equal(mem0.protocol[field], treatment.protocol[field], f"paired protocol {field}")
    # ``recent_window=4`` is the shared replay configuration default.  This
    # comparator separately binds the frozen LongMemEval implementation hash
    # and reconstructs treatment prompts solely from ``retrieved_chunks``;
    # completed-haystack QA therefore has no live recent-turn tail in either
    # arm.  Keep configured and effective values distinct in the public pair.
    prompt_contract = {
        "max_prompt_tokens": left["max_prompt_tokens"],
        "prompt_cap_semantics": treatment.protocol["prompt_cap_semantics"],
        "prompt_token_proxy_identity": left["prompt_token_proxy_identity"],
        "responder_output_token_reserve": left["responder_output_token_reserve"],
        "qa_prompt_builder": "memory_condense.eval.benchmark.build_qa_prompt",
        "configured_recent_window": left["recent_window"],
        "effective_recent_window": MEM0_EFFECTIVE_RECENT_WINDOW,
        "recent_window_semantics": MEM0_RECENT_WINDOW_SEMANTICS,
    }
    return {
        **{field: left[field] for field in common_hashes},
        "benchmark_split": "validation",
        "responder_model": left["responder_model"],
        "judge_model": left["judge_model"],
        "recent_window": left["recent_window"],
        "configured_recent_window": left["recent_window"],
        "effective_recent_window": MEM0_EFFECTIVE_RECENT_WINDOW,
        "recent_window_semantics": MEM0_RECENT_WINDOW_SEMANTICS,
        "prompt_contract": prompt_contract,
        "prompt_contract_sha256": canonical_sha256(prompt_contract),
    }


def _public_arm_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "num_questions",
        "judge_accuracy",
        "mean_f1",
        "exact_match_rate",
        "mean_context_tokens",
        "mean_prompt_token_proxy",
        "mean_request_token_proxy",
        "p95_prompt_token_proxy",
        "max_prompt_token_proxy_observed",
        "context_to_prompt_ratio",
        "context_token_distribution",
        "prompt_token_proxy_distribution",
        "request_token_proxy_distribution",
    )
    return {field: metrics[field] for field in fields}


def _validate_paired_population_identity(
    treatment: _ValidatedArm,
    mem0: _ValidatedArm,
) -> dict[str, Any]:
    """Bind both arms to the same exact composed million-token samples.

    Matching question IDs and question text is insufficient: two reports can
    retain the same questions while retrieving from different haystacks.  The
    source campaign hashes every composed ``BenchmarkSample`` (including all
    turns, source IDs, questions, answers, and dates), and the Mem0 campaign
    reconstructs that same sample before building its raw add stream.  Require
    those content identities to agree for every paired question.
    """

    treatment_sources = _mapping(
        treatment.report["question_sources"],
        "treatment.question_sources",
    )
    mem0_sources = _mapping(
        mem0.report["question_sources"],
        "mem0.question_sources",
    )
    mem0_population = _mapping(
        mem0.report["population_identity"],
        "mem0.population_identity",
    )
    sample_hashes = _mapping(
        mem0_population["sample_sha256_by_offset"],
        "mem0.population_identity.sample_sha256_by_offset",
    )

    question_bindings: dict[str, dict[str, Any]] = {}
    treatment_hash_by_offset: dict[str, str] = {}
    for row in treatment.rows:
        question_id = str(row["question_id"])
        treatment_source = _mapping(
            treatment_sources[question_id],
            f"treatment.question_sources[{question_id!r}]",
        )
        mem0_source = _mapping(
            mem0_sources[question_id],
            f"mem0.question_sources[{question_id!r}]",
        )
        offset = _integer(
            mem0_source["sample_offset"],
            f"mem0.question_sources[{question_id!r}].sample_offset",
        )
        offset_key = str(offset)
        mem0_sample_sha256 = _sha256(
            sample_hashes[offset_key],
            f"mem0.population_identity.sample_sha256_by_offset[{offset_key}]",
        )
        treatment_sample_sha256 = _sha256(
            treatment_source["sample_sha256"],
            f"treatment.question_sources[{question_id!r}].sample_sha256",
        )
        _must_equal(
            treatment_sample_sha256,
            mem0_sample_sha256,
            f"paired question {question_id} composed sample_sha256",
        )
        previous = treatment_hash_by_offset.setdefault(
            offset_key,
            treatment_sample_sha256,
        )
        _must_equal(
            treatment_sample_sha256,
            previous,
            f"paired Mem0 offset {offset} treatment sample_sha256",
        )
        question_bindings[question_id] = {
            "sample_offset": offset,
            "sample_sha256": mem0_sample_sha256,
        }

    expected_offsets = {str(offset) for offset in FROZEN_OFFSETS}
    if set(treatment_hash_by_offset) != expected_offsets:
        raise PairedComparisonError(
            "paired composed-sample offset coverage mismatch"
        )
    sample_sha256_by_offset = {
        str(offset): sample_hashes[str(offset)] for offset in FROZEN_OFFSETS
    }
    return {
        "sample_offsets": list(FROZEN_OFFSETS),
        "sample_sha256_by_offset": sample_sha256_by_offset,
        "sample_set_sha256": canonical_sha256(sample_sha256_by_offset),
        "question_to_sample_binding_sha256": canonical_sha256(
            question_bindings
        ),
    }


def compare_campaign_reports(
    treatment_campaign: Mapping[str, Any],
    mem0_campaign: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate and compare two strict merged campaign dictionaries.

    File loading is deliberately out of scope: callers must first use each
    arm's strict merger and pass its public result.  This core still rejects
    schema drift, duplicate/missing questions, non-finite values, secret
    material, forged primitive metrics, aggregate drift, and identity drift.
    """

    treatment = _validate_treatment(treatment_campaign)
    mem0 = _validate_mem0(mem0_campaign)
    shared_identity = _validate_shared_identity(treatment, mem0)
    treatment_ids = tuple(str(row["question_id"]) for row in treatment.rows)
    mem0_ids = tuple(str(row["question_id"]) for row in mem0.rows)
    if treatment_ids != mem0_ids:
        missing = sorted(set(treatment_ids) - set(mem0_ids))
        extra = sorted(set(mem0_ids) - set(treatment_ids))
        raise PairedComparisonError(
            "campaign question populations differ: "
            f"missing_from_mem0={missing!r}, extra_in_mem0={extra!r}"
        )
    question_ids_sha256 = canonical_sha256(list(treatment_ids))
    mem0_population = _mapping(
        mem0.report["population_identity"], "mem0.population_identity"
    )
    _must_equal(
        mem0_population["question_ids_sha256"],
        question_ids_sha256,
        "paired question population hash",
    )
    paired_population = _validate_paired_population_identity(treatment, mem0)

    paired_rows: list[dict[str, Any]] = []
    wins = ties = losses = 0
    for treatment_row, mem0_row in zip(treatment.rows, mem0.rows, strict=True):
        for field in ("question", "gold_answer", "category"):
            _must_equal(
                mem0_row[field],
                treatment_row[field],
                f"paired question {treatment_row['question_id']} {field}",
            )
        treatment_messages = build_qa_prompt(
            mem0_row["dated_question"],
            treatment_row["retrieved_chunks"],
        )
        _must_equal(
            treatment_row["prompt_token_proxy"],
            count_chat_prompt_token_proxy(treatment_messages),
            f"paired question {treatment_row['question_id']} treatment prompt contract",
        )
        treatment_correct = bool(treatment_row["judge_correct"])
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
        treatment_context = int(treatment_row["context_tokens"])
        mem0_context = int(mem0_row["context_tokens"])
        treatment_prompt = int(treatment_row["prompt_token_proxy"])
        mem0_prompt = int(mem0_row["prompt_token_proxy"])
        paired_rows.append(
            {
                "question_id": treatment_row["question_id"],
                "outcome": outcome,
                "treatment": {
                    "predicted_answer": treatment_row["predicted_answer"],
                    "judge_correct": treatment_correct,
                    "f1": treatment_row["f1"],
                    "exact_match": treatment_row["exact_match"],
                    "context_tokens": treatment_context,
                    "prompt_token_proxy": treatment_prompt,
                },
                "mem0": {
                    "predicted_answer": mem0_row["prediction"],
                    "judge_correct": mem0_correct,
                    "f1": mem0_row["f1"],
                    "exact_match": mem0_row["exact_match"],
                    "context_tokens": mem0_context,
                    "prompt_token_proxy": mem0_prompt,
                },
                "treatment_minus_mem0": {
                    "judge_correct": int(treatment_correct) - int(mem0_correct),
                    "f1": float(treatment_row["f1"]) - float(mem0_row["f1"]),
                    "exact_match": int(bool(treatment_row["exact_match"]))
                    - int(bool(mem0_row["exact_match"])),
                    "context_tokens": treatment_context - mem0_context,
                    "prompt_token_proxy": treatment_prompt - mem0_prompt,
                },
                "mem0_over_treatment": {
                    "context_token_ratio": _safe_ratio(mem0_context, treatment_context),
                    "prompt_token_proxy_ratio": _safe_ratio(mem0_prompt, treatment_prompt),
                },
            }
        )

    treatment_metrics = _public_arm_metrics(treatment.metrics)
    mem0_metrics = _public_arm_metrics(mem0.metrics)
    metric_deltas = {
        field: float(treatment.metrics[field]) - float(mem0.metrics[field])
        for field in (
            "judge_accuracy",
            "mean_f1",
            "exact_match_rate",
            "mean_context_tokens",
            "mean_prompt_token_proxy",
            "mean_request_token_proxy",
        )
    }
    token_ratios = {
        "mem0_over_treatment_mean_context_tokens": _safe_ratio(
            mem0.metrics["mean_context_tokens"],
            treatment.metrics["mean_context_tokens"],
        ),
        "mem0_over_treatment_mean_prompt_token_proxy": _safe_ratio(
            mem0.metrics["mean_prompt_token_proxy"],
            treatment.metrics["mean_prompt_token_proxy"],
        ),
        "mem0_over_treatment_mean_request_token_proxy": _safe_ratio(
            mem0.metrics["mean_request_token_proxy"],
            treatment.metrics["mean_request_token_proxy"],
        ),
    }

    treatment_integrity = bool(
        treatment.report["claim_profile_verified"]
        and treatment.report["locked_population_verified"]
    )
    mem0_production = bool(mem0.report["production_binding_certified"])
    mem0_integrity = bool(
        mem0.report["locked_population_verified"]
        and mem0.report["local_comparison_protocol_verified"]
    )
    certified = treatment_integrity and mem0_integrity and mem0_production
    blocking_reasons: list[str] = []
    if not treatment_integrity:
        blocking_reasons.append("treatment_locked_claim_profile_not_verified")
    if not mem0_integrity:
        blocking_reasons.append("mem0_locked_comparison_protocol_not_verified")
    if not mem0_production:
        blocking_reasons.append("mem0_production_binding_certified_false")

    result = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "report_type": COMPARISON_REPORT_TYPE,
        "comparison_scope": "same_locked_100q_longmemeval_population",
        "input_hashes": {
            "treatment_campaign_canonical_sha256": treatment.canonical_sha256,
            "mem0_campaign_canonical_sha256": mem0.canonical_sha256,
            "treatment_campaign_input_set_sha256": treatment.report[
                "input_set_sha256"
            ],
            "mem0_campaign_input_set_sha256": mem0.report["input_set_sha256"],
            "question_ids_sha256": question_ids_sha256,
        },
        "shared_identity": shared_identity,
        "paired_population_identity": paired_population,
        "metric_comparison": {
            "valid": True,
            "status": "paired_metrics_recomputed_from_primitive_rows",
            "num_questions": FROZEN_QUESTION_COUNT,
        },
        "certification": {
            "certified": certified,
            "status": "certified" if certified else "metric_only_noncertified",
            "blocking_reasons": blocking_reasons,
            "treatment_locked_claim_profile_verified": treatment_integrity,
            "mem0_local_comparison_protocol_verified": mem0_integrity,
            "mem0_production_binding_certified": mem0_production,
        },
        "arm_metrics": {
            "treatment": treatment_metrics,
            "mem0": mem0_metrics,
        },
        "paired_judge_outcomes": {
            "treatment_wins": wins,
            "ties": ties,
            "treatment_losses": losses,
        },
        "treatment_minus_mem0": metric_deltas,
        "token_ratios": token_ratios,
        "provenance_comparison": {
            "comparable": False,
            "status": "not_applicable_to_mem0",
            "treatment": {
                "status": "not_scored_by_campaign_comparator",
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
        "state_contracts": {
            "mem0_local_request_token_state_contract_satisfied": mem0.report[
                "local_request_token_state_contract_satisfied"
            ],
            "mem0_zero_persisted_transformer_token_state_verified": mem0.report[
                "zero_persisted_transformer_token_state_verified"
            ],
            "mem0_external_provider_persistence_certified": mem0.report[
                "external_provider_persistence_certified"
            ],
        },
        "question_results": paired_rows,
    }
    _walk_json(result, "paired comparison result")
    return result


__all__ = [
    "COMPARISON_REPORT_TYPE",
    "COMPARISON_SCHEMA_VERSION",
    "PairedComparisonError",
    "canonical_sha256",
    "compare_campaign_reports",
]
