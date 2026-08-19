"""Strict reports and campaign merging for the frozen Mem0 comparison arm.

The comparison is deliberately a two-stage protocol:

* an isolated Mem0 environment ingests and searches one locked 1M-token shard,
  writes a content-addressed retrieval artifact, erases its owned state, and
  exits; then
* the frozen v3 scoring environment consumes that artifact and performs one
  responder and one judge call per question.

This module imports neither Mem0 nor its optional dependencies.  It treats all
run artifacts as hostile input, reconstructs the frozen validation population,
and recomputes the metrics and local prompt-token cap from primitive fields.
Mem0 request-window attribution remains a diagnostic and is never promoted to
exact evidence provenance or source coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .source_compat import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.eval.benchmark import exact_match, f1_score
from memory_condense.eval.campaign import (
    LockedValidationPlan,
    build_locked_validation_plan,
)
from memory_condense.eval.mem0_adapter import (
    MEM0AI_PIN,
    MEM0_API_VERSION,
    MEM0_ATTRIBUTION_KIND,
    MEM0_BM25_MODEL,
    MEM0_CERTIFIED_RENDERING,
    MEM0_OFFICIAL_THRESHOLD,
    MEM0_OFFICIAL_TOP_K,
    MEM0_PROVIDER_USAGE_STATUS,
    MEM0_SPACY_MODEL,
    Mem0AdapterStats,
)
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    file_sha256,
    implementation_sha256,
)

from .policy import Mem0ComparisonPolicy, load_mem0_comparison_policy
from .preflight import load_source_validation_plan, tool_implementation_sha256
from .prompt_pack import pack_mem0_prompt, validate_source_evaluation_identity
from .protocol import RawStressShard, build_raw_stress_shards


SHARD_SCHEMA_VERSION = 1
SHARD_REPORT_TYPE = "mem0_longmemeval_stress_shard"
RETRIEVAL_ARTIFACT_TYPE = "mem0_longmemeval_retrieval_artifact"
CAMPAIGN_REPORT_TYPE = "mem0_longmemeval_campaign"
ARM_ID = "mem0_oss_2_0_18_direct_1m_v1"
RUN_STATUS = "complete"

INPUT_ORDER_PROTOCOL = (
    "locked-record-order+official-within-record-date-sort+"
    "consecutive-1-or-2-turn-slices-v1"
)
SEARCH_PROTOCOL = "mem0-top200-threshold0.1-rerank-false-explain-false-v1"
TWO_STAGE_PROTOCOL = "isolated-mem0-retrieval+frozen-v3-scoring-v1"
INGESTION_PROTOCOL = "mem0-official-longmemeval-consecutive-slices-v1"
# These are JSON-object receipts written by ``run_shard``.  An earlier draft
# of this merger described them as JSONL and consequently could not consume a
# report produced by the runner it was meant to validate.
RETRIEVAL_TRACE_FORMAT = "memory-condense-mem0-retrieval-trace-v1"
SCORING_TRACE_FORMAT = "memory-condense-mem0-scoring-trace-v1"
RETRIEVAL_ARTIFACT_FORMAT = "memory-condense-mem0-retrieval-artifact-v1"
SCORING_RECEIPT_FORMAT = "memory-condense-mem0-scoring-receipt-v1"

SOURCE_COVERAGE_STATUS = "unavailable_exact_source_provenance"
SOURCE_COVERAGE_REASON = (
    "mem0_request_window_attribution_is_not_exact_evidence_provenance"
)

# These are properties of the frozen v3 LongMemEval-S validation population,
# not configurable benchmark defaults.
FROZEN_OFFSETS = tuple(range(0, 100, 10))
FROZEN_QUESTION_COUNT = 100
FROZEN_RAW_PAIRS = 24_928
FROZEN_SKIPPED_EMPTY_PAIRS = 5
FROZEN_ADD_OPERATIONS = 24_923
FROZEN_SEARCH_OPERATIONS = 100
FROZEN_RESPONDER_CALLS = 100
FROZEN_JUDGE_CALLS = 100

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
    "private_key",
    "signing_key",
    "connection_string",
    "credentials",
}
_SECRET_VALUE_RE = re.compile(
    r"(?:^\s*(?:bearer|basic)\s+\S+|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk|ghp|github_pat|xox[baprs]|AIza)[-_][A-Za-z0-9_-]{8,}|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)


class Mem0ReportError(ValueError):
    """A Mem0 artifact cannot support the locked comparison."""


@dataclass(frozen=True, slots=True)
class ExpectedMem0Shard:
    """Reconstructed primitive identities and counts for one frozen shard."""

    sample_offset: int
    sample_id: str
    sample_sha256: str
    num_turns: int
    transcript_tokens: int
    questions: tuple[dict[str, Any], ...]
    history_sample_ids: tuple[str, ...]
    raw_history_bundle_sha256: str
    contributor_ids_sha256: str
    records: int
    raw_sessions: int
    raw_turns: int
    raw_pairs: int
    skipped_empty_pairs: int
    expected_adds: int

    @property
    def question_ids(self) -> tuple[str, ...]:
        return tuple(str(row["question_id"]) for row in self.questions)


@dataclass(frozen=True, slots=True)
class FrozenMem0Population:
    """All identities needed to validate the ten comparison reports."""

    plan: LockedValidationPlan
    shards: Mapping[int, ExpectedMem0Shard]
    mem0_policy_path: Path
    mem0_environment_lock_path: Path
    mem0_policy_sha256: str
    mem0_environment_lock_sha256: str
    mem0_tool_implementation_sha256: str
    source_evaluation_identity: Mapping[str, Any]
    mem0_policy: Mem0ComparisonPolicy


@dataclass(frozen=True, slots=True)
class ValidatedMem0Shard:
    """Validated report primitives used by the arm-specific merger."""

    report: dict[str, Any]
    report_name: str
    report_sha256: str
    sample_offset: int
    identity: dict[str, Any]
    model_identity: dict[str, Any]
    config: dict[str, Any]
    runtime_identity: dict[str, Any]
    questions: tuple[dict[str, Any], ...]
    retrieval_artifact_sha256: str
    retrieval_trace_sha256: str
    scoring_trace_sha256: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(child) for child in value]
    return value


def _json_constant(value: str) -> None:
    raise Mem0ReportError(f"non-finite JSON number {value!r} is not allowed")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Mem0ReportError(f"{label} must be an object")
    return dict(value)


def _require_exact_fields(
    value: Mapping[str, Any], fields: Iterable[str], label: str
) -> tuple[str, ...]:
    """Reject schema smuggling before an input object can be republished."""

    ordered = tuple(fields)
    expected = set(ordered)
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise Mem0ReportError(
            f"{label} fields do not match the native contract: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return ordered


def _field(value: Mapping[str, Any], name: str, label: str) -> Any:
    if name not in value:
        raise Mem0ReportError(f"{label}.{name} is required")
    return value[name]


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise Mem0ReportError(f"{label} must be an array")
    return list(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Mem0ReportError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _string(value, label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise Mem0ReportError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Mem0ReportError(f"{label} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Mem0ReportError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Mem0ReportError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise Mem0ReportError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise Mem0ReportError(f"{label} must be <= {maximum}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise Mem0ReportError(f"{label} must be a boolean")
    return value


def _must_equal(actual: Any, expected: Any, label: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise Mem0ReportError(
            f"{label} mismatch: expected {_canonical_json(expected)}, "
            f"got {_canonical_json(actual)}"
        )


def _reject_secret_material(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in _FORBIDDEN_SECRET_KEYS or key.endswith(
                (
                    "_password",
                    "_secret",
                    "_api_key",
                    "_authorization",
                    "_auth_token",
                    "_secret_key",
                    "_access_token",
                    "_refresh_token",
                    "_private_key",
                    "_signing_key",
                    "_connection_string",
                )
            ):
                if child not in (None, "", "<redacted>"):
                    raise Mem0ReportError(
                        f"{label} contains unredacted secret field {raw_key!r}"
                    )
            _reject_secret_material(child, f"{label}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secret_material(child, f"{label}[{index}]")
    elif isinstance(value, str) and any(
        marker in label.casefold()
        for marker in (
            "config",
            "identity",
            "binding",
            "environment",
            "header",
            "credential",
        )
    ):
        if value != "<redacted>" and _SECRET_VALUE_RE.search(value):
            raise Mem0ReportError(
                f"{label} contains credential-shaped secret material"
            )


def _raw_shape(shard: RawStressShard) -> tuple[int, int, int]:
    bundle = _mapping(shard.raw_history_bundle, "raw history bundle")
    records = _list(bundle.get("records"), "raw history bundle.records")
    sessions = 0
    turns = 0
    for record_index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"raw history bundle.records[{record_index}]")
        raw_sessions = _list(
            record.get("haystack_sessions"),
            f"raw history bundle.records[{record_index}].haystack_sessions",
        )
        sessions += len(raw_sessions)
        for session_index, raw_session in enumerate(raw_sessions):
            session = _list(
                raw_session,
                "raw history bundle.records"
                f"[{record_index}].haystack_sessions[{session_index}]",
            )
            turns += len(session)
    return len(records), sessions, turns


def reconstruct_frozen_mem0_population(
    *,
    benchmark_file: str | Path,
    split_manifest: str | Path,
    policy_manifest: str | Path,
    mem0_policy_manifest: str | Path,
    mem0_environment_lock: str | Path,
) -> FrozenMem0Population:
    """Rebuild the exact ten-shard population before reading run reports."""

    plan = build_locked_validation_plan(
        benchmark_file=benchmark_file,
        benchmark_format="longmemeval",
        split_manifest=split_manifest,
        policy_manifest=policy_manifest,
    )
    if plan.claim_profile_verified is not True:
        raise Mem0ReportError("the source v3 validation claim profile is not verified")
    if plan.sample_offsets != FROZEN_OFFSETS:
        raise Mem0ReportError(
            "the source v3 policy does not contain the exact ten frozen offsets"
        )
    if len(plan.question_ids) != FROZEN_QUESTION_COUNT:
        raise Mem0ReportError(
            "the source v3 policy does not reconstruct exactly 100 questions"
        )

    source_plan = load_source_validation_plan(
        benchmark_file=benchmark_file,
        split_manifest=split_manifest,
        policy_manifest=policy_manifest,
    )
    for label, actual, wanted in (
        ("source policy", source_plan.policy_manifest_sha256, plan.policy_manifest_sha256),
        ("source implementation", source_plan.implementation_sha256, plan.implementation_sha256),
        ("source environment", source_plan.environment_lock_sha256, plan.environment_lock_sha256),
        ("source offsets", source_plan.sample_offsets, plan.sample_offsets),
    ):
        _must_equal(actual, wanted, f"reconstructed {label}")
    mem0_policy_path = Path(mem0_policy_manifest).resolve()
    mem0_lock_path = Path(mem0_environment_lock).resolve()

    expected: dict[int, ExpectedMem0Shard] = {}
    all_ids: list[str] = []
    total_raw_pairs = 0
    total_skipped = 0
    total_adds = 0
    target_tokens = int(plan.evaluation["stress_context_tokens"])
    question_count = int(plan.evaluation["stress_questions"])
    raw_shards = list(
        build_raw_stress_shards(
            benchmark_file=benchmark_file,
            split_manifest=split_manifest,
            sample_offsets=plan.sample_offsets,
            target_tokens=target_tokens,
            max_questions=question_count,
        )
    )
    if len(raw_shards) != len(plan.sample_offsets):
        raise Mem0ReportError("raw Mem0 shard reconstruction count mismatch")
    for offset, raw in zip(plan.sample_offsets, raw_shards, strict=True):
        locked = plan.shards[offset]
        if raw.sample_sha256 != locked.sample_sha256:
            raise Mem0ReportError(f"reconstructed Mem0 shard {offset} sample mismatch")
        expected_ids = tuple(str(row["question_id"]) for row in locked.questions)
        if raw.question_ids != expected_ids:
            raise Mem0ReportError(
                f"reconstructed Mem0 shard {offset} question population mismatch"
            )
        if len(raw.add_batches) != raw.add_counts.add_requests:
            raise Mem0ReportError(
                f"reconstructed Mem0 shard {offset} add sequence mismatch"
            )
        records, raw_sessions, raw_turns = _raw_shape(raw)
        raw_questions = {
            question.question_id: question
            for question in raw.parsed_sample.questions
        }
        enriched_questions: list[dict[str, Any]] = []
        for locked_question in locked.questions:
            question_id = str(locked_question["question_id"])
            question = raw_questions[question_id]
            expected_locked = {
                "question_id": question.question_id,
                "question": question.question,
                "gold_answer": question.answer,
                "category": question.category,
            }
            if dict(locked_question) != expected_locked:
                raise Mem0ReportError(
                    f"reconstructed Mem0 shard {offset} question metadata mismatch"
                )
            enriched_questions.append(
                {
                    **expected_locked,
                    "dated_question": question.dated_question,
                }
            )
        expected[offset] = ExpectedMem0Shard(
            sample_offset=offset,
            sample_id=locked.sample_id,
            sample_sha256=locked.sample_sha256,
            num_turns=locked.num_turns,
            transcript_tokens=locked.transcript_tokens,
            questions=tuple(enriched_questions),
            history_sample_ids=raw.history_sample_ids,
            raw_history_bundle_sha256=raw.raw_history_bundle_sha256,
            contributor_ids_sha256=canonical_sha256(list(raw.history_sample_ids)),
            records=records,
            raw_sessions=raw_sessions,
            raw_turns=raw_turns,
            raw_pairs=raw.add_counts.raw_pairs,
            skipped_empty_pairs=raw.add_counts.skipped_empty_pairs,
            expected_adds=raw.add_counts.add_requests,
        )
        all_ids.extend(expected_ids)
        total_raw_pairs += raw.add_counts.raw_pairs
        total_skipped += raw.add_counts.skipped_empty_pairs
        total_adds += raw.add_counts.add_requests

    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(plan.question_ids):
        raise Mem0ReportError("reconstructed Mem0 shards do not cover the population")
    frozen_totals = (total_raw_pairs, total_skipped, total_adds)
    expected_totals = (
        FROZEN_RAW_PAIRS,
        FROZEN_SKIPPED_EMPTY_PAIRS,
        FROZEN_ADD_OPERATIONS,
    )
    if frozen_totals != expected_totals:
        raise Mem0ReportError(
            "reconstructed raw/add totals do not match the frozen v3 population: "
            f"{frozen_totals!r} != {expected_totals!r}"
        )

    mem0_policy = load_mem0_comparison_policy(
        mem0_policy_path,
        source_plan=source_plan,
        mem0_environment_lock=mem0_lock_path,
        expected_shards=tuple(raw_shards),
    )
    source_evaluation_identity = validate_source_evaluation_identity(
        source_plan.evaluation_identity
    )
    return FrozenMem0Population(
        plan=plan,
        shards=expected,
        mem0_policy_path=mem0_policy_path,
        mem0_environment_lock_path=mem0_lock_path,
        mem0_policy_sha256=mem0_policy.sha256,
        mem0_environment_lock_sha256=mem0_policy.environment_lock_sha256,
        mem0_tool_implementation_sha256=(
            mem0_policy.tool_implementation_sha256
        ),
        source_evaluation_identity=source_evaluation_identity,
        mem0_policy=mem0_policy,
    )


def _expected_identity(population: FrozenMem0Population) -> dict[str, str]:
    plan = population.plan
    return {
        "source_validation_policy_sha256": plan.policy_manifest_sha256,
        "source_implementation_sha256": plan.implementation_sha256,
        "source_environment_lock_sha256": plan.environment_lock_sha256,
        "mem0_policy_sha256": population.mem0_policy_sha256,
        "mem0_environment_lock_sha256": population.mem0_environment_lock_sha256,
        "mem0_tool_implementation_sha256": (
            population.mem0_tool_implementation_sha256
        ),
    }


def _validate_model_and_config(
    report: Mapping[str, Any],
    *,
    identity: Mapping[str, str],
    plan: LockedValidationPlan,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_identity = _mapping(report.get("model_identity"), f"{label}.model_identity")
    required_models = {
        "mem0_llm_model_id",
        "mem0_embedder_model_id",
        "responder_model",
        "judge_model",
    }
    for field in required_models:
        _string(model_identity.get(field), f"{label}.model_identity.{field}")
    _must_equal(
        model_identity["responder_model"],
        plan.evaluation["responder_model"],
        f"{label}.model_identity.responder_model",
    )
    _must_equal(
        model_identity["judge_model"],
        plan.evaluation["judge_model"],
        f"{label}.model_identity.judge_model",
    )
    _reject_secret_material(model_identity, f"{label}.model_identity")
    _must_equal(
        identity.get("model_identity_sha256"),
        canonical_sha256(model_identity),
        f"{label}.identity.model_identity_sha256",
    )

    config = _mapping(report.get("config"), f"{label}.config")
    required_config = {
        "mem0ai_version": MEM0AI_PIN,
        "mem0_api_version": MEM0_API_VERSION,
        "responder_model": plan.evaluation["responder_model"],
        "judge_model": plan.evaluation["judge_model"],
        "max_prompt_tokens": plan.evaluation["max_prompt_tokens"],
        "responder_output_token_reserve": plan.evaluation[
            "responder_output_token_reserve"
        ],
        "provider_retries": 0,
        "top_k": MEM0_OFFICIAL_TOP_K,
        "threshold": MEM0_OFFICIAL_THRESHOLD,
        "rerank": False,
        "explain": False,
        "rendering_mode": MEM0_CERTIFIED_RENDERING,
        "input_order_protocol": INPUT_ORDER_PROTOCOL,
        "prompt_cap_semantics": plan.evaluation["prompt_cap_semantics"],
        "prompt_token_proxy_identity": plan.evaluation[
            "prompt_token_proxy_identity"
        ],
    }
    for field, expected in required_config.items():
        _must_equal(config.get(field), expected, f"{label}.config.{field}")
    _reject_secret_material(config, f"{label}.config")
    _must_equal(
        identity.get("config_sha256"),
        canonical_sha256(config),
        f"{label}.identity.config_sha256",
    )
    return model_identity, config


def _validate_protocol(
    value: Any,
    *,
    offset: int,
    plan: LockedValidationPlan,
    label: str,
) -> dict[str, Any]:
    protocol = _mapping(value, label)
    expected = {
        "comparison_protocol": TWO_STAGE_PROTOCOL,
        "benchmark_format": "longmemeval",
        "benchmark_split": "validation",
        "stress_context_tokens": plan.evaluation["stress_context_tokens"],
        "stress_questions": plan.evaluation["stress_questions"],
        "stress_question_offset": 0,
        "sample_offset": offset,
        "max_samples": 1,
        "use_judge": True,
        "max_provider_calls": 2 * int(plan.evaluation["stress_questions"]),
        "official_longmemeval_protocol": True,
        "input_order_protocol": INPUT_ORDER_PROTOCOL,
        "search_protocol": SEARCH_PROTOCOL,
    }
    for field, wanted in expected.items():
        _must_equal(protocol.get(field), wanted, f"{label}.{field}")
    return protocol


def _validate_provenance(value: Any, label: str) -> dict[str, Any]:
    provenance = _mapping(value, label)
    expected = {
        "attribution_kind": MEM0_ATTRIBUTION_KIND,
        "supports_exact_source_provenance": False,
        "source_coverage_status": SOURCE_COVERAGE_STATUS,
        "source_coverage": None,
        "request_window_diagnostic_only": True,
        "source_session_date_exposure": "diagnostics_only_not_model_input",
        "retrieved_created_at_exposure": "answer_prompt_date_headings",
        "source_coverage_reason": SOURCE_COVERAGE_REASON,
    }
    for field, wanted in expected.items():
        _must_equal(_field(provenance, field, label), wanted, f"{label}.{field}")
    return provenance


def _validate_sample(value: Any, expected: ExpectedMem0Shard, label: str) -> dict[str, Any]:
    sample = _mapping(value, label)
    expected_values = {
        "sample_id": expected.sample_id,
        "sample_sha256": expected.sample_sha256,
        "num_turns": expected.num_turns,
        "transcript_tokens": expected.transcript_tokens,
        "num_questions": len(expected.questions),
        "question_ids": list(expected.question_ids),
        "question_ids_sha256": canonical_sha256(list(expected.question_ids)),
    }
    for field, wanted in expected_values.items():
        _must_equal(sample.get(field), wanted, f"{label}.{field}")
    return sample


def _validate_raw_receipt(
    value: Any, expected: ExpectedMem0Shard, label: str
) -> dict[str, Any]:
    receipt = _mapping(value, label)
    expected_values = {
        "raw_history_bundle_sha256": expected.raw_history_bundle_sha256,
        "contributor_ids_sha256": expected.contributor_ids_sha256,
        "records": expected.records,
        "raw_sessions": expected.raw_sessions,
        "raw_turns": expected.raw_turns,
        "raw_pairs": expected.raw_pairs,
        "skipped_empty_pairs": expected.skipped_empty_pairs,
        "whitespace_only_pairs": 0,
    }
    for field, wanted in expected_values.items():
        _must_equal(receipt.get(field), wanted, f"{label}.{field}")
    return receipt


def _validate_runtime_identity(value: Any, label: str) -> dict[str, Any]:
    runtime = _mapping(value, label)
    for field, wanted in {
        "protocol": "mem0-oss-2.0.18-certified-local-v1",
        "certified": True,
        "local_owned_state": True,
        "on_disk": True,
    }.items():
        _must_equal(runtime.get(field), wanted, f"{label}.{field}")
    _sha256(runtime.get("stable_config_sha256"), f"{label}.stable_config_sha256")
    _sha256(
        runtime.get("effective_config_sha256"),
        f"{label}.effective_config_sha256",
    )
    runtime_config = _mapping(runtime.get("config"), f"{label}.config")
    _reject_secret_material(runtime_config, f"{label}.config")
    stack = _mapping(runtime.get("stack"), f"{label}.stack")
    versions = _mapping(stack.get("dependency_versions"), f"{label}.stack.dependency_versions")
    _must_equal(versions.get("mem0ai"), MEM0AI_PIN, f"{label}.stack.dependency_versions.mem0ai")
    _string(stack.get("bm25_model"), f"{label}.stack.bm25_model")
    _string(stack.get("spacy_model"), f"{label}.stack.spacy_model")
    _must_equal(stack.get("bm25_operational"), True, f"{label}.stack.bm25_operational")
    _must_equal(
        stack.get("entity_extraction_operational"),
        True,
        f"{label}.stack.entity_extraction_operational",
    )
    return runtime


def _validate_ingestion_receipt(
    value: Any, expected: ExpectedMem0Shard, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _mapping(value, label)
    exact = {
        "protocol": INGESTION_PROTOCOL,
        "expected_adds": expected.expected_adds,
        "attempted_adds": expected.expected_adds,
        "completed_adds": expected.expected_adds,
        "failed_adds": 0,
        "fresh_owned_state": True,
        "on_disk": True,
        "cleanup_complete": True,
        "state_removed": True,
        "active_scopes_after_cleanup": 0,
        "ledger_memories_after_cleanup": 0,
        "persisted_transformer_token_state": False,
    }
    for field, wanted in exact.items():
        _must_equal(receipt.get(field), wanted, f"{label}.{field}")
    _integer(receipt.get("returned_memories"), f"{label}.returned_memories")
    _integer(receipt.get("unique_memories"), f"{label}.unique_memories")
    _integer(
        receipt.get("raw_message_token_proxy"),
        f"{label}.raw_message_token_proxy",
    )
    _number(receipt.get("latency_s"), f"{label}.latency_s", minimum=0.0)
    runtime = _validate_runtime_identity(
        receipt.get("runtime_identity"), f"{label}.runtime_identity"
    )
    _must_equal(
        receipt.get("runtime_identity_sha256"),
        canonical_sha256(runtime),
        f"{label}.runtime_identity_sha256",
    )
    return receipt, runtime


def _validate_mem0_usage(value: Any, label: str) -> dict[str, Any]:
    usage = _mapping(value, label)
    expected = {
        "provider_prompt_tokens": None,
        "provider_completion_tokens": None,
        "provider_calls": None,
        "provider_usage_status": MEM0_PROVIDER_USAGE_STATUS,
    }
    for field, wanted in expected.items():
        _must_equal(_field(usage, field, label), wanted, f"{label}.{field}")
    return usage


def _safe_sibling(path: Path, filename: Any, label: str) -> Path:
    name = _string(filename, label)
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
        raise Mem0ReportError(f"{label} must be a sibling basename")
    resolved = (path.parent / relative).resolve()
    if resolved.parent != path.parent.resolve():
        raise Mem0ReportError(f"{label} escapes the report directory")
    if not resolved.is_file():
        raise Mem0ReportError(f"{label} does not name an existing file")
    return resolved


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload, parse_constant=_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Mem0ReportError(f"cannot read {label} {path}: {exc}") from exc
    return _mapping(value, label), payload


def _validate_trace(
    value: Any,
    *,
    report_path: Path,
    expected_format: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    receipt = _mapping(value, label)
    _must_equal(receipt.get("format"), expected_format, f"{label}.format")
    trace_path = _safe_sibling(report_path, receipt.get("filename"), f"{label}.filename")
    try:
        payload = trace_path.read_bytes()
    except OSError as exc:
        raise Mem0ReportError(f"cannot read {label} file: {exc}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    _must_equal(receipt.get("sha256"), digest, f"{label}.sha256")
    _must_equal(receipt.get("bytes"), len(payload), f"{label}.bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Mem0ReportError(f"{label} is not UTF-8 JSONL") from exc
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise Mem0ReportError(f"{label} must contain non-empty JSONL records")
    for index, line in enumerate(lines):
        try:
            row = json.loads(line, parse_constant=_json_constant)
        except json.JSONDecodeError as exc:
            raise Mem0ReportError(f"{label} line {index + 1} is invalid JSON") from exc
        _mapping(row, f"{label} line {index + 1}")
    _must_equal(receipt.get("lines"), len(lines), f"{label}.lines")
    return receipt, digest


def _validate_pool(value: Any, label: str) -> list[dict[str, Any]]:
    rows = _list(value, label)
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    prior_rank = 0
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{label}[{index}]")
        rank = _integer(row.get("rank"), f"{label}[{index}].rank", minimum=1)
        if rank <= prior_rank:
            raise Mem0ReportError(f"{label} ranks must be strictly increasing")
        prior_rank = rank
        memory_id = _string(row.get("memory_id"), f"{label}[{index}].memory_id")
        if memory_id in ids:
            raise Mem0ReportError(f"{label} repeats memory ID {memory_id!r}")
        ids.add(memory_id)
        if not isinstance(row.get("text"), str):
            raise Mem0ReportError(f"{label}[{index}].text must be a string")
        created_at = row.get("created_at")
        if not isinstance(created_at, str) or not created_at.strip():
            raise Mem0ReportError(
                f"{label}[{index}].created_at must be a non-empty string"
            )
        score = _field(row, "score", f"{label}[{index}]")
        if score is not None:
            _number(score, f"{label}[{index}].score")
        _must_equal(
            row.get("attribution_kind"),
            MEM0_ATTRIBUTION_KIND,
            f"{label}[{index}].attribution_kind",
        )
        normalized.append(row)
    return normalized


def _validate_row_provenance(value: Any, label: str) -> dict[str, Any]:
    """Validate the intentionally weak per-search attribution declaration."""

    provenance = _mapping(value, label)
    _must_equal(
        _field(provenance, "kind", label),
        MEM0_ATTRIBUTION_KIND,
        f"{label}.kind",
    )
    _must_equal(
        _field(provenance, "supports_exact_source_provenance", label),
        False,
        f"{label}.supports_exact_source_provenance",
    )
    return provenance


def _validate_retrieval_row(
    value: Any,
    *,
    expected_question: Mapping[str, Any],
    prompt_cap: int,
    label: str,
) -> dict[str, Any]:
    row = _mapping(value, label)
    _must_equal(
        row.get("format"),
        "memory-condense-mem0-retrieval-row-v1",
        f"{label}.format",
    )
    question_id = _string(row.get("question_id"), f"{label}.question_id")
    _must_equal(question_id, expected_question["question_id"], f"{label}.question_id")
    query = _string(row.get("query"), f"{label}.query")
    _must_equal(
        query,
        expected_question.get("dated_question", expected_question["question"]),
        f"{label}.query",
    )
    context = row.get("context")
    if not isinstance(context, str):
        raise Mem0ReportError(f"{label}.context must be a string")
    _must_equal(row.get("context_sha256"), text_sha256(context), f"{label}.context_sha256")
    context_tokens = count_tokens(context)
    _must_equal(row.get("context_tokens"), context_tokens, f"{label}.context_tokens")

    messages_raw = _list(row.get("messages"), f"{label}.messages")
    if len(messages_raw) != 2:
        raise Mem0ReportError(f"{label}.messages must contain exactly two messages")
    messages: list[dict[str, str]] = []
    for index, raw_message in enumerate(messages_raw):
        message = _mapping(raw_message, f"{label}.messages[{index}]")
        role = _string(message.get("role"), f"{label}.messages[{index}].role")
        content = message.get("content")
        if not isinstance(content, str):
            raise Mem0ReportError(f"{label}.messages[{index}].content must be a string")
        messages.append({"role": role, "content": content})
    _must_equal([message["role"] for message in messages], ["system", "user"], f"{label}.messages roles")
    _must_equal(row.get("messages_sha256"), canonical_sha256(messages), f"{label}.messages_sha256")
    prompt_proxy = count_chat_prompt_token_proxy(messages)
    _must_equal(row.get("prompt_token_proxy"), prompt_proxy, f"{label}.prompt_token_proxy")
    _must_equal(row.get("max_prompt_token_proxy"), prompt_cap, f"{label}.max_prompt_token_proxy")
    _must_equal(
        row.get("residual_prompt_token_proxy"),
        prompt_cap - prompt_proxy,
        f"{label}.residual_prompt_token_proxy",
    )
    _must_equal(
        row.get("responder_output_token_reserve"),
        256,
        f"{label}.responder_output_token_reserve",
    )
    _must_equal(
        row.get("request_token_proxy"),
        prompt_proxy + 256,
        f"{label}.request_token_proxy",
    )
    if prompt_proxy > prompt_cap:
        raise Mem0ReportError(
            f"{label} exceeds the locked local prompt-token cap: "
            f"{prompt_proxy} > {prompt_cap}"
        )

    raw_pool = _validate_pool(row.get("raw_pool"), f"{label}.raw_pool")
    packed_pool = _validate_pool(row.get("packed_pool"), f"{label}.packed_pool")
    _must_equal(row.get("raw_memory_count"), len(raw_pool), f"{label}.raw_memory_count")
    _must_equal(row.get("packed_memory_count"), len(packed_pool), f"{label}.packed_memory_count")
    _must_equal(row.get("raw_pool_sha256"), canonical_sha256(raw_pool), f"{label}.raw_pool_sha256")
    _must_equal(row.get("packed_pool_sha256"), canonical_sha256(packed_pool), f"{label}.packed_pool_sha256")
    raw_by_id = {str(item["memory_id"]): item for item in raw_pool}
    prior_index = -1
    raw_ids = [str(item["memory_id"]) for item in raw_pool]
    for packed in packed_pool:
        memory_id = str(packed["memory_id"])
        if memory_id not in raw_by_id or packed != raw_by_id[memory_id]:
            raise Mem0ReportError(f"{label}.packed_pool is not a subset of raw_pool")
        index = raw_ids.index(memory_id)
        if index <= prior_index:
            raise Mem0ReportError(
                f"{label}.packed_pool is not an order-preserving subset"
            )
        prior_index = index
    _must_equal(
        row.get("raw_memory_tokens"),
        sum(count_tokens(str(item["text"])) for item in raw_pool),
        f"{label}.raw_memory_tokens",
    )
    _must_equal(
        row.get("packed_memory_tokens"),
        sum(count_tokens(str(item["text"])) for item in packed_pool),
        f"{label}.packed_memory_tokens",
    )
    _number(row.get("search_latency_s"), f"{label}.search_latency_s", minimum=0.0)
    _must_equal(
        row.get("rendering_mode"),
        MEM0_CERTIFIED_RENDERING,
        f"{label}.rendering_mode",
    )
    for field in (
        "official_longmemeval_protocol",
        "official_search_protocol",
        "independently_verified",
        "adapter_comparison_certified",
    ):
        _must_equal(row.get(field), True, f"{label}.{field}")
    try:
        source_identity = validate_source_evaluation_identity(
            row.get("source_evaluation_identity")
        )
    except (TypeError, ValueError) as exc:
        raise Mem0ReportError(
            f"{label}.source_evaluation_identity is invalid: {exc}"
        ) from exc
    _must_equal(
        row.get("source_evaluation_identity_sha256"),
        canonical_sha256(source_identity),
        f"{label}.source_evaluation_identity_sha256",
    )
    _must_equal(
        row.get("prompt_token_proxy_identity"),
        source_identity["prompt_token_proxy_identity"],
        f"{label}.prompt_token_proxy_identity",
    )
    _validate_row_provenance(row.get("provenance"), f"{label}.provenance")
    if "source_coverage" in row and row["source_coverage"] is not None:
        raise Mem0ReportError(f"{label}.source_coverage must remain unavailable")
    self_hash_payload = dict(row)
    reported_self_hash = self_hash_payload.pop("retrieval_row_sha256", None)
    _must_equal(
        reported_self_hash,
        canonical_sha256(self_hash_payload),
        f"{label}.retrieval_row_sha256",
    )
    return row


def _validate_retrieval_artifact(
    descriptor_value: Any,
    *,
    report_path: Path,
    expected: ExpectedMem0Shard,
    expected_identity: Mapping[str, str],
    model_identity: Mapping[str, Any],
    config: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    ingestion_receipt: Mapping[str, Any],
    mem0_usage: Mapping[str, Any],
    provenance: Mapping[str, Any],
    prompt_cap: int,
    label: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], str]:
    descriptor = _mapping(descriptor_value, label)
    _must_equal(descriptor.get("format"), RETRIEVAL_ARTIFACT_FORMAT, f"{label}.format")
    artifact_path = _safe_sibling(report_path, descriptor.get("filename"), f"{label}.filename")
    artifact, payload = _load_json(artifact_path, "Mem0 retrieval artifact")
    _reject_secret_material(artifact, "Mem0 retrieval artifact")
    digest = hashlib.sha256(payload).hexdigest()
    _must_equal(descriptor.get("sha256"), digest, f"{label}.sha256")
    _must_equal(descriptor.get("bytes"), len(payload), f"{label}.bytes")
    _must_equal(descriptor.get("num_questions"), len(expected.questions), f"{label}.num_questions")
    _must_equal(
        descriptor.get("question_ids_sha256"),
        canonical_sha256(list(expected.question_ids)),
        f"{label}.question_ids_sha256",
    )

    _must_equal(artifact.get("schema_version"), SHARD_SCHEMA_VERSION, "retrieval artifact.schema_version")
    _must_equal(artifact.get("artifact_type"), RETRIEVAL_ARTIFACT_TYPE, "retrieval artifact.artifact_type")
    _must_equal(artifact.get("arm_id"), ARM_ID, "retrieval artifact.arm_id")
    _must_equal(artifact.get("run_status"), RUN_STATUS, "retrieval artifact.run_status")
    _must_equal(artifact.get("sample_offset"), expected.sample_offset, "retrieval artifact.sample_offset")
    _must_equal(artifact.get("identity"), dict(expected_identity), "retrieval artifact.identity")
    _must_equal(artifact.get("model_identity"), dict(model_identity), "retrieval artifact.model_identity")
    _must_equal(artifact.get("config"), dict(config), "retrieval artifact.config")
    _validate_sample(artifact.get("sample"), expected, "retrieval artifact.sample")
    _must_equal(artifact.get("raw_input_receipt"), dict(raw_receipt), "retrieval artifact.raw_input_receipt")
    _must_equal(artifact.get("ingestion_receipt"), dict(ingestion_receipt), "retrieval artifact.ingestion_receipt")
    _must_equal(artifact.get("mem0_usage"), dict(mem0_usage), "retrieval artifact.mem0_usage")
    _must_equal(artifact.get("provenance"), dict(provenance), "retrieval artifact.provenance")

    trace, trace_digest = _validate_trace(
        artifact.get("trace"),
        report_path=artifact_path,
        expected_format=RETRIEVAL_TRACE_FORMAT,
        label="retrieval artifact.trace",
    )
    _must_equal(descriptor.get("trace"), trace, f"{label}.trace")

    raw_rows = _list(artifact.get("question_results"), "retrieval artifact.question_results")
    if len(raw_rows) != len(expected.questions):
        raise Mem0ReportError("retrieval artifact must contain exactly ten question rows")
    rows: list[dict[str, Any]] = []
    for index, expected_question in enumerate(expected.questions):
        rows.append(
            _validate_retrieval_row(
                raw_rows[index],
                expected_question=expected_question,
                prompt_cap=prompt_cap,
                label=f"retrieval artifact.question_results[{index}]",
            )
        )
    if tuple(str(row["question_id"]) for row in rows) != expected.question_ids:
        raise Mem0ReportError("retrieval artifact question order is not frozen order")
    return artifact, tuple(rows), trace_digest


def _validate_usage(value: Any, label: str) -> dict[str, int | float]:
    usage = _mapping(value, label)
    _require_exact_fields(
        usage,
        (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "elapsed_s",
            "calls",
        ),
        label,
    )
    return {
        "input_tokens": _integer(usage.get("input_tokens"), f"{label}.input_tokens"),
        "output_tokens": _integer(usage.get("output_tokens"), f"{label}.output_tokens"),
        "cache_read_input_tokens": _integer(
            usage.get("cache_read_input_tokens"),
            f"{label}.cache_read_input_tokens",
        ),
        "elapsed_s": _number(usage.get("elapsed_s"), f"{label}.elapsed_s", minimum=0.0),
        "calls": _integer(usage.get("calls"), f"{label}.calls"),
    }


def _judge_verdict(value: Any, label: str) -> bool:
    reasoning = _string(value, label)
    match = _JUDGE_RE.match(reasoning)
    if match is None:
        raise Mem0ReportError(f"{label} has no exact binary judge verdict")
    remainder = match.group("remainder").lstrip(" \t\r\n,.:;-—")
    if remainder.casefold().startswith("or ") or remainder.startswith("/"):
        raise Mem0ReportError(f"{label} contains an ambiguous judge verdict")
    return match.group(1).casefold() == "correct"


_RETRIEVAL_BIND_FIELDS = (
    "question_id",
    "query",
    "context_sha256",
    "context_tokens",
    "messages_sha256",
    "prompt_token_proxy",
    "max_prompt_token_proxy",
    "residual_prompt_token_proxy",
    "raw_memory_count",
    "raw_memory_tokens",
    "raw_pool_sha256",
    "packed_memory_count",
    "packed_memory_tokens",
    "packed_pool_sha256",
    "search_latency_s",
    "provenance",
)


def _validate_scored_question(
    value: Any,
    *,
    retrieval_row: Mapping[str, Any],
    expected_question: Mapping[str, Any],
    prompt_cap: int,
    output_reserve: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, int | float], dict[str, int | float]]:
    row = _mapping(value, label)
    _must_equal(
        row.get("retrieval_row_sha256"),
        retrieval_row.get("retrieval_row_sha256"),
        f"{label}.retrieval_row_sha256",
    )
    for field in _RETRIEVAL_BIND_FIELDS:
        _must_equal(row.get(field), retrieval_row.get(field), f"{label}.{field}")
    for field in ("question_id", "question", "gold_answer", "category"):
        _must_equal(row.get(field), expected_question.get(field), f"{label}.{field}")
    prediction = _string(row.get("predicted_answer"), f"{label}.predicted_answer")
    gold = str(expected_question["gold_answer"])
    recomputed_f1 = f1_score(prediction, gold)
    recomputed_em = exact_match(prediction, gold)
    reported_f1 = _number(row.get("f1"), f"{label}.f1", minimum=0.0, maximum=1.0)
    if not math.isclose(reported_f1, recomputed_f1, rel_tol=0.0, abs_tol=1e-15):
        raise Mem0ReportError(f"{label}.f1 disagrees with the answer strings")
    _must_equal(row.get("exact_match"), recomputed_em, f"{label}.exact_match")
    verdict = _judge_verdict(row.get("judge_reasoning"), f"{label}.judge_reasoning")
    _must_equal(row.get("judge_correct"), verdict, f"{label}.judge_correct")
    _must_equal(
        _field(row, "source_coverage", label),
        None,
        f"{label}.source_coverage",
    )

    prompt_proxy = int(retrieval_row["prompt_token_proxy"])
    _must_equal(
        row.get("request_token_proxy"),
        prompt_proxy + output_reserve,
        f"{label}.request_token_proxy",
    )
    if prompt_proxy > prompt_cap:
        raise Mem0ReportError(f"{label} exceeds the prompt-token cap")
    responder = _validate_usage(row.get("responder_usage"), f"{label}.responder_usage")
    judge = _validate_usage(row.get("judge_usage"), f"{label}.judge_usage")
    if responder["calls"] != 1 or judge["calls"] != 1:
        raise Mem0ReportError(
            f"{label} must bind one completed responder and one judge call"
        )
    provider_input = int(responder["input_tokens"])
    expected_compliance: bool | None = (
        None if provider_input == 0 else provider_input <= prompt_cap
    )
    _must_equal(
        _field(row, "provider_prompt_budget_compliant", label),
        expected_compliance,
        f"{label}.provider_prompt_budget_compliant",
    )
    if expected_compliance is False:
        raise Mem0ReportError(f"{label} provider input usage exceeds the prompt cap")
    row["f1"] = recomputed_f1
    row["exact_match"] = recomputed_em
    row["judge_correct"] = verdict
    return row, responder, judge


def _validate_scoring_receipt(
    value: Any,
    *,
    report_path: Path,
    artifact_sha256: str,
    question_count: int,
    plan: LockedValidationPlan,
    label: str,
) -> tuple[dict[str, Any], str]:
    receipt = _mapping(value, label)
    expected = {
        "format": SCORING_RECEIPT_FORMAT,
        "retrieval_artifact_sha256": artifact_sha256,
        "scoring_environment_lock_sha256": plan.environment_lock_sha256,
        "responder_model": plan.evaluation["responder_model"],
        "judge_model": plan.evaluation["judge_model"],
        "provider_retries": 0,
        "authorized_provider_calls": 2 * question_count,
        "responder_attempted": question_count,
        "responder_completed": question_count,
        "responder_failed": 0,
        "judge_attempted": question_count,
        "judge_completed": question_count,
        "judge_failed": 0,
        "persisted_transformer_token_state": False,
    }
    for field, wanted in expected.items():
        _must_equal(receipt.get(field), wanted, f"{label}.{field}")
    _number(receipt.get("elapsed_s"), f"{label}.elapsed_s", minimum=0.0)
    _must_equal(
        receipt.get("prompt_token_proxy_identity"),
        plan.evaluation["prompt_token_proxy_identity"],
        f"{label}.prompt_token_proxy_identity",
    )
    trace, trace_digest = _validate_trace(
        receipt.get("trace"),
        report_path=report_path,
        expected_format=SCORING_TRACE_FORMAT,
        label=f"{label}.trace",
    )
    receipt["trace"] = trace
    return receipt, trace_digest


def _validate_json_trace_descriptor(
    value: Any,
    *,
    owner_path: Path,
    expected_format: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate one native ``run_shard`` JSON-object trace binding."""

    descriptor = _mapping(value, label)
    trace_path = _safe_sibling(
        owner_path, descriptor.get("filename"), f"{label}.filename"
    )
    trace, payload = _load_json(trace_path, label)
    digest = hashlib.sha256(payload).hexdigest()
    _must_equal(descriptor.get("sha256"), digest, f"{label}.sha256")
    _must_equal(descriptor.get("bytes"), len(payload), f"{label}.bytes")
    _must_equal(trace.get("format"), expected_format, f"{label} payload.format")
    _must_equal(trace.get("status"), RUN_STATUS, f"{label} payload.status")
    return descriptor, trace, digest


def _validate_native_events(value: Any, label: str) -> list[dict[str, Any]]:
    events = [
        _mapping(row, f"{label}[{index}]")
        for index, row in enumerate(_list(value, label))
    ]
    if not events:
        raise Mem0ReportError(f"{label} must not be empty")
    _must_equal(
        [row.get("sequence") for row in events],
        list(range(1, len(events) + 1)),
        f"{label} sequence",
    )
    for index, row in enumerate(events):
        _string(row.get("event"), f"{label}[{index}].event")
    return events


def _validate_extraction_calls(
    value: Any, *, expected_calls: int, label: str
) -> dict[str, Any]:
    receipt = _mapping(value, label)
    exact = {
        "kind": "mem0_memory_llm_generate_response_logical_calls",
        "boundary": "Memory.llm.generate_response",
        "count_semantics": "local_logical_wrapper_calls_not_http_attempts",
        "external_http_attempts_certified": False,
        "authorized_local_wrapper_retries": 0,
        "external_retry_attempts_certified": False,
        "authorized": expected_calls,
        "attempted": expected_calls,
        "completed": expected_calls,
        "failed": 0,
        "rejected": 0,
        "infer_true_adds_started": expected_calls,
        "infer_true_adds_exactly_one_call": expected_calls,
        "one_logical_call_per_infer_true_add_certified": True,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }
    for field, wanted in exact.items():
        _must_equal(receipt.get(field), wanted, f"{label}.{field}")
    return receipt


def _validate_native_call_budget(
    value: Any, *, expected_calls: int, label: str
) -> dict[str, Any]:
    receipt = _mapping(value, label)
    exact = {
        "authorized": expected_calls,
        "attempted": expected_calls,
        "completed": expected_calls,
        "failed": 0,
    }
    if set(receipt) != set(exact):
        raise Mem0ReportError(f"{label} fields do not match the native contract")
    for field, wanted in exact.items():
        _must_equal(receipt.get(field), wanted, f"{label}.{field}")
    return receipt


def _validate_stateless_provider_contracts(value: Any, label: str) -> dict[str, Any]:
    contracts = _mapping(value, label)
    if set(contracts) != {"responder", "judge"}:
        raise Mem0ReportError(
            f"{label} must contain exactly responder and judge contracts"
        )
    for name in ("responder", "judge"):
        receipt = _mapping(contracts[name], f"{label}.{name}")
        for field, wanted in {
            "contract": "stateless-request-token-state-v1",
            "persisted_request_token_state": False,
            "retained_request_token_state_bytes": 0,
            "request_token_state_evidence_kind": (
                "local_injected_request_token_state_contract"
            ),
            "external_provider_persistence_certified": False,
        }.items():
            _must_equal(receipt.get(field), wanted, f"{label}.{name}.{field}")
        _reject_secret_material(receipt, f"{label}.{name}")
    return contracts


def _validate_execution_binding(value: Any, label: str) -> dict[str, Any]:
    receipt = _mapping(value, label)
    exact = {
        "kind": "injected_nonproduction",
        "trusted_runtime_binding_receipt_sha256": None,
        "comparison_certified": False,
        "external_http_attempts_certified": False,
        "external_provider_persistence_certified": False,
    }
    if set(receipt) != set(exact):
        raise Mem0ReportError(f"{label} fields do not match the native contract")
    for field, wanted in exact.items():
        _must_equal(receipt.get(field), wanted, f"{label}.{field}")
    return receipt


def _validate_environment_lock_receipt(
    value: Any,
    *,
    expected_sha256: str,
    expected_filename: str | None,
    label: str,
) -> dict[str, Any]:
    receipt = _mapping(value, label)
    exact_fields = {
        "filename",
        "authorized_sha256",
        "sha256_before",
        "sha256_after",
        "unchanged",
    }
    if set(receipt) != exact_fields:
        raise Mem0ReportError(f"{label} fields do not match the native contract")
    filename = _string(receipt.get("filename"), f"{label}.filename")
    if Path(filename).name != filename:
        raise Mem0ReportError(f"{label}.filename must be a basename")
    if expected_filename is not None:
        _must_equal(filename, expected_filename, f"{label}.filename")
    for field in ("authorized_sha256", "sha256_before", "sha256_after"):
        _must_equal(receipt.get(field), expected_sha256, f"{label}.{field}")
    _must_equal(receipt.get("unchanged"), True, f"{label}.unchanged")
    return receipt


def _validate_native_model_config(
    document: Mapping[str, Any],
    *,
    population: FrozenMem0Population,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = _mapping(document.get("identity"), f"{label}.identity")
    expected_identity = _expected_identity(population)
    identity_fields = _require_exact_fields(
        identity,
        (
            *expected_identity,
            "mem0_stable_config_sha256",
            "extraction_model_identity",
            "extraction_model_identity_sha256",
            "embedder_model_identity",
            "embedder_model_identity_sha256",
            "scoring_policy_sha256",
            "source_evaluation_identity_sha256",
        ),
        f"{label}.identity",
    )
    for field, wanted in expected_identity.items():
        _must_equal(identity.get(field), wanted, f"{label}.identity.{field}")
    stable_config_sha = _sha256(
        identity.get("mem0_stable_config_sha256"),
        f"{label}.identity.mem0_stable_config_sha256",
    )
    _must_equal(
        stable_config_sha,
        population.mem0_policy.stable_config_sha256,
        f"{label}.identity.mem0_stable_config_sha256",
    )
    scoring_policy_sha = _sha256(
        identity.get("scoring_policy_sha256"),
        f"{label}.identity.scoring_policy_sha256",
    )
    _must_equal(
        scoring_policy_sha,
        population.mem0_policy.sha256,
        f"{label}.identity.scoring_policy_sha256",
    )
    _must_equal(
        identity.get("source_evaluation_identity_sha256"),
        canonical_sha256(population.source_evaluation_identity),
        f"{label}.identity.source_evaluation_identity_sha256",
    )
    for prefix, policy_value in (
        ("extraction", population.mem0_policy.extraction_identity),
        ("embedder", population.mem0_policy.embedder_identity),
    ):
        expected_model_identity = _plain_json(policy_value)
        field = f"{prefix}_model_identity"
        digest_field = f"{field}_sha256"
        _must_equal(
            identity.get(field),
            expected_model_identity,
            f"{label}.identity.{field}",
        )
        _must_equal(
            identity.get(digest_field),
            canonical_sha256(expected_model_identity),
            f"{label}.identity.{digest_field}",
        )
    embedder_identity = _plain_json(population.mem0_policy.embedder_identity)
    for field, wanted in {
        "execution": "local_offline",
        "network_calls_authorized": 0,
        "runtime_probe_required": True,
    }.items():
        _must_equal(
            embedder_identity.get(field),
            wanted,
            f"{label}.identity.embedder_model_identity.{field}",
        )

    source_identity = population.source_evaluation_identity
    policy_scoring = _plain_json(population.mem0_policy.scoring)
    model_identity = _mapping(
        document.get("model_identity"), f"{label}.model_identity"
    )
    model_identity_fields = _require_exact_fields(
        model_identity,
        (
            "responder_model",
            "responder_model_identity_sha256",
            "judge_model",
            "judge_model_identity_sha256",
        ),
        f"{label}.model_identity",
    )
    for field, wanted in {
        "responder_model": source_identity["responder_model"],
        "judge_model": source_identity["judge_model"],
    }.items():
        _must_equal(
            model_identity.get(field), wanted, f"{label}.model_identity.{field}"
        )
    for field in (
        "responder_model_identity_sha256",
        "judge_model_identity_sha256",
    ):
        _sha256(model_identity.get(field), f"{label}.model_identity.{field}")
    _must_equal(
        model_identity.get("responder_model_identity_sha256"),
        policy_scoring["responder_identity_sha256"],
        f"{label}.model_identity.responder_model_identity_sha256",
    )
    _must_equal(
        model_identity.get("judge_model_identity_sha256"),
        policy_scoring["judge_identity_sha256"],
        f"{label}.model_identity.judge_model_identity_sha256",
    )
    _reject_secret_material(model_identity, f"{label}.model_identity")
    _must_equal(
        document.get("model_identity_sha256"),
        canonical_sha256(model_identity),
        f"{label}.model_identity_sha256",
    )

    config = _mapping(document.get("config"), f"{label}.config")
    expected_config = {
        "max_prompt_tokens": source_identity["max_prompt_tokens"],
        "responder_max_output_tokens": source_identity[
            "responder_output_token_reserve"
        ],
        "judge_max_output_tokens": policy_scoring["judge_max_output_tokens"],
        "authorized_local_wrapper_retries": 0,
        "external_retry_attempts_certified": False,
        "mem0_top_k": MEM0_OFFICIAL_TOP_K,
        "mem0_threshold": MEM0_OFFICIAL_THRESHOLD,
        "rendering_mode": MEM0_CERTIFIED_RENDERING,
    }
    config_fields = _require_exact_fields(
        config, expected_config, f"{label}.config"
    )
    for field, wanted in expected_config.items():
        _must_equal(config.get(field), wanted, f"{label}.config.{field}")
    _reject_secret_material(config, f"{label}.config")
    _must_equal(
        document.get("config_sha256"),
        canonical_sha256(config),
        f"{label}.config_sha256",
    )
    projected_identity = {field: identity[field] for field in identity_fields}
    projected_identity["mem0_stable_config_sha256"] = stable_config_sha
    return (
        projected_identity,
        {field: model_identity[field] for field in model_identity_fields},
        {field: config[field] for field in config_fields},
    )


def _validate_native_protocol(
    value: Any,
    *,
    population: FrozenMem0Population,
    label: str,
) -> dict[str, Any]:
    protocol = _mapping(value, label)
    source = population.source_evaluation_identity
    expected = {
        "split": "validation",
        "stress_context_tokens": source["stress_context_tokens"],
        "stress_questions": source["stress_questions"],
        "stress_question_offset": source["stress_question_offset"],
        "answer_prompt_calls_per_question": 1,
        "judge_calls_per_question": 1,
        "authorized_local_wrapper_retries": 0,
        "external_retry_attempts_certified": False,
    }
    for field, wanted in expected.items():
        _must_equal(protocol.get(field), wanted, f"{label}.{field}")
    return protocol


def _validate_native_sample(
    value: Any, expected: ExpectedMem0Shard, label: str
) -> dict[str, Any]:
    sample = _mapping(value, label)
    exact = {
        "sample_id": expected.sample_id,
        "sample_sha256": expected.sample_sha256,
        "raw_history_bundle_sha256": expected.raw_history_bundle_sha256,
        "history_sample_ids": list(expected.history_sample_ids),
        "question_ids": list(expected.question_ids),
    }
    for field, wanted in exact.items():
        _must_equal(sample.get(field), wanted, f"{label}.{field}")
    return sample


def _validate_native_raw_receipt(
    value: Any, expected: ExpectedMem0Shard, label: str
) -> dict[str, Any]:
    receipt = _mapping(value, label)
    exact = {
        "format": "memory-condense-mem0-raw-shard-v1",
        "sample_offset": expected.sample_offset,
        "sample_id": expected.sample_id,
        "sample_sha256": expected.sample_sha256,
        "raw_history_bundle_sha256": expected.raw_history_bundle_sha256,
        "history_samples": len(expected.history_sample_ids),
        "questions": len(expected.questions),
        "question_ids": list(expected.question_ids),
        "turns": expected.num_turns,
        "transcript_tokens": expected.transcript_tokens,
        "raw_pairs": expected.raw_pairs,
        "skipped_empty_pairs": expected.skipped_empty_pairs,
        "mem0_add_requests": expected.expected_adds,
        "whitespace_only_pairs": 0,
    }
    for field, wanted in exact.items():
        _must_equal(receipt.get(field), wanted, f"{label}.{field}")
    return receipt


def _validate_native_ingestion(
    value: Any, expected: ExpectedMem0Shard, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _mapping(value, label)
    exact = {
        "raw_pairs": expected.raw_pairs,
        "skipped_empty_pairs": expected.skipped_empty_pairs,
        "authorized_add_operations": expected.expected_adds,
        "attempted_add_operations": expected.expected_adds,
        "completed_add_operations": expected.expected_adds,
        "failed_add_operations": 0,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
        "one_scope": True,
        "comparison_certified": False,
    }
    for field, wanted in exact.items():
        _must_equal(receipt.get(field), wanted, f"{label}.{field}")
    _sha256(receipt.get("user_scope_sha256"), f"{label}.user_scope_sha256")
    extraction = _validate_extraction_calls(
        receipt.get("extraction_model_calls"),
        expected_calls=expected.expected_adds,
        label=f"{label}.extraction_model_calls",
    )
    return receipt, extraction


def _validate_native_usage(
    value: Any, expected: ExpectedMem0Shard, label: str
) -> dict[str, Any]:
    usage = _mapping(value, label)
    fields = set(Mem0AdapterStats.__dataclass_fields__)
    if set(usage) != fields:
        raise Mem0ReportError(
            f"{label} fields mismatch: missing={sorted(fields - set(usage))!r}, "
            f"extra={sorted(set(usage) - fields)!r}"
        )
    exact_counts = {
        "add_calls": expected.expected_adds,
        "add_attempted_calls": expected.expected_adds,
        "add_completed_calls": expected.expected_adds,
        "add_failed_calls": 0,
        "search_calls": len(expected.questions),
        "released_scopes": 0,
    }
    for field, wanted in exact_counts.items():
        _must_equal(usage.get(field), wanted, f"{label}.{field}")
    for field in (
        "add_raw_message_tokens",
        "search_query_tokens",
        "search_raw_memory_tokens",
        "search_context_tokens",
        "search_prompt_token_proxy",
        "search_prompt_tokens",
        "add_returned_memories",
        "unique_ledger_memories",
        "search_returned_memories",
        "search_packed_memories",
    ):
        _integer(usage.get(field), f"{label}.{field}")
    for field in ("add_latency_s", "search_latency_s"):
        _number(usage.get(field), f"{label}.{field}", minimum=0.0)
    for field in ("provider_prompt_tokens", "provider_completion_tokens"):
        _must_equal(usage.get(field), None, f"{label}.{field}")
    _must_equal(
        usage.get("provider_usage_status"),
        MEM0_PROVIDER_USAGE_STATUS,
        f"{label}.provider_usage_status",
    )
    if not isinstance(usage.get("token_counter_identity"), str):
        raise Mem0ReportError(f"{label}.token_counter_identity must be a string")
    _boolean(
        usage.get("token_counter_identity_verified"),
        f"{label}.token_counter_identity_verified",
    )
    return usage


def _validate_native_provenance(value: Any, label: str) -> dict[str, Any]:
    provenance = _mapping(value, label)
    for field, wanted in {
        "attribution_kind": MEM0_ATTRIBUTION_KIND,
        "supports_exact_source_provenance": False,
        "source_session_date_exposure": "diagnostics_only_not_model_input",
        "retrieved_created_at_exposure": "answer_prompt_date_headings",
        "provider_usage_status": MEM0_PROVIDER_USAGE_STATUS,
        "external_http_attempts_certified": False,
        "external_retry_attempts_certified": False,
        "external_provider_persistence_certified": False,
    }.items():
        _must_equal(provenance.get(field), wanted, f"{label}.{field}")
    return provenance


def _validate_native_retrieval_artifact(
    descriptor_value: Any,
    *,
    report_path: Path,
    report_identity: Mapping[str, Any],
    expected: ExpectedMem0Shard,
    population: FrozenMem0Population,
    report_raw_receipt: Mapping[str, Any],
    report_ingestion: Mapping[str, Any],
    report_usage: Mapping[str, Any],
    report_provenance: Mapping[str, Any],
    label: str,
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    str,
    str,
]:
    descriptor = _mapping(descriptor_value, label)
    artifact_path = _safe_sibling(
        report_path, descriptor.get("filename"), f"{label}.filename"
    )
    artifact, payload = _load_json(artifact_path, "Mem0 retrieval artifact")
    artifact_digest = hashlib.sha256(payload).hexdigest()
    _must_equal(descriptor.get("sha256"), artifact_digest, f"{label}.sha256")
    _must_equal(descriptor.get("bytes"), len(payload), f"{label}.bytes")
    _must_equal(
        descriptor.get("question_ids"),
        list(expected.question_ids),
        f"{label}.question_ids",
    )

    content_digest = _sha256(
        artifact.get("content_sha256"), "retrieval artifact.content_sha256"
    )
    content = dict(artifact)
    del content["content_sha256"]
    _must_equal(
        content_digest,
        canonical_sha256(content),
        "retrieval artifact.content_sha256",
    )
    for field, wanted in {
        "format": RETRIEVAL_ARTIFACT_FORMAT,
        "status": RUN_STATUS,
        "certification_status": "injected_nonproduction",
        "comparison_certified": False,
        "sample_offset": expected.sample_offset,
        "sample_id": expected.sample_id,
        "sample_sha256": expected.sample_sha256,
        "raw_history_bundle_sha256": expected.raw_history_bundle_sha256,
        "history_sample_ids_sha256": expected.contributor_ids_sha256,
        "question_ids": list(expected.question_ids),
        "question_ids_sha256": canonical_sha256(list(expected.question_ids)),
    }.items():
        _must_equal(artifact.get(field), wanted, f"retrieval artifact.{field}")
    execution_binding = _validate_execution_binding(
        artifact.get("execution_binding"), "retrieval artifact.execution_binding"
    )

    artifact_identity = _mapping(
        artifact.get("identity"), "retrieval artifact.identity"
    )
    for field in (
        "source_validation_policy_sha256",
        "source_implementation_sha256",
        "source_environment_lock_sha256",
        "mem0_policy_sha256",
        "mem0_tool_implementation_sha256",
        "mem0_environment_lock_sha256",
        "mem0_stable_config_sha256",
        "extraction_model_identity",
        "extraction_model_identity_sha256",
        "embedder_model_identity",
        "embedder_model_identity_sha256",
    ):
        _must_equal(
            artifact_identity.get(field),
            report_identity.get(field),
            f"retrieval artifact.identity.{field}",
        )
    runtime_probe = _mapping(
        artifact_identity.get("runtime_model_identity_probe"),
        "retrieval artifact.identity.runtime_model_identity_probe",
    )
    expected_runtime_probe = {
        "kind": "unavailable_injected_nonproduction",
        "extraction_model_identity_sha256": report_identity[
            "extraction_model_identity_sha256"
        ],
        "embedder_model_identity_sha256": report_identity[
            "embedder_model_identity_sha256"
        ],
        "before_match": False,
        "after_match": False,
        "comparison_certified": False,
    }
    if set(runtime_probe) != set(expected_runtime_probe):
        raise Mem0ReportError(
            "retrieval artifact runtime model probe fields do not match the native contract"
        )
    for field, wanted in expected_runtime_probe.items():
        _must_equal(
            runtime_probe.get(field),
            wanted,
            f"retrieval artifact.identity.runtime_model_identity_probe.{field}",
        )
    _must_equal(
        artifact_identity.get("source_evaluation_identity"),
        dict(population.source_evaluation_identity),
        "retrieval artifact.identity.source_evaluation_identity",
    )
    _must_equal(
        artifact_identity.get("source_evaluation_identity_sha256"),
        canonical_sha256(population.source_evaluation_identity),
        "retrieval artifact.identity.source_evaluation_identity_sha256",
    )
    runtime = _validate_runtime_identity(
        artifact_identity.get("runtime_identity"),
        "retrieval artifact.identity.runtime_identity",
    )
    expected_runtime_fields = {
        "protocol",
        "config",
        "stack",
        "stable_config_sha256",
        "effective_config_sha256",
        "local_owned_state",
        "on_disk",
        "certified",
        "persisted_request_token_state",
        "retained_request_token_state_bytes",
        "request_token_state_evidence_kind",
        "external_provider_persistence_certified",
    }
    if set(runtime) != expected_runtime_fields:
        raise Mem0ReportError(
            "retrieval artifact runtime identity fields do not match the native contract"
        )
    _must_equal(
        runtime.get("stable_config_sha256"),
        report_identity["mem0_stable_config_sha256"],
        "retrieval artifact runtime stable_config_sha256",
    )
    observed_stable_payload = {
        "protocol": runtime["protocol"],
        "config": runtime["config"],
        "stack": runtime["stack"],
    }
    _must_equal(
        canonical_sha256(observed_stable_payload),
        report_identity["mem0_stable_config_sha256"],
        "retrieval artifact runtime stable payload SHA-256",
    )
    _must_equal(
        observed_stable_payload,
        _plain_json(population.mem0_policy.stable_payload),
        "retrieval artifact runtime stable payload policy binding",
    )
    for field, wanted in {
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }.items():
        _must_equal(
            runtime.get(field),
            wanted,
            f"retrieval artifact runtime.{field}",
        )
    stack = _mapping(runtime.get("stack"), "retrieval artifact runtime.stack")
    _must_equal(
        stack.get("bm25_model"),
        MEM0_BM25_MODEL,
        "retrieval artifact runtime.stack.bm25_model",
    )
    _must_equal(
        stack.get("spacy_model"),
        MEM0_SPACY_MODEL,
        "retrieval artifact runtime.stack.spacy_model",
    )

    protocol = _mapping(artifact.get("protocol"), "retrieval artifact.protocol")
    for field, wanted in {
        "input_order": INPUT_ORDER_PROTOCOL,
        "official_longmemeval_protocol": True,
        "official_search_protocol": True,
        "top_k": MEM0_OFFICIAL_TOP_K,
        "threshold": MEM0_OFFICIAL_THRESHOLD,
        "rendering_mode": MEM0_CERTIFIED_RENDERING,
        "max_prompt_tokens": population.source_evaluation_identity[
            "max_prompt_tokens"
        ],
    }.items():
        _must_equal(protocol.get(field), wanted, f"retrieval artifact.protocol.{field}")

    raw_receipt = _validate_native_raw_receipt(
        artifact.get("raw_input_receipt"),
        expected,
        "retrieval artifact.raw_input_receipt",
    )
    _must_equal(
        raw_receipt,
        dict(report_raw_receipt),
        "retrieval artifact.raw_input_receipt report binding",
    )
    ingestion, extraction = _validate_native_ingestion(
        artifact.get("ingestion_receipt"),
        expected,
        "retrieval artifact.ingestion_receipt",
    )
    _must_equal(
        ingestion,
        dict(report_ingestion),
        "retrieval artifact.ingestion_receipt report binding",
    )
    search = _mapping(artifact.get("search_receipt"), "retrieval artifact.search_receipt")
    for field, wanted in {
        "authorized_search_operations": len(expected.questions),
        "completed_search_operations": len(expected.questions),
        "failed_search_operations": 0,
    }.items():
        _must_equal(search.get(field), wanted, f"retrieval artifact.search_receipt.{field}")
    usage = _validate_native_usage(
        artifact.get("mem0_usage"), expected, "retrieval artifact.mem0_usage"
    )
    _must_equal(usage, dict(report_usage), "retrieval artifact.mem0_usage report binding")
    provenance = _validate_native_provenance(
        artifact.get("provenance"), "retrieval artifact.provenance"
    )
    _must_equal(
        provenance,
        dict(report_provenance),
        "retrieval artifact.provenance report binding",
    )

    trace_descriptor, trace, trace_digest = _validate_json_trace_descriptor(
        artifact.get("retrieval_trace"),
        owner_path=artifact_path,
        expected_format=RETRIEVAL_TRACE_FORMAT,
        label="retrieval artifact.retrieval_trace",
    )
    _reject_secret_material(trace, "Mem0 retrieval trace")
    _must_equal(
        descriptor.get("retrieval_trace"),
        trace_descriptor,
        f"{label}.retrieval_trace",
    )
    _must_equal(
        trace.get("execution_binding"),
        execution_binding,
        "retrieval trace.execution_binding",
    )
    for field, wanted in {
        "certification_status": "injected_nonproduction",
        "comparison_certified": False,
    }.items():
        _must_equal(trace.get(field), wanted, f"retrieval trace.{field}")
    for field, wanted in {
        "stage": "retrieval",
        "sample_offset": expected.sample_offset,
        "sample_id": expected.sample_id,
        "sample_sha256": expected.sample_sha256,
    }.items():
        _must_equal(trace.get(field), wanted, f"retrieval trace.{field}")
    _number(trace.get("elapsed_s"), "retrieval trace.elapsed_s", minimum=0.0)
    cleanup = _mapping(trace.get("cleanup"), "retrieval trace.cleanup")
    for field, wanted in {
        "attempted": True,
        "completed": True,
        "state_absent_before": True,
        "state_absent_after": True,
        "active_scope_cleared": True,
        "extraction_meter_restore_attempted": True,
        "extraction_meter_restored_before_cleanup": True,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
        "adapter_closed": True,
        "ledger_empty": True,
        "registered_scopes_empty": True,
        "scope_protocol_empty": True,
        "backend_closed_or_cleared": True,
        "owned_state_path_absent": True,
    }.items():
        _must_equal(cleanup.get(field), wanted, f"retrieval trace.cleanup.{field}")
    environment_lock = _validate_environment_lock_receipt(
        artifact.get("environment_lock"),
        expected_sha256=population.mem0_environment_lock_sha256,
        expected_filename=population.mem0_environment_lock_path.name,
        label="retrieval artifact.environment_lock",
    )
    _must_equal(
        trace.get("environment_lock"),
        environment_lock,
        "retrieval trace.environment_lock",
    )
    _must_equal(
        cleanup.get("environment_lock"),
        environment_lock,
        "retrieval trace.cleanup.environment_lock",
    )
    cleanup_extraction = _validate_extraction_calls(
        cleanup.get("extraction_model_calls"),
        expected_calls=expected.expected_adds,
        label="retrieval trace.cleanup.extraction_model_calls",
    )
    _must_equal(
        cleanup_extraction,
        extraction,
        "retrieval trace cleanup extraction receipt binding",
    )

    raw_rows = _list(artifact.get("retrieval_rows"), "retrieval artifact.retrieval_rows")
    if len(raw_rows) != len(expected.questions):
        raise Mem0ReportError(
            "retrieval artifact must contain exactly the frozen question count"
        )
    rows: list[dict[str, Any]] = []
    prompt_cap = int(population.source_evaluation_identity["max_prompt_tokens"])
    for index, expected_question in enumerate(expected.questions):
        row = _validate_retrieval_row(
            raw_rows[index],
            expected_question=expected_question,
            prompt_cap=prompt_cap,
            label=f"retrieval artifact.retrieval_rows[{index}]",
        )
        _must_equal(
            row.get("source_evaluation_identity"),
            dict(population.source_evaluation_identity),
            f"retrieval artifact.retrieval_rows[{index}].source_evaluation_identity",
        )
        independent_result = {
            "query": row["query"],
            "raw_pool": row["raw_pool"],
            "official_longmemeval_protocol": True,
            "official_search_protocol": True,
            "rendering_mode": MEM0_CERTIFIED_RENDERING,
            "certified_rendering": True,
            "comparison_certified": True,
            "runtime_identity": runtime,
            "attribution_kind": MEM0_ATTRIBUTION_KIND,
            "supports_exact_source_provenance": False,
        }
        rebuilt_pack = pack_mem0_prompt(
            str(row["query"]),
            independent_result,
            max_prompt_tokens=prompt_cap,
            evaluation_identity=population.source_evaluation_identity,
        )
        rebuilt = rebuilt_pack.to_retrieval_row(
            question_id=str(row["question_id"]),
            search_latency_s=float(row["search_latency_s"]),
        )
        _must_equal(
            row,
            rebuilt,
            f"retrieval artifact.retrieval_rows[{index}] independent prompt rebuild",
        )
        rows.append(row)

    events = _validate_native_events(trace.get("events"), "retrieval trace.events")
    event_names = [str(row["event"]) for row in events]
    expected_event_names = [
        "authorization_verified",
        "prepared_corpus_built",
        "extraction_meter_installed",
        "ingest_complete",
        *(["search_complete"] * len(rows)),
        "cleanup_complete",
    ]
    _must_equal(event_names, expected_event_names, "retrieval trace event protocol")
    prepared = events[1]
    _must_equal(prepared.get("batches"), expected.expected_adds, "retrieval trace prepared batches")
    batch_hashes = _list(
        prepared.get("ordered_batch_hashes"),
        "retrieval trace prepared ordered_batch_hashes",
    )
    if len(batch_hashes) != expected.expected_adds:
        raise Mem0ReportError("retrieval trace prepared batch hash count mismatch")
    for index, digest in enumerate(batch_hashes):
        _sha256(digest, f"retrieval trace prepared batch hash[{index}]")
    _must_equal(
        events[2].get("authorized_logical_calls"),
        expected.expected_adds,
        "retrieval trace extraction authorization",
    )
    _must_equal(
        events[2].get("authorized_local_wrapper_retries"),
        0,
        "retrieval trace extraction authorized local-wrapper retries",
    )
    _must_equal(
        events[2].get("external_http_attempts_certified"),
        False,
        "retrieval trace extraction HTTP-attempt status",
    )
    _must_equal(
        events[2].get("external_retry_attempts_certified"),
        False,
        "retrieval trace extraction retry-attempt status",
    )
    _must_equal(events[3].get("add_operations"), expected.expected_adds, "retrieval trace add count")
    _must_equal(
        events[3].get("logical_extraction_calls"),
        expected.expected_adds,
        "retrieval trace extraction call count",
    )
    for index, row in enumerate(rows):
        event = events[4 + index]
        for field, wanted in {
            "question_id": row["question_id"],
            "query_sha256": text_sha256(str(row["query"])),
            "raw_memory_count": row["raw_memory_count"],
            "raw_pool_sha256": row["raw_pool_sha256"],
            "retrieval_row_sha256": row["retrieval_row_sha256"],
        }.items():
            _must_equal(event.get(field), wanted, f"retrieval trace search event {index}.{field}")
    _must_equal(
        events[-1].get("state_absent_after"),
        True,
        "retrieval trace cleanup event state_absent_after",
    )
    return artifact, tuple(rows), runtime, artifact_digest, trace_digest


_NATIVE_RETRIEVAL_FIELD_MAP = {
    "context": "context",
    "context_sha256": "context_sha256",
    "context_tokens": "context_tokens",
    "messages": "messages",
    "messages_sha256": "messages_sha256",
    "prompt_token_proxy": "prompt_token_proxy",
    "max_prompt_tokens": "max_prompt_token_proxy",
    "residual_prompt_tokens": "residual_prompt_token_proxy",
    "prompt_token_proxy_identity": "prompt_token_proxy_identity",
    "raw_pool_count": "raw_memory_count",
    "raw_pool_sha256": "raw_pool_sha256",
    "raw_memory_tokens": "raw_memory_tokens",
    "packed_count": "packed_memory_count",
    "packed_memory_tokens": "packed_memory_tokens",
    "packed_pool_sha256": "packed_pool_sha256",
    "search_latency_s": "search_latency_s",
}

_NATIVE_SCORED_QUESTION_FIELDS = (
    "question_index",
    "question_id",
    "question",
    "dated_question",
    "gold_answer",
    "prediction",
    "category",
    "retrieval_row_sha256",
    "query_sha256",
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
    "responder_usage",
    "judge_usage",
)


def _validate_native_scored_question(
    value: Any,
    *,
    index: int,
    retrieval_row: Mapping[str, Any],
    expected_question: Mapping[str, Any],
    prompt_cap: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, int | float], dict[str, int | float]]:
    row = _mapping(value, label)
    _require_exact_fields(row, _NATIVE_SCORED_QUESTION_FIELDS, label)
    _must_equal(row.get("question_index"), index + 1, f"{label}.question_index")
    for field, wanted in {
        "question_id": expected_question["question_id"],
        "question": expected_question["question"],
        "dated_question": expected_question["dated_question"],
        "gold_answer": expected_question["gold_answer"],
        "category": expected_question["category"],
        "retrieval_row_sha256": retrieval_row["retrieval_row_sha256"],
        "query_sha256": text_sha256(str(retrieval_row["query"])),
        "attribution_kind": MEM0_ATTRIBUTION_KIND,
        "supports_exact_source_provenance": False,
    }.items():
        _must_equal(row.get(field), wanted, f"{label}.{field}")
    for report_field, retrieval_field in _NATIVE_RETRIEVAL_FIELD_MAP.items():
        _must_equal(
            row.get(report_field),
            retrieval_row.get(retrieval_field),
            f"{label}.{report_field}",
        )

    prediction = _string(row.get("prediction"), f"{label}.prediction")
    _must_equal(prediction, prediction.strip(), f"{label}.prediction normalization")
    gold = str(expected_question["gold_answer"])
    recomputed_f1 = f1_score(prediction, gold)
    reported_f1 = _number(
        row.get("f1"), f"{label}.f1", minimum=0.0, maximum=1.0
    )
    if not math.isclose(reported_f1, recomputed_f1, rel_tol=0.0, abs_tol=1e-15):
        raise Mem0ReportError(f"{label}.f1 disagrees with the answer strings")
    _must_equal(
        row.get("exact_match"),
        exact_match(prediction, gold),
        f"{label}.exact_match",
    )
    verdict = _judge_verdict(row.get("judge_reasoning"), f"{label}.judge_reasoning")
    _must_equal(row.get("judge_correct"), verdict, f"{label}.judge_correct")

    responder = _validate_usage(row.get("responder_usage"), f"{label}.responder_usage")
    judge = _validate_usage(row.get("judge_usage"), f"{label}.judge_usage")
    if responder["calls"] != 1 or judge["calls"] != 1:
        raise Mem0ReportError(
            f"{label} must bind one responder and one judge call"
        )
    if int(responder["input_tokens"]) <= 0:
        raise Mem0ReportError(f"{label} has no exact responder input-token usage")
    if int(responder["input_tokens"]) > prompt_cap:
        raise Mem0ReportError(f"{label} responder input usage exceeds the prompt cap")
    if int(judge["input_tokens"]) <= 0:
        raise Mem0ReportError(f"{label} has no exact judge input-token usage")
    projected = {field: row[field] for field in _NATIVE_SCORED_QUESTION_FIELDS}
    projected["f1"] = recomputed_f1
    projected["exact_match"] = exact_match(prediction, gold)
    projected["judge_correct"] = verdict
    projected["responder_usage"] = responder
    projected["judge_usage"] = judge
    return projected, responder, judge


def _validate_usage_total(
    value: Any,
    rows: Sequence[Mapping[str, int | float]],
    label: str,
) -> dict[str, int | float]:
    total = _validate_usage(value, label)
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "calls",
    ):
        _must_equal(
            total[field],
            sum(int(row[field]) for row in rows),
            f"{label}.{field}",
        )
    expected_elapsed = sum(float(row["elapsed_s"]) for row in rows)
    if not math.isclose(
        float(total["elapsed_s"]),
        expected_elapsed,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise Mem0ReportError(f"{label}.elapsed_s disagrees with question rows")
    return total


def _validate_native_scoring_receipt(
    value: Any,
    *,
    report_path: Path,
    expected: ExpectedMem0Shard,
    population: FrozenMem0Population,
    artifact_sha256: str,
    artifact_bytes: int,
    question_rows: Sequence[Mapping[str, Any]],
    responder_rows: Sequence[Mapping[str, int | float]],
    judge_rows: Sequence[Mapping[str, int | float]],
    report_execution_binding: Mapping[str, Any],
    report_environment_lock: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], str]:
    receipt = _mapping(value, label)
    question_count = len(expected.questions)
    for field, wanted in {
        "retrieval_artifact_sha256": artifact_sha256,
        "source_environment_lock_sha256": population.plan.environment_lock_sha256,
        "answer_judge_logical_wrapper_calls": 2 * question_count,
        "authorized_local_wrapper_retries": 0,
        "external_http_attempts_certified": False,
        "external_retry_attempts_certified": False,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }.items():
        _must_equal(receipt.get(field), wanted, f"{label}.{field}")
    responder_budget = _validate_native_call_budget(
        receipt.get("responder_logical_wrapper_calls"),
        expected_calls=question_count,
        label=f"{label}.responder_logical_wrapper_calls",
    )
    judge_budget = _validate_native_call_budget(
        receipt.get("judge_logical_wrapper_calls"),
        expected_calls=question_count,
        label=f"{label}.judge_logical_wrapper_calls",
    )
    stateless_contracts = _validate_stateless_provider_contracts(
        receipt.get("stateless_provider_contracts"),
        f"{label}.stateless_provider_contracts",
    )
    _validate_usage_total(
        receipt.get("responder_usage"), responder_rows, f"{label}.responder_usage"
    )
    _validate_usage_total(
        receipt.get("judge_usage"), judge_rows, f"{label}.judge_usage"
    )

    _descriptor, trace, trace_digest = _validate_json_trace_descriptor(
        receipt.get("scoring_trace"),
        owner_path=report_path,
        expected_format=SCORING_TRACE_FORMAT,
        label=f"{label}.scoring_trace",
    )
    _reject_secret_material(trace, "Mem0 scoring trace")
    for field, wanted in {
        "certification_status": "injected_nonproduction",
        "comparison_certified": False,
        "stage": "scoring",
        "sample_offset": expected.sample_offset,
        "sample_id": expected.sample_id,
        "sample_sha256": expected.sample_sha256,
        "retrieval_artifact_sha256": artifact_sha256,
        "mem0_state_touched": False,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
        "external_http_attempts_certified": False,
        "external_retry_attempts_certified": False,
    }.items():
        _must_equal(trace.get(field), wanted, f"scoring trace.{field}")
    _must_equal(
        trace.get("execution_binding"),
        dict(report_execution_binding),
        "scoring trace.execution_binding",
    )
    _must_equal(
        trace.get("environment_lock"),
        dict(report_environment_lock),
        "scoring trace.environment_lock",
    )
    _number(trace.get("elapsed_s"), "scoring trace.elapsed_s", minimum=0.0)
    _must_equal(
        _validate_native_call_budget(
            trace.get("responder_logical_wrapper_calls"),
            expected_calls=question_count,
            label="scoring trace.responder_logical_wrapper_calls",
        ),
        responder_budget,
        "scoring trace responder budget binding",
    )
    _must_equal(
        _validate_native_call_budget(
            trace.get("judge_logical_wrapper_calls"),
            expected_calls=question_count,
            label="scoring trace.judge_logical_wrapper_calls",
        ),
        judge_budget,
        "scoring trace judge budget binding",
    )
    _must_equal(
        _validate_stateless_provider_contracts(
            trace.get("stateless_provider_contracts"),
            "scoring trace.stateless_provider_contracts",
        ),
        stateless_contracts,
        "scoring trace stateless provider contract binding",
    )
    events = _validate_native_events(trace.get("events"), "scoring trace.events")
    _must_equal(
        [str(row["event"]) for row in events],
        [
            "authorization_verified",
            "retrieval_artifact_verified",
            *(["question_scored"] * question_count),
            "call_budgets_closed",
        ],
        "scoring trace event protocol",
    )
    _must_equal(
        events[1].get("sha256"), artifact_sha256, "scoring trace artifact sha256"
    )
    _must_equal(events[1].get("bytes"), artifact_bytes, "scoring trace artifact bytes")
    for index, row in enumerate(question_rows):
        event = events[2 + index]
        for field, wanted in {
            "question_id": row["question_id"],
            "retrieval_row_sha256": row["retrieval_row_sha256"],
            "prediction_sha256": text_sha256(str(row["prediction"])),
            "responder_logical_wrapper_calls_completed": index + 1,
            "judge_logical_wrapper_calls_completed": index + 1,
        }.items():
            _must_equal(event.get(field), wanted, f"scoring trace question event {index}.{field}")
    _must_equal(
        events[-1].get("responder_logical_wrapper_calls"),
        responder_budget,
        "scoring trace closed responder budget",
    )
    _must_equal(
        events[-1].get("judge_logical_wrapper_calls"),
        judge_budget,
        "scoring trace closed judge budget",
    )
    return receipt, trace_digest


def validate_mem0_shard_report(
    report: Mapping[str, Any],
    *,
    report_path: str | Path,
    expected: ExpectedMem0Shard,
    population: FrozenMem0Population,
    report_sha256: str | None = None,
) -> ValidatedMem0Shard:
    """Validate one native Stage-B report against reconstructed primitives."""

    population.mem0_policy.recheck()
    path = Path(report_path).resolve()
    label = f"report[{path.as_posix()}]"
    document = dict(report)
    _reject_secret_material(document, label)
    on_disk, report_payload = _load_json(path, "Mem0 Stage-B report")
    _must_equal(document, on_disk, f"{label} caller/file binding")
    observed_report_sha = hashlib.sha256(report_payload).hexdigest()
    if report_sha256 is not None:
        _must_equal(report_sha256, observed_report_sha, f"{label} file SHA-256")
    for field, wanted in {
        "schema_version": SHARD_SCHEMA_VERSION,
        "report_type": SHARD_REPORT_TYPE,
        "arm_id": ARM_ID,
        "run_status": RUN_STATUS,
        "certification_status": "injected_nonproduction",
        "comparison_certified": False,
        "sample_offset": expected.sample_offset,
    }.items():
        _must_equal(document.get(field), wanted, f"{label}.{field}")
    report_execution_binding = _validate_execution_binding(
        document.get("execution_binding"), f"{label}.execution_binding"
    )
    report_environment_lock = _validate_environment_lock_receipt(
        document.get("environment_lock"),
        expected_sha256=population.plan.environment_lock_sha256,
        expected_filename="pixi.lock",
        label=f"{label}.environment_lock",
    )

    identity, native_model_identity, config = _validate_native_model_config(
        document, population=population, label=label
    )
    _validate_native_protocol(
        document.get("evaluation_protocol"),
        population=population,
        label=f"{label}.evaluation_protocol",
    )
    samples = _list(document.get("samples"), f"{label}.samples")
    if len(samples) != 1:
        raise Mem0ReportError(f"{label}.samples must contain exactly one shard")
    _validate_native_sample(samples[0], expected, f"{label}.samples[0]")
    raw_receipt = _validate_native_raw_receipt(
        document.get("raw_input_receipt"), expected, f"{label}.raw_input_receipt"
    )
    ingestion, _extraction = _validate_native_ingestion(
        document.get("ingestion_receipt"), expected, f"{label}.ingestion_receipt"
    )
    mem0_usage = _validate_native_usage(
        document.get("mem0_usage"), expected, f"{label}.mem0_usage"
    )
    provenance = _validate_native_provenance(
        document.get("provenance"), f"{label}.provenance"
    )
    (
        _artifact,
        retrieval_rows,
        runtime,
        artifact_digest,
        retrieval_trace_digest,
    ) = _validate_native_retrieval_artifact(
        document.get("retrieval_artifact"),
        report_path=path,
        report_identity=identity,
        expected=expected,
        population=population,
        report_raw_receipt=raw_receipt,
        report_ingestion=ingestion,
        report_usage=mem0_usage,
        report_provenance=provenance,
        label=f"{label}.retrieval_artifact",
    )

    raw_questions = _list(
        document.get("question_results"), f"{label}.question_results"
    )
    if len(raw_questions) != len(expected.questions):
        raise Mem0ReportError(f"{label} has the wrong scored-question count")
    questions: list[dict[str, Any]] = []
    responder_rows: list[dict[str, int | float]] = []
    judge_rows: list[dict[str, int | float]] = []
    prompt_cap = int(population.source_evaluation_identity["max_prompt_tokens"])
    for index, expected_question in enumerate(expected.questions):
        question, responder, judge = _validate_native_scored_question(
            raw_questions[index],
            index=index,
            retrieval_row=retrieval_rows[index],
            expected_question=expected_question,
            prompt_cap=prompt_cap,
            label=f"{label}.question_results[{index}]",
        )
        questions.append(question)
        responder_rows.append(responder)
        judge_rows.append(judge)
    _must_equal(
        [row["question_id"] for row in questions],
        list(expected.question_ids),
        f"{label} scored question order",
    )
    retrieval_descriptor = _mapping(
        document.get("retrieval_artifact"), f"{label}.retrieval_artifact"
    )
    _scoring, scoring_trace_digest = _validate_native_scoring_receipt(
        document.get("scoring_receipt"),
        report_path=path,
        expected=expected,
        population=population,
        artifact_sha256=artifact_digest,
        artifact_bytes=_integer(
            retrieval_descriptor.get("bytes"),
            f"{label}.retrieval_artifact.bytes",
        ),
        question_rows=questions,
        responder_rows=responder_rows,
        judge_rows=judge_rows,
        report_execution_binding=report_execution_binding,
        report_environment_lock=report_environment_lock,
        label=f"{label}.scoring_receipt",
    )

    population.mem0_policy.recheck()
    model_identity = dict(native_model_identity)
    return ValidatedMem0Shard(
        report=document,
        report_name=path.name,
        report_sha256=observed_report_sha,
        sample_offset=expected.sample_offset,
        identity=dict(identity),
        model_identity=model_identity,
        config=config,
        runtime_identity=runtime,
        questions=tuple(questions),
        retrieval_artifact_sha256=artifact_digest,
        retrieval_trace_sha256=retrieval_trace_digest,
        scoring_trace_sha256=scoring_trace_digest,
    )


def _distribution(values: Iterable[int]) -> dict[str, int | float | list[int]]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": 0, "mean": 0.0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "values": []}

    def nearest(q: float) -> int:
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": math.fsum(float(value) for value in ordered) / len(ordered),
        "p50": nearest(0.50),
        "p90": nearest(0.90),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
        "max": ordered[-1],
        "values": ordered,
    }


def _sum_usage(rows: Iterable[Mapping[str, Any]]) -> dict[str, int | float]:
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


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return math.fsum(rows) / len(rows) if rows else 0.0


def _load_report(path: Path) -> tuple[dict[str, Any], str]:
    report, payload = _load_json(path, "Mem0 shard report")
    return report, hashlib.sha256(payload).hexdigest()


def _assert_population_unchanged(population: FrozenMem0Population) -> None:
    population.mem0_policy.recheck()
    plan = population.plan
    checks = (
        (plan.dataset_path, plan.dataset_sha256, "dataset"),
        (plan.split_manifest_path, plan.split_manifest_sha256, "split manifest"),
        (plan.policy_manifest_path, plan.policy_manifest_sha256, "source v3 policy"),
    )
    for path, wanted, label in checks:
        try:
            actual = file_sha256(path)
        except OSError as exc:
            raise Mem0ReportError(f"cannot recheck {label}: {exc}") from exc
        if actual != wanted:
            raise Mem0ReportError(f"{label} changed during campaign merge")
    if implementation_sha256() != plan.implementation_sha256:
        raise Mem0ReportError("source v3 implementation changed during merge")
    if environment_lock_sha256() != plan.environment_lock_sha256:
        raise Mem0ReportError("source v3 environment lock changed during merge")
    if tool_implementation_sha256() != population.mem0_tool_implementation_sha256:
        raise Mem0ReportError("Mem0 comparison tooling changed during merge")


def merge_mem0_shard_reports(
    report_paths: Iterable[str | Path],
    *,
    benchmark_file: str | Path,
    split_manifest: str | Path,
    policy_manifest: str | Path,
    mem0_policy_manifest: str | Path,
    mem0_environment_lock: str | Path,
) -> dict[str, Any]:
    """Validate and merge exactly ten locked Mem0 comparison shards."""

    paths = [Path(value).resolve() for value in report_paths]
    if len(paths) != len(FROZEN_OFFSETS):
        raise Mem0ReportError("the Mem0 campaign requires exactly ten shard reports")
    if len(set(paths)) != len(paths):
        raise Mem0ReportError("the Mem0 campaign repeats a report path")
    population = reconstruct_frozen_mem0_population(
        benchmark_file=benchmark_file,
        split_manifest=split_manifest,
        policy_manifest=policy_manifest,
        mem0_policy_manifest=mem0_policy_manifest,
        mem0_environment_lock=mem0_environment_lock,
    )

    validated: list[ValidatedMem0Shard] = []
    observed_offsets: set[int] = set()
    for path in paths:
        report, digest = _load_report(path)
        offset = _integer(report.get("sample_offset"), f"report[{path}].sample_offset")
        if offset in observed_offsets:
            raise Mem0ReportError(f"duplicate Mem0 sample_offset {offset}")
        try:
            expected = population.shards[offset]
        except KeyError as exc:
            raise Mem0ReportError(f"unexpected Mem0 sample_offset {offset}") from exc
        observed_offsets.add(offset)
        validated.append(
            validate_mem0_shard_report(
                report,
                report_path=path,
                expected=expected,
                population=population,
                report_sha256=digest,
            )
        )
    if observed_offsets != set(FROZEN_OFFSETS):
        missing = sorted(set(FROZEN_OFFSETS) - observed_offsets)
        raise Mem0ReportError(
            "the Mem0 campaign is missing frozen offsets: "
            + ", ".join(str(value) for value in missing)
        )
    validated.sort(key=lambda shard: shard.sample_offset)

    first = validated[0]
    stable_runtime = dict(first.runtime_identity)
    stable_runtime.pop("effective_config_sha256", None)
    effective_runtime_hashes: set[str] = set()
    artifact_hashes: set[str] = set()
    question_by_id: dict[str, dict[str, Any]] = {}
    question_sources: dict[str, dict[str, Any]] = {}
    for shard in validated:
        _must_equal(shard.identity, first.identity, f"shard {shard.sample_offset} identity")
        _must_equal(shard.model_identity, first.model_identity, f"shard {shard.sample_offset} model identity")
        _must_equal(shard.config, first.config, f"shard {shard.sample_offset} config")
        runtime = dict(shard.runtime_identity)
        effective = _sha256(
            runtime.pop("effective_config_sha256", None),
            f"shard {shard.sample_offset} runtime effective_config_sha256",
        )
        _must_equal(runtime, stable_runtime, f"shard {shard.sample_offset} stable runtime identity")
        if effective in effective_runtime_hashes:
            raise Mem0ReportError("Mem0 shards reused an effective owned-state config")
        effective_runtime_hashes.add(effective)
        if shard.retrieval_artifact_sha256 in artifact_hashes:
            raise Mem0ReportError("Mem0 shards reused a retrieval artifact")
        artifact_hashes.add(shard.retrieval_artifact_sha256)
        for row in shard.questions:
            question_id = str(row["question_id"])
            if question_id in question_by_id:
                raise Mem0ReportError(f"duplicate Mem0 question_id {question_id!r}")
            question_by_id[question_id] = row
            question_sources[question_id] = {
                "sample_offset": shard.sample_offset,
                "report_name": shard.report_name,
                "report_sha256": shard.report_sha256,
                "retrieval_artifact_sha256": shard.retrieval_artifact_sha256,
            }

    if set(question_by_id) != set(population.plan.question_ids):
        raise Mem0ReportError("Mem0 question IDs do not equal the frozen population")
    if len(question_by_id) != FROZEN_QUESTION_COUNT:
        raise Mem0ReportError("Mem0 campaign does not contain exactly 100 questions")
    questions = [question_by_id[key] for key in sorted(question_by_id)]

    raw_pairs = sum(
        int(shard.report["raw_input_receipt"]["raw_pairs"])
        for shard in validated
    )
    skipped = sum(
        int(shard.report["raw_input_receipt"]["skipped_empty_pairs"])
        for shard in validated
    )
    adds = sum(
        int(shard.report["ingestion_receipt"]["completed_add_operations"])
        for shard in validated
    )
    logical_extraction_calls = sum(
        int(
            shard.report["ingestion_receipt"]["extraction_model_calls"][
                "completed"
            ]
        )
        for shard in validated
    )
    searches = len(questions)
    responder_usage_rows = [
        _validate_usage(row["responder_usage"], "merged responder usage")
        for row in questions
    ]
    judge_usage_rows = [
        _validate_usage(row["judge_usage"], "merged judge usage")
        for row in questions
    ]
    responder_calls = sum(int(row["calls"]) for row in responder_usage_rows)
    judge_calls = sum(int(row["calls"]) for row in judge_usage_rows)
    observed_totals = (raw_pairs, skipped, adds, searches, responder_calls, judge_calls)
    frozen_totals = (
        sum(shard.raw_pairs for shard in population.shards.values()),
        sum(shard.skipped_empty_pairs for shard in population.shards.values()),
        sum(shard.expected_adds for shard in population.shards.values()),
        sum(len(shard.questions) for shard in population.shards.values()),
        sum(len(shard.questions) for shard in population.shards.values()),
        sum(len(shard.questions) for shard in population.shards.values()),
    )
    if observed_totals != frozen_totals:
        raise Mem0ReportError(
            f"Mem0 operation totals disagree with the frozen campaign: "
            f"{observed_totals!r} != {frozen_totals!r}"
        )

    prompt_cap = int(population.source_evaluation_identity["max_prompt_tokens"])
    output_reserve = int(
        population.source_evaluation_identity["responder_output_token_reserve"]
    )
    context_distribution = _distribution(
        int(row["context_tokens"]) for row in questions
    )
    prompt_distribution = _distribution(
        int(row["prompt_token_proxy"]) for row in questions
    )
    request_distribution = _distribution(
        int(row["prompt_token_proxy"]) + output_reserve for row in questions
    )
    prompt_compliant = all(
        int(row["prompt_token_proxy"]) <= prompt_cap for row in questions
    )
    if not prompt_compliant:
        raise Mem0ReportError("merged Mem0 prompts exceed the frozen token cap")
    provider_rows = [
        int(row["input_tokens"])
        for row in responder_usage_rows
        if int(row["input_tokens"]) > 0
    ]
    provider_compliant = (
        all(value <= prompt_cap for value in provider_rows)
        if provider_rows
        else None
    )
    provider_status = (
        "unavailable"
        if not provider_rows
        else "complete"
        if len(provider_rows) == len(questions)
        else "partial"
    )

    mean_f1 = _mean(float(row["f1"]) for row in questions)
    exact_match_rate = _mean(1.0 if row["exact_match"] else 0.0 for row in questions)
    judge_accuracy = _mean(1.0 if row["judge_correct"] else 0.0 for row in questions)
    accuracy_target = float(population.source_evaluation_identity["accuracy_target"])
    metric_target_met = judge_accuracy >= accuracy_target
    ordered_inputs = [
        {
            "sample_offset": shard.sample_offset,
            "sample_sha256": population.shards[
                shard.sample_offset
            ].sample_sha256,
            "name": shard.report_name,
            "sha256": shard.report_sha256,
            "retrieval_artifact_sha256": shard.retrieval_artifact_sha256,
            "retrieval_trace_sha256": shard.retrieval_trace_sha256,
            "scoring_trace_sha256": shard.scoring_trace_sha256,
        }
        for shard in validated
    ]
    input_set_sha256 = canonical_sha256(
        [row["sha256"] for row in ordered_inputs]
    )
    common_question_results = [
        {
            "question_id": row["question_id"],
            "predicted_answer": row["prediction"],
            "judge_correct": row["judge_correct"],
            "f1": row["f1"],
            "exact_match": row["exact_match"],
            "context_tokens": row["context_tokens"],
            "prompt_token_proxy": row["prompt_token_proxy"],
            "responder_usage": row["responder_usage"],
            "judge_usage": row["judge_usage"],
        }
        for row in questions
    ]

    _assert_population_unchanged(population)
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "report_type": CAMPAIGN_REPORT_TYPE,
        "arm_id": ARM_ID,
        "run_status": RUN_STATUS,
        "inputs": ordered_inputs,
        "input_count": len(ordered_inputs),
        "input_set_sha256": input_set_sha256,
        "identity": first.identity,
        "model_identity": first.model_identity,
        "runtime_model_identity_probe": {
            "kind": "unavailable_injected_nonproduction",
            "extraction_model_identity_sha256": first.identity[
                "extraction_model_identity_sha256"
            ],
            "embedder_model_identity_sha256": first.identity[
                "embedder_model_identity_sha256"
            ],
            "before_match": False,
            "after_match": False,
            "comparison_certified": False,
        },
        "config": first.config,
        "benchmark": "longmemeval",
        "dataset_sha256": population.plan.dataset_sha256,
        "split_manifest_sha256": population.plan.split_manifest_sha256,
        "benchmark_split": "validation",
        "implementation_sha256": population.plan.implementation_sha256,
        "environment_lock_sha256": population.plan.environment_lock_sha256,
        "policy_manifest_sha256": population.plan.policy_manifest_sha256,
        "responder_model": population.source_evaluation_identity[
            "responder_model"
        ],
        "judge_model": population.source_evaluation_identity["judge_model"],
        "recent_window": population.source_evaluation_identity["recent_window"],
        "max_prompt_tokens": prompt_cap,
        "prompt_token_proxy_identity": population.source_evaluation_identity[
            "prompt_token_proxy_identity"
        ],
        "responder_output_token_reserve": output_reserve,
        "evaluation_protocol": dict(population.source_evaluation_identity),
        "population_identity": {
            "question_ids_sha256": canonical_sha256(
                sorted(population.plan.question_ids)
            ),
            "sample_offsets": list(FROZEN_OFFSETS),
            "sample_sha256_by_offset": {
                str(offset): population.shards[offset].sample_sha256
                for offset in FROZEN_OFFSETS
            },
        },
        "prompt_identity": {
            "max_prompt_tokens": prompt_cap,
            "prompt_cap_semantics": population.source_evaluation_identity[
                "prompt_cap_semantics"
            ],
            "prompt_token_proxy_identity": population.source_evaluation_identity[
                "prompt_token_proxy_identity"
            ],
            "responder_output_token_reserve": output_reserve,
        },
        "sample_offsets": list(FROZEN_OFFSETS),
        "num_samples": len(validated),
        "num_questions": len(questions),
        "question_results": questions,
        "common_question_result_schema": (
            "memory-condense-common-qa-result-v1"
        ),
        "common_question_results": common_question_results,
        "question_sources": {
            key: question_sources[key] for key in sorted(question_sources)
        },
        "raw_input_totals": {
            "raw_pairs": raw_pairs,
            "skipped_empty_pairs": skipped,
        },
        "operation_totals": {
            "mem0_adds": adds,
            "mem0_searches": searches,
            "responder_logical_wrapper_calls": responder_calls,
            "judge_logical_wrapper_calls": judge_calls,
            "answer_judge_logical_wrapper_calls": (
                responder_calls + judge_calls
            ),
            "mem0_local_logical_wrapper_calls": logical_extraction_calls,
            "mem0_logical_extraction_call_boundary": (
                "Memory.llm.generate_response"
            ),
            "external_http_attempts_certified": False,
            "underlying_mem0_provider_calls": None,
            "underlying_mem0_provider_usage_status": MEM0_PROVIDER_USAGE_STATUS,
        },
        "mean_f1": mean_f1,
        "exact_match_rate": exact_match_rate,
        "judge_accuracy": judge_accuracy,
        "mean_context_tokens": context_distribution["mean"],
        "mean_prompt_token_proxy": prompt_distribution["mean"],
        "p95_prompt_token_proxy": prompt_distribution["p95"],
        "context_token_distribution": context_distribution,
        "prompt_token_proxy_distribution": prompt_distribution,
        "request_token_proxy_distribution": request_distribution,
        "max_prompt_token_proxy_observed": prompt_distribution["max"],
        "prompt_token_proxy_budget_compliance": prompt_compliant,
        "provider_prompt_budget_compliance": provider_compliant,
        "provider_input_usage_status": (
            "local_injected_receipts_" + provider_status
        ),
        "external_provider_usage_certified": False,
        "responder_usage": _sum_usage(responder_usage_rows),
        "judge_usage": _sum_usage(judge_usage_rows),
        "provenance": {
            "attribution_kind": MEM0_ATTRIBUTION_KIND,
            "supports_exact_source_provenance": False,
            "source_session_date_exposure": (
                "diagnostics_only_not_model_input"
            ),
            "retrieved_created_at_exposure": "answer_prompt_date_headings",
            "source_coverage_status": SOURCE_COVERAGE_STATUS,
            "source_coverage": None,
            "request_window_diagnostic_only": True,
            "source_coverage_reason": SOURCE_COVERAGE_REASON,
        },
        "source_coverage_status": SOURCE_COVERAGE_STATUS,
        "source_coverage": None,
        "exact_provenance_requirement_met": False,
        "local_request_token_state_contract_satisfied": True,
        "zero_persisted_transformer_token_state_verified": False,
        "external_provider_persistence_certified": False,
        "production_binding_certified": False,
        "certification_status": "injected_core_nonproduction",
        "locked_population_verified": True,
        "local_comparison_protocol_verified": True,
        "accuracy_target": accuracy_target,
        "min_target_questions": population.source_evaluation_identity[
            "min_target_questions"
        ],
        "metric_accuracy_target_met": metric_target_met,
        "accuracy_target_met": False,
        "target_status": (
            "metric_passed_noncertified"
            if metric_target_met
            else "metric_failed_noncertified"
        ),
    }


def _is_equal_to_or_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    """Atomically create a flushed report without clobbering a racing writer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing campaign report {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_mem0_campaign_report(
    report: Mapping[str, Any],
    output: str | Path,
    *,
    protected_inputs: Iterable[str | Path] = (),
) -> Path:
    """Create a deterministic campaign JSON document exactly once.

    ``protected_inputs`` lets orchestrators bind the source shard reports and
    lock/policy paths they consumed.  Existing inputs are already protected by
    no-clobber publication; directory inputs are additionally protected from
    descendant output paths.
    """

    path = Path(output).resolve(strict=False)
    repository = Path(__file__).resolve().parents[2]
    protected = (
        *(
            (Path(value).resolve(strict=False), "caller-protected input")
            for value in protected_inputs
        ),
        (Path(__file__).resolve().parent, "Mem0 tool implementation root"),
        (
            (repository / "src" / "memory_condense").resolve(strict=False),
            "source implementation root",
        ),
    )
    for protected_path, label in protected:
        if _is_equal_to_or_within(path, protected_path):
            raise ValueError(
                f"campaign output {path} equals or descends from protected "
                f"{label} {protected_path}"
            )
    _reject_secret_material(report, "Mem0 campaign report")
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_create_bytes(path, payload)
    return path


__all__ = [
    "ARM_ID",
    "CAMPAIGN_REPORT_TYPE",
    "ExpectedMem0Shard",
    "FrozenMem0Population",
    "Mem0ReportError",
    "RETRIEVAL_ARTIFACT_FORMAT",
    "RETRIEVAL_ARTIFACT_TYPE",
    "SHARD_REPORT_TYPE",
    "SCORING_RECEIPT_FORMAT",
    "SOURCE_COVERAGE_STATUS",
    "ValidatedMem0Shard",
    "canonical_sha256",
    "merge_mem0_shard_reports",
    "reconstruct_frozen_mem0_population",
    "save_mem0_campaign_report",
    "text_sha256",
    "validate_mem0_shard_report",
]
