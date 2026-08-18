"""Merge locked benchmark shards into one auditable campaign report.

The validation campaign is deliberately sharded so provider failures do not
discard an entire run.  A shard's ``insufficient_questions`` status is
therefore expected: only the merged, question-weighted result is allowed to
make the target claim.  This module is intentionally independent of the main
evaluation CLI so completed shard artifacts can be merged without importing
model or retrieval code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from memory_condense._tokenizer import tokenizer_proxy_identity
from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
)
from memory_condense.eval.context_stress import (
    compose_context_stress_sample,
    transcript_tokens,
)
from memory_condense.eval.cache_receipts import (
    cache_receipts_sha256,
    validated_cache_receipts,
)
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    file_sha256,
    implementation_sha256,
    project_root,
)
from memory_condense.eval.sample_identity import sample_sha256
from memory_condense.eval.validation_profile import (
    ValidationClaimProfileError,
    claimed_validation_profile,
    validate_longmemeval_claim_profile,
)
from memory_condense.loader import load_benchmark


class CampaignMergeError(ValueError):
    """A shard cannot participate in a locked validation campaign."""


@dataclass(frozen=True, slots=True)
class ExpectedStressShard:
    sample_offset: int
    sample_id: str
    sample_sha256: str
    num_turns: int
    transcript_tokens: int
    questions: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class LockedValidationPlan:
    dataset_path: Path
    split_manifest_path: Path
    policy_manifest_path: Path
    selection_artifact_path: Path
    dataset_sha256: str
    split_manifest_sha256: str
    policy_manifest_sha256: str
    implementation_sha256: str
    environment_lock_sha256: str
    selection_artifact_sha256: str
    retrieval: dict[str, Any]
    evaluation: dict[str, Any]
    sample_offsets: tuple[int, ...]
    shards: dict[int, ExpectedStressShard]
    question_ids: frozenset[str]
    claim_profile: str
    claim_profile_verified: bool


def _safe_repository_file(
    value: object,
    *,
    label: str,
    repository_root: str | Path | None,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CampaignMergeError(f"{label} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignMergeError(f"{label} must be a safe repository-relative path")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else project_root().resolve()
    )
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CampaignMergeError(f"{label} must stay within the repository") from exc
    if not candidate.is_file():
        raise CampaignMergeError(f"{label} does not name an existing file")
    return candidate


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload, parse_constant=_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignMergeError(f"cannot read {label} {path}: {exc}") from exc
    return _require_mapping(value, label), payload


def build_locked_validation_plan(
    *,
    benchmark_file: str | Path,
    benchmark_format: str,
    split_manifest: str | Path,
    policy_manifest: str | Path,
    repository_root: str | Path | None = None,
) -> LockedValidationPlan:
    """Reconstruct every frozen stress shard from locked source artifacts."""

    dataset_path = Path(benchmark_file).resolve()
    split_path = Path(split_manifest).resolve()
    policy_path = Path(policy_manifest).resolve()
    policy, policy_bytes = _load_json_object(policy_path, "policy manifest")

    if policy.get("format") != "memory-condense-retrieval-policy-v1":
        raise CampaignMergeError("validation policy manifest format mismatch")
    if policy.get("status") != "validation_frozen":
        raise CampaignMergeError("validation policy is not frozen")
    if policy.get("split") != "validation":
        raise CampaignMergeError("validation policy must bind split='validation'")
    if policy.get("split_manifest") != split_path.name:
        raise CampaignMergeError("validation policy split-manifest identity mismatch")

    try:
        dataset_digest = file_sha256(dataset_path)
        split_digest = file_sha256(split_path)
    except OSError as exc:
        raise CampaignMergeError(
            f"cannot hash locked validation source artifact: {exc}"
        ) from exc
    policy_digest = hashlib.sha256(policy_bytes).hexdigest()
    code_digest = implementation_sha256()
    environment_digest = environment_lock_sha256()
    expected_hashes = {
        "dataset_sha256": dataset_digest,
        "split_manifest_sha256": split_digest,
        "implementation_sha256": code_digest,
        "environment_lock_sha256": environment_digest,
    }
    for field, actual in expected_hashes.items():
        if policy.get(field) != actual:
            raise CampaignMergeError(f"validation policy {field} mismatch")

    if policy.get("selection_artifact_required") is not True:
        raise CampaignMergeError(
            "validation policy must require its development selection artifact"
        )
    selection_path = _safe_repository_file(
        policy.get("selection_artifact"),
        label="policy selection_artifact",
        repository_root=repository_root,
    )
    selection_digest = _require_sha256(
        policy.get("selection_artifact_sha256"),
        "policy.selection_artifact_sha256",
    )
    if file_sha256(selection_path) != selection_digest:
        raise CampaignMergeError("validation policy selection artifact hash mismatch")

    frozen_retrieval = dict(
        _require_mapping(policy.get("retrieval"), "policy.retrieval")
    )
    if not frozen_retrieval:
        raise CampaignMergeError("validation policy retrieval identity is empty")

    evaluation = _require_mapping(policy.get("evaluation"), "policy.evaluation")
    frozen_evaluation = dict(evaluation)
    try:
        claim_profile = claimed_validation_profile(policy)
        if claim_profile:
            validate_longmemeval_claim_profile(policy, frozen_evaluation)
    except ValidationClaimProfileError as exc:
        raise CampaignMergeError(str(exc)) from exc
    raw_offsets = frozen_evaluation.pop("sample_offsets", None)
    offsets = _require_list(raw_offsets, "policy.evaluation.sample_offsets")
    if (
        not offsets
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in offsets
        )
        or len(set(offsets)) != len(offsets)
    ):
        raise CampaignMergeError(
            "policy.evaluation.sample_offsets must be unique non-negative integers"
        )
    if frozen_evaluation.get("use_judge") is not True:
        raise CampaignMergeError("validation evaluation must enable the judge")
    if frozen_evaluation.get("provider_retries") != 0:
        raise CampaignMergeError("validation evaluation must freeze provider_retries=0")
    if frozen_evaluation.get("stress_question_offset") != 0:
        raise CampaignMergeError(
            "validation evaluation must freeze stress_question_offset=0"
        )
    if frozen_evaluation.get("max_samples") != 1:
        raise CampaignMergeError("validation evaluation must freeze max_samples=1")
    _require_nonempty_string(
        frozen_evaluation.get("responder_model"),
        "policy.evaluation.responder_model",
    )
    _require_nonempty_string(
        frozen_evaluation.get("judge_model"),
        "policy.evaluation.judge_model",
    )
    _require_nonempty_string(
        frozen_evaluation.get("embedding_device"),
        "policy.evaluation.embedding_device",
    )
    frozen_format = _require_nonempty_string(
        frozen_evaluation.get("benchmark_format"),
        "policy.evaluation.benchmark_format",
    )
    if frozen_format != "longmemeval":
        raise CampaignMergeError(
            "LongMemEval validation must freeze benchmark_format='longmemeval'"
        )
    if benchmark_format != frozen_format:
        raise CampaignMergeError(
            "campaign benchmark format does not match the frozen validation policy"
        )
    _require_int(
        frozen_evaluation.get("max_prompt_tokens"),
        "policy.evaluation.max_prompt_tokens",
        minimum=1,
    )
    if frozen_evaluation.get("prompt_cap_semantics") != (
        "local_prompt_token_proxy_with_provider_usage_postcheck_v1"
    ):
        raise CampaignMergeError(
            "validation evaluation must freeze the prompt-token-proxy cap "
            "and provider-usage postcheck semantics"
        )
    frozen_proxy_identity = _require_mapping(
        frozen_evaluation.get("prompt_token_proxy_identity"),
        "policy.evaluation.prompt_token_proxy_identity",
    )
    if _canonical_json(frozen_proxy_identity) != _canonical_json(
        tokenizer_proxy_identity()
    ):
        raise CampaignMergeError(
            "validation prompt-token-proxy tokenizer identity mismatch"
        )
    output_reserve = _require_int(
        frozen_evaluation.get("responder_output_token_reserve"),
        "policy.evaluation.responder_output_token_reserve",
        minimum=1,
    )
    if output_reserve != BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE:
        raise CampaignMergeError(
            "validation responder output reserve does not match the frozen "
            "remote benchmark protocol"
        )
    _require_float(
        frozen_evaluation.get("accuracy_target"),
        "policy.evaluation.accuracy_target",
        minimum=0.0,
        maximum=1.0,
    )
    target_tokens = _require_int(
        frozen_evaluation.get("stress_context_tokens"),
        "policy.evaluation.stress_context_tokens",
        minimum=1,
    )
    questions_per_shard = _require_int(
        frozen_evaluation.get("stress_questions"),
        "policy.evaluation.stress_questions",
        minimum=1,
    )
    min_questions = _require_int(
        frozen_evaluation.get("min_target_questions"),
        "policy.evaluation.min_target_questions",
        minimum=1,
    )
    _require_int(
        frozen_evaluation.get("recent_window"),
        "policy.evaluation.recent_window",
        minimum=0,
    )
    provider_authorization = _require_int(
        frozen_evaluation.get("max_provider_calls"),
        "policy.evaluation.max_provider_calls",
        minimum=0,
    )
    expected_provider_calls = 2 * questions_per_shard
    if provider_authorization != expected_provider_calls:
        raise CampaignMergeError(
            "validation evaluation must authorize exactly one responder and "
            "one judge call per stress question"
        )

    try:
        samples = load_benchmark(dataset_path, benchmark_format)
        manifest = load_split_manifest(split_path)
        validation = select_locked_split(
            samples,
            dataset_path=dataset_path,
            manifest=manifest,
            split="validation",
        )
    except (OSError, ValueError) as exc:
        raise CampaignMergeError(
            f"cannot reconstruct the locked validation population: {exc}"
        ) from exc
    shards: dict[int, ExpectedStressShard] = {}
    expected_question_ids: set[str] = set()
    for raw_offset in offsets:
        offset = int(raw_offset)
        if offset >= len(validation):
            raise CampaignMergeError(
                f"validation sample offset {offset} is outside the locked split"
            )
        try:
            stress_sample = compose_context_stress_sample(
                validation[offset:],
                target_tokens=target_tokens,
                max_questions=questions_per_shard,
                question_offset=0,
            )
        except ValueError as exc:
            raise CampaignMergeError(
                f"cannot reconstruct validation stress shard at offset {offset}: {exc}"
            ) from exc
        if len(stress_sample.questions) != questions_per_shard:
            raise CampaignMergeError(
                f"validation stress shard {offset} has "
                f"{len(stress_sample.questions)} questions; expected "
                f"{questions_per_shard}"
            )
        expected_questions = tuple(
            {
                "question_id": question.question_id,
                "question": question.question,
                "gold_answer": question.answer,
                "category": question.category,
            }
            for question in stress_sample.questions
        )
        shard_ids = {str(row["question_id"]) for row in expected_questions}
        overlap = expected_question_ids & shard_ids
        if overlap:
            raise CampaignMergeError(
                "validation stress plan repeats question IDs: "
                + ", ".join(sorted(overlap))
            )
        expected_question_ids.update(shard_ids)
        shards[offset] = ExpectedStressShard(
            sample_offset=offset,
            sample_id=stress_sample.sample_id,
            sample_sha256=sample_sha256(stress_sample),
            num_turns=len(stress_sample.turns),
            transcript_tokens=transcript_tokens(stress_sample),
            questions=expected_questions,
        )

    full_validation_ids = {
        question.question_id for sample in validation for question in sample.questions
    }
    if expected_question_ids != full_validation_ids:
        missing = full_validation_ids - expected_question_ids
        extra = expected_question_ids - full_validation_ids
        raise CampaignMergeError(
            "validation stress plan does not cover the exact locked population: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if len(expected_question_ids) != min_questions:
        raise CampaignMergeError(
            "validation policy min_target_questions does not equal the locked "
            f"population ({min_questions} != {len(expected_question_ids)})"
        )
    claim_profile_verified = False
    if claim_profile:
        try:
            validate_longmemeval_claim_profile(
                policy,
                evaluation,
                population_size=len(expected_question_ids),
            )
        except ValidationClaimProfileError as exc:
            raise CampaignMergeError(str(exc)) from exc
        claim_profile_verified = True

    return LockedValidationPlan(
        dataset_path=dataset_path,
        split_manifest_path=split_path,
        policy_manifest_path=policy_path,
        selection_artifact_path=selection_path,
        dataset_sha256=dataset_digest,
        split_manifest_sha256=split_digest,
        policy_manifest_sha256=policy_digest,
        implementation_sha256=code_digest,
        environment_lock_sha256=environment_digest,
        selection_artifact_sha256=selection_digest,
        retrieval=frozen_retrieval,
        evaluation=frozen_evaluation,
        sample_offsets=tuple(int(value) for value in offsets),
        shards=shards,
        question_ids=frozenset(expected_question_ids),
        claim_profile=claim_profile,
        claim_profile_verified=claim_profile_verified,
    )


def _assert_locked_plan_unchanged(plan: LockedValidationPlan) -> None:
    checks = (
        (plan.dataset_path, plan.dataset_sha256, "dataset"),
        (plan.split_manifest_path, plan.split_manifest_sha256, "split manifest"),
        (plan.policy_manifest_path, plan.policy_manifest_sha256, "policy manifest"),
        (
            plan.selection_artifact_path,
            plan.selection_artifact_sha256,
            "selection artifact",
        ),
    )
    for path, expected, label in checks:
        try:
            actual = file_sha256(path)
        except OSError as exc:
            raise CampaignMergeError(
                f"cannot recheck {label} during campaign verification: {exc}"
            ) from exc
        if actual != expected:
            raise CampaignMergeError(f"{label} changed during campaign verification")
    if implementation_sha256() != plan.implementation_sha256:
        raise CampaignMergeError("implementation changed during campaign verification")
    if environment_lock_sha256() != plan.environment_lock_sha256:
        raise CampaignMergeError("environment lock changed during campaign verification")


_HASH_FIELDS = (
    "dataset_sha256",
    "split_manifest_sha256",
    "implementation_sha256",
    "environment_lock_sha256",
    "policy_manifest_sha256",
)
_QUESTION_ERROR_FIELDS = {
    "error",
    "errors",
    "exception",
    "provider_error",
    "responder_error",
    "judge_error",
}
_BINARY_JUDGE_VERDICT = re.compile(
    r"^\s*(CORRECT|INCORRECT)\b(?P<remainder>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _json_constant(value: str) -> None:
    raise CampaignMergeError(f"non-finite JSON number {value!r} is not allowed")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignMergeError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CampaignMergeError(f"{label} must be a JSON array")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignMergeError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_nonempty_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CampaignMergeError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CampaignMergeError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignMergeError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignMergeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignMergeError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise CampaignMergeError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise CampaignMergeError(f"{label} must be <= {maximum}")
    return result


def _revalidate_locked_claim_profile(plan: LockedValidationPlan) -> bool:
    """Derive certification eligibility without trusting a cached boolean.

    ``LockedValidationPlan`` is a public dataclass, so callers can construct or
    replace one directly.  A ``claim_profile_verified=True`` flag must never be
    enough to turn a diagnostic population into a certified result.  Re-read
    the still-hash-locked policy and independently check its exact evaluation,
    offsets, population, and ten-question shard partition.
    """

    policy, _policy_bytes = _load_json_object(
        plan.policy_manifest_path,
        "policy manifest",
    )
    try:
        source_profile = claimed_validation_profile(policy)
    except ValidationClaimProfileError as exc:
        raise CampaignMergeError(str(exc)) from exc
    if source_profile != plan.claim_profile:
        raise CampaignMergeError(
            "locked plan claim profile disagrees with its policy manifest"
        )

    source_evaluation = dict(
        _require_mapping(policy.get("evaluation"), "policy.evaluation")
    )
    raw_offsets = source_evaluation.pop("sample_offsets", None)
    offsets = _require_list(raw_offsets, "policy.evaluation.sample_offsets")
    if source_evaluation != plan.evaluation or tuple(offsets) != plan.sample_offsets:
        raise CampaignMergeError(
            "locked plan evaluation disagrees with its policy manifest"
        )

    if not source_profile:
        if plan.claim_profile_verified:
            raise CampaignMergeError(
                "locked plan claims a verified profile absent from its policy"
            )
        return False

    full_evaluation = {
        **source_evaluation,
        "sample_offsets": list(offsets),
    }
    try:
        validate_longmemeval_claim_profile(
            policy,
            full_evaluation,
            population_size=len(plan.question_ids),
        )
    except ValidationClaimProfileError as exc:
        raise CampaignMergeError(str(exc)) from exc
    if not plan.claim_profile_verified:
        raise CampaignMergeError(
            "locked plan did not verify its declared claim profile"
        )

    if set(plan.shards) != set(plan.sample_offsets):
        raise CampaignMergeError(
            "locked claim profile shard map does not match its offset plan"
        )
    questions_per_shard = int(plan.evaluation["stress_questions"])
    reconstructed_ids: set[str] = set()
    for offset in plan.sample_offsets:
        shard = plan.shards[offset]
        if shard.sample_offset != offset:
            raise CampaignMergeError(
                "locked claim profile shard has the wrong sample offset"
            )
        if len(shard.questions) != questions_per_shard:
            raise CampaignMergeError(
                "locked claim profile does not contain exact ten-question shards"
            )
        shard_ids = {
            str(question.get("question_id", "")) for question in shard.questions
        }
        if "" in shard_ids or len(shard_ids) != len(shard.questions):
            raise CampaignMergeError(
                "locked claim profile shard has invalid or duplicate question IDs"
            )
        if reconstructed_ids & shard_ids:
            raise CampaignMergeError(
                "locked claim profile repeats questions across shards"
            )
        reconstructed_ids.update(shard_ids)
    if reconstructed_ids != set(plan.question_ids):
        raise CampaignMergeError(
            "locked claim profile shards do not equal its exact population"
        )
    return True


def _load_report(path: Path) -> tuple[dict[str, Any], str, str, str]:
    resolved = path.resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise CampaignMergeError(f"cannot read report {resolved}: {exc}") from exc
    digest = _file_sha256(payload)
    try:
        report = json.loads(payload, parse_constant=_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignMergeError(f"invalid JSON report {resolved}: {exc}") from exc
    return (
        _require_mapping(report, f"report {resolved}"),
        digest,
        resolved.as_posix(),
        resolved.name,
    )


def _identity(report: dict[str, Any], label: str) -> dict[str, Any]:
    config = _require_mapping(report.get("config"), f"{label}.config")
    identity: dict[str, Any] = {
        field: _require_sha256(report.get(field), f"{label}.{field}")
        for field in _HASH_FIELDS
    }
    split = _require_nonempty_string(
        report.get("benchmark_split"), f"{label}.benchmark_split"
    )
    if split != "validation":
        raise CampaignMergeError(
            f"{label}.benchmark_split must be 'validation', got {split!r}"
        )
    identity["benchmark_split"] = split
    identity["benchmark"] = _require_nonempty_string(
        report.get("benchmark"), f"{label}.benchmark"
    )
    identity["chunker_config"] = _require_mapping(
        config.get("chunker"), f"{label}.config.chunker"
    )
    identity["retrieval_config"] = _require_mapping(
        config.get("retrieval"), f"{label}.config.retrieval"
    )
    identity["responder_model"] = _require_nonempty_string(
        config.get("responder_model"), f"{label}.config.responder_model"
    )
    identity["judge_model"] = _require_nonempty_string(
        config.get("judge_model"), f"{label}.config.judge_model"
    )
    identity["max_prompt_tokens"] = _require_int(
        config.get("max_prompt_tokens"),
        f"{label}.config.max_prompt_tokens",
        minimum=1,
    )
    embedding_device = config.get("embedding_device")
    if embedding_device is not None and not isinstance(embedding_device, str):
        raise CampaignMergeError(
            f"{label}.config.embedding_device must be a string or null"
        )
    identity["embedding_device"] = embedding_device
    identity["recent_window"] = _require_int(
        config.get("recent_window"), f"{label}.config.recent_window"
    )
    raw_proxy_identity = report.get("prompt_token_proxy_identity")
    identity["prompt_token_proxy_identity"] = (
        {
            "schema": "legacy-cl100k-message-content-only-v0",
            "encoding": "cl100k_base",
        }
        if raw_proxy_identity is None
        else dict(
            _require_mapping(
                raw_proxy_identity,
                f"{label}.prompt_token_proxy_identity",
            )
        )
    )
    identity["responder_output_token_reserve"] = _require_int(
        report.get("responder_output_token_reserve", 0),
        f"{label}.responder_output_token_reserve",
    )
    raw_protocol = report.get("evaluation_protocol", {})
    protocol = _require_mapping(raw_protocol, f"{label}.evaluation_protocol")
    protocol_common = dict(protocol)
    protocol_common.pop("sample_offset", None)
    identity["evaluation_protocol"] = protocol_common
    return identity


def _ensure_same_identity(
    expected: dict[str, Any], actual: dict[str, Any], label: str
) -> None:
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if _canonical_json(actual_value) != _canonical_json(expected_value):
            raise CampaignMergeError(
                f"locked campaign identity drift in {label}.{field}: "
                f"expected {_canonical_json(expected_value)}, "
                f"got {_canonical_json(actual_value)}"
            )


def _assert_policy_retrieval_identity(
    identity: dict[str, Any],
    plan: LockedValidationPlan,
    label: str,
) -> None:
    """Match a report's nested config to the policy's flattened identity."""

    retrieval = identity["retrieval_config"]
    chunker = identity["chunker_config"]
    actual: dict[str, Any] = {}
    for field in plan.retrieval:
        if field == "chunker_min_tokens":
            value = chunker.get("min_tokens")
        elif field == "chunker_max_tokens":
            value = chunker.get("max_tokens")
        elif field == "max_prompt_tokens":
            value = identity["max_prompt_tokens"]
        else:
            value = retrieval.get(field)
        actual[field] = value
    if _canonical_json(actual) != _canonical_json(plan.retrieval):
        raise CampaignMergeError(
            f"{label}.config does not match the frozen retrieval policy"
        )


def _has_error(question: dict[str, Any]) -> str | None:
    for field, value in question.items():
        is_error_field = field in _QUESTION_ERROR_FIELDS or field.endswith("_error")
        if is_error_field and value not in (None, "", False, 0, [], {}):
            return field
    status = question.get("status")
    if isinstance(status, str) and status.casefold() in {
        "error",
        "failed",
        "provider_error",
    }:
        return "status"
    return None


def _locked_judge_verdict(value: Any, label: str) -> bool:
    reasoning = _require_nonempty_string(value, label)
    match = _BINARY_JUDGE_VERDICT.match(reasoning)
    if match is None:
        raise CampaignMergeError(f"{label} has no exact binary judge verdict")
    remainder = match.group("remainder").lstrip(" \t\r\n,.:;-—")
    if remainder.casefold().startswith("or ") or remainder.startswith("/"):
        raise CampaignMergeError(f"{label} contains an ambiguous judge verdict")
    return match.group(1).casefold() == "correct"


def _validate_usage(value: Any, label: str) -> dict[str, int | float]:
    usage = _require_mapping(value, label)
    return {
        "input_tokens": _require_int(
            usage.get("input_tokens"), f"{label}.input_tokens"
        ),
        "output_tokens": _require_int(
            usage.get("output_tokens"), f"{label}.output_tokens"
        ),
        "cache_read_input_tokens": _require_int(
            usage.get("cache_read_input_tokens"),
            f"{label}.cache_read_input_tokens",
        ),
        "elapsed_s": _require_float(
            usage.get("elapsed_s"), f"{label}.elapsed_s", minimum=0.0
        ),
        "calls": _require_int(usage.get("calls"), f"{label}.calls"),
    }


def _validate_question(
    value: Any,
    label: str,
    *,
    prompt_cap: int,
    output_token_reserve: int,
    require_proxy_fields: bool,
) -> tuple[dict[str, Any], dict[str, int | float], dict[str, int | float]]:
    question = dict(_require_mapping(value, label))
    question_id = _require_nonempty_string(
        question.get("question_id"), f"{label}.question_id"
    )
    error_field = _has_error(question)
    if error_field is not None:
        raise CampaignMergeError(
            f"{label} ({question_id}) contains a per-question error in {error_field}"
        )
    _require_float(question.get("f1"), f"{label}.f1", minimum=0.0, maximum=1.0)
    _require_bool(question.get("exact_match"), f"{label}.exact_match")
    _require_bool(question.get("judge_correct"), f"{label}.judge_correct")
    _require_int(question.get("context_tokens"), f"{label}.context_tokens")
    legacy_prompt_tokens = _require_int(
        question.get("prompt_tokens"), f"{label}.prompt_tokens"
    )
    raw_prompt_proxy = question.get("prompt_token_proxy")
    if raw_prompt_proxy is None:
        if require_proxy_fields:
            raise CampaignMergeError(
                f"{label}.prompt_token_proxy is required for locked validation"
            )
        prompt_proxy = legacy_prompt_tokens
    else:
        prompt_proxy = _require_int(
            raw_prompt_proxy,
            f"{label}.prompt_token_proxy",
        )
        if prompt_proxy != legacy_prompt_tokens:
            raise CampaignMergeError(
                f"{label}.prompt_tokens compatibility alias disagrees with "
                "prompt_token_proxy"
            )
    if prompt_proxy > prompt_cap:
        raise CampaignMergeError(
            f"{label} ({question_id}) exceeds the locked prompt cap: "
            f"{prompt_proxy} > {prompt_cap}"
        )
    responder_usage = _validate_usage(
        question.get("responder_usage"), f"{label}.responder_usage"
    )
    judge_usage = _validate_usage(
        question.get("judge_usage"), f"{label}.judge_usage"
    )
    expected_provider_compliance = (
        None
        if int(responder_usage["input_tokens"]) <= 0
        else int(responder_usage["input_tokens"]) <= prompt_cap
    )
    reported_provider_compliance = question.get(
        "provider_prompt_budget_compliant"
    )
    if require_proxy_fields or reported_provider_compliance is not None:
        if reported_provider_compliance is not None:
            _require_bool(
                reported_provider_compliance,
                f"{label}.provider_prompt_budget_compliant",
            )
        if reported_provider_compliance != expected_provider_compliance:
            raise CampaignMergeError(
                f"{label}.provider_prompt_budget_compliant disagrees with "
                "provider-reported input usage"
            )
    if expected_provider_compliance is False:
        raise CampaignMergeError(
            f"{label} ({question_id}) provider input usage exceeds the locked "
            f"prompt cap: {responder_usage['input_tokens']} > {prompt_cap}"
        )
    raw_reserve = question.get("responder_output_token_reserve")
    raw_request_proxy = question.get("request_token_proxy")
    if require_proxy_fields and (raw_reserve is None or raw_request_proxy is None):
        raise CampaignMergeError(
            f"{label} must report responder output reserve and request-token proxy"
        )
    if raw_reserve is not None:
        reported_reserve = _require_int(
            raw_reserve,
            f"{label}.responder_output_token_reserve",
        )
        if reported_reserve != output_token_reserve:
            raise CampaignMergeError(
                f"{label}.responder_output_token_reserve disagrees with the "
                "locked protocol"
            )
    if raw_request_proxy is not None:
        request_proxy = _require_int(
            raw_request_proxy,
            f"{label}.request_token_proxy",
        )
        if request_proxy != prompt_proxy + output_token_reserve:
            raise CampaignMergeError(
                f"{label}.request_token_proxy does not include the exact "
                "prompt proxy plus output reserve"
            )
    question["prompt_token_proxy"] = prompt_proxy
    question["prompt_tokens"] = prompt_proxy
    question["provider_prompt_budget_compliant"] = expected_provider_compliance
    return question, responder_usage, judge_usage


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
    paths = [Path(path) for path in report_paths]
    if not paths:
        raise CampaignMergeError("at least one report is required")

    loaded = [_load_report(path) for path in paths]
    # Argument order and artifact location cannot change floating-point
    # reduction order or campaign identity.
    loaded.sort(key=lambda row: (row[1], row[3]))

    expected_identity: dict[str, Any] | None = None
    input_rows: list[dict[str, Any]] = []
    questions_by_id: dict[str, dict[str, Any]] = {}
    question_sources: dict[str, dict[str, str]] = {}
    responder_usage_by_question: dict[str, dict[str, int | float]] = {}
    judge_usage_by_question: dict[str, dict[str, int | float]] = {}
    cache_receipts_by_sample: dict[
        str, dict[str, list[dict[str, object]]]
    ] = {}
    observed_compiled_cache_keys: set[str] = set()
    observed_causal_cache_keys: set[str] = set()
    sample_count = 0
    observed_offsets: set[int] = set()

    for report, digest, path_label, portable_name in loaded:
        label = f"report[{path_label}]"
        identity = _identity(report, label)
        if expected_identity is None:
            expected_identity = identity
            if locked_plan is not None:
                for field in _HASH_FIELDS:
                    expected = getattr(locked_plan, field)
                    if identity[field] != expected:
                        raise CampaignMergeError(
                            f"{label}.{field} does not match the independently "
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
                    locked_plan.evaluation.get(
                        "responder_output_token_reserve"
                    )
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
        else:
            _ensure_same_identity(expected_identity, identity, label)

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
        samples = _require_list(report.get("samples"), f"{label}.samples")
        declared_samples = _require_int(
            report.get("num_samples"), f"{label}.num_samples"
        )
        if declared_samples != len(samples):
            raise CampaignMergeError(
                f"{label}.num_samples={declared_samples} but contains "
                f"{len(samples)} sample rows"
            )

        expected_shard: ExpectedStressShard | None = None
        if locked_plan is not None:
            protocol = _require_mapping(
                report.get("evaluation_protocol"),
                f"{label}.evaluation_protocol",
            )
            sample_offset = _require_int(
                protocol.get("sample_offset"),
                f"{label}.evaluation_protocol.sample_offset",
            )
            if sample_offset in observed_offsets:
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
            observed_offsets.add(sample_offset)
            if len(samples) != 1:
                raise CampaignMergeError(
                    f"{label} must contain exactly one reconstructed stress sample"
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
                        expected_implementation_sha256=(
                            locked_plan.implementation_sha256
                            if locked_plan is not None
                            else None
                        ),
                        expected_environment_lock_sha256=(
                            locked_plan.environment_lock_sha256
                            if locked_plan is not None
                            else None
                        ),
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
                actual_receipt_sha256 = cache_receipts_sha256(
                    sample_cache_receipts
                )
                if reported_receipt_sha256 != actual_receipt_sha256:
                    raise CampaignMergeError(
                        f"{sample_label}.cache_receipts_sha256 does not match "
                        "the exact cache receipt pair"
                    )
                compiled_key = str(
                    sample_cache_receipts["compiled"][0]["cache_key"]
                )
                causal_key = str(
                    sample_cache_receipts["causal"][0]["cache_key"]
                )
                if compiled_key in observed_compiled_cache_keys:
                    raise CampaignMergeError(
                        "locked validation shards reuse a compiled cache entry"
                    )
                if causal_key in observed_causal_cache_keys:
                    raise CampaignMergeError(
                        "locked validation shards reuse a causal cache entry"
                    )
                observed_compiled_cache_keys.add(compiled_key)
                observed_causal_cache_keys.add(causal_key)
                cache_receipts_by_sample[
                    expected_shard.sample_sha256
                ] = sample_cache_receipts
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
            report_question_count += len(rows)
            sample_count += 1
            for question_index, raw_question in enumerate(rows):
                question_label = (
                    f"{sample_label}.question_results[{question_index}]"
                )
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
                    for field in ("question", "gold_answer", "category"):
                        if question.get(field) != expected_question[field]:
                            raise CampaignMergeError(
                                f"{question_label}.{field} does not match the "
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
                if question_id in questions_by_id:
                    prior = question_sources[question_id]
                    raise CampaignMergeError(
                        f"duplicate question_id {question_id!r} in {path_label}; "
                        f"already present in {prior['report_name']}"
                    )
                questions_by_id[question_id] = question
                report_prompt_counts.append(
                    int(question["prompt_token_proxy"])
                )
                report_provider_compliances.append(
                    question["provider_prompt_budget_compliant"]
                )
                question_sources[question_id] = {
                    "report_name": portable_name,
                    "report_sha256": digest,
                    "sample_id": sample_id,
                    "sample_sha256": str(sample.get("sample_sha256") or ""),
                }
                responder_usage_by_question[question_id] = responder_usage
                judge_usage_by_question[question_id] = judge_usage

        declared_report_questions = _require_int(
            report.get("num_questions"), f"{label}.num_questions"
        )
        if declared_report_questions != report_question_count:
            raise CampaignMergeError(
                f"{label}.num_questions={declared_report_questions} but "
                f"contains {report_question_count} question rows"
            )
        observed_max = _require_int(
            report.get("max_prompt_tokens_observed"),
            f"{label}.max_prompt_tokens_observed",
        )
        recomputed_max = max(report_prompt_counts, default=0)
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
            for value in report_provider_compliances
            if value is not None
        ]
        expected_provider_compliance = (
            all(available_provider_rows) if available_provider_rows else None
        )
        expected_provider_status = (
            "unavailable"
            if not available_provider_rows
            else "complete"
            if len(available_provider_rows) == len(report_provider_compliances)
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
        input_rows.append(
            {
                "name": portable_name,
                "sha256": digest,
                "num_samples": len(samples),
                "num_questions": report_question_count,
                "target_status": shard_target_status,
            }
        )

    assert expected_identity is not None
    question_ids = sorted(questions_by_id)
    questions = [questions_by_id[question_id] for question_id in question_ids]
    num_questions = len(questions)
    if locked_plan is not None:
        if observed_offsets != set(locked_plan.sample_offsets):
            missing_offsets = set(locked_plan.sample_offsets) - observed_offsets
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

    mean_f1 = _mean(float(question["f1"]) for question in questions)
    exact_match_rate = _mean(
        1.0 if question["exact_match"] else 0.0 for question in questions
    )
    judge_accuracy = _mean(
        1.0 if question["judge_correct"] else 0.0 for question in questions
    )
    context_distribution = _distribution(
        int(question["context_tokens"]) for question in questions
    )
    prompt_proxy_distribution = _distribution(
        int(question["prompt_token_proxy"]) for question in questions
    )
    request_proxy_distribution = _distribution(
        int(question["prompt_token_proxy"])
        + int(expected_identity["responder_output_token_reserve"])
        for question in questions
    )
    provider_input_counts = [
        int(responder_usage_by_question[question_id]["input_tokens"])
        for question_id in question_ids
        if int(responder_usage_by_question[question_id]["input_tokens"]) > 0
    ]
    provider_input_distribution = _distribution(provider_input_counts)
    provider_prompt_budget_compliance = (
        all(
            count <= int(expected_identity["max_prompt_tokens"])
            for count in provider_input_counts
        )
        if provider_input_counts
        else None
    )
    provider_input_usage_status = (
        "unavailable"
        if not provider_input_counts
        else "complete"
        if len(provider_input_counts) == num_questions
        else "partial"
    )
    transcript_distribution = _distribution(
        int(question.get("transcript_tokens", 0)) for question in questions
    )
    metric_target_met = judge_accuracy >= accuracy_target
    target_met = bool(
        locked_plan is not None
        and claim_profile_verified
        and metric_target_met
    )
    target_status = (
        "passed"
        if target_met
        else "unverified_claim_profile"
        if locked_plan is not None and not claim_profile_verified
        else "failed"
        if locked_plan is not None
        else "unverified_population"
    )
    ordered_inputs = sorted(input_rows, key=lambda row: (row["sha256"], row["name"]))
    input_set_sha256 = hashlib.sha256(
        _canonical_json(sorted(row["sha256"] for row in ordered_inputs)).encode(
            "utf-8"
        )
    ).hexdigest()

    # This is deliberately the final operation before emitting a certified
    # result. A source, policy, environment, or implementation that changed
    # while shard reports were being inspected cannot retain claim status.
    if locked_plan is not None:
        _assert_locked_plan_unchanged(locked_plan)

    return {
        "schema_version": 1,
        "report_type": "benchmark_campaign",
        "inputs": ordered_inputs,
        "input_count": len(ordered_inputs),
        "input_set_sha256": input_set_sha256,
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
            digest: cache_receipts_by_sample[digest]
            for digest in sorted(cache_receipts_by_sample)
        },
        "num_samples": sample_count,
        "num_questions": num_questions,
        "question_results": questions,
        "question_sources": {
            question_id: question_sources[question_id]
            for question_id in question_ids
        },
        "mean_f1": mean_f1,
        "exact_match_rate": exact_match_rate,
        "judge_accuracy": judge_accuracy,
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
        "provider_input_token_distribution": provider_input_distribution,
        "prompt_token_distribution": prompt_proxy_distribution,
        "transcript_token_distribution": transcript_distribution,
        "prompt_token_proxy_budget_compliance": True,
        "provider_prompt_budget_compliance": (
            provider_prompt_budget_compliance
        ),
        "provider_input_usage_status": provider_input_usage_status,
        "prompt_budget_compliance": True,
        "responder_usage": _sum_usage(
            responder_usage_by_question[question_id]
            for question_id in question_ids
        ),
        "judge_usage": _sum_usage(
            judge_usage_by_question[question_id] for question_id in question_ids
        ),
        "by_category": _category_metrics(questions),
        "accuracy_target": accuracy_target,
        "min_target_questions": min_questions,
        "accuracy_target_met": target_met,
        "metric_accuracy_target_met": metric_target_met,
        "locked_population_verified": locked_plan is not None,
        "target_status": target_status,
    }


def save_campaign_report(report: dict[str, Any], output: str | Path) -> Path:
    """Write a deterministic campaign JSON document."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (path.parent / f"{path.name}.sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n",
        encoding="ascii",
    )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge locked benchmark validation shards"
    )
    parser.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        required=True,
        help="Benchmark report JSON files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-questions", type=int, default=100)
    parser.add_argument("--accuracy-target", type=float, default=0.95)
    parser.add_argument("--benchmark-file", type=Path)
    parser.add_argument("--benchmark-format", default="auto")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument(
        "--allow-unverified-summary",
        action="store_true",
        help=(
            "Merge metrics without certifying the population; target_status "
            "will remain unverified_population"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if any(path.resolve() == output for path in args.reports):
        parser.error("--output must not overwrite an input report")
    try:
        verification_paths = (
            args.benchmark_file,
            args.split_manifest,
            args.policy_manifest,
        )
        if any(verification_paths) and not all(verification_paths):
            raise CampaignMergeError(
                "--benchmark-file, --split-manifest, and --policy-manifest "
                "must be supplied together"
            )
        if not all(verification_paths) and not args.allow_unverified_summary:
            raise CampaignMergeError(
                "locked certification requires --benchmark-file, "
                "--split-manifest, and --policy-manifest; use "
                "--allow-unverified-summary only for non-claim diagnostics"
            )
        locked_plan = (
            build_locked_validation_plan(
                benchmark_file=args.benchmark_file,
                benchmark_format=args.benchmark_format,
                split_manifest=args.split_manifest,
                policy_manifest=args.policy_manifest,
                repository_root=args.repository_root,
            )
            if all(verification_paths)
            else None
        )
        report = merge_benchmark_reports(
            args.reports,
            min_questions=args.min_questions,
            accuracy_target=args.accuracy_target,
            locked_plan=locked_plan,
        )
        path = save_campaign_report(report, output)
    except CampaignMergeError as exc:
        parser.error(str(exc))
    print(
        f"Merged {report['input_count']} shards / {report['num_questions']} "
        f"questions: judge={report['judge_accuracy']:.1%}, "
        f"target={report['target_status']}; saved {path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    raise SystemExit(main())
