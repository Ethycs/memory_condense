"""Locked validation plan construction and immutable-source rechecks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
)
from memory_condense.eval.campaign_models import (
    CampaignMergeError,
    ExpectedStressShard,
    LockedValidationPlan,
)
from memory_condense.eval.campaign_validation import (
    _canonical_json,
    _json_constant,
    _require_float,
    _require_int,
    _require_list,
    _require_mapping,
    _require_nonempty_string,
    _require_sha256,
)
from memory_condense.eval.context_stress import (
    compose_context_stress_sample,
    transcript_tokens,
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
from memory_condense.ingest.loader import load_benchmark


def _campaign_override(name: str, default: Any) -> Any:
    """Resolve a helper monkeypatched through the compatibility facade."""

    facade = sys.modules.get("memory_condense.eval.campaign")
    return getattr(facade, name, default)


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
            transcript_tokens=_campaign_override(
                "transcript_tokens", transcript_tokens
            )(stress_sample),
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
