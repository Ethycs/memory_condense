"""Provider-free launch contracts for the locked resumable Mem0 full100 arm.

This module is the trusted orchestration boundary, not a provider client.  It
reconstructs the exact validation100 population, verifies a current v3 Mem0
policy, materializes one immutable resume plan per namespace, and audits any
journals without provider calls.  Its separate live function is the only
production-capable extraction entrypoint: it holds the namespace lease,
reconstructs sealed launch authority, and issues one exact in-process segment
capability.  Provider-free artifacts themselves always grant zero calls.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .policy import (
    MEM0_LOCKED_ADD_OPERATIONS,
    MEM0_LOCKED_EXTRACTION_CALLS,
    MEM0_LOCKED_NAMESPACE_COUNT,
    MEM0_LOCKED_QUESTION_COUNT,
    MEM0_LOCKED_SEARCH_OPERATIONS,
    MEM0_POLICY_FORMAT,
    Mem0ComparisonPolicy,
    canonical_json_sha256,
    load_mem0_comparison_policy,
)
from .preflight import (
    SourceValidationPlan,
    load_source_validation_plan,
    tool_implementation_sha256,
)
from .protocol import RawStressShard, build_raw_stress_shards
from .resumable import (
    AppendOnlyResumeJournal,
    JournalLease,
    SNAPSHOT_ROOT_MARKER,
    ResumeAmbiguityError,
    ResumePlan,
    ResumableShardError,
    deterministic_user_scope,
    publish_sealed_json,
    read_journal,
    read_sealed_json,
    replay_journal,
    state_tree_receipt,
    verify_immutable_state_snapshot,
    _is_reparse_or_link,
    _read_record_segments,
    _path_identity_sha256,
    _snapshot_root_marker_from_header,
)
from .resumable_runner import (
    DEFAULT_SEGMENT_ADDS,
    RESUMABLE_LIVE_LAUNCH_AUTHORITY_FORMAT,
    ResumableSegmentResult,
    _OneUseSegmentAuthorizationIssuer,
    _prepare_locked_ingest_state,
    _run_resumable_ingest_segment_locked,
    _validate_terminal_stage,
    build_resume_plan,
)


PREFLIGHT_FORMAT = "memory-condense-mem0-resumable-launch-preflight-v1"
SHARD_FORMAT = "memory-condense-mem0-resumable-shard-launch-v1"
MANIFEST_FORMAT = "memory-condense-mem0-resumable-launch-manifest-v1"
REPLAY_FORMAT = "memory-condense-mem0-resumable-launch-replay-v1"

PREFLIGHT_NAME = "mem0-resumable-launch-preflight-v1.json"
MANIFEST_NAME = "mem0-resumable-launch-manifest-v1.json"
REPLAY_NAME = "mem0-resumable-launch-replay-v1.json"
SHARD_LAUNCH_NAME = "launch-v1.json"

LOCKED_SAMPLE_OFFSETS = tuple(range(0, 100, 10))
LOCKED_ADD_COUNTS = (2_548, 2_405, 2_457, 2_542, 2_521, 2_483, 2_390, 2_483, 2_516, 2_578)
LOCKED_TARGET_TOKENS = 1_000_000
LOCKED_QUESTIONS_PER_SHARD = 10

ANSWER_PROMPT_TOKEN_CAP = 7_232
ANSWER_OUTPUT_TOKEN_RESERVE = 768
ANSWER_COMPLETE_REQUEST_TOKEN_CAP = 8_000
JUDGE_PROMPT_TOKEN_CAP = 8_000
JUDGE_OUTPUT_TOKEN_RESERVE = 1_024
JUDGE_COMPLETE_REQUEST_TOKEN_CAP = 9_024

PROSPECTIVE_PROVIDER_CALLS = (
    MEM0_LOCKED_EXTRACTION_CALLS
    + MEM0_LOCKED_QUESTION_COUNT
    + MEM0_LOCKED_QUESTION_COUNT
)
WRITE_METERING_MISSING_FIELDS = (
    "extraction_provider_input_tokens",
    "extraction_provider_output_tokens",
    "embedding_operations",
    "embedding_input_token_proxy",
    "persisted_storage_bytes",
    "extraction_latency_s",
    "embedding_latency_s",
    "storage_latency_s",
)


class Mem0ResumableLaunchError(ValueError):
    """A launch input, sealed artifact, or journal escaped the locked arm."""


@dataclass(frozen=True, slots=True)
class LockedLaunchInputs:
    benchmark_file: Path
    split_manifest: Path
    source_policy_manifest: Path
    source_repository_root: Path
    mem0_policy_manifest: Path
    expected_mem0_policy_sha256: str
    mem0_environment_lock: Path
    tool_root: Path


@dataclass(frozen=True, slots=True)
class ShardLaunchBinding:
    sample_offset: int
    sample_id: str
    sample_sha256: str
    raw_history_bundle_sha256: str
    question_ids: tuple[str, ...]
    authorization_sha256: str
    plan: ResumePlan


@dataclass(frozen=True, slots=True)
class LockedLaunchContext:
    source_plan: SourceValidationPlan
    mem0_policy_sha256: str
    mem0_tool_implementation_sha256: str
    mem0_environment_lock_sha256: str
    shards: tuple[ShardLaunchBinding, ...]


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise Mem0ResumableLaunchError(f"{label} must be lowercase SHA-256")
    return value


def _strict_json(value: Any, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise Mem0ResumableLaunchError(f"{label} is not strict JSON") from exc


def _sealed_payload_sha256(value: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(raw).hexdigest()


def _verify_policy_header_and_digest(
    path: str | Path, expected_sha256: str
) -> str:
    """Reject stale policy formats and byte substitutions before source work."""

    expected = _require_sha256(expected_sha256, "expected Mem0 policy SHA-256")
    target = Path(path).resolve()
    raw = target.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise Mem0ResumableLaunchError("Mem0 policy byte SHA-256 mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Mem0ResumableLaunchError("Mem0 policy is not JSON") from exc
    if type(value) is not dict or value.get("format") != MEM0_POLICY_FORMAT:
        raise Mem0ResumableLaunchError(
            f"Mem0 launch requires exact policy format {MEM0_POLICY_FORMAT}"
        )
    return actual


def _authoritative_common_parent_contract() -> tuple[dict[str, Any], str]:
    """Load the dependency-light score-plane request identity."""

    from .common_parent_contract import COMPARISON_SEMANTICS, EXACT_ACCOUNTING

    accounting = _strict_json(EXACT_ACCOUNTING, "authoritative exact accounting")
    if type(accounting) is not dict:
        raise Mem0ResumableLaunchError(
            "authoritative EXACT_ACCOUNTING must be an object"
        )
    if not isinstance(COMPARISON_SEMANTICS, str) or not COMPARISON_SEMANTICS:
        raise Mem0ResumableLaunchError(
            "authoritative COMPARISON_SEMANTICS must be non-empty text"
        )
    return accounting, COMPARISON_SEMANTICS


def _require_common_parent_accounting() -> tuple[dict[str, Any], str]:
    accounting, semantics = _authoritative_common_parent_contract()
    expected = {
        "answer_complete_request_token_cap": ANSWER_COMPLETE_REQUEST_TOKEN_CAP,
        "answer_max_prompt_tokens": ANSWER_PROMPT_TOKEN_CAP,
        "answer_output_token_reserve": ANSWER_OUTPUT_TOKEN_RESERVE,
        "judge_complete_envelope_token_cap": JUDGE_COMPLETE_REQUEST_TOKEN_CAP,
        "judge_max_prompt_tokens": JUDGE_PROMPT_TOKEN_CAP,
        "judge_model": "codex_sdk/gpt-5.6-sol",
        "judge_output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
        "responder_model": "codex_sdk/gpt-5.6-terra",
        "retained_transformer_token_state_bytes": 0,
        "sdk_retries": 0,
    }
    if accounting != expected or semantics != "common_parent":
        raise Mem0ResumableLaunchError(
            "launch accounting diverged from authoritative common-parent contract"
        )
    return accounting, semantics


def _validate_source_plan(plan: SourceValidationPlan) -> None:
    if plan.sample_offsets != LOCKED_SAMPLE_OFFSETS:
        raise Mem0ResumableLaunchError("source plan is not exact validation100 order")
    if plan.target_tokens != LOCKED_TARGET_TOKENS:
        raise Mem0ResumableLaunchError("source plan is not the 1M-token workload")
    if plan.questions_per_shard != LOCKED_QUESTIONS_PER_SHARD:
        raise Mem0ResumableLaunchError("source plan must bind ten questions per shard")
    evaluation = plan.evaluation_identity
    exact = {
        "provider_retries": 0,
        "stress_context_tokens": LOCKED_TARGET_TOKENS,
        "stress_questions": LOCKED_QUESTIONS_PER_SHARD,
        "stress_question_offset": 0,
        "max_samples": 1,
        "min_target_questions": MEM0_LOCKED_QUESTION_COUNT,
    }
    for field, expected in exact.items():
        if evaluation.get(field) != expected:
            raise Mem0ResumableLaunchError(
                f"source evaluation identity {field} mismatch"
            )
    if tuple(evaluation.get("sample_offsets", ())) != LOCKED_SAMPLE_OFFSETS:
        raise Mem0ResumableLaunchError(
            "source evaluation sample-offset order mismatch"
        )


def _verify_source_import_origin(repository_root: Path) -> None:
    spec = importlib.util.find_spec("memory_condense")
    locations = tuple(spec.submodule_search_locations or ()) if spec else ()
    expected = (repository_root.resolve() / "src" / "memory_condense").resolve()
    if len(locations) != 1 or Path(locations[0]).resolve() != expected:
        raise Mem0ResumableLaunchError(
            "imported memory_condense package is not the source tree being hashed"
        )


def _binding_from_policy(
    policy: Mem0ComparisonPolicy, shard: RawStressShard
) -> ShardLaunchBinding:
    authorization = policy.retrieval_authorization(shard)
    authorization_sha256 = canonical_json_sha256(asdict(authorization))
    plan = build_resume_plan(
        shard=shard,
        authorization=authorization,
        authorization_sha256=authorization_sha256,
    )
    return ShardLaunchBinding(
        sample_offset=shard.sample_offset,
        sample_id=shard.parsed_sample.sample_id,
        sample_sha256=shard.sample_sha256,
        raw_history_bundle_sha256=shard.raw_history_bundle_sha256,
        question_ids=shard.question_ids,
        authorization_sha256=authorization_sha256,
        plan=plan,
    )


def _validate_context(context: LockedLaunchContext) -> None:
    _require_common_parent_accounting()
    _validate_source_plan(context.source_plan)
    for value, label in (
        (context.mem0_policy_sha256, "Mem0 policy"),
        (context.mem0_tool_implementation_sha256, "Mem0 tool"),
        (context.mem0_environment_lock_sha256, "Mem0 lock"),
    ):
        _require_sha256(value, label)
    if len(context.shards) != MEM0_LOCKED_NAMESPACE_COUNT:
        raise Mem0ResumableLaunchError("launch must contain exactly ten namespaces")
    offsets = tuple(row.sample_offset for row in context.shards)
    if offsets != LOCKED_SAMPLE_OFFSETS:
        raise Mem0ResumableLaunchError("launch shard order is not validation100 order")
    question_ids = tuple(
        question_id for row in context.shards for question_id in row.question_ids
    )
    if (
        len(question_ids) != MEM0_LOCKED_QUESTION_COUNT
        or len(set(question_ids)) != MEM0_LOCKED_QUESTION_COUNT
    ):
        raise Mem0ResumableLaunchError(
            "launch questions are not the unique validation100 population"
        )
    scopes = tuple(row.plan.user_scope for row in context.shards)
    if len(set(scopes)) != MEM0_LOCKED_NAMESPACE_COUNT:
        raise Mem0ResumableLaunchError("Mem0 namespaces are not unique per shard")
    add_counts = tuple(row.plan.authorized_add_operations for row in context.shards)
    if add_counts != LOCKED_ADD_COUNTS:
        raise Mem0ResumableLaunchError("locked per-shard add counts changed")
    if sum(add_counts) != MEM0_LOCKED_ADD_OPERATIONS:
        raise Mem0ResumableLaunchError("locked add population changed")
    if sum(row.plan.authorized_extraction_calls for row in context.shards) != MEM0_LOCKED_EXTRACTION_CALLS:
        raise Mem0ResumableLaunchError("locked extraction population changed")
    if sum(row.plan.authorized_search_operations for row in context.shards) != MEM0_LOCKED_SEARCH_OPERATIONS:
        raise Mem0ResumableLaunchError("locked search population changed")
    for row in context.shards:
        plan = row.plan
        if plan.user_scope != deterministic_user_scope(row.authorization_sha256):
            raise Mem0ResumableLaunchError(
                f"shard {row.sample_offset} namespace is not derived from "
                "authorization SHA-256"
            )
        exact = {
            "authorization_sha256": row.authorization_sha256,
            "mem0_policy_sha256": context.mem0_policy_sha256,
            "source_validation_policy_sha256": context.source_plan.policy_manifest_sha256,
            "source_implementation_sha256": context.source_plan.implementation_sha256,
            "source_environment_lock_sha256": context.source_plan.environment_lock_sha256,
            "mem0_tool_implementation_sha256": context.mem0_tool_implementation_sha256,
            "mem0_environment_lock_sha256": context.mem0_environment_lock_sha256,
            "sample_offset": row.sample_offset,
            "sample_sha256": row.sample_sha256,
            "raw_history_bundle_sha256": row.raw_history_bundle_sha256,
            "authorized_search_operations": LOCKED_QUESTIONS_PER_SHARD,
        }
        for field, expected in exact.items():
            if getattr(plan, field) != expected:
                raise Mem0ResumableLaunchError(
                    f"shard {row.sample_offset} resume plan {field} mismatch"
                )
        if len(row.question_ids) != LOCKED_QUESTIONS_PER_SHARD:
            raise Mem0ResumableLaunchError(
                f"shard {row.sample_offset} question population changed"
            )
        if plan.authorized_add_operations != plan.authorized_extraction_calls:
            raise Mem0ResumableLaunchError(
                f"shard {row.sample_offset} add/extraction budget diverged"
            )


def load_locked_launch_context(inputs: LockedLaunchInputs) -> LockedLaunchContext:
    """Reconstruct and authenticate the only launchable full100 population."""

    initial_policy_sha = _verify_policy_header_and_digest(
        inputs.mem0_policy_manifest, inputs.expected_mem0_policy_sha256
    )
    tool_root = inputs.tool_root.resolve()
    if tool_root != Path(__file__).resolve().parent:
        raise Mem0ResumableLaunchError(
            "tool root must be the package containing this launch implementation"
        )
    _verify_source_import_origin(inputs.source_repository_root)
    source_plan = load_source_validation_plan(
        benchmark_file=inputs.benchmark_file,
        split_manifest=inputs.split_manifest,
        policy_manifest=inputs.source_policy_manifest,
        repository_root=inputs.source_repository_root,
    )
    _validate_source_plan(source_plan)
    raw_shards = build_raw_stress_shards(
        benchmark_file=inputs.benchmark_file,
        split_manifest=inputs.split_manifest,
        sample_offsets=LOCKED_SAMPLE_OFFSETS,
        target_tokens=LOCKED_TARGET_TOKENS,
        max_questions=LOCKED_QUESTIONS_PER_SHARD,
    )
    policy = load_mem0_comparison_policy(
        inputs.mem0_policy_manifest,
        source_plan=source_plan,
        mem0_environment_lock=inputs.mem0_environment_lock,
        expected_shards=raw_shards,
        tool_root=tool_root,
    )
    if policy.sha256 != initial_policy_sha:
        raise Mem0ResumableLaunchError("Mem0 policy changed during launch preflight")
    bindings = tuple(_binding_from_policy(policy, shard) for shard in raw_shards)
    policy.recheck()
    if hashlib.sha256(inputs.mem0_policy_manifest.read_bytes()).hexdigest() != initial_policy_sha:
        raise Mem0ResumableLaunchError("Mem0 policy changed after plan construction")
    context = LockedLaunchContext(
        source_plan=source_plan,
        mem0_policy_sha256=policy.sha256,
        mem0_tool_implementation_sha256=policy.tool_implementation_sha256,
        mem0_environment_lock_sha256=policy.environment_lock_sha256,
        shards=bindings,
    )
    _validate_context(context)
    recheck_locked_launch_inputs(inputs, context)
    return context


def recheck_locked_launch_inputs(
    inputs: LockedLaunchInputs, context: LockedLaunchContext
) -> None:
    """Re-hash every mutable external identity without rebuilding the corpus."""

    _validate_context(context)
    policy_sha = _verify_policy_header_and_digest(
        inputs.mem0_policy_manifest, inputs.expected_mem0_policy_sha256
    )
    if policy_sha != context.mem0_policy_sha256:
        raise Mem0ResumableLaunchError("Mem0 policy changed after reconstruction")
    tool_root = inputs.tool_root.resolve()
    if (
        tool_root != Path(__file__).resolve().parent
        or tool_implementation_sha256(tool_root)
        != context.mem0_tool_implementation_sha256
    ):
        raise Mem0ResumableLaunchError("Mem0 tool changed after reconstruction")
    if (
        hashlib.sha256(inputs.mem0_environment_lock.read_bytes()).hexdigest()
        != context.mem0_environment_lock_sha256
    ):
        raise Mem0ResumableLaunchError("Mem0 environment lock changed after reconstruction")
    final_source = load_source_validation_plan(
        benchmark_file=inputs.benchmark_file,
        split_manifest=inputs.split_manifest,
        policy_manifest=inputs.source_policy_manifest,
        repository_root=inputs.source_repository_root,
    )
    if final_source != context.source_plan:
        raise Mem0ResumableLaunchError("source inputs changed after reconstruction")


def _common_parent_budget() -> dict[str, Any]:
    accounting, semantics = _require_common_parent_accounting()
    return {
        "comparison_semantics": semantics,
        "exact_accounting_sha256": canonical_json_sha256(accounting),
        "answer": {
            "calls": MEM0_LOCKED_QUESTION_COUNT,
            "complete_request_token_cap": accounting[
                "answer_complete_request_token_cap"
            ],
            "max_prompt_tokens": accounting["answer_max_prompt_tokens"],
            "model": accounting["responder_model"],
            "output_token_reserve": accounting["answer_output_token_reserve"],
        },
        "judge": {
            "calls": MEM0_LOCKED_QUESTION_COUNT,
            "complete_request_token_cap": accounting[
                "judge_complete_envelope_token_cap"
            ],
            "max_prompt_tokens": accounting["judge_max_prompt_tokens"],
            "model": accounting["judge_model"],
            "output_token_reserve": accounting["judge_output_token_reserve"],
        },
        "retained_transformer_token_state_bytes": accounting[
            "retained_transformer_token_state_bytes"
        ],
        "sdk_retries": accounting["sdk_retries"],
    }


def _provider_and_cost_contract() -> dict[str, Any]:
    return {
        "provider_call_authorization": {
            "authorization_granted": False,
            "authorization_source": None,
            "physical_provider_calls_performed": 0,
            "prospective_hard_call_ceiling": PROSPECTIVE_PROVIDER_CALLS,
            "prospective_calls": {
                "mem0_extraction": MEM0_LOCKED_EXTRACTION_CALLS,
                "common_parent_answer": MEM0_LOCKED_QUESTION_COUNT,
                "common_parent_judge": MEM0_LOCKED_QUESTION_COUNT,
            },
            "sdk_retries": 0,
            "live_segment_entrypoint_exposed": True,
            "reason": (
                "provider-free artifacts grant zero calls; the separate live "
                "entrypoint reconstructs these seals and issues one exact "
                "in-process segment capability under the journal lease"
            ),
        },
        "cost_accounting": {
            "preflight_cost_incurred": False,
            "provider_usage_zero_fill_authorized": False,
            "write_cost_comparison_eligible": False,
            "write_metering_status": (
                "implementation_complete_live_observations_pending"
            ),
            "missing_write_metering_fields": list(WRITE_METERING_MISSING_FIELDS),
            "logical_extraction_calls_are_not_provider_token_usage": True,
            "post_call_usage_and_price_schedule_required": True,
        },
    }


def build_preflight_payload(context: LockedLaunchContext) -> dict[str, Any]:
    """Build the deterministic, gold-free, zero-call launch preflight."""

    _validate_context(context)
    question_ids = [
        question_id for row in context.shards for question_id in row.question_ids
    ]
    rows = []
    for row in context.shards:
        rows.append(
            {
                "sample_offset": row.sample_offset,
                "sample_id": row.sample_id,
                "sample_sha256": row.sample_sha256,
                "raw_history_bundle_sha256": row.raw_history_bundle_sha256,
                "question_ids": list(row.question_ids),
                "question_ids_sha256": canonical_json_sha256(list(row.question_ids)),
                "authorization_sha256": row.authorization_sha256,
                "resume_plan_sha256": row.plan.sha256,
                "ordered_batches_sha256": row.plan.as_dict()[
                    "ordered_batches_sha256"
                ],
                "namespace": row.plan.user_scope,
                "namespace_sha256": hashlib.sha256(
                    row.plan.user_scope.encode("utf-8")
                ).hexdigest(),
                "cross_namespace_reads_authorized": False,
                "authorized_add_operations": row.plan.authorized_add_operations,
                "authorized_extraction_calls": row.plan.authorized_extraction_calls,
                "authorized_search_operations": row.plan.authorized_search_operations,
            }
        )
    payload = {
        "format": PREFLIGHT_FORMAT,
        "status": "provider_free_materialization_ready",
        "gold_handling": {
            "references_loaded_for_source_validation": True,
            "references_persisted_in_launch_artifacts": False,
            "references_exposed_to_provider": False,
        },
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "source": {
            "dataset_sha256": context.source_plan.dataset_sha256,
            "split_manifest_sha256": context.source_plan.split_manifest_sha256,
            "source_policy_sha256": context.source_plan.policy_manifest_sha256,
            "source_implementation_sha256": context.source_plan.implementation_sha256,
            "source_environment_lock_sha256": context.source_plan.environment_lock_sha256,
            "target_tokens_per_shard": LOCKED_TARGET_TOKENS,
            "sample_offsets": list(LOCKED_SAMPLE_OFFSETS),
        },
        "mem0": {
            "policy_format": MEM0_POLICY_FORMAT,
            "policy_sha256": context.mem0_policy_sha256,
            "tool_implementation_sha256": context.mem0_tool_implementation_sha256,
            "environment_lock_sha256": context.mem0_environment_lock_sha256,
            "segment_adds": DEFAULT_SEGMENT_ADDS,
            "namespace_count": MEM0_LOCKED_NAMESPACE_COUNT,
            "cross_namespace_reads_authorized": False,
            "provider_sdk_retries": 0,
        },
        "population": {
            "questions": MEM0_LOCKED_QUESTION_COUNT,
            "question_ids_sha256": canonical_json_sha256(question_ids),
            "add_operations": MEM0_LOCKED_ADD_OPERATIONS,
            "logical_extraction_calls": MEM0_LOCKED_EXTRACTION_CALLS,
            "search_operations": MEM0_LOCKED_SEARCH_OPERATIONS,
        },
        "common_parent_request_budget": _common_parent_budget(),
        **_provider_and_cost_contract(),
        "shards": rows,
    }
    return _strict_json(payload, "launch preflight")


def _shard_relative_dir(sample_offset: int) -> str:
    return f"shard-{sample_offset:03d}"


def build_shard_payload(
    context: LockedLaunchContext,
    row: ShardLaunchBinding,
    *,
    preflight_sha256: str,
) -> dict[str, Any]:
    _validate_context(context)
    _require_sha256(preflight_sha256, "preflight SHA-256")
    if row not in context.shards:
        raise Mem0ResumableLaunchError("shard is not in the locked launch context")
    relative = _shard_relative_dir(row.sample_offset)
    payload = {
        "format": SHARD_FORMAT,
        "status": "provider_free_plan_only",
        "preflight_sha256": preflight_sha256,
        "sample_offset": row.sample_offset,
        "sample_id": row.sample_id,
        "question_ids": list(row.question_ids),
        "authorization_sha256": row.authorization_sha256,
        "namespace": {
            "user_scope": row.plan.user_scope,
            "user_scope_sha256": hashlib.sha256(
                row.plan.user_scope.encode("utf-8")
            ).hexdigest(),
            "one_namespace_per_shard": True,
            "cross_namespace_reads_authorized": False,
        },
        "ordering": {
            "sample_offset_order": list(LOCKED_SAMPLE_OFFSETS),
            "question_ids_sha256": canonical_json_sha256(list(row.question_ids)),
            "ordered_batches_sha256": row.plan.as_dict()[
                "ordered_batches_sha256"
            ],
            "ordered_batch_count": row.plan.authorized_add_operations,
        },
        "resume_plan": row.plan.as_dict(),
        "paths": {
            "journal_run_root_relative": f"{relative}/resume.jsonl",
            "runner_paths_relative_to_journal_parent": {
                "owned_state": "owned-state",
                "snapshot_root": "snapshots",
                "terminal_stage": "terminal-stage.json",
                "retrieval_artifact": "retrieval.json",
                "retrieval_trace": "retrieval.trace.json",
            },
        },
        "operation_budget": {
            "segment_adds": DEFAULT_SEGMENT_ADDS,
            "add_operations": row.plan.authorized_add_operations,
            "logical_extraction_calls": row.plan.authorized_extraction_calls,
            "search_operations": row.plan.authorized_search_operations,
            "provider_sdk_retries": 0,
        },
        "provider_call_authorization": {
            "authorization_granted": False,
            "physical_provider_calls_performed": 0,
            "live_segment_entrypoint_exposed": True,
        },
        "retained_transformer_token_state_bytes": 0,
    }
    return _strict_json(payload, f"shard {row.sample_offset} launch")


def _read_expected_sealed(
    path: str | Path, expected_sha256: str, label: str
) -> dict[str, Any]:
    try:
        receipt = read_sealed_json(path, expected_sha256=expected_sha256)
    except (OSError, ResumableShardError) as exc:
        raise Mem0ResumableLaunchError(f"{label} is not a valid sealed artifact") from exc
    payload = receipt["payload"]
    if type(payload) is not dict:
        raise Mem0ResumableLaunchError(f"{label} payload must be an object")
    return receipt


def _verify_preflight(
    context: LockedLaunchContext,
    path: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    receipt = _read_expected_sealed(path, expected_sha256, "launch preflight")
    expected = build_preflight_payload(context)
    if receipt["payload"] != expected:
        raise Mem0ResumableLaunchError(
            "sealed launch preflight is not the current reconstructed contract"
        )
    return receipt


def materialize_launch(
    *,
    context: LockedLaunchContext,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    run_root: str | Path,
    dry_run: bool = False,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Publish deterministic per-shard plans and a zero-authority manifest."""

    preflight = _verify_preflight(
        context, preflight_path, expected_preflight_sha256
    )
    root = Path(run_root)
    if not dry_run:
        copied = publish_sealed_json(root / PREFLIGHT_NAME, preflight["payload"])
        if copied["sha256"] != preflight["sha256"]:
            raise Mem0ResumableLaunchError(
                "materialized preflight digest differs from verified source"
            )
    shard_payloads = tuple(
        build_shard_payload(
            context, row, preflight_sha256=preflight["sha256"]
        )
        for row in context.shards
    )
    artifact_rows = []
    for row, payload in zip(context.shards, shard_payloads, strict=True):
        relative_path = f"{_shard_relative_dir(row.sample_offset)}/{SHARD_LAUNCH_NAME}"
        expected_file_sha = _sealed_payload_sha256(payload)
        if not dry_run:
            receipt = publish_sealed_json(root / Path(relative_path), payload)
            if receipt["sha256"] != expected_file_sha:
                raise Mem0ResumableLaunchError("published shard launch digest changed")
        artifact_rows.append(
            {
                "sample_offset": row.sample_offset,
                "path": relative_path,
                "artifact_sha256": expected_file_sha,
                "payload_sha256": canonical_json_sha256(payload),
                "resume_plan_sha256": row.plan.sha256,
                "namespace_sha256": hashlib.sha256(
                    row.plan.user_scope.encode("utf-8")
                ).hexdigest(),
            }
        )
    manifest = {
        "format": MANIFEST_FORMAT,
        "status": "provider_free_plans_materialized" if not dry_run else "provider_free_dry_run",
        "preflight_path": PREFLIGHT_NAME,
        "preflight_sha256": preflight["sha256"],
        "mem0_policy_sha256": context.mem0_policy_sha256,
        "mem0_tool_implementation_sha256": context.mem0_tool_implementation_sha256,
        "mem0_environment_lock_sha256": context.mem0_environment_lock_sha256,
        "sample_offsets": list(LOCKED_SAMPLE_OFFSETS),
        "common_parent_request_budget": _common_parent_budget(),
        **_provider_and_cost_contract(),
        "gold_handling": {
            "references_loaded_for_source_validation": True,
            "references_persisted_in_launch_artifacts": False,
            "references_exposed_to_provider": False,
        },
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "shard_manifests": artifact_rows,
    }
    manifest = _strict_json(manifest, "launch manifest")
    if not dry_run:
        publish_sealed_json(root / MANIFEST_NAME, manifest)
    return manifest, shard_payloads


def _path_under_run_root(root: Path, relative_value: Any, label: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise Mem0ResumableLaunchError(f"{label} must be a relative path")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise Mem0ResumableLaunchError(f"{label} is unsafe")
    target = (root / relative).resolve(strict=False)
    base = root.resolve(strict=False)
    if target == base or base not in target.parents:
        raise Mem0ResumableLaunchError(f"{label} escapes the run root")
    return target


def _path_present(path: Path) -> bool:
    """Treat broken links and Windows reparse points as occupied paths."""

    return path.exists() or _is_reparse_or_link(path)


def _verify_snapshot_root_marker(
    snapshot_root: Path, header: Mapping[str, Any]
) -> None:
    if _is_reparse_or_link(snapshot_root) or not snapshot_root.is_dir():
        raise Mem0ResumableLaunchError("snapshot root is not a plain directory")
    expected = _snapshot_root_marker_from_header(header)
    try:
        receipt = read_sealed_json(snapshot_root / SNAPSHOT_ROOT_MARKER)
    except (OSError, ResumableShardError) as exc:
        raise Mem0ResumableLaunchError("snapshot-root marker is invalid") from exc
    if receipt["payload"] != expected:
        raise Mem0ResumableLaunchError("snapshot-root marker identity changed")


def _verify_latest_snapshot(
    *,
    journal_path: Path,
    snapshot_root: Path,
    state: Any,
) -> None:
    _verify_snapshot_root_marker(snapshot_root, state.entries[0])
    seal = state.latest_prefix_seal
    if seal is None:
        return
    relative = seal.get("snapshot_path")
    if not isinstance(relative, str):
        raise Mem0ResumableLaunchError("latest snapshot path is invalid")
    snapshot = (journal_path.parent / Path(relative)).resolve(strict=False)
    if snapshot_root not in snapshot.parents:
        raise Mem0ResumableLaunchError("latest snapshot escaped its fixed root")
    try:
        verified = verify_immutable_state_snapshot(
            snapshot,
            expected_authority_sha256=seal["snapshot_authority_sha256"],
            expected_manifest_sha256=seal["snapshot_manifest_sha256"],
            expected_tree_sha256=seal["snapshot_tree_sha256"],
            expected_ownership_token_sha256=seal["ownership_token_sha256"],
        )
    except (OSError, ResumableShardError) as exc:
        raise Mem0ResumableLaunchError("latest immutable snapshot is invalid") from exc
    authority = verified["snapshot_authority"]
    if authority.get("journal_path_sha256") != _path_identity_sha256(journal_path):
        raise Mem0ResumableLaunchError("snapshot authority journal path changed")


def _verify_working_state_if_present(state_path: Path, state: Any) -> None:
    if not _path_present(state_path):
        return
    seal = state.latest_prefix_seal
    if seal is None:
        raise Mem0ResumableLaunchError(
            "working state exists without an immutable prefix"
        )
    try:
        receipt = state_tree_receipt(state_path)
    except (OSError, ResumableShardError) as exc:
        raise Mem0ResumableLaunchError("working state tree is invalid") from exc
    if (
        receipt["snapshot_tree_sha256"] != seal["snapshot_tree_sha256"]
        or receipt["ownership_token_sha256"] != seal["ownership_token_sha256"]
    ):
        raise Mem0ResumableLaunchError(
            "working state differs from the immutable prefix"
        )


def _verify_terminal_files(
    *,
    state: Any,
    stage_path: Path,
    artifact_path: Path,
    trace_path: Path,
    expected_artifact_relative: str,
    expected_trace_relative: str,
) -> None:
    terminal = state.terminal_search
    stage_sidecar = stage_path.with_name(stage_path.name + ".sha256")
    artifact_sidecar = artifact_path.with_name(artifact_path.name + ".sha256")
    trace_sidecar = trace_path.with_name(trace_path.name + ".sha256")
    if _path_present(artifact_sidecar) or _path_present(trace_sidecar):
        raise Mem0ResumableLaunchError(
            "official terminal outputs cannot carry sealed-JSON sidecars"
        )
    if terminal is None:
        if any(
            _path_present(path)
            for path in (stage_path, stage_sidecar, artifact_path, trace_path)
        ):
            raise Mem0ResumableLaunchError(
                "terminal files exist before terminal search authority"
            )
        return
    stage_exists = _path_present(stage_path)
    sidecar_exists = _path_present(stage_sidecar)
    if stage_exists != sidecar_exists:
        raise Mem0ResumableLaunchError("terminal staging bundle is partial")
    if state.checkpoint_gc is None and (
        state.terminal_published is None or stage_exists
    ):
        try:
            stage = read_sealed_json(
                stage_path, expected_sha256=terminal["terminal_stage_sha256"]
            )
            _validate_terminal_stage(artifact=stage["payload"], state=state)
        except (OSError, ResumableShardError) as exc:
            raise Mem0ResumableLaunchError("terminal staging artifact is invalid") from exc
    elif state.checkpoint_gc is not None and stage_exists:
        raise Mem0ResumableLaunchError("terminal stage remained after checkpoint GC")
    published = state.terminal_published
    if published is None:
        artifact_exists = _path_present(artifact_path)
        trace_exists = _path_present(trace_path)
        if artifact_exists != trace_exists:
            raise Mem0ResumableLaunchError(
                "unjournaled terminal output transaction is partial"
            )
        if artifact_exists and state.active_state_removed is None:
            raise Mem0ResumableLaunchError(
                "terminal outputs predate active-state removal authority"
            )
        return
    if (
        published.get("official_artifact_path") != expected_artifact_relative
        or published.get("official_trace_path") != expected_trace_relative
    ):
        raise Mem0ResumableLaunchError("terminal output paths escaped launch plan")
    receipts = []
    try:
        for target, field in (
            (artifact_path, "official_artifact_sha256"),
            (trace_path, "official_trace_sha256"),
        ):
            if _is_reparse_or_link(target) or not target.is_file():
                raise Mem0ResumableLaunchError("official terminal output is absent")
            receipts.append((hashlib.sha256(target.read_bytes()).hexdigest(), field))
    except OSError as exc:
        raise Mem0ResumableLaunchError(
            "official terminal output could not be authenticated"
        ) from exc
    if any(published[field] != digest for digest, field in receipts):
        raise Mem0ResumableLaunchError("official terminal output digest changed")


def _journal_status(
    root: Path,
    payload: Mapping[str, Any],
    row: ShardLaunchBinding,
    *,
    expected_live_launch_authority: Mapping[str, Any],
) -> dict[str, Any]:
    paths = payload.get("paths")
    if type(paths) is not dict:
        raise Mem0ResumableLaunchError("shard launch paths changed type")
    journal_path = _path_under_run_root(
        root, paths.get("journal_run_root_relative"), "journal path"
    )
    runner = paths.get("runner_paths_relative_to_journal_parent")
    expected_runner = {
        "owned_state": "owned-state",
        "snapshot_root": "snapshots",
        "terminal_stage": "terminal-stage.json",
        "retrieval_artifact": "retrieval.json",
        "retrieval_trace": "retrieval.trace.json",
    }
    if runner != expected_runner:
        raise Mem0ResumableLaunchError("runner-relative paths changed")
    state_path = journal_path.parent / expected_runner["owned_state"]
    snapshot_root = journal_path.parent / expected_runner["snapshot_root"]
    stage_path = journal_path.parent / expected_runner["terminal_stage"]
    artifact_path = journal_path.parent / expected_runner["retrieval_artifact"]
    trace_path = journal_path.parent / expected_runner["retrieval_trace"]
    records = journal_path.with_name(journal_path.name + ".records")
    journal_present = _path_present(journal_path)
    records_present = _path_present(records)
    if not journal_present and not records_present:
        orphan_candidates = (
            state_path,
            snapshot_root,
            stage_path,
            stage_path.with_name(stage_path.name + ".sha256"),
            artifact_path,
            artifact_path.with_name(artifact_path.name + ".sha256"),
            trace_path,
            trace_path.with_name(trace_path.name + ".sha256"),
        )
        if any(_path_present(candidate) for candidate in orphan_candidates):
            raise Mem0ResumableLaunchError(
                f"shard {row.sample_offset} has state without an authoritative journal"
            )
        return {
            "sample_offset": row.sample_offset,
            "status": "not_started",
            "committed_prefix": 0,
            "sealed_prefix": 0,
            "journal_tail_sha256": None,
            "provider_calls_performed_by_replay": 0,
        }
    if journal_present and not records_present:
        raise Mem0ResumableLaunchError(
            f"shard {row.sample_offset} journal lacks its atomic record root"
        )
    projection_repair_required = not journal_present
    try:
        # This phase is a byte-preserving audit.  The mutable runner's
        # ``AppendOnlyResumeJournal.replay`` may repair a truncated JSONL
        # projection from atomic records, so audit the authoritative records
        # directly when the projection is absent and otherwise require exact
        # projection/record equality through ``read_journal``.
        entries = (
            _read_record_segments(journal_path)
            if projection_repair_required
            else read_journal(journal_path)
        )
        state = replay_journal(
            entries, expected_plan=row.plan
        )
    except (OSError, ResumableShardError) as exc:
        raise Mem0ResumableLaunchError(
            f"shard {row.sample_offset} resume journal failed strict replay"
        ) from exc
    header = state.entries[0]
    if header.get("journal_path_sha256") != _path_identity_sha256(journal_path):
        raise Mem0ResumableLaunchError("journal was moved from its authorized path")
    if (
        header.get("owned_state_path") != expected_runner["owned_state"]
        or header.get("snapshot_root_path") != expected_runner["snapshot_root"]
    ):
        raise Mem0ResumableLaunchError("journal state paths escaped launch plan")
    for seal in (
        entry for entry in state.entries if entry.get("kind") == "prefix_sealed"
    ):
        attestation = seal.get("write_usage_attestation")
        authorization = (
            attestation.get("segment_authorization_receipt")
            if isinstance(attestation, dict)
            else None
        )
        authority = (
            authorization.get("live_launch_authority")
            if isinstance(authorization, dict)
            else None
        )
        if authority != expected_live_launch_authority:
            raise Mem0ResumableLaunchError(
                "journal segment was not issued from this sealed launch"
            )
    if state.checkpoint_gc is None and state.terminal_published is None:
        _verify_latest_snapshot(
            journal_path=journal_path,
            snapshot_root=snapshot_root,
            state=state,
        )
    elif state.checkpoint_gc is None and _path_present(snapshot_root):
        # Once official outputs are authoritative, the runner may have removed
        # this root and crashed before journaling checkpoint_gc.  If it remains,
        # it must still be the fully authenticated root.
        _verify_latest_snapshot(
            journal_path=journal_path,
            snapshot_root=snapshot_root,
            state=state,
        )
    elif _path_present(snapshot_root):
        raise Mem0ResumableLaunchError("snapshot root remained after checkpoint GC")
    _verify_terminal_files(
        state=state,
        stage_path=stage_path,
        artifact_path=artifact_path,
        trace_path=trace_path,
        expected_artifact_relative=expected_runner["retrieval_artifact"],
        expected_trace_relative=expected_runner["retrieval_trace"],
    )
    if state.active_state_removed is not None or state.cleanup_closed is not None:
        if _path_present(state_path):
            raise Mem0ResumableLaunchError("working state remained after removal")
    else:
        _verify_working_state_if_present(state_path, state)
    if state.externally_ambiguous:
        raise ResumeAmbiguityError(
            f"shard {row.sample_offset} contains work beyond its sealed prefix"
        )
    if projection_repair_required:
        status = "journal_projection_repair_required"
    elif state.cleanup_closed is not None:
        if state.terminal_published is None or state.checkpoint_gc is None:
            raise Mem0ResumableLaunchError("terminal cleanup lacks publication closure")
        status = "terminal_closed"
    elif state.terminal_published is not None:
        status = (
            "terminal_gc_recovery_required"
            if state.checkpoint_gc is None
            and (not _path_present(snapshot_root) or not _path_present(stage_path))
            else "terminal_cleanup_required"
        )
    elif state.terminal_search is not None:
        status = "terminal_publication_required"
    elif state.requires_rollback:
        status = "presend_rollback_available"
    else:
        state.require_resumable()
        status = (
            "terminal_ready"
            if state.committed_prefix == state.plan.authorized_add_operations
            else "next_segment_ready"
        )
    return {
        "sample_offset": row.sample_offset,
        "status": status,
        "committed_prefix": state.committed_prefix,
        "sealed_prefix": state.sealed_prefix,
        "journal_tail_sha256": state.entries[-1]["entry_sha256"],
        "provider_calls_performed_by_replay": 0,
    }


def replay_launch(
    *,
    context: LockedLaunchContext,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    launch_manifest_path: str | Path,
    expected_launch_manifest_sha256: str,
    run_root: str | Path,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rebuild all contracts and strictly replay any present journals."""

    root = Path(run_root).resolve(strict=False)
    fixed_preflight = root / PREFLIGHT_NAME
    fixed_manifest = root / MANIFEST_NAME
    fixed_output = root / REPLAY_NAME
    if Path(preflight_path).resolve(strict=False) != fixed_preflight:
        raise Mem0ResumableLaunchError(
            "replay preflight must be the fixed run-root artifact"
        )
    if Path(launch_manifest_path).resolve(strict=False) != fixed_manifest:
        raise Mem0ResumableLaunchError(
            "replay manifest must be the fixed run-root artifact"
        )
    preflight = _verify_preflight(
        context, fixed_preflight, expected_preflight_sha256
    )
    launch = _read_expected_sealed(
        fixed_manifest,
        expected_launch_manifest_sha256,
        "launch manifest",
    )
    expected_manifest, expected_shards = materialize_launch(
        context=context,
        preflight_path=fixed_preflight,
        expected_preflight_sha256=preflight["sha256"],
        run_root=run_root,
        dry_run=True,
    )
    # Dry-run status is an execution summary, not a semantic manifest field.
    expected_manifest["status"] = "provider_free_plans_materialized"
    if launch["payload"] != expected_manifest:
        raise Mem0ResumableLaunchError(
            "sealed launch manifest is not the current reconstructed contract"
        )
    rows = launch["payload"].get("shard_manifests")
    if not isinstance(rows, list) or len(rows) != MEM0_LOCKED_NAMESPACE_COUNT:
        raise Mem0ResumableLaunchError("launch manifest shard population changed")
    statuses = []
    for manifest_row, binding, expected_payload in zip(
        rows, context.shards, expected_shards, strict=True
    ):
        if type(manifest_row) is not dict or manifest_row.get("sample_offset") != binding.sample_offset:
            raise Mem0ResumableLaunchError("launch manifest shard order changed")
        shard_path = _path_under_run_root(
            root, manifest_row.get("path"), "shard launch path"
        )
        shard_receipt = _read_expected_sealed(
            shard_path,
            manifest_row.get("artifact_sha256"),
            f"shard {binding.sample_offset} launch",
        )
        if shard_receipt["payload"] != expected_payload:
            raise Mem0ResumableLaunchError(
                f"shard {binding.sample_offset} launch contract changed"
            )
        expected_live_authority = {
            "format": RESUMABLE_LIVE_LAUNCH_AUTHORITY_FORMAT,
            "preflight_sha256": preflight["sha256"],
            "launch_manifest_sha256": launch["sha256"],
            "shard_launch_sha256": shard_receipt["sha256"],
            "shard_launch_payload_sha256": manifest_row["payload_sha256"],
            "plan_sha256": binding.plan.sha256,
            "authorization_sha256": binding.plan.authorization_sha256,
            "journal_path_sha256": _path_identity_sha256(
                root
                / _shard_relative_dir(binding.sample_offset)
                / "resume.jsonl"
            ),
            "sample_offset": binding.sample_offset,
            "namespace": binding.plan.user_scope,
            "namespace_sha256": hashlib.sha256(
                binding.plan.user_scope.encode("utf-8")
            ).hexdigest(),
            "mem0_policy_sha256": context.mem0_policy_sha256,
            "mem0_tool_implementation_sha256": (
                context.mem0_tool_implementation_sha256
            ),
            "mem0_environment_lock_sha256": (
                context.mem0_environment_lock_sha256
            ),
            "retained_transformer_token_state_bytes": 0,
        }
        statuses.append(
            _journal_status(
                root,
                expected_payload,
                binding,
                expected_live_launch_authority=expected_live_authority,
            )
        )
    payload = {
        "format": REPLAY_FORMAT,
        "status": "provider_free_replay_complete",
        "preflight_sha256": preflight["sha256"],
        "launch_manifest_sha256": launch["sha256"],
        "mem0_policy_sha256": context.mem0_policy_sha256,
        "gold_handling": {
            "references_loaded_for_source_validation": True,
            "references_persisted_in_replay_artifact": False,
            "references_exposed_to_provider": False,
        },
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "provider_call_authorization_granted": False,
        "shards": statuses,
    }
    payload = _strict_json(payload, "launch replay")
    if not dry_run:
        destination = (
            Path(output_path).resolve(strict=False)
            if output_path is not None
            else fixed_output
        )
        if destination != fixed_output:
            raise Mem0ResumableLaunchError(
                "replay receipt must use the fixed run-root artifact path"
            )
        publish_sealed_json(fixed_output, payload)
    return payload


def run_locked_live_segment(
    *,
    inputs: LockedLaunchInputs,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    launch_manifest_path: str | Path,
    expected_launch_manifest_sha256: str,
    run_root: str | Path,
    sample_offset: int,
    authorized_provider_calls: int,
) -> ResumableSegmentResult:
    """Advance exactly one shard segment from sealed launch authority.

    This is the only production-capable extraction entrypoint.  It derives
    the shard, namespace, plan, and one-use provider grant from the current
    v3 policy and the exact sealed preflight/manifest/shard artifacts.  One
    journal lease spans reconstruction, replay/rollback, every provider send,
    transport close, immutable checkpoint publication, journal sealing, and
    the final post-publication audit.
    """

    if sample_offset not in LOCKED_SAMPLE_OFFSETS:
        raise Mem0ResumableLaunchError(
            "live segment offset is not in locked validation100 order"
        )
    if isinstance(authorized_provider_calls, bool) or not isinstance(
        authorized_provider_calls, int
    ):
        raise Mem0ResumableLaunchError(
            "live segment requires an explicit integer provider-call grant"
        )
    root = Path(run_root).resolve(strict=False)
    fixed_preflight = root / PREFLIGHT_NAME
    fixed_manifest = root / MANIFEST_NAME
    if Path(preflight_path).resolve(strict=False) != fixed_preflight:
        raise Mem0ResumableLaunchError(
            "live preflight must be the fixed run-root artifact"
        )
    if Path(launch_manifest_path).resolve(strict=False) != fixed_manifest:
        raise Mem0ResumableLaunchError(
            "live manifest must be the fixed run-root artifact"
        )
    journal_path = root / _shard_relative_dir(sample_offset) / "resume.jsonl"
    lease = JournalLease(journal_path)
    with lease:
        # Reconstruct from mutable sources only after the exclusive namespace
        # lease is held; no caller-supplied shard/plan can enter this surface.
        context = load_locked_launch_context(inputs)
        preflight = _verify_preflight(
            context, fixed_preflight, expected_preflight_sha256
        )
        manifest = _read_expected_sealed(
            fixed_manifest,
            expected_launch_manifest_sha256,
            "launch manifest",
        )
        replay_launch(
            context=context,
            preflight_path=fixed_preflight,
            expected_preflight_sha256=preflight["sha256"],
            launch_manifest_path=fixed_manifest,
            expected_launch_manifest_sha256=manifest["sha256"],
            run_root=root,
            dry_run=True,
        )
        binding = next(
            row for row in context.shards if row.sample_offset == sample_offset
        )
        manifest_rows = manifest["payload"].get("shard_manifests")
        if not isinstance(manifest_rows, list):  # pragma: no cover - replay checked.
            raise Mem0ResumableLaunchError("launch shard manifest is invalid")
        manifest_row = next(
            (
                row
                for row in manifest_rows
                if isinstance(row, dict)
                and row.get("sample_offset") == sample_offset
            ),
            None,
        )
        if manifest_row is None:
            raise Mem0ResumableLaunchError("live shard is absent from launch manifest")
        shard_launch_path = _path_under_run_root(
            root, manifest_row.get("path"), "live shard launch path"
        )
        shard_launch = _read_expected_sealed(
            shard_launch_path,
            manifest_row.get("artifact_sha256"),
            f"shard {sample_offset} launch",
        )
        expected_shard_payload = build_shard_payload(
            context, binding, preflight_sha256=preflight["sha256"]
        )
        if shard_launch["payload"] != expected_shard_payload:
            raise Mem0ResumableLaunchError("live shard launch payload changed")
        if manifest_row.get("payload_sha256") != canonical_json_sha256(
            expected_shard_payload
        ):
            raise Mem0ResumableLaunchError("live shard payload digest changed")

        # Rebuild the raw population and policy authorization under the same
        # lease.  Context bindings are receipts, never executable input.
        raw_shards = build_raw_stress_shards(
            benchmark_file=inputs.benchmark_file,
            split_manifest=inputs.split_manifest,
            sample_offsets=LOCKED_SAMPLE_OFFSETS,
            target_tokens=LOCKED_TARGET_TOKENS,
            max_questions=LOCKED_QUESTIONS_PER_SHARD,
        )
        policy = load_mem0_comparison_policy(
            inputs.mem0_policy_manifest,
            source_plan=context.source_plan,
            mem0_environment_lock=inputs.mem0_environment_lock,
            expected_shards=raw_shards,
            tool_root=inputs.tool_root.resolve(),
        )
        shard = next(
            row for row in raw_shards if row.sample_offset == sample_offset
        )
        authorization = policy.retrieval_authorization(shard)
        if _binding_from_policy(policy, shard) != binding:
            raise Mem0ResumableLaunchError(
                "live policy authorization differs from sealed shard plan"
            )
        plan = binding.plan
        paths = expected_shard_payload["paths"]
        if paths.get("journal_run_root_relative") != (
            f"{_shard_relative_dir(sample_offset)}/resume.jsonl"
        ):
            raise Mem0ResumableLaunchError("live journal path changed")
        runner_paths = paths.get("runner_paths_relative_to_journal_parent")
        expected_runner_paths = {
            "owned_state": "owned-state",
            "snapshot_root": "snapshots",
            "terminal_stage": "terminal-stage.json",
            "retrieval_artifact": "retrieval.json",
            "retrieval_trace": "retrieval.trace.json",
        }
        if runner_paths != expected_runner_paths:
            raise Mem0ResumableLaunchError("live runner paths changed")
        launch_authority = {
            "format": RESUMABLE_LIVE_LAUNCH_AUTHORITY_FORMAT,
            "preflight_sha256": preflight["sha256"],
            "launch_manifest_sha256": manifest["sha256"],
            "shard_launch_sha256": shard_launch["sha256"],
            "shard_launch_payload_sha256": manifest_row["payload_sha256"],
            "plan_sha256": plan.sha256,
            "authorization_sha256": plan.authorization_sha256,
            "journal_path_sha256": _path_identity_sha256(journal_path),
            "sample_offset": sample_offset,
            "namespace": plan.user_scope,
            "namespace_sha256": hashlib.sha256(
                plan.user_scope.encode("utf-8")
            ).hexdigest(),
            "mem0_policy_sha256": context.mem0_policy_sha256,
            "mem0_tool_implementation_sha256": (
                context.mem0_tool_implementation_sha256
            ),
            "mem0_environment_lock_sha256": (
                context.mem0_environment_lock_sha256
            ),
            "retained_transformer_token_state_bytes": 0,
        }
        journal = AppendOnlyResumeJournal(journal_path, plan)
        state = _prepare_locked_ingest_state(
            journal=journal,
            owned_state_relative=runner_paths["owned_state"],
            snapshot_root_relative=runner_paths["snapshot_root"],
        )
        issuer = _OneUseSegmentAuthorizationIssuer(
            plan=plan,
            journal_path=journal_path,
            lease=lease,
            live_launch_authority=launch_authority,
        )
        grant = issuer.issue(
            state=state,
            authorized_provider_calls=authorized_provider_calls,
        )
        result = _run_resumable_ingest_segment_locked(
            shard=shard,
            authorization=authorization,
            plan=plan,
            journal_path=journal_path,
            owned_state_relative=runner_paths["owned_state"],
            snapshot_root_relative=runner_paths["snapshot_root"],
            segment_adds=DEFAULT_SEGMENT_ADDS,
            journal=journal,
            state=state,
            segment_authorization=grant,
        )

        # Provider work is not declared successful until every mutable source
        # and sealed launch artifact is re-authenticated and the newly sealed
        # journal passes the provider-free launch replay.
        policy.recheck()
        recheck_locked_launch_inputs(inputs, context)
        _verify_preflight(context, fixed_preflight, preflight["sha256"])
        final_manifest = _read_expected_sealed(
            fixed_manifest, manifest["sha256"], "launch manifest"
        )
        final_shard = _read_expected_sealed(
            shard_launch_path,
            shard_launch["sha256"],
            f"shard {sample_offset} launch",
        )
        if final_manifest != manifest or final_shard != shard_launch:
            raise Mem0ResumableLaunchError(
                "sealed launch authority changed during live segment"
            )
        replay_launch(
            context=context,
            preflight_path=fixed_preflight,
            expected_preflight_sha256=preflight["sha256"],
            launch_manifest_path=fixed_manifest,
            expected_launch_manifest_sha256=manifest["sha256"],
            run_root=root,
            dry_run=True,
        )
        return result


__all__ = [
    "ANSWER_COMPLETE_REQUEST_TOKEN_CAP",
    "ANSWER_OUTPUT_TOKEN_RESERVE",
    "ANSWER_PROMPT_TOKEN_CAP",
    "JUDGE_COMPLETE_REQUEST_TOKEN_CAP",
    "JUDGE_OUTPUT_TOKEN_RESERVE",
    "JUDGE_PROMPT_TOKEN_CAP",
    "LOCKED_ADD_COUNTS",
    "LOCKED_SAMPLE_OFFSETS",
    "LockedLaunchContext",
    "LockedLaunchInputs",
    "MANIFEST_FORMAT",
    "MANIFEST_NAME",
    "Mem0ResumableLaunchError",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "PROSPECTIVE_PROVIDER_CALLS",
    "REPLAY_FORMAT",
    "REPLAY_NAME",
    "SHARD_FORMAT",
    "SHARD_LAUNCH_NAME",
    "ShardLaunchBinding",
    "WRITE_METERING_MISSING_FIELDS",
    "build_preflight_payload",
    "build_shard_payload",
    "load_locked_launch_context",
    "materialize_launch",
    "recheck_locked_launch_inputs",
    "replay_launch",
    "run_locked_live_segment",
]
