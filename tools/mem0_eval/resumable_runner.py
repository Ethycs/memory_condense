"""One-segment-at-a-time exact Mem0 shard execution.

Each invocation either advances one bounded ingestion segment and publishes an
immutable checkpoint, or (at the full prefix) performs provider-free search.
It never retries an uncovered HTTP send.  The historical adapter owns normal
window/ledger behavior; this coordinator only rehydrates its sealed prefix and
meters the real suffix.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .policy import canonical_json_sha256 as policy_json_sha256
from .production_binding import ProductionBindingError
from .protocol import RawStressShard, shard_receipt, validate_raw_stress_shard
from .resumable import (
    AppendOnlyResumeJournal,
    JournalLease,
    RESUME_CHECKPOINT_GC_FORMAT,
    RESUME_CLEANUP_FORMAT,
    RESUME_PUBLICATION_FORMAT,
    RESUME_STATE_REMOVAL_FORMAT,
    RESUME_TERMINAL_FORMAT,
    ReplayState,
    ResumeAmbiguityError,
    ResumePlan,
    ResumableShardError,
    append_active_state_removed,
    append_checkpoint_gc,
    append_cleanup_closed,
    append_commit,
    append_intent,
    append_prefix_sealed,
    append_presend_rollback,
    append_send_attempt,
    append_terminal_published,
    append_terminal_search,
    canonical_json_sha256,
    checkpoint_authority_from_replay,
    create_immutable_state_snapshot,
    prefix_close_receipt,
    prefix_zero_restore_receipt,
    publish_sealed_json,
    read_sealed_json,
    reconcile_published_checkpoint,
    rehydration_material,
    remove_verified_snapshot_root,
    restore_snapshot_to_fresh_state,
    _path_identity_sha256,
    _safe_relative,
    state_tree_receipt,
    suffix_counter_seed,
    validate_write_usage_attestation,
    verify_completed_add_receipts,
    verify_immutable_state_snapshot,
)
from .resumable_runtime import (
    Mem0WriteActivityMeter,
    ResumableExactMem0AdapterFactory,
    suspend_resumable_adapter,
)
from .run_shard import (
    INPUT_ORDER_PROTOCOL,
    MEM0_ATTRIBUTION_KIND,
    MEM0_CERTIFIED_RENDERING,
    MEM0_OFFICIAL_THRESHOLD,
    MEM0_OFFICIAL_TOP_K,
    MEM0_PROVIDER_USAGE_STATUS,
    RETRIEVAL_ARTIFACT_FORMAT,
    RETRIEVAL_TRACE_FORMAT,
    LogicalExtractionCallMeter,
    RetrievalStageResult,
    RetrievalStageAuthorization,
    _atomic_create_payloads,
    _candidate_payload,
    _default_prompt_packer,
    _environment_lock_snapshot,
    _pack_payload,
    _render_json_bytes,
    _result_protocol_identity,
    _runtime_identity,
    _stats_snapshot,
    _validate_retrieval_authorization,
    _validated_model_identities,
    build_adapter_prepared_corpus,
    install_memory_llm_extraction_meter,
)
from .source_compat import _memory_id, _remove_owned_state, _response_rows


RESUMABLE_SEGMENT_FORMAT = "memory-condense-mem0-resumable-segment-v2"
RESUMABLE_TERMINAL_RESULT_FORMAT = (
    "memory-condense-mem0-resumable-terminal-result-v2"
)
RESUMABLE_TERMINAL_TRACE_FORMAT = (
    "memory-condense-mem0-resumable-terminal-trace-v2"
)
RESUMABLE_EXECUTION_BINDING_KIND = "exact_mem0_resumable_execution_v2"
DEFAULT_SEGMENT_ADDS = 256
RESUMABLE_SEGMENT_AUTHORIZATION_FORMAT = (
    "memory-condense-mem0-one-use-segment-authorization-v1"
)
RESUMABLE_LIVE_LAUNCH_AUTHORITY_FORMAT = (
    "memory-condense-mem0-live-launch-authority-v1"
)
RESUMABLE_WRITE_USAGE_FORMAT = (
    "memory-condense-mem0-complete-write-usage-attestation-v1"
)
_SEGMENT_AUTHORIZATION_CONSTRUCTOR = object()


def _validate_live_launch_authority(
    value: Mapping[str, Any],
    *,
    plan: ResumePlan,
    journal_path: str | os.PathLike[str],
) -> Mapping[str, Any]:
    try:
        row = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResumableShardError("live launch authority is not strict JSON") from exc
    expected_keys = {
        "format",
        "preflight_sha256",
        "launch_manifest_sha256",
        "shard_launch_sha256",
        "shard_launch_payload_sha256",
        "plan_sha256",
        "authorization_sha256",
        "journal_path_sha256",
        "sample_offset",
        "namespace",
        "namespace_sha256",
        "mem0_policy_sha256",
        "mem0_tool_implementation_sha256",
        "mem0_environment_lock_sha256",
        "retained_transformer_token_state_bytes",
    }
    if not isinstance(row, dict) or set(row) != expected_keys:
        raise ResumableShardError("live launch authority fields changed")
    exact = {
        "format": RESUMABLE_LIVE_LAUNCH_AUTHORITY_FORMAT,
        "plan_sha256": plan.sha256,
        "authorization_sha256": plan.authorization_sha256,
        "journal_path_sha256": _path_identity_sha256(journal_path),
        "sample_offset": plan.sample_offset,
        "namespace": plan.user_scope,
        "namespace_sha256": hashlib.sha256(
            plan.user_scope.encode("utf-8")
        ).hexdigest(),
        "mem0_policy_sha256": plan.mem0_policy_sha256,
        "mem0_tool_implementation_sha256": (
            plan.mem0_tool_implementation_sha256
        ),
        "mem0_environment_lock_sha256": plan.mem0_environment_lock_sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    if any(row.get(field) != expected for field, expected in exact.items()):
        raise ResumableShardError("live launch authority binding changed")
    for field in (
        "preflight_sha256",
        "launch_manifest_sha256",
        "shard_launch_sha256",
        "shard_launch_payload_sha256",
    ):
        value = row.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ResumableShardError(f"live launch authority {field} is invalid")
    return MappingProxyType(row)


class _OneUseSegmentAuthorization:
    """Unforgeable-in-process capability consumed by one locked segment."""

    __slots__ = (
        "_body",
        "_consumed",
        "_journal_path",
        "_lease",
        "_lock",
        "_receipt_sha256",
    )

    def __init__(
        self,
        constructor: object,
        body: Mapping[str, Any],
        *,
        journal_path: str | os.PathLike[str],
        lease: JournalLease,
    ) -> None:
        if constructor is not _SEGMENT_AUTHORIZATION_CONSTRUCTOR:
            raise ResumableShardError(
                "segment authorization must be issued by the code-owned issuer"
            )
        self._body = MappingProxyType(dict(body))
        self._receipt_sha256 = canonical_json_sha256(body)
        self._consumed = False
        self._journal_path = Path(journal_path)
        self._lease = lease
        self._lock = threading.Lock()

    def consume(
        self,
        *,
        state: ReplayState,
        journal_path: str | os.PathLike[str],
        segment_adds: int,
    ) -> Mapping[str, Any]:
        with self._lock:
            if self._consumed:
                raise ResumableShardError(
                    "segment authorization was already consumed"
                )
            if (
                Path(journal_path) != self._journal_path
                or not self._lease.held_for(self._journal_path)
            ):
                raise ResumableShardError(
                    "segment authorization lost its journal lease"
                )
            plan = state.plan
            prefix_before = state.committed_prefix
            launch_authority = dict(self._body.get("live_launch_authority", {}))
            expected = {
                "format": RESUMABLE_SEGMENT_AUTHORIZATION_FORMAT,
                "plan_sha256": plan.sha256,
                "authorization_sha256": plan.authorization_sha256,
                "journal_path_sha256": _path_identity_sha256(Path(journal_path)),
                "prefix_before": prefix_before,
                "prefix_after": prefix_before + segment_adds,
                "generation": state.generation,
                "prior_checkpoint_authority_sha256": (
                    state.checkpoint_authority_sha256
                ),
                "authorized_provider_calls": segment_adds,
                "authorized_add_operations": segment_adds,
                "provider_retries": 0,
                "namespace": plan.user_scope,
                "retained_transformer_token_state_bytes": 0,
                "live_launch_authority": launch_authority,
                "live_launch_authority_sha256": canonical_json_sha256(
                    launch_authority
                ),
            }
            if dict(self._body) != expected:
                raise ResumableShardError("segment authorization binding changed")
            self._consumed = True
            return MappingProxyType(
                {**expected, "receipt_sha256": self._receipt_sha256}
            )


class _OneUseSegmentAuthorizationIssuer:
    """Issue once, and only while the exact journal lease is held."""

    __slots__ = (
        "_issued",
        "_journal_path",
        "_launch_authority",
        "_lease",
        "_lock",
        "_plan",
    )

    def __init__(
        self,
        *,
        plan: ResumePlan,
        journal_path: str | os.PathLike[str],
        lease: JournalLease,
        live_launch_authority: Mapping[str, Any],
    ) -> None:
        self._plan = plan
        self._journal_path = Path(journal_path)
        self._lease = lease
        self._launch_authority = _validate_live_launch_authority(
            live_launch_authority,
            plan=plan,
            journal_path=self._journal_path,
        )
        self._issued = False
        self._lock = threading.Lock()

    def issue(
        self,
        *,
        state: ReplayState,
        authorized_provider_calls: int,
    ) -> _OneUseSegmentAuthorization:
        with self._lock:
            if self._issued:
                raise ResumableShardError("segment issuer is one-use")
            if not self._lease.held_for(self._journal_path):
                raise ResumableShardError(
                    "segment authorization requires the live journal lease"
                )
            state.require_resumable()
            if state.plan != self._plan:
                raise ResumableShardError("segment issuer plan changed")
            prefix = state.committed_prefix
            remaining = self._plan.authorized_add_operations - prefix
            expected = min(DEFAULT_SEGMENT_ADDS, remaining)
            if (
                isinstance(authorized_provider_calls, bool)
                or not isinstance(authorized_provider_calls, int)
                or authorized_provider_calls != expected
                or expected <= 0
            ):
                raise ResumableShardError(
                    "explicit provider-call authorization must equal the exact next segment"
                )
            body = {
                "format": RESUMABLE_SEGMENT_AUTHORIZATION_FORMAT,
                "plan_sha256": self._plan.sha256,
                "authorization_sha256": self._plan.authorization_sha256,
                "journal_path_sha256": _path_identity_sha256(self._journal_path),
                "prefix_before": prefix,
                "prefix_after": prefix + expected,
                "generation": state.generation,
                "prior_checkpoint_authority_sha256": (
                    state.checkpoint_authority_sha256
                ),
                "authorized_provider_calls": expected,
                "authorized_add_operations": expected,
                "provider_retries": 0,
                "namespace": self._plan.user_scope,
                "retained_transformer_token_state_bytes": 0,
                "live_launch_authority": dict(self._launch_authority),
                "live_launch_authority_sha256": canonical_json_sha256(
                    dict(self._launch_authority)
                ),
            }
            self._issued = True
            return _OneUseSegmentAuthorization(
                _SEGMENT_AUTHORIZATION_CONSTRUCTOR,
                body,
                journal_path=self._journal_path,
                lease=self._lease,
            )


@dataclass(frozen=True, slots=True)
class ResumableSegmentResult:
    action: str
    prefix_before: int
    prefix_after: int
    segment_adds: int
    checkpoint_authority_sha256: str
    journal_tail_sha256: str
    state_tree_sha256: str | None
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ResumableTerminalResult:
    action: str
    artifact_path: Path
    trace_path: Path
    artifact_sha256: str
    artifact_bytes: int
    trace_sha256: str
    trace_bytes: int
    journal_tail_sha256: str
    checkpoint_authority_sha256: str
    receipt_sha256: str


def _source_ref_dict(ref: Any) -> dict[str, Any]:
    return {
        "sample_id": ref.sample_id,
        "source": ref.source,
        "session": ref.session,
        "session_index": ref.session_index,
        "original_session_index": ref.original_session_index,
        "batch_index": ref.batch_index,
        "date": ref.date,
        "turn_start": ref.turn_start,
        "turn_count": ref.turn_count,
        "roles": list(ref.roles),
    }


def prepared_batch_sha256(batch: Any) -> str:
    """Bind one literal ordered provider add without retaining its text."""

    return canonical_json_sha256(
        {
            "source_ref": _source_ref_dict(batch.ref),
            "messages": [list(message) for message in batch.messages],
        }
    )


def build_resume_plan(
    *,
    shard: RawStressShard,
    authorization: RetrievalStageAuthorization,
    authorization_sha256: str,
) -> ResumePlan:
    corpus = build_adapter_prepared_corpus(shard)
    return ResumePlan(
        authorization_sha256=authorization_sha256,
        mem0_policy_sha256=authorization.mem0_policy_sha256,
        source_validation_policy_sha256=(
            authorization.source_validation_policy_sha256
        ),
        source_implementation_sha256=authorization.source_implementation_sha256,
        source_environment_lock_sha256=(
            authorization.source_environment_lock_sha256
        ),
        mem0_tool_implementation_sha256=(
            authorization.mem0_tool_implementation_sha256
        ),
        mem0_environment_lock_sha256=authorization.mem0_environment_lock_sha256,
        sample_offset=shard.sample_offset,
        sample_sha256=shard.sample_sha256,
        raw_history_bundle_sha256=shard.raw_history_bundle_sha256,
        ordered_batch_sha256s=tuple(
            prepared_batch_sha256(batch) for batch in corpus.batches
        ),
        authorized_add_operations=authorization.authorized_add_operations,
        authorized_extraction_calls=authorization.authorized_extraction_calls,
        authorized_search_operations=authorization.authorized_search_operations,
        user_scope=(
            "longmemeval:resumable:"
            f"{authorization_sha256[:32]}"
        ),
    )


def _validate_plan_against_inputs(
    state: ReplayState,
    *,
    shard: RawStressShard,
    authorization: RetrievalStageAuthorization,
    batch_sha256s: Sequence[str],
) -> None:
    validate_raw_stress_shard(shard)
    plan = state.plan
    pairs = {
        "sample_offset": (plan.sample_offset, shard.sample_offset),
        "sample_sha256": (plan.sample_sha256, shard.sample_sha256),
        "raw_history_bundle_sha256": (
            plan.raw_history_bundle_sha256,
            shard.raw_history_bundle_sha256,
        ),
        "authorized_add_operations": (
            plan.authorized_add_operations,
            authorization.authorized_add_operations,
        ),
        "authorized_extraction_calls": (
            plan.authorized_extraction_calls,
            authorization.authorized_extraction_calls,
        ),
        "authorized_search_operations": (
            plan.authorized_search_operations,
            authorization.authorized_search_operations,
        ),
        "mem0_policy_sha256": (
            plan.mem0_policy_sha256,
            authorization.mem0_policy_sha256,
        ),
        "mem0_tool_implementation_sha256": (
            plan.mem0_tool_implementation_sha256,
            authorization.mem0_tool_implementation_sha256,
        ),
        "mem0_environment_lock_sha256": (
            plan.mem0_environment_lock_sha256,
            authorization.mem0_environment_lock_sha256,
        ),
    }
    changed = [label for label, values in pairs.items() if values[0] != values[1]]
    if changed:
        raise ResumableShardError(f"resumable input identities changed: {changed}")
    verify_completed_add_receipts(
        state, ordered_batch_sha256s=tuple(batch_sha256s)
    )


def _http_request_sha256(request: Any) -> str:
    method = getattr(request, "method", None)
    url = getattr(request, "url", None)
    content: Any
    try:
        content = request.content
    except BaseException:
        reader = getattr(request, "read", None)
        if not callable(reader):
            raise ProductionBindingError("HTTP request body cannot be hashed")
        content = reader()
    if not isinstance(method, str) or url is None or not isinstance(
        content, (bytes, bytearray)
    ):
        raise ProductionBindingError("HTTP request identity is incomplete")
    return canonical_json_sha256(
        {
            "method": method.upper(),
            "url": str(url),
            "body_sha256": hashlib.sha256(bytes(content)).hexdigest(),
            "body_bytes": len(content),
        }
    )


class DurableHTTPJournalBoundary:
    """Arm exactly one intent, then mark it inside the HTTP cap lock."""

    def __init__(self, journal: AppendOnlyResumeJournal) -> None:
        self._journal = journal
        self._state: ReplayState | None = None
        self._ordinal: int | None = None
        self._lock = threading.Lock()

    def arm(self, state: ReplayState, *, ordinal: int) -> None:
        with self._lock:
            if self._state is not None:
                raise ResumableShardError("HTTP send boundary is already armed")
            if state.pending_intent is None or state.pending_send_attempt is not None:
                raise ResumableShardError("HTTP boundary requires a fresh intent")
            if state.pending_intent["ordinal"] != ordinal:
                raise ResumableShardError("HTTP boundary ordinal mismatch")
            self._state = state
            self._ordinal = ordinal

    def before_http_send(self, request: Any) -> None:
        with self._lock:
            state = self._state
            if state is None or self._ordinal is None:
                raise ResumableShardError("unarmed HTTP send was rejected")
            if state.pending_send_attempt is not None:
                raise ResumeAmbiguityError("a second HTTP send was attempted for one add")
            self._state = state = append_send_attempt(
                self._journal,
                state,
                request_sha256=_http_request_sha256(request),
            )

    def consume_after_response(self) -> ReplayState:
        with self._lock:
            state = self._state
            if state is None or state.pending_send_attempt is None:
                raise ResumableShardError(
                    "Mem0 add returned without one durable HTTP send marker"
                )
            self._state = None
            self._ordinal = None
            return state

    @property
    def state(self) -> ReplayState | None:
        return self._state


def _stats_dict(stats: Any) -> dict[str, Any]:
    fields = getattr(type(stats), "__dataclass_fields__", None)
    if not isinstance(fields, Mapping):
        raise ResumableShardError("Mem0 adapter stats type changed")
    return {name: getattr(stats, name) for name in fields}


def _set_adapter_stats_from_state(adapter: Any, state: ReplayState) -> None:
    from memory_condense.eval.mem0_adapter import Mem0AdapterStats

    material = rehydration_material(state)
    values = material["adapter_stats"]
    if values is not None:
        fields = Mem0AdapterStats.__dataclass_fields__
        adapter._stats = Mem0AdapterStats(
            **{name: value for name, value in values.items() if fields[name].init}
        )
        if _stats_dict(adapter._stats) != values:
            raise ResumableShardError(
                "source-layout derived adapter stats differ from checkpoint"
            )
    else:
        adapter._stats = Mem0AdapterStats(
            token_counter_identity=adapter.stats.token_counter_identity,
            token_counter_identity_verified=(
                adapter.stats.token_counter_identity_verified
            ),
        )


def _cumulative_receipt(
    receipt: Mapping[str, Any],
    *,
    prefix: int,
    full_authorized: int,
    logical: bool,
    prior_cumulative: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = json.loads(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    segment_attempted = int(row.get("attempted", -1))
    segment_completed = int(row.get("completed", -1))
    segment_failed = int(row.get("failed", -1))
    segment_rejected = int(row.get("rejected", -1))
    body = {
        "kind": (
            "resumable_cumulative_logical_extraction"
            if logical
            else "resumable_cumulative_http_transport"
        ),
        "authorized": full_authorized,
        "seeded_prefix": prefix,
        "segment_authorized": row.get("authorized"),
        "attempted": prefix + segment_attempted,
        "completed": prefix + segment_completed,
        "failed": segment_failed,
        "rejected": segment_rejected,
        # Retain the complete observed segment receipt, not only a digest.  A
        # checkpoint must prove the exact model/route/cap boundary (and the
        # logical infer supervisor), while the digest provides a compact
        # binding for receipts that consume this cumulative projection.
        "segment_receipt": row,
        "segment_receipt_sha256": policy_json_sha256(row),
        "retries_authorized": 0,
    }
    if logical:
        body.update(
            infer_true_adds_started=(
                prefix + int(row.get("infer_true_adds_started", -1))
            ),
            infer_true_adds_exactly_one_call=(
                prefix
                + int(row.get("infer_true_adds_exactly_one_call", -1))
            ),
        )
    else:
        usage_fields = (
            "provider_usage_records",
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_total_tokens",
        )
        for field in usage_fields:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ResumableShardError(
                    f"segment transport {field} is incomplete"
                )
        latency = row.get("provider_latency_s")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or float(latency) < 0
        ):
            raise ResumableShardError(
                "segment transport provider latency is incomplete"
            )
        if (
            row["provider_usage_records"] != segment_completed
            or row["provider_total_tokens"]
            != row["provider_input_tokens"] + row["provider_output_tokens"]
            or row.get("provider_usage_status") != "provider_reported_exact"
        ):
            raise ResumableShardError(
                "segment transport provider usage does not close"
            )
        prior = dict(prior_cumulative or {})
        if prefix:
            if not prior:
                raise ResumableShardError(
                    "resumed transport usage omitted its sealed prefix"
                )
            prior_records = int(prior.get("provider_usage_records", -1))
            if prior_records != prefix:
                raise ResumableShardError(
                    "resumed transport provider-usage prefix changed"
                )
        else:
            prior_records = 0
        prior_input = int(prior.get("provider_input_tokens", 0))
        prior_output = int(prior.get("provider_output_tokens", 0))
        prior_total = int(prior.get("provider_total_tokens", 0))
        prior_latency = float(prior.get("provider_latency_s", 0.0))
        if any(value < 0 for value in (prior_input, prior_output, prior_total)) or (
            prior_total != prior_input + prior_output
        ) or prior_latency < 0:
            raise ResumableShardError(
                "resumed transport provider usage is invalid"
            )
        body.update(
            provider_usage_status="provider_reported_exact",
            provider_usage_records=prior_records + row["provider_usage_records"],
            provider_input_tokens=prior_input + row["provider_input_tokens"],
            provider_output_tokens=prior_output + row["provider_output_tokens"],
            provider_total_tokens=prior_total + row["provider_total_tokens"],
            provider_latency_s=prior_latency + float(row["provider_latency_s"]),
        )
    return body


def _complete_write_usage_attestation(
    *,
    state: ReplayState,
    segment_authorization_receipt: Mapping[str, Any],
    segment_write_activity_receipt: Mapping[str, Any],
    suspended: Mapping[str, Any],
) -> dict[str, Any]:
    authorization_receipt = dict(segment_authorization_receipt)
    authorization = dict(authorization_receipt)
    authorization_sha = authorization.pop("receipt_sha256", None)
    if authorization_sha != canonical_json_sha256(authorization):
        raise ResumableShardError("segment authorization receipt changed")
    activity_receipt = dict(segment_write_activity_receipt)
    activity = dict(activity_receipt)
    activity_sha = activity.pop("receipt_sha256", None)
    if activity_sha != canonical_json_sha256(activity):
        raise ResumableShardError("segment write-activity receipt changed")
    if (
        activity.get("wrappers_installed") is not True
        or activity.get("wrappers_restored") is not True
        or activity.get("embedding_attempted")
        != activity.get("embedding_completed")
        or activity.get("embedding_failed") != 0
        or activity.get("storage_attempted")
        != activity.get("storage_completed")
        or activity.get("storage_failed") != 0
    ):
        raise ResumableShardError("segment write activity did not close")
    closure = suspended.get("transport_closure")
    if not isinstance(closure, Mapping):
        raise ResumableShardError("suspension omitted transport closure")
    closure = dict(closure)
    closure_sha = closure.get("receipt_sha256")
    if closure_sha != suspended.get("transport_closure_sha256"):
        raise ResumableShardError("suspension transport-closure digest changed")
    latest = state.commits[-1]
    stats = latest["adapter_stats"]
    transport = latest["transport_receipt"]
    prior_usage = (
        state.latest_prefix_seal.get("write_usage_attestation")
        if state.latest_prefix_seal is not None
        else None
    )
    prior_observed: Mapping[str, Any] = {}
    prior_sha: str | None = None
    if prior_usage is not None:
        verified_prior = validate_write_usage_attestation(
            prior_usage,
            state=state,
            expected_committed_prefix=state.sealed_prefix,
            expected_generation=state.latest_prefix_seal["generation"],
        )
        prior_observed = verified_prior["observed"]
        prior_sha = verified_prior["receipt_sha256"]
    tree = suspended.get("owned_state_tree")
    if not isinstance(tree, Mapping):
        raise ResumableShardError("suspension omitted owned state tree")
    storage_bytes = tree.get("total_bytes")
    persisted_count = suspended.get("namespace_persisted_memory_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (storage_bytes, persisted_count)
    ):
        raise ResumableShardError("suspension write-state counts are invalid")
    observed = {
        "add_attempted": stats["add_attempted_calls"],
        "add_completed": stats["add_completed_calls"],
        "add_failed": stats["add_failed_calls"],
        "extraction_attempted": transport["attempted"],
        "extraction_completed": transport["completed"],
        "extraction_failed": transport["failed"],
        "extraction_raw_message_token_proxy": stats["add_raw_message_tokens"],
        "extraction_provider_input_tokens": transport[
            "provider_input_tokens"
        ],
        "extraction_provider_output_tokens": transport[
            "provider_output_tokens"
        ],
        "extraction_usage_status": transport["provider_usage_status"],
        "embedding_operations": int(
            prior_observed.get("embedding_operations", 0)
        )
        + int(activity["embedding_completed"]),
        "embedding_input_token_proxy": int(
            prior_observed.get("embedding_input_token_proxy", 0)
        )
        + int(activity["embedding_input_token_proxy"]),
        "returned_memory_count": stats["add_returned_memories"],
        "persisted_memory_count": persisted_count,
        "persisted_storage_bytes": storage_bytes,
        "add_latency_s": float(stats["add_latency_s"]),
        "extraction_latency_s": float(transport["provider_latency_s"]),
        "embedding_latency_s": float(
            prior_observed.get("embedding_latency_s", 0.0)
        )
        + float(activity["embedding_latency_s"]),
        "storage_latency_s": float(
            prior_observed.get("storage_latency_s", 0.0)
        )
        + float(activity["storage_latency_s"]),
    }
    body = {
        "format": RESUMABLE_WRITE_USAGE_FORMAT,
        "plan_sha256": state.plan.sha256,
        "authorization_sha256": state.plan.authorization_sha256,
        "generation": state.generation,
        "committed_prefix": state.committed_prefix,
        "prior_write_usage_attestation_sha256": prior_sha,
        "segment_authorization_receipt": authorization_receipt,
        "segment_authorization_receipt_sha256": authorization_sha,
        "segment_write_activity_receipt": activity_receipt,
        "segment_write_activity_receipt_sha256": activity_sha,
        "transport_closure_receipt_sha256": closure_sha,
        "observed": observed,
        "observed_sha256": canonical_json_sha256(observed),
        "retained_transformer_token_state_bytes": 0,
    }
    attestation = {**body, "receipt_sha256": canonical_json_sha256(body)}
    return validate_write_usage_attestation(
        attestation,
        state=state,
        expected_storage_bytes=storage_bytes,
    )


class _SegmentAddController:
    def __init__(
        self,
        *,
        adapter: Any,
        real_add: Any,
        batches: Sequence[Any],
        primer_count: int,
        prefix: int,
        journal: AppendOnlyResumeJournal,
        state: ReplayState,
        boundary: DurableHTTPJournalBoundary,
        meter: LogicalExtractionCallMeter,
        factory: ResumableExactMem0AdapterFactory,
    ) -> None:
        self.adapter = adapter
        self.real_add = real_add
        self.batches = tuple(batches)
        self.primer_count = primer_count
        self.prefix = prefix
        self.journal = journal
        self.state = state
        self.boundary = boundary
        self.meter = meter
        self.factory = factory
        self.sealed_transport_receipt = (
            dict(state.commits[-1]["transport_receipt"])
            if prefix and state.commits
            else None
        )
        self.calls = 0
        self.real_calls = 0
        self.pending: dict[str, Any] | None = None
        self._stats_reset = False

    def _reset_sealed_adapter_stats(self) -> None:
        if self._stats_reset:
            return
        _set_adapter_stats_from_state(self.adapter, self.state)
        self._stats_reset = True

    def finalize_pending(self) -> None:
        pending = self.pending
        if pending is None:
            return
        after = _stats_dict(self.adapter.stats)
        before = pending["stats_before"]
        latency = float(after["add_latency_s"]) - float(before["add_latency_s"])
        raw_tokens = int(after["add_raw_message_tokens"]) - int(
            before["add_raw_message_tokens"]
        )
        logical = _cumulative_receipt(
            self.meter.receipt(),
            prefix=self.prefix,
            full_authorized=self.state.plan.authorized_extraction_calls,
            logical=True,
        )
        transport = _cumulative_receipt(
            self.factory.transport_receipt(),
            prefix=self.prefix,
            full_authorized=self.state.plan.authorized_extraction_calls,
            logical=False,
            prior_cumulative=self.sealed_transport_receipt,
        )
        material = rehydration_material(self.state)
        source = pending["source_ref"]
        prior_unique: list[dict[str, Any]] = []
        for ref in material["request_window_deque"]:
            if ref not in prior_unique:
                prior_unique.append(ref)
        window = list(prior_unique)
        if source not in window:
            window.append(source)
        self.state = append_commit(
            self.journal,
            self.state,
            response_sha256=pending["response_sha256"],
            returned_memory_ids=pending["returned_memory_ids"],
            source_ref=source,
            request_window_refs=window,
            adapter_stats=after,
            logical_meter_receipt=logical,
            transport_receipt=transport,
            add_latency_s=latency,
            raw_message_tokens=raw_tokens,
            scope_protocol=True,
        )
        self.pending = None

    def __call__(self, messages: Any, **kwargs: Any) -> Any:
        self.finalize_pending()
        if self.calls >= len(self.batches):
            raise ResumableShardError("adapter exceeded the prepared segment")
        batch = self.batches[self.calls]
        self.calls += 1
        expected_messages = [
            {"role": role, "content": content} for role, content in batch.messages
        ]
        if messages != expected_messages:
            raise ResumableShardError("Mem0 add messages differ from locked corpus")
        if kwargs != {"user_id": self.state.plan.user_scope, "infer": True}:
            raise ResumableShardError("Mem0 add scope/infer contract changed")
        if self.calls <= self.primer_count:
            return {"results": []}

        self._reset_sealed_adapter_stats()
        ordinal = self.prefix + self.real_calls
        self.real_calls += 1
        self.state = append_intent(
            self.journal,
            self.state,
            ordinal=ordinal,
            session_sha256=canonical_json_sha256(_source_ref_dict(batch.ref)),
        )
        self.boundary.arm(self.state, ordinal=ordinal)
        stats_before = _stats_dict(self.adapter.stats)
        response = self.real_add(messages, **kwargs)
        self.state = self.boundary.consume_after_response()

        rows = _response_rows(response, operation="add")
        ids: list[str] = []
        for row in rows:
            memory_id = _memory_id(row)
            if memory_id is None:
                raise ResumableShardError("Mem0 add result omitted memory ID")
            ids.append(memory_id)
        response_receipt = {
            "ordinal": ordinal,
            "returned_memory_ids": ids,
            "row_count": len(rows),
        }
        self.pending = {
            "response_sha256": canonical_json_sha256(response_receipt),
            "returned_memory_ids": ids,
            "source_ref": _source_ref_dict(batch.ref),
            "stats_before": stats_before,
        }
        return response


def _primer_start(batches: Sequence[Any], prefix: int) -> int:
    messages = 0
    start = prefix
    while start > 0 and messages < 10:
        start -= 1
        messages += int(batches[start].ref.turn_count)
    return start


def _preseed_adapter(adapter: Any, state: ReplayState) -> None:
    from memory_condense.eval.mem0_adapter import SourceRef

    material = rehydration_material(state)
    ledger: dict[tuple[str, str], list[Any]] = {}
    for row in material["ledger_projection"]:
        ledger[(state.plan.user_scope, row["memory_id"])] = [
            SourceRef(**{**ref, "roles": tuple(ref["roles"])})
            for ref in row["source_refs"]
        ]
    adapter._ledger = ledger
    adapter._scopes = []
    adapter._scope_protocol = {}
    adapter._active_scope = None


def _adapter_ledger_projection(adapter: Any, state: ReplayState) -> list[dict[str, Any]]:
    """Project the live historical ledger into the sealed replay schema."""

    ledger = getattr(adapter, "ledger", None)
    if not isinstance(ledger, Mapping):
        raise ResumableShardError("Mem0 adapter ledger is not a mapping")
    rows: list[dict[str, Any]] = []
    for key, refs in ledger.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] != state.plan.user_scope
            or not isinstance(key[1], str)
            or not key[1]
        ):
            raise ResumableShardError("Mem0 adapter ledger key changed")
        if not isinstance(refs, Sequence) or isinstance(
            refs, (str, bytes, bytearray)
        ):
            raise ResumableShardError("Mem0 adapter ledger refs changed")
        rows.append(
            {
                "user_scope": key[0],
                "memory_id": key[1],
                "source_refs": [_source_ref_dict(ref) for ref in refs],
            }
        )
    return sorted(rows, key=lambda row: row["memory_id"])


def _assert_live_adapter_matches_rehydration(
    adapter: Any, state: ReplayState
) -> None:
    """Prove that the mutable adapter exactly matches the journal projection."""

    material = rehydration_material(state)
    if _stats_dict(adapter.stats) != material["adapter_stats"]:
        raise ResumableShardError("adapter stats differ from sealed replay")
    if _adapter_ledger_projection(adapter, state) != material["ledger_projection"]:
        raise ResumableShardError("adapter ledger differs from sealed replay")
    scope = state.plan.user_scope
    if getattr(adapter, "active_user_scope", None) != scope:
        raise ResumableShardError("adapter active scope differs from frozen plan")
    if getattr(adapter, "_scopes", None) != [scope]:
        raise ResumableShardError("adapter registered scopes differ from frozen plan")
    if getattr(adapter, "_scope_protocol", None) != {
        scope: material["scope_protocol"]
    }:
        raise ResumableShardError("adapter scope protocol differs from sealed replay")


def _safe_retire_working_state(
    *, state_path: Path, expected_tree_sha256: str | None
) -> None:
    if not state_path.exists():
        return
    receipt = state_tree_receipt(state_path)
    if (
        expected_tree_sha256 is not None
        and receipt["snapshot_tree_sha256"] != expected_tree_sha256
    ):
        raise ResumableShardError(
            "existing working state differs from the immutable checkpoint"
        )
    marker = state_path / ".memory-condense-owned-state"
    token = marker.read_text(encoding="utf-8")
    retired = state_path.with_name(
        f".{state_path.name}.retired-{receipt['snapshot_tree_sha256'][:16]}"
    )
    if retired.exists():
        raise ResumableShardError("deterministic retired-state path already exists")
    os.rename(state_path, retired)
    _remove_owned_state(retired, token)


def _restore_working_state(
    *, journal: AppendOnlyResumeJournal, state: ReplayState, state_path: Path
) -> Mapping[str, Any] | None:
    seal = state.latest_prefix_seal
    if seal is None:
        _safe_retire_working_state(state_path=state_path, expected_tree_sha256=None)
        return None
    _safe_retire_working_state(
        state_path=state_path,
        expected_tree_sha256=seal["snapshot_tree_sha256"],
    )
    return restore_snapshot_to_fresh_state(
        snapshot_path=journal.path.parent / seal["snapshot_path"],
        destination_state_dir=state_path,
        expected_authority_sha256=seal["snapshot_authority_sha256"],
        expected_manifest_sha256=seal["snapshot_manifest_sha256"],
        expected_tree_sha256=seal["snapshot_tree_sha256"],
        expected_ownership_token_sha256=seal["ownership_token_sha256"],
    )


def _new_or_replayed_journal(
    *,
    journal: AppendOnlyResumeJournal,
    owned_state_relative: str,
    snapshot_root_relative: str,
) -> ReplayState:
    records = journal.path.with_name(journal.path.name + ".records")
    if journal.path.exists() or records.exists():
        state = journal.replay()
    else:
        state = journal.create(
            owned_state_path=owned_state_relative,
            snapshot_root_path=snapshot_root_relative,
        )
    if state.committed_prefix > state.sealed_prefix and state.pending_intent is None:
        state = reconcile_published_checkpoint(journal, state)
    return state


def _run_resumable_ingest_segment_locked(
    *,
    shard: RawStressShard,
    authorization: RetrievalStageAuthorization,
    plan: ResumePlan,
    journal_path: str | os.PathLike[str],
    owned_state_relative: str,
    snapshot_root_relative: str,
    segment_adds: int = DEFAULT_SEGMENT_ADDS,
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    segment_authorization: _OneUseSegmentAuthorization,
) -> ResumableSegmentResult:
    """Advance one suffix while the caller retains the journal lease."""

    if segment_adds != DEFAULT_SEGMENT_ADDS:
        raise ResumableShardError("production segment cadence must be exactly 256 adds")
    corpus = build_adapter_prepared_corpus(shard)
    batch_hashes = tuple(prepared_batch_sha256(batch) for batch in corpus.batches)
    _validate_plan_against_inputs(
        state,
        shard=shard,
        authorization=authorization,
        batch_sha256s=batch_hashes,
    )
    state_path = journal.path.parent / Path(owned_state_relative)
    state.require_resumable()
    prefix = state.committed_prefix
    total = plan.authorized_add_operations
    if prefix == total:
        raise ResumableShardError(
            "full prefix is ready for the separate terminal search operation"
        )
    end = min(total, prefix + segment_adds)
    exact_segment = end - prefix
    segment_authorization_receipt = segment_authorization.consume(
        state=state,
        journal_path=journal_path,
        segment_adds=exact_segment,
    )
    _restore_working_state(journal=journal, state=state, state_path=state_path)

    boundary = DurableHTTPJournalBoundary(journal)
    factory = ResumableExactMem0AdapterFactory(
        authorization,
        segment_authorized_calls=exact_segment,
        adopt_existing_state=(prefix > 0),
        user_scope=plan.user_scope,
        before_http_send=boundary.before_http_send,
        expected_ownership_token_sha256=(
            state.latest_prefix_seal["ownership_token_sha256"]
            if state.latest_prefix_seal is not None
            else None
        ),
    )
    adapter = factory(state_path)
    _preseed_adapter(adapter, state)
    meter = LogicalExtractionCallMeter(authorized=exact_segment)
    restore_meter = install_memory_llm_extraction_meter(adapter, meter)
    write_meter = Mem0WriteActivityMeter()
    restore_write_meter = write_meter.install(adapter)
    owned = adapter._backend
    real_add = owned.add
    primer_start = _primer_start(corpus.batches, prefix)
    selected = tuple(corpus.batches[primer_start:prefix]) + tuple(
        corpus.batches[prefix:end]
    )
    controller = _SegmentAddController(
        adapter=adapter,
        real_add=real_add,
        batches=selected,
        primer_count=prefix - primer_start,
        prefix=prefix,
        journal=journal,
        state=state,
        boundary=boundary,
        meter=meter,
        factory=factory,
    )
    owned.add = controller
    suspended: Mapping[str, Any] | None = None
    try:
        from memory_condense.eval.mem0_adapter import _PreparedCorpus

        selected_corpus = _PreparedCorpus(
            sample_id=corpus.sample_id,
            batches=selected,
            raw_pair_count=len(selected),
            skipped_empty_pair_count=0,
            official_longmemeval_protocol=True,
        )
        adapter._ingest_prepared(selected_corpus)
        controller.finalize_pending()
        if controller.real_calls != exact_segment:
            raise ResumableShardError("segment did not execute its exact suffix")
        state = controller.state
        meter.assert_complete()
        _assert_live_adapter_matches_rehydration(adapter, state)
        owned.add = real_add
        restore_meter()
        restore_write_meter()
        write_meter.assert_closed()
        suspended = suspend_resumable_adapter(adapter)
    except BaseException:
        try:
            owned.add = real_add
        except BaseException:
            pass
        try:
            restore_meter()
        except BaseException:
            pass
        try:
            restore_write_meter()
        except BaseException:
            pass
        try:
            suspend_resumable_adapter(adapter)
        except BaseException:
            pass
        raise
    assert suspended is not None
    write_usage = _complete_write_usage_attestation(
        state=state,
        segment_authorization_receipt=segment_authorization_receipt,
        segment_write_activity_receipt=write_meter.receipt(),
        suspended=suspended,
    )
    close = prefix_close_receipt(
        state,
        history_sqlite_closed=bool(suspended["history_sqlite_closed"]),
        qdrant_local_collections_closed=int(
            suspended["qdrant_local_collections_closed"]
        ),
        qdrant_clients_closed=int(suspended["qdrant_clients_closed"]),
        transport_closed=bool(suspended["transport_closed"]),
        transport_closure_receipt=suspended["transport_closure"],
        write_usage_attestation=write_usage,
        expected_storage_bytes=int(suspended["owned_state_tree"]["total_bytes"]),
    )
    authority = checkpoint_authority_from_replay(
        state, handles_closed_receipt=close
    )
    snapshot = create_immutable_state_snapshot(
        journal_path=journal.path,
        owned_state_dir=state_path,
        snapshot_root=journal.path.parent / Path(snapshot_root_relative),
        committed_prefix=state.committed_prefix,
        checkpoint_authority=authority,
    )
    state = append_prefix_sealed(journal, state, snapshot_receipt=snapshot)
    body = {
        "format": RESUMABLE_SEGMENT_FORMAT,
        "action": "prefix_checkpointed",
        "prefix_before": prefix,
        "prefix_after": state.committed_prefix,
        "segment_adds": exact_segment,
        "checkpoint_authority_sha256": state.checkpoint_authority_sha256,
        "journal_tail_sha256": state.entries[-1]["entry_sha256"],
        "state_tree_sha256": snapshot["snapshot_tree_sha256"],
        "factory_receipt_sha256": factory.binding_receipt()["receipt_sha256"],
        "suspend_receipt_sha256": suspended["receipt_sha256"],
        "segment_authorization_receipt_sha256": (
            segment_authorization_receipt["receipt_sha256"]
        ),
        "transport_closure_receipt_sha256": suspended[
            "transport_closure_sha256"
        ],
        "write_usage_attestation_sha256": write_usage["receipt_sha256"],
    }
    receipt_sha = canonical_json_sha256(body)
    return ResumableSegmentResult(
        action=body["action"],
        prefix_before=prefix,
        prefix_after=state.committed_prefix,
        segment_adds=exact_segment,
        checkpoint_authority_sha256=state.checkpoint_authority_sha256,
        journal_tail_sha256=state.entries[-1]["entry_sha256"],
        state_tree_sha256=snapshot["snapshot_tree_sha256"],
        receipt_sha256=receipt_sha,
    )


def _prepare_locked_ingest_state(
    *,
    journal: AppendOnlyResumeJournal,
    owned_state_relative: str,
    snapshot_root_relative: str,
) -> ReplayState:
    state = _new_or_replayed_journal(
        journal=journal,
        owned_state_relative=owned_state_relative,
        snapshot_root_relative=snapshot_root_relative,
    )
    state_path = journal.path.parent / Path(owned_state_relative)
    if state.pending_send_attempt is not None:
        raise ResumeAmbiguityError("uncovered provider send cannot be retried")
    if state.requires_rollback:
        restored = _restore_working_state(
            journal=journal, state=state, state_path=state_path
        )
        restore = (
            prefix_zero_restore_receipt(
                state, destination_state_dir=state_path
            )
            if state.sealed_prefix == 0
            else restored
        )
        if restore is None:
            raise ResumableShardError("nonzero rollback omitted restore receipt")
        state = append_presend_rollback(
            journal, state, restore_receipt=restore
        )
    return state


def run_resumable_ingest_segment(**_kwargs: Any) -> ResumableSegmentResult:
    """Reject the former unsealed live surface.

    Live provider authority must be derived by ``resumable_launch`` from the
    current sealed preflight, manifest, and per-shard launch artifact.  Keeping
    this symbol as a fail-closed compatibility shim prevents older callers
    from silently bypassing that provenance boundary.
    """

    raise ResumableShardError(
        "unsealed segment execution is disabled; use run_locked_live_segment"
    )


def _journal_relative_target(
    journal: AppendOnlyResumeJournal, value: str, *, label: str
) -> tuple[str, Path]:
    relative = _safe_relative(value, label)
    root = Path(os.path.abspath(journal.path.parent))
    target = Path(os.path.abspath(root / Path(relative)))
    if target == root or root not in target.parents:
        raise ResumableShardError(f"{label} escaped the journal directory")
    return relative, target


def _closed_logical_extraction_receipt(total: int) -> dict[str, Any]:
    return {
        "kind": "mem0_memory_llm_generate_response_logical_calls",
        "boundary": "Memory.llm.generate_response",
        "count_semantics": "local_logical_wrapper_calls_not_http_attempts",
        "external_http_attempts_certified": False,
        "authorized_local_wrapper_retries": 0,
        "external_retry_attempts_certified": False,
        "authorized": total,
        "attempted": total,
        "completed": total,
        "failed": 0,
        "rejected": 0,
        "infer_true_adds_started": total,
        "infer_true_adds_exactly_one_call": total,
        "one_logical_call_per_infer_true_add_certified": True,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }


def _terminal_execution_binding(state: ReplayState) -> dict[str, Any]:
    latest = state.commits[-1]
    seal = state.latest_prefix_seal
    if seal is None:
        raise ResumableShardError("terminal execution omitted the full prefix seal")
    write_usage = validate_write_usage_attestation(
        seal.get("write_usage_attestation"),
        state=state,
        expected_committed_prefix=state.committed_prefix,
        expected_generation=seal["generation"],
    )
    body = {
        "kind": RESUMABLE_EXECUTION_BINDING_KIND,
        "comparison_certified": True,
        "external_http_attempts_certified": True,
        "external_provider_persistence_certified": False,
        "authorization_sha256": state.plan.authorization_sha256,
        "plan_sha256": state.plan.sha256,
        "checkpoint_authority_sha256": state.checkpoint_authority_sha256,
        "full_prefix": state.committed_prefix,
        "active_commit_entry_sha256": latest["entry_sha256"],
        "logical_meter_receipt_sha256": latest[
            "logical_meter_receipt_sha256"
        ],
        "transport_receipt_sha256": latest["transport_receipt_sha256"],
        "transport_closure_receipt_sha256": seal[
            "transport_closure_receipt_sha256"
        ],
        "write_usage_attestation_sha256": write_usage["receipt_sha256"],
        "source_implementation_sha256": (
            state.plan.source_implementation_sha256
        ),
        "source_environment_lock_sha256": (
            state.plan.source_environment_lock_sha256
        ),
        "mem0_tool_implementation_sha256": (
            state.plan.mem0_tool_implementation_sha256
        ),
        "mem0_environment_lock_sha256": (
            state.plan.mem0_environment_lock_sha256
        ),
    }
    return {**body, "receipt_sha256": canonical_json_sha256(body)}


def _rehydrate_terminal_adapter(
    *, adapter: Any, state: ReplayState, batches: Sequence[Any]
) -> None:
    """Rebuild the source-owned ten-message deque without provider/store adds."""

    _preseed_adapter(adapter, state)
    prefix = state.committed_prefix
    start = _primer_start(batches, prefix)
    primer = tuple(batches[start:prefix])
    if not primer:
        raise ResumableShardError("full-prefix terminal replay omitted its primer")
    owned = adapter._backend
    real_add = owned.add
    calls = 0

    def primer_add(messages: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        if calls >= len(primer):
            raise ResumableShardError("terminal primer exceeded its sealed suffix")
        batch = primer[calls]
        calls += 1
        expected_messages = [
            {"role": role, "content": content} for role, content in batch.messages
        ]
        if messages != expected_messages:
            raise ResumableShardError("terminal primer messages changed")
        if kwargs != {"user_id": state.plan.user_scope, "infer": True}:
            raise ResumableShardError("terminal primer scope/infer contract changed")
        return {"results": []}

    owned.add = primer_add
    try:
        from memory_condense.eval.mem0_adapter import _PreparedCorpus

        corpus = _PreparedCorpus(
            sample_id=primer[0].ref.sample_id,
            batches=primer,
            raw_pair_count=len(primer),
            skipped_empty_pair_count=0,
            official_longmemeval_protocol=True,
        )
        result = adapter._ingest_prepared(corpus)
        if calls != len(primer) or tuple(result.returned_memory_ids):
            raise ResumableShardError("terminal primer mutated returned-memory state")
    finally:
        owned.add = real_add
    _set_adapter_stats_from_state(adapter, state)
    _assert_live_adapter_matches_rehydration(adapter, state)


def _perform_terminal_search(
    *,
    shard: RawStressShard,
    authorization: RetrievalStageAuthorization,
    state: ReplayState,
    journal: AppendOnlyResumeJournal,
    state_path: Path,
    max_prompt_tokens: int,
    prompt_packer: Any,
) -> dict[str, Any]:
    """Run all local searches with an extraction transport that permits zero calls."""

    corpus = build_adapter_prepared_corpus(shard)
    factory = ResumableExactMem0AdapterFactory(
        authorization,
        segment_authorized_calls=0,
        adopt_existing_state=True,
        user_scope=state.plan.user_scope,
        before_http_send=None,
        expected_ownership_token_sha256=(
            state.latest_prefix_seal["ownership_token_sha256"]
        ),
    )
    adapter = factory(state_path)
    suspended: Mapping[str, Any] | None = None
    search_events: list[dict[str, Any]] = []
    try:
        _rehydrate_terminal_adapter(
            adapter=adapter, state=state, batches=corpus.batches
        )
        identity_reader = getattr(adapter, "_runtime_identity_snapshot", None)
        if not callable(identity_reader):
            raise ResumableShardError("terminal adapter omitted runtime identity")
        runtime_identity = _runtime_identity(
            identity_reader(),
            authorization.mem0_stable_config_sha256,
            authorization.mem0_stable_payload,
        )
        retrieval_rows: list[dict[str, Any]] = []
        previous_search_latency = 0.0
        for index, question in enumerate(shard.parsed_sample.questions, start=1):
            query = question.dated_question
            result = adapter.search(
                query,
                max_prompt_tokens=1_000_000_000,
                prompt_renderer=lambda rendered_query, _context: rendered_query,
                user_scope=state.plan.user_scope,
                threshold=MEM0_OFFICIAL_THRESHOLD,
                rendering_mode=MEM0_CERTIFIED_RENDERING,
            )
            search_runtime = _result_protocol_identity(
                result,
                authorization.mem0_stable_config_sha256,
                authorization.mem0_stable_payload,
            )
            if canonical_json_sha256(search_runtime) != canonical_json_sha256(
                runtime_identity
            ):
                raise ResumableShardError(
                    "Mem0 runtime identity changed during terminal search"
                )
            raw_pool = [
                _candidate_payload(candidate, rank)
                for rank, candidate in enumerate(
                    getattr(result, "raw_pool", ()), start=1
                )
            ]
            if len({row["memory_id"] for row in raw_pool}) != len(raw_pool):
                raise ResumableShardError("terminal raw pool repeated a memory ID")
            current_latency = float(
                getattr(getattr(result, "stats", None), "search_latency_s", 0.0)
            )
            latency = max(0.0, current_latency - previous_search_latency)
            previous_search_latency = current_latency
            packed = prompt_packer(
                query,
                result,
                max_prompt_tokens=max_prompt_tokens,
                evaluation_identity=authorization.source_evaluation_identity,
            )
            row = _pack_payload(
                packed,
                raw_pool,
                question_id=question.question_id,
                search_latency_s=latency,
            )
            retrieval_rows.append(row)
            search_events.append(
                {
                    "sequence": index,
                    "question_id": question.question_id,
                    "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                    "raw_memory_count": row["raw_memory_count"],
                    "raw_pool_sha256": row["raw_pool_sha256"],
                    "retrieval_row_sha256": row["retrieval_row_sha256"],
                }
            )
        stats = _stats_dict(adapter.stats)
        if stats["search_calls"] != state.plan.authorized_search_operations:
            raise ResumableShardError("terminal search count did not close")
        if any(
            stats[field] != state.plan.authorized_add_operations
            for field in (
                "add_calls",
                "add_attempted_calls",
                "add_completed_calls",
            )
        ) or stats["add_failed_calls"] != 0:
            raise ResumableShardError("terminal search changed sealed add accounting")
        transport = adapter._production_extraction_transport
        transport.assert_call_budget_closed()
        receipt = transport.transport_receipt()
        if any(receipt[field] != 0 for field in ("authorized", "attempted", "completed", "failed", "rejected")):
            raise ResumableShardError("terminal search touched extraction transport")
        suspended = suspend_resumable_adapter(adapter)
        adapter._ledger.clear()
        adapter._scopes.clear()
        adapter._scope_protocol.clear()
        adapter._active_scope = None
        adapter._backend = None
        return {
            "runtime_identity": runtime_identity,
            "retrieval_rows": retrieval_rows,
            "mem0_usage": stats,
            "search_events": search_events,
            "factory_receipt": dict(factory.binding_receipt()),
            "suspend_receipt": dict(suspended),
        }
    except BaseException:
        if suspended is None:
            try:
                suspend_resumable_adapter(adapter)
            except BaseException:
                pass
        raise


def _terminal_stage_payload(
    *,
    shard: RawStressShard,
    authorization: RetrievalStageAuthorization,
    state: ReplayState,
    search: Mapping[str, Any],
) -> dict[str, Any]:
    extraction_identity, embedder_identity = _validated_model_identities(
        authorization
    )
    execution = _terminal_execution_binding(state)
    factory_receipt = dict(search["factory_receipt"])
    suspend_receipt = dict(search["suspend_receipt"])
    execution_body = dict(execution)
    execution_body.pop("receipt_sha256")
    execution_body.update(
        terminal_factory_receipt_sha256=factory_receipt["receipt_sha256"],
        terminal_suspend_receipt_sha256=suspend_receipt["receipt_sha256"],
    )
    execution = {
        **execution_body,
        "receipt_sha256": canonical_json_sha256(execution_body),
    }
    runtime_identity = {
        **dict(search["runtime_identity"]),
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }
    final_stats = dict(search["mem0_usage"])
    latest = state.commits[-1]
    seal = state.latest_prefix_seal
    assert seal is not None
    write_usage = validate_write_usage_attestation(
        seal["write_usage_attestation"],
        state=state,
        expected_committed_prefix=state.committed_prefix,
        expected_generation=seal["generation"],
    )
    observed_write = write_usage["observed"]
    final_stats["provider_prompt_tokens"] = observed_write[
        "extraction_provider_input_tokens"
    ]
    final_stats["provider_completion_tokens"] = observed_write[
        "extraction_provider_output_tokens"
    ]
    final_stats["provider_usage_status"] = observed_write[
        "extraction_usage_status"
    ]
    result = {
        "format": RESUMABLE_TERMINAL_RESULT_FORMAT,
        "status": "complete",
        "certification_status": "exact_resumable_production",
        "comparison_certified": True,
        "execution_binding": execution,
        "sample_offset": shard.sample_offset,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_sha256": shard.sample_sha256,
        "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
        "history_sample_ids_sha256": canonical_json_sha256(
            list(shard.history_sample_ids)
        ),
        "question_ids": list(shard.question_ids),
        "question_ids_sha256": canonical_json_sha256(list(shard.question_ids)),
        "identity": {
            "source_validation_policy_sha256": (
                authorization.source_validation_policy_sha256
            ),
            "source_implementation_sha256": (
                authorization.source_implementation_sha256
            ),
            "source_environment_lock_sha256": (
                authorization.source_environment_lock_sha256
            ),
            "mem0_policy_sha256": authorization.mem0_policy_sha256,
            "mem0_tool_implementation_sha256": (
                authorization.mem0_tool_implementation_sha256
            ),
            "mem0_environment_lock_sha256": (
                authorization.mem0_environment_lock_sha256
            ),
            "mem0_stable_config_sha256": (
                authorization.mem0_stable_config_sha256
            ),
            "extraction_model_identity": extraction_identity,
            "extraction_model_identity_sha256": canonical_json_sha256(
                extraction_identity
            ),
            "embedder_model_identity": embedder_identity,
            "embedder_model_identity_sha256": canonical_json_sha256(
                embedder_identity
            ),
            "runtime_model_identity_probe": {
                "kind": "exact_resumable_factory_bound",
                "bound_embedder_receipt_sha256": factory_receipt[
                    "bound_embedder"
                ]["receipt_sha256"],
                "bound_bm25_receipt_sha256": factory_receipt["bound_bm25"][
                    "receipt_sha256"
                ],
                "comparison_certified": True,
            },
            "source_evaluation_identity": json.loads(
                json.dumps(authorization.source_evaluation_identity)
            ),
            "source_evaluation_identity_sha256": canonical_json_sha256(
                authorization.source_evaluation_identity
            ),
            "runtime_identity": runtime_identity,
        },
        "protocol": {
            "input_order": INPUT_ORDER_PROTOCOL,
            "official_longmemeval_protocol": True,
            "official_search_protocol": True,
            "top_k": MEM0_OFFICIAL_TOP_K,
            "threshold": MEM0_OFFICIAL_THRESHOLD,
            "rendering_mode": MEM0_CERTIFIED_RENDERING,
            "max_prompt_tokens": 8_000,
        },
        "raw_input_receipt": shard_receipt(shard),
        "ingestion_receipt": {
            "raw_pairs": shard.add_counts.raw_pairs,
            "skipped_empty_pairs": shard.add_counts.skipped_empty_pairs,
            "authorized_add_operations": (
                authorization.authorized_add_operations
            ),
            "attempted_add_operations": final_stats["add_attempted_calls"],
            "completed_add_operations": final_stats["add_completed_calls"],
            "failed_add_operations": final_stats["add_failed_calls"],
            "extraction_model_calls": _closed_logical_extraction_receipt(
                authorization.authorized_extraction_calls
            ),
            "persisted_request_token_state": False,
            "retained_request_token_state_bytes": 0,
            "request_token_state_evidence_kind": (
                "local_injected_request_token_state_contract"
            ),
            "external_provider_persistence_certified": False,
            "one_scope": True,
            "user_scope_sha256": hashlib.sha256(
                state.plan.user_scope.encode()
            ).hexdigest(),
            "comparison_certified": True,
        },
        "resumable_closure": {
            "plan_sha256": state.plan.sha256,
            "resume_plan": state.plan.as_dict(),
            "checkpoint_authority_sha256": (
                state.checkpoint_authority_sha256
            ),
            "active_commit_entry_sha256": latest["entry_sha256"],
            "journal_tail_entry_sha256": state.entries[-1]["entry_sha256"],
            "journal_chain_sha256": canonical_json_sha256(
                [row["entry_sha256"] for row in state.entries]
            ),
            "commit_population_sha256": canonical_json_sha256(
                [dict(row) for row in state.commits]
            ),
            "full_prefix_seal": dict(state.latest_prefix_seal),
            "logical_meter_receipt": latest["logical_meter_receipt"],
            "transport_receipt": latest["transport_receipt"],
            "transport_closure_receipt_sha256": seal[
                "transport_closure_receipt_sha256"
            ],
            "write_usage_attestation": write_usage,
            "write_usage_attestation_sha256": write_usage["receipt_sha256"],
            "factory_receipt": factory_receipt,
            "suspend_receipt": suspend_receipt,
        },
        "search_receipt": {
            "authorized_search_operations": (
                authorization.authorized_search_operations
            ),
            "completed_search_operations": final_stats["search_calls"],
            "failed_search_operations": 0,
            "extraction_transport_calls_during_search": 0,
        },
        "mem0_usage": final_stats,
        "write_usage_attestation": write_usage,
        "provenance": {
            "attribution_kind": MEM0_ATTRIBUTION_KIND,
            "supports_exact_source_provenance": False,
            "source_session_date_exposure": "diagnostics_only_not_model_input",
            "retrieved_created_at_exposure": "answer_prompt_date_headings",
            "provider_usage_status": observed_write[
                "extraction_usage_status"
            ],
            "external_http_attempts_certified": True,
            "external_retry_attempts_certified": True,
            "external_provider_persistence_certified": False,
        },
        "retrieval_rows": list(search["retrieval_rows"]),
    }
    trace = {
        "format": RESUMABLE_TERMINAL_TRACE_FORMAT,
        "status": "search_staged",
        "sample_offset": shard.sample_offset,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_sha256": shard.sample_sha256,
        "plan_sha256": state.plan.sha256,
        "checkpoint_authority_sha256": state.checkpoint_authority_sha256,
        "events": list(search["search_events"]),
        "completed_search_operations": len(search["search_events"]),
        "extraction_transport_calls": 0,
        "handles_closed_receipt_sha256": suspend_receipt["receipt_sha256"],
        "checkpoint_retained": True,
        "transport_closure_receipt_sha256": seal[
            "transport_closure_receipt_sha256"
        ],
        "write_usage_attestation_sha256": write_usage["receipt_sha256"],
    }
    result_sha = canonical_json_sha256(result)
    trace_sha = canonical_json_sha256(trace)
    body = {
        "format": RESUME_TERMINAL_FORMAT,
        "plan_sha256": state.plan.sha256,
        "authorization_sha256": state.plan.authorization_sha256,
        "committed_prefix": state.committed_prefix,
        "full_checkpoint_authority_sha256": (
            state.checkpoint_authority_sha256
        ),
        "completed_search_operations": len(search["search_events"]),
        "extraction_calls_closed": True,
        "provider_retries": 0,
        "transport_closure_receipt_sha256": seal[
            "transport_closure_receipt_sha256"
        ],
        "write_usage_attestation_sha256": write_usage["receipt_sha256"],
        "terminal_result_sha256": result_sha,
        "terminal_trace_sha256": trace_sha,
        "result": result,
        "trace": trace,
    }
    return body


def _validate_terminal_stage(
    *, artifact: Mapping[str, Any], state: ReplayState
) -> dict[str, Any]:
    value = json.loads(json.dumps(artifact, sort_keys=True, allow_nan=False))
    expected_keys = {
        "format",
        "plan_sha256",
        "authorization_sha256",
        "committed_prefix",
        "full_checkpoint_authority_sha256",
        "completed_search_operations",
        "extraction_calls_closed",
        "provider_retries",
        "transport_closure_receipt_sha256",
        "write_usage_attestation_sha256",
        "terminal_result_sha256",
        "terminal_trace_sha256",
        "result",
        "trace",
    }
    if set(value) != expected_keys or value.get("format") != RESUME_TERMINAL_FORMAT:
        raise ResumableShardError("terminal staging bundle schema changed")
    required = {
        "plan_sha256": state.plan.sha256,
        "authorization_sha256": state.plan.authorization_sha256,
        "committed_prefix": state.plan.authorized_add_operations,
        "full_checkpoint_authority_sha256": (
            state.checkpoint_authority_sha256
        ),
        "completed_search_operations": state.plan.authorized_search_operations,
        "extraction_calls_closed": True,
        "provider_retries": 0,
        "transport_closure_receipt_sha256": state.latest_prefix_seal[
            "transport_closure_receipt_sha256"
        ],
        "write_usage_attestation_sha256": state.latest_prefix_seal[
            "write_usage_attestation_sha256"
        ],
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ResumableShardError(f"terminal staging {field} mismatch")
    result = value.get("result")
    trace = value.get("trace")
    if not isinstance(result, dict) or not isinstance(trace, dict):
        raise ResumableShardError("terminal staging payloads are invalid")
    if canonical_json_sha256(result) != value.get("terminal_result_sha256"):
        raise ResumableShardError("terminal result digest mismatch")
    if canonical_json_sha256(trace) != value.get("terminal_trace_sha256"):
        raise ResumableShardError("terminal trace digest mismatch")
    if (
        result.get("format") != RESUMABLE_TERMINAL_RESULT_FORMAT
        or result.get("sample_sha256") != state.plan.sample_sha256
        or result.get("resumable_closure", {}).get(
            "checkpoint_authority_sha256"
        )
        != state.checkpoint_authority_sha256
        or trace.get("format") != RESUMABLE_TERMINAL_TRACE_FORMAT
        or trace.get("checkpoint_authority_sha256")
        != state.checkpoint_authority_sha256
        or result.get("write_usage_attestation", {}).get("receipt_sha256")
        != value["write_usage_attestation_sha256"]
        or result.get("resumable_closure", {}).get(
            "write_usage_attestation_sha256"
        )
        != value["write_usage_attestation_sha256"]
        or trace.get("write_usage_attestation_sha256")
        != value["write_usage_attestation_sha256"]
        or trace.get("transport_closure_receipt_sha256")
        != value["transport_closure_receipt_sha256"]
    ):
        raise ResumableShardError("terminal staging identity mismatch")
    return value


def _remove_terminal_working_state(
    *,
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    state_path: Path,
    terminal_stage_sha256: str,
) -> dict[str, Any]:
    seal = state.latest_prefix_seal
    if seal is None or seal["committed_prefix"] != state.plan.authorized_add_operations:
        raise ResumableShardError("terminal state removal omitted the full checkpoint")
    snapshot = journal.path.parent / Path(seal["snapshot_path"])
    verify_immutable_state_snapshot(
        snapshot,
        expected_authority_sha256=seal["snapshot_authority_sha256"],
        expected_manifest_sha256=seal["snapshot_manifest_sha256"],
        expected_tree_sha256=seal["snapshot_tree_sha256"],
        expected_ownership_token_sha256=seal["ownership_token_sha256"],
    )
    if state_path.exists():
        tree = state_tree_receipt(state_path)
        if tree["ownership_token_sha256"] != seal["ownership_token_sha256"]:
            raise ResumableShardError(
                "terminal working-state ownership differs from checkpoint"
            )
        token = (state_path / ".memory-condense-owned-state").read_text(
            encoding="utf-8"
        ).strip()
        _remove_owned_state(state_path, token)
    if state_path.exists():
        raise ResumableShardError("terminal working state remained after removal")
    body = {
        "format": RESUME_STATE_REMOVAL_FORMAT,
        "plan_sha256": state.plan.sha256,
        "terminal_stage_sha256": terminal_stage_sha256,
        "owned_state_path_sha256": _path_identity_sha256(state_path),
        "owned_state_removed": True,
        "snapshots_retained": snapshot.is_dir(),
    }
    if body["snapshots_retained"] is not True:
        raise ResumableShardError(
            "full checkpoint disappeared before terminal publication"
        )
    return body


def _official_terminal_payloads(
    *,
    stage: Mapping[str, Any],
    state: ReplayState,
    artifact_target: Path,
    trace_target: Path,
    environment_lock_path: Path,
    environment_lock_sha256: str,
) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    result = dict(stage["result"])
    execution = dict(result["execution_binding"])
    cleanup = {
        "attempted": True,
        "completed": True,
        "state_absent_before": False,
        "state_absent_after": True,
        "active_scope_cleared": True,
        "adapter_closed": True,
        "ledger_empty": True,
        "registered_scopes_empty": True,
        "scope_protocol_empty": True,
        "backend_closed_or_cleared": True,
        "owned_state_path_absent": True,
        "extraction_meter_restore_attempted": False,
        "extraction_meter_restored_before_cleanup": True,
        "resumable_zero_call_search": True,
        "terminal_stage_retained_until_publication": True,
        "full_checkpoint_retained_until_publication": True,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
        "environment_lock": {
            "filename": environment_lock_path.name,
            "authorized_sha256": environment_lock_sha256,
            "sha256_before": environment_lock_sha256,
            "sha256_after": environment_lock_sha256,
            "unchanged": True,
        },
    }
    trace = {
        "format": RETRIEVAL_TRACE_FORMAT,
        "status": "complete",
        "certification_status": "exact_resumable_production",
        "comparison_certified": True,
        "execution_binding": execution,
        "stage": "retrieval",
        "sample_offset": result["sample_offset"],
        "sample_id": result["sample_id"],
        "sample_sha256": result["sample_sha256"],
        "events": [
            {
                "sequence": 1,
                "event": "full_prefix_checkpoint_verified",
                "checkpoint_authority_sha256": (
                    state.checkpoint_authority_sha256
                ),
            },
            *[
                {**row, "sequence": index + 1, "event": "search_complete"}
                for index, row in enumerate(stage["trace"]["events"], start=1)
            ],
            {
                "sequence": len(stage["trace"]["events"]) + 2,
                "event": "terminal_stage_sealed",
                "terminal_result_sha256": stage["terminal_result_sha256"],
                "terminal_trace_sha256": stage["terminal_trace_sha256"],
            },
            {
                "sequence": len(stage["trace"]["events"]) + 3,
                "event": "active_state_removed_with_checkpoint_retained",
            },
        ],
        "cleanup": cleanup,
        "environment_lock": cleanup["environment_lock"],
        "elapsed_s": 0.0,
        # This is the byte digest recorded by the durable terminal-search
        # transition, not an ambiguously named canonical-content digest.
        "resumable_terminal_stage_file_sha256": state.terminal_search[
            "terminal_stage_sha256"
        ],
    }
    trace_payload = _render_json_bytes(trace)
    trace_sha = hashlib.sha256(trace_payload).hexdigest()
    artifact = {
        **result,
        "format": RETRIEVAL_ARTIFACT_FORMAT,
        "retrieval_trace": {
            "filename": trace_target.name,
            "sha256": trace_sha,
            "bytes": len(trace_payload),
        },
        "environment_lock": cleanup["environment_lock"],
        "resumable_terminal": {
            "terminal_stage_file_sha256": state.terminal_search[
                "terminal_stage_sha256"
            ],
            "stage_result_sha256": stage["terminal_result_sha256"],
            "stage_trace_sha256": stage["terminal_trace_sha256"],
            "checkpoint_authority_sha256": state.checkpoint_authority_sha256,
            "active_state_removed_before_publication": True,
            "full_checkpoint_retained_until_publication": True,
        },
    }
    artifact["content_sha256"] = canonical_json_sha256(artifact)
    artifact_payload = _render_json_bytes(artifact)
    return artifact_payload, trace_payload, artifact, trace


def _publish_or_verify_terminal_outputs(
    *,
    artifact_target: Path,
    trace_target: Path,
    artifact_payload: bytes,
    trace_payload: bytes,
) -> tuple[tuple[str, int], tuple[str, int]]:
    expected_artifact = (
        hashlib.sha256(artifact_payload).hexdigest(), len(artifact_payload)
    )
    expected_trace = (
        hashlib.sha256(trace_payload).hexdigest(), len(trace_payload)
    )
    artifact_exists = artifact_target.exists()
    trace_exists = trace_target.exists()
    if artifact_exists != trace_exists:
        raise ResumableShardError("terminal output transaction is partial")
    if not artifact_exists:
        trace_receipt, artifact_receipt = _atomic_create_payloads(
            (
                (trace_target, trace_payload),
                (artifact_target, artifact_payload),
            )
        )
        if trace_receipt != expected_trace or artifact_receipt != expected_artifact:
            raise ResumableShardError("terminal publication receipt changed")
    else:
        if (
            artifact_target.is_symlink()
            or trace_target.is_symlink()
            or not artifact_target.is_file()
            or not trace_target.is_file()
            or artifact_target.read_bytes() != artifact_payload
            or trace_target.read_bytes() != trace_payload
        ):
            raise ResumableShardError("existing terminal outputs differ from staging")
    return expected_artifact, expected_trace


def _remove_terminal_stage(
    *, stage_target: Path, expected_sha256: str
) -> bool:
    sidecar = stage_target.with_name(stage_target.name + ".sha256")
    if not stage_target.exists() and not sidecar.exists():
        return True
    read_sealed_json(stage_target, expected_sha256=expected_sha256)
    stage_target.unlink()
    sidecar.unlink()
    if stage_target.exists() or sidecar.exists():
        raise ResumableShardError("terminal staging bundle remained after GC")
    return True


def _verified_official_output_receipts(
    *, state: ReplayState, artifact_target: Path, trace_target: Path
) -> tuple[tuple[str, int], tuple[str, int]]:
    published = state.terminal_published
    if published is None:
        raise ResumableShardError("official terminal outputs are not journaled")
    artifact = artifact_target.read_bytes()
    trace = trace_target.read_bytes()
    artifact_receipt = (hashlib.sha256(artifact).hexdigest(), len(artifact))
    trace_receipt = (hashlib.sha256(trace).hexdigest(), len(trace))
    if (
        published["official_artifact_sha256"] != artifact_receipt[0]
        or published["official_trace_sha256"] != trace_receipt[0]
    ):
        raise ResumableShardError("published terminal output digest changed")
    return artifact_receipt, trace_receipt


def _finish_terminal_gc(
    *,
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    stage_target: Path,
) -> ReplayState:
    if state.checkpoint_gc is None:
        snapshot_gc = remove_verified_snapshot_root(journal, state)
        if snapshot_gc["snapshots_removed"] is not True:
            raise ResumableShardError("terminal snapshot GC did not close")
        terminal = state.terminal_search
        assert terminal is not None
        _remove_terminal_stage(
            stage_target=stage_target,
            expected_sha256=terminal["terminal_stage_sha256"],
        )
        state = append_checkpoint_gc(
            journal,
            state,
            checkpoint_gc_receipt={
                "format": RESUME_CHECKPOINT_GC_FORMAT,
                "publication_receipt_sha256": state.terminal_published[
                    "publication_receipt_sha256"
                ],
                "snapshots_removed": True,
                "terminal_stage_removed": True,
            },
        )
    if state.cleanup_closed is None:
        state = append_cleanup_closed(
            journal,
            state,
            cleanup_receipt={
                "format": RESUME_CLEANUP_FORMAT,
                "checkpoint_gc_receipt_sha256": state.checkpoint_gc[
                    "checkpoint_gc_receipt_sha256"
                ],
                "owned_state_removed": True,
                "snapshots_removed": True,
                "terminal_stage_removed": True,
                "official_outputs_retained": True,
            },
        )
    return state


def _run_resumable_terminal_stage_locked(
    *,
    shard: RawStressShard,
    authorization: RetrievalStageAuthorization,
    plan: ResumePlan,
    journal_path: str | os.PathLike[str],
    owned_state_relative: str,
    snapshot_root_relative: str,
    terminal_stage_relative: str,
    artifact_relative: str,
    trace_relative: str,
    mem0_environment_lock_path: str | os.PathLike[str],
    max_prompt_tokens: int = 8_000,
    prompt_packer: Any = _default_prompt_packer,
) -> ResumableTerminalResult:
    """Finish a full prefix without another extraction-provider call.

    The safe order is immutable terminal staging, active-state removal while
    the full checkpoint remains, atomic official artifact/trace publication,
    checkpoint/stage GC, and a final closed journal marker.
    """

    if max_prompt_tokens != 8_000:
        raise ResumableShardError("terminal prompt cap must remain exactly 8000")
    if not callable(prompt_packer):
        raise TypeError("terminal prompt_packer must be callable")
    lock_target, lock_sha = _environment_lock_snapshot(
        mem0_environment_lock_path,
        label="mem0_environment_lock_path",
    )
    _validate_retrieval_authorization(
        shard,
        authorization,
        computed_mem0_environment_lock_sha256=lock_sha,
    )
    journal = AppendOnlyResumeJournal(journal_path, plan)
    state = _new_or_replayed_journal(
        journal=journal,
        owned_state_relative=owned_state_relative,
        snapshot_root_relative=snapshot_root_relative,
    )
    corpus = build_adapter_prepared_corpus(shard)
    batch_hashes = tuple(prepared_batch_sha256(row) for row in corpus.batches)
    _validate_plan_against_inputs(
        state,
        shard=shard,
        authorization=authorization,
        batch_sha256s=batch_hashes,
    )
    if canonical_json_sha256(asdict(authorization)) != plan.authorization_sha256:
        raise ResumableShardError("terminal authorization digest differs from plan")
    if state.pending_intent is not None or state.pending_send_attempt is not None:
        raise ResumeAmbiguityError("terminal search found uncovered add work")
    if (
        state.committed_prefix != plan.authorized_add_operations
        or state.sealed_prefix != state.committed_prefix
    ):
        raise ResumableShardError("terminal search requires the sealed full prefix")
    state_path_relative, state_path = _journal_relative_target(
        journal, owned_state_relative, label="owned_state_relative"
    )
    snapshot_relative, snapshot_root = _journal_relative_target(
        journal, snapshot_root_relative, label="snapshot_root_relative"
    )
    stage_relative, stage_target = _journal_relative_target(
        journal, terminal_stage_relative, label="terminal_stage_relative"
    )
    artifact_relative, artifact_target = _journal_relative_target(
        journal, artifact_relative, label="artifact_relative"
    )
    trace_relative, trace_target = _journal_relative_target(
        journal, trace_relative, label="trace_relative"
    )
    targets = {
        state_path,
        snapshot_root,
        stage_target,
        artifact_target,
        trace_target,
        journal.path,
        lock_target,
    }
    if len(targets) != 7:
        raise ResumableShardError("terminal paths must be distinct")
    if any(
        left in right.parents or right in left.parents
        for index, left in enumerate(targets)
        for right in tuple(targets)[index + 1 :]
    ):
        raise ResumableShardError("terminal paths must not be nested")

    # Official publication is a durable recovery boundary.  Once it exists,
    # no checkpoint or staging payload is needed to finish provider-free GC.
    if state.terminal_published is not None:
        if (
            state.terminal_published["official_artifact_path"] != artifact_relative
            or state.terminal_published["official_trace_path"] != trace_relative
        ):
            raise ResumableShardError("official terminal output path changed")
        artifact_receipt, trace_receipt = _verified_official_output_receipts(
            state=state,
            artifact_target=artifact_target,
            trace_target=trace_target,
        )
        state = _finish_terminal_gc(
            journal=journal, state=state, stage_target=stage_target
        )
    else:
        stage_receipt: dict[str, Any]
        if stage_target.exists():
            stage_receipt = read_sealed_json(stage_target)
            stage = _validate_terminal_stage(
                artifact=stage_receipt["payload"], state=state
            )
        else:
            if state.terminal_search is not None:
                raise ResumeAmbiguityError(
                    "terminal staging disappeared before official publication"
                )
            seal = state.latest_prefix_seal
            assert seal is not None
            if state_path.exists():
                active_tree = state_tree_receipt(state_path)
                if (
                    active_tree["ownership_token_sha256"]
                    != seal["ownership_token_sha256"]
                ):
                    raise ResumableShardError(
                        "terminal active-state ownership differs from checkpoint"
                    )
                token = (state_path / ".memory-condense-owned-state").read_text(
                    encoding="utf-8"
                ).strip()
                _remove_owned_state(state_path, token)
            restore_snapshot_to_fresh_state(
                snapshot_path=journal.path.parent / Path(seal["snapshot_path"]),
                destination_state_dir=state_path,
                expected_authority_sha256=seal["snapshot_authority_sha256"],
                expected_manifest_sha256=seal["snapshot_manifest_sha256"],
                expected_tree_sha256=seal["snapshot_tree_sha256"],
                expected_ownership_token_sha256=seal[
                    "ownership_token_sha256"
                ],
            )
            search = _perform_terminal_search(
                shard=shard,
                authorization=authorization,
                state=state,
                journal=journal,
                state_path=state_path,
                max_prompt_tokens=max_prompt_tokens,
                prompt_packer=prompt_packer,
            )
            stage = _terminal_stage_payload(
                shard=shard,
                authorization=authorization,
                state=state,
                search=search,
            )
            stage_receipt = publish_sealed_json(stage_target, stage)
            stage = _validate_terminal_stage(
                artifact=stage_receipt["payload"], state=state
            )
        if state.terminal_search is None:
            state = append_terminal_search(
                journal,
                state,
                terminal_stage_path=stage_relative,
                terminal_stage_sha256=stage_receipt["sha256"],
                terminal_result_sha256=stage["terminal_result_sha256"],
                terminal_trace_sha256=stage["terminal_trace_sha256"],
                completed_search_operations=stage[
                    "completed_search_operations"
                ],
            )
        elif state.terminal_search["terminal_stage_sha256"] != stage_receipt[
            "sha256"
        ]:
            raise ResumableShardError("journaled terminal stage digest changed")
        if state.active_state_removed is None:
            removal = _remove_terminal_working_state(
                journal=journal,
                state=state,
                state_path=state_path,
                terminal_stage_sha256=stage_receipt["sha256"],
            )
            state = append_active_state_removed(
                journal, state, removal_receipt=removal
            )
        elif state_path.exists():
            raise ResumableShardError(
                "journal says terminal active state was removed but it exists"
            )
        artifact_payload, trace_payload, _artifact, _trace = (
            _official_terminal_payloads(
                stage=stage,
                state=state,
                artifact_target=artifact_target,
                trace_target=trace_target,
                environment_lock_path=lock_target,
                environment_lock_sha256=lock_sha,
            )
        )
        artifact_receipt, trace_receipt = _publish_or_verify_terminal_outputs(
            artifact_target=artifact_target,
            trace_target=trace_target,
            artifact_payload=artifact_payload,
            trace_payload=trace_payload,
        )
        state = append_terminal_published(
            journal,
            state,
            publication_receipt={
                "format": RESUME_PUBLICATION_FORMAT,
                "terminal_stage_sha256": stage_receipt["sha256"],
                "official_artifact_path": artifact_relative,
                "official_artifact_sha256": artifact_receipt[0],
                "official_trace_path": trace_relative,
                "official_trace_sha256": trace_receipt[0],
                "outputs_verified": True,
            },
        )
        state = _finish_terminal_gc(
            journal=journal, state=state, stage_target=stage_target
        )

    observed_lock_target, lock_after = _environment_lock_snapshot(
        lock_target, label="mem0_environment_lock_path post-terminal"
    )
    if observed_lock_target != lock_target or lock_after != lock_sha:
        raise ResumableShardError("Mem0 environment lock changed during terminal run")
    artifact_receipt, trace_receipt = _verified_official_output_receipts(
        state=state,
        artifact_target=artifact_target,
        trace_target=trace_target,
    )
    if state.cleanup_closed is None or state_path.exists() or snapshot_root.exists():
        raise ResumableShardError("terminal lifecycle did not close")
    body = {
        "action": "terminal_published_and_cleaned",
        "artifact_sha256": artifact_receipt[0],
        "artifact_bytes": artifact_receipt[1],
        "trace_sha256": trace_receipt[0],
        "trace_bytes": trace_receipt[1],
        "journal_tail_sha256": state.entries[-1]["entry_sha256"],
        "checkpoint_authority_sha256": state.checkpoint_authority_sha256,
    }
    return ResumableTerminalResult(
        action=body["action"],
        artifact_path=artifact_target,
        trace_path=trace_target,
        artifact_sha256=artifact_receipt[0],
        artifact_bytes=artifact_receipt[1],
        trace_sha256=trace_receipt[0],
        trace_bytes=trace_receipt[1],
        journal_tail_sha256=body["journal_tail_sha256"],
        checkpoint_authority_sha256=body[
            "checkpoint_authority_sha256"
        ],
        receipt_sha256=canonical_json_sha256(body),
    )


def run_resumable_terminal_stage(
    *,
    shard: RawStressShard,
    authorization: RetrievalStageAuthorization,
    plan: ResumePlan,
    journal_path: str | os.PathLike[str],
    owned_state_relative: str,
    snapshot_root_relative: str,
    terminal_stage_relative: str,
    artifact_relative: str,
    trace_relative: str,
    mem0_environment_lock_path: str | os.PathLike[str],
    max_prompt_tokens: int = 8_000,
    prompt_packer: Any = _default_prompt_packer,
) -> ResumableTerminalResult:
    """Hold the shard lease through terminal search and final publication."""

    with JournalLease(journal_path):
        return _run_resumable_terminal_stage_locked(
            shard=shard,
            authorization=authorization,
            plan=plan,
            journal_path=journal_path,
            owned_state_relative=owned_state_relative,
            snapshot_root_relative=snapshot_root_relative,
            terminal_stage_relative=terminal_stage_relative,
            artifact_relative=artifact_relative,
            trace_relative=trace_relative,
            mem0_environment_lock_path=mem0_environment_lock_path,
            max_prompt_tokens=max_prompt_tokens,
            prompt_packer=prompt_packer,
        )


__all__ = [
    "DEFAULT_SEGMENT_ADDS",
    "DurableHTTPJournalBoundary",
    "RESUMABLE_EXECUTION_BINDING_KIND",
    "RESUMABLE_SEGMENT_FORMAT",
    "RESUMABLE_TERMINAL_RESULT_FORMAT",
    "RESUMABLE_TERMINAL_TRACE_FORMAT",
    "ResumableSegmentResult",
    "ResumableTerminalResult",
    "build_resume_plan",
    "prepared_batch_sha256",
    "run_resumable_ingest_segment",
    "run_resumable_terminal_stage",
]
