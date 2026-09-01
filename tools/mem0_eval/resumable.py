"""Fail-closed, append-only resumability for the locked Mem0 shard.

This module owns the durable control plane only.  It never imports Mem0,
loads a model, or opens a provider socket at import time.  Production code
must bind the pure journal/snapshot contracts here to the exact factory and
send boundary in :mod:`tools.mem0_eval.production_binding`.

The journal is deliberately conservative.  An ``intent`` with no durable
``send_attempt`` can roll back to the latest sealed prefix.  Once a send was
attempted, a missing commit is externally ambiguous and the campaign cannot
retry that add.  A commit which was not followed by a closed-handle immutable
snapshot is also not resumable: provider work cannot be replayed merely to
repair local state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import tempfile
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .policy import (
    MEM0_EXTRACTION_GATEWAY_URL,
    MEM0_EXTRACTION_MODEL,
    MEM0_EXTRACTION_PROVIDER,
    MEM0_EXTRACTION_REVISION,
    MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
)


RESUME_JOURNAL_FORMAT = "memory-condense-mem0-resume-journal-v2"
RESUME_PLAN_FORMAT = "memory-condense-mem0-resume-plan-v1"
RESUME_SNAPSHOT_FORMAT = "memory-condense-mem0-prefix-snapshot-v2"
RESUME_SNAPSHOT_AUTHORITY_FORMAT = (
    "memory-condense-mem0-prefix-snapshot-authority-v2"
)
RESUME_RESTORE_FORMAT = "memory-condense-mem0-restored-state-v1"
RESUME_ROOT_MARKER_FORMAT = "memory-condense-mem0-snapshot-root-v1"
RESUME_REHYDRATION_FORMAT = "memory-condense-mem0-rehydration-v1"
RESUME_PREFIX_CLOSE_FORMAT = "memory-condense-mem0-prefix-close-v2"
RESUME_RECORD_ROOT_FORMAT = "memory-condense-mem0-journal-record-root-v1"
RESUME_TERMINAL_FORMAT = "memory-condense-mem0-terminal-search-v2"
RESUME_CLEANUP_FORMAT = "memory-condense-mem0-terminal-cleanup-v1"
RESUME_STATE_REMOVAL_FORMAT = "memory-condense-mem0-active-state-removal-v1"
RESUME_PUBLICATION_FORMAT = "memory-condense-mem0-terminal-publication-v1"
RESUME_CHECKPOINT_GC_FORMAT = "memory-condense-mem0-checkpoint-gc-v1"
RESUME_WRITE_USAGE_FORMAT = (
    "memory-condense-mem0-complete-write-usage-attestation-v1"
)
OWNERSHIP_MARKER = ".memory-condense-owned-state"
SNAPSHOT_ROOT_MARKER = ".memory-condense-mem0-snapshot-root"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
_WINDOW_MESSAGES = 10
_RESUME_SEGMENT_ADDS = 256
_PROCESS_LEASE_REGISTRY: set[str] = set()
_PROCESS_LEASE_REGISTRY_LOCK = threading.Lock()
_WINDOW_REF_KEYS = {
    "sample_id",
    "source",
    "session",
    "session_index",
    "original_session_index",
    "batch_index",
    "date",
    "turn_start",
    "turn_count",
    "roles",
}
_ADAPTER_STATS_KEYS = {
    "add_calls",
    "add_attempted_calls",
    "add_completed_calls",
    "add_failed_calls",
    "search_calls",
    "add_latency_s",
    "search_latency_s",
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
    "released_scopes",
    "provider_prompt_tokens",
    "provider_completion_tokens",
    "provider_usage_status",
    "token_counter_identity",
    "token_counter_identity_verified",
}
_WRITE_USAGE_OBSERVED_KEYS = {
    "add_attempted",
    "add_completed",
    "add_failed",
    "extraction_attempted",
    "extraction_completed",
    "extraction_failed",
    "extraction_raw_message_token_proxy",
    "extraction_provider_input_tokens",
    "extraction_provider_output_tokens",
    "extraction_usage_status",
    "embedding_operations",
    "embedding_input_token_proxy",
    "returned_memory_count",
    "persisted_memory_count",
    "persisted_storage_bytes",
    "add_latency_s",
    "extraction_latency_s",
    "embedding_latency_s",
    "storage_latency_s",
}
_ENTRY_PAYLOAD_KEYS = {
    "header": {
        "plan",
        "plan_sha256",
        "owned_state_path",
        "snapshot_root_path",
        "snapshot_root_ownership_token",
        "snapshot_root_marker_sha256",
        "journal_path_sha256",
        "empty_prefix_authority_sha256",
        "append_only",
        "provider_retries",
        "resume_semantics",
    },
    "intent": {
        "generation",
        "ordinal",
        "batch_sha256",
        "committed_prefix_before",
        "session_sha256",
    },
    "send_attempt": {
        "generation",
        "ordinal",
        "intent_entry_sha256",
        "global_attempt_ordinal",
        "request_sha256",
        "prior_checkpoint_authority_sha256",
    },
    "commit": {
        "generation",
        "ordinal",
        "intent_entry_sha256",
        "send_attempt_entry_sha256",
        "batch_sha256",
        "response_sha256",
        "returned_memory_ids",
        "source_ref",
        "request_window_refs",
        "request_window_sha256",
        "ledger_projection_sha256",
        "user_scope",
        "scope_protocol",
        "attribution_kind",
        "date_exposure_kind",
        "request_window_messages",
        "adapter_stats",
        "adapter_stats_sha256",
        "logical_meter_receipt",
        "logical_meter_receipt_sha256",
        "transport_receipt",
        "transport_receipt_sha256",
        "add_latency_s",
        "raw_message_tokens",
        "cumulative_logical_attempted",
        "cumulative_logical_completed",
        "cumulative_logical_failed",
        "cumulative_logical_rejected",
        "cumulative_http_attempted",
        "cumulative_http_completed",
        "cumulative_http_failed",
        "cumulative_http_rejected",
    },
    "prefix_sealed": {
        "generation",
        "committed_prefix",
        "active_commit_entry_sha256",
        "snapshot_path",
        "snapshot_manifest_sha256",
        "snapshot_tree_sha256",
        "ownership_token_sha256",
        "handles_closed_receipt_sha256",
        "transport_closure_receipt_sha256",
        "write_usage_attestation",
        "write_usage_attestation_sha256",
        "snapshot_authority_sha256",
        "snapshot_authority_artifact_sha256",
        "snapshot_receipt_sha256",
        "rehydration_sha256",
        "cumulative_extraction_attempted",
        "cumulative_extraction_completed",
        "cumulative_http_attempted",
        "cumulative_http_completed",
        "failures",
        "rejections",
    },
    "rollback": {
        "generation",
        "intent_entry_sha256",
        "restore_prefix",
        "restore_authority_sha256",
        "restore_receipt_sha256",
        "restored_snapshot_tree_sha256",
        "next_generation",
        "reason",
    },
    "terminal_search": {
        "generation",
        "completed_search_operations",
        "terminal_stage_path",
        "terminal_stage_sha256",
        "terminal_result_sha256",
        "terminal_trace_sha256",
        "committed_prefix",
        "full_checkpoint_authority_sha256",
        "extraction_calls_closed",
        "provider_retries",
        "transport_closure_receipt_sha256",
        "write_usage_attestation_sha256",
    },
    "active_state_removed": {
        "generation",
        "terminal_search_entry_sha256",
        "state_removal_receipt_sha256",
        "owned_state_removed",
        "snapshots_retained",
    },
    "terminal_published": {
        "generation",
        "terminal_search_entry_sha256",
        "official_artifact_path",
        "official_artifact_sha256",
        "official_trace_path",
        "official_trace_sha256",
        "publication_receipt_sha256",
        "outputs_verified",
    },
    "checkpoint_gc": {
        "generation",
        "terminal_published_entry_sha256",
        "active_state_removed_entry_sha256",
        "checkpoint_gc_receipt_sha256",
        "snapshots_removed",
        "terminal_stage_removed",
    },
    "cleanup_closed": {
        "generation",
        "checkpoint_gc_entry_sha256",
        "cleanup_receipt_sha256",
        "owned_state_removed",
        "snapshots_removed",
        "terminal_stage_removed",
        "official_outputs_retained",
    },
}
_KINDS = {
    "header",
    "intent",
    "send_attempt",
    "commit",
    "prefix_sealed",
    "rollback",
    "terminal_search",
    "active_state_removed",
    "terminal_published",
    "checkpoint_gc",
    "cleanup_closed",
}


class ResumableShardError(RuntimeError):
    """The resume journal, snapshot, or state transition is invalid."""


class ResumeAmbiguityError(ResumableShardError):
    """A provider send or local mutation cannot safely be replayed."""


class ResumeJournalLocked(ResumableShardError):
    """Another process currently owns the resume journal lease."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResumableShardError(f"value is not canonical JSON: {exc}") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_json(value: Any, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise ResumableShardError(f"{label} is not strict JSON") from exc


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ResumableShardError(f"{label} must be a lowercase SHA-256")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResumableShardError(f"{label} must be non-empty text")
    return value.strip()


def _require_count(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResumableShardError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResumableShardError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ResumableShardError(f"{label} must be finite and non-negative")
    return result


def _source_ref(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResumableShardError(f"{label} must be an object")
    row = _strict_json(value, label)
    if not isinstance(row, dict):  # pragma: no cover - guarded above
        raise ResumableShardError(f"{label} must be an object")
    _exact_keys(row, _WINDOW_REF_KEYS, label)
    for field in ("sample_id", "source", "session", "date"):
        row[field] = _require_text(row.get(field), f"{label}.{field}")
    for field in (
        "session_index",
        "original_session_index",
        "batch_index",
        "turn_start",
    ):
        row[field] = _require_count(row.get(field), f"{label}.{field}")
    row["turn_count"] = _require_count(
        row.get("turn_count"), f"{label}.turn_count", minimum=1
    )
    roles = row.get("roles")
    if (
        not isinstance(roles, list)
        or len(roles) != row["turn_count"]
        or any(not isinstance(role, str) or not role for role in roles)
    ):
        raise ResumableShardError(
            f"{label}.roles must contain one non-empty role per message"
        )
    return row


def _source_refs(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ResumableShardError(f"{label} must be an array")
    return [_source_ref(row, f"{label}[{index}]") for index, row in enumerate(value)]


def _merge_source_refs(*groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            row = _source_ref(value, "source ref")
            digest = canonical_json_sha256(row)
            if digest not in seen:
                seen.add(digest)
                result.append(row)
    return result


def _adapter_stats(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResumableShardError(f"{label} must be an object")
    row = _strict_json(value, label)
    if not isinstance(row, dict):  # pragma: no cover
        raise ResumableShardError(f"{label} must be an object")
    _exact_keys(row, _ADAPTER_STATS_KEYS, label)
    for field in (
        "add_calls",
        "add_attempted_calls",
        "add_completed_calls",
        "add_failed_calls",
        "search_calls",
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
        "released_scopes",
    ):
        row[field] = _require_count(row.get(field), f"{label}.{field}")
    for field in ("add_latency_s", "search_latency_s"):
        row[field] = _require_finite(row.get(field), f"{label}.{field}")
    for field in ("provider_prompt_tokens", "provider_completion_tokens"):
        if row.get(field) is not None:
            row[field] = _require_count(row[field], f"{label}.{field}")
    row["provider_usage_status"] = _require_text(
        row.get("provider_usage_status"), f"{label}.provider_usage_status"
    )
    row["token_counter_identity"] = _require_text(
        row.get("token_counter_identity"), f"{label}.token_counter_identity"
    )
    if not isinstance(row.get("token_counter_identity_verified"), bool):
        raise ResumableShardError(
            f"{label}.token_counter_identity_verified must be boolean"
        )
    return row


def _observed_counter_receipt(
    value: Any,
    *,
    label: str,
    expected: int,
    logical: bool,
    authorized_total: int | None = None,
    expected_seeded_prefix: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResumableShardError(f"{label} must be an object")
    row = _strict_json(value, label)
    if not isinstance(row, dict):  # pragma: no cover
        raise ResumableShardError(f"{label} must be an object")
    for field in ("attempted", "completed", "failed", "rejected"):
        _require_count(row.get(field), f"{label}.{field}")
    if row["attempted"] != expected or row["completed"] != expected:
        raise ResumableShardError(f"{label} cumulative completion mismatch")
    if row["failed"] != 0 or row["rejected"] != 0:
        raise ResumableShardError(f"{label} contains failures or rejections")
    if logical:
        for field in (
            "infer_true_adds_started",
            "infer_true_adds_exactly_one_call",
        ):
            if row.get(field) != expected:
                raise ResumableShardError(f"{label}.{field} mismatch")
    else:
        if authorized_total is None or expected_seeded_prefix is None:
            raise ResumableShardError(
                f"{label} requires the exact plan and sealed-prefix authority"
            )
        authorized_total = _require_count(
            authorized_total, f"{label}.authorized_total", minimum=1
        )
        expected_seeded_prefix = _require_count(
            expected_seeded_prefix, f"{label}.expected_seeded_prefix"
        )
        if not expected_seeded_prefix <= expected <= authorized_total:
            raise ResumableShardError(f"{label} prefix authority is impossible")
        _exact_keys(
            row,
            {
                "kind",
                "authorized",
                "seeded_prefix",
                "segment_authorized",
                "attempted",
                "completed",
                "failed",
                "rejected",
                "segment_receipt",
                "segment_receipt_sha256",
                "retries_authorized",
                "provider_usage_status",
                "provider_usage_records",
                "provider_input_tokens",
                "provider_output_tokens",
                "provider_total_tokens",
                "provider_latency_s",
            },
            label,
        )
        if (
            row["kind"] != "resumable_cumulative_http_transport"
            or row["authorized"] != authorized_total
            or row["seeded_prefix"] != expected_seeded_prefix
            or row["segment_authorized"]
            != min(
                _RESUME_SEGMENT_ADDS,
                authorized_total - expected_seeded_prefix,
            )
            or row["attempted"] != row["provider_usage_records"]
            or row["provider_usage_status"] != "provider_reported_exact"
            or row["provider_total_tokens"]
            != row["provider_input_tokens"] + row["provider_output_tokens"]
            or row["retries_authorized"] != 0
        ):
            raise ResumableShardError(f"{label} cumulative transport changed")
        for field in (
            "authorized",
            "seeded_prefix",
            "segment_authorized",
            "provider_usage_records",
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_total_tokens",
        ):
            _require_count(row.get(field), f"{label}.{field}")
        _require_finite(row.get("provider_latency_s"), f"{label}.provider_latency_s")
        segment = row.get("segment_receipt")
        if not isinstance(segment, dict):
            raise ResumableShardError(f"{label} segment receipt is invalid")
        if canonical_json_sha256(segment) != row["segment_receipt_sha256"]:
            raise ResumableShardError(f"{label} segment receipt digest mismatch")
        _exact_keys(
            segment,
            {
                "kind",
                "role",
                "authorized",
                "attempted",
                "completed",
                "failed",
                "rejected",
                "retries_authorized",
                "provider_usage_status",
                "provider_usage_records",
                "provider_input_tokens",
                "provider_output_tokens",
                "provider_total_tokens",
                "provider_latency_s",
                "production_eligible",
                "provider",
                "model",
                "revision",
                "route_identity_sha256",
                "request_identity_sha256",
                "gateway_url",
                "max_completion_tokens",
                "sampling_parameters_omitted",
                "sdk_retries",
                "http_transport_retries",
                "follow_redirects",
                "trust_env",
                "cap_boundary",
                "external_http_attempts_certified",
                "external_provider_persistence_certified",
            },
            f"{label}.segment_receipt",
        )
        segment_expected = expected - row["seeded_prefix"]
        exact = {
            "kind": "local_transport_send_cap",
            "role": "extraction",
            "authorized": row["segment_authorized"],
            "attempted": segment_expected,
            "completed": segment_expected,
            "failed": 0,
            "rejected": 0,
            "retries_authorized": 0,
            "provider_usage_status": "provider_reported_exact",
            "provider_usage_records": segment_expected,
            "production_eligible": True,
            "provider": MEM0_EXTRACTION_PROVIDER,
            "model": MEM0_EXTRACTION_MODEL,
            "revision": MEM0_EXTRACTION_REVISION,
            "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
            "gateway_url": MEM0_EXTRACTION_GATEWAY_URL,
            "max_completion_tokens": 2_000,
            "sampling_parameters_omitted": True,
            "sdk_retries": 0,
            "http_transport_retries": 0,
            "follow_redirects": False,
            "trust_env": False,
            "cap_boundary": "httpx.BaseTransport.handle_request",
            "external_http_attempts_certified": True,
            "external_provider_persistence_certified": False,
        }
        for field, wanted in exact.items():
            if segment.get(field) != wanted:
                raise ResumableShardError(
                    f"{label}.segment_receipt.{field} mismatch"
                )
        expected_request_identity_sha256 = canonical_json_sha256(
            {
                "format": "memory-condense-mem0-extraction-request-v1",
                "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 2_000,
                "sampling_parameters": "omitted",
                "sdk_retries": 0,
                "http_transport_retries": 0,
                "follow_redirects": False,
                "trust_env": False,
                "timeout_seconds": 600.0,
                "connect_timeout_seconds": 30.0,
                "cap_boundary": "httpx.BaseTransport.handle_request",
            }
        )
        if segment.get("request_identity_sha256") != (
            expected_request_identity_sha256
        ):
            raise ResumableShardError(
                f"{label}.segment_receipt.request_identity_sha256 mismatch"
            )
        for field in (
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_total_tokens",
        ):
            _require_count(segment.get(field), f"{label}.segment_receipt.{field}")
        if segment["provider_total_tokens"] != (
            segment["provider_input_tokens"] + segment["provider_output_tokens"]
        ):
            raise ResumableShardError(
                f"{label}.segment provider token usage does not close"
            )
        _require_finite(
            segment.get("provider_latency_s"),
            f"{label}.segment_receipt.provider_latency_s",
        )
    return row


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ResumableShardError(
            f"{label} fields mismatch: missing={sorted(expected - set(value))!r}, "
            f"extra={sorted(set(value) - expected)!r}"
        )


def _safe_relative(value: Any, label: str) -> str:
    text = _require_text(value, label)
    if "\\" in text or ":" in text or "\x00" in text:
        raise ResumableShardError(
            f"{label} must not contain backslashes, drive/ADS colons, or NUL"
        )
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    parts = posix.parts
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ResumableShardError(f"{label} must be a normalized relative path")
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    for part in parts:
        stem = part.split(".", 1)[0].casefold()
        if part.endswith((" ", ".")) or stem in reserved:
            raise ResumableShardError(f"{label} contains a Windows-unsafe name")
    if posix.as_posix() != text:
        raise ResumableShardError(f"{label} must use normalized POSIX separators")
    return text


def _lexical_absolute(value: str | os.PathLike[str]) -> Path:
    """Return an absolute path without resolving links or junctions."""

    return Path(os.path.abspath(os.fspath(value)))


def _path_identity_sha256(value: str | os.PathLike[str]) -> str:
    text = _lexical_absolute(value).as_posix()
    if os.name == "nt":
        text = text.casefold()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_reparse_or_link(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    is_junction = getattr(path, "is_junction", lambda: False)
    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or is_junction()
        or (
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    )


def _assert_no_link_ancestors(
    value: str | os.PathLike[str],
    *,
    label: str,
    require_final: bool,
) -> Path:
    """Inspect the lexical path before any resolving operation.

    On Windows, ``Path.resolve`` follows junctions.  Walking the absolute
    lexical components with ``lstat`` first prevents a caller from smuggling
    the state or snapshot root through a symlink/reparse boundary.
    """

    path = _lexical_absolute(value)
    if require_final and not path.exists():
        raise ResumableShardError(f"{label} does not exist")
    chain = tuple(reversed(path.parents)) + (path,)
    for component in chain:
        if component.exists() and _is_reparse_or_link(component):
            raise ResumableShardError(f"{label} traverses a link/reparse point")
    return path


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability barrier on supported platforms."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Windows does not generally expose directory handles through os.open.
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class ResumePlan:
    """Frozen identity and operation budget for one resumable shard."""

    authorization_sha256: str
    mem0_policy_sha256: str
    source_validation_policy_sha256: str
    source_implementation_sha256: str
    source_environment_lock_sha256: str
    mem0_tool_implementation_sha256: str
    mem0_environment_lock_sha256: str
    sample_offset: int
    sample_sha256: str
    raw_history_bundle_sha256: str
    ordered_batch_sha256s: tuple[str, ...]
    authorized_add_operations: int
    authorized_extraction_calls: int
    authorized_search_operations: int
    user_scope: str

    def __post_init__(self) -> None:
        for field in (
            "authorization_sha256",
            "mem0_policy_sha256",
            "source_validation_policy_sha256",
            "source_implementation_sha256",
            "source_environment_lock_sha256",
            "mem0_tool_implementation_sha256",
            "mem0_environment_lock_sha256",
            "sample_sha256",
            "raw_history_bundle_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        _require_count(self.sample_offset, "sample_offset")
        adds = _require_count(
            self.authorized_add_operations,
            "authorized_add_operations",
            minimum=1,
        )
        extraction = _require_count(
            self.authorized_extraction_calls,
            "authorized_extraction_calls",
            minimum=1,
        )
        _require_count(
            self.authorized_search_operations,
            "authorized_search_operations",
            minimum=1,
        )
        if adds != extraction:
            raise ResumableShardError(
                "one infer=True extraction call is required per add"
            )
        if len(self.ordered_batch_sha256s) != adds:
            raise ResumableShardError("ordered batch count differs from add budget")
        for index, digest in enumerate(self.ordered_batch_sha256s):
            _require_sha256(digest, f"ordered_batch_sha256s[{index}]")
        if len(set(self.ordered_batch_sha256s)) != len(
            self.ordered_batch_sha256s
        ):
            # Identical message slices are valid, but the receipt hash also binds
            # ordinal in production.  Requiring distinct row hashes here would
            # reject real repeated content, so only the tuple position is trusted.
            pass
        scope = _require_text(self.user_scope, "user_scope")
        if len(scope) > 256 or any(ord(char) < 32 for char in scope):
            raise ResumableShardError("user_scope contains unsafe characters")

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": RESUME_PLAN_FORMAT,
            "authorization_sha256": self.authorization_sha256,
            "mem0_policy_sha256": self.mem0_policy_sha256,
            "source_validation_policy_sha256": (
                self.source_validation_policy_sha256
            ),
            "source_implementation_sha256": self.source_implementation_sha256,
            "source_environment_lock_sha256": (
                self.source_environment_lock_sha256
            ),
            "mem0_tool_implementation_sha256": (
                self.mem0_tool_implementation_sha256
            ),
            "mem0_environment_lock_sha256": (
                self.mem0_environment_lock_sha256
            ),
            "sample_offset": self.sample_offset,
            "sample_sha256": self.sample_sha256,
            "raw_history_bundle_sha256": self.raw_history_bundle_sha256,
            "ordered_batch_sha256s": list(self.ordered_batch_sha256s),
            "ordered_batches_sha256": canonical_json_sha256(
                list(self.ordered_batch_sha256s)
            ),
            "authorized_add_operations": self.authorized_add_operations,
            "authorized_extraction_calls": self.authorized_extraction_calls,
            "authorized_search_operations": self.authorized_search_operations,
            "user_scope": self.user_scope,
            "user_scope_sha256": hashlib.sha256(
                self.user_scope.encode("utf-8")
            ).hexdigest(),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.as_dict())


def deterministic_user_scope(authorization_sha256: str) -> str:
    digest = _require_sha256(authorization_sha256, "authorization_sha256")
    return f"longmemeval:resumable:{digest[:32]}"


def _entry(kind: str, sequence: int, previous: str | None, **payload: Any) -> dict[str, Any]:
    if kind not in _KINDS:
        raise ResumableShardError(f"unknown journal entry kind {kind!r}")
    _require_count(sequence, "journal sequence")
    if sequence == 0:
        if previous is not None:
            raise ResumableShardError("journal header cannot have a predecessor")
    else:
        _require_sha256(previous, "previous_entry_sha256")
    body = {
        "format": RESUME_JOURNAL_FORMAT,
        "kind": kind,
        "sequence": sequence,
        "previous_entry_sha256": previous,
        **_strict_json(payload, "journal payload"),
    }
    return {**body, "entry_sha256": canonical_json_sha256(body)}


def _line_bytes(entry: Mapping[str, Any]) -> bytes:
    return _canonical_json(entry) + b"\n"


def _validate_entry_digest(entry: Mapping[str, Any], index: int) -> str:
    value = dict(entry)
    digest = value.pop("entry_sha256", None)
    _require_sha256(digest, f"journal[{index}].entry_sha256")
    if canonical_json_sha256(value) != digest:
        raise ResumableShardError(f"journal[{index}] entry digest mismatch")
    return digest


@dataclass(frozen=True, slots=True)
class ReplayState:
    plan: ResumePlan
    entries: tuple[Mapping[str, Any], ...]
    generation: int
    committed_prefix: int
    sealed_prefix: int
    latest_prefix_seal: Mapping[str, Any] | None
    pending_intent: Mapping[str, Any] | None
    pending_send_attempt: Mapping[str, Any] | None
    terminal_search: Mapping[str, Any] | None
    active_state_removed: Mapping[str, Any] | None
    terminal_published: Mapping[str, Any] | None
    checkpoint_gc: Mapping[str, Any] | None
    cleanup_closed: Mapping[str, Any] | None
    commits: tuple[Mapping[str, Any], ...]

    @property
    def resume_safe(self) -> bool:
        return (
            self.cleanup_closed is None
            and self.terminal_search is None
            and self.pending_intent is None
            and self.committed_prefix == self.sealed_prefix
            and (
                self.latest_prefix_seal is not None
                or self.committed_prefix == 0
            )
        )

    @property
    def requires_rollback(self) -> bool:
        return (
            self.pending_intent is not None
            and self.pending_send_attempt is None
            and self.committed_prefix == self.sealed_prefix
            and (
                self.latest_prefix_seal is not None
                or self.committed_prefix == 0
            )
        )

    @property
    def checkpoint_authority_sha256(self) -> str:
        if self.latest_prefix_seal is not None:
            return _require_sha256(
                self.latest_prefix_seal.get("snapshot_authority_sha256"),
                "snapshot_authority_sha256",
            )
        return _require_sha256(
            self.entries[0].get("empty_prefix_authority_sha256"),
            "empty_prefix_authority_sha256",
        )

    @property
    def externally_ambiguous(self) -> bool:
        return self.pending_send_attempt is not None or (
            self.committed_prefix > self.sealed_prefix
        )

    def require_resumable(self) -> None:
        if self.externally_ambiguous:
            raise ResumeAmbiguityError(
                "journal contains provider work beyond the latest immutable prefix"
            )
        if not self.resume_safe:
            raise ResumableShardError("journal is not at a sealed resumable prefix")


@dataclass(frozen=True, slots=True)
class SuffixCounterSeed:
    """Full-population cumulative counters installed before suffix callables."""

    committed_prefix: int
    authorized_total: int
    remaining: int
    logical_attempted: int
    logical_completed: int
    http_attempted: int
    http_completed: int
    infer_true_adds_started: int
    infer_true_adds_exactly_one_call: int
    failed: int
    rejected: int
    zero_remaining_send_denied: bool

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(
            {field: getattr(self, field) for field in self.__dataclass_fields__}
        )


def suffix_counter_seed(state: ReplayState) -> SuffixCounterSeed:
    state.require_resumable()
    prefix = state.committed_prefix
    total = state.plan.authorized_extraction_calls
    if prefix > total:
        raise ResumableShardError("resume prefix exceeds the extraction budget")
    if prefix:
        latest = state.commits[-1]
        prior_seals = [
            entry
            for entry in state.entries
            if entry.get("kind") == "prefix_sealed"
            and int(entry.get("committed_prefix", -1)) < prefix
        ]
        segment_seeded_prefix = (
            int(prior_seals[-1]["committed_prefix"]) if prior_seals else 0
        )
        _observed_counter_receipt(
            latest["logical_meter_receipt"],
            label="sealed logical meter",
            expected=prefix,
            logical=True,
        )
        _observed_counter_receipt(
            latest["transport_receipt"],
            label="sealed transport",
            expected=prefix,
            logical=False,
            authorized_total=total,
            expected_seeded_prefix=segment_seeded_prefix,
        )
    remaining = total - prefix
    return SuffixCounterSeed(
        committed_prefix=prefix,
        authorized_total=total,
        remaining=remaining,
        logical_attempted=prefix,
        logical_completed=prefix,
        http_attempted=prefix,
        http_completed=prefix,
        infer_true_adds_started=prefix,
        infer_true_adds_exactly_one_call=prefix,
        failed=0,
        rejected=0,
        zero_remaining_send_denied=(remaining == 0),
    )


def _ledger_projection(
    *,
    user_scope: str,
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "user_scope": user_scope,
            "memory_id": memory_id,
            "source_refs": [_source_ref(ref, "ledger source ref") for ref in refs],
        }
        for memory_id, refs in sorted(ledger.items())
    ]


def _derive_rehydration_material(
    commits: Sequence[Mapping[str, Any]],
    *,
    plan: ResumePlan,
) -> dict[str, Any]:
    recent: list[dict[str, Any]] = []
    ledger: dict[str, list[dict[str, Any]]] = {}
    response_ids_by_add: list[list[str]] = []
    scope_protocol: bool | None = None
    adapter_stats: dict[str, Any] | None = None
    for ordinal, commit in enumerate(commits):
        if commit.get("ordinal") != ordinal:
            raise ResumableShardError("rehydration commits are not contiguous")
        source_ref = _source_ref(commit.get("source_ref"), "commit source_ref")
        prior_unique = _merge_source_refs(recent)
        expected_window = _merge_source_refs(prior_unique, (source_ref,))
        observed_window = _source_refs(
            commit.get("request_window_refs"), "commit request_window_refs"
        )
        if observed_window != expected_window:
            raise ResumableShardError(
                f"commit {ordinal} request-window reconstruction mismatch"
            )
        if canonical_json_sha256(observed_window) != commit.get(
            "request_window_sha256"
        ):
            raise ResumableShardError(f"commit {ordinal} request-window digest mismatch")
        ids = commit.get("returned_memory_ids")
        if not isinstance(ids, list) or any(
            not isinstance(memory_id, str) or not memory_id.strip()
            for memory_id in ids
        ):
            raise ResumableShardError(f"commit {ordinal} memory IDs are invalid")
        exact_ids = list(ids)
        response_ids_by_add.append(exact_ids)
        for memory_id in exact_ids:
            ledger[memory_id] = _merge_source_refs(
                ledger.get(memory_id, ()), expected_window
            )
        projection = _ledger_projection(user_scope=plan.user_scope, ledger=ledger)
        if canonical_json_sha256(projection) != commit.get(
            "ledger_projection_sha256"
        ):
            raise ResumableShardError(f"commit {ordinal} ledger projection mismatch")
        observed_protocol = commit.get("scope_protocol")
        if not isinstance(observed_protocol, bool):
            raise ResumableShardError("commit scope_protocol must be boolean")
        if scope_protocol is None:
            scope_protocol = observed_protocol
        elif observed_protocol != scope_protocol:
            raise ResumableShardError("commit scope protocol changed within a shard")
        if commit.get("user_scope") != plan.user_scope:
            raise ResumableShardError("commit user scope differs from the frozen plan")
        if commit.get("attribution_kind") != "request_window_non_evidence":
            raise ResumableShardError("commit attribution kind changed")
        if commit.get("date_exposure_kind") != "diagnostics_only_not_model_input":
            raise ResumableShardError("commit date exposure kind changed")
        if commit.get("request_window_messages") != _WINDOW_MESSAGES:
            raise ResumableShardError("commit request-window bound changed")
        stats = _adapter_stats(commit.get("adapter_stats"), "commit adapter_stats")
        if canonical_json_sha256(stats) != commit.get("adapter_stats_sha256"):
            raise ResumableShardError("commit adapter-stats digest mismatch")
        expected_adds = ordinal + 1
        if any(
            stats[field] != expected_adds
            for field in ("add_calls", "add_attempted_calls", "add_completed_calls")
        ) or stats["add_failed_calls"] != 0:
            raise ResumableShardError("commit adapter add-call accounting mismatch")
        if stats["add_returned_memories"] != sum(
            len(row) for row in response_ids_by_add
        ):
            raise ResumableShardError("commit returned-memory accounting mismatch")
        if stats["unique_ledger_memories"] != len(ledger):
            raise ResumableShardError("commit unique-ledger accounting mismatch")
        if stats["add_raw_message_tokens"] != sum(
            int(row["raw_message_tokens"]) for row in commits[: ordinal + 1]
        ):
            raise ResumableShardError("commit raw-message token accounting mismatch")
        if not math.isclose(
            stats["add_latency_s"],
            sum(float(row["add_latency_s"]) for row in commits[: ordinal + 1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ResumableShardError("commit add latency accounting mismatch")
        if any(
            stats[field] != 0
            for field in (
                "search_calls",
                "search_query_tokens",
                "search_raw_memory_tokens",
                "search_context_tokens",
                "search_prompt_token_proxy",
                "search_prompt_tokens",
                "search_returned_memories",
                "search_packed_memories",
                "released_scopes",
            )
        ) or stats["search_latency_s"] != 0.0:
            raise ResumableShardError("commit contains premature search/release stats")
        if adapter_stats is not None and (
            stats["token_counter_identity"]
            != adapter_stats["token_counter_identity"]
            or stats["token_counter_identity_verified"]
            != adapter_stats["token_counter_identity_verified"]
        ):
            raise ResumableShardError("token-counter identity changed within a shard")
        adapter_stats = stats
        recent.extend([source_ref] * source_ref["turn_count"])
        recent = recent[-_WINDOW_MESSAGES:]
    projection = _ledger_projection(user_scope=plan.user_scope, ledger=ledger)
    body = {
        "format": RESUME_REHYDRATION_FORMAT,
        "plan_sha256": plan.sha256,
        "committed_prefix": len(commits),
        "user_scope": plan.user_scope,
        "scope_protocol": scope_protocol,
        "request_window_messages": _WINDOW_MESSAGES,
        "request_window_deque": recent,
        "ordered_response_ids_by_add": response_ids_by_add,
        "ledger_projection": projection,
        "ledger_projection_sha256": canonical_json_sha256(projection),
        "adapter_stats": adapter_stats,
        "adapter_stats_sha256": (
            canonical_json_sha256(adapter_stats)
            if adapter_stats is not None
            else None
        ),
    }
    return {**body, "rehydration_sha256": canonical_json_sha256(body)}


def rehydration_material(state: ReplayState) -> dict[str, Any]:
    """Return the fully inspectable local adapter state for exact adoption."""

    return _derive_rehydration_material(state.commits, plan=state.plan)


def _plan_from_payload(value: Any) -> ResumePlan:
    if not isinstance(value, Mapping):
        raise ResumableShardError("journal header plan must be an object")
    plan = dict(value)
    _exact_keys(
        plan,
        {
            "format",
            "authorization_sha256",
            "mem0_policy_sha256",
            "source_validation_policy_sha256",
            "source_implementation_sha256",
            "source_environment_lock_sha256",
            "mem0_tool_implementation_sha256",
            "mem0_environment_lock_sha256",
            "sample_offset",
            "sample_sha256",
            "raw_history_bundle_sha256",
            "ordered_batch_sha256s",
            "ordered_batches_sha256",
            "authorized_add_operations",
            "authorized_extraction_calls",
            "authorized_search_operations",
            "user_scope",
            "user_scope_sha256",
        },
        "journal header plan",
    )
    if plan.pop("format") != RESUME_PLAN_FORMAT:
        raise ResumableShardError("resume-plan format mismatch")
    ordered_digest = plan.pop("ordered_batches_sha256")
    scope_digest = plan.pop("user_scope_sha256")
    ordered = plan.get("ordered_batch_sha256s")
    if not isinstance(ordered, list):
        raise ResumableShardError("ordered_batch_sha256s must be an array")
    if canonical_json_sha256(ordered) != ordered_digest:
        raise ResumableShardError("ordered batch aggregate digest mismatch")
    scope = plan.get("user_scope")
    if not isinstance(scope, str) or hashlib.sha256(scope.encode()).hexdigest() != scope_digest:
        raise ResumableShardError("user-scope digest mismatch")
    plan["ordered_batch_sha256s"] = tuple(ordered)
    return ResumePlan(**plan)


def new_journal_header(
    plan: ResumePlan,
    *,
    owned_state_path: str,
    snapshot_root_path: str,
    journal_path_sha256: str,
    snapshot_root_ownership_token: str,
) -> dict[str, Any]:
    state = _safe_relative(owned_state_path, "owned_state_path")
    snapshots = _safe_relative(snapshot_root_path, "snapshot_root_path")
    if state == snapshots or state.startswith(snapshots + "/") or snapshots.startswith(
        state + "/"
    ):
        raise ResumableShardError("state and snapshot paths must not overlap")
    token = _require_text(
        snapshot_root_ownership_token, "snapshot_root_ownership_token"
    )
    if not _TOKEN_RE.fullmatch(token):
        raise ResumableShardError(
            "snapshot_root_ownership_token must be 32 lowercase hex characters"
        )
    journal_sha = _require_sha256(journal_path_sha256, "journal_path_sha256")
    marker_body = {
        "format": RESUME_ROOT_MARKER_FORMAT,
        "plan_sha256": plan.sha256,
        "authorization_sha256": plan.authorization_sha256,
        "journal_path_sha256": journal_sha,
        "owned_state_path": state,
        "snapshot_root_path": snapshots,
        "snapshot_root_ownership_token": token,
    }
    marker = {**marker_body, "marker_sha256": canonical_json_sha256(marker_body)}
    empty_body = {
        "format": RESUME_SNAPSHOT_AUTHORITY_FORMAT,
        "plan_sha256": plan.sha256,
        "authorization_sha256": plan.authorization_sha256,
        "generation": 0,
        "committed_prefix": 0,
        "journal_header_sha256": "pending",
        "snapshot_root_marker_sha256": marker["marker_sha256"],
        "owned_state_absent": True,
        "cumulative_extraction_attempted": 0,
        "cumulative_extraction_completed": 0,
        "cumulative_http_attempted": 0,
        "cumulative_http_completed": 0,
        "failures": 0,
        "rejections": 0,
    }
    # The empty-prefix authority intentionally cannot include the eventual
    # header digest without forming a hash cycle.  It is instead embedded in
    # the authenticated header and binds every external plan/root identity.
    empty_body.pop("journal_header_sha256")
    body = {
        "plan": plan.as_dict(),
        "plan_sha256": plan.sha256,
        "owned_state_path": state,
        "snapshot_root_path": snapshots,
        "snapshot_root_ownership_token": token,
        "snapshot_root_marker_sha256": marker["marker_sha256"],
        "journal_path_sha256": journal_sha,
        "empty_prefix_authority_sha256": canonical_json_sha256(empty_body),
        "append_only": True,
        "provider_retries": 0,
        "resume_semantics": "checkpoint_authority_fail_closed_v2",
    }
    return _entry("header", 0, None, **body)


def _snapshot_root_marker_from_header(
    header: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "format": RESUME_ROOT_MARKER_FORMAT,
        "plan_sha256": _require_sha256(
            header.get("plan_sha256"), "header plan_sha256"
        ),
        "authorization_sha256": _require_sha256(
            dict(header.get("plan", {})).get("authorization_sha256"),
            "header authorization_sha256",
        ),
        "journal_path_sha256": _require_sha256(
            header.get("journal_path_sha256"), "header journal_path_sha256"
        ),
        "owned_state_path": _safe_relative(
            header.get("owned_state_path"), "header owned_state_path"
        ),
        "snapshot_root_path": _safe_relative(
            header.get("snapshot_root_path"), "header snapshot_root_path"
        ),
        "snapshot_root_ownership_token": _require_text(
            header.get("snapshot_root_ownership_token"),
            "header snapshot_root_ownership_token",
        ),
    }
    if not _TOKEN_RE.fullmatch(body["snapshot_root_ownership_token"]):
        raise ResumableShardError("header snapshot-root token is invalid")
    return {**body, "marker_sha256": canonical_json_sha256(body)}


def _empty_prefix_authority_from_header(
    header: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _plan_from_payload(header.get("plan"))
    body = {
        "format": RESUME_SNAPSHOT_AUTHORITY_FORMAT,
        "plan_sha256": plan.sha256,
        "authorization_sha256": plan.authorization_sha256,
        "generation": 0,
        "committed_prefix": 0,
        "snapshot_root_marker_sha256": _require_sha256(
            header.get("snapshot_root_marker_sha256"),
            "snapshot_root_marker_sha256",
        ),
        "owned_state_absent": True,
        "cumulative_extraction_attempted": 0,
        "cumulative_extraction_completed": 0,
        "cumulative_http_attempted": 0,
        "cumulative_http_completed": 0,
        "failures": 0,
        "rejections": 0,
    }
    return {**body, "authority_sha256": canonical_json_sha256(body)}


def _sealed_json_bytes(value: Mapping[str, Any]) -> tuple[bytes, str]:
    payload = _canonical_json(value) + b"\n"
    return payload, hashlib.sha256(payload).hexdigest()


def _publish_file_no_clobber(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        if os.write(descriptor, payload) != len(payload):
            raise ResumableShardError(f"short durable write for {path.name}")
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def publish_sealed_json(
    path: str | os.PathLike[str], value: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish or byte-verify one canonical JSON document plus sidecar.

    The data file is authoritative; a crash between its durable creation and
    sidecar creation is recoverable only when the existing bytes are exactly
    the requested canonical payload.  Different existing bytes are never
    replaced.
    """

    target = _assert_no_link_ancestors(
        path, label="sealed JSON target", require_final=False
    )
    if target == target.parent:
        raise ResumableShardError("sealed JSON target cannot be a filesystem root")
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _assert_no_link_ancestors(
        target, label="sealed JSON target", require_final=False
    )
    payload, digest = _sealed_json_bytes(_strict_json(value, "sealed JSON value"))
    sidecar = target.with_name(target.name + ".sha256")
    created = False
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise ResumableShardError(
                "refusing to replace a different sealed JSON artifact"
            )
    else:
        if sidecar.exists():
            raise ResumableShardError("sealed JSON sidecar exists without its artifact")
        _publish_file_no_clobber(target, payload)
        created = True
        _fsync_directory(target.parent)
    expected_sidecar = _sidecar_bytes(target, digest)
    if sidecar.exists():
        if (
            sidecar.is_symlink()
            or not sidecar.is_file()
            or sidecar.read_bytes() != expected_sidecar
        ):
            raise ResumableShardError("sealed JSON digest sidecar mismatch")
    else:
        _publish_file_no_clobber(sidecar, expected_sidecar)
        _fsync_directory(target.parent)
    return {
        "path": str(target),
        "sha256": digest,
        "bytes": len(payload),
        "created": created,
        "payload": json.loads(payload),
    }


def read_sealed_json(
    path: str | os.PathLike[str], *, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Read one exact canonical JSON+sidecar artifact without repairing it."""

    target = _assert_no_link_ancestors(
        path, label="sealed JSON artifact", require_final=True
    )
    if target.is_symlink() or not target.is_file():
        raise ResumableShardError("sealed JSON artifact is not a regular file")
    raw = target.read_bytes()
    if not raw.endswith(b"\n"):
        raise ResumableShardError("sealed JSON artifact is truncated")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumableShardError("sealed JSON artifact is not JSON") from exc
    if type(value) is not dict or raw != _canonical_json(value) + b"\n":
        raise ResumableShardError("sealed JSON artifact is not canonical")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256, "expected sealed JSON SHA"
    ):
        raise ResumableShardError("sealed JSON artifact digest mismatch")
    sidecar = target.with_name(target.name + ".sha256")
    if (
        sidecar.is_symlink()
        or not sidecar.is_file()
        or sidecar.read_bytes() != _sidecar_bytes(target, digest)
    ):
        raise ResumableShardError("sealed JSON artifact sidecar mismatch")
    return {
        "path": str(target),
        "sha256": digest,
        "bytes": len(raw),
        "payload": value,
    }


def _ensure_snapshot_root(
    *,
    journal_path: Path,
    header: Mapping[str, Any],
) -> Path:
    if _path_identity_sha256(journal_path) != header.get("journal_path_sha256"):
        raise ResumableShardError("journal path identity differs from header")
    root_relative = _safe_relative(
        header.get("snapshot_root_path"), "snapshot_root_path"
    )
    root = _assert_no_link_ancestors(
        journal_path.parent / Path(root_relative),
        label="snapshot root",
        require_final=False,
    )
    marker_value = _snapshot_root_marker_from_header(header)
    if marker_value["marker_sha256"] != header.get("snapshot_root_marker_sha256"):
        raise ResumableShardError("snapshot-root marker digest differs from header")
    root.parent.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        root.mkdir()
        _fsync_directory(root.parent)
    if not root.is_dir() or _is_reparse_or_link(root):
        raise ResumableShardError("snapshot root is not a plain owned directory")
    marker = root / SNAPSHOT_ROOT_MARKER
    sidecar = marker.with_name(marker.name + ".sha256")
    payload, digest = _sealed_json_bytes(marker_value)
    if not marker.exists() and not sidecar.exists():
        _publish_file_no_clobber(marker, payload)
        _publish_file_no_clobber(sidecar, _sidecar_bytes(marker, digest))
        _fsync_directory(root)
    if marker.is_symlink() or sidecar.is_symlink():
        raise ResumableShardError("snapshot-root marker cannot be a symlink")
    if marker.read_bytes() != payload or sidecar.read_bytes() != _sidecar_bytes(
        marker, digest
    ):
        raise ResumableShardError("snapshot-root marker identity mismatch")
    return root


def _parse_journal_bytes(raw: bytes) -> tuple[dict[str, Any], ...]:
    if not raw or not raw.endswith(b"\n"):
        raise ResumableShardError("journal is empty or has a truncated final line")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
            raise ResumableShardError(f"journal line {index} is truncated or empty")
        payload = line[:-1]
        if payload.endswith(b"\r"):
            raise ResumableShardError("journal must use LF line endings")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResumableShardError(f"journal line {index} is not JSON") from exc
        if type(value) is not dict or _canonical_json(value) != payload:
            raise ResumableShardError(f"journal line {index} is not canonical JSON")
        rows.append(value)
    return tuple(rows)


def read_journal(path: str | os.PathLike[str]) -> tuple[dict[str, Any], ...]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ResumableShardError("resume journal must be a regular file")
    projected = _parse_journal_bytes(target.read_bytes())
    records = target.with_name(target.name + ".records")
    if records.exists():
        authoritative = _read_record_segments(target)
        if tuple(projected) != tuple(authoritative):
            raise ResumableShardError(
                "journal JSONL projection differs from atomic record chain"
            )
        return authoritative
    return projected


def _record_root_marker(
    *, journal_path: Path, header: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "format": RESUME_RECORD_ROOT_FORMAT,
        "plan_sha256": header["plan_sha256"],
        "journal_path_sha256": header["journal_path_sha256"],
        "snapshot_root_ownership_token": header[
            "snapshot_root_ownership_token"
        ],
        "header_entry_sha256": header["entry_sha256"],
    }
    return {**body, "marker_sha256": canonical_json_sha256(body)}


def _record_root(path: Path) -> Path:
    return path.with_name(path.name + ".records")


def _initialize_record_root(path: Path, header: Mapping[str, Any]) -> Path:
    root = _assert_no_link_ancestors(
        _record_root(path), label="journal record root", require_final=False
    )
    if root.exists():
        raise FileExistsError("journal atomic-record root already exists")
    root.mkdir()
    marker = root / "root-marker.json"
    value = _record_root_marker(journal_path=path, header=header)
    payload, digest = _sealed_json_bytes(value)
    _publish_file_no_clobber(marker, payload)
    _publish_file_no_clobber(
        marker.with_name(marker.name + ".sha256"),
        _sidecar_bytes(marker, digest),
    )
    _fsync_directory(root)
    _fsync_directory(root.parent)
    return root


def _verify_record_root(path: Path, header: Mapping[str, Any]) -> Path:
    root = _assert_no_link_ancestors(
        _record_root(path), label="journal record root", require_final=True
    )
    if not root.is_dir():
        raise ResumableShardError("journal record root is not a directory")
    marker = root / "root-marker.json"
    sidecar = marker.with_name(marker.name + ".sha256")
    expected = _record_root_marker(journal_path=path, header=header)
    payload, digest = _sealed_json_bytes(expected)
    if (
        marker.read_bytes() != payload
        or sidecar.read_bytes() != _sidecar_bytes(marker, digest)
    ):
        raise ResumableShardError("journal record-root marker mismatch")
    return root


def _record_name(entry: Mapping[str, Any]) -> str:
    return f"{int(entry['sequence']):08d}-{entry['kind']}.jsonl"


def _publish_record(path: Path, entry: Mapping[str, Any]) -> None:
    root = _verify_record_root(path, entry if entry.get("kind") == "header" else _read_record_segments(path)[0])
    target = root / _record_name(entry)
    payload = _line_bytes(entry)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if stream.write(payload) != len(payload):
                raise ResumableShardError("atomic journal record write was short")
            stream.flush()
            os.fsync(stream.fileno())
        _rename_directory_no_clobber(temporary, target)
        _fsync_directory(root)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_record_segments(path: Path) -> tuple[dict[str, Any], ...]:
    root = _assert_no_link_ancestors(
        _record_root(path), label="journal record root", require_final=True
    )
    candidates = sorted(root.glob("*.jsonl"), key=lambda item: item.name)
    if not candidates:
        raise ResumableShardError("journal record root has no header segment")
    rows: list[dict[str, Any]] = []
    for expected_sequence, candidate in enumerate(candidates):
        if _is_reparse_or_link(candidate) or not candidate.is_file():
            raise ResumableShardError("journal record segment is not regular")
        parsed = _parse_journal_bytes(candidate.read_bytes())
        if len(parsed) != 1:
            raise ResumableShardError("journal record segment has multiple entries")
        entry = parsed[0]
        if entry.get("sequence") != expected_sequence:
            raise ResumableShardError("journal record segment sequence has a gap")
        if candidate.name != _record_name(entry):
            raise ResumableShardError("journal record filename differs from entry")
        rows.append(entry)
    _verify_record_root(path, rows[0])
    return tuple(rows)


def _validate_common_chain(entries: Sequence[Mapping[str, Any]]) -> None:
    if not entries:
        raise ResumableShardError("journal has no header")
    previous: str | None = None
    for index, entry in enumerate(entries):
        if entry.get("format") != RESUME_JOURNAL_FORMAT:
            raise ResumableShardError(f"journal[{index}] format mismatch")
        if entry.get("kind") not in _KINDS:
            raise ResumableShardError(f"journal[{index}] kind is invalid")
        if entry.get("sequence") != index:
            raise ResumableShardError(f"journal[{index}] sequence mismatch")
        if entry.get("previous_entry_sha256") != previous:
            raise ResumableShardError(f"journal[{index}] predecessor mismatch")
        kind = str(entry.get("kind"))
        _exact_keys(
            entry,
            {
                "format",
                "kind",
                "sequence",
                "previous_entry_sha256",
                "entry_sha256",
                *_ENTRY_PAYLOAD_KEYS[kind],
            },
            f"journal[{index}] {kind}",
        )
        previous = _validate_entry_digest(entry, index)


def replay_journal(
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_plan: ResumePlan | None = None,
) -> ReplayState:
    """Replay and authenticate every transition in an append-only journal."""

    _validate_common_chain(entries)
    header = dict(entries[0])
    if header.get("kind") != "header":
        raise ResumableShardError("journal entry zero must be the header")
    plan = _plan_from_payload(header["plan"])
    if header.get("plan_sha256") != plan.sha256:
        raise ResumableShardError("journal plan digest mismatch")
    if expected_plan is not None and plan != expected_plan:
        raise ResumableShardError("journal plan differs from reconstructed plan")
    _safe_relative(header.get("owned_state_path"), "owned_state_path")
    _safe_relative(header.get("snapshot_root_path"), "snapshot_root_path")
    marker = _snapshot_root_marker_from_header(header)
    if marker["marker_sha256"] != header.get("snapshot_root_marker_sha256"):
        raise ResumableShardError("journal snapshot-root marker digest mismatch")
    empty = _empty_prefix_authority_from_header(header)
    if empty["authority_sha256"] != header.get("empty_prefix_authority_sha256"):
        raise ResumableShardError("journal empty-prefix authority digest mismatch")
    if header.get("append_only") is not True or header.get("provider_retries") != 0:
        raise ResumableShardError("journal header retry/append-only policy mismatch")
    if header.get("resume_semantics") != "checkpoint_authority_fail_closed_v2":
        raise ResumableShardError("journal resume semantics mismatch")

    generation = 0
    committed_prefix = 0
    sealed_prefix = 0
    latest_seal: Mapping[str, Any] | None = None
    pending_intent: Mapping[str, Any] | None = None
    pending_send: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    active_removed: Mapping[str, Any] | None = None
    published: Mapping[str, Any] | None = None
    checkpoint_gc: Mapping[str, Any] | None = None
    cleanup: Mapping[str, Any] | None = None
    active_commits: list[Mapping[str, Any]] = []
    commits_by_generation: dict[int, list[Mapping[str, Any]]] = {0: active_commits}

    for index, raw_entry in enumerate(entries[1:], start=1):
        entry = dict(raw_entry)
        kind = entry["kind"]
        if cleanup is not None:
            raise ResumableShardError("nothing may follow cleanup closure")
        if checkpoint_gc is not None and kind != "cleanup_closed":
            raise ResumableShardError("only final closure may follow checkpoint GC")
        if (
            published is not None
            and checkpoint_gc is None
            and kind != "checkpoint_gc"
        ):
            raise ResumableShardError("only checkpoint GC may follow publication")
        if (
            active_removed is not None
            and published is None
            and kind != "terminal_published"
        ):
            raise ResumableShardError(
                "only terminal publication may follow active-state removal"
            )
        if terminal is not None and active_removed is None and kind != "active_state_removed":
            raise ResumableShardError(
                "only active-state removal may follow terminal staging"
            )
        entry_generation = _require_count(
            entry.get("generation"), f"journal[{index}].generation"
        )
        if entry_generation != generation:
            raise ResumableShardError(f"journal[{index}] generation mismatch")

        if kind == "intent":
            if pending_intent is not None:
                raise ResumableShardError("a second intent appeared before commit")
            if committed_prefix != sealed_prefix and latest_seal is not None:
                # Multiple adds per clean prefix are allowed; they are simply
                # non-resumable until the next prefix seal.
                pass
            ordinal = _require_count(entry.get("ordinal"), "intent ordinal")
            if ordinal != committed_prefix:
                raise ResumableShardError("intent ordinal is not the next add")
            if ordinal >= plan.authorized_add_operations:
                raise ResumableShardError("intent exceeds the authorized add budget")
            if entry.get("batch_sha256") != plan.ordered_batch_sha256s[ordinal]:
                raise ResumableShardError("intent batch digest mismatch")
            if entry.get("committed_prefix_before") != committed_prefix:
                raise ResumableShardError("intent prefix receipt mismatch")
            _require_sha256(entry.get("session_sha256"), "intent session_sha256")
            pending_intent = MappingProxyType(entry)
            pending_send = None
        elif kind == "send_attempt":
            if pending_intent is None or pending_send is not None:
                raise ResumableShardError("send attempt has no unique pending intent")
            if entry.get("ordinal") != pending_intent.get("ordinal"):
                raise ResumableShardError("send attempt ordinal mismatch")
            if entry.get("intent_entry_sha256") != pending_intent.get("entry_sha256"):
                raise ResumableShardError("send attempt intent receipt mismatch")
            if entry.get("global_attempt_ordinal") != committed_prefix:
                raise ResumableShardError("send attempt global count mismatch")
            _require_sha256(entry.get("request_sha256"), "request_sha256")
            expected_checkpoint = (
                latest_seal.get("snapshot_authority_sha256")
                if latest_seal is not None
                else header.get("empty_prefix_authority_sha256")
            )
            if entry.get("prior_checkpoint_authority_sha256") != expected_checkpoint:
                raise ResumableShardError(
                    "send attempt prior-checkpoint authority mismatch"
                )
            pending_send = MappingProxyType(entry)
        elif kind == "commit":
            if pending_intent is None or pending_send is None:
                raise ResumableShardError("commit requires intent and send attempt")
            ordinal = pending_intent["ordinal"]
            if entry.get("ordinal") != ordinal:
                raise ResumableShardError("commit ordinal mismatch")
            if entry.get("intent_entry_sha256") != pending_intent["entry_sha256"]:
                raise ResumableShardError("commit intent receipt mismatch")
            if entry.get("send_attempt_entry_sha256") != pending_send["entry_sha256"]:
                raise ResumableShardError("commit send receipt mismatch")
            if entry.get("batch_sha256") != plan.ordered_batch_sha256s[ordinal]:
                raise ResumableShardError("commit batch digest mismatch")
            _require_sha256(entry.get("response_sha256"), "response_sha256")
            _require_sha256(
                entry.get("request_window_sha256"), "request_window_sha256"
            )
            memory_ids = entry.get("returned_memory_ids")
            if (
                not isinstance(memory_ids, list)
                or any(
                    not isinstance(memory_id, str) or not memory_id.strip()
                    for memory_id in memory_ids
                )
            ):
                raise ResumableShardError("commit memory IDs are invalid")
            _source_ref(entry.get("source_ref"), "commit source_ref")
            _source_refs(entry.get("request_window_refs"), "request_window_refs")
            if entry.get("user_scope") != plan.user_scope:
                raise ResumableShardError("commit user scope mismatch")
            if not isinstance(entry.get("scope_protocol"), bool):
                raise ResumableShardError("commit scope_protocol must be boolean")
            if entry.get("request_window_messages") != _WINDOW_MESSAGES:
                raise ResumableShardError("commit request window changed")
            stats = _adapter_stats(entry.get("adapter_stats"), "adapter_stats")
            if canonical_json_sha256(stats) != entry.get("adapter_stats_sha256"):
                raise ResumableShardError("commit adapter-stats receipt mismatch")
            _require_finite(entry.get("add_latency_s"), "add_latency_s")
            _require_count(entry.get("raw_message_tokens"), "raw_message_tokens")
            expected_total = ordinal + 1
            logical_receipt = _observed_counter_receipt(
                entry.get("logical_meter_receipt"),
                label="logical_meter_receipt",
                expected=expected_total,
                logical=True,
            )
            transport_receipt = _observed_counter_receipt(
                entry.get("transport_receipt"),
                label="transport_receipt",
                expected=expected_total,
                logical=False,
                authorized_total=plan.authorized_extraction_calls,
                expected_seeded_prefix=sealed_prefix,
            )
            if canonical_json_sha256(logical_receipt) != entry.get(
                "logical_meter_receipt_sha256"
            ):
                raise ResumableShardError("logical-meter receipt digest mismatch")
            if canonical_json_sha256(transport_receipt) != entry.get(
                "transport_receipt_sha256"
            ):
                raise ResumableShardError("transport receipt digest mismatch")
            for field in (
                "cumulative_logical_attempted",
                "cumulative_logical_completed",
                "cumulative_http_attempted",
                "cumulative_http_completed",
            ):
                if entry.get(field) != expected_total:
                    raise ResumableShardError(f"commit {field} mismatch")
            for field in (
                "cumulative_logical_failed",
                "cumulative_logical_rejected",
                "cumulative_http_failed",
                "cumulative_http_rejected",
            ):
                if entry.get(field) != 0:
                    raise ResumableShardError(f"commit {field} must be zero")
            active_commits.append(MappingProxyType(entry))
            committed_prefix += 1
            pending_intent = None
            pending_send = None
        elif kind == "prefix_sealed":
            if pending_intent is not None:
                raise ResumableShardError("cannot seal a prefix with a pending add")
            if entry.get("committed_prefix") != committed_prefix:
                raise ResumableShardError("prefix seal count mismatch")
            if committed_prefix <= sealed_prefix:
                raise ResumableShardError("prefix seal must advance")
            if entry.get("active_commit_entry_sha256") != active_commits[-1].get(
                "entry_sha256"
            ):
                raise ResumableShardError("prefix seal commit receipt mismatch")
            for field in (
                "snapshot_manifest_sha256",
                "snapshot_tree_sha256",
                "ownership_token_sha256",
                "handles_closed_receipt_sha256",
                "transport_closure_receipt_sha256",
                "write_usage_attestation_sha256",
            ):
                _require_sha256(entry.get(field), f"prefix seal {field}")
            usage = entry.get("write_usage_attestation")
            if not isinstance(usage, dict):
                raise ResumableShardError(
                    "prefix seal omitted write-usage attestation"
                )
            usage_body = dict(usage)
            usage_sha = usage_body.pop("receipt_sha256", None)
            if (
                usage_sha != canonical_json_sha256(usage_body)
                or usage_sha != entry.get("write_usage_attestation_sha256")
                or usage.get("transport_closure_receipt_sha256")
                != entry.get("transport_closure_receipt_sha256")
            ):
                raise ResumableShardError(
                    "prefix seal write-usage receipt changed"
                )
            snapshot_path = _safe_relative(
                entry.get("snapshot_path"), "prefix snapshot_path"
            )
            expected_snapshot_path = (
                f"{header['snapshot_root_path']}/prefix-{committed_prefix:06d}"
            )
            if snapshot_path != expected_snapshot_path:
                raise ResumableShardError("prefix snapshot path is not plan-bound")
            for field in (
                "snapshot_authority_sha256",
                "snapshot_authority_artifact_sha256",
                "snapshot_receipt_sha256",
                "rehydration_sha256",
            ):
                _require_sha256(entry.get(field), f"prefix seal {field}")
            expected_rehydration = _derive_rehydration_material(
                active_commits, plan=plan
            )
            if entry.get("rehydration_sha256") != expected_rehydration[
                "rehydration_sha256"
            ]:
                raise ResumableShardError("prefix seal rehydration receipt mismatch")
            for field in (
                "cumulative_extraction_attempted",
                "cumulative_extraction_completed",
                "cumulative_http_attempted",
                "cumulative_http_completed",
            ):
                if entry.get(field) != committed_prefix:
                    raise ResumableShardError(f"prefix seal {field} mismatch")
            if entry.get("failures") != 0 or entry.get("rejections") != 0:
                raise ResumableShardError("prefix seal has failed/rejected calls")
            sealed_prefix = committed_prefix
            latest_seal = MappingProxyType(entry)
        elif kind == "rollback":
            if pending_intent is None or pending_send is not None:
                raise ResumableShardError(
                    "rollback is allowed only for a provably pre-send intent"
                )
            if committed_prefix != sealed_prefix:
                raise ResumableShardError("rollback has no clean sealed prefix")
            if entry.get("intent_entry_sha256") != pending_intent["entry_sha256"]:
                raise ResumableShardError("rollback intent receipt mismatch")
            if entry.get("restore_prefix") != sealed_prefix:
                raise ResumableShardError("rollback prefix receipt mismatch")
            expected_authority = (
                latest_seal.get("snapshot_authority_sha256")
                if latest_seal is not None
                else header.get("empty_prefix_authority_sha256")
            )
            if entry.get("restore_authority_sha256") != expected_authority:
                raise ResumableShardError("rollback authority receipt mismatch")
            _require_sha256(
                entry.get("restore_receipt_sha256"), "rollback restore receipt"
            )
            expected_tree = (
                latest_seal.get("snapshot_tree_sha256")
                if latest_seal is not None
                else canonical_json_sha256({"owned_state_absent": True})
            )
            if entry.get("restored_snapshot_tree_sha256") != expected_tree:
                raise ResumableShardError("rollback snapshot receipt mismatch")
            if entry.get("next_generation") != generation + 1:
                raise ResumableShardError("rollback generation did not advance")
            pending_intent = None
            pending_send = None
            generation += 1
            active_commits = list(active_commits[:sealed_prefix])
            commits_by_generation[generation] = active_commits
        elif kind == "terminal_search":
            if pending_intent is not None:
                raise ResumableShardError("terminal search has a pending add")
            if committed_prefix != plan.authorized_add_operations:
                raise ResumableShardError("terminal search occurred before full ingest")
            if sealed_prefix != committed_prefix or latest_seal is None:
                raise ResumableShardError("terminal search requires a sealed full prefix")
            if entry.get("completed_search_operations") != (
                plan.authorized_search_operations
            ):
                raise ResumableShardError("terminal search count mismatch")
            for field in (
                "terminal_stage_sha256",
                "terminal_result_sha256",
                "terminal_trace_sha256",
            ):
                _require_sha256(entry.get(field), field)
            _safe_relative(entry.get("terminal_stage_path"), "terminal stage path")
            if entry.get("full_checkpoint_authority_sha256") != (
                latest_seal.get("snapshot_authority_sha256")
            ):
                raise ResumableShardError("terminal stage checkpoint changed")
            if entry.get("extraction_calls_closed") is not True:
                raise ResumableShardError("terminal stage extraction cap is open")
            if entry.get("provider_retries") != 0:
                raise ResumableShardError("terminal stage provider retries changed")
            if entry.get("transport_closure_receipt_sha256") != (
                latest_seal.get("transport_closure_receipt_sha256")
            ):
                raise ResumableShardError(
                    "terminal transport-closure receipt changed"
                )
            if entry.get("write_usage_attestation_sha256") != (
                latest_seal.get("write_usage_attestation_sha256")
            ):
                raise ResumableShardError(
                    "terminal write-usage attestation changed"
                )
            terminal = MappingProxyType(entry)
        elif kind == "active_state_removed":
            if terminal is None or active_removed is not None:
                raise ResumableShardError("active-state removal is out of order")
            if entry.get("terminal_search_entry_sha256") != terminal.get(
                "entry_sha256"
            ):
                raise ResumableShardError("state-removal terminal receipt mismatch")
            _require_sha256(
                entry.get("state_removal_receipt_sha256"),
                "state-removal receipt",
            )
            if entry.get("owned_state_removed") is not True:
                raise ResumableShardError("active working state was not removed")
            if entry.get("snapshots_retained") is not True:
                raise ResumableShardError(
                    "full checkpoint was removed before output publication"
                )
            active_removed = MappingProxyType(entry)
        elif kind == "terminal_published":
            if terminal is None or active_removed is None or published is not None:
                raise ResumableShardError("terminal publication is out of order")
            if entry.get("terminal_search_entry_sha256") != terminal.get(
                "entry_sha256"
            ):
                raise ResumableShardError("publication terminal receipt mismatch")
            for field in (
                "official_artifact_sha256",
                "official_trace_sha256",
                "publication_receipt_sha256",
            ):
                _require_sha256(entry.get(field), field)
            _safe_relative(
                entry.get("official_artifact_path"), "official artifact path"
            )
            _safe_relative(entry.get("official_trace_path"), "official trace path")
            if entry.get("outputs_verified") is not True:
                raise ResumableShardError("terminal outputs were not verified")
            published = MappingProxyType(entry)
        elif kind == "checkpoint_gc":
            if published is None or active_removed is None or checkpoint_gc is not None:
                raise ResumableShardError("checkpoint GC is out of order")
            if entry.get("terminal_published_entry_sha256") != published.get(
                "entry_sha256"
            ) or entry.get("active_state_removed_entry_sha256") != active_removed.get(
                "entry_sha256"
            ):
                raise ResumableShardError("checkpoint GC predecessor mismatch")
            _require_sha256(
                entry.get("checkpoint_gc_receipt_sha256"),
                "checkpoint GC receipt",
            )
            if entry.get("snapshots_removed") is not True:
                raise ResumableShardError("checkpoint GC retained snapshots")
            if entry.get("terminal_stage_removed") is not True:
                raise ResumableShardError("checkpoint GC retained terminal staging")
            checkpoint_gc = MappingProxyType(entry)
        elif kind == "cleanup_closed":
            if checkpoint_gc is None:
                raise ResumableShardError("cleanup occurred before checkpoint GC")
            if entry.get("checkpoint_gc_entry_sha256") != checkpoint_gc.get(
                "entry_sha256"
            ):
                raise ResumableShardError("cleanup checkpoint-GC receipt mismatch")
            if entry.get("owned_state_removed") is not True:
                raise ResumableShardError("cleanup did not remove owned state")
            if entry.get("snapshots_removed") is not True:
                raise ResumableShardError("cleanup did not remove snapshots")
            if entry.get("terminal_stage_removed") is not True:
                raise ResumableShardError("cleanup retained terminal staging")
            if entry.get("official_outputs_retained") is not True:
                raise ResumableShardError("cleanup removed official outputs")
            _require_sha256(
                entry.get("cleanup_receipt_sha256"), "cleanup receipt SHA-256"
            )
            cleanup = MappingProxyType(entry)
        elif kind == "header":
            raise ResumableShardError("journal contains a second header")

    _derive_rehydration_material(active_commits, plan=plan)
    replayed = ReplayState(
        plan=plan,
        entries=tuple(MappingProxyType(dict(row)) for row in entries),
        generation=generation,
        committed_prefix=committed_prefix,
        sealed_prefix=sealed_prefix,
        latest_prefix_seal=latest_seal,
        pending_intent=pending_intent,
        pending_send_attempt=pending_send,
        terminal_search=terminal,
        active_state_removed=active_removed,
        terminal_published=published,
        checkpoint_gc=checkpoint_gc,
        cleanup_closed=cleanup,
        commits=tuple(active_commits),
    )
    for seal in (
        row for row in replayed.entries if row.get("kind") == "prefix_sealed"
    ):
        validate_write_usage_attestation(
            seal.get("write_usage_attestation"),
            state=replayed,
            expected_committed_prefix=seal.get("committed_prefix"),
            expected_generation=seal.get("generation"),
        )
    return replayed


class JournalLease(AbstractContextManager["JournalLease"]):
    """One-process advisory lease released automatically when a process dies."""

    def __init__(self, journal_path: str | os.PathLike[str]) -> None:
        journal = _lexical_absolute(journal_path)
        self._journal_path = journal
        self.path = journal.with_name(journal.name + ".lock")
        self._registry_key = _path_identity_sha256(self.path)
        self._registry_held = False
        self._handle: Any | None = None

    def __enter__(self) -> "JournalLease":
        if self._handle is not None or self._registry_held:
            raise ResumeJournalLocked("resume journal lease cannot be re-entered")
        _assert_no_link_ancestors(
            self._journal_path, label="resume journal lease target", require_final=False
        )
        _assert_no_link_ancestors(
            self.path, label="resume journal lease", require_final=False
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_link_ancestors(
            self.path, label="resume journal lease", require_final=False
        )
        with _PROCESS_LEASE_REGISTRY_LOCK:
            if self._registry_key in _PROCESS_LEASE_REGISTRY:
                raise ResumeJournalLocked("resume journal lease is already held")
            _PROCESS_LEASE_REGISTRY.add(self._registry_key)
            self._registry_held = True
        descriptor: int | None = None
        handle: Any | None = None
        try:
            descriptor = os.open(
                self.path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            if _is_reparse_or_link(self.path) or not os.path.samestat(
                os.fstat(descriptor), os.lstat(self.path)
            ):
                raise ResumableShardError(
                    "resume journal lease changed during acquisition"
                )
            handle = os.fdopen(descriptor, "r+b")
            descriptor = None
            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - Windows is the campaign platform.
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError, ResumableShardError) as exc:
            if handle is not None:
                handle.close()
            if descriptor is not None:
                os.close(descriptor)
            with _PROCESS_LEASE_REGISTRY_LOCK:
                _PROCESS_LEASE_REGISTRY.discard(self._registry_key)
                self._registry_held = False
            raise ResumeJournalLocked("resume journal lease is already held") from exc
        self._handle = handle
        return self

    def __exit__(self, *_args: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            with _PROCESS_LEASE_REGISTRY_LOCK:
                _PROCESS_LEASE_REGISTRY.discard(self._registry_key)
                self._registry_held = False

    def held_for(self, journal_path: str | os.PathLike[str]) -> bool:
        """Return whether this live lease is bound to exactly ``journal_path``."""

        journal = _lexical_absolute(journal_path)
        expected = journal.with_name(journal.name + ".lock")
        return (
            self._handle is not None
            and self._registry_held
            and self.path == expected
            and self._journal_path == journal
        )


class AppendOnlyResumeJournal:
    """Publish-once header plus durable, canonical, hash-chained appends."""

    def __init__(self, path: str | os.PathLike[str], plan: ResumePlan) -> None:
        self.path = _assert_no_link_ancestors(
            path, label="resume journal", require_final=False
        )
        self.plan = plan
        self._lock = threading.Lock()

    def create(self, *, owned_state_path: str, snapshot_root_path: str) -> ReplayState:
        token = secrets.token_hex(16)
        header = new_journal_header(
            self.plan,
            owned_state_path=owned_state_path,
            snapshot_root_path=snapshot_root_path,
            journal_path_sha256=_path_identity_sha256(self.path),
            snapshot_root_ownership_token=token,
        )
        payload = _line_bytes(header)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or _record_root(self.path).exists():
            raise FileExistsError("resume journal or its atomic records already exist")
        _initialize_record_root(self.path, header)
        _publish_record(self.path, header)
        descriptor: int | None = None
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
        _fsync_directory(self.path.parent)
        _ensure_snapshot_root(journal_path=self.path, header=header)
        return replay_journal((header,), expected_plan=self.plan)

    def replay(self) -> ReplayState:
        entries = self._reconcile_projection_from_records()
        state = replay_journal(entries, expected_plan=self.plan)
        header = state.entries[0]
        relative = _safe_relative(
            header.get("snapshot_root_path"), "snapshot_root_path"
        )
        root = _assert_no_link_ancestors(
            self.path.parent / Path(relative),
            label="snapshot root",
            require_final=False,
        )
        if state.checkpoint_gc is not None:
            if root.exists():
                raise ResumableShardError(
                    "journal says checkpoint GC completed but snapshot root exists"
                )
        elif state.terminal_published is not None:
            # Crash recovery may observe the root either before GC or after
            # deletion but before the checkpoint_gc journal acknowledgement.
            # Never recreate it once official outputs are authoritative.
            if root.exists():
                _ensure_snapshot_root(journal_path=self.path, header=header)
        else:
            if not root.exists():
                raise ResumableShardError(
                    "snapshot root disappeared before terminal publication"
                )
            _ensure_snapshot_root(journal_path=self.path, header=header)
        return state

    def _reconcile_projection_from_records(self) -> tuple[dict[str, Any], ...]:
        authoritative = _read_record_segments(self.path)
        expected = b"".join(_line_bytes(entry) for entry in authoritative)
        raw = self.path.read_bytes() if self.path.exists() else b""
        if raw == expected:
            return authoritative
        safe_prefix = False
        try:
            projected = _parse_journal_bytes(raw)
        except ResumableShardError:
            last_newline = raw.rfind(b"\n")
            complete = raw[: last_newline + 1] if last_newline >= 0 else b""
            try:
                prefix_rows = (
                    _parse_journal_bytes(complete) if complete else tuple()
                )
            except ResumableShardError:
                prefix_rows = tuple()
            safe_prefix = tuple(prefix_rows) == authoritative[: len(prefix_rows)]
        else:
            safe_prefix = tuple(projected) == authoritative[: len(projected)]
        if not safe_prefix:
            raise ResumableShardError(
                "journal projection corruption is not an atomic-record prefix"
            )
        record_root = _verify_record_root(self.path, authoritative[0])
        if raw:
            digest = hashlib.sha256(raw).hexdigest()
            evidence = record_root / f"projection-recovery-{digest}.bin"
            if not evidence.exists():
                _publish_file_no_clobber(evidence, raw)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".projection", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                if stream.write(expected) != len(expected):
                    raise ResumableShardError("journal projection repair was short")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        if self.path.read_bytes() != expected:
            raise ResumableShardError("journal projection repair did not persist")
        return authoritative

    def append(self, kind: str, **payload: Any) -> ReplayState:
        with self._lock:
            entries = self._reconcile_projection_from_records()
            before_bytes = b"".join(_line_bytes(entry) for entry in entries)
            replay_journal(entries, expected_plan=self.plan)
            previous = entries[-1]["entry_sha256"]
            entry = _entry(kind, len(entries), previous, **payload)
            line = _line_bytes(entry)
            _publish_record(self.path, entry)
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0),
            )
            try:
                written = os.write(descriptor, line)
                if written != len(line):
                    raise ResumableShardError("journal append was short")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            observed = self.path.read_bytes()
            if observed != before_bytes + line:
                # The atomic segment is already authoritative.  Repair the
                # JSONL projection now, or on restart if the process dies here.
                self._reconcile_projection_from_records()
            return replay_journal((*entries, entry), expected_plan=self.plan)


def append_intent(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    ordinal: int,
    session_sha256: str,
) -> ReplayState:
    if state.pending_intent is not None or state.terminal_search is not None:
        raise ResumableShardError("cannot append an intent in the current state")
    if ordinal != state.committed_prefix:
        raise ResumableShardError("intent ordinal differs from committed prefix")
    if ordinal >= state.plan.authorized_add_operations:
        raise ResumableShardError("intent exceeds the authorized add budget")
    return journal.append(
        "intent",
        generation=state.generation,
        ordinal=ordinal,
        batch_sha256=state.plan.ordered_batch_sha256s[ordinal],
        committed_prefix_before=state.committed_prefix,
        session_sha256=_require_sha256(session_sha256, "session_sha256"),
    )


def append_send_attempt(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    request_sha256: str,
) -> ReplayState:
    intent = state.pending_intent
    if intent is None or state.pending_send_attempt is not None:
        raise ResumableShardError("send attempt requires one pending intent")
    return journal.append(
        "send_attempt",
        generation=state.generation,
        ordinal=intent["ordinal"],
        intent_entry_sha256=intent["entry_sha256"],
        global_attempt_ordinal=state.committed_prefix,
        request_sha256=_require_sha256(request_sha256, "request_sha256"),
        prior_checkpoint_authority_sha256=state.checkpoint_authority_sha256,
    )


def append_commit(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    response_sha256: str,
    returned_memory_ids: Sequence[str],
    source_ref: Mapping[str, Any],
    request_window_refs: Sequence[Mapping[str, Any]],
    adapter_stats: Mapping[str, Any],
    logical_meter_receipt: Mapping[str, Any],
    transport_receipt: Mapping[str, Any],
    add_latency_s: float,
    raw_message_tokens: int,
    scope_protocol: bool,
) -> ReplayState:
    intent = state.pending_intent
    send = state.pending_send_attempt
    if intent is None or send is None:
        raise ResumableShardError("commit requires one intent and send attempt")
    ordinal = int(intent["ordinal"])
    cumulative = ordinal + 1
    ids = list(returned_memory_ids)
    if any(
        not isinstance(value, str) or not value.strip()
        for value in ids
    ):
        raise ResumableShardError(
            "returned memory IDs must be exact non-empty strings"
        )
    source = _source_ref(source_ref, "source_ref")
    window = [_source_ref(row, "request_window_ref") for row in request_window_refs]
    prior = rehydration_material(state)
    expected_window = _merge_source_refs(
        _merge_source_refs(prior["request_window_deque"]), (source,)
    )
    if window != expected_window:
        raise ResumableShardError("request window differs from locked ten-message tail")
    ledger: dict[str, list[dict[str, Any]]] = {
        row["memory_id"]: list(row["source_refs"])
        for row in prior["ledger_projection"]
    }
    for memory_id in ids:
        ledger[memory_id] = _merge_source_refs(ledger.get(memory_id, ()), window)
    ledger_projection = _ledger_projection(
        user_scope=state.plan.user_scope, ledger=ledger
    )
    stats = _adapter_stats(adapter_stats, "adapter_stats")
    logical = _observed_counter_receipt(
        logical_meter_receipt,
        label="logical_meter_receipt",
        expected=cumulative,
        logical=True,
    )
    transport = _observed_counter_receipt(
        transport_receipt,
        label="transport_receipt",
        expected=cumulative,
        logical=False,
        authorized_total=state.plan.authorized_extraction_calls,
        expected_seeded_prefix=state.sealed_prefix,
    )
    if not isinstance(scope_protocol, bool):
        raise ResumableShardError("scope_protocol must be boolean")
    return journal.append(
        "commit",
        generation=state.generation,
        ordinal=ordinal,
        intent_entry_sha256=intent["entry_sha256"],
        send_attempt_entry_sha256=send["entry_sha256"],
        batch_sha256=intent["batch_sha256"],
        response_sha256=_require_sha256(response_sha256, "response_sha256"),
        returned_memory_ids=ids,
        source_ref=source,
        request_window_refs=window,
        request_window_sha256=canonical_json_sha256(window),
        ledger_projection_sha256=canonical_json_sha256(ledger_projection),
        user_scope=state.plan.user_scope,
        scope_protocol=scope_protocol,
        attribution_kind="request_window_non_evidence",
        date_exposure_kind="diagnostics_only_not_model_input",
        request_window_messages=_WINDOW_MESSAGES,
        adapter_stats=stats,
        adapter_stats_sha256=canonical_json_sha256(stats),
        logical_meter_receipt=logical,
        logical_meter_receipt_sha256=canonical_json_sha256(logical),
        transport_receipt=transport,
        transport_receipt_sha256=canonical_json_sha256(transport),
        add_latency_s=_require_finite(add_latency_s, "add_latency_s"),
        raw_message_tokens=_require_count(raw_message_tokens, "raw_message_tokens"),
        cumulative_logical_attempted=cumulative,
        cumulative_logical_completed=cumulative,
        cumulative_logical_failed=0,
        cumulative_logical_rejected=0,
        cumulative_http_attempted=cumulative,
        cumulative_http_completed=cumulative,
        cumulative_http_failed=0,
        cumulative_http_rejected=0,
    )


def _ownership_token(root: Path) -> str:
    marker = root / OWNERSHIP_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ResumableShardError("owned state has no regular ownership marker")
    token = marker.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ResumableShardError("owned-state ownership token is invalid")
    return token


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    casefolded: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        _safe_relative(relative, "state/snapshot entry")
        folded = relative.casefold()
        if folded in casefolded:
            raise ResumableShardError("state/snapshot tree has a case-collision")
        casefolded.add(folded)
        if any(":" in part for part in Path(relative).parts):
            raise ResumableShardError("state/snapshot tree contains an ADS path")
        is_junction = getattr(path, "is_junction", lambda: False)
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        if (
            path.is_symlink()
            or is_junction()
            or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        ):
            raise ResumableShardError(
                "state/snapshot trees cannot contain links or reparse points"
            )
        if path.is_file():
            metadata = path.stat(follow_symlinks=False)
            if metadata.st_nlink != 1:
                raise ResumableShardError(
                    "state/snapshot trees cannot contain hard-linked files"
                )
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
            files[relative] = {
                "type": "file",
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
        elif path.is_dir():
            files[relative] = {"type": "directory"}
        else:
            raise ResumableShardError("state/snapshot tree has a special file")
    if files.get(OWNERSHIP_MARKER, {}).get("type") != "file":
        raise ResumableShardError("state/snapshot manifest omits ownership marker")
    return files


def state_tree_receipt(root: str | os.PathLike[str]) -> dict[str, Any]:
    path = _assert_no_link_ancestors(
        root, label="owned state", require_final=True
    )
    if not path.is_dir():
        raise ResumableShardError("owned state must be a regular directory")
    token = _ownership_token(path)
    manifest = _file_manifest(path)
    manifest_sha = canonical_json_sha256(manifest)
    body = {
        "path_name": path.name,
        "ownership_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "manifest": manifest,
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_tree_sha256": canonical_json_sha256(
            {
                "ownership_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "manifest_sha256": manifest_sha,
            }
        ),
        "file_count": len(manifest),
        "total_bytes": sum(
            int(row.get("bytes", 0)) for row in manifest.values()
        ),
    }
    return body


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_file():
            # Windows rejects FlushFileBuffers on a read-only handle.  These
            # files are owned staging copies, so open read/write solely for the
            # durability barrier without changing their bytes.
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())


def validate_write_usage_attestation(
    value: Any,
    *,
    state: ReplayState,
    expected_storage_bytes: int | None = None,
    expected_committed_prefix: int | None = None,
    expected_generation: int | None = None,
) -> dict[str, Any]:
    """Authenticate one complete, cumulative, gold-free write receipt."""

    if not isinstance(value, Mapping):
        raise ResumableShardError("write-usage attestation must be an object")
    row = _strict_json(value, "write-usage attestation")
    if not isinstance(row, dict):  # pragma: no cover
        raise ResumableShardError("write-usage attestation must be an object")
    _exact_keys(
        row,
        {
            "format",
            "plan_sha256",
            "authorization_sha256",
            "generation",
            "committed_prefix",
            "prior_write_usage_attestation_sha256",
            "segment_authorization_receipt",
            "segment_authorization_receipt_sha256",
            "segment_write_activity_receipt",
            "segment_write_activity_receipt_sha256",
            "transport_closure_receipt_sha256",
            "observed",
            "observed_sha256",
            "retained_transformer_token_state_bytes",
            "receipt_sha256",
        },
        "write-usage attestation",
    )
    body = dict(row)
    supplied = body.pop("receipt_sha256", None)
    if supplied != canonical_json_sha256(body):
        raise ResumableShardError("write-usage attestation digest mismatch")
    committed_prefix = (
        state.committed_prefix
        if expected_committed_prefix is None
        else _require_count(
            expected_committed_prefix, "expected write-usage prefix", minimum=1
        )
    )
    generation = (
        state.generation
        if expected_generation is None
        else _require_count(expected_generation, "expected write-usage generation")
    )
    required = {
        "format": RESUME_WRITE_USAGE_FORMAT,
        "plan_sha256": state.plan.sha256,
        "authorization_sha256": state.plan.authorization_sha256,
        "generation": generation,
        "committed_prefix": committed_prefix,
        "retained_transformer_token_state_bytes": 0,
    }
    for field, expected in required.items():
        if row.get(field) != expected:
            raise ResumableShardError(
                f"write-usage attestation {field} mismatch"
            )
    for field in (
        "segment_authorization_receipt_sha256",
        "segment_write_activity_receipt_sha256",
        "transport_closure_receipt_sha256",
        "observed_sha256",
    ):
        _require_sha256(row.get(field), f"write-usage {field}")
    prior = row.get("prior_write_usage_attestation_sha256")
    prior_seals = [
        entry
        for entry in state.entries
        if entry.get("kind") == "prefix_sealed"
        and int(entry.get("committed_prefix", -1)) < row["committed_prefix"]
    ]
    expected_prior = (
        prior_seals[-1].get("write_usage_attestation_sha256")
        if prior_seals
        else None
    )
    if prior != expected_prior:
        raise ResumableShardError("write-usage prior-seal chain mismatch")
    segment_authorization = row.get("segment_authorization_receipt")
    if not isinstance(segment_authorization, dict):
        raise ResumableShardError(
            "write-usage segment authorization is invalid"
        )
    authorization_body = dict(segment_authorization)
    authorization_sha = authorization_body.pop("receipt_sha256", None)
    if (
        authorization_sha != canonical_json_sha256(authorization_body)
        or authorization_sha != row["segment_authorization_receipt_sha256"]
    ):
        raise ResumableShardError(
            "write-usage segment authorization changed"
        )
    _exact_keys(
        segment_authorization,
        {
            "format",
            "plan_sha256",
            "authorization_sha256",
            "journal_path_sha256",
            "prefix_before",
            "prefix_after",
            "generation",
            "prior_checkpoint_authority_sha256",
            "authorized_provider_calls",
            "authorized_add_operations",
            "provider_retries",
            "namespace",
            "retained_transformer_token_state_bytes",
            "live_launch_authority",
            "live_launch_authority_sha256",
            "receipt_sha256",
        },
        "segment authorization receipt",
    )
    prior_prefix = prior_seals[-1]["committed_prefix"] if prior_seals else 0
    prior_authority = (
        prior_seals[-1]["snapshot_authority_sha256"]
        if prior_seals
        else state.entries[0]["empty_prefix_authority_sha256"]
    )
    segment_calls = committed_prefix - prior_prefix
    exact_segment_calls = min(
        _RESUME_SEGMENT_ADDS,
        state.plan.authorized_add_operations - prior_prefix,
    )
    launch_authority = authorization_body.get("live_launch_authority")
    if not isinstance(launch_authority, dict):
        raise ResumableShardError("write-usage live launch authority is invalid")
    _exact_keys(
        launch_authority,
        {
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
        },
        "write-usage live launch authority",
    )
    exact_launch = {
        "format": "memory-condense-mem0-live-launch-authority-v1",
        "plan_sha256": state.plan.sha256,
        "authorization_sha256": state.plan.authorization_sha256,
        "journal_path_sha256": state.entries[0]["journal_path_sha256"],
        "sample_offset": state.plan.sample_offset,
        "namespace": state.plan.user_scope,
        "namespace_sha256": hashlib.sha256(
            state.plan.user_scope.encode("utf-8")
        ).hexdigest(),
        "mem0_policy_sha256": state.plan.mem0_policy_sha256,
        "mem0_tool_implementation_sha256": (
            state.plan.mem0_tool_implementation_sha256
        ),
        "mem0_environment_lock_sha256": state.plan.mem0_environment_lock_sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    if any(
        launch_authority.get(field) != expected
        for field, expected in exact_launch.items()
    ):
        raise ResumableShardError("write-usage live launch authority changed")
    for field in (
        "preflight_sha256",
        "launch_manifest_sha256",
        "shard_launch_sha256",
        "shard_launch_payload_sha256",
    ):
        _require_sha256(
            launch_authority.get(field), f"write-usage live launch {field}"
        )
    launch_sha = authorization_body.get("live_launch_authority_sha256")
    if launch_sha != canonical_json_sha256(launch_authority):
        raise ResumableShardError("write-usage live launch digest mismatch")
    expected_segment = {
        "format": "memory-condense-mem0-one-use-segment-authorization-v1",
        "plan_sha256": state.plan.sha256,
        "authorization_sha256": state.plan.authorization_sha256,
        "journal_path_sha256": state.entries[0]["journal_path_sha256"],
        "prefix_before": prior_prefix,
        "prefix_after": committed_prefix,
        "generation": generation,
        "prior_checkpoint_authority_sha256": prior_authority,
        "authorized_provider_calls": segment_calls,
        "authorized_add_operations": segment_calls,
        "provider_retries": 0,
        "namespace": state.plan.user_scope,
        "retained_transformer_token_state_bytes": 0,
        "live_launch_authority": launch_authority,
        "live_launch_authority_sha256": launch_sha,
    }
    if (
        authorization_body != expected_segment
        or segment_calls != exact_segment_calls
        or exact_segment_calls <= 0
    ):
        raise ResumableShardError(
            "write-usage segment authorization binding changed"
        )
    activity = row.get("segment_write_activity_receipt")
    if not isinstance(activity, dict):
        raise ResumableShardError("write-usage activity receipt is invalid")
    activity_body = dict(activity)
    activity_sha = activity_body.pop("receipt_sha256", None)
    if (
        activity_sha != canonical_json_sha256(activity_body)
        or activity_sha != row["segment_write_activity_receipt_sha256"]
    ):
        raise ResumableShardError("write-usage activity receipt changed")
    _exact_keys(
        activity,
        {
            "format",
            "embedding_attempted",
            "embedding_completed",
            "embedding_failed",
            "embedding_input_token_proxy",
            "embedding_latency_s",
            "storage_attempted",
            "storage_completed",
            "storage_failed",
            "storage_latency_s",
            "wrappers_installed",
            "wrappers_restored",
            "receipt_sha256",
        },
        "segment write-activity receipt",
    )
    if (
        activity.get("format")
        != "memory-condense-mem0-resumable-write-activity-v1"
        or activity.get("wrappers_installed") is not True
        or activity.get("wrappers_restored") is not True
    ):
        raise ResumableShardError("write-usage activity boundary changed")
    for kind in ("embedding", "storage"):
        attempted = _require_count(
            activity.get(f"{kind}_attempted"), f"write-usage {kind} attempted"
        )
        completed = _require_count(
            activity.get(f"{kind}_completed"), f"write-usage {kind} completed"
        )
        failed = _require_count(
            activity.get(f"{kind}_failed"), f"write-usage {kind} failed"
        )
        if attempted != completed or failed:
            raise ResumableShardError(
                f"write-usage {kind} activity did not close"
            )
    _require_count(
        activity.get("embedding_input_token_proxy"),
        "write-usage segment embedding token proxy",
    )
    _require_finite(
        activity.get("embedding_latency_s"),
        "write-usage segment embedding latency",
    )
    _require_finite(
        activity.get("storage_latency_s"),
        "write-usage segment storage latency",
    )
    observed = row.get("observed")
    if not isinstance(observed, dict):
        raise ResumableShardError("write-usage observed fields are invalid")
    _exact_keys(observed, _WRITE_USAGE_OBSERVED_KEYS, "write-usage observed")
    if canonical_json_sha256(observed) != row["observed_sha256"]:
        raise ResumableShardError("write-usage observed digest mismatch")
    prefix = committed_prefix
    exact_counts = {
        "add_attempted": prefix,
        "add_completed": prefix,
        "add_failed": 0,
        "extraction_attempted": prefix,
        "extraction_completed": prefix,
        "extraction_failed": 0,
    }
    for field, expected in exact_counts.items():
        if observed.get(field) != expected:
            raise ResumableShardError(f"write-usage {field} mismatch")
    count_fields = (
        "extraction_raw_message_token_proxy",
        "extraction_provider_input_tokens",
        "extraction_provider_output_tokens",
        "embedding_operations",
        "embedding_input_token_proxy",
        "returned_memory_count",
        "persisted_memory_count",
        "persisted_storage_bytes",
    )
    for field in count_fields:
        _require_count(observed.get(field), f"write-usage {field}")
    if observed.get("extraction_usage_status") != "provider_reported_exact":
        raise ResumableShardError("write-usage provider status is not exact")
    for field in (
        "add_latency_s",
        "extraction_latency_s",
        "embedding_latency_s",
        "storage_latency_s",
    ):
        _require_finite(observed.get(field), f"write-usage {field}")
    if observed["persisted_memory_count"] > observed["returned_memory_count"]:
        raise ResumableShardError(
            "write-usage persisted memories exceed returned memories"
        )
    if (
        expected_storage_bytes is not None
        and observed["persisted_storage_bytes"]
        != _require_count(expected_storage_bytes, "expected storage bytes")
    ):
        raise ResumableShardError("write-usage storage bytes differ from state tree")
    return row


def prefix_close_receipt(
    state: ReplayState,
    *,
    history_sqlite_closed: bool,
    qdrant_local_collections_closed: int,
    qdrant_clients_closed: int,
    transport_closed: bool,
    transport_closure_receipt: Mapping[str, Any],
    write_usage_attestation: Mapping[str, Any],
    expected_storage_bytes: int,
) -> dict[str, Any]:
    """Seal observed, non-destructive quiescence for a prefix snapshot."""

    if state.pending_intent is not None or not state.commits:
        raise ResumableShardError("prefix close requires a clean committed tail")
    if not isinstance(history_sqlite_closed, bool) or not history_sqlite_closed:
        raise ResumableShardError("history SQLite handle is not closed")
    local_closed = _require_count(
        qdrant_local_collections_closed,
        "qdrant_local_collections_closed",
        minimum=1,
    )
    clients_closed = _require_count(
        qdrant_clients_closed, "qdrant_clients_closed", minimum=1
    )
    if not isinstance(transport_closed, bool) or not transport_closed:
        raise ResumableShardError("extraction transport is not closed")
    latest = state.commits[-1]
    closure = _strict_json(
        transport_closure_receipt, "transport-closure receipt"
    )
    if not isinstance(closure, dict):
        raise ResumableShardError("transport-closure receipt must be an object")
    closure_body = dict(closure)
    closure_sha = closure_body.pop("receipt_sha256", None)
    if closure_sha != canonical_json_sha256(closure_body):
        raise ResumableShardError("transport-closure receipt digest mismatch")
    _exact_keys(
        closure,
        {
            "format",
            "segment_authorized_calls",
            "transport_closed",
            "budget_closed_exactly",
            "provider_usage_complete",
            "sdk_retries",
            "http_transport_retries",
            "transport_receipt",
            "transport_receipt_sha256",
            "receipt_sha256",
        },
        "transport-closure receipt",
    )
    if (
        closure.get("format")
        != "memory-condense-mem0-resumable-transport-closure-v1"
        or
        closure.get("transport_closed") is not True
        or closure.get("budget_closed_exactly") is not True
        or closure.get("provider_usage_complete") is not True
        or closure.get("segment_authorized_calls")
        != latest["transport_receipt"]["segment_authorized"]
        or closure.get("transport_receipt_sha256")
        != latest["transport_receipt"]["segment_receipt_sha256"]
        or closure.get("transport_receipt")
        != latest["transport_receipt"]["segment_receipt"]
    ):
        raise ResumableShardError(
            "transport closure differs from the committed segment transport"
        )
    usage = validate_write_usage_attestation(
        write_usage_attestation,
        state=state,
        expected_storage_bytes=expected_storage_bytes,
    )
    if usage["transport_closure_receipt_sha256"] != closure_sha:
        raise ResumableShardError(
            "write usage differs from transport closure"
        )
    body = {
        "format": RESUME_PREFIX_CLOSE_FORMAT,
        "plan_sha256": state.plan.sha256,
        "generation": state.generation,
        "committed_prefix": state.committed_prefix,
        "history_sqlite_closed": True,
        "qdrant_local_collections_closed": local_closed,
        "qdrant_clients_closed": clients_closed,
        "transport_closed": True,
        "transport_closure_receipt": closure,
        "transport_closure_receipt_sha256": closure_sha,
        "write_usage_attestation": usage,
        "write_usage_attestation_sha256": usage["receipt_sha256"],
        "state_deleted": False,
        "logical_meter_receipt_sha256": latest[
            "logical_meter_receipt_sha256"
        ],
        "transport_receipt_sha256": latest["transport_receipt_sha256"],
        "adapter_stats_sha256": latest["adapter_stats_sha256"],
        "cumulative_logical_attempted": state.committed_prefix,
        "cumulative_logical_completed": state.committed_prefix,
        "cumulative_http_attempted": state.committed_prefix,
        "cumulative_http_completed": state.committed_prefix,
        "infer_true_adds_started": state.committed_prefix,
        "infer_true_adds_exactly_one_call": state.committed_prefix,
        "failures": 0,
        "rejections": 0,
    }
    return {**body, "receipt_sha256": canonical_json_sha256(body)}


def _validate_prefix_close_receipt(
    state: ReplayState, value: Any
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResumableShardError("handles-closed receipt must be an object")
    receipt = _strict_json(value, "handles-closed receipt")
    if not isinstance(receipt, dict):  # pragma: no cover
        raise ResumableShardError("handles-closed receipt must be an object")
    expected = prefix_close_receipt(
        state,
        history_sqlite_closed=receipt.get("history_sqlite_closed"),
        qdrant_local_collections_closed=receipt.get(
            "qdrant_local_collections_closed"
        ),
        qdrant_clients_closed=receipt.get("qdrant_clients_closed"),
        transport_closed=receipt.get("transport_closed"),
        transport_closure_receipt=receipt.get("transport_closure_receipt"),
        write_usage_attestation=receipt.get("write_usage_attestation"),
        expected_storage_bytes=receipt.get("write_usage_attestation", {})
        .get("observed", {})
        .get("persisted_storage_bytes"),
    )
    if receipt != expected:
        raise ResumableShardError("handles-closed receipt fields/identity mismatch")
    return receipt


def checkpoint_authority_from_replay(
    state: ReplayState,
    *,
    handles_closed_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a quiescent state copy to every exact committed-prefix receipt."""

    if state.pending_intent is not None or not state.commits:
        raise ResumableShardError("checkpoint authority requires a clean commit tail")
    if state.committed_prefix <= state.sealed_prefix:
        raise ResumableShardError("checkpoint authority must advance the sealed prefix")
    active = state.commits[-1]
    tail = state.entries[-1]
    if tail.get("entry_sha256") != active.get("entry_sha256"):
        raise ResumableShardError("checkpoint authority tail is not the active commit")
    prefix_batches = list(
        state.plan.ordered_batch_sha256s[: state.committed_prefix]
    )
    close_receipt = _validate_prefix_close_receipt(
        state, handles_closed_receipt
    )
    rehydration = rehydration_material(state)
    header = state.entries[0]
    root_marker = _snapshot_root_marker_from_header(header)
    body = {
        "format": RESUME_SNAPSHOT_AUTHORITY_FORMAT,
        "plan_sha256": state.plan.sha256,
        "authorization_sha256": state.plan.authorization_sha256,
        "mem0_policy_sha256": state.plan.mem0_policy_sha256,
        "source_implementation_sha256": state.plan.source_implementation_sha256,
        "source_environment_lock_sha256": (
            state.plan.source_environment_lock_sha256
        ),
        "mem0_tool_implementation_sha256": (
            state.plan.mem0_tool_implementation_sha256
        ),
        "mem0_environment_lock_sha256": (
            state.plan.mem0_environment_lock_sha256
        ),
        "generation": state.generation,
        "committed_prefix": state.committed_prefix,
        "owned_state_path": header["owned_state_path"],
        "snapshot_root_path": header["snapshot_root_path"],
        "journal_path_sha256": header["journal_path_sha256"],
        "snapshot_root_marker": root_marker,
        "snapshot_root_marker_sha256": root_marker["marker_sha256"],
        "active_commit_entry_sha256": active["entry_sha256"],
        "journal_tail_entry_sha256": tail["entry_sha256"],
        "ordered_prefix_sha256": canonical_json_sha256(prefix_batches),
        "handles_closed_receipt": close_receipt,
        "handles_closed_receipt_sha256": close_receipt["receipt_sha256"],
        "rehydration": rehydration,
        "rehydration_sha256": rehydration["rehydration_sha256"],
        "cumulative_extraction_attempted": close_receipt[
            "cumulative_logical_attempted"
        ],
        "cumulative_extraction_completed": close_receipt[
            "cumulative_logical_completed"
        ],
        "cumulative_http_attempted": close_receipt[
            "cumulative_http_attempted"
        ],
        "cumulative_http_completed": close_receipt[
            "cumulative_http_completed"
        ],
        "infer_true_adds_started": close_receipt["infer_true_adds_started"],
        "infer_true_adds_exactly_one_call": close_receipt[
            "infer_true_adds_exactly_one_call"
        ],
        "failures": close_receipt["failures"],
        "rejections": close_receipt["rejections"],
    }
    return {**body, "authority_sha256": canonical_json_sha256(body)}


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def _write_snapshot_authority(root: Path, authority: Mapping[str, Any]) -> str:
    target = root / "prefix-authority.json"
    payload = _canonical_json(authority) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    target.write_bytes(payload)
    target.with_name(target.name + ".sha256").write_bytes(
        _sidecar_bytes(target, digest)
    )
    _fsync_tree(root)
    return digest


def _read_snapshot_authority(root: Path) -> tuple[dict[str, Any], str]:
    target = root / "prefix-authority.json"
    sidecar = target.with_name(target.name + ".sha256")
    if target.is_symlink() or sidecar.is_symlink():
        raise ResumableShardError("snapshot authority cannot be a symlink")
    raw = target.read_bytes()
    if not raw.endswith(b"\n"):
        raise ResumableShardError("snapshot authority is truncated")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumableShardError("snapshot authority is not JSON") from exc
    if type(value) is not dict or raw != _canonical_json(value) + b"\n":
        raise ResumableShardError("snapshot authority is not canonical JSON")
    digest = hashlib.sha256(raw).hexdigest()
    if sidecar.read_bytes() != _sidecar_bytes(target, digest):
        raise ResumableShardError("snapshot authority sidecar mismatch")
    body = dict(value)
    authority_sha = body.pop("authority_sha256", None)
    if authority_sha != canonical_json_sha256(body):
        raise ResumableShardError("snapshot internal authority digest mismatch")
    if body.get("format") != RESUME_SNAPSHOT_AUTHORITY_FORMAT:
        raise ResumableShardError("snapshot authority format mismatch")
    _exact_keys(
        value,
        {
            "format",
            "plan_sha256",
            "authorization_sha256",
            "mem0_policy_sha256",
            "source_implementation_sha256",
            "source_environment_lock_sha256",
            "mem0_tool_implementation_sha256",
            "mem0_environment_lock_sha256",
            "generation",
            "committed_prefix",
            "owned_state_path",
            "snapshot_root_path",
            "journal_path_sha256",
            "snapshot_root_marker",
            "snapshot_root_marker_sha256",
            "active_commit_entry_sha256",
            "journal_tail_entry_sha256",
            "ordered_prefix_sha256",
            "handles_closed_receipt",
            "handles_closed_receipt_sha256",
            "rehydration",
            "rehydration_sha256",
            "cumulative_extraction_attempted",
            "cumulative_extraction_completed",
            "cumulative_http_attempted",
            "cumulative_http_completed",
            "infer_true_adds_started",
            "infer_true_adds_exactly_one_call",
            "failures",
            "rejections",
            "state_manifest_sha256",
            "state_tree_sha256",
            "ownership_token_sha256",
            "file_count",
            "total_bytes",
            "authority_sha256",
        },
        "snapshot authority",
    )
    for field in (
        "plan_sha256",
        "authorization_sha256",
        "mem0_policy_sha256",
        "source_implementation_sha256",
        "source_environment_lock_sha256",
        "mem0_tool_implementation_sha256",
        "mem0_environment_lock_sha256",
        "journal_path_sha256",
        "snapshot_root_marker_sha256",
        "active_commit_entry_sha256",
        "journal_tail_entry_sha256",
        "ordered_prefix_sha256",
        "handles_closed_receipt_sha256",
        "rehydration_sha256",
        "state_manifest_sha256",
        "state_tree_sha256",
        "ownership_token_sha256",
    ):
        _require_sha256(value.get(field), f"snapshot authority {field}")
    marker_value = value.get("snapshot_root_marker")
    if not isinstance(marker_value, dict):
        raise ResumableShardError("snapshot authority root marker is not an object")
    marker_body = dict(marker_value)
    marker_sha = marker_body.pop("marker_sha256", None)
    if (
        marker_sha != canonical_json_sha256(marker_body)
        or marker_sha != value["snapshot_root_marker_sha256"]
    ):
        raise ResumableShardError("snapshot authority root marker is invalid")
    rehydration = value.get("rehydration")
    if not isinstance(rehydration, dict):
        raise ResumableShardError("snapshot rehydration material is not an object")
    rehydration_body = dict(rehydration)
    rehydration_sha = rehydration_body.pop("rehydration_sha256", None)
    if (
        rehydration_sha != canonical_json_sha256(rehydration_body)
        or rehydration_sha != value["rehydration_sha256"]
    ):
        raise ResumableShardError("snapshot rehydration receipt is invalid")
    close_receipt = value.get("handles_closed_receipt")
    if not isinstance(close_receipt, dict):
        raise ResumableShardError("snapshot close receipt is not an object")
    close_body = dict(close_receipt)
    close_sha = close_body.pop("receipt_sha256", None)
    if (
        close_sha != canonical_json_sha256(close_body)
        or close_sha != value["handles_closed_receipt_sha256"]
    ):
        raise ResumableShardError("snapshot close receipt is invalid")
    return value, digest


def _rename_directory_no_clobber(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"immutable snapshot already exists: {destination}")
    # On the campaign's Windows platform os.rename is a same-volume atomic
    # no-replace operation.  The precheck is repeated immediately before it;
    # a racing creator causes FileExistsError rather than replacement.
    os.rename(source, destination)


def create_immutable_state_snapshot(
    *,
    journal_path: str | os.PathLike[str],
    owned_state_dir: str | os.PathLike[str],
    snapshot_root: str | os.PathLike[str],
    committed_prefix: int,
    checkpoint_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a closed owned-state tree into one immutable publish-once prefix."""

    journal = _assert_no_link_ancestors(
        journal_path, label="resume journal", require_final=True
    )
    state = _assert_no_link_ancestors(
        owned_state_dir, label="owned state", require_final=True
    )
    root = _assert_no_link_ancestors(
        snapshot_root, label="snapshot root", require_final=True
    )
    prefix = _require_count(committed_prefix, "committed_prefix", minimum=1)
    if checkpoint_authority is None:
        raise ResumableShardError("snapshot requires committed-prefix authority")
    authority_input = _strict_json(
        checkpoint_authority, "checkpoint_authority"
    )
    if not isinstance(authority_input, dict):
        raise ResumableShardError("checkpoint authority must be an object")
    authority_body = dict(authority_input)
    supplied_authority_sha = authority_body.pop("authority_sha256", None)
    if supplied_authority_sha != canonical_json_sha256(authority_body):
        raise ResumableShardError("checkpoint authority digest mismatch")
    if authority_body.get("format") != RESUME_SNAPSHOT_AUTHORITY_FORMAT:
        raise ResumableShardError("checkpoint authority format mismatch")
    if authority_body.get("committed_prefix") != prefix:
        raise ResumableShardError("checkpoint authority prefix mismatch")
    closed_sha = _require_sha256(
        authority_body.get("handles_closed_receipt_sha256"),
        "checkpoint authority close receipt",
    )
    if _path_identity_sha256(journal) != authority_body.get("journal_path_sha256"):
        raise ResumableShardError("checkpoint authority journal path mismatch")
    expected_state = journal.parent / Path(
        _safe_relative(authority_body.get("owned_state_path"), "owned_state_path")
    )
    expected_root = journal.parent / Path(
        _safe_relative(
            authority_body.get("snapshot_root_path"), "snapshot_root_path"
        )
    )
    if state != _lexical_absolute(expected_state):
        raise ResumableShardError("owned state path differs from checkpoint plan")
    if root != _lexical_absolute(expected_root):
        raise ResumableShardError("snapshot root path differs from checkpoint plan")
    root_marker = authority_body.get("snapshot_root_marker")
    if not isinstance(root_marker, dict):
        raise ResumableShardError("checkpoint authority omitted snapshot-root marker")
    marker_payload, marker_artifact_sha = _sealed_json_bytes(root_marker)
    marker = root / SNAPSHOT_ROOT_MARKER
    if (
        root_marker.get("marker_sha256")
        != authority_body.get("snapshot_root_marker_sha256")
        or marker.read_bytes() != marker_payload
        or marker.with_name(marker.name + ".sha256").read_bytes()
        != _sidecar_bytes(marker, marker_artifact_sha)
    ):
        raise ResumableShardError("snapshot-root marker differs from checkpoint")
    if not state.is_dir():
        raise ResumableShardError("snapshot source is not a regular directory")
    if _paths_overlap(state, root):
        raise ResumableShardError("snapshot root and working state overlap")
    _ownership_token(state)
    target = root / f"prefix-{prefix:06d}"
    if target.exists():
        raise FileExistsError(f"immutable prefix snapshot already exists: {target}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".staging", dir=root)
    )
    try:
        staged_state = staging / "state"
        staged_state.mkdir()
        for source in sorted(state.rglob("*"), key=lambda value: value.as_posix()):
            relative = source.relative_to(state)
            destination = staged_state / relative
            if source.is_symlink():
                raise ResumableShardError("snapshot source contains a symlink")
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        _fsync_tree(staged_state)
        before = state_tree_receipt(state)
        staged = state_tree_receipt(staged_state)
        for field in (
            "ownership_token_sha256",
            "snapshot_manifest_sha256",
            "snapshot_tree_sha256",
            "file_count",
            "total_bytes",
        ):
            if staged[field] != before[field]:
                raise ResumableShardError(f"snapshot copy {field} mismatch")
        enriched_body = {
            **authority_body,
            "state_manifest_sha256": staged["snapshot_manifest_sha256"],
            "state_tree_sha256": staged["snapshot_tree_sha256"],
            "ownership_token_sha256": staged["ownership_token_sha256"],
            "file_count": staged["file_count"],
            "total_bytes": staged["total_bytes"],
        }
        enriched = {
            **enriched_body,
            "authority_sha256": canonical_json_sha256(enriched_body),
        }
        authority_artifact_sha = _write_snapshot_authority(staging, enriched)
        _rename_directory_no_clobber(staging, target)
        _fsync_directory(root)
        published = state_tree_receipt(target / "state")
        expected_published = {**staged, "path_name": "state"}
        if published != expected_published:
            raise ResumableShardError("published prefix snapshot changed")
        observed_authority, observed_artifact_sha = _read_snapshot_authority(target)
        if observed_authority != enriched or observed_artifact_sha != authority_artifact_sha:
            raise ResumableShardError("published snapshot authority changed")
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    body = {
        "format": RESUME_SNAPSHOT_FORMAT,
        "committed_prefix": prefix,
        "snapshot_path": target.relative_to(journal.parent).as_posix(),
        "snapshot_manifest_sha256": published["snapshot_manifest_sha256"],
        "snapshot_tree_sha256": published["snapshot_tree_sha256"],
        "ownership_token_sha256": published["ownership_token_sha256"],
        "handles_closed_receipt_sha256": closed_sha,
        "transport_closure_receipt_sha256": authority_body[
            "handles_closed_receipt"
        ]["transport_closure_receipt_sha256"],
        "write_usage_attestation": authority_body["handles_closed_receipt"] [
            "write_usage_attestation"
        ],
        "write_usage_attestation_sha256": authority_body[
            "handles_closed_receipt"
        ]["write_usage_attestation_sha256"],
        "snapshot_authority_sha256": enriched["authority_sha256"],
        "snapshot_authority_artifact_sha256": authority_artifact_sha,
        "active_commit_entry_sha256": enriched["active_commit_entry_sha256"],
        "journal_tail_entry_sha256": enriched["journal_tail_entry_sha256"],
        "generation": enriched["generation"],
        "file_count": published["file_count"],
        "total_bytes": published["total_bytes"],
    }
    return {**body, "snapshot_receipt_sha256": canonical_json_sha256(body)}


def verify_immutable_state_snapshot(
    snapshot_path: str | os.PathLike[str],
    *,
    expected_authority_sha256: str,
    expected_manifest_sha256: str,
    expected_tree_sha256: str,
    expected_ownership_token_sha256: str,
) -> dict[str, Any]:
    root = _assert_no_link_ancestors(
        snapshot_path, label="immutable snapshot", require_final=True
    )
    if not root.is_dir():
        raise ResumableShardError("immutable snapshot is not a directory")
    authority, authority_artifact_sha = _read_snapshot_authority(root)
    if authority.get("authority_sha256") != _require_sha256(
        expected_authority_sha256, "expected_authority_sha256"
    ):
        raise ResumableShardError("immutable snapshot authority mismatch")
    receipt = state_tree_receipt(root / "state")
    for field, expected in {
        "snapshot_manifest_sha256": _require_sha256(
            expected_manifest_sha256, "expected_manifest_sha256"
        ),
        "snapshot_tree_sha256": _require_sha256(
            expected_tree_sha256, "expected_tree_sha256"
        ),
        "ownership_token_sha256": _require_sha256(
            expected_ownership_token_sha256,
            "expected_ownership_token_sha256",
        ),
    }.items():
        if receipt[field] != expected:
            raise ResumableShardError(f"immutable snapshot {field} mismatch")
    for field, observed in {
        "state_manifest_sha256": receipt["snapshot_manifest_sha256"],
        "state_tree_sha256": receipt["snapshot_tree_sha256"],
        "ownership_token_sha256": receipt["ownership_token_sha256"],
    }.items():
        if authority.get(field) != observed:
            raise ResumableShardError(f"snapshot authority {field} mismatch")
    return {
        **receipt,
        "snapshot_authority": authority,
        "snapshot_authority_artifact_sha256": authority_artifact_sha,
    }


def append_prefix_sealed(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    snapshot_receipt: Mapping[str, Any],
) -> ReplayState:
    if state.pending_intent is not None or not state.commits:
        raise ResumableShardError("prefix seal requires a completed add boundary")
    snapshot = dict(snapshot_receipt)
    _exact_keys(
        snapshot,
        {
            "format",
            "committed_prefix",
            "snapshot_path",
            "snapshot_manifest_sha256",
            "snapshot_tree_sha256",
            "ownership_token_sha256",
            "handles_closed_receipt_sha256",
            "transport_closure_receipt_sha256",
            "write_usage_attestation",
            "write_usage_attestation_sha256",
            "snapshot_authority_sha256",
            "snapshot_authority_artifact_sha256",
            "active_commit_entry_sha256",
            "journal_tail_entry_sha256",
            "generation",
            "file_count",
            "total_bytes",
            "snapshot_receipt_sha256",
        },
        "snapshot receipt",
    )
    receipt_body = dict(snapshot)
    supplied_receipt_sha = receipt_body.pop("snapshot_receipt_sha256")
    if supplied_receipt_sha != canonical_json_sha256(receipt_body):
        raise ResumableShardError("snapshot receipt self-digest mismatch")
    if snapshot.get("format") != RESUME_SNAPSHOT_FORMAT:
        raise ResumableShardError("snapshot receipt format mismatch")
    if snapshot.get("committed_prefix") != state.committed_prefix:
        raise ResumableShardError("snapshot prefix differs from journal prefix")
    if snapshot.get("active_commit_entry_sha256") != state.commits[-1][
        "entry_sha256"
    ]:
        raise ResumableShardError("snapshot active-commit receipt mismatch")
    if snapshot.get("journal_tail_entry_sha256") != state.entries[-1][
        "entry_sha256"
    ]:
        raise ResumableShardError("snapshot journal-tail receipt mismatch")
    if snapshot.get("generation") != state.generation:
        raise ResumableShardError("snapshot generation mismatch")
    expected_path = (
        f"{state.entries[0]['snapshot_root_path']}/"
        f"prefix-{state.committed_prefix:06d}"
    )
    if snapshot.get("snapshot_path") != expected_path:
        raise ResumableShardError("snapshot path differs from frozen prefix path")
    rehydration = rehydration_material(state)
    usage = validate_write_usage_attestation(
        snapshot.get("write_usage_attestation"),
        state=state,
        expected_storage_bytes=snapshot.get("total_bytes"),
    )
    if usage["receipt_sha256"] != snapshot.get(
        "write_usage_attestation_sha256"
    ) or usage["transport_closure_receipt_sha256"] != snapshot.get(
        "transport_closure_receipt_sha256"
    ):
        raise ResumableShardError("snapshot write-usage digest mismatch")
    return journal.append(
        "prefix_sealed",
        generation=state.generation,
        committed_prefix=state.committed_prefix,
        active_commit_entry_sha256=state.commits[-1]["entry_sha256"],
        snapshot_path=_safe_relative(snapshot.get("snapshot_path"), "snapshot_path"),
        snapshot_manifest_sha256=_require_sha256(
            snapshot.get("snapshot_manifest_sha256"),
            "snapshot_manifest_sha256",
        ),
        snapshot_tree_sha256=_require_sha256(
            snapshot.get("snapshot_tree_sha256"), "snapshot_tree_sha256"
        ),
        ownership_token_sha256=_require_sha256(
            snapshot.get("ownership_token_sha256"), "ownership_token_sha256"
        ),
        handles_closed_receipt_sha256=_require_sha256(
            snapshot.get("handles_closed_receipt_sha256"),
            "handles_closed_receipt_sha256",
        ),
        transport_closure_receipt_sha256=_require_sha256(
            snapshot.get("transport_closure_receipt_sha256"),
            "transport_closure_receipt_sha256",
        ),
        write_usage_attestation=usage,
        write_usage_attestation_sha256=usage["receipt_sha256"],
        snapshot_authority_sha256=_require_sha256(
            snapshot.get("snapshot_authority_sha256"),
            "snapshot_authority_sha256",
        ),
        snapshot_authority_artifact_sha256=_require_sha256(
            snapshot.get("snapshot_authority_artifact_sha256"),
            "snapshot_authority_artifact_sha256",
        ),
        snapshot_receipt_sha256=supplied_receipt_sha,
        rehydration_sha256=rehydration["rehydration_sha256"],
        cumulative_extraction_attempted=state.committed_prefix,
        cumulative_extraction_completed=state.committed_prefix,
        cumulative_http_attempted=state.committed_prefix,
        cumulative_http_completed=state.committed_prefix,
        failures=0,
        rejections=0,
    )


def reconcile_published_checkpoint(
    journal: AppendOnlyResumeJournal,
    state: ReplayState | None = None,
) -> ReplayState:
    """Acknowledge an atomically published checkpoint after a process crash.

    The immutable directory is the commit authority.  ``prefix_sealed`` is a
    journal mirror: its absence after the no-clobber directory rename cannot
    turn already-checkpointed provider work into an ambiguity.
    """

    current = state or journal.replay()
    if current.pending_intent is not None:
        raise ResumeAmbiguityError(
            "cannot reconcile a checkpoint with a pending provider operation"
        )
    if current.committed_prefix == current.sealed_prefix:
        return current
    if current.committed_prefix < 1 or not current.commits:
        raise ResumeAmbiguityError("there is no committed prefix to reconcile")
    header = current.entries[0]
    expected_relative = (
        f"{header['snapshot_root_path']}/"
        f"prefix-{current.committed_prefix:06d}"
    )
    target = _assert_no_link_ancestors(
        journal.path.parent / Path(expected_relative),
        label="published checkpoint",
        require_final=False,
    )
    if not target.is_dir():
        raise ResumeAmbiguityError(
            "committed provider work has no valid published checkpoint"
        )
    authority, authority_artifact_sha = _read_snapshot_authority(target)
    close_receipt = authority.get("handles_closed_receipt")
    if not isinstance(close_receipt, Mapping):
        raise ResumeAmbiguityError("published checkpoint omitted close receipt")
    expected_base = checkpoint_authority_from_replay(
        current, handles_closed_receipt=close_receipt
    )
    expected_body = dict(expected_base)
    expected_body.pop("authority_sha256")
    for key, expected in expected_body.items():
        if authority.get(key) != expected:
            raise ResumeAmbiguityError(
                f"published checkpoint authority differs at {key}"
            )
    verified = verify_immutable_state_snapshot(
        target,
        expected_authority_sha256=authority["authority_sha256"],
        expected_manifest_sha256=authority["state_manifest_sha256"],
        expected_tree_sha256=authority["state_tree_sha256"],
        expected_ownership_token_sha256=authority["ownership_token_sha256"],
    )
    validate_write_usage_attestation(
        authority["handles_closed_receipt"]["write_usage_attestation"],
        state=current,
        expected_storage_bytes=verified["total_bytes"],
    )
    body = {
        "format": RESUME_SNAPSHOT_FORMAT,
        "committed_prefix": current.committed_prefix,
        "snapshot_path": expected_relative,
        "snapshot_manifest_sha256": verified["snapshot_manifest_sha256"],
        "snapshot_tree_sha256": verified["snapshot_tree_sha256"],
        "ownership_token_sha256": verified["ownership_token_sha256"],
        "handles_closed_receipt_sha256": authority[
            "handles_closed_receipt_sha256"
        ],
        "transport_closure_receipt_sha256": authority[
            "handles_closed_receipt"
        ]["transport_closure_receipt_sha256"],
        "write_usage_attestation": authority["handles_closed_receipt"] [
            "write_usage_attestation"
        ],
        "write_usage_attestation_sha256": authority[
            "handles_closed_receipt"
        ]["write_usage_attestation_sha256"],
        "snapshot_authority_sha256": authority["authority_sha256"],
        "snapshot_authority_artifact_sha256": authority_artifact_sha,
        "active_commit_entry_sha256": authority[
            "active_commit_entry_sha256"
        ],
        "journal_tail_entry_sha256": authority["journal_tail_entry_sha256"],
        "generation": authority["generation"],
        "file_count": verified["file_count"],
        "total_bytes": verified["total_bytes"],
    }
    receipt = {**body, "snapshot_receipt_sha256": canonical_json_sha256(body)}
    return append_prefix_sealed(journal, current, snapshot_receipt=receipt)


def prefix_zero_restore_receipt(
    state: ReplayState,
    *,
    destination_state_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Prove that a pre-send prefix-zero rollback has no mutable state."""

    if state.committed_prefix != 0:
        raise ResumableShardError("prefix-zero restore proof requires prefix zero")
    destination = _assert_no_link_ancestors(
        destination_state_dir,
        label="prefix-zero destination",
        require_final=False,
    )
    expected = _lexical_absolute(
        state.entries[0]["owned_state_path"]
    )
    # Header paths are relative to the journal parent.  A caller can only use
    # this helper through the journal-bound overload below, so the absolute
    # path itself is sealed as an identity rather than trusted on replay.
    del expected
    if destination.exists():
        raise ResumableShardError(
            "prefix-zero rollback requires verified absence of working state"
        )
    absent_tree_sha = canonical_json_sha256({"owned_state_absent": True})
    body = {
        "format": RESUME_RESTORE_FORMAT,
        "plan_sha256": state.plan.sha256,
        "generation": state.generation,
        "restored_prefix": 0,
        "restore_authority_sha256": state.checkpoint_authority_sha256,
        "destination_path_sha256": _path_identity_sha256(destination),
        "owned_state_absent": True,
        "restored_snapshot_manifest_sha256": None,
        "restored_snapshot_tree_sha256": absent_tree_sha,
        "restored_ownership_token_sha256": None,
    }
    return {**body, "restore_receipt_sha256": canonical_json_sha256(body)}


def _validate_restore_receipt(
    state: ReplayState, value: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _strict_json(value, "restore receipt")
    if not isinstance(receipt, dict):
        raise ResumableShardError("restore receipt must be an object")
    _exact_keys(
        receipt,
        {
            "format",
            "plan_sha256",
            "generation",
            "restored_prefix",
            "restore_authority_sha256",
            "destination_path_sha256",
            "owned_state_absent",
            "restored_snapshot_manifest_sha256",
            "restored_snapshot_tree_sha256",
            "restored_ownership_token_sha256",
            "restore_receipt_sha256",
        },
        "restore receipt",
    )
    body = dict(receipt)
    supplied = body.pop("restore_receipt_sha256", None)
    if supplied != canonical_json_sha256(body):
        raise ResumableShardError("restore receipt self-digest mismatch")
    if (
        receipt.get("format") != RESUME_RESTORE_FORMAT
        or receipt.get("plan_sha256") != state.plan.sha256
        or receipt.get("generation") != state.generation
        or receipt.get("restored_prefix") != state.sealed_prefix
        or receipt.get("restore_authority_sha256")
        != state.checkpoint_authority_sha256
    ):
        raise ResumableShardError("restore receipt identity mismatch")
    _require_sha256(receipt.get("destination_path_sha256"), "restore destination")
    expected_tree = (
        state.latest_prefix_seal.get("snapshot_tree_sha256")
        if state.latest_prefix_seal is not None
        else canonical_json_sha256({"owned_state_absent": True})
    )
    if receipt.get("restored_snapshot_tree_sha256") != expected_tree:
        raise ResumableShardError("restored tree differs from checkpoint")
    if state.sealed_prefix == 0:
        if (
            receipt.get("owned_state_absent") is not True
            or receipt.get("restored_snapshot_manifest_sha256") is not None
            or receipt.get("restored_ownership_token_sha256") is not None
        ):
            raise ResumableShardError("prefix-zero restore receipt is not empty")
    else:
        if receipt.get("owned_state_absent") is not False:
            raise ResumableShardError("nonzero restore receipt claims absent state")
        _require_sha256(
            receipt.get("restored_snapshot_manifest_sha256"),
            "restored manifest",
        )
        _require_sha256(
            receipt.get("restored_ownership_token_sha256"),
            "restored ownership token",
        )
    return receipt


def append_presend_rollback(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    restore_receipt: Mapping[str, Any],
) -> ReplayState:
    if not state.requires_rollback:
        raise ResumeAmbiguityError(
            "rollback is allowed only for an intent proven not to reach send"
        )
    intent = state.pending_intent
    assert intent is not None
    restored = _validate_restore_receipt(state, restore_receipt)
    expected_destination = journal.path.parent / Path(
        state.entries[0]["owned_state_path"]
    )
    if restored["destination_path_sha256"] != _path_identity_sha256(
        expected_destination
    ):
        raise ResumableShardError("restore destination differs from frozen state path")
    return journal.append(
        "rollback",
        generation=state.generation,
        intent_entry_sha256=intent["entry_sha256"],
        restore_prefix=state.sealed_prefix,
        restore_authority_sha256=state.checkpoint_authority_sha256,
        restore_receipt_sha256=restored["restore_receipt_sha256"],
        restored_snapshot_tree_sha256=restored[
            "restored_snapshot_tree_sha256"
        ],
        next_generation=state.generation + 1,
        reason="intent_durable_send_boundary_not_reached",
    )


def append_terminal_search(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    terminal_stage_path: str,
    terminal_stage_sha256: str,
    terminal_result_sha256: str,
    terminal_trace_sha256: str,
    completed_search_operations: int,
) -> ReplayState:
    if state.committed_prefix != state.plan.authorized_add_operations:
        raise ResumableShardError("cannot search before every add is committed")
    if state.sealed_prefix != state.committed_prefix:
        raise ResumableShardError("cannot search before the full prefix is sealed")
    if completed_search_operations != state.plan.authorized_search_operations:
        raise ResumableShardError("terminal search count differs from authorization")
    return journal.append(
        "terminal_search",
        generation=state.generation,
        completed_search_operations=completed_search_operations,
        terminal_stage_path=_safe_relative(
            terminal_stage_path, "terminal_stage_path"
        ),
        terminal_stage_sha256=_require_sha256(
            terminal_stage_sha256, "terminal_stage_sha256"
        ),
        terminal_result_sha256=_require_sha256(
            terminal_result_sha256, "terminal_result_sha256"
        ),
        terminal_trace_sha256=_require_sha256(
            terminal_trace_sha256, "terminal_trace_sha256"
        ),
        committed_prefix=state.committed_prefix,
        full_checkpoint_authority_sha256=state.checkpoint_authority_sha256,
        extraction_calls_closed=True,
        provider_retries=0,
        transport_closure_receipt_sha256=state.latest_prefix_seal[
            "transport_closure_receipt_sha256"
        ],
        write_usage_attestation_sha256=state.latest_prefix_seal[
            "write_usage_attestation_sha256"
        ],
    )


def _exact_receipt(
    value: Mapping[str, Any],
    *,
    label: str,
    expected_keys: set[str],
    expected_format: str,
) -> tuple[dict[str, Any], str]:
    receipt = _strict_json(value, label)
    if not isinstance(receipt, dict):
        raise ResumableShardError(f"{label} must be an object")
    _exact_keys(receipt, expected_keys, label)
    if receipt.get("format") != expected_format:
        raise ResumableShardError(f"{label} format mismatch")
    return receipt, canonical_json_sha256(receipt)


def append_active_state_removed(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    removal_receipt: Mapping[str, Any],
) -> ReplayState:
    if state.terminal_search is None or state.active_state_removed is not None:
        raise ResumableShardError("active-state removal requires terminal staging")
    receipt, digest = _exact_receipt(
        removal_receipt,
        label="active-state removal receipt",
        expected_keys={
            "format",
            "plan_sha256",
            "terminal_stage_sha256",
            "owned_state_path_sha256",
            "owned_state_removed",
            "snapshots_retained",
        },
        expected_format=RESUME_STATE_REMOVAL_FORMAT,
    )
    if (
        receipt["plan_sha256"] != state.plan.sha256
        or receipt["terminal_stage_sha256"]
        != state.terminal_search["terminal_stage_sha256"]
        or receipt["owned_state_removed"] is not True
        or receipt["snapshots_retained"] is not True
    ):
        raise ResumableShardError("active-state removal receipt identity mismatch")
    _require_sha256(receipt["owned_state_path_sha256"], "owned-state path SHA")
    return journal.append(
        "active_state_removed",
        generation=state.generation,
        terminal_search_entry_sha256=state.terminal_search["entry_sha256"],
        state_removal_receipt_sha256=digest,
        owned_state_removed=True,
        snapshots_retained=True,
    )


def append_terminal_published(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    publication_receipt: Mapping[str, Any],
) -> ReplayState:
    if state.active_state_removed is None or state.terminal_published is not None:
        raise ResumableShardError("terminal publication is out of order")
    receipt, digest = _exact_receipt(
        publication_receipt,
        label="terminal publication receipt",
        expected_keys={
            "format",
            "terminal_stage_sha256",
            "official_artifact_path",
            "official_artifact_sha256",
            "official_trace_path",
            "official_trace_sha256",
            "outputs_verified",
        },
        expected_format=RESUME_PUBLICATION_FORMAT,
    )
    if (
        receipt["terminal_stage_sha256"]
        != state.terminal_search["terminal_stage_sha256"]
        or receipt["outputs_verified"] is not True
    ):
        raise ResumableShardError("terminal publication receipt identity mismatch")
    artifact_path = _safe_relative(
        receipt["official_artifact_path"], "official artifact path"
    )
    trace_path = _safe_relative(
        receipt["official_trace_path"], "official trace path"
    )
    artifact_sha = _require_sha256(
        receipt["official_artifact_sha256"], "official artifact SHA"
    )
    trace_sha = _require_sha256(
        receipt["official_trace_sha256"], "official trace SHA"
    )
    return journal.append(
        "terminal_published",
        generation=state.generation,
        terminal_search_entry_sha256=state.terminal_search["entry_sha256"],
        official_artifact_path=artifact_path,
        official_artifact_sha256=artifact_sha,
        official_trace_path=trace_path,
        official_trace_sha256=trace_sha,
        publication_receipt_sha256=digest,
        outputs_verified=True,
    )


def append_checkpoint_gc(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    checkpoint_gc_receipt: Mapping[str, Any],
) -> ReplayState:
    if (
        state.active_state_removed is None
        or state.terminal_published is None
        or state.checkpoint_gc is not None
    ):
        raise ResumableShardError("checkpoint GC is out of order")
    receipt, digest = _exact_receipt(
        checkpoint_gc_receipt,
        label="checkpoint GC receipt",
        expected_keys={
            "format",
            "publication_receipt_sha256",
            "snapshots_removed",
            "terminal_stage_removed",
        },
        expected_format=RESUME_CHECKPOINT_GC_FORMAT,
    )
    if (
        receipt["publication_receipt_sha256"]
        != state.terminal_published["publication_receipt_sha256"]
        or receipt["snapshots_removed"] is not True
        or receipt["terminal_stage_removed"] is not True
    ):
        raise ResumableShardError("checkpoint GC receipt identity mismatch")
    return journal.append(
        "checkpoint_gc",
        generation=state.generation,
        terminal_published_entry_sha256=state.terminal_published[
            "entry_sha256"
        ],
        active_state_removed_entry_sha256=state.active_state_removed[
            "entry_sha256"
        ],
        checkpoint_gc_receipt_sha256=digest,
        snapshots_removed=True,
        terminal_stage_removed=True,
    )


def append_cleanup_closed(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
    *,
    cleanup_receipt: Mapping[str, Any],
) -> ReplayState:
    if state.checkpoint_gc is None or state.cleanup_closed is not None:
        raise ResumableShardError("cleanup requires completed checkpoint GC")
    receipt, receipt_sha = _exact_receipt(
        cleanup_receipt,
        label="cleanup receipt",
        expected_keys={
            "format",
            "checkpoint_gc_receipt_sha256",
            "owned_state_removed",
            "snapshots_removed",
            "terminal_stage_removed",
            "official_outputs_retained",
        },
        expected_format=RESUME_CLEANUP_FORMAT,
    )
    if (
        receipt["checkpoint_gc_receipt_sha256"]
        != state.checkpoint_gc["checkpoint_gc_receipt_sha256"]
        or any(
            receipt[field] is not True
            for field in (
                "owned_state_removed",
                "snapshots_removed",
                "terminal_stage_removed",
                "official_outputs_retained",
            )
        )
    ):
        raise ResumableShardError("cleanup receipt is not terminally closed")
    return journal.append(
        "cleanup_closed",
        generation=state.generation,
        checkpoint_gc_entry_sha256=state.checkpoint_gc["entry_sha256"],
        cleanup_receipt_sha256=receipt_sha,
        owned_state_removed=True,
        snapshots_removed=True,
        terminal_stage_removed=True,
        official_outputs_retained=True,
    )


def verify_completed_add_receipts(
    state: ReplayState,
    *,
    ordered_batch_sha256s: Sequence[str],
) -> None:
    """Recheck every active commit against the freshly rebuilt corpus."""

    observed = tuple(ordered_batch_sha256s)
    if observed != state.plan.ordered_batch_sha256s:
        raise ResumableShardError("rebuilt ordered batch population changed")
    if len(state.commits) != state.committed_prefix:
        raise ResumableShardError("active commit population is incomplete")
    for ordinal, commit in enumerate(state.commits):
        if commit.get("ordinal") != ordinal:
            raise ResumableShardError("active commit ordinal is not contiguous")
        if commit.get("batch_sha256") != observed[ordinal]:
            raise ResumableShardError("active commit batch identity changed")
        if commit.get("cumulative_http_completed") != ordinal + 1:
            raise ResumableShardError("active commit HTTP prefix changed")


def restore_snapshot_to_fresh_state(
    *,
    snapshot_path: str | os.PathLike[str],
    destination_state_dir: str | os.PathLike[str],
    expected_authority_sha256: str,
    expected_manifest_sha256: str,
    expected_tree_sha256: str,
    expected_ownership_token_sha256: str,
) -> dict[str, Any]:
    """Copy one verified immutable snapshot to a fresh working directory."""

    source = _assert_no_link_ancestors(
        snapshot_path, label="immutable snapshot", require_final=True
    )
    destination = _assert_no_link_ancestors(
        destination_state_dir,
        label="resume destination",
        require_final=False,
    )
    if destination.exists():
        raise FileExistsError("resume destination state already exists")
    if not destination.parent.is_dir() or _paths_overlap(source, destination):
        raise ResumableShardError("resume destination is unsafe")
    verified = verify_immutable_state_snapshot(
        source,
        expected_authority_sha256=expected_authority_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_tree_sha256=expected_tree_sha256,
        expected_ownership_token_sha256=expected_ownership_token_sha256,
    )
    before = {
        field: verified[field]
        for field in (
            "snapshot_manifest_sha256",
            "snapshot_tree_sha256",
            "ownership_token_sha256",
            "file_count",
            "total_bytes",
        )
    }
    source_state = source / "state"
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent
        )
    )
    try:
        for source_path in sorted(
            source_state.rglob("*"), key=lambda value: value.as_posix()
        ):
            relative = source_path.relative_to(source_state)
            target = staging / relative
            if source_path.is_symlink():
                raise ResumableShardError("immutable snapshot contains a symlink")
            if source_path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source_path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
        _fsync_tree(staging)
        staged = state_tree_receipt(staging)
        for field in (
            "snapshot_manifest_sha256",
            "snapshot_tree_sha256",
            "ownership_token_sha256",
        ):
            if staged[field] != before[field]:
                raise ResumableShardError(f"restored state {field} mismatch")
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    after = state_tree_receipt(destination)
    for field in (
        "snapshot_manifest_sha256",
        "snapshot_tree_sha256",
        "ownership_token_sha256",
    ):
        if after[field] != before[field]:
            raise ResumableShardError(f"published restored state {field} mismatch")
    authority = verified["snapshot_authority"]
    body = {
        "format": RESUME_RESTORE_FORMAT,
        "plan_sha256": authority["plan_sha256"],
        "generation": authority["generation"],
        "restored_prefix": authority["committed_prefix"],
        "restore_authority_sha256": authority["authority_sha256"],
        "destination_path_sha256": _path_identity_sha256(destination),
        "owned_state_absent": False,
        "restored_snapshot_manifest_sha256": after[
            "snapshot_manifest_sha256"
        ],
        "restored_snapshot_tree_sha256": after["snapshot_tree_sha256"],
        "restored_ownership_token_sha256": after[
            "ownership_token_sha256"
        ],
    }
    return {**body, "restore_receipt_sha256": canonical_json_sha256(body)}


def remove_verified_snapshot_root(
    journal: AppendOnlyResumeJournal,
    state: ReplayState,
) -> dict[str, Any]:
    """Remove only the fully verified plan-owned checkpoint population.

    This operation is legal only after official terminal outputs are durably
    published.  If a prior process already completed the deletion but crashed
    before journaling it, absence is accepted because every checkpoint was
    previously authenticated by the immutable prefix-seal chain.
    """

    if state.terminal_published is None:
        raise ResumableShardError(
            "snapshot GC requires verified official terminal publication"
        )
    header = state.entries[0]
    if _path_identity_sha256(journal.path) != header.get("journal_path_sha256"):
        raise ResumableShardError("snapshot GC journal path identity changed")
    relative = _safe_relative(
        header.get("snapshot_root_path"), "snapshot_root_path"
    )
    root = _assert_no_link_ancestors(
        journal.path.parent / Path(relative),
        label="snapshot GC root",
        require_final=False,
    )
    if root == journal.path.parent or journal.path.parent not in root.parents:
        raise ResumableShardError("snapshot GC root escaped the journal directory")
    if not root.exists():
        return {
            "format": "memory-condense-mem0-snapshot-gc-v1",
            "plan_sha256": state.plan.sha256,
            "snapshot_root_path_sha256": _path_identity_sha256(root),
            "verified_snapshot_count": len(
                [row for row in state.entries if row.get("kind") == "prefix_sealed"]
            ),
            "already_absent": True,
            "snapshots_removed": True,
        }

    root = _ensure_snapshot_root(journal_path=journal.path, header=header)
    seals = [row for row in state.entries if row.get("kind") == "prefix_sealed"]
    if not seals or seals[-1].get("committed_prefix") != state.plan.authorized_add_operations:
        raise ResumableShardError("snapshot GC omitted the sealed full prefix")
    expected: dict[str, Mapping[str, Any]] = {}
    for seal in seals:
        snapshot_relative = _safe_relative(
            seal.get("snapshot_path"), "sealed snapshot path"
        )
        snapshot = _lexical_absolute(journal.path.parent / Path(snapshot_relative))
        if snapshot.parent != root:
            raise ResumableShardError("sealed snapshot is outside the owned root")
        if snapshot.name in expected:
            raise ResumableShardError("sealed snapshot path is repeated")
        expected[snapshot.name] = seal
    allowed_files = {SNAPSHOT_ROOT_MARKER, SNAPSHOT_ROOT_MARKER + ".sha256"}
    observed_dirs: set[str] = set()
    for child in root.iterdir():
        if child.name in allowed_files:
            if child.is_symlink() or not child.is_file():
                raise ResumableShardError("snapshot-root marker type changed")
            continue
        if child.name not in expected or child.is_symlink() or not child.is_dir():
            raise ResumableShardError("snapshot root contains an unowned entry")
        observed_dirs.add(child.name)
    if observed_dirs != set(expected):
        raise ResumableShardError("snapshot root population differs from the journal")
    for name, seal in expected.items():
        verify_immutable_state_snapshot(
            root / name,
            expected_authority_sha256=seal["snapshot_authority_sha256"],
            expected_manifest_sha256=seal["snapshot_manifest_sha256"],
            expected_tree_sha256=seal["snapshot_tree_sha256"],
            expected_ownership_token_sha256=seal["ownership_token_sha256"],
        )
    shutil.rmtree(root)
    if root.exists():
        raise ResumableShardError("verified snapshot root remained after GC")
    _fsync_directory(root.parent)
    return {
        "format": "memory-condense-mem0-snapshot-gc-v1",
        "plan_sha256": state.plan.sha256,
        "snapshot_root_path_sha256": _path_identity_sha256(root),
        "verified_snapshot_count": len(expected),
        "already_absent": False,
        "snapshots_removed": True,
    }


__all__ = [
    "AppendOnlyResumeJournal",
    "JournalLease",
    "OWNERSHIP_MARKER",
    "RESUME_CHECKPOINT_GC_FORMAT",
    "RESUME_CLEANUP_FORMAT",
    "RESUME_JOURNAL_FORMAT",
    "RESUME_PLAN_FORMAT",
    "RESUME_PREFIX_CLOSE_FORMAT",
    "RESUME_PUBLICATION_FORMAT",
    "RESUME_RESTORE_FORMAT",
    "RESUME_SNAPSHOT_FORMAT",
    "RESUME_STATE_REMOVAL_FORMAT",
    "RESUME_TERMINAL_FORMAT",
    "ReplayState",
    "ResumeAmbiguityError",
    "ResumeJournalLocked",
    "ResumePlan",
    "ResumableShardError",
    "append_active_state_removed",
    "append_checkpoint_gc",
    "append_cleanup_closed",
    "append_commit",
    "append_intent",
    "append_prefix_sealed",
    "append_presend_rollback",
    "append_send_attempt",
    "append_terminal_published",
    "append_terminal_search",
    "canonical_json_sha256",
    "checkpoint_authority_from_replay",
    "create_immutable_state_snapshot",
    "deterministic_user_scope",
    "new_journal_header",
    "prefix_close_receipt",
    "prefix_zero_restore_receipt",
    "publish_sealed_json",
    "read_journal",
    "read_sealed_json",
    "reconcile_published_checkpoint",
    "rehydration_material",
    "remove_verified_snapshot_root",
    "replay_journal",
    "restore_snapshot_to_fresh_state",
    "state_tree_receipt",
    "suffix_counter_seed",
    "verify_completed_add_receipts",
    "verify_immutable_state_snapshot",
]
