"""Provider-free scaffold for the nonofficial Mem0 ``infer=False`` arm.

This module deliberately does not import Mem0.  Its callable runner accepts an
injected, Memory-shaped test double and exercises the exact raw LongMemEval add
boundaries already derived by :mod:`tools.mem0_eval.protocol`.  The shipped CLI
does not bind a runtime, so these artifacts are always labelled test-only and
cannot be confused with the official ``infer=True`` comparison arm.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from memory_condense.eval.sample_identity import canonical_sha256

from .protocol import (
    CompositeAddBatch,
    RawStressShard,
    add_batches_sha256,
    validate_raw_stress_shard,
)


DIRECT_STORE_SCHEMA_VERSION = 2
DIRECT_STORE_SHARD_FORMAT = (
    "memory-condense-mem0-direct-store-retrieval-shard-v2"
)
DIRECT_STORE_ARM_ID = "mem0_direct_store_infer_false_nonofficial_v1"
DIRECT_STORE_ARM_LABEL = (
    "nonofficial_mem0_direct_store_infer_false_retrieval_ablation"
)
DIRECT_STORE_INGESTION_PROTOCOL = (
    "mem0-longmemeval-consecutive-slices-infer-false-v1"
)
DIRECT_STORE_SEARCH_PROTOCOL = (
    "mem0-top200-threshold0.1-rerank-false-explain-false-v1"
)
DIRECT_STORE_PROVENANCE_KIND = "exact_direct_store_add_batch"
DIRECT_STORE_RUNTIME_FORMAT = (
    "memory-condense-mem0-direct-store-injected-runtime-v2"
)
DIRECT_STORE_EXTRACTION_RECEIPT_FORMAT = (
    "memory-condense-mem0-zero-extraction-deny-receipt-v2"
)
DIRECT_STORE_CLEANUP_FORMAT = (
    "memory-condense-mem0-direct-store-injected-cleanup-v2"
)
FROZEN_SAMPLE_OFFSETS = tuple(range(0, 100, 10))
QUESTIONS_PER_SHARD = 10
SEARCH_TOP_K = 200
SEARCH_THRESHOLD = 0.1


class DirectStoreError(RuntimeError):
    """The nonofficial direct-store scaffold could not close exactly."""


class DirectStoreBackend(Protocol):
    """Small Memory-shaped seam used only through dependency injection."""

    llm: Any

    def add(self, *args: Any, **kwargs: Any) -> Any: ...

    def search(self, *args: Any, **kwargs: Any) -> Any: ...

    def runtime_receipt(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...

    def cleanup_receipt(self) -> Mapping[str, Any]: ...


def _canonical_copy(value: Any) -> Any:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DirectStoreError("value is not strict canonical JSON") from exc


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise DirectStoreError(
            f"{label} fields mismatch: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DirectStoreError(f"{label} must be normalized non-empty text")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DirectStoreError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise DirectStoreError(f"{label} must be a finite non-negative number")
    return float(value)


def _validate_runtime_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectStoreError("injected backend omitted its runtime receipt")
    receipt = _canonical_copy(value)
    assert isinstance(receipt, dict)
    expected = {
        "format",
        "execution_kind",
        "backend_label",
        "actual_mem0_executed",
        "dependency_versions",
        "local_only",
        "network_calls_authorized",
        "network_calls_observed",
        "provider_calls_authorized",
        "provider_calls_observed",
    }
    _exact_keys(receipt, expected, "runtime receipt")
    required = {
        "format": DIRECT_STORE_RUNTIME_FORMAT,
        "execution_kind": "injected_test_double",
        "actual_mem0_executed": False,
        "local_only": True,
        "network_calls_authorized": 0,
        "network_calls_observed": 0,
        "provider_calls_authorized": 0,
        "provider_calls_observed": 0,
    }
    for field, expected_value in required.items():
        if receipt.get(field) != expected_value:
            raise DirectStoreError(f"runtime receipt {field} mismatch")
    _nonempty(receipt.get("backend_label"), "runtime backend_label")
    versions = receipt.get("dependency_versions")
    if not isinstance(versions, Mapping):
        raise DirectStoreError("runtime dependency_versions must be a mapping")
    for name, version in versions.items():
        _nonempty(name, "dependency name")
        _nonempty(version, f"dependency version for {name}")
    return receipt


def _validate_cleanup_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectStoreError("injected backend omitted its cleanup receipt")
    receipt = _canonical_copy(value)
    assert isinstance(receipt, dict)
    _exact_keys(
        receipt,
        {
            "format",
            "closed",
            "owned_state_removed",
            "network_calls_observed",
            "provider_calls_observed",
        },
        "cleanup receipt",
    )
    required = {
        "format": DIRECT_STORE_CLEANUP_FORMAT,
        "closed": True,
        "owned_state_removed": True,
        "network_calls_observed": 0,
        "provider_calls_observed": 0,
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise DirectStoreError(f"cleanup receipt {field} mismatch")
    return receipt


def _same_callable(left: Any, right: Any) -> bool:
    if left is right:
        return True
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
        and getattr(left, "__func__", None) is not None
    )


@dataclass(slots=True)
class ZeroExtractionDenyMeter:
    """Deny and count every logical extraction call while supervising adds."""

    authorized_adds: int
    extraction_attempted: int = 0
    extraction_completed: int = 0
    extraction_failed: int = 0
    extraction_rejected: int = 0
    infer_false_adds_started: int = 0
    infer_false_adds_completed: int = 0
    infer_false_adds_failed: int = 0
    _inside_add: bool = False
    _installed: bool = False

    def __post_init__(self) -> None:
        _nonnegative_int(self.authorized_adds, "authorized_adds")

    def wrap_generate(self, callback: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(callback):
            raise DirectStoreError(
                "Memory.llm.generate_response must be callable for deny metering"
            )

        def denied(*_args: Any, **_kwargs: Any) -> Any:
            self.extraction_attempted += 1
            self.extraction_rejected += 1
            raise DirectStoreError(
                "infer=False direct-store ingestion attempted logical extraction"
            )

        return denied

    def wrap_add(self, callback: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(callback):
            raise DirectStoreError("Memory.add must be callable")

        def supervised(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("infer") is not False:
                raise DirectStoreError(
                    "direct-store ingestion requires explicit infer=False"
                )
            if self._inside_add:
                raise DirectStoreError("nested Memory.add calls are not allowed")
            if self.infer_false_adds_started >= self.authorized_adds:
                raise DirectStoreError("direct-store add authorization exhausted")
            extraction_before = self.extraction_attempted
            self._inside_add = True
            self.infer_false_adds_started += 1
            operation_error: BaseException | None = None
            try:
                result = callback(*args, **kwargs)
            except BaseException as exc:
                operation_error = exc
                self.infer_false_adds_failed += 1
                raise
            else:
                if self.extraction_attempted != extraction_before:
                    self.infer_false_adds_failed += 1
                    raise DirectStoreError(
                        "infer=False add attempted extraction even though the "
                        "backend swallowed the deny exception"
                    )
                self.infer_false_adds_completed += 1
                return result
            finally:
                self._inside_add = False
                if (
                    operation_error is not None
                    and self.extraction_attempted != extraction_before
                ):
                    operation_error.add_note(
                        "infer=False add also attempted forbidden extraction"
                    )

        return supervised

    def assert_complete(self) -> None:
        observed = {
            "extraction_attempted": self.extraction_attempted,
            "extraction_completed": self.extraction_completed,
            "extraction_failed": self.extraction_failed,
            "extraction_rejected": self.extraction_rejected,
            "infer_false_adds_started": self.infer_false_adds_started,
            "infer_false_adds_completed": self.infer_false_adds_completed,
            "infer_false_adds_failed": self.infer_false_adds_failed,
        }
        expected = {
            "extraction_attempted": 0,
            "extraction_completed": 0,
            "extraction_failed": 0,
            "extraction_rejected": 0,
            "infer_false_adds_started": self.authorized_adds,
            "infer_false_adds_completed": self.authorized_adds,
            "infer_false_adds_failed": 0,
        }
        if observed != expected or self._inside_add or not self._installed:
            raise DirectStoreError(
                "zero-extraction/add receipt did not close exactly: "
                f"expected={expected!r}, observed={observed!r}"
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "format": DIRECT_STORE_EXTRACTION_RECEIPT_FORMAT,
            "boundary": "Memory.llm.generate_response",
            "mode": "deny_all_during_infer_false_direct_store",
            "authorized": 0,
            "attempted": self.extraction_attempted,
            "completed": self.extraction_completed,
            "failed": self.extraction_failed,
            "rejected": self.extraction_rejected,
            "infer_false_adds_authorized": self.authorized_adds,
            "infer_false_adds_started": self.infer_false_adds_started,
            "infer_false_adds_completed": self.infer_false_adds_completed,
            "infer_false_adds_failed": self.infer_false_adds_failed,
            "zero_extraction_calls_certified": (
                self._installed
                and not self._inside_add
                and self.extraction_attempted == 0
                and self.extraction_completed == 0
                and self.extraction_failed == 0
                and self.extraction_rejected == 0
                and self.infer_false_adds_started == self.authorized_adds
                and self.infer_false_adds_completed == self.authorized_adds
                and self.infer_false_adds_failed == 0
            ),
            "external_http_attempts_certified": False,
            "external_provider_persistence_certified": False,
        }


def install_zero_extraction_deny_meter(
    backend: DirectStoreBackend,
    meter: ZeroExtractionDenyMeter,
) -> Callable[[], None]:
    """Patch the injected Memory-shaped boundary and return a strict restore."""

    if meter._installed:
        raise DirectStoreError("zero-extraction meter can be installed only once")
    llm = getattr(backend, "llm", None)
    if llm is None:
        raise DirectStoreError("injected Memory-shaped backend omitted llm")
    original_generate = getattr(llm, "generate_response", None)
    original_add = getattr(backend, "add", None)
    wrapped_generate = meter.wrap_generate(original_generate)
    wrapped_add = meter.wrap_add(original_add)
    generate_installed = False
    add_installed = False
    try:
        setattr(llm, "generate_response", wrapped_generate)
        generate_installed = True
        if getattr(llm, "generate_response", None) is not wrapped_generate:
            raise DirectStoreError("could not install extraction deny wrapper")
        setattr(backend, "add", wrapped_add)
        add_installed = True
        if getattr(backend, "add", None) is not wrapped_add:
            raise DirectStoreError("could not install infer=False add supervisor")
    except BaseException:
        if add_installed:
            setattr(backend, "add", original_add)
        if generate_installed:
            setattr(llm, "generate_response", original_generate)
        raise
    meter._installed = True
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            raise DirectStoreError("direct-store wrappers were already restored")
        errors: list[str] = []
        if getattr(backend, "add", None) is not wrapped_add:
            errors.append("Memory.add wrapper changed before restoration")
        if getattr(llm, "generate_response", None) is not wrapped_generate:
            errors.append("extraction deny wrapper changed before restoration")
        try:
            setattr(backend, "add", original_add)
        except BaseException as exc:
            errors.append(f"Memory.add restoration failed: {type(exc).__name__}")
        try:
            setattr(llm, "generate_response", original_generate)
        except BaseException as exc:
            errors.append(
                "extraction wrapper restoration failed: "
                f"{type(exc).__name__}"
            )
        if not _same_callable(getattr(backend, "add", None), original_add):
            errors.append("Memory.add restoration could not be verified")
        if not _same_callable(
            getattr(llm, "generate_response", None), original_generate
        ):
            errors.append("extraction wrapper restoration could not be verified")
        restored = not errors
        if errors:
            raise DirectStoreError("; ".join(errors))

    return restore


def _batch_source(batch: CompositeAddBatch) -> dict[str, Any]:
    return {
        "source_sample_id": batch.source_sample_id,
        "source": batch.source,
        "date": batch.date,
        "session_index": batch.session_index,
        "original_session_index": batch.original_session_index,
        "batch_index": batch.batch_index,
        "turn_start": batch.turn_start,
        "message_count": len(batch.messages),
        "roles": [role for role, _content in batch.messages],
    }


def _messages_payload(batch: CompositeAddBatch) -> list[dict[str, str]]:
    return [
        {"role": role, "content": content}
        for role, content in batch.messages
    ]


def _response_rows(response: Any, operation: str) -> list[Mapping[str, Any]]:
    rows: Any
    if isinstance(response, Mapping) and set(response) == {"results"}:
        rows = response["results"]
    elif isinstance(response, Sequence) and not isinstance(
        response, (str, bytes, bytearray)
    ):
        rows = response
    elif isinstance(response, Mapping) and (
        "id" in response or "memory_id" in response
    ):
        rows = [response]
    else:
        raise DirectStoreError(f"{operation} returned an unsupported response")
    if not isinstance(rows, Sequence) or isinstance(
        rows, (str, bytes, bytearray)
    ):
        raise DirectStoreError(f"{operation} results must be a sequence")
    normalized: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DirectStoreError(f"{operation} result {index} is not a mapping")
        normalized.append(row)
    return normalized


def _memory_id(row: Mapping[str, Any], label: str) -> str:
    first = row.get("id")
    second = row.get("memory_id")
    if first is not None and second is not None and str(first) != str(second):
        raise DirectStoreError(f"{label} has conflicting memory IDs")
    value = first if first is not None else second
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise DirectStoreError(f"{label} omitted its memory ID")
    return _nonempty(str(value), f"{label} memory ID")


def _add_memory_ids(response: Any) -> list[str]:
    rows = _response_rows(response, "direct-store add")
    if not rows:
        raise DirectStoreError("infer=False add returned no stored memory IDs")
    ids = [_memory_id(row, f"add result {index}") for index, row in enumerate(rows)]
    if len(ids) != len(set(ids)):
        raise DirectStoreError("one infer=False add returned duplicate memory IDs")
    return ids


def _candidate_payload(
    row: Mapping[str, Any],
    *,
    rank: int,
    source_by_memory_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    memory_id = _memory_id(row, f"search result {rank}")
    source = source_by_memory_id.get(memory_id)
    if source is None:
        raise DirectStoreError(
            f"search returned memory {memory_id!r} outside the exact add ledger"
        )
    first_text = row.get("memory")
    second_text = row.get("text")
    if (
        first_text is not None
        and second_text is not None
        and first_text != second_text
    ):
        raise DirectStoreError(f"search result {rank} has conflicting memory text")
    text = first_text if first_text is not None else second_text
    if not isinstance(text, str) or not text.strip():
        raise DirectStoreError(f"search result {rank} has no memory text")
    score_value = row.get("score")
    if score_value is None:
        score = None
    elif (
        isinstance(score_value, bool)
        or not isinstance(score_value, (int, float))
        or not math.isfinite(float(score_value))
    ):
        raise DirectStoreError(f"search result {rank} has invalid score")
    else:
        score = float(score_value)
    created_at = row.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        raise DirectStoreError(f"search result {rank} has no created_at")
    payload = {
        "rank": rank,
        "memory_id": memory_id,
        "text": text,
        "score": score,
        "created_at": created_at.strip(),
        "provenance_kind": DIRECT_STORE_PROVENANCE_KIND,
        "source": _canonical_copy(source),
    }
    payload["candidate_sha256"] = canonical_sha256(payload)
    return payload


def _arm_receipt() -> dict[str, Any]:
    return {
        "arm_id": DIRECT_STORE_ARM_ID,
        "label": DIRECT_STORE_ARM_LABEL,
        "official_mem0_comparison": False,
        "infer": False,
        "extraction_enabled": False,
        "result_class": "injected_test_only",
        "benchmark_result_eligible": False,
    }


def _artifact_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    digest = body.pop("artifact_sha256", None)
    if digest is None:
        raise DirectStoreError("direct-store artifact omitted artifact_sha256")
    if digest != canonical_sha256(body):
        raise DirectStoreError("direct-store artifact SHA-256 mismatch")
    return body


def run_injected_direct_store_shard(
    shard: RawStressShard,
    *,
    backend_factory: Callable[[], DirectStoreBackend],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one test-only retrieval shard with exact ``infer=False`` calls.

    No official or benchmark-eligible receipt can be emitted here.  A real
    Mem0 binding needs a separate audited launcher; the public CLI therefore
    exposes preflight and merge only.
    """

    if not callable(backend_factory):
        raise TypeError("backend_factory must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")
    source_receipt = validate_raw_stress_shard(shard)
    if shard.sample_offset not in FROZEN_SAMPLE_OFFSETS:
        raise DirectStoreError("sample offset is outside the frozen 100Q campaign")
    if len(shard.question_ids) != QUESTIONS_PER_SHARD:
        raise DirectStoreError("a direct-store shard must contain exactly 10 questions")

    backend = backend_factory()
    if backend is None:
        raise DirectStoreError("backend_factory returned no backend")
    meter = ZeroExtractionDenyMeter(shard.add_counts.add_requests)
    runtime_before: dict[str, Any] | None = None
    restore: Callable[[], None] | None = None
    primary_error: BaseException | None = None
    cleanup_receipt: dict[str, Any] | None = None
    add_ledger: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    source_by_memory_id: dict[str, Mapping[str, Any]] = {}
    total_add_seconds = 0.0
    total_search_seconds = 0.0
    scope = (
        f"direct-store-{shard.sample_offset:03d}-{shard.sample_sha256[:16]}"
    )

    try:
        runtime_reader = getattr(backend, "runtime_receipt", None)
        if not callable(runtime_reader):
            raise DirectStoreError("injected backend omitted runtime_receipt()")
        runtime_before = _validate_runtime_receipt(runtime_reader())
        restore = install_zero_extraction_deny_meter(backend, meter)
        for ordinal, batch in enumerate(shard.add_batches, start=1):
            source = _batch_source(batch)
            batch_id = f"add-{ordinal:05d}"
            messages = _messages_payload(batch)
            request = {
                "messages": messages,
                "user_id": scope,
                "infer": False,
            }
            started = float(clock())
            response = backend.add(messages, user_id=scope, infer=False)
            elapsed = max(0.0, float(clock()) - started)
            total_add_seconds += elapsed
            memory_ids = _add_memory_ids(response)
            duplicate = next(
                (memory_id for memory_id in memory_ids if memory_id in source_by_memory_id),
                None,
            )
            if duplicate is not None:
                raise DirectStoreError(
                    f"memory ID {duplicate!r} was returned by multiple add calls"
                )
            source_row = {
                "batch_id": batch_id,
                "add_ordinal": ordinal,
                **source,
                "messages_sha256": canonical_sha256(messages),
                "add_request_sha256": canonical_sha256(request),
            }
            for memory_id in memory_ids:
                source_by_memory_id[memory_id] = source_row
            ledger_row = {
                **source_row,
                "returned_memory_ids": memory_ids,
            }
            ledger_row["ledger_row_sha256"] = canonical_sha256(ledger_row)
            add_ledger.append(ledger_row)

        for ordinal, question in enumerate(shard.parsed_sample.questions, start=1):
            question_id = _nonempty(question.question_id, "question_id")
            query = _nonempty(question.question, f"question {question_id}")
            started = float(clock())
            response = backend.search(
                query,
                top_k=SEARCH_TOP_K,
                filters={"user_id": scope},
                threshold=SEARCH_THRESHOLD,
                rerank=False,
                explain=False,
            )
            elapsed = max(0.0, float(clock()) - started)
            total_search_seconds += elapsed
            candidates = [
                _candidate_payload(
                    row,
                    rank=rank,
                    source_by_memory_id=source_by_memory_id,
                )
                for rank, row in enumerate(
                    _response_rows(response, "direct-store search"), start=1
                )
            ]
            candidate_ids = [row["memory_id"] for row in candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise DirectStoreError(
                    f"search for {question_id!r} returned duplicate memory IDs"
                )
            if len(candidates) > SEARCH_TOP_K:
                raise DirectStoreError("search returned more than the frozen top-k")
            search_row = {
                "question_ordinal": ordinal,
                "question_id": question_id,
                "query": query,
                "query_sha256": canonical_sha256(query),
                "search_kwargs": {
                    "top_k": SEARCH_TOP_K,
                    "filters": {"user_id": scope},
                    "threshold": SEARCH_THRESHOLD,
                    "rerank": False,
                    "explain": False,
                },
                "search_elapsed_seconds": elapsed,
                "candidates": candidates,
                "candidates_sha256": canonical_sha256(candidates),
            }
            search_row["search_row_sha256"] = canonical_sha256(search_row)
            searches.append(search_row)

        meter.assert_complete()
        runtime_after = _validate_runtime_receipt(runtime_reader())
        if runtime_after != runtime_before:
            raise DirectStoreError("injected backend runtime receipt changed during run")
        validate_raw_stress_shard(shard)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if restore is not None:
            try:
                restore()
            except BaseException as exc:
                cleanup_errors.append(exc)
        close = getattr(backend, "close", None)
        if not callable(close):
            cleanup_errors.append(
                DirectStoreError("injected backend omitted close()")
            )
        else:
            try:
                close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        receipt_reader = getattr(backend, "cleanup_receipt", None)
        if not callable(receipt_reader):
            cleanup_errors.append(
                DirectStoreError("injected backend omitted cleanup_receipt()")
            )
        else:
            try:
                cleanup_receipt = _validate_cleanup_receipt(receipt_reader())
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if primary_error is not None:
                for error in cleanup_errors:
                    primary_error.add_note(
                        f"cleanup failure: {type(error).__name__}: {error}"
                    )
            elif len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            else:
                raise BaseExceptionGroup(
                    "direct-store cleanup failed", cleanup_errors
                )

    assert cleanup_receipt is not None
    assert runtime_before is not None
    extraction_receipt = meter.receipt()
    artifact: dict[str, Any] = {
        "format": DIRECT_STORE_SHARD_FORMAT,
        "schema_version": DIRECT_STORE_SCHEMA_VERSION,
        "status": "complete_injected_test_only",
        "arm": _arm_receipt(),
        "sample": {
            "sample_offset": shard.sample_offset,
            "sample_id": shard.parsed_sample.sample_id,
            "sample_sha256": shard.sample_sha256,
            "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
            "history_sample_ids": list(shard.history_sample_ids),
            "history_sample_ids_sha256": source_receipt[
                "history_sample_ids_sha256"
            ],
            "ordered_question_ids": list(shard.question_ids),
            "ordered_question_ids_sha256": source_receipt[
                "question_ids_sha256"
            ],
            "sorted_unique_question_ids_sha256": canonical_sha256(
                sorted(set(shard.question_ids))
            ),
        },
        "protocol": {
            "ingestion": DIRECT_STORE_INGESTION_PROTOCOL,
            "search": DIRECT_STORE_SEARCH_PROTOCOL,
            "official_longmemeval_add_boundaries_preserved": True,
            "official_mem0_extraction_protocol": False,
            "infer": False,
            "source_dates_supplied_to_mem0": False,
            "source_metadata_supplied_to_mem0": False,
        },
        "runtime": runtime_before,
        "cleanup": cleanup_receipt,
        "extraction": extraction_receipt,
        "ingestion": {
            "scope": scope,
            "raw_pairs": shard.add_counts.raw_pairs,
            "skipped_empty_pairs": shard.add_counts.skipped_empty_pairs,
            "add_operations": len(add_ledger),
            "add_batches_sha256": add_batches_sha256(shard.add_batches),
            "add_ledger_sha256": canonical_sha256(add_ledger),
            "returned_memory_ids": len(source_by_memory_id),
            "add_elapsed_seconds": total_add_seconds,
            "add_ledger": add_ledger,
        },
        "retrieval": {
            "search_operations": len(searches),
            "search_elapsed_seconds": total_search_seconds,
            "ordered_question_ids": [row["question_id"] for row in searches],
            "ordered_question_ids_sha256": canonical_sha256(
                [row["question_id"] for row in searches]
            ),
            "sorted_unique_question_ids_sha256": canonical_sha256(
                sorted({row["question_id"] for row in searches})
            ),
            "searches_sha256": canonical_sha256(searches),
            "searches": searches,
        },
        "provider_calls": {
            "authorized": 0,
            "attempted": 0,
            "completed": 0,
        },
        "network_calls": {
            "authorized": 0,
            "observed": 0,
            "evidence_kind": "injected_test_double_receipt_not_os_attestation",
        },
        "limitations": {
            "actual_mem0_executed": False,
            "official_mem0_comparison": False,
            "resource_matching_completed": False,
            "extraction_quality_measured": False,
            "answer_generation_or_judging_performed": False,
            "runtime_claim_is_noncertifying": True,
        },
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return validate_direct_store_shard_artifact(artifact, shard)


def validate_direct_store_shard_artifact(
    value: Mapping[str, Any],
    shard: RawStressShard,
) -> dict[str, Any]:
    """Independently validate one test-only shard against its raw source."""

    if not isinstance(value, Mapping):
        raise DirectStoreError("direct-store shard artifact must be a mapping")
    artifact = _canonical_copy(value)
    assert isinstance(artifact, dict)
    _artifact_body(artifact)
    _exact_keys(
        artifact,
        {
            "format",
            "schema_version",
            "status",
            "arm",
            "sample",
            "protocol",
            "runtime",
            "cleanup",
            "extraction",
            "ingestion",
            "retrieval",
            "provider_calls",
            "network_calls",
            "limitations",
            "artifact_sha256",
        },
        "direct-store shard artifact",
    )
    if artifact["format"] != DIRECT_STORE_SHARD_FORMAT:
        raise DirectStoreError("direct-store shard format mismatch")
    if artifact["schema_version"] != DIRECT_STORE_SCHEMA_VERSION:
        raise DirectStoreError("direct-store shard schema version mismatch")
    if artifact["status"] != "complete_injected_test_only":
        raise DirectStoreError("direct-store shard status mismatch")
    if artifact["arm"] != _arm_receipt():
        raise DirectStoreError("direct-store shard arm label mismatch")

    source_receipt = validate_raw_stress_shard(shard)
    sample = artifact.get("sample")
    if not isinstance(sample, Mapping):
        raise DirectStoreError("direct-store sample receipt is invalid")
    expected_sample = {
        "sample_offset": shard.sample_offset,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_sha256": shard.sample_sha256,
        "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
        "history_sample_ids": list(shard.history_sample_ids),
        "history_sample_ids_sha256": source_receipt[
            "history_sample_ids_sha256"
        ],
        "ordered_question_ids": list(shard.question_ids),
        "ordered_question_ids_sha256": source_receipt[
            "question_ids_sha256"
        ],
        "sorted_unique_question_ids_sha256": canonical_sha256(
            sorted(set(shard.question_ids))
        ),
    }
    if sample != expected_sample:
        raise DirectStoreError("direct-store sample receipt mismatch")
    expected_protocol = {
        "ingestion": DIRECT_STORE_INGESTION_PROTOCOL,
        "search": DIRECT_STORE_SEARCH_PROTOCOL,
        "official_longmemeval_add_boundaries_preserved": True,
        "official_mem0_extraction_protocol": False,
        "infer": False,
        "source_dates_supplied_to_mem0": False,
        "source_metadata_supplied_to_mem0": False,
    }
    if artifact.get("protocol") != expected_protocol:
        raise DirectStoreError("direct-store protocol receipt mismatch")
    _validate_runtime_receipt(artifact.get("runtime"))
    _validate_cleanup_receipt(artifact.get("cleanup"))

    extraction = artifact.get("extraction")
    if not isinstance(extraction, Mapping):
        raise DirectStoreError("zero-extraction receipt is invalid")
    expected_extraction = {
        "format": DIRECT_STORE_EXTRACTION_RECEIPT_FORMAT,
        "boundary": "Memory.llm.generate_response",
        "mode": "deny_all_during_infer_false_direct_store",
        "authorized": 0,
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "rejected": 0,
        "infer_false_adds_authorized": shard.add_counts.add_requests,
        "infer_false_adds_started": shard.add_counts.add_requests,
        "infer_false_adds_completed": shard.add_counts.add_requests,
        "infer_false_adds_failed": 0,
        "zero_extraction_calls_certified": True,
        "external_http_attempts_certified": False,
        "external_provider_persistence_certified": False,
    }
    if extraction != expected_extraction:
        raise DirectStoreError("zero-extraction receipt did not close exactly")

    ingestion = artifact.get("ingestion")
    if not isinstance(ingestion, Mapping):
        raise DirectStoreError("direct-store ingestion receipt is invalid")
    _exact_keys(
        ingestion,
        {
            "scope",
            "raw_pairs",
            "skipped_empty_pairs",
            "add_operations",
            "add_batches_sha256",
            "add_ledger_sha256",
            "returned_memory_ids",
            "add_elapsed_seconds",
            "add_ledger",
        },
        "direct-store ingestion receipt",
    )
    expected_scope = (
        f"direct-store-{shard.sample_offset:03d}-{shard.sample_sha256[:16]}"
    )
    for field, expected in {
        "scope": expected_scope,
        "raw_pairs": shard.add_counts.raw_pairs,
        "skipped_empty_pairs": shard.add_counts.skipped_empty_pairs,
        "add_operations": shard.add_counts.add_requests,
        "add_batches_sha256": add_batches_sha256(shard.add_batches),
    }.items():
        if ingestion.get(field) != expected:
            raise DirectStoreError(f"direct-store ingestion {field} mismatch")
    _nonnegative_number(
        ingestion.get("add_elapsed_seconds"), "add_elapsed_seconds"
    )
    ledger = ingestion.get("add_ledger")
    if not isinstance(ledger, list) or len(ledger) != len(shard.add_batches):
        raise DirectStoreError("direct-store add ledger length mismatch")
    known: dict[str, dict[str, Any]] = {}
    for ordinal, (row, batch) in enumerate(
        zip(ledger, shard.add_batches), start=1
    ):
        if not isinstance(row, Mapping):
            raise DirectStoreError(f"add ledger row {ordinal} is invalid")
        source = _batch_source(batch)
        messages = _messages_payload(batch)
        source_row = {
            "batch_id": f"add-{ordinal:05d}",
            "add_ordinal": ordinal,
            **source,
            "messages_sha256": canonical_sha256(messages),
            "add_request_sha256": canonical_sha256(
                {
                    "messages": messages,
                    "user_id": expected_scope,
                    "infer": False,
                }
            ),
        }
        memory_ids = row.get("returned_memory_ids")
        if (
            not isinstance(memory_ids, list)
            or not memory_ids
            or any(not isinstance(item, str) or not item for item in memory_ids)
            or len(memory_ids) != len(set(memory_ids))
        ):
            raise DirectStoreError(
                f"add ledger row {ordinal} returned-memory IDs are invalid"
            )
        expected_row = {**source_row, "returned_memory_ids": memory_ids}
        expected_row["ledger_row_sha256"] = canonical_sha256(expected_row)
        if row != expected_row:
            raise DirectStoreError(f"add ledger row {ordinal} mismatch")
        for memory_id in memory_ids:
            if memory_id in known:
                raise DirectStoreError(
                    "one memory ID is attributed to more than one add batch"
                )
            known[memory_id] = source_row
    if ingestion.get("add_ledger_sha256") != canonical_sha256(ledger):
        raise DirectStoreError("direct-store add-ledger SHA-256 mismatch")
    if ingestion.get("returned_memory_ids") != len(known):
        raise DirectStoreError("returned-memory count mismatch")

    retrieval = artifact.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise DirectStoreError("direct-store retrieval receipt is invalid")
    _exact_keys(
        retrieval,
        {
            "search_operations",
            "search_elapsed_seconds",
            "ordered_question_ids",
            "ordered_question_ids_sha256",
            "sorted_unique_question_ids_sha256",
            "searches_sha256",
            "searches",
        },
        "direct-store retrieval receipt",
    )
    searches = retrieval.get("searches")
    if not isinstance(searches, list) or len(searches) != len(shard.question_ids):
        raise DirectStoreError("direct-store search count mismatch")
    expected_question_ids = list(shard.question_ids)
    if retrieval.get("ordered_question_ids") != expected_question_ids:
        raise DirectStoreError("direct-store question ordering mismatch")
    if retrieval.get("ordered_question_ids_sha256") != canonical_sha256(
        expected_question_ids
    ):
        raise DirectStoreError("direct-store ordered-question SHA-256 mismatch")
    if retrieval.get("sorted_unique_question_ids_sha256") != canonical_sha256(
        sorted(set(expected_question_ids))
    ):
        raise DirectStoreError(
            "direct-store sorted-unique-question SHA-256 mismatch"
        )
    if retrieval.get("search_operations") != len(expected_question_ids):
        raise DirectStoreError("direct-store search operation count mismatch")
    _nonnegative_number(
        retrieval.get("search_elapsed_seconds"), "search_elapsed_seconds"
    )
    for ordinal, (row, question) in enumerate(
        zip(searches, shard.parsed_sample.questions), start=1
    ):
        if not isinstance(row, Mapping):
            raise DirectStoreError(f"search row {ordinal} is invalid")
        _exact_keys(
            row,
            {
                "question_ordinal",
                "question_id",
                "query",
                "query_sha256",
                "search_kwargs",
                "search_elapsed_seconds",
                "candidates",
                "candidates_sha256",
                "search_row_sha256",
            },
            f"search row {ordinal}",
        )
        body = dict(row)
        digest = body.pop("search_row_sha256", None)
        if digest != canonical_sha256(body):
            raise DirectStoreError(f"search row {ordinal} SHA-256 mismatch")
        if row.get("question_ordinal") != ordinal:
            raise DirectStoreError(f"search row {ordinal} ordinal mismatch")
        if row.get("question_id") != question.question_id:
            raise DirectStoreError(f"search row {ordinal} question ID mismatch")
        if row.get("query") != question.question:
            raise DirectStoreError(f"search row {ordinal} query mismatch")
        if row.get("query_sha256") != canonical_sha256(question.question):
            raise DirectStoreError(f"search row {ordinal} query SHA-256 mismatch")
        if row.get("search_kwargs") != {
            "top_k": SEARCH_TOP_K,
            "filters": {"user_id": expected_scope},
            "threshold": SEARCH_THRESHOLD,
            "rerank": False,
            "explain": False,
        }:
            raise DirectStoreError(f"search row {ordinal} kwargs mismatch")
        _nonnegative_number(
            row.get("search_elapsed_seconds"),
            f"search row {ordinal} elapsed seconds",
        )
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) > SEARCH_TOP_K:
            raise DirectStoreError(f"search row {ordinal} candidates are invalid")
        seen_candidates: set[str] = set()
        for rank, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, Mapping):
                raise DirectStoreError(
                    f"search row {ordinal} candidate {rank} is invalid"
                )
            _exact_keys(
                candidate,
                {
                    "rank",
                    "memory_id",
                    "text",
                    "score",
                    "created_at",
                    "provenance_kind",
                    "source",
                    "candidate_sha256",
                },
                f"search row {ordinal} candidate {rank}",
            )
            candidate_body = dict(candidate)
            candidate_digest = candidate_body.pop("candidate_sha256", None)
            if candidate_digest != canonical_sha256(candidate_body):
                raise DirectStoreError(
                    f"search row {ordinal} candidate {rank} SHA-256 mismatch"
                )
            memory_id = _nonempty(
                candidate.get("memory_id"),
                f"search row {ordinal} candidate {rank} memory_id",
            )
            if memory_id not in known or memory_id in seen_candidates:
                raise DirectStoreError(
                    f"search row {ordinal} candidate provenance is invalid"
                )
            seen_candidates.add(str(memory_id))
            if candidate.get("rank") != rank:
                raise DirectStoreError(
                    f"search row {ordinal} candidate rank mismatch"
                )
            if candidate.get("provenance_kind") != DIRECT_STORE_PROVENANCE_KIND:
                raise DirectStoreError(
                    f"search row {ordinal} candidate provenance kind mismatch"
                )
            if candidate.get("source") != known[str(memory_id)]:
                raise DirectStoreError(
                    f"search row {ordinal} candidate source mismatch"
                )
            text = candidate.get("text")
            if not isinstance(text, str) or not text.strip():
                raise DirectStoreError(
                    f"search row {ordinal} candidate {rank} text is invalid"
                )
            score = candidate.get("score")
            if score is not None and (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise DirectStoreError(
                    f"search row {ordinal} candidate {rank} score is invalid"
                )
            _nonempty(
                candidate.get("created_at"),
                f"search row {ordinal} candidate {rank} created_at",
            )
        if row.get("candidates_sha256") != canonical_sha256(candidates):
            raise DirectStoreError(
                f"search row {ordinal} candidates SHA-256 mismatch"
            )
    if retrieval.get("searches_sha256") != canonical_sha256(searches):
        raise DirectStoreError("direct-store searches SHA-256 mismatch")

    if artifact.get("provider_calls") != {
        "authorized": 0,
        "attempted": 0,
        "completed": 0,
    }:
        raise DirectStoreError("direct-store provider-call receipt mismatch")
    if artifact.get("network_calls") != {
        "authorized": 0,
        "observed": 0,
        "evidence_kind": "injected_test_double_receipt_not_os_attestation",
    }:
        raise DirectStoreError("direct-store network-call receipt mismatch")
    if artifact.get("limitations") != {
        "actual_mem0_executed": False,
        "official_mem0_comparison": False,
        "resource_matching_completed": False,
        "extraction_quality_measured": False,
        "answer_generation_or_judging_performed": False,
        "runtime_claim_is_noncertifying": True,
    }:
        raise DirectStoreError("direct-store limitations receipt mismatch")
    return artifact


def _atomic_create(path: Path, payload: bytes) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with staging.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, target)
        except FileExistsError as exc:
            raise DirectStoreError(f"refusing to overwrite {target}") from exc
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def save_direct_store_artifact(value: Mapping[str, Any], path: str | Path) -> None:
    """Save one already self-hashed artifact atomically without clobbering."""

    payload = json.dumps(
        _canonical_copy(value),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    _atomic_create(Path(path), payload)


__all__ = [
    "DIRECT_STORE_ARM_ID",
    "DIRECT_STORE_ARM_LABEL",
    "DIRECT_STORE_CLEANUP_FORMAT",
    "DIRECT_STORE_EXTRACTION_RECEIPT_FORMAT",
    "DIRECT_STORE_RUNTIME_FORMAT",
    "DIRECT_STORE_SCHEMA_VERSION",
    "DIRECT_STORE_SHARD_FORMAT",
    "DirectStoreError",
    "FROZEN_SAMPLE_OFFSETS",
    "QUESTIONS_PER_SHARD",
    "ZeroExtractionDenyMeter",
    "install_zero_extraction_deny_meter",
    "run_injected_direct_store_shard",
    "save_direct_store_artifact",
    "validate_direct_store_shard_artifact",
]
