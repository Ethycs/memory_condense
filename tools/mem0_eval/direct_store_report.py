"""Frozen 100Q population and merge schema for the direct-store scaffold."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from memory_condense.eval.sample_identity import canonical_sha256

from .direct_store import (
    DIRECT_STORE_ARM_ID,
    DIRECT_STORE_ARM_LABEL,
    DIRECT_STORE_SCHEMA_VERSION,
    DirectStoreError,
    FROZEN_SAMPLE_OFFSETS,
    QUESTIONS_PER_SHARD,
    save_direct_store_artifact,
    validate_direct_store_shard_artifact,
)
from .protocol import (
    RawStressShard,
    build_raw_stress_shards,
    shard_receipt,
    validate_raw_stress_shard,
)
from .preflight import tool_implementation_sha256


DIRECT_STORE_POPULATION_FORMAT = (
    "memory-condense-mem0-direct-store-population-preflight-v2"
)
DIRECT_STORE_CAMPAIGN_FORMAT = (
    "memory-condense-mem0-direct-store-retrieval-campaign-v2"
)
FROZEN_QUESTION_COUNT = 100
FROZEN_RAW_PAIRS = 24_928
FROZEN_SKIPPED_EMPTY_PAIRS = 5
FROZEN_ADD_OPERATIONS = 24_923
FROZEN_SEARCH_OPERATIONS = 100
FROZEN_DATASET_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
FROZEN_SPLIT_MANIFEST_SHA256 = (
    "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
)
FROZEN_ORDERED_QUESTION_IDS_SHA256 = (
    "7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1"
)
FROZEN_SORTED_UNIQUE_QUESTION_IDS_SHA256 = (
    "dd8addf6bba1bd83d7ce4c9427e2e8a86cf0eacbf04d13b3cf13cc8d287dd99c"
)


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


def _strict_json_copy(value: Any) -> Any:
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
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DirectStoreError("report value is not strict canonical JSON") from exc


def _file_sha256(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise DirectStoreError(f"cannot read {label}: {exc}") from exc
    return digest.hexdigest()


def _runtime_environment_receipt() -> dict[str, Any]:
    lock_path = Path(__file__).resolve().parent / "pixi.lock"
    present = lock_path.is_file()
    return {
        "status": "unresolved",
        "isolated_environment_lock": "tools/mem0_eval/pixi.lock",
        "isolated_environment_lock_present": present,
        "isolated_environment_lock_sha256": (
            _file_sha256(lock_path, "isolated Mem0 environment lock")
            if present
            else None
        ),
        "actual_mem0_runtime_environment_frozen": False,
        "actual_mem0_runtime_verified": False,
        "actual_mem0_executed": False,
        "retrieval_execution_authorized": False,
        "resource_matching_completed": False,
    }


def _population_identity_sha256(receipt: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "source_coordinates": receipt["source_coordinates"],
            "sample_offsets": receipt["sample_offsets"],
            "ordered_question_ids": receipt["ordered_question_ids"],
            "ordered_question_ids_sha256": receipt[
                "ordered_question_ids_sha256"
            ],
            "sorted_unique_question_ids_sha256": receipt[
                "sorted_unique_question_ids_sha256"
            ],
            "totals": receipt["totals"],
            "shards": receipt["shards"],
        }
    )


def validate_frozen_direct_store_population(
    shards: Sequence[RawStressShard],
) -> dict[str, Any]:
    """Validate the exact ten-shard, 24,923-add source population."""

    return _build_population_receipt(
        shards,
        source_files_verified=False,
        source_files_rechecked=False,
        tool_sha256=tool_implementation_sha256(),
        tool_rechecked=False,
    )


def _validated_population_rows(
    shards: Sequence[RawStressShard],
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    if len(shards) != len(FROZEN_SAMPLE_OFFSETS):
        raise DirectStoreError("direct-store campaign requires exactly ten shards")
    offsets = tuple(shard.sample_offset for shard in shards)
    if offsets != FROZEN_SAMPLE_OFFSETS:
        raise DirectStoreError(
            "direct-store shards must be supplied in frozen offset order 0..90"
        )
    receipts: list[dict[str, Any]] = []
    question_ids: list[str] = []
    raw_pairs = 0
    skipped = 0
    adds = 0
    for shard in shards:
        validate_raw_stress_shard(shard)
        if len(shard.question_ids) != QUESTIONS_PER_SHARD:
            raise DirectStoreError(
                f"offset {shard.sample_offset} does not contain ten questions"
            )
        receipts.append(shard_receipt(shard))
        question_ids.extend(shard.question_ids)
        raw_pairs += shard.add_counts.raw_pairs
        skipped += shard.add_counts.skipped_empty_pairs
        adds += shard.add_counts.add_requests
    if len(question_ids) != FROZEN_QUESTION_COUNT:
        raise DirectStoreError("direct-store population is not exactly 100Q")
    if len(question_ids) != len(set(question_ids)):
        raise DirectStoreError("direct-store population repeats question IDs")
    ordered_question_ids_sha256 = canonical_sha256(question_ids)
    sorted_unique_question_ids_sha256 = canonical_sha256(
        sorted(set(question_ids))
    )
    if ordered_question_ids_sha256 != FROZEN_ORDERED_QUESTION_IDS_SHA256:
        raise DirectStoreError(
            "direct-store ordered-question population SHA-256 mismatch"
        )
    if (
        sorted_unique_question_ids_sha256
        != FROZEN_SORTED_UNIQUE_QUESTION_IDS_SHA256
    ):
        raise DirectStoreError(
            "direct-store sorted-unique-question population SHA-256 mismatch"
        )
    observed: dict[str, int] = {
        "raw_pairs": raw_pairs,
        "skipped_empty_pairs": skipped,
        "add_operations": adds,
        "search_operations": len(question_ids),
    }
    required = {
        "raw_pairs": FROZEN_RAW_PAIRS,
        "skipped_empty_pairs": FROZEN_SKIPPED_EMPTY_PAIRS,
        "add_operations": FROZEN_ADD_OPERATIONS,
        "search_operations": FROZEN_SEARCH_OPERATIONS,
    }
    if observed != required:
        raise DirectStoreError(
            "direct-store frozen population totals mismatch: "
            f"expected={required!r}, observed={observed!r}"
        )
    return receipts, question_ids, required


def _build_population_receipt(
    shards: Sequence[RawStressShard],
    *,
    source_files_verified: bool,
    source_files_rechecked: bool,
    tool_sha256: str,
    tool_rechecked: bool,
) -> dict[str, Any]:
    if not isinstance(tool_sha256, str) or len(tool_sha256) != 64:
        raise DirectStoreError("direct-store tool identity is not SHA-256")
    receipts, question_ids, required = _validated_population_rows(shards)
    ordered_question_ids_sha256 = canonical_sha256(question_ids)
    sorted_unique_question_ids_sha256 = canonical_sha256(
        sorted(set(question_ids))
    )
    receipt: dict[str, Any] = {
        "format": DIRECT_STORE_POPULATION_FORMAT,
        "schema_version": DIRECT_STORE_SCHEMA_VERSION,
        "status": (
            "ready_provider_free_population_and_tool_receipt"
            if source_files_verified and source_files_rechecked and tool_rechecked
            else "validated_injected_population_only"
        ),
        "arm": _arm_receipt(),
        "source_coordinates": {
            "dataset_sha256": FROZEN_DATASET_SHA256,
            "split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
        },
        "source_file_verification": {
            "verified_before_population": source_files_verified,
            "rechecked_after_population": source_files_rechecked,
        },
        "tool_identity": {
            "kind": "tools-mem0-eval-python-tree-v1",
            "root": "tools/mem0_eval",
            "scope": "recursive-*.py",
            "hash_protocol": (
                "length-prefixed-relative-path-and-bytes-sha256-v1"
            ),
            "tool_implementation_sha256": tool_sha256,
            "rechecked_after_population": tool_rechecked,
        },
        "actual_mem0_runtime_environment": _runtime_environment_receipt(),
        "sample_offsets": list(FROZEN_SAMPLE_OFFSETS),
        "ordered_question_ids": question_ids,
        "ordered_question_ids_sha256": ordered_question_ids_sha256,
        "sorted_unique_question_ids_sha256": (
            sorted_unique_question_ids_sha256
        ),
        "totals": required,
        "shards": receipts,
        "provider_calls_authorized": 0,
        "network_calls_authorized": 0,
        "actual_mem0_executed": False,
    }
    receipt["population_sha256"] = _population_identity_sha256(receipt)
    receipt["preflight_sha256"] = canonical_sha256(receipt)
    return receipt


def build_frozen_direct_store_population_preflight(
    *,
    benchmark_file: str | Path,
    split_manifest: str | Path,
) -> tuple[tuple[RawStressShard, ...], dict[str, Any]]:
    """Reconstruct the locked population once without importing Mem0."""

    dataset_path = Path(benchmark_file).resolve()
    split_path = Path(split_manifest).resolve()
    dataset_before = _file_sha256(dataset_path, "LongMemEval dataset")
    split_before = _file_sha256(split_path, "locked split manifest")
    if dataset_before != FROZEN_DATASET_SHA256:
        raise DirectStoreError("frozen LongMemEval dataset SHA-256 mismatch")
    if split_before != FROZEN_SPLIT_MANIFEST_SHA256:
        raise DirectStoreError("frozen split-manifest SHA-256 mismatch")
    tool_before = tool_implementation_sha256()
    runtime_before = _runtime_environment_receipt()
    shards = build_raw_stress_shards(
        benchmark_file=dataset_path,
        split_manifest=split_path,
        sample_offsets=FROZEN_SAMPLE_OFFSETS,
        target_tokens=1_000_000,
        max_questions=QUESTIONS_PER_SHARD,
    )
    dataset_after = _file_sha256(dataset_path, "LongMemEval dataset")
    split_after = _file_sha256(split_path, "locked split manifest")
    tool_after = tool_implementation_sha256()
    runtime_after = _runtime_environment_receipt()
    if dataset_after != dataset_before or split_after != split_before:
        raise DirectStoreError("frozen source files changed during preflight")
    if tool_after != tool_before:
        raise DirectStoreError("direct-store tool implementation changed during preflight")
    if runtime_after != runtime_before:
        raise DirectStoreError(
            "actual Mem0 runtime-environment status changed during preflight"
        )
    return shards, _build_population_receipt(
        shards,
        source_files_verified=True,
        source_files_rechecked=True,
        tool_sha256=tool_before,
        tool_rechecked=True,
    )


def _load_artifact(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        copied = _strict_json_copy(value)
        assert isinstance(copied, dict)
        return copied
    path = Path(value)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DirectStoreError(f"cannot load direct-store artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DirectStoreError(f"direct-store artifact {path} is not an object")
    return payload


def _campaign_from_validated(
    artifacts: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    shard_rows: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    add_seconds = 0.0
    search_seconds = 0.0
    returned_memories = 0
    for artifact in artifacts:
        sample = artifact["sample"]
        ingestion = artifact["ingestion"]
        retrieval = artifact["retrieval"]
        shard_rows.append(
            {
                "sample_offset": sample["sample_offset"],
                "sample_id": sample["sample_id"],
                "sample_sha256": sample["sample_sha256"],
                "artifact_sha256": artifact["artifact_sha256"],
                "add_ledger_sha256": ingestion["add_ledger_sha256"],
                "searches_sha256": retrieval["searches_sha256"],
                "runtime_sha256": canonical_sha256(artifact["runtime"]),
                "cleanup_sha256": canonical_sha256(artifact["cleanup"]),
            }
        )
        add_seconds += float(ingestion["add_elapsed_seconds"])
        search_seconds += float(retrieval["search_elapsed_seconds"])
        returned_memories += int(ingestion["returned_memory_ids"])
        searches.extend(
            {
                "sample_offset": sample["sample_offset"],
                **_strict_json_copy(search),
            }
            for search in retrieval["searches"]
        )

    question_ids = [row["question_id"] for row in searches]
    campaign: dict[str, Any] = {
        "format": DIRECT_STORE_CAMPAIGN_FORMAT,
        "schema_version": DIRECT_STORE_SCHEMA_VERSION,
        "status": "complete_injected_test_only",
        "arm": _arm_receipt(),
        "population_sha256": population["population_sha256"],
        "sample_offsets": list(FROZEN_SAMPLE_OFFSETS),
        "ordered_question_ids": question_ids,
        "ordered_question_ids_sha256": canonical_sha256(question_ids),
        "sorted_unique_question_ids_sha256": canonical_sha256(
            sorted(set(question_ids))
        ),
        "totals": {
            "raw_pairs": FROZEN_RAW_PAIRS,
            "skipped_empty_pairs": FROZEN_SKIPPED_EMPTY_PAIRS,
            "add_operations": FROZEN_ADD_OPERATIONS,
            "returned_memory_ids": returned_memories,
            "search_operations": FROZEN_SEARCH_OPERATIONS,
            "extraction_calls": 0,
            "provider_calls": 0,
            "network_calls_observed": 0,
            "add_elapsed_seconds": add_seconds,
            "search_elapsed_seconds": search_seconds,
        },
        "shards": shard_rows,
        "shards_sha256": canonical_sha256(shard_rows),
        "retrieval_rows": searches,
        "retrieval_rows_sha256": canonical_sha256(searches),
        "limitations": {
            "actual_mem0_executed": False,
            "official_mem0_comparison": False,
            "resource_matching_completed": False,
            "answer_generation_or_judging_performed": False,
            "benchmark_result_eligible": False,
        },
    }
    campaign["campaign_sha256"] = canonical_sha256(campaign)
    return campaign


def merge_direct_store_retrieval_shards(
    values: Sequence[Mapping[str, Any] | str | Path],
    *,
    expected_shards: Sequence[RawStressShard],
) -> dict[str, Any]:
    """Validate and merge exactly ten ordered, test-only retrieval shards."""

    population = validate_frozen_direct_store_population(expected_shards)
    if len(values) != len(expected_shards):
        raise DirectStoreError("merge requires exactly one artifact per shard")
    validated = [
        validate_direct_store_shard_artifact(_load_artifact(value), shard)
        for value, shard in zip(values, expected_shards)
    ]
    offsets = tuple(row["sample"]["sample_offset"] for row in validated)
    if offsets != FROZEN_SAMPLE_OFFSETS:
        raise DirectStoreError("artifact order does not match frozen shard order")
    runtime_hashes = {canonical_sha256(row["runtime"]) for row in validated}
    if len(runtime_hashes) != 1:
        raise DirectStoreError("injected runtime identity differs across shards")
    campaign = _campaign_from_validated(validated, population)
    if (
        campaign["ordered_question_ids"]
        != population["ordered_question_ids"]
    ):
        raise DirectStoreError("merged retrieval order differs from locked 100Q order")
    if (
        campaign["ordered_question_ids_sha256"]
        != FROZEN_ORDERED_QUESTION_IDS_SHA256
    ):
        raise DirectStoreError("merged ordered-question SHA-256 mismatch")
    if (
        campaign["sorted_unique_question_ids_sha256"]
        != FROZEN_SORTED_UNIQUE_QUESTION_IDS_SHA256
    ):
        raise DirectStoreError("merged sorted-unique-question SHA-256 mismatch")
    if len(campaign["retrieval_rows"]) != FROZEN_SEARCH_OPERATIONS:
        raise DirectStoreError("merged campaign does not contain 100 searches")
    return campaign


def validate_direct_store_campaign_report(
    value: Mapping[str, Any],
    *,
    shard_artifacts: Sequence[Mapping[str, Any] | str | Path],
    expected_shards: Sequence[RawStressShard],
) -> dict[str, Any]:
    """Rebuild a campaign from its source shards and require bytewise JSON parity."""

    observed = _strict_json_copy(value)
    if not isinstance(observed, dict):
        raise DirectStoreError("direct-store campaign report must be an object")
    body = dict(observed)
    digest = body.pop("campaign_sha256", None)
    if digest != canonical_sha256(body):
        raise DirectStoreError("direct-store campaign SHA-256 mismatch")
    expected = merge_direct_store_retrieval_shards(
        shard_artifacts,
        expected_shards=expected_shards,
    )
    if observed != expected:
        raise DirectStoreError("direct-store campaign report does not rebuild")
    return observed


def save_direct_store_campaign_report(
    value: Mapping[str, Any], path: str | Path
) -> None:
    save_direct_store_artifact(value, path)


__all__ = [
    "DIRECT_STORE_CAMPAIGN_FORMAT",
    "DIRECT_STORE_POPULATION_FORMAT",
    "FROZEN_ADD_OPERATIONS",
    "FROZEN_DATASET_SHA256",
    "FROZEN_ORDERED_QUESTION_IDS_SHA256",
    "FROZEN_QUESTION_COUNT",
    "FROZEN_RAW_PAIRS",
    "FROZEN_SEARCH_OPERATIONS",
    "FROZEN_SPLIT_MANIFEST_SHA256",
    "FROZEN_SORTED_UNIQUE_QUESTION_IDS_SHA256",
    "FROZEN_SKIPPED_EMPTY_PAIRS",
    "build_frozen_direct_store_population_preflight",
    "merge_direct_store_retrieval_shards",
    "save_direct_store_campaign_report",
    "validate_direct_store_campaign_report",
    "validate_frozen_direct_store_population",
]
