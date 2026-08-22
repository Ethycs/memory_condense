"""Execution and strict ten-shard merge for cumulative 1M validation."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memory_condense.eval._recall_guarded_cumulative_validation_shard import (
    CURRENT_SOURCE_FORMAT,
    CURRENT_SOURCE_SCOPE,
    CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
    LOCKED_100Q_OFFSETS,
    LOCKED_LONGMEMEVAL_DATASET_SHA256,
    LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
    LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    LOCKED_QUESTIONS_PER_SHARD,
    LOCKED_VALIDATION_POLICY_MANIFEST_SHA256,
    MAX_CONTEXT_TOKENS,
    MAX_PROMPT_TOKENS,
    RESPONDER_OUTPUT_TOKEN_RESERVE,
    SOURCE_ROUTER_MAX_SOURCES,
    SOURCE_ROUTER_RRF_CONSTANT,
    STAGE_IDS,
    VALIDATION_CAMPAIGN_FORMAT,
    VALIDATION_EXTERNAL_RECONSTRUCTION_FORMAT,
    VALIDATION_MERGED_QUESTION_FORMAT,
    VALIDATION_MERGED_RETRIEVAL_FORMAT,
    VALIDATION_POLICY_ATTESTATION_FORMAT,
    VALIDATION_SHARD_QUESTION_FORMAT,
    VALIDATION_SHARD_REFERENCE_FORMAT,
    VALIDATION_SHARD_RETRIEVAL_FORMAT,
    CombinedCumulativeStoreReceipt,
    LockedCumulativePopulationPlan,
    PreparedRecallGuardedCumulativeStore,
    ValidationShardPreflight,
    FrozenValidationPolicy,
    _MERGED_QUESTION_FIELDS,
    _SHARD_QUESTION_FIELDS,
    _assert_gold_blind_schema,
    _atomic_write_json,
    _canonical_json_bytes,
    _closure_policy,
    _episode_policy,
    _question_part,
    _read_canonical_json,
    _require_exact_keys,
    _require_sha256,
    _representative_policy,
    _self_hashed,
    _validate_policy_binding,
    _validate_question_part,
    _validate_sealed_question_payload,
    _validation_execution_policy,
    build_locked_cumulative_population_identity,
    environment_lock_sha256,
    identity_sha256,
    implementation_sha256,
    load_frozen_validation_policy,
    merge_locked_cumulative_shard_identities,
    retrieve_recall_guarded_cumulative_packet,
    shard_output_root,
    validate_current_source_receipt,
    validate_locked_cumulative_population_identity,
    validate_locked_cumulative_shard_identity,
    validate_validation_shard_retrieval,
)

def run_locked_validation_shard_retrieval(
    *,
    prepared: PreparedRecallGuardedCumulativeStore,
    preflight: ValidationShardPreflight,
    selector: Any,
    representative_linker: Any,
    source_store_receipt: Mapping[str, Any],
    source_store_mode: str,
    combined_store_mode: str,
) -> tuple[dict[str, Any], str]:
    """Run missing S0--S3 questions and publish one canonical shard artifact."""

    if implementation_sha256() != preflight.retrieval_implementation_sha256:
        raise RuntimeError("retrieval implementation changed after preflight")
    if environment_lock_sha256() != preflight.environment_lock_sha256:
        raise RuntimeError("retrieval environment changed after preflight")
    selected_source = validate_current_source_receipt(
        source_store_receipt,
        sample=preflight.sample,
        expected_device=preflight.source_embedding_device,
    )
    if (
        prepared.receipt.source_database_sha256
        != selected_source["database_sha256"]
        or prepared.receipt.retrieval_policy_sha256
        != preflight.policy.retrieval_policy_sha256
    ):
        raise RuntimeError("retrieval store changed its source or policy")
    prepared.condenser.set_context_candidate_selector(selector)
    artifact_id = prepared.compilation.artifact.artifact_id
    parts_dir = preflight.shard_root / "retrieval-parts"
    parts: list[dict[str, Any]] = []
    part_hashes: list[str] = []
    for local_ordinal, question in enumerate(preflight.sample.questions):
        global_ordinal = preflight.sample_offset + local_ordinal
        path = parts_dir / f"q{global_ordinal:03d}.json"
        if path.exists():
            part, digest = _read_canonical_json(path)
            _validate_question_part(
                part,
                question=question,
                local_ordinal=local_ordinal,
                preflight=preflight,
                source_store_receipt_sha256=selected_source["receipt_sha256"],
                combined_store_receipt_sha256=prepared.receipt.receipt_sha256,
                compilation_receipt_sha256=(
                    prepared.compilation.receipt_sha256
                ),
            )
            print(
                f"Question {global_ordinal + 1}/100 {question.question_id}: "
                "verified checkpoint hit",
                flush=True,
            )
        else:
            print(
                f"Question {global_ordinal + 1}/100 {question.question_id}: "
                "running S0-S3",
                flush=True,
            )
            started = time.perf_counter()
            result = retrieve_recall_guarded_cumulative_packet(
                prepared.condenser,
                query=question.question,
                prompt_question=question.dated_question,
                retrieval=preflight.policy.config.retrieval,
                artifact_id=artifact_id,
                max_context_tokens=MAX_CONTEXT_TOKENS,
                max_prompt_tokens=MAX_PROMPT_TOKENS,
                responder_output_token_reserve=RESPONDER_OUTPUT_TOKEN_RESERVE,
                episode_policy=_episode_policy(artifact_id),
                representative_linker=representative_linker,
                representative_policy=_representative_policy(artifact_id),
                source_router_max_sources=SOURCE_ROUTER_MAX_SOURCES,
                source_router_rrf_constant=SOURCE_ROUTER_RRF_CONSTANT,
                closure_policy=_closure_policy(),
                require_certified_coverage_runtime=True,
                require_owned_representative_runtime=True,
            )
            part = _question_part(
                result,
                question=question,
                local_ordinal=local_ordinal,
                preflight=preflight,
                source_store_receipt_sha256=selected_source["receipt_sha256"],
                combined_store_receipt_sha256=prepared.receipt.receipt_sha256,
                compilation_receipt_sha256=prepared.compilation.receipt_sha256,
                elapsed_seconds=time.perf_counter() - started,
            )
            _validate_question_part(
                part,
                question=question,
                local_ordinal=local_ordinal,
                preflight=preflight,
                source_store_receipt_sha256=selected_source["receipt_sha256"],
                combined_store_receipt_sha256=prepared.receipt.receipt_sha256,
                compilation_receipt_sha256=(
                    prepared.compilation.receipt_sha256
                ),
            )
            digest = _atomic_write_json(path, part)
            statuses = ",".join(
                str(stage["stage_receipt"]["admission_status"])
                for stage in part["stages"]
            )
            print(
                f"  published; statuses={statuses}; "
                f"elapsed={part['elapsed_seconds']:.1f}s",
                flush=True,
            )
        parts.append(part)
        part_hashes.append(digest)
    retrieval: dict[str, Any] = {
        "format": VALIDATION_SHARD_RETRIEVAL_FORMAT,
        "campaign_format": VALIDATION_CAMPAIGN_FORMAT,
        "population_identity": dict(preflight.population_identity),
        "population_identity_sha256": preflight.population_identity[
            "population_identity_sha256"
        ],
        "shard_identity": dict(preflight.shard_identity),
        "shard_identity_sha256": preflight.shard_identity[
            "shard_identity_sha256"
        ],
        "shard_offset": preflight.sample_offset,
        "validation_policy_attestation": dict(preflight.policy.attestation),
        "validation_policy_attestation_sha256": (
            preflight.policy.attestation_sha256
        ),
        "validation_policy_manifest_sha256": (
            LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
        ),
        "validation_execution_policy": dict(
            preflight.policy.execution_policy
        ),
        "validation_execution_policy_sha256": (
            preflight.policy.execution_policy_sha256
        ),
        "retrieval_policy_sha256": preflight.policy.retrieval_policy_sha256,
        "retrieval_implementation_sha256": (
            preflight.retrieval_implementation_sha256
        ),
        "environment_lock_sha256": preflight.environment_lock_sha256,
        "source_embedding_device": preflight.source_embedding_device,
        "source_timestamp_semantics": CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
        "source_store_mode": source_store_mode,
        "source_store_receipt": selected_source,
        "source_store_receipt_sha256": selected_source["receipt_sha256"],
        "combined_store_mode": combined_store_mode,
        "combined_store_receipt": asdict(prepared.receipt),
        "combined_store_receipt_sha256": prepared.receipt.receipt_sha256,
        "compilation_receipt_sha256": prepared.compilation.receipt_sha256,
        "transcript_tokens": preflight.shard_identity["transcript_tokens"],
        "turn_count": preflight.shard_identity["turn_count"],
        "question_count": len(parts),
        "stage_ids": list(STAGE_IDS),
        "question_part_sha256s": part_hashes,
        "questions": parts,
        "provider_calls": 0,
        "gold_fields_present": False,
    }
    validate_validation_shard_retrieval(retrieval, preflight=preflight)
    if implementation_sha256() != preflight.retrieval_implementation_sha256:
        raise RuntimeError("retrieval implementation changed during shard run")
    if environment_lock_sha256() != preflight.environment_lock_sha256:
        raise RuntimeError("retrieval environment changed during shard run")
    path = preflight.shard_root / "retrieval.json"
    digest = _atomic_write_json(path, retrieval)
    print(f"Validation shard retrieval published: {path} ({digest})", flush=True)
    return retrieval, digest


_SHARD_REFERENCE_FIELDS = frozenset(
    {
        "format",
        "ordinal",
        "shard_offset",
        "shard_identity",
        "shard_identity_sha256",
        "shard_retrieval_sha256",
        "source_store_receipt",
        "source_store_receipt_sha256",
        "combined_store_receipt",
        "combined_store_receipt_sha256",
        "compilation_receipt_sha256",
        "source_question_part_sha256s",
        "question_count",
    }
)
_MERGED_RETRIEVAL_FIELDS = frozenset(
    {
        "format",
        "campaign_format",
        "population_identity",
        "population_identity_sha256",
        "shard_count",
        "question_count",
        "stage_ids",
        "validation_policy_attestation",
        "validation_policy_attestation_sha256",
        "validation_policy_manifest_sha256",
        "validation_execution_policy",
        "validation_execution_policy_sha256",
        "retrieval_policy_sha256",
        "retrieval_implementation_sha256",
        "environment_lock_sha256",
        "source_embedding_device",
        "source_timestamp_semantics",
        "ordered_shard_retrieval_sha256s",
        "shards",
        "external_reconstruction_receipt",
        "external_reconstruction_receipt_sha256",
        "question_part_sha256s",
        "questions",
        "provider_calls",
        "gold_fields_present",
    }
)


def _validate_embedded_policy_surface(retrieval: Mapping[str, Any]) -> None:
    if retrieval.get("validation_policy_manifest_sha256") != (
        LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
    ):
        raise ValueError("merged retrieval changed its frozen policy manifest")
    raw_attestation = retrieval.get("validation_policy_attestation")
    if not isinstance(raw_attestation, Mapping):
        raise ValueError("merged retrieval omitted its policy attestation")
    attestation = dict(raw_attestation)
    declared = _require_sha256(
        attestation.pop("attestation_sha256", None),
        "merged policy attestation SHA-256",
    )
    if (
        declared != identity_sha256(attestation)
        or declared != retrieval.get("validation_policy_attestation_sha256")
        or raw_attestation.get("format")
        != VALIDATION_POLICY_ATTESTATION_FORMAT
        or raw_attestation.get("manifest_sha256")
        != LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
        or raw_attestation.get("manifest_status") != "validation_frozen"
        or raw_attestation.get("manifest_split") != "validation"
        or raw_attestation.get("manifest_claim_profile")
        != "longmemeval-s-1m-100q-95-v1"
    ):
        raise ValueError("merged retrieval policy attestation changed")
    for name in (
        "manifest_declared_implementation_sha256",
        "manifest_declared_environment_lock_sha256",
        "manifest_retrieval_controls_sha256",
        "resolved_retrieval_policy_sha256",
    ):
        _require_sha256(raw_attestation.get(name), f"policy attestation {name}")
    retrieval_policy_sha = _require_sha256(
        retrieval.get("retrieval_policy_sha256"),
        "merged retrieval policy SHA-256",
    )
    if raw_attestation.get("resolved_retrieval_policy_sha256") != (
        retrieval_policy_sha
    ):
        raise ValueError("merged retrieval changed its resolved policy")
    expected_execution = _validation_execution_policy(
        policy_attestation_sha256=declared,
        retrieval_policy_sha256=retrieval_policy_sha,
    )
    if (
        retrieval.get("validation_execution_policy") != expected_execution
        or retrieval.get("validation_execution_policy_sha256")
        != identity_sha256(expected_execution)
    ):
        raise ValueError("merged retrieval execution policy changed")


def _validate_source_receipt_surface(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("merged shard omitted its source-store receipt")
    receipt = dict(value)
    declared = _require_sha256(
        receipt.pop("receipt_sha256", None),
        "source-store receipt SHA-256",
    )
    if (
        declared != identity_sha256(receipt)
        or receipt.get("format") != CURRENT_SOURCE_FORMAT
        or receipt.get("source_scope") != CURRENT_SOURCE_SCOPE
        or receipt.get("timestamp_semantics")
        != CURRENT_SOURCE_TIMESTAMP_SEMANTICS
        or receipt.get("selected_store_entry")
        != f"stores/{receipt.get('base_store_key')}"
        or receipt.get("selected_query_entry")
        != f"query-inputs/{receipt.get('query_input_key')}"
    ):
        raise ValueError("merged shard source-store receipt changed")
    for name in (
        "base_store_key",
        "store_manifest_sha256",
        "store_artifact_sha256",
        "database_sha256",
        "index_sha256",
        "corpus_sha256",
        "deterministic_turn_ids_sha256",
        "turn_sequence_sha256",
        "chunk_sequence_sha256",
        "source_streams_sha256",
        "embedding_identity_sha256",
        "build_runtime_identity_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
        "query_input_key",
        "query_manifest_sha256",
        "query_artifact_sha256",
    ):
        _require_sha256(receipt.get(name), f"source-store receipt {name}")
    return {**receipt, "receipt_sha256": declared}


def validate_merged_validation_retrieval(
    retrieval: Mapping[str, Any],
) -> None:
    """Validate the self-contained merged artifact without benchmark gold.

    The ten source shard files are checked against the hash-locked dataset by
    :func:`merge_locked_validation_retrievals` before publication.  This
    self-contained validator is the safe entry point for the later fixed-stage
    responder: it rechecks every embedded identity, physical-store receipt,
    original part hash, normalized part hash, typed ladder, prompt, and budget.
    """

    _require_exact_keys(
        retrieval,
        _MERGED_RETRIEVAL_FIELDS,
        "merged validation retrieval",
    )
    if (
        retrieval.get("format") != VALIDATION_MERGED_RETRIEVAL_FORMAT
        or retrieval.get("campaign_format") != VALIDATION_CAMPAIGN_FORMAT
        or retrieval.get("shard_count") != len(LOCKED_100Q_OFFSETS)
        or retrieval.get("question_count") != 100
        or tuple(retrieval.get("stage_ids", ())) != STAGE_IDS
        or retrieval.get("source_timestamp_semantics")
        != CURRENT_SOURCE_TIMESTAMP_SEMANTICS
        or retrieval.get("provider_calls") != 0
        or retrieval.get("gold_fields_present") is not False
    ):
        raise ValueError("merged validation retrieval changed campaign controls")
    population = retrieval.get("population_identity")
    if not isinstance(population, Mapping):
        raise ValueError("merged validation retrieval omitted its population")
    normalized_population = validate_locked_cumulative_population_identity(
        population,
        plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    )
    population_sha = str(normalized_population["population_identity_sha256"])
    if retrieval.get("population_identity_sha256") != population_sha:
        raise ValueError("merged validation population digest changed")
    _validate_embedded_policy_surface(retrieval)
    implementation = _require_sha256(
        retrieval.get("retrieval_implementation_sha256"),
        "merged retrieval implementation SHA-256",
    )
    environment = _require_sha256(
        retrieval.get("environment_lock_sha256"),
        "merged retrieval environment-lock SHA-256",
    )
    device = retrieval.get("source_embedding_device")
    if not isinstance(device, str) or not device or device != device.casefold():
        raise ValueError("merged retrieval source device is not normalized")
    references = retrieval.get("shards")
    ordered_shard_hashes = retrieval.get("ordered_shard_retrieval_sha256s")
    if (
        not isinstance(references, list)
        or len(references) != len(LOCKED_100Q_OFFSETS)
        or not isinstance(ordered_shard_hashes, list)
        or len(ordered_shard_hashes) != len(references)
    ):
        raise ValueError("merged validation shard population is incomplete")
    normalized_references: list[dict[str, Any]] = []
    for ordinal, (offset, raw_reference) in enumerate(
        zip(LOCKED_100Q_OFFSETS, references, strict=True)
    ):
        if not isinstance(raw_reference, Mapping):
            raise ValueError("merged validation shard reference must be an object")
        reference = dict(raw_reference)
        _require_exact_keys(
            reference,
            _SHARD_REFERENCE_FIELDS,
            "merged validation shard reference",
        )
        shard = reference.get("shard_identity")
        if not isinstance(shard, Mapping):
            raise ValueError("merged shard reference omitted its identity")
        normalized_shard = validate_locked_cumulative_shard_identity(shard)
        shard_sha = str(normalized_shard["shard_identity_sha256"])
        source_receipt = _validate_source_receipt_surface(
            reference.get("source_store_receipt")
        )
        raw_combined = reference.get("combined_store_receipt")
        if not isinstance(raw_combined, Mapping):
            raise ValueError("merged shard omitted its combined-store receipt")
        combined = CombinedCumulativeStoreReceipt(**dict(raw_combined))
        source_part_hashes = reference.get("source_question_part_sha256s")
        if (
            reference.get("format") != VALIDATION_SHARD_REFERENCE_FORMAT
            or reference.get("ordinal") != ordinal
            or reference.get("shard_offset") != offset
            or normalized_shard["construction"]["sample_offset"] != offset
            or reference.get("shard_identity_sha256") != shard_sha
            or shard_sha
            != normalized_population["ordered_shard_identity_sha256s"][ordinal]
            or reference.get("shard_retrieval_sha256")
            != ordered_shard_hashes[ordinal]
            or reference.get("source_store_receipt_sha256")
            != source_receipt["receipt_sha256"]
            or reference.get("combined_store_receipt_sha256")
            != combined.receipt_sha256
            or reference.get("compilation_receipt_sha256")
            != combined.compilation_receipt_sha256
            or combined.source_database_sha256
            != source_receipt["database_sha256"]
            or combined.turn_count != source_receipt["turn_count"]
            or combined.chunk_count != source_receipt["chunk_count"]
            or combined.retrieval_policy_sha256
            != retrieval["retrieval_policy_sha256"]
            or reference.get("question_count") != LOCKED_QUESTIONS_PER_SHARD
            or not isinstance(source_part_hashes, list)
            or len(source_part_hashes) != LOCKED_QUESTIONS_PER_SHARD
        ):
            raise ValueError("merged validation shard reference changed")
        _require_sha256(
            reference["shard_retrieval_sha256"],
            "source shard retrieval SHA-256",
        )
        for digest in source_part_hashes:
            _require_sha256(digest, "source question-part SHA-256")
        normalized_references.append(reference)
    if len(set(ordered_shard_hashes)) != len(ordered_shard_hashes):
        raise ValueError("merged validation repeats a shard retrieval")
    raw_reconstruction = retrieval.get("external_reconstruction_receipt")
    if not isinstance(raw_reconstruction, Mapping):
        raise ValueError("merged validation omitted external reconstruction receipt")
    reconstruction = dict(raw_reconstruction)
    _require_exact_keys(
        reconstruction,
        frozenset(
            {
                "format",
                "dataset_sha256",
                "split_manifest_sha256",
                "split",
                "population_identity_sha256",
                "ordered_shard_identity_sha256s",
                "ordered_shard_retrieval_sha256s",
                "validation_policy_manifest_sha256",
                "validation_policy_attestation_sha256",
                "retrieval_policy_sha256",
                "retrieval_implementation_sha256",
                "environment_lock_sha256",
                "source_embedding_device",
                "verification",
                "provider_calls",
                "gold_fields_present",
                "receipt_sha256",
            }
        ),
        "external reconstruction receipt",
    )
    declared_reconstruction = _require_sha256(
        reconstruction.pop("receipt_sha256", None),
        "external reconstruction receipt SHA-256",
    )
    expected_reconstruction = {
        "format": VALIDATION_EXTERNAL_RECONSTRUCTION_FORMAT,
        "dataset_sha256": LOCKED_LONGMEMEVAL_DATASET_SHA256,
        "split_manifest_sha256": LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
        "split": "validation",
        "population_identity_sha256": population_sha,
        "ordered_shard_identity_sha256s": normalized_population[
            "ordered_shard_identity_sha256s"
        ],
        "ordered_shard_retrieval_sha256s": ordered_shard_hashes,
        "validation_policy_manifest_sha256": (
            LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
        ),
        "validation_policy_attestation_sha256": retrieval[
            "validation_policy_attestation_sha256"
        ],
        "retrieval_policy_sha256": retrieval["retrieval_policy_sha256"],
        "retrieval_implementation_sha256": implementation,
        "environment_lock_sha256": environment,
        "source_embedding_device": device,
        "verification": (
            "fresh hash-locked dataset/split reconstruction plus exact ordered "
            "canonical shard artifact validation"
        ),
        "provider_calls": 0,
        "gold_fields_present": False,
    }
    if (
        reconstruction != expected_reconstruction
        or declared_reconstruction != identity_sha256(reconstruction)
        or retrieval.get("external_reconstruction_receipt_sha256")
        != declared_reconstruction
    ):
        raise ValueError("external reconstruction receipt changed")
    questions = retrieval.get("questions")
    part_hashes = retrieval.get("question_part_sha256s")
    if (
        not isinstance(questions, list)
        or len(questions) != 100
        or not isinstance(part_hashes, list)
        or len(part_hashes) != len(questions)
    ):
        raise ValueError("merged validation question population is incomplete")
    observed_hashes = [
        hashlib.sha256(_canonical_json_bytes(question)).hexdigest()
        for question in questions
    ]
    if part_hashes != observed_hashes:
        raise ValueError("merged validation normalized part hashes changed")
    seen_question_ids: set[str] = set()
    for ordinal, raw_question in enumerate(questions):
        if not isinstance(raw_question, Mapping):
            raise ValueError("merged validation question must be an object")
        question = dict(raw_question)
        _require_exact_keys(
            question,
            _MERGED_QUESTION_FIELDS,
            "merged validation question",
        )
        shard_ordinal, local_ordinal = divmod(
            ordinal, LOCKED_QUESTIONS_PER_SHARD
        )
        reference = normalized_references[shard_ordinal]
        shard = reference["shard_identity"]
        probe = shard["ordered_question_probes"][local_ordinal]
        source_part_sha = reference["source_question_part_sha256s"][
            local_ordinal
        ]
        expected = {
            "format": VALIDATION_MERGED_QUESTION_FORMAT,
            "population_identity_sha256": population_sha,
            "shard_identity_sha256": reference["shard_identity_sha256"],
            "shard_offset": reference["shard_offset"],
            "local_ordinal": local_ordinal,
            "ordinal": ordinal,
            "question_id_sha256": identity_sha256(
                {"question_id": question.get("question_id")}
            ),
            "question_sha256": probe["retrieval_query_sha256"],
            "dated_question_sha256": probe["prompt_question_sha256"],
            "probe_identity_sha256": probe["probe_identity_sha256"],
            "validation_policy_manifest_sha256": (
                LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
            ),
            "validation_policy_attestation_sha256": retrieval[
                "validation_policy_attestation_sha256"
            ],
            "validation_execution_policy_sha256": retrieval[
                "validation_execution_policy_sha256"
            ],
            "retrieval_policy_sha256": retrieval["retrieval_policy_sha256"],
            "retrieval_implementation_sha256": implementation,
            "environment_lock_sha256": environment,
            "source_store_receipt_sha256": reference[
                "source_store_receipt_sha256"
            ],
            "combined_store_receipt_sha256": reference[
                "combined_store_receipt_sha256"
            ],
            "compilation_receipt_sha256": reference[
                "compilation_receipt_sha256"
            ],
            "source_shard_retrieval_sha256": reference[
                "shard_retrieval_sha256"
            ],
            "source_question_part_sha256": source_part_sha,
            "stage_ids": list(STAGE_IDS),
            "provider_calls": 0,
        }
        if any(question.get(name) != value for name, value in expected.items()):
            raise ValueError("merged validation question cross-binding changed")
        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("merged validation question ID is missing")
        if question_id in seen_question_ids:
            raise ValueError("merged validation question IDs must be unique")
        seen_question_ids.add(question_id)
        source_projection = dict(question)
        source_projection["format"] = VALIDATION_SHARD_QUESTION_FORMAT
        source_projection.pop("source_shard_retrieval_sha256")
        source_projection.pop("source_question_part_sha256")
        if hashlib.sha256(_canonical_json_bytes(source_projection)).hexdigest() != (
            source_part_sha
        ):
            raise ValueError("merged question changed its original shard part")
        _validate_sealed_question_payload(question)
    _assert_gold_blind_schema(retrieval, label="merged validation retrieval")


def merged_question_store_receipts(
    retrieval: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return each normalized question's rigorously resolved local store.

    This helper intentionally validates the complete artifact first.  It lets
    the fixed-stage responder retain a one-pass question loop while avoiding a
    fictitious aggregate store receipt.
    """

    validate_merged_validation_retrieval(retrieval)
    references = retrieval["shards"]
    return tuple(
        dict(references[ordinal // LOCKED_QUESTIONS_PER_SHARD][
            "combined_store_receipt"
        ])
        for ordinal in range(int(retrieval["question_count"]))
    )


def _shard_reference(
    retrieval: Mapping[str, Any],
    *,
    ordinal: int,
    retrieval_sha256: str,
) -> dict[str, Any]:
    return {
        "format": VALIDATION_SHARD_REFERENCE_FORMAT,
        "ordinal": ordinal,
        "shard_offset": retrieval["shard_offset"],
        "shard_identity": dict(retrieval["shard_identity"]),
        "shard_identity_sha256": retrieval["shard_identity_sha256"],
        "shard_retrieval_sha256": retrieval_sha256,
        "source_store_receipt": dict(retrieval["source_store_receipt"]),
        "source_store_receipt_sha256": retrieval[
            "source_store_receipt_sha256"
        ],
        "combined_store_receipt": dict(retrieval["combined_store_receipt"]),
        "combined_store_receipt_sha256": retrieval[
            "combined_store_receipt_sha256"
        ],
        "compilation_receipt_sha256": retrieval[
            "compilation_receipt_sha256"
        ],
        "source_question_part_sha256s": list(
            retrieval["question_part_sha256s"]
        ),
        "question_count": retrieval["question_count"],
    }


@dataclass(frozen=True, slots=True)
class ReconstructedValidationShardSet:
    """Ten canonical shards verified against freshly reconstructed inputs."""

    population_identity: Mapping[str, Any]
    policy: FrozenValidationPolicy
    shard_retrievals: tuple[Mapping[str, Any], ...]
    shard_retrieval_sha256s: tuple[str, ...]
    contexts: tuple[ValidationShardPreflight, ...]
    retrieval_implementation_sha256: str
    environment_lock_sha256: str
    source_embedding_device: str


def reconstruct_and_validate_locked_validation_retrievals(
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    policy_path: str | Path,
    output_root: str | Path,
    shard_retrieval_paths: Sequence[str | Path] | None = None,
    device: str = "cuda",
    plan: LockedCumulativePopulationPlan = LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
) -> ReconstructedValidationShardSet:
    """Apply the filesystem/source-backed validation required before merge."""

    if plan != LOCKED_LONGMEMEVAL_VALIDATION_PLAN:
        raise ValueError("validation merge requires the exact locked 100Q plan")
    samples, shard_identities, population = (
        build_locked_cumulative_population_identity(
            dataset_path,
            split_manifest_path,
            plan=plan,
        )
    )
    reconstructed = merge_locked_cumulative_shard_identities(
        shard_identities,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        plan=plan,
    )
    if reconstructed != population:
        raise RuntimeError("locked population merge disagrees with reconstruction")
    policy = load_frozen_validation_policy(policy_path, device=device)
    _validate_policy_binding(policy)
    root = Path(output_root).resolve()
    paths = (
        tuple(
            shard_output_root(root, offset) / "retrieval.json"
            for offset in plan.shard_offsets
        )
        if shard_retrieval_paths is None
        else tuple(Path(path).resolve() for path in shard_retrieval_paths)
    )
    if len(paths) != len(plan.shard_offsets):
        raise ValueError("merge requires exactly ten ordered shard retrieval paths")
    active_implementation = implementation_sha256()
    active_environment = environment_lock_sha256()
    retrievals: list[Mapping[str, Any]] = []
    digests: list[str] = []
    contexts: list[ValidationShardPreflight] = []
    for index, (offset, path, sample, shard_identity) in enumerate(
        zip(plan.shard_offsets, paths, samples, shard_identities, strict=True)
    ):
        retrieval, digest = _read_canonical_json(path)
        context = ValidationShardPreflight(
            sample=sample,
            shard_identity=shard_identity,
            population_identity=population,
            policy=policy,
            sample_offset=offset,
            shard_root=path.parent,
            qwen_prefix_model_dir=Path(),
            qwen_choice_model_dir=Path(),
            retrieval_implementation_sha256=active_implementation,
            environment_lock_sha256=active_environment,
            source_embedding_device=str(device).casefold(),
        )
        validate_validation_shard_retrieval(retrieval, preflight=context)
        if retrieval["shard_offset"] != offset:
            raise ValueError(f"shard retrieval {index} is out of order")
        retrievals.append(retrieval)
        digests.append(digest)
        contexts.append(context)
    if len(set(digests)) != len(digests):
        raise ValueError("merge received a repeated shard retrieval artifact")
    return ReconstructedValidationShardSet(
        population_identity=population,
        policy=policy,
        shard_retrievals=tuple(retrievals),
        shard_retrieval_sha256s=tuple(digests),
        contexts=tuple(contexts),
        retrieval_implementation_sha256=active_implementation,
        environment_lock_sha256=active_environment,
        source_embedding_device=str(device).casefold(),
    )


def merge_locked_validation_retrievals(
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    policy_path: str | Path,
    output_root: str | Path,
    output_path: str | Path | None = None,
    shard_retrieval_paths: Sequence[str | Path] | None = None,
    device: str = "cuda",
    plan: LockedCumulativePopulationPlan = LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
) -> tuple[dict[str, Any], str]:
    """Strictly merge the ten exact ordered shard retrieval artifacts."""

    verified = reconstruct_and_validate_locked_validation_retrievals(
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        policy_path=policy_path,
        output_root=output_root,
        shard_retrieval_paths=shard_retrieval_paths,
        device=device,
        plan=plan,
    )
    population = verified.population_identity
    policy = verified.policy
    shard_retrievals = verified.shard_retrievals
    shard_hashes = verified.shard_retrieval_sha256s
    contexts = verified.contexts
    active_implementation = verified.retrieval_implementation_sha256
    active_environment = verified.environment_lock_sha256
    root = Path(output_root).resolve()
    references = [
        _shard_reference(retrieval, ordinal=index, retrieval_sha256=digest)
        for index, (retrieval, digest) in enumerate(
            zip(shard_retrievals, shard_hashes, strict=True)
        )
    ]
    questions: list[dict[str, Any]] = []
    part_hashes: list[str] = []
    for retrieval, retrieval_sha, context in zip(
        shard_retrievals, shard_hashes, contexts, strict=True
    ):
        for local_ordinal, (raw_question, source_part_sha) in enumerate(
            zip(
                retrieval["questions"],
                retrieval["question_part_sha256s"],
                strict=True,
            )
        ):
            normalized = dict(raw_question)
            normalized["format"] = VALIDATION_MERGED_QUESTION_FORMAT
            normalized["source_shard_retrieval_sha256"] = retrieval_sha
            normalized["source_question_part_sha256"] = source_part_sha
            _validate_question_part(
                normalized,
                question=context.sample.questions[local_ordinal],
                local_ordinal=local_ordinal,
                preflight=context,
                source_store_receipt_sha256=retrieval[
                    "source_store_receipt_sha256"
                ],
                combined_store_receipt_sha256=retrieval[
                    "combined_store_receipt_sha256"
                ],
                compilation_receipt_sha256=retrieval[
                    "compilation_receipt_sha256"
                ],
                merged=True,
                source_shard_retrieval_sha256=retrieval_sha,
                source_question_part_sha256=source_part_sha,
            )
            questions.append(normalized)
            part_hashes.append(
                hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
            )
    if len(questions) != 100:
        raise RuntimeError("locked merge did not produce exactly 100 questions")
    reconstruction_receipt = _self_hashed(
        {
            "format": VALIDATION_EXTERNAL_RECONSTRUCTION_FORMAT,
            "dataset_sha256": LOCKED_LONGMEMEVAL_DATASET_SHA256,
            "split_manifest_sha256": (
                LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256
            ),
            "split": "validation",
            "population_identity_sha256": population[
                "population_identity_sha256"
            ],
            "ordered_shard_identity_sha256s": population[
                "ordered_shard_identity_sha256s"
            ],
            "ordered_shard_retrieval_sha256s": list(shard_hashes),
            "validation_policy_manifest_sha256": (
                LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
            ),
            "validation_policy_attestation_sha256": (
                policy.attestation_sha256
            ),
            "retrieval_policy_sha256": policy.retrieval_policy_sha256,
            "retrieval_implementation_sha256": active_implementation,
            "environment_lock_sha256": active_environment,
            "source_embedding_device": str(device).casefold(),
            "verification": (
                "fresh hash-locked dataset/split reconstruction plus exact "
                "ordered canonical shard artifact validation"
            ),
            "provider_calls": 0,
            "gold_fields_present": False,
        },
        "receipt_sha256",
    )
    merged: dict[str, Any] = {
        "format": VALIDATION_MERGED_RETRIEVAL_FORMAT,
        "campaign_format": VALIDATION_CAMPAIGN_FORMAT,
        "population_identity": dict(population),
        "population_identity_sha256": population[
            "population_identity_sha256"
        ],
        "shard_count": len(references),
        "question_count": len(questions),
        "stage_ids": list(STAGE_IDS),
        "validation_policy_attestation": dict(policy.attestation),
        "validation_policy_attestation_sha256": policy.attestation_sha256,
        "validation_policy_manifest_sha256": (
            LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
        ),
        "validation_execution_policy": dict(policy.execution_policy),
        "validation_execution_policy_sha256": (
            policy.execution_policy_sha256
        ),
        "retrieval_policy_sha256": policy.retrieval_policy_sha256,
        "retrieval_implementation_sha256": active_implementation,
        "environment_lock_sha256": active_environment,
        "source_embedding_device": str(device).casefold(),
        "source_timestamp_semantics": CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
        "ordered_shard_retrieval_sha256s": list(shard_hashes),
        "shards": references,
        "external_reconstruction_receipt": reconstruction_receipt,
        "external_reconstruction_receipt_sha256": reconstruction_receipt[
            "receipt_sha256"
        ],
        "question_part_sha256s": part_hashes,
        "questions": questions,
        "provider_calls": 0,
        "gold_fields_present": False,
    }
    validate_merged_validation_retrieval(merged)
    if implementation_sha256() != active_implementation:
        raise RuntimeError("retrieval implementation changed during merge")
    if environment_lock_sha256() != active_environment:
        raise RuntimeError("retrieval environment changed during merge")
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else root / "retrieval.json"
    )
    digest = _atomic_write_json(destination, merged)
    print(f"Merged validation retrieval published: {destination} ({digest})", flush=True)
    return merged, digest
