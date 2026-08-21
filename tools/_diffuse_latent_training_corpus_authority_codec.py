"""Explicit canonical codecs for non-authoritative corpus candidates."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.eval._diffuse_latent_training_corpus_models import (
    LatentTrainingFileIdentity,
)

from tools._diffuse_latent_training_corpus_authority_models import (
    CANDIDATE_PUBLICATION_FORMAT,
    DECLARED_EXECUTION_FORMAT,
    PHASE_CANDIDATE_FORMAT,
    PRODUCTION_CANDIDATE_FORMAT,
    PRODUCTION_EXTERNAL_LOCK_FORMAT,
    DeclaredProductionExecutionCoordinates,
    ProductionCandidatePublicationReceipt,
    ProductionCorpusCandidateReceipt,
    ProductionExternalLock,
    ProductionLatentTrainingCorpusError,
    ProductionPhaseCandidateReceipt,
)


MAX_CANDIDATE_RECEIPT_BYTES = 4 * 1024 * 1024

_LOCK_KEYS = {
    "dataset_sha256", "split_manifest_sha256", "treatment_file_sha256",
    "sanitized_projection_sha256", "analysis_ordered_question_ids_sha256",
    "fit_count", "fit_ordered_question_ids_sha256", "validation_count",
    "validation_ordered_question_ids_sha256", "excluded_confirmation_count",
    "excluded_confirmation_ordered_question_ids_sha256",
    "compilation_policy_sha256", "episode_policy_sha256",
    "closure_policy_sha256", "base_arm_sha256",
    "episode_primary_arm_sha256", "matched_controls_sha256",
    "representative_policy_controls_sha256", "retrieval_config_sha256",
    "evaluation_policy_sha256", "fusion_caps_sha256",
    "qwen_atom_feature_caps_sha256", "bge_model_id", "bge_model_revision",
    "bge_checkpoint_sha256", "qwen_model_id", "qwen_model_revision",
    "qwen_checkpoint_sha256", "retrieval_prefix_layers",
    "retrieval_attention_layer", "feature_prefix_layers", "feature_output_layer",
    "runtime_mode", "device", "episodic_route", "closure_routing_scope",
    "scorer_labels_present", "evaluator_label_schema_present", "format",
    "lock_sha256",
}
_EXECUTION_KEYS = {
    "launcher_relative_path", "launcher_sha256", "source_commit",
    "package_implementation_sha256", "corpus_implementation_sha256",
    "route_implementation_sha256",
    "runtime_binding_sha256", "ordered_legacy_input_provider_identities_sha256",
    "representative_linker_identity_sha256",
    "representative_policy_factory_identity_sha256", "bge_checkpoint_sha256",
    "qwen_retrieval_checkpoint_sha256", "qwen_feature_checkpoint_sha256",
    "qwen_retrieval_contract_sha256", "qwen_feature_contract_sha256",
    "tracked_worktree_clean_attested", "local_checkpoint_bytes_verified_attested",
    "runtime_binding_rederived_attested", "retrieval_qwen_execution_attested",
    "feature_qwen_execution_attested", "production_authorized", "format",
    "coordinates_sha256",
}
_CANDIDATE_KEYS = {
    "generic_root_manifest_sha256", "generic_root_manifest_bytes",
    "generic_corpus_sha256", "generic_inventory_sha256",
    "generic_population_projection_sha256", "generic_implementation_sha256",
    "generic_fit_partition_sha256", "generic_fit_manifest_file_sha256",
    "generic_fit_manifest_file_bytes", "generic_validation_partition_sha256",
    "generic_validation_manifest_file_sha256",
    "generic_validation_manifest_file_bytes", "external_lock",
    "declared_execution", "source_treatment_exact_type_verified",
    "production_authorized", "d1_eligible", "validation_eligible",
    "retrieval_qwen_execution_attested", "feature_qwen_execution_attested",
    "scorer_labels_present", "evaluator_label_schema_present", "format",
    "candidate_sha256",
}
_PHASE_KEYS = {
    "phase", "generic_corpus_sha256", "generic_root_manifest_sha256",
    "production_candidate_sha256", "production_candidate_file_sha256",
    "production_candidate_file_bytes", "partition_sha256",
    "partition_file_sha256", "partition_file_bytes", "row_count",
    "ordered_question_ids_sha256", "inventory", "inventory_sha256",
    "source_treatment_exact_type_verified", "production_authorized",
    "d1_eligible", "validation_eligible", "optimizer_updates_authorized",
    "validation_diagnostics_authorized", "checkpoint_selection_authorized",
    "retrieval_qwen_execution_attested", "feature_qwen_execution_attested",
    "sibling_partition_present", "scorer_labels_present",
    "evaluator_label_schema_present", "format", "phase_candidate_sha256",
}
_PUBLICATION_KEYS = {
    "generic_corpus_sha256", "generic_root_manifest_sha256",
    "production_candidate_sha256", "production_candidate_file_sha256",
    "production_candidate_file_bytes", "fit_phase_candidate_sha256",
    "validation_phase_candidate_sha256", "source_commit",
    "source_treatment_exact_type_verified", "production_authorized",
    "d1_eligible", "validation_eligible", "retrieval_qwen_execution_attested",
    "feature_qwen_execution_attested", "scorer_labels_present",
    "evaluator_label_schema_present", "format", "receipt_sha256",
}
_FILE_KEYS = {"relative_path", "sha256", "bytes"}


def _reject_constant(value: str) -> None:
    raise ProductionLatentTrainingCorpusError(
        f"unsupported JSON constant {value!r}"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProductionLatentTrainingCorpusError("duplicate JSON object key")
        result[key] = value
    return result


def canonical_candidate_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(value)).encode("utf-8")


def _reject_nonfinite(value: Any, label: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ProductionLatentTrainingCorpusError(f"{label} contains non-finite JSON")
    if type(value) is list:
        for item in value:
            _reject_nonfinite(item, label)
    elif type(value) is dict:
        for item in value.values():
            _reject_nonfinite(item, label)


def _loads(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise TypeError(f"{label} payload must be exact bytes")
    if not payload or len(payload) > MAX_CANDIDATE_RECEIPT_BYTES:
        raise ProductionLatentTrainingCorpusError(f"{label} exceeds its byte cap")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionLatentTrainingCorpusError(
            f"{label} is not strict UTF-8 JSON"
        ) from exc
    if type(value) is not dict or canonical_candidate_bytes(value) != payload:
        raise ProductionLatentTrainingCorpusError(
            f"{label} must be one canonical JSON object"
        )
    _reject_nonfinite(value, label)
    return value


def _object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ProductionLatentTrainingCorpusError(
            f"{label} does not match its closed schema"
        )
    return value


def _array(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise ProductionLatentTrainingCorpusError(f"{label} must be a JSON array")
    return value


def encode_external_lock(value: ProductionExternalLock) -> bytes:
    if type(value) is not ProductionExternalLock:
        raise TypeError("external lock has the wrong exact type")
    value.__post_init__()
    return canonical_candidate_bytes(value.identity_payload())


def _decode_external_lock(value: Any) -> ProductionExternalLock:
    row = _object(value, _LOCK_KEYS, "external lock")
    if row["format"] != PRODUCTION_EXTERNAL_LOCK_FORMAT:
        raise ProductionLatentTrainingCorpusError("unsupported external-lock format")
    return ProductionExternalLock(**row)


def decode_external_lock(payload: bytes) -> ProductionExternalLock:
    return _decode_external_lock(_loads(payload, "external lock"))


def _decode_execution(value: Any) -> DeclaredProductionExecutionCoordinates:
    row = _object(value, _EXECUTION_KEYS, "declared execution")
    if row["format"] != DECLARED_EXECUTION_FORMAT:
        raise ProductionLatentTrainingCorpusError("unsupported execution format")
    return DeclaredProductionExecutionCoordinates(**row)


def encode_production_candidate(value: ProductionCorpusCandidateReceipt) -> bytes:
    if type(value) is not ProductionCorpusCandidateReceipt:
        raise TypeError("production candidate has the wrong exact type")
    value.__post_init__()
    return canonical_candidate_bytes(value.identity_payload())


def decode_production_candidate(payload: bytes) -> ProductionCorpusCandidateReceipt:
    row = _object(_loads(payload, "production candidate"), _CANDIDATE_KEYS, "production candidate")
    if row["format"] != PRODUCTION_CANDIDATE_FORMAT:
        raise ProductionLatentTrainingCorpusError("unsupported candidate format")
    values = dict(row)
    values["external_lock"] = _decode_external_lock(values["external_lock"])
    values["declared_execution"] = _decode_execution(values["declared_execution"])
    return ProductionCorpusCandidateReceipt(**values)


def _decode_file(value: Any, label: str) -> LatentTrainingFileIdentity:
    row = _object(value, _FILE_KEYS, label)
    return LatentTrainingFileIdentity(
        row["relative_path"], row["sha256"], row["bytes"]
    )


def encode_phase_candidate(value: ProductionPhaseCandidateReceipt) -> bytes:
    if type(value) is not ProductionPhaseCandidateReceipt:
        raise TypeError("phase candidate has the wrong exact type")
    value.__post_init__()
    return canonical_candidate_bytes(value.identity_payload())


def decode_phase_candidate(payload: bytes) -> ProductionPhaseCandidateReceipt:
    row = _object(_loads(payload, "phase candidate"), _PHASE_KEYS, "phase candidate")
    if row["format"] != PHASE_CANDIDATE_FORMAT:
        raise ProductionLatentTrainingCorpusError("unsupported phase-candidate format")
    values = dict(row)
    values["inventory"] = tuple(
        _decode_file(item, f"phase inventory[{index}]")
        for index, item in enumerate(_array(values["inventory"], "phase inventory"))
    )
    return ProductionPhaseCandidateReceipt(**values)


def encode_candidate_publication(
    value: ProductionCandidatePublicationReceipt,
) -> bytes:
    if type(value) is not ProductionCandidatePublicationReceipt:
        raise TypeError("candidate publication has the wrong exact type")
    value.__post_init__()
    return canonical_candidate_bytes(value.identity_payload())


def decode_candidate_publication(
    payload: bytes,
) -> ProductionCandidatePublicationReceipt:
    row = _object(
        _loads(payload, "candidate publication"),
        _PUBLICATION_KEYS,
        "candidate publication",
    )
    if row["format"] != CANDIDATE_PUBLICATION_FORMAT:
        raise ProductionLatentTrainingCorpusError("unsupported publication format")
    return ProductionCandidatePublicationReceipt(**row)


__all__ = [
    "MAX_CANDIDATE_RECEIPT_BYTES", "canonical_candidate_bytes",
    "decode_candidate_publication", "decode_external_lock",
    "decode_phase_candidate", "decode_production_candidate",
    "encode_candidate_publication", "encode_external_lock",
    "encode_phase_candidate", "encode_production_candidate",
]
