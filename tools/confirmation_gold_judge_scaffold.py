#!/usr/bin/env python3
"""Open confirmation gold only after a complete prediction plane is sealed.

This module is an evaluator-only boundary.  It has two deliberately inert
operations:

* ``compile-plan`` verifies the frozen policy, label-free treatment and
  preflight, and a complete ordered prediction plane *before* it reads the
  benchmark.  It then publishes exactly one question/reference/prediction row
  per prediction for an external Sol judge.
* ``score`` verifies an externally produced, sealed verdict plane and reports
  both the full-population score and the sensitivity score after excluding
  identities recorded in the exposure audit.

There is no provider client, provider flag, retry loop, or network path here.
All JSON artifacts owned by this module are canonical and filename-sidecar
sealed.  Existing artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from tools.plan_confirmation_treatment_pipeline import (  # noqa: E402
    FORMAT as PIPELINE_PREFLIGHT_FORMAT,
    compile_confirmation_pipeline_preflight,
)
from tools.v4_population_firebreak.analysis import (  # noqa: E402
    _partition_metadata,
    _scan_dataset_metadata,
)
from tools.v4_population_firebreak.canonical import (  # noqa: E402
    FileSnapshot,
    assert_snapshot_unchanged,
    canonical_json_bytes,
    canonical_sha256,
    exact_keys,
    parse_json_bytes,
    publish_no_clobber,
    read_snapshot,
    require_int,
    require_list,
    require_mapping,
    require_sha256,
    require_text,
)
from tools.v4_population_firebreak.population import (  # noqa: E402
    Partition,
    _answer_text,
    _parse_record,
)
from tools.v4_population_firebreak.treatment import (  # noqa: E402
    CONFIRMATION_TREATMENT_INPUT_FORMAT,
    ConfirmationTreatmentInput,
    _decode_treatment_sample,
)


PREDICTIONS_FORMAT = "memory-condense-confirmation-predictions-v1"
PREDICTION_EXECUTOR_FORMAT = "memory-condense-confirmation-policy-v5-r3-executor-v1"
RUN_MANIFEST_FORMAT = f"{PREDICTION_EXECUTOR_FORMAT}-run-manifest-v1"
PHASE_CHECKPOINT_FORMAT = f"{PREDICTION_EXECUTOR_FORMAT}-phase-checkpoint-v1"
PHASE_ARTIFACT_FORMAT = f"{PREDICTION_EXECUTOR_FORMAT}-phase-artifact-binding-v1"
PROVIDER_REQUIREMENT_FORMAT = f"{PREDICTION_EXECUTOR_FORMAT}-provider-requirement-v1"
PROVIDER_ACCOUNTING_FORMAT = f"{PREDICTION_EXECUTOR_FORMAT}-provider-accounting-v1"
PREDICTION_HANDOFF_FORMAT = f"{PREDICTION_EXECUTOR_FORMAT}-prediction-handoff-v1"
RUN_MANIFEST_NAME = "confirmation-policy-v5-r3-run-manifest-v1.json"
PHASE_DIRECTORY_NAME = "confirmation-policy-v5-r3-phases"
PREDICTION_HANDOFF_NAME = "confirmation-policy-v5-r3-prediction-handoff-v1.json"
JUDGE_PLAN_FORMAT = "memory-condense-confirmation-sol-judge-plan-v1"
JUDGE_PLAN_ROW_FORMAT = f"{JUDGE_PLAN_FORMAT}-row-v1"
JUDGE_RESULTS_FORMAT = "memory-condense-confirmation-sol-judge-results-v1"
SCORE_REPORT_FORMAT = "memory-condense-confirmation-score-report-v1"
POLICY_FREEZE_FORMAT = "memory-condense-policy-v5-r3-confirmation-freeze-v1"
POLICY_FREEZE_STATUS = "confirmation_candidate_frozen"
POLICY_TREATMENT_FORMAT = "memory-condense-policy-v5-r3-treatment-projection-v1"

_TREATMENT_KEYS = {
    "format",
    "role",
    "dataset_sha256",
    "split_manifest_sha256",
    "sample_count",
    "ordered_question_ids_sha256",
    "ordered_normalized_sample_bindings_sha256",
    "ordered_raw_record_bindings_sha256",
    "sanitized_projection_sha256",
    "samples",
}
_PREDICTION_KEYS = {
    "format",
    "status",
    "policy_manifest_sha256",
    "treatment_file_sha256",
    "treatment_preflight_sha256",
    "sample_count",
    "ordered_question_ids_sha256",
    "predictions",
}
_PREDICTION_ROW_KEYS = {"question_id", "prediction"}
_HANDOFF_KEYS = {
    "format",
    "status",
    "run_manifest_sha256",
    "prediction_phase_checkpoint_sha256",
    "predictions",
    "question_count",
    "ordered_question_ids_sha256",
    "completed_phase_checkpoint_sha256s",
    "provider_accounting",
    "safety",
    "handoff_identity_sha256",
}
_HANDOFF_PROVIDER_ACCOUNTING_KEYS = {
    "terra_required_calls",
    "terra_physical_calls",
    "terra_checkpoint_finalization_physical_calls",
    "terra_retry_limit",
    "sol_calls",
}
_HANDOFF_SAFETY = {
    "gold_or_reference_opened": False,
    "evaluation_process_started": False,
    "prediction_mutation_available": False,
}
_RUN_MANIFEST_KEYS = {
    "format",
    "policy_id",
    "readiness_sha256",
    "policy_manifest_sha256",
    "runtime_policy_sha256",
    "treatment_input_sha256",
    "treatment_preflight_sha256",
    "question_count",
    "namespace_count",
    "memory_workload",
    "ordered_question_ids_sha256",
    "phase_dag",
    "runtime",
    "safety",
    "run_identity_sha256",
}
_RUN_MEMORY_WORKLOAD_KEYS = {
    "target_memory_tokens_per_namespace",
    "namespace_count",
    "question_count",
    "namespace_sizes",
    "suffix_haystack_overlap_permitted",
    "probe_membership_separate_from_haystack_membership",
}
_RUN_RUNTIME_KEYS = {
    "factory",
    "qwen_prefix_model_dir",
    "qwen_choice_model_dir",
    "api_key_env",
    "retry_limit",
}
_RUN_SAFETY = {
    "gold_or_reference_path_available": False,
    "judge_import_available": False,
    "provider_authorization_inherited_from_readiness": False,
    "phase_provider_release_required": True,
    "prediction_and_evaluation_processes_separate": True,
}
_PHASE_CHECKPOINT_KEYS = {
    "format",
    "phase_id",
    "status",
    "run_manifest_sha256",
    "adapter_identity_sha256",
    "dependency_checkpoint_sha256s",
    "logical_question_count",
    "artifacts",
    "provider_requirement",
    "provider_accounting",
    "metadata",
    "checkpoint_identity_sha256",
}
_PHASE_ARTIFACT_KEYS = {
    "format",
    "path",
    "role",
    "sha256",
    "artifact_binding_sha256",
}
_PROVIDER_REQUIREMENT_KEYS = {
    "format",
    "provider_class",
    "required_total_calls",
    "checkpointed_calls",
    "remaining_calls",
    "retry_limit",
    "requirement_receipt_sha256",
}
_PROVIDER_ACCOUNTING_KEYS = {
    "format",
    "provider_class",
    "required_total_calls",
    "checkpointed_calls_before",
    "remaining_calls_before",
    "authorized_provider_calls",
    "physical_provider_calls",
    "completed_calls_after",
    "remaining_calls_after",
    "retry_limit",
    "accounting_receipt_sha256",
}
_PHASE_DEFINITIONS = (
    ("namespace_ingest", None),
    ("staged_cumulative_s0_s3", None),
    ("s0_terra_answer", "terra"),
    ("protected_s0", None),
    ("query_expansion", "terra"),
    ("query_direct_answer", "terra"),
    ("evidence_map", "terra"),
    ("source_streams", None),
    ("adaptive_source_map", "terra"),
    ("adaptive_evidence_solver", "terra"),
    ("adaptive_tail", "terra"),
    ("typed_final", "terra"),
    ("specialist_v3", "terra"),
    ("semantic_residual_local_global", None),
    ("terminal_v5_answer", "terra"),
    ("numeric_v5_overlay", None),
    ("prediction_seal", None),
)
_PRODUCTION_PHASE_API = {
    "namespace_ingest": (
        "tools.confirmation_namespace_store_adapter",
        "execute_confirmation_namespaces",
    ),
    "staged_cumulative_s0_s3": (
        "tools.confirmation_staged_cumulative_coordinator",
        "execute_staged_confirmation_cumulative",
    ),
    "s0_terra_answer": (
        "tools.confirmation_terra_completion_lifecycle",
        "run_provider_completion",
    ),
    "protected_s0": (
        "tools.confirmation_protected_s0_plane",
        "publish_protected_s0_answer_plane",
    ),
    "query_expansion": (
        "tools.confirmation_query_expansion_adapter",
        "run_confirmation_query_expansion_provider",
    ),
    "query_direct_answer": (
        "tools.confirmation_query_payload_parent",
        "run_confirmation_query_payload_provider",
    ),
    "evidence_map": (
        "tools.confirmation_evidence_map_parent",
        "run_confirmation_evidence_map_provider",
    ),
    "source_streams": (
        "tools.confirmation_source_streams",
        "materialize_confirmation_source_streams",
    ),
    "adaptive_source_map": (
        "tools.confirmation_adaptive_source_map",
        "run_confirmation_adaptive_source_map_provider",
    ),
    "adaptive_evidence_solver": (
        "tools.confirmation_adaptive_tail",
        "materialize_confirmation_adaptive_evidence",
    ),
    "adaptive_tail": (
        "tools.confirmation_adaptive_tail",
        "materialize_confirmation_adaptive_tail",
    ),
    "typed_final": (
        "tools.confirmation_typed_final",
        "materialize_confirmation_typed_final",
    ),
    "specialist_v3": (
        "tools.confirmation_specialist_v3",
        "replay_confirmation_specialist_v3",
    ),
    "semantic_residual_local_global": (
        "tools.confirmation_semantic_planes",
        "materialize_confirmation_semantic_planes",
    ),
    "terminal_v5_answer": (
        "tools.confirmation_terra_completion_lifecycle",
        "run_provider_completion",
    ),
    "numeric_v5_overlay": (
        "tools.materialize_confirmation_numeric_v5_overlay",
        "materialize_confirmation_numeric_v5_overlay",
    ),
    "prediction_seal": (
        "tools.materialize_confirmation_prediction_plane",
        "materialize_confirmation_prediction_plane",
    ),
}
_FORBIDDEN_PREDICTION_METADATA_KEYS = frozenset(
    {
        "gold",
        "gold_answer",
        "gold_path",
        "reference",
        "reference_answer",
        "reference_path",
        "dataset_path",
        "split_manifest_path",
        "exposure_audit_path",
        "judge_plan_path",
        "judge_results_path",
    }
)
_POLICY_FREEZE_KEYS = {
    "claim_profile",
    "confirmation_population",
    "format",
    "freeze_date",
    "implementation",
    "provider_accounting",
    "status",
    "treatment_policy",
    "treatment_projection_sha256",
    "validation_lineage",
    "validation_result",
    "manifest_identity_sha256",
}
_POLICY_TREATMENT_KEYS = {
    "arbitration_priority",
    "confirmation_guards",
    "confirmation_population_static_root",
    "format",
    "full100_policy_bindings",
    "numeric_frontier_policy",
    "policy_id",
    "responder_runtime",
    "typed_final_validator_policy_format",
}
_POLICY_STATIC_ROOT_KEYS = {
    "dataset_sha256",
    "split_manifest_sha256",
    "sample_count",
    "ordered_question_ids_sha256",
    "ordered_normalized_sample_bindings_sha256",
    "ordered_raw_record_bindings_sha256",
}
_POLICY_VALIDATION_RESULT_KEYS = {
    "accuracy",
    "correct",
    "miss_ordinals",
    "question_count",
    "score_complete",
    "report_only",
    "runtime_use_forbidden",
}
_POLICY_CONFIRMATION_GUARDS = {
    "confirmation_role_fixed": True,
    "confirmation_tuning_forbidden": True,
    "gold_or_reference_available_during_prediction": False,
    "judge_available_before_all_predictions_freeze": False,
    "policy_change_requires_new_version": True,
    "question_local_gold_blind_routing_only": True,
    "treatment_projection_only_runtime_input": True,
    "validation_artifacts_runtime_use_forbidden": True,
    "validation_ordinals_runtime_use_forbidden": True,
    "validation_question_ids_runtime_use_forbidden": True,
}
_JUDGE_PLAN_KEYS = {
    "format",
    "status",
    "bindings",
    "population",
    "exposure_audit",
    "execution",
    "rows",
    "plan_identity_sha256",
}
_JUDGE_PLAN_BINDING_KEYS = {
    "policy_manifest_sha256",
    "treatment_file_sha256",
    "treatment_preflight_sha256",
    "predictions_file_sha256",
    "prediction_handoff_sha256",
    "prediction_run_manifest_sha256",
    "dataset_sha256",
    "split_manifest_sha256",
}
_JUDGE_PLAN_POPULATION_KEYS = {
    "question_count",
    "ordered_question_ids_sha256",
}
_JUDGE_PLAN_EXPOSURE_KEYS = {
    "audit_sha256",
    "potentially_exposed_count",
    "ordered_potentially_exposed_ids_sha256",
    "membership_emitted_to_judge_rows",
    "answer_values_emitted",
}
_JUDGE_PLAN_EXECUTION_KEYS = {
    "provider_class",
    "would_call_count",
    "count_basis",
    "physical_provider_calls",
    "provider_execution_available",
    "authorization_released",
}
_JUDGE_PLAN_ROW_KEYS = {
    "format",
    "question_id",
    "question",
    "reference_answer",
    "prediction",
    "row_receipt_sha256",
}
_JUDGE_RESULTS_KEYS = {
    "format",
    "status",
    "judge_plan_sha256",
    "sample_count",
    "ordered_question_ids_sha256",
    "rows",
}
_JUDGE_RESULT_ROW_KEYS = {"question_id", "verdict"}
_VERDICTS = frozenset({"correct", "incorrect"})


class ConfirmationJudgeError(ValueError):
    """The sealed confirmation lifecycle is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class SealedJson:
    path: Path
    snapshot: FileSnapshot
    sidecar: FileSnapshot
    payload: dict[str, Any]

    @property
    def sha256(self) -> str:
        return self.snapshot.sha256


@dataclass(frozen=True, slots=True)
class PredictionGate:
    policy: SealedJson
    treatment_artifact: SealedJson
    preflight: SealedJson
    handoff: SealedJson
    run_manifest: SealedJson
    checkpoints: tuple[SealedJson, ...]
    predictions_artifact: SealedJson
    treatment: ConfirmationTreatmentInput
    treatment_samples: tuple[dict[str, Any], ...]
    question_ids: tuple[str, ...]
    predictions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExposureLock:
    snapshot: FileSnapshot
    exposed_confirmation_ids: frozenset[str]
    ordered_exposed_ids_sha256: str


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationJudgeError(message)


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def _require_expected_sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)
    except ValueError as exc:
        raise ConfirmationJudgeError(str(exc)) from exc


def read_sealed_json(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> SealedJson:
    """Read canonical JSON plus an exact filename-bearing digest sidecar."""

    target = Path(path)
    sidecar_path = target.with_name(target.name + ".sha256")
    _require(target.is_file() and not target.is_symlink(), f"{label} is not a regular file")
    _require(
        sidecar_path.is_file() and not sidecar_path.is_symlink(),
        f"{label} digest sidecar is missing or invalid",
    )
    snapshot = read_snapshot(target, label)
    sidecar = read_snapshot(sidecar_path, f"{label} digest sidecar")
    expected = _require_expected_sha(expected_sha256, f"expected {label} SHA-256")
    _require(snapshot.sha256 == expected, f"{label} differs from its external seal")
    _require(
        sidecar.payload == _sidecar_bytes(target, snapshot.sha256),
        f"{label} digest sidecar is invalid",
    )
    try:
        parsed = require_mapping(parse_json_bytes(snapshot.payload, label), label)
    except ValueError as exc:
        raise ConfirmationJudgeError(str(exc)) from exc
    _require(
        snapshot.payload == canonical_json_bytes(parsed) + b"\n",
        f"{label} is not canonical JSON",
    )
    return SealedJson(target.resolve(), snapshot, sidecar, parsed)


def publish_sealed_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> tuple[SealedJson, bool]:
    """Publish once, or reuse only byte-identical canonical JSON and sidecar."""

    target = Path(path)
    sidecar = target.with_name(target.name + ".sha256")
    value = dict(payload)
    raw = canonical_json_bytes(value) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    if target.exists() or target.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        existing = read_sealed_json(
            target,
            expected_sha256=digest,
            label="existing output artifact",
        )
        _require(existing.payload == value, "refusing to replace a different artifact")
        return existing, False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfirmationJudgeError("cannot create artifact directory") from exc
    created_target = False
    try:
        publish_no_clobber(target, raw)
        created_target = True
        publish_no_clobber(sidecar, _sidecar_bytes(target, digest))
    except (OSError, ValueError) as exc:
        if created_target and not sidecar.exists():
            try:
                if target.is_file() and not target.is_symlink() and target.read_bytes() == raw:
                    target.unlink()
            except OSError:
                pass
        raise ConfirmationJudgeError("cannot publish sealed artifact") from exc
    return read_sealed_json(
        target,
        expected_sha256=digest,
        label="published output artifact",
    ), True


def _identity_body(value: Mapping[str, Any], key: str, label: str) -> str:
    declared = _require_expected_sha(value.get(key), f"{label} identity")
    body = {field: item for field, item in value.items() if field != key}
    _require(canonical_sha256(body) == declared, f"{label} identity differs")
    return declared


def _assert_prediction_gold_blind(value: object, path: str = "prediction") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(
                str(key).casefold() not in _FORBIDDEN_PREDICTION_METADATA_KEYS,
                f"prediction metadata exposes evaluator field at {path}.{key}",
            )
            _assert_prediction_gold_blind(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_prediction_gold_blind(item, f"{path}[{index}]")


def _bound_path(root: Path, relative: object, label: str) -> Path:
    text = require_text(relative, f"{label} path")
    candidate = Path(text)
    _require(not candidate.is_absolute(), f"{label} path must be relative")
    resolved = (root / candidate).resolve()
    try:
        canonical_relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ConfirmationJudgeError(f"{label} path escapes the prediction run") from exc
    _require(
        str(canonical_relative).replace("\\", "/") == text,
        f"{label} path is not canonical",
    )
    return resolved


def _verify_bound_file(
    root: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, FileSnapshot, FileSnapshot]:
    exact_keys(binding, _PHASE_ARTIFACT_KEYS, f"{label} binding")
    _require(binding.get("format") == PHASE_ARTIFACT_FORMAT, f"{label} format changed")
    _identity_body(binding, "artifact_binding_sha256", f"{label} binding")
    require_text(binding.get("role"), f"{label} role")
    expected = _require_expected_sha(binding.get("sha256"), f"{label} SHA-256")
    target = _bound_path(root, binding.get("path"), label)
    sidecar_path = target.with_name(target.name + ".sha256")
    _require(target.is_file() and not target.is_symlink(), f"{label} is not a regular file")
    _require(
        sidecar_path.is_file() and not sidecar_path.is_symlink(),
        f"{label} digest sidecar is missing or invalid",
    )
    snapshot = read_snapshot(target, label)
    sidecar = read_snapshot(sidecar_path, f"{label} digest sidecar")
    _require(snapshot.sha256 == expected, f"{label} differs from its checkpoint binding")
    _require(
        sidecar.payload == _sidecar_bytes(target, expected),
        f"{label} digest sidecar is invalid",
    )
    return target, snapshot, sidecar


def _checkpoint_path(root: Path, index: int, phase_id: str) -> Path:
    return root / PHASE_DIRECTORY_NAME / f"{index:02d}-{phase_id}.json"


def _decode_treatment(artifact: SealedJson) -> tuple[ConfirmationTreatmentInput, tuple[dict[str, Any], ...]]:
    value = artifact.payload
    try:
        exact_keys(value, _TREATMENT_KEYS, "confirmation treatment input")
        _require(
            require_text(value["format"], "confirmation treatment format")
            == CONFIRMATION_TREATMENT_INPUT_FORMAT,
            "unsupported confirmation treatment format",
        )
        _require(value["role"] == "confirmation", "treatment role must be confirmation")
        count = require_int(value["sample_count"], "confirmation sample count", minimum=1)
        dataset_sha = require_sha256(value["dataset_sha256"], "treatment dataset SHA-256")
        split_sha = require_sha256(value["split_manifest_sha256"], "treatment split SHA-256")
        ordered_sha = require_sha256(
            value["ordered_question_ids_sha256"], "treatment ordered IDs SHA-256"
        )
        normalized_sha = require_sha256(
            value["ordered_normalized_sample_bindings_sha256"],
            "treatment normalized bindings SHA-256",
        )
        raw_sha = require_sha256(
            value["ordered_raw_record_bindings_sha256"],
            "treatment raw bindings SHA-256",
        )
        projection_sha = require_sha256(
            value["sanitized_projection_sha256"], "treatment projection SHA-256"
        )
        raw_samples = require_list(value["samples"], "confirmation treatment samples")
    except ValueError as exc:
        raise ConfirmationJudgeError(str(exc)) from exc
    _require(len(raw_samples) == count, "treatment sample count is incomplete")
    _require(canonical_sha256(raw_samples) == projection_sha, "treatment projection seal differs")
    samples = tuple(
        _decode_treatment_sample(row, index, role_label="confirmation")
        for index, row in enumerate(raw_samples)
    )
    ids = tuple(sample.sample_id for sample in samples)
    _require(len(ids) == len(set(ids)), "treatment question IDs repeat")
    _require(canonical_sha256(list(ids)) == ordered_sha, "treatment order seal differs")
    treatment = ConfirmationTreatmentInput(
        file_sha256=artifact.sha256,
        sanitized_projection_sha256=projection_sha,
        dataset_sha256=dataset_sha,
        split_manifest_sha256=split_sha,
        ordered_question_ids_sha256=ordered_sha,
        ordered_normalized_sample_bindings_sha256=normalized_sha,
        ordered_raw_record_bindings_sha256=raw_sha,
        samples=samples,
    )
    return treatment, tuple(dict(row) for row in raw_samples)


def _validate_policy_manifest(
    artifact: SealedJson,
    treatment: ConfirmationTreatmentInput,
) -> None:
    """Authenticate the freeze contract, not merely an arbitrary pinned file."""

    manifest = artifact.payload
    exact_keys(manifest, _POLICY_FREEZE_KEYS, "frozen policy manifest")
    _require(manifest["format"] == POLICY_FREEZE_FORMAT, "unsupported policy freeze format")
    _require(manifest["status"] == POLICY_FREEZE_STATUS, "policy is not frozen for confirmation")
    body = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_identity_sha256"
    }
    _require(
        require_sha256(
            manifest["manifest_identity_sha256"],
            "policy freeze manifest identity",
        )
        == canonical_sha256(body),
        "policy freeze manifest identity differs",
    )

    validation = require_mapping(
        manifest["validation_result"],
        "policy freeze validation result",
    )
    exact_keys(
        validation,
        _POLICY_VALIDATION_RESULT_KEYS,
        "policy freeze validation result",
    )
    _require(
        validation["runtime_use_forbidden"] is True,
        "validation result is not forbidden at confirmation runtime",
    )
    _require(
        validation["report_only"] is True,
        "validation result is not report-only",
    )

    policy = require_mapping(
        manifest["treatment_policy"],
        "policy freeze treatment projection",
    )
    exact_keys(policy, _POLICY_TREATMENT_KEYS, "policy freeze treatment projection")
    _require(
        policy["format"] == POLICY_TREATMENT_FORMAT,
        "unsupported policy treatment projection",
    )
    _require(policy["policy_id"] == "policy-v5-r3", "policy treatment ID changed")
    _require(
        require_sha256(
            manifest["treatment_projection_sha256"],
            "policy treatment projection identity",
        )
        == canonical_sha256(policy),
        "policy treatment projection identity differs",
    )

    guards = require_mapping(
        policy["confirmation_guards"],
        "policy confirmation guards",
    )
    exact_keys(guards, set(_POLICY_CONFIRMATION_GUARDS), "policy confirmation guards")
    _require(
        guards == _POLICY_CONFIRMATION_GUARDS,
        "policy confirmation guards differ from the freeze contract",
    )
    static_root = require_mapping(
        policy["confirmation_population_static_root"],
        "policy confirmation population static root",
    )
    exact_keys(
        static_root,
        _POLICY_STATIC_ROOT_KEYS,
        "policy confirmation population static root",
    )
    expected_root = {
        "dataset_sha256": treatment.dataset_sha256,
        "split_manifest_sha256": treatment.split_manifest_sha256,
        "sample_count": len(treatment.samples),
        "ordered_question_ids_sha256": treatment.ordered_question_ids_sha256,
        "ordered_normalized_sample_bindings_sha256": (
            treatment.ordered_normalized_sample_bindings_sha256
        ),
        "ordered_raw_record_bindings_sha256": (
            treatment.ordered_raw_record_bindings_sha256
        ),
    }
    _require(
        static_root == expected_root,
        "policy confirmation population static root differs from treatment",
    )


def _verify_preflight(artifact: SealedJson, treatment: ConfirmationTreatmentInput) -> None:
    value = artifact.payload
    _require(value.get("format") == PIPELINE_PREFLIGHT_FORMAT, "unsupported pipeline preflight")
    _require(value.get("gold_loaded") is False, "pipeline preflight is not gold-blind")
    _require(value.get("physical_provider_calls") == 0, "pipeline preflight reports provider calls")
    _require(
        value.get("provider_execution_available") is False,
        "pipeline preflight contains a provider execution path",
    )
    sizes = value.get("namespace_sizes")
    _require(type(sizes) is list, "pipeline preflight namespace schedule is absent")
    try:
        replay = compile_confirmation_pipeline_preflight(treatment, namespace_sizes=sizes)
    except ValueError as exc:
        raise ConfirmationJudgeError(f"pipeline preflight replay failed: {exc}") from exc
    _require(value == replay, "pipeline preflight is not the deterministic treatment replay")


def _decode_predictions(
    artifact: SealedJson,
    *,
    policy_sha256: str,
    treatment_sha256: str,
    preflight_sha256: str,
    ordered_question_ids_sha256: str,
    question_ids: tuple[str, ...],
) -> tuple[str, ...]:
    value = artifact.payload
    try:
        exact_keys(value, _PREDICTION_KEYS, "confirmation predictions")
        _require(value["format"] == PREDICTIONS_FORMAT, "unsupported predictions format")
        _require(value["status"] == "complete", "prediction plane is not complete")
        count = require_int(value["sample_count"], "prediction sample count", minimum=1)
        policy_binding = require_sha256(
            value["policy_manifest_sha256"], "prediction policy SHA-256"
        )
        treatment_binding = require_sha256(
            value["treatment_file_sha256"], "prediction treatment SHA-256"
        )
        preflight_binding = require_sha256(
            value["treatment_preflight_sha256"], "prediction preflight SHA-256"
        )
        ordered_binding = require_sha256(
            value["ordered_question_ids_sha256"], "prediction ordered IDs SHA-256"
        )
        rows = require_list(value["predictions"], "prediction rows")
    except ValueError as exc:
        raise ConfirmationJudgeError(str(exc)) from exc
    _require(policy_binding == policy_sha256, "predictions bind another frozen policy")
    _require(treatment_binding == treatment_sha256, "predictions bind another treatment")
    _require(preflight_binding == preflight_sha256, "predictions bind another preflight")
    _require(ordered_binding == ordered_question_ids_sha256, "predictions bind another order")
    _require(count == len(question_ids) == len(rows), "prediction plane is incomplete")
    predictions: list[str] = []
    observed_ids: list[str] = []
    for index, raw in enumerate(rows):
        row = require_mapping(raw, f"prediction row {index}")
        exact_keys(row, _PREDICTION_ROW_KEYS, f"prediction row {index}")
        question_id = require_text(row["question_id"], f"prediction row {index} ID")
        prediction = require_text(row["prediction"], f"prediction row {index} value")
        _require(question_id == question_id.strip(), f"prediction row {index} ID is not canonical")
        _require(prediction == prediction.strip(), f"prediction row {index} value is not canonical")
        observed_ids.append(question_id)
        predictions.append(prediction)
    _require(tuple(observed_ids) == question_ids, "prediction rows are missing or reordered")
    return tuple(predictions)


def _validate_run_manifest(
    artifact: SealedJson,
    *,
    policy_sha256: str,
    treatment: ConfirmationTreatmentInput,
    treatment_sha256: str,
    preflight: SealedJson,
) -> Mapping[str, str]:
    value = artifact.payload
    exact_keys(value, _RUN_MANIFEST_KEYS, "prediction run manifest")
    _require(value.get("format") == RUN_MANIFEST_FORMAT, "prediction run manifest format changed")
    _require(value.get("policy_id") == "policy-v5-r3", "prediction run policy ID changed")
    _identity_body(value, "run_identity_sha256", "prediction run manifest")
    _require_expected_sha(value.get("readiness_sha256"), "prediction run readiness SHA-256")
    _require_expected_sha(value.get("runtime_policy_sha256"), "prediction runtime policy SHA-256")
    _require(
        value.get("policy_manifest_sha256") == policy_sha256,
        "prediction run binds another frozen policy",
    )
    _require(
        value.get("treatment_input_sha256") == treatment_sha256,
        "prediction run binds another treatment",
    )
    _require(
        value.get("treatment_preflight_sha256") == preflight.sha256,
        "prediction run binds another preflight",
    )
    count = require_int(value.get("question_count"), "prediction run question count", minimum=1)
    namespace_count = require_int(
        value.get("namespace_count"), "prediction run namespace count", minimum=1
    )
    _require(count == len(treatment.samples), "prediction run population count changed")
    _require(
        value.get("ordered_question_ids_sha256")
        == treatment.ordered_question_ids_sha256,
        "prediction run ordered question root changed",
    )

    workload = require_mapping(value.get("memory_workload"), "prediction memory workload")
    exact_keys(workload, _RUN_MEMORY_WORKLOAD_KEYS, "prediction memory workload")
    namespace_sizes = require_list(workload.get("namespace_sizes"), "prediction namespace sizes")
    _require(
        workload.get("target_memory_tokens_per_namespace") == 1_000_000
        and workload.get("namespace_count") == namespace_count
        and workload.get("question_count") == count
        and namespace_sizes == preflight.payload.get("namespace_sizes")
        and len(namespace_sizes) == namespace_count
        and all(type(size) is int and size > 0 for size in namespace_sizes)
        and sum(namespace_sizes) == count
        and workload.get("suffix_haystack_overlap_permitted") is True
        and workload.get("probe_membership_separate_from_haystack_membership") is True,
        "prediction memory workload changed",
    )

    phase_dag = require_list(value.get("phase_dag"), "prediction phase DAG")
    _require(len(phase_dag) == len(_PHASE_DEFINITIONS), "prediction phase DAG is incomplete")
    prior: list[str] = []
    adapter_identities: dict[str, str] = {}
    for index, ((expected_id, expected_provider), raw) in enumerate(
        zip(_PHASE_DEFINITIONS, phase_dag, strict=True)
    ):
        row = require_mapping(raw, f"prediction phase DAG row {index}")
        exact_keys(
            row,
            {
                "dependencies",
                "phase_id",
                "production_adapter_identity_sha256",
                "production_api",
                "provider_class",
            },
            f"prediction phase DAG row {index}",
        )
        adapter_identities[expected_id] = _require_expected_sha(
            row.get("production_adapter_identity_sha256"),
            f"prediction phase DAG row {index} production adapter identity",
        )
        _require(
            row.get("phase_id") == expected_id
            and row.get("provider_class") == expected_provider
            and row.get("dependencies") == prior
            and row.get("production_api") == list(_PRODUCTION_PHASE_API[expected_id]),
            f"prediction phase DAG row {index} changed",
        )
        prior.append(expected_id)
    _require(
        len(set(adapter_identities.values())) == len(adapter_identities),
        "prediction production adapter identities repeat",
    )

    runtime = require_mapping(value.get("runtime"), "prediction runtime")
    exact_keys(runtime, _RUN_RUNTIME_KEYS, "prediction runtime")
    _require(
        runtime.get("factory")
        == "tools.confirmation_production_runtime.build_confirmation_production_runtime"
        and type(runtime.get("qwen_prefix_model_dir")) is str
        and bool(runtime.get("qwen_prefix_model_dir"))
        and Path(str(runtime["qwen_prefix_model_dir"])).is_absolute()
        and type(runtime.get("qwen_choice_model_dir")) is str
        and bool(runtime.get("qwen_choice_model_dir"))
        and Path(str(runtime["qwen_choice_model_dir"])).is_absolute()
        and type(runtime.get("api_key_env")) is str
        and bool(runtime.get("api_key_env"))
        and runtime.get("retry_limit") == 0,
        "prediction runtime changed",
    )
    safety = require_mapping(value.get("safety"), "prediction run safety")
    exact_keys(safety, set(_RUN_SAFETY), "prediction run safety")
    _require(dict(safety) == _RUN_SAFETY, "prediction run safety changed")
    return adapter_identities


def _verify_provider_receipts(
    checkpoint: Mapping[str, Any],
    *,
    expected_provider: str | None,
    phase_id: str,
) -> Mapping[str, Any]:
    requirement = require_mapping(
        checkpoint.get("provider_requirement"), f"{phase_id} provider requirement"
    )
    exact_keys(requirement, _PROVIDER_REQUIREMENT_KEYS, f"{phase_id} provider requirement")
    _identity_body(
        requirement, "requirement_receipt_sha256", f"{phase_id} provider requirement"
    )
    required = require_int(
        requirement.get("required_total_calls"), f"{phase_id} required calls"
    )
    checkpointed = require_int(
        requirement.get("checkpointed_calls"), f"{phase_id} checkpointed calls"
    )
    remaining = require_int(
        requirement.get("remaining_calls"), f"{phase_id} remaining calls"
    )
    _require(
        requirement.get("format") == PROVIDER_REQUIREMENT_FORMAT
        and requirement.get("provider_class") == expected_provider
        and checkpointed + remaining == required
        and requirement.get("retry_limit") == 0
        and (expected_provider is not None or required == 0),
        f"{phase_id} provider requirement changed",
    )

    accounting = require_mapping(
        checkpoint.get("provider_accounting"), f"{phase_id} provider accounting"
    )
    exact_keys(accounting, _PROVIDER_ACCOUNTING_KEYS, f"{phase_id} provider accounting")
    _identity_body(
        accounting, "accounting_receipt_sha256", f"{phase_id} provider accounting"
    )
    total = require_int(
        accounting.get("required_total_calls"), f"{phase_id} accounting total"
    )
    before = require_int(
        accounting.get("checkpointed_calls_before"), f"{phase_id} calls before"
    )
    remaining_before = require_int(
        accounting.get("remaining_calls_before"), f"{phase_id} calls remaining before"
    )
    authorized = require_int(
        accounting.get("authorized_provider_calls"), f"{phase_id} authorized calls"
    )
    physical = require_int(
        accounting.get("physical_provider_calls"), f"{phase_id} physical calls"
    )
    completed = require_int(
        accounting.get("completed_calls_after"), f"{phase_id} completed calls"
    )
    remaining_after = require_int(
        accounting.get("remaining_calls_after"), f"{phase_id} calls remaining after"
    )
    _require(
        accounting.get("format") == PROVIDER_ACCOUNTING_FORMAT
        and accounting.get("provider_class") == expected_provider
        and before + remaining_before == total
        and authorized == remaining_before
        and physical == remaining_before
        and completed == total
        and remaining_after == 0
        and accounting.get("retry_limit") == 0
        and total == required
        and before == checkpointed
        and remaining_before == remaining,
        f"{phase_id} provider accounting changed",
    )
    return accounting


def _verify_checkpoint(
    artifact: SealedJson,
    *,
    root: Path,
    index: int,
    phase_id: str,
    expected_provider: str | None,
    expected_adapter_identity_sha256: str,
    run_manifest_sha256: str,
    question_count: int,
    earlier: Sequence[SealedJson],
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    value = artifact.payload
    exact_keys(value, _PHASE_CHECKPOINT_KEYS, f"prediction checkpoint {phase_id}")
    _require(
        value.get("format") == PHASE_CHECKPOINT_FORMAT
        and value.get("phase_id") == phase_id
        and value.get("status") == "complete"
        and value.get("run_manifest_sha256") == run_manifest_sha256
        and value.get("logical_question_count") == question_count,
        f"prediction checkpoint {phase_id} changed",
    )
    _require(
        value.get("adapter_identity_sha256") == expected_adapter_identity_sha256,
        f"prediction checkpoint {phase_id} adapter differs from immutable production provenance",
    )
    dependencies = require_mapping(
        value.get("dependency_checkpoint_sha256s"), f"{phase_id} dependencies"
    )
    expected_dependencies = {
        earlier_id: earlier_artifact.sha256
        for (earlier_id, _), earlier_artifact in zip(
            _PHASE_DEFINITIONS[:index], earlier, strict=True
        )
    }
    _require(
        dict(dependencies) == expected_dependencies,
        f"prediction checkpoint {phase_id} dependency chain changed",
    )
    artifacts = require_list(value.get("artifacts"), f"{phase_id} artifacts")
    bindings: list[Mapping[str, Any]] = []
    roles: set[str] = set()
    for artifact_index, raw in enumerate(artifacts):
        binding = require_mapping(raw, f"{phase_id} artifact {artifact_index}")
        role = require_text(binding.get("role"), f"{phase_id} artifact role")
        _require(role not in roles, f"{phase_id} artifact roles repeat")
        roles.add(role)
        _verify_bound_file(root, binding, label=f"{phase_id} artifact {role}")
        bindings.append(binding)
    accounting = _verify_provider_receipts(
        value, expected_provider=expected_provider, phase_id=phase_id
    )
    _assert_prediction_gold_blind(value.get("metadata"), path=f"phase[{phase_id}].metadata")
    _identity_body(value, "checkpoint_identity_sha256", f"prediction checkpoint {phase_id}")
    return accounting, tuple(bindings)


def _open_prediction_handoff(
    *,
    path: str | Path,
    expected_sha256: str,
    policy_sha256: str,
    treatment: ConfirmationTreatmentInput,
    treatment_sha256: str,
    preflight: SealedJson,
) -> tuple[SealedJson, SealedJson, tuple[SealedJson, ...], SealedJson, tuple[str, ...]]:
    """Authenticate the complete prediction provenance chain without gold access."""

    handoff = read_sealed_json(
        path,
        expected_sha256=expected_sha256,
        label="sealed prediction handoff",
    )
    _require(
        handoff.path.name == PREDICTION_HANDOFF_NAME,
        "prediction handoff filename changed",
    )
    value = handoff.payload
    exact_keys(value, _HANDOFF_KEYS, "prediction handoff")
    _require(
        value.get("format") == PREDICTION_HANDOFF_FORMAT
        and value.get("status") == "predictions_sealed_evaluation_unopened",
        "prediction handoff is not evaluation-ready",
    )
    _identity_body(value, "handoff_identity_sha256", "prediction handoff")
    root = handoff.path.parent.resolve()

    run_sha = _require_expected_sha(
        value.get("run_manifest_sha256"), "prediction handoff run manifest SHA-256"
    )
    run_manifest = read_sealed_json(
        root / RUN_MANIFEST_NAME,
        expected_sha256=run_sha,
        label="prediction run manifest",
    )
    adapter_identities = _validate_run_manifest(
        run_manifest,
        policy_sha256=policy_sha256,
        treatment=treatment,
        treatment_sha256=treatment_sha256,
        preflight=preflight,
    )

    question_count = require_int(
        value.get("question_count"), "prediction handoff question count", minimum=1
    )
    ordered_root = _require_expected_sha(
        value.get("ordered_question_ids_sha256"),
        "prediction handoff ordered question root",
    )
    _require(
        question_count == len(treatment.samples)
        and ordered_root == treatment.ordered_question_ids_sha256,
        "prediction handoff population changed",
    )
    declared_checkpoint_shas = require_list(
        value.get("completed_phase_checkpoint_sha256s"),
        "prediction handoff checkpoint bindings",
    )
    _require(
        len(declared_checkpoint_shas) == len(_PHASE_DEFINITIONS),
        "prediction handoff does not bind all 17 checkpoints",
    )
    checkpoint_shas = tuple(
        _require_expected_sha(item, f"prediction checkpoint binding {index}")
        for index, item in enumerate(declared_checkpoint_shas)
    )
    _require(
        len(set(checkpoint_shas)) == len(checkpoint_shas),
        "prediction checkpoint bindings repeat",
    )

    phase_root = root / PHASE_DIRECTORY_NAME
    _require(
        phase_root.is_dir() and not phase_root.is_symlink(),
        "prediction checkpoint directory is missing or unsafe",
    )
    expected_names = {
        name
        for index, (phase_id, _) in enumerate(_PHASE_DEFINITIONS)
        for name in (
            f"{index:02d}-{phase_id}.json",
            f"{index:02d}-{phase_id}.json.sha256",
        )
    }
    _require(
        {item.name for item in phase_root.iterdir()} == expected_names,
        "prediction checkpoint directory contains missing or foreign state",
    )

    checkpoints: list[SealedJson] = []
    checkpoint_accounting: list[Mapping[str, Any]] = []
    final_bindings: tuple[Mapping[str, Any], ...] = ()
    for index, ((phase_id, provider_class), checkpoint_sha) in enumerate(
        zip(_PHASE_DEFINITIONS, checkpoint_shas, strict=True)
    ):
        checkpoint = read_sealed_json(
            _checkpoint_path(root, index, phase_id),
            expected_sha256=checkpoint_sha,
            label=f"prediction checkpoint {phase_id}",
        )
        accounting, bindings = _verify_checkpoint(
            checkpoint,
            root=root,
            index=index,
            phase_id=phase_id,
            expected_provider=provider_class,
            expected_adapter_identity_sha256=adapter_identities[phase_id],
            run_manifest_sha256=run_manifest.sha256,
            question_count=question_count,
            earlier=checkpoints,
        )
        checkpoints.append(checkpoint)
        checkpoint_accounting.append(accounting)
        if phase_id == "prediction_seal":
            final_bindings = bindings

    final = checkpoints[-1]
    _require(
        value.get("prediction_phase_checkpoint_sha256") == final.sha256,
        "prediction handoff final checkpoint binding changed",
    )
    prediction_bindings = [
        binding for binding in final_bindings if binding.get("role") == "sealed_predictions"
    ]
    _require(
        len(prediction_bindings) == 1
        and isinstance(value.get("predictions"), Mapping)
        and dict(value["predictions"]) == dict(prediction_bindings[0]),
        "prediction handoff does not bind the final prediction artifact",
    )

    terra_rows = [
        row for row in checkpoint_accounting if row.get("provider_class") == "terra"
    ]
    expected_accounting = {
        "terra_required_calls": sum(int(row["required_total_calls"]) for row in terra_rows),
        "terra_physical_calls": sum(int(row["completed_calls_after"]) for row in terra_rows),
        "terra_checkpoint_finalization_physical_calls": sum(
            int(row["physical_provider_calls"]) for row in terra_rows
        ),
        "terra_retry_limit": 0,
        "sol_calls": 0,
    }
    handoff_accounting = require_mapping(
        value.get("provider_accounting"), "prediction handoff provider accounting"
    )
    exact_keys(
        handoff_accounting,
        _HANDOFF_PROVIDER_ACCOUNTING_KEYS,
        "prediction handoff provider accounting",
    )
    _require(
        dict(handoff_accounting) == expected_accounting,
        "prediction handoff provider accounting changed",
    )
    safety = require_mapping(value.get("safety"), "prediction handoff safety")
    exact_keys(safety, set(_HANDOFF_SAFETY), "prediction handoff safety")
    _require(dict(safety) == _HANDOFF_SAFETY, "prediction handoff safety changed")

    prediction_binding = prediction_bindings[0]
    prediction_path, _, _ = _verify_bound_file(
        root, prediction_binding, label="sealed predictions"
    )
    predictions_artifact = read_sealed_json(
        prediction_path,
        expected_sha256=str(prediction_binding["sha256"]),
        label="complete confirmation predictions",
    )
    question_ids = tuple(sample.sample_id for sample in treatment.samples)
    predictions = _decode_predictions(
        predictions_artifact,
        policy_sha256=policy_sha256,
        treatment_sha256=treatment_sha256,
        preflight_sha256=preflight.sha256,
        ordered_question_ids_sha256=treatment.ordered_question_ids_sha256,
        question_ids=question_ids,
    )
    return handoff, run_manifest, tuple(checkpoints), predictions_artifact, predictions


def verify_prediction_gate(
    *,
    policy_manifest_path: str | Path,
    expected_policy_manifest_sha256: str,
    treatment_input_path: str | Path,
    expected_treatment_input_sha256: str,
    treatment_preflight_path: str | Path,
    expected_treatment_preflight_sha256: str,
    prediction_handoff_path: str | Path,
    expected_prediction_handoff_sha256: str,
) -> PredictionGate:
    """Verify the handoff and its complete ancestry; this function cannot read gold."""

    policy = read_sealed_json(
        policy_manifest_path,
        expected_sha256=expected_policy_manifest_sha256,
        label="frozen policy manifest",
    )
    treatment_artifact = read_sealed_json(
        treatment_input_path,
        expected_sha256=expected_treatment_input_sha256,
        label="label-free confirmation treatment",
    )
    treatment, treatment_samples = _decode_treatment(treatment_artifact)
    _validate_policy_manifest(policy, treatment)
    preflight = read_sealed_json(
        treatment_preflight_path,
        expected_sha256=expected_treatment_preflight_sha256,
        label="label-free confirmation preflight",
    )
    _verify_preflight(preflight, treatment)
    question_ids = tuple(sample.sample_id for sample in treatment.samples)
    handoff, run_manifest, checkpoints, predictions_artifact, predictions = (
        _open_prediction_handoff(
            path=prediction_handoff_path,
            expected_sha256=expected_prediction_handoff_sha256,
            policy_sha256=policy.sha256,
            treatment=treatment,
            treatment_sha256=treatment_artifact.sha256,
            preflight=preflight,
        )
    )
    _require(
        run_manifest.payload.get("policy_manifest_sha256") == policy.sha256,
        "prediction handoff ancestry binds another frozen policy",
    )
    return PredictionGate(
        policy=policy,
        treatment_artifact=treatment_artifact,
        preflight=preflight,
        handoff=handoff,
        run_manifest=run_manifest,
        checkpoints=checkpoints,
        predictions_artifact=predictions_artifact,
        treatment=treatment,
        treatment_samples=treatment_samples,
        question_ids=question_ids,
        predictions=predictions,
    )


def _open_confirmation_gold(
    gate: PredictionGate,
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
) -> tuple[FileSnapshot, FileSnapshot, tuple[tuple[str, str], ...]]:
    """Evaluator-only adapter used only after :func:`verify_prediction_gate`."""

    dataset = read_snapshot(dataset_path, "confirmation benchmark dataset")
    split = read_snapshot(split_manifest_path, "confirmation split manifest")
    _require(dataset.sha256 == gate.treatment.dataset_sha256, "dataset differs from treatment lock")
    _require(split.sha256 == gate.treatment.split_manifest_sha256, "split differs from treatment lock")
    try:
        manifest = require_mapping(
            parse_json_bytes(split.payload, "confirmation split manifest"),
            "confirmation split manifest",
        )
        exact_keys(
            manifest,
            {"format", "dataset_sha256", "salt", "algorithm", "splits"},
            "confirmation split manifest",
        )
        _require(
            manifest["dataset_sha256"] == dataset.sha256,
            "split manifest does not bind the benchmark dataset",
        )
        _require(
            manifest["format"] == "memory-condense-locked-benchmark-split-v1",
            "unsupported confirmation split format",
        )
        _require(
            manifest["algorithm"] == "stratified-largest-remainder-v1",
            "unsupported confirmation split algorithm",
        )
        split_salt = require_text(manifest["salt"], "confirmation split salt")
        raw_counts = require_mapping(manifest["splits"], "confirmation split counts")
        _require(
            list(raw_counts) == ["development", "validation", "confirmation"],
            "confirmation split order or names differ",
        )
        counts = {
            name: require_int(value, f"confirmation split count {name}", minimum=1)
            for name, value in raw_counts.items()
        }
        metadata = _scan_dataset_metadata(dataset.payload)
    except ValueError as exc:
        raise ConfirmationJudgeError(f"cannot reconstruct confirmation population: {exc}") from exc
    _require(
        len(metadata) == sum(counts.values()),
        "split counts do not cover the benchmark record population",
    )
    metadata_ids = [item.sample_id for item in metadata]
    _require(len(metadata_ids) == len(set(metadata_ids)), "benchmark repeats a question ID")
    try:
        selected = _partition_metadata(metadata, counts, split_salt)
    except ValueError as exc:
        raise ConfirmationJudgeError(f"cannot reconstruct confirmation partition: {exc}") from exc
    confirmation_metadata = selected["confirmation"]
    _require(
        len(confirmation_metadata) == len(gate.question_ids)
        and canonical_sha256([item.sample_id for item in confirmation_metadata])
        == gate.treatment.ordered_question_ids_sha256,
        "benchmark confirmation membership differs from treatment lock",
    )

    opened: list[tuple[str, str]] = []
    confirmation_samples = []
    for index, (metadata_row, treatment_projection) in enumerate(
        zip(confirmation_metadata, gate.treatment_samples, strict=True)
    ):
        record = require_mapping(
            parse_json_bytes(
                dataset.payload[metadata_row.start : metadata_row.end],
                f"confirmation benchmark record {index}",
            ),
            f"confirmation benchmark record {index}",
        )
        projected = _parse_record(record, metadata_row.index)
        _require(
            projected is not None
            and projected.sample_id == metadata_row.sample_id
            and projected.category == metadata_row.category
            and projected.treatment_projection == treatment_projection,
            f"benchmark record differs from treatment at confirmation row {index}",
        )
        confirmation_samples.append(projected)
        question_value = treatment_projection["questions"][0]
        question = require_text(question_value["question"], f"confirmation question {index}")
        question_date = question_value["question_date"]
        dated_question = (
            question
            if question_date is None
            else f"[Question asked at {question_date}]\n{question}"
        )
        reference = _answer_text(record.get("answer"))
        _require(bool(reference), f"benchmark gold is absent at confirmation row {index}")
        opened.append((dated_question, reference))
    confirmation = Partition(
        name="confirmation",
        samples=tuple(confirmation_samples),
    )
    _require(
        confirmation.ordered_ids_sha256 == gate.treatment.ordered_question_ids_sha256
        and confirmation.ordered_normalized_bindings_sha256
        == gate.treatment.ordered_normalized_sample_bindings_sha256
        and confirmation.ordered_raw_bindings_sha256
        == gate.treatment.ordered_raw_record_bindings_sha256,
        "benchmark confirmation source bindings differ from treatment lock",
    )
    return dataset, split, tuple(opened)


def _load_exposure_lock(
    path: str | Path,
    *,
    expected_sha256: str,
    question_ids: tuple[str, ...],
    expected_exposed_count: int,
    expected_ordered_exposed_ids_sha256: str,
) -> ExposureLock:
    snapshot = read_snapshot(path, "confirmation exposure audit")
    expected_audit = _require_expected_sha(expected_sha256, "expected exposure audit SHA-256")
    _require(snapshot.sha256 == expected_audit, "exposure audit differs from its external seal")
    value = require_mapping(parse_json_bytes(snapshot.payload, "confirmation exposure audit"), "confirmation exposure audit")
    rows = require_list(value.get("numeric_answers"), "exposure audit numeric metadata")
    audit_ids: list[str] = []
    for index, raw in enumerate(rows):
        row = require_mapping(raw, f"exposure audit row {index}")
        audit_ids.append(require_text(row.get("question_id"), f"exposure audit row {index} ID"))
    _require(len(audit_ids) == len(set(audit_ids)), "exposure audit repeats an identity")
    exposed = frozenset(audit_ids) & frozenset(question_ids)
    ordered = [question_id for question_id in question_ids if question_id in exposed]
    expected_count = require_int(expected_exposed_count, "expected exposed count")
    expected_root = _require_expected_sha(
        expected_ordered_exposed_ids_sha256,
        "expected ordered exposed IDs SHA-256",
    )
    digest = canonical_sha256(ordered)
    _require(len(ordered) == expected_count, "exposure sensitivity count differs from its lock")
    _require(digest == expected_root, "exposure sensitivity identities differ from their lock")
    return ExposureLock(snapshot, exposed, digest)


def _judge_row(question_id: str, question: str, reference: str, prediction: str) -> dict[str, Any]:
    body = {
        "format": JUDGE_PLAN_ROW_FORMAT,
        "question_id": question_id,
        "question": question,
        "reference_answer": reference,
        "prediction": prediction,
    }
    return {**body, "row_receipt_sha256": canonical_sha256(body)}


def compile_confirmation_judge_plan(
    *,
    policy_manifest_path: str | Path,
    expected_policy_manifest_sha256: str,
    treatment_input_path: str | Path,
    expected_treatment_input_sha256: str,
    treatment_preflight_path: str | Path,
    expected_treatment_preflight_sha256: str,
    prediction_handoff_path: str | Path,
    expected_prediction_handoff_sha256: str,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    exposure_audit_path: str | Path,
    expected_exposure_audit_sha256: str,
    expected_exposed_count: int,
    expected_ordered_exposed_ids_sha256: str,
) -> dict[str, Any]:
    """Compile exactly N inert Sol rows after the complete prediction gate."""

    # This ordering is the core firebreak: neither dataset nor exposure path is
    # passed to a reader until every label-free input and prediction is exact.
    gate = verify_prediction_gate(
        policy_manifest_path=policy_manifest_path,
        expected_policy_manifest_sha256=expected_policy_manifest_sha256,
        treatment_input_path=treatment_input_path,
        expected_treatment_input_sha256=expected_treatment_input_sha256,
        treatment_preflight_path=treatment_preflight_path,
        expected_treatment_preflight_sha256=expected_treatment_preflight_sha256,
        prediction_handoff_path=prediction_handoff_path,
        expected_prediction_handoff_sha256=expected_prediction_handoff_sha256,
    )
    dataset, split, opened = _open_confirmation_gold(
        gate,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
    )
    exposure = _load_exposure_lock(
        exposure_audit_path,
        expected_sha256=expected_exposure_audit_sha256,
        question_ids=gate.question_ids,
        expected_exposed_count=expected_exposed_count,
        expected_ordered_exposed_ids_sha256=expected_ordered_exposed_ids_sha256,
    )
    rows = [
        _judge_row(question_id, question, reference, prediction)
        for question_id, prediction, (question, reference) in zip(
            gate.question_ids, gate.predictions, opened, strict=True
        )
    ]
    count = len(rows)
    body: dict[str, Any] = {
        "format": JUDGE_PLAN_FORMAT,
        "status": "compiled",
        "bindings": {
            "policy_manifest_sha256": gate.policy.sha256,
            "treatment_file_sha256": gate.treatment_artifact.sha256,
            "treatment_preflight_sha256": gate.preflight.sha256,
            "predictions_file_sha256": gate.predictions_artifact.sha256,
            "prediction_handoff_sha256": gate.handoff.sha256,
            "prediction_run_manifest_sha256": gate.run_manifest.sha256,
            "dataset_sha256": dataset.sha256,
            "split_manifest_sha256": split.sha256,
        },
        "population": {
            "question_count": count,
            "ordered_question_ids_sha256": gate.treatment.ordered_question_ids_sha256,
        },
        "exposure_audit": {
            "audit_sha256": exposure.snapshot.sha256,
            "potentially_exposed_count": len(exposure.exposed_confirmation_ids),
            "ordered_potentially_exposed_ids_sha256": exposure.ordered_exposed_ids_sha256,
            "membership_emitted_to_judge_rows": False,
            "answer_values_emitted": False,
        },
        "execution": {
            "provider_class": "sol",
            "would_call_count": count,
            "count_basis": "one-call-per-sealed-confirmation-prediction",
            "physical_provider_calls": 0,
            "provider_execution_available": False,
            "authorization_released": False,
        },
        "rows": rows,
    }
    plan = {**body, "plan_identity_sha256": canonical_sha256(body)}
    for sealed, label in (
        (gate.policy, "frozen policy manifest"),
        (gate.treatment_artifact, "label-free confirmation treatment"),
        (gate.preflight, "label-free confirmation preflight"),
        (gate.handoff, "sealed prediction handoff"),
        (gate.run_manifest, "prediction run manifest"),
        (gate.predictions_artifact, "complete confirmation predictions"),
    ):
        assert_snapshot_unchanged(sealed.snapshot, label)
        assert_snapshot_unchanged(sealed.sidecar, f"{label} digest sidecar")
    for index, sealed in enumerate(gate.checkpoints):
        phase_id = _PHASE_DEFINITIONS[index][0]
        assert_snapshot_unchanged(sealed.snapshot, f"prediction checkpoint {phase_id}")
        assert_snapshot_unchanged(
            sealed.sidecar, f"prediction checkpoint {phase_id} digest sidecar"
        )
    assert_snapshot_unchanged(dataset, "confirmation benchmark dataset")
    assert_snapshot_unchanged(split, "confirmation split manifest")
    assert_snapshot_unchanged(exposure.snapshot, "confirmation exposure audit")
    return plan


def publish_confirmation_judge_plan(output_path: str | Path, **kwargs: Any) -> tuple[SealedJson, bool]:
    return publish_sealed_json(output_path, compile_confirmation_judge_plan(**kwargs))


def _validate_judge_plan(artifact: SealedJson) -> tuple[str, ...]:
    plan = artifact.payload
    exact_keys(plan, _JUDGE_PLAN_KEYS, "confirmation judge plan")
    _require(plan["format"] == JUDGE_PLAN_FORMAT, "unsupported confirmation judge plan")
    _require(plan["status"] == "compiled", "confirmation judge plan is not compiled")
    body = {key: value for key, value in plan.items() if key != "plan_identity_sha256"}
    _require(
        require_sha256(plan["plan_identity_sha256"], "judge plan identity")
        == canonical_sha256(body),
        "judge plan identity differs",
    )
    bindings = require_mapping(plan["bindings"], "judge plan bindings")
    exact_keys(bindings, _JUDGE_PLAN_BINDING_KEYS, "judge plan bindings")
    for key, value in bindings.items():
        require_sha256(value, f"judge plan binding {key}")
    population = require_mapping(plan["population"], "judge plan population")
    exact_keys(population, _JUDGE_PLAN_POPULATION_KEYS, "judge plan population")
    count = require_int(population["question_count"], "judge plan question count", minimum=1)
    ordered_sha = require_sha256(population["ordered_question_ids_sha256"], "judge plan order")
    exposure = require_mapping(plan["exposure_audit"], "judge plan exposure binding")
    exact_keys(exposure, _JUDGE_PLAN_EXPOSURE_KEYS, "judge plan exposure binding")
    require_sha256(exposure["audit_sha256"], "judge plan exposure audit SHA-256")
    exposed_count = require_int(exposure["potentially_exposed_count"], "judge plan exposed count")
    _require(exposed_count <= count, "judge plan exposed count exceeds its population")
    require_sha256(
        exposure["ordered_potentially_exposed_ids_sha256"],
        "judge plan exposed order",
    )
    _require(exposure["membership_emitted_to_judge_rows"] is False, "judge rows reveal exposure membership")
    _require(exposure["answer_values_emitted"] is False, "judge plan exposure binding reveals values")
    execution = require_mapping(plan["execution"], "judge plan execution")
    exact_keys(execution, _JUDGE_PLAN_EXECUTION_KEYS, "judge plan execution")
    _require(execution["provider_class"] == "sol", "judge plan provider class changed")
    _require(execution["would_call_count"] == count, "judge plan call count is not exact")
    _require(
        execution["count_basis"] == "one-call-per-sealed-confirmation-prediction",
        "judge plan call-count basis changed",
    )
    _require(execution["physical_provider_calls"] == 0, "scaffold reports provider execution")
    _require(execution["provider_execution_available"] is False, "scaffold contains provider execution")
    _require(execution["authorization_released"] is False, "judge plan claims provider authorization")
    raw_rows = require_list(plan["rows"], "judge plan rows")
    _require(len(raw_rows) == count, "judge plan rows are incomplete")
    ids: list[str] = []
    for index, raw in enumerate(raw_rows):
        row = require_mapping(raw, f"judge plan row {index}")
        exact_keys(row, _JUDGE_PLAN_ROW_KEYS, f"judge plan row {index}")
        _require(row["format"] == JUDGE_PLAN_ROW_FORMAT, f"judge plan row {index} format changed")
        question_id = require_text(row["question_id"], f"judge plan row {index} ID")
        body = {key: value for key, value in row.items() if key != "row_receipt_sha256"}
        _require(
            require_sha256(row["row_receipt_sha256"], f"judge plan row {index} receipt")
            == canonical_sha256(body),
            f"judge plan row {index} identity differs",
        )
        for key in ("question", "reference_answer", "prediction"):
            require_text(row[key], f"judge plan row {index} {key}")
        ids.append(question_id)
    _require(len(ids) == len(set(ids)), "judge plan repeats a question ID")
    _require(canonical_sha256(ids) == ordered_sha, "judge plan rows are reordered")
    return tuple(ids)


def _decode_judge_results(
    artifact: SealedJson,
    *,
    judge_plan_sha256: str,
    question_ids: tuple[str, ...],
) -> tuple[bool, ...]:
    value = artifact.payload
    exact_keys(value, _JUDGE_RESULTS_KEYS, "confirmation judge results")
    _require(value["format"] == JUDGE_RESULTS_FORMAT, "unsupported judge results format")
    _require(value["status"] == "complete", "judge verdict plane is not complete")
    _require(
        require_sha256(value["judge_plan_sha256"], "judge result plan binding")
        == judge_plan_sha256,
        "judge results bind another plan",
    )
    count = require_int(value["sample_count"], "judge result sample count", minimum=1)
    _require(count == len(question_ids), "judge verdict plane has the wrong population size")
    _require(
        require_sha256(value["ordered_question_ids_sha256"], "judge result order")
        == canonical_sha256(list(question_ids)),
        "judge results bind another order",
    )
    rows = require_list(value["rows"], "judge result rows")
    _require(len(rows) == count, "judge verdict plane is incomplete")
    verdicts: list[bool] = []
    observed: list[str] = []
    for index, raw in enumerate(rows):
        row = require_mapping(raw, f"judge result row {index}")
        exact_keys(row, _JUDGE_RESULT_ROW_KEYS, f"judge result row {index}")
        question_id = require_text(row["question_id"], f"judge result row {index} ID")
        verdict = require_text(row["verdict"], f"judge result row {index} verdict")
        _require(verdict in _VERDICTS, f"judge result row {index} verdict is invalid")
        observed.append(question_id)
        verdicts.append(verdict == "correct")
    _require(tuple(observed) == question_ids, "judge result rows are missing or reordered")
    return tuple(verdicts)


def _aggregate(values: Sequence[bool]) -> dict[str, Any]:
    total = len(values)
    correct = sum(values)
    incorrect = total - correct
    percent = (
        Decimal(correct * 100) / Decimal(total)
        if total
        else Decimal(0)
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {
        "question_count": total,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "accuracy_fraction": f"{correct}/{total}",
        "accuracy_percent": format(percent, ".2f"),
    }


def compile_confirmation_score_report(
    *,
    judge_plan_path: str | Path,
    expected_judge_plan_sha256: str,
    judge_results_path: str | Path,
    expected_judge_results_sha256: str,
    exposure_audit_path: str | Path,
    expected_exposure_audit_sha256: str,
) -> dict[str, Any]:
    plan = read_sealed_json(
        judge_plan_path,
        expected_sha256=expected_judge_plan_sha256,
        label="confirmation judge plan",
    )
    question_ids = _validate_judge_plan(plan)
    results = read_sealed_json(
        judge_results_path,
        expected_sha256=expected_judge_results_sha256,
        label="complete confirmation judge results",
    )
    verdicts = _decode_judge_results(
        results,
        judge_plan_sha256=plan.sha256,
        question_ids=question_ids,
    )
    exposure_binding = require_mapping(plan.payload["exposure_audit"], "judge plan exposure binding")
    exposure = _load_exposure_lock(
        exposure_audit_path,
        expected_sha256=expected_exposure_audit_sha256,
        question_ids=question_ids,
        expected_exposed_count=int(exposure_binding["potentially_exposed_count"]),
        expected_ordered_exposed_ids_sha256=str(
            exposure_binding["ordered_potentially_exposed_ids_sha256"]
        ),
    )
    _require(
        exposure.snapshot.sha256 == exposure_binding["audit_sha256"],
        "score exposure audit differs from the judge-plan binding",
    )
    non_exposed = [
        verdict
        for question_id, verdict in zip(question_ids, verdicts, strict=True)
        if question_id not in exposure.exposed_confirmation_ids
    ]
    exposed = [
        verdict
        for question_id, verdict in zip(question_ids, verdicts, strict=True)
        if question_id in exposure.exposed_confirmation_ids
    ]
    report = {
        "format": SCORE_REPORT_FORMAT,
        "status": "scored",
        "bindings": {
            "judge_plan_sha256": plan.sha256,
            "judge_results_sha256": results.sha256,
            "exposure_audit_sha256": exposure.snapshot.sha256,
        },
        "full_population": _aggregate(verdicts),
        "non_exposed_sensitivity": {
            **_aggregate(non_exposed),
            "claim": "excludes-identities-in-recorded-answer-metadata-audit",
        },
        "potentially_exposed_sensitivity": {
            **_aggregate(exposed),
            "claim": "identities-in-recorded-answer-metadata-audit",
        },
        "scaffold_provider_calls": 0,
    }
    for sealed, label in (
        (plan, "confirmation judge plan"),
        (results, "complete confirmation judge results"),
    ):
        assert_snapshot_unchanged(sealed.snapshot, label)
        assert_snapshot_unchanged(sealed.sidecar, f"{label} digest sidecar")
    assert_snapshot_unchanged(exposure.snapshot, "confirmation exposure audit")
    return report


def publish_confirmation_score_report(output_path: str | Path, **kwargs: Any) -> tuple[SealedJson, bool]:
    return publish_sealed_json(output_path, compile_confirmation_score_report(**kwargs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("compile-plan", help="compile inert Sol request rows")
    plan.add_argument("--policy-manifest", type=Path, required=True)
    plan.add_argument("--expected-policy-manifest-sha256", required=True)
    plan.add_argument("--treatment-input", type=Path, required=True)
    plan.add_argument("--expected-treatment-input-sha256", required=True)
    plan.add_argument("--treatment-preflight", type=Path, required=True)
    plan.add_argument("--expected-treatment-preflight-sha256", required=True)
    plan.add_argument("--prediction-handoff", type=Path, required=True)
    plan.add_argument("--expected-prediction-handoff-sha256", required=True)
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--split-manifest", type=Path, required=True)
    plan.add_argument("--exposure-audit", type=Path, required=True)
    plan.add_argument("--expected-exposure-audit-sha256", required=True)
    plan.add_argument("--expected-exposed-count", type=int, required=True)
    plan.add_argument("--expected-ordered-exposed-ids-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser("score", help="score a sealed external verdict plane")
    score.add_argument("--judge-plan", type=Path, required=True)
    score.add_argument("--expected-judge-plan-sha256", required=True)
    score.add_argument("--judge-results", type=Path, required=True)
    score.add_argument("--expected-judge-results-sha256", required=True)
    score.add_argument("--exposure-audit", type=Path, required=True)
    score.add_argument("--expected-exposure-audit-sha256", required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "compile-plan":
        artifact, created = publish_confirmation_judge_plan(
            args.output,
            policy_manifest_path=args.policy_manifest,
            expected_policy_manifest_sha256=args.expected_policy_manifest_sha256,
            treatment_input_path=args.treatment_input,
            expected_treatment_input_sha256=args.expected_treatment_input_sha256,
            treatment_preflight_path=args.treatment_preflight,
            expected_treatment_preflight_sha256=args.expected_treatment_preflight_sha256,
            prediction_handoff_path=args.prediction_handoff,
            expected_prediction_handoff_sha256=args.expected_prediction_handoff_sha256,
            dataset_path=args.dataset,
            split_manifest_path=args.split_manifest,
            exposure_audit_path=args.exposure_audit,
            expected_exposure_audit_sha256=args.expected_exposure_audit_sha256,
            expected_exposed_count=args.expected_exposed_count,
            expected_ordered_exposed_ids_sha256=args.expected_ordered_exposed_ids_sha256,
        )
        return {
            "created": created,
            "artifact_sha256": artifact.sha256,
            "would_call_count": artifact.payload["execution"]["would_call_count"],
            "physical_provider_calls": 0,
        }
    if args.command == "score":
        artifact, created = publish_confirmation_score_report(
            args.output,
            judge_plan_path=args.judge_plan,
            expected_judge_plan_sha256=args.expected_judge_plan_sha256,
            judge_results_path=args.judge_results,
            expected_judge_results_sha256=args.expected_judge_results_sha256,
            exposure_audit_path=args.exposure_audit,
            expected_exposure_audit_sha256=args.expected_exposure_audit_sha256,
        )
        return {
            "created": created,
            "artifact_sha256": artifact.sha256,
            "full_population": artifact.payload["full_population"],
            "non_exposed_sensitivity": artifact.payload["non_exposed_sensitivity"],
            "physical_provider_calls": 0,
        }
    raise ConfirmationJudgeError("unknown command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except (ConfirmationJudgeError, ValueError) as exc:
        print(f"confirmation judge scaffold failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConfirmationJudgeError",
    "JUDGE_PLAN_FORMAT",
    "JUDGE_RESULTS_FORMAT",
    "PREDICTIONS_FORMAT",
    "SCORE_REPORT_FORMAT",
    "build_parser",
    "compile_confirmation_judge_plan",
    "compile_confirmation_score_report",
    "main",
    "publish_confirmation_judge_plan",
    "publish_confirmation_score_report",
    "publish_sealed_json",
    "read_sealed_json",
    "verify_prediction_gate",
]
