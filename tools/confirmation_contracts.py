#!/usr/bin/env python3
"""Gold-neutral sealed contracts shared by confirmation prediction stages.

This module owns only canonical artifact I/O and authentication of the frozen
policy, sanitized treatment, and deterministic namespace preflight.  It has no
benchmark reader, reference-answer decoder, judge surface, or provider path.
Prediction modules may depend on this module; they must never depend on the
post-prediction gold/judge scaffold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.plan_confirmation_treatment_pipeline import (
    FORMAT as PIPELINE_PREFLIGHT_FORMAT,
    compile_confirmation_pipeline_preflight,
)
from tools.confirmation_canonical import (
    FileSnapshot,
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
from tools.confirmation_treatment import (
    CONFIRMATION_TREATMENT_INPUT_FORMAT,
    ConfirmationTreatmentInput,
    _decode_treatment_sample,
)


PREDICTIONS_FORMAT = "memory-condense-confirmation-predictions-v1"
POLICY_TREATMENT_FORMAT = "memory-condense-policy-v5-r3-treatment-projection-v1"
RUNTIME_POLICY_FORMAT = "memory-condense-policy-v5-r3-confirmation-runtime-policy-v1"
RUNTIME_POLICY_STATUS = "sanitized_prediction_runtime_policy"

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
_RUNTIME_POLICY_KEYS = {
    "format",
    "runtime_policy_identity_sha256",
    "source_policy_manifest_sha256",
    "status",
    "treatment_policy",
    "treatment_projection_sha256",
}
_RUNTIME_FORBIDDEN_KEYS = {
    "claim_profile",
    "confirmation_population",
    "freeze_date",
    "implementation",
    "manifest_identity_sha256",
    "miss_ordinals",
    "provider_accounting",
    "validation_lineage",
    "validation_result",
}
_RUNTIME_ALLOWED_VALIDATION_GUARDS = {
    "validation_artifacts_runtime_use_forbidden",
    "validation_ordinals_runtime_use_forbidden",
    "validation_question_ids_runtime_use_forbidden",
}
_RUNTIME_ALLOWED_EVALUATOR_GUARDS = {
    "gold_or_reference_available_during_prediction": False,
    "judge_available_before_all_predictions_freeze": False,
    "question_local_gold_blind_routing_only": True,
}


class ConfirmationContractError(ValueError):
    """A shared, gold-neutral confirmation contract failed closed."""


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
class RuntimePolicy:
    """A sealed runtime projection whose policy identity remains the source freeze.

    ``sha256`` deliberately means the immutable source-policy identity used by
    prediction artifacts.  ``runtime_policy_sha256`` is the separate seal on
    the sanitized file that the prediction process is permitted to open.
    """

    artifact: SealedJson
    source_policy_manifest_sha256: str

    @property
    def path(self) -> Path:
        return self.artifact.path

    @property
    def snapshot(self) -> FileSnapshot:
        return self.artifact.snapshot

    @property
    def sidecar(self) -> FileSnapshot:
        return self.artifact.sidecar

    @property
    def payload(self) -> dict[str, Any]:
        return self.artifact.payload

    @property
    def sha256(self) -> str:
        return self.source_policy_manifest_sha256

    @property
    def runtime_policy_sha256(self) -> str:
        return self.artifact.sha256


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationContractError(message)


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def _require_expected_sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)
    except ValueError as exc:
        raise ConfirmationContractError(str(exc)) from exc


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
        raise ConfirmationContractError(str(exc)) from exc
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
        raise ConfirmationContractError("cannot create artifact directory") from exc
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
        raise ConfirmationContractError("cannot publish sealed artifact") from exc
    return read_sealed_json(
        target,
        expected_sha256=digest,
        label="published output artifact",
    ), True


def decode_treatment(
    artifact: SealedJson,
) -> tuple[ConfirmationTreatmentInput, tuple[dict[str, Any], ...]]:
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
        raise ConfirmationContractError(str(exc)) from exc
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


def _validate_treatment_policy(
    policy: Mapping[str, Any],
    treatment: ConfirmationTreatmentInput,
) -> None:
    exact_keys(policy, _POLICY_TREATMENT_KEYS, "policy treatment projection")
    _require(policy["format"] == POLICY_TREATMENT_FORMAT, "unsupported policy treatment projection")
    _require(policy["policy_id"] == "policy-v5-r3", "policy treatment ID changed")
    guards = require_mapping(policy["confirmation_guards"], "policy confirmation guards")
    exact_keys(guards, set(_POLICY_CONFIRMATION_GUARDS), "policy confirmation guards")
    _require(
        guards == _POLICY_CONFIRMATION_GUARDS,
        "policy confirmation guards differ from the freeze contract",
    )
    static_root = require_mapping(
        policy["confirmation_population_static_root"],
        "policy confirmation population static root",
    )
    exact_keys(static_root, _POLICY_STATIC_ROOT_KEYS, "policy confirmation population static root")
    expected_root = {
        "dataset_sha256": treatment.dataset_sha256,
        "split_manifest_sha256": treatment.split_manifest_sha256,
        "sample_count": len(treatment.samples),
        "ordered_question_ids_sha256": treatment.ordered_question_ids_sha256,
        "ordered_normalized_sample_bindings_sha256": treatment.ordered_normalized_sample_bindings_sha256,
        "ordered_raw_record_bindings_sha256": treatment.ordered_raw_record_bindings_sha256,
    }
    _require(static_root == expected_root, "policy confirmation population static root differs from treatment")


def _assert_runtime_policy_safe(value: object, path: str = "runtime_policy") -> None:
    """Reject evaluator/runtime-location material from the sealed projection."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            _require(key not in _RUNTIME_FORBIDDEN_KEYS, f"runtime policy exposes forbidden field: {path}.{raw_key}")
            _require(
                not key.endswith("_path") and not key.endswith("_paths"),
                f"runtime policy exposes a filesystem field: {path}.{raw_key}",
            )
            if "validation" in key:
                _require(
                    key in _RUNTIME_ALLOWED_VALIDATION_GUARDS and child is True,
                    f"runtime policy exposes validation material: {path}.{raw_key}",
                )
            if key in _RUNTIME_ALLOWED_EVALUATOR_GUARDS:
                _require(
                    child is _RUNTIME_ALLOWED_EVALUATOR_GUARDS[key],
                    f"runtime policy changes evaluator guard: {path}.{raw_key}",
                )
            elif any(token in key for token in ("gold", "reference", "judge")):
                _require(
                    child is False,
                    f"runtime policy exposes evaluator material: {path}.{raw_key}",
                )
            _assert_runtime_policy_safe(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_runtime_policy_safe(child, f"{path}[{index}]")


def validate_runtime_policy_payload(
    value: Mapping[str, Any],
    treatment: ConfirmationTreatmentInput,
) -> str:
    """Validate the exact sanitized runtime-policy value and source binding."""

    exact_keys(value, _RUNTIME_POLICY_KEYS, "confirmation runtime policy")
    _require(value["format"] == RUNTIME_POLICY_FORMAT, "unsupported confirmation runtime policy")
    _require(value["status"] == RUNTIME_POLICY_STATUS, "confirmation runtime policy is not released")
    body = {
        key: item
        for key, item in value.items()
        if key != "runtime_policy_identity_sha256"
    }
    _require(
        require_sha256(value["runtime_policy_identity_sha256"], "runtime policy identity")
        == canonical_sha256(body),
        "runtime policy identity differs",
    )
    source_sha = require_sha256(
        value["source_policy_manifest_sha256"],
        "runtime source policy manifest SHA-256",
    )
    policy = require_mapping(value["treatment_policy"], "runtime treatment policy")
    _require(
        require_sha256(value["treatment_projection_sha256"], "runtime treatment projection identity")
        == canonical_sha256(policy),
        "runtime treatment projection identity differs",
    )
    _validate_treatment_policy(policy, treatment)
    _assert_runtime_policy_safe(value)
    return source_sha


def validate_runtime_policy(
    artifact: SealedJson,
    treatment: ConfirmationTreatmentInput,
) -> RuntimePolicy:
    """Authenticate the only policy artifact prediction processes may open."""

    source_sha = validate_runtime_policy_payload(artifact.payload, treatment)
    return RuntimePolicy(
        artifact=artifact,
        source_policy_manifest_sha256=source_sha,
    )


def read_runtime_policy(
    path: str | Path,
    *,
    expected_runtime_policy_sha256: str,
    treatment: ConfirmationTreatmentInput,
) -> RuntimePolicy:
    artifact = read_sealed_json(
        path,
        expected_sha256=expected_runtime_policy_sha256,
        label="sanitized confirmation runtime policy",
    )
    return validate_runtime_policy(artifact, treatment)


def verify_preflight(
    artifact: SealedJson,
    treatment: ConfirmationTreatmentInput,
) -> None:
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
        raise ConfirmationContractError(f"pipeline preflight replay failed: {exc}") from exc
    _require(value == replay, "pipeline preflight is not the deterministic treatment replay")


# Compatibility spellings while callers migrate from the evaluator module.
_decode_treatment = decode_treatment
_verify_preflight = verify_preflight


__all__ = [
    "ConfirmationContractError",
    "ConfirmationTreatmentInput",
    "POLICY_TREATMENT_FORMAT",
    "PREDICTIONS_FORMAT",
    "RUNTIME_POLICY_FORMAT",
    "RUNTIME_POLICY_STATUS",
    "RuntimePolicy",
    "SealedJson",
    "decode_treatment",
    "publish_sealed_json",
    "read_runtime_policy",
    "read_sealed_json",
    "validate_runtime_policy",
    "validate_runtime_policy_payload",
    "verify_preflight",
]
