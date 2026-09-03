#!/usr/bin/env python3
"""Readiness-gated raw-data exporter for confirmation policy-v5-r3.

This entrypoint is intentionally separate from the prediction executor.  It is
the only production process in the confirmation workflow that accepts a raw
dataset or split manifest.  Its output is a sealed, label-free treatment
projection plus the fixed 20-by-10 namespace preflight consumed by
``tools.run_confirmation_policy_v5_r3``.  It also performs the one-way
extraction from the full policy freeze into the minimal sealed runtime policy;
the prediction entrypoint has no full-freeze path or loader.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools import confirmation_contracts as contracts
from tools.plan_confirmation_treatment_pipeline import (
    SealedConfirmationPipelinePlan,
    publish_confirmation_pipeline_preflight,
    uniform_namespace_sizes,
)
from tools.run_confirmation_policy_v5_r3 import (
    ConfirmationExecutorError,
    GitState,
    OfflineTestVerifier,
    VerifiedReadiness,
    _default_git_state,
    _default_offline_test_verifier,
    _file_sha256,
    _mapping,
    _require,
    verify_confirmation_readiness,
)
from tools.v4_population_firebreak.canonical import (
    canonical_sha256,
    publish_no_clobber,
)
from tools.v4_population_firebreak.verifier import (
    export_confirmation_treatment_input,
)


FORMAT = "memory-condense-confirmation-policy-v5-r3-treatment-export-v1"
TREATMENT_EXPORT_RECEIPT_FORMAT = f"{FORMAT}-receipt-v1"
TREATMENT_EXPORT_RECEIPT_NAME = "confirmation-treatment-export-receipt-v1.json"
TREATMENT_INPUT_NAME = "confirmation-treatment-input-v1.json"
TREATMENT_PREFLIGHT_NAME = "confirmation-treatment-pipeline-preflight-v1.json"
RUNTIME_POLICY_NAME = "confirmation-runtime-policy-v1.json"
CONFIRMATION_QUESTION_COUNT = 200
CONFIRMATION_NAMESPACE_COUNT = 20
CONFIRMATION_NAMESPACE_SIZE = 10
POLICY_FREEZE_FORMAT = "memory-condense-policy-v5-r3-confirmation-freeze-v1"
POLICY_FREEZE_STATUS = "confirmation_candidate_frozen"
_POLICY_FREEZE_KEYS = {
    "claim_profile",
    "confirmation_population",
    "format",
    "freeze_date",
    "implementation",
    "manifest_identity_sha256",
    "provider_accounting",
    "status",
    "treatment_policy",
    "treatment_projection_sha256",
    "validation_lineage",
    "validation_result",
}
_VALIDATION_RESULT_KEYS = {
    "accuracy",
    "correct",
    "miss_ordinals",
    "question_count",
    "report_only",
    "runtime_use_forbidden",
    "score_complete",
}


TreatmentExporter = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ConfirmationTreatmentExport:
    readiness: VerifiedReadiness
    treatment_artifact: contracts.SealedJson
    runtime_policy: contracts.RuntimePolicy
    preflight: SealedConfirmationPipelinePlan
    export_receipt: contracts.SealedJson
    created: bool


def _publish_file_sidecar(path: Path, digest: str) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    raw = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.exists() or sidecar.is_symlink():
        _require(
            sidecar.is_file()
            and not sidecar.is_symlink()
            and sidecar.read_bytes() == raw,
            "treatment digest sidecar differs",
        )
        return False
    try:
        publish_no_clobber(sidecar, raw)
    except (OSError, ValueError) as exc:
        raise ConfirmationExecutorError("cannot publish treatment digest sidecar") from exc
    return True


def _validate_full_policy_for_export(
    artifact: contracts.SealedJson,
) -> Mapping[str, Any]:
    """Authenticate the full freeze only inside this non-provider process."""

    value = artifact.payload
    _require(set(value) == _POLICY_FREEZE_KEYS, "frozen policy manifest schema changed")
    _require(value.get("format") == POLICY_FREEZE_FORMAT, "unsupported frozen policy manifest")
    _require(value.get("status") == POLICY_FREEZE_STATUS, "policy is not frozen for confirmation")
    body = {
        key: item for key, item in value.items() if key != "manifest_identity_sha256"
    }
    _require(
        value.get("manifest_identity_sha256") == canonical_sha256(body),
        "frozen policy manifest identity differs",
    )
    validation = _mapping(value.get("validation_result"), "policy validation result")
    _require(set(validation) == _VALIDATION_RESULT_KEYS, "policy validation-result schema changed")
    _require(
        validation.get("report_only") is True
        and validation.get("runtime_use_forbidden") is True,
        "policy validation result is not report-only runtime-forbidden state",
    )
    treatment_policy = _mapping(
        value.get("treatment_policy"), "frozen treatment policy projection"
    )
    _require(
        value.get("treatment_projection_sha256")
        == canonical_sha256(treatment_policy),
        "frozen treatment policy projection identity differs",
    )
    return treatment_policy


def export_confirmation_treatment_after_readiness(
    *,
    repository_root: str | Path,
    output_root: str | Path,
    readiness_path: str | Path,
    expected_readiness_sha256: str,
    expected_policy_manifest_sha256: str,
    policy_manifest_path: str | Path | None = None,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    git_state: GitState = _default_git_state,
    offline_test_verifier: OfflineTestVerifier = _default_offline_test_verifier,
    treatment_exporter: TreatmentExporter = export_confirmation_treatment_input,
) -> ConfirmationTreatmentExport:
    """Verify v2 first, then emit only the sealed sanitized projection."""

    # Do not resolve, stat, or inspect either raw-data path above this call.
    readiness = verify_confirmation_readiness(
        repository_root=repository_root,
        readiness_path=readiness_path,
        expected_readiness_sha256=expected_readiness_sha256,
        expected_policy_manifest_sha256=expected_policy_manifest_sha256,
        git_state=git_state,
        offline_test_verifier=offline_test_verifier,
    )
    _require(policy_manifest_path is not None, "full policy manifest path is required by the standalone exporter")
    full_policy = contracts.read_sealed_json(
        policy_manifest_path,
        expected_sha256=expected_policy_manifest_sha256,
        label="frozen confirmation policy manifest",
    )
    treatment_policy = _validate_full_policy_for_export(full_policy)
    root = Path(output_root).resolve()
    _require(
        root != readiness.repository_root,
        "treatment export root cannot be the repository root",
    )
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfirmationExecutorError("cannot create treatment export root") from exc

    treatment_path = root / TREATMENT_INPUT_NAME
    existed = treatment_path.exists()
    receipt_value = treatment_exporter(
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        output_path=treatment_path,
    )
    receipt = _mapping(receipt_value, "firebreak export receipt")
    treatment_sha = _file_sha256(treatment_path, "sanitized confirmation treatment")
    _require(
        _mapping(receipt.get("treatment_input"), "treatment export input").get(
            "file_sha256"
        )
        == treatment_sha,
        "firebreak export receipt binds another treatment",
    )
    _publish_file_sidecar(treatment_path, treatment_sha)
    treatment_artifact = contracts.read_sealed_json(
        treatment_path,
        expected_sha256=treatment_sha,
        label="sanitized confirmation treatment",
    )
    treatment, _rows = contracts.decode_treatment(treatment_artifact)
    _require(
        len(treatment.samples) == CONFIRMATION_QUESTION_COUNT,
        "confirmation treatment count changed",
    )
    namespace_sizes = uniform_namespace_sizes(
        len(treatment.samples), CONFIRMATION_NAMESPACE_SIZE
    )
    _require(
        namespace_sizes
        == (CONFIRMATION_NAMESPACE_SIZE,) * CONFIRMATION_NAMESPACE_COUNT,
        "confirmation namespace schedule changed",
    )
    preflight, _preflight_created = publish_confirmation_pipeline_preflight(
        root / TREATMENT_PREFLIGHT_NAME,
        treatment,
        namespace_sizes=namespace_sizes,
    )
    runtime_body = {
        "format": contracts.RUNTIME_POLICY_FORMAT,
        "source_policy_manifest_sha256": full_policy.sha256,
        "status": contracts.RUNTIME_POLICY_STATUS,
        "treatment_policy": dict(treatment_policy),
        "treatment_projection_sha256": canonical_sha256(treatment_policy),
    }
    runtime_payload = {
        **runtime_body,
        "runtime_policy_identity_sha256": canonical_sha256(runtime_body),
    }
    contracts.validate_runtime_policy_payload(runtime_payload, treatment)
    runtime_artifact, runtime_created = contracts.publish_sealed_json(
        root / RUNTIME_POLICY_NAME,
        runtime_payload,
    )
    runtime_policy = contracts.validate_runtime_policy(runtime_artifact, treatment)
    body = {
        "format": TREATMENT_EXPORT_RECEIPT_FORMAT,
        "status": "sanitized_treatment_exported_provider_free",
        "readiness_sha256": readiness.artifact.sha256,
        "policy_manifest_sha256": expected_policy_manifest_sha256,
        "runtime_policy_path": runtime_artifact.path.name,
        "runtime_policy_sha256": runtime_artifact.sha256,
        "firebreak_export_receipt": dict(receipt),
        "treatment_input_path": treatment_path.name,
        "treatment_input_sha256": treatment_artifact.sha256,
        "treatment_preflight_path": Path(preflight.path).name,
        "treatment_preflight_sha256": preflight.sha256,
        "question_count": CONFIRMATION_QUESTION_COUNT,
        "namespace_count": CONFIRMATION_NAMESPACE_COUNT,
        "namespace_sizes": list(namespace_sizes),
        "physical_provider_calls": 0,
        "gold_or_reference_emitted": False,
    }
    export_receipt, receipt_created = contracts.publish_sealed_json(
        root / TREATMENT_EXPORT_RECEIPT_NAME,
        {**body, "receipt_identity_sha256": canonical_sha256(body)},
    )
    return ConfirmationTreatmentExport(
        readiness=readiness,
        treatment_artifact=treatment_artifact,
        runtime_policy=runtime_policy,
        preflight=preflight,
        export_receipt=export_receipt,
        created=(not existed) and runtime_created and receipt_created,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--expected-readiness-sha256", required=True)
    parser.add_argument("--expected-policy-manifest-sha256", required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exported = export_confirmation_treatment_after_readiness(
            repository_root=args.repository_root,
            output_root=args.output_root,
            readiness_path=args.readiness,
            expected_readiness_sha256=args.expected_readiness_sha256,
            expected_policy_manifest_sha256=args.expected_policy_manifest_sha256,
            policy_manifest_path=args.policy_manifest,
            dataset_path=args.dataset,
            split_manifest_path=args.split_manifest,
        )
    except (ConfirmationExecutorError, ValueError, OSError) as exc:
        print(f"confirmation treatment export failed: {exc}")
        return 2
    print(
        json.dumps(
            {
                "format": TREATMENT_EXPORT_RECEIPT_FORMAT,
                "status": "sanitized_treatment_exported_provider_free",
                "created": exported.created,
                "treatment_input_path": str(exported.treatment_artifact.path),
                "treatment_input_sha256": exported.treatment_artifact.sha256,
                "treatment_preflight_path": str(exported.preflight.path),
                "treatment_preflight_sha256": exported.preflight.sha256,
                "runtime_policy_path": str(exported.runtime_policy.path),
                "runtime_policy_sha256": exported.runtime_policy.runtime_policy_sha256,
                "source_policy_manifest_sha256": exported.runtime_policy.sha256,
                "export_receipt_path": str(exported.export_receipt.path),
                "export_receipt_sha256": exported.export_receipt.sha256,
                "physical_provider_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
