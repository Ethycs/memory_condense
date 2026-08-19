"""Runtime-level receipts for one certified three-arm diffuse comparison."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.diffuse_longmemeval_matched import (
    MATCHED_BOUNDARY_MODES,
    DiffuseLongMemEvalMatchedSuiteReceipt,
    validate_matched_diffuse_retrieval_phases,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DIFFUSE_RUNTIME_RESULT_FORMAT,
    DiffuseLongMemEvalRuntimeResult,
    ResidencyPreflightObservation,
)


DIFFUSE_MATCHED_RUNTIME_SUITE_FORMAT = (
    "memory-condense-longmemeval-diffuse-matched-runtime-suite-v1"
)
RESIDENT_PREFLIGHT_POLICY = "cuda-mem-get-info-min-free-v1"
STAGED_PREFLIGHT_POLICY = "bge-close-before-qwen-load-v1"
MINIMUM_RESIDENT_FREE_BYTES = 3072 * 1024 * 1024
_CUDA_DEVICE_RE = re.compile(r"^cuda(?::[0-9]+)?$")


def _digest(value: object, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        normalized = int(value)  # type: ignore[arg-type]
        exact = float(value) == float(normalized)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not exact or normalized < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return normalized


def _preflight_payload(
    observation: ResidencyPreflightObservation,
) -> dict[str, object]:
    return {
        name: getattr(observation, name)
        for name in observation.__dataclass_fields__
    }


def _validate_preflight_observations(
    observations: Sequence[ResidencyPreflightObservation],
) -> tuple[str, str, int]:
    frozen = tuple(observations)
    if len(frozen) != len(MATCHED_BOUNDARY_MODES):
        raise ValueError("runtime suite requires one preflight per arm")
    if any(type(item) is not ResidencyPreflightObservation for item in frozen):
        raise TypeError("runtime suite requires exact preflight observations")
    policies = {item.policy for item in frozen}
    devices = {str(item.device).casefold() for item in frozen}
    required_values = {item.required_free_bytes for item in frozen}
    if len(policies) != 1:
        raise ValueError("matched runtime arms changed residency policy")
    if len(devices) != 1:
        raise ValueError("matched runtime arms changed Qwen device")
    if len(required_values) != 1:
        raise ValueError("matched runtime arms changed the residency threshold")
    policy = next(iter(policies))
    device = next(iter(devices))
    if _CUDA_DEVICE_RE.fullmatch(device) is None:
        raise ValueError("certified matched runtime requires an explicit CUDA device")

    if policy == RESIDENT_PREFLIGHT_POLICY:
        required = _exact_int(
            next(iter(required_values)),
            "resident required_free_bytes",
            minimum=MINIMUM_RESIDENT_FREE_BYTES,
        )
        for item in frozen:
            if item.embedding_released_before_qwen_load is not False:
                raise ValueError("resident runtime cannot release BGE before Qwen")
            if item.observed_free_bytes is None or item.observed_total_bytes is None:
                raise ValueError("resident runtime requires observed CUDA memory")
            free = _exact_int(item.observed_free_bytes, "observed_free_bytes")
            total = _exact_int(item.observed_total_bytes, "observed_total_bytes")
            if free < required:
                raise ValueError("resident CUDA free memory is below its threshold")
            if total < free:
                raise ValueError("observed CUDA total memory is below free memory")
        return policy, device, required

    if policy == STAGED_PREFLIGHT_POLICY:
        required = _exact_int(
            next(iter(required_values)),
            "staged required_free_bytes",
        )
        if required != 0:
            raise ValueError("staged runtime cannot carry a resident-memory threshold")
        for item in frozen:
            if item.embedding_released_before_qwen_load is not True:
                raise ValueError("staged runtime must release BGE before Qwen")
            if item.observed_free_bytes is not None or item.observed_total_bytes is not None:
                raise ValueError("staged runtime cannot claim a CUDA memory observation")
        return policy, device, required

    raise ValueError("unsupported matched runtime residency policy")


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalMatchedRuntimeSuiteReceipt:
    """Self-contained runtime attestation layered over the matched phases."""

    sample_id: str
    runtime_binding_sha256: str
    runtime_binding_certified: bool
    residency_policy: str
    residency_device: str
    required_free_bytes: int
    runtime_result_receipt_sha256s: tuple[str, ...]
    preflight_observations: tuple[ResidencyPreflightObservation, ...]
    matched_suite: DiffuseLongMemEvalMatchedSuiteReceipt
    format: str = DIFFUSE_MATCHED_RUNTIME_SUITE_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != DIFFUSE_MATCHED_RUNTIME_SUITE_FORMAT:
            raise ValueError("unsupported matched runtime suite format")
        if not str(self.sample_id).strip():
            raise ValueError("sample_id must be non-empty")
        _digest(self.runtime_binding_sha256, "runtime_binding_sha256")
        if self.runtime_binding_certified is not True:
            raise ValueError("runtime_binding_certified must be true")
        receipts = tuple(self.runtime_result_receipt_sha256s)
        if len(receipts) != len(MATCHED_BOUNDARY_MODES):
            raise ValueError("runtime receipts must cover every matched arm")
        for index, value in enumerate(receipts):
            _digest(value, f"runtime_result_receipt_sha256s[{index}]")
        object.__setattr__(self, "runtime_result_receipt_sha256s", receipts)
        observations = tuple(self.preflight_observations)
        policy, device, required = _validate_preflight_observations(observations)
        if self.residency_policy != policy or self.residency_device != device:
            raise ValueError("aggregate residency identity changed")
        if self.required_free_bytes != required:
            raise ValueError("aggregate residency threshold changed")
        object.__setattr__(self, "preflight_observations", observations)
        if type(self.matched_suite) is not DiffuseLongMemEvalMatchedSuiteReceipt:
            raise TypeError("matched_suite must be the exact matched-phase receipt")
        matched_expected = identity_sha256(
            self.matched_suite.identity_payload(include_receipt=False)
        )
        if self.matched_suite.receipt_sha256 != matched_expected:
            raise ValueError("matched-phase suite receipt does not match")
        if self.matched_suite.sample_id != self.sample_id:
            raise ValueError("runtime suite and matched phases name different samples")
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("matched runtime suite receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "sample_id": self.sample_id,
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "runtime_binding_certified": self.runtime_binding_certified,
            "residency_policy": self.residency_policy,
            "residency_device": self.residency_device,
            "required_free_bytes": self.required_free_bytes,
            "runtime_result_receipt_sha256s": list(
                self.runtime_result_receipt_sha256s
            ),
            "preflight_observations": [
                {
                    **_preflight_payload(item),
                    "receipt_sha256": item.receipt_sha256,
                }
                for item in self.preflight_observations
            ],
            "matched_suite_receipt_sha256": self.matched_suite.receipt_sha256,
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def validate_matched_diffuse_runtime_results(
    results: Sequence[DiffuseLongMemEvalRuntimeResult],
) -> DiffuseLongMemEvalMatchedRuntimeSuiteReceipt:
    """Validate runtime attestations, then delegate phase matching."""

    supplied = tuple(results)
    if len(supplied) != len(MATCHED_BOUNDARY_MODES):
        raise ValueError("matched runtime suite requires exactly three results")
    if any(type(item) is not DiffuseLongMemEvalRuntimeResult for item in supplied):
        raise TypeError("matched runtime suite requires exact runtime results")
    for item in supplied:
        if item.format != DIFFUSE_RUNTIME_RESULT_FORMAT:
            raise ValueError("runtime result has an unsupported format")
        if item.runtime_binding_certified is not True:
            raise ValueError("matched runtime result is not certified")
        expected = identity_sha256(item.identity_payload(include_receipt=False))
        if item.receipt_sha256 != expected:
            raise ValueError("runtime result receipt does not match")

    matched = validate_matched_diffuse_retrieval_phases(
        tuple(item.phase for item in supplied)
    )
    by_mode = {
        item.phase.arm.compilation.boundary_mode: item for item in supplied
    }
    if set(by_mode) != set(MATCHED_BOUNDARY_MODES) or len(by_mode) != len(supplied):
        raise ValueError("runtime suite requires each canonical arm exactly once")
    ordered = tuple(by_mode[mode] for mode in MATCHED_BOUNDARY_MODES)
    if tuple(item.phase.receipt_sha256 for item in ordered) != (
        matched.retrieval_phase_receipt_sha256s
    ):
        raise ValueError("matched receipt does not bind the supplied runtime phases")
    bindings = {item.runtime_binding_sha256 for item in ordered}
    if len(bindings) != 1:
        raise ValueError("matched runtime arms changed runtime binding")
    observations = tuple(item.residency_preflight for item in ordered)
    policy, device, required = _validate_preflight_observations(observations)
    return DiffuseLongMemEvalMatchedRuntimeSuiteReceipt(
        sample_id=matched.sample_id,
        runtime_binding_sha256=next(iter(bindings)),
        runtime_binding_certified=True,
        residency_policy=policy,
        residency_device=device,
        required_free_bytes=required,
        runtime_result_receipt_sha256s=tuple(
            item.receipt_sha256 for item in ordered
        ),
        preflight_observations=observations,
        matched_suite=matched,
    )


__all__ = [
    "DIFFUSE_MATCHED_RUNTIME_SUITE_FORMAT",
    "MINIMUM_RESIDENT_FREE_BYTES",
    "RESIDENT_PREFLIGHT_POLICY",
    "STAGED_PREFLIGHT_POLICY",
    "DiffuseLongMemEvalMatchedRuntimeSuiteReceipt",
    "validate_matched_diffuse_runtime_results",
]
