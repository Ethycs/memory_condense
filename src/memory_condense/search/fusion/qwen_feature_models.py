"""Sealed, text-free receipts for resident Qwen atom-feature operations.

These identities attest the exact operation inputs and bounded execution
contract.  They deliberately do not claim a cryptographic digest of the
transient ``[N, D]`` feature tensor or verified model behavior.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from memory_condense.domain._discourse_identity import (
    _nonempty,
    _sha256,
    _strict_int,
    normalize_fields,
)
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.fusion.models import (
    AuthoritativeHyperedge,
    FusionAtomRef,
    FusionCaps,
)


_MACHINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,127}$")
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")
_EXECUTION_DTYPES = frozenset(
    {"torch.float16", "torch.bfloat16", "torch.float32"}
)


def _positive_int(value: object, label: str) -> int:
    normalized = _strict_int(value, label)
    if normalized < 1:
        raise ValueError(f"{label} must be positive")
    return normalized


def _nonnegative_int(value: object, label: str) -> int:
    normalized = _strict_int(value, label)
    if normalized < 0:
        raise ValueError(f"{label} must be non-negative")
    return normalized


def _nonnegative_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be a finite non-negative float")
    return value


def _machine_id(value: object, label: str) -> str:
    normalized = _nonempty(value, label)
    if not _MACHINE_ID.fullmatch(normalized):
        raise ValueError(f"{label} must be a bounded machine identifier")
    return normalized


def _cuda_device(value: object, label: str) -> str:
    normalized = _nonempty(value, label).casefold()
    if not _CUDA_DEVICE.fullmatch(normalized):
        raise ValueError(f"{label} must be a canonical indexed CUDA device")
    return normalized


def _execution_dtype(value: object, label: str) -> str:
    normalized = _nonempty(value, label)
    if normalized not in _EXECUTION_DTYPES:
        raise ValueError(f"{label} must be a supported CUDA execution dtype")
    return normalized


def _exact_values(
    values: object,
    expected_type: type,
    label: str,
) -> tuple[object, ...]:
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence") from exc
    if any(type(value) is not expected_type for value in normalized):
        raise TypeError(f"{label} must contain exact {expected_type.__name__} values")
    return normalized


@dataclass(frozen=True, slots=True)
class QwenAtomFeatureCaps(SealedIdentity):
    """Hard row, batching, workspace, and invariance bounds for Qwen."""

    _SEAL_FIELD = "caps_sha256"
    _SEAL_MISMATCH = "Qwen atom feature caps SHA-256 does not match its contents"

    max_row_tokens: int = 128
    max_query_tail_tokens: int = 64
    max_rows_per_forward: int = 4
    max_workspace_tokens: int = 512
    max_evidence_characters: int = 4096
    max_query_characters: int = 2048
    batch_invariance_atol: float = 1e-3
    batch_invariance_rtol: float = 1e-3
    caps_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            max_row_tokens=_positive_int,
            max_query_tail_tokens=_positive_int,
            max_rows_per_forward=_positive_int,
            max_workspace_tokens=_positive_int,
            max_evidence_characters=_positive_int,
            max_query_characters=_positive_int,
            batch_invariance_atol=_nonnegative_float,
            batch_invariance_rtol=_nonnegative_float,
        )
        if self.max_query_tail_tokens >= self.max_row_tokens:
            raise ValueError("max_query_tail_tokens must be below max_row_tokens")
        if self.max_row_tokens > self.max_workspace_tokens:
            raise ValueError("one maximum-length row must fit the Qwen workspace")
        self._seal()


@dataclass(frozen=True, slots=True)
class QwenAtomRowReceipt(SealedIdentity):
    """Text-free token-count receipt for one exact packet atom row."""

    _SEAL_FIELD = "row_receipt_sha256"
    _SEAL_MISMATCH = "Qwen atom row receipt SHA-256 does not match its contents"

    row_index: int
    atom_id: str
    atom_identity_sha256: str
    span_identity_sha256: str
    quote_sha256: str
    evidence_character_count: int
    query_character_count: int
    prefix_tokens: int
    evidence_tokens_observed: int
    evidence_tokens_admitted: int
    query_tail_tokens: int
    total_row_tokens: int
    readout_end_index: int
    evidence_truncated: bool
    row_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            row_index=_nonnegative_int,
            atom_id=_nonempty,
            atom_identity_sha256=_sha256,
            span_identity_sha256=_sha256,
            quote_sha256=_sha256,
            evidence_character_count=_positive_int,
            query_character_count=_positive_int,
            prefix_tokens=_positive_int,
            evidence_tokens_observed=_positive_int,
            evidence_tokens_admitted=_positive_int,
            query_tail_tokens=_positive_int,
            total_row_tokens=_positive_int,
            readout_end_index=_nonnegative_int,
        )
        if type(self.evidence_truncated) is not bool:
            raise TypeError("evidence_truncated must be a boolean")
        if self.evidence_tokens_observed not in {
            self.evidence_tokens_admitted,
            self.evidence_tokens_admitted + 1,
        }:
            raise ValueError("bounded evidence-token observation is incoherent")
        if self.evidence_truncated != (
            self.evidence_tokens_observed == self.evidence_tokens_admitted + 1
        ):
            raise ValueError("evidence_truncated disagrees with bounded observation")
        if self.readout_end_index != self.total_row_tokens - 1:
            raise ValueError("readout_end_index must identify the final row token")
        if self.total_row_tokens != (
            self.prefix_tokens
            + self.evidence_tokens_admitted
            + self.query_tail_tokens
        ):
            raise ValueError("row token counts do not add up")
        self._seal()


@dataclass(frozen=True, slots=True)
class QwenAtomBatchReceipt(SealedIdentity):
    """Closed padded-workspace arithmetic for one exhaustive Qwen forward."""

    _SEAL_FIELD = "batch_receipt_sha256"
    _SEAL_MISMATCH = "Qwen atom batch receipt SHA-256 does not match its contents"

    batch_index: int
    start_row: int
    row_count: int
    padded_width: int
    padded_workspace_tokens: int
    batch_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            batch_index=_nonnegative_int,
            start_row=_nonnegative_int,
            row_count=_positive_int,
            padded_width=_positive_int,
            padded_workspace_tokens=_positive_int,
        )
        if self.padded_workspace_tokens != self.row_count * self.padded_width:
            raise ValueError("batch workspace must equal row_count * padded_width")
        self._seal()


@dataclass(frozen=True, slots=True)
class QwenAtomFeatureProviderReceipt(SealedIdentity):
    """Owned Qwen runtime declaration for query-conditioned atom features."""

    _SEAL_FIELD = "provider_receipt_sha256"
    _SEAL_MISMATCH = "Qwen atom feature provider SHA-256 does not match its contents"

    implementation_sha256: str
    model_id: str
    model_revision: str
    checkpoint_sha256: str
    verified_files_sha256: str
    tokenizer_identity_sha256: str
    retained_layers: int
    output_layer: int
    hidden_dim: int
    device: str
    execution_dtype: str
    provider_id: str = "qwen3_prefix.query_readout_last.v1"
    pooling: str = "last_token"
    prompt_template_sha256: str = ""
    truncation_rule: str = "evidence_prefix_only"
    checkpoint_status: str = "checkpoint_files_verified"
    model_behavior_verified: bool = False
    exclusive_synchronous_ownership_required: bool = True
    exclusive_synchronous_ownership_verified: bool = False
    general_concurrency_safe: bool = False
    legacy_hook_paths_serialized: bool = False
    loaded_parameter_content_attested: bool = False
    loaded_tokenizer_content_attested: bool = False
    loaded_module_runtime_constants_attested: bool = False
    tokenizer_behavior_verified: bool = False
    supported_structural_mutation_checks_attested: bool = True
    supported_mutation_scope: str = "structure_parameter_metadata_bounded_scalar_fields"
    execution_gate_scope: str = "fusion_provider_only"
    provider_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            implementation_sha256=_sha256,
            model_id=_machine_id,
            model_revision=_machine_id,
            checkpoint_sha256=_sha256,
            verified_files_sha256=_sha256,
            tokenizer_identity_sha256=_sha256,
            retained_layers=_positive_int,
            output_layer=_nonnegative_int,
            hidden_dim=_positive_int,
            device=_cuda_device,
            execution_dtype=_execution_dtype,
            provider_id=_machine_id,
            prompt_template_sha256=_sha256,
        )
        if self.output_layer >= self.retained_layers:
            raise ValueError("Qwen output layer lies outside the retained prefix")
        expected = {
            "provider_id": "qwen3_prefix.query_readout_last.v1",
            "pooling": "last_token",
            "truncation_rule": "evidence_prefix_only",
            "checkpoint_status": "checkpoint_files_verified",
            "model_behavior_verified": False,
            "exclusive_synchronous_ownership_required": True,
            "exclusive_synchronous_ownership_verified": False,
            "general_concurrency_safe": False,
            "legacy_hook_paths_serialized": False,
            "loaded_parameter_content_attested": False,
            "loaded_tokenizer_content_attested": False,
            "loaded_module_runtime_constants_attested": False,
            "tokenizer_behavior_verified": False,
            "supported_structural_mutation_checks_attested": True,
            "supported_mutation_scope": "structure_parameter_metadata_bounded_scalar_fields",
            "execution_gate_scope": "fusion_provider_only",
        }
        boolean_fields = (
            "model_behavior_verified",
            "exclusive_synchronous_ownership_required",
            "exclusive_synchronous_ownership_verified",
            "general_concurrency_safe",
            "legacy_hook_paths_serialized",
            "loaded_parameter_content_attested",
            "loaded_tokenizer_content_attested",
            "loaded_module_runtime_constants_attested",
            "tokenizer_behavior_verified",
            "supported_structural_mutation_checks_attested",
        )
        if any(type(getattr(self, name)) is not bool for name in boolean_fields):
            raise TypeError("Qwen provider claim flags must be booleans")
        _machine_id(self.supported_mutation_scope, "supported_mutation_scope")
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("Qwen provider receipt overstates its exact claim boundary")
        self._seal()


@dataclass(frozen=True, slots=True)
class QwenAtomFeatureOperationReceipt(SealedIdentity):
    """Tensor-free receipt for one exhaustive resident feature operation."""

    _SEAL_FIELD = "operation_sha256"
    _SEAL_MISMATCH = "Qwen atom feature operation SHA-256 does not match its contents"

    packet_receipt_sha256: str
    closure_plan_sha256: str
    query_program_sha256: str
    query_sha256: str
    closure_policy_sha256: str
    snapshot_sha256: str
    caps: FusionCaps
    feature_caps: QwenAtomFeatureCaps
    provider: QwenAtomFeatureProviderReceipt
    atoms: tuple[FusionAtomRef, ...]
    hyperedges: tuple[AuthoritativeHyperedge, ...]
    rows: tuple[QwenAtomRowReceipt, ...]
    batches: tuple[QwenAtomBatchReceipt, ...]
    feature_shape: tuple[int, int]
    feature_device: str
    feature_execution_dtype: str
    qwen_forward_count: int
    primary_qwen_forward_count: int
    batch_invariance_forward_count: int
    max_observed_padded_workspace_tokens: int
    runtime_batch_invariance_attested: bool = False
    operation_format: str = "qwen_atom_feature_execution_smoke_v1"
    operation_kind: str = "feature_execution_smoke"
    qwen_executed: bool = True
    router_executed: bool = False
    matched_pair_produced: bool = False
    performance_attested: bool = False
    feature_tensor_sha256: None = None
    steered_tensor_produced: bool = False
    steered_tensor_sha256: None = None
    feature_tensor_content_attested: bool = False
    operation_inputs_attested: bool = True
    retrieval_route_attested: bool = False
    retained_request_tensor_bytes: int = 0
    operation_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            packet_receipt_sha256=_sha256,
            closure_plan_sha256=_sha256,
            query_program_sha256=_sha256,
            query_sha256=_sha256,
            closure_policy_sha256=_sha256,
            snapshot_sha256=_sha256,
            feature_device=_cuda_device,
            feature_execution_dtype=_execution_dtype,
            qwen_forward_count=_positive_int,
            primary_qwen_forward_count=_positive_int,
            batch_invariance_forward_count=_nonnegative_int,
            max_observed_padded_workspace_tokens=_positive_int,
            retained_request_tensor_bytes=_nonnegative_int,
        )
        if type(self.caps) is not FusionCaps:
            raise TypeError("caps must be an exact FusionCaps")
        if type(self.feature_caps) is not QwenAtomFeatureCaps:
            raise TypeError("feature_caps must be an exact QwenAtomFeatureCaps")
        if type(self.provider) is not QwenAtomFeatureProviderReceipt:
            raise TypeError("provider must be an exact QwenAtomFeatureProviderReceipt")
        self.caps._seal()
        self.feature_caps._seal()
        self.provider._seal()
        object.__setattr__(
            self,
            "atoms",
            _exact_values(self.atoms, FusionAtomRef, "atoms"),
        )
        object.__setattr__(
            self,
            "hyperedges",
            _exact_values(self.hyperedges, AuthoritativeHyperedge, "hyperedges"),
        )
        object.__setattr__(
            self,
            "rows",
            _exact_values(self.rows, QwenAtomRowReceipt, "rows"),
        )
        object.__setattr__(
            self,
            "batches",
            _exact_values(self.batches, QwenAtomBatchReceipt, "batches"),
        )
        for value in (*self.atoms, *self.hyperedges, *self.rows, *self.batches):
            value._seal()
        try:
            shape = tuple(self.feature_shape)
        except TypeError as exc:
            raise TypeError("feature_shape must be [N, D]") from exc
        if len(shape) != 2:
            raise ValueError("feature_shape must be [N, D]")
        shape = tuple(_positive_int(value, "feature_shape") for value in shape)
        object.__setattr__(self, "feature_shape", shape)
        self._validate_claim_boundary()
        self._validate_atom_rows()
        self._validate_topology()
        self._validate_execution()
        self._seal()

    def _validate_claim_boundary(self) -> None:
        if self.feature_tensor_sha256 is not None or self.steered_tensor_sha256 is not None:
            raise ValueError("tranche-A smoke cannot claim full request tensor hashes")
        flags = {
            "qwen_executed": True,
            "router_executed": False,
            "matched_pair_produced": False,
            "performance_attested": False,
            "steered_tensor_produced": False,
            "feature_tensor_content_attested": False,
            "operation_inputs_attested": True,
            "retrieval_route_attested": False,
            "runtime_batch_invariance_attested": False,
        }
        if any(type(getattr(self, name)) is not bool for name in flags):
            raise TypeError("operation attestation flags must be booleans")
        if any(getattr(self, name) != expected for name, expected in flags.items()):
            raise ValueError("resident operation overstates its claim boundary")
        if (
            self.operation_format != "qwen_atom_feature_execution_smoke_v1"
            or self.operation_kind != "feature_execution_smoke"
        ):
            raise ValueError("tranche-A operation must be an execution smoke")
        if self.retained_request_tensor_bytes != 0:
            raise ValueError("returned operation receipts cannot retain request tensors")

    def _validate_atom_rows(self) -> None:
        atom_ids = tuple(item.atom_id for item in self.atoms)
        if not atom_ids or len(atom_ids) != len(set(atom_ids)):
            raise ValueError("operation atoms must be a non-empty unique sequence")
        if len(atom_ids) > self.caps.max_atoms:
            raise ValueError("operation atom count exceeds FusionCaps.max_atoms")
        if tuple(item.row_index for item in self.rows) != tuple(range(len(atom_ids))):
            raise ValueError("row indices must exhaust packet order from zero")
        if tuple(item.atom_id for item in self.rows) != atom_ids:
            raise ValueError("row receipts must preserve exact packet atom order")
        atom_by_id = {item.atom_id: item for item in self.atoms}
        if any(
            row.atom_identity_sha256 != atom_by_id[row.atom_id].atom_identity_sha256
            or row.span_identity_sha256 != atom_by_id[row.atom_id].span_identity_sha256
            or row.quote_sha256 != atom_by_id[row.atom_id].quote_sha256
            for row in self.rows
        ):
            raise ValueError("row receipts disagree with their exact atom identities")
        if any(row.query_tail_tokens > self.feature_caps.max_query_tail_tokens for row in self.rows):
            raise ValueError("row query tail exceeds Qwen feature caps")
        if any(row.total_row_tokens > self.feature_caps.max_row_tokens for row in self.rows):
            raise ValueError("row length exceeds Qwen feature caps")
        if any(
            row.evidence_character_count > self.feature_caps.max_evidence_characters
            for row in self.rows
        ):
            raise ValueError("row evidence characters exceed Qwen feature caps")
        if any(
            row.query_character_count > self.feature_caps.max_query_characters
            for row in self.rows
        ):
            raise ValueError("row query characters exceed Qwen feature caps")
        if len({row.query_character_count for row in self.rows}) != 1:
            raise ValueError("all atom rows must bind the same exact query length")
        if len({row.prefix_tokens for row in self.rows}) != 1:
            raise ValueError("all atom rows must bind one exact evidence prefix")
        if len({row.query_tail_tokens for row in self.rows}) != 1:
            raise ValueError("all atom rows must bind one exact query/readout tail")
        for row in self.rows:
            evidence_budget = (
                self.feature_caps.max_row_tokens
                - row.prefix_tokens
                - row.query_tail_tokens
            )
            if evidence_budget < 1:
                raise ValueError("row caps leave no evidence-token budget")
            if row.evidence_tokens_admitted != min(
                row.evidence_tokens_observed,
                evidence_budget,
            ):
                raise ValueError("row did not apply exact prefix-only evidence truncation")
        if any(row.prefix_tokens + row.query_tail_tokens >= row.total_row_tokens for row in self.rows):
            raise ValueError("every row must admit at least one evidence token")

    def _validate_topology(self) -> None:
        known_atoms = {item.atom_id for item in self.atoms}
        if len(self.hyperedges) > self.caps.max_hyperedges:
            raise ValueError("operation hyperedges exceed FusionCaps.max_hyperedges")
        if len({item.bundle_id for item in self.hyperedges}) != len(self.hyperedges):
            raise ValueError("operation hyperedge bundle IDs must be unique")
        if any(atom_id not in known_atoms for edge in self.hyperedges for atom_id in edge.atom_ids):
            raise ValueError("operation hyperedge references an unknown atom")
        topology_links = sum(
            len(edge.atom_ids) * (len(edge.atom_ids) - 1) // 2
            for edge in self.hyperedges
        )
        if topology_links > self.caps.max_topology_links:
            raise ValueError("operation topology exceeds FusionCaps.max_topology_links")

    def _validate_execution(self) -> None:
        if self.feature_shape != (len(self.atoms), self.provider.hidden_dim):
            raise ValueError("feature_shape disagrees with exact atoms/provider width")
        if self.feature_shape[1] > self.caps.max_hidden_dim:
            raise ValueError("feature width exceeds FusionCaps.max_hidden_dim")
        if self.feature_device != self.provider.device:
            raise ValueError("feature device disagrees with provider device")
        if self.feature_execution_dtype != self.provider.execution_dtype:
            raise ValueError("feature dtype disagrees with provider dtype")
        if self.batch_invariance_forward_count != 0:
            raise ValueError("ordinary feature smoke cannot run invariance diagnostics")
        if self.qwen_forward_count != (
            self.primary_qwen_forward_count
        ):
            raise ValueError("total Qwen forwards must equal primary forwards")
        minimum_forwards = math.ceil(
            len(self.rows) / self.feature_caps.max_rows_per_forward
        )
        if self.primary_qwen_forward_count < minimum_forwards:
            raise ValueError("primary Qwen forwards cannot cover all bounded rows")
        if len(self.batches) != self.primary_qwen_forward_count:
            raise ValueError("batch receipts must cover every primary Qwen forward")
        if tuple(item.batch_index for item in self.batches) != tuple(
            range(len(self.batches))
        ):
            raise ValueError("batch indices must be contiguous from zero")
        expected_start = 0
        for batch in self.batches:
            if batch.start_row != expected_start:
                raise ValueError("batch receipts must exhaust rows without gaps")
            if batch.row_count > self.feature_caps.max_rows_per_forward:
                raise ValueError("batch row count exceeds Qwen feature caps")
            if batch.padded_width > self.feature_caps.max_row_tokens:
                raise ValueError("batch padded width exceeds Qwen feature caps")
            if batch.padded_workspace_tokens > self.feature_caps.max_workspace_tokens:
                raise ValueError("batch padded workspace exceeds Qwen feature caps")
            covered_rows = self.rows[batch.start_row : batch.start_row + batch.row_count]
            if len(covered_rows) != batch.row_count:
                raise ValueError("batch receipt extends beyond the row sequence")
            if batch.padded_width != max(row.total_row_tokens for row in covered_rows):
                raise ValueError("batch padded width disagrees with its rows")
            expected_start += batch.row_count
        if expected_start != len(self.rows):
            raise ValueError("batch receipts must process every row exactly once")
        observed_max = max(batch.padded_workspace_tokens for batch in self.batches)
        if self.max_observed_padded_workspace_tokens != observed_max:
            raise ValueError("maximum observed workspace disagrees with batches")
        if (
            self.max_observed_padded_workspace_tokens
            > self.feature_caps.max_workspace_tokens
        ):
            raise ValueError("observed Qwen workspace exceeds feature caps")
        if not all(
            math.isfinite(value)
            for value in (
                self.feature_caps.batch_invariance_atol,
                self.feature_caps.batch_invariance_rtol,
            )
        ):  # pragma: no cover - feature caps already guard this
            raise ValueError("batch invariance tolerances must be finite")


__all__ = [
    "QwenAtomBatchReceipt",
    "QwenAtomFeatureCaps",
    "QwenAtomFeatureOperationReceipt",
    "QwenAtomFeatureProviderReceipt",
    "QwenAtomRowReceipt",
]
