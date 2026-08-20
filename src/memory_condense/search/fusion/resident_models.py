"""Text-free sealed models for the GPU-resident matched fusion path.

These models are intentionally distinct from the CPU-hashed reference
``EvidenceFusionPlan``.  They bind the exact tranche-A feature sub-operation,
the bounded route matrices, and extractive plans without retaining request
tensors, route-matrix values, token IDs, or source/query text.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Literal, Sequence

from memory_condense.domain._discourse_identity import (
    _nonempty,
    _sha256,
    _strict_int,
    identity_sha256,
    normalize_fields,
)
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.fusion.models import (
    AuthoritativeHyperedge,
    ExtractiveGroup,
    FusionAtomRef,
    FusionCaps,
    LatentMembership,
    RouterArchitectureReceipt,
    RouterStateReceipt,
)
from memory_condense.search.fusion.qwen_feature_models import (
    QwenAtomFeatureOperationReceipt,
)


ResidentFusionMode = Literal["topology_only", "latent_router"]
_ROUTER_DTYPES = frozenset(
    {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
)


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact integer")
    normalized = _strict_int(value, label)
    if normalized < 1:
        raise ValueError(f"{label} must be positive")
    return normalized


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact integer")
    normalized = _strict_int(value, label)
    if normalized < 0:
        raise ValueError(f"{label} must be non-negative")
    return normalized


def _cuda_device(value: str, label: str) -> str:
    normalized = _nonempty(value, label)
    prefix, separator, index = normalized.partition(":")
    if prefix != "cuda" or separator != ":" or not index.isdigit():
        raise ValueError(f"{label} must be an indexed CUDA device")
    return f"cuda:{int(index)}"


def _router_dtype(value: str, label: str) -> str:
    normalized = _nonempty(value, label)
    if normalized not in _ROUTER_DTYPES:
        raise ValueError(f"{label} is not a supported router dtype")
    return normalized


def _optional_sha256(value: str | None, label: str) -> str | None:
    return None if value is None else _sha256(value, label)


def _shape2(value: object, label: str) -> tuple[int, int]:
    try:
        shape = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be a two-dimensional shape") from exc
    if len(shape) != 2:
        raise ValueError(f"{label} must be a two-dimensional shape")
    return (
        _positive_int(shape[0], label),
        _positive_int(shape[1], label),
    )


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
    for value in normalized:
        value._seal()  # type: ignore[attr-defined]
    return normalized


def _require_plain_identity_tree(value: object, label: str) -> None:
    """Reject scalar subclasses or mutable containers in returned receipts."""

    value_type = type(value)
    if value is None or value_type in {str, int, float, bool}:
        return
    if value_type is tuple:
        for index, item in enumerate(value):
            _require_plain_identity_tree(item, f"{label}[{index}]")
        return
    if is_dataclass(value):
        for item in fields(value):
            _require_plain_identity_tree(
                getattr(value, item.name),
                f"{label}.{item.name}",
            )
        return
    raise TypeError(f"{label} contains an unsupported identity value")


def resident_matched_input_sha256(
    *,
    feature_suboperation_sha256: str,
    router_runtime_sha256: str,
    implementation_sha256: str,
) -> str:
    """Derive the only valid matched-input identity for a resident pair."""

    return identity_sha256(
        {
            "format": "memory-condense-qwen-resident-matched-input-v1",
            "feature_suboperation_sha256": _sha256(
                feature_suboperation_sha256,
                "feature_suboperation_sha256",
            ),
            "router_runtime_sha256": _sha256(
                router_runtime_sha256,
                "router_runtime_sha256",
            ),
            "implementation_sha256": _sha256(
                implementation_sha256,
                "implementation_sha256",
            ),
        }
    )


def resident_values_sha256(kind: str, values: Sequence[SealedIdentity]) -> str:
    """Bind an ordered sequence of already-sealed, text-free identities."""

    if not kind or any(character.isspace() for character in kind):
        raise ValueError("resident identity kind must be a machine token")
    for value in values:
        if not isinstance(value, SealedIdentity):
            raise TypeError("resident identity sequences require sealed values")
        value._seal()
    return identity_sha256(
        {
            "format": "memory-condense-resident-sequence-v1",
            "kind": kind,
            "values": [value.identity_payload() for value in values],
        }
    )


def resident_atom_order_sha256(atom_order: Sequence[str]) -> str:
    normalized = tuple(_nonempty(value, "atom_order") for value in atom_order)
    return identity_sha256(
        {
            "format": "memory-condense-resident-atom-order-v1",
            "atom_ids": list(normalized),
        }
    )


@dataclass(frozen=True, slots=True)
class ResidentRouterRuntimeReceipt(SealedIdentity):
    """One sealed router's resident device/dtype and bounded runtime identity."""

    _SEAL_FIELD = "runtime_sha256"
    _SEAL_MISMATCH = "resident router runtime SHA-256 does not match its contents"

    architecture: RouterArchitectureReceipt
    state: RouterStateReceipt
    device: str
    execution_dtype: str
    max_atoms: int
    max_hidden_dim: int
    max_route_cells: int
    sealed_for_inference: bool = True
    runtime_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.architecture) is not RouterArchitectureReceipt:
            raise TypeError("architecture must be an exact RouterArchitectureReceipt")
        if type(self.state) is not RouterStateReceipt:
            raise TypeError("state must be an exact RouterStateReceipt")
        self.architecture._seal()
        self.state._seal()
        normalize_fields(
            self,
            device=_cuda_device,
            execution_dtype=_router_dtype,
            max_atoms=_positive_int,
            max_hidden_dim=_positive_int,
            max_route_cells=_positive_int,
        )
        if type(self.sealed_for_inference) is not bool or not self.sealed_for_inference:
            raise ValueError("resident router runtime must be sealed for inference")
        if self.state.parameter_count != self.architecture.parameter_count:
            raise ValueError("router state parameter count disagrees with architecture")
        if self.state.parameter_dtypes != (self.execution_dtype,):
            raise ValueError("resident router state must use one execution dtype")
        if self.architecture.hidden_dim > self.max_hidden_dim:
            raise ValueError("router hidden width exceeds its resident bound")
        if self.architecture.num_latents * self.max_atoms > self.max_route_cells:
            raise ValueError("router resident bounds exceed max_route_cells")
        _require_plain_identity_tree(self, "resident router runtime")
        self._seal()


@dataclass(frozen=True, slots=True)
class ResidentEvidenceFusionPlan(SealedIdentity):
    """One resident extractive arm over the exact feature sub-operation."""

    _SEAL_FIELD = "plan_sha256"
    _SEAL_MISMATCH = "resident fusion plan SHA-256 does not match its contents"

    mode: ResidentFusionMode
    feature_suboperation_sha256: str
    matched_input_sha256: str
    caps: FusionCaps
    atoms: tuple[FusionAtomRef, ...]
    hyperedges: tuple[AuthoritativeHyperedge, ...]
    memberships: tuple[LatentMembership, ...]
    groups: tuple[ExtractiveGroup, ...]
    atom_order: tuple[str, ...]
    router_runtime: ResidentRouterRuntimeReceipt | None = None
    extraction_matrix_sha256: str | None = None
    reinjection_matrix_sha256: str | None = None
    extraction_shape: tuple[int, int] | tuple[()] = ()
    reinjection_shape: tuple[int, int] | tuple[()] = ()
    plan_retained_request_tensor_bytes: int = 0
    plan_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in {
            "topology_only",
            "latent_router",
        }:
            raise ValueError("resident fusion mode is unsupported")
        normalize_fields(
            self,
            feature_suboperation_sha256=_sha256,
            matched_input_sha256=_sha256,
            plan_retained_request_tensor_bytes=_nonnegative_int,
        )
        if type(self.caps) is not FusionCaps:
            raise TypeError("caps must be an exact FusionCaps")
        self.caps._seal()
        object.__setattr__(self, "atoms", _exact_values(self.atoms, FusionAtomRef, "atoms"))
        object.__setattr__(
            self,
            "hyperedges",
            _exact_values(self.hyperedges, AuthoritativeHyperedge, "hyperedges"),
        )
        object.__setattr__(
            self,
            "memberships",
            _exact_values(self.memberships, LatentMembership, "memberships"),
        )
        object.__setattr__(
            self,
            "groups",
            _exact_values(self.groups, ExtractiveGroup, "groups"),
        )
        try:
            atom_order = tuple(self.atom_order)
        except TypeError as exc:
            raise TypeError("atom_order must be a sequence") from exc
        if any(type(value) is not str for value in atom_order):
            raise TypeError("atom_order must contain exact strings")
        object.__setattr__(self, "atom_order", atom_order)
        for shape_name in ("extraction_shape", "reinjection_shape"):
            try:
                shape = tuple(getattr(self, shape_name))
            except TypeError as exc:
                raise TypeError(f"{shape_name} must be a sequence") from exc
            object.__setattr__(self, shape_name, shape)
        self._validate_atom_set()
        self._validate_topology()
        self._validate_mode()
        if self.plan_retained_request_tensor_bytes != 0:
            raise ValueError("resident plans cannot retain request tensors")
        _require_plain_identity_tree(self, "resident fusion plan")
        self._seal()

    def _validate_atom_set(self) -> None:
        atom_ids = tuple(item.atom_id for item in self.atoms)
        if not atom_ids or len(atom_ids) != len(set(atom_ids)):
            raise ValueError("resident plan atoms must be non-empty and unique")
        if len(atom_ids) > self.caps.max_atoms:
            raise ValueError("resident plan atom count exceeds max_atoms")
        if self.atom_order != tuple(
            atom_id for group in self.groups for atom_id in group.atom_ids
        ):
            raise ValueError("resident groups must concatenate to atom_order")
        if len(self.atom_order) != len(atom_ids) or set(self.atom_order) != set(atom_ids):
            raise ValueError("resident atom_order must preserve the exact atom set")
        if tuple(group.group_index for group in self.groups) != tuple(
            range(len(self.groups))
        ):
            raise ValueError("resident group indices must be contiguous from zero")
        if len(self.groups) > self.caps.max_groups:
            raise ValueError("resident group count exceeds max_groups")
        if any(len(group.atom_ids) > self.caps.max_group_atoms for group in self.groups):
            raise ValueError("resident group exceeds max_group_atoms")

    def _validate_topology(self) -> None:
        known_atoms = {item.atom_id for item in self.atoms}
        if len(self.hyperedges) > self.caps.max_hyperedges:
            raise ValueError("resident hyperedge count exceeds max_hyperedges")
        if len({item.bundle_id for item in self.hyperedges}) != len(self.hyperedges):
            raise ValueError("resident hyperedge bundle IDs must be unique")
        if any(
            atom_id not in known_atoms
            for edge in self.hyperedges
            for atom_id in edge.atom_ids
        ):
            raise ValueError("resident hyperedge references an unknown atom")
        links = sum(
            len(edge.atom_ids) * (len(edge.atom_ids) - 1) // 2
            for edge in self.hyperedges
        )
        if links > self.caps.max_topology_links:
            raise ValueError("resident topology exceeds max_topology_links")

    def _validate_mode(self) -> None:
        if self.mode == "topology_only":
            if self.memberships:
                raise ValueError("topology-only resident plans cannot carry memberships")
            if any(group.latent_index is not None for group in self.groups):
                raise ValueError("topology-only resident groups cannot name latent slots")
            if self.router_runtime is not None:
                raise ValueError("topology-only resident plans cannot bind a router")
            if self.extraction_matrix_sha256 is not None or self.reinjection_matrix_sha256 is not None:
                raise ValueError("topology-only resident plans cannot bind route hashes")
            if self.extraction_shape or self.reinjection_shape:
                raise ValueError("topology-only resident plans cannot bind route shapes")
            return
        if type(self.router_runtime) is not ResidentRouterRuntimeReceipt:
            raise TypeError("latent resident plans require ResidentRouterRuntimeReceipt")
        self.router_runtime._seal()
        object.__setattr__(
            self,
            "extraction_matrix_sha256",
            _optional_sha256(self.extraction_matrix_sha256, "extraction_matrix_sha256"),
        )
        object.__setattr__(
            self,
            "reinjection_matrix_sha256",
            _optional_sha256(self.reinjection_matrix_sha256, "reinjection_matrix_sha256"),
        )
        if self.extraction_matrix_sha256 is None or self.reinjection_matrix_sha256 is None:
            raise ValueError("latent resident plans require both route hashes")
        object.__setattr__(
            self,
            "extraction_shape",
            _shape2(self.extraction_shape, "extraction_shape"),
        )
        object.__setattr__(
            self,
            "reinjection_shape",
            _shape2(self.reinjection_shape, "reinjection_shape"),
        )
        latent_count = self.router_runtime.architecture.num_latents
        atom_count = len(self.atoms)
        hidden_dim = self.router_runtime.architecture.hidden_dim
        route_cells = latent_count * atom_count
        if hidden_dim > self.caps.max_hidden_dim:
            raise ValueError("resident router width exceeds FusionCaps")
        if latent_count > self.caps.max_latents:
            raise ValueError("resident latent count exceeds FusionCaps")
        if route_cells > self.caps.max_route_cells:
            raise ValueError("resident K*N exceeds FusionCaps")
        if atom_count > self.router_runtime.max_atoms:
            raise ValueError("resident atom count exceeds router bounds")
        if route_cells > self.router_runtime.max_route_cells:
            raise ValueError("resident K*N exceeds router bounds")
        if self.extraction_shape != (latent_count, atom_count):
            raise ValueError("latent resident extraction shape must be [K, N]")
        if self.reinjection_shape != (atom_count, latent_count):
            raise ValueError("latent resident reinjection shape must be [N, K]")
        known_atoms = {item.atom_id for item in self.atoms}
        coordinates = {
            (membership.atom_id, membership.latent_index)
            for membership in self.memberships
        }
        if len(coordinates) != len(self.memberships):
            raise ValueError("resident latent memberships must be unique")
        if any(membership.atom_id not in known_atoms for membership in self.memberships):
            raise ValueError("resident membership references an unknown atom")
        if any(
            membership.latent_index >= latent_count
            for membership in self.memberships
        ):
            raise ValueError("resident membership references an unknown latent slot")
        counts = {
            atom_id: sum(item.atom_id == atom_id for item in self.memberships)
            for atom_id in known_atoms
        }
        if any(count < 1 for count in counts.values()):
            raise ValueError("every resident atom requires a latent membership")
        if any(
            count > self.caps.max_latent_memberships_per_atom
            for count in counts.values()
        ):
            raise ValueError("resident memberships exceed the per-atom cap")
        if any(group.latent_index is None for group in self.groups):
            raise ValueError("resident latent groups must name unlabeled slots")
        if any(
            (atom_id, group.latent_index) not in coordinates
            for group in self.groups
            for atom_id in group.atom_ids
        ):
            raise ValueError("resident group slots must be retained memberships")


@dataclass(frozen=True, slots=True)
class QwenResidentFusionOperationReceipt(SealedIdentity):
    """Outer B receipt joining one A feature sub-operation to one router run."""

    _SEAL_FIELD = "operation_sha256"
    _SEAL_MISMATCH = "resident fusion operation SHA-256 does not match its contents"

    feature_suboperation: QwenAtomFeatureOperationReceipt
    implementation_sha256: str
    router_runtime: ResidentRouterRuntimeReceipt
    matched_input_sha256: str
    topology_plan_sha256: str
    latent_plan_sha256: str
    extraction_matrix_sha256: str
    reinjection_matrix_sha256: str
    extraction_shape: tuple[int, int]
    reinjection_shape: tuple[int, int]
    topology_groups_sha256: str
    latent_memberships_sha256: str
    latent_groups_sha256: str
    latent_atom_order_sha256: str
    route_matrix_canonical_dtype: str = "float32-le"
    route_weight_normalization_policy: str = "source_dtype_softmax_sum_v1"
    router_forward_count: int = 1
    operation_format: str = "qwen_resident_matched_fusion_v1"
    operation_kind: str = "resident_matched_fusion"
    qwen_executed: bool = True
    router_executed: bool = True
    matched_pair_produced: bool = True
    bounded_route_matrix_content_attested: bool = True
    route_matrix_values_retained: bool = False
    feature_tensor_content_attested: bool = False
    steered_tensor_produced: bool = True
    steered_tensor_content_attested: bool = False
    single_feature_workspace_attested: bool = True
    topology_reencode_count: int = 0
    operation_inputs_attested: bool = True
    retrieval_route_attested: bool = False
    performance_attested: bool = False
    feature_tensor_sha256: None = None
    steered_tensor_sha256: None = None
    retained_request_tensor_bytes: int = 0
    operation_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.feature_suboperation) is not QwenAtomFeatureOperationReceipt:
            raise TypeError("feature_suboperation must be the exact tranche-A receipt")
        if type(self.router_runtime) is not ResidentRouterRuntimeReceipt:
            raise TypeError("router_runtime must be ResidentRouterRuntimeReceipt")
        self.feature_suboperation._seal()
        self.router_runtime._seal()
        normalize_fields(
            self,
            implementation_sha256=_sha256,
            matched_input_sha256=_sha256,
            topology_plan_sha256=_sha256,
            latent_plan_sha256=_sha256,
            extraction_matrix_sha256=_sha256,
            reinjection_matrix_sha256=_sha256,
            topology_groups_sha256=_sha256,
            latent_memberships_sha256=_sha256,
            latent_groups_sha256=_sha256,
            latent_atom_order_sha256=_sha256,
            router_forward_count=_positive_int,
            topology_reencode_count=_nonnegative_int,
            retained_request_tensor_bytes=_nonnegative_int,
        )
        object.__setattr__(
            self,
            "extraction_shape",
            _shape2(self.extraction_shape, "extraction_shape"),
        )
        object.__setattr__(
            self,
            "reinjection_shape",
            _shape2(self.reinjection_shape, "reinjection_shape"),
        )
        self._validate_join()
        self._validate_claims()
        _require_plain_identity_tree(self, "resident fusion operation")
        self._seal()

    def _validate_join(self) -> None:
        feature = self.feature_suboperation
        runtime = self.router_runtime
        atom_count = len(feature.atoms)
        latent_count = runtime.architecture.num_latents
        if runtime.architecture.hidden_dim != feature.feature_shape[1]:
            raise ValueError("resident router width disagrees with Qwen features")
        if runtime.device != feature.feature_device:
            raise ValueError("resident router and Qwen feature devices differ")
        if runtime.execution_dtype != feature.feature_execution_dtype:
            raise ValueError("resident router and Qwen feature dtypes differ")
        if self.extraction_shape != (latent_count, atom_count):
            raise ValueError("resident extraction shape must be [K, N]")
        if self.reinjection_shape != (atom_count, latent_count):
            raise ValueError("resident reinjection shape must be [N, K]")
        if (
            type(self.route_matrix_canonical_dtype) is not str
            or self.route_matrix_canonical_dtype != "float32-le"
        ):
            raise ValueError("resident route matrices require canonical float32-le")
        if (
            type(self.route_weight_normalization_policy) is not str
            or self.route_weight_normalization_policy
            != "source_dtype_softmax_sum_v1"
        ):
            raise ValueError("resident route normalization policy is unsupported")
        if latent_count > feature.caps.max_latents:
            raise ValueError("resident latent count exceeds FusionCaps")
        if latent_count * atom_count > feature.caps.max_route_cells:
            raise ValueError("resident K*N exceeds FusionCaps")
        if atom_count > runtime.max_atoms:
            raise ValueError("resident atom count exceeds router bounds")
        if latent_count * atom_count > runtime.max_route_cells:
            raise ValueError("resident K*N exceeds router bounds")
        expected_matched_input = resident_matched_input_sha256(
            feature_suboperation_sha256=feature.operation_sha256,
            router_runtime_sha256=runtime.runtime_sha256,
            implementation_sha256=self.implementation_sha256,
        )
        if self.matched_input_sha256 != expected_matched_input:
            raise ValueError("resident matched-input identity is not canonical")

    def _validate_claims(self) -> None:
        if self.feature_tensor_sha256 is not None or self.steered_tensor_sha256 is not None:
            raise ValueError("resident fusion cannot bind full request tensor hashes")
        expected = {
            "qwen_executed": True,
            "router_executed": True,
            "matched_pair_produced": True,
            "bounded_route_matrix_content_attested": True,
            "route_matrix_values_retained": False,
            "feature_tensor_content_attested": False,
            "steered_tensor_produced": True,
            "steered_tensor_content_attested": False,
            "single_feature_workspace_attested": True,
            "operation_inputs_attested": True,
            "retrieval_route_attested": False,
            "performance_attested": False,
        }
        if any(type(getattr(self, name)) is not bool for name in expected):
            raise TypeError("resident operation attestation flags must be booleans")
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("resident operation overstates its claim boundary")
        if any(
            type(value) is not str
            for value in (self.operation_format, self.operation_kind)
        ) or (
            self.operation_format != "qwen_resident_matched_fusion_v1"
            or self.operation_kind != "resident_matched_fusion"
        ):
            raise ValueError("resident operation format is unsupported")
        if self.router_forward_count != 1:
            raise ValueError("resident fusion requires exactly one router forward")
        if self.topology_reencode_count != 0:
            raise ValueError("resident topology control cannot re-encode features")
        if self.retained_request_tensor_bytes != 0:
            raise ValueError("resident operation cannot retain request tensors")


@dataclass(frozen=True, slots=True)
class MatchedEvidenceFusionPairReceipt(SealedIdentity):
    """Sealed join over the outer operation and both resident plans."""

    _SEAL_FIELD = "pair_sha256"
    _SEAL_MISMATCH = "matched resident pair SHA-256 does not match its contents"

    operation_sha256: str
    feature_suboperation_sha256: str
    topology_plan_sha256: str
    latent_plan_sha256: str
    matched_input_sha256: str
    exact_atom_set_shared: bool = True
    exact_hyperedges_shared: bool = True
    feature_operation_shared: bool = True
    pair_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            operation_sha256=_sha256,
            feature_suboperation_sha256=_sha256,
            topology_plan_sha256=_sha256,
            latent_plan_sha256=_sha256,
            matched_input_sha256=_sha256,
        )
        expected = {
            "exact_atom_set_shared": True,
            "exact_hyperedges_shared": True,
            "feature_operation_shared": True,
        }
        if any(type(getattr(self, name)) is not bool for name in expected):
            raise TypeError("matched-pair claim flags must be booleans")
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("matched-pair receipt must bind one exact shared input")
        _require_plain_identity_tree(self, "matched-pair receipt")
        self._seal()


@dataclass(frozen=True, slots=True)
class MatchedEvidenceFusionPair:
    """Exact tensor-free public result for one atomic resident operation."""

    topology_only: ResidentEvidenceFusionPlan
    latent_router: ResidentEvidenceFusionPlan
    operation: QwenResidentFusionOperationReceipt
    receipt: MatchedEvidenceFusionPairReceipt

    def __post_init__(self) -> None:
        expected_types = (
            (self.topology_only, ResidentEvidenceFusionPlan, "topology_only"),
            (self.latent_router, ResidentEvidenceFusionPlan, "latent_router"),
            (self.operation, QwenResidentFusionOperationReceipt, "operation"),
            (self.receipt, MatchedEvidenceFusionPairReceipt, "receipt"),
        )
        for value, expected_type, label in expected_types:
            if type(value) is not expected_type:
                raise TypeError(f"{label} has the wrong resident-owned type")
            value._seal()
            _require_plain_identity_tree(value, label)
        if self.topology_only.mode != "topology_only":
            raise ValueError("matched control must be topology_only")
        if self.latent_router.mode != "latent_router":
            raise ValueError("matched treatment must be latent_router")
        feature_sha = self.operation.feature_suboperation.operation_sha256
        shared_plan_fields = ("matched_input_sha256", "caps", "atoms", "hyperedges")
        if any(
            getattr(self.topology_only, name) != getattr(self.latent_router, name)
            for name in shared_plan_fields
        ):
            raise ValueError("resident plans do not share one exact matched input")
        if (
            self.topology_only.feature_suboperation_sha256 != feature_sha
            or self.latent_router.feature_suboperation_sha256 != feature_sha
        ):
            raise ValueError("resident plans do not bind the feature sub-operation")
        feature = self.operation.feature_suboperation
        if (
            self.topology_only.caps != feature.caps
            or self.topology_only.atoms != feature.atoms
            or self.topology_only.hyperedges != feature.hyperedges
        ):
            raise ValueError("resident plans do not match the feature sub-operation")
        if (
            self.operation.topology_plan_sha256 != self.topology_only.plan_sha256
            or self.operation.latent_plan_sha256 != self.latent_router.plan_sha256
        ):
            raise ValueError("resident operation does not bind both plan identities")
        expected_receipt = {
            "operation_sha256": self.operation.operation_sha256,
            "feature_suboperation_sha256": feature_sha,
            "topology_plan_sha256": self.topology_only.plan_sha256,
            "latent_plan_sha256": self.latent_router.plan_sha256,
            "matched_input_sha256": self.operation.matched_input_sha256,
        }
        if any(getattr(self.receipt, name) != value for name, value in expected_receipt.items()):
            raise ValueError("matched-pair receipt does not bind its exact values")
        if self.operation.matched_input_sha256 != self.topology_only.matched_input_sha256:
            raise ValueError("resident operation and plans disagree on matched input")
        treatment = self.latent_router
        if (
            treatment.router_runtime != self.operation.router_runtime
            or treatment.extraction_matrix_sha256
            != self.operation.extraction_matrix_sha256
            or treatment.reinjection_matrix_sha256
            != self.operation.reinjection_matrix_sha256
            or treatment.extraction_shape != self.operation.extraction_shape
            or treatment.reinjection_shape != self.operation.reinjection_shape
        ):
            raise ValueError("resident treatment route provenance disagrees with operation")
        if self.operation.topology_groups_sha256 != resident_values_sha256(
            "topology_groups", self.topology_only.groups
        ):
            raise ValueError("resident operation topology groups disagree with control")
        if self.operation.latent_memberships_sha256 != resident_values_sha256(
            "latent_memberships", self.latent_router.memberships
        ):
            raise ValueError("resident operation memberships disagree with treatment")
        if self.operation.latent_groups_sha256 != resident_values_sha256(
            "latent_groups", self.latent_router.groups
        ):
            raise ValueError("resident operation latent groups disagree with treatment")
        if self.operation.latent_atom_order_sha256 != resident_atom_order_sha256(
            self.latent_router.atom_order
        ):
            raise ValueError("resident operation atom order disagrees with treatment")


__all__ = [
    "MatchedEvidenceFusionPair",
    "MatchedEvidenceFusionPairReceipt",
    "QwenResidentFusionOperationReceipt",
    "ResidentEvidenceFusionPlan",
    "ResidentFusionMode",
    "ResidentRouterRuntimeReceipt",
    "resident_matched_input_sha256",
]
