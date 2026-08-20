"""Text-free identities emitted by bounded post-retrieval fusion.

The values in this module are receipts and extractive ordering instructions.
They never contain evidence text, node features, attention matrices, residuals,
or learned parameters.  Exact source bytes remain in the ``EvidencePacket``;
the plan binds them by digest and only rearranges their existing atom IDs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from memory_condense.domain._discourse_identity import (
    _finite,
    _nonempty,
    _sha256,
    _strict_int,
    _unique_nonempty,
    normalize_fields,
)
from memory_condense.domain.sealed import SealedIdentity


FusionMode = Literal["topology_only", "latent_router"]
RouterTrainingStatus = Literal["untrained", "trained_declared"]
_MACHINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,127}$")
_POOLING_MODES = frozenset({"mean", "cls", "last_token", "attention_pool", "none"})
_ROUTER_PARAMETER_DTYPES = frozenset(
    {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
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


def _optional_sha256(value: str | None, label: str) -> str | None:
    return None if value is None else _sha256(value, label)


def _shape(value: tuple[int, ...], label: str) -> tuple[int, ...]:
    normalized = tuple(_positive_int(item, label) for item in value)
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    return normalized


def _unit_weight(value: float, label: str) -> float:
    normalized = float(_finite(value, label))
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return normalized


@dataclass(frozen=True, slots=True)
class FusionCaps(SealedIdentity):
    """Hard bounds shared by topology-only and latent-router arms."""

    _SEAL_FIELD = "caps_sha256"
    _SEAL_MISMATCH = "fusion caps SHA-256 does not match its contents"

    max_atoms: int = 64
    max_latents: int = 16
    max_hidden_dim: int = 4096
    max_route_cells: int = 1024
    max_topology_links: int = 2048
    max_hyperedges: int = 64
    max_groups: int = 64
    max_group_atoms: int = 16
    max_latent_memberships_per_atom: int = 2
    caps_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            max_atoms=_positive_int,
            max_latents=_positive_int,
            max_hidden_dim=_positive_int,
            max_route_cells=_positive_int,
            max_topology_links=_nonnegative_int,
            max_hyperedges=_positive_int,
            max_groups=_positive_int,
            max_group_atoms=_positive_int,
            max_latent_memberships_per_atom=_positive_int,
        )
        self._seal()


@dataclass(frozen=True, slots=True)
class FusionAtomRef(SealedIdentity):
    """Text-free binding to one exact selected packet atom and source span."""

    _SEAL_FIELD = "ref_sha256"
    _SEAL_MISMATCH = "fusion atom ref SHA-256 does not match its contents"

    atom_id: str
    atom_identity_sha256: str
    span_identity_sha256: str
    quote_sha256: str
    ref_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            atom_id=_nonempty,
            atom_identity_sha256=_sha256,
            span_identity_sha256=_sha256,
            quote_sha256=_sha256,
        )
        self._seal()


@dataclass(frozen=True, slots=True)
class DeclaredFeatureExtractorIdentity(SealedIdentity):
    """Caller-declared, text-free identity of the node-feature extractor."""

    _SEAL_FIELD = "extractor_identity_sha256"
    _SEAL_MISMATCH = "feature extractor identity SHA-256 does not match its contents"

    extractor_id: str
    implementation_sha256: str
    checkpoint_sha256: str
    output_layer: int
    pooling: str
    max_input_tokens_per_atom: int
    hidden_dim: int
    extractor_identity_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            extractor_id=_nonempty,
            implementation_sha256=_sha256,
            checkpoint_sha256=_sha256,
            output_layer=_nonnegative_int,
            pooling=_nonempty,
            max_input_tokens_per_atom=_positive_int,
            hidden_dim=_positive_int,
        )
        if not _MACHINE_ID.fullmatch(self.extractor_id):
            raise ValueError("extractor_id must be a bounded machine identifier")
        if self.pooling not in _POOLING_MODES:
            raise ValueError("pooling must be a supported closed value")
        self._seal()


@dataclass(frozen=True, slots=True)
class NodeFeatureReceipt(SealedIdentity):
    """Exact ordered atom/query binding for one declared feature batch."""

    _SEAL_FIELD = "feature_receipt_sha256"
    _SEAL_MISMATCH = "node feature receipt SHA-256 does not match its contents"

    extractor: DeclaredFeatureExtractorIdentity
    ordered_atom_ids: tuple[str, ...]
    query_sha256: str
    tensor_sha256: str
    tensor_shape: tuple[int, ...]
    tensor_dtype: str = "float32-le"
    feature_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.extractor) is not DeclaredFeatureExtractorIdentity:
            raise TypeError("extractor must be DeclaredFeatureExtractorIdentity")
        normalize_fields(
            self,
            ordered_atom_ids=_unique_nonempty,
            query_sha256=_sha256,
            tensor_sha256=_sha256,
        )
        object.__setattr__(self, "tensor_shape", _shape(tuple(self.tensor_shape), "tensor_shape"))
        if len(self.tensor_shape) != 2:
            raise ValueError("node feature tensor shape must be [N, D]")
        if self.tensor_shape != (
            len(self.ordered_atom_ids),
            self.extractor.hidden_dim,
        ):
            raise ValueError("node feature shape disagrees with atom IDs or extractor")
        if self.tensor_dtype != "float32-le":
            raise ValueError("node feature receipt requires canonical float32-le")
        self._seal()


@dataclass(frozen=True, slots=True)
class RouterArchitectureReceipt(SealedIdentity):
    """Explicit identity of the exact two-pass K-latent router contract."""

    _SEAL_FIELD = "architecture_sha256"
    _SEAL_MISMATCH = "router architecture SHA-256 does not match its contents"

    hidden_dim: int
    num_latents: int
    num_heads: int
    parameter_count: int
    algorithm: str = "k_latent_two_pass_cross_attention_v1"
    phase_count: int = 2
    extraction_query_axis: str = "latent_k"
    extraction_key_value_axis: str = "evidence_n"
    reinjection_query_axis: str = "evidence_n"
    reinjection_key_value_axis: str = "updated_latent_k"
    attention_weight_reduction: str = "mean_over_heads"
    extraction_residual_rule: str = "none"
    residual_rule: str = "x_plus_reinjected_update"
    architecture_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            hidden_dim=_positive_int,
            num_latents=_positive_int,
            num_heads=_positive_int,
            parameter_count=_positive_int,
            algorithm=_nonempty,
            phase_count=_positive_int,
            extraction_query_axis=_nonempty,
            extraction_key_value_axis=_nonempty,
            reinjection_query_axis=_nonempty,
            reinjection_key_value_axis=_nonempty,
            attention_weight_reduction=_nonempty,
            extraction_residual_rule=_nonempty,
            residual_rule=_nonempty,
        )
        expected = {
            "algorithm": "k_latent_two_pass_cross_attention_v1",
            "phase_count": 2,
            "extraction_query_axis": "latent_k",
            "extraction_key_value_axis": "evidence_n",
            "reinjection_query_axis": "evidence_n",
            "reinjection_key_value_axis": "updated_latent_k",
            "attention_weight_reduction": "mean_over_heads",
            "extraction_residual_rule": "none",
            "residual_rule": "x_plus_reinjected_update",
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("router architecture must be the exact two-pass contract")
        if self.hidden_dim % self.num_heads:
            raise ValueError("router hidden_dim must be divisible by num_heads")
        expected_parameters = (
            self.num_latents * self.hidden_dim
            + 8 * self.hidden_dim * self.hidden_dim
            + 8 * self.hidden_dim
        )
        if self.parameter_count != expected_parameters:
            raise ValueError("router parameter_count disagrees with two biased MHA blocks")
        self._seal()


@dataclass(frozen=True, slots=True)
class RouterStateReceipt(SealedIdentity):
    """Identity of loaded parameters, distinct from a checkpoint-file receipt."""

    _SEAL_FIELD = "state_receipt_sha256"
    _SEAL_MISMATCH = "router state receipt SHA-256 does not match its contents"

    loaded_parameter_bytes_sha256: str
    operational_float32_sha256: str
    parameter_count: int
    parameter_dtypes: tuple[str, ...]
    training_status: RouterTrainingStatus
    state_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            loaded_parameter_bytes_sha256=_sha256,
            operational_float32_sha256=_sha256,
            parameter_count=_positive_int,
            parameter_dtypes=_unique_nonempty,
        )
        object.__setattr__(self, "parameter_dtypes", tuple(sorted(self.parameter_dtypes)))
        if not self.parameter_dtypes:
            raise ValueError("router state requires at least one parameter dtype")
        if any(value not in _ROUTER_PARAMETER_DTYPES for value in self.parameter_dtypes):
            raise ValueError("router parameter dtype is unsupported")
        if self.training_status not in {"untrained", "trained_declared"}:
            raise ValueError("router training status must be explicit")
        self._seal()


@dataclass(frozen=True, slots=True)
class AuthoritativeHyperedge(SealedIdentity):
    """One selected bundle as an undirected atom co-membership witness.

    Unit and relation IDs are copied only as bundle witnesses.  A closure plan
    does not retain relation direction or member roles, so this receipt never
    invents either.
    """

    _SEAL_FIELD = "hyperedge_sha256"
    _SEAL_MISMATCH = "fusion hyperedge SHA-256 does not match its contents"

    bundle_id: str
    atom_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    unit_witness_ids: tuple[str, ...] = ()
    relation_witness_ids: tuple[str, ...] = ()
    required: bool = False
    utility: float = 0.0
    hyperedge_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            bundle_id=_nonempty,
            atom_ids=_unique_nonempty,
            obligation_ids=_unique_nonempty,
            unit_witness_ids=_unique_nonempty,
            relation_witness_ids=_unique_nonempty,
            utility=_finite,
        )
        if not self.atom_ids:
            raise ValueError("an authoritative hyperedge requires at least one atom")
        if not isinstance(self.required, bool):
            raise ValueError("hyperedge required must be a boolean")
        self._seal()


@dataclass(frozen=True, slots=True)
class LatentMembership(SealedIdentity):
    """Unlabeled sparse receipt for one atom-to-latent route."""

    _SEAL_FIELD = "membership_sha256"
    _SEAL_MISMATCH = "latent membership SHA-256 does not match its contents"

    atom_id: str
    latent_index: int
    extraction_weight: float
    reinjection_weight: float
    joint_weight: float
    membership_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            atom_id=_nonempty,
            latent_index=_nonnegative_int,
        )
        object.__setattr__(
            self,
            "extraction_weight",
            _unit_weight(self.extraction_weight, "extraction_weight"),
        )
        object.__setattr__(
            self,
            "reinjection_weight",
            _unit_weight(self.reinjection_weight, "reinjection_weight"),
        )
        object.__setattr__(
            self,
            "joint_weight",
            _unit_weight(self.joint_weight, "joint_weight"),
        )
        expected = math.sqrt(self.extraction_weight * self.reinjection_weight)
        if not math.isclose(self.joint_weight, expected, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("joint_weight must be the geometric route weight")
        self._seal()


@dataclass(frozen=True, slots=True)
class ExtractiveGroup(SealedIdentity):
    """A deterministic group/order over existing atoms; never a synopsis."""

    _SEAL_FIELD = "group_sha256"
    _SEAL_MISMATCH = "extractive group SHA-256 does not match its contents"

    group_index: int
    atom_ids: tuple[str, ...]
    latent_index: int | None = None
    group_sha256: str = ""

    def __post_init__(self) -> None:
        normalize_fields(
            self,
            group_index=_nonnegative_int,
            atom_ids=_unique_nonempty,
        )
        if not self.atom_ids:
            raise ValueError("an extractive group requires at least one atom")
        if self.latent_index is not None:
            object.__setattr__(
                self,
                "latent_index",
                _nonnegative_int(self.latent_index, "latent_index"),
            )
        self._seal()


@dataclass(frozen=True, slots=True)
class EvidenceFusionPlan(SealedIdentity):
    """Immutable, text-free output of post-retrieval evidence fusion."""

    _SEAL_FIELD = "fusion_sha256"
    _SEAL_MISMATCH = "evidence fusion SHA-256 does not match its contents"

    mode: FusionMode
    packet_receipt_sha256: str
    closure_plan_sha256: str
    query_program_sha256: str
    query_sha256: str
    closure_policy_sha256: str
    snapshot_sha256: str
    matched_input_sha256: str
    caps: FusionCaps
    atoms: tuple[FusionAtomRef, ...]
    hyperedges: tuple[AuthoritativeHyperedge, ...]
    memberships: tuple[LatentMembership, ...]
    groups: tuple[ExtractiveGroup, ...]
    atom_order: tuple[str, ...]
    node_features: NodeFeatureReceipt | None = None
    router_architecture: RouterArchitectureReceipt | None = None
    router_state: RouterStateReceipt | None = None
    steered_features_sha256: str | None = None
    extraction_matrix_sha256: str | None = None
    reinjection_matrix_sha256: str | None = None
    extraction_shape: tuple[int, ...] = ()
    reinjection_shape: tuple[int, ...] = ()
    plan_retained_request_tensor_bytes: int = 0
    fusion_sha256: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"topology_only", "latent_router"}:
            raise ValueError("fusion mode must be topology_only or latent_router")
        normalize_fields(
            self,
            packet_receipt_sha256=_sha256,
            closure_plan_sha256=_sha256,
            query_program_sha256=_sha256,
            query_sha256=_sha256,
            closure_policy_sha256=_sha256,
            snapshot_sha256=_sha256,
            matched_input_sha256=_sha256,
            plan_retained_request_tensor_bytes=_nonnegative_int,
        )
        if type(self.caps) is not FusionCaps:
            raise TypeError("caps must be FusionCaps")
        self._freeze_collections()
        self._validate_atom_set()
        self._validate_topology()
        self._validate_shared_features()
        self._validate_route_fields()
        if self.plan_retained_request_tensor_bytes != 0:
            raise ValueError("fusion plans cannot retain request tensors")
        self._seal()

    def _freeze_collections(self) -> None:
        collections = (
            ("atoms", FusionAtomRef),
            ("hyperedges", AuthoritativeHyperedge),
            ("memberships", LatentMembership),
            ("groups", ExtractiveGroup),
        )
        for name, expected_type in collections:
            values = tuple(getattr(self, name))
            if any(type(value) is not expected_type for value in values):
                raise TypeError(f"{name} must contain {expected_type.__name__} values")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "extraction_shape", tuple(self.extraction_shape))
        object.__setattr__(self, "reinjection_shape", tuple(self.reinjection_shape))

    def _validate_atom_set(self) -> None:
        atoms = tuple(self.atoms)
        atom_ids = tuple(item.atom_id for item in atoms)
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError("fusion atom IDs must be unique")
        if len(atom_ids) > self.caps.max_atoms:
            raise ValueError("fusion atom count exceeds max_atoms")
        normalize_fields(self, atom_order=_unique_nonempty)
        if len(self.atom_order) != len(atom_ids) or set(self.atom_order) != set(atom_ids):
            raise ValueError("fusion atom_order must preserve the exact atom set")
        groups = tuple(self.groups)
        if len(groups) > self.caps.max_groups:
            raise ValueError("fusion group count exceeds max_groups")
        if tuple(item.group_index for item in groups) != tuple(range(len(groups))):
            raise ValueError("fusion group indices must be contiguous from zero")
        grouped = tuple(atom_id for group in groups for atom_id in group.atom_ids)
        if grouped != self.atom_order:
            raise ValueError("extractive groups must partition atom_order exactly")
        if any(len(group.atom_ids) > self.caps.max_group_atoms for group in groups):
            raise ValueError("an extractive group exceeds max_group_atoms")

    def _validate_topology(self) -> None:
        known_atoms = {item.atom_id for item in self.atoms}
        hyperedges = tuple(self.hyperedges)
        if len(hyperedges) > self.caps.max_hyperedges:
            raise ValueError("fusion hyperedge count exceeds max_hyperedges")
        if len({item.bundle_id for item in hyperedges}) != len(hyperedges):
            raise ValueError("fusion hyperedge bundle IDs must be unique")
        if any(atom_id not in known_atoms for edge in hyperedges for atom_id in edge.atom_ids):
            raise ValueError("fusion hyperedge references an unknown atom")
        topology_links = sum(
            len(edge.atom_ids) * (len(edge.atom_ids) - 1) // 2
            for edge in hyperedges
        )
        if topology_links > self.caps.max_topology_links:
            raise ValueError("fusion topology links exceed max_topology_links")

    def _validate_shared_features(self) -> None:
        if type(self.node_features) is not NodeFeatureReceipt:
            raise TypeError("fusion plans require NodeFeatureReceipt")
        atom_count = len(self.atoms)
        if atom_count < 1:
            raise ValueError("fusion requires at least one selected atom")
        if self.node_features.ordered_atom_ids != tuple(
            item.atom_id for item in self.atoms
        ):
            raise ValueError("node feature receipt must preserve exact atom order")
        if self.node_features.query_sha256 != self.query_sha256:
            raise ValueError("node feature receipt query does not match fusion query")
        if self.node_features.tensor_shape[1] > self.caps.max_hidden_dim:
            raise ValueError("feature hidden dimension exceeds max_hidden_dim")

    def _validate_route_fields(self) -> None:
        route_digests = (
            self.steered_features_sha256,
            self.extraction_matrix_sha256,
            self.reinjection_matrix_sha256,
        )
        if self.mode == "topology_only":
            if any(value is not None for value in route_digests):
                raise ValueError("topology-only plans cannot bind latent route state")
            if self.router_architecture is not None:
                raise ValueError("topology-only plans cannot identify a router")
            if self.router_state is not None:
                raise ValueError("topology-only plans cannot identify a router")
            if self.extraction_shape or self.reinjection_shape:
                raise ValueError("topology-only plans cannot carry route matrix shapes")
            if self.memberships:
                raise ValueError("topology-only plans cannot carry latent memberships")
            if any(group.latent_index is not None for group in self.groups):
                raise ValueError("topology-only groups cannot name latent slots")
            return

        if any(value is None for value in route_digests):
            raise ValueError("latent-router plans require every route digest")
        for name in (
            "steered_features_sha256",
            "extraction_matrix_sha256",
            "reinjection_matrix_sha256",
        ):
            object.__setattr__(self, name, _optional_sha256(getattr(self, name), name))
        if type(self.router_architecture) is not RouterArchitectureReceipt:
            raise TypeError("latent-router plans require RouterArchitectureReceipt")
        if type(self.router_state) is not RouterStateReceipt:
            raise TypeError("latent-router plans require RouterStateReceipt")
        if self.router_state.parameter_count != self.router_architecture.parameter_count:
            raise ValueError("router state parameter count disagrees with architecture")
        object.__setattr__(
            self,
            "extraction_shape",
            _shape(self.extraction_shape, "extraction_shape"),
        )
        object.__setattr__(
            self,
            "reinjection_shape",
            _shape(self.reinjection_shape, "reinjection_shape"),
        )
        atom_count = len(self.atoms)
        if len(self.extraction_shape) != 2:
            raise ValueError("extraction shape must be [K, N]")
        latent_count, extraction_atoms = self.extraction_shape
        if extraction_atoms != atom_count:
            raise ValueError("extraction shape must be [K, N]")
        if self.reinjection_shape != (atom_count, latent_count):
            raise ValueError("reinjection shape must be [N, K]")
        if latent_count > self.caps.max_latents:
            raise ValueError("latent count exceeds max_latents")
        if self.router_architecture.hidden_dim != self.node_features.tensor_shape[1]:
            raise ValueError("router architecture hidden_dim disagrees with features")
        if self.router_architecture.num_latents != latent_count:
            raise ValueError("router architecture num_latents disagrees with matrices")
        if latent_count * atom_count > self.caps.max_route_cells:
            raise ValueError("K*N exceeds max_route_cells")
        memberships = tuple(self.memberships)
        coordinates = {(item.atom_id, item.latent_index) for item in memberships}
        if len(coordinates) != len(memberships):
            raise ValueError("latent memberships must be unique")
        known_atoms = {item.atom_id for item in self.atoms}
        if any(item.atom_id not in known_atoms for item in memberships):
            raise ValueError("latent membership references an unknown atom")
        if any(item.latent_index >= latent_count for item in memberships):
            raise ValueError("latent membership references an unknown slot")
        counts = {
            atom_id: sum(item.atom_id == atom_id for item in memberships)
            for atom_id in known_atoms
        }
        if any(count < 1 for count in counts.values()):
            raise ValueError("every atom requires a retained latent membership")
        if any(
            count > self.caps.max_latent_memberships_per_atom
            for count in counts.values()
        ):
            raise ValueError("latent memberships exceed the per-atom cap")
        if any(group.latent_index is None for group in self.groups):
            raise ValueError("latent-router groups must name an unlabeled slot")
        if any(
            group.latent_index is not None and group.latent_index >= latent_count
            for group in self.groups
        ):
            raise ValueError("extractive group references an unknown latent slot")
        membership_coordinates = {
            (item.atom_id, item.latent_index) for item in memberships
        }
        if any(
            (atom_id, group.latent_index) not in membership_coordinates
            for group in self.groups
            for atom_id in group.atom_ids
        ):
            raise ValueError("group latent must be a retained membership for every atom")


__all__ = [
    "AuthoritativeHyperedge",
    "DeclaredFeatureExtractorIdentity",
    "EvidenceFusionPlan",
    "ExtractiveGroup",
    "FusionAtomRef",
    "FusionCaps",
    "FusionMode",
    "LatentMembership",
    "NodeFeatureReceipt",
    "RouterArchitectureReceipt",
    "RouterStateReceipt",
    "RouterTrainingStatus",
]
