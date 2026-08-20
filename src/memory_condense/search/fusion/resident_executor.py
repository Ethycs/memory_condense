"""Exact private GPU executor for one Qwen-to-latent resident operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory_condense.search.fusion.latent_router import (
    LatentEvidenceRouter,
    LatentRouterForward,
)
from memory_condense.search.fusion.models import (
    ExtractiveGroup,
    FusionCaps,
    LatentMembership,
)
from memory_condense.search.fusion.planning_core import (
    _latent_memberships_and_groups,
)
from memory_condense.search.fusion.qwen_feature_executor import (
    _validate_feature_output,
)
from memory_condense.search.fusion.qwen_feature_models import (
    QwenAtomFeatureCaps,
    QwenAtomFeatureProviderReceipt,
)
from memory_condense.search.fusion.resident_models import (
    ResidentRouterRuntimeReceipt,
)
from memory_condense.search.fusion.tensor_identity import (
    CANONICAL_TENSOR_DTYPE,
    canonical_float32_tensor,
)


_PINNED_ROUTER_TYPE = LatentEvidenceRouter
_PINNED_ROUTER_FORWARD_TYPE = LatentRouterForward
_PINNED_ROUTER_FORWARD_INIT = LatentRouterForward.__init__
_PINNED_ROUTE_ONE = LatentEvidenceRouter.route_one
_PINNED_ASSERT_INFERENCE_SEAL = LatentEvidenceRouter._assert_inference_seal
_PINNED_RUNTIME_FINGERPRINT = LatentEvidenceRouter._runtime_fingerprint
_PINNED_RUNTIME_STRUCTURE = LatentEvidenceRouter._runtime_structure
_PINNED_REJECT_RUNTIME_HOOKS = LatentEvidenceRouter._reject_runtime_hooks
_PINNED_RESIDENT_RUNTIME_GETTER = (
    LatentEvidenceRouter.resident_runtime_receipt.fget
)
_PINNED_FEATURE_OUTPUT_VALIDATOR = _validate_feature_output
_PINNED_CANONICALIZE = canonical_float32_tensor
_PINNED_REDUCE_ROUTES = _latent_memberships_and_groups


@dataclass(frozen=True, slots=True)
class _ResidentRouteExecution:
    """Tensor-free private result retaining only bounded route matrix values."""

    primary_forward_count: int
    router_forward_count: int
    extraction_matrix_sha256: str
    reinjection_matrix_sha256: str
    extraction_shape: tuple[int, int]
    reinjection_shape: tuple[int, int]
    memberships: tuple[LatentMembership, ...]
    groups: tuple[ExtractiveGroup, ...]
    atom_order: tuple[str, ...]
    route_matrix_canonical_dtype: str = CANONICAL_TENSOR_DTYPE

    def __post_init__(self) -> None:
        if (
            isinstance(self.primary_forward_count, bool)
            or not isinstance(self.primary_forward_count, int)
            or self.primary_forward_count < 1
        ):
            raise ValueError("resident execution requires primary Qwen forwards")
        if type(self.router_forward_count) is not int or self.router_forward_count != 1:
            raise ValueError("resident execution requires exactly one router forward")
        if self.route_matrix_canonical_dtype != CANONICAL_TENSOR_DTYPE:
            raise ValueError("resident route matrices require canonical float32-le")
        if any(type(value) is not LatentMembership for value in self.memberships):
            raise TypeError("resident execution memberships changed type")
        if any(type(value) is not ExtractiveGroup for value in self.groups):
            raise TypeError("resident execution groups changed type")
        object.__setattr__(self, "memberships", tuple(self.memberships))
        object.__setattr__(self, "groups", tuple(self.groups))
        atom_order = tuple(self.atom_order)
        if any(type(atom_id) is not str for atom_id in atom_order):
            raise TypeError("resident execution atom order changed type")
        object.__setattr__(self, "atom_order", atom_order)
        for shape_name in ("extraction_shape", "reinjection_shape"):
            shape = tuple(getattr(self, shape_name))
            if (
                len(shape) != 2
                or any(type(dimension) is not int or dimension < 1 for dimension in shape)
            ):
                raise ValueError("resident execution route shapes must be positive 2-D")
            object.__setattr__(self, shape_name, shape)


def _validate_route_tensor(
    value: Any,
    *,
    torch: Any,
    expected_shape: tuple[int, int],
    runtime: ResidentRouterRuntimeReceipt,
    label: str,
) -> None:
    try:
        if type(value) is not torch.Tensor:
            raise TypeError(f"{label} returned a foreign tensor type")
        if tuple(int(item) for item in value.shape) != expected_shape:
            raise ValueError(f"{label} has the wrong shape")
        if str(value.device) != runtime.device:
            raise ValueError(f"{label} left the resident router device")
        if str(value.dtype) != runtime.execution_dtype:
            raise ValueError(f"{label} changed resident router dtype")
        if bool(value.requires_grad) or value.grad_fn is not None:
            raise RuntimeError(f"{label} retained an autograd graph")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{label} contains non-finite values")
    finally:
        value = None


def _assert_resident_executor_seams() -> None:
    if (
        LatentEvidenceRouter is not _PINNED_ROUTER_TYPE
        or LatentRouterForward is not _PINNED_ROUTER_FORWARD_TYPE
        or _PINNED_ROUTER_FORWARD_TYPE.__init__ is not _PINNED_ROUTER_FORWARD_INIT
        or _PINNED_ROUTER_TYPE.route_one is not _PINNED_ROUTE_ONE
        or _PINNED_ROUTER_TYPE._assert_inference_seal
        is not _PINNED_ASSERT_INFERENCE_SEAL
        or _PINNED_ROUTER_TYPE._runtime_fingerprint
        is not _PINNED_RUNTIME_FINGERPRINT
        or _PINNED_ROUTER_TYPE._runtime_structure is not _PINNED_RUNTIME_STRUCTURE
        or _PINNED_ROUTER_TYPE._reject_runtime_hooks
        is not _PINNED_REJECT_RUNTIME_HOOKS
        or _PINNED_ROUTER_TYPE.resident_runtime_receipt.fget
        is not _PINNED_RESIDENT_RUNTIME_GETTER
        or _validate_feature_output is not _PINNED_FEATURE_OUTPUT_VALIDATOR
        or canonical_float32_tensor is not _PINNED_CANONICALIZE
        or _latent_memberships_and_groups is not _PINNED_REDUCE_ROUTES
        or _validate_route_tensor is not _PINNED_VALIDATE_ROUTE_TENSOR
    ):
        raise RuntimeError("resident executor implementation seams were replaced")


def _execute_qwen_resident_route(
    *,
    encoder: Any,
    torch: Any,
    output_layer: int,
    provider_receipt: QwenAtomFeatureProviderReceipt,
    feature_caps: QwenAtomFeatureCaps,
    rows: tuple[Any, ...],
    batches: tuple[tuple[int, int, int, int], ...],
    gate_token: Any,
    router: LatentEvidenceRouter,
    router_runtime: ResidentRouterRuntimeReceipt,
    atom_ids: tuple[str, ...],
    topology_degree: dict[str, int],
    caps: FusionCaps,
) -> _ResidentRouteExecution:
    """Fill one ``[N,D]``, route its exact storage once, and shed full tensors."""

    _assert_resident_executor_seams()
    if type(router) is not _PINNED_ROUTER_TYPE:
        raise TypeError("resident execution requires the exact owned router")
    if type(router_runtime) is not ResidentRouterRuntimeReceipt:
        raise TypeError("resident execution requires exact router runtime identity")
    provider_receipt._seal()
    feature_caps._seal()
    router_runtime._seal()
    if type(caps) is not FusionCaps:
        raise TypeError("resident execution requires exact FusionCaps")
    caps._seal()
    if _PINNED_RESIDENT_RUNTIME_GETTER(router) != router_runtime:
        raise RuntimeError("resident router runtime changed before execution")

    feature_tensor = None
    batch_features = None
    routed = None
    steered = None
    extraction = None
    reinjection = None
    canonical_extraction = None
    canonical_reinjection = None
    try:
        feature_tensor = torch.empty(
            (len(rows), provider_receipt.hidden_dim),
            device=provider_receipt.device,
            dtype=encoder.dtype,
        )
        for start, count, _width, _workspace in batches:
            stop = start + count
            batch_features = encoder._encode_selected_layer_final_readout(
                tuple(row.token_ids for row in rows[start:stop]),
                layer=output_layer,
                _gate_token=gate_token,
            )
            _PINNED_FEATURE_OUTPUT_VALIDATOR(
                batch_features,
                torch=torch,
                expected_rows=count,
                provider_receipt=provider_receipt,
                label="resident Qwen feature batch",
            )
            feature_tensor[start:stop].copy_(batch_features)
            batch_features = None
        if not bool(torch.isfinite(feature_tensor).all().item()):
            raise ValueError("resident Qwen feature tensor contains non-finite values")
        if feature_tensor.layout is not torch.strided or not feature_tensor.is_contiguous():
            raise RuntimeError("resident feature workspace must be contiguous strided storage")
        if int(feature_tensor.storage_offset()) != 0:
            raise RuntimeError("resident feature workspace must start at storage offset zero")
        if bool(feature_tensor.requires_grad) or feature_tensor.grad_fn is not None:
            raise RuntimeError("resident feature workspace retained an autograd graph")

        storage_identity = (
            id(feature_tensor),
            int(feature_tensor.data_ptr()),
            int(feature_tensor._version),
            tuple(int(value) for value in feature_tensor.shape),
            str(feature_tensor.device),
            str(feature_tensor.dtype),
            tuple(int(value) for value in feature_tensor.stride()),
            int(feature_tensor.storage_offset()),
            str(feature_tensor.layout),
        )
        routed = _PINNED_ROUTE_ONE(router, feature_tensor)
        if type(routed) is not _PINNED_ROUTER_FORWARD_TYPE:
            raise TypeError("resident router returned an unsupported result")
        if storage_identity != (
            id(feature_tensor),
            int(feature_tensor.data_ptr()),
            int(feature_tensor._version),
            tuple(int(value) for value in feature_tensor.shape),
            str(feature_tensor.device),
            str(feature_tensor.dtype),
            tuple(int(value) for value in feature_tensor.stride()),
            int(feature_tensor.storage_offset()),
            str(feature_tensor.layout),
        ):
            raise RuntimeError("resident router mutated or replaced feature storage")

        atom_count = len(rows)
        latent_count = router_runtime.architecture.num_latents
        steered = routed.steered_nodes
        extraction = routed.extraction_attention
        reinjection = routed.reinjection_attention
        _PINNED_VALIDATE_ROUTE_TENSOR(
            steered,
            torch=torch,
            expected_shape=(atom_count, provider_receipt.hidden_dim),
            runtime=router_runtime,
            label="resident steered nodes",
        )
        _PINNED_VALIDATE_ROUTE_TENSOR(
            extraction,
            torch=torch,
            expected_shape=(latent_count, atom_count),
            runtime=router_runtime,
            label="resident extraction attention",
        )
        _PINNED_VALIDATE_ROUTE_TENSOR(
            reinjection,
            torch=torch,
            expected_shape=(atom_count, latent_count),
            runtime=router_runtime,
            label="resident reinjection attention",
        )

        # Release the full [N,D] output before any device-to-host matrix copy.
        routed = None
        steered = None
        feature_tensor = None
        canonical_extraction = _PINNED_CANONICALIZE(
            extraction,
            label="resident extraction attention",
        )
        extraction = None
        canonical_reinjection = _PINNED_CANONICALIZE(
            reinjection,
            label="resident reinjection attention",
        )
        reinjection = None
        memberships, groups, atom_order = _PINNED_REDUCE_ROUTES(
            atom_ids,
            canonical_extraction,
            canonical_reinjection,
            topology_degree,
            caps,
            source_dtype=router_runtime.execution_dtype,
        )
        return _ResidentRouteExecution(
            primary_forward_count=len(batches),
            router_forward_count=1,
            extraction_matrix_sha256=canonical_extraction.tensor_sha256,
            reinjection_matrix_sha256=canonical_reinjection.tensor_sha256,
            extraction_shape=canonical_extraction.shape,
            reinjection_shape=canonical_reinjection.shape,
            memberships=memberships,
            groups=groups,
            atom_order=atom_order,
        )
    finally:
        batch_features = None
        routed = None
        steered = None
        extraction = None
        reinjection = None
        canonical_extraction = None
        canonical_reinjection = None
        feature_tensor = None
        rows = ()
        batches = ()
        gate_token = None


_PINNED_VALIDATE_ROUTE_TENSOR = _validate_route_tensor


__all__ = [
    "_ResidentRouteExecution",
    "_assert_resident_executor_seams",
    "_execute_qwen_resident_route",
    "_validate_route_tensor",
]
