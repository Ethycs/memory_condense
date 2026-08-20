"""Atomic resident Qwen-to-latent matched fusion builder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import memory_condense.search.fusion.latent_router as latent_router_module
import memory_condense.search.fusion.models as fusion_models_module
import memory_condense.search.fusion.planning_core as planning_core_module
import memory_condense.search.fusion.qwen_feature_models as feature_models_module
import memory_condense.search.fusion.qwen_features as qwen_features_module
import memory_condense.search.fusion.resident_executor as resident_executor_module
import memory_condense.search.fusion.resident_models as resident_models_module
import memory_condense.search.fusion.tensor_identity as tensor_identity_module
import memory_condense.domain._discourse_identity as discourse_identity_module
import memory_condense.domain.sealed as sealed_identity_module
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.discourse import ClosurePlan, EvidencePacket
from memory_condense.search.fusion.latent_router import (
    LatentEvidenceRouter,
    LatentRouterForward,
)
from memory_condense.search.fusion.models import FusionCaps
from memory_condense.search.fusion.planning_core import (
    _preflight_topology,
    _topology_atom_groups,
)
from memory_condense.search.fusion.qwen_feature_models import (
    QwenAtomFeatureCaps,
    QwenAtomFeatureOperationReceipt,
)
from memory_condense.search.fusion.qwen_features import QwenAtomFeatureProvider
from memory_condense.search.fusion.resident_executor import (
    _ResidentRouteExecution,
    _assert_resident_executor_seams,
    _execute_qwen_resident_route,
)
from memory_condense.search.fusion.resident_models import (
    MatchedEvidenceFusionPair,
    MatchedEvidenceFusionPairReceipt,
    QwenResidentFusionOperationReceipt,
    ResidentEvidenceFusionPlan,
    ResidentRouterRuntimeReceipt,
    resident_atom_order_sha256,
    resident_matched_input_sha256,
    resident_values_sha256,
)


_PINNED_PROVIDER_TYPE = QwenAtomFeatureProvider
_PINNED_PROVIDER_ASSERT_STATE = QwenAtomFeatureProvider._assert_provider_state
_PINNED_PROVIDER_ASSERT_IMPLEMENTATION = (
    QwenAtomFeatureProvider._assert_implementation
)
_PINNED_PROVIDER_ASSERT_RUNTIME = QwenAtomFeatureProvider._assert_runtime
_PINNED_PROVIDER_RECEIPT_GETTER = QwenAtomFeatureProvider.receipt.fget
_PINNED_ROUTER_TYPE = LatentEvidenceRouter
_PINNED_ROUTER_FORWARD_TYPE = LatentRouterForward
_PINNED_ROUTER_ROUTE_ONE = LatentEvidenceRouter.route_one
_PINNED_ROUTER_ASSERT_SEAL = LatentEvidenceRouter._assert_inference_seal
_PINNED_ROUTER_REJECT_GLOBAL_HOOKS = (
    LatentEvidenceRouter._reject_global_runtime_hooks
)
_PINNED_ROUTER_RUNTIME_GETTER = LatentEvidenceRouter.resident_runtime_receipt.fget
_PINNED_ROUTER_STATE_GETTER = LatentEvidenceRouter.state_receipt.fget
_PINNED_ROUTER_RUNTIME_FINGERPRINT = LatentEvidenceRouter._runtime_fingerprint
_PINNED_ROUTER_RUNTIME_STRUCTURE = LatentEvidenceRouter._runtime_structure
_PINNED_ROUTER_REJECT_HOOKS = LatentEvidenceRouter._reject_runtime_hooks
_PINNED_REQUIRE_TORCH = latent_router_module._require_torch
_PINNED_FEATURE_IMPLEMENTATION = qwen_features_module._pinned_owned_implementation()
_PINNED_EXECUTOR_SEAM_ASSERT = _assert_resident_executor_seams
_PINNED_VALUES_SHA256 = resident_values_sha256
_PINNED_ATOM_ORDER_SHA256 = resident_atom_order_sha256
_PINNED_IDENTITY_SHA256 = identity_sha256
_PINNED_RESIDENT_MATCHED_INPUT_SHA256 = resident_matched_input_sha256


def _matched_input_sha256(
    *,
    feature_suboperation: QwenAtomFeatureOperationReceipt,
    router_runtime: ResidentRouterRuntimeReceipt,
    implementation_sha256: str,
) -> str:
    return _PINNED_RESIDENT_MATCHED_INPUT_SHA256(
        feature_suboperation_sha256=feature_suboperation.operation_sha256,
        router_runtime_sha256=router_runtime.runtime_sha256,
        implementation_sha256=implementation_sha256,
    )


_PINNED_MATCHED_INPUT_SHA256 = _matched_input_sha256


@dataclass(frozen=True, slots=True)
class _OwnedResidentImplementation:
    execute: Any
    execution_type: type
    topology_groups: Any
    topology_preflight: Any
    resident_plan_type: type
    operation_type: type
    pair_receipt_type: type
    pair_type: type
    matched_input_sha256: Any


_PINNED_RESIDENT_IMPLEMENTATION = _OwnedResidentImplementation(
    execute=_execute_qwen_resident_route,
    execution_type=_ResidentRouteExecution,
    topology_groups=_topology_atom_groups,
    topology_preflight=_preflight_topology,
    resident_plan_type=ResidentEvidenceFusionPlan,
    operation_type=QwenResidentFusionOperationReceipt,
    pair_receipt_type=MatchedEvidenceFusionPairReceipt,
    pair_type=MatchedEvidenceFusionPair,
    matched_input_sha256=_PINNED_MATCHED_INPUT_SHA256,
)


def _pinned_resident_implementation(
    _value: _OwnedResidentImplementation = _PINNED_RESIDENT_IMPLEMENTATION,
) -> _OwnedResidentImplementation:
    return _value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resident_implementation_sha256(
    provider: QwenAtomFeatureProvider,
) -> str:
    return _PINNED_IDENTITY_SHA256(
        {
            "format": "memory-condense-qwen-resident-matched-implementation-v1",
            "source_files": {
                "qwen_matched": _file_sha256(Path(__file__)),
                "resident_models": _file_sha256(
                    Path(resident_models_module.__file__)
                ),
                "resident_executor": _file_sha256(
                    Path(resident_executor_module.__file__)
                ),
                "planning_core": _file_sha256(Path(planning_core_module.__file__)),
                "latent_router": _file_sha256(Path(latent_router_module.__file__)),
                "tensor_identity": _file_sha256(
                    Path(tensor_identity_module.__file__)
                ),
                "fusion_models": _file_sha256(Path(fusion_models_module.__file__)),
                "qwen_feature_models": _file_sha256(
                    Path(feature_models_module.__file__)
                ),
                "sealed_identity": _file_sha256(
                    Path(sealed_identity_module.__file__)
                ),
                "discourse_identity": _file_sha256(
                    Path(discourse_identity_module.__file__)
                ),
            },
            "feature_provider_implementation_sha256": (
                _PINNED_PROVIDER_RECEIPT_GETTER(provider).implementation_sha256
            ),
            "runtime_versions": {
                "torch": str(getattr(provider._torch, "__version__", "")),
            },
            "router_algorithm": "k_latent_two_pass_cross_attention_v1",
            "feature_workspace_count": 1,
            "topology_reencode_count": 0,
        }
    )


def _resident_owned_objects() -> tuple[Any, ...]:
    return (
        build_qwen_matched_fusion_pair,
        _pinned_resident_implementation,
        _file_sha256,
        _resident_implementation_sha256,
        _matched_input_sha256,
        _snapshot_resident_router,
        _assert_resident_implementation,
        _PINNED_PROVIDER_TYPE,
        _PINNED_PROVIDER_ASSERT_STATE,
        _PINNED_PROVIDER_ASSERT_IMPLEMENTATION,
        _PINNED_PROVIDER_ASSERT_RUNTIME,
        _PINNED_PROVIDER_RECEIPT_GETTER,
        _PINNED_ROUTER_TYPE,
        _PINNED_ROUTER_FORWARD_TYPE,
        _PINNED_ROUTER_FORWARD_TYPE.__init__,
        _PINNED_ROUTER_ROUTE_ONE,
        _PINNED_ROUTER_ASSERT_SEAL,
        _PINNED_ROUTER_REJECT_GLOBAL_HOOKS,
        _PINNED_ROUTER_RUNTIME_GETTER,
        _PINNED_ROUTER_STATE_GETTER,
        _PINNED_ROUTER_RUNTIME_FINGERPRINT,
        _PINNED_ROUTER_RUNTIME_STRUCTURE,
        _PINNED_ROUTER_REJECT_HOOKS,
        _PINNED_REQUIRE_TORCH,
        _PINNED_FEATURE_IMPLEMENTATION,
        _PINNED_EXECUTOR_SEAM_ASSERT,
        _PINNED_VALUES_SHA256,
        _PINNED_ATOM_ORDER_SHA256,
        _PINNED_IDENTITY_SHA256,
        _PINNED_RESIDENT_MATCHED_INPUT_SHA256,
        _PINNED_MATCHED_INPUT_SHA256,
        _PINNED_RESIDENT_IMPLEMENTATION,
        _execute_qwen_resident_route,
        _ResidentRouteExecution,
        _ResidentRouteExecution.__init__,
        _ResidentRouteExecution.__post_init__,
        _topology_atom_groups,
        _preflight_topology,
        planning_core_module._adjacency,
        planning_core_module._float32,
        planning_core_module._matrix_rows,
        planning_core_module._validate_attention_rows,
        planning_core_module._latent_memberships_and_groups,
        ResidentEvidenceFusionPlan,
        ResidentEvidenceFusionPlan.__post_init__,
        ResidentEvidenceFusionPlan.__init__,
        ResidentEvidenceFusionPlan._validate_atom_set,
        ResidentEvidenceFusionPlan._validate_topology,
        ResidentEvidenceFusionPlan._validate_mode,
        ResidentRouterRuntimeReceipt,
        ResidentRouterRuntimeReceipt.__init__,
        ResidentRouterRuntimeReceipt.__post_init__,
        QwenResidentFusionOperationReceipt,
        QwenResidentFusionOperationReceipt.__init__,
        QwenResidentFusionOperationReceipt.__post_init__,
        QwenResidentFusionOperationReceipt._validate_join,
        QwenResidentFusionOperationReceipt._validate_claims,
        MatchedEvidenceFusionPairReceipt,
        MatchedEvidenceFusionPairReceipt.__init__,
        MatchedEvidenceFusionPairReceipt.__post_init__,
        MatchedEvidenceFusionPair,
        MatchedEvidenceFusionPair.__init__,
        MatchedEvidenceFusionPair.__post_init__,
        resident_values_sha256,
        resident_atom_order_sha256,
        resident_matched_input_sha256,
        resident_models_module._positive_int,
        resident_models_module._nonnegative_int,
        resident_models_module._cuda_device,
        resident_models_module._router_dtype,
        resident_models_module._optional_sha256,
        resident_models_module._shape2,
        resident_models_module._exact_values,
        resident_models_module._require_plain_identity_tree,
        sealed_identity_module.SealedIdentity._seal,
        sealed_identity_module.SealedIdentity.identity_payload,
    )


def _resident_owned_fingerprint() -> tuple[Any, ...]:
    return tuple(
        (
            id(value),
            id(getattr(value, "__code__", None)),
            id(getattr(value, "__defaults__", None)),
            tuple(id(item) for item in (getattr(value, "__defaults__", None) or ())),
            tuple(
                sorted(
                    (name, id(item))
                    for name, item in (
                        getattr(value, "__kwdefaults__", None) or {}
                    ).items()
                )
            ),
        )
        for value in _resident_owned_objects()
    )


def _assert_resident_implementation() -> None:
    if _resident_owned_fingerprint() != _RESIDENT_OWNED_FINGERPRINT:
        raise RuntimeError("owned resident fusion implementation was replaced")
    if (
        qwen_features_module.QwenAtomFeatureProvider is not _PINNED_PROVIDER_TYPE
        or _PINNED_PROVIDER_TYPE._assert_provider_state
        is not _PINNED_PROVIDER_ASSERT_STATE
        or _PINNED_PROVIDER_TYPE._assert_implementation
        is not _PINNED_PROVIDER_ASSERT_IMPLEMENTATION
        or _PINNED_PROVIDER_TYPE._assert_runtime
        is not _PINNED_PROVIDER_ASSERT_RUNTIME
        or _PINNED_PROVIDER_TYPE.receipt.fget
        is not _PINNED_PROVIDER_RECEIPT_GETTER
        or latent_router_module.LatentEvidenceRouter is not _PINNED_ROUTER_TYPE
        or latent_router_module.LatentRouterForward
        is not _PINNED_ROUTER_FORWARD_TYPE
        or _PINNED_ROUTER_TYPE.route_one is not _PINNED_ROUTER_ROUTE_ONE
        or _PINNED_ROUTER_TYPE._assert_inference_seal
        is not _PINNED_ROUTER_ASSERT_SEAL
        or _PINNED_ROUTER_TYPE._reject_global_runtime_hooks
        is not _PINNED_ROUTER_REJECT_GLOBAL_HOOKS
        or _PINNED_ROUTER_TYPE.resident_runtime_receipt.fget
        is not _PINNED_ROUTER_RUNTIME_GETTER
        or _PINNED_ROUTER_TYPE.state_receipt.fget
        is not _PINNED_ROUTER_STATE_GETTER
        or _PINNED_ROUTER_TYPE._runtime_fingerprint
        is not _PINNED_ROUTER_RUNTIME_FINGERPRINT
        or _PINNED_ROUTER_TYPE._runtime_structure
        is not _PINNED_ROUTER_RUNTIME_STRUCTURE
        or _PINNED_ROUTER_TYPE._reject_runtime_hooks
        is not _PINNED_ROUTER_REJECT_HOOKS
        or latent_router_module._require_torch is not _PINNED_REQUIRE_TORCH
        or resident_executor_module._execute_qwen_resident_route
        is not _PINNED_RESIDENT_IMPLEMENTATION.execute
        or resident_executor_module._assert_resident_executor_seams
        is not _PINNED_EXECUTOR_SEAM_ASSERT
        or planning_core_module._topology_atom_groups
        is not _PINNED_RESIDENT_IMPLEMENTATION.topology_groups
        or planning_core_module._preflight_topology
        is not _PINNED_RESIDENT_IMPLEMENTATION.topology_preflight
        or resident_models_module.resident_values_sha256
        is not _PINNED_VALUES_SHA256
        or resident_models_module.resident_atom_order_sha256
        is not _PINNED_ATOM_ORDER_SHA256
        or resident_models_module.resident_matched_input_sha256
        is not _PINNED_RESIDENT_MATCHED_INPUT_SHA256
        or resident_models_module.identity_sha256 is not _PINNED_IDENTITY_SHA256
    ):
        raise RuntimeError("owned resident fusion dependency seams were replaced")
    _PINNED_EXECUTOR_SEAM_ASSERT()


def _snapshot_resident_router(
    provider: QwenAtomFeatureProvider,
    router: LatentEvidenceRouter,
    *,
    atom_count: int,
    caps: FusionCaps,
) -> ResidentRouterRuntimeReceipt:
    if type(router) is not _PINNED_ROUTER_TYPE:
        raise TypeError("resident fusion requires the exact owned LatentEvidenceRouter")
    _PINNED_ROUTER_ASSERT_SEAL(router)
    _PINNED_ROUTER_REJECT_GLOBAL_HOOKS(router)
    runtime = _PINNED_ROUTER_RUNTIME_GETTER(router)
    if type(runtime) is not ResidentRouterRuntimeReceipt:
        raise TypeError("resident router omitted its exact runtime receipt")
    runtime._seal()
    feature = _PINNED_PROVIDER_RECEIPT_GETTER(provider)
    feature._seal()
    if router._torch is not provider._torch:
        raise RuntimeError("Qwen and resident router must share one Torch runtime")
    if runtime.device != feature.device:
        raise ValueError("Qwen and resident router must share one CUDA device")
    if runtime.execution_dtype != feature.execution_dtype:
        raise ValueError("Qwen and resident router must share one execution dtype")
    architecture = runtime.architecture
    if architecture.hidden_dim != feature.hidden_dim:
        raise ValueError("Qwen feature width and resident router width differ")
    if atom_count > caps.max_atoms or atom_count > runtime.max_atoms:
        raise MemoryError("resident atom count exceeds an owned bound")
    if architecture.hidden_dim > caps.max_hidden_dim:
        raise MemoryError("resident hidden width exceeds FusionCaps")
    if architecture.num_latents > caps.max_latents:
        raise MemoryError("resident latent count exceeds FusionCaps")
    route_cells = architecture.num_latents * atom_count
    if route_cells > caps.max_route_cells or route_cells > runtime.max_route_cells:
        raise MemoryError("resident K*N exceeds an owned route-cell bound")
    return runtime


def build_qwen_matched_fusion_pair(
    packet: EvidencePacket,
    plan: ClosurePlan,
    *,
    provider: QwenAtomFeatureProvider,
    router: LatentEvidenceRouter,
    caps: FusionCaps,
    feature_caps: QwenAtomFeatureCaps,
) -> MatchedEvidenceFusionPair:
    """Build one topology control and one latent treatment from one GPU workspace."""

    _assert_resident_implementation()
    if type(provider) is not _PINNED_PROVIDER_TYPE:
        raise TypeError("provider must be the exact owned QwenAtomFeatureProvider")
    if type(router) is not _PINNED_ROUTER_TYPE:
        raise TypeError("router must be the exact owned LatentEvidenceRouter")
    if type(caps) is not FusionCaps:
        raise TypeError("caps must be an exact FusionCaps")
    if type(feature_caps) is not QwenAtomFeatureCaps:
        raise TypeError("feature_caps must be an exact QwenAtomFeatureCaps")
    caps._seal()
    feature_caps._seal()
    _PINNED_PROVIDER_ASSERT_STATE(provider)
    provider_receipt = _PINNED_PROVIDER_RECEIPT_GETTER(provider)
    encoder = provider._encoder
    feature_owned = provider._implementation
    if feature_owned is not _PINNED_FEATURE_IMPLEMENTATION:
        raise RuntimeError("provider does not retain the pinned feature implementation")
    resident_owned = _pinned_resident_implementation()
    implementation_sha256 = _resident_implementation_sha256(provider)

    rows: tuple[Any, ...] = ()
    batches: tuple[tuple[int, int, int, int], ...] = ()
    execution: _ResidentRouteExecution | None = None
    with provider._gate_factory(encoder) as gate_token:
        _PINNED_PROVIDER_ASSERT_STATE(provider)
        feature_owned.preflight_packet(
            packet,
            plan,
            caps,
            feature_caps,
            hidden_dim=provider_receipt.hidden_dim,
        )
        router_runtime = _snapshot_resident_router(
            provider,
            router,
            atom_count=len(packet.atoms),
            caps=caps,
        )
        feature_owned.validate_packet_plan(packet, plan)
        inputs = feature_owned.capture_operation_inputs(
            packet,
            plan,
            caps,
            feature_caps,
        )
        atom_ids = tuple(item.atom_id for item in inputs.atoms)
        topology_groups, topology_order, topology_degree = (
            resident_owned.topology_groups(
                atom_ids,
                inputs.hyperedges,
                inputs.caps,
            )
        )
        resident_owned.topology_preflight(
            inputs.hyperedges,
            topology_groups,
            inputs.caps,
        )
        _PINNED_PROVIDER_ASSERT_IMPLEMENTATION(provider)
        _PINNED_PROVIDER_ASSERT_RUNTIME(provider)
        try:
            rows = feature_owned.build_atom_rows(
                encoder.tokenizer,
                inputs.atom_values,
                inputs.query,
                inputs.feature_caps,
            )
            batches = feature_owned.batch_rows(rows, inputs.feature_caps)
            batch_receipts = tuple(
                feature_owned.batch_receipt_type(
                    batch_index=index,
                    start_row=start,
                    row_count=count,
                    padded_width=width,
                    padded_workspace_tokens=workspace,
                )
                for index, (start, count, width, workspace) in enumerate(batches)
            )
            execution = resident_owned.execute(
                encoder=encoder,
                torch=provider._torch,
                output_layer=provider._output_layer,
                provider_receipt=provider_receipt,
                feature_caps=inputs.feature_caps,
                rows=rows,
                batches=batches,
                gate_token=gate_token,
                router=router,
                router_runtime=router_runtime,
                atom_ids=atom_ids,
                topology_degree=topology_degree,
                caps=inputs.caps,
            )
            if type(execution) is not resident_owned.execution_type:
                raise RuntimeError("resident executor returned an unsupported artifact")
            row_receipts = tuple(row.receipt for row in rows)
            primary_forward_count = execution.primary_forward_count
            router_forward_count = execution.router_forward_count
            extraction_sha256 = execution.extraction_matrix_sha256
            reinjection_sha256 = execution.reinjection_matrix_sha256
            extraction_shape = execution.extraction_shape
            reinjection_shape = execution.reinjection_shape
            memberships = execution.memberships
            latent_groups = execution.groups
            latent_order = execution.atom_order
            route_dtype = execution.route_matrix_canonical_dtype
        finally:
            execution = None
            rows = ()
            batches = ()
            feature_owned.revalidate_operation_inputs(
                inputs,
                packet,
                plan,
                caps,
                feature_caps,
                hidden_dim=provider_receipt.hidden_dim,
            )
            _assert_resident_implementation()
            _PINNED_PROVIDER_ASSERT_IMPLEMENTATION(provider)
            _PINNED_PROVIDER_ASSERT_RUNTIME(provider)
            if _snapshot_resident_router(
                provider,
                router,
                atom_count=len(inputs.atoms),
                caps=inputs.caps,
            ) != router_runtime:
                raise RuntimeError("resident router runtime changed during execution")
            if _resident_implementation_sha256(provider) != implementation_sha256:
                raise RuntimeError("resident implementation digest changed during execution")

        feature_suboperation = feature_owned.operation_receipt_type(
            packet_receipt_sha256=inputs.packet_receipt_sha256,
            closure_plan_sha256=inputs.closure_plan_sha256,
            query_program_sha256=inputs.query_program_sha256,
            query_sha256=inputs.query_sha256,
            closure_policy_sha256=inputs.closure_policy_sha256,
            snapshot_sha256=inputs.snapshot_sha256,
            caps=inputs.caps,
            feature_caps=inputs.feature_caps,
            provider=provider_receipt,
            atoms=inputs.atoms,
            hyperedges=inputs.hyperedges,
            rows=row_receipts,
            batches=batch_receipts,
            feature_shape=(len(inputs.atoms), provider_receipt.hidden_dim),
            feature_device=provider_receipt.device,
            feature_execution_dtype=provider_receipt.execution_dtype,
            qwen_forward_count=primary_forward_count,
            primary_qwen_forward_count=primary_forward_count,
            batch_invariance_forward_count=0,
            runtime_batch_invariance_attested=False,
            max_observed_padded_workspace_tokens=max(
                batch.padded_workspace_tokens for batch in batch_receipts
            ),
        )
        if type(feature_suboperation) is not QwenAtomFeatureOperationReceipt:
            raise RuntimeError("resident builder omitted the exact feature sub-operation")
        matched_input = resident_owned.matched_input_sha256(
            feature_suboperation=feature_suboperation,
            router_runtime=router_runtime,
            implementation_sha256=implementation_sha256,
        )
        topology_plan = resident_owned.resident_plan_type(
            mode="topology_only",
            feature_suboperation_sha256=feature_suboperation.operation_sha256,
            matched_input_sha256=matched_input,
            caps=inputs.caps,
            atoms=inputs.atoms,
            hyperedges=inputs.hyperedges,
            memberships=(),
            groups=topology_groups,
            atom_order=topology_order,
        )
        latent_plan = resident_owned.resident_plan_type(
            mode="latent_router",
            feature_suboperation_sha256=feature_suboperation.operation_sha256,
            matched_input_sha256=matched_input,
            caps=inputs.caps,
            atoms=inputs.atoms,
            hyperedges=inputs.hyperedges,
            memberships=memberships,
            groups=latent_groups,
            atom_order=latent_order,
            router_runtime=router_runtime,
            extraction_matrix_sha256=extraction_sha256,
            reinjection_matrix_sha256=reinjection_sha256,
            extraction_shape=extraction_shape,
            reinjection_shape=reinjection_shape,
        )
        operation = resident_owned.operation_type(
            feature_suboperation=feature_suboperation,
            implementation_sha256=implementation_sha256,
            router_runtime=router_runtime,
            matched_input_sha256=matched_input,
            topology_plan_sha256=topology_plan.plan_sha256,
            latent_plan_sha256=latent_plan.plan_sha256,
            extraction_matrix_sha256=extraction_sha256,
            reinjection_matrix_sha256=reinjection_sha256,
            extraction_shape=extraction_shape,
            reinjection_shape=reinjection_shape,
            route_matrix_canonical_dtype=route_dtype,
            topology_groups_sha256=_PINNED_VALUES_SHA256(
                "topology_groups", topology_groups
            ),
            latent_memberships_sha256=_PINNED_VALUES_SHA256(
                "latent_memberships", memberships
            ),
            latent_groups_sha256=_PINNED_VALUES_SHA256(
                "latent_groups", latent_groups
            ),
            latent_atom_order_sha256=_PINNED_ATOM_ORDER_SHA256(latent_order),
            router_forward_count=router_forward_count,
        )
        pair_receipt = resident_owned.pair_receipt_type(
            operation_sha256=operation.operation_sha256,
            feature_suboperation_sha256=feature_suboperation.operation_sha256,
            topology_plan_sha256=topology_plan.plan_sha256,
            latent_plan_sha256=latent_plan.plan_sha256,
            matched_input_sha256=matched_input,
        )
        return resident_owned.pair_type(
            topology_only=topology_plan,
            latent_router=latent_plan,
            operation=operation,
            receipt=pair_receipt,
        )


_RESIDENT_OWNED_FINGERPRINT = _resident_owned_fingerprint()


__all__ = ["build_qwen_matched_fusion_pair"]
