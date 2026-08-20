"""Build provenance-safe extractive plans over an already packed packet."""

from __future__ import annotations

from typing import Any

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain.discourse import ClosurePlan, EvidencePacket
from memory_condense.search.fusion.feature_batch import NodeFeatureBatch
from memory_condense.search.fusion.models import (
    AuthoritativeHyperedge,
    EvidenceFusionPlan,
    FusionAtomRef,
    FusionCaps,
    FusionMode,
    RouterArchitectureReceipt,
    RouterStateReceipt,
)
from memory_condense.search.fusion.planning_core import (
    _latent_memberships_and_groups,
    _preflight_topology,
    _topology_atom_groups,
)
from memory_condense.search.fusion.tensor_identity import (
    CanonicalTensor,
    canonical_float32_tensor,
    tensor_shape,
)


def _validate_packet_plan(packet: EvidencePacket, plan: ClosurePlan) -> None:
    if not isinstance(packet, EvidencePacket):
        raise TypeError("packet must be EvidencePacket")
    if not isinstance(plan, ClosurePlan):
        raise TypeError("plan must be ClosurePlan")
    if packet.receipt.plan_sha256 != plan.plan_sha256:
        raise ValueError("packet receipt does not bind the supplied closure plan")

    plan_atoms = {item.atom_id: item for item in plan.atoms}
    if len(plan_atoms) != len(plan.atoms):  # pragma: no cover - plan guards this
        raise ValueError("closure plan atom IDs are not unique")
    for atom in packet.atoms:
        planned = plan_atoms.get(atom.atom_id)
        if planned is None or identity_sha256(atom.identity_payload()) != identity_sha256(
            planned.identity_payload()
        ):
            raise ValueError("packet atom does not exactly match the closure plan")

    plan_bundles = {item.bundle_id: item for item in plan.bundles}
    selected_atoms = {item.atom_id for item in packet.atoms}
    for bundle in packet.bundles:
        planned = plan_bundles.get(bundle.bundle_id)
        if planned is None or identity_sha256(bundle.identity_payload()) != identity_sha256(
            planned.identity_payload()
        ):
            raise ValueError("packet bundle does not exactly match the closure plan")
        if any(atom_id not in selected_atoms for atom_id in bundle.atom_ids):
            raise ValueError("packet bundle references an unselected atom")


def _atom_refs(packet: EvidencePacket) -> tuple[FusionAtomRef, ...]:
    return tuple(
        FusionAtomRef(
            atom_id=atom.atom_id,
            atom_identity_sha256=identity_sha256(atom.identity_payload()),
            span_identity_sha256=identity_sha256(atom.span.identity_payload()),
            quote_sha256=atom.span.quote_sha256,
        )
        for atom in packet.atoms
    )


def _authoritative_hyperedges(
    packet: EvidencePacket,
) -> tuple[AuthoritativeHyperedge, ...]:
    """Project only the selected bundle hypergraph; infer no relation direction."""

    return tuple(
        AuthoritativeHyperedge(
            bundle_id=bundle.bundle_id,
            atom_ids=bundle.atom_ids,
            obligation_ids=bundle.obligation_ids,
            unit_witness_ids=bundle.unit_ids,
            relation_witness_ids=bundle.relation_ids,
            required=bundle.required,
            utility=bundle.utility,
        )
        for bundle in packet.bundles
    )


def _preflight_features(
    node_features: Any,
    *,
    atom_count: int,
    caps: FusionCaps,
) -> tuple[int, int]:
    shape = tensor_shape(node_features, label="node_features")
    if len(shape) != 2:
        raise ValueError("node_features must have shape [N, D]")
    nodes, width = shape
    if nodes != atom_count:
        raise ValueError("node_features must preserve the exact packet atom set")
    if nodes < 1 or width < 1:
        raise ValueError("node_features must have positive N and D")
    if nodes > caps.max_atoms:
        raise MemoryError("node feature count exceeds fusion max_atoms")
    if width > caps.max_hidden_dim:
        raise MemoryError("node feature width exceeds fusion max_hidden_dim")
    return nodes, width


def _matched_input_sha256(
    *,
    packet: EvidencePacket,
    plan: ClosurePlan,
    caps: FusionCaps,
    atoms: tuple[FusionAtomRef, ...],
    hyperedges: tuple[AuthoritativeHyperedge, ...],
    node_feature_receipt_sha256: str,
    node_features: CanonicalTensor,
) -> str:
    return identity_sha256(
        {
            "packet_receipt_sha256": packet.receipt.receipt_sha256,
            "closure_plan_sha256": plan.plan_sha256,
            "query_program_sha256": plan.query_program.program_sha256,
            "query_sha256": quote_sha256(plan.query_program.query),
            "closure_policy_sha256": plan.policy.policy_sha256,
            "snapshot_sha256": plan.snapshot.snapshot_sha256,
            "caps": caps.identity_payload(),
            "atoms": [item.identity_payload() for item in atoms],
            "hyperedges": [item.identity_payload() for item in hyperedges],
            "node_feature_receipt_sha256": node_feature_receipt_sha256,
            "node_features_sha256": node_features.tensor_sha256,
        }
    )


def build_evidence_fusion_plan(
    packet: EvidencePacket,
    plan: ClosurePlan,
    *,
    node_features: NodeFeatureBatch,
    mode: FusionMode = "topology_only",
    caps: FusionCaps | None = None,
    router: Any | None = None,
) -> EvidenceFusionPlan:
    """Plan an extractive atom order without changing or generating evidence.

    ``node_features`` must follow the packet's exact atom order.  Both modes
    bind the same feature tensor so a matched control differs only by whether
    the exact two-pass latent router is invoked.
    """

    if not isinstance(packet, EvidencePacket):
        raise TypeError("packet must be EvidencePacket")
    if not isinstance(plan, ClosurePlan):
        raise TypeError("plan must be ClosurePlan")
    if mode not in {"topology_only", "latent_router"}:
        raise ValueError("mode must be topology_only or latent_router")
    active_caps = FusionCaps() if caps is None else caps
    if type(active_caps) is not FusionCaps:
        raise TypeError("caps must be FusionCaps")
    if len(packet.atoms) > active_caps.max_atoms:
        raise MemoryError("packet atom count exceeds fusion max_atoms")
    if len(packet.bundles) > active_caps.max_hyperedges:
        raise MemoryError("packet bundle count exceeds fusion max_hyperedges")
    raw_topology_links = sum(
        len(bundle.atom_ids) * (len(bundle.atom_ids) - 1) // 2
        for bundle in packet.bundles
    )
    if raw_topology_links > active_caps.max_topology_links:
        raise MemoryError("packet bundle co-memberships exceed max_topology_links")
    _validate_packet_plan(packet, plan)
    if type(node_features) is not NodeFeatureBatch:
        raise TypeError("node_features must be an owned NodeFeatureBatch")
    feature_receipt = node_features.receipt
    expected_atom_ids = tuple(item.atom_id for item in packet.atoms)
    if feature_receipt.ordered_atom_ids != expected_atom_ids:
        raise ValueError("node feature receipt atom order does not match packet")
    if feature_receipt.query_sha256 != quote_sha256(plan.query_program.query):
        raise ValueError("node feature receipt query does not match closure query")
    atom_count, hidden_dim = _preflight_features(
        node_features.values,
        atom_count=len(packet.atoms),
        caps=active_caps,
    )
    if feature_receipt.tensor_shape != (atom_count, hidden_dim):
        raise ValueError("node feature values do not match their sealed receipt")
    snapshot = node_features.detached_snapshot()
    features = canonical_float32_tensor(
        snapshot,
        label="node_features",
        retain_values=False,
    )
    if (
        feature_receipt.tensor_sha256 != features.tensor_sha256
        or feature_receipt.tensor_shape != features.shape
        or feature_receipt.tensor_dtype != features.dtype
    ):
        raise ValueError("node feature values do not match their sealed receipt")
    atoms = _atom_refs(packet)
    atom_ids = tuple(item.atom_id for item in atoms)
    hyperedges = _authoritative_hyperedges(packet)
    topology_groups, topology_order, degree = _topology_atom_groups(
        atom_ids,
        hyperedges,
        active_caps,
    )
    _preflight_topology(hyperedges, topology_groups, active_caps)
    matched_input = _matched_input_sha256(
        packet=packet,
        plan=plan,
        caps=active_caps,
        atoms=atoms,
        hyperedges=hyperedges,
        node_feature_receipt_sha256=feature_receipt.feature_receipt_sha256,
        node_features=features,
    )
    common = {
        "packet_receipt_sha256": packet.receipt.receipt_sha256,
        "closure_plan_sha256": plan.plan_sha256,
        "query_program_sha256": plan.query_program.program_sha256,
        "query_sha256": quote_sha256(plan.query_program.query),
        "closure_policy_sha256": plan.policy.policy_sha256,
        "snapshot_sha256": plan.snapshot.snapshot_sha256,
        "matched_input_sha256": matched_input,
        "caps": active_caps,
        "atoms": atoms,
        "hyperedges": hyperedges,
        "node_features": feature_receipt,
        "plan_retained_request_tensor_bytes": 0,
    }
    if mode == "topology_only":
        return EvidenceFusionPlan(
            mode=mode,
            memberships=(),
            groups=topology_groups,
            atom_order=topology_order,
            **common,
        )

    if router is None:
        raise ValueError("latent_router mode requires an exact latent router")
    from memory_condense.search.fusion.latent_router import LatentEvidenceRouter

    if type(router) is not LatentEvidenceRouter:
        raise TypeError("latent_router mode requires the owned LatentEvidenceRouter")
    architecture = getattr(router, "architecture_receipt", None)
    if type(architecture) is not RouterArchitectureReceipt:
        raise TypeError("router must expose RouterArchitectureReceipt")
    if architecture.hidden_dim != hidden_dim:
        raise ValueError("router hidden_dim does not match node features")
    if architecture.num_latents > active_caps.max_latents:
        raise MemoryError("router latent count exceeds fusion max_latents")
    if architecture.num_latents * atom_count > active_caps.max_route_cells:
        raise MemoryError("K*N exceeds fusion max_route_cells")
    if int(getattr(router, "max_atoms", 0)) < atom_count:
        raise MemoryError("router max_atoms is below the packet atom count")
    if int(getattr(router, "max_hidden_dim", 0)) < hidden_dim:
        raise MemoryError("router max_hidden_dim is below the feature width")
    if int(getattr(router, "max_route_cells", 0)) < architecture.num_latents * atom_count:
        raise MemoryError("router max_route_cells is below K*N")
    state_before = getattr(router, "state_receipt", None)
    if type(state_before) is not RouterStateReceipt:
        raise TypeError("router must expose RouterStateReceipt")
    if state_before.parameter_count != architecture.parameter_count:
        raise ValueError("router state parameter count disagrees with architecture")
    route_one = getattr(router, "route_one", None)
    if not callable(route_one):
        raise TypeError("router must expose route_one(node_features)")
    routed = route_one(snapshot)
    from memory_condense.search.fusion.latent_router import LatentRouterForward

    if type(routed) is not LatentRouterForward:
        raise TypeError("owned router returned an unsupported result")
    route_source_dtypes = {
        str(getattr(value, "dtype", ""))
        for value in (
            routed.steered_nodes,
            routed.extraction_attention,
            routed.reinjection_attention,
        )
    }
    if len(route_source_dtypes) != 1:
        raise ValueError("router outputs must share one execution dtype")
    route_source_dtype = next(iter(route_source_dtypes))
    features_after = canonical_float32_tensor(
        snapshot,
        label="node_features",
        retain_values=False,
    )
    if features_after.tensor_sha256 != features.tensor_sha256:
        raise RuntimeError("router mutated the node feature snapshot")

    steered = canonical_float32_tensor(
        getattr(routed, "steered_nodes", None),
        label="steered_nodes",
        retain_values=False,
    )
    extraction = canonical_float32_tensor(
        getattr(routed, "extraction_attention", None),
        label="extraction_attention",
    )
    reinjection = canonical_float32_tensor(
        getattr(routed, "reinjection_attention", None),
        label="reinjection_attention",
    )
    expected_extraction = (architecture.num_latents, atom_count)
    if steered.shape != features.shape:
        raise ValueError("steered node shape must remain [N, D]")
    if extraction.shape != expected_extraction:
        raise ValueError("extraction attention must have shape [K, N]")
    if reinjection.shape != (atom_count, architecture.num_latents):
        raise ValueError("reinjection attention must have shape [N, K]")
    memberships, groups, atom_order = _latent_memberships_and_groups(
        atom_ids,
        extraction,
        reinjection,
        degree,
        active_caps,
        source_dtype=route_source_dtype,
    )
    result = EvidenceFusionPlan(
        mode=mode,
        memberships=memberships,
        groups=groups,
        atom_order=atom_order,
        router_architecture=architecture,
        router_state=state_before,
        steered_features_sha256=steered.tensor_sha256,
        extraction_matrix_sha256=extraction.tensor_sha256,
        reinjection_matrix_sha256=reinjection.tensor_sha256,
        extraction_shape=extraction.shape,
        reinjection_shape=reinjection.shape,
        **common,
    )
    # No tensor or full matrix is reachable from the sealed result.
    del routed, steered, extraction, reinjection, features_after, snapshot
    return result


def validate_matched_fusion_pair(
    topology_only: EvidenceFusionPlan,
    latent_router: EvidenceFusionPlan,
) -> str:
    """Fail unless two plans form a single-factor matched ablation pair."""

    if type(topology_only) is not EvidenceFusionPlan or type(
        latent_router
    ) is not EvidenceFusionPlan:
        raise TypeError("matched fusion values must be EvidenceFusionPlan")
    if topology_only.mode != "topology_only" or latent_router.mode != "latent_router":
        raise ValueError("matched pair must be topology_only then latent_router")
    shared = (
        "packet_receipt_sha256",
        "closure_plan_sha256",
        "query_program_sha256",
        "query_sha256",
        "closure_policy_sha256",
        "snapshot_sha256",
        "matched_input_sha256",
        "caps",
        "atoms",
        "hyperedges",
        "node_features",
    )
    if any(
        getattr(topology_only, name) != getattr(latent_router, name)
        for name in shared
    ):
        raise ValueError("fusion plans are not a matched single-factor pair")
    return topology_only.matched_input_sha256


__all__ = ["build_evidence_fusion_plan", "validate_matched_fusion_pair"]
