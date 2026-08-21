from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from memory_condense.domain._discourse_identity import (
    canonical_json,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    ClosureScopeWitness,
    DiscourseSnapshot,
    EvidenceAtom,
    EvidenceBundle,
    EvidenceObligation,
    EvidencePacket,
    EvidenceSpan,
    ObligationResult,
    QueryProgram,
)
from memory_condense.search.fusion.models import FusionCaps
from memory_condense.search.fusion.planner import (
    _atom_refs,
    _authoritative_hyperedges,
)
from memory_condense.search.fusion.resident_models import resident_values_sha256
from memory_condense.search.fusion.training_targets import (
    AtomPositionPairTarget,
    DirectCoBundleNeighborhood,
    LatentRouterStructuralTargetReceipt,
    LatentRouterStructuralTargets,
    build_latent_router_structural_targets,
)


def _atom(
    position: int,
    text: str,
    atom_id: str,
    *,
    provenance_variant: bool = False,
) -> EvidenceAtom:
    prefix = "changed-provenance" if provenance_variant else "target-fixture"
    role = "assistant" if provenance_variant else "user"
    span = EvidenceSpan(
        chunk_id=f"{prefix}-chunk-{position}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=position + 1,
        source_id=f"{prefix}-source",
        turn_start_char=position * (17 if provenance_variant else 11),
        turn_id=f"{prefix}-turn-{position}",
        role=role,
        created_at=f"2026-08-{position + 1:02d}T00:00:00Z",
    )
    return EvidenceAtom(
        atom_id=atom_id,
        span=span,
        text=text,
        label=f"SECRET-LABEL-{position}",
    )


def _fixture(
    groups: tuple[tuple[int, ...], ...],
    *,
    texts: tuple[str, ...] | None = None,
    atom_ids: tuple[str, ...] | None = None,
    bundle_ids: tuple[str, ...] | None = None,
    metadata_variant: bool = False,
    provenance_variant: bool = False,
) -> tuple[EvidencePacket, ClosurePlan]:
    atom_count = 1 + max(position for group in groups for position in group)
    if texts is None:
        texts = tuple(
            f"Sensitive structural fixture evidence number {position}."
            for position in range(atom_count)
        )
    if atom_ids is None:
        atom_ids = tuple(f"opaque-atom-{position}" for position in range(atom_count))
    if bundle_ids is None:
        bundle_ids = tuple(f"opaque-bundle-{position:03d}" for position in range(len(groups)))
    assert len(texts) == len(atom_ids) == atom_count
    assert len(bundle_ids) == len(groups)
    atoms = tuple(
        _atom(
            position,
            text,
            atom_ids[position],
            provenance_variant=provenance_variant,
        )
        for position, text in enumerate(texts)
    )

    if metadata_variant:
        obligations = (
            EvidenceObligation(
                obligation_id="answer",
                kind="changed-answer-kind",
                required=False,
                weight=3.0,
                dependencies=("auxiliary",),
            ),
            EvidenceObligation(
                obligation_id="auxiliary",
                kind="changed-auxiliary-kind",
                required=False,
                weight=2.0,
            ),
        )
        selected_obligation = "auxiliary"
        unit_id = "unit-metadata-variant"
        relation_id = "relation-metadata-variant"
        required = False
        utility = 73.25
        results = (
            ObligationResult(obligation_id="answer", status="not_found"),
            ObligationResult(
                obligation_id="auxiliary",
                status="satisfied",
                unit_ids=(unit_id,),
                relation_ids=(relation_id,),
                bundle_ids=bundle_ids,
            ),
        )
    else:
        obligations = (
            EvidenceObligation(
                obligation_id="answer",
                kind="answer_fact",
                required=True,
                weight=1.0,
            ),
            EvidenceObligation(
                obligation_id="auxiliary",
                kind="supporting_fact",
                required=False,
                weight=0.5,
                dependencies=("answer",),
            ),
        )
        selected_obligation = "answer"
        unit_id = "unit-original"
        relation_id = "relation-original"
        required = True
        utility = 1.0
        results = (
            ObligationResult(
                obligation_id="answer",
                status="satisfied",
                unit_ids=(unit_id,),
                relation_ids=(relation_id,),
                bundle_ids=bundle_ids,
            ),
            ObligationResult(obligation_id="auxiliary", status="not_found"),
        )
    bundles = tuple(
        EvidenceBundle(
            bundle_id=bundle_id,
            atom_ids=tuple(atoms[position].atom_id for position in group),
            obligation_ids=(selected_obligation,),
            unit_ids=(unit_id,),
            relation_ids=(relation_id,),
            required=required,
            utility=utility,
        )
        for bundle_id, group in zip(bundle_ids, groups, strict=True)
    )
    program = QueryProgram(
        query="SECRET QUERY: derive only structure, never this prose.",
        intent="structural-fixture",
        subject_terms=("SECRET-SUBJECT",),
        obligations=obligations,
    )
    plan = ClosurePlan(
        query_program=program,
        policy=ClosurePolicy(max_bundles=max(128, len(bundles)), beam_width=256),
        snapshot=DiscourseSnapshot(
            max_turn_ordinal=atom_count,
            chunk_count=atom_count,
            graph_revision=1,
            schema_version=1,
            artifact_ids=("target-fixture-artifact",),
            source_content_sha256="1" * 64,
            graph_content_sha256="2" * 64,
        ),
        seeds=(),
        atoms=atoms,
        bundles=bundles,
        obligation_results=results,
        visited_episode_ids=(),
        visited_unit_ids=(unit_id,),
        visited_relation_ids=(relation_id,),
        stopping_reason="complete",
        complete_claimed=True,
        scope_witnesses=(
            ClosureScopeWitness(
                kind="structural_fixture_scope",
                subject_id="target-fixture",
                requested_limit=atom_count,
                returned_count=atom_count,
                exhaustive=True,
            ),
        ),
        artifact_id="target-fixture-artifact",
    )
    context = "SECRET PACKED CONTEXT that must never enter target receipts."
    receipt = ClosureReceipt(
        plan_sha256=plan.plan_sha256,
        context_sha256=quote_sha256(context),
        selected_bundle_ids=tuple(bundle.bundle_id for bundle in plan.bundles),
        selected_atom_ids=tuple(atom.atom_id for atom in plan.atoms),
        dropped_bundle_reasons={},
        context_token_proxy=8,
        max_context_token_proxy=128,
        tokenizer_identity="fixture-tokenizer",
        stopping_reason="complete",
        complete_claimed=True,
    )
    return EvidencePacket(context, plan.atoms, plan.bundles, receipt), plan


def _caps(
    packet: EvidencePacket,
    **updates: int,
) -> FusionCaps:
    values = {
        "max_atoms": len(packet.atoms),
        "max_hyperedges": max(1, len(packet.bundles)),
        "max_topology_links": sum(
            len(bundle.atom_ids) * (len(bundle.atom_ids) - 1) // 2
            for bundle in packet.bundles
        ),
    }
    values.update(updates)
    return FusionCaps(**values)


def _build(packet: EvidencePacket, plan: ClosurePlan, **caps: int):
    return build_latent_router_structural_targets(
        packet,
        plan,
        caps=_caps(packet, **caps),
    )


def _coordinates(
    pairs: tuple[AtomPositionPairTarget, ...],
) -> tuple[tuple[int, int], ...]:
    return tuple((item.left_position, item.right_position) for item in pairs)


def _reseal_receipt(receipt: ClosureReceipt, **updates: object) -> None:
    for name, value in updates.items():
        object.__setattr__(receipt, name, value)
    object.__setattr__(receipt, "receipt_sha256", "")
    receipt._seal()


def _rebind_plan_receipt(packet: EvidencePacket, plan: ClosurePlan) -> None:
    object.__setattr__(plan, "plan_sha256", "")
    plan._seal()
    _reseal_receipt(packet.receipt, plan_sha256=plan.plan_sha256)


def test_overlapping_bundles_deduplicate_sorted_pairs_and_build_neighborhoods() -> None:
    packet, plan = _fixture(((0, 1), (0, 1), (1, 2)))

    receipt = _build(packet, plan)
    targets = receipt.structural_targets

    assert type(receipt) is LatentRouterStructuralTargetReceipt
    assert _coordinates(targets.positive_pairs) == ((0, 1), (1, 2))
    assert _coordinates(targets.negative_pairs) == ((0, 2),)
    assert tuple(item.direct_co_bundle_target for item in targets.positive_pairs) == (
        1,
        1,
    )
    assert tuple(item.direct_co_bundle_target for item in targets.negative_pairs) == (0,)
    assert tuple(item.member_positions for item in targets.neighborhoods) == (
        (0, 1),
        (0, 1, 2),
        (1, 2),
    )
    assert targets.positive_pair_count == 2
    assert targets.negative_pair_count == 1
    assert receipt.packet_receipt_sha256 == packet.receipt.receipt_sha256
    assert receipt.closure_plan_sha256 == plan.plan_sha256
    assert receipt.ordered_atom_refs_sha256 == resident_values_sha256(
        "latent_router_training_packet_atom_refs",
        _atom_refs(packet),
    )
    assert receipt.authoritative_hyperedges_sha256 == resident_values_sha256(
        "latent_router_training_authoritative_hyperedges",
        _authoritative_hyperedges(packet),
    )
    assert targets.positive_pair_sequence_sha256 == resident_values_sha256(
        "latent_router_training_positive_pairs",
        targets.positive_pairs,
    )
    assert targets.negative_pair_sequence_sha256 == resident_values_sha256(
        "latent_router_training_negative_pairs",
        targets.negative_pairs,
    )
    assert receipt.ordered_atom_refs_sha256 != resident_values_sha256(
        "latent_router_training_packet_atom_refs",
        tuple(reversed(_atom_refs(packet))),
    )
    assert receipt.authoritative_hyperedges_sha256 != resident_values_sha256(
        "latent_router_training_authoritative_hyperedges",
        tuple(reversed(_authoritative_hyperedges(packet))),
    )
    assert targets.positive_pair_sequence_sha256 != resident_values_sha256(
        "latent_router_training_positive_pairs",
        tuple(reversed(targets.positive_pairs)),
    )
    assert targets.target_sha256 == identity_sha256(
        targets.identity_payload(include_receipt=False)
    )
    assert receipt.target_receipt_sha256 == identity_sha256(
        receipt.identity_payload(include_receipt=False)
    )


def test_singleton_and_duplicate_text_atoms_remain_distinct_positions() -> None:
    packet, plan = _fixture(((0,),))
    singleton = _build(packet, plan).structural_targets
    assert singleton.atom_count == 1
    assert singleton.positive_pairs == singleton.negative_pairs == ()
    assert singleton.positive_pair_sequence_sha256 != (
        singleton.negative_pair_sequence_sha256
    )
    assert tuple(item.member_positions for item in singleton.neighborhoods) == ((0,),)

    duplicate = "The exact same evidence bytes appear twice."
    packet, plan = _fixture(((0,), (1,)), texts=(duplicate, duplicate))
    targets = _build(packet, plan).structural_targets
    assert packet.atoms[0].text == packet.atoms[1].text
    assert packet.atoms[0].atom_id != packet.atoms[1].atom_id
    assert _coordinates(targets.positive_pairs) == ()
    assert _coordinates(targets.negative_pairs) == ((0, 1),)
    assert tuple(item.member_positions for item in targets.neighborhoods) == ((0,), (1,))


def test_pair_sequence_digests_and_combined_target_reject_tampering() -> None:
    packet, plan = _fixture(((0, 1), (1, 2)))
    receipt = _build(packet, plan)
    targets = receipt.structural_targets

    with pytest.raises(ValueError, match="canonical sequence"):
        replace(
            targets,
            positive_pair_sequence_sha256="0" * 64,
            target_sha256="",
        )
    with pytest.raises(ValueError, match="target SHA-256"):
        replace(targets, target_sha256="0" * 64)
    with pytest.raises(ValueError, match="target receipt SHA-256"):
        replace(receipt, ordered_atom_refs_sha256="0" * 64)


@pytest.mark.parametrize(
    ("left", "right", "match"),
    (
        (-1, 1, "non-negative"),
        (1, 0, "left_position < right_position"),
        (0, 2, "out-of-range"),
    ),
)
def test_aggregate_reconstructs_and_rejects_invalid_pair_children(
    left: int,
    right: int,
    match: str,
) -> None:
    pair = AtomPositionPairTarget(0, 1, 0)
    object.__setattr__(pair, "left_position", left)
    object.__setattr__(pair, "right_position", right)
    object.__setattr__(pair, "pair_sha256", "")

    with pytest.raises(ValueError, match=match):
        LatentRouterStructuralTargets(
            atom_count=2,
            positive_pairs=(),
            negative_pairs=(pair,),
            neighborhoods=(
                DirectCoBundleNeighborhood(0, (0,)),
                DirectCoBundleNeighborhood(1, (1,)),
            ),
            positive_pair_count=0,
            negative_pair_count=1,
        )


def test_aggregate_reconstructs_and_rejects_a_stale_valid_pair_seal() -> None:
    pair = AtomPositionPairTarget(0, 1, 0)
    object.__setattr__(pair, "direct_co_bundle_target", 1)

    with pytest.raises(ValueError, match="atom-position pair SHA-256"):
        LatentRouterStructuralTargets(
            atom_count=2,
            positive_pairs=(pair,),
            negative_pairs=(),
            neighborhoods=(
                DirectCoBundleNeighborhood(0, (0, 1)),
                DirectCoBundleNeighborhood(1, (0, 1)),
            ),
            positive_pair_count=1,
            negative_pair_count=0,
        )


def test_outer_receipt_reconstructs_stale_aggregate_and_neighborhood_trees() -> None:
    packet, plan = _fixture(((0, 1),))
    receipt = _build(packet, plan)
    targets = receipt.structural_targets
    object.__setattr__(targets, "positive_pairs", ())
    object.__setattr__(targets, "positive_pair_count", 0)
    object.__setattr__(targets, "positive_pair_sequence_sha256", "")
    object.__setattr__(targets, "target_sha256", "")
    with pytest.raises(ValueError, match="exhaust the unordered complement"):
        replace(
            receipt,
            structural_targets=targets,
            target_receipt_sha256="",
        )

    packet, plan = _fixture(((0, 1), (1, 2)))
    targets = _build(packet, plan).structural_targets
    object.__setattr__(targets.neighborhoods[0], "member_positions", (0,))
    object.__setattr__(targets.neighborhoods[0], "neighborhood_sha256", "")
    with pytest.raises(ValueError, match="self plus direct co-bundle"):
        replace(targets, target_sha256="")


@pytest.mark.parametrize(
    ("groups", "positive", "negative", "neighborhoods"),
    (
        (
            ((0, 1, 2),),
            ((0, 1), (0, 2), (1, 2)),
            (),
            ((0, 1, 2), (0, 1, 2), (0, 1, 2)),
        ),
        (
            ((0,), (1,), (2,)),
            (),
            ((0, 1), (0, 2), (1, 2)),
            ((0,), (1,), (2,)),
        ),
    ),
)
def test_all_positive_and_all_negative_packets_are_closed_cases(
    groups: tuple[tuple[int, ...], ...],
    positive: tuple[tuple[int, int], ...],
    negative: tuple[tuple[int, int], ...],
    neighborhoods: tuple[tuple[int, ...], ...],
) -> None:
    packet, plan = _fixture(groups)
    targets = _build(packet, plan).structural_targets
    assert _coordinates(targets.positive_pairs) == positive
    assert _coordinates(targets.negative_pairs) == negative
    assert tuple(item.member_positions for item in targets.neighborhoods) == neighborhoods


def test_target_numerics_ignore_utility_required_obligation_and_graph_metadata() -> None:
    baseline_packet, baseline_plan = _fixture(((0, 1), (1, 2)))
    variant_packet, variant_plan = _fixture(
        ((0, 1), (1, 2)),
        metadata_variant=True,
    )

    baseline = _build(baseline_packet, baseline_plan)
    variant = _build(variant_packet, variant_plan)

    assert baseline.structural_targets == variant.structural_targets
    assert baseline.structural_targets.target_sha256 == (
        variant.structural_targets.target_sha256
    )
    assert baseline.target_receipt_sha256 != variant.target_receipt_sha256
    assert baseline.closure_plan_sha256 != variant.closure_plan_sha256
    assert baseline.authoritative_hyperedges_sha256 != (
        variant.authoritative_hyperedges_sha256
    )


def test_target_numerics_ignore_provenance_text_and_consistent_opaque_id_changes() -> None:
    groups = ((0, 1), (1, 2))
    baseline_packet, baseline_plan = _fixture(groups)
    provenance_packet, provenance_plan = _fixture(
        groups,
        texts=(
            "Changed source bytes alpha.",
            "Changed source bytes beta.",
            "Changed source bytes gamma.",
        ),
        provenance_variant=True,
    )
    renamed_packet, renamed_plan = _fixture(
        groups,
        atom_ids=("renamed-z", "renamed-y", "renamed-x"),
        bundle_ids=("renamed-bundle-z", "renamed-bundle-a"),
    )

    baseline = _build(baseline_packet, baseline_plan)
    provenance = _build(provenance_packet, provenance_plan)
    renamed = _build(renamed_packet, renamed_plan)

    assert baseline.structural_targets == provenance.structural_targets
    assert baseline.structural_targets == renamed.structural_targets
    assert baseline.ordered_atom_refs_sha256 != provenance.ordered_atom_refs_sha256
    assert baseline.ordered_atom_refs_sha256 != renamed.ordered_atom_refs_sha256
    assert baseline.target_receipt_sha256 != provenance.target_receipt_sha256
    assert baseline.target_receipt_sha256 != renamed.target_receipt_sha256


@pytest.mark.parametrize("tamper", ("receipt", "plan", "packet_atom", "packet_bundle", "caps"))
def test_builder_rejects_tampered_or_cross_body_inputs(tamper: str) -> None:
    packet, plan = _fixture(((0, 1), (1, 2)))
    caps = _caps(packet)
    if tamper == "receipt":
        object.__setattr__(
            packet.receipt,
            "selected_atom_ids",
            tuple(reversed(packet.receipt.selected_atom_ids)),
        )
    elif tamper == "plan":
        object.__setattr__(plan.query_program, "query", "tampered query")
    elif tamper == "packet_atom":
        forged_atom = replace(packet.atoms[0], label="forged packet-only label")
        packet = EvidencePacket(
            packet.context,
            (forged_atom, *packet.atoms[1:]),
            packet.bundles,
            packet.receipt,
        )
    elif tamper == "packet_bundle":
        forged_bundle = replace(packet.bundles[0], utility=99.0)
        packet = EvidencePacket(
            packet.context,
            packet.atoms,
            (forged_bundle, *packet.bundles[1:]),
            packet.receipt,
        )
    else:
        object.__setattr__(caps, "max_atoms", 99)

    with pytest.raises((TypeError, ValueError), match="SHA-256|exactly match|closure plan"):
        build_latent_router_structural_targets(packet, plan, caps=caps)


def test_packet_plan_join_uses_canonical_identity_for_signed_zero() -> None:
    packet, original_plan = _fixture(((0, 1),))
    planned_bundle = replace(original_plan.bundles[0], utility=0.0)
    plan = replace(
        original_plan,
        bundles=(planned_bundle,),
        plan_sha256="",
    )
    receipt = replace(
        packet.receipt,
        plan_sha256=plan.plan_sha256,
        receipt_sha256="",
    )
    forged_bundle = replace(planned_bundle, utility=-0.0)
    assert forged_bundle.identity_payload() == planned_bundle.identity_payload()
    assert identity_sha256(forged_bundle.identity_payload()) != identity_sha256(
        planned_bundle.identity_payload()
    )
    packet = EvidencePacket(
        packet.context,
        packet.atoms,
        (forged_bundle,),
        receipt,
    )

    with pytest.raises(ValueError, match="bundle does not exactly match"):
        build_latent_router_structural_targets(packet, plan, caps=_caps(packet))


@pytest.mark.parametrize(
    "updates",
    (
        {"retained_request_token_state_bytes": 1},
        {"responder_output_token_reserve": 1},
        {"prompt_token_proxy": 5},
        {
            "prompt_token_proxy": 5,
            "max_prompt_token_proxy": 20,
            "responder_output_token_reserve": 2,
            "prompt_workspace_token_proxy": 8,
            "base_messages_sha256": "a" * 64,
            "evidence_message_role": "user",
            "evidence_prefix_sha256": "b" * 64,
            "evidence_suffix_sha256": "c" * 64,
            "prompt_messages_sha256": "d" * 64,
        },
        {
            "prompt_token_proxy": 5,
            "max_prompt_token_proxy": 6,
            "responder_output_token_reserve": 2,
            "prompt_workspace_token_proxy": 7,
            "base_messages_sha256": "a" * 64,
            "evidence_message_role": "user",
            "evidence_prefix_sha256": "b" * 64,
            "evidence_suffix_sha256": "c" * 64,
            "prompt_messages_sha256": "d" * 64,
        },
        {"context_token_proxy": 129},
        {"stopping_reason": "not_found", "complete_claimed": True},
        {"stopping_reason": "evil-stop", "complete_claimed": False},
    ),
)
def test_resealed_closure_receipt_cannot_bypass_semantic_invariants(
    updates: dict[str, object],
) -> None:
    packet, plan = _fixture(((0, 1),))
    _reseal_receipt(packet.receipt, **updates)

    with pytest.raises(ValueError):
        build_latent_router_structural_targets(packet, plan, caps=_caps(packet))


def test_coherently_resealed_receipt_must_match_plan_derived_packing_outcome() -> None:
    packet, plan = _fixture(((0, 1),))
    _reseal_receipt(
        packet.receipt,
        stopping_reason="not_found",
        complete_claimed=False,
    )

    with pytest.raises(ValueError, match="outcome disagrees"):
        build_latent_router_structural_targets(packet, plan, caps=_caps(packet))


@pytest.mark.parametrize("fault", ("selected", "unknown", "reason"))
def test_dropped_bundle_join_is_an_exact_closed_partition(fault: str) -> None:
    packet, plan = _fixture(((0, 1), (2,)))
    selected_bundle = packet.bundles[0]
    dropped_bundle = packet.bundles[1]
    if fault == "selected":
        dropped = {selected_bundle.bundle_id: "hard_budget"}
    elif fault == "unknown":
        dropped = {"unknown-bundle": "hard_budget"}
    else:
        dropped = {dropped_bundle.bundle_id: "invented_reason"}
    if fault == "reason":
        atoms = packet.atoms[:2]
        bundles = (selected_bundle,)
        receipt = replace(
            packet.receipt,
            selected_atom_ids=tuple(atom.atom_id for atom in atoms),
            selected_bundle_ids=(selected_bundle.bundle_id,),
            dropped_bundle_reasons=dropped,
            receipt_sha256="",
        )
        packet = EvidencePacket(packet.context, atoms, bundles, receipt)
    else:
        receipt = replace(
            packet.receipt,
            dropped_bundle_reasons=dropped,
            receipt_sha256="",
        )
        packet = EvidencePacket(packet.context, packet.atoms, packet.bundles, receipt)

    with pytest.raises(ValueError, match="selected and dropped|both selected|unsupported"):
        build_latent_router_structural_targets(packet, plan, caps=_caps(packet))


def test_selected_packet_atoms_must_equal_selected_bundle_union() -> None:
    packet, plan = _fixture(((0, 1), (2,)))
    selected = (packet.bundles[0],)
    dropped = packet.bundles[1]
    receipt = replace(
        packet.receipt,
        selected_bundle_ids=(selected[0].bundle_id,),
        dropped_bundle_reasons={dropped.bundle_id: "lower_utility"},
        receipt_sha256="",
    )
    packet = EvidencePacket(packet.context, packet.atoms, selected, receipt)

    with pytest.raises(ValueError, match="selected bundle atom union"):
        build_latent_router_structural_targets(packet, plan, caps=_caps(packet))


def test_canonical_dropped_bundle_subset_retains_exact_structural_packet() -> None:
    packet, plan = _fixture(((0, 1), (2,)))
    atoms = packet.atoms[:2]
    selected_bundle = packet.bundles[0]
    dropped_bundle = packet.bundles[1]
    receipt = replace(
        packet.receipt,
        selected_atom_ids=tuple(atom.atom_id for atom in atoms),
        selected_bundle_ids=(selected_bundle.bundle_id,),
        dropped_bundle_reasons={dropped_bundle.bundle_id: "lower_utility"},
        receipt_sha256="",
    )
    packet = EvidencePacket(packet.context, atoms, (selected_bundle,), receipt)

    targets = _build(packet, plan).structural_targets
    assert _coordinates(targets.positive_pairs) == ((0, 1),)
    assert targets.negative_pairs == ()
    assert tuple(item.member_positions for item in targets.neighborhoods) == (
        (0, 1),
        (0, 1),
    )


def test_consistent_packet_and_receipt_permutation_cannot_redefine_positions() -> None:
    packet, plan = _fixture(((0, 1), (2,)))
    receipt = replace(
        packet.receipt,
        selected_atom_ids=tuple(reversed(packet.receipt.selected_atom_ids)),
        selected_bundle_ids=tuple(reversed(packet.receipt.selected_bundle_ids)),
        receipt_sha256="",
    )
    packet = EvidencePacket(
        packet.context,
        tuple(reversed(packet.atoms)),
        tuple(reversed(packet.bundles)),
        receipt,
    )

    with pytest.raises(ValueError, match="authoritative packing order|authoritative plan order"):
        build_latent_router_structural_targets(packet, plan, caps=_caps(packet))


@pytest.mark.parametrize(
    "fault",
    (
        "confidence",
        "dependency_cycle",
        "snapshot",
        "bundle_required",
        "completion",
        "stopping_reason",
    ),
)
def test_resealed_plan_cannot_bypass_owned_semantic_invariants(fault: str) -> None:
    packet, plan = _fixture(((0, 1),))
    if fault == "confidence":
        object.__setattr__(plan.policy, "min_relation_confidence", 1.5)
    elif fault == "dependency_cycle":
        answer, auxiliary = plan.query_program.obligations
        object.__setattr__(answer, "dependencies", ("auxiliary",))
        object.__setattr__(auxiliary, "dependencies", ("answer",))
        object.__setattr__(plan.query_program, "program_sha256", "")
        plan.query_program._seal()
    elif fault == "snapshot":
        object.__setattr__(plan.snapshot, "chunk_count", -1)
        object.__setattr__(plan.snapshot, "snapshot_sha256", "")
        plan.snapshot._seal()
    elif fault == "bundle_required":
        object.__setattr__(plan.bundles[0], "required", False)
    elif fault == "completion":
        object.__setattr__(plan, "complete_claimed", False)
    else:
        object.__setattr__(plan, "stopping_reason", "evil-stop")
        object.__setattr__(plan, "complete_claimed", False)
    _rebind_plan_receipt(packet, plan)

    with pytest.raises(ValueError):
        build_latent_router_structural_targets(packet, plan, caps=_caps(packet))


def test_receipt_selected_counts_are_capped_before_id_or_plan_traversal() -> None:
    packet, plan = _fixture(((0, 1),))
    object.__setattr__(plan.query_program, "query", object())
    _reseal_receipt(
        packet.receipt,
        selected_atom_ids=tuple(f"oversized-{index}" for index in range(65)),
    )

    with pytest.raises(MemoryError, match="selected atom count"):
        build_latent_router_structural_targets(
            packet,
            plan,
            caps=FusionCaps(max_atoms=64),
        )


def test_exact_subtype_scalar_and_frozen_boundaries() -> None:
    packet, plan = _fixture(((0, 1),))

    @dataclass(frozen=True, slots=True)
    class PacketSubtype(EvidencePacket):
        injected_text: str = "SENSITIVE INJECTED TEXT"

    packet_subtype = PacketSubtype(
        packet.context,
        packet.atoms,
        packet.bundles,
        packet.receipt,
    )
    with pytest.raises(TypeError, match="exact EvidencePacket"):
        build_latent_router_structural_targets(
            packet_subtype,
            plan,
            caps=_caps(packet),
        )

    pair = AtomPositionPairTarget(0, 1, 1)

    @dataclass(frozen=True, slots=True)
    class PairSubtype(AtomPositionPairTarget):
        injected_text: str = "SENSITIVE INJECTED TEXT"

    injected = PairSubtype(0, 1, 1)
    with pytest.raises(TypeError, match="exact AtomPositionPairTarget"):
        LatentRouterStructuralTargets(
            atom_count=2,
            positive_pairs=(injected,),
            negative_pairs=(),
            neighborhoods=(
                DirectCoBundleNeighborhood(0, (0, 1)),
                DirectCoBundleNeighborhood(1, (0, 1)),
            ),
            positive_pair_count=1,
            negative_pair_count=0,
        )
    with pytest.raises(FrozenInstanceError):
        pair.left_position = 7  # type: ignore[misc]
    with pytest.raises(TypeError, match="exact integer"):
        AtomPositionPairTarget(True, 1, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact integer"):
        AtomPositionPairTarget(0, 1, True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact tuple"):
        DirectCoBundleNeighborhood(0, [0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique and ascending"):
        DirectCoBundleNeighborhood(1, (1, 0))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(pair, pair_sha256="0" * 64)


@pytest.mark.parametrize("owner", ("plan", "receipt"))
def test_resealed_bool_as_int_is_rejected_across_owned_inputs(owner: str) -> None:
    packet, plan = _fixture(((0, 1),))
    if owner == "plan":
        plan = replace(plan, complete_claimed=1, plan_sha256="")  # type: ignore[arg-type]
        receipt = replace(
            packet.receipt,
            plan_sha256=plan.plan_sha256,
            receipt_sha256="",
        )
    else:
        receipt = replace(
            packet.receipt,
            complete_claimed=1,  # type: ignore[arg-type]
            receipt_sha256="",
        )
    packet = EvidencePacket(packet.context, packet.atoms, packet.bundles, receipt)

    with pytest.raises(TypeError, match="exact boolean"):
        build_latent_router_structural_targets(packet, plan, caps=_caps(packet))


def test_resealed_plan_cannot_hide_a_stale_nested_witness_seal() -> None:
    packet, plan = _fixture(((0, 1),))
    object.__setattr__(plan.scope_witnesses[0], "kind", "mutated-witness-kind")
    plan = replace(plan, plan_sha256="")
    receipt = replace(
        packet.receipt,
        plan_sha256=plan.plan_sha256,
        receipt_sha256="",
    )
    packet = EvidencePacket(packet.context, packet.atoms, packet.bundles, receipt)

    with pytest.raises(ValueError, match="scope witness SHA-256"):
        build_latent_router_structural_targets(packet, plan, caps=_caps(packet))


def test_public_target_constructors_reject_work_above_static_caps_early() -> None:
    with pytest.raises(MemoryError, match="at or above 64"):
        AtomPositionPairTarget(0, 64, 1)
    with pytest.raises(MemoryError, match="below 64"):
        DirectCoBundleNeighborhood(64, (64,))
    with pytest.raises(MemoryError, match="cannot exceed 64"):
        DirectCoBundleNeighborhood(0, tuple(range(65)))

    pair = AtomPositionPairTarget(0, 1, 1)
    neighborhood = DirectCoBundleNeighborhood(0, (0,))
    with pytest.raises(MemoryError, match="pair count exceeds 2016"):
        LatentRouterStructuralTargets(
            atom_count=2,
            positive_pairs=(pair,) * 2_017,
            negative_pairs=(),
            neighborhoods=(
                DirectCoBundleNeighborhood(0, (0, 1)),
                DirectCoBundleNeighborhood(1, (0, 1)),
            ),
            positive_pair_count=2_017,
            negative_pair_count=0,
        )
    with pytest.raises(MemoryError, match="neighborhood count exceeds 64"):
        LatentRouterStructuralTargets(
            atom_count=1,
            positive_pairs=(),
            negative_pairs=(),
            neighborhoods=(neighborhood,) * 65,
            positive_pair_count=0,
            negative_pair_count=0,
        )


def test_packet_count_caps_reject_before_nested_plan_or_body_traversal() -> None:
    packet, plan = _fixture(((0,),))
    object.__setattr__(plan.query_program, "query", object())
    object.__setattr__(packet, "atoms", packet.atoms * 65)

    with pytest.raises(MemoryError, match="exceeds 64"):
        build_latent_router_structural_targets(
            packet,
            plan,
            caps=FusionCaps(max_atoms=65),
        )


def test_caps_enforce_operation_limits_and_absolute_pair_boundary() -> None:
    packet, plan = _fixture(((0, 1, 2),))
    with pytest.raises(MemoryError, match="max_atoms"):
        _build(packet, plan, max_atoms=2)
    two_bundle_packet, two_bundle_plan = _fixture(((0,), (1,)))
    with pytest.raises(MemoryError, match="max_hyperedges"):
        _build(two_bundle_packet, two_bundle_plan, max_hyperedges=1)
    with pytest.raises(MemoryError, match="max_topology_links"):
        _build(packet, plan, max_topology_links=2)

    positions_64 = tuple(range(64))
    packet_64, plan_64 = _fixture((positions_64,))
    targets_64 = _build(packet_64, plan_64).structural_targets
    assert targets_64.atom_count == 64
    assert targets_64.positive_pair_count == 2_016
    assert targets_64.negative_pair_count == 0

    positions_65 = tuple(range(65))
    packet_65, plan_65 = _fixture((positions_65,))
    with pytest.raises(MemoryError, match="exceeds 64"):
        build_latent_router_structural_targets(
            packet_65,
            plan_65,
            caps=_caps(packet_65),
        )


def test_receipt_is_text_free_tensor_free_and_builder_accepts_no_labels() -> None:
    packet, plan = _fixture(((0, 1), (1, 2)))
    receipt = _build(packet, plan)
    encoded = canonical_json(receipt.identity_payload())
    forbidden = (
        packet.context,
        plan.query_program.query,
        *(atom.text for atom in packet.atoms),
        *(atom.label for atom in packet.atoms),
        *(atom.atom_id for atom in packet.atoms),
        *(bundle.bundle_id for bundle in packet.bundles),
        "unit-original",
        "relation-original",
        "answer_fact",
        "supporting_fact",
    )
    assert all(value not in encoded for value in forbidden)

    def assert_plain(value: object) -> None:
        if type(value) in {str, int}:
            return
        if type(value) is tuple:
            for item in value:
                assert_plain(item)
            return
        assert hasattr(value, "__dataclass_fields__")
        for name in value.__dataclass_fields__:
            assert_plain(getattr(value, name))

    assert_plain(receipt)
    signature = inspect.signature(build_latent_router_structural_targets)
    assert tuple(signature.parameters) == ("packet", "plan", "caps")
    assert signature.parameters["caps"].kind is inspect.Parameter.KEYWORD_ONLY


def test_training_target_module_keeps_torch_and_transformers_cold() -> None:
    code = (
        "import importlib, sys; "
        "importlib.import_module('memory_condense.search.fusion.training_targets'); "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False", result.stdout
