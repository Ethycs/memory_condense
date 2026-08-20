from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace

import pytest

from memory_condense.domain._discourse_identity import canonical_json
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
    make_atom_id,
    quote_sha256,
)
from memory_condense.search.fusion import (
    AuthoritativeHyperedge,
    DeclaredFeatureExtractorIdentity,
    FusionCaps,
    LatentEvidenceRouter,
    NodeFeatureBatch,
    NodeFeatureReceipt,
    build_evidence_fusion_plan,
    validate_matched_fusion_pair,
)
from memory_condense.search.fusion.tensor_identity import canonical_float32_tensor


torch = pytest.importorskip("torch")


def _atom(index: int, text: str) -> EvidenceAtom:
    span = EvidenceSpan(
        chunk_id=f"chunk-{index}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=index,
        source_id="fusion-fixture",
        turn_id=f"turn-{index}",
        role="user",
        created_at=f"2026-08-{index:02d}",
    )
    return EvidenceAtom(
        atom_id=make_atom_id(span),
        span=span,
        text=text,
        label=f"fixture-{index}",
    )


def _packet_and_plan(
    *,
    query: str = "How do these exact observations relate?",
) -> tuple[EvidencePacket, ClosurePlan]:
    atoms = (
        _atom(1, "Alpha evidence remains exact."),
        _atom(2, "Beta evidence shares one bundle."),
        _atom(3, "Gamma evidence is a routing hypothesis."),
    )
    bundles = (
        EvidenceBundle(
            bundle_id="bundle-primary",
            atom_ids=(atoms[0].atom_id, atoms[1].atom_id),
            obligation_ids=("answer",),
            unit_ids=("unit-primary",),
            relation_ids=("relation-primary",),
            required=True,
            utility=2.0,
        ),
        EvidenceBundle(
            bundle_id="bundle-routing",
            atom_ids=(atoms[2].atom_id,),
            obligation_ids=(),
            unit_ids=("unit-routing",),
            required=False,
            utility=0.25,
        ),
    )
    program = QueryProgram(
        query=query,
        intent="relate",
        subject_terms=("observations",),
        obligations=(
            EvidenceObligation(
                obligation_id="answer",
                kind="answer_fact",
                required=True,
                weight=1.0,
            ),
        ),
    )
    plan = ClosurePlan(
        query_program=program,
        policy=ClosurePolicy(max_bundles=8, beam_width=16),
        snapshot=DiscourseSnapshot(
            max_turn_ordinal=3,
            chunk_count=3,
            graph_revision=1,
            schema_version=1,
            artifact_ids=("fixture-artifact",),
            source_content_sha256="1" * 64,
            graph_content_sha256="2" * 64,
        ),
        seeds=(),
        atoms=atoms,
        bundles=bundles,
        obligation_results=(
            ObligationResult(
                obligation_id="answer",
                status="satisfied",
                unit_ids=("unit-primary",),
                relation_ids=("relation-primary",),
                bundle_ids=("bundle-primary",),
            ),
        ),
        visited_episode_ids=(),
        visited_unit_ids=("unit-primary", "unit-routing"),
        visited_relation_ids=("relation-primary",),
        stopping_reason="complete",
        complete_claimed=True,
        scope_witnesses=(
            ClosureScopeWitness(
                kind="fixture_scope",
                subject_id="fusion-fixture",
                requested_limit=3,
                returned_count=3,
                exhaustive=True,
            ),
        ),
        artifact_id="fixture-artifact",
    )
    context = "fixture packed context"
    receipt = ClosureReceipt(
        plan_sha256=plan.plan_sha256,
        context_sha256=quote_sha256(context),
        selected_bundle_ids=tuple(item.bundle_id for item in plan.bundles),
        selected_atom_ids=tuple(item.atom_id for item in plan.atoms),
        dropped_bundle_reasons={},
        context_token_proxy=3,
        max_context_token_proxy=16,
        tokenizer_identity="fixture-tokenizer",
        stopping_reason="complete",
        complete_claimed=True,
    )
    return EvidencePacket(context, plan.atoms, plan.bundles, receipt), plan


def _extractor() -> DeclaredFeatureExtractorIdentity:
    return DeclaredFeatureExtractorIdentity(
        extractor_id="fixture.encoder.v1",
        implementation_sha256="3" * 64,
        checkpoint_sha256="4" * 64,
        output_layer=2,
        pooling="mean",
        max_input_tokens_per_atom=64,
        hidden_dim=4,
    )


def _batch(packet: EvidencePacket, plan: ClosurePlan, *, delta: float = 0.0):
    values = torch.tensor(
        [
            [1.0 + delta, 0.0, 0.5, -0.5],
            [0.0, 1.0, 0.5, -0.25],
            [0.5, 0.5, 1.0, 0.25],
        ],
        dtype=torch.float32,
    )
    return NodeFeatureBatch.create(
        values,
        ordered_atom_ids=tuple(item.atom_id for item in packet.atoms),
        query=plan.query_program.query,
        extractor=_extractor(),
    )


def _caps(**updates: int) -> FusionCaps:
    values = {
        "max_atoms": 3,
        "max_latents": 2,
        "max_hidden_dim": 4,
        "max_route_cells": 6,
        "max_topology_links": 2,
        "max_hyperedges": 2,
        "max_groups": 3,
        "max_group_atoms": 2,
        "max_latent_memberships_per_atom": 2,
    }
    values.update(updates)
    return FusionCaps(**values)


def _router() -> LatentEvidenceRouter:
    torch.manual_seed(17)
    return LatentEvidenceRouter(
        hidden_dim=4,
        num_latents=2,
        num_heads=2,
        max_atoms=3,
        max_route_cells=6,
    ).seal_for_inference()


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if is_dataclass(value):
        return any(_contains_tensor(getattr(value, item.name)) for item in fields(value))
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    return False


def test_topology_plan_is_text_free_self_hashed_and_preserves_exact_atoms() -> None:
    packet, closure = _packet_and_plan()
    batch = _batch(packet, closure)

    first = build_evidence_fusion_plan(
        packet,
        closure,
        node_features=batch,
        caps=_caps(),
    )
    second = build_evidence_fusion_plan(
        packet,
        closure,
        node_features=batch,
        caps=_caps(),
    )

    assert first.mode == "topology_only"
    assert first.fusion_sha256 == second.fusion_sha256
    assert tuple(item.atom_id for item in first.atoms) == tuple(
        item.atom_id for item in packet.atoms
    )
    assert set(first.atom_order) == {item.atom_id for item in packet.atoms}
    assert first.plan_retained_request_tensor_bytes == 0
    assert not _contains_tensor(first)
    routing = next(edge for edge in first.hyperedges if edge.bundle_id == "bundle-routing")
    assert routing.obligation_ids == ()
    assert routing.unit_witness_ids == ("unit-routing",)
    encoded = canonical_json(first.identity_payload())
    assert closure.query_program.query not in encoded
    assert all(atom.text not in encoded for atom in packet.atoms)
    assert all(atom.label not in encoded for atom in packet.atoms)

    with pytest.raises(ValueError, match="SHA-256"):
        replace(first, fusion_sha256="0" * 64)


def test_exact_latent_router_builds_a_matched_single_factor_plan() -> None:
    packet, closure = _packet_and_plan()
    batch = _batch(packet, closure)
    caps = _caps()
    topology = build_evidence_fusion_plan(
        packet,
        closure,
        node_features=batch,
        caps=caps,
    )
    router = _router()

    latent = build_evidence_fusion_plan(
        packet,
        closure,
        node_features=batch,
        mode="latent_router",
        caps=caps,
        router=router,
    )
    repeated = build_evidence_fusion_plan(
        packet,
        closure,
        node_features=batch,
        mode="latent_router",
        caps=caps,
        router=router,
    )

    assert validate_matched_fusion_pair(topology, latent) == topology.matched_input_sha256
    assert latent.fusion_sha256 == repeated.fusion_sha256
    assert latent.extraction_shape == (2, 3)
    assert latent.reinjection_shape == (3, 2)
    assert latent.router_architecture.phase_count == 2
    assert latent.router_architecture.extraction_query_axis == "latent_k"
    assert latent.router_architecture.extraction_key_value_axis == "evidence_n"
    assert latent.router_architecture.reinjection_query_axis == "evidence_n"
    assert latent.router_architecture.reinjection_key_value_axis == "updated_latent_k"
    assert latent.router_architecture.extraction_residual_rule == "none"
    assert latent.router_architecture.residual_rule == "x_plus_reinjected_update"
    assert latent.router_state.training_status == "untrained"
    assert latent.router_state.parameter_dtypes == ("torch.float32",)
    assert tuple(item.atom_id for item in latent.atoms) == tuple(
        item.atom_id for item in topology.atoms
    )
    assert set(latent.atom_order) == set(topology.atom_order)
    assert all(group.latent_index is not None for group in latent.groups)
    assert not _contains_tensor(latent)


def test_router_has_exact_kn_nk_shapes_and_a_sealed_bounded_inference_path() -> None:
    torch.manual_seed(3)
    unsealed = LatentEvidenceRouter(
        4,
        num_latents=2,
        num_heads=2,
        max_atoms=3,
        max_route_cells=6,
    )
    values = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="sealed"):
        unsealed.route_one(values)
    with pytest.raises(ValueError, match=r"\[1, N"):
        unsealed.forward(torch.randn(2, 3, 4))

    router = unsealed.seal_for_inference()
    routed = router.route_one(values)
    assert tuple(routed.extraction_attention.shape) == (2, 3)
    assert tuple(routed.reinjection_attention.shape) == (3, 2)
    assert tuple(routed.steered_nodes.shape) == (3, 4)
    assert torch.allclose(
        routed.extraction_attention.sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.allclose(
        routed.reinjection_attention.sum(dim=-1),
        torch.ones(3),
        atol=1e-6,
    )
    with pytest.raises(RuntimeError, match="only supports route_one"):
        router.forward(torch.randn(2, 3, 4))
    with pytest.raises(RuntimeError, match="sealed"):
        router.train()
    with pytest.raises(RuntimeError, match="sealed"):
        router.to(dtype=torch.float16)
    with pytest.raises(RuntimeError, match="does not expose its module"):
        _ = router.module
    with pytest.raises(RuntimeError, match="does not expose parameters"):
        router.parameters()
    with pytest.raises(RuntimeError, match="does not expose mutable state"):
        router.state_dict()
    with pytest.raises(AttributeError):
        router.route_one = lambda _value: None
    with pytest.raises(MemoryError, match="max_hidden_dim"):
        LatentEvidenceRouter(
            8,
            num_latents=2,
            num_heads=2,
            max_atoms=3,
            max_hidden_dim=4,
            max_route_cells=6,
        )


def test_seal_retires_raw_parameter_and_numpy_storage_aliases() -> None:
    torch.manual_seed(19)
    router = LatentEvidenceRouter(
        4,
        num_latents=2,
        num_heads=2,
        max_atoms=3,
        max_route_cells=6,
    )
    captured_parameter = next(router.parameters())
    router.seal_for_inference()
    sealed_state = router.state_receipt
    with torch.no_grad():
        captured_parameter.data.add_(1.0)
    captured_parameter.detach().numpy().reshape(-1)[0] += 1.0
    routed = router.route_one(torch.randn(3, 4))
    assert tuple(routed.steered_nodes.shape) == (3, 4)
    assert router.state_receipt == sealed_state


def test_seal_retires_captured_modules_and_detects_shared_class_mutation() -> None:
    prehooked = LatentEvidenceRouter(
        4,
        num_latents=2,
        num_heads=2,
        max_atoms=3,
        max_route_cells=6,
    )
    prehooked.module.register_forward_pre_hook(lambda _module, _inputs: None)
    with pytest.raises(RuntimeError, match="execution hooks"):
        prehooked.seal_for_inference()

    torch.manual_seed(23)
    hooked = LatentEvidenceRouter(
        4,
        num_latents=2,
        num_heads=2,
        max_atoms=3,
        max_route_cells=6,
    )
    hooked_module = hooked.module
    hooked.seal_for_inference()
    hooked_module.extract_attention.register_forward_hook(
        lambda _module, _inputs, output: output
    )
    assert tuple(hooked.route_one(torch.randn(3, 4)).steered_nodes.shape) == (3, 4)

    shadowed = LatentEvidenceRouter(
        4,
        num_latents=2,
        num_heads=2,
        max_atoms=3,
        max_route_cells=6,
    )
    shadowed_module = shadowed.module
    shadowed.seal_for_inference()
    shadowed_module.forward = lambda _value: None
    assert tuple(shadowed.route_one(torch.randn(3, 4)).steered_nodes.shape) == (3, 4)

    class_patched = LatentEvidenceRouter(
        4,
        num_latents=2,
        num_heads=2,
        max_atoms=3,
        max_route_cells=6,
    )
    class_patched_module = class_patched.module
    module_type = type(class_patched_module)
    original_forward = module_type.forward
    class_patched.seal_for_inference()
    try:
        module_type.forward = lambda _self, _value: None
        with pytest.raises(RuntimeError, match="runtime changed"):
            class_patched.route_one(torch.randn(3, 4))
    finally:
        module_type.forward = original_forward


def test_feature_receipt_rejects_mutation_order_and_query_mismatch() -> None:
    packet, closure = _packet_and_plan()
    batch = _batch(packet, closure)
    batch.values[0, 0] += 1.0
    with pytest.raises(ValueError, match="sealed receipt"):
        build_evidence_fusion_plan(
            packet,
            closure,
            node_features=batch,
            caps=_caps(),
        )

    wrong_order = NodeFeatureBatch.create(
        torch.randn(3, 4),
        ordered_atom_ids=tuple(reversed(tuple(item.atom_id for item in packet.atoms))),
        query=closure.query_program.query,
        extractor=_extractor(),
    )
    with pytest.raises(ValueError, match="atom order"):
        build_evidence_fusion_plan(
            packet,
            closure,
            node_features=wrong_order,
            caps=_caps(),
        )

    wrong_query = NodeFeatureBatch.create(
        torch.randn(3, 4),
        ordered_atom_ids=tuple(item.atom_id for item in packet.atoms),
        query="a different query",
        extractor=_extractor(),
    )
    with pytest.raises(ValueError, match="query"):
        build_evidence_fusion_plan(
            packet,
            closure,
            node_features=wrong_query,
            caps=_caps(),
        )


def test_feature_shape_caps_run_before_clone_or_materialization() -> None:
    packet, closure = _packet_and_plan()
    valid = _batch(packet, closure)

    class CloneBomb:
        shape = (10_000_000, 4)

        def __init__(self) -> None:
            self.clone_called = False
            self.tolist_called = False

        def clone(self):
            self.clone_called = True
            raise AssertionError("clone ran before caps")

        def tolist(self):
            self.tolist_called = True
            raise AssertionError("tolist ran before caps")

    create_bomb = CloneBomb()
    with pytest.raises(ValueError, match="shape disagrees"):
        NodeFeatureBatch.create(
            create_bomb,
            ordered_atom_ids=tuple(item.atom_id for item in packet.atoms),
            query=closure.query_program.query,
            extractor=_extractor(),
        )
    assert create_bomb.clone_called is False
    assert create_bomb.tolist_called is False

    planner_bomb = CloneBomb()
    injected = NodeFeatureBatch(values=planner_bomb, receipt=valid.receipt)
    with pytest.raises(ValueError, match="exact packet atom set"):
        build_evidence_fusion_plan(
            packet,
            closure,
            node_features=injected,
            caps=_caps(),
        )
    assert planner_bomb.clone_called is False
    assert planner_bomb.tolist_called is False


def test_fail_closed_caps_plan_mismatch_and_non_owned_router() -> None:
    packet, closure = _packet_and_plan()
    batch = _batch(packet, closure)
    with pytest.raises(MemoryError, match="max_hyperedges"):
        build_evidence_fusion_plan(
            packet,
            closure,
            node_features=batch,
            caps=_caps(max_hyperedges=1),
        )

    _other_packet, other_closure = _packet_and_plan(query="another query")
    with pytest.raises(ValueError, match="receipt"):
        build_evidence_fusion_plan(
            packet,
            other_closure,
            node_features=batch,
            caps=_caps(),
        )

    with pytest.raises(TypeError, match="owned LatentEvidenceRouter"):
        build_evidence_fusion_plan(
            packet,
            closure,
            node_features=batch,
            mode="latent_router",
            caps=_caps(),
            router=object(),
        )


def test_matched_validator_rejects_different_feature_batches() -> None:
    packet, closure = _packet_and_plan()
    caps = _caps()
    topology = build_evidence_fusion_plan(
        packet,
        closure,
        node_features=_batch(packet, closure, delta=0.5),
        caps=caps,
    )
    latent = build_evidence_fusion_plan(
        packet,
        closure,
        node_features=_batch(packet, closure),
        mode="latent_router",
        caps=caps,
        router=_router(),
    )
    with pytest.raises(ValueError, match="not a matched"):
        validate_matched_fusion_pair(topology, latent)


def test_tensor_hash_binds_shape_and_nested_inputs_are_frozen() -> None:
    row = canonical_float32_tensor([[1.0, 2.0]], label="row")
    column = canonical_float32_tensor([[1.0], [2.0]], label="column")
    assert row.tensor_sha256 != column.tensor_sha256
    tensor_values = torch.tensor([[1.25, -2.5], [3.75, 4.5]], dtype=torch.float32)
    vectorized = canonical_float32_tensor(
        tensor_values,
        label="vectorized",
        retain_values=False,
    )
    scalar_reference = canonical_float32_tensor(
        tensor_values.tolist(),
        label="scalar-reference",
    )
    assert vectorized.tensor_sha256 == scalar_reference.tensor_sha256
    assert vectorized.shape == scalar_reference.shape == (2, 2)
    assert vectorized.flat_values == ()

    atom_ids = ["atom-a", "atom-b"]
    edge = AuthoritativeHyperedge(
        bundle_id="bundle-a",
        atom_ids=atom_ids,
        obligation_ids=(),
    )
    atom_ids.append("atom-c")
    assert edge.atom_ids == ("atom-a", "atom-b")


def test_declared_extractor_cannot_embed_free_form_text() -> None:
    with pytest.raises(ValueError, match="machine identifier"):
        DeclaredFeatureExtractorIdentity(
            extractor_id="the answer is hidden here",
            implementation_sha256="3" * 64,
            checkpoint_sha256="4" * 64,
            output_layer=0,
            pooling="mean",
            max_input_tokens_per_atom=8,
            hidden_dim=4,
        )
    with pytest.raises(ValueError, match="supported closed"):
        DeclaredFeatureExtractorIdentity(
            extractor_id="fixture.encoder",
            implementation_sha256="3" * 64,
            checkpoint_sha256="4" * 64,
            output_layer=0,
            pooling="question-specific prose",
            max_input_tokens_per_atom=8,
            hidden_dim=4,
        )


def test_feature_receipt_subclass_cannot_inject_raw_text_into_plan() -> None:
    packet, closure = _packet_and_plan()
    batch = _batch(packet, closure)

    @dataclass(frozen=True, slots=True)
    class TextBearingReceipt(NodeFeatureReceipt):
        raw_source_text: str = "SENSITIVE SOURCE PROSE"

    base = batch.receipt
    injected = TextBearingReceipt(
        extractor=base.extractor,
        ordered_atom_ids=base.ordered_atom_ids,
        query_sha256=base.query_sha256,
        tensor_sha256=base.tensor_sha256,
        tensor_shape=base.tensor_shape,
    )
    with pytest.raises(TypeError, match="receipt must be NodeFeatureReceipt"):
        NodeFeatureBatch(values=batch.values, receipt=injected)
