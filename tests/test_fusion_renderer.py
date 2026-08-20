from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType

import pytest

import memory_condense.domain.discourse as discourse_module
import memory_condense.search.fusion.render_models as render_models_module
import memory_condense.search.fusion.renderer as renderer_module
import memory_condense.search.packing.evidence_packet as evidence_packet_module

from memory_condense.domain._discourse_identity import (
    identity_sha256,
    make_atom_id,
    quote_sha256,
)
from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
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
from memory_condense.search.fusion.models import (
    AuthoritativeHyperedge,
    ExtractiveGroup,
    FusionAtomRef,
    FusionCaps,
    LatentMembership,
    RouterArchitectureReceipt,
    RouterStateReceipt,
)
from memory_condense.search.fusion.renderer import (
    _preflight_packet,
    _render_structural_matched_contexts,
    render_matched_fusion_contexts,
)
from memory_condense.search.fusion.resident_models import (
    ResidentEvidenceFusionPlan,
    ResidentRouterRuntimeReceipt,
)
from memory_condense.search.packing.evidence_packet import (
    EvidencePromptBudget,
    normalize_evidence_prompt_budget,
    pack_evidence_plan,
    render_evidence_context,
    render_grouped_evidence_context,
)
@dataclass(frozen=True, slots=True)
class _Fixture:
    packet: EvidencePacket
    plan: ClosurePlan
    topology_only: ResidentEvidenceFusionPlan
    latent_router: ResidentEvidenceFusionPlan
    base_messages: tuple[dict[str, str], ...]
    evidence_message_role: str
    evidence_prefix: str
    evidence_suffix: str


class _InjectInt(int):
    def __format__(self, _format_spec: str) -> str:
        return "1 | latent_index=forged"


def _atom(index: int, text: str) -> EvidenceAtom:
    span = EvidenceSpan(
        chunk_id=f"render-chunk-{index}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=index,
        source_id="render-source",
        turn_id=f"render-turn-{index}",
        role="user",
    )
    return EvidenceAtom(
        atom_id=make_atom_id(span),
        span=span,
        text=text,
        label=f"render-label-{index}",
    )


def _plan() -> ClosurePlan:
    atoms = (
        _atom(1, "Alpha evidence remains exact."),
        _atom(2, "Beta evidence remains exact."),
    )
    bundle = EvidenceBundle(
        bundle_id="render-bundle",
        atom_ids=tuple(atom.atom_id for atom in atoms),
        obligation_ids=("answer",),
        unit_ids=("render-unit",),
        required=True,
        utility=1.0,
    )
    program = QueryProgram(
        query="Which evidence is related?",
        intent="relate",
        subject_terms=("evidence",),
        obligations=(
            EvidenceObligation(
                obligation_id="answer",
                kind="answer_fact",
                required=True,
                weight=1.0,
            ),
        ),
    )
    return ClosurePlan(
        query_program=program,
        policy=ClosurePolicy(max_bundles=2, beam_width=4),
        snapshot=DiscourseSnapshot(
            max_turn_ordinal=2,
            chunk_count=2,
            graph_revision=1,
            schema_version=1,
            artifact_ids=("render-artifact",),
            source_content_sha256="1" * 64,
            graph_content_sha256="2" * 64,
        ),
        seeds=(),
        atoms=atoms,
        bundles=(bundle,),
        obligation_results=(
            ObligationResult(
                obligation_id="answer",
                status="satisfied",
                unit_ids=("render-unit",),
                bundle_ids=(bundle.bundle_id,),
            ),
        ),
        visited_episode_ids=(),
        visited_unit_ids=("render-unit",),
        visited_relation_ids=(),
        stopping_reason="complete",
        complete_claimed=True,
        scope_witnesses=(
            ClosureScopeWitness(
                kind="render_scope",
                subject_id="render-source",
                requested_limit=2,
                returned_count=2,
                exhaustive=True,
            ),
        ),
        artifact_id="render-artifact",
    )


def _refs(packet: EvidencePacket) -> tuple[FusionAtomRef, ...]:
    return tuple(
        FusionAtomRef(
            atom_id=atom.atom_id,
            atom_identity_sha256=identity_sha256(atom.identity_payload()),
            span_identity_sha256=identity_sha256(atom.span.identity_payload()),
            quote_sha256=atom.span.quote_sha256,
        )
        for atom in packet.atoms
    )


def _edges(packet: EvidencePacket) -> tuple[AuthoritativeHyperedge, ...]:
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


def _structural_plans(
    packet: EvidencePacket,
) -> tuple[ResidentEvidenceFusionPlan, ResidentEvidenceFusionPlan]:
    refs = _refs(packet)
    edges = _edges(packet)
    atom_ids = tuple(item.atom_id for item in refs)
    caps = FusionCaps(
        max_atoms=2,
        max_latents=2,
        max_hidden_dim=4,
        max_route_cells=4,
        max_topology_links=1,
        max_hyperedges=1,
        max_groups=2,
        max_group_atoms=2,
        max_latent_memberships_per_atom=1,
    )
    architecture = RouterArchitectureReceipt(
        hidden_dim=4,
        num_latents=2,
        num_heads=2,
        parameter_count=168,
    )
    state = RouterStateReceipt(
        loaded_parameter_bytes_sha256="8" * 64,
        operational_float32_sha256="9" * 64,
        parameter_count=168,
        parameter_dtypes=("torch.float32",),
        training_status="untrained",
    )
    runtime = ResidentRouterRuntimeReceipt(
        architecture=architecture,
        state=state,
        device="cuda:0",
        execution_dtype="torch.float32",
        max_atoms=2,
        max_hidden_dim=4,
        max_route_cells=4,
    )
    topology_groups = (
        ExtractiveGroup(group_index=0, atom_ids=atom_ids),
    )
    memberships = (
        LatentMembership(
            atom_id=atom_ids[0],
            latent_index=1,
            extraction_weight=0.5,
            reinjection_weight=0.5,
            joint_weight=0.5,
        ),
        LatentMembership(
            atom_id=atom_ids[1],
            latent_index=0,
            extraction_weight=0.5,
            reinjection_weight=0.5,
            joint_weight=0.5,
        ),
    )
    latent_groups = (
        ExtractiveGroup(
            group_index=0,
            atom_ids=(atom_ids[1],),
            latent_index=0,
        ),
        ExtractiveGroup(
            group_index=1,
            atom_ids=(atom_ids[0],),
            latent_index=1,
        ),
    )
    shared = {
        "feature_suboperation_sha256": "3" * 64,
        "matched_input_sha256": "4" * 64,
        "caps": caps,
        "atoms": refs,
        "hyperedges": edges,
    }
    control = ResidentEvidenceFusionPlan(
        mode="topology_only",
        memberships=(),
        groups=topology_groups,
        atom_order=atom_ids,
        **shared,
    )
    latent = ResidentEvidenceFusionPlan(
        mode="latent_router",
        memberships=memberships,
        groups=latent_groups,
        atom_order=(atom_ids[1], atom_ids[0]),
        router_runtime=runtime,
        extraction_matrix_sha256="b" * 64,
        reinjection_matrix_sha256="c" * 64,
        extraction_shape=(2, 2),
        reinjection_shape=(2, 2),
        **shared,
    )
    return control, latent


def _fixture(
    *,
    context_room: bool = True,
    prompt_room: bool = True,
    prompt_enabled: bool = True,
    exact_candidate_caps: bool = False,
    single_context_overflow: bool = False,
) -> _Fixture:
    plan = _plan()
    context = render_evidence_context(plan.atoms, plan.bundles)
    atom_ids = tuple(atom.atom_id for atom in plan.atoms)
    topology_context = render_grouped_evidence_context(
        plan.atoms,
        plan.bundles,
        (atom_ids,),
    )
    latent_context = render_grouped_evidence_context(
        plan.atoms,
        plan.bundles,
        ((atom_ids[1],), (atom_ids[0],)),
    )
    topology_context_tokens = count_tokens(topology_context)
    latent_context_tokens = count_tokens(latent_context)
    if single_context_overflow:
        assert topology_context_tokens < latent_context_tokens
        context_cap = topology_context_tokens
    elif exact_candidate_caps:
        context_cap = max(topology_context_tokens, latent_context_tokens)
    else:
        context_cap = count_tokens(context) + (256 if context_room else 0)
    base_messages = (
        {"role": "system", "content": "Answer only from exact evidence."},
        {"role": "user", "content": plan.query_program.query},
    )
    role = "system"
    prefix = "#"
    suffix = "\nAnswer concisely."
    reserve = 13
    if prompt_enabled:
        original_prompt_tokens = count_chat_prompt_token_proxy(
            (
                *base_messages,
                {"role": role, "content": prefix + context + suffix},
            )
        )
        candidate_prompt_tokens = tuple(
            count_chat_prompt_token_proxy(
                (
                    *base_messages,
                    {"role": role, "content": prefix + candidate + suffix},
                )
            )
            for candidate in (topology_context, latent_context)
        )
        if exact_candidate_caps:
            prompt_cap = max(candidate_prompt_tokens) + reserve
        else:
            prompt_cap = (
                original_prompt_tokens
                + reserve
                + (256 if prompt_room else 0)
            )
        packet = pack_evidence_plan(
            plan,
            max_context_tokens=context_cap,
            base_messages=base_messages,
            evidence_message_role=role,
            evidence_prefix=prefix,
            evidence_suffix=suffix,
            max_prompt_tokens=prompt_cap,
            output_token_reserve=reserve,
        )
    else:
        packet = pack_evidence_plan(plan, max_context_tokens=context_cap)
    topology_only, latent_router = _structural_plans(packet)
    return _Fixture(
        packet=packet,
        plan=plan,
        topology_only=topology_only,
        latent_router=latent_router,
        base_messages=base_messages,
        evidence_message_role=role,
        evidence_prefix=prefix,
        evidence_suffix=suffix,
    )


def _render(fixture: _Fixture):
    return _render_structural_matched_contexts(
        fixture.packet,
        fixture.topology_only,
        fixture.latent_router,
        base_messages=fixture.base_messages,
        evidence_message_role=fixture.evidence_message_role,
        evidence_prefix=fixture.evidence_prefix,
        evidence_suffix=fixture.evidence_suffix,
    )


def _contains_tensor(value: object) -> bool:
    if hasattr(value, "shape") and hasattr(value, "device"):
        return True
    if is_dataclass(value):
        return any(_contains_tensor(getattr(value, item.name)) for item in fields(value))
    if type(value) is tuple:
        return any(_contains_tensor(item) for item in value)
    return False


def _dataclass_field_names(value: object) -> set[str]:
    if not is_dataclass(value):
        return set()
    names = {item.name for item in fields(value)}
    for item in fields(value):
        child = getattr(value, item.name)
        if is_dataclass(child):
            names.update(_dataclass_field_names(child))
        elif type(child) is tuple:
            for member in child:
                names.update(_dataclass_field_names(member))
    return names


def test_grouped_renderer_shares_legacy_grammar_and_consumes_explicit_order() -> None:
    fixture = _fixture()
    atom_ids = tuple(atom.atom_id for atom in fixture.packet.atoms)
    assert render_evidence_context(
        fixture.packet.atoms, fixture.packet.bundles
    ) == fixture.packet.context
    grouped = render_grouped_evidence_context(
        fixture.packet.atoms,
        fixture.packet.bundles,
        ((atom_ids[1],), (atom_ids[0],)),
    )
    assert grouped.index(fixture.packet.atoms[1].text) < grouped.index(
        fixture.packet.atoms[0].text
    )
    assert grouped.count("### Evidence group G1") == 1
    assert grouped.count("### Evidence group G2") == 1
    assert grouped.count("[E1 |") == grouped.count("[E2 |") == 1
    assert "latent" not in grouped.lower()
    assert "B1=render-bundle" in grouped


def test_renderer_rejects_format_injecting_nested_integer_subclasses() -> None:
    fixture = _fixture()
    injected_span = replace(
        fixture.packet.atoms[0].span,
        ordinal=_InjectInt(fixture.packet.atoms[0].span.ordinal),
    )
    injected_atom = replace(fixture.packet.atoms[0], span=injected_span)
    injected_atoms = (injected_atom, fixture.packet.atoms[1])
    assert type(injected_atom.span.ordinal) is _InjectInt

    with pytest.raises(TypeError, match="ordinal must be an exact integer"):
        render_grouped_evidence_context(
            injected_atoms,
            fixture.packet.bundles,
            (tuple(atom.atom_id for atom in injected_atoms),),
        )
    injected_packet = EvidencePacket(
        context=fixture.packet.context,
        atoms=injected_atoms,
        bundles=fixture.packet.bundles,
        receipt=fixture.packet.receipt,
    )
    with pytest.raises(TypeError, match="ordinal must be an exact integer"):
        _preflight_packet(injected_packet)


def test_renderer_rejects_receipt_counter_integer_subclasses() -> None:
    fixture = _fixture()
    injected_receipt = replace(
        fixture.packet.receipt,
        context_token_proxy=_InjectInt(
            fixture.packet.receipt.context_token_proxy
        ),
        receipt_sha256="",
    )
    assert type(injected_receipt.context_token_proxy) is _InjectInt
    injected_packet = replace(fixture.packet, receipt=injected_receipt)

    with pytest.raises(
        TypeError,
        match="context_token_proxy must be an exact integer",
    ):
        _preflight_packet(injected_packet)


def test_legacy_renderer_has_an_exact_non_circular_byte_golden() -> None:
    later_text = "Later exact bytes."
    later_span = EvidenceSpan(
        chunk_id="chunk-later",
        start_char=0,
        end_char=len(later_text),
        quote_sha256=quote_sha256(later_text),
        ordinal=2,
        source_id="source|later]",
        role="assistant",
        created_at="2026-08-20]",
    )
    later = EvidenceAtom(
        atom_id=make_atom_id(later_span),
        span=later_span,
        text=later_text,
        label="later|label]",
    )
    earlier_text = "Earlier exact bytes."
    earlier_span = EvidenceSpan(
        chunk_id="chunk|earlier]",
        start_char=0,
        end_char=len(earlier_text),
        quote_sha256=quote_sha256(earlier_text),
        ordinal=1,
        source_id="source\nearlier",
        role="user",
        created_at="2026-08-19",
    )
    earlier = EvidenceAtom(
        atom_id=make_atom_id(earlier_span),
        span=earlier_span,
        text=earlier_text,
        label="earlier]label",
    )
    bundles = (
        EvidenceBundle(
            bundle_id="bundle|all]",
            atom_ids=(later.atom_id, earlier.atom_id),
            obligation_ids=("answer|fact]",),
        ),
        EvidenceBundle(
            bundle_id="bundle-later",
            atom_ids=(later.atom_id,),
            obligation_ids=("detail",),
        ),
    )

    assert render_evidence_context((later, earlier), bundles) == (
        "## Source-grounded evidence\n\n"
        "[E1 | bundles=B1 | source=source earlier | ordinal=1 | "
        "chunk=chunk/earlier) | role=user | date=2026-08-19 | "
        "label=earlier)label]\nEarlier exact bytes.\n\n"
        "[E2 | bundles=B1,B2 | source=source/later) | ordinal=2 | "
        "chunk=chunk-later | role=assistant | date=2026-08-20) | "
        "label=later/label)]\nLater exact bytes.\n\n"
        "Bundle map:\n"
        "B1=bundle/all); obligations=answer/fact)\n"
        "B2=bundle-later; obligations=detail"
    )


def test_provider_free_core_returns_only_deterministic_structural_views() -> None:
    fixture = _fixture()
    first = _render(fixture)
    second = _render(fixture)

    assert first == second
    assert first.pair_wide_fallback_applied is False
    assert not hasattr(first, "receipt")
    assert not hasattr(first, "matched_pair_sha256")
    public_types = (
        render_models_module.FusionRenderArmReceipt,
        render_models_module.MatchedFusionRenderReceipt,
        render_models_module.RenderedFusionContext,
        render_models_module.MatchedFusionContexts,
    )
    assert not isinstance(first, public_types)
    assert not any(
        forbidden in name
        for name in _dataclass_field_names(first)
        for forbidden in (
            "receipt",
            "execution",
            "performance",
            "operation",
            "matched_pair",
        )
    )
    assert first.topology_only.effective_context != (
        first.latent_router.effective_context
    )
    for rendered in (first.topology_only, first.latent_router):
        for atom in fixture.packet.atoms:
            assert rendered.effective_context.count(atom.text) == 1
        assert rendered.effective_context.count("B1=render-bundle") == 1
        assert rendered.candidate.context_token_proxy == count_tokens(
            rendered.candidate.context
        )
        messages = (
            *fixture.base_messages,
            {
                "role": fixture.evidence_message_role,
                "content": (
                    fixture.evidence_prefix
                    + rendered.candidate.context
                    + fixture.evidence_suffix
                ),
            },
        )
        assert rendered.candidate.prompt.token_proxy == (
            count_chat_prompt_token_proxy(messages)
        )
    assert not _contains_tensor(first)


def test_one_context_overflow_falls_both_arms_back_to_exact_original_bytes() -> None:
    fixture = _fixture(single_context_overflow=True, prompt_room=True)
    rendered = _render(fixture)

    assert rendered.pair_wide_fallback_applied is True
    assert (
        rendered.topology_only.context_overflow,
        rendered.latent_router.context_overflow,
    ) == (False, True)
    assert rendered.topology_only.effective_context == fixture.packet.context
    assert rendered.latent_router.effective_context == fixture.packet.context
    assert rendered.topology_only.candidate.context_sha256 != (
        quote_sha256(rendered.topology_only.effective_context)
    )


def test_prompt_overflow_falls_both_arms_back_after_full_candidate_recount() -> None:
    fixture = _fixture(context_room=True, prompt_room=False)
    rendered = _render(fixture)

    assert rendered.pair_wide_fallback_applied is True
    assert rendered.topology_only.context_overflow is False
    assert rendered.latent_router.context_overflow is False
    assert (
        rendered.topology_only.prompt_overflow
        or rendered.latent_router.prompt_overflow
    )
    assert rendered.topology_only.effective_context == (
        rendered.latent_router.effective_context
    )
    assert rendered.topology_only.effective_context == fixture.packet.context
    assert rendered.topology_only.candidate.prompt.token_proxy > 0
    assert rendered.latent_router.candidate.prompt.token_proxy > 0


def test_exact_candidate_caps_are_inclusive_without_fallback() -> None:
    fixture = _fixture(exact_candidate_caps=True)
    rendered = _render(fixture)

    assert rendered.pair_wide_fallback_applied is False
    assert max(
        rendered.topology_only.candidate.context_token_proxy,
        rendered.latent_router.candidate.context_token_proxy,
    ) == fixture.packet.receipt.max_context_token_proxy
    assert max(
        rendered.topology_only.candidate.prompt.workspace_token_proxy,
        rendered.latent_router.candidate.prompt.workspace_token_proxy,
    ) == fixture.packet.receipt.max_prompt_token_proxy
    assert rendered.topology_only.context_overflow is False
    assert rendered.latent_router.context_overflow is False
    assert rendered.topology_only.prompt_overflow is False
    assert rendered.latent_router.prompt_overflow is False


def test_renderer_rejects_disabled_or_mismatched_prompt_framing() -> None:
    disabled = _fixture(prompt_enabled=False)
    with pytest.raises(ValueError, match="requires full prompt accounting"):
        _render(disabled)

    fixture = _fixture()
    with pytest.raises(ValueError, match="framing differs"):
        _render_structural_matched_contexts(
            fixture.packet,
            fixture.topology_only,
            fixture.latent_router,
            base_messages=fixture.base_messages,
            evidence_message_role=fixture.evidence_message_role,
            evidence_prefix=fixture.evidence_prefix + "changed",
            evidence_suffix=fixture.evidence_suffix,
        )


def test_public_renderer_rejects_non_pair_without_minting_execution_evidence() -> None:
    fixture = _fixture()
    with pytest.raises(TypeError, match="pair must be an exact"):
        render_matched_fusion_contexts(fixture.packet, object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    (
        "_assert_renderer_implementation",
        "_renderer_owned_objects",
        "_renderer_owned_fingerprint",
        "_renderer_implementation_sha256",
        "_render_structural_matched_contexts",
        "_arm_receipt",
        "_PINNED_ARM_RECEIPT_TYPE",
        "_PINNED_MATCHED_RECEIPT_TYPE",
        "_PINNED_RENDERED_CONTEXT_TYPE",
        "_PINNED_MATCHED_CONTEXTS_TYPE",
    ),
)
def test_public_renderer_rejects_stable_owned_seam_replacement(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    fixture = _fixture()
    if name.startswith("_PINNED_"):
        replacement = object()
    elif name == "_renderer_owned_fingerprint":
        replacement = (
            lambda *args, **kwargs: renderer_module._RENDERER_OWNED_FINGERPRINT
        )
    else:
        replacement = lambda *args, **kwargs: None
    monkeypatch.setattr(renderer_module, name, replacement)

    with pytest.raises(RuntimeError, match="owned matched fusion renderer"):
        render_matched_fusion_contexts(
            fixture.packet,
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "bypass_name",
    ("_assert_renderer_implementation", "_renderer_owned_fingerprint"),
)
def test_public_renderer_rejects_guard_bypass_plus_core_replacement(
    monkeypatch: pytest.MonkeyPatch,
    bypass_name: str,
) -> None:
    fixture = _fixture()
    bypass = (
        (lambda *args, **kwargs: renderer_module._RENDERER_OWNED_FINGERPRINT)
        if bypass_name == "_renderer_owned_fingerprint"
        else (lambda *args, **kwargs: None)
    )
    monkeypatch.setattr(renderer_module, bypass_name, bypass)
    monkeypatch.setattr(
        renderer_module,
        "_render_structural_matched_contexts",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(RuntimeError, match="owned matched fusion renderer"):
        render_matched_fusion_contexts(
            fixture.packet,
            object(),  # type: ignore[arg-type]
        )


def test_noncanonical_sealed_packet_context_rejects_instead_of_falling_back() -> None:
    fixture = _fixture()
    forged_context = fixture.packet.context + "\nforged but internally receipted"
    prompt_messages = (
        *fixture.base_messages,
        {
            "role": fixture.evidence_message_role,
            "content": fixture.evidence_prefix + forged_context + fixture.evidence_suffix,
        },
    )
    prompt_tokens = count_chat_prompt_token_proxy(prompt_messages)
    forged_receipt = replace(
        fixture.packet.receipt,
        context_sha256=quote_sha256(forged_context),
        context_token_proxy=count_tokens(forged_context),
        prompt_token_proxy=prompt_tokens,
        prompt_workspace_token_proxy=(
            prompt_tokens + fixture.packet.receipt.responder_output_token_reserve
        ),
        prompt_messages_sha256=identity_sha256(list(prompt_messages)),
        receipt_sha256="",
    )
    forged_packet = EvidencePacket(
        context=forged_context,
        atoms=fixture.packet.atoms,
        bundles=fixture.packet.bundles,
        receipt=forged_receipt,
    )
    with pytest.raises(ValueError, match="not the canonical exact packet rendering"):
        _render_structural_matched_contexts(
            forged_packet,
            fixture.topology_only,
            fixture.latent_router,
            base_messages=fixture.base_messages,
            evidence_message_role=fixture.evidence_message_role,
            evidence_prefix=fixture.evidence_prefix,
            evidence_suffix=fixture.evidence_suffix,
        )


def test_resealed_reversed_packet_atom_order_rejects_canonical_fallback_claim() -> None:
    fixture = _fixture()
    reversed_atoms = tuple(reversed(fixture.packet.atoms))
    reversed_receipt = replace(
        fixture.packet.receipt,
        selected_atom_ids=tuple(atom.atom_id for atom in reversed_atoms),
        receipt_sha256="",
    )
    reversed_packet = EvidencePacket(
        context=fixture.packet.context,
        atoms=reversed_atoms,
        bundles=fixture.packet.bundles,
        receipt=reversed_receipt,
    )

    with pytest.raises(ValueError, match="canonical source order"):
        _preflight_packet(reversed_packet)


def test_prompt_budget_is_immutable_and_detaches_advertised_mapping_sequences() -> None:
    source = [
        {"role": "system", "content": "immutable prompt"},
    ]
    budget = normalize_evidence_prompt_budget(
        base_messages=source,
        evidence_message_role="user",
        evidence_prefix="",
        evidence_suffix="",
        max_prompt_tokens=100,
        output_token_reserve=0,
    )
    assert type(budget) is EvidencePromptBudget
    assert budget.base_messages == (("system", "immutable prompt"),)
    source[0]["content"] = "mutated"
    source.append({"role": "user", "content": "late"})
    assert budget.messages("context")[0]["content"] == "immutable prompt"
    proxy = MappingProxyType({"role": "user", "content": "question"})
    fixture = _fixture()
    # A general Sequence[Mapping] is detached before identity/counting.
    with pytest.raises(ValueError, match="framing differs"):
        _render_structural_matched_contexts(
            fixture.packet,
            fixture.topology_only,
            fixture.latent_router,
            base_messages=(proxy,),
            evidence_message_role=fixture.evidence_message_role,
            evidence_prefix=fixture.evidence_prefix,
            evidence_suffix=fixture.evidence_suffix,
        )
    with pytest.raises(TypeError, match="exact integer"):
        EvidencePromptBudget(
            base_messages=(),
            evidence_message_role="user",
            evidence_prefix="",
            evidence_suffix="",
            max_prompt_tokens=100.0,  # type: ignore[arg-type]
            output_token_reserve=0,
        )
    with pytest.raises(TypeError, match="exact tuple"):
        EvidencePromptBudget(
            base_messages=({"role": "system", "content": "mutable"},),  # type: ignore[arg-type]
            evidence_message_role="user",
            evidence_prefix="",
            evidence_suffix="",
            max_prompt_tokens=100,
            output_token_reserve=0,
        )
    with pytest.raises(TypeError, match="exact integer"):
        EvidencePromptBudget(
            base_messages=(),
            evidence_message_role="user",
            evidence_prefix="",
            evidence_suffix="",
            max_prompt_tokens=True,  # type: ignore[arg-type]
            output_token_reserve=0,
        )
    with pytest.raises(TypeError, match="exact string"):
        budget.messages(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("target", "name"),
    (
        (evidence_packet_module, "_label_scalar"),
        (evidence_packet_module, "_atom_sort_key"),
        (discourse_module, "evidence_span_sort_key"),
        (
            render_models_module.FusionRenderArmReceipt,
            "_validate_budget_accounting",
        ),
        (
            render_models_module.FusionRenderArmReceipt,
            "_validate_effective_view",
        ),
    ),
)
def test_owned_renderer_dependency_replacement_rejects(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    name: str,
) -> None:
    fixture = _fixture()
    monkeypatch.setattr(target, name, lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="owned matched fusion renderer"):
        _render(fixture)


def test_group_partition_duplicates_are_rejected_not_deduplicated() -> None:
    fixture = _fixture()
    atom_ids = tuple(atom.atom_id for atom in fixture.packet.atoms)
    with pytest.raises(ValueError, match="partition the exact atom set once"):
        render_grouped_evidence_context(
            fixture.packet.atoms,
            fixture.packet.bundles,
            ((atom_ids[0], atom_ids[0]),),
        )
