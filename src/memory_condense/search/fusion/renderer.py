"""Deterministic matched rendering over exact resident fusion plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import memory_condense.domain._discourse_identity as discourse_identity_module
import memory_condense.domain._tokenizer as tokenizer_module
import memory_condense.domain.discourse as discourse_module
import memory_condense.domain.sealed as sealed_identity_module
import memory_condense.search.fusion.models as fusion_models_module
import memory_condense.search.fusion.render_models as render_models_module
import memory_condense.search.fusion.resident_models as resident_models_module
import memory_condense.search.packing.evidence_packet as evidence_packet_module
from memory_condense.domain._discourse_identity import (
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain._tokenizer import (
    DEFAULT_ENCODING,
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import (
    ClosureReceipt,
    EvidenceAtom,
    EvidenceBundle,
    EvidencePacket,
    EvidenceSpan,
)
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.fusion.models import (
    AuthoritativeHyperedge,
    FusionAtomRef,
)
from memory_condense.search.fusion.render_models import (
    FusionRenderArmReceipt,
    MatchedFusionContexts,
    MatchedFusionRenderReceipt,
    RenderedFusionContext,
)
from memory_condense.search.fusion.resident_models import (
    MatchedEvidenceFusionPair,
    ResidentEvidenceFusionPlan,
    resident_atom_order_sha256,
    resident_values_sha256,
)
from memory_condense.search.packing.evidence_packet import (
    EvidencePromptBudget,
    normalize_evidence_prompt_budget,
    render_evidence_context,
    render_grouped_evidence_context,
)


_PINNED_IDENTITY_SHA256 = identity_sha256
_PINNED_QUOTE_SHA256 = quote_sha256
_PINNED_COUNT_TOKENS = count_tokens
_PINNED_COUNT_CHAT_PROMPT = count_chat_prompt_token_proxy
_PINNED_TOKENIZER_IDENTITY = tokenizer_proxy_identity
_PINNED_NORMALIZE_PROMPT = normalize_evidence_prompt_budget
_PINNED_LEGACY_RENDERER = render_evidence_context
_PINNED_GROUP_RENDERER = render_grouped_evidence_context
_PINNED_RENDER_GROUPS = evidence_packet_module._render_evidence_groups
_PINNED_VALIDATE_RENDER_INPUTS = evidence_packet_module._validate_render_inputs
_PINNED_BUNDLE_LABELS = evidence_packet_module._bundle_labels
_PINNED_LABEL_SCALAR = evidence_packet_module._label_scalar
_PINNED_ATOM_SORT_KEY = evidence_packet_module._atom_sort_key
_PINNED_EVIDENCE_SPAN_SORT_KEY = discourse_module.evidence_span_sort_key
_PINNED_HEADER = evidence_packet_module._HEADER
_PINNED_VALUES_SHA256 = resident_values_sha256
_PINNED_ATOM_ORDER_SHA256 = resident_atom_order_sha256
_PINNED_PACKET_TYPE = EvidencePacket
_PINNED_RECEIPT_TYPE = ClosureReceipt
_PINNED_ATOM_TYPE = EvidenceAtom
_PINNED_BUNDLE_TYPE = EvidenceBundle
_PINNED_SPAN_TYPE = EvidenceSpan
_PINNED_PAIR_TYPE = MatchedEvidenceFusionPair
_PINNED_PAIR_POST_INIT = MatchedEvidenceFusionPair.__post_init__
_PINNED_PLAN_TYPE = ResidentEvidenceFusionPlan
_PINNED_REF_TYPE = FusionAtomRef
_PINNED_EDGE_TYPE = AuthoritativeHyperedge
_PINNED_PROMPT_BUDGET_TYPE = EvidencePromptBudget
_PINNED_PROMPT_BUDGET_INIT = EvidencePromptBudget.__init__
_PINNED_PROMPT_BUDGET_POST_INIT = EvidencePromptBudget.__post_init__
_PINNED_PROMPT_MESSAGES = EvidencePromptBudget.messages
_PINNED_PROMPT_TOKENS = EvidencePromptBudget.prompt_tokens
_PINNED_BASE_MESSAGES_SHA256 = EvidencePromptBudget.base_messages_sha256.fget
_PINNED_PREFIX_SHA256 = EvidencePromptBudget.evidence_prefix_sha256.fget
_PINNED_SUFFIX_SHA256 = EvidencePromptBudget.evidence_suffix_sha256.fget
_PINNED_PROMPT_MESSAGES_SHA256 = EvidencePromptBudget.prompt_messages_sha256
_PINNED_ARM_RECEIPT_TYPE = FusionRenderArmReceipt
_PINNED_ARM_RECEIPT_INIT = FusionRenderArmReceipt.__init__
_PINNED_ARM_RECEIPT_POST_INIT = FusionRenderArmReceipt.__post_init__
_PINNED_ARM_VALIDATE_BUDGET = FusionRenderArmReceipt._validate_budget_accounting
_PINNED_ARM_VALIDATE_EFFECTIVE = FusionRenderArmReceipt._validate_effective_view
_PINNED_MATCHED_RECEIPT_TYPE = MatchedFusionRenderReceipt
_PINNED_MATCHED_RECEIPT_INIT = MatchedFusionRenderReceipt.__init__
_PINNED_MATCHED_RECEIPT_POST_INIT = MatchedFusionRenderReceipt.__post_init__
_PINNED_RENDERED_CONTEXT_TYPE = RenderedFusionContext
_PINNED_RENDERED_CONTEXT_INIT = RenderedFusionContext.__init__
_PINNED_RENDERED_CONTEXT_POST_INIT = RenderedFusionContext.__post_init__
_PINNED_MATCHED_CONTEXTS_TYPE = MatchedFusionContexts
_PINNED_MATCHED_CONTEXTS_INIT = MatchedFusionContexts.__init__
_PINNED_MATCHED_CONTEXTS_POST_INIT = MatchedFusionContexts.__post_init__

_RENDERER_POLICY = {
    "format": "memory-condense-matched-fusion-render-policy-v1",
    "group_heading": "### Evidence group G{group_index_plus_one}",
    "evidence_numbering": "global_from_one",
    "bundle_labels": "packet_order_from_one",
    "latent_metadata_rendered": False,
    "fallback": "atomic_pair_original_packet_context",
    "budget_overflow_only_fallback": True,
    "candidate_prompt_always_recounted": True,
}
_RENDERER_POLICY_SHA256 = _PINNED_IDENTITY_SHA256(_RENDERER_POLICY)


@dataclass(frozen=True, slots=True)
class _PromptMeasurement:
    messages_sha256: str
    token_proxy: int
    workspace_token_proxy: int


@dataclass(frozen=True, slots=True)
class _CandidateRender:
    groups_sha256: str
    atom_order_sha256: str
    context: str
    context_sha256: str
    context_token_proxy: int
    prompt: _PromptMeasurement


@dataclass(frozen=True, slots=True)
class _StructuralArmRender:
    candidate: _CandidateRender
    context_overflow: bool
    prompt_overflow: bool
    effective_context: str


@dataclass(frozen=True, slots=True)
class _StructuralMatchedRender:
    topology_only: _StructuralArmRender
    latent_router: _StructuralArmRender
    original_context_token_proxy: int
    original_prompt: _PromptMeasurement
    renderer_implementation_sha256: str
    prompt_frame_sha256: str
    pair_wide_fallback_applied: bool


@dataclass(frozen=True, slots=True)
class _StructuralInputSnapshot:
    packet_receipt_sha256: str
    packet_context_sha256: str
    topology_plan_sha256: str
    latent_plan_sha256: str
    topology_groups_sha256: str
    latent_groups_sha256: str
    topology_atom_order_sha256: str
    latent_atom_order_sha256: str
    atom_refs_sha256: str
    hyperedges_sha256: str


@dataclass(frozen=True, slots=True)
class _CertifiedInputSnapshot:
    structural: _StructuralInputSnapshot
    matched_pair_sha256: str
    feature_operation_sha256: str
    feature_packet_receipt_sha256: str
    feature_closure_plan_sha256: str


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _renderer_implementation_sha256() -> str:
    return _PINNED_IDENTITY_SHA256(
        {
            "format": "memory-condense-matched-fusion-renderer-implementation-v1",
            "renderer_policy_sha256": _RENDERER_POLICY_SHA256,
            "source_files": {
                "renderer": _file_sha256(Path(__file__)),
                "render_models": _file_sha256(
                    Path(render_models_module.__file__)
                ),
                "evidence_packet": _file_sha256(
                    Path(evidence_packet_module.__file__)
                ),
                "tokenizer": _file_sha256(Path(tokenizer_module.__file__)),
                "discourse": _file_sha256(Path(discourse_module.__file__)),
                "fusion_models": _file_sha256(
                    Path(fusion_models_module.__file__)
                ),
                "resident_models": _file_sha256(
                    Path(resident_models_module.__file__)
                ),
                "sealed_identity": _file_sha256(
                    Path(sealed_identity_module.__file__)
                ),
                "discourse_identity": _file_sha256(
                    Path(discourse_identity_module.__file__)
                ),
            },
        }
    )


def _detach_base_messages(
    base_messages: Sequence[Mapping[str, str]] | None,
) -> tuple[dict[str, str], ...]:
    if base_messages is None:
        return ()
    if isinstance(base_messages, (str, bytes)):
        raise TypeError("base_messages must be a sequence of mappings")
    try:
        raw_messages = tuple(base_messages)
    except TypeError as exc:
        raise TypeError("base_messages must be a sequence of mappings") from exc
    detached: list[dict[str, str]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"base message {index} must be a mapping")
        if set(message) != {"role", "content"}:
            raise ValueError("base messages require only role and content")
        role = message["role"]
        content = message["content"]
        if type(role) is not str or type(content) is not str:
            raise TypeError("base message role and content must be exact strings")
        detached.append({"role": role, "content": content})
    return tuple(detached)


def _packet_atom_refs(packet: EvidencePacket) -> tuple[FusionAtomRef, ...]:
    return tuple(
        _PINNED_REF_TYPE(
            atom_id=atom.atom_id,
            atom_identity_sha256=_PINNED_IDENTITY_SHA256(atom.identity_payload()),
            span_identity_sha256=_PINNED_IDENTITY_SHA256(
                atom.span.identity_payload()
            ),
            quote_sha256=atom.span.quote_sha256,
        )
        for atom in packet.atoms
    )


def _packet_hyperedges(
    packet: EvidencePacket,
) -> tuple[AuthoritativeHyperedge, ...]:
    return tuple(
        _PINNED_EDGE_TYPE(
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


def _preflight_framing(
    *,
    encoding: str,
    base_messages: Sequence[Mapping[str, str]] | None,
    evidence_message_role: str,
    evidence_prefix: str,
    evidence_suffix: str,
) -> tuple[str, tuple[dict[str, str], ...]]:
    if type(encoding) is not str or not encoding or encoding != encoding.strip():
        raise ValueError("encoding must be an exact non-empty unpadded string")
    for name, value in (
        ("evidence_message_role", evidence_message_role),
        ("evidence_prefix", evidence_prefix),
        ("evidence_suffix", evidence_suffix),
    ):
        if type(value) is not str:
            raise TypeError(f"{name} must be an exact string")
    return encoding, _detach_base_messages(base_messages)


def _validate_closure_receipt_schema(receipt: ClosureReceipt) -> None:
    for name in (
        "plan_sha256",
        "context_sha256",
        "tokenizer_identity",
        "stopping_reason",
        "receipt_sha256",
    ):
        if type(getattr(receipt, name)) is not str:
            raise TypeError(f"closure receipt {name} must be an exact string")
    for name in ("selected_bundle_ids", "selected_atom_ids"):
        values = getattr(receipt, name)
        if type(values) is not tuple or any(
            type(value) is not str for value in values
        ):
            raise TypeError(
                f"closure receipt {name} must be an exact tuple of strings"
            )
    for name in (
        "context_token_proxy",
        "max_context_token_proxy",
        "retained_request_token_state_bytes",
        "responder_output_token_reserve",
    ):
        if type(getattr(receipt, name)) is not int:
            raise TypeError(f"closure receipt {name} must be an exact integer")
    if type(receipt.complete_claimed) is not bool:
        raise TypeError("closure receipt complete_claimed must be an exact boolean")
    for name in (
        "prompt_token_proxy",
        "max_prompt_token_proxy",
        "prompt_workspace_token_proxy",
    ):
        value = getattr(receipt, name)
        if value is not None and type(value) is not int:
            raise TypeError(
                f"closure receipt {name} must be an exact optional integer"
            )
    for name in (
        "base_messages_sha256",
        "evidence_message_role",
        "evidence_prefix_sha256",
        "evidence_suffix_sha256",
        "prompt_messages_sha256",
    ):
        value = getattr(receipt, name)
        if value is not None and type(value) is not str:
            raise TypeError(
                f"closure receipt {name} must be an exact optional string"
            )


def _preflight_packet(packet: object) -> EvidencePacket:
    if type(packet) is not _PINNED_PACKET_TYPE:
        raise TypeError("packet must be an exact EvidencePacket")
    if type(packet.receipt) is not _PINNED_RECEIPT_TYPE:
        raise TypeError("packet receipt must be an exact ClosureReceipt")
    _validate_closure_receipt_schema(packet.receipt)
    if type(packet.context) is not str:
        raise TypeError("packet context must be an exact string")
    if type(packet.atoms) is not tuple or any(
        type(atom) is not _PINNED_ATOM_TYPE for atom in packet.atoms
    ):
        raise TypeError("packet atoms must be an exact tuple of EvidenceAtom values")
    if type(packet.bundles) is not tuple or any(
        type(bundle) is not _PINNED_BUNDLE_TYPE for bundle in packet.bundles
    ):
        raise TypeError(
            "packet bundles must be an exact tuple of EvidenceBundle values"
        )
    if not packet.atoms:
        raise ValueError("matched fusion rendering requires packet atoms")
    _PINNED_VALIDATE_RENDER_INPUTS(packet.atoms, packet.bundles)

    packet.receipt._seal()
    if _PINNED_QUOTE_SHA256(packet.context) != packet.receipt.context_sha256:
        raise ValueError("packet context no longer matches its closure receipt")
    packet_atom_ids = tuple(atom.atom_id for atom in packet.atoms)
    if (
        len(packet_atom_ids) != len(set(packet_atom_ids))
        or packet.receipt.selected_atom_ids != packet_atom_ids
    ):
        raise ValueError("packet atom order disagrees with its closure receipt")
    canonical_atom_ids = tuple(
        atom.atom_id
        for atom in sorted(packet.atoms, key=_PINNED_ATOM_SORT_KEY)
    )
    if packet_atom_ids != canonical_atom_ids:
        raise ValueError("packet atoms are not in canonical source order")
    if packet.receipt.selected_bundle_ids != tuple(
        bundle.bundle_id for bundle in packet.bundles
    ):
        raise ValueError("packet bundle order disagrees with its closure receipt")
    if any(
        _PINNED_QUOTE_SHA256(atom.text) != atom.span.quote_sha256
        for atom in packet.atoms
    ):
        raise ValueError("packet atom text no longer matches its source span")
    if any(
        atom_id not in set(packet_atom_ids)
        for bundle in packet.bundles
        for atom_id in bundle.atom_ids
    ):
        raise ValueError("packet bundle references an unknown selected atom")
    if _PINNED_LEGACY_RENDERER(packet.atoms, packet.bundles) != packet.context:
        raise ValueError("packet context is not the canonical exact packet rendering")
    return packet


def _preflight_structural_plans(
    packet: object,
    topology_only: object,
    latent_router: object,
) -> tuple[
    EvidencePacket,
    ResidentEvidenceFusionPlan,
    ResidentEvidenceFusionPlan,
    tuple[FusionAtomRef, ...],
    tuple[AuthoritativeHyperedge, ...],
]:
    packet = _preflight_packet(packet)
    for plan in (topology_only, latent_router):
        if type(plan) is not _PINNED_PLAN_TYPE:
            raise TypeError("fusion arms must be exact ResidentEvidenceFusionPlan values")
        plan._seal()
    assert isinstance(topology_only, ResidentEvidenceFusionPlan)
    assert isinstance(latent_router, ResidentEvidenceFusionPlan)
    if topology_only.mode != "topology_only":
        raise ValueError("structural control plan must be topology_only")
    if latent_router.mode != "latent_router":
        raise ValueError("structural treatment plan must be latent_router")

    shared_plan_fields = (
        "feature_suboperation_sha256",
        "matched_input_sha256",
        "caps",
        "atoms",
        "hyperedges",
    )
    if any(
        getattr(topology_only, name) != getattr(latent_router, name)
        for name in shared_plan_fields
    ):
        raise ValueError("structural fusion plans do not share one exact input")

    atoms = _packet_atom_refs(packet)
    hyperedges = _packet_hyperedges(packet)
    packet_atom_ids = tuple(atom.atom_id for atom in packet.atoms)
    for plan in (topology_only, latent_router):
        if plan.atoms != atoms or plan.hyperedges != hyperedges:
            raise ValueError("fusion plan does not preserve the exact packet")
        grouped = tuple(
            atom_id for group in plan.groups for atom_id in group.atom_ids
        )
        if grouped != plan.atom_order or (
            len(grouped) != len(packet_atom_ids)
            or len(set(grouped)) != len(grouped)
            or set(grouped) != set(packet_atom_ids)
        ):
            raise ValueError("fusion groups must partition packet atoms exactly once")
    return packet, topology_only, latent_router, atoms, hyperedges


def _preflight_packet_pair(
    packet: object,
    pair: object,
) -> tuple[
    EvidencePacket,
    MatchedEvidenceFusionPair,
    tuple[FusionAtomRef, ...],
    tuple[AuthoritativeHyperedge, ...],
]:
    if type(pair) is not _PINNED_PAIR_TYPE:
        raise TypeError("pair must be an exact MatchedEvidenceFusionPair")
    _PINNED_PAIR_POST_INIT(pair)
    packet, _topology, _latent, atoms, hyperedges = (
        _preflight_structural_plans(
            packet,
            pair.topology_only,
            pair.latent_router,
        )
    )
    feature = pair.operation.feature_suboperation
    if feature.packet_receipt_sha256 != packet.receipt.receipt_sha256:
        raise ValueError("fusion feature operation belongs to another packet")
    if feature.closure_plan_sha256 != packet.receipt.plan_sha256:
        raise ValueError("packet and fusion feature operation bind different plans")
    if feature.atoms != atoms:
        raise ValueError("packet atom identities differ from the fusion operation")
    if feature.hyperedges != hyperedges:
        raise ValueError("packet hyperedges differ from the fusion operation")
    return packet, pair, atoms, hyperedges


def _structural_input_snapshot(
    packet: EvidencePacket,
    topology_only: ResidentEvidenceFusionPlan,
    latent_router: ResidentEvidenceFusionPlan,
    atoms: tuple[FusionAtomRef, ...],
    hyperedges: tuple[AuthoritativeHyperedge, ...],
) -> _StructuralInputSnapshot:
    return _StructuralInputSnapshot(
        packet_receipt_sha256=packet.receipt.receipt_sha256,
        packet_context_sha256=_PINNED_QUOTE_SHA256(packet.context),
        topology_plan_sha256=topology_only.plan_sha256,
        latent_plan_sha256=latent_router.plan_sha256,
        topology_groups_sha256=_PINNED_VALUES_SHA256(
            "fusion_render_groups", topology_only.groups
        ),
        latent_groups_sha256=_PINNED_VALUES_SHA256(
            "fusion_render_groups", latent_router.groups
        ),
        topology_atom_order_sha256=_PINNED_ATOM_ORDER_SHA256(
            topology_only.atom_order
        ),
        latent_atom_order_sha256=_PINNED_ATOM_ORDER_SHA256(
            latent_router.atom_order
        ),
        atom_refs_sha256=_PINNED_VALUES_SHA256(
            "fusion_render_atom_refs", atoms
        ),
        hyperedges_sha256=_PINNED_VALUES_SHA256(
            "fusion_render_hyperedges", hyperedges
        ),
    )


def _certified_input_snapshot(
    packet: EvidencePacket,
    pair: MatchedEvidenceFusionPair,
    atoms: tuple[FusionAtomRef, ...],
    hyperedges: tuple[AuthoritativeHyperedge, ...],
) -> _CertifiedInputSnapshot:
    feature = pair.operation.feature_suboperation
    return _CertifiedInputSnapshot(
        structural=_structural_input_snapshot(
            packet,
            pair.topology_only,
            pair.latent_router,
            atoms,
            hyperedges,
        ),
        matched_pair_sha256=pair.receipt.pair_sha256,
        feature_operation_sha256=feature.operation_sha256,
        feature_packet_receipt_sha256=feature.packet_receipt_sha256,
        feature_closure_plan_sha256=feature.closure_plan_sha256,
    )


def _prompt_budget(
    packet: EvidencePacket,
    *,
    encoding: str,
    base_messages: tuple[dict[str, str], ...],
    evidence_message_role: str,
    evidence_prefix: str,
    evidence_suffix: str,
) -> tuple[EvidencePromptBudget, str]:
    receipt = packet.receipt
    required_prompt_fields = (
        receipt.prompt_token_proxy,
        receipt.max_prompt_token_proxy,
        receipt.prompt_workspace_token_proxy,
        receipt.base_messages_sha256,
        receipt.evidence_message_role,
        receipt.evidence_prefix_sha256,
        receipt.evidence_suffix_sha256,
        receipt.prompt_messages_sha256,
    )
    if any(value is None for value in required_prompt_fields):
        raise ValueError("matched fusion renderer requires full prompt accounting")
    tokenizer_body = _PINNED_TOKENIZER_IDENTITY(encoding)
    tokenizer_identity = (
        f"{tokenizer_body['encoding']}:"
        f"{_PINNED_IDENTITY_SHA256(tokenizer_body)}"
    )
    if tokenizer_identity != receipt.tokenizer_identity:
        raise ValueError("renderer tokenizer identity differs from the packet")
    assert receipt.max_prompt_token_proxy is not None
    budget = _PINNED_NORMALIZE_PROMPT(
        base_messages=base_messages,
        evidence_message_role=evidence_message_role,
        evidence_prefix=evidence_prefix,
        evidence_suffix=evidence_suffix,
        max_prompt_tokens=receipt.max_prompt_token_proxy,
        output_token_reserve=receipt.responder_output_token_reserve,
    )
    if type(budget) is not _PINNED_PROMPT_BUDGET_TYPE:
        raise RuntimeError("full prompt accounting did not produce an owned budget")
    assert receipt.base_messages_sha256 is not None
    assert receipt.evidence_message_role is not None
    assert receipt.evidence_prefix_sha256 is not None
    assert receipt.evidence_suffix_sha256 is not None
    expected = (
        (_PINNED_BASE_MESSAGES_SHA256(budget), receipt.base_messages_sha256),
        (budget.evidence_message_role, receipt.evidence_message_role),
        (_PINNED_PREFIX_SHA256(budget), receipt.evidence_prefix_sha256),
        (_PINNED_SUFFIX_SHA256(budget), receipt.evidence_suffix_sha256),
    )
    if any(left != right for left, right in expected):
        raise ValueError("renderer prompt framing differs from the packet receipt")
    prompt_frame_sha256 = _PINNED_IDENTITY_SHA256(
        {
            "format": "memory-condense-evidence-prompt-frame-v1",
            "tokenizer_identity": tokenizer_identity,
            "base_messages_sha256": _PINNED_BASE_MESSAGES_SHA256(budget),
            "evidence_message_role": budget.evidence_message_role,
            "evidence_prefix_sha256": _PINNED_PREFIX_SHA256(budget),
            "evidence_suffix_sha256": _PINNED_SUFFIX_SHA256(budget),
            "max_prompt_token_proxy": budget.max_prompt_tokens,
            "responder_output_token_reserve": budget.output_token_reserve,
        }
    )
    return budget, prompt_frame_sha256


def _measure_prompt(
    context: str,
    *,
    budget: EvidencePromptBudget,
    encoding: str,
) -> _PromptMeasurement:
    messages = _PINNED_PROMPT_MESSAGES(budget, context)
    prompt_tokens = _PINNED_COUNT_CHAT_PROMPT(messages, encoding=encoding)
    return _PromptMeasurement(
        messages_sha256=_PINNED_IDENTITY_SHA256(list(messages)),
        token_proxy=prompt_tokens,
        workspace_token_proxy=prompt_tokens + budget.output_token_reserve,
    )


def _measure_original(
    packet: EvidencePacket,
    *,
    budget: EvidencePromptBudget,
    encoding: str,
) -> tuple[int, _PromptMeasurement]:
    context_tokens = _PINNED_COUNT_TOKENS(packet.context, encoding=encoding)
    prompt = _measure_prompt(packet.context, budget=budget, encoding=encoding)
    receipt = packet.receipt
    expected = (
        (context_tokens, receipt.context_token_proxy),
        (prompt.messages_sha256, receipt.prompt_messages_sha256),
        (prompt.token_proxy, receipt.prompt_token_proxy),
        (prompt.workspace_token_proxy, receipt.prompt_workspace_token_proxy),
    )
    if any(left != right for left, right in expected):
        raise ValueError("original packet budget recount differs from its receipt")
    return context_tokens, prompt


def _render_candidate(
    packet: EvidencePacket,
    plan: ResidentEvidenceFusionPlan,
    *,
    budget: EvidencePromptBudget,
    encoding: str,
) -> _CandidateRender:
    atom_groups = tuple(group.atom_ids for group in plan.groups)
    context = _PINNED_GROUP_RENDERER(packet.atoms, packet.bundles, atom_groups)
    context_tokens = _PINNED_COUNT_TOKENS(context, encoding=encoding)
    # The complete prompt is recounted even if the context cap already fails.
    prompt = _measure_prompt(context, budget=budget, encoding=encoding)
    return _CandidateRender(
        groups_sha256=_PINNED_VALUES_SHA256("fusion_render_groups", plan.groups),
        atom_order_sha256=_PINNED_ATOM_ORDER_SHA256(plan.atom_order),
        context=context,
        context_sha256=_PINNED_QUOTE_SHA256(context),
        context_token_proxy=context_tokens,
        prompt=prompt,
    )


def _arm_receipt(
    arm: _StructuralArmRender,
    *,
    plan: ResidentEvidenceFusionPlan,
    packet: EvidencePacket,
    matched_pair_sha256: str,
    implementation_sha256: str,
    prompt_frame_sha256: str,
    original_context_tokens: int,
    original_prompt: _PromptMeasurement,
    pair_fallback: bool,
) -> FusionRenderArmReceipt:
    candidate = arm.candidate
    receipt = packet.receipt
    assert receipt.max_prompt_token_proxy is not None
    context_compliant = (
        candidate.context_token_proxy <= receipt.max_context_token_proxy
    )
    prompt_compliant = (
        candidate.prompt.workspace_token_proxy <= receipt.max_prompt_token_proxy
    )
    original_order_sha256 = _PINNED_ATOM_ORDER_SHA256(
        receipt.selected_atom_ids
    )
    if pair_fallback:
        effective_order_sha256 = original_order_sha256
        effective_context_sha256 = receipt.context_sha256
        effective_context_tokens = original_context_tokens
        effective_prompt = original_prompt
    else:
        effective_order_sha256 = candidate.atom_order_sha256
        effective_context_sha256 = candidate.context_sha256
        effective_context_tokens = candidate.context_token_proxy
        effective_prompt = candidate.prompt
    return _PINNED_ARM_RECEIPT_TYPE(
        mode=plan.mode,
        packet_receipt_sha256=receipt.receipt_sha256,
        matched_pair_sha256=matched_pair_sha256,
        fusion_plan_sha256=plan.plan_sha256,
        renderer_implementation_sha256=implementation_sha256,
        prompt_frame_sha256=prompt_frame_sha256,
        groups_sha256=candidate.groups_sha256,
        candidate_atom_order_sha256=candidate.atom_order_sha256,
        original_atom_order_sha256=original_order_sha256,
        effective_atom_order_sha256=effective_order_sha256,
        candidate_context_sha256=candidate.context_sha256,
        candidate_context_token_proxy=candidate.context_token_proxy,
        candidate_prompt_messages_sha256=candidate.prompt.messages_sha256,
        candidate_prompt_token_proxy=candidate.prompt.token_proxy,
        candidate_prompt_workspace_token_proxy=(
            candidate.prompt.workspace_token_proxy
        ),
        original_context_sha256=receipt.context_sha256,
        original_context_token_proxy=original_context_tokens,
        original_prompt_messages_sha256=original_prompt.messages_sha256,
        original_prompt_token_proxy=original_prompt.token_proxy,
        original_prompt_workspace_token_proxy=(
            original_prompt.workspace_token_proxy
        ),
        effective_context_sha256=effective_context_sha256,
        effective_context_token_proxy=effective_context_tokens,
        effective_prompt_messages_sha256=effective_prompt.messages_sha256,
        effective_prompt_token_proxy=effective_prompt.token_proxy,
        effective_prompt_workspace_token_proxy=(
            effective_prompt.workspace_token_proxy
        ),
        max_context_token_proxy=receipt.max_context_token_proxy,
        max_prompt_token_proxy=receipt.max_prompt_token_proxy,
        responder_output_token_reserve=receipt.responder_output_token_reserve,
        context_cap_compliant=context_compliant,
        prompt_cap_compliant=prompt_compliant,
        pair_wide_fallback_applied=pair_fallback,
        plan_applied=not pair_fallback,
    )


def _render_structural_matched_contexts(
    packet: EvidencePacket,
    topology_only: ResidentEvidenceFusionPlan,
    latent_router: ResidentEvidenceFusionPlan,
    *,
    encoding: str = DEFAULT_ENCODING,
    base_messages: Sequence[Mapping[str, str]] | None = None,
    evidence_message_role: str = "user",
    evidence_prefix: str = "",
    evidence_suffix: str = "",
) -> _StructuralMatchedRender:
    """Provider-free structural rendering without execution attestations."""
    _assert_renderer_implementation()
    implementation_sha256 = _renderer_implementation_sha256()
    encoding, detached_messages = _preflight_framing(
        encoding=encoding,
        base_messages=base_messages,
        evidence_message_role=evidence_message_role,
        evidence_prefix=evidence_prefix,
        evidence_suffix=evidence_suffix,
    )
    packet, topology_only, latent_router, atoms, hyperedges = (
        _preflight_structural_plans(packet, topology_only, latent_router)
    )
    input_snapshot = _structural_input_snapshot(
        packet,
        topology_only,
        latent_router,
        atoms,
        hyperedges,
    )
    budget, prompt_frame_sha256 = _prompt_budget(
        packet,
        encoding=encoding,
        base_messages=detached_messages,
        evidence_message_role=evidence_message_role,
        evidence_prefix=evidence_prefix,
        evidence_suffix=evidence_suffix,
    )
    # Original bytes and the full original prompt are independently recounted
    # before either candidate can become an effective output.
    original_context_tokens, original_prompt = _measure_original(
        packet,
        budget=budget,
        encoding=encoding,
    )
    topology = _render_candidate(
        packet,
        topology_only,
        budget=budget,
        encoding=encoding,
    )
    latent = _render_candidate(
        packet,
        latent_router,
        budget=budget,
        encoding=encoding,
    )
    max_context = packet.receipt.max_context_token_proxy
    max_prompt = budget.max_prompt_tokens
    topology_context_overflow = topology.context_token_proxy > max_context
    topology_prompt_overflow = topology.prompt.workspace_token_proxy > max_prompt
    latent_context_overflow = latent.context_token_proxy > max_context
    latent_prompt_overflow = latent.prompt.workspace_token_proxy > max_prompt
    pair_fallback = any(
        (
            topology_context_overflow,
            topology_prompt_overflow,
            latent_context_overflow,
            latent_prompt_overflow,
        )
    )
    result = _StructuralMatchedRender(
        topology_only=_StructuralArmRender(
            candidate=topology,
            context_overflow=topology_context_overflow,
            prompt_overflow=topology_prompt_overflow,
            effective_context=packet.context if pair_fallback else topology.context,
        ),
        latent_router=_StructuralArmRender(
            candidate=latent,
            context_overflow=latent_context_overflow,
            prompt_overflow=latent_prompt_overflow,
            effective_context=packet.context if pair_fallback else latent.context,
        ),
        original_context_token_proxy=original_context_tokens,
        original_prompt=original_prompt,
        renderer_implementation_sha256=implementation_sha256,
        prompt_frame_sha256=prompt_frame_sha256,
        pair_wide_fallback_applied=pair_fallback,
    )
    _assert_renderer_implementation()
    if _renderer_implementation_sha256() != implementation_sha256:
        raise RuntimeError("matched fusion renderer changed during execution")
    packet_after, topology_after, latent_after, atoms_after, hyperedges_after = (
        _preflight_structural_plans(packet, topology_only, latent_router)
    )
    after_snapshot = _structural_input_snapshot(
        packet_after,
        topology_after,
        latent_after,
        atoms_after,
        hyperedges_after,
    )
    if after_snapshot != input_snapshot:
        raise RuntimeError("structural fusion rendering inputs changed during execution")
    return result


def _render_matched_fusion_contexts_impl(
    packet: EvidencePacket,
    pair: MatchedEvidenceFusionPair,
    guard,
    implementation_digest,
    preflight_pair,
    certified_snapshot,
    preflight_framing,
    structural_renderer,
    arm_receipt_builder,
    matched_receipt_type,
    rendered_context_type,
    matched_contexts_type,
    *,
    encoding: str = DEFAULT_ENCODING,
    base_messages: Sequence[Mapping[str, str]] | None = None,
    evidence_message_role: str = "user",
    evidence_prefix: str = "",
    evidence_suffix: str = "",
) -> MatchedFusionContexts:
    """Certify matched rendering only after validating one executed pair."""
    guard()
    implementation_sha256 = implementation_digest()
    packet, pair, atoms, hyperedges = preflight_pair(packet, pair)
    input_snapshot = certified_snapshot(packet, pair, atoms, hyperedges)
    encoding, detached_messages = preflight_framing(
        encoding=encoding,
        base_messages=base_messages,
        evidence_message_role=evidence_message_role,
        evidence_prefix=evidence_prefix,
        evidence_suffix=evidence_suffix,
    )
    structural = structural_renderer(
        packet,
        pair.topology_only,
        pair.latent_router,
        encoding=encoding,
        base_messages=detached_messages,
        evidence_message_role=evidence_message_role,
        evidence_prefix=evidence_prefix,
        evidence_suffix=evidence_suffix,
    )
    if structural.renderer_implementation_sha256 != implementation_sha256:
        raise RuntimeError("certified and structural renderer identities differ")
    topology = structural.topology_only
    latent = structural.latent_router
    pair_fallback = structural.pair_wide_fallback_applied
    topology_receipt = arm_receipt_builder(
        topology,
        plan=pair.topology_only,
        packet=packet,
        matched_pair_sha256=pair.receipt.pair_sha256,
        implementation_sha256=implementation_sha256,
        prompt_frame_sha256=structural.prompt_frame_sha256,
        original_context_tokens=structural.original_context_token_proxy,
        original_prompt=structural.original_prompt,
        pair_fallback=pair_fallback,
    )
    latent_receipt = arm_receipt_builder(
        latent,
        plan=pair.latent_router,
        packet=packet,
        matched_pair_sha256=pair.receipt.pair_sha256,
        implementation_sha256=implementation_sha256,
        prompt_frame_sha256=structural.prompt_frame_sha256,
        original_context_tokens=structural.original_context_token_proxy,
        original_prompt=structural.original_prompt,
        pair_fallback=pair_fallback,
    )
    matched_receipt = matched_receipt_type(
        packet_receipt_sha256=packet.receipt.receipt_sha256,
        matched_pair_sha256=pair.receipt.pair_sha256,
        renderer_implementation_sha256=implementation_sha256,
        prompt_frame_sha256=structural.prompt_frame_sha256,
        topology_render_receipt_sha256=(
            topology_receipt.render_receipt_sha256
        ),
        latent_render_receipt_sha256=latent_receipt.render_receipt_sha256,
        topology_context_overflow=topology.context_overflow,
        topology_prompt_overflow=topology.prompt_overflow,
        latent_context_overflow=latent.context_overflow,
        latent_prompt_overflow=latent.prompt_overflow,
        pair_wide_fallback_applied=pair_fallback,
    )
    result = matched_contexts_type(
        topology_only=rendered_context_type(
            context=topology.effective_context,
            receipt=topology_receipt,
        ),
        latent_router=rendered_context_type(
            context=latent.effective_context,
            receipt=latent_receipt,
        ),
        receipt=matched_receipt,
    )
    guard()
    if implementation_digest() != implementation_sha256:
        raise RuntimeError("matched fusion renderer changed during execution")
    packet_after, pair_after, atoms_after, hyperedges_after = (
        preflight_pair(packet, pair)
    )
    after_snapshot = certified_snapshot(
        packet_after,
        pair_after,
        atoms_after,
        hyperedges_after,
    )
    if after_snapshot != input_snapshot:
        raise RuntimeError("certified fusion rendering inputs changed during execution")
    return result


def _make_public_renderer(
    implementation,
    guard,
    implementation_digest,
    preflight_pair,
    certified_snapshot,
    preflight_framing,
    structural_renderer,
    arm_receipt_builder,
    matched_receipt_type,
    rendered_context_type,
    matched_contexts_type,
):
    def render_matched_fusion_contexts(
        packet: EvidencePacket,
        pair: MatchedEvidenceFusionPair,
        *,
        encoding: str = DEFAULT_ENCODING,
        base_messages: Sequence[Mapping[str, str]] | None = None,
        evidence_message_role: str = "user",
        evidence_prefix: str = "",
        evidence_suffix: str = "",
    ) -> MatchedFusionContexts:
        """Render a certified pair through captured owned implementation seams."""
        return implementation(
            packet,
            pair,
            guard,
            implementation_digest,
            preflight_pair,
            certified_snapshot,
            preflight_framing,
            structural_renderer,
            arm_receipt_builder,
            matched_receipt_type,
            rendered_context_type,
            matched_contexts_type,
            encoding=encoding,
            base_messages=base_messages,
            evidence_message_role=evidence_message_role,
            evidence_prefix=evidence_prefix,
            evidence_suffix=evidence_suffix,
        )

    return render_matched_fusion_contexts


def _renderer_owned_objects() -> tuple[object, ...]:
    return (
        render_matched_fusion_contexts,
        _render_matched_fusion_contexts_impl,
        _make_public_renderer,
        _renderer_owned_objects,
        _renderer_owned_fingerprint,
        _assert_renderer_dependencies,
        _make_renderer_guard,
        _assert_renderer_implementation,
        _render_structural_matched_contexts,
        _file_sha256,
        _renderer_implementation_sha256,
        _detach_base_messages,
        _packet_atom_refs,
        _packet_hyperedges,
        _preflight_framing,
        _validate_closure_receipt_schema,
        _preflight_packet,
        _preflight_structural_plans,
        _preflight_packet_pair,
        _structural_input_snapshot,
        _certified_input_snapshot,
        _prompt_budget,
        _measure_prompt,
        _measure_original,
        _render_candidate,
        _arm_receipt,
        _PromptMeasurement,
        _PromptMeasurement.__init__,
        _CandidateRender,
        _CandidateRender.__init__,
        _StructuralArmRender,
        _StructuralArmRender.__init__,
        _StructuralMatchedRender,
        _StructuralMatchedRender.__init__,
        _StructuralInputSnapshot,
        _StructuralInputSnapshot.__init__,
        _CertifiedInputSnapshot,
        _CertifiedInputSnapshot.__init__,
        _PINNED_IDENTITY_SHA256,
        _PINNED_QUOTE_SHA256,
        _PINNED_COUNT_TOKENS,
        _PINNED_COUNT_CHAT_PROMPT,
        _PINNED_TOKENIZER_IDENTITY,
        _PINNED_NORMALIZE_PROMPT,
        _PINNED_LEGACY_RENDERER,
        _PINNED_GROUP_RENDERER,
        _PINNED_RENDER_GROUPS,
        _PINNED_VALIDATE_RENDER_INPUTS,
        _PINNED_BUNDLE_LABELS,
        _PINNED_LABEL_SCALAR,
        _PINNED_ATOM_SORT_KEY,
        _PINNED_EVIDENCE_SPAN_SORT_KEY,
        _PINNED_HEADER,
        _PINNED_VALUES_SHA256,
        _PINNED_ATOM_ORDER_SHA256,
        _PINNED_PAIR_TYPE,
        _PINNED_PAIR_POST_INIT,
        _PINNED_PLAN_TYPE,
        _PINNED_PACKET_TYPE,
        _PINNED_RECEIPT_TYPE,
        _PINNED_ATOM_TYPE,
        _PINNED_BUNDLE_TYPE,
        _PINNED_SPAN_TYPE,
        _PINNED_REF_TYPE,
        _PINNED_EDGE_TYPE,
        _PINNED_PROMPT_BUDGET_TYPE,
        _PINNED_PROMPT_BUDGET_INIT,
        _PINNED_PROMPT_BUDGET_POST_INIT,
        _PINNED_PROMPT_MESSAGES,
        _PINNED_PROMPT_TOKENS,
        _PINNED_BASE_MESSAGES_SHA256,
        _PINNED_PREFIX_SHA256,
        _PINNED_SUFFIX_SHA256,
        _PINNED_PROMPT_MESSAGES_SHA256,
        _PINNED_ARM_RECEIPT_TYPE,
        _PINNED_ARM_RECEIPT_INIT,
        _PINNED_ARM_RECEIPT_POST_INIT,
        _PINNED_ARM_VALIDATE_BUDGET,
        _PINNED_ARM_VALIDATE_EFFECTIVE,
        _PINNED_MATCHED_RECEIPT_TYPE,
        _PINNED_MATCHED_RECEIPT_INIT,
        _PINNED_MATCHED_RECEIPT_POST_INIT,
        _PINNED_RENDERED_CONTEXT_TYPE,
        _PINNED_RENDERED_CONTEXT_INIT,
        _PINNED_RENDERED_CONTEXT_POST_INIT,
        _PINNED_MATCHED_CONTEXTS_TYPE,
        _PINNED_MATCHED_CONTEXTS_INIT,
        _PINNED_MATCHED_CONTEXTS_POST_INIT,
        SealedIdentity._seal,
        SealedIdentity.identity_payload,
    )


def _renderer_owned_fingerprint(
    values: tuple[object, ...],
) -> tuple[tuple[object, ...], ...]:
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
        for value in values
    )


def _assert_renderer_dependencies() -> None:
    if (
        evidence_packet_module.EvidencePromptBudget
        is not _PINNED_PROMPT_BUDGET_TYPE
        or evidence_packet_module.EvidenceAtom is not _PINNED_ATOM_TYPE
        or evidence_packet_module.EvidenceBundle is not _PINNED_BUNDLE_TYPE
        or evidence_packet_module.EvidenceSpan is not _PINNED_SPAN_TYPE
        or discourse_module.EvidenceAtom is not _PINNED_ATOM_TYPE
        or discourse_module.EvidenceBundle is not _PINNED_BUNDLE_TYPE
        or discourse_module.EvidenceSpan is not _PINNED_SPAN_TYPE
        or evidence_packet_module.normalize_evidence_prompt_budget
        is not _PINNED_NORMALIZE_PROMPT
        or evidence_packet_module.render_evidence_context
        is not _PINNED_LEGACY_RENDERER
        or evidence_packet_module.render_grouped_evidence_context
        is not _PINNED_GROUP_RENDERER
        or evidence_packet_module._render_evidence_groups
        is not _PINNED_RENDER_GROUPS
        or evidence_packet_module._validate_render_inputs
        is not _PINNED_VALIDATE_RENDER_INPUTS
        or evidence_packet_module._bundle_labels is not _PINNED_BUNDLE_LABELS
        or evidence_packet_module._label_scalar is not _PINNED_LABEL_SCALAR
        or evidence_packet_module._atom_sort_key is not _PINNED_ATOM_SORT_KEY
        or evidence_packet_module.evidence_span_sort_key
        is not _PINNED_EVIDENCE_SPAN_SORT_KEY
        or discourse_module.evidence_span_sort_key
        is not _PINNED_EVIDENCE_SPAN_SORT_KEY
        or evidence_packet_module._HEADER != _PINNED_HEADER
        or _PINNED_PROMPT_BUDGET_TYPE.__init__
        is not _PINNED_PROMPT_BUDGET_INIT
        or _PINNED_PROMPT_BUDGET_TYPE.__post_init__
        is not _PINNED_PROMPT_BUDGET_POST_INIT
        or _PINNED_PROMPT_BUDGET_TYPE.messages is not _PINNED_PROMPT_MESSAGES
        or _PINNED_PROMPT_BUDGET_TYPE.prompt_tokens is not _PINNED_PROMPT_TOKENS
        or _PINNED_PROMPT_BUDGET_TYPE.base_messages_sha256.fget
        is not _PINNED_BASE_MESSAGES_SHA256
        or _PINNED_PROMPT_BUDGET_TYPE.evidence_prefix_sha256.fget
        is not _PINNED_PREFIX_SHA256
        or _PINNED_PROMPT_BUDGET_TYPE.evidence_suffix_sha256.fget
        is not _PINNED_SUFFIX_SHA256
        or _PINNED_PROMPT_BUDGET_TYPE.prompt_messages_sha256
        is not _PINNED_PROMPT_MESSAGES_SHA256
        or tokenizer_module.count_tokens is not _PINNED_COUNT_TOKENS
        or tokenizer_module.count_chat_prompt_token_proxy
        is not _PINNED_COUNT_CHAT_PROMPT
        or tokenizer_module.tokenizer_proxy_identity
        is not _PINNED_TOKENIZER_IDENTITY
        or resident_models_module.resident_values_sha256
        is not _PINNED_VALUES_SHA256
        or resident_models_module.resident_atom_order_sha256
        is not _PINNED_ATOM_ORDER_SHA256
        or resident_models_module.MatchedEvidenceFusionPair
        is not _PINNED_PAIR_TYPE
        or _PINNED_PAIR_TYPE.__post_init__ is not _PINNED_PAIR_POST_INIT
        or render_models_module.FusionRenderArmReceipt
        is not _PINNED_ARM_RECEIPT_TYPE
        or _PINNED_ARM_RECEIPT_TYPE.__init__ is not _PINNED_ARM_RECEIPT_INIT
        or _PINNED_ARM_RECEIPT_TYPE.__post_init__
        is not _PINNED_ARM_RECEIPT_POST_INIT
        or _PINNED_ARM_RECEIPT_TYPE._validate_budget_accounting
        is not _PINNED_ARM_VALIDATE_BUDGET
        or _PINNED_ARM_RECEIPT_TYPE._validate_effective_view
        is not _PINNED_ARM_VALIDATE_EFFECTIVE
        or render_models_module.MatchedFusionRenderReceipt
        is not _PINNED_MATCHED_RECEIPT_TYPE
        or _PINNED_MATCHED_RECEIPT_TYPE.__init__
        is not _PINNED_MATCHED_RECEIPT_INIT
        or _PINNED_MATCHED_RECEIPT_TYPE.__post_init__
        is not _PINNED_MATCHED_RECEIPT_POST_INIT
        or render_models_module.RenderedFusionContext
        is not _PINNED_RENDERED_CONTEXT_TYPE
        or _PINNED_RENDERED_CONTEXT_TYPE.__init__
        is not _PINNED_RENDERED_CONTEXT_INIT
        or _PINNED_RENDERED_CONTEXT_TYPE.__post_init__
        is not _PINNED_RENDERED_CONTEXT_POST_INIT
        or render_models_module.MatchedFusionContexts
        is not _PINNED_MATCHED_CONTEXTS_TYPE
        or _PINNED_MATCHED_CONTEXTS_TYPE.__init__
        is not _PINNED_MATCHED_CONTEXTS_INIT
        or _PINNED_MATCHED_CONTEXTS_TYPE.__post_init__
        is not _PINNED_MATCHED_CONTEXTS_POST_INIT
        or discourse_identity_module.identity_sha256
        is not _PINNED_IDENTITY_SHA256
        or discourse_identity_module.quote_sha256 is not _PINNED_QUOTE_SHA256
    ):
        raise RuntimeError("owned matched fusion renderer dependency was replaced")


def _make_renderer_guard(
    owned_objects,
    owned_fingerprint,
    assert_dependencies,
):
    expected: tuple[tuple[object, ...], ...] | None = None

    def assert_renderer_implementation() -> None:
        if expected is None:
            raise RuntimeError("matched fusion renderer guard is not sealed")
        try:
            observed = owned_fingerprint(owned_objects())
        except Exception as exc:
            raise RuntimeError("owned matched fusion renderer was replaced") from exc
        if observed != expected:
            raise RuntimeError("owned matched fusion renderer was replaced")
        try:
            assert_dependencies()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "owned matched fusion renderer dependency was replaced"
            ) from exc

    def seal() -> tuple[tuple[object, ...], ...]:
        nonlocal expected
        if expected is not None:
            raise RuntimeError("matched fusion renderer guard is already sealed")
        assert_dependencies()
        expected = owned_fingerprint(owned_objects())
        return expected

    return assert_renderer_implementation, seal


_assert_renderer_implementation, _seal_renderer_guard = _make_renderer_guard(
    _renderer_owned_objects,
    _renderer_owned_fingerprint,
    _assert_renderer_dependencies,
)
render_matched_fusion_contexts = _make_public_renderer(
    _render_matched_fusion_contexts_impl,
    _assert_renderer_implementation,
    _renderer_implementation_sha256,
    _preflight_packet_pair,
    _certified_input_snapshot,
    _preflight_framing,
    _render_structural_matched_contexts,
    _arm_receipt,
    _PINNED_MATCHED_RECEIPT_TYPE,
    _PINNED_RENDERED_CONTEXT_TYPE,
    _PINNED_MATCHED_CONTEXTS_TYPE,
)
_RENDERER_OWNED_FINGERPRINT = _seal_renderer_guard()
del _seal_renderer_guard


__all__ = ["render_matched_fusion_contexts"]
