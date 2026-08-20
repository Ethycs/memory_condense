"""Text-free receipts and text-bearing views for matched fusion rendering."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Literal

from memory_condense.domain._discourse_identity import (
    _sha256,
    _strict_int,
    normalize_fields,
    quote_sha256,
)
from memory_condense.domain.sealed import SealedIdentity


FusionRenderMode = Literal["topology_only", "latent_router"]


def _exact_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    return _sha256(value, label)


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    normalized = _strict_int(value, label)
    if normalized < 0:
        raise ValueError(f"{label} must be non-negative")
    return normalized


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact boolean")
    return value


def _require_plain_identity_tree(value: object, label: str) -> None:
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


@dataclass(frozen=True, slots=True)
class FusionRenderArmReceipt(SealedIdentity):
    """One text-free candidate/effective rendering receipt."""

    _SEAL_FIELD = "render_receipt_sha256"
    _SEAL_MISMATCH = "fusion render receipt SHA-256 does not match its contents"

    mode: FusionRenderMode
    packet_receipt_sha256: str
    matched_pair_sha256: str
    fusion_plan_sha256: str
    renderer_implementation_sha256: str
    prompt_frame_sha256: str
    groups_sha256: str
    candidate_atom_order_sha256: str
    original_atom_order_sha256: str
    effective_atom_order_sha256: str
    candidate_context_sha256: str
    candidate_context_token_proxy: int
    candidate_prompt_messages_sha256: str
    candidate_prompt_token_proxy: int
    candidate_prompt_workspace_token_proxy: int
    original_context_sha256: str
    original_context_token_proxy: int
    original_prompt_messages_sha256: str
    original_prompt_token_proxy: int
    original_prompt_workspace_token_proxy: int
    effective_context_sha256: str
    effective_context_token_proxy: int
    effective_prompt_messages_sha256: str
    effective_prompt_token_proxy: int
    effective_prompt_workspace_token_proxy: int
    max_context_token_proxy: int
    max_prompt_token_proxy: int
    responder_output_token_reserve: int
    context_cap_compliant: bool
    prompt_cap_compliant: bool
    pair_wide_fallback_applied: bool
    plan_applied: bool
    render_format: str = "fusion_render_arm_v1"
    exact_atom_set_preserved: bool = True
    exact_bundle_set_preserved: bool = True
    exact_evidence_bytes_preserved: bool = True
    retained_request_tensor_bytes: int = 0
    render_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in {
            "topology_only",
            "latent_router",
        }:
            raise ValueError("fusion render mode is unsupported")
        if type(self.render_format) is not str or self.render_format != (
            "fusion_render_arm_v1"
        ):
            raise ValueError("fusion render arm format is unsupported")
        sha_fields = (
            "packet_receipt_sha256",
            "matched_pair_sha256",
            "fusion_plan_sha256",
            "renderer_implementation_sha256",
            "prompt_frame_sha256",
            "groups_sha256",
            "candidate_atom_order_sha256",
            "original_atom_order_sha256",
            "effective_atom_order_sha256",
            "candidate_context_sha256",
            "candidate_prompt_messages_sha256",
            "original_context_sha256",
            "original_prompt_messages_sha256",
            "effective_context_sha256",
            "effective_prompt_messages_sha256",
        )
        for name in sha_fields:
            object.__setattr__(self, name, _exact_sha256(getattr(self, name), name))
        int_fields = (
            "candidate_context_token_proxy",
            "candidate_prompt_token_proxy",
            "candidate_prompt_workspace_token_proxy",
            "original_context_token_proxy",
            "original_prompt_token_proxy",
            "original_prompt_workspace_token_proxy",
            "effective_context_token_proxy",
            "effective_prompt_token_proxy",
            "effective_prompt_workspace_token_proxy",
            "max_context_token_proxy",
            "max_prompt_token_proxy",
            "responder_output_token_reserve",
            "retained_request_tensor_bytes",
        )
        for name in int_fields:
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        bool_fields = (
            "context_cap_compliant",
            "prompt_cap_compliant",
            "pair_wide_fallback_applied",
            "plan_applied",
            "exact_atom_set_preserved",
            "exact_bundle_set_preserved",
            "exact_evidence_bytes_preserved",
        )
        for name in bool_fields:
            object.__setattr__(self, name, _exact_bool(getattr(self, name), name))
        self._validate_budget_accounting()
        self._validate_effective_view()
        if not (
            self.exact_atom_set_preserved
            and self.exact_bundle_set_preserved
            and self.exact_evidence_bytes_preserved
        ):
            raise ValueError("fusion renderer cannot weaken extractive preservation")
        if self.retained_request_tensor_bytes != 0:
            raise ValueError("fusion render receipts cannot retain request tensors")
        _require_plain_identity_tree(self, "fusion render receipt")
        self._seal()

    def _validate_budget_accounting(self) -> None:
        context_compliant = (
            self.candidate_context_token_proxy <= self.max_context_token_proxy
        )
        if self.context_cap_compliant != context_compliant:
            raise ValueError("candidate context compliance disagrees with its count")
        if self.candidate_prompt_workspace_token_proxy != (
            self.candidate_prompt_token_proxy + self.responder_output_token_reserve
        ):
            raise ValueError("candidate prompt workspace accounting changed")
        if self.original_prompt_workspace_token_proxy != (
            self.original_prompt_token_proxy + self.responder_output_token_reserve
        ):
            raise ValueError("original prompt workspace accounting changed")
        if self.effective_prompt_workspace_token_proxy != (
            self.effective_prompt_token_proxy + self.responder_output_token_reserve
        ):
            raise ValueError("effective prompt workspace accounting changed")
        prompt_compliant = (
            self.candidate_prompt_workspace_token_proxy
            <= self.max_prompt_token_proxy
        )
        if self.prompt_cap_compliant != prompt_compliant:
            raise ValueError("candidate prompt compliance disagrees with its count")
        if (
            self.original_prompt_workspace_token_proxy > self.max_prompt_token_proxy
            or self.effective_prompt_workspace_token_proxy > self.max_prompt_token_proxy
        ):
            raise ValueError("original or effective prompt exceeds its hard cap")
        if self.original_context_token_proxy > self.max_context_token_proxy:
            raise ValueError("original packet context exceeds its hard cap")
        if self.effective_context_token_proxy > self.max_context_token_proxy:
            raise ValueError("effective context exceeds its hard cap")

    def _validate_effective_view(self) -> None:
        if self.plan_applied == self.pair_wide_fallback_applied:
            raise ValueError("plan application must be inverse of pair fallback")
        if self.pair_wide_fallback_applied:
            expected = (
                (self.effective_atom_order_sha256, self.original_atom_order_sha256),
                (self.effective_context_sha256, self.original_context_sha256),
                (
                    self.effective_context_token_proxy,
                    self.original_context_token_proxy,
                ),
                (
                    self.effective_prompt_messages_sha256,
                    self.original_prompt_messages_sha256,
                ),
                (
                    self.effective_prompt_token_proxy,
                    self.original_prompt_token_proxy,
                ),
                (
                    self.effective_prompt_workspace_token_proxy,
                    self.original_prompt_workspace_token_proxy,
                ),
            )
        else:
            expected = (
                (self.effective_atom_order_sha256, self.candidate_atom_order_sha256),
                (self.effective_context_sha256, self.candidate_context_sha256),
                (
                    self.effective_context_token_proxy,
                    self.candidate_context_token_proxy,
                ),
                (
                    self.effective_prompt_messages_sha256,
                    self.candidate_prompt_messages_sha256,
                ),
                (
                    self.effective_prompt_token_proxy,
                    self.candidate_prompt_token_proxy,
                ),
                (
                    self.effective_prompt_workspace_token_proxy,
                    self.candidate_prompt_workspace_token_proxy,
                ),
            )
        if any(left != right for left, right in expected):
            raise ValueError("effective fusion rendering does not match its source")


@dataclass(frozen=True, slots=True)
class MatchedFusionRenderReceipt(SealedIdentity):
    """Text-free atomic join over both candidate/effective arm receipts."""

    _SEAL_FIELD = "matched_render_sha256"
    _SEAL_MISMATCH = "matched render SHA-256 does not match its contents"

    packet_receipt_sha256: str
    matched_pair_sha256: str
    renderer_implementation_sha256: str
    prompt_frame_sha256: str
    topology_render_receipt_sha256: str
    latent_render_receipt_sha256: str
    topology_context_overflow: bool
    topology_prompt_overflow: bool
    latent_context_overflow: bool
    latent_prompt_overflow: bool
    pair_wide_fallback_applied: bool
    render_format: str = "matched_fusion_render_v1"
    atomic_pair_fallback_attested: bool = True
    shared_prompt_frame_attested: bool = True
    matched_render_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.render_format) is not str or self.render_format != (
            "matched_fusion_render_v1"
        ):
            raise ValueError("matched fusion render format is unsupported")
        normalize_fields(
            self,
            packet_receipt_sha256=_exact_sha256,
            matched_pair_sha256=_exact_sha256,
            renderer_implementation_sha256=_exact_sha256,
            prompt_frame_sha256=_exact_sha256,
            topology_render_receipt_sha256=_exact_sha256,
            latent_render_receipt_sha256=_exact_sha256,
        )
        bool_fields = (
            "topology_context_overflow",
            "topology_prompt_overflow",
            "latent_context_overflow",
            "latent_prompt_overflow",
            "pair_wide_fallback_applied",
            "atomic_pair_fallback_attested",
            "shared_prompt_frame_attested",
        )
        for name in bool_fields:
            object.__setattr__(self, name, _exact_bool(getattr(self, name), name))
        overflow = any(
            (
                self.topology_context_overflow,
                self.topology_prompt_overflow,
                self.latent_context_overflow,
                self.latent_prompt_overflow,
            )
        )
        if self.pair_wide_fallback_applied != overflow:
            raise ValueError("pair-wide fallback must equal the candidate overflow union")
        if not self.atomic_pair_fallback_attested or not self.shared_prompt_frame_attested:
            raise ValueError("matched renderer cannot weaken its shared-arm contract")
        _require_plain_identity_tree(self, "matched fusion render receipt")
        self._seal()


@dataclass(frozen=True, slots=True)
class RenderedFusionContext:
    """One effective text context plus its text-free receipt."""

    context: str
    receipt: FusionRenderArmReceipt

    def __post_init__(self) -> None:
        if type(self.context) is not str:
            raise TypeError("rendered fusion context must be an exact string")
        if type(self.receipt) is not FusionRenderArmReceipt:
            raise TypeError("rendered fusion context requires its exact receipt")
        self.receipt._seal()
        if quote_sha256(self.context) != self.receipt.effective_context_sha256:
            raise ValueError("rendered fusion context disagrees with its receipt")


@dataclass(frozen=True, slots=True)
class MatchedFusionContexts:
    """Atomic matched control/treatment contexts and their shared receipt."""

    topology_only: RenderedFusionContext
    latent_router: RenderedFusionContext
    receipt: MatchedFusionRenderReceipt

    def __post_init__(self) -> None:
        if type(self.topology_only) is not RenderedFusionContext:
            raise TypeError("topology_only must be an exact RenderedFusionContext")
        if type(self.latent_router) is not RenderedFusionContext:
            raise TypeError("latent_router must be an exact RenderedFusionContext")
        if type(self.receipt) is not MatchedFusionRenderReceipt:
            raise TypeError("receipt must be an exact MatchedFusionRenderReceipt")
        control = self.topology_only.receipt
        treatment = self.latent_router.receipt
        control._seal()
        treatment._seal()
        self.receipt._seal()
        if control.mode != "topology_only" or treatment.mode != "latent_router":
            raise ValueError("matched render arms are in the wrong modes")
        shared = (
            "packet_receipt_sha256",
            "matched_pair_sha256",
            "renderer_implementation_sha256",
            "prompt_frame_sha256",
            "pair_wide_fallback_applied",
            "original_atom_order_sha256",
            "original_context_sha256",
            "original_context_token_proxy",
            "original_prompt_messages_sha256",
            "original_prompt_token_proxy",
            "original_prompt_workspace_token_proxy",
            "max_context_token_proxy",
            "max_prompt_token_proxy",
            "responder_output_token_reserve",
        )
        if any(getattr(control, name) != getattr(treatment, name) for name in shared):
            raise ValueError("matched render arms do not share one exact input")
        joins = {
            "packet_receipt_sha256": control.packet_receipt_sha256,
            "matched_pair_sha256": control.matched_pair_sha256,
            "renderer_implementation_sha256": control.renderer_implementation_sha256,
            "prompt_frame_sha256": control.prompt_frame_sha256,
            "topology_render_receipt_sha256": control.render_receipt_sha256,
            "latent_render_receipt_sha256": treatment.render_receipt_sha256,
            "topology_context_overflow": not control.context_cap_compliant,
            "topology_prompt_overflow": not control.prompt_cap_compliant,
            "latent_context_overflow": not treatment.context_cap_compliant,
            "latent_prompt_overflow": not treatment.prompt_cap_compliant,
            "pair_wide_fallback_applied": control.pair_wide_fallback_applied,
        }
        if any(getattr(self.receipt, name) != value for name, value in joins.items()):
            raise ValueError("matched render receipt does not bind both exact arms")
        if self.receipt.pair_wide_fallback_applied and (
            self.topology_only.context != self.latent_router.context
        ):
            raise ValueError("pair-wide fallback contexts must be byte-identical")


__all__ = [
    "FusionRenderArmReceipt",
    "FusionRenderMode",
    "MatchedFusionContexts",
    "MatchedFusionRenderReceipt",
    "RenderedFusionContext",
]
