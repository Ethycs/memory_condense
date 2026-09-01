"""Deterministic closure, consensus, and answer-operation executors.

All functions consume only a question-derived :class:`TypedOperatorSpec` and a
sealed :class:`TypedEvidencePacket`.  They make no model call, retain no model
state, and return content-addressed receipts.  Unsupported synthesis/direct
generation remains explicitly non-deterministic instead of being guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_operator_adapter import (
    ConflictPolicy,
    ContentCoherence,
    EvidenceStatus,
    NumericRole,
    TypedEvidenceItem,
    TypedEvidencePacket,
    TypedItemKind,
    ValueAuthority,
)
from .typed_numeric_semantics import NumericQualifier
from .typed_operator_spec import (
    AnswerShape,
    ComparisonMode,
    SlotKind,
    TemporalMode,
    TypedOperatorSpec,
    normalized_terms,
)


CLOSURE_FORMAT = "memory-condense-typed-slot-closure-v1"
CONSENSUS_FORMAT = "memory-condense-typed-evidence-consensus-v2"
EXECUTION_FORMAT = "memory-condense-typed-operator-execution-v2"
PRESERVATION_FORMAT = "memory-condense-candidate-preservation-v1"


class ExecutionStatus(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    CONFLICTED = "conflicted"
    NON_DETERMINISTIC = "non_deterministic"


class ExecutorKind(str, Enum):
    NUMERIC = "numeric"
    TIME = "time"
    SET = "set"
    STATE = "state"
    NONE = "none"


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(row) is not str or not row for row in values):
        raise MatchedEvalContractError(f"{label} must be an exact text tuple")
    if len(set(values)) != len(values):
        raise MatchedEvalContractError(f"{label} must be ordered and unique")
    return values


@dataclass(frozen=True, slots=True)
class SlotBindingReceipt:
    slot_id: str
    item_receipt_sha256s: tuple[str, ...]
    handle_ids: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.slot_id, "slot binding slot")
        _ordered_unique(self.item_receipt_sha256s, "slot binding items")
        _ordered_unique(self.handle_ids, "slot binding handles")
        if not self.item_receipt_sha256s or not self.handle_ids:
            raise MatchedEvalContractError("slot binding requires evidence")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("slot binding receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "handle_ids": list(self.handle_ids),
            "item_receipt_sha256s": list(self.item_receipt_sha256s),
            "slot_id": self.slot_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SlotClosureReceipt:
    operator_spec_receipt_sha256: str
    evidence_packet_receipt_sha256: str
    frontier_receipt_sha256: str
    bindings: tuple[SlotBindingReceipt, ...]
    bound_slot_ids: tuple[str, ...]
    missing_slot_ids: tuple[str, ...]
    conflicted_slot_ids: tuple[str, ...]
    usable_item_receipt_sha256s: tuple[str, ...]
    sufficient: bool
    complete_frontier_required: bool
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.operator_spec_receipt_sha256, "closure operator spec"),
            (self.evidence_packet_receipt_sha256, "closure evidence packet"),
            (self.frontier_receipt_sha256, "closure frontier"),
        ):
            require_sha256(value, label)
        if type(self.bindings) is not tuple or any(type(row) is not SlotBindingReceipt for row in self.bindings):
            raise MatchedEvalContractError("closure bindings changed type")
        bound = _ordered_unique(self.bound_slot_ids, "bound slots")
        missing = _ordered_unique(self.missing_slot_ids, "missing slots")
        conflicted = _ordered_unique(self.conflicted_slot_ids, "conflicted slots")
        _ordered_unique(self.usable_item_receipt_sha256s, "usable item receipts")
        if set(bound) & set(missing) or set(bound) & set(conflicted) or set(missing) & set(conflicted):
            raise MatchedEvalContractError("closure slot partitions overlap")
        if tuple(row.slot_id for row in self.bindings) != bound:
            raise MatchedEvalContractError("closure bindings disagree with bound slots")
        if type(self.sufficient) is not bool or type(self.complete_frontier_required) is not bool:
            raise MatchedEvalContractError("closure flags must be exact")
        if self.provider_prompt_count != 0 or self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("closure must remain provider-free and zero-state")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("slot closure receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="slot_closure")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "bindings": [row.projection() for row in self.bindings],
            "bound_slot_ids": list(self.bound_slot_ids),
            "complete_frontier_required": self.complete_frontier_required,
            "conflicted_slot_ids": list(self.conflicted_slot_ids),
            "evidence_packet_receipt_sha256": self.evidence_packet_receipt_sha256,
            "format": CLOSURE_FORMAT,
            "frontier_receipt_sha256": self.frontier_receipt_sha256,
            "missing_slot_ids": list(self.missing_slot_ids),
            "operator_spec_receipt_sha256": self.operator_spec_receipt_sha256,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "sufficient": self.sufficient,
            "usable_item_receipt_sha256s": list(self.usable_item_receipt_sha256s),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ConsensusGroupReceipt:
    semantic_key_sha256: str
    item_receipt_sha256s: tuple[str, ...]
    source_group_handles: tuple[str, ...]
    support_count: int
    cross_group_support_count: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.semantic_key_sha256, "consensus semantic key")
        _ordered_unique(self.item_receipt_sha256s, "consensus items")
        _ordered_unique(self.source_group_handles, "consensus source groups")
        if (
            type(self.support_count) is not int
            or type(self.cross_group_support_count) is not int
            or self.support_count != len(self.item_receipt_sha256s)
            or self.cross_group_support_count != len(self.source_group_handles)
        ):
            raise MatchedEvalContractError("consensus counts changed")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("consensus group receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "cross_group_support_count": self.cross_group_support_count,
            "item_receipt_sha256s": list(self.item_receipt_sha256s),
            "semantic_key_sha256": self.semantic_key_sha256,
            "source_group_handles": list(self.source_group_handles),
            "support_count": self.support_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class EvidenceConsensusReceipt:
    evidence_packet_receipt_sha256: str
    groups: tuple[ConsensusGroupReceipt, ...]
    quarantined_item_receipt_sha256s: tuple[str, ...]
    fail_open_used: bool
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.evidence_packet_receipt_sha256, "consensus packet")
        if type(self.groups) is not tuple or any(type(row) is not ConsensusGroupReceipt for row in self.groups):
            raise MatchedEvalContractError("consensus groups changed type")
        _ordered_unique(self.quarantined_item_receipt_sha256s, "quarantined consensus items")
        if type(self.fail_open_used) is not bool:
            raise MatchedEvalContractError("consensus fail-open flag must be exact")
        if self.provider_prompt_count != 0 or self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("consensus must remain provider-free and zero-state")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("consensus receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_consensus")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "evidence_packet_receipt_sha256": self.evidence_packet_receipt_sha256,
            "fail_open_used": self.fail_open_used,
            "format": CONSENSUS_FORMAT,
            "groups": [row.projection() for row in self.groups],
            "provider_prompt_count": 0,
            "quarantined_item_receipt_sha256s": list(self.quarantined_item_receipt_sha256s),
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class OperatorExecutionReceipt:
    operator_spec_receipt_sha256: str
    evidence_packet_receipt_sha256: str
    closure_receipt_sha256: str
    consensus_receipt_sha256: str
    executor: ExecutorKind
    status: ExecutionStatus
    prediction: str
    numeric_result: float | None
    used_item_receipt_sha256s: tuple[str, ...]
    used_handle_ids: tuple[str, ...]
    missing_slot_ids: tuple[str, ...]
    reason: str
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.operator_spec_receipt_sha256, "execution operator spec"),
            (self.evidence_packet_receipt_sha256, "execution evidence packet"),
            (self.closure_receipt_sha256, "execution closure"),
            (self.consensus_receipt_sha256, "execution consensus"),
        ):
            require_sha256(value, label)
        if type(self.executor) is not ExecutorKind or type(self.status) is not ExecutionStatus:
            raise MatchedEvalContractError("execution enums must be canonical")
        if type(self.prediction) is not str:
            raise MatchedEvalContractError("execution prediction must be exact text")
        if self.status is ExecutionStatus.SUPPORTED and not self.prediction:
            raise MatchedEvalContractError("supported execution requires a prediction")
        if self.status is not ExecutionStatus.SUPPORTED and self.prediction:
            raise MatchedEvalContractError("unsupported execution cannot emit a prediction")
        if self.numeric_result is not None and type(self.numeric_result) not in {int, float}:
            raise MatchedEvalContractError("execution numeric result is invalid")
        _ordered_unique(self.used_item_receipt_sha256s, "execution items")
        _ordered_unique(self.used_handle_ids, "execution handles")
        _ordered_unique(self.missing_slot_ids, "execution missing slots")
        require_text(self.reason, "execution reason")
        if self.provider_prompt_count != 0 or self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("execution must remain provider-free and zero-state")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("execution receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_execution")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "closure_receipt_sha256": self.closure_receipt_sha256,
            "consensus_receipt_sha256": self.consensus_receipt_sha256,
            "evidence_packet_receipt_sha256": self.evidence_packet_receipt_sha256,
            "executor": self.executor.value,
            "format": EXECUTION_FORMAT,
            "missing_slot_ids": list(self.missing_slot_ids),
            "numeric_result": self.numeric_result,
            "operator_spec_receipt_sha256": self.operator_spec_receipt_sha256,
            "prediction": self.prediction,
            "provider_prompt_count": 0,
            "reason": self.reason,
            "retained_transformer_token_state_bytes": 0,
            "status": self.status.value,
            "used_handle_ids": list(self.used_handle_ids),
            "used_item_receipt_sha256s": list(self.used_item_receipt_sha256s),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class CandidatePreservationReceipt:
    operator_spec_receipt_sha256: str
    evidence_packet_receipt_sha256: str
    candidate_prediction_sha256: str
    required_specificity_terms: tuple[str, ...]
    missing_specificity_terms: tuple[str, ...]
    required_personalization_anchors: tuple[str, ...]
    missing_personalization_anchors: tuple[str, ...]
    preserves_required_content: bool
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.operator_spec_receipt_sha256, "preservation operator spec"),
            (self.evidence_packet_receipt_sha256, "preservation packet"),
            (self.candidate_prediction_sha256, "preservation candidate prediction"),
        ):
            require_sha256(value, label)
        required_specific = _ordered_unique(self.required_specificity_terms, "required specificity")
        missing_specific = _ordered_unique(self.missing_specificity_terms, "missing specificity")
        required_personal = _ordered_unique(self.required_personalization_anchors, "required personalization")
        missing_personal = _ordered_unique(self.missing_personalization_anchors, "missing personalization")
        if not set(missing_specific) <= set(required_specific) or not set(missing_personal) <= set(required_personal):
            raise MatchedEvalContractError("preservation missing anchors escaped requirements")
        if type(self.preserves_required_content) is not bool:
            raise MatchedEvalContractError("preservation flag must be exact")
        if self.preserves_required_content != (not missing_specific and not missing_personal):
            raise MatchedEvalContractError("preservation decision changed")
        if self.provider_prompt_count != 0 or self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("preservation must remain provider-free and zero-state")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("preservation receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="candidate_preservation")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidate_prediction_sha256": self.candidate_prediction_sha256,
            "evidence_packet_receipt_sha256": self.evidence_packet_receipt_sha256,
            "format": PRESERVATION_FORMAT,
            "missing_personalization_anchors": list(self.missing_personalization_anchors),
            "missing_specificity_terms": list(self.missing_specificity_terms),
            "operator_spec_receipt_sha256": self.operator_spec_receipt_sha256,
            "preserves_required_content": self.preserves_required_content,
            "provider_prompt_count": 0,
            "required_personalization_anchors": list(self.required_personalization_anchors),
            "required_specificity_terms": list(self.required_specificity_terms),
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _usable(packet: TypedEvidencePacket) -> tuple[TypedEvidenceItem, ...]:
    return tuple(
        row for row in packet.items
        if row.included
        and (
            row.content_coherence is not ContentCoherence.CONFLICT
            or packet.conflict_policy is ConflictPolicy.FAIL_OPEN
        )
        and row.status is not EvidenceStatus.CANCELLED
        and (
            row.status is not EvidenceStatus.PROPOSED
            or packet.operator_spec.include_proposed
        )
    )


def build_slot_closure(spec: TypedOperatorSpec, packet: TypedEvidencePacket) -> SlotClosureReceipt:
    if type(spec) is not TypedOperatorSpec or type(packet) is not TypedEvidencePacket:
        raise TypeError("slot closure requires exact typed inputs")
    if packet.operator_spec.receipt_sha256 != spec.receipt_sha256:
        raise MatchedEvalContractError("slot closure spec/packet binding changed")
    usable = _usable(packet)
    bindings: list[SlotBindingReceipt] = []
    bound: list[str] = []
    missing: list[str] = []
    conflicted: list[str] = []
    conflict_items = tuple(row for row in packet.items if row.content_conflict)
    for slot in spec.required_slots:
        supporting = tuple(row for row in usable if slot.slot_id in row.supported_slot_ids)
        conflict_for_slot = tuple(
            row for row in conflict_items
            if set(slot.match_terms) & set(normalized_terms(row.summary))
        )
        if supporting and conflict_for_slot:
            conflicted.append(slot.slot_id)
        elif supporting:
            bound.append(slot.slot_id)
            item_receipts = tuple(row.receipt_sha256 for row in supporting)
            handles = tuple(dict.fromkeys(handle for row in supporting for handle in row.handle_ids))
            bindings.append(SlotBindingReceipt(slot.slot_id, item_receipts, handles))
        elif conflict_for_slot:
            conflicted.append(slot.slot_id)
        else:
            missing.append(slot.slot_id)
    frontier_ok = not spec.requires_complete_frontier or packet.frontier.closed
    sufficient = bool(
        usable
        and not missing
        and not conflicted
        and frontier_ok
    )
    return SlotClosureReceipt(
        spec.receipt_sha256, packet.receipt_sha256,
        packet.frontier.receipt_sha256, tuple(bindings), tuple(bound),
        tuple(missing), tuple(conflicted),
        tuple(row.receipt_sha256 for row in usable), sufficient,
        spec.requires_complete_frontier,
    )


def _semantic_projection(item: TypedEvidenceItem) -> dict[str, Any]:
    # Summary text is used only as a last-resort identity when no structured
    # entity/group/date/value exists.  This prevents repeated excerpts from
    # being counted as independent events simply because they have new handles.
    structured = any(
        value is not None
        for value in (
            item.entity_key, item.group_key, item.numeric_value, item.date,
            item.relation, item.participant_count,
        )
    )
    return {
        "date": item.date,
        "entity_key": item.entity_key,
        "fallback_terms": None if structured else list(normalized_terms(item.summary)),
        "group_key": item.group_key,
        "kind": item.kind.value,
        "numeric_role": item.numeric_role.value,
        "numeric_qualifier": item.numeric_qualifier.value,
        "numeric_value": item.numeric_value,
        "participant_count": item.participant_count,
        "relation": item.relation,
        "status": item.status.value,
        "unit": item.unit,
    }


def build_evidence_consensus(packet: TypedEvidencePacket) -> EvidenceConsensusReceipt:
    if type(packet) is not TypedEvidencePacket:
        raise TypeError("consensus requires an exact packet")
    binding_by_handle = {row.handle_id: row for row in packet.handles}
    usable = _usable(packet)
    grouped: dict[str, list[TypedEvidenceItem]] = {}
    for item in usable:
        grouped.setdefault(identity_sha256(_semantic_projection(item)), []).append(item)
    groups: list[ConsensusGroupReceipt] = []
    for semantic_key, items in grouped.items():
        source_groups = tuple(
            dict.fromkeys(
                binding_by_handle[handle].source_group_handle
                for item in items for handle in item.handle_ids
            )
        )
        groups.append(
            ConsensusGroupReceipt(
                semantic_key,
                tuple(row.receipt_sha256 for row in items),
                source_groups,
                len(items),
                len(source_groups),
            )
        )
    quarantined = tuple(
        row.receipt_sha256 for row in packet.items
        if row.content_conflict and packet.conflict_policy is ConflictPolicy.QUARANTINE
    )
    return EvidenceConsensusReceipt(
        packet.receipt_sha256, tuple(groups), quarantined,
        packet.conflict_policy is ConflictPolicy.FAIL_OPEN,
    )


def _format_number(value: float, unit: str | None = None) -> str:
    scalar = str(int(value)) if float(value).is_integer() else f"{value:.10f}".rstrip("0").rstrip(".")
    if unit == "$" or unit == "USD":
        return "$" + scalar
    return scalar if unit is None else f"{scalar} {unit}"


def _execution(
    spec: TypedOperatorSpec,
    packet: TypedEvidencePacket,
    closure: SlotClosureReceipt,
    consensus: EvidenceConsensusReceipt,
    *,
    executor: ExecutorKind,
    status: ExecutionStatus,
    prediction: str = "",
    numeric_result: float | None = None,
    used: tuple[TypedEvidenceItem, ...] = (),
    reason: str,
) -> OperatorExecutionReceipt:
    handles = tuple(dict.fromkeys(handle for row in used for handle in row.handle_ids))
    return OperatorExecutionReceipt(
        spec.receipt_sha256, packet.receipt_sha256, closure.receipt_sha256,
        consensus.receipt_sha256, executor, status, prediction,
        numeric_result, tuple(row.receipt_sha256 for row in used), handles,
        closure.missing_slot_ids, reason,
    )


def _preflight(spec: TypedOperatorSpec, packet: TypedEvidencePacket, executor: ExecutorKind) -> tuple[SlotClosureReceipt, EvidenceConsensusReceipt, OperatorExecutionReceipt | None]:
    closure = build_slot_closure(spec, packet)
    consensus = build_evidence_consensus(packet)
    if closure.conflicted_slot_ids:
        return closure, consensus, _execution(
            spec, packet, closure, consensus, executor=executor,
            status=ExecutionStatus.CONFLICTED,
            reason="required_slot_has_content_conflict",
        )
    if not closure.sufficient:
        return closure, consensus, _execution(
            spec, packet, closure, consensus, executor=executor,
            status=ExecutionStatus.INSUFFICIENT,
            reason=(
                "frontier_not_closed"
                if spec.requires_complete_frontier and not packet.frontier.closed
                else "required_slot_or_usable_evidence_missing"
            ),
        )
    return closure, consensus, None


def execute_numeric(spec: TypedOperatorSpec, packet: TypedEvidencePacket) -> OperatorExecutionReceipt:
    closure, consensus, blocked = _preflight(spec, packet, ExecutorKind.NUMERIC)
    if blocked is not None:
        return blocked
    numeric_slot_ids = {
        slot.slot_id for slot in spec.required_slots if slot.requires_numeric
    }
    values = tuple(
        row
        for row in _usable(packet)
        if row.numeric_value is not None
        and (
            not numeric_slot_ids
            or numeric_slot_ids & set(row.supported_slot_ids)
        )
    )
    if not values:
        return _execution(spec, packet, closure, consensus, executor=ExecutorKind.NUMERIC, status=ExecutionStatus.INSUFFICIENT, reason="no_numeric_operands")
    if any(
        row.numeric_qualifier is not NumericQualifier.EXACT for row in values
    ):
        return _execution(
            spec,
            packet,
            closure,
            consensus,
            executor=ExecutorKind.NUMERIC,
            status=ExecutionStatus.INSUFFICIENT,
            reason="qualified_numeric_operands_require_model",
        )
    if spec.comparison_mode is ComparisonMode.MAX_ENTITY:
        grouped: dict[str, list[TypedEvidenceItem]] = {}
        for row in values:
            key = row.group_key or row.entity_key
            if key:
                grouped.setdefault(key, []).append(row)
        scores: list[tuple[float, str, tuple[TypedEvidenceItem, ...]]] = []
        for key, rows in grouped.items():
            deltas = [float(row.numeric_value) for row in rows if row.numeric_role is NumericRole.DELTA]
            baseline = [float(row.numeric_value) for row in rows if row.numeric_role is NumericRole.BASELINE]
            end = [float(row.numeric_value) for row in rows if row.numeric_role is NumericRole.END]
            if deltas:
                score = sum(deltas)
            elif baseline and end:
                score = end[-1] - baseline[0]
            else:
                continue
            scores.append((score, key, tuple(rows)))
        if not scores:
            return _execution(spec, packet, closure, consensus, executor=ExecutorKind.NUMERIC, status=ExecutionStatus.INSUFFICIENT, reason="comparison_group_missing_complete_values")
        scores.sort(key=lambda row: (-row[0], row[1].casefold()))
        top = tuple(row for row in scores if row[0] == scores[0][0])
        if len(top) != 1:
            return _execution(spec, packet, closure, consensus, executor=ExecutorKind.NUMERIC, status=ExecutionStatus.CONFLICTED, reason="comparison_tie")
        score, key, used = top[0]
        return _execution(spec, packet, closure, consensus, executor=ExecutorKind.NUMERIC, status=ExecutionStatus.SUPPORTED, prediction=key, numeric_result=score, used=used, reason="deterministic_max_entity")

    if spec.comparison_mode in {ComparisonMode.DIFFERENCE, ComparisonMode.BOOLEAN_GREATER}:
        side_slots = tuple(
            slot
            for slot in spec.required_slots
            if slot.kind is SlotKind.COMPARISON_SIDE
        )
        if len(side_slots) != 2:
            return _execution(
                spec,
                packet,
                closure,
                consensus,
                executor=ExecutorKind.NUMERIC,
                status=ExecutionStatus.INSUFFICIENT,
                reason="comparison_requires_two_ordered_side_slots",
            )
        side_rows = tuple(
            tuple(row for row in values if slot.slot_id in row.supported_slot_ids)
            for slot in side_slots
        )
        if any(not rows for rows in side_rows):
            return _execution(
                spec,
                packet,
                closure,
                consensus,
                executor=ExecutorKind.NUMERIC,
                status=ExecutionStatus.INSUFFICIENT,
                reason="comparison_side_missing_value",
            )
        if set(side_rows[0]) & set(side_rows[1]):
            return _execution(
                spec,
                packet,
                closure,
                consensus,
                executor=ExecutorKind.NUMERIC,
                status=ExecutionStatus.CONFLICTED,
                reason="comparison_item_binds_multiple_sides",
            )

        sides: list[tuple[str, float, str | None, tuple[TypedEvidenceItem, ...]]] = []
        for slot, rows in zip(side_slots, side_rows, strict=True):
            baseline = {
                (float(row.numeric_value), row.unit)
                for row in rows
                if row.numeric_role is NumericRole.BASELINE
            }
            end = {
                (float(row.numeric_value), row.unit)
                for row in rows
                if row.numeric_role is NumericRole.END
            }
            deltas = {
                (float(row.numeric_value), row.unit)
                for row in rows
                if row.numeric_role is NumericRole.DELTA
            }
            operands = {
                (float(row.numeric_value), row.unit)
                for row in rows
                if row.numeric_role in {NumericRole.OPERAND, NumericRole.NONE}
            }
            scalar: float | None = None
            unit: str | None = None
            if baseline or end:
                if len(baseline) == len(end) == 1 and not deltas and not operands:
                    baseline_value, baseline_unit = next(iter(baseline))
                    end_value, end_unit = next(iter(end))
                    if baseline_unit == end_unit:
                        scalar = end_value - baseline_value
                        unit = end_unit
            elif len(deltas) == 1 and not operands:
                scalar, unit = next(iter(deltas))
            elif len(operands) == 1 and not deltas:
                scalar, unit = next(iter(operands))
            if scalar is None:
                return _execution(
                    spec,
                    packet,
                    closure,
                    consensus,
                    executor=ExecutorKind.NUMERIC,
                    status=ExecutionStatus.CONFLICTED,
                    reason="comparison_side_value_conflict",
                )
            sides.append((slot.label, scalar, unit, rows))

        left, right = sides
        if left[2] != right[2]:
            return _execution(
                spec,
                packet,
                closure,
                consensus,
                executor=ExecutorKind.NUMERIC,
                status=ExecutionStatus.CONFLICTED,
                reason="comparison_side_unit_conflict",
            )
        delta = left[1] - right[1]
        if spec.comparison_mode is ComparisonMode.BOOLEAN_GREATER:
            prediction = "Yes" if delta > 0 else "No"
        else:
            prediction = _format_number(abs(delta), left[2])
        return _execution(spec, packet, closure, consensus, executor=ExecutorKind.NUMERIC, status=ExecutionStatus.SUPPORTED, prediction=prediction, numeric_result=delta, used=tuple((*left[3], *right[3])), reason="deterministic_ordered_side_comparison")

    # Count/aggregate operands are deduplicated by semantic projection before
    # reduction, so repeated map/source renderings do not inflate the total.
    unique: dict[str, TypedEvidenceItem] = {}
    for row in values:
        unique.setdefault(identity_sha256(_semantic_projection(row)), row)
    operands = tuple(unique.values())
    total = sum(float(row.numeric_value) for row in operands)
    return _execution(spec, packet, closure, consensus, executor=ExecutorKind.NUMERIC, status=ExecutionStatus.SUPPORTED, prediction=_format_number(total, operands[0].unit), numeric_result=total, used=operands, reason="deterministic_numeric_reduction")


def _parse_datetime(value: str) -> datetime:
    candidates = (
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
        "%B %Y", "%b %Y",
    )
    for format_string in candidates:
        try:
            return datetime.strptime(value, format_string)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise MatchedEvalContractError("typed evidence date is not executable") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def execute_time(spec: TypedOperatorSpec, packet: TypedEvidencePacket) -> OperatorExecutionReceipt:
    closure, consensus, blocked = _preflight(spec, packet, ExecutorKind.TIME)
    if blocked is not None:
        return blocked
    dated: list[tuple[datetime, TypedEvidenceItem]] = []
    for row in _usable(packet):
        if row.date is not None:
            try:
                dated.append((_parse_datetime(row.date), row))
            except MatchedEvalContractError:
                continue
    dated.sort(key=lambda pair: (pair[0], pair[1].receipt_sha256))
    if not dated:
        return _execution(spec, packet, closure, consensus, executor=ExecutorKind.TIME, status=ExecutionStatus.INSUFFICIENT, reason="no_executable_dates")
    if spec.temporal_mode is TemporalMode.INTERVAL:
        if len(dated) >= 2:
            start, end = dated[0][0], dated[-1][0]
            used = (dated[0][1], dated[-1][1])
        elif spec.query_timestamp is not None:
            asked_date = spec.query_timestamp.replace("/", "-").split(" (")[0]
            start, end = dated[0][0], _parse_datetime(asked_date)
            used = (dated[0][1],)
        else:
            return _execution(spec, packet, closure, consensus, executor=ExecutorKind.TIME, status=ExecutionStatus.INSUFFICIENT, reason="interval_requires_two_boundaries")
        if end < start:
            return _execution(spec, packet, closure, consensus, executor=ExecutorKind.TIME, status=ExecutionStatus.CONFLICTED, reason="interval_boundary_order_conflict")
        days = (end - start).days
        months = (end.year - start.year) * 12 + end.month - start.month
        if end.day < start.day:
            months -= 1
        prediction = f"{months} months" if months >= 1 else f"{days} days"
        numeric = float(months if months >= 1 else days)
        return _execution(spec, packet, closure, consensus, executor=ExecutorKind.TIME, status=ExecutionStatus.SUPPORTED, prediction=prediction, numeric_result=numeric, used=used, reason="deterministic_interval")
    if spec.temporal_mode is TemporalMode.RELATIVE_SELECT:
        # RELATIVE_SELECT must never fall through to the generic chronological
        # timeline below.  A question such as ``10 days ago`` is an exact-day
        # lookup, while the v1 spec can currently express only a bounded
        # lookback window.  Treat an unexpressed target as insufficient instead
        # of labelling a concatenation of every dated row as a supported
        # deterministic answer.  A semantic candidate arbiter may still answer
        # from an independently validated exact-day citation.
        if spec.query_timestamp is None or spec.temporal_window_days is None:
            return _execution(
                spec,
                packet,
                closure,
                consensus,
                executor=ExecutorKind.TIME,
                status=ExecutionStatus.INSUFFICIENT,
                reason="relative_selection_target_unresolved",
            )
        asked = _parse_datetime(
            spec.query_timestamp.replace("/", "-").split(" (")[0]
        )
        selected = tuple(
            row
            for moment, row in dated
            if 0 <= (asked - moment).days <= spec.temporal_window_days
        )
        if not selected:
            return _execution(
                spec,
                packet,
                closure,
                consensus,
                executor=ExecutorKind.TIME,
                status=ExecutionStatus.INSUFFICIENT,
                reason="relative_window_empty",
            )
        prediction = ", ".join(row.entity_key or row.summary for row in selected)
        return _execution(
            spec,
            packet,
            closure,
            consensus,
            executor=ExecutorKind.TIME,
            status=ExecutionStatus.SUPPORTED,
            prediction=prediction,
            used=selected,
            reason="deterministic_relative_selection",
        )
    ordered = tuple(row for _moment, row in dated)
    prediction = " → ".join(row.entity_key or row.summary for row in ordered)
    return _execution(spec, packet, closure, consensus, executor=ExecutorKind.TIME, status=ExecutionStatus.SUPPORTED, prediction=prediction, used=ordered, reason="deterministic_timeline_order")


def execute_set(spec: TypedOperatorSpec, packet: TypedEvidencePacket) -> OperatorExecutionReceipt:
    closure, consensus, blocked = _preflight(spec, packet, ExecutorKind.SET)
    if blocked is not None:
        return blocked
    unique: dict[str, TypedEvidenceItem] = {}
    for row in _usable(packet):
        if row.kind not in {TypedItemKind.MEMBER, TypedItemKind.EVENT, TypedItemKind.DIRECT}:
            continue
        label = row.entity_key or row.summary
        unique.setdefault(" ".join(normalized_terms(label)), row)
    selected = tuple(unique.values())
    if not selected:
        return _execution(spec, packet, closure, consensus, executor=ExecutorKind.SET, status=ExecutionStatus.INSUFFICIENT, reason="no_set_members")
    if spec.cardinality is not None and len(selected) != spec.cardinality:
        return _execution(spec, packet, closure, consensus, executor=ExecutorKind.SET, status=ExecutionStatus.INSUFFICIENT, reason="set_cardinality_not_closed")
    if spec.ordering == "chronological":
        if any(row.date is None for row in selected):
            return _execution(spec, packet, closure, consensus, executor=ExecutorKind.SET, status=ExecutionStatus.INSUFFICIENT, reason="ordered_set_missing_dates")
        selected = tuple(sorted(selected, key=lambda row: (_parse_datetime(row.date or ""), row.receipt_sha256)))
    prediction = ", ".join(row.entity_key or row.summary for row in selected)
    return _execution(spec, packet, closure, consensus, executor=ExecutorKind.SET, status=ExecutionStatus.SUPPORTED, prediction=prediction, used=selected, reason="deterministic_set_join")


def execute_state(spec: TypedOperatorSpec, packet: TypedEvidencePacket) -> OperatorExecutionReceipt:
    closure, consensus, blocked = _preflight(spec, packet, ExecutorKind.STATE)
    if blocked is not None:
        return blocked
    candidates: list[tuple[datetime, int, TypedEvidenceItem]] = []
    for row in _usable(packet):
        if row.date is None:
            continue
        try:
            moment = _parse_datetime(row.date)
        except MatchedEvalContractError:
            continue
        authority = 1 if row.value_authority is ValueAuthority.EXPLICIT else 0
        candidates.append((moment, authority, row))
    if not candidates:
        return _execution(spec, packet, closure, consensus, executor=ExecutorKind.STATE, status=ExecutionStatus.INSUFFICIENT, reason="state_chain_has_no_dated_state")
    candidates.sort(key=lambda row: (row[0], row[1], row[2].receipt_sha256))
    latest_time = candidates[-1][0]
    latest = tuple(row for moment, _authority, row in candidates if moment == latest_time)
    explicit = tuple(row for row in latest if row.value_authority is ValueAuthority.EXPLICIT)
    selected = explicit or latest
    values = {(row.numeric_value, row.unit, row.entity_key or row.summary) for row in selected}
    if len(values) != 1:
        return _execution(spec, packet, closure, consensus, executor=ExecutorKind.STATE, status=ExecutionStatus.CONFLICTED, reason="latest_state_conflict")
    row = selected[-1]
    if row.numeric_value is not None:
        prediction = _format_number(float(row.numeric_value), row.unit)
        numeric = float(row.numeric_value)
    else:
        prediction = row.entity_key or row.summary
        numeric = None
    return _execution(spec, packet, closure, consensus, executor=ExecutorKind.STATE, status=ExecutionStatus.SUPPORTED, prediction=prediction, numeric_result=numeric, used=(row,), reason="deterministic_latest_explicit_state")


def execute_typed_operator(spec: TypedOperatorSpec, packet: TypedEvidencePacket) -> OperatorExecutionReceipt:
    """Dispatch only operations with a deterministic evidence-table meaning."""

    if spec.temporal_mode is TemporalMode.LATEST_STATE:
        return execute_state(spec, packet)
    if spec.temporal_mode is not TemporalMode.NONE:
        return execute_time(spec, packet)
    if spec.answer_shape is AnswerShape.SET_LIST:
        return execute_set(spec, packet)
    if spec.answer_shape in {AnswerShape.NUMBER, AnswerShape.BOOLEAN} or spec.comparison_mode is not ComparisonMode.NONE:
        return execute_numeric(spec, packet)
    closure = build_slot_closure(spec, packet)
    consensus = build_evidence_consensus(packet)
    return _execution(
        spec, packet, closure, consensus, executor=ExecutorKind.NONE,
        status=ExecutionStatus.NON_DETERMINISTIC,
        reason="direct_or_synthesis_requires_candidate_arbiter",
    )


def assess_candidate_preservation(
    spec: TypedOperatorSpec,
    packet: TypedEvidencePacket,
    candidate_prediction: str,
) -> CandidatePreservationReceipt:
    """Block generic rewrites that erase selected evidence detail.

    Requirements come only from question flags and the selected typed items;
    callers cannot inject a post-hoc target list.
    """

    if type(spec) is not TypedOperatorSpec or type(packet) is not TypedEvidencePacket:
        raise TypeError("candidate preservation requires exact typed inputs")
    if packet.operator_spec.receipt_sha256 != spec.receipt_sha256:
        raise MatchedEvalContractError("candidate preservation spec/packet changed")
    if type(candidate_prediction) is not str or not candidate_prediction or candidate_prediction.strip() != candidate_prediction:
        raise ValueError("candidate prediction must be non-empty exact text")
    usable = _usable(packet)
    required_specific = tuple(
        dict.fromkeys(
            term for row in usable for term in row.specificity_terms
        )
    ) if spec.specificity_required else ()
    required_personal = tuple(
        dict.fromkeys(
            term for row in usable for term in row.personalization_anchors
        )
    ) if spec.personalization_required else ()
    present = set(normalized_terms(candidate_prediction))
    missing_specific = tuple(row for row in required_specific if row not in present)
    missing_personal = tuple(row for row in required_personal if row not in present)
    return CandidatePreservationReceipt(
        spec.receipt_sha256, packet.receipt_sha256,
        identity_sha256({"prediction": candidate_prediction}),
        required_specific, missing_specific, required_personal, missing_personal,
        not missing_specific and not missing_personal,
    )


__all__ = [
    "CandidatePreservationReceipt", "ConsensusGroupReceipt",
    "EvidenceConsensusReceipt", "ExecutionStatus", "ExecutorKind",
    "OperatorExecutionReceipt", "SlotBindingReceipt", "SlotClosureReceipt",
    "assess_candidate_preservation", "build_evidence_consensus",
    "build_slot_closure", "execute_numeric", "execute_set", "execute_state",
    "execute_time", "execute_typed_operator",
]
