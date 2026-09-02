"""Provider-free adapter from sealed retrieval outputs to typed evidence items.

The adapter is intentionally split into a provider-visible, opaque projection
and a local provenance ledger.  Raw source IDs, namespaces, citations, and
artifact paths never enter the former.  Evidence from unrelated histories is
not classified by source-ID prefixes: opaque source-group handles expose only
candidate-to-candidate co-membership, while semantic fit is derived from text.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Sequence

from memory_condense.domain._tokenizer import count_tokens

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .query_evidence_map_solver_v2_live import (
    MAP_ITEM_FORMAT,
    EvidenceMapPlanRow,
    PayloadEvidenceAlias,
    ValidatedMapItem,
    VerifiedEvidenceMapRow,
)
from .source_history_fact_union import FactUnionEnvelope, LaneAdmission
from .typed_operator_spec import (
    RequiredSlot,
    TypedOperatorSpec,
    normalized_terms,
)
from .typed_numeric_semantics import (
    NumericQualifier,
    numeric_mentions,
    single_numeric_mention,
)


FORMAT = "memory-condense-typed-evidence-packet-v2"
COMPACT_FINAL_PROVIDER_FORMAT = (
    "memory-condense-typed-memory-final-arm-v1-compact-provider-evidence-v1"
)
COMPACT_FINAL_PROVIDER_FORMAT_V2 = (
    "memory-condense-typed-memory-final-arm-v1-compact-provider-evidence-v2"
)
HANDLE_FORMAT = "memory-condense-opaque-evidence-handle-v1"
BINDING_FORMAT = "memory-condense-local-evidence-binding-v1"
ITEM_FORMAT = "memory-condense-typed-evidence-item-v2"
REJECTION_FORMAT = "memory-condense-typed-item-rejection-v1"
FRONTIER_FORMAT = "memory-condense-evidence-frontier-v1"
CONTRIBUTION_FORMAT = "memory-condense-typed-evidence-contribution-v2"
HARD_PROMPT_TOKEN_CAP = 8_000
DEFAULT_OUTPUT_TOKEN_RESERVE = 768


class EvidenceOrigin(str, Enum):
    MAP = "map"
    SOURCE_FACT = "source_fact"
    DIRECT_POINTER = "direct_pointer"
    MEM0 = "mem0"


class ProvenanceGrade(str, Enum):
    EXACT_CITATION = "exact_citation"
    EXACT_FACT_UNION = "exact_fact_union"
    DIRECT_POINTER = "direct_pointer"
    INFERRED_MEMORY = "inferred_memory"
    REQUEST_WINDOW_ONLY = "request_window_only"


class TypedItemKind(str, Enum):
    DIRECT = "direct"
    OPERAND = "operand"
    EVENT = "event"
    MEMBER = "member"
    CLAIM = "claim"
    STATE = "state"


class NumericRole(str, Enum):
    NONE = "none"
    OPERAND = "operand"
    BASELINE = "baseline"
    END = "end"
    DELTA = "delta"


class EvidenceStatus(str, Enum):
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    CURRENT = "current"
    PROPOSED = "proposed"
    CANCELLED = "cancelled"


class ValueAuthority(str, Enum):
    EXPLICIT = "explicit"
    DERIVED = "derived"


class ContentCoherence(str, Enum):
    MATCH = "match"
    UNRESOLVED = "unresolved"
    CONFLICT = "conflict"


class ConflictPolicy(str, Enum):
    QUARANTINE = "quarantine"
    FAIL_OPEN = "fail_open"


class FrontierMode(str, Enum):
    EXHAUSTIVE = "exhaustive"
    BOUNDED = "bounded"
    OPEN = "open"


class ProviderPayloadMode(str, Enum):
    """Provider serialization that owns the packet's construction budget."""

    CANONICAL = "canonical"
    COMPACT_FINAL = "compact_final"
    COMPACT_FINAL_V2 = "compact_final_v2"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(row) is not str or not row for row in values):
        raise MatchedEvalContractError(f"{label} must be an exact text tuple")
    if len(set(values)) != len(values):
        raise MatchedEvalContractError(f"{label} must be ordered and unique")
    return values


_HANDLE_RE = re.compile(r"^H[0-9]{3,6}$")
_GROUP_RE = re.compile(r"^G[0-9]{3,6}$")


@dataclass(frozen=True, slots=True)
class OpaqueEvidenceHandle:
    handle_id: str
    origin: EvidenceOrigin
    provenance_grade: ProvenanceGrade
    source_group_handle: str
    binding_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.handle_id) is not str or _HANDLE_RE.fullmatch(self.handle_id) is None:
            raise MatchedEvalContractError("evidence handle must be opaque")
        if type(self.origin) is not EvidenceOrigin:
            raise MatchedEvalContractError("evidence origin must be canonical")
        if type(self.provenance_grade) is not ProvenanceGrade:
            raise MatchedEvalContractError("evidence provenance grade must be canonical")
        if (
            type(self.source_group_handle) is not str
            or _GROUP_RE.fullmatch(self.source_group_handle) is None
        ):
            raise MatchedEvalContractError("source group handle must be opaque")
        require_sha256(self.binding_receipt_sha256, "handle binding receipt")

    def projection(self) -> dict[str, str]:
        return {
            "binding_receipt_sha256": self.binding_receipt_sha256,
            "format": HANDLE_FORMAT,
            "handle_id": self.handle_id,
            "origin": self.origin.value,
            "provenance_grade": self.provenance_grade.value,
            "source_group_handle": self.source_group_handle,
        }


@dataclass(frozen=True, slots=True)
class EvidenceHandleBinding:
    """Prompt-external provenance for one opaque evidence handle."""

    handle_id: str
    origin: EvidenceOrigin
    provenance_grade: ProvenanceGrade
    source_group_handle: str
    sealed_artifact_sha256: str
    parent_receipt_sha256: str
    evidence_receipt_sha256: str
    payload_sha256: str
    citation_sha256: str
    citation_char_count: int
    local_source_locator_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if _HANDLE_RE.fullmatch(self.handle_id) is None:
            raise MatchedEvalContractError("local evidence handle must be opaque")
        if type(self.origin) is not EvidenceOrigin:
            raise MatchedEvalContractError("local evidence origin must be canonical")
        if type(self.provenance_grade) is not ProvenanceGrade:
            raise MatchedEvalContractError("local provenance grade must be canonical")
        allowed_grades = {
            EvidenceOrigin.MAP: {ProvenanceGrade.EXACT_CITATION},
            EvidenceOrigin.SOURCE_FACT: {ProvenanceGrade.EXACT_FACT_UNION},
            EvidenceOrigin.DIRECT_POINTER: {ProvenanceGrade.DIRECT_POINTER},
            EvidenceOrigin.MEM0: {
                ProvenanceGrade.INFERRED_MEMORY,
                ProvenanceGrade.REQUEST_WINDOW_ONLY,
            },
        }
        if self.provenance_grade not in allowed_grades[self.origin]:
            raise MatchedEvalContractError("origin/provenance grade overstates its source")
        if _GROUP_RE.fullmatch(self.source_group_handle) is None:
            raise MatchedEvalContractError("local source group must be opaque")
        for value, label in (
            (self.sealed_artifact_sha256, "binding artifact"),
            (self.parent_receipt_sha256, "binding parent"),
            (self.evidence_receipt_sha256, "binding evidence"),
            (self.payload_sha256, "binding payload"),
            (self.citation_sha256, "binding citation"),
            (self.local_source_locator_sha256, "binding local source locator"),
        ):
            require_sha256(value, label)
        if type(self.citation_char_count) is not int or self.citation_char_count < 0:
            raise MatchedEvalContractError("citation character count is invalid")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("local evidence binding receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "citation_char_count": self.citation_char_count,
            "citation_sha256": self.citation_sha256,
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "format": BINDING_FORMAT,
            "handle_id": self.handle_id,
            "local_source_locator_sha256": self.local_source_locator_sha256,
            "origin": self.origin.value,
            "provenance_grade": self.provenance_grade.value,
            "parent_receipt_sha256": self.parent_receipt_sha256,
            "payload_sha256": self.payload_sha256,
            "sealed_artifact_sha256": self.sealed_artifact_sha256,
            "source_group_handle": self.source_group_handle,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def opaque(self) -> OpaqueEvidenceHandle:
        return OpaqueEvidenceHandle(
            self.handle_id,
            self.origin,
            self.provenance_grade,
            self.source_group_handle,
            self.receipt_sha256,
        )


@dataclass(frozen=True, slots=True)
class TypedEvidenceItem:
    item_id: str
    handle_ids: tuple[str, ...]
    kind: TypedItemKind
    summary: str
    entity_key: str | None
    group_key: str | None
    numeric_value: float | None
    numeric_role: NumericRole
    numeric_qualifier: NumericQualifier
    unit: str | None
    date: str | None
    status: EvidenceStatus
    relation: str | None
    participant_count: int | None
    value_authority: ValueAuthority
    included: bool
    supported_slot_ids: tuple[str, ...]
    content_coherence: ContentCoherence
    content_conflict: bool
    conflict_receipt_sha256: str | None
    specificity_terms: tuple[str, ...]
    personalization_anchors: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.item_id, "typed item ID")
        _ordered_unique(self.handle_ids, "typed item handles")
        if not self.handle_ids or any(_HANDLE_RE.fullmatch(row) is None for row in self.handle_ids):
            raise MatchedEvalContractError("typed item requires opaque handles")
        if type(self.kind) is not TypedItemKind:
            raise MatchedEvalContractError("typed item kind must be canonical")
        require_text(self.summary, "typed item summary")
        for value, label in (
            (self.entity_key, "typed entity key"),
            (self.group_key, "typed group key"),
            (self.unit, "typed unit"),
            (self.date, "typed date"),
            (self.relation, "typed relation"),
        ):
            if value is not None:
                require_text(value, label)
        if self.numeric_value is not None and type(self.numeric_value) not in {int, float}:
            raise MatchedEvalContractError("typed numeric value is invalid")
        if type(self.numeric_role) is not NumericRole:
            raise MatchedEvalContractError("typed numeric role must be canonical")
        if type(self.numeric_qualifier) is not NumericQualifier:
            raise MatchedEvalContractError("typed numeric qualifier must be canonical")
        if (
            self.numeric_value is None
            and self.numeric_qualifier is not NumericQualifier.EXACT
        ):
            raise MatchedEvalContractError(
                "non-numeric item cannot carry a numeric qualifier"
            )
        if type(self.status) is not EvidenceStatus:
            raise MatchedEvalContractError("typed status must be canonical")
        if self.participant_count is not None and (
            type(self.participant_count) is not int or self.participant_count < 0
        ):
            raise MatchedEvalContractError("participant count is invalid")
        if type(self.value_authority) is not ValueAuthority:
            raise MatchedEvalContractError("value authority must be canonical")
        if type(self.included) is not bool:
            raise MatchedEvalContractError("typed inclusion flag must be exact")
        _ordered_unique(self.supported_slot_ids, "supported slot IDs")
        if type(self.content_coherence) is not ContentCoherence:
            raise MatchedEvalContractError("content coherence must be canonical")
        if type(self.content_conflict) is not bool:
            raise MatchedEvalContractError("content conflict flag must be exact")
        if self.content_conflict != (self.content_coherence is ContentCoherence.CONFLICT):
            raise MatchedEvalContractError("content conflict flag disagrees with coherence")
        if self.content_conflict:
            require_sha256(self.conflict_receipt_sha256 or "", "content conflict receipt")
        elif self.conflict_receipt_sha256 is not None:
            raise MatchedEvalContractError("non-conflicting item cannot carry conflict receipt")
        _ordered_unique(self.specificity_terms, "specificity terms")
        _ordered_unique(self.personalization_anchors, "personalization anchors")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("typed item receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_evidence_item")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "content_coherence": self.content_coherence.value,
            "content_conflict": self.content_conflict,
            "conflict_receipt_sha256": self.conflict_receipt_sha256,
            "date": self.date,
            "entity_key": self.entity_key,
            "format": ITEM_FORMAT,
            "group_key": self.group_key,
            "handle_ids": list(self.handle_ids),
            "included": self.included,
            "item_id": self.item_id,
            "kind": self.kind.value,
            "numeric_role": self.numeric_role.value,
            "numeric_qualifier": self.numeric_qualifier.value,
            "numeric_value": self.numeric_value,
            "participant_count": self.participant_count,
            "personalization_anchors": list(self.personalization_anchors),
            "relation": self.relation,
            "specificity_terms": list(self.specificity_terms),
            "status": self.status.value,
            "summary": self.summary,
            "supported_slot_ids": list(self.supported_slot_ids),
            "unit": self.unit,
            "value_authority": self.value_authority.value,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class RejectedTypedItem:
    source_index: int
    reason: str
    raw_item_sha256: str
    rejection_sha256: str

    def __post_init__(self) -> None:
        if type(self.source_index) is not int:
            raise MatchedEvalContractError("typed rejection index must be exact")
        require_text(self.reason, "typed rejection reason")
        require_sha256(self.raw_item_sha256, "typed rejected item")
        require_sha256(self.rejection_sha256, "typed rejection")

    def projection(self) -> dict[str, Any]:
        return {
            "raw_item_sha256": self.raw_item_sha256,
            "reason": self.reason,
            "rejection_sha256": self.rejection_sha256,
            "source_index": self.source_index,
        }


def _rejected(index: int, reason: str, raw: object) -> RejectedTypedItem:
    try:
        raw_sha = identity_sha256(raw)
    except (TypeError, ValueError):
        raw_sha = _text_sha256(repr(raw))
    body = {
        "format": REJECTION_FORMAT,
        "raw_item_sha256": raw_sha,
        "reason": reason,
        "source_index": index,
    }
    return RejectedTypedItem(index, reason, raw_sha, identity_sha256(body))


@dataclass(frozen=True, slots=True)
class EvidenceFrontierReceipt:
    mode: FrontierMode
    available_handle_ids: tuple[str, ...]
    represented_handle_ids: tuple[str, ...]
    omitted_handle_ids: tuple[str, ...]
    rejected_item_receipt_sha256s: tuple[str, ...]
    unresolved_slot_ids: tuple[str, ...]
    truncated: bool
    closed: bool
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.mode) is not FrontierMode:
            raise MatchedEvalContractError("frontier mode must be canonical")
        available = _ordered_unique(self.available_handle_ids, "available handles")
        represented = _ordered_unique(self.represented_handle_ids, "represented handles")
        omitted = _ordered_unique(self.omitted_handle_ids, "omitted handles")
        _ordered_unique(self.rejected_item_receipt_sha256s, "rejected item receipts")
        _ordered_unique(self.unresolved_slot_ids, "unresolved slots")
        if not set(represented) <= set(available) or set(omitted) != set(available) - set(represented):
            raise MatchedEvalContractError("frontier handle partition changed")
        if type(self.truncated) is not bool or type(self.closed) is not bool:
            raise MatchedEvalContractError("frontier flags must be exact")
        expected_closed = bool(
            self.mode is FrontierMode.EXHAUSTIVE
            and not self.truncated
            and not omitted
            and not self.unresolved_slot_ids
        )
        if self.closed != expected_closed:
            raise MatchedEvalContractError("frontier closure is not justified")
        if self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("frontier retained transformer state")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("frontier receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "available_handle_ids": list(self.available_handle_ids),
            "closed": self.closed,
            "format": FRONTIER_FORMAT,
            "mode": self.mode.value,
            "omitted_handle_ids": list(self.omitted_handle_ids),
            "rejected_item_receipt_sha256s": list(self.rejected_item_receipt_sha256s),
            "represented_handle_ids": list(self.represented_handle_ids),
            "retained_transformer_token_state_bytes": 0,
            "truncated": self.truncated,
            "unresolved_slot_ids": list(self.unresolved_slot_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ParsedTypedItems:
    accepted_items: tuple[TypedEvidenceItem, ...]
    rejected_items: tuple[RejectedTypedItem, ...]
    parse_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class TypedEvidenceContribution:
    """One sealed mechanism contribution before the common typed union.

    Handle IDs must already be unique across contributions.  This avoids
    rewriting either local provenance receipts or typed-item citations during
    composition.
    """

    mechanism_id: str
    bindings: tuple[EvidenceHandleBinding, ...]
    parsed: ParsedTypedItems
    sealed_artifact_sha256: str
    frontier_mode: FrontierMode
    truncated: bool
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.mechanism_id, "typed contribution mechanism")
        if type(self.bindings) is not tuple or any(
            type(row) is not EvidenceHandleBinding for row in self.bindings
        ):
            raise MatchedEvalContractError("typed contribution bindings changed type")
        if type(self.parsed) is not ParsedTypedItems:
            raise MatchedEvalContractError("typed contribution parse changed type")
        require_sha256(self.parsed.parse_receipt_sha256, "typed contribution parse")
        require_sha256(self.sealed_artifact_sha256, "typed contribution artifact")
        if any(
            row.sealed_artifact_sha256 != self.sealed_artifact_sha256
            for row in self.bindings
        ):
            raise MatchedEvalContractError(
                "typed contribution binding escaped its sealed artifact"
            )
        if type(self.frontier_mode) is not FrontierMode or type(self.truncated) is not bool:
            raise MatchedEvalContractError("typed contribution frontier changed")
        handle_ids = tuple(row.handle_id for row in self.bindings)
        if len(set(handle_ids)) != len(handle_ids):
            raise MatchedEvalContractError("typed contribution handles repeat")
        if any(
            not set(row.handle_ids) <= set(handle_ids)
            for row in self.parsed.accepted_items
        ):
            raise MatchedEvalContractError("typed contribution item escaped its bindings")
        if self.provider_prompt_count != 0 or self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("typed contribution must be provider-free and zero-state")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("typed contribution receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_evidence_contribution")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "binding_receipt_sha256s": [row.receipt_sha256 for row in self.bindings],
            "format": CONTRIBUTION_FORMAT,
            "frontier_mode": self.frontier_mode.value,
            "mechanism_id": self.mechanism_id,
            "parse_receipt_sha256": self.parsed.parse_receipt_sha256,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "sealed_artifact_sha256": self.sealed_artifact_sha256,
            "truncated": self.truncated,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def compact_evidence_content_projection(
    items: Sequence[TypedEvidenceItem],
    bindings: Sequence[EvidenceHandleBinding],
) -> dict[str, Any]:
    """Provider-safe selected content with exact summary bytes and opaque IDs."""

    exact_items = tuple(items)
    exact_bindings = tuple(bindings)
    if not (
        all(type(row) is TypedEvidenceItem for row in exact_items)
        and all(type(row) is EvidenceHandleBinding for row in exact_bindings)
    ):
        raise MatchedEvalContractError("compact evidence content changed type")
    represented = {handle for item in exact_items for handle in item.handle_ids}
    retained_bindings = tuple(
        row for row in exact_bindings if row.handle_id in represented
    )
    if {row.handle_id for row in retained_bindings} != represented:
        raise MatchedEvalContractError(
            "compact evidence content escaped its opaque bindings"
        )
    handles = [
        {
            "group_handle": row.source_group_handle,
            "handle_id": row.handle_id,
            "origin": row.origin.value,
            "provenance_grade": row.provenance_grade.value,
        }
        for row in retained_bindings
    ]
    compact_items: list[dict[str, Any]] = []
    for item in exact_items:
        value: dict[str, Any] = {
            "content_coherence": item.content_coherence.value,
            "handle_ids": list(item.handle_ids),
            "included": item.included,
            "kind": item.kind.value,
            "status": item.status.value,
            "summary": item.summary,
            "supported_slot_ids": list(item.supported_slot_ids),
            "value_authority": item.value_authority.value,
        }
        optional = {
            "date": item.date,
            "entity_key": item.entity_key,
            "group_key": item.group_key,
            "numeric_role": (
                None
                if item.numeric_role.value == "none"
                else item.numeric_role.value
            ),
            "numeric_qualifier": (
                item.numeric_qualifier.value
                if item.numeric_value is not None
                else None
            ),
            "numeric_value": item.numeric_value,
            "participant_count": item.participant_count,
            "personalization_anchors": (
                list(item.personalization_anchors)
                if item.personalization_anchors
                else None
            ),
            "relation": item.relation,
            "specificity_terms": (
                list(item.specificity_terms)
                if item.specificity_terms
                else None
            ),
            "unit": item.unit,
        }
        value.update(
            {key: child for key, child in optional.items() if child is not None}
        )
        compact_items.append(value)
    result = {"handles": handles, "items": compact_items}
    assert_gold_blind(result, path="compact_typed_evidence_content")
    return result


def _compact_final_evidence_content_projection_v2(
    items: Sequence[TypedEvidenceItem],
    bindings: Sequence[EvidenceHandleBinding],
    *,
    slot_alias_by_id: dict[str, str],
) -> dict[str, Any]:
    """Version-two provider content without stable local identity fields."""

    exact_items = tuple(items)
    exact_bindings = tuple(bindings)
    if not (
        all(type(row) is TypedEvidenceItem for row in exact_items)
        and all(type(row) is EvidenceHandleBinding for row in exact_bindings)
    ):
        raise MatchedEvalContractError("compact evidence content changed type")
    represented = {handle for item in exact_items for handle in item.handle_ids}
    retained_bindings = tuple(
        row for row in exact_bindings if row.handle_id in represented
    )
    if {row.handle_id for row in retained_bindings} != represented:
        raise MatchedEvalContractError(
            "compact evidence content escaped its opaque bindings"
        )

    grade_by_origin = {
        EvidenceOrigin.DIRECT_POINTER: ProvenanceGrade.DIRECT_POINTER,
        EvidenceOrigin.MAP: ProvenanceGrade.EXACT_CITATION,
        EvidenceOrigin.SOURCE_FACT: ProvenanceGrade.EXACT_FACT_UNION,
    }
    handles: list[dict[str, Any]] = []
    for row in retained_bindings:
        handle: dict[str, Any] = {
            "group_handle": row.source_group_handle,
            "handle_id": row.handle_id,
            "origin": row.origin.value,
        }
        if grade_by_origin.get(row.origin) is not row.provenance_grade:
            handle["provenance_grade"] = row.provenance_grade.value
        handles.append(handle)

    group_keys = tuple(
        dict.fromkeys(
            item.group_key
            for item in exact_items
            if item.group_key is not None
        )
    )
    group_alias_by_id = {
        group_key: f"K{index:03d}"
        for index, group_key in enumerate(group_keys, start=1)
    }
    compact_items: list[dict[str, Any]] = []
    for item in exact_items:
        try:
            supported_slot_ids = [
                slot_alias_by_id[slot_id] for slot_id in item.supported_slot_ids
            ]
        except KeyError as exc:
            raise MatchedEvalContractError(
                "compact evidence item escaped provider slot aliases"
            ) from exc
        value: dict[str, Any] = {
            "handle_ids": list(item.handle_ids),
            "kind": item.kind.value,
            "summary": item.summary,
            "value_authority": item.value_authority.value,
        }
        if item.content_coherence is not ContentCoherence.MATCH:
            value["content_coherence"] = item.content_coherence.value
        if not item.included:
            value["included"] = False
        if item.status is not EvidenceStatus.UNKNOWN:
            value["status"] = item.status.value
        if supported_slot_ids:
            value["supported_slot_ids"] = supported_slot_ids
        optional = {
            "date": item.date,
            "entity_key": item.entity_key,
            "group_key": (
                None
                if item.group_key is None
                else group_alias_by_id.get(item.group_key, item.group_key)
            ),
            "numeric_role": (
                None
                if item.numeric_role is NumericRole.NONE
                else item.numeric_role.value
            ),
            "numeric_qualifier": (
                item.numeric_qualifier.value
                if item.numeric_value is not None
                and item.numeric_qualifier is not NumericQualifier.EXACT
                else None
            ),
            "numeric_value": item.numeric_value,
            "participant_count": item.participant_count,
            "personalization_anchors": (
                list(item.personalization_anchors)
                if item.personalization_anchors
                else None
            ),
            "relation": item.relation,
            "specificity_terms": (
                list(item.specificity_terms)
                if item.specificity_terms
                else None
            ),
            "unit": item.unit,
        }
        value.update(
            {key: child for key, child in optional.items() if child is not None}
        )
        compact_items.append(value)
    result = {"handles": handles, "items": compact_items}
    assert_gold_blind(result, path="compact_typed_evidence_content_v2")
    return result


def _compact_provider_projection(
    *,
    operator_spec: TypedOperatorSpec,
    frontier: EvidenceFrontierReceipt,
    conflict_policy: ConflictPolicy,
    items: Sequence[TypedEvidenceItem],
    bindings: Sequence[EvidenceHandleBinding],
) -> dict[str, Any]:
    """Historical compact-final v1 projection; keep byte-identical."""

    spec = operator_spec
    operator = {
        "absence_decision_requires_closed_frontier": (
            spec.absence_decision_requires_closed_frontier
        ),
        "answer_shape": spec.answer_shape.value,
        "cardinality": spec.cardinality,
        "comparison_mode": spec.comparison_mode.value,
        "include_proposed": spec.include_proposed,
        "operation": spec.operation,
        "ordering": spec.ordering,
        "personalization_required": spec.personalization_required,
        "query_timestamp": spec.query_timestamp,
        "required_evidence_role": spec.required_evidence_role,
        "required_slots": [
            {
                "kind": slot.kind.value,
                "label": slot.label,
                "match_terms": list(slot.match_terms),
                "minimum_match_term_count": slot.minimum_match_term_count,
                "relation_constraint": slot.relation_constraint,
                "requires_numeric": slot.requires_numeric,
                "slot_id": slot.slot_id,
            }
            for slot in spec.required_slots
        ],
        "requires_all_slots": spec.requires_all_slots,
        "requires_complete_frontier": spec.requires_complete_frontier,
        "specificity_required": spec.specificity_required,
        "style": spec.style.value,
        "temporal_mode": spec.temporal_mode.value,
        "temporal_window_days": spec.temporal_window_days,
    }
    compact_frontier = {
        "available_handle_ids": list(frontier.available_handle_ids),
        "closed": frontier.closed,
        "mode": frontier.mode.value,
        "omitted_handle_ids": list(frontier.omitted_handle_ids),
        "rejected_item_count": len(frontier.rejected_item_receipt_sha256s),
        "represented_handle_ids": list(frontier.represented_handle_ids),
        "truncated": frontier.truncated,
        "unresolved_slot_ids": list(frontier.unresolved_slot_ids),
    }
    value = {
        "conflict_policy": conflict_policy.value,
        "format": COMPACT_FINAL_PROVIDER_FORMAT,
        "frontier": compact_frontier,
        **compact_evidence_content_projection(items, bindings),
        "operator_spec": operator,
    }
    assert_gold_blind(value, path="compact_typed_evidence")
    return value


def _compact_provider_projection_v2(
    *,
    operator_spec: TypedOperatorSpec,
    frontier: EvidenceFrontierReceipt,
    conflict_policy: ConflictPolicy,
    items: Sequence[TypedEvidenceItem],
    bindings: Sequence[EvidenceHandleBinding],
) -> dict[str, Any]:
    spec = operator_spec
    slot_alias_by_id = {
        slot.slot_id: f"S{index:03d}"
        for index, slot in enumerate(spec.required_slots, start=1)
    }
    operator = {
        "absence_decision_requires_closed_frontier": (
            spec.absence_decision_requires_closed_frontier
        ),
        "answer_shape": spec.answer_shape.value,
        "cardinality": spec.cardinality,
        "comparison_mode": spec.comparison_mode.value,
        "include_proposed": spec.include_proposed,
        "operation": spec.operation,
        "ordering": spec.ordering,
        "personalization_required": spec.personalization_required,
        "query_timestamp": spec.query_timestamp,
        "required_evidence_role": spec.required_evidence_role,
        "required_slots": [
            {
                "kind": slot.kind.value,
                "label": slot.label,
                "match_terms": list(slot.match_terms),
                "minimum_match_term_count": slot.minimum_match_term_count,
                "relation_constraint": slot.relation_constraint,
                "requires_numeric": slot.requires_numeric,
                "slot_id": slot_alias_by_id[slot.slot_id],
            }
            for slot in spec.required_slots
        ],
        "requires_all_slots": spec.requires_all_slots,
        "requires_complete_frontier": spec.requires_complete_frontier,
        "specificity_required": spec.specificity_required,
        "style": spec.style.value,
        "temporal_mode": spec.temporal_mode.value,
        "temporal_window_days": spec.temporal_window_days,
    }
    try:
        unresolved_slot_ids = [
            slot_alias_by_id[slot_id] for slot_id in frontier.unresolved_slot_ids
        ]
    except KeyError as exc:
        raise MatchedEvalContractError(
            "compact frontier escaped provider slot aliases"
        ) from exc
    compact_frontier = {
        "closed": frontier.closed,
        "mode": frontier.mode.value,
        "omitted_handle_ids": list(frontier.omitted_handle_ids),
        "rejected_item_count": len(frontier.rejected_item_receipt_sha256s),
        "truncated": frontier.truncated,
        "unresolved_slot_ids": unresolved_slot_ids,
    }
    value = {
        "conflict_policy": conflict_policy.value,
        "defaults": {
            "item": {
                "content_coherence": ContentCoherence.MATCH.value,
                "included": True,
                "numeric_qualifier_when_numeric": NumericQualifier.EXACT.value,
                "status": EvidenceStatus.UNKNOWN.value,
                "supported_slot_ids": [],
            },
            "provenance_grade_by_origin": {
                origin.value: grade.value
                for origin, grade in (
                    (
                        EvidenceOrigin.DIRECT_POINTER,
                        ProvenanceGrade.DIRECT_POINTER,
                    ),
                    (EvidenceOrigin.MAP, ProvenanceGrade.EXACT_CITATION),
                    (
                        EvidenceOrigin.SOURCE_FACT,
                        ProvenanceGrade.EXACT_FACT_UNION,
                    ),
                )
            },
        },
        "format": COMPACT_FINAL_PROVIDER_FORMAT_V2,
        "frontier": compact_frontier,
        **_compact_final_evidence_content_projection_v2(
            items,
            bindings,
            slot_alias_by_id=slot_alias_by_id,
        ),
        "operator_spec": operator,
    }
    assert_gold_blind(value, path="compact_typed_evidence")
    return value


@dataclass(frozen=True, slots=True)
class TypedEvidencePacket:
    operator_spec: TypedOperatorSpec
    handles: tuple[OpaqueEvidenceHandle, ...]
    local_bindings: tuple[EvidenceHandleBinding, ...]
    items: tuple[TypedEvidenceItem, ...]
    rejected_items: tuple[RejectedTypedItem, ...]
    frontier: EvidenceFrontierReceipt
    conflict_policy: ConflictPolicy
    sealed_input_artifact_sha256s: tuple[str, ...]
    output_token_reserve: int
    provider_payload_token_proxy: int
    provider_payload_mode: ProviderPayloadMode = ProviderPayloadMode.CANONICAL
    hard_prompt_token_cap: Literal[8000] = HARD_PROMPT_TOKEN_CAP
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.operator_spec) is not TypedOperatorSpec:
            raise MatchedEvalContractError("packet requires an exact operator spec")
        if type(self.handles) is not tuple or any(type(row) is not OpaqueEvidenceHandle for row in self.handles):
            raise MatchedEvalContractError("packet handles changed type")
        if type(self.local_bindings) is not tuple or any(type(row) is not EvidenceHandleBinding for row in self.local_bindings):
            raise MatchedEvalContractError("packet bindings changed type")
        if type(self.items) is not tuple or any(type(row) is not TypedEvidenceItem for row in self.items):
            raise MatchedEvalContractError("packet items changed type")
        if type(self.rejected_items) is not tuple or any(type(row) is not RejectedTypedItem for row in self.rejected_items):
            raise MatchedEvalContractError("packet rejections changed type")
        if type(self.frontier) is not EvidenceFrontierReceipt:
            raise MatchedEvalContractError("packet frontier changed type")
        if type(self.conflict_policy) is not ConflictPolicy:
            raise MatchedEvalContractError("packet conflict policy changed type")
        if type(self.provider_payload_mode) is not ProviderPayloadMode:
            raise MatchedEvalContractError("packet provider payload mode changed")
        ids = tuple(row.handle_id for row in self.handles)
        binding_ids = tuple(row.handle_id for row in self.local_bindings)
        if len(set(ids)) != len(ids) or ids != binding_ids:
            raise MatchedEvalContractError("opaque handles lost exact local bindings")
        if any(row.binding_receipt_sha256 != binding.receipt_sha256 for row, binding in zip(self.handles, self.local_bindings, strict=True)):
            raise MatchedEvalContractError("opaque handle receipt lost its local binding")
        if any(not set(row.handle_ids) <= set(ids) for row in self.items):
            raise MatchedEvalContractError("typed item cites an unknown handle")
        slot_ids = {row.slot_id for row in self.operator_spec.required_slots}
        if any(not set(row.supported_slot_ids) <= slot_ids for row in self.items):
            raise MatchedEvalContractError("typed item escaped question-derived slots")
        _ordered_unique(self.sealed_input_artifact_sha256s, "sealed input artifacts")
        for value in self.sealed_input_artifact_sha256s:
            require_sha256(value, "sealed input artifact")
        if self.hard_prompt_token_cap != HARD_PROMPT_TOKEN_CAP:
            raise MatchedEvalContractError("typed packet hard cap must remain 8k")
        if type(self.output_token_reserve) is not int or not 1 <= self.output_token_reserve < HARD_PROMPT_TOKEN_CAP:
            raise MatchedEvalContractError("typed packet output reserve is invalid")
        expected_tokens = count_tokens(self.render_provider_payload())
        if self.provider_payload_token_proxy != expected_tokens:
            raise MatchedEvalContractError("typed packet token proxy changed")
        if expected_tokens + self.output_token_reserve > HARD_PROMPT_TOKEN_CAP:
            raise MatchedEvalContractError("typed packet exceeds the hard 8k envelope")
        if self.provider_prompt_count != 0 or self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("typed packet must remain provider-free and zero-state")
        if self.gold_loaded is not False:
            raise MatchedEvalContractError("typed packet cannot load gold")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("typed packet receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_evidence_packet")

    def provider_projection(self) -> dict[str, Any]:
        if self.provider_payload_mode in {
            ProviderPayloadMode.COMPACT_FINAL,
            ProviderPayloadMode.COMPACT_FINAL_V2,
        }:
            projector = (
                _compact_provider_projection_v2
                if self.provider_payload_mode is ProviderPayloadMode.COMPACT_FINAL_V2
                else _compact_provider_projection
            )
            return projector(
                operator_spec=self.operator_spec,
                frontier=self.frontier,
                conflict_policy=self.conflict_policy,
                items=self.items,
                bindings=self.local_bindings,
            )
        value = {
            "conflict_policy": self.conflict_policy.value,
            "format": FORMAT,
            "frontier": self.frontier.projection(),
            "gold_loaded": False,
            "handles": [row.projection() for row in self.handles],
            "items": [row.projection() for row in self.items],
            "operator_spec": self.operator_spec.projection(),
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
        }
        assert_gold_blind(value, path="typed_provider_projection")
        return value

    def render_provider_payload(self) -> str:
        return json.dumps(
            self.provider_projection(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "conflict_policy": self.conflict_policy.value,
            "format": FORMAT,
            "frontier": self.frontier.projection(),
            "gold_loaded": False,
            "handles": [row.projection() for row in self.handles],
            "hard_prompt_token_cap": 8_000,
            "items": [row.projection() for row in self.items],
            "local_bindings": [row.projection() for row in self.local_bindings],
            "operator_spec": self.operator_spec.projection(),
            "output_token_reserve": self.output_token_reserve,
            "provider_payload_token_proxy": self.provider_payload_token_proxy,
            "provider_prompt_count": 0,
            "rejected_items": [row.projection() for row in self.rejected_items],
            "retained_transformer_token_state_bytes": 0,
            "sealed_input_artifact_sha256s": list(self.sealed_input_artifact_sha256s),
        }
        if self.provider_payload_mode is not ProviderPayloadMode.CANONICAL:
            value["provider_payload_mode"] = self.provider_payload_mode.value
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def compact_typed_evidence_projection(
    packet: TypedEvidencePacket,
) -> dict[str, Any]:
    if type(packet) is not TypedEvidencePacket:
        raise TypeError("packet must be an exact TypedEvidencePacket")
    projector = (
        _compact_provider_projection_v2
        if packet.provider_payload_mode is ProviderPayloadMode.COMPACT_FINAL_V2
        else _compact_provider_projection
    )
    return projector(
        operator_spec=packet.operator_spec,
        frontier=packet.frontier,
        conflict_policy=packet.conflict_policy,
        items=packet.items,
        bindings=packet.local_bindings,
    )


# A negative fact is still a fact.  In particular, Boolean ``No`` answers and
# active obligations such as ``has not yet picked up`` must remain usable
# evidence.  Only language saying that the value itself is missing is an
# insufficiency marker; item-to-item contradictions are assessed separately.
_META_INSUFFICIENCY_RE = re.compile(
    r"\b(?:unknown|unspecified|not stated|not identified)\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?"
    r"(?:,?\s+(?:19|20)\d{2})?\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(?:19|20)\d{2}\b",
    re.I,
)


def _slot_supported(slot: RequiredSlot, summary: str, numeric_value: float | None, participant_count: int | None) -> bool:
    terms = set(normalized_terms(summary))
    matched = sum(term in terms for term in slot.match_terms)
    if matched < slot.minimum_match_term_count:
        return False
    if slot.requires_numeric and numeric_value is None:
        return False
    if slot.relation_constraint == "participant_singular" and participant_count != 1:
        return False
    return not bool(_META_INSUFFICIENCY_RE.search(summary))


def _typed_kind(value: str) -> TypedItemKind:
    normalized = value.casefold()
    if "operand" in normalized or "numeric" in normalized:
        return TypedItemKind.OPERAND
    if "event" in normalized or "timeline" in normalized or "temporal" in normalized:
        return TypedItemKind.EVENT
    if "member" in normalized or "set" in normalized:
        return TypedItemKind.MEMBER
    if "state" in normalized:
        return TypedItemKind.STATE
    if "claim" in normalized or "synth" in normalized:
        return TypedItemKind.CLAIM
    return TypedItemKind.DIRECT


def conservative_numeric_value(summary: str) -> float | None:
    """Return one unambiguous scalar mention, otherwise ``None``.

    This compatibility API is deliberately unbound to a question, but still
    shares the canonical numeric semantics that reject calendar and rank
    numerals.  Evidence adapters use the bound API directly.
    """

    if type(summary) is not str:
        raise TypeError("numeric inference summary must be exact text")
    mention = single_numeric_mention(summary)
    return None if mention is None else mention.value


def _inferred_number(
    summary: str,
    *,
    operator_spec: TypedOperatorSpec,
    dated_question: str,
):
    return single_numeric_mention(
        summary,
        operator_spec=operator_spec,
        question=dated_question,
    )


def _inferred_role(summary: str) -> NumericRole:
    if re.search(r"\b(?:baseline|started|initial(?:ly)?)\b", summary, re.I):
        return NumericRole.BASELINE
    if re.search(r"\b(?:ended|ending|current|now|reached|grew to)\b", summary, re.I):
        return NumericRole.END
    if re.search(r"\b(?:increase|gain|grew by|decrease|loss|delta)\b", summary, re.I):
        return NumericRole.DELTA
    return NumericRole.OPERAND


def _inferred_status(summary: str) -> EvidenceStatus:
    if re.search(r"\b(?:cancelled|canceled|abandoned)\b", summary, re.I):
        return EvidenceStatus.CANCELLED
    if re.search(
        r"\b(?:needs?\s+to|still\s+needs?|not\s+yet|awaiting|pending)\b",
        summary,
        re.I,
    ):
        return EvidenceStatus.CURRENT
    if re.search(r"\b(?:plan|planned|proposed|intend)\b", summary, re.I):
        return EvidenceStatus.PROPOSED
    if re.search(r"\b(?:current|currently|now|latest)\b", summary, re.I):
        return EvidenceStatus.CURRENT
    if re.search(r"\b(?:completed|finished|did|bought|paid|spent|went|visited)\b", summary, re.I):
        return EvidenceStatus.COMPLETED
    return EvidenceStatus.UNKNOWN


def _content_assessment(spec: TypedOperatorSpec, summary: str, supported: tuple[str, ...], participant_count: int | None) -> tuple[ContentCoherence, str | None]:
    reasons: list[str] = []
    if any(
        slot.relation_constraint == "participant_singular"
        and "friend" in set(normalized_terms(summary))
        and participant_count not in {None, 1}
        for slot in spec.required_slots
    ):
        reasons.append("participant_cardinality_mismatch")
    if reasons:
        body = {
            "content_derived_reasons": reasons,
            "operator_spec_receipt_sha256": spec.receipt_sha256,
            "summary_sha256": _text_sha256(summary),
        }
        return ContentCoherence.CONFLICT, identity_sha256(body)
    if _META_INSUFFICIENCY_RE.search(summary):
        return ContentCoherence.UNRESOLVED, None
    if supported or not spec.required_slots:
        return ContentCoherence.MATCH, None
    return ContentCoherence.UNRESOLVED, None


_RAW_OPTIONAL = frozenset(
    {
        "date", "entity_key", "group_key", "included", "kind", "numeric_role",
        "numeric_qualifier", "numeric_value", "participant_count",
        "personalization_anchors", "relation", "specificity_terms", "status",
        "unit", "value_authority",
    }
)


def _parse_one(raw: object, *, index: int, spec: TypedOperatorSpec, known_handles: set[str]) -> TypedEvidenceItem:
    if type(raw) is not dict or not {"handle_ids", "summary"} <= set(raw) or not set(raw) <= _RAW_OPTIONAL | {"handle_ids", "summary"}:
        raise ValueError("item_schema")
    handles_raw = raw["handle_ids"]
    summary = raw["summary"]
    if type(handles_raw) is not list or not handles_raw or any(type(row) is not str for row in handles_raw):
        raise ValueError("handle_schema")
    handles = tuple(handles_raw)
    if len(set(handles)) != len(handles) or not set(handles) <= known_handles:
        raise ValueError("unknown_or_repeated_handle")
    if type(summary) is not str or not summary or summary.strip() != summary:
        raise ValueError("summary_schema")
    try:
        kind = TypedItemKind(raw.get("kind", TypedItemKind.DIRECT.value))
        numeric_role = NumericRole(raw.get("numeric_role", NumericRole.NONE.value))
        status = (
            EvidenceStatus(raw["status"])
            if "status" in raw
            else _inferred_status(summary)
        )
        authority = ValueAuthority(raw.get("value_authority", ValueAuthority.EXPLICIT.value))
    except ValueError as exc:
        raise ValueError("enum_schema") from exc
    numeric_value = raw.get("numeric_value")
    if numeric_value is not None and type(numeric_value) not in {int, float}:
        raise ValueError("numeric_schema")
    compatible_mentions = numeric_mentions(summary, operator_spec=spec)
    matching_mentions = tuple(
        mention
        for mention in compatible_mentions
        if numeric_value is not None
        and abs(float(numeric_value) - mention.value) <= 1e-9
    )
    # A raw typed value is never stronger than the cited summary.  Keep the
    # evidence item, but strip a calendar/duration/rank contaminant rather than
    # rejecting the citation wholesale.
    if numeric_value is not None and not matching_mentions:
        numeric_value = None
        numeric_role = NumericRole.NONE
    try:
        inferred_qualifier = (
            matching_mentions[0].qualifier
            if len(matching_mentions) == 1
            else NumericQualifier.EXACT
        )
        numeric_qualifier = NumericQualifier(
            raw.get("numeric_qualifier", inferred_qualifier.value)
        )
    except ValueError as exc:
        raise ValueError("enum_schema") from exc
    if numeric_value is None:
        numeric_qualifier = NumericQualifier.EXACT
    elif (
        "numeric_qualifier" in raw
        and len(matching_mentions) == 1
        and numeric_qualifier is not matching_mentions[0].qualifier
    ):
        raise ValueError("numeric_qualifier_semantics")
    participant_count = raw.get("participant_count")
    if participant_count is not None and (type(participant_count) is not int or participant_count < 0):
        raise ValueError("participant_schema")
    included = raw.get("included", True)
    if type(included) is not bool:
        raise ValueError("included_schema")
    text_fields: dict[str, str | None] = {}
    for key in ("entity_key", "group_key", "unit", "date", "relation"):
        value = raw.get(key)
        if value is not None and (type(value) is not str or not value or value.strip() != value):
            raise ValueError(f"{key}_schema")
        text_fields[key] = value
    term_fields: dict[str, tuple[str, ...]] = {}
    for key in ("specificity_terms", "personalization_anchors"):
        value = raw.get(key, [])
        if type(value) is not list or any(type(row) is not str or not row for row in value):
            raise ValueError(f"{key}_schema")
        normalized = tuple(dict.fromkeys(term for row in value for term in normalized_terms(row)))
        term_fields[key] = normalized
    supported = tuple(
        slot.slot_id
        for slot in spec.required_slots
        if _slot_supported(slot, summary, numeric_value, participant_count)
    )
    coherence, conflict_receipt = _content_assessment(spec, summary, supported, participant_count)
    body = {
        "format": ITEM_FORMAT,
        "handle_ids": list(handles),
        "source_index": index,
        "summary_sha256": _text_sha256(summary),
        "supported_slot_ids": list(supported),
    }
    item_id = identity_sha256(body)
    return TypedEvidenceItem(
        item_id, handles, kind, summary, text_fields["entity_key"],
        text_fields["group_key"], numeric_value, numeric_role, numeric_qualifier,
        text_fields["unit"], text_fields["date"], status,
        text_fields["relation"], participant_count, authority, included,
        supported, coherence, coherence is ContentCoherence.CONFLICT,
        conflict_receipt, term_fields["specificity_terms"],
        term_fields["personalization_anchors"],
    )


def parse_typed_items(
    raw_items: object,
    *,
    operator_spec: TypedOperatorSpec,
    bindings: tuple[EvidenceHandleBinding, ...],
) -> ParsedTypedItems:
    """Validate each item independently; one bad item never rejects siblings."""

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    if type(bindings) is not tuple or any(type(row) is not EvidenceHandleBinding for row in bindings):
        raise TypeError("bindings must be an exact tuple")
    known = {row.handle_id for row in bindings}
    if len(known) != len(bindings):
        raise MatchedEvalContractError("typed bindings repeat handles")
    if type(raw_items) is not list:
        rejected = (_rejected(-1, "root_schema", raw_items),)
        body = {"accepted_item_receipt_sha256s": [], "format": f"{ITEM_FORMAT}-parse", "rejected_item_receipt_sha256s": [rejected[0].rejection_sha256]}
        return ParsedTypedItems((), rejected, identity_sha256(body))
    accepted: list[TypedEvidenceItem] = []
    rejected: list[RejectedTypedItem] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        try:
            item = _parse_one(raw, index=index, spec=operator_spec, known_handles=known)
        except (MatchedEvalContractError, TypeError, ValueError) as exc:
            reason = str(exc) or type(exc).__name__
            rejected.append(_rejected(index, reason, raw))
            continue
        if item.item_id in seen:
            rejected.append(_rejected(index, "duplicate_item", raw))
            continue
        seen.add(item.item_id)
        accepted.append(item)
    body = {
        "accepted_item_receipt_sha256s": [row.receipt_sha256 for row in accepted],
        "format": f"{ITEM_FORMAT}-parse",
        "rejected_item_receipt_sha256s": [row.rejection_sha256 for row in rejected],
    }
    return ParsedTypedItems(tuple(accepted), tuple(rejected), identity_sha256(body))


def build_frontier_receipt(
    spec: TypedOperatorSpec,
    bindings: tuple[EvidenceHandleBinding, ...],
    items: tuple[TypedEvidenceItem, ...],
    rejected: tuple[RejectedTypedItem, ...],
    *,
    mode: FrontierMode,
    truncated: bool = False,
    conflict_policy: ConflictPolicy = ConflictPolicy.QUARANTINE,
) -> EvidenceFrontierReceipt:
    if type(mode) is not FrontierMode or type(conflict_policy) is not ConflictPolicy:
        raise TypeError("frontier controls must be canonical")
    available = tuple(row.handle_id for row in bindings)
    represented_set = {handle for item in items for handle in item.handle_ids}
    represented = tuple(row for row in available if row in represented_set)
    omitted = tuple(row for row in available if row not in represented_set)
    usable = tuple(
        row for row in items
        if row.included
        and (not row.content_conflict or conflict_policy is ConflictPolicy.FAIL_OPEN)
    )
    supported = {slot_id for row in usable for slot_id in row.supported_slot_ids}
    unresolved = tuple(row.slot_id for row in spec.required_slots if row.slot_id not in supported)
    capacity_rejected = any(row.reason.startswith("hard_8k_") for row in rejected)
    effective_truncated = bool(truncated or capacity_rejected)
    closed = bool(
        mode is FrontierMode.EXHAUSTIVE
        and not effective_truncated
        and not omitted
        and not unresolved
    )
    return EvidenceFrontierReceipt(
        mode, available, represented, omitted,
        tuple(row.rejection_sha256 for row in rejected), unresolved,
        effective_truncated, closed,
    )


def _binding(
    *,
    index: int,
    origin: EvidenceOrigin,
    group_handle: str,
    artifact_sha256: str,
    parent_receipt_sha256: str,
    evidence_receipt_sha256: str,
    payload: str,
    citation: str,
    local_source_locator: object,
) -> EvidenceHandleBinding:
    grade = {
        EvidenceOrigin.MAP: ProvenanceGrade.EXACT_CITATION,
        EvidenceOrigin.SOURCE_FACT: ProvenanceGrade.EXACT_FACT_UNION,
        EvidenceOrigin.DIRECT_POINTER: ProvenanceGrade.DIRECT_POINTER,
        EvidenceOrigin.MEM0: ProvenanceGrade.INFERRED_MEMORY,
    }[origin]
    return EvidenceHandleBinding(
        f"H{index:03d}", origin, grade, group_handle, artifact_sha256,
        parent_receipt_sha256, evidence_receipt_sha256,
        _text_sha256(payload), _text_sha256(citation), len(citation),
        identity_sha256(local_source_locator),
    )


def _map_item_sealed(item: ValidatedMapItem) -> bool:
    body = {
        "alias": item.alias,
        "candidate": item.candidate,
        "citation": item.citation,
        "citation_match": item.citation_match,
        "format": MAP_ITEM_FORMAT,
        "item_id": item.item_id,
        "kind": item.kind,
        "source_index": item.source_index,
    }
    return item.item_sha256 == identity_sha256(body)


def _auto_raw_item(
    item: ValidatedMapItem,
    handle_id: str,
    *,
    operator_spec: TypedOperatorSpec,
    dated_question: str,
) -> dict[str, Any]:
    summary = item.candidate
    mention = _inferred_number(
        summary,
        operator_spec=operator_spec,
        dated_question=dated_question,
    )
    role = _inferred_role(summary) if mention is not None else NumericRole.NONE
    date_match = _DATE_RE.search(summary)
    participant_count = None
    if re.search(r"\bwith\s+(?:a|one)\s+friend\b", summary, re.I):
        participant_count = 1
    elif re.search(r"\bwith\s+friends\b", summary, re.I):
        participant_count = 2
    raw: dict[str, Any] = {
        "handle_ids": [handle_id],
        "included": not bool(re.search(r"\b(?:exclude|excluded|ineligible)\b", summary, re.I)),
        "kind": _typed_kind(item.kind).value,
        "numeric_role": role.value,
        "status": _inferred_status(summary).value,
        "summary": summary,
        "value_authority": ValueAuthority.EXPLICIT.value,
    }
    if mention is not None:
        raw["numeric_qualifier"] = mention.qualifier.value
        raw["numeric_value"] = mention.value
        if mention.unit is not None:
            raw["unit"] = mention.unit
    if date_match is not None:
        raw["date"] = date_match.group(0)
    if participant_count is not None:
        raw["participant_count"] = participant_count
    return raw


def _group_handles(keys: Sequence[object], *, start: int = 1) -> tuple[str, ...]:
    if type(start) is not int or not 1 <= start <= 999_999:
        raise MatchedEvalContractError("opaque source-group start is invalid")
    ordinal_by_key: dict[str, str] = {}
    result: list[str] = []
    for key in keys:
        digest = identity_sha256(key)
        ordinal = start + len(ordinal_by_key)
        if ordinal > 999_999:
            raise MatchedEvalContractError("opaque source-group allocation overflow")
        ordinal_by_key.setdefault(digest, f"G{ordinal:03d}")
        result.append(ordinal_by_key[digest])
    return tuple(result)


def _source_group_key(admission: LaneAdmission) -> object:
    # The group relation is computed locally and rendered only as Gnnn.  It is
    # never compared with a question ID or exposed as a source/namespace hash.
    return {
        "origin_source_locators": sorted(
            identity_sha256({"namespace": row.namespace_id, "source": row.source_id})
            for row in admission.union_fact.origins
        )
    }


def _packet_with_budget_salvage(
    *,
    spec: TypedOperatorSpec,
    bindings: tuple[EvidenceHandleBinding, ...],
    parsed: ParsedTypedItems,
    mode: FrontierMode,
    conflict_policy: ConflictPolicy,
    sealed_artifacts: tuple[str, ...],
    output_token_reserve: int,
    truncated: bool,
    provider_payload_mode: ProviderPayloadMode = ProviderPayloadMode.CANONICAL,
) -> TypedEvidencePacket:
    def projection_for(
        items: tuple[TypedEvidenceItem, ...],
        rejected_items: tuple[RejectedTypedItem, ...],
    ) -> tuple[EvidenceFrontierReceipt, tuple[OpaqueEvidenceHandle, ...], dict[str, Any]]:
        frontier = build_frontier_receipt(
            spec,
            bindings,
            items,
            rejected_items,
            mode=mode,
            truncated=truncated,
            conflict_policy=conflict_policy,
        )
        handles = tuple(row.opaque() for row in bindings)
        if provider_payload_mode in {
            ProviderPayloadMode.COMPACT_FINAL,
            ProviderPayloadMode.COMPACT_FINAL_V2,
        }:
            projector = (
                _compact_provider_projection_v2
                if provider_payload_mode is ProviderPayloadMode.COMPACT_FINAL_V2
                else _compact_provider_projection
            )
            projection = projector(
                operator_spec=spec,
                frontier=frontier,
                conflict_policy=conflict_policy,
                items=items,
                bindings=bindings,
            )
        else:
            projection = {
                "conflict_policy": conflict_policy.value,
                "format": FORMAT,
                "frontier": frontier.projection(),
                "gold_loaded": False,
                "handles": [row.projection() for row in handles],
                "items": [row.projection() for row in items],
                "operator_spec": spec.projection(),
                "provider_prompt_count": 0,
                "retained_transformer_token_state_bytes": 0,
            }
        return frontier, handles, projection

    fit_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], bool] = {}

    def fits(
        items: tuple[TypedEvidenceItem, ...],
        rejected_items: tuple[RejectedTypedItem, ...],
        *,
        memoize: bool = True,
    ) -> bool:
        cache_key = (
            tuple(row.receipt_sha256 for row in items),
            tuple(row.rejection_sha256 for row in rejected_items),
        )
        if memoize and cache_key in fit_cache:
            return fit_cache[cache_key]
        _frontier, _handles, projection = projection_for(items, rejected_items)
        rendered = json.dumps(
            projection,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        result = (
            count_tokens(rendered) + output_token_reserve
            <= HARD_PROMPT_TOKEN_CAP
        )
        if memoize:
            fit_cache[cache_key] = result
        return result

    kept: list[TypedEvidenceItem] = []
    rejected = list(parsed.rejected_items)
    for item in parsed.accepted_items:
        trial = tuple((*kept, item))
        if not fits(trial, tuple(rejected)):
            rejected.append(_rejected(len(rejected), "hard_8k_item_overflow", item.projection()))
        else:
            kept.append(item)

        # A newly recorded rejection grows the provider-visible frontier by one
        # receipt.  Account for that growth immediately: otherwise a trial that
        # fit earlier can become invalid only at final packet construction.  If
        # an eviction is required, remove the latest admitted item so callers
        # that deliberately place protected/fair-share minima first cannot have
        # those minima silently starved by later fill items.
        while not fits(tuple(kept), tuple(rejected)):
            if not kept:
                raise MatchedEvalContractError(
                    "typed packet metadata exceeds the hard 8k envelope"
                )
            evicted = kept.pop()
            rejected.append(
                _rejected(
                    len(rejected),
                    "hard_8k_frontier_growth_overflow",
                    evicted.projection(),
                )
            )

    final_items = tuple(kept)
    # Rebuild and retokenize the final population even if it was probed during
    # salvage.  The cache removes only redundant exploratory work; this exact
    # verification remains the authoritative hard-cap check.
    if not fits(final_items, tuple(rejected), memoize=False):
        raise MatchedEvalContractError(
            "typed packet metadata exceeds the hard 8k envelope"
        )
    frontier, handles, projection = projection_for(
        final_items, tuple(rejected)
    )
    rendered = json.dumps(
        projection,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return TypedEvidencePacket(
        spec, handles, bindings, final_items, tuple(rejected), frontier,
        conflict_policy, sealed_artifacts, output_token_reserve,
        count_tokens(rendered), provider_payload_mode,
    )


def build_typed_evidence_packet(
    operator_spec: TypedOperatorSpec,
    bindings: tuple[EvidenceHandleBinding, ...],
    parsed: ParsedTypedItems,
    *,
    sealed_input_artifact_sha256s: tuple[str, ...],
    frontier_mode: FrontierMode = FrontierMode.BOUNDED,
    conflict_policy: ConflictPolicy = ConflictPolicy.QUARANTINE,
    output_token_reserve: int = DEFAULT_OUTPUT_TOKEN_RESERVE,
    truncated: bool = False,
    provider_payload_mode: ProviderPayloadMode = ProviderPayloadMode.CANONICAL,
) -> TypedEvidencePacket:
    """Build a hard-capped packet from already typed, independently parsed items."""

    return _packet_with_budget_salvage(
        spec=operator_spec, bindings=bindings, parsed=parsed, mode=frontier_mode,
        conflict_policy=conflict_policy,
        sealed_artifacts=sealed_input_artifact_sha256s,
        output_token_reserve=output_token_reserve,
        truncated=truncated,
        provider_payload_mode=provider_payload_mode,
    )


def merge_typed_evidence_contributions(
    operator_spec: TypedOperatorSpec,
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    conflict_policy: ConflictPolicy = ConflictPolicy.QUARANTINE,
    output_token_reserve: int = DEFAULT_OUTPUT_TOKEN_RESERVE,
) -> TypedEvidencePacket:
    """Union sealed mechanism outputs without upgrading the weakest frontier."""

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    if (
        type(contributions) is not tuple
        or not contributions
        or any(type(row) is not TypedEvidenceContribution for row in contributions)
    ):
        raise TypeError("contributions must be a non-empty exact tuple")
    mechanism_ids = tuple(row.mechanism_id for row in contributions)
    if len(set(mechanism_ids)) != len(mechanism_ids):
        raise MatchedEvalContractError("typed contribution mechanisms repeat")
    bindings = tuple(binding for row in contributions for binding in row.bindings)
    handle_ids = tuple(row.handle_id for row in bindings)
    if len(set(handle_ids)) != len(handle_ids):
        raise MatchedEvalContractError(
            "typed contribution handles collide; allocate global opaque handles before merge"
        )
    seen_groups: set[str] = set()
    for contribution in contributions:
        contribution_groups = {
            row.source_group_handle for row in contribution.bindings
        }
        if seen_groups & contribution_groups:
            raise MatchedEvalContractError(
                "typed contribution source groups collide; allocate global opaque groups before merge"
            )
        seen_groups.update(contribution_groups)
    accepted = tuple(
        item for row in contributions for item in row.parsed.accepted_items
    )
    if len({row.item_id for row in accepted}) != len(accepted):
        raise MatchedEvalContractError("typed contribution item receipts collide")
    rejected = tuple(
        item for row in contributions for item in row.parsed.rejected_items
    )
    parsed = ParsedTypedItems(
        accepted,
        rejected,
        identity_sha256(
            {
                "contribution_receipt_sha256s": [
                    row.receipt_sha256 for row in contributions
                ],
                "format": f"{ITEM_FORMAT}-contribution-union",
            }
        ),
    )
    modes = {row.frontier_mode for row in contributions}
    if FrontierMode.OPEN in modes:
        merged_mode = FrontierMode.OPEN
    elif FrontierMode.BOUNDED in modes:
        merged_mode = FrontierMode.BOUNDED
    else:
        merged_mode = FrontierMode.EXHAUSTIVE
    artifacts = tuple(
        dict.fromkeys(row.sealed_artifact_sha256 for row in contributions)
    )
    return _packet_with_budget_salvage(
        spec=operator_spec,
        bindings=bindings,
        parsed=parsed,
        mode=merged_mode,
        conflict_policy=conflict_policy,
        sealed_artifacts=artifacts,
        output_token_reserve=output_token_reserve,
        truncated=any(row.truncated for row in contributions),
    )


def adapt_verified_evidence(
    operator_spec: TypedOperatorSpec,
    map_plan_row: EvidenceMapPlanRow,
    map_row: VerifiedEvidenceMapRow,
    *,
    map_artifact_sha256: str,
    fact_envelope: FactUnionEnvelope | None = None,
    source_artifact_sha256: str | None = None,
    frontier_mode: FrontierMode = FrontierMode.BOUNDED,
    conflict_policy: ConflictPolicy = ConflictPolicy.QUARANTINE,
    output_token_reserve: int = DEFAULT_OUTPUT_TOKEN_RESERVE,
    handle_start: int = 1,
    group_start: int = 1,
) -> TypedEvidencePacket:
    """Adapt the exact V2 plan/terminal row and optional packed source facts.

    This is the integration seam for the existing adaptive plan builder.  It
    consumes its already sealed objects and performs no provider operation.
    """

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    if type(map_plan_row) is not EvidenceMapPlanRow or type(map_row) is not VerifiedEvidenceMapRow:
        raise TypeError("adapter requires exact V2 map plan and terminal rows")
    require_sha256(map_artifact_sha256, "V2 map artifact")
    if (
        type(handle_start) is not int
        or not 1 <= handle_start <= 999_999
        or type(group_start) is not int
        or not 1 <= group_start <= 999_999
    ):
        raise MatchedEvalContractError("opaque handle/group allocation start is invalid")
    packet = map_plan_row.direct_plan_row.adapter.source.packet
    if not (
        operator_spec.question_sha256 == map_row.dated_question_sha256
        == packet.dated_question_sha256
        and operator_spec.style.value == map_row.route_id
        and map_row.map_plan_row_receipt_sha256 == map_plan_row.receipt_sha256
        and map_row.ordinal == map_plan_row.ordinal
    ):
        raise MatchedEvalContractError("typed adapter escaped its V2 question/route binding")
    dated_question = getattr(packet, "dated_question", "")
    if type(dated_question) is not str:
        dated_question = ""
    alias_by_name = {row.alias: row for row in map_plan_row.aliases}
    if len(alias_by_name) != len(map_plan_row.aliases):
        raise MatchedEvalContractError("V2 aliases repeat")

    valid_map: list[tuple[ValidatedMapItem, PayloadEvidenceAlias]] = []
    rejected: list[RejectedTypedItem] = []
    for index, item in enumerate(map_row.accepted_items):
        alias = alias_by_name.get(item.alias)
        if type(item) is not ValidatedMapItem or not _map_item_sealed(item):
            rejected.append(_rejected(index, "map_item_seal", repr(item)))
        elif alias is None:
            rejected.append(_rejected(index, "map_item_alias_binding", item.projection()))
        else:
            valid_map.append((item, alias))

    source_admissions = tuple(
        admission
        for pack in (() if fact_envelope is None else fact_envelope.lane_packs)
        for admission in pack.admissions
    )
    if fact_envelope is not None:
        if type(fact_envelope) is not FactUnionEnvelope:
            raise TypeError("fact_envelope must be exact")
        if source_artifact_sha256 is None:
            raise MatchedEvalContractError("source facts require a sealed artifact")
        require_sha256(source_artifact_sha256, "source fact artifact")
        if fact_envelope.hard_prompt_token_cap != HARD_PROMPT_TOKEN_CAP or fact_envelope.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("source envelope escaped 8k/zero-state invariants")
        require_sha256(fact_envelope.receipt_sha256, "source fact envelope receipt")
    elif source_artifact_sha256 is not None:
        raise MatchedEvalContractError("source artifact supplied without a fact envelope")

    valid_source: list[tuple[int, LaneAdmission]] = []
    for source_index, admission in enumerate(source_admissions):
        if type(admission) is not LaneAdmission:
            rejected.append(_rejected(source_index, "source_admission_type", repr(admission)))
            continue
        variants = admission.union_fact.fact_variants
        if not variants or any(type(row) is not str or not row for row in variants):
            rejected.append(_rejected(source_index, "source_fact_variants", repr(admission)))
            continue
        try:
            require_sha256(admission.receipt_sha256, "source admission receipt")
            require_sha256(
                admission.union_fact.receipt_sha256,
                "source union fact receipt",
            )
        except MatchedEvalContractError:
            rejected.append(_rejected(source_index, "source_fact_seal", repr(admission)))
            continue
        valid_source.append((source_index, admission))

    source_keys: list[object] = [
        {"source_locator": alias.source_id} for _item, alias in valid_map
    ] + [_source_group_key(row) for _index, row in valid_source]
    groups = _group_handles(source_keys, start=group_start)
    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, Any]] = []
    next_handle = handle_start
    for pair_index, (item, alias) in enumerate(valid_map):
        binding = _binding(
            index=next_handle, origin=EvidenceOrigin.MAP,
            group_handle=groups[pair_index], artifact_sha256=map_artifact_sha256,
            parent_receipt_sha256=map_row.map_parse_receipt_sha256,
            evidence_receipt_sha256=item.item_sha256, payload=item.candidate,
            citation=item.citation,
            local_source_locator={"alias": alias.alias, "evidence_id": alias.evidence_id, "source_id": alias.source_id},
        )
        bindings.append(binding)
        raw_items.append(
            _auto_raw_item(
                item,
                binding.handle_id,
                operator_spec=operator_spec,
                dated_question=dated_question,
            )
        )
        next_handle += 1
    for source_position, (source_index, admission) in enumerate(valid_source):
        variants = admission.union_fact.fact_variants
        summary = " | ".join(variants)
        binding = _binding(
            index=next_handle, origin=EvidenceOrigin.SOURCE_FACT,
            group_handle=groups[len(valid_map) + source_position],
            artifact_sha256=source_artifact_sha256 or "",
            parent_receipt_sha256=fact_envelope.receipt_sha256 if fact_envelope else "",
            evidence_receipt_sha256=admission.receipt_sha256,
            payload=summary, citation=summary,
            local_source_locator={"admission_alias": admission.alias, "union_fact_receipt_sha256": admission.union_fact.receipt_sha256},
        )
        bindings.append(binding)
        pseudo = ValidatedMapItem(
            f"F{source_index + 1:03d}", source_index,
            admission.union_fact.owner_lane.value, admission.alias, summary,
            summary, "source_fact", identity_sha256({
                "alias": admission.alias, "candidate": summary, "citation": summary,
                "citation_match": "source_fact", "format": MAP_ITEM_FORMAT,
                "item_id": f"F{source_index + 1:03d}",
                "kind": admission.union_fact.owner_lane.value,
                "source_index": source_index,
            }),
        )
        raw_items.append(
            _auto_raw_item(
                pseudo,
                binding.handle_id,
                operator_spec=operator_spec,
                dated_question=dated_question,
            )
        )
        next_handle += 1
    parsed = parse_typed_items(raw_items, operator_spec=operator_spec, bindings=tuple(bindings))
    merged = ParsedTypedItems(
        parsed.accepted_items,
        tuple((*rejected, *parsed.rejected_items)),
        identity_sha256({
            "accepted_item_receipt_sha256s": [row.receipt_sha256 for row in parsed.accepted_items],
            "format": f"{ITEM_FORMAT}-adapted-parse",
            "rejected_item_receipt_sha256s": [row.rejection_sha256 for row in (*rejected, *parsed.rejected_items)],
        }),
    )
    artifacts = (map_artifact_sha256,) if source_artifact_sha256 is None else (map_artifact_sha256, source_artifact_sha256)
    return _packet_with_budget_salvage(
        spec=operator_spec, bindings=tuple(bindings), parsed=merged,
        mode=frontier_mode, conflict_policy=conflict_policy,
        sealed_artifacts=artifacts, output_token_reserve=output_token_reserve,
        truncated=bool(fact_envelope is not None and any(pack.not_admitted_union_fact_ids for pack in fact_envelope.lane_packs)),
    )


def typed_packet_from_adaptive_plan_row(
    operator_spec: TypedOperatorSpec,
    adaptive_row: object,
    *,
    map_artifact_sha256: str,
    source_artifact_sha256: str | None = None,
    frontier_mode: FrontierMode = FrontierMode.BOUNDED,
    conflict_policy: ConflictPolicy = ConflictPolicy.QUARANTINE,
    handle_start: int = 1,
    group_start: int = 1,
) -> TypedEvidencePacket:
    """Narrow seam used immediately after adaptive plan construction.

    The import is local so this provider-free module does not broaden the live
    module import graph merely to name the convenience type.
    """

    from .adaptive_evidence_solver_live import AdaptiveEvidenceSolverPlanRow

    if type(adaptive_row) is not AdaptiveEvidenceSolverPlanRow:
        raise TypeError("adaptive_row must be an exact AdaptiveEvidenceSolverPlanRow")
    return adapt_verified_evidence(
        operator_spec, adaptive_row.map_plan_row, adaptive_row.map_row,
        map_artifact_sha256=map_artifact_sha256,
        fact_envelope=adaptive_row.fact_envelope,
        source_artifact_sha256=source_artifact_sha256,
        frontier_mode=frontier_mode,
        conflict_policy=conflict_policy,
        handle_start=handle_start,
        group_start=group_start,
    )


__all__ = [
    "COMPACT_FINAL_PROVIDER_FORMAT", "COMPACT_FINAL_PROVIDER_FORMAT_V2",
    "ConflictPolicy", "ContentCoherence",
    "DEFAULT_OUTPUT_TOKEN_RESERVE",
    "EvidenceFrontierReceipt", "EvidenceHandleBinding", "EvidenceOrigin",
    "EvidenceStatus", "FrontierMode", "HARD_PROMPT_TOKEN_CAP",
    "NumericQualifier", "NumericRole", "OpaqueEvidenceHandle", "ParsedTypedItems",
    "ProviderPayloadMode", "ProvenanceGrade",
    "RejectedTypedItem", "TypedEvidenceContribution", "TypedEvidenceItem", "TypedEvidencePacket",
    "TypedItemKind", "ValueAuthority", "adapt_verified_evidence",
    "build_frontier_receipt", "build_typed_evidence_packet",
    "compact_evidence_content_projection", "compact_typed_evidence_projection",
    "conservative_numeric_value",
    "merge_typed_evidence_contributions", "parse_typed_items",
    "typed_packet_from_adaptive_plan_row",
]
