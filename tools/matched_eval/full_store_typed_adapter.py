"""Provider-free adapter from full-store slot closure to typed evidence.

The full-store scanner owns exact raw citations and local source/partition
locators.  This adapter remaps its candidate and source-group identities into
caller-owned opaque H/G ranges, converts each selected citation into one typed
item, and retains the exact local citation projection outside the provider
payload.  Scanner dates remain useful for relative-memory ordering, but their
authority is explicit only for a textual date and derived for row timestamps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import FullStoreSlotClosureResult
from .typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    EvidenceStatus,
    FrontierMode,
    NumericRole,
    ParsedTypedItems,
    ProvenanceGrade,
    TypedEvidenceContribution,
    TypedItemKind,
    ValueAuthority,
    parse_typed_items,
)
from .typed_operator_spec import (
    AnswerShape,
    SlotKind,
    TemporalMode,
    TypedOperatorSpec,
    normalized_terms,
)
from .typed_numeric_semantics import single_numeric_mention


FORMAT = "memory-condense-full-store-typed-adapter-v2"
MECHANISM_ID = "full_store_slot_closure_v1"
_MAX = 999_999
def _require(ok: object, message: str) -> None:
    if not ok:
        raise MatchedEvalContractError(message)


def _number(
    text: str,
    *,
    numeric_slot_supported: bool,
    operator_spec: TypedOperatorSpec | None = None,
    dated_question: str | None = None,
) -> float | None:
    if not numeric_slot_supported:
        return None
    mention = single_numeric_mention(
        text,
        operator_spec=operator_spec,
        question=dated_question,
    )
    return None if mention is None else mention.value


def _role(text: str, value: float | None) -> NumericRole:
    if value is None:
        return NumericRole.NONE
    if re.search(r"\b(?:initial(?:ly)?|baseline|started)\b", text, re.I):
        return NumericRole.BASELINE
    if re.search(r"\b(?:current|now|ended|reached|grew to)\b", text, re.I):
        return NumericRole.END
    if re.search(r"\b(?:increase|gain|grew by|decrease|loss|delta)\b", text, re.I):
        return NumericRole.DELTA
    return NumericRole.OPERAND


def _status(text: str) -> EvidenceStatus:
    if re.search(r"\b(?:cancelled|canceled|abandoned)\b", text, re.I):
        return EvidenceStatus.CANCELLED
    if re.search(
        r"\b(?:needs?\s+to|still\s+needs?|not\s+yet|awaiting|pending)\b",
        text,
        re.I,
    ):
        return EvidenceStatus.CURRENT
    if re.search(r"\b(?:plan|planned|proposed|intend)\b", text, re.I):
        return EvidenceStatus.PROPOSED
    if re.search(r"\b(?:current|currently|now|latest)\b", text, re.I):
        return EvidenceStatus.CURRENT
    if re.search(r"\b(?:completed|finished|bought|paid|spent|went|visited|watched)\b", text, re.I):
        return EvidenceStatus.COMPLETED
    return EvidenceStatus.UNKNOWN


def _kind(spec: TypedOperatorSpec, supported: tuple[str, ...]) -> TypedItemKind:
    slot_by_id = {row.slot_id: row for row in spec.required_slots}
    kinds = {slot_by_id[row].kind for row in supported if row in slot_by_id}
    if SlotKind.OPERAND in kinds or spec.answer_shape is AnswerShape.NUMBER:
        return TypedItemKind.OPERAND
    if spec.temporal_mode is TemporalMode.LATEST_STATE:
        return TypedItemKind.STATE
    if spec.temporal_mode is not TemporalMode.NONE:
        return TypedItemKind.EVENT
    if spec.answer_shape is AnswerShape.SET_LIST:
        return TypedItemKind.MEMBER
    return TypedItemKind.DIRECT


def _unit(text: str) -> str | None:
    if "$" in text:
        return "$"
    if "%" in text or re.search(r"\bpercent\b", text, re.I):
        return "%"
    return None


def _specific_slot_entity(
    spec: TypedOperatorSpec,
    supported: tuple[str, ...],
    text: str,
) -> str | None:
    """Promote only a specific operand/comparison label present in evidence."""

    generic = {
        "boundary",
        "comparison",
        "constraint",
        "event",
        "numeric",
        "operand",
        "participant",
        "predicate",
        "question",
        "role",
        "side",
        "state",
        "value",
    }
    terms = set(normalized_terms(text))
    by_id = {row.slot_id: row for row in spec.required_slots}
    for slot_id in supported:
        slot = by_id.get(slot_id)
        if slot is None or slot.kind not in {
            SlotKind.OPERAND,
            SlotKind.COMPARISON_SIDE,
        }:
            continue
        label_terms = set(normalized_terms(slot.label))
        if label_terms and not label_terms & generic and label_terms <= terms:
            return slot.label
    return None


def _raw_item(
    spec: TypedOperatorSpec,
    candidate: Any,
    handle_id: str,
    *,
    dated_question: str | None = None,
) -> dict[str, Any]:
    text = candidate.quote
    slots = tuple(candidate.supported_slot_ids)
    slot_by_id = {row.slot_id: row for row in spec.required_slots}
    numeric_slot_supported = any(
        slot_by_id[row].requires_numeric for row in slots if row in slot_by_id
    ) or (
        spec.answer_shape is AnswerShape.NUMBER and not spec.required_slots
    )
    mention = (
        single_numeric_mention(
            text,
            operator_spec=spec,
            question=dated_question,
        )
        if numeric_slot_supported
        else None
    )
    numeric = None if mention is None else mention.value
    entity = _specific_slot_entity(spec, slots, text)
    raw: dict[str, Any] = {
        "handle_ids": [handle_id],
        "included": True,
        "kind": _kind(spec, slots).value,
        "numeric_role": _role(text, numeric).value,
        "status": _status(text).value,
        "summary": text,
        "value_authority": (
            ValueAuthority.DERIVED.value
            if candidate.event_date_basis == "row_created_at"
            else ValueAuthority.EXPLICIT.value
        ),
    }
    if numeric is not None:
        raw["numeric_qualifier"] = mention.qualifier.value
        raw["numeric_value"] = numeric
    if candidate.event_date is not None:
        raw["date"] = candidate.event_date
    if entity is not None:
        raw["entity_key"] = entity
        if numeric is not None:
            raw["group_key"] = entity
    unit = mention.unit if mention is not None else _unit(text)
    if unit is not None:
        raw["unit"] = unit
    if candidate.role:
        raw["relation"] = (
            f"memory_role:{candidate.role};date_basis:"
            f"{candidate.event_date_basis or 'none'}"
        )
    return raw


@dataclass(frozen=True, slots=True)
class FullStoreTypedAudit:
    contribution_receipt_sha256: str
    closure_receipt_sha256: str
    local_citation_bindings: tuple[dict[str, Any], ...]
    local_story_key_receipt_sha256_by_group: tuple[tuple[str, str], ...]
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.contribution_receipt_sha256, "full-store contribution")
        require_sha256(self.closure_receipt_sha256, "full-store closure")
        _require(
            type(self.local_citation_bindings) is tuple
            and all(type(row) is dict for row in self.local_citation_bindings),
            "full-store local citation audit changed type",
        )
        _require(
            type(self.local_story_key_receipt_sha256_by_group) is tuple
            and len({row[0] for row in self.local_story_key_receipt_sha256_by_group})
            == len(self.local_story_key_receipt_sha256_by_group),
            "full-store local story bindings repeat",
        )
        for group, receipt in self.local_story_key_receipt_sha256_by_group:
            require_text(group, "full-store opaque group")
            require_sha256(receipt, "full-store local story receipt")
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "full-store typed audit must remain provider-free/gold-blind/zero-state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "full-store typed audit changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="full_store_typed_audit")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "closure_receipt_sha256": self.closure_receipt_sha256,
            "contribution_receipt_sha256": self.contribution_receipt_sha256,
            "format": FORMAT,
            "gold_loaded": False,
            "local_citation_bindings": list(self.local_citation_bindings),
            "local_story_key_receipt_sha256_by_group": [
                {"group_handle": group, "local_story_key_receipt_sha256": receipt}
                for group, receipt in self.local_story_key_receipt_sha256_by_group
            ],
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def adapt_full_store_slot_closure(
    operator_spec: TypedOperatorSpec,
    result: FullStoreSlotClosureResult,
    *,
    closure_artifact_sha256: str,
    handle_start: int,
    group_start: int,
    mechanism_id: str = MECHANISM_ID,
) -> tuple[TypedEvidenceContribution, FullStoreTypedAudit]:
    """Adapt selected exact closure citations into a bounded contribution."""

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    if type(result) is not FullStoreSlotClosureResult:
        raise TypeError("result must be exact FullStoreSlotClosureResult")
    require_sha256(closure_artifact_sha256, "full-store closure artifact")
    require_text(mechanism_id, "full-store mechanism")
    _require(
        result.operator_spec.receipt_sha256 == operator_spec.receipt_sha256
        and result.receipt.typed_operator_spec_receipt_sha256
        == operator_spec.receipt_sha256,
        "full-store adapter escaped its operator spec",
    )
    _require(
        type(handle_start) is int
        and type(group_start) is int
        and 1 <= handle_start <= _MAX
        and 1 <= group_start <= _MAX,
        "full-store opaque range start changed",
    )
    _require(
        handle_start + len(result.candidates) - 1 <= _MAX,
        "full-store handle allocation overflow",
    )
    old_groups = tuple(dict.fromkeys(row.source_group_handle for row in result.candidates))
    _require(
        group_start + len(old_groups) - 1 <= _MAX,
        "full-store group allocation overflow",
    )
    remap = {old: f"G{group_start + index:03d}" for index, old in enumerate(old_groups)}
    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, Any]] = []
    story_by_group: dict[str, str] = {}
    local_rows: list[dict[str, Any]] = []
    for offset, (candidate, local) in enumerate(
        zip(result.candidates, result.local_bindings, strict=True)
    ):
        handle_id = f"H{handle_start + offset:03d}"
        group = remap[candidate.source_group_handle]
        local_projection = local.projection()
        bindings.append(
            EvidenceHandleBinding(
                handle_id,
                EvidenceOrigin.MAP,
                ProvenanceGrade.EXACT_CITATION,
                group,
                closure_artifact_sha256,
                result.receipt.receipt_sha256,
                candidate.candidate_id,
                quote_sha256(candidate.quote),
                candidate.quote_sha256,
                len(candidate.quote),
                identity_sha256(local_projection),
            )
        )
        raw_items.append(
            _raw_item(
                operator_spec,
                candidate,
                handle_id,
                dated_question=result.dated_question,
            )
        )
        exact_story_key = identity_sha256(
            {"namespace_id": local.namespace_id, "source_id": local.source_id}
        )
        previous = story_by_group.setdefault(group, exact_story_key)
        _require(previous == exact_story_key, "full-store opaque group mixed sources")
        local_rows.append(
            {
                "candidate": candidate.projection(),
                "candidate_id": candidate.candidate_id,
                "handle_id": handle_id,
                "local_citation_binding": local_projection,
                "opaque_group_handle": group,
            }
        )
    parsed = parse_typed_items(
        raw_items,
        operator_spec=operator_spec,
        bindings=tuple(bindings),
    )
    parsed = ParsedTypedItems(
        parsed.accepted_items,
        parsed.rejected_items,
        identity_sha256(
            {
                "closure_receipt_sha256": result.receipt.receipt_sha256,
                "format": f"{FORMAT}-parse-v1",
                "parser_receipt_sha256": parsed.parse_receipt_sha256,
                "selected_candidate_ids": [row.candidate_id for row in result.candidates],
            }
        ),
    )
    contribution = TypedEvidenceContribution(
        mechanism_id,
        tuple(bindings),
        parsed,
        closure_artifact_sha256,
        FrontierMode.BOUNDED,
        result.receipt.selection_truncated,
    )
    audit = FullStoreTypedAudit(
        contribution.receipt_sha256,
        result.receipt.receipt_sha256,
        tuple(local_rows),
        tuple(sorted(story_by_group.items())),
    )
    return contribution, audit


__all__ = [
    "FORMAT",
    "MECHANISM_ID",
    "FullStoreTypedAudit",
    "adapt_full_store_slot_closure",
]
