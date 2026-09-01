"""Gold-blind authority guard between local map evidence and specialist lanes.

Specialist proofs validate an operation inside their own selected scope.  They
do not establish that a lexical/direct-pointer scope outranks stronger local
exact citations already supporting the protected parent.  This module exposes
one provider-free guard for that seam.  It protects only question-bound numeric
parents supported by map/exact-citation evidence when every replacement handle
is weaker direct-pointer evidence.

The guard is intentionally asymmetric and conservative.  It can prove an
explicit current total, an explicit duration, or a bounded local cardinality
lower bound that contradicts a smaller replacement.  Any mixed provenance,
unit mismatch, target ambiguity, conflicting authoritative value, or merely
possible larger cardinality produces no protection proof.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256

from .contracts import assert_gold_blind, identity_sha256, require_sha256, require_text
from .typed_numeric_semantics import (
    NumericDimension,
    NumericMention,
    NumericQualifier,
    expected_numeric_dimension,
    numeric_mentions,
)
from .typed_operator_spec import normalized_terms


FORMAT = "memory-condense-cross-plane-authority-proof-v1"
RESOLUTION_FORMAT = f"{FORMAT}-resolution-v1"

_COUNT_TARGET_RE = re.compile(
    r"\bhow\s+many\s+(?P<target>.+?)\s+"
    r"(?:have|has|had|do|does|did|are|is|were|was|will|would|can|could)\b",
    re.IGNORECASE,
)
_DURATION_TARGET_RE = re.compile(
    r"\bhow\s+long\s+(?:have|has|had)\s+(?:i|we)\s+(?:been\s+)?"
    r"(?P<target>.+?)[?.!]*$",
    re.IGNORECASE,
)
_DECLARED_TOTAL_RE = re.compile(
    r"\b(?:current(?:ly)?|current[- ]count|count|total|in\s+all|altogether)\b",
    re.IGNORECASE,
)
_TARGET_META_TERMS = frozenset(
    {
        "amount",
        "ask",
        "count",
        "different",
        "first",
        "kind",
        "long",
        "many",
        "number",
        "question",
        "total",
        "type",
    }
)
_CANCELLED_STATUSES = frozenset({"cancelled", "proposed", "superseded"})
_SAFE_PROOF_KINDS = frozenset(
    {
        "numeric_operand_groups",
        "temporal_interval",
        "temporal_relative",
    }
)


class CrossPlaneAuthorityError(ValueError):
    """Raised when an authority proof or its receipt changes identity."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise CrossPlaneAuthorityError(message)


def _exact_dict(value: object) -> dict[str, Any] | None:
    return dict(value) if type(value) is dict else None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _unit(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        return None
    folded = value.strip().casefold()
    if folded in {"$", "dollar", "dollars", "usd"}:
        return "$"
    if folded in {"%", "percent", "percentage"}:
        return "%"
    if folded in {"lb", "lbs", "pound", "pounds"}:
        return "lb"
    return folded.rstrip("s")


def _question_body(dated_question: str) -> str:
    return dated_question.rsplit("]\n", 1)[-1].strip()


def _target_terms(
    dated_question: str,
    dimension: NumericDimension,
) -> tuple[str, ...]:
    body = _question_body(dated_question)
    match = (
        _COUNT_TARGET_RE.search(body)
        if dimension is NumericDimension.COUNT
        else _DURATION_TARGET_RE.search(body)
        if dimension is NumericDimension.DURATION
        else None
    )
    if match is None:
        return ()
    return tuple(
        "current" if term == "currently" else term
        for term in normalized_terms(body)
        if term not in _TARGET_META_TERMS and not term.isdigit()
    )


def _target_matches(anchors: Sequence[str], target_terms: Sequence[str]) -> bool:
    target = set(target_terms)
    overlap = target & set(anchors)
    required = 1 if len(target) == 1 else min(2, len(target))
    return len(overlap) >= required


def _one_numeric_mention(
    prediction: str,
    *,
    dated_question: str,
    dimension: NumericDimension,
) -> NumericMention | None:
    mentions = numeric_mentions(
        prediction,
        question=dated_question,
        expected_dimension=dimension,
    )
    return mentions[0] if len(mentions) == 1 else None


def _same_scalar(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-9


@dataclass(frozen=True, slots=True)
class _AuthoritativeEvidence:
    handle_id: str
    group_handle: str
    numeric_value: float
    numeric_qualifier: str
    numeric_role: str
    status: str
    unit: str | None
    summary: str
    provider_item_sha256: str
    contract_item_receipt_sha256s: tuple[str, ...]

    def projection(self) -> dict[str, Any]:
        return {
            "contract_item_receipt_sha256s": list(
                self.contract_item_receipt_sha256s
            ),
            "group_handle": self.group_handle,
            "handle_id": self.handle_id,
            "numeric_qualifier": self.numeric_qualifier,
            "numeric_role": self.numeric_role,
            "numeric_value": self.numeric_value,
            "provider_item_sha256": self.provider_item_sha256,
            "status": self.status,
            "summary_sha256": quote_sha256(self.summary),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class CrossPlaneAuthorityProtection:
    prediction: str
    basis: Literal[
        "exact_current_total",
        "explicit_duration",
        "exact_declared_total",
        "bounded_cardinality_lower_bound",
    ]
    parent_support_handle_ids: tuple[str, ...]
    replacement_handle_ids: tuple[str, ...]
    proof_json: str
    proof_receipt_sha256: str
    source_completion_sha256: str
    scope_receipt_sha256: str
    receipt_sha256: str = ""
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        require_text(self.prediction, "authority protected prediction")
        _require(
            self.basis
            in {
                "exact_current_total",
                "explicit_duration",
                "exact_declared_total",
                "bounded_cardinality_lower_bound",
            },
            "authority protection basis changed",
        )
        for values, label in (
            (self.parent_support_handle_ids, "parent"),
            (self.replacement_handle_ids, "replacement"),
        ):
            _require(
                bool(values)
                and len(values) == len(set(values))
                and all(type(value) is str and value for value in values),
                f"authority {label} handles changed",
            )
        _require(
            not set(self.parent_support_handle_ids)
            & set(self.replacement_handle_ids),
            "authority planes overlap",
        )
        for value in (
            self.proof_receipt_sha256,
            self.source_completion_sha256,
            self.scope_receipt_sha256,
        ):
            require_sha256(value, "authority receipt")
        proof = self.proof
        _require(
            self.proof_json == _canonical_json(proof)
            and self.proof_receipt_sha256 == identity_sha256(proof),
            "authority proof changed",
        )
        _require(
            self.provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0,
            "authority proof escaped zero-call zero-state boundary",
        )
        computed = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == computed, "authority resolution changed")
        object.__setattr__(self, "receipt_sha256", computed)
        assert_gold_blind(self.projection(), path="cross_plane_authority_resolution")

    @property
    def proof(self) -> dict[str, Any]:
        try:
            value = json.loads(self.proof_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CrossPlaneAuthorityError("authority proof changed encoding") from exc
        _require(type(value) is dict, "authority proof changed type")
        return dict(value)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "basis": self.basis,
            "format": RESOLUTION_FORMAT,
            "parent_support_handle_ids": list(self.parent_support_handle_ids),
            "prediction": self.prediction,
            "prediction_sha256": quote_sha256(self.prediction),
            "proof": self.proof,
            "proof_receipt_sha256": self.proof_receipt_sha256,
            "provider_calls": 0,
            "replacement_handle_ids": list(self.replacement_handle_ids),
            "retained_transformer_token_state_bytes": 0,
            "scope_receipt_sha256": self.scope_receipt_sha256,
            "source_completion_sha256": self.source_completion_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _authoritative_evidence(
    *,
    provider_input: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
    allowed: tuple[str, ...],
    target_terms: tuple[str, ...],
    dimension: NumericDimension,
) -> tuple[_AuthoritativeEvidence, ...] | None:
    typed = _exact_dict(provider_input.get("typed_evidence"))
    by_handle = _exact_dict(validation_contract.get("by_handle"))
    if typed is None or by_handle is None:
        return None
    inventory = typed.get("handles")
    items = typed.get("items")
    if type(inventory) is not list or type(items) is not list:
        return None
    exact_map: dict[str, dict[str, Any]] = {}
    for raw in inventory:
        row = _exact_dict(raw)
        if row is None:
            return None
        handle = row.get("handle_id")
        if (
            handle in allowed
            and row.get("origin") == "map"
            and row.get("provenance_grade") == "exact_citation"
            and type(row.get("group_handle")) is str
            and row["group_handle"]
        ):
            exact_map[str(handle)] = row

    output: list[_AuthoritativeEvidence] = []
    seen_handles: set[str] = set()
    for raw in items:
        item = _exact_dict(raw)
        if item is None:
            return None
        handles = item.get("handle_ids")
        if (
            type(handles) is not list
            or len(handles) != 1
            or handles[0] not in exact_map
        ):
            continue
        handle = str(handles[0])
        if handle in seen_handles:
            return None
        seen_handles.add(handle)
        contract = _exact_dict(by_handle.get(handle))
        anchors = None if contract is None else contract.get("answer_anchor_terms")
        receipts = (
            None if contract is None else contract.get("usable_item_receipt_sha256s")
        )
        semantic = None if contract is None else contract.get("semantic_rows")
        numeric = item.get("numeric_value")
        if (
            contract is None
            or type(anchors) is not list
            or type(receipts) is not list
            or not receipts
            or type(semantic) is not list
            or type(numeric) not in {int, float}
            or not math.isfinite(float(numeric))
            or item.get("included") is not True
            or item.get("content_coherence") != "match"
            or item.get("value_authority") != "explicit"
            or item.get("status") in _CANCELLED_STATUSES
            or "authored_by_assistant" in str(item.get("relation", ""))
            or any(type(value) is not str for value in anchors)
            or any(type(value) is not str for value in receipts)
        ):
            continue
        item_unit = _unit(item.get("unit"))
        if (
            dimension is NumericDimension.COUNT
            and item_unit is not None
        ) or (
            dimension is NumericDimension.DURATION
            and item_unit
            not in {"second", "minute", "hour", "day", "week", "month", "year"}
        ):
            continue
        matching_semantic = tuple(
            row
            for raw_row in semantic
            if (row := _exact_dict(raw_row)) is not None
            and type(row.get("numeric_value")) in {int, float}
            and _same_scalar(float(row["numeric_value"]), float(numeric))
            and _unit(row.get("unit")) == item_unit
            and row.get("item_receipt_sha256") in receipts
        )
        if len(matching_semantic) != 1:
            continue
        semantic_anchors = tuple(
            dict.fromkeys(
                str(term)
                for row in matching_semantic
                for key in (
                    "action_concepts",
                    "entity_terms",
                    "relation_terms",
                    "summary_terms",
                )
                for term in row.get(key, [])
                if type(term) is str
            )
        )
        if not _target_matches((*anchors, *semantic_anchors), target_terms):
            continue
        output.append(
            _AuthoritativeEvidence(
                handle_id=handle,
                group_handle=str(exact_map[handle]["group_handle"]),
                numeric_value=float(numeric),
                numeric_qualifier=str(item.get("numeric_qualifier", "exact")),
                numeric_role=str(item.get("numeric_role", "none")),
                status=str(item.get("status", "unknown")),
                unit=item_unit,
                summary=str(item.get("summary", "")),
                provider_item_sha256=identity_sha256(item),
                contract_item_receipt_sha256s=tuple(receipts),
            )
        )
    return tuple(output)


def _weaker_replacement_evidence(
    *,
    provider_input: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
    allowed: tuple[str, ...],
    replacement_handles: tuple[str, ...],
) -> tuple[dict[str, Any], ...] | None:
    typed = _exact_dict(provider_input.get("typed_evidence"))
    by_handle = _exact_dict(validation_contract.get("by_handle"))
    if typed is None or by_handle is None:
        return None
    inventory = typed.get("handles")
    if type(inventory) is not list:
        return None
    rows = {
        str(row["handle_id"]): row
        for raw in inventory
        if (row := _exact_dict(raw)) is not None
        and type(row.get("handle_id")) is str
    }
    output: list[dict[str, Any]] = []
    for handle in replacement_handles:
        row = rows.get(handle)
        contract = _exact_dict(by_handle.get(handle))
        receipts = (
            None if contract is None else contract.get("usable_item_receipt_sha256s")
        )
        if (
            handle not in allowed
            or row is None
            or row.get("origin") != "direct_pointer"
            or row.get("provenance_grade") != "direct_pointer"
            or type(row.get("group_handle")) is not str
            or contract is None
            or type(receipts) is not list
            or not receipts
            or any(type(value) is not str for value in receipts)
        ):
            return None
        output.append(
            {
                "contract_item_receipt_sha256s": list(receipts),
                "group_handle": row["group_handle"],
                "handle_id": handle,
                "inventory_row_sha256": identity_sha256(row),
                "origin": "direct_pointer",
                "provenance_grade": "direct_pointer",
            }
        )
    return tuple(output)


def _select_authority_basis(
    *,
    evidence: tuple[_AuthoritativeEvidence, ...],
    parent: NumericMention,
    replacement: NumericMention,
    dimension: NumericDimension,
) -> tuple[str, tuple[_AuthoritativeEvidence, ...], float | None] | None:
    current = tuple(
        row
        for row in evidence
        if row.status == "current" or row.numeric_role == "end"
    )
    if current:
        values = {(row.numeric_value, row.unit) for row in current}
        if len(values) != 1:
            return None
        value, unit = next(iter(values))
        if (
            _same_scalar(value, parent.value)
            and unit == parent.unit
            and not _same_scalar(replacement.value, parent.value)
        ):
            return "exact_current_total", current, None
        return None

    if dimension is NumericDimension.DURATION:
        values = {(row.numeric_value, row.unit) for row in evidence}
        if len(values) != 1:
            return None
        value, unit = next(iter(values))
        if (
            _same_scalar(value, parent.value)
            and unit == parent.unit
            and parent.qualifier
            in {NumericQualifier.EXACT, NumericQualifier.APPROXIMATE}
            and not _same_scalar(replacement.value, parent.value)
        ):
            return "explicit_duration", evidence, None
        return None

    declared = tuple(row for row in evidence if _DECLARED_TOTAL_RE.search(row.summary))
    if declared:
        values = {(row.numeric_value, row.unit) for row in declared}
        if len(values) != 1:
            return None
        value, unit = next(iter(values))
        if (
            _same_scalar(value, parent.value)
            and unit == parent.unit
            and not _same_scalar(replacement.value, parent.value)
        ):
            return "exact_declared_total", declared, None
        return None

    operands = tuple(
        row
        for row in evidence
        if row.numeric_role == "operand"
        and row.numeric_qualifier == "exact"
        and row.unit is None
    )
    operand_receipts = tuple(
        receipt
        for row in operands
        for receipt in row.contract_item_receipt_sha256s
    )
    if (
        not operands
        or len({row.provider_item_sha256 for row in operands}) != len(operands)
        or len(operand_receipts) != len(set(operand_receipts))
    ):
        return None
    lower_bound = sum(row.numeric_value for row in operands)
    parent_supported = (
        parent.qualifier is NumericQualifier.EXACT
        and _same_scalar(parent.value, lower_bound)
    ) or (
        parent.qualifier is NumericQualifier.LOWER_BOUND
        and parent.value <= lower_bound + 1e-9
    )
    if parent_supported and replacement.value < lower_bound - 1e-9:
        return "bounded_cardinality_lower_bound", operands, lower_bound
    return None


def protect_parent_from_cross_plane_authority(
    *,
    dated_question: str,
    parent_prediction: str,
    replacement_prediction: str,
    replacement_used_handle_ids: Sequence[str],
    replacement_proof_kind: str,
    provider_input: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
    allowed_handle_ids: Sequence[str],
    answer_plan_receipt_sha256: str,
    base_scope_receipt_sha256: str,
    source_completion_sha256: str,
) -> CrossPlaneAuthorityProtection | None:
    """Protect a stronger local parent, or abstain without changing the answer."""

    require_text(dated_question, "authority dated question")
    require_text(parent_prediction, "authority parent prediction")
    require_text(replacement_prediction, "authority replacement prediction")
    require_text(replacement_proof_kind, "authority replacement proof")
    for value in (
        answer_plan_receipt_sha256,
        base_scope_receipt_sha256,
        source_completion_sha256,
    ):
        require_sha256(value, "authority binding")
    allowed = tuple(allowed_handle_ids)
    replacement_handles = tuple(replacement_used_handle_ids)
    protected = _exact_dict(provider_input.get("protected_parent_fallback"))
    if (
        not allowed
        or len(allowed) != len(set(allowed))
        or any(type(value) is not str or not value for value in allowed)
        or not replacement_handles
        or len(replacement_handles) != len(set(replacement_handles))
        or any(type(value) is not str or not value for value in replacement_handles)
        or replacement_proof_kind not in _SAFE_PROOF_KINDS
        or provider_input.get("dated_question") != dated_question
        or protected is None
        or protected.get("prediction") != parent_prediction
        or protected.get("prediction_sha256") != quote_sha256(parent_prediction)
    ):
        return None
    dimension = expected_numeric_dimension(question=dated_question)
    if dimension not in {NumericDimension.COUNT, NumericDimension.DURATION}:
        return None
    target_terms = _target_terms(dated_question, dimension)
    parent = _one_numeric_mention(
        parent_prediction,
        dated_question=dated_question,
        dimension=dimension,
    )
    replacement = _one_numeric_mention(
        replacement_prediction,
        dated_question=dated_question,
        dimension=dimension,
    )
    if not target_terms or parent is None or replacement is None:
        return None
    weaker = _weaker_replacement_evidence(
        provider_input=provider_input,
        validation_contract=validation_contract,
        allowed=allowed,
        replacement_handles=replacement_handles,
    )
    authoritative = _authoritative_evidence(
        provider_input=provider_input,
        validation_contract=validation_contract,
        allowed=allowed,
        target_terms=target_terms,
        dimension=dimension,
    )
    if weaker is None or authoritative is None or not authoritative:
        return None
    selected = _select_authority_basis(
        evidence=authoritative,
        parent=parent,
        replacement=replacement,
        dimension=dimension,
    )
    if selected is None:
        return None
    basis, support, lower_bound = selected
    support_handles = tuple(row.handle_id for row in support)
    if set(support_handles) & set(replacement_handles):
        return None
    typed = _exact_dict(provider_input.get("typed_evidence"))
    assert typed is not None
    frontier = _exact_dict(typed.get("frontier"))
    proof_body = {
        "allowed_handle_ids_sha256": identity_sha256(list(allowed)),
        "answer_plan_receipt_sha256": answer_plan_receipt_sha256,
        "basis": basis,
        "base_scope_receipt_sha256": base_scope_receipt_sha256,
        "expected_numeric_dimension": dimension.value,
        "format": FORMAT,
        "frontier": None
        if frontier is None
        else {
            "closed": frontier.get("closed"),
            "mode": frontier.get("mode"),
            "truncated": frontier.get("truncated"),
        },
        "lower_bound": lower_bound,
        "parent_evidence": [row.projection() for row in support],
        "parent_numeric": {
            "qualifier": parent.qualifier.value,
            "unit": parent.unit,
            "value": parent.value,
        },
        "parent_prediction_sha256": quote_sha256(parent_prediction),
        "provider_calls": 0,
        "provider_input_sha256": identity_sha256(provider_input),
        "question_sha256": quote_sha256(dated_question),
        "replacement_evidence": list(weaker),
        "replacement_numeric": {
            "qualifier": replacement.qualifier.value,
            "unit": replacement.unit,
            "value": replacement.value,
        },
        "replacement_prediction_sha256": quote_sha256(replacement_prediction),
        "replacement_proof_kind": replacement_proof_kind,
        "retained_transformer_token_state_bytes": 0,
        "source_completion_sha256": source_completion_sha256,
        "target_terms": list(target_terms),
        "validation_contract_sha256": identity_sha256(validation_contract),
    }
    assert_gold_blind(proof_body, path="cross_plane_authority_proof")
    proof_receipt = identity_sha256(proof_body)
    return CrossPlaneAuthorityProtection(
        prediction=parent_prediction,
        basis=basis,  # type: ignore[arg-type]
        parent_support_handle_ids=support_handles,
        replacement_handle_ids=replacement_handles,
        proof_json=_canonical_json(proof_body),
        proof_receipt_sha256=proof_receipt,
        source_completion_sha256=source_completion_sha256,
        scope_receipt_sha256=base_scope_receipt_sha256,
    )


__all__ = [
    "FORMAT",
    "RESOLUTION_FORMAT",
    "CrossPlaneAuthorityError",
    "CrossPlaneAuthorityProtection",
    "protect_parent_from_cross_plane_authority",
]
