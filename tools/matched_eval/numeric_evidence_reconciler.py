"""Provider-free numeric reconciliation over sealed typed provider input.

This module consumes the same compact, gold-blind provider input that would be
sent to the typed final answer model.  It does not reopen local stores, inspect
benchmark identities, call a provider, or integrate with the answer wrapper.
Its only output is an inspectable proof receipt for deterministic numeric
relationships that are already explicit in the sealed evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_memory_final_arm import PROMPT_ROW_FORMAT
from .typed_numeric_semantics import NumericQualifier
from .typed_operator_adapter import (
    COMPACT_FINAL_PROVIDER_FORMAT,
    ContentCoherence,
    EvidenceOrigin,
    EvidenceStatus,
    NumericRole,
    ProvenanceGrade,
    TypedItemKind,
    ValueAuthority,
)
from .typed_operator_spec import (
    AnswerShape,
    ComparisonMode,
    SlotKind,
    TemporalMode,
    normalized_terms,
)


FORMAT = "memory-condense-sealed-numeric-evidence-reconciliation-v1"
CONTRIBUTION_FORMAT = f"{FORMAT}-contribution-v1"
REJECTION_FORMAT = f"{FORMAT}-rejection-v1"
POLICY_ID = "sealed_typed_numeric_reconciliation_v1"


class NumericEvidenceReconcilerError(MatchedEvalContractError):
    """The sealed input or reconciliation proof violated its contract."""


class _ContributionConflict(ValueError):
    """Two sealed rows claimed the same typed fact with incompatible values."""


class ReconciliationStatus(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    CONFLICTED = "conflicted"


class ReconciliationMode(str, Enum):
    NONE = "none"
    DIRECT_TOTAL = "direct_current_or_end_total"
    CARDINALITY_SUM = "exact_cardinality_sum"
    RECURRING_PLUS_ADDITIONS = "recurring_base_plus_distinct_additions"
    COMPARISON = "sealed_two_sided_comparison"


class UnitFamily(str, Enum):
    ITEM = "item"
    EVENT = "event"
    RECURRING = "recurring_frequency"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DURATION = "duration"
    MEASURE = "measure"
    UNKNOWN = "unknown"


_HANDLE_RE = re.compile(r"^H[0-9]{3,6}$")
_GROUP_RE = re.compile(r"^G[0-9]{3,6}$")
_FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "ordinal",
        "ordinals",
        "question_id",
        "question_ids",
        "reference",
        "reference_answer",
        "gold",
        "gold_answer",
    }
)
_AMBIGUOUS_TEMPORAL_RE = re.compile(
    r"\b(?:ambiguous|potential(?:ly)?|unspecified|unclear|unknown)\b"
    r".{0,32}\b(?:eligible|eligibility|timing|window)\b|"
    r"\b(?:eligible|eligibility|timing|window)\b"
    r".{0,32}\b(?:ambiguous|potential(?:ly)?|unspecified|unclear|unknown)\b",
    re.I,
)
_EXPLICIT_TEMPORAL_RE = re.compile(
    r"\btemporally\s+eligible\b|\bwithin\s+the\s+(?:requested\s+)?"
    r"(?:time\s+)?window\b",
    re.I,
)
_RELATIVE_TEMPORAL_RE = re.compile(
    r"\b(?:since|within|during\s+the\s+(?:last|past)|in\s+the\s+(?:last|past)|"
    r"before|after)\b",
    re.I,
)
_QUESTION_TIMESTAMP_RE = re.compile(
    r"\[Question asked at (?P<value>[^\]]+)\]", re.I
)
_COUNT_NOUN_RE = re.compile(
    r"\bhow\s+many(?:\s+total)?\s+(?P<noun>[a-z][a-z-]*)\b", re.I
)
_RELATION_PART_RE = re.compile(
    r"^\s*(?P<key>[a-z][a-z0-9_-]*)(?:\s*=\s*(?P<value>[^;]+))?\s*$",
    re.I,
)

_EVENT_NOUNS = frozenset(
    {
        "appointment",
        "appointments",
        "event",
        "events",
        "meeting",
        "meetings",
        "occasion",
        "occasions",
        "occurrence",
        "occurrences",
        "race",
        "races",
        "run",
        "runs",
        "session",
        "sessions",
        "time",
        "times",
        "trip",
        "trips",
        "visit",
        "visits",
        "workout",
        "workouts",
    }
)
_DURATION_NOUNS = frozenset(
    {
        "day",
        "days",
        "hour",
        "hours",
        "minute",
        "minutes",
        "month",
        "months",
        "second",
        "seconds",
        "week",
        "weeks",
        "year",
        "years",
    }
)
_MEASURE_UNITS = frozenset(
    {
        "foot",
        "feet",
        "g",
        "gram",
        "grams",
        "inch",
        "inches",
        "kg",
        "kilogram",
        "kilograms",
        "km",
        "lb",
        "lbs",
        "meter",
        "meters",
        "mile",
        "miles",
        "oz",
        "ounce",
        "ounces",
        "pound",
        "pounds",
    }
)
_UNIT_ALIASES = {
    "children": "child",
    "feet": "foot",
    "items": "item",
    "people": "person",
    "persons": "person",
    "pieces": "piece",
    "stories": "story",
    "twins": "twin",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NumericEvidenceReconcilerError(message)


def _forbid_identity_fields(value: object, path: str = "provider_input") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold()
            if (
                normalized in _FORBIDDEN_IDENTITY_KEYS
                or normalized.endswith("_ordinal")
                or normalized in {"questionid", "benchmark_ordinal"}
            ):
                raise NumericEvidenceReconcilerError(
                    f"numeric reconciler forbids benchmark identity field {path}.{key}"
                )
            _forbid_identity_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _forbid_identity_fields(child, f"{path}[{index}]")


def _exact_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise NumericEvidenceReconcilerError(f"{label} must be non-empty exact text")
    return value


def _exact_list(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise NumericEvidenceReconcilerError(f"{label} must be an exact list")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _exact_text(value, label)


def _ordered_unique_texts(value: object, label: str) -> tuple[str, ...]:
    rows = _exact_list(value, label)
    if any(type(row) is not str or not row for row in rows):
        raise NumericEvidenceReconcilerError(f"{label} contains invalid text")
    result = tuple(rows)
    if len(set(result)) != len(result):
        raise NumericEvidenceReconcilerError(f"{label} must be ordered and unique")
    return result


def _finite_number(value: object, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise NumericEvidenceReconcilerError(f"{label} must be a finite number")
    return float(value)


def _normalize_unit_token(value: str) -> str:
    token = re.sub(r"[\s-]+", "_", value.casefold().strip())
    token = token.strip("_.,")
    return _UNIT_ALIASES.get(token, token[:-1] if token.endswith("s") else token)


@dataclass(frozen=True, slots=True)
class _UnitSemantics:
    family: UnitFamily
    canonical: str | None
    numerator_family: UnitFamily | None = None
    period: str | None = None


def _unit_semantics(unit: str | None) -> _UnitSemantics:
    if unit is None:
        return _UnitSemantics(UnitFamily.UNKNOWN, None)
    folded = unit.casefold().strip()
    recurring = re.fullmatch(
        r"(?P<numerator>[a-z][a-z _-]*?)(?:\s*/\s*|\s+per\s+|_per_)"
        r"(?P<period>day|week|month|year)s?",
        folded,
    )
    if recurring is not None:
        numerator_raw = recurring.group("numerator")
        numerator = _unit_semantics(numerator_raw)
        if numerator.family not in {UnitFamily.ITEM, UnitFamily.EVENT}:
            return _UnitSemantics(UnitFamily.UNKNOWN, _normalize_unit_token(folded))
        period = recurring.group("period")
        return _UnitSemantics(
            UnitFamily.RECURRING,
            f"{numerator.canonical}/{period}",
            numerator.family,
            period,
        )
    token = _normalize_unit_token(folded)
    if token in {"$", "usd", "dollar"}:
        return _UnitSemantics(UnitFamily.CURRENCY, "$")
    if token in {"%", "percent", "percentage"}:
        return _UnitSemantics(UnitFamily.PERCENTAGE, "%")
    if token in {_normalize_unit_token(row) for row in _DURATION_NOUNS}:
        return _UnitSemantics(UnitFamily.DURATION, token)
    if token in {_normalize_unit_token(row) for row in _MEASURE_UNITS}:
        return _UnitSemantics(UnitFamily.MEASURE, token)
    if folded in _EVENT_NOUNS or token in {
        _normalize_unit_token(row) for row in _EVENT_NOUNS
    }:
        return _UnitSemantics(UnitFamily.EVENT, token)
    # A sealed non-measure/count noun is an item unit.  This admits exact
    # plural quantities such as children/twins while retaining a hard
    # item-versus-event boundary.
    if re.fullmatch(r"[a-z][a-z0-9_]*", token):
        return _UnitSemantics(UnitFamily.ITEM, token)
    return _UnitSemantics(UnitFamily.UNKNOWN, token or None)


@dataclass(frozen=True, slots=True)
class _Handle:
    handle_id: str
    group_handle: str
    origin: EvidenceOrigin
    provenance_grade: ProvenanceGrade


@dataclass(frozen=True, slots=True)
class _Slot:
    slot_id: str
    kind: SlotKind
    label: str
    match_terms: tuple[str, ...]
    requires_numeric: bool


@dataclass(frozen=True, slots=True)
class _Operator:
    answer_shape: AnswerShape
    comparison_mode: ComparisonMode
    temporal_mode: TemporalMode
    operation: str
    cardinality: int | None
    include_proposed: bool
    query_timestamp: str | None
    temporal_window_days: int | None
    slots: tuple[_Slot, ...]
    receipt_sha256: str

    @property
    def numeric_slot_ids(self) -> frozenset[str]:
        return frozenset(row.slot_id for row in self.slots if row.requires_numeric)


@dataclass(frozen=True, slots=True)
class _Item:
    projection_sha256: str
    handle_ids: tuple[str, ...]
    group_handles: tuple[str, ...]
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
    value_authority: ValueAuthority
    included: bool
    supported_slot_ids: tuple[str, ...]
    content_coherence: ContentCoherence


@dataclass(frozen=True, slots=True)
class NumericContributionProof:
    semantic_key_sha256: str
    numeric_role: str
    numeric_value: float
    unit: str | None
    unit_family: str
    handle_ids: tuple[str, ...]
    supported_slot_ids: tuple[str, ...]
    item_projection_sha256s: tuple[str, ...]
    temporal_bases: tuple[str, ...]
    corroborated_duplicate_count: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.semantic_key_sha256, "numeric semantic key")
        require_text(self.numeric_role, "numeric contribution role")
        _finite_number(self.numeric_value, "numeric contribution value")
        if self.unit is not None:
            require_text(self.unit, "numeric contribution unit")
        UnitFamily(self.unit_family)
        if (
            type(self.handle_ids) is not tuple
            or not self.handle_ids
            or len(set(self.handle_ids)) != len(self.handle_ids)
            or any(_HANDLE_RE.fullmatch(row) is None for row in self.handle_ids)
        ):
            raise NumericEvidenceReconcilerError(
                "numeric contribution handles must be opaque and unique"
            )
        for values, label in (
            (self.supported_slot_ids, "numeric contribution slots"),
            (self.item_projection_sha256s, "numeric contribution items"),
            (self.temporal_bases, "numeric contribution temporal bases"),
        ):
            if type(values) is not tuple or len(set(values)) != len(values):
                raise NumericEvidenceReconcilerError(f"{label} must be ordered unique")
        for digest in self.item_projection_sha256s:
            require_sha256(digest, "numeric contribution item")
        if (
            type(self.corroborated_duplicate_count) is not int
            or self.corroborated_duplicate_count < 0
            or self.corroborated_duplicate_count
            != len(self.item_projection_sha256s) - 1
        ):
            raise NumericEvidenceReconcilerError(
                "numeric contribution corroboration accounting changed"
            )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise NumericEvidenceReconcilerError("numeric contribution receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="numeric_contribution")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "corroborated_duplicate_count": self.corroborated_duplicate_count,
            "format": CONTRIBUTION_FORMAT,
            "handle_ids": list(self.handle_ids),
            "item_projection_sha256s": list(self.item_projection_sha256s),
            "numeric_role": self.numeric_role,
            "numeric_value": self.numeric_value,
            "semantic_key_sha256": self.semantic_key_sha256,
            "supported_slot_ids": list(self.supported_slot_ids),
            "temporal_bases": list(self.temporal_bases),
            "unit": self.unit,
            "unit_family": self.unit_family,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class RejectedNumericOperandProof:
    item_projection_sha256: str
    handle_ids: tuple[str, ...]
    reason: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.item_projection_sha256, "rejected numeric item")
        if (
            type(self.handle_ids) is not tuple
            or len(set(self.handle_ids)) != len(self.handle_ids)
            or any(_HANDLE_RE.fullmatch(row) is None for row in self.handle_ids)
        ):
            raise NumericEvidenceReconcilerError(
                "rejected numeric handles must be opaque and unique"
            )
        require_text(self.reason, "numeric rejection reason")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise NumericEvidenceReconcilerError("numeric rejection receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="numeric_rejection")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": REJECTION_FORMAT,
            "handle_ids": list(self.handle_ids),
            "item_projection_sha256": self.item_projection_sha256,
            "reason": self.reason,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class NumericEvidenceReconciliationReceipt:
    sealed_provider_input_sha256: str
    typed_evidence_sha256: str
    operator_spec_sha256: str
    status: ReconciliationStatus
    mode: ReconciliationMode
    reason: str
    numeric_result: float | None
    unit: str | None
    comparison_relation: str | None
    boolean_result: bool | None
    contributions: tuple[NumericContributionProof, ...]
    rejected_operands: tuple[RejectedNumericOperandProof, ...]
    used_handle_ids: tuple[str, ...]
    numeric_candidate_count: int
    ignored_non_numeric_count: int
    deduplicated_item_count: int
    provider_prompt_count: int = 0
    retained_transformer_token_state_bytes: int = 0
    gold_loaded: bool = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.sealed_provider_input_sha256, "sealed numeric provider input"),
            (self.typed_evidence_sha256, "sealed numeric typed evidence"),
            (self.operator_spec_sha256, "sealed numeric operator spec"),
        ):
            require_sha256(value, label)
        if type(self.status) is not ReconciliationStatus:
            raise NumericEvidenceReconcilerError("numeric status must be canonical")
        if type(self.mode) is not ReconciliationMode:
            raise NumericEvidenceReconcilerError("numeric mode must be canonical")
        require_text(self.reason, "numeric reconciliation reason")
        if self.numeric_result is not None:
            _finite_number(self.numeric_result, "numeric reconciliation result")
        if self.unit is not None:
            require_text(self.unit, "numeric reconciliation unit")
        if self.comparison_relation not in {
            None,
            "left_greater",
            "equal",
            "left_less",
        }:
            raise NumericEvidenceReconcilerError("comparison relation is invalid")
        if self.boolean_result is not None and type(self.boolean_result) is not bool:
            raise NumericEvidenceReconcilerError("numeric boolean result is invalid")
        if type(self.contributions) is not tuple or any(
            type(row) is not NumericContributionProof for row in self.contributions
        ):
            raise NumericEvidenceReconcilerError("numeric contributions changed type")
        if type(self.rejected_operands) is not tuple or any(
            type(row) is not RejectedNumericOperandProof
            for row in self.rejected_operands
        ):
            raise NumericEvidenceReconcilerError("numeric rejections changed type")
        if (
            type(self.used_handle_ids) is not tuple
            or len(set(self.used_handle_ids)) != len(self.used_handle_ids)
            or any(_HANDLE_RE.fullmatch(row) is None for row in self.used_handle_ids)
        ):
            raise NumericEvidenceReconcilerError("numeric used handles changed")
        expected_handles = tuple(
            dict.fromkeys(
                handle for row in self.contributions for handle in row.handle_ids
            )
        )
        if self.used_handle_ids != expected_handles:
            raise NumericEvidenceReconcilerError(
                "numeric used handles do not reconcile to contributions"
            )
        for value, label in (
            (self.numeric_candidate_count, "numeric candidate count"),
            (self.ignored_non_numeric_count, "ignored non-numeric count"),
            (self.deduplicated_item_count, "deduplicated numeric item count"),
        ):
            if type(value) is not int or value < 0:
                raise NumericEvidenceReconcilerError(f"{label} is invalid")
        expected_dedup = sum(
            row.corroborated_duplicate_count for row in self.contributions
        )
        if self.deduplicated_item_count != expected_dedup:
            raise NumericEvidenceReconcilerError(
                "numeric deduplication accounting changed"
            )
        if self.provider_prompt_count != 0:
            raise NumericEvidenceReconcilerError("numeric reconciler called a provider")
        if self.retained_transformer_token_state_bytes != 0:
            raise NumericEvidenceReconcilerError(
                "numeric reconciler retained transformer token state"
            )
        if self.gold_loaded is not False:
            raise NumericEvidenceReconcilerError("numeric reconciler loaded gold")
        if self.status is ReconciliationStatus.SUPPORTED:
            if self.numeric_result is None or not self.contributions:
                raise NumericEvidenceReconcilerError(
                    "supported numeric receipt lost its proof"
                )
        elif self.numeric_result is not None or self.contributions:
            raise NumericEvidenceReconcilerError(
                "unsupported numeric receipt carried a result or used proof"
            )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise NumericEvidenceReconcilerError("numeric reconciliation receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        projection = self.projection()
        _forbid_identity_fields(projection, path="numeric_reconciliation")
        assert_gold_blind(projection, path="numeric_reconciliation")

    @property
    def supported(self) -> bool:
        return self.status is ReconciliationStatus.SUPPORTED

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "boolean_result": self.boolean_result,
            "comparison_relation": self.comparison_relation,
            "contributions": [row.projection() for row in self.contributions],
            "deduplicated_item_count": self.deduplicated_item_count,
            "format": FORMAT,
            "gold_loaded": False,
            "ignored_non_numeric_count": self.ignored_non_numeric_count,
            "mode": self.mode.value,
            "numeric_candidate_count": self.numeric_candidate_count,
            "numeric_result": self.numeric_result,
            "operator_spec_sha256": self.operator_spec_sha256,
            "policy_id": POLICY_ID,
            "provider_prompt_count": 0,
            "reason": self.reason,
            "rejected_operands": [row.projection() for row in self.rejected_operands],
            "retained_transformer_token_state_bytes": 0,
            "sealed_provider_input_sha256": self.sealed_provider_input_sha256,
            "status": self.status.value,
            "typed_evidence_sha256": self.typed_evidence_sha256,
            "unit": self.unit,
            "used_handle_ids": list(self.used_handle_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _parse_handles(raw: object) -> tuple[_Handle, ...]:
    rows = _exact_list(raw, "typed evidence handles")
    output: list[_Handle] = []
    for row in rows:
        if type(row) is not dict or set(row) != {
            "group_handle",
            "handle_id",
            "origin",
            "provenance_grade",
        }:
            raise NumericEvidenceReconcilerError(
                "typed evidence handle schema changed"
            )
        handle = _exact_text(row["handle_id"], "typed evidence handle")
        group = _exact_text(row["group_handle"], "typed evidence group handle")
        if _HANDLE_RE.fullmatch(handle) is None or _GROUP_RE.fullmatch(group) is None:
            raise NumericEvidenceReconcilerError("typed evidence handle is not opaque")
        try:
            origin = EvidenceOrigin(row["origin"])
            provenance = ProvenanceGrade(row["provenance_grade"])
        except ValueError as exc:
            raise NumericEvidenceReconcilerError(
                "typed evidence handle enum changed"
            ) from exc
        output.append(_Handle(handle, group, origin, provenance))
    if len({row.handle_id for row in output}) != len(output):
        raise NumericEvidenceReconcilerError("typed evidence handles are repeated")
    return tuple(output)


def _parse_slots(raw: object) -> tuple[_Slot, ...]:
    rows = _exact_list(raw, "numeric operator slots")
    output: list[_Slot] = []
    required = {
        "kind",
        "label",
        "match_terms",
        "minimum_match_term_count",
        "relation_constraint",
        "requires_numeric",
        "slot_id",
    }
    for row in rows:
        if type(row) is not dict or set(row) != required:
            raise NumericEvidenceReconcilerError("numeric operator slot schema changed")
        slot_id = _exact_text(row["slot_id"], "numeric slot ID")
        label = _exact_text(row["label"], "numeric slot label")
        terms = _ordered_unique_texts(row["match_terms"], "numeric slot terms")
        if type(row["requires_numeric"]) is not bool:
            raise NumericEvidenceReconcilerError("numeric slot flag changed type")
        try:
            kind = SlotKind(row["kind"])
        except ValueError as exc:
            raise NumericEvidenceReconcilerError("numeric slot kind changed") from exc
        output.append(
            _Slot(slot_id, kind, label, terms, row["requires_numeric"])
        )
    if len({row.slot_id for row in output}) != len(output):
        raise NumericEvidenceReconcilerError("numeric operator slots are repeated")
    return tuple(output)


def _parse_operator(raw: object) -> _Operator:
    if type(raw) is not dict:
        raise NumericEvidenceReconcilerError("typed numeric operator must be an object")
    required = {
        "absence_decision_requires_closed_frontier",
        "answer_shape",
        "cardinality",
        "comparison_mode",
        "include_proposed",
        "operation",
        "ordering",
        "personalization_required",
        "query_timestamp",
        "required_evidence_role",
        "required_slots",
        "requires_all_slots",
        "requires_complete_frontier",
        "specificity_required",
        "style",
        "temporal_mode",
        "temporal_window_days",
    }
    if set(raw) != required:
        raise NumericEvidenceReconcilerError("typed numeric operator schema changed")
    try:
        answer_shape = AnswerShape(raw["answer_shape"])
        comparison = ComparisonMode(raw["comparison_mode"])
        temporal = TemporalMode(raw["temporal_mode"])
    except ValueError as exc:
        raise NumericEvidenceReconcilerError("typed numeric operator enum changed") from exc
    operation = _exact_text(raw["operation"], "typed numeric operation")
    cardinality = raw["cardinality"]
    if cardinality is not None and (
        type(cardinality) is not int or cardinality < 1
    ):
        raise NumericEvidenceReconcilerError("typed numeric cardinality is invalid")
    include_proposed = raw["include_proposed"]
    if type(include_proposed) is not bool:
        raise NumericEvidenceReconcilerError("typed numeric include-proposed changed")
    query_timestamp = _optional_text(
        raw["query_timestamp"], "typed numeric query timestamp"
    )
    window = raw["temporal_window_days"]
    if window is not None and (type(window) is not int or window < 1):
        raise NumericEvidenceReconcilerError("typed numeric temporal window is invalid")
    slots = _parse_slots(raw["required_slots"])
    return _Operator(
        answer_shape,
        comparison,
        temporal,
        operation,
        cardinality,
        include_proposed,
        query_timestamp,
        window,
        slots,
        identity_sha256(raw),
    )


def _parse_items(
    raw: object,
    *,
    handles: Sequence[_Handle],
    operator: _Operator,
) -> tuple[_Item, ...]:
    rows = _exact_list(raw, "typed numeric items")
    handle_map = {row.handle_id: row for row in handles}
    known_slots = {row.slot_id for row in operator.slots}
    required = {
        "content_coherence",
        "handle_ids",
        "included",
        "kind",
        "status",
        "summary",
        "supported_slot_ids",
        "value_authority",
    }
    optional = {
        "date",
        "entity_key",
        "group_key",
        "numeric_role",
        "numeric_qualifier",
        "numeric_value",
        "participant_count",
        "personalization_anchors",
        "relation",
        "specificity_terms",
        "unit",
    }
    output: list[_Item] = []
    for row in rows:
        if type(row) is not dict or not required <= set(row) or not set(row) <= required | optional:
            raise NumericEvidenceReconcilerError("typed numeric item schema changed")
        item_handles = _ordered_unique_texts(
            row["handle_ids"], "typed numeric item handles"
        )
        if not item_handles or not set(item_handles) <= set(handle_map):
            raise NumericEvidenceReconcilerError("typed numeric item cites unknown handles")
        slots = _ordered_unique_texts(
            row["supported_slot_ids"], "typed numeric item slots"
        )
        if not set(slots) <= known_slots:
            raise NumericEvidenceReconcilerError("typed numeric item escaped operator slots")
        summary = _exact_text(row["summary"], "typed numeric summary")
        numeric_raw = row.get("numeric_value")
        numeric = (
            None
            if numeric_raw is None
            else _finite_number(numeric_raw, "typed numeric value")
        )
        try:
            kind = TypedItemKind(row["kind"])
            role = NumericRole(row.get("numeric_role", NumericRole.NONE.value))
            qualifier = NumericQualifier(
                row.get("numeric_qualifier", NumericQualifier.EXACT.value)
            )
            status = EvidenceStatus(row["status"])
            authority = ValueAuthority(row["value_authority"])
            coherence = ContentCoherence(row["content_coherence"])
        except ValueError as exc:
            raise NumericEvidenceReconcilerError("typed numeric item enum changed") from exc
        included = row["included"]
        if type(included) is not bool:
            raise NumericEvidenceReconcilerError("typed numeric inclusion changed")
        text = {
            key: _optional_text(row.get(key), f"typed numeric {key}")
            for key in ("entity_key", "group_key", "unit", "date", "relation")
        }
        group_handles = tuple(
            dict.fromkeys(handle_map[value].group_handle for value in item_handles)
        )
        output.append(
            _Item(
                identity_sha256(row),
                item_handles,
                group_handles,
                kind,
                summary,
                text["entity_key"],
                text["group_key"],
                numeric,
                role,
                qualifier,
                text["unit"],
                text["date"],
                status,
                text["relation"],
                authority,
                included,
                slots,
                coherence,
            )
        )
    return tuple(output)


def _parse_input(
    provider_input: Mapping[str, Any],
    sealed_provider_input_sha256: str,
) -> tuple[str, Mapping[str, Any], _Operator, tuple[_Item, ...], tuple[_Handle, ...]]:
    if type(provider_input) is not dict:
        raise NumericEvidenceReconcilerError(
            "numeric reconciler requires an exact provider-input object"
        )
    require_sha256(sealed_provider_input_sha256, "sealed provider input")
    _forbid_identity_fields(provider_input)
    assert_gold_blind(provider_input, path="numeric_provider_input")
    observed = identity_sha256(provider_input)
    if observed != sealed_provider_input_sha256:
        raise NumericEvidenceReconcilerError("sealed provider input SHA-256 mismatch")
    if provider_input.get("format") != PROMPT_ROW_FORMAT:
        raise NumericEvidenceReconcilerError("numeric provider-input format changed")
    question = _exact_text(
        provider_input.get("dated_question"), "numeric dated question"
    )
    typed = provider_input.get("typed_evidence")
    if type(typed) is not dict or typed.get("format") != COMPACT_FINAL_PROVIDER_FORMAT:
        raise NumericEvidenceReconcilerError("compact typed evidence format changed")
    required_typed = {
        "conflict_policy",
        "format",
        "frontier",
        "handles",
        "items",
        "operator_spec",
    }
    if set(typed) != required_typed:
        raise NumericEvidenceReconcilerError("compact typed evidence schema changed")
    operator = _parse_operator(typed["operator_spec"])
    handles = _parse_handles(typed["handles"])
    items = _parse_items(typed["items"], handles=handles, operator=operator)
    frontier = typed["frontier"]
    if type(frontier) is not dict or type(frontier.get("closed")) is not bool:
        raise NumericEvidenceReconcilerError("typed numeric frontier schema changed")
    return question, typed, operator, items, handles


def _relation_fields(value: str | None) -> tuple[dict[str, str], frozenset[str]]:
    fields: dict[str, str] = {}
    flags: set[str] = set()
    if value is None:
        return fields, frozenset()
    for raw in value.split(";"):
        match = _RELATION_PART_RE.fullmatch(raw)
        if match is None:
            continue
        key = match.group("key").casefold().replace("-", "_")
        child = match.group("value")
        if child is None:
            flags.add(key)
        else:
            fields[key] = child.strip().casefold()
    return fields, frozenset(flags)


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.replace("/", "-").split(" (")[0]
    for format_string in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m",
        "%B %d, %Y",
        "%B %d %Y",
        "%B %Y",
    ):
        try:
            return datetime.strptime(normalized, format_string)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _question_timestamp(question: str, operator: _Operator) -> datetime | None:
    explicit = _parse_datetime(operator.query_timestamp)
    if explicit is not None:
        return explicit
    match = _QUESTION_TIMESTAMP_RE.search(question)
    return None if match is None else _parse_datetime(match.group("value"))


def _temporal_basis(
    item: _Item,
    *,
    question: str,
    operator: _Operator,
) -> tuple[bool, str]:
    temporal_required = (
        operator.temporal_mode is not TemporalMode.NONE
        or operator.temporal_window_days is not None
        or _RELATIVE_TEMPORAL_RE.search(question) is not None
    )
    if not temporal_required:
        return True, "not_required"
    if item.numeric_role is NumericRole.END or item.status is EvidenceStatus.CURRENT:
        return True, "sealed_current_or_end"
    fields, flags = _relation_fields(item.relation)
    marker = fields.get("temporal_eligibility")
    if marker in {"eligible", "included", "true"} or "temporal_eligible" in flags:
        return True, "explicit_typed_eligibility"
    if marker in {"ineligible", "excluded", "false"}:
        return False, "explicit_temporal_ineligibility"
    if _AMBIGUOUS_TEMPORAL_RE.search(item.summary):
        return False, "ambiguous_temporal_eligibility"
    if _EXPLICIT_TEMPORAL_RE.search(item.summary):
        return True, "explicit_summary_eligibility"
    moment = _parse_datetime(item.date)
    query = _question_timestamp(question, operator)
    if moment is None or query is None or moment > query:
        return False, "ambiguous_temporal_eligibility"
    if operator.temporal_window_days is not None:
        if (query - moment).days <= operator.temporal_window_days:
            return True, "dated_closed_window"
        return False, "outside_temporal_window"
    if operator.temporal_mode in {
        TemporalMode.LATEST_STATE,
        TemporalMode.ORDER,
    }:
        return True, "dated_temporal_comparator"
    # A date alone does not establish an uncompiled since/before/after bound.
    return False, "ambiguous_temporal_eligibility"


def _rejection(item: _Item, reason: str) -> RejectedNumericOperandProof:
    return RejectedNumericOperandProof(
        item.projection_sha256,
        item.handle_ids,
        reason,
    )


def _screen_numeric_items(
    items: Sequence[_Item],
    *,
    question: str,
    operator: _Operator,
) -> tuple[
    tuple[tuple[_Item, str], ...],
    tuple[RejectedNumericOperandProof, ...],
    int,
]:
    admitted: list[tuple[_Item, str]] = []
    rejected: list[RejectedNumericOperandProof] = []
    ignored = 0
    numeric_slots = operator.numeric_slot_ids
    for item in items:
        if item.numeric_value is None:
            ignored += 1
            continue
        reason: str | None = None
        if not item.included:
            reason = "item_not_included"
        elif item.content_coherence is not ContentCoherence.MATCH:
            reason = "item_not_content_coherent"
        elif item.status is EvidenceStatus.CANCELLED:
            reason = "cancelled_numeric_operand"
        elif item.status is EvidenceStatus.PROPOSED and not operator.include_proposed:
            reason = "proposed_numeric_operand_excluded"
        elif item.numeric_qualifier is not NumericQualifier.EXACT:
            reason = "non_exact_numeric_operand"
        elif item.value_authority is not ValueAuthority.EXPLICIT:
            reason = "non_explicit_numeric_value"
        elif (
            any(handle.startswith("H700") for handle in item.handle_ids)
            and not numeric_slots.intersection(item.supported_slot_ids)
        ):
            reason = "generic_lexical_h700_operand"
        elif numeric_slots and not numeric_slots.intersection(item.supported_slot_ids):
            reason = "numeric_slot_unbound"
        if reason is not None:
            rejected.append(_rejection(item, reason))
            continue
        eligible, temporal = _temporal_basis(
            item,
            question=question,
            operator=operator,
        )
        if not eligible:
            rejected.append(_rejection(item, temporal))
            continue
        admitted.append((item, temporal))
    return tuple(admitted), tuple(rejected), ignored


def _semantic_key(item: _Item) -> dict[str, Any]:
    fields, _flags = _relation_fields(item.relation)
    explicit = (
        fields.get("corroborates")
        or fields.get("event_key")
        or fields.get("dedup_key")
        or fields.get("cardinality_key")
    )
    if explicit is not None:
        identity: dict[str, Any] = {"explicit_key": explicit}
    elif item.group_key is not None:
        identity = {"group_key": item.group_key}
    elif item.entity_key is not None and item.date is not None:
        identity = {
            "date": item.date,
            "entity_terms": list(normalized_terms(item.entity_key)),
        }
    else:
        identity = {"summary_terms": list(normalized_terms(item.summary))}
    return {
        "identity": identity,
        "numeric_role": item.numeric_role.value,
    }


def _contributions(
    rows: Sequence[tuple[_Item, str]],
) -> tuple[NumericContributionProof, ...]:
    grouped: dict[str, list[tuple[_Item, str]]] = {}
    for item, temporal in rows:
        key = identity_sha256(_semantic_key(item))
        grouped.setdefault(key, []).append((item, temporal))
    output: list[NumericContributionProof] = []
    for key, members in grouped.items():
        first = members[0][0]
        semantics = _unit_semantics(first.unit)
        if any(
            child.numeric_value != first.numeric_value
            or child.numeric_role is not first.numeric_role
            or _unit_semantics(child.unit) != semantics
            for child, _temporal in members[1:]
        ):
            raise _ContributionConflict(
                "same_semantic_fact_has_incompatible_numeric_claims"
            )
        item_hashes = tuple(
            dict.fromkeys(child.projection_sha256 for child, _ in members)
        )
        output.append(
            NumericContributionProof(
                key,
                first.numeric_role.value,
                float(first.numeric_value or 0.0),
                semantics.canonical,
                semantics.family.value,
                tuple(
                    dict.fromkeys(
                        handle for child, _ in members for handle in child.handle_ids
                    )
                ),
                tuple(
                    dict.fromkeys(
                        slot
                        for child, _ in members
                        for slot in child.supported_slot_ids
                    )
                ),
                item_hashes,
                tuple(dict.fromkeys(temporal for _child, temporal in members)),
                len(item_hashes) - 1,
            )
        )
    return tuple(output)


def _expected_count_family(question: str) -> tuple[UnitFamily | None, str | None]:
    match = _COUNT_NOUN_RE.search(question)
    if match is None:
        return None, None
    noun = match.group("noun").casefold()
    semantics = _unit_semantics(noun)
    if noun in _DURATION_NOUNS:
        return UnitFamily.DURATION, semantics.canonical
    if noun in _EVENT_NOUNS:
        return UnitFamily.EVENT, semantics.canonical
    return UnitFamily.ITEM, semantics.canonical


def _relation_is_recurring_addition(item: _Item) -> bool:
    fields, flags = _relation_fields(item.relation)
    return (
        fields.get("frequency_addition") in {"true", "recurring", "included"}
        or fields.get("recurrence") in {"recurring", "frequency_addition"}
        or bool({"frequency_addition", "recurring_addition", "recurring"} & flags)
    )


def _relation_is_recurring_base(item: _Item) -> bool:
    fields, flags = _relation_fields(item.relation)
    return (
        fields.get("recurrence") in {"base", "recurring_base"}
        or fields.get("frequency_base") in {"true", "base"}
        or bool({"frequency_base", "recurring_base"} & flags)
    )


def _relation_for_delta(value: float) -> str:
    if value > 0:
        return "left_greater"
    if value < 0:
        return "left_less"
    return "equal"


def _empty_receipt(
    *,
    sealed_sha: str,
    typed_sha: str,
    operator_sha: str,
    status: ReconciliationStatus,
    mode: ReconciliationMode,
    reason: str,
    rejected: Sequence[RejectedNumericOperandProof],
    numeric_candidate_count: int,
    ignored_non_numeric_count: int,
) -> NumericEvidenceReconciliationReceipt:
    return NumericEvidenceReconciliationReceipt(
        sealed_sha,
        typed_sha,
        operator_sha,
        status,
        mode,
        reason,
        None,
        None,
        None,
        None,
        (),
        tuple(rejected),
        (),
        numeric_candidate_count,
        ignored_non_numeric_count,
        0,
    )


def _supported_receipt(
    *,
    sealed_sha: str,
    typed_sha: str,
    operator_sha: str,
    mode: ReconciliationMode,
    reason: str,
    numeric_result: float,
    unit: str | None,
    contributions: Sequence[NumericContributionProof],
    rejected: Sequence[RejectedNumericOperandProof],
    numeric_candidate_count: int,
    ignored_non_numeric_count: int,
    comparison_relation: str | None = None,
    boolean_result: bool | None = None,
) -> NumericEvidenceReconciliationReceipt:
    exact = tuple(contributions)
    return NumericEvidenceReconciliationReceipt(
        sealed_sha,
        typed_sha,
        operator_sha,
        ReconciliationStatus.SUPPORTED,
        mode,
        reason,
        numeric_result,
        unit,
        comparison_relation,
        boolean_result,
        exact,
        tuple(rejected),
        tuple(dict.fromkeys(handle for row in exact for handle in row.handle_ids)),
        numeric_candidate_count,
        ignored_non_numeric_count,
        sum(row.corroborated_duplicate_count for row in exact),
    )


def _direct_total(
    admitted: Sequence[tuple[_Item, str]],
) -> tuple[str, tuple[NumericContributionProof, ...], float | None, str | None]:
    selected = tuple(
        row
        for row in admitted
        if row[0].numeric_role is NumericRole.END
        or row[0].status is EvidenceStatus.CURRENT
    )
    if not selected:
        return "not_applicable", (), None, None
    try:
        proof = _contributions(selected)
    except _ContributionConflict:
        return "conflicting_current_or_end_totals", (), None, None
    values = {(row.numeric_value, row.unit) for row in proof}
    if len(values) != 1:
        return "conflicting_current_or_end_totals", (), None, None
    value, unit = next(iter(values))
    return "supported", proof, value, unit


def _recurring_total(
    admitted: Sequence[tuple[_Item, str]],
) -> tuple[str, tuple[NumericContributionProof, ...], float | None, str | None]:
    baselines = tuple(
        row
        for row in admitted
        if row[0].numeric_role is NumericRole.BASELINE
        and (
            _unit_semantics(row[0].unit).family is UnitFamily.RECURRING
            or _relation_is_recurring_base(row[0])
        )
    )
    if not baselines:
        return "not_applicable", (), None, None
    try:
        base_proof = _contributions(baselines)
    except _ContributionConflict:
        return "recurring_base_conflict", (), None, None
    base_values = {(row.numeric_value, row.unit) for row in base_proof}
    if len(base_values) != 1:
        return "recurring_base_conflict", (), None, None
    base_value, base_unit = next(iter(base_values))
    base_semantics = _unit_semantics(base_unit)
    if base_semantics.family is not UnitFamily.RECURRING:
        return "recurring_base_unit_unsealed", (), None, None
    additions: list[tuple[_Item, str]] = []
    for row in admitted:
        item = row[0]
        if item.numeric_role not in {NumericRole.DELTA, NumericRole.OPERAND}:
            continue
        semantics = _unit_semantics(item.unit)
        if semantics.family is UnitFamily.RECURRING:
            if semantics.canonical != base_semantics.canonical:
                return "recurring_period_or_unit_conflict", (), None, None
            additions.append(row)
        elif (
            semantics.family is base_semantics.numerator_family
            and _relation_is_recurring_addition(item)
        ):
            additions.append(row)
        else:
            return "ambiguous_one_off_frequency_addition", (), None, None
    if not additions:
        return "recurring_addition_missing", (), None, None
    try:
        addition_proof = _contributions(additions)
    except _ContributionConflict:
        return "recurring_addition_conflict", (), None, None
    total = float(base_value) + sum(row.numeric_value for row in addition_proof)
    return "supported", (*base_proof, *addition_proof), total, base_unit


def _side_scalar(
    rows: Sequence[tuple[_Item, str]],
) -> tuple[str, float | None, str | None, tuple[NumericContributionProof, ...]]:
    try:
        proof = _contributions(rows)
    except _ContributionConflict:
        return "comparison_side_value_conflict", None, None, ()
    baselines = tuple(row for row in proof if row.numeric_role == NumericRole.BASELINE.value)
    ends = tuple(row for row in proof if row.numeric_role == NumericRole.END.value)
    deltas = tuple(row for row in proof if row.numeric_role == NumericRole.DELTA.value)
    operands = tuple(
        row
        for row in proof
        if row.numeric_role in {NumericRole.OPERAND.value, NumericRole.NONE.value}
    )
    if baselines or ends:
        if len(baselines) == len(ends) == 1 and not deltas and not operands:
            if baselines[0].unit != ends[0].unit:
                return "comparison_side_unit_conflict", None, None, ()
            return (
                "supported",
                ends[0].numeric_value - baselines[0].numeric_value,
                ends[0].unit,
                proof,
            )
        if not baselines and len(ends) == 1 and not deltas and not operands:
            return "supported", ends[0].numeric_value, ends[0].unit, proof
        return "comparison_side_value_conflict", None, None, ()
    values = (*deltas, *operands)
    if not values:
        return "comparison_side_missing_value", None, None, ()
    units = {(row.unit_family, row.unit) for row in values}
    if len(units) != 1:
        return "comparison_side_unit_conflict", None, None, ()
    return "supported", sum(row.numeric_value for row in values), values[0].unit, proof


def _comparison(
    admitted: Sequence[tuple[_Item, str]],
    operator: _Operator,
) -> tuple[
    str,
    tuple[NumericContributionProof, ...],
    float | None,
    str | None,
    str | None,
    bool | None,
]:
    if operator.comparison_mode is ComparisonMode.NONE:
        return "not_applicable", (), None, None, None, None
    if operator.comparison_mode not in {
        ComparisonMode.DIFFERENCE,
        ComparisonMode.BOOLEAN_GREATER,
    }:
        return "unsupported_comparison_mode", (), None, None, None, None
    side_slots = tuple(row for row in operator.slots if row.kind is SlotKind.COMPARISON_SIDE)
    if len(side_slots) != 2 or any(not row.requires_numeric for row in side_slots):
        return "comparison_requires_two_sealed_numeric_sides", (), None, None, None, None
    side_ids = {row.slot_id for row in side_slots}
    if any(len(side_ids.intersection(item.supported_slot_ids)) != 1 for item, _ in admitted):
        return "comparison_item_binds_zero_or_multiple_sides", (), None, None, None, None
    side_rows = tuple(
        tuple(row for row in admitted if slot.slot_id in row[0].supported_slot_ids)
        for slot in side_slots
    )
    if any(not rows for rows in side_rows):
        return "comparison_side_missing_value", (), None, None, None, None
    left_status, left_value, left_unit, left_proof = _side_scalar(side_rows[0])
    right_status, right_value, right_unit, right_proof = _side_scalar(side_rows[1])
    if left_status != "supported":
        return left_status, (), None, None, None, None
    if right_status != "supported":
        return right_status, (), None, None, None, None
    if left_unit != right_unit:
        return "comparison_side_unit_conflict", (), None, None, None, None
    _require(left_value is not None and right_value is not None, "comparison scalar disappeared")
    delta = left_value - right_value
    relation = _relation_for_delta(delta)
    boolean = (
        delta > 0
        if operator.comparison_mode is ComparisonMode.BOOLEAN_GREATER
        else None
    )
    return "supported", (*left_proof, *right_proof), delta, left_unit, relation, boolean


def _cardinality_sum(
    admitted: Sequence[tuple[_Item, str]],
    *,
    question: str,
) -> tuple[str, tuple[NumericContributionProof, ...], float | None, str | None]:
    selected = tuple(
        row for row in admitted if row[0].numeric_role is NumericRole.OPERAND
    )
    if not selected:
        return "exact_cardinality_operands_missing", (), None, None
    try:
        proof = _contributions(selected)
    except _ContributionConflict:
        return "cardinality_operand_conflict", (), None, None
    families = {UnitFamily(row.unit_family) for row in proof}
    if UnitFamily.UNKNOWN in families:
        return "cardinality_operand_unit_unsealed", (), None, None
    if not families <= {UnitFamily.ITEM, UnitFamily.EVENT}:
        return "cardinality_operand_not_item_or_event", (), None, None
    if len(families) != 1:
        return "item_event_unit_conflict", (), None, None
    expected, expected_unit = _expected_count_family(question)
    actual = next(iter(families))
    if expected is UnitFamily.DURATION:
        return "duration_question_is_not_cardinality", (), None, None
    if expected is not None and expected is not actual:
        return "question_operand_unit_family_conflict", (), None, None
    if any(row.numeric_value < 0 or not row.numeric_value.is_integer() for row in proof):
        return "cardinality_operand_not_nonnegative_integer", (), None, None
    unit = expected_unit or proof[0].unit
    total = sum(row.numeric_value for row in proof)
    return "supported", proof, total, unit


def reconcile_sealed_numeric_evidence(
    provider_input: Mapping[str, Any],
    *,
    sealed_provider_input_sha256: str,
) -> NumericEvidenceReconciliationReceipt:
    """Reconcile only deterministic numeric facts in a sealed provider input."""

    question, typed, operator, items, _handles = _parse_input(
        provider_input,
        sealed_provider_input_sha256,
    )
    typed_sha = identity_sha256(typed)
    admitted, rejected, ignored = _screen_numeric_items(
        items,
        question=question,
        operator=operator,
    )
    candidate_count = sum(item.numeric_value is not None for item in items)
    common = {
        "sealed_sha": sealed_provider_input_sha256,
        "typed_sha": typed_sha,
        "operator_sha": operator.receipt_sha256,
        "rejected": rejected,
        "numeric_candidate_count": candidate_count,
        "ignored_non_numeric_count": ignored,
    }
    if operator.comparison_mode is not ComparisonMode.NONE:
        status, proof, value, unit, relation, boolean = _comparison(
            admitted, operator
        )
        if status == "supported":
            return _supported_receipt(
                **common,
                mode=ReconciliationMode.COMPARISON,
                reason="both_comparison_sides_sealed",
                numeric_result=float(value or 0.0),
                unit=unit,
                contributions=proof,
                comparison_relation=relation,
                boolean_result=boolean,
            )
        return _empty_receipt(
            **common,
            status=ReconciliationStatus.CONFLICTED,
            mode=ReconciliationMode.COMPARISON,
            reason=status,
        )

    status, proof, value, unit = _direct_total(admitted)
    if status == "supported":
        return _supported_receipt(
            **common,
            mode=ReconciliationMode.DIRECT_TOTAL,
            reason="one_corroborated_current_or_end_total",
            numeric_result=float(value or 0.0),
            unit=unit,
            contributions=proof,
        )
    if status != "not_applicable":
        return _empty_receipt(
            **common,
            status=ReconciliationStatus.CONFLICTED,
            mode=ReconciliationMode.DIRECT_TOTAL,
            reason=status,
        )

    status, proof, value, unit = _recurring_total(admitted)
    if status == "supported":
        return _supported_receipt(
            **common,
            mode=ReconciliationMode.RECURRING_PLUS_ADDITIONS,
            reason="recurring_base_plus_deduplicated_recurring_additions",
            numeric_result=float(value or 0.0),
            unit=unit,
            contributions=proof,
        )
    if status != "not_applicable":
        return _empty_receipt(
            **common,
            status=ReconciliationStatus.CONFLICTED,
            mode=ReconciliationMode.RECURRING_PLUS_ADDITIONS,
            reason=status,
        )

    status, proof, value, unit = _cardinality_sum(admitted, question=question)
    if status == "supported":
        return _supported_receipt(
            **common,
            mode=ReconciliationMode.CARDINALITY_SUM,
            reason="exact_deduplicated_cardinality_operands",
            numeric_result=float(value or 0.0),
            unit=unit,
            contributions=proof,
        )
    return _empty_receipt(
        **common,
        status=ReconciliationStatus.INSUFFICIENT,
        mode=ReconciliationMode.CARDINALITY_SUM,
        reason=(
            status
            if admitted
            else (
                rejected[0].reason
                if rejected
                else "no_sealed_numeric_operands"
            )
        ),
    )


__all__ = [
    "CONTRIBUTION_FORMAT",
    "FORMAT",
    "NumericContributionProof",
    "NumericEvidenceReconcilerError",
    "NumericEvidenceReconciliationReceipt",
    "POLICY_ID",
    "REJECTION_FORMAT",
    "ReconciliationMode",
    "ReconciliationStatus",
    "RejectedNumericOperandProof",
    "UnitFamily",
    "reconcile_sealed_numeric_evidence",
]
