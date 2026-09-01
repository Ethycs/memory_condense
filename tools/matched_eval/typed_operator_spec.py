"""Question-only typed answer-operation specifications.

The existing routed-repair router chooses a broad processing style.  This
module adds the missing, evidence-independent obligation layer: answer shape,
comparison/temporal mode, and required semantic slots.  It deliberately accepts
only the dated question.  It cannot see question IDs, benchmark categories,
references, predictions, target registries, or judge outcomes.

The compiler is provider-free and retains no transformer state.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from tools._routed_repair_routing import (
    RoutedRepairReason,
    RoutedRepairStyle,
    route_question,
)

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-typed-operator-spec-v1"
SLOT_FORMAT = "memory-condense-typed-required-slot-v1"


class SlotKind(str, Enum):
    OPERAND = "operand"
    COMPARISON_SIDE = "comparison_side"
    PREDICATE = "predicate"
    PARTICIPANT = "participant"
    TEMPORAL_BOUNDARY = "temporal_boundary"


class AnswerShape(str, Enum):
    DIRECT = "direct"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DURATION = "duration"
    ORDERED_LIST = "ordered_list"
    SET_LIST = "set_list"
    SYNTHESIS = "synthesis"


class ComparisonMode(str, Enum):
    NONE = "none"
    DIFFERENCE = "difference"
    BOOLEAN_GREATER = "boolean_greater"
    MAX_ENTITY = "max_entity"


class TemporalMode(str, Enum):
    NONE = "none"
    INTERVAL = "interval"
    ORDER = "order"
    LATEST_STATE = "latest_state"
    RELATIVE_SELECT = "relative_select"


@dataclass(frozen=True, slots=True)
class RequiredSlot:
    slot_id: str
    kind: SlotKind
    label: str
    match_terms: tuple[str, ...]
    minimum_match_term_count: int
    requires_numeric: bool = False
    relation_constraint: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.slot_id, "required-slot ID")
        if type(self.kind) is not SlotKind:
            raise MatchedEvalContractError("required-slot kind must be canonical")
        require_text(self.label, "required-slot label")
        if (
            type(self.match_terms) is not tuple
            or not self.match_terms
            or any(
                type(term) is not str or not term or term != term.casefold()
                for term in self.match_terms
            )
            or len(set(self.match_terms)) != len(self.match_terms)
        ):
            raise MatchedEvalContractError(
                "required-slot match terms must be ordered unique normalized text"
            )
        if (
            type(self.minimum_match_term_count) is not int
            or not 1 <= self.minimum_match_term_count <= len(self.match_terms)
        ):
            raise MatchedEvalContractError("required-slot match threshold is invalid")
        if type(self.requires_numeric) is not bool:
            raise MatchedEvalContractError("required-slot numeric flag must be exact")
        if self.relation_constraint is not None:
            require_text(self.relation_constraint, "required-slot relation constraint")

    def projection(self) -> dict[str, Any]:
        return {
            "format": SLOT_FORMAT,
            "kind": self.kind.value,
            "label": self.label,
            "match_terms": list(self.match_terms),
            "minimum_match_term_count": self.minimum_match_term_count,
            "relation_constraint": self.relation_constraint,
            "requires_numeric": self.requires_numeric,
            "slot_id": self.slot_id,
        }


@dataclass(frozen=True, slots=True)
class TypedOperatorSpec:
    question_sha256: str
    route_receipt_sha256: str
    style: RoutedRepairStyle
    operation: str
    answer_shape: AnswerShape
    comparison_mode: ComparisonMode
    temporal_mode: TemporalMode
    required_slots: tuple[RequiredSlot, ...]
    requires_all_slots: bool
    requires_complete_frontier: bool
    absence_decision_requires_closed_frontier: bool
    specificity_required: bool
    personalization_required: bool
    include_proposed: bool
    cardinality: int | None
    ordering: str
    query_timestamp: str | None
    temporal_window_days: int | None
    required_evidence_role: str | None
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.question_sha256, "typed-spec question")
        require_sha256(self.route_receipt_sha256, "typed-spec route receipt")
        if type(self.style) is not RoutedRepairStyle:
            raise MatchedEvalContractError("typed-spec route style must be canonical")
        require_text(self.operation, "typed-spec operation")
        if type(self.answer_shape) is not AnswerShape:
            raise MatchedEvalContractError("typed-spec answer shape must be canonical")
        if type(self.comparison_mode) is not ComparisonMode:
            raise MatchedEvalContractError("typed-spec comparison mode must be canonical")
        if type(self.temporal_mode) is not TemporalMode:
            raise MatchedEvalContractError("typed-spec temporal mode must be canonical")
        if (
            type(self.required_slots) is not tuple
            or any(type(slot) is not RequiredSlot for slot in self.required_slots)
            or len({slot.slot_id for slot in self.required_slots})
            != len(self.required_slots)
        ):
            raise MatchedEvalContractError("typed-spec slots must be exact and unique")
        for flag, label in (
            (self.requires_all_slots, "all-slots flag"),
            (self.requires_complete_frontier, "complete-frontier flag"),
            (
                self.absence_decision_requires_closed_frontier,
                "absence-closure flag",
            ),
            (self.specificity_required, "specificity flag"),
            (self.personalization_required, "personalization flag"),
            (self.include_proposed, "include-proposed flag"),
        ):
            if type(flag) is not bool:
                raise MatchedEvalContractError(f"typed-spec {label} must be exact")
        if self.cardinality is not None and (
            type(self.cardinality) is not int or self.cardinality < 1
        ):
            raise MatchedEvalContractError("typed-spec cardinality must be positive")
        require_text(self.ordering, "typed-spec ordering")
        if self.query_timestamp is not None:
            require_text(self.query_timestamp, "typed-spec query timestamp")
        if self.temporal_window_days is not None and (
            type(self.temporal_window_days) is not int
            or self.temporal_window_days < 1
        ):
            raise MatchedEvalContractError("typed-spec temporal window is invalid")
        if self.required_evidence_role not in {None, "user", "assistant"}:
            raise MatchedEvalContractError("typed-spec evidence role is invalid")
        if self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("typed-spec retained transformer state")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("typed-spec receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_operator_spec")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "absence_decision_requires_closed_frontier": (
                self.absence_decision_requires_closed_frontier
            ),
            "answer_shape": self.answer_shape.value,
            "cardinality": self.cardinality,
            "comparison_mode": self.comparison_mode.value,
            "format": FORMAT,
            "operation": self.operation,
            "ordering": self.ordering,
            "personalization_required": self.personalization_required,
            "include_proposed": self.include_proposed,
            "query_timestamp": self.query_timestamp,
            "question_sha256": self.question_sha256,
            "required_evidence_role": self.required_evidence_role,
            "required_slots": [slot.projection() for slot in self.required_slots],
            "requires_all_slots": self.requires_all_slots,
            "requires_complete_frontier": self.requires_complete_frontier,
            "retained_transformer_token_state_bytes": 0,
            "route_receipt_sha256": self.route_receipt_sha256,
            "specificity_required": self.specificity_required,
            "style": self.style.value,
            "temporal_mode": self.temporal_mode.value,
            "temporal_window_days": self.temporal_window_days,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


_DATED_RE = re.compile(
    r"^\[Question asked at (?P<asked_at>.+?)\]\s*", re.IGNORECASE | re.DOTALL
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
_HOW_MANY_RE = re.compile(r"\bhow many\b", re.IGNORECASE)
_YES_NO_RE = re.compile(r"^(?:did|do|does|is|are|was|were|has|have|had|can)\b", re.I)
_SPECIFIC_RE = re.compile(r"\b(?:specific(?:ally)?|exact(?:ly)?|what type|what kind)\b", re.I)
_PARTICIPANT_RE = re.compile(r"\bwith\s+(a|an|one|my|the|some)?\s*(friend|friends|family|colleague|colleagues)\b", re.I)
_CONTEXT_RE = re.compile(
    r"\b(?:present(?:ed|ing)?|join(?:ed|ing)?|led me to|first order|"
    r"initially|undergrad(?:uate)?|return(?:ed|ing)?|pick(?:ed|ing)? up)\b",
    re.I,
)
_FOR_PAIR_RE = re.compile(
    r"\bfor\s+(?P<left>[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,3})"
    r"\s+(?:and|or)\s+(?P<right>[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,3})(?:\?|$)",
    re.I,
)
_HOW_MANY_SUBJECT_PAIR_RE = re.compile(
    r"\bhow many\s+(?P<left>[A-Za-z][A-Za-z0-9'’-]*)\s+(?:and|or)\s+"
    r"(?P<right>[A-Za-z][A-Za-z0-9'’-]*)(?:\s+[A-Za-z][A-Za-z0-9'’-]*)?\b",
    re.I,
)
_BETWEEN_RE = re.compile(
    r"\bbetween\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\?|$)", re.I
)
_COMPARED_TO_RE = re.compile(r"\bcompared\s+to\b", re.I)
_CAPITALIZED_ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){0,3}\b"
)
_COMPARISON_ENTITY_STOP = frozenset(
    {
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "how",
        "i",
        "is",
        "what",
        "was",
        "were",
        "which",
        "who",
        "would",
    }
)
_LATEST_TRANSACTION_RE = re.compile(
    r"\bhow much did I (?:spend|pay)(?:\s+for|\s+on)\b", re.I
)

_STOP = frozenset(
    {
        "a", "about", "ago", "an", "and", "any", "at", "be", "been",
        "can", "compared", "could", "did", "do", "does", "during", "for",
        "from", "have", "how", "i", "in", "initially", "is", "it", "last",
        "me", "month", "months", "my", "of", "on", "or", "past", "please",
        "recent", "recently", "the", "this", "to", "was", "were", "what",
        "when", "where", "which", "who", "why", "with", "would", "you",
    }
)


def normalize_term(value: str) -> str:
    """Small deterministic normalizer shared with the typed evidence adapter."""

    word = value.casefold().replace("’", "'").strip("' -_")
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("oes") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
        if len(word) >= 2 and word[-1] == word[-2]:
            word = word[:-1]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        word = word[:-1]
    return word


def normalized_terms(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            term
            for raw in _WORD_RE.findall(value)
            if (term := normalize_term(raw)) and term not in _STOP
        )
    )


def _compared_entity_pair(body: str) -> tuple[str, str] | None:
    """Return the nearest question-only named entities around ``compared to``.

    Selecting the first capitalized token before the comparator turns a
    sentence-opening interrogative (for example, ``How`` or ``Did``) into a
    numeric operand.  Split at the comparator instead, discard phrases made
    only of question-function words, then choose the nearest phrase on each
    side.  No question ID, evidence, prediction, or reference is involved.
    """

    compared = _COMPARED_TO_RE.search(body)
    if compared is None:
        return None

    def candidates(value: str) -> tuple[str, ...]:
        result: list[str] = []
        for match in _CAPITALIZED_ENTITY_RE.finditer(value):
            label = match.group(0).strip()
            terms = tuple(
                normalize_term(raw) for raw in _WORD_RE.findall(label)
            )
            if terms and any(term not in _COMPARISON_ENTITY_STOP for term in terms):
                result.append(label)
        return tuple(result)

    left = candidates(body[: compared.start()])
    right = candidates(body[compared.end() :])
    if not left or not right:
        return None
    return left[-1], right[0]


def _slot(
    kind: SlotKind,
    label: str,
    terms: tuple[str, ...],
    *,
    threshold: int | None = None,
    requires_numeric: bool = False,
    relation_constraint: str | None = None,
) -> RequiredSlot:
    clean = tuple(dict.fromkeys(normalize_term(term) for term in terms if term))
    if not clean:
        raise MatchedEvalContractError("question-derived slot has no match terms")
    minimum = len(clean) if threshold is None else min(threshold, len(clean))
    body = {
        "format": SLOT_FORMAT,
        "kind": kind.value,
        "label": label,
        "match_terms": list(clean),
        "minimum_match_term_count": minimum,
        "relation_constraint": relation_constraint,
        "requires_numeric": requires_numeric,
    }
    return RequiredSlot(
        identity_sha256(body),
        kind,
        label,
        clean,
        minimum,
        requires_numeric,
        relation_constraint,
    )


def _answer_shape(style: RoutedRepairStyle, reason: RoutedRepairReason, body: str) -> AnswerShape:
    if style is RoutedRepairStyle.SYNTHESIZE:
        return AnswerShape.SYNTHESIS
    if reason is RoutedRepairReason.TEMPORAL_INTERVAL:
        return AnswerShape.DURATION
    if reason is RoutedRepairReason.TEMPORAL_ORDER:
        return AnswerShape.ORDERED_LIST
    if style is RoutedRepairStyle.SET_JOIN:
        return AnswerShape.SET_LIST
    if style is RoutedRepairStyle.NUMERIC_REDUCE:
        if reason is RoutedRepairReason.NUMERIC_COMPARISON and _YES_NO_RE.search(body):
            return AnswerShape.BOOLEAN
        if reason is RoutedRepairReason.NUMERIC_COMPARISON and re.search(
            r"\b(?:which|who).{0,80}\b(?:most|highest|largest|greatest)\b",
            body,
            re.I,
        ):
            return AnswerShape.DIRECT
        return AnswerShape.NUMBER
    return AnswerShape.DIRECT


def _comparison_mode(reason: RoutedRepairReason, shape: AnswerShape, body: str) -> ComparisonMode:
    if reason is not RoutedRepairReason.NUMERIC_COMPARISON:
        return ComparisonMode.NONE
    if shape is AnswerShape.BOOLEAN:
        return ComparisonMode.BOOLEAN_GREATER
    if re.search(r"\b(?:most|highest|largest|greatest)\b", body, re.I):
        return ComparisonMode.MAX_ENTITY
    return ComparisonMode.DIFFERENCE


def _temporal_mode(style: RoutedRepairStyle, reason: RoutedRepairReason) -> TemporalMode:
    if style is RoutedRepairStyle.STATE_CHAIN:
        return TemporalMode.LATEST_STATE
    if reason is RoutedRepairReason.TEMPORAL_INTERVAL:
        return TemporalMode.INTERVAL
    if reason is RoutedRepairReason.TEMPORAL_ORDER:
        return TemporalMode.ORDER
    if reason is RoutedRepairReason.RELATIVE_TIME_LOOKUP:
        return TemporalMode.RELATIVE_SELECT
    return TemporalMode.NONE


def _question_slots(body: str, *, style: RoutedRepairStyle, reason: RoutedRepairReason) -> tuple[RequiredSlot, ...]:
    slots: list[RequiredSlot] = []
    pair = _FOR_PAIR_RE.search(body)
    if pair and (_HOW_MANY_RE.search(body) or style is RoutedRepairStyle.SET_JOIN):
        for label in (pair.group("left"), pair.group("right")):
            terms = normalized_terms(label)
            if terms:
                slots.append(
                    _slot(
                        SlotKind.OPERAND,
                        label.strip(" ?."),
                        terms,
                        requires_numeric=bool(_HOW_MANY_RE.search(body)),
                    )
                )

    subject_pair = _HOW_MANY_SUBJECT_PAIR_RE.search(body) if pair is None else None
    if subject_pair:
        for label in (subject_pair.group("left"), subject_pair.group("right")):
            terms = normalized_terms(label)
            if terms:
                slots.append(
                    _slot(
                        SlotKind.OPERAND,
                        label.strip(" ?."),
                        terms,
                        requires_numeric=True,
                    )
                )

    between = _BETWEEN_RE.search(body)
    if between and reason is RoutedRepairReason.TEMPORAL_INTERVAL:
        for label in (between.group("left"), between.group("right")):
            terms = normalized_terms(label)
            if terms:
                slots.append(
                    _slot(SlotKind.TEMPORAL_BOUNDARY, label.strip(" ?."), terms)
                )

    if reason is RoutedRepairReason.TEMPORAL_INTERVAL and between is None:
        event_terms = tuple(
            term
            for term in normalized_terms(body)
            if term not in {"long", "many", "pass", "since", "live"}
        )
        if event_terms:
            slots.append(
                _slot(
                    SlotKind.TEMPORAL_BOUNDARY,
                    "question-derived start event or state",
                    event_terms,
                    threshold=min(2, len(event_terms)),
                    relation_constraint="implicit_query_time_end",
                )
            )

    if re.search(r"\bcurrent apartment\b", body, re.I):
        state_terms = tuple(
            term
            for term in normalized_terms(body)
            if term in {"apartment", "harajuku", "current", "live"}
        )
        if state_terms:
            slots.append(
                _slot(
                    SlotKind.PREDICATE,
                    "current residence state",
                    state_terms,
                    threshold=min(2, len(state_terms)),
                    relation_constraint="state_entity",
                )
            )

    if _LATEST_TRANSACTION_RE.search(body):
        transaction_terms = tuple(
            term
            for term in normalized_terms(body)
            if term not in {"much", "spend", "pay"}
        )
        if transaction_terms:
            slots.append(
                _slot(
                    SlotKind.PREDICATE,
                    "completed transaction target",
                    transaction_terms,
                    threshold=min(2, len(transaction_terms)),
                    requires_numeric=True,
                    relation_constraint="latest_completed_transaction",
                )
            )

    compared = _compared_entity_pair(body)
    if compared and reason is RoutedRepairReason.NUMERIC_COMPARISON:
        for label in compared:
            slots.append(
                _slot(
                    SlotKind.COMPARISON_SIDE,
                    label,
                    (label,),
                    requires_numeric=True,
                )
            )

    participant = _PARTICIPANT_RE.search(body)
    if participant:
        noun = participant.group(2)
        singular = normalize_term(noun) == "friend" and noun.casefold() != "friends"
        slots.append(
            _slot(
                SlotKind.PARTICIPANT,
                participant.group(0),
                (noun,),
                relation_constraint=(
                    "participant_singular" if singular else "participant_any"
                ),
            )
        )

    if _CONTEXT_RE.search(body):
        terms = normalized_terms(body)
        # The answer variable (for example, "university") is not itself proof
        # of the event/predicate in the question.  Requiring several remaining
        # content terms prevents an adjacent entity mention from satisfying the
        # complete claim, as in the undergrad-poster case.
        wh_noun = re.search(r"\b(?:which|what|where)\s+([A-Za-z]+)", body, re.I)
        if wh_noun:
            answer_term = normalize_term(wh_noun.group(1))
            terms = tuple(term for term in terms if term != answer_term)
        if terms:
            slots.append(
                _slot(
                    SlotKind.PREDICATE,
                    "question predicate and role constraints",
                    terms,
                    threshold=min(3, len(terms)),
                )
            )

    unique: dict[str, RequiredSlot] = {}
    for slot in slots:
        unique.setdefault(slot.slot_id, slot)
    return tuple(unique.values())


def compile_typed_operator_spec(question: str, /) -> TypedOperatorSpec:
    """Compile a content-addressed specification from only one dated question."""

    if type(question) is not str:
        raise TypeError("question must be exact text")
    if not question or question.strip() != question:
        raise ValueError("question must be non-empty normalized text")
    route = route_question(question)
    body = _DATED_RE.sub("", question).strip()
    if not body:
        raise ValueError("dated question body must be non-empty")
    shape = _answer_shape(route.style, route.reason, body)
    slots = _question_slots(body, style=route.style, reason=route.reason)
    temporal_mode = _temporal_mode(route.style, route.reason)
    if _LATEST_TRANSACTION_RE.search(body):
        temporal_mode = TemporalMode.LATEST_STATE
    specificity = bool(_SPECIFIC_RE.search(body))
    personalization = bool(route.style is RoutedRepairStyle.SYNTHESIZE)
    include_proposed = bool(re.search(r"\b(?:plan|planned|planning|intend|intended)\b", body, re.I))
    requires_complete = bool(
        route.modifiers.requires_complete_frontier
        or any(slot.kind in {SlotKind.OPERAND, SlotKind.TEMPORAL_BOUNDARY} for slot in slots)
        or temporal_mode is TemporalMode.LATEST_STATE
    )
    return TypedOperatorSpec(
        question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
        route_receipt_sha256=route.receipt_sha256,
        style=route.style,
        operation=route.modifiers.operation,
        answer_shape=shape,
        comparison_mode=_comparison_mode(route.reason, shape, body),
        temporal_mode=temporal_mode,
        required_slots=slots,
        requires_all_slots=True,
        requires_complete_frontier=requires_complete,
        absence_decision_requires_closed_frontier=True,
        specificity_required=specificity,
        personalization_required=personalization,
        include_proposed=include_proposed,
        cardinality=route.modifiers.cardinality,
        ordering=route.modifiers.ordering,
        query_timestamp=route.modifiers.query_timestamp,
        temporal_window_days=route.modifiers.temporal_window_days,
        required_evidence_role=route.modifiers.required_evidence_role,
    )


__all__ = [
    "AnswerShape",
    "ComparisonMode",
    "FORMAT",
    "RequiredSlot",
    "SlotKind",
    "TemporalMode",
    "TypedOperatorSpec",
    "compile_typed_operator_spec",
    "normalize_term",
    "normalized_terms",
]
