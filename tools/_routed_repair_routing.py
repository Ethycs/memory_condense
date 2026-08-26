"""Gold-blind question routing for the locked full-source repair.

The router deliberately accepts only question text.  It compiles an evidence
processing style and inspectable modifiers; it does not predict whether the
available evidence is sufficient and cannot see benchmark categories, source
labels, question IDs, references, predictions, or judge outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from memory_condense.search.closure.compiler import infer_intent
from memory_condense.search.selectors.set_program import (
    SetOrdering,
    SetQuantifier,
    compile_set_program,
)


ROUTED_REPAIR_ROUTING_FORMAT = "memory-condense-routed-repair-routing-v1"


class RoutedRepairStyle(str, Enum):
    """Evidence representation and answer operation requested by a question."""

    EXTRACT = "direct_extract"
    STATE_CHAIN = "state_chain"
    TIMELINE = "temporal_timeline"
    NUMERIC_REDUCE = "numeric_reduce"
    SET_JOIN = "set_join"
    SYNTHESIZE = "synthesize"


class RoutedRepairReason(str, Enum):
    """Stable first-match reason for a route decision."""

    CURRENT_SYNTHESIS_REQUEST = "current_synthesis_request"
    TEMPORAL_INTERVAL = "temporal_interval"
    NUMERIC_AGGREGATE = "numeric_aggregate"
    NUMERIC_COMPARISON = "numeric_comparison"
    TEMPORAL_ORDER = "temporal_order"
    RELATIVE_TIME_LOOKUP = "relative_time_lookup"
    EXPLICIT_SET = "explicit_set"
    STATE_RESOLUTION = "state_resolution"
    DIRECT_FALLBACK = "direct_fallback"


@dataclass(frozen=True, slots=True)
class RoutedRepairModifiers:
    """Question-derived modifiers consumed by retrieval and fact rendering."""

    operation: str
    requires_complete_frontier: bool
    requires_temporal_metadata: bool
    cardinality: int | None
    ordinal: int | None
    ordering: str
    required_evidence_role: str | None
    required_evidence_role_basis: str | None
    query_timestamp: str | None
    temporal_window_days: int | None
    retrospective: bool

    def __post_init__(self) -> None:
        if not self.operation:
            raise ValueError("routing operation must be non-empty")
        if self.ordering not in {item.value for item in SetOrdering}:
            raise ValueError("routing ordering is invalid")
        for name in ("cardinality", "ordinal", "temporal_window_days"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"routing {name} must be a positive integer")
        if self.required_evidence_role not in {None, "user", "assistant"}:
            raise ValueError("routing required evidence role is invalid")
        if (self.required_evidence_role is None) != (
            self.required_evidence_role_basis is None
        ):
            raise ValueError("routing required evidence role and basis must agree")

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "requires_complete_frontier": self.requires_complete_frontier,
            "requires_temporal_metadata": self.requires_temporal_metadata,
            "cardinality": self.cardinality,
            "ordinal": self.ordinal,
            "ordering": self.ordering,
            "required_evidence_role": self.required_evidence_role,
            "required_evidence_role_basis": self.required_evidence_role_basis,
            "query_timestamp": self.query_timestamp,
            "temporal_window_days": self.temporal_window_days,
            "retrospective": self.retrospective,
        }


@dataclass(frozen=True, slots=True)
class RoutedRepairReceipt:
    """Content-addressed, provider-free route decision for one question."""

    question_sha256: str
    style: RoutedRepairStyle
    reason: RoutedRepairReason
    modifiers: RoutedRepairModifiers
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not _is_sha256(self.question_sha256):
            raise ValueError("routing question_sha256 must be lowercase SHA-256")
        expected = _identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("routing receipt digest does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": ROUTED_REPAIR_ROUTING_FORMAT,
            "question_sha256": self.question_sha256,
            "style": self.style.value,
            "reason": self.reason.value,
            "modifiers": self.modifiers.as_dict(),
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


_DATED_QUESTION_RE = re.compile(
    r"^\[Question asked at (?P<asked_at>.+?)\]\s*", re.IGNORECASE | re.DOTALL
)
_RETROSPECTIVE_RE = re.compile(
    r"\b(?:remind me|previous (?:chat|conversation)|prior (?:chat|conversation)|"
    r"going back to|thinking back to|follow(?:ing)? up on|we discussed earlier|"
    r"you (?:had )?(?:mentioned|provided|recommended|suggested|said|gave|told)|"
    r"finally decided)\b",
    re.IGNORECASE,
)
_EXPLICIT_ASSISTANT_PAST_RE = re.compile(
    r"\b(?:you|the assistant)\s+"
    r"(?:(?:had|specifically|previously|earlier)\s+)*"
    r"(?:mentioned|provided|recommended|"
    r"suggested|said|gave|told|advised|listed|shared)\b",
    re.IGNORECASE,
)
_SYNTHESIS_RE = re.compile(
    r"\b(?:recommend|suggest|suggestions?|ideas?|what should (?:i|we)|"
    r"could there be (?:a|any) reason|why)\b",
    re.IGNORECASE,
)
_TEMPORAL_INTERVAL_RE = re.compile(
    r"(?:\bhow many\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
    r"[^?.!]{0,144}\b(?:ago|passed|before|after|between|since|until|when)\b|"
    r"\bhow (?:long|much time)\b)",
    re.IGNORECASE,
)
_NUMERIC_COMPARISON_RE = re.compile(
    r"\b(?:difference|how (?:much|many) (?:more|less)|"
    r"older than|younger than|faster than|slower than|more expensive|"
    r"less expensive|higher percentage|lower percentage|what percentage|"
    r"compared (?:with|to)|versus|vs\.?)\b",
    re.IGNORECASE,
)
_NUMERIC_AGGREGATE_RE = re.compile(
    r"\b(?:total|sum|combined|in all|altogether)\b",
    re.IGNORECASE,
)
_NUMERIC_SUPERLATIVE_RE = re.compile(
    r"\b(?:gain(?:ed)?|lose|lost|spend|spent|cost|have|had|hold|held|"
    r"receive(?:d)?|read|watch(?:ed)?)\b[^?.!]{0,80}"
    r"\b(?:the\s+)?(?:most|fewest|highest|lowest)\b",
    re.IGNORECASE,
)
_PAIR_ORDER_RE = re.compile(
    r"\b(?:which|who|what)\b[^?.!]{0,160}\b(?:first|earlier|later)\b",
    re.IGNORECASE,
)
_EXPLICIT_ORDER_RE = re.compile(
    r"\b(?:order of|in (?:what|which) order|in (?:chronological )?order|"
    r"chronological(?:ly)?|earliest to latest|latest to earliest|"
    r"first to last|last to first|starting (?:with|from) (?:the )?"
    r"(?:earliest|latest|oldest|newest))\b",
    re.IGNORECASE,
)
_RELATIVE_TIME_RE = re.compile(
    r"\b(?:last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|\d+)\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
    re.IGNORECASE,
)
_EXPLICIT_MULTI_MEMBER_RE = re.compile(
    r"\b(?:what|which)\s+are\s+(?:the\s+)?(?:two|three|four|five|six|"
    r"seven|eight|nine|ten|\d+)\s+[^?.!]+",
    re.IGNORECASE,
)
_STATE_RE = re.compile(
    r"\b(?:currently|current|right now|at present|at the moment|latest|"
    r"newest|most recently|previous\b[^?.!]{0,80}\bbefore|"
    r"before\b[^?.!]{0,80}\b(?:updated|changed|current)|initially|"
    r"switch(?:ed)? to|updated|no longer|finally decided)\b",
    re.IGNORECASE,
)
_TIME_SCOPE_RE = re.compile(
    r"\b(?:today|yesterday|currently|current|since|ago|past|last|previous|"
    r"before|after|between|until|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)
_ORDINAL_RE = re.compile(r"\b(?P<ordinal>\d{1,3})(?:st|nd|rd|th)\b", re.IGNORECASE)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _identity_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _ordinal(body: str) -> int | None:
    match = _ORDINAL_RE.search(body)
    return int(match.group("ordinal")) if match is not None else None


def route_question(question: str, /) -> RoutedRepairReceipt:
    """Route one question without accepting any benchmark or answer metadata."""

    if type(question) is not str:
        raise TypeError("question must be a string")
    stripped = question.strip()
    if not stripped:
        raise ValueError("question must be non-empty")
    body = _DATED_QUESTION_RE.sub("", stripped).strip()
    if not body:
        raise ValueError("dated question body must be non-empty")

    program = compile_set_program(stripped)
    explicit_assistant_past = bool(_EXPLICIT_ASSISTANT_PAST_RE.search(body))
    required_evidence_role = program.required_evidence_role
    required_evidence_role_basis = program.required_evidence_role_basis
    if required_evidence_role is None and explicit_assistant_past:
        required_evidence_role = "assistant"
        required_evidence_role_basis = (
            "explicit_retrospective_assistant_attribution"
        )
    retrospective = bool(
        _RETROSPECTIVE_RE.search(body) or required_evidence_role is not None
    )
    current_synthesis = bool(
        not retrospective
        and (
            infer_intent(body) == "recommend"
            or _SYNTHESIS_RE.search(body)
        )
    )
    temporal_interval = bool(_TEMPORAL_INTERVAL_RE.search(body))
    numeric_comparison = bool(
        _NUMERIC_COMPARISON_RE.search(body)
        or _NUMERIC_SUPERLATIVE_RE.search(body)
    )
    numeric_aggregate = bool(
        program.quantifier is SetQuantifier.COUNT
        or _NUMERIC_AGGREGATE_RE.search(body)
    )
    temporal_order = bool(
        _EXPLICIT_ORDER_RE.search(body)
        or _PAIR_ORDER_RE.search(body)
        or program.ordering is not SetOrdering.NONE
    )
    relative_time = bool(_RELATIVE_TIME_RE.search(body))
    explicit_set = bool(
        program.quantifier in {SetQuantifier.ALL, SetQuantifier.FIXED}
        or _EXPLICIT_MULTI_MEMBER_RE.search(body)
    )
    state_resolution = bool(_STATE_RE.search(body))

    if current_synthesis:
        style = RoutedRepairStyle.SYNTHESIZE
        reason = RoutedRepairReason.CURRENT_SYNTHESIS_REQUEST
        operation = "preference_or_causal_synthesis"
    elif temporal_interval:
        style = RoutedRepairStyle.TIMELINE
        reason = RoutedRepairReason.TEMPORAL_INTERVAL
        operation = "interval"
    elif numeric_comparison:
        style = RoutedRepairStyle.NUMERIC_REDUCE
        reason = RoutedRepairReason.NUMERIC_COMPARISON
        operation = "compare_or_calculate"
    elif numeric_aggregate:
        style = RoutedRepairStyle.NUMERIC_REDUCE
        reason = RoutedRepairReason.NUMERIC_AGGREGATE
        operation = "count_or_aggregate"
    elif temporal_order:
        style = RoutedRepairStyle.TIMELINE
        reason = RoutedRepairReason.TEMPORAL_ORDER
        operation = "order_or_select"
    elif relative_time:
        style = RoutedRepairStyle.TIMELINE
        reason = RoutedRepairReason.RELATIVE_TIME_LOOKUP
        operation = "relative_time_lookup"
    elif explicit_set:
        style = RoutedRepairStyle.SET_JOIN
        reason = RoutedRepairReason.EXPLICIT_SET
        operation = "deduplicated_member_join"
    elif state_resolution:
        style = RoutedRepairStyle.STATE_CHAIN
        reason = RoutedRepairReason.STATE_RESOLUTION
        operation = "latest_or_prior_state"
    else:
        style = RoutedRepairStyle.EXTRACT
        reason = RoutedRepairReason.DIRECT_FALLBACK
        operation = "single_supported_fact"

    complete = bool(
        program.requires_completeness
        or style in {RoutedRepairStyle.NUMERIC_REDUCE, RoutedRepairStyle.SET_JOIN}
        or reason in {
            RoutedRepairReason.TEMPORAL_INTERVAL,
            RoutedRepairReason.TEMPORAL_ORDER,
        }
    )
    temporal = bool(
        style in {RoutedRepairStyle.STATE_CHAIN, RoutedRepairStyle.TIMELINE}
        or program.temporal_window_days is not None
        or _TIME_SCOPE_RE.search(body)
    )
    modifiers = RoutedRepairModifiers(
        operation=operation,
        requires_complete_frontier=complete,
        requires_temporal_metadata=temporal,
        cardinality=program.cardinality,
        ordinal=_ordinal(body),
        ordering=program.ordering.value,
        required_evidence_role=required_evidence_role,
        required_evidence_role_basis=required_evidence_role_basis,
        query_timestamp=program.query_timestamp,
        temporal_window_days=program.temporal_window_days,
        retrospective=retrospective,
    )
    return RoutedRepairReceipt(
        question_sha256=_question_sha256(question),
        style=style,
        reason=reason,
        modifiers=modifiers,
    )


__all__ = [
    "ROUTED_REPAIR_ROUTING_FORMAT",
    "RoutedRepairModifiers",
    "RoutedRepairReason",
    "RoutedRepairReceipt",
    "RoutedRepairStyle",
    "route_question",
]
