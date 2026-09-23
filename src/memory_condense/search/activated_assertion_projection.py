"""Question-conditioned fact projection inside already activated sources.

This module is deliberately smaller than a retriever.  Its caller has already
selected opaque source identities and hydrated exact chunks from those
sources.  The projector identifies compact assertion-bearing excerpts, gives
numeric/date/action/URL/ordinal evidence independent selection opportunities,
and only then removes protected or duplicate evidence.

No source identifier is tokenized or otherwise interpreted.  ``created_at``
is retained as source metadata but is never used as an event date.  The module
accepts neither answers nor benchmark labels and performs no provider work.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from memory_condense.domain.discourse import identity_sha256, quote_sha256


FORMAT = "memory-condense-activated-assertion-projection-v1"
POLICY_FORMAT = f"{FORMAT}-policy"
HINT_FORMAT = f"{FORMAT}-question-hint"
ROLE_ROUTE_FORMAT = f"{FORMAT}-role-route"
FACT_FORMAT = f"{FORMAT}-fact"
QUOTE_SPAN_FORMAT = f"{FORMAT}-quote-span"
LANE_AUDIT_FORMAT = f"{FORMAT}-lane-audit"
CANDIDATE_AUDIT_FORMAT = f"{FORMAT}-candidate-audit"
MAX_OUTPUT_CHUNKS = 40


class AssertionRoleMode(str, Enum):
    """Question-derived authorship scope for local facts."""

    USER = "user"
    ASSISTANT = "assistant"
    MIXED = "mixed"


class AssertionReason(str, Enum):
    """Independent reason lanes retained before cross-lane deduplication."""

    ASSERTION = "assertion"
    NUMERIC = "numeric"
    DATE = "date"
    ACTION = "action"
    URL = "url"
    ORDINAL = "ordinal"


DEFAULT_LANE_BUDGETS: tuple[tuple[AssertionReason, int], ...] = (
    (AssertionReason.ASSERTION, 10),
    (AssertionReason.NUMERIC, 8),
    (AssertionReason.DATE, 6),
    (AssertionReason.ACTION, 8),
    (AssertionReason.URL, 4),
    (AssertionReason.ORDINAL, 4),
)


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be non-empty exact text")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _ordered_unique_text(
    values: Sequence[str], label: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be an ordered text sequence")
    result = tuple(values)
    if any(type(value) is not str or not value for value in result):
        raise ValueError(f"{label} must contain non-empty exact text")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must be ordered and unique")
    if not allow_empty and not result:
        raise ValueError(f"{label} cannot be empty")
    return result


def _question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ActivatedAssertionCandidate:
    """One exact hydrated chunk offered by the caller."""

    chunk_id: str
    source_id: str
    role: str
    created_at: str
    text: str
    token_count: int

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "candidate chunk ID")
        # The source value is validated only as an opaque equality key.
        _require_text(self.source_id, "candidate source ID")
        _require_text(self.role, "candidate role")
        if type(self.created_at) is not str:
            raise TypeError("candidate created_at must be exact text")
        _require_text(self.text, "candidate text")
        if (
            isinstance(self.token_count, bool)
            or type(self.token_count) is not int
            or self.token_count < 0
        ):
            raise ValueError("candidate token count must be non-negative")

    def identity_projection(self) -> dict[str, Any]:
        """Return exact, text-bound input identity without exposing source text."""

        return {
            "chunk_id": self.chunk_id,
            "created_at_sha256": quote_sha256(self.created_at),
            "role": self.role,
            "source_id_sha256": quote_sha256(self.source_id),
            "text_sha256": quote_sha256(self.text),
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class ActivatedAssertionPolicy:
    """Frozen, content-addressed limits for the local assertion plane."""

    max_output_chunks: int = MAX_OUTPUT_CHUNKS
    max_output_tokens: int = 2_400
    lane_budgets: tuple[tuple[AssertionReason, int], ...] = DEFAULT_LANE_BUDGETS
    include_proposed: bool = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_output_chunks, bool)
            or type(self.max_output_chunks) is not int
            or not 1 <= self.max_output_chunks <= MAX_OUTPUT_CHUNKS
        ):
            raise ValueError("max_output_chunks must be between 1 and 40")
        if (
            isinstance(self.max_output_tokens, bool)
            or type(self.max_output_tokens) is not int
            or self.max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be positive")
        if type(self.lane_budgets) is not tuple or not self.lane_budgets:
            raise TypeError("lane_budgets must be a non-empty exact tuple")
        lanes: list[AssertionReason] = []
        total = 0
        for row in self.lane_budgets:
            if (
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not AssertionReason
                or isinstance(row[1], bool)
                or type(row[1]) is not int
                or row[1] < 0
            ):
                raise ValueError("lane budget rows must be (AssertionReason, int)")
            lanes.append(row[0])
            total += row[1]
        if len(lanes) != len(set(lanes)):
            raise ValueError("lane budgets repeat a reason")
        if total < 1 or total > self.max_output_chunks:
            raise ValueError("lane budget sum must fit max_output_chunks")
        if type(self.include_proposed) is not bool:
            raise TypeError("include_proposed must be exact bool")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("assertion policy receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": POLICY_FORMAT,
            "include_proposed": self.include_proposed,
            "lane_budgets": [
                {"budget": budget, "reason": reason.value}
                for reason, budget in self.lane_budgets
            ],
            "max_output_chunks": self.max_output_chunks,
            "max_output_tokens": self.max_output_tokens,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class QuestionAssertionHint:
    """Optional primitives compiled from the same dated question upstream.

    The question digest prevents accidental reuse across prompts.  Callers are
    responsible for deriving ``obligation_terms`` and ``role_mode`` from the
    question alone; the type intentionally has no answer, prediction, ID, or
    benchmark-label field.
    """

    dated_question_sha256: str
    obligation_terms: tuple[str, ...] = ()
    role_mode: AssertionRoleMode | None = None
    include_proposed: bool | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.dated_question_sha256, "hint question")
        if type(self.obligation_terms) is not tuple:
            raise TypeError("hint obligation terms must be an exact tuple")
        _ordered_unique_text(self.obligation_terms, "hint obligation terms")
        if any(term != _normalize_term(term) for term in self.obligation_terms):
            raise ValueError("hint obligation terms must be normalized")
        if self.role_mode is not None and type(self.role_mode) is not AssertionRoleMode:
            raise TypeError("hint role mode must be canonical")
        if (
            self.include_proposed is not None
            and type(self.include_proposed) is not bool
        ):
            raise TypeError("hint include_proposed must be exact bool or None")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("question assertion hint receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "dated_question_sha256": self.dated_question_sha256,
            "format": HINT_FORMAT,
            "include_proposed": self.include_proposed,
            "obligation_terms": list(self.obligation_terms),
            "role_mode": None if self.role_mode is None else self.role_mode.value,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class AssertionRoleRoute:
    inferred_mode: AssertionRoleMode
    effective_mode: AssertionRoleMode
    reasons: tuple[str, ...]
    hint_applied: bool

    def __post_init__(self) -> None:
        if type(self.inferred_mode) is not AssertionRoleMode:
            raise TypeError("inferred role mode must be canonical")
        if type(self.effective_mode) is not AssertionRoleMode:
            raise TypeError("effective role mode must be canonical")
        if type(self.reasons) is not tuple:
            raise TypeError("role-route reasons must be an exact tuple")
        _ordered_unique_text(self.reasons, "role-route reasons", allow_empty=False)
        if type(self.hint_applied) is not bool:
            raise TypeError("role-route hint flag must be exact")

    def projection(self) -> dict[str, Any]:
        return {
            "effective_mode": self.effective_mode.value,
            "format": ROLE_ROUTE_FORMAT,
            "hint_applied": self.hint_applied,
            "inferred_mode": self.inferred_mode.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class AssertionQuoteSpan:
    """One exact source span participating in an assembled fact excerpt."""

    quote: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        _require_text(self.quote, "assertion quote span")
        if (
            type(self.start_char) is not int
            or type(self.end_char) is not int
            or not 0 <= self.start_char < self.end_char
            or self.end_char - self.start_char != len(self.quote)
        ):
            raise ValueError("assertion quote span coordinates changed")

    def projection(self) -> dict[str, Any]:
        return {
            "end_char": self.end_char,
            "format": QUOTE_SPAN_FORMAT,
            "quote": self.quote,
            "quote_sha256": quote_sha256(self.quote),
            "start_char": self.start_char,
        }


@dataclass(frozen=True, slots=True)
class SelectedAssertionFact:
    chunk_id: str
    source_handle: str
    role: str
    source_created_at: str
    quote: str
    quote_start_char: int
    quote_end_char: int
    quote_spans: tuple[AssertionQuoteSpan, ...]
    input_text_sha256: str
    input_token_count: int
    fact_token_count: int
    inclusion_reasons: tuple[AssertionReason, ...]
    selection_routes: tuple[AssertionReason, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "selected fact chunk ID")
        if re.fullmatch(r"G[0-9]{6}", self.source_handle) is None:
            raise ValueError("selected fact source handle changed")
        _require_text(self.role, "selected fact role")
        if type(self.source_created_at) is not str:
            raise TypeError("selected fact created_at must be exact text")
        _require_text(self.quote, "selected fact quote")
        if (
            type(self.quote_start_char) is not int
            or type(self.quote_end_char) is not int
            or not 0 <= self.quote_start_char < self.quote_end_char
        ):
            raise ValueError("selected fact quote coordinates changed")
        if (
            type(self.quote_spans) is not tuple
            or not self.quote_spans
            or any(type(row) is not AssertionQuoteSpan for row in self.quote_spans)
        ):
            raise ValueError("selected fact quote spans changed")
        if (
            self.quote_spans[0].start_char != self.quote_start_char
            or self.quote_spans[-1].end_char != self.quote_end_char
            or any(
                left.end_char > right.start_char
                for left, right in zip(self.quote_spans, self.quote_spans[1:])
            )
            or self.quote != "\n".join(row.quote for row in self.quote_spans)
        ):
            raise ValueError("selected fact quote assembly changed")
        _require_sha256(self.input_text_sha256, "selected fact input text")
        for value, label in (
            (self.input_token_count, "selected fact input tokens"),
            (self.fact_token_count, "selected fact tokens"),
        ):
            if isinstance(value, bool) or type(value) is not int or value < 0:
                raise ValueError(f"{label} must be non-negative")
        if (
            type(self.inclusion_reasons) is not tuple
            or not self.inclusion_reasons
            or any(type(row) is not AssertionReason for row in self.inclusion_reasons)
            or len(set(self.inclusion_reasons)) != len(self.inclusion_reasons)
        ):
            raise ValueError("selected fact reasons changed")
        if (
            type(self.selection_routes) is not tuple
            or not self.selection_routes
            or any(type(row) is not AssertionReason for row in self.selection_routes)
            or len(set(self.selection_routes)) != len(self.selection_routes)
        ):
            raise ValueError("selected fact routes changed")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("selected fact receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "created_at_semantics": "source_metadata_only_not_event_time",
            "fact_token_count": self.fact_token_count,
            "format": FACT_FORMAT,
            "inclusion_reasons": [row.value for row in self.inclusion_reasons],
            "input_text_sha256": self.input_text_sha256,
            "input_token_count": self.input_token_count,
            "quote": self.quote,
            "quote_end_char": self.quote_end_char,
            "quote_coordinate_semantics": (
                "single_exact_source_span"
                if len(self.quote_spans) == 1
                else "outer_envelope_exact_spans_are_authoritative"
            ),
            "quote_spans": [row.projection() for row in self.quote_spans],
            "quote_sha256": quote_sha256(self.quote),
            "quote_start_char": self.quote_start_char,
            "role": self.role,
            "selection_routes": [row.value for row in self.selection_routes],
            "source_created_at": self.source_created_at,
            "source_handle": self.source_handle,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class AssertionLaneAudit:
    reason: AssertionReason
    budget: int
    ranked_candidate_chunk_ids: tuple[str, ...]
    selected_before_dedup_chunk_ids: tuple[str, ...]
    retained_after_dedup_chunk_ids: tuple[str, ...]
    protected_duplicate_chunk_ids: tuple[str, ...]
    exact_duplicate_chunk_ids: tuple[str, ...]
    semantic_duplicate_chunk_ids: tuple[str, ...]
    refilled_chunk_ids: tuple[str, ...]
    packing_refilled_chunk_ids: tuple[str, ...]
    token_unpacked_chunk_ids: tuple[str, ...]
    packed_chunk_ids: tuple[str, ...]
    unfilled_slots: int

    def __post_init__(self) -> None:
        if type(self.reason) is not AssertionReason:
            raise TypeError("lane-audit reason must be canonical")
        if (
            isinstance(self.budget, bool)
            or type(self.budget) is not int
            or self.budget < 0
        ):
            raise ValueError("lane-audit budget must be non-negative")
        for name in (
            "ranked_candidate_chunk_ids",
            "selected_before_dedup_chunk_ids",
            "retained_after_dedup_chunk_ids",
            "protected_duplicate_chunk_ids",
            "exact_duplicate_chunk_ids",
            "semantic_duplicate_chunk_ids",
            "refilled_chunk_ids",
            "packing_refilled_chunk_ids",
            "token_unpacked_chunk_ids",
            "packed_chunk_ids",
        ):
            value = getattr(self, name)
            if type(value) is not tuple:
                raise TypeError(f"lane-audit {name} must be an exact tuple")
            _ordered_unique_text(value, f"lane-audit {name}")
        if (
            isinstance(self.unfilled_slots, bool)
            or type(self.unfilled_slots) is not int
            or not 0 <= self.unfilled_slots <= self.budget
        ):
            raise ValueError("lane-audit unfilled slots changed")
        if len(self.packed_chunk_ids) + self.unfilled_slots != self.budget:
            raise ValueError("lane-audit packed count disagrees with budget")

    def projection(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "exact_duplicate_chunk_ids": list(self.exact_duplicate_chunk_ids),
            "format": LANE_AUDIT_FORMAT,
            "packed_chunk_ids": list(self.packed_chunk_ids),
            "packing_refilled_chunk_ids": list(
                self.packing_refilled_chunk_ids
            ),
            "protected_duplicate_chunk_ids": list(
                self.protected_duplicate_chunk_ids
            ),
            "ranked_candidate_chunk_ids": list(self.ranked_candidate_chunk_ids),
            "reason": self.reason.value,
            "refilled_chunk_ids": list(self.refilled_chunk_ids),
            "retained_after_dedup_chunk_ids": list(
                self.retained_after_dedup_chunk_ids
            ),
            "selected_before_dedup_chunk_ids": list(
                self.selected_before_dedup_chunk_ids
            ),
            "semantic_duplicate_chunk_ids": list(
                self.semantic_duplicate_chunk_ids
            ),
            "token_unpacked_chunk_ids": list(self.token_unpacked_chunk_ids),
            "unfilled_slots": self.unfilled_slots,
        }


@dataclass(frozen=True, slots=True)
class AssertionCandidateAudit:
    input_ordinal: int
    chunk_id: str
    source_handle: str | None
    source_id_sha256: str
    role: str
    text_sha256: str
    eligible_reasons: tuple[AssertionReason, ...]
    selection_routes: tuple[AssertionReason, ...]
    included: bool
    omission_reasons: tuple[str, ...]
    duplicate_of_chunk_id: str | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.input_ordinal, bool)
            or type(self.input_ordinal) is not int
            or self.input_ordinal < 0
        ):
            raise ValueError("candidate-audit ordinal must be non-negative")
        _require_text(self.chunk_id, "candidate-audit chunk ID")
        if self.source_handle is not None and re.fullmatch(
            r"G[0-9]{6}", self.source_handle
        ) is None:
            raise ValueError("candidate-audit source handle changed")
        _require_sha256(self.source_id_sha256, "candidate-audit source")
        _require_text(self.role, "candidate-audit role")
        _require_sha256(self.text_sha256, "candidate-audit text")
        for name in ("eligible_reasons", "selection_routes"):
            value = getattr(self, name)
            if (
                type(value) is not tuple
                or any(type(row) is not AssertionReason for row in value)
                or len(value) != len(set(value))
            ):
                raise ValueError(f"candidate-audit {name} changed")
        if type(self.included) is not bool:
            raise TypeError("candidate-audit included flag must be exact")
        if type(self.omission_reasons) is not tuple:
            raise TypeError("candidate-audit omissions must be an exact tuple")
        _ordered_unique_text(self.omission_reasons, "candidate-audit omissions")
        if self.duplicate_of_chunk_id is not None:
            _require_text(self.duplicate_of_chunk_id, "candidate-audit duplicate")
        if self.included and self.omission_reasons:
            raise ValueError("included candidate cannot carry omissions")

    def projection(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "duplicate_of_chunk_id": self.duplicate_of_chunk_id,
            "eligible_reasons": [row.value for row in self.eligible_reasons],
            "format": CANDIDATE_AUDIT_FORMAT,
            "included": self.included,
            "input_ordinal": self.input_ordinal,
            "omission_reasons": list(self.omission_reasons),
            "role": self.role,
            "selection_routes": [row.value for row in self.selection_routes],
            "source_handle": self.source_handle,
            "source_id_sha256": self.source_id_sha256,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True, slots=True)
class ActivatedAssertionProjection:
    dated_question_sha256: str
    policy: ActivatedAssertionPolicy
    question_hint_receipt_sha256: str | None
    effective_include_proposed: bool
    role_route: AssertionRoleRoute
    active_source_bindings: tuple[tuple[str, str], ...]
    candidate_population_sha256: str
    protected_chunk_ids: tuple[str, ...]
    selected_facts: tuple[SelectedAssertionFact, ...]
    lane_audits: tuple[AssertionLaneAudit, ...]
    candidate_audits: tuple[AssertionCandidateAudit, ...]
    payload_token_count: int
    token_budget_exhausted: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.dated_question_sha256, "projection question")
        if type(self.policy) is not ActivatedAssertionPolicy:
            raise TypeError("projection policy must be exact")
        if self.question_hint_receipt_sha256 is not None:
            _require_sha256(self.question_hint_receipt_sha256, "projection hint")
        if type(self.effective_include_proposed) is not bool:
            raise TypeError("projection include_proposed flag must be exact")
        if type(self.role_route) is not AssertionRoleRoute:
            raise TypeError("projection role route must be exact")
        if (
            type(self.active_source_bindings) is not tuple
            or any(
                type(row) is not tuple
                or len(row) != 2
                or re.fullmatch(r"G[0-9]{6}", row[0]) is None
                or _require_sha256(row[1], "active source identity") != row[1]
                for row in self.active_source_bindings
            )
        ):
            raise ValueError("active source bindings changed")
        _require_sha256(self.candidate_population_sha256, "candidate population")
        if type(self.protected_chunk_ids) is not tuple:
            raise TypeError("projection protected chunk IDs must be an exact tuple")
        _ordered_unique_text(self.protected_chunk_ids, "protected chunk IDs")
        if type(self.selected_facts) is not tuple or any(
            type(row) is not SelectedAssertionFact for row in self.selected_facts
        ):
            raise TypeError("projection selected facts must be an exact tuple")
        if len({row.chunk_id for row in self.selected_facts}) != len(
            self.selected_facts
        ):
            raise ValueError("projection selected facts repeat a chunk")
        if type(self.lane_audits) is not tuple or any(
            type(row) is not AssertionLaneAudit for row in self.lane_audits
        ):
            raise TypeError("projection lane audits must be an exact tuple")
        if tuple(row.reason for row in self.lane_audits) != tuple(
            reason for reason, _budget in self.policy.lane_budgets
        ):
            raise ValueError("projection lane audits disagree with policy")
        if type(self.candidate_audits) is not tuple or any(
            type(row) is not AssertionCandidateAudit for row in self.candidate_audits
        ):
            raise TypeError("projection candidate audits must be an exact tuple")
        if len(self.selected_facts) > self.policy.max_output_chunks:
            raise ValueError("selected facts exceed output chunk cap")
        if (
            type(self.payload_token_count) is not int
            or self.payload_token_count < 0
            or self.payload_token_count > self.policy.max_output_tokens
        ):
            raise ValueError("selected facts exceed output token cap")
        if type(self.token_budget_exhausted) is not bool:
            raise TypeError("token budget exhausted flag must be exact")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("activated assertion projection receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "active_source_bindings": [
                {"source_handle": handle, "source_id_sha256": digest}
                for handle, digest in self.active_source_bindings
            ],
            "candidate_audits": [row.projection() for row in self.candidate_audits],
            "candidate_population_sha256": self.candidate_population_sha256,
            "dated_question_sha256": self.dated_question_sha256,
            "effective_include_proposed": self.effective_include_proposed,
            "format": FORMAT,
            "gold_loaded": False,
            "lane_audits": [row.projection() for row in self.lane_audits],
            "new_provider_calls": 0,
            "payload_token_count": self.payload_token_count,
            "policy": self.policy.projection(),
            "protected_chunk_ids": list(self.protected_chunk_ids),
            "question_hint_receipt_sha256": self.question_hint_receipt_sha256,
            "role_route": self.role_route.projection(),
            "selected_facts": [row.projection() for row in self.selected_facts],
            "token_budget_exhausted": self.token_budget_exhausted,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


_DATED_QUESTION_RE = re.compile(
    r"^\s*\[Question asked at .+?\]\s*", re.IGNORECASE | re.DOTALL
)
_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|I'm|I've|I'd|I'll|my|mine|me|we|we're|we've|we'd|we'll|our|ours)\b",
    re.IGNORECASE,
)
_USER_SCOPE_RE = re.compile(
    r"\b(?:I|I'm|I've|I'd|I'll|my|mine|we|we're|we've|we'd|we'll|our|ours)\b",
    re.IGNORECASE,
)
_ASSISTANT_RECALL_RE = re.compile(
    r"\b(?:you|the\s+assistant)\b[^?.!\n]{0,96}\b(?:said|told|gave|"
    r"provided|recommended|suggested|mentioned|listed|sent|shared|advised)\b|"
    r"\b(?:what|which|where)\b[^?.!\n]{0,80}\b(?:did|had)\s+(?:you|the\s+assistant)\b",
    re.IGNORECASE,
)
_ASSISTANT_ARTIFACT_QUERY_RE = re.compile(
    r"\b(?:recommendations?|suggestions?|options?|list|urls?|links?|websites?|"
    r"webpages?)\b",
    re.IGNORECASE,
)
_USER_CURRENT_RE = re.compile(
    r"\b(?:currently|right\s+now|at\s+present|at\s+the\s+moment|"
    r"my\s+current|do\s+i\s+have|am\s+i|have\s+i)\b",
    re.IGNORECASE,
)
_USER_COUNT_RE = re.compile(
    r"\bhow\s+many\b[^?.!\n]{0,120}\b(?:i|my|mine|we|our)\b|"
    r"\b(?:i|my|mine|we|our)\b[^?.!\n]{0,120}\bhow\s+many\b",
    re.IGNORECASE,
)
_USER_PREFERENCE_RE = re.compile(
    r"\b(?:my\s+(?:favorite|favourite|preference)|"
    r"(?:do|did)\s+i\s+(?:like|love|prefer|enjoy)|"
    r"i\s+(?:like|love|prefer|enjoy)(?:ed)?)\b",
    re.IGNORECASE,
)
_USER_AUTOBIOGRAPHICAL_RE = re.compile(
    r"\b(?:i|we)\b[^?.!\n]{0,96}\b(?:bought|purchased|ordered|visited|"
    r"attended|completed|finished|started|joined|participated|used|replaced|"
    r"upgraded|read|watched|wrote|made|created|planted|grew|installed|moved|"
    r"lived|worked|studied|received|got|paid|spent|chose|selected|decided)\b",
    re.IGNORECASE,
)
_QUESTION_LEAD_RE = re.compile(
    r"^\s*(?:who|what|when|where|why|which|how|can|could|would|should|"
    r"do|does|did|is|are|was|were|have|has|had|will)\b",
    re.IGNORECASE,
)
# Proposal filtering is deliberately anchored to a first-person intent or
# hypothetical predicate.  Modal language in a completed assertion's reason
# clause ("I bought it because it would last") is not itself a proposal.
_PROPOSED_RE = re.compile(
    r"\b(?:i|we)\s+(?:(?:plan(?:s|ned|ning)?|intend(?:s|ed|ing)?|"
    r"hope(?:s|d|ing)?|want(?:s|ed|ing)?|expect(?:s|ed|ing)?|"
    r"consider(?:s|ed|ing)?)\s+(?:to|(?:about|on)\s+)|"
    r"(?:might|may|could|would|should|will)\s+(?:not\s+)?[^\s?.!,;:]+)|"
    r"\b(?:i['’]m|we['’]re|i\s+am|we\s+are)\s+"
    r"(?:going|planning|hoping|intending|thinking)\s+(?:to|about|of)\b|"
    r"\b(?:i|we)['’]d\s+like\s+to\b|"
    r"\b(?:i|we)['’]ll\s+[^\s?.!,;:]+|"
    r"\bif\s+(?:i|we)\s+[^\s?.!,;:]+",
    re.IGNORECASE,
)
_COMPLETED_FIRST_PERSON_PREFIX_RE = re.compile(
    r"^\s*(?:i|we)\s+(?:bought|purchased|ordered|visited|attended|completed|"
    r"finished|started|joined|participated|used|replaced|upgraded|read|watched|"
    r"wrote|made|created|planted|grew|installed|moved|lived|worked|studied|"
    r"received|got|paid|spent|chose|selected|decided|preferred|liked|loved|"
    r"enjoyed)\b",
    re.IGNORECASE,
)
_ASSISTANT_RECALLED_MODAL_RE = re.compile(
    r"^\s*i\s+would\s+(?:recommend|suggest|provide|choose|use|send|share|list)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>()]+|\b[a-z0-9][a-z0-9.-]+\.[a-z]{2,}"
    r"(?:/[^\s<>()]*)?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(?:\d{1,2}(?:st|nd|rd|th)?(?:,?\s+"
    r"(?:19|20)\d{2})?|(?:19|20)\d{2})\b|"
    r"\b(?:today|yesterday|last\s+(?:Monday|Tuesday|Wednesday|Thursday|"
    r"Friday|Saturday|Sunday)|(?:a|an|one|two|three|four|five|six|seven|"
    r"eight|nine|ten|\d+)\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|st|nd|rd|th)?(?!\w)|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand)\b",
    re.IGNORECASE,
)
_ORDINAL_RE = re.compile(
    r"\b(?:\d{1,4}(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth|eleventh|twelfth|earliest|latest|oldest|"
    r"newest|previous|next)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(?:bought|buy|purchased|purchase|ordered|order|visited|visit|"
    r"attended|attend|completed|complete|finished|finish|started|start|"
    r"joined|join|participated|participate|used|use|replaced|replace|"
    r"upgraded|upgrade|read|watched|watch|wrote|write|made|make|created|"
    r"create|planted|plant|grew|grow|installed|install|moved|move|lived|"
    r"live|worked|work|studied|study|received|receive|got|get|paid|pay|"
    r"spent|spend|recommended|recommend|suggested|suggest|provided|provide|"
    r"listed|list|sent|send|shared|share|mentioned|mention|told|tell|gave|"
    r"give|chose|choose|selected|select|decided|decide|preferred|prefer|"
    r"liked|like|loved|love|enjoyed|enjoy)\b",
    re.IGNORECASE,
)
_LIST_STRUCTURE_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+|(?:^|\n)\s*[^\n:]{1,48}:\s*",
    re.MULTILINE,
)
_ABBREVIATIONS = frozenset(
    {
        "dr.",
        "e.g.",
        "etc.",
        "i.e.",
        "jr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "u.k.",
        "u.s.",
        "vs.",
    }
)
_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
    }
)


def _normalize_term(value: str) -> str:
    word = (
        unicodedata.normalize("NFKC", value)
        .casefold()
        .replace("’", "'")
        .strip("'-_")
    )
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


def _terms(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            term
            for raw in _WORD_RE.findall(value)
            if (term := _normalize_term(raw)) and term not in _STOP_TERMS
        )
    )


def _semantic_text_key(value: str) -> str:
    return " ".join(_normalize_term(raw) for raw in _WORD_RE.findall(value))


def route_assertion_roles(
    dated_question: str,
    /,
    *,
    hint: QuestionAssertionHint | None = None,
) -> AssertionRoleRoute:
    """Choose a conservative authorship scope from question text only."""

    _require_text(dated_question, "dated question")
    digest = _question_sha256(dated_question)
    if hint is not None:
        if type(hint) is not QuestionAssertionHint:
            raise TypeError("question hint must be exact")
        if hint.dated_question_sha256 != digest:
            raise ValueError("question hint escaped its dated question")
    body = _DATED_QUESTION_RE.sub("", dated_question, count=1).strip()
    if not body:
        raise ValueError("dated question body must be non-empty")

    reasons: list[str] = []
    assistant = bool(_ASSISTANT_RECALL_RE.search(body))
    if assistant:
        reasons.append("explicit_assistant_recall")
    user_checks = (
        ("user_current_state", _USER_CURRENT_RE),
        ("user_count", _USER_COUNT_RE),
        ("user_preference", _USER_PREFERENCE_RE),
        ("user_autobiographical", _USER_AUTOBIOGRAPHICAL_RE),
    )
    user = False
    for label, pattern in user_checks:
        if pattern.search(body):
            reasons.append(label)
            user = True
    # Object pronoun ``me`` alone is compatible with assistant recall (for
    # example, "Which URL did you give me?") and is not an autobiographical
    # signal.  It remains valid inside a first-person source assertion.
    if not user and _USER_SCOPE_RE.search(body):
        reasons.append("first_person_memory_scope")
        user = True

    if assistant and user:
        inferred = AssertionRoleMode.MIXED
        reasons.append("combined_user_and_assistant_scope")
    elif assistant:
        inferred = AssertionRoleMode.ASSISTANT
    elif user:
        inferred = AssertionRoleMode.USER
    else:
        inferred = AssertionRoleMode.MIXED
        reasons.append(
            "ambiguous_assistant_artifact"
            if _ASSISTANT_ARTIFACT_QUERY_RE.search(body)
            else "no_exclusive_authorship_signal"
        )
    effective = hint.role_mode if hint is not None and hint.role_mode else inferred
    if effective is not inferred:
        reasons.append("question_bound_role_hint")
    return AssertionRoleRoute(
        inferred_mode=inferred,
        effective_mode=effective,
        reasons=tuple(dict.fromkeys(reasons)),
        hint_applied=bool(hint is not None and hint.role_mode is not None),
    )


@dataclass(frozen=True, slots=True)
class _Clause:
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _FactExcerpt:
    text: str
    start: int
    end: int
    spans: tuple[_Clause, ...]


@dataclass(frozen=True, slots=True)
class _EligibleCandidate:
    input_ordinal: int
    row: ActivatedAssertionCandidate
    source_handle: str
    fact: _FactExcerpt
    reasons: tuple[AssertionReason, ...]
    overlap_count: int
    fact_token_count: int


def _sentence_spans(line: str) -> tuple[tuple[int, int], ...]:
    """Return conservative sentence spans without splitting numeric values."""

    url_core_spans: list[tuple[int, int]] = []
    for match in _URL_RE.finditer(line):
        end = match.end()
        while end > match.start() and line[end - 1] in ".,!;:'\"’”)]}":
            end -= 1
        if match.start() < end:
            url_core_spans.append((match.start(), end))

    boundaries: list[int] = []
    for index, character in enumerate(line):
        if character not in ".!?":
            continue
        if any(start <= index < end for start, end in url_core_spans):
            continue
        cursor = index + 1
        while cursor < len(line) and line[cursor] in "\"'’”)]}":
            cursor += 1
        following = line[cursor : cursor + 1]
        if following and not following.isspace():
            # This covers decimal points and punctuation clusters until their
            # final member, without rewriting the exact source text.
            continue
        if character == ".":
            token_match = re.search(r"[^\s]+$", line[: index + 1])
            token = "" if token_match is None else token_match.group(0).casefold()
            if token in _ABBREVIATIONS or re.fullmatch(
                r"(?:[a-z]\.){1,4}", token
            ):
                continue
        boundaries.append(cursor)
    spans: list[tuple[int, int]] = []
    start = 0
    for end in boundaries:
        if start < end:
            spans.append((start, end))
        start = end
    if start < len(line):
        spans.append((start, len(line)))
    return tuple(spans)


def _clauses(text: str) -> tuple[_Clause, ...]:
    output: list[_Clause] = []
    line_start = 0
    for line in text.splitlines(keepends=True):
        raw_line = line.rstrip("\r\n")
        pieces = _sentence_spans(raw_line)
        for raw_start, raw_end in pieces:
            raw = raw_line[raw_start:raw_end]
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())
            start = line_start + raw_start + leading
            end = line_start + raw_end - trailing
            if start < end and any(
                character.isalnum() for character in text[start:end]
            ):
                output.append(_Clause(text[start:end], start, end))
        line_start += len(line)
    if not output and text:
        stripped = text.strip()
        if stripped:
            start = text.find(stripped)
            output.append(_Clause(stripped, start, start + len(stripped)))
    return tuple(output)


def _pure_question(clause: _Clause) -> bool:
    without_urls = _URL_RE.sub("", clause.text)
    return "?" in without_urls or (
        _QUESTION_LEAD_RE.search(without_urls) is not None
        and without_urls.rstrip().endswith("?")
    )


def _feature_reasons(text: str) -> tuple[AssertionReason, ...]:
    checks = (
        (AssertionReason.ASSERTION, True),
        (AssertionReason.NUMERIC, _NUMBER_RE.search(text) is not None),
        (AssertionReason.DATE, _DATE_RE.search(text) is not None),
        (AssertionReason.ACTION, _ACTION_RE.search(text) is not None),
        (AssertionReason.URL, _URL_RE.search(text) is not None),
        (AssertionReason.ORDINAL, _ORDINAL_RE.search(text) is not None),
    )
    return tuple(reason for reason, active in checks if active)


def _grounded_user_value(
    clause: _Clause,
    *,
    question_terms: frozenset[str],
) -> bool:
    """Admit short answer-value turns without treating all prose as a fact."""

    clause_terms = set(_terms(clause.text))
    if len(clause_terms) > 12:
        return False
    typed_value = any(
        reason is not AssertionReason.ASSERTION
        for reason in _feature_reasons(clause.text)
    )
    # The source has already passed activation.  Very short categorical values
    # often share no surface token with a typed question ("What color?" ->
    # "Blue."), so retain them rather than forcing lexical overlap.
    return typed_value or bool(clause_terms & question_terms) or len(clause_terms) <= 4


def _is_proposed_clause(
    clause: _Clause,
    *,
    role: str,
    route: AssertionRoleRoute,
) -> bool:
    if _PROPOSED_RE.search(clause.text) is None:
        return False
    if _COMPLETED_FIRST_PERSON_PREFIX_RE.search(clause.text):
        return False
    return not (
        role == "assistant"
        and "explicit_assistant_recall" in route.reasons
        and _ASSISTANT_RECALLED_MODAL_RE.search(clause.text) is not None
    )


def _permitted_blocks(
    source_text: str,
    clauses: Sequence[_Clause],
    permitted: Sequence[_Clause],
) -> tuple[_Clause, ...]:
    """Join only adjacent permitted clauses into exact contiguous excerpts."""

    permitted_spans = {(clause.start, clause.end) for clause in permitted}
    groups: list[list[_Clause]] = []
    current: list[_Clause] = []
    for clause in clauses:
        if (clause.start, clause.end) in permitted_spans:
            current.append(clause)
            continue
        if current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return tuple(
        _Clause(
            source_text[group[0].start : group[-1].end],
            group[0].start,
            group[-1].end,
        )
        for group in groups
    )


def _best_clause(
    row: ActivatedAssertionCandidate,
    *,
    route: AssertionRoleRoute,
    question_terms: frozenset[str],
    policy: ActivatedAssertionPolicy,
    include_proposed: bool,
) -> tuple[_FactExcerpt | None, str | None]:
    role = row.role.casefold()
    if role not in {"user", "assistant"}:
        return None, "role_mismatch"

    clauses = _clauses(row.text)
    declarative = tuple(clause for clause in clauses if not _pure_question(clause))
    if not declarative:
        return None, "pure_question"

    if role == "user":
        declarative_spans = {(clause.start, clause.end) for clause in declarative}
        grounded_rows: list[_Clause] = []
        anchored = False
        filtered_proposal = False
        for clause in clauses:
            if (clause.start, clause.end) not in declarative_spans:
                anchored = False
                continue
            proposed = _is_proposed_clause(clause, role=role, route=route)
            direct = bool(_FIRST_PERSON_RE.search(clause.text)) or (
                _grounded_user_value(clause, question_terms=question_terms)
            )
            continuation = anchored
            if (include_proposed or not proposed) and (direct or continuation):
                grounded_rows.append(clause)
                anchored = direct or continuation
            else:
                filtered_proposal = filtered_proposal or bool(
                    proposed and (direct or continuation)
                )
                anchored = False
        grounded = tuple(grounded_rows)
        if not grounded:
            return (
                None,
                "proposed_or_hypothetical"
                if filtered_proposal
                else "not_first_person_or_grounded_value",
            )
        permitted = tuple(
            clause
            for clause in grounded
            if include_proposed
            or not _is_proposed_clause(clause, role=role, route=route)
        )
        if not permitted:
            return None, "proposed_or_hypothetical"
    else:
        permitted = tuple(
            clause
            for clause in declarative
            if include_proposed
            or not _is_proposed_clause(clause, role=role, route=route)
        )
        if not permitted:
            return None, "proposed_or_hypothetical"

    # Preserve an assistant list or URL as one exact unit only when every
    # parsed clause survived the question/proposal filters.
    if role == "assistant" and len(permitted) == len(clauses) and (
        _LIST_STRUCTURE_RE.search(row.text) or _URL_RE.search(row.text)
    ):
        stripped = row.text.strip()
        start = row.text.find(stripped)
        span = _Clause(stripped, start, start + len(stripped))
        return _FactExcerpt(
            text=stripped,
            start=span.start,
            end=span.end,
            spans=(span,),
        ), None

    blocks = _permitted_blocks(row.text, clauses, permitted)
    if not blocks:
        return None, "no_safe_declarative_block"
    # Every block contains only permitted clauses.  Joining non-contiguous
    # blocks with a newline preserves all operands without copying the filtered
    # question/proposal text between them; ``quote_spans`` binds each byte range.
    return _FactExcerpt(
        text="\n".join(block.text for block in blocks),
        start=blocks[0].start,
        end=blocks[-1].end,
        spans=blocks,
    ), None


def _fair_source_order(
    candidates: Sequence[_EligibleCandidate],
    *,
    reason: AssertionReason,
    source_handles: Sequence[str],
    route: AssertionRoleRoute,
) -> tuple[_EligibleCandidate, ...]:
    by_source: dict[str, list[_EligibleCandidate]] = defaultdict(list)
    for candidate in candidates:
        if reason in candidate.reasons:
            by_source[candidate.source_handle].append(candidate)
    for rows in by_source.values():
        rows.sort(
            key=lambda row: (
                (
                    route.effective_mode is not AssertionRoleMode.MIXED
                    and row.row.role.casefold() != route.effective_mode.value
                ),
                -row.overlap_count,
                -len(row.reasons),
                row.fact_token_count,
                row.input_ordinal,
                row.row.chunk_id,
            )
        )
    output: list[_EligibleCandidate] = []
    depth = 0
    while True:
        round_rows: list[_EligibleCandidate] = []
        for source_handle in source_handles:
            rows = by_source.get(source_handle, ())
            if depth < len(rows):
                round_rows.append(rows[depth])
        if not round_rows:
            break
        if route.effective_mode in {
            AssertionRoleMode.USER,
            AssertionRoleMode.ASSISTANT,
        }:
            primary_role = route.effective_mode.value
            round_rows.sort(
                key=lambda row: row.row.role.casefold() != primary_role
            )
        output.extend(round_rows)
        depth += 1
    return tuple(output)


class _TokenCounts:
    def __init__(self, callback: Callable[[str], int]) -> None:
        if not callable(callback):
            raise TypeError("count_tokens must be callable")
        self._callback = callback
        self._cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        if text not in self._cache:
            value = self._callback(text)
            if isinstance(value, bool) or type(value) is not int or value < 0:
                raise ValueError("count_tokens must return a non-negative integer")
            self._cache[text] = value
        return self._cache[text]


def _fact_budget_line(candidate: _EligibleCandidate) -> str:
    # Keep conversation time as explicitly labelled provenance. Feature
    # extraction above never reads it, so it cannot silently satisfy a date
    # lane or become an asserted event time.
    provenance = [candidate.source_handle, candidate.row.role.casefold()]
    if candidate.row.created_at:
        provenance.append(f"source-time={candidate.row.created_at}")
    return f"[{' | '.join(provenance)}] {candidate.fact.text}"


def project_activated_assertions(
    dated_question: str,
    active_source_ids: Sequence[str],
    candidates: Sequence[ActivatedAssertionCandidate],
    /,
    *,
    protected_chunk_ids: Sequence[str] = (),
    policy: ActivatedAssertionPolicy = ActivatedAssertionPolicy(),
    question_hint: QuestionAssertionHint | None = None,
    count_tokens: Callable[[str], int],
) -> ActivatedAssertionProjection:
    """Project bounded exact facts from already activated opaque sources.

    Every lane takes its own fair-source prefix before any protected, exact, or
    normalized-text duplicate is removed.  A lane that loses a selected row to
    deduplication refills only from its own tail.  The shared token budget is a
    final skip-and-continue packing step; it never changes the lane receipts.
    """

    _require_text(dated_question, "dated question")
    question_digest = _question_sha256(dated_question)
    sources = _ordered_unique_text(
        active_source_ids, "active source IDs", allow_empty=False
    )
    protected = _ordered_unique_text(protected_chunk_ids, "protected chunk IDs")
    if type(policy) is not ActivatedAssertionPolicy:
        raise TypeError("policy must be an exact ActivatedAssertionPolicy")
    if question_hint is not None:
        if type(question_hint) is not QuestionAssertionHint:
            raise TypeError("question_hint must be exact")
        if question_hint.dated_question_sha256 != question_digest:
            raise ValueError("question hint escaped its dated question")
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be an ordered candidate sequence")
    population = tuple(candidates)
    if any(type(row) is not ActivatedAssertionCandidate for row in population):
        raise TypeError(
            "candidates must contain exact ActivatedAssertionCandidate rows"
        )
    chunk_ids = tuple(row.chunk_id for row in population)
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("candidate population repeats a chunk ID")

    route = route_assertion_roles(dated_question, hint=question_hint)
    body = _DATED_QUESTION_RE.sub("", dated_question, count=1).strip()
    query_terms = set(_terms(body))
    if question_hint is not None:
        query_terms.update(question_hint.obligation_terms)
    frozen_question_terms = frozenset(query_terms)
    effective_include_proposed = (
        policy.include_proposed
        if question_hint is None or question_hint.include_proposed is None
        else question_hint.include_proposed
    )
    source_handle_by_id = {
        source_id: f"G{index:06d}" for index, source_id in enumerate(sources, 1)
    }
    source_handles = tuple(source_handle_by_id.values())
    token_counts = _TokenCounts(count_tokens)

    eligible: list[_EligibleCandidate] = []
    eligibility_omission: dict[str, str] = {}
    reasons_by_chunk: dict[str, tuple[AssertionReason, ...]] = {}
    for ordinal, row in enumerate(population):
        # Exact equality is the only operation ever performed on source IDs.
        source_handle = source_handle_by_id.get(row.source_id)
        if source_handle is None:
            eligibility_omission[row.chunk_id] = "inactive_source"
            continue
        observed_tokens = token_counts(row.text)
        if observed_tokens != row.token_count:
            raise ValueError(f"candidate token count changed: {row.chunk_id}")
        fact, omission = _best_clause(
            row,
            route=route,
            question_terms=frozen_question_terms,
            policy=policy,
            include_proposed=effective_include_proposed,
        )
        if fact is None:
            assert omission is not None
            eligibility_omission[row.chunk_id] = omission
            continue
        reasons = _feature_reasons(fact.text)
        fact_tokens = token_counts(fact.text)
        semantic_key = _semantic_text_key(fact.text)
        if not semantic_key:
            eligibility_omission[row.chunk_id] = "empty_normalized_fact"
            continue
        candidate = _EligibleCandidate(
            input_ordinal=ordinal,
            row=row,
            source_handle=source_handle,
            fact=fact,
            reasons=reasons,
            overlap_count=len(set(_terms(fact.text)) & frozen_question_terms),
            fact_token_count=fact_tokens,
        )
        eligible.append(candidate)
        reasons_by_chunk[row.chunk_id] = reasons

    eligible_by_id = {row.row.chunk_id: row for row in eligible}
    ranked_by_lane = {
        reason: _fair_source_order(
            eligible,
            reason=reason,
            source_handles=source_handles,
            route=route,
        )
        for reason, _budget in policy.lane_budgets
    }
    initial_by_lane = {
        reason: ranked_by_lane[reason][:budget]
        for reason, budget in policy.lane_budgets
    }

    protected_set = set(protected)
    claimed_ids = set(protected)
    retained_by_lane: dict[AssertionReason, list[_EligibleCandidate]] = {
        reason: [] for reason, _budget in policy.lane_budgets
    }
    duplicates_by_lane: dict[AssertionReason, dict[str, list[str]]] = {
        reason: {
            "protected": [],
            "exact": [],
            "semantic": [],
            "refill": [],
            "packing_refill": [],
            "token_unpacked": [],
        }
        for reason, _budget in policy.lane_budgets
    }
    selection_routes: dict[str, list[AssertionReason]] = defaultdict(list)
    union: list[_EligibleCandidate] = []

    def consider(
        candidate: _EligibleCandidate,
        reason: AssertionReason,
        *,
        refill: bool,
    ) -> bool:
        chunk_id = candidate.row.chunk_id
        selection_routes[chunk_id].append(reason)
        if chunk_id in claimed_ids:
            kind = "protected" if chunk_id in protected_set else "exact"
            duplicates_by_lane[reason][kind].append(chunk_id)
            return False
        # Text equality is not occurrence identity. Repeated statements can be
        # distinct events, so only exact chunk IDs participate in deduplication.
        claimed_ids.add(chunk_id)
        retained_by_lane[reason].append(candidate)
        union.append(candidate)
        if refill:
            duplicates_by_lane[reason]["refill"].append(chunk_id)
        return True

    # All independent prefixes exist before this cross-lane dedup pass.
    for reason, _budget in policy.lane_budgets:
        for candidate in initial_by_lane[reason]:
            consider(candidate, reason, refill=False)

    # A vacancy remains owned by the lane that lost it.
    for reason, budget in policy.lane_budgets:
        needed = budget - len(retained_by_lane[reason])
        if needed <= 0:
            continue
        for candidate in ranked_by_lane[reason][budget:]:
            if consider(candidate, reason, refill=True):
                needed -= 1
                if needed == 0:
                    break

    primary_union = tuple(union)
    retained_route_by_chunk = {
        row.row.chunk_id: reason
        for reason, rows in retained_by_lane.items()
        for row in rows
    }
    packed: list[_EligibleCandidate] = []
    packed_by_lane: dict[AssertionReason, list[_EligibleCandidate]] = {
        reason: [] for reason, _budget in policy.lane_budgets
    }
    token_omissions: set[str] = set()

    def try_pack(candidate: _EligibleCandidate, reason: AssertionReason) -> bool:
        if len(packed) >= policy.max_output_chunks:
            token_omissions.add(candidate.row.chunk_id)
            duplicates_by_lane[reason]["token_unpacked"].append(
                candidate.row.chunk_id
            )
            return False
        proposal = (*packed, candidate)
        rendered = "\n".join(_fact_budget_line(row) for row in proposal)
        if token_counts(rendered) <= policy.max_output_tokens:
            packed.append(candidate)
            packed_by_lane[reason].append(candidate)
            return True
        token_omissions.add(candidate.row.chunk_id)
        duplicates_by_lane[reason]["token_unpacked"].append(
            candidate.row.chunk_id
        )
        return False

    # Packing preserves lane ownership.  When a retained fact cannot fit, the
    # same lane continues down its ranked tail until a smaller, deduplicated
    # fact fills the vacancy or the tail is exhausted.
    for candidate in primary_union:
        reason = retained_route_by_chunk[candidate.row.chunk_id]
        if try_pack(candidate, reason):
            continue
        for replacement in ranked_by_lane[reason]:
            replacement_id = replacement.row.chunk_id
            if reason in selection_routes.get(replacement_id, ()):
                continue
            if not consider(replacement, reason, refill=False):
                continue
            retained_route_by_chunk[replacement_id] = reason
            if try_pack(replacement, reason):
                duplicates_by_lane[reason]["packing_refill"].append(
                    replacement_id
                )
                break

    final_payload = "\n".join(_fact_budget_line(row) for row in packed)
    payload_tokens = token_counts(final_payload) if final_payload else 0
    packed_ids = {row.row.chunk_id for row in packed}
    selected_facts = tuple(
        SelectedAssertionFact(
            chunk_id=row.row.chunk_id,
            source_handle=row.source_handle,
            role=row.row.role.casefold(),
            source_created_at=row.row.created_at,
            quote=row.fact.text,
            quote_start_char=row.fact.start,
            quote_end_char=row.fact.end,
            quote_spans=tuple(
                AssertionQuoteSpan(
                    quote=span.text,
                    start_char=span.start,
                    end_char=span.end,
                )
                for span in row.fact.spans
            ),
            input_text_sha256=quote_sha256(row.row.text),
            input_token_count=row.row.token_count,
            fact_token_count=row.fact_token_count,
            inclusion_reasons=row.reasons,
            selection_routes=tuple(dict.fromkeys(selection_routes[row.row.chunk_id])),
        )
        for row in packed
    )

    lane_audits = tuple(
        AssertionLaneAudit(
            reason=reason,
            budget=budget,
            ranked_candidate_chunk_ids=tuple(
                row.row.chunk_id for row in ranked_by_lane[reason]
            ),
            selected_before_dedup_chunk_ids=tuple(
                row.row.chunk_id for row in initial_by_lane[reason]
            ),
            retained_after_dedup_chunk_ids=tuple(
                row.row.chunk_id for row in retained_by_lane[reason]
            ),
            protected_duplicate_chunk_ids=tuple(
                duplicates_by_lane[reason]["protected"]
            ),
            exact_duplicate_chunk_ids=tuple(
                duplicates_by_lane[reason]["exact"]
            ),
            semantic_duplicate_chunk_ids=tuple(
                duplicates_by_lane[reason]["semantic"]
            ),
            refilled_chunk_ids=tuple(duplicates_by_lane[reason]["refill"]),
            packing_refilled_chunk_ids=tuple(
                duplicates_by_lane[reason]["packing_refill"]
            ),
            token_unpacked_chunk_ids=tuple(
                dict.fromkeys(duplicates_by_lane[reason]["token_unpacked"])
            ),
            packed_chunk_ids=tuple(
                row.row.chunk_id for row in packed_by_lane[reason]
            ),
            unfilled_slots=max(0, budget - len(packed_by_lane[reason])),
        )
        for reason, budget in policy.lane_budgets
    )

    candidate_audits: list[AssertionCandidateAudit] = []
    for ordinal, row in enumerate(population):
        source_handle = source_handle_by_id.get(row.source_id)
        reasons = reasons_by_chunk.get(row.chunk_id, ())
        routes = tuple(dict.fromkeys(selection_routes.get(row.chunk_id, ())))
        omission_reasons: list[str] = []
        duplicate_of = None
        if row.chunk_id in eligibility_omission:
            omission_reasons.append(eligibility_omission[row.chunk_id])
        elif row.chunk_id in protected_set:
            omission_reasons.append("protected_exact_duplicate")
        elif duplicate_of is not None and row.chunk_id not in retained_route_by_chunk:
            omission_reasons.append("normalized_text_duplicate")
        elif row.chunk_id not in retained_route_by_chunk:
            omission_reasons.append("lane_budget_not_selected")
        elif row.chunk_id in token_omissions:
            omission_reasons.append("token_budget_unpacked")
        included = row.chunk_id in packed_ids
        if included and omission_reasons:
            raise AssertionError("included assertion carries an omission")
        candidate_audits.append(
            AssertionCandidateAudit(
                input_ordinal=ordinal,
                chunk_id=row.chunk_id,
                source_handle=source_handle,
                source_id_sha256=quote_sha256(row.source_id),
                role=row.role.casefold(),
                text_sha256=quote_sha256(row.text),
                eligible_reasons=reasons,
                selection_routes=routes,
                included=included,
                omission_reasons=tuple(omission_reasons),
                duplicate_of_chunk_id=duplicate_of,
            )
        )

    population_sha = identity_sha256(
        [row.identity_projection() for row in population]
    )
    return ActivatedAssertionProjection(
        dated_question_sha256=question_digest,
        policy=policy,
        question_hint_receipt_sha256=(
            None if question_hint is None else question_hint.receipt_sha256
        ),
        effective_include_proposed=effective_include_proposed,
        role_route=route,
        active_source_bindings=tuple(
            (source_handle_by_id[source_id], quote_sha256(source_id))
            for source_id in sources
        ),
        candidate_population_sha256=population_sha,
        protected_chunk_ids=protected,
        selected_facts=selected_facts,
        lane_audits=lane_audits,
        candidate_audits=tuple(candidate_audits),
        payload_token_count=payload_tokens,
        token_budget_exhausted=bool(token_omissions),
    )


__all__ = [
    "ActivatedAssertionCandidate",
    "ActivatedAssertionPolicy",
    "ActivatedAssertionProjection",
    "AssertionCandidateAudit",
    "AssertionLaneAudit",
    "AssertionReason",
    "AssertionQuoteSpan",
    "AssertionRoleMode",
    "AssertionRoleRoute",
    "DEFAULT_LANE_BUDGETS",
    "MAX_OUTPUT_CHUNKS",
    "QuestionAssertionHint",
    "SelectedAssertionFact",
    "project_activated_assertions",
    "route_assertion_roles",
]
