"""Provider-free conjunctive event sufficiency with exact provenance.

This module is an additive answer-stage overlay.  It does not retrieve evidence
and it does not turn lexical overlap or story compatibility into an event
join.  Instead it models a question as required event edges, binds every edge
to an exact selected source span, and accepts a response value only when all
edges occur on one *proven* event-identity component.

Two notions of closure are intentionally independent:

``packing_closed``
    Every retained retrieval branch fitted the payload policy.

``support_frontier_closed``
    Every member of the declared support population was assessed by an exact
    typed-claim enumeration or a hard typed exclusion.  Embedding similarity,
    IDF pruning, candidate co-membership, and packing closure are not support
    certificates.

Consequently a closed packet alone can never authorize abstention.  Even a
support-closed abstention is scoped only to the requested conjunctive event;
it never licenses an inference that the underlying semantic fact is absent
from memory.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Sequence

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_operator_adapter import EvidenceHandleBinding, TypedEvidenceItem


OBLIGATION_FORMAT = "memory-condense-conjunctive-event-obligation-v1"
OVERLAY_FORMAT = "memory-condense-conjunctive-event-obligation-overlay-v1"
SOURCE_FORMAT = "memory-condense-exact-event-evidence-source-v1"
ANCHOR_FORMAT = "memory-condense-exact-event-anchor-v1"
CLAIM_FORMAT = "memory-condense-event-claim-binding-v1"
IDENTITY_LINK_FORMAT = "memory-condense-event-identity-link-receipt-v1"
SUPPORT_CELL_FORMAT = "memory-condense-event-support-cell-certificate-v1"
CLOSURE_FORMAT = "memory-condense-conjunctive-event-closure-receipt-v1"
ADVISORY_FORMAT = "memory-condense-scoped-insufficiency-advisory-v1"
DECISION_FORMAT = "memory-condense-conjunctive-event-decision-v1"
MECHANISM_ID = "conjunctive_event_identity_and_sufficiency_v1"

_HANDLE_RE = re.compile(r"^H[0-9]{3,6}$")
_GROUP_RE = re.compile(r"^G[0-9]{3,6}$")
_RELATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DATED_RE = re.compile(
    r"^\[Question asked at .+?\]\s*", re.IGNORECASE | re.DOTALL
)
_PRESENTATION_QUESTION_RE = re.compile(
    r"^(?:at\s+)?which\s+(?P<venue_type>[A-Za-z][A-Za-z0-9_-]*)\s+"
    r"did\s+(?P<actor>I|we)\s+(?P<action>present)\s+"
    r"(?:a|an|the|my|our)\s+(?P<theme>[A-Za-z][A-Za-z0-9_-]*)\s+"
    r"(?:for|on|about)\s+(?:my|our|the)?\s*(?P<context>.+?)\??$",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.DOTALL)
_FIRST_PERSON_RE = re.compile(r"\b(?:I(?:['’](?:m|ve|d|ll))?|my|we|our)\b", re.I)
_PRESENT_RE = re.compile(r"\bpresent(?:ed|ing|s)?\b", re.I)
_POSTER_RE = re.compile(r"\bposter(?:s)?\b", re.I)
_PROJECT_CHAIN_RE = re.compile(
    r"\bposter(?:s)?\b.{0,160}?\b(?:for|on|about)\s+"
    r"(?:my|our|the)?\s*(?P<chain>[^.!?]{0,120}?\bproject(?:s)?\b)",
    re.I | re.DOTALL,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


def _ordered_unique_sha256(
    values: tuple[str, ...], label: str
) -> tuple[str, ...]:
    _require(type(values) is tuple, f"{label} must be an immutable tuple")
    for value in values:
        require_sha256(value, label)
    _require(len(set(values)) == len(values), f"{label} must be ordered and unique")
    return values


def _canonical_value(value: str) -> str:
    exact = " ".join(require_text(value, "event value").casefold().split())
    aliases = {
        "i": "user",
        "me": "user",
        "myself": "user",
        "undergrad": "undergraduate",
        "presented": "present",
        "presenting": "present",
        "posters": "poster",
        "projects": "project",
    }
    return aliases.get(exact, exact)


def _question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


class ClaimAssertionBasis(str, Enum):
    DETERMINISTIC_EXACT_PHRASE = "deterministic_exact_phrase"
    VALIDATED_TYPED_ASSERTION = "validated_typed_assertion"


class EventIdentityStatus(str, Enum):
    PROVEN_SAME_EVENT = "proven_same_event"
    COMPATIBLE_UNPROVEN = "compatible_unproven"
    PROVEN_DISTINCT_EVENT = "proven_distinct_event"


class EventIdentityBasis(str, Enum):
    EXACT_SHARED_EVENT_IDENTIFIER = "exact_shared_event_identifier"
    EXPLICIT_CROSS_SOURCE_REFERENCE = "explicit_cross_source_reference"
    STORY_CANDIDATE_COMEMBERSHIP = "story_candidate_comembership"
    TEMPORAL_SEMANTIC_COMPATIBILITY = "temporal_semantic_compatibility"
    EXPLICIT_DISTINCT_EVENT_ASSERTION = "explicit_distinct_event_assertion"


class SupportCellDisposition(str, Enum):
    EXACT_TYPED_CLAIMS_ENUMERATED = "exact_typed_claims_enumerated"
    HARD_TYPED_NO_SUPPORT = "hard_typed_no_support"
    UNRESOLVED = "unresolved"


class HardSupportExclusionBasis(str, Enum):
    EXACT_ROLE_MISMATCH = "exact_role_mismatch"
    EXACT_EVENT_TYPE_MISMATCH = "exact_event_type_mismatch"
    EXACT_RELATION_SCHEMA_MISMATCH = "exact_relation_schema_mismatch"
    EXACT_TEMPORAL_SCOPE_MISMATCH = "exact_temporal_scope_mismatch"


class EventDecisionDisposition(str, Enum):
    KEEP_PARENT = "keep_parent"
    REPLACE = "replace"
    ABSTAIN = "abstain"


class EventDecisionReason(str, Enum):
    COMPLETE_EVENT_MATCHES_PARENT = "complete_event_matches_parent"
    COMPLETE_EVENT_REPLACES_PARENT = "complete_event_replaces_parent"
    SUPPORT_CLOSED_EVENT_UNRESOLVED = "support_closed_event_unresolved"
    SUPPORT_OPEN_EVENT_UNRESOLVED = "support_open_event_unresolved"
    SUPPORT_CLOSED_EVENT_CONFLICT = "support_closed_event_conflict"
    SUPPORT_OPEN_EVENT_CONFLICT = "support_open_event_conflict"


@dataclass(frozen=True, slots=True)
class EventEdgeObligationV1:
    """One required edge in an all-of, same-event question program."""

    relation: str
    required_value: str | None
    answer_variable: bool
    answer_value_type: str | None = None
    obligation_id: str = ""
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.relation) is str and _RELATION_RE.fullmatch(self.relation),
            "event obligation relation must be canonical snake case",
        )
        _require(type(self.answer_variable) is bool, "answer-variable flag changed")
        if self.answer_variable:
            _require(
                self.required_value is None and self.answer_value_type is not None,
                "event answer variable must have only a value type",
            )
            require_text(self.answer_value_type or "", "event answer value type")
        else:
            _require(
                self.required_value is not None and self.answer_value_type is None,
                "event constraint must have only a required value",
            )
            require_text(self.required_value or "", "event required value")
        body = self.projection(include_identity=False, include_receipt=False)
        expected_id = identity_sha256(body)
        if self.obligation_id:
            _require(self.obligation_id == expected_id, "event obligation ID changed")
        object.__setattr__(self, "obligation_id", expected_id)
        expected_receipt = identity_sha256(
            self.projection(include_identity=True, include_receipt=False)
        )
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected_receipt,
                "event obligation receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected_receipt)

    def projection(
        self,
        *,
        include_identity: bool = True,
        include_receipt: bool = True,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "answer_value_type": self.answer_value_type,
            "answer_variable": self.answer_variable,
            "format": OBLIGATION_FORMAT,
            "relation": self.relation,
            "required_value": self.required_value,
            "same_event_required": True,
        }
        if include_identity:
            value["obligation_id"] = self.obligation_id
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ConjunctiveEventObligationOverlayV1:
    """Question-only event program; every edge is mandatory on one event."""

    question: str
    obligations: tuple[EventEdgeObligationV1, ...]
    question_sha256: str = ""
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.question, "conjunctive-event question")
        _require(
            type(self.obligations) is tuple
            and len(self.obligations) >= 2
            and all(type(row) is EventEdgeObligationV1 for row in self.obligations),
            "conjunctive-event obligations changed type",
        )
        _require(
            len({row.obligation_id for row in self.obligations})
            == len(self.obligations),
            "conjunctive-event obligations repeat",
        )
        _require(
            sum(row.answer_variable for row in self.obligations) == 1,
            "conjunctive-event program requires exactly one answer variable",
        )
        expected_question = _question_sha256(self.question)
        if self.question_sha256:
            _require(
                self.question_sha256 == expected_question,
                "conjunctive-event question digest changed",
            )
        object.__setattr__(self, "question_sha256", expected_question)
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "conjunctive-event overlay escaped its provider-free zero-state boundary",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "event overlay receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="conjunctive_event_overlay")

    @property
    def answer_obligation(self) -> EventEdgeObligationV1:
        return next(row for row in self.obligations if row.answer_variable)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "composition": "all_required_edges_on_one_proven_event_identity_component",
            "format": OVERLAY_FORMAT,
            "lexical_n_of_m_shortcut_allowed": False,
            "mechanism_id": MECHANISM_ID,
            "obligations": [row.projection() for row in self.obligations],
            "provider_prompt_count": 0,
            "question": self.question,
            "question_sha256": self.question_sha256,
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def compile_conjunctive_event_obligation_overlay(
    question: str,
) -> ConjunctiveEventObligationOverlayV1 | None:
    """Compile the supported presentation-event form from question text only.

    The compiler is deliberately narrow and returns ``None`` for unsupported
    syntax.  It does not fall back to an N-of-M bag of words.
    """

    require_text(question, "conjunctive-event question")
    body = _DATED_RE.sub("", question).strip()
    match = _PRESENTATION_QUESTION_RE.fullmatch(body)
    if match is None:
        return None
    actor = "user" if match.group("actor").casefold() == "i" else "users"
    context_terms = tuple(
        _canonical_value(row)
        for row in re.findall(r"[A-Za-z0-9_-]+", match.group("context"))
    )
    if not context_terms:
        return None
    head = context_terms[-1]
    qualifiers = tuple(dict.fromkeys(context_terms[:-1]))
    obligations = [
        EventEdgeObligationV1("actor", actor, False),
        EventEdgeObligationV1(
            "action", _canonical_value(match.group("action")), False
        ),
        EventEdgeObligationV1(
            "theme", _canonical_value(match.group("theme")), False
        ),
        EventEdgeObligationV1("theme_about", head, False),
    ]
    obligations.extend(
        EventEdgeObligationV1("theme_about_qualifier", row, False)
        for row in qualifiers
    )
    obligations.append(
        EventEdgeObligationV1(
            "venue",
            None,
            True,
            _canonical_value(match.group("venue_type")),
        )
    )
    return ConjunctiveEventObligationOverlayV1(question, tuple(obligations))


@dataclass(frozen=True, slots=True)
class ExactEventEvidenceSourceV1:
    """One exact selected source and its prompt-external lineage receipt."""

    handle_id: str
    source_group_handle: str
    exact_text: str
    role: str
    lineage_receipt_sha256: str
    exact_text_sha256: str = ""
    source_member_sha256: str = ""
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(_HANDLE_RE.fullmatch(self.handle_id) is not None, "source handle changed")
        _require(
            _GROUP_RE.fullmatch(self.source_group_handle) is not None,
            "source group handle changed",
        )
        require_text(self.exact_text, "exact event source text")
        _require(self.role in {"user", "assistant", "unknown"}, "source role changed")
        require_sha256(self.lineage_receipt_sha256, "source lineage receipt")
        expected_text = hashlib.sha256(self.exact_text.encode("utf-8")).hexdigest()
        if self.exact_text_sha256:
            _require(self.exact_text_sha256 == expected_text, "source text digest changed")
        object.__setattr__(self, "exact_text_sha256", expected_text)
        member_body = {
            "exact_text_sha256": expected_text,
            "handle_id": self.handle_id,
            "lineage_receipt_sha256": self.lineage_receipt_sha256,
            "role": self.role,
            "source_group_handle": self.source_group_handle,
        }
        expected_member = identity_sha256(member_body)
        if self.source_member_sha256:
            _require(
                self.source_member_sha256 == expected_member,
                "source member identity changed",
            )
        object.__setattr__(self, "source_member_sha256", expected_member)
        expected = identity_sha256(self.local_projection(include_text=False, include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "event source receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def local_projection(
        self,
        *,
        include_text: bool = True,
        include_receipt: bool = True,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "exact_text_sha256": self.exact_text_sha256,
            "format": SOURCE_FORMAT,
            "handle_id": self.handle_id,
            "lineage_receipt_sha256": self.lineage_receipt_sha256,
            "role": self.role,
            "source_group_handle": self.source_group_handle,
            "source_member_sha256": self.source_member_sha256,
        }
        if include_text:
            value["exact_text"] = self.exact_text
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def exact_event_sources_from_typed_evidence(
    items: Sequence[TypedEvidenceItem],
    bindings: Sequence[EvidenceHandleBinding],
) -> tuple[ExactEventEvidenceSourceV1, ...]:
    """Adapt exact one-handle typed items without weakening provenance.

    A typed item is admissible only when its summary is byte-identical to the
    local binding's citation.  Multi-handle summaries and compressed or
    paraphrased text fail closed because no single source span can authorize
    their event edges.
    """

    exact_items = tuple(items)
    exact_bindings = tuple(bindings)
    _require(
        all(type(row) is TypedEvidenceItem for row in exact_items),
        "typed event items changed type",
    )
    _require(
        all(type(row) is EvidenceHandleBinding for row in exact_bindings),
        "typed event bindings changed type",
    )
    binding_by_handle = {row.handle_id: row for row in exact_bindings}
    _require(
        len(binding_by_handle) == len(exact_bindings),
        "typed event bindings repeat handles",
    )
    sources: list[ExactEventEvidenceSourceV1] = []
    seen_handles: set[str] = set()
    for item in exact_items:
        _require(
            len(item.handle_ids) == 1,
            "typed event item must bind one exact source handle",
        )
        handle = item.handle_ids[0]
        _require(handle in binding_by_handle, "typed event item has no local binding")
        binding = binding_by_handle[handle]
        exact_sha = hashlib.sha256(item.summary.encode("utf-8")).hexdigest()
        _require(
            exact_sha == binding.citation_sha256
            and len(item.summary) == binding.citation_char_count,
            "typed event summary is not the exact bound citation",
        )
        _require(handle not in seen_handles, "typed event source handle repeats")
        seen_handles.add(handle)
        relation = item.relation or ""
        role = (
            "user"
            if "authored_by_user" in relation or "memory_role:user" in relation
            else "assistant"
            if "authored_by_assistant" in relation
            or "memory_role:assistant" in relation
            else "unknown"
        )
        sources.append(
            ExactEventEvidenceSourceV1(
                handle,
                binding.source_group_handle,
                item.summary,
                role,
                binding.receipt_sha256,
            )
        )
    return tuple(sources)


@dataclass(frozen=True, slots=True)
class ExactEventAnchorV1:
    """A byte-exact event span within one selected source."""

    source: ExactEventEvidenceSourceV1
    span_start: int
    span_end: int
    exact_quote: str
    event_key_sha256: str = ""
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.source) is ExactEventEvidenceSourceV1, "event anchor source changed")
        _require(
            type(self.span_start) is int
            and type(self.span_end) is int
            and 0 <= self.span_start < self.span_end <= len(self.source.exact_text),
            "event anchor coordinates changed",
        )
        require_text(self.exact_quote, "exact event anchor quote")
        _require(
            self.source.exact_text[self.span_start : self.span_end] == self.exact_quote,
            "event anchor quote is not the exact source span",
        )
        key_body = {
            "format": ANCHOR_FORMAT,
            "source_member_sha256": self.source.source_member_sha256,
            "span_end": self.span_end,
            "span_start": self.span_start,
        }
        expected_key = identity_sha256(key_body)
        if self.event_key_sha256:
            _require(self.event_key_sha256 == expected_key, "event key changed")
        object.__setattr__(self, "event_key_sha256", expected_key)
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "event anchor receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "event_key_sha256": self.event_key_sha256,
            "exact_quote": self.exact_quote,
            "exact_quote_sha256": hashlib.sha256(
                self.exact_quote.encode("utf-8")
            ).hexdigest(),
            "format": ANCHOR_FORMAT,
            "handle_id": self.source.handle_id,
            "lineage_receipt_sha256": self.source.lineage_receipt_sha256,
            "source_group_handle": self.source.source_group_handle,
            "source_member_sha256": self.source.source_member_sha256,
            "source_text_sha256": self.source.exact_text_sha256,
            "span_end": self.span_end,
            "span_start": self.span_start,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class EventClaimBindingV1:
    """One typed event edge grounded in an exact selected source substring."""

    anchor: ExactEventAnchorV1
    obligation_id: str
    relation: str
    value: str
    value_type: str | None
    quote_start: int
    quote_end: int
    exact_quote: str
    assertion_basis: ClaimAssertionBasis
    semantic_assertion_receipt_sha256: str
    claim_id: str = ""
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.anchor) is ExactEventAnchorV1, "event claim anchor changed")
        require_sha256(self.obligation_id, "event claim obligation")
        _require(
            type(self.relation) is str and _RELATION_RE.fullmatch(self.relation),
            "event claim relation changed",
        )
        require_text(self.value, "event claim value")
        if self.value_type is not None:
            require_text(self.value_type, "event claim value type")
        _require(
            type(self.quote_start) is int
            and type(self.quote_end) is int
            and self.anchor.span_start <= self.quote_start < self.quote_end <= self.anchor.span_end,
            "event claim quote coordinates changed",
        )
        require_text(self.exact_quote, "exact event claim quote")
        source = self.anchor.source.exact_text
        _require(
            source[self.quote_start : self.quote_end] == self.exact_quote,
            "event claim quote is not the exact source span",
        )
        _require(
            type(self.assertion_basis) is ClaimAssertionBasis,
            "event claim assertion basis changed",
        )
        require_sha256(
            self.semantic_assertion_receipt_sha256,
            "event semantic assertion receipt",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "event claim escaped its provider-free zero-state boundary",
        )
        body = self.projection(include_identity=False, include_receipt=False)
        expected_claim = identity_sha256(body)
        if self.claim_id:
            _require(self.claim_id == expected_claim, "event claim ID changed")
        object.__setattr__(self, "claim_id", expected_claim)
        expected = identity_sha256(
            self.projection(include_identity=True, include_receipt=False)
        )
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "event claim receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="event_claim_binding")

    @property
    def event_key_sha256(self) -> str:
        return self.anchor.event_key_sha256

    @property
    def source_member_sha256(self) -> str:
        return self.anchor.source.source_member_sha256

    def projection(
        self,
        *,
        include_identity: bool = True,
        include_receipt: bool = True,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "anchor_receipt_sha256": self.anchor.receipt_sha256,
            "assertion_basis": self.assertion_basis.value,
            "event_key_sha256": self.event_key_sha256,
            "exact_quote": self.exact_quote,
            "exact_quote_sha256": hashlib.sha256(
                self.exact_quote.encode("utf-8")
            ).hexdigest(),
            "format": CLAIM_FORMAT,
            "handle_id": self.anchor.source.handle_id,
            "lineage_receipt_sha256": self.anchor.source.lineage_receipt_sha256,
            "obligation_id": self.obligation_id,
            "provider_prompt_count": 0,
            "quote_end": self.quote_end,
            "quote_start": self.quote_start,
            "relation": self.relation,
            "retained_transformer_token_state_bytes": 0,
            "semantic_assertion_receipt_sha256": (
                self.semantic_assertion_receipt_sha256
            ),
            "source_group_handle": self.anchor.source.source_group_handle,
            "source_member_sha256": self.source_member_sha256,
            "source_text_sha256": self.anchor.source.exact_text_sha256,
            "value": self.value,
            "value_type": self.value_type,
        }
        if include_identity:
            value["claim_id"] = self.claim_id
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _exact_span(text: str, match: re.Match[str], group: int | str = 0) -> tuple[int, int, str]:
    start, end = match.span(group)
    return start, end, text[start:end]


def _event_anchor(source: ExactEventEvidenceSourceV1, start: int, end: int) -> ExactEventAnchorV1:
    while start < end and source.exact_text[start].isspace():
        start += 1
    while end > start and source.exact_text[end - 1].isspace():
        end -= 1
    return ExactEventAnchorV1(source, start, end, source.exact_text[start:end])


def _bind_extracted_claim(
    overlay: ConjunctiveEventObligationOverlayV1,
    anchor: ExactEventAnchorV1,
    obligation: EventEdgeObligationV1,
    *,
    value: str,
    value_type: str | None,
    span: tuple[int, int, str],
) -> EventClaimBindingV1:
    start, end, quote = span
    assertion_receipt = identity_sha256(
        {
            "extractor": "question_bound_presentation_exact_phrase_v1",
            "obligation_id": obligation.obligation_id,
            "overlay_receipt_sha256": overlay.receipt_sha256,
            "quote_end": end,
            "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
            "quote_start": start,
            "source_member_sha256": anchor.source.source_member_sha256,
            "value": value,
            "value_type": value_type,
        }
    )
    return EventClaimBindingV1(
        anchor,
        obligation.obligation_id,
        obligation.relation,
        value,
        value_type,
        start,
        end,
        quote,
        ClaimAssertionBasis.DETERMINISTIC_EXACT_PHRASE,
        assertion_receipt,
    )


def extract_deterministic_presentation_event_claims(
    overlay: ConjunctiveEventObligationOverlayV1,
    sources: Sequence[ExactEventEvidenceSourceV1],
) -> tuple[EventClaimBindingV1, ...]:
    """Extract the supported presentation-event edges without a model call.

    This is intentionally an exact, narrow extractor.  Unsupported phrasing
    remains unresolved; it is never converted to partial lexical satisfaction.
    """

    _require(
        type(overlay) is ConjunctiveEventObligationOverlayV1,
        "event extraction overlay changed type",
    )
    exact_sources = tuple(sources)
    _require(
        all(type(row) is ExactEventEvidenceSourceV1 for row in exact_sources),
        "event extraction sources changed type",
    )
    obligations_by_relation: dict[str, list[EventEdgeObligationV1]] = {}
    for obligation in overlay.obligations:
        obligations_by_relation.setdefault(obligation.relation, []).append(obligation)

    claims: list[EventClaimBindingV1] = []
    for source in exact_sources:
        for sentence_match in _SENTENCE_RE.finditer(source.exact_text):
            anchor = _event_anchor(source, *sentence_match.span())
            text = source.exact_text
            local_start, local_end = anchor.span_start, anchor.span_end
            sentence = text[local_start:local_end]

            def absolute(match: re.Match[str], group: int | str = 0) -> tuple[int, int, str]:
                start, end = match.span(group)
                start += local_start
                end += local_start
                return start, end, text[start:end]

            first_person = _FIRST_PERSON_RE.search(sentence)
            if source.role == "user" and first_person is not None:
                for obligation in obligations_by_relation.get("actor", []):
                    if _canonical_value(obligation.required_value or "") == "user":
                        claims.append(
                            _bind_extracted_claim(
                                overlay,
                                anchor,
                                obligation,
                                value="user",
                                value_type=None,
                                span=absolute(first_person),
                            )
                        )

            present = _PRESENT_RE.search(sentence)
            if present is not None:
                for obligation in obligations_by_relation.get("action", []):
                    if _canonical_value(obligation.required_value or "") == "present":
                        claims.append(
                            _bind_extracted_claim(
                                overlay,
                                anchor,
                                obligation,
                                value="present",
                                value_type=None,
                                span=absolute(present),
                            )
                        )

            poster = _POSTER_RE.search(sentence)
            if poster is not None:
                for obligation in obligations_by_relation.get("theme", []):
                    if _canonical_value(obligation.required_value or "") == "poster":
                        claims.append(
                            _bind_extracted_claim(
                                overlay,
                                anchor,
                                obligation,
                                value="poster",
                                value_type=None,
                                span=absolute(poster),
                            )
                        )

            chain = _PROJECT_CHAIN_RE.search(sentence)
            if chain is not None:
                project_match = re.search(r"\bproject(?:s)?\b", chain.group("chain"), re.I)
                _require(project_match is not None, "project-chain extractor changed")
                chain_start = local_start + chain.start("chain")
                project_start = chain_start + project_match.start()
                project_end = chain_start + project_match.end()
                project_span = (project_start, project_end, text[project_start:project_end])
                for obligation in obligations_by_relation.get("theme_about", []):
                    if _canonical_value(obligation.required_value or "") == "project":
                        claims.append(
                            _bind_extracted_claim(
                                overlay,
                                anchor,
                                obligation,
                                value="project",
                                value_type=None,
                                span=project_span,
                            )
                        )
                chain_text = chain.group("chain")
                for obligation in obligations_by_relation.get(
                    "theme_about_qualifier", []
                ):
                    wanted = _canonical_value(obligation.required_value or "")
                    aliases = {
                        "undergraduate": r"\bundergrad(?:uate)?\b",
                        "course": r"\bcourse\b",
                        "research": r"\bresearch\b",
                    }
                    pattern = aliases.get(wanted, rf"\b{re.escape(wanted)}\b")
                    qualifier = re.search(pattern, chain_text, re.I)
                    if qualifier is None:
                        continue
                    start = chain_start + qualifier.start()
                    end = chain_start + qualifier.end()
                    claims.append(
                        _bind_extracted_claim(
                            overlay,
                            anchor,
                            obligation,
                            value=wanted,
                            value_type=None,
                            span=(start, end, text[start:end]),
                        )
                    )

            for obligation in obligations_by_relation.get("venue", []):
                if not obligation.answer_variable:
                    continue
                venue_type = re.escape(obligation.answer_value_type or "")
                venue_re = re.compile(
                    rf"\b(?:at|to|hosted\s+by)\s+"
                    rf"(?P<value>(?:[A-Z][A-Za-z.'’\-]*\s+){{0,5}}{venue_type})\b",
                    re.I,
                )
                venue = venue_re.search(sentence)
                if venue is None:
                    continue
                value = venue.group("value")
                claims.append(
                    _bind_extracted_claim(
                        overlay,
                        anchor,
                        obligation,
                        value=value,
                        value_type=obligation.answer_value_type,
                        span=absolute(venue, "value"),
                    )
                )
    return tuple(claims)


@dataclass(frozen=True, slots=True)
class EventIdentityLinkReceiptV1:
    """A typed identity relation; compatibility is never identity proof."""

    left_event_key_sha256: str
    right_event_key_sha256: str
    status: EventIdentityStatus
    basis: EventIdentityBasis
    witness_receipt_sha256: str | None
    identity_proven: bool
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.left_event_key_sha256, "left event key")
        require_sha256(self.right_event_key_sha256, "right event key")
        _require(
            self.left_event_key_sha256 < self.right_event_key_sha256,
            "event identity keys must be distinct and canonically ordered",
        )
        _require(type(self.status) is EventIdentityStatus, "event identity status changed")
        _require(type(self.basis) is EventIdentityBasis, "event identity basis changed")
        proven_bases = {
            EventIdentityBasis.EXACT_SHARED_EVENT_IDENTIFIER,
            EventIdentityBasis.EXPLICIT_CROSS_SOURCE_REFERENCE,
        }
        compatible_bases = {
            EventIdentityBasis.STORY_CANDIDATE_COMEMBERSHIP,
            EventIdentityBasis.TEMPORAL_SEMANTIC_COMPATIBILITY,
        }
        if self.status is EventIdentityStatus.PROVEN_SAME_EVENT:
            _require(self.basis in proven_bases, "compatible evidence cannot prove event identity")
            require_sha256(self.witness_receipt_sha256 or "", "event identity witness")
        elif self.status is EventIdentityStatus.COMPATIBLE_UNPROVEN:
            _require(self.basis in compatible_bases, "event compatibility basis changed")
            _require(self.witness_receipt_sha256 is None, "compatibility cannot carry identity proof")
        else:
            _require(
                self.basis is EventIdentityBasis.EXPLICIT_DISTINCT_EVENT_ASSERTION,
                "distinct-event basis changed",
            )
            require_sha256(self.witness_receipt_sha256 or "", "distinct-event witness")
        expected_proven = self.status is EventIdentityStatus.PROVEN_SAME_EVENT
        _require(self.identity_proven is expected_proven, "event identity proof flag changed")
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "event identity link escaped its provider-free zero-state boundary",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "event identity receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "basis": self.basis.value,
            "format": IDENTITY_LINK_FORMAT,
            "identity_proven": self.identity_proven,
            "left_event_key_sha256": self.left_event_key_sha256,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "right_event_key_sha256": self.right_event_key_sha256,
            "status": self.status.value,
            "witness_receipt_sha256": self.witness_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def event_identity_link(
    left_event_key_sha256: str,
    right_event_key_sha256: str,
    *,
    status: EventIdentityStatus,
    basis: EventIdentityBasis,
    witness_receipt_sha256: str | None = None,
) -> EventIdentityLinkReceiptV1:
    left, right = sorted((left_event_key_sha256, right_event_key_sha256))
    return EventIdentityLinkReceiptV1(
        left,
        right,
        status,
        basis,
        witness_receipt_sha256,
        status is EventIdentityStatus.PROVEN_SAME_EVENT,
    )


@dataclass(frozen=True, slots=True)
class EventSupportCellCertificateV1:
    """Exact typed-support accounting for one complete population cell."""

    cell_id_sha256: str
    population_member_sha256s: tuple[str, ...]
    assessed_member_sha256s: tuple[str, ...]
    claim_receipt_sha256s: tuple[str, ...]
    disposition: SupportCellDisposition
    hard_exclusion_basis: HardSupportExclusionBasis | None
    assessment_receipt_sha256: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.cell_id_sha256, "support cell ID")
        members = _ordered_unique_sha256(
            self.population_member_sha256s, "support cell members"
        )
        assessed = _ordered_unique_sha256(
            self.assessed_member_sha256s, "support assessed members"
        )
        _ordered_unique_sha256(
            self.claim_receipt_sha256s, "support cell claim receipts"
        )
        _require(bool(members), "support cell cannot be empty")
        _require(set(assessed) <= set(members), "support assessment escaped its cell")
        _require(
            type(self.disposition) is SupportCellDisposition,
            "support cell disposition changed",
        )
        if self.disposition is SupportCellDisposition.UNRESOLVED:
            _require(
                self.hard_exclusion_basis is None
                and self.assessment_receipt_sha256 is None,
                "unresolved support cell cannot carry a closure certificate",
            )
        else:
            _require(set(assessed) == set(members), "closed support cell was not fully assessed")
            require_sha256(
                self.assessment_receipt_sha256 or "", "support assessment receipt"
            )
            if self.disposition is SupportCellDisposition.HARD_TYPED_NO_SUPPORT:
                _require(
                    type(self.hard_exclusion_basis) is HardSupportExclusionBasis,
                    "hard no-support cell requires an exact typed basis",
                )
                _require(
                    not self.claim_receipt_sha256s,
                    "hard no-support cell cannot contain support claims",
                )
            else:
                _require(
                    self.hard_exclusion_basis is None,
                    "enumerated support cell cannot carry an exclusion basis",
                )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "support cell receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def support_closed(self) -> bool:
        return self.disposition is not SupportCellDisposition.UNRESOLVED

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "assessed_member_sha256s": list(self.assessed_member_sha256s),
            "assessment_receipt_sha256": self.assessment_receipt_sha256,
            "cell_id_sha256": self.cell_id_sha256,
            "claim_receipt_sha256s": list(self.claim_receipt_sha256s),
            "disposition": self.disposition.value,
            "format": SUPPORT_CELL_FORMAT,
            "hard_exclusion_basis": (
                self.hard_exclusion_basis.value
                if self.hard_exclusion_basis is not None
                else None
            ),
            "population_member_sha256s": list(self.population_member_sha256s),
            "support_closed": self.support_closed,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ConjunctiveEventClosureReceiptV1:
    """Separate packing accounting from exact typed-support accounting."""

    population_identity_sha256: str
    population_member_sha256s: tuple[str, ...]
    cells: tuple[EventSupportCellCertificateV1, ...]
    packing_closed: bool
    support_frontier_closed: bool
    semantic_absence_may_be_inferred: Literal[False] = False
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.population_identity_sha256, "event support population")
        members = _ordered_unique_sha256(
            self.population_member_sha256s, "event support population members"
        )
        _require(bool(members), "event support population cannot be empty")
        _require(
            type(self.cells) is tuple
            and bool(self.cells)
            and all(type(row) is EventSupportCellCertificateV1 for row in self.cells),
            "event support cells changed type",
        )
        cell_members = tuple(
            member for cell in self.cells for member in cell.population_member_sha256s
        )
        _require(
            len(set(cell_members)) == len(cell_members)
            and set(cell_members) == set(members),
            "event support cells do not exactly partition the population",
        )
        _require(
            type(self.packing_closed) is bool
            and type(self.support_frontier_closed) is bool,
            "event closure flags changed type",
        )
        expected_support = all(row.support_closed for row in self.cells)
        _require(
            self.support_frontier_closed is expected_support,
            "support closure is not justified by every population cell",
        )
        _require(
            self.semantic_absence_may_be_inferred is False,
            "conjunctive support closure cannot prove semantic absence",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "event closure escaped its provider-free zero-state boundary",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "event closure receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="conjunctive_event_closure")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "cells": [row.projection() for row in self.cells],
            "format": CLOSURE_FORMAT,
            "packing_closed": self.packing_closed,
            "population_identity_sha256": self.population_identity_sha256,
            "population_member_sha256s": list(self.population_member_sha256s),
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "scope": "requested_conjunctive_event_binding_over_declared_typed_support_population",
            "semantic_absence_may_be_inferred": False,
            "support_frontier_closed": self.support_frontier_closed,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def build_conjunctive_event_closure_receipt(
    population_identity_sha256: str,
    sources: Sequence[ExactEventEvidenceSourceV1],
    claims: Sequence[EventClaimBindingV1],
    *,
    packing_closed: bool,
    unresolved_member_sha256s: Sequence[str] = (),
    hard_exclusions: dict[str, HardSupportExclusionBasis] | None = None,
) -> ConjunctiveEventClosureReceiptV1:
    """Build one-member cells for an exact declared typed support population."""

    require_sha256(population_identity_sha256, "event support population")
    exact_sources = tuple(sources)
    exact_claims = tuple(claims)
    _require(
        bool(exact_sources)
        and all(type(row) is ExactEventEvidenceSourceV1 for row in exact_sources),
        "event closure sources changed type",
    )
    _require(
        all(type(row) is EventClaimBindingV1 for row in exact_claims),
        "event closure claims changed type",
    )
    members = tuple(row.source_member_sha256 for row in exact_sources)
    _ordered_unique_sha256(members, "event closure source members")
    unresolved = set(unresolved_member_sha256s)
    exclusions = dict(hard_exclusions or {})
    _require(unresolved <= set(members), "unresolved support member is outside population")
    _require(set(exclusions) <= set(members), "hard support exclusion is outside population")
    _require(not unresolved & set(exclusions), "support member cannot be unresolved and excluded")
    claims_by_member: dict[str, tuple[EventClaimBindingV1, ...]] = {
        member: tuple(row for row in exact_claims if row.source_member_sha256 == member)
        for member in members
    }
    _require(
        all(row.source_member_sha256 in set(members) for row in exact_claims),
        "event closure claim escaped its population",
    )
    cells: list[EventSupportCellCertificateV1] = []
    for member in members:
        member_claims = claims_by_member[member]
        cell_id = identity_sha256(
            {
                "format": SUPPORT_CELL_FORMAT,
                "population_identity_sha256": population_identity_sha256,
                "population_member_sha256s": [member],
            }
        )
        if member in unresolved:
            cells.append(
                EventSupportCellCertificateV1(
                    cell_id,
                    (member,),
                    (),
                    tuple(row.receipt_sha256 for row in member_claims),
                    SupportCellDisposition.UNRESOLVED,
                    None,
                    None,
                )
            )
            continue
        if member in exclusions:
            _require(not member_claims, "hard-excluded support member has event claims")
            basis = exclusions[member]
            assessment = identity_sha256(
                {
                    "basis": basis.value,
                    "cell_id_sha256": cell_id,
                    "member_sha256": member,
                    "policy": "hard_typed_exclusion_only_v1",
                }
            )
            cells.append(
                EventSupportCellCertificateV1(
                    cell_id,
                    (member,),
                    (member,),
                    (),
                    SupportCellDisposition.HARD_TYPED_NO_SUPPORT,
                    basis,
                    assessment,
                )
            )
            continue
        assessment = identity_sha256(
            {
                "cell_id_sha256": cell_id,
                "claim_receipt_sha256s": [
                    row.receipt_sha256 for row in member_claims
                ],
                "member_sha256": member,
                "policy": "exact_typed_claim_enumeration_v1",
            }
        )
        cells.append(
            EventSupportCellCertificateV1(
                cell_id,
                (member,),
                (member,),
                tuple(row.receipt_sha256 for row in member_claims),
                SupportCellDisposition.EXACT_TYPED_CLAIMS_ENUMERATED,
                None,
                assessment,
            )
        )
    return ConjunctiveEventClosureReceiptV1(
        population_identity_sha256,
        members,
        tuple(cells),
        packing_closed,
        all(row.support_closed for row in cells),
    )


@dataclass(frozen=True, slots=True)
class ScopedInsufficiencyAdvisoryV1:
    """Canonical scoped abstention; never a semantic-absence assertion."""

    answer_value_type: str
    unmet_obligation_ids: tuple[str, ...]
    overlay_receipt_sha256: str
    closure_receipt_sha256: str
    text: str
    semantic_absence_may_be_inferred: Literal[False] = False
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.answer_value_type, "insufficiency answer value type")
        _ordered_unique_sha256(
            self.unmet_obligation_ids, "insufficiency unmet obligations"
        )
        _require(bool(self.unmet_obligation_ids), "insufficiency advisory needs unmet edges")
        require_sha256(self.overlay_receipt_sha256, "insufficiency overlay")
        require_sha256(self.closure_receipt_sha256, "insufficiency closure")
        expected_text = canonical_scoped_insufficiency_text(self.answer_value_type)
        _require(self.text == expected_text, "scoped insufficiency text changed")
        _require(
            self.semantic_absence_may_be_inferred is False,
            "scoped insufficiency cannot imply semantic absence",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "insufficiency advisory escaped its provider-free zero-state boundary",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "insufficiency advisory changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "answer_value_type": self.answer_value_type,
            "closure_receipt_sha256": self.closure_receipt_sha256,
            "format": ADVISORY_FORMAT,
            "overlay_receipt_sha256": self.overlay_receipt_sha256,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "scope": "requested_conjunctive_event_binding_only",
            "semantic_absence_may_be_inferred": False,
            "text": self.text,
            "unmet_obligation_ids": list(self.unmet_obligation_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def canonical_scoped_insufficiency_text(answer_value_type: str) -> str:
    label = _canonical_value(answer_value_type).replace("_", " ")
    return (
        f"Available evidence is insufficient to determine the {label}; "
        "it does not establish the full event in one proven-identity evidence bundle."
    )


@dataclass(frozen=True, slots=True)
class ConjunctiveEventDecisionV1:
    disposition: EventDecisionDisposition
    reason: EventDecisionReason
    parent_hypothesis_sha256: str
    terminal_response_text: str | None
    terminal_authorized: bool
    complete_event_component_sha256: str | None
    supporting_claim_receipt_sha256s: tuple[str, ...]
    ignored_compatible_identity_link_receipt_sha256s: tuple[str, ...]
    advisory: ScopedInsufficiencyAdvisoryV1 | None
    overlay_receipt_sha256: str
    closure_receipt_sha256: str
    semantic_absence_may_be_inferred: Literal[False] = False
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.disposition) is EventDecisionDisposition
            and type(self.reason) is EventDecisionReason,
            "conjunctive-event decision enum changed",
        )
        require_sha256(self.parent_hypothesis_sha256, "event parent hypothesis")
        _require(type(self.terminal_authorized) is bool, "terminal authorization changed")
        if self.terminal_response_text is not None:
            require_text(self.terminal_response_text, "event terminal response")
        if self.complete_event_component_sha256 is not None:
            require_sha256(
                self.complete_event_component_sha256, "complete event component"
            )
        _ordered_unique_sha256(
            self.supporting_claim_receipt_sha256s, "supporting event claims"
        )
        _ordered_unique_sha256(
            self.ignored_compatible_identity_link_receipt_sha256s,
            "ignored compatible event links",
        )
        require_sha256(self.overlay_receipt_sha256, "event decision overlay")
        require_sha256(self.closure_receipt_sha256, "event decision closure")
        if self.disposition is EventDecisionDisposition.ABSTAIN:
            _require(
                type(self.advisory) is ScopedInsufficiencyAdvisoryV1
                and self.terminal_authorized
                and self.terminal_response_text == self.advisory.text,
                "abstention decision is not backed by its scoped advisory",
            )
            _require(
                self.reason
                in {
                    EventDecisionReason.SUPPORT_CLOSED_EVENT_UNRESOLVED,
                    EventDecisionReason.SUPPORT_CLOSED_EVENT_CONFLICT,
                }
                and self.advisory.overlay_receipt_sha256
                == self.overlay_receipt_sha256
                and self.advisory.closure_receipt_sha256
                == self.closure_receipt_sha256,
                "abstention advisory escaped its closed support decision",
            )
        else:
            _require(self.advisory is None, "non-abstention decision carries advisory")
        if self.disposition is EventDecisionDisposition.REPLACE:
            _require(
                self.reason is EventDecisionReason.COMPLETE_EVENT_REPLACES_PARENT
                and self.terminal_authorized
                and self.terminal_response_text is not None
                and self.complete_event_component_sha256 is not None
                and bool(self.supporting_claim_receipt_sha256s),
                "replacement lacks one complete event witness",
            )
        if self.disposition is EventDecisionDisposition.KEEP_PARENT:
            if self.terminal_authorized:
                _require(
                    self.reason is EventDecisionReason.COMPLETE_EVENT_MATCHES_PARENT
                    and self.terminal_response_text is not None
                    and self.complete_event_component_sha256 is not None
                    and bool(self.supporting_claim_receipt_sha256s),
                    "authorized parent lacks one complete event witness",
                )
            else:
                _require(
                    self.reason
                    in {
                        EventDecisionReason.SUPPORT_OPEN_EVENT_UNRESOLVED,
                        EventDecisionReason.SUPPORT_OPEN_EVENT_CONFLICT,
                    }
                    and self.terminal_response_text is None
                    and self.complete_event_component_sha256 is None
                    and not self.supporting_claim_receipt_sha256s,
                    "open support cannot authorize the parent",
                )
        if self.reason in {
            EventDecisionReason.SUPPORT_OPEN_EVENT_UNRESOLVED,
            EventDecisionReason.SUPPORT_OPEN_EVENT_CONFLICT,
        }:
            _require(
                self.disposition is EventDecisionDisposition.KEEP_PARENT
                and not self.terminal_authorized
                and self.terminal_response_text is None,
                "open unresolved support cannot authorize the parent",
            )
        _require(
            self.semantic_absence_may_be_inferred is False,
            "event decision cannot infer semantic absence",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "event decision escaped its provider-free zero-state boundary",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "event decision receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="conjunctive_event_decision")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "advisory": self.advisory.projection() if self.advisory is not None else None,
            "closure_receipt_sha256": self.closure_receipt_sha256,
            "complete_event_component_sha256": self.complete_event_component_sha256,
            "disposition": self.disposition.value,
            "format": DECISION_FORMAT,
            "ignored_compatible_identity_link_receipt_sha256s": list(
                self.ignored_compatible_identity_link_receipt_sha256s
            ),
            "overlay_receipt_sha256": self.overlay_receipt_sha256,
            "parent_hypothesis_sha256": self.parent_hypothesis_sha256,
            "provider_prompt_count": 0,
            "reason": self.reason.value,
            "retained_transformer_token_state_bytes": 0,
            "semantic_absence_may_be_inferred": False,
            "supporting_claim_receipt_sha256s": list(
                self.supporting_claim_receipt_sha256s
            ),
            "terminal_authorized": self.terminal_authorized,
            "terminal_response_text": self.terminal_response_text,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _event_components(
    event_keys: tuple[str, ...],
    links: tuple[EventIdentityLinkReceiptV1, ...],
) -> dict[str, tuple[str, ...]]:
    parent = {key: key for key in event_keys}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for link in links:
        if link.status is EventIdentityStatus.PROVEN_SAME_EVENT:
            union(link.left_event_key_sha256, link.right_event_key_sha256)
    result: dict[str, list[str]] = {}
    for key in event_keys:
        result.setdefault(find(key), []).append(key)
    return {
        root: tuple(sorted(values))
        for root, values in sorted(result.items())
    }


def decide_conjunctive_event(
    overlay: ConjunctiveEventObligationOverlayV1,
    claims: Sequence[EventClaimBindingV1],
    identity_links: Sequence[EventIdentityLinkReceiptV1],
    closure: ConjunctiveEventClosureReceiptV1,
    *,
    parent_hypothesis: str,
) -> ConjunctiveEventDecisionV1:
    """Return a receipt-bound keep/replace/abstain decision."""

    _require(type(overlay) is ConjunctiveEventObligationOverlayV1, "event overlay changed")
    exact_claims = tuple(claims)
    exact_links = tuple(identity_links)
    _require(
        all(type(row) is EventClaimBindingV1 for row in exact_claims),
        "event decision claims changed type",
    )
    _require(
        all(type(row) is EventIdentityLinkReceiptV1 for row in exact_links),
        "event decision identity links changed type",
    )
    _require(type(closure) is ConjunctiveEventClosureReceiptV1, "event closure changed")
    require_text(parent_hypothesis, "event parent hypothesis")
    obligations = {row.obligation_id: row for row in overlay.obligations}
    for claim in exact_claims:
        _require(claim.obligation_id in obligations, "event claim escaped its overlay")
        obligation = obligations[claim.obligation_id]
        _require(claim.relation == obligation.relation, "event claim relation changed")
        if obligation.answer_variable:
            _require(
                claim.value_type is not None
                and _canonical_value(claim.value_type)
                == _canonical_value(obligation.answer_value_type or ""),
                "event variable claim has the wrong value type",
            )
        else:
            _require(
                claim.value_type is None
                and _canonical_value(claim.value)
                == _canonical_value(obligation.required_value or ""),
                "event constraint claim does not satisfy its exact obligation",
            )

    members = set(closure.population_member_sha256s)
    _require(
        all(row.source_member_sha256 in members for row in exact_claims),
        "event decision claim escaped the support population",
    )
    cell_by_member = {
        member: cell
        for cell in closure.cells
        for member in cell.population_member_sha256s
    }
    supplied_claim_receipts = {row.receipt_sha256 for row in exact_claims}
    certified_claim_receipts = {
        receipt for cell in closure.cells for receipt in cell.claim_receipt_sha256s
    }
    _require(
        supplied_claim_receipts == certified_claim_receipts,
        "event decision claims disagree with support certificates",
    )
    for claim in exact_claims:
        _require(
            claim.receipt_sha256
            in cell_by_member[claim.source_member_sha256].claim_receipt_sha256s,
            "event claim is not certified in its source cell",
        )

    event_keys = tuple(dict.fromkeys(row.event_key_sha256 for row in exact_claims))
    event_key_set = set(event_keys)
    _require(
        all(
            row.left_event_key_sha256 in event_key_set
            and row.right_event_key_sha256 in event_key_set
            for row in exact_links
        ),
        "event identity link escaped the claim population",
    )
    components = _event_components(event_keys, exact_links)
    answer_obligation = overlay.answer_obligation
    complete: list[tuple[str, str, tuple[EventClaimBindingV1, ...]]] = []
    best_covered: set[str] = set()
    for keys in components.values():
        rows = tuple(row for row in exact_claims if row.event_key_sha256 in keys)
        covered = {row.obligation_id for row in rows}
        if len(covered) > len(best_covered):
            best_covered = covered
        if set(obligations) <= covered:
            variable_rows = tuple(
                row for row in rows if row.obligation_id == answer_obligation.obligation_id
            )
            for variable in variable_rows:
                component_sha = identity_sha256(
                    {
                        "event_keys": list(keys),
                        "format": DECISION_FORMAT,
                        "overlay_receipt_sha256": overlay.receipt_sha256,
                    }
                )
                supporting = tuple(
                    row
                    for row in rows
                    if row.obligation_id in obligations
                    and (
                        row.obligation_id != answer_obligation.obligation_id
                        or _canonical_value(row.value) == _canonical_value(variable.value)
                    )
                )
                complete.append((component_sha, variable.value, supporting))

    unique_values: dict[str, tuple[str, str, tuple[EventClaimBindingV1, ...]]] = {}
    for component_sha, value, rows in complete:
        unique_values.setdefault(_canonical_value(value), (component_sha, value, rows))
    ignored_links = tuple(
        row.receipt_sha256
        for row in exact_links
        if row.status is EventIdentityStatus.COMPATIBLE_UNPROVEN
    )
    parent_sha = hashlib.sha256(parent_hypothesis.encode("utf-8")).hexdigest()

    if len(unique_values) == 1:
        component_sha, value, rows = next(iter(unique_values.values()))
        matches_parent = _canonical_value(value) == _canonical_value(parent_hypothesis)
        return ConjunctiveEventDecisionV1(
            EventDecisionDisposition.KEEP_PARENT
            if matches_parent
            else EventDecisionDisposition.REPLACE,
            EventDecisionReason.COMPLETE_EVENT_MATCHES_PARENT
            if matches_parent
            else EventDecisionReason.COMPLETE_EVENT_REPLACES_PARENT,
            parent_sha,
            parent_hypothesis if matches_parent else value,
            True,
            component_sha,
            tuple(dict.fromkeys(row.receipt_sha256 for row in rows)),
            ignored_links,
            None,
            overlay.receipt_sha256,
            closure.receipt_sha256,
        )

    unmet = tuple(
        row.obligation_id for row in overlay.obligations if row.obligation_id not in best_covered
    )
    if not unmet:
        # Conflicting complete components cover every edge, but still cannot
        # determine one response value.
        unmet = (answer_obligation.obligation_id,)
    conflict = len(unique_values) > 1
    if closure.support_frontier_closed:
        advisory = ScopedInsufficiencyAdvisoryV1(
            answer_obligation.answer_value_type or "requested value",
            unmet,
            overlay.receipt_sha256,
            closure.receipt_sha256,
            canonical_scoped_insufficiency_text(
                answer_obligation.answer_value_type or "requested value"
            ),
        )
        return ConjunctiveEventDecisionV1(
            EventDecisionDisposition.ABSTAIN,
            EventDecisionReason.SUPPORT_CLOSED_EVENT_CONFLICT
            if conflict
            else EventDecisionReason.SUPPORT_CLOSED_EVENT_UNRESOLVED,
            parent_sha,
            advisory.text,
            True,
            None,
            (),
            ignored_links,
            advisory,
            overlay.receipt_sha256,
            closure.receipt_sha256,
        )

    return ConjunctiveEventDecisionV1(
        EventDecisionDisposition.KEEP_PARENT,
        EventDecisionReason.SUPPORT_OPEN_EVENT_CONFLICT
        if conflict
        else EventDecisionReason.SUPPORT_OPEN_EVENT_UNRESOLVED,
        parent_sha,
        None,
        False,
        None,
        (),
        ignored_links,
        None,
        overlay.receipt_sha256,
        closure.receipt_sha256,
    )


def decide_typed_conjunctive_event(
    question: str,
    items: Sequence[TypedEvidenceItem],
    bindings: Sequence[EvidenceHandleBinding],
    *,
    population_identity_sha256: str,
    parent_hypothesis: str,
    packing_closed: bool,
    support_enumerated_handle_ids: Sequence[str] = (),
    hard_exclusions_by_handle: dict[str, HardSupportExclusionBasis] | None = None,
    identity_links: Sequence[EventIdentityLinkReceiptV1] = (),
) -> ConjunctiveEventDecisionV1:
    """Consume existing typed evidence and emit one fail-closed decision.

    Support closure is opt-in per opaque handle.  Any adapted source that is
    neither explicitly enumerated nor hard-excluded remains unresolved, even
    when ``packing_closed`` is true.
    """

    overlay = compile_conjunctive_event_obligation_overlay(question)
    _require(overlay is not None, "question has no supported event overlay")
    sources = exact_event_sources_from_typed_evidence(items, bindings)
    claims = extract_deterministic_presentation_event_claims(overlay, sources)
    source_by_handle = {row.handle_id: row for row in sources}
    enumerated = set(support_enumerated_handle_ids)
    exclusions_by_handle = dict(hard_exclusions_by_handle or {})
    _require(
        enumerated <= set(source_by_handle),
        "enumerated typed support handle is outside the adapted sources",
    )
    _require(
        set(exclusions_by_handle) <= set(source_by_handle),
        "hard-excluded typed support handle is outside the adapted sources",
    )
    _require(
        not enumerated & set(exclusions_by_handle),
        "typed support handle cannot be enumerated and excluded",
    )
    unresolved_members = tuple(
        source.source_member_sha256
        for source in sources
        if source.handle_id not in enumerated
        and source.handle_id not in exclusions_by_handle
    )
    exclusions_by_member = {
        source_by_handle[handle].source_member_sha256: basis
        for handle, basis in exclusions_by_handle.items()
    }
    closure = build_conjunctive_event_closure_receipt(
        population_identity_sha256,
        sources,
        claims,
        packing_closed=packing_closed,
        unresolved_member_sha256s=unresolved_members,
        hard_exclusions=exclusions_by_member,
    )
    return decide_conjunctive_event(
        overlay,
        claims,
        identity_links,
        closure,
        parent_hypothesis=parent_hypothesis,
    )


__all__ = [
    "ADVISORY_FORMAT",
    "ANCHOR_FORMAT",
    "CLAIM_FORMAT",
    "CLOSURE_FORMAT",
    "DECISION_FORMAT",
    "IDENTITY_LINK_FORMAT",
    "MECHANISM_ID",
    "OBLIGATION_FORMAT",
    "OVERLAY_FORMAT",
    "SOURCE_FORMAT",
    "SUPPORT_CELL_FORMAT",
    "ClaimAssertionBasis",
    "ConjunctiveEventClosureReceiptV1",
    "ConjunctiveEventDecisionV1",
    "ConjunctiveEventObligationOverlayV1",
    "EventClaimBindingV1",
    "EventDecisionDisposition",
    "EventDecisionReason",
    "EventEdgeObligationV1",
    "EventIdentityBasis",
    "EventIdentityLinkReceiptV1",
    "EventIdentityStatus",
    "EventSupportCellCertificateV1",
    "ExactEventAnchorV1",
    "ExactEventEvidenceSourceV1",
    "HardSupportExclusionBasis",
    "ScopedInsufficiencyAdvisoryV1",
    "SupportCellDisposition",
    "build_conjunctive_event_closure_receipt",
    "canonical_scoped_insufficiency_text",
    "compile_conjunctive_event_obligation_overlay",
    "decide_conjunctive_event",
    "decide_typed_conjunctive_event",
    "event_identity_link",
    "exact_event_sources_from_typed_evidence",
    "extract_deterministic_presentation_event_claims",
]
