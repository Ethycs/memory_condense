"""Provider-free compression for complete action-linked SET evidence.

The compressor sits *after* retrieval selection.  It accepts exact selected
snippets, binds caller-declared member/support spans back to those bytes, and
only then deduplicates equivalent members.  Satisfying a question-only
explicit cardinality closes only this selected-scope member witness.  It does
not make the upstream retrieval population exhaustive, so the exported typed
contribution always remains ``BOUNDED`` and cannot authorize semantic absence.

The input type is retrieval-neutral on purpose.  A semantic-tree, EM, or
classical lane can expose the same exact snippet contract without this module
depending on its internal index representation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_action_semantics import (
    canonical_action_concepts,
    completed_action_concepts,
)
from .typed_downstream_operator import DownstreamOperatorOverlay
from .typed_operator_adapter import (
    ConflictPolicy,
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ParsedTypedItems,
    ProviderPayloadMode,
    ProvenanceGrade,
    TypedEvidenceContribution,
    TypedEvidenceItem,
    TypedEvidencePacket,
    TypedItemKind,
    ValueAuthority,
    build_typed_evidence_packet,
    parse_typed_items,
)
from .typed_operator_spec import AnswerShape, TypedOperatorSpec, normalized_terms


FORMAT = "memory-condense-post-selection-action-set-compressor-v1"
DEMAND_FORMAT = f"{FORMAT}-demand"
SNIPPET_FORMAT = f"{FORMAT}-selected-exact-snippet"
PROPOSAL_FORMAT = f"{FORMAT}-exact-member-proposal"
BOUND_CANDIDATE_FORMAT = f"{FORMAT}-bound-candidate"
FACT_FORMAT = f"{FORMAT}-compressed-fact"
EXCLUSION_FORMAT = f"{FORMAT}-post-selection-exclusion"
CLOSURE_FORMAT = f"{FORMAT}-support-closure"
PROVIDER_FORMAT = f"{FORMAT}-provider-payload"
MECHANISM_ID = "post_selection_action_linked_set_fact_compressor_v1"
DEFAULT_PAYLOAD_TOKEN_CAP = 2_048

_DATED_RE = re.compile(
    r"^\[Question asked at (?P<asked_at>.+?)\]\s*", re.IGNORECASE | re.DOTALL
)
_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_GROUP_RE = re.compile(r"^G[0-9]{3,6}$")
_HANDLE_RE = re.compile(r"^H[0-9]{3,6}$")
_RELATION_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "two",
        "was",
        "were",
        "what",
        "which",
        "who",
        "with",
    }
)
_RELATED_ACTIVITY_RE = re.compile(
    r"\brelated\s+to\s+(?P<member>[^\W\d_]+(?:[-'][^\W\d_]+)?)\b",
    re.IGNORECASE | re.UNICODE,
)
_PHOTOGRAPHY_SURFACE_RE = re.compile(
    r"\b(?:photography|photographs?|photos?)\b", re.IGNORECASE
)


class ActionSetCompressionError(MatchedEvalContractError):
    """Raised when selected bytes, typed support, or closure diverges."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ActionSetCompressionError(message)


def _receipt(value: Mapping[str, Any], declared: str, label: str) -> str:
    expected = identity_sha256(value)
    if declared and declared != expected:
        raise ActionSetCompressionError(f"{label} changed")
    return expected


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(row) is str and bool(row) for row in values)
        and len(set(values)) == len(values),
        f"{label} must be an ordered unique text tuple",
    )
    return values


@dataclass(frozen=True, slots=True)
class ActionLinkedSetDemand:
    """Question-only action/object demand plus explicit SET cardinality."""

    question_sha256: str
    operator_spec_receipt_sha256: str
    downstream_overlay_receipt_sha256: str
    cardinality: int
    action_concepts: tuple[str, ...]
    relation_anchor_terms: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.question_sha256, "action-set demand question"),
            (self.operator_spec_receipt_sha256, "action-set demand operator"),
            (self.downstream_overlay_receipt_sha256, "action-set demand overlay"),
        ):
            require_sha256(value, label)
        _require(
            type(self.cardinality) is int and self.cardinality >= 1,
            "action-set demand requires positive explicit cardinality",
        )
        _ordered_unique(self.action_concepts, "action-set demand actions")
        _ordered_unique(self.relation_anchor_terms, "action-set relation anchors")
        _require(
            bool(self.action_concepts) and bool(self.relation_anchor_terms),
            "action-set demand requires action and relation anchors",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "action-set demand receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="action_linked_set_demand")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "action_concepts": list(self.action_concepts),
            "cardinality": self.cardinality,
            "downstream_overlay_receipt_sha256": (
                self.downstream_overlay_receipt_sha256
            ),
            "format": DEMAND_FORMAT,
            "operator_spec_receipt_sha256": self.operator_spec_receipt_sha256,
            "question_sha256": self.question_sha256,
            "relation_anchor_terms": list(self.relation_anchor_terms),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _relation_tail_terms(question_body: str, actions: tuple[str, ...]) -> tuple[str, ...]:
    words = tuple(_WORD_RE.finditer(question_body))
    action_positions = tuple(
        index
        for index, match in enumerate(words)
        if set(canonical_action_concepts(match.group(0))) & set(actions)
    )
    _require(bool(action_positions), "action-set question has no action surface")
    tail = " ".join(match.group(0) for match in words[action_positions[-1] + 1 :])
    terms = tuple(
        term
        for term in normalized_terms(tail)
        if term not in _RELATION_STOP_TERMS
    )
    return tuple(dict.fromkeys(terms))


def compile_action_linked_set_demand(
    dated_question: str,
    operator_spec: TypedOperatorSpec,
    downstream_overlay: DownstreamOperatorOverlay,
) -> ActionLinkedSetDemand:
    """Compile the compressor's demand from question-only sealed inputs."""

    require_text(dated_question, "action-set dated question")
    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be an exact TypedOperatorSpec")
    if type(downstream_overlay) is not DownstreamOperatorOverlay:
        raise TypeError("downstream_overlay must be an exact DownstreamOperatorOverlay")
    _require(
        quote_sha256(dated_question) == operator_spec.question_sha256
        == downstream_overlay.question_sha256
        and downstream_overlay.legacy_operator_spec_receipt_sha256
        == operator_spec.receipt_sha256,
        "action-set question/spec/overlay binding changed",
    )
    _require(
        operator_spec.answer_shape is AnswerShape.SET_LIST
        and downstream_overlay.effective_set_cardinality is not None,
        "action-set compressor requires an explicit-cardinality SET question",
    )
    body = _DATED_RE.sub("", dated_question).strip()
    actions = canonical_action_concepts(body)
    _require(bool(actions), "action-set question has no canonical action")
    anchors = _relation_tail_terms(body, actions)
    _require(bool(anchors), "action-set question has no relation-object anchors")
    return ActionLinkedSetDemand(
        question_sha256=operator_spec.question_sha256,
        operator_spec_receipt_sha256=operator_spec.receipt_sha256,
        downstream_overlay_receipt_sha256=downstream_overlay.receipt_sha256,
        cardinality=downstream_overlay.effective_set_cardinality,
        action_concepts=actions,
        relation_anchor_terms=anchors,
    )


@dataclass(frozen=True, slots=True)
class SelectedExactSnippet:
    """One exact snippet admitted by an upstream selector before compression."""

    selection_ordinal: int
    candidate_id: str
    source_group_handle: str
    quote: str
    quote_sha256: str
    role: str
    created_at: str
    evidence_receipt_sha256: str
    local_binding_receipt_sha256: str
    local_source_locator_sha256: str
    selection_receipt_sha256: str
    token_count: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.selection_ordinal) is int and self.selection_ordinal >= 0,
            "selected snippet ordinal changed",
        )
        for value, label in (
            (self.candidate_id, "selected snippet candidate"),
            (self.quote_sha256, "selected snippet quote"),
            (self.evidence_receipt_sha256, "selected snippet evidence"),
            (self.local_binding_receipt_sha256, "selected snippet local binding"),
            (self.local_source_locator_sha256, "selected snippet source locator"),
            (self.selection_receipt_sha256, "selected snippet selection"),
        ):
            require_sha256(value, label)
        _require(
            _GROUP_RE.fullmatch(self.source_group_handle) is not None,
            "selected snippet source handle must be opaque",
        )
        require_text(self.quote, "selected snippet quote")
        require_text(self.role, "selected snippet role")
        require_text(self.created_at, "selected snippet created-at")
        _require(
            self.quote_sha256 == quote_sha256(self.quote)
            and type(self.token_count) is int
            and self.token_count == count_tokens(self.quote),
            "selected snippet lost exact quote bytes",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "selected snippet receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "created_at": self.created_at,
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "format": SNIPPET_FORMAT,
            "local_binding_receipt_sha256": self.local_binding_receipt_sha256,
            "local_source_locator_sha256": self.local_source_locator_sha256,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "selected_before_fact_compression": True,
            "selection_ordinal": self.selection_ordinal,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "source_group_handle": self.source_group_handle,
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ExactActionSetFactProposal:
    """Exact coordinates for a member and its action/object support quote."""

    selected_candidate_id: str
    member_text: str
    member_start_char: int
    member_end_char: int
    support_start_char: int
    support_end_char: int
    action_concept: str
    member_derivation: Literal["exact_span", "lexical_normalization"] = (
        "exact_span"
    )
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.selected_candidate_id, "action-set proposal candidate")
        require_text(self.member_text, "action-set proposal member")
        _require(
            all(
                type(value) is int
                for value in (
                    self.member_start_char,
                    self.member_end_char,
                    self.support_start_char,
                    self.support_end_char,
                )
            )
            and 0 <= self.support_start_char <= self.member_start_char
            < self.member_end_char <= self.support_end_char,
            "action-set proposal coordinates changed",
        )
        require_text(self.action_concept, "action-set proposal action")
        _require(
            canonical_action_concepts(self.action_concept)
            == (self.action_concept,),
            "action-set proposal action is not canonical",
        )
        _require(
            self.member_derivation in {"exact_span", "lexical_normalization"},
            "action-set proposal member derivation changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "action-set proposal receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "action_concept": self.action_concept,
            "format": PROPOSAL_FORMAT,
            "member_end_char": self.member_end_char,
            "member_start_char": self.member_start_char,
            "member_text": self.member_text,
            "member_derivation": self.member_derivation,
            "selected_candidate_id": self.selected_candidate_id,
            "support_end_char": self.support_end_char,
            "support_start_char": self.support_start_char,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def locate_exact_action_set_fact(
    snippet: SelectedExactSnippet,
    *,
    member_text: str,
    support_quote: str,
    action_concept: str,
) -> ExactActionSetFactProposal:
    """Locate unique exact member/support occurrences inside a selected snippet."""

    if type(snippet) is not SelectedExactSnippet:
        raise TypeError("snippet must be an exact SelectedExactSnippet")
    require_text(member_text, "action-set located member")
    require_text(support_quote, "action-set located support quote")
    support_start = snippet.quote.find(support_quote)
    _require(
        support_start >= 0
        and snippet.quote.find(support_quote, support_start + 1) < 0,
        "action-set support quote is absent or ambiguous",
    )
    relative_member = support_quote.find(member_text)
    _require(
        relative_member >= 0
        and support_quote.find(member_text, relative_member + 1) < 0,
        "action-set member is absent or ambiguous in its support quote",
    )
    member_start = support_start + relative_member
    return ExactActionSetFactProposal(
        selected_candidate_id=snippet.candidate_id,
        member_text=member_text,
        member_start_char=member_start,
        member_end_char=member_start + len(member_text),
        support_start_char=support_start,
        support_end_char=support_start + len(support_quote),
        action_concept=action_concept,
        member_derivation="exact_span",
    )


def _lexically_normalized_activity_member(surface: str) -> str | None:
    """Return a conservative category label for one exact activity surface."""

    terms = set(normalized_terms(surface))
    if terms & {"photo", "photograph", "photography"}:
        return "photography"
    return None


def infer_exact_action_set_fact_proposals(
    demand: ActionLinkedSetDemand,
    selected_snippets: tuple[SelectedExactSnippet, ...],
) -> tuple[ExactActionSetFactProposal, ...]:
    """Infer conservative member spans from exact action/object snippets.

    The extractor intentionally has only two mechanisms: an explicit
    ``related to <activity>`` complement and a lexical normalization from an
    exact photo/photograph surface to the activity category ``photography``.
    It does not use source IDs, question IDs, answers, or benchmark labels.
    """

    if type(demand) is not ActionLinkedSetDemand:
        raise TypeError("demand must be an exact ActionLinkedSetDemand")
    _require(
        type(selected_snippets) is tuple
        and all(type(row) is SelectedExactSnippet for row in selected_snippets),
        "action-set inference snippets changed immutable type",
    )
    proposals: list[ExactActionSetFactProposal] = []
    for snippet in selected_snippets:
        support_terms = set(normalized_terms(snippet.quote))
        actions = tuple(
            action
            for action in demand.action_concepts
            if action in completed_action_concepts(snippet.quote)
        )
        if not actions or not set(demand.relation_anchor_terms) <= support_terms:
            continue

        member_match = _RELATED_ACTIVITY_RE.search(snippet.quote)
        member_text: str | None = None
        member_surface: str | None = None
        member_start = -1
        derivation: Literal["exact_span", "lexical_normalization"] = "exact_span"
        if member_match is not None:
            member_surface = member_match.group("member")
            member_text = member_surface.casefold()
            member_start = member_match.start("member")
        else:
            photo_match = _PHOTOGRAPHY_SURFACE_RE.search(snippet.quote)
            if photo_match is not None:
                member_surface = photo_match.group(0)
                normalized = _lexically_normalized_activity_member(member_surface)
                if normalized is not None:
                    member_text = normalized
                    member_start = photo_match.start()
                    derivation = (
                        "exact_span"
                        if member_surface.casefold() == normalized
                        else "lexical_normalization"
                    )
        if member_text is None or member_surface is None:
            continue
        proposals.append(
            ExactActionSetFactProposal(
                selected_candidate_id=snippet.candidate_id,
                member_text=member_text,
                member_start_char=member_start,
                member_end_char=member_start + len(member_surface),
                support_start_char=0,
                support_end_char=len(snippet.quote),
                action_concept=actions[0],
                member_derivation=derivation,
            )
        )
    return tuple(proposals)


@dataclass(frozen=True, slots=True)
class BoundActionSetCandidate:
    pre_dedup_ordinal: int
    selected_candidate_id: str
    selected_source_group_handle: str
    selected_quote_sha256: str
    selected_evidence_receipt_sha256: str
    selected_local_binding_receipt_sha256: str
    proposal_receipt_sha256: str
    member_text: str
    member_surface_text: str
    member_derivation: Literal["exact_span", "lexical_normalization"]
    member_key: str
    member_start_char: int
    member_end_char: int
    action_concept: str
    relation_anchor_terms: tuple[str, ...]
    support_quote: str
    support_quote_sha256: str
    support_start_char: int
    support_end_char: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.pre_dedup_ordinal) is int and self.pre_dedup_ordinal >= 0,
            "bound action-set candidate ordinal changed",
        )
        for value, label in (
            (self.selected_candidate_id, "bound candidate selection"),
            (self.selected_quote_sha256, "bound candidate selected quote"),
            (self.selected_evidence_receipt_sha256, "bound candidate evidence"),
            (self.selected_local_binding_receipt_sha256, "bound candidate binding"),
            (self.proposal_receipt_sha256, "bound candidate proposal"),
            (self.support_quote_sha256, "bound candidate support quote"),
        ):
            require_sha256(value, label)
        _require(
            _GROUP_RE.fullmatch(self.selected_source_group_handle) is not None,
            "bound candidate source handle changed",
        )
        for value, label in (
            (self.member_text, "bound candidate member"),
            (self.member_surface_text, "bound candidate member surface"),
            (self.member_key, "bound candidate member key"),
            (self.action_concept, "bound candidate action"),
            (self.support_quote, "bound candidate support quote"),
        ):
            require_text(value, label)
        _ordered_unique(self.relation_anchor_terms, "bound candidate relation anchors")
        _require(
            self.member_derivation in {"exact_span", "lexical_normalization"}
            and self.member_surface_text
            in self.support_quote,
            "bound candidate member derivation/surface changed",
        )
        _require(
            self.support_quote_sha256 == quote_sha256(self.support_quote),
            "bound candidate support quote digest changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "bound action-set candidate receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "action_concept": self.action_concept,
            "format": BOUND_CANDIDATE_FORMAT,
            "member_end_char": self.member_end_char,
            "member_derivation": self.member_derivation,
            "member_key": self.member_key,
            "member_start_char": self.member_start_char,
            "member_surface_text": self.member_surface_text,
            "member_text": self.member_text,
            "pre_dedup_ordinal": self.pre_dedup_ordinal,
            "proposal_receipt_sha256": self.proposal_receipt_sha256,
            "relation_anchor_terms": list(self.relation_anchor_terms),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_evidence_receipt_sha256": (
                self.selected_evidence_receipt_sha256
            ),
            "selected_local_binding_receipt_sha256": (
                self.selected_local_binding_receipt_sha256
            ),
            "selected_quote_sha256": self.selected_quote_sha256,
            "selected_source_group_handle": self.selected_source_group_handle,
            "support_end_char": self.support_end_char,
            "support_quote_sha256": self.support_quote_sha256,
            "support_start_char": self.support_start_char,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class CompressedActionSetFact:
    member_text: str
    member_key: str
    action_concept: str
    relation_anchor_terms: tuple[str, ...]
    handle_ids: tuple[str, ...]
    bound_candidate_receipt_sha256s: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.member_text, "compressed set member")
        require_text(self.member_key, "compressed set member key")
        require_text(self.action_concept, "compressed set action")
        _ordered_unique(self.relation_anchor_terms, "compressed set relation anchors")
        _ordered_unique(self.handle_ids, "compressed set handles")
        _require(
            bool(self.handle_ids)
            and all(_HANDLE_RE.fullmatch(value) is not None for value in self.handle_ids),
            "compressed set handles must be opaque",
        )
        _ordered_unique(
            self.bound_candidate_receipt_sha256s,
            "compressed set candidate receipts",
        )
        for value in self.bound_candidate_receipt_sha256s:
            require_sha256(value, "compressed set candidate")
        _require(
            len(self.handle_ids) == len(self.bound_candidate_receipt_sha256s),
            "compressed set fact lost support handles",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "compressed action-set fact receipt",
            ),
        )

    @property
    def summary(self) -> str:
        anchors = " ".join(self.relation_anchor_terms)
        return f"{self.member_text} led the user to {self.action_concept} {anchors}."

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "action_concept": self.action_concept,
            "bound_candidate_receipt_sha256s": list(
                self.bound_candidate_receipt_sha256s
            ),
            "format": FACT_FORMAT,
            "handle_ids": list(self.handle_ids),
            "member_key": self.member_key,
            "member_text": self.member_text,
            "relation_anchor_terms": list(self.relation_anchor_terms),
            "summary": self.summary,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class PostSelectionSnippetExclusion:
    selected_candidate_id: str
    selected_quote_sha256: str
    selected_local_binding_receipt_sha256: str
    reason: Literal["no_exact_action_linked_set_proposal_after_selection"] = (
        "no_exact_action_linked_set_proposal_after_selection"
    )
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.selected_candidate_id, "post-selection excluded candidate"),
            (self.selected_quote_sha256, "post-selection excluded quote"),
            (
                self.selected_local_binding_receipt_sha256,
                "post-selection excluded binding",
            ),
        ):
            require_sha256(value, label)
        _require(
            self.reason == "no_exact_action_linked_set_proposal_after_selection",
            "post-selection exclusion reason changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "post-selection exclusion receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, str]:
        value = {
            "format": EXCLUSION_FORMAT,
            "reason": self.reason,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_local_binding_receipt_sha256": (
                self.selected_local_binding_receipt_sha256
            ),
            "selected_quote_sha256": self.selected_quote_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ActionSetSupportClosureReceipt:
    demand_receipt_sha256: str
    selection_receipt_sha256: str
    selected_snippet_count: int
    candidate_count_before_dedup: int
    distinct_supported_member_count: int
    explicit_cardinality: int
    bound_candidate_receipt_sha256s: tuple[str, ...]
    compressed_fact_receipt_sha256s: tuple[str, ...]
    post_selection_exclusion_receipt_sha256s: tuple[str, ...]
    explicit_cardinality_satisfied: bool
    support_frontier_closed: bool
    closure_basis: Literal[
        "explicit_cardinality_satisfied_by_exact_action_linked_members",
        "explicit_cardinality_not_satisfied",
    ]
    semantic_absence_may_be_inferred: Literal[False] = False
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.demand_receipt_sha256, "action-set closure demand")
        require_sha256(self.selection_receipt_sha256, "action-set closure selection")
        for value, label in (
            (self.selected_snippet_count, "action-set closure selected count"),
            (
                self.candidate_count_before_dedup,
                "action-set closure pre-dedup count",
            ),
            (
                self.distinct_supported_member_count,
                "action-set closure distinct count",
            ),
            (self.explicit_cardinality, "action-set closure cardinality"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        _require(self.explicit_cardinality >= 1, "action-set closure cardinality changed")
        for values, label in (
            (self.bound_candidate_receipt_sha256s, "action-set closure candidates"),
            (self.compressed_fact_receipt_sha256s, "action-set closure facts"),
            (
                self.post_selection_exclusion_receipt_sha256s,
                "action-set closure exclusions",
            ),
        ):
            _ordered_unique(values, label)
            for value in values:
                require_sha256(value, label)
        satisfied = self.distinct_supported_member_count == self.explicit_cardinality
        _require(
            type(self.explicit_cardinality_satisfied) is bool
            and self.explicit_cardinality_satisfied == satisfied
            and type(self.support_frontier_closed) is bool
            and self.support_frontier_closed == satisfied
            and self.closure_basis
            == (
                "explicit_cardinality_satisfied_by_exact_action_linked_members"
                if satisfied
                else "explicit_cardinality_not_satisfied"
            ),
            "action-set support closure is not justified",
        )
        _require(
            self.candidate_count_before_dedup
            == len(self.bound_candidate_receipt_sha256s)
            and self.distinct_supported_member_count
            == len(self.compressed_fact_receipt_sha256s)
            and self.semantic_absence_may_be_inferred is False
            and self.retained_transformer_token_state_bytes == 0,
            "action-set closure accounting/state changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "action-set support closure receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="action_set_support_closure")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "bound_candidate_receipt_sha256s": list(
                self.bound_candidate_receipt_sha256s
            ),
            "candidate_count_before_dedup": self.candidate_count_before_dedup,
            "closure_basis": self.closure_basis,
            "compressed_fact_receipt_sha256s": list(
                self.compressed_fact_receipt_sha256s
            ),
            "demand_receipt_sha256": self.demand_receipt_sha256,
            "distinct_supported_member_count": (
                self.distinct_supported_member_count
            ),
            "explicit_cardinality": self.explicit_cardinality,
            "explicit_cardinality_satisfied": self.explicit_cardinality_satisfied,
            "format": CLOSURE_FORMAT,
            "post_selection_exclusion_receipt_sha256s": list(
                self.post_selection_exclusion_receipt_sha256s
            ),
            "retained_transformer_token_state_bytes": 0,
            "selected_snippet_count": self.selected_snippet_count,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "semantic_absence_may_be_inferred": False,
            "support_frontier_closed": self.support_frontier_closed,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ActionLinkedSetCompression:
    demand: ActionLinkedSetDemand
    selection_receipt_sha256: str
    selected_snippet_receipt_sha256s: tuple[str, ...]
    bound_candidates: tuple[BoundActionSetCandidate, ...]
    facts: tuple[CompressedActionSetFact, ...]
    post_selection_exclusions: tuple[PostSelectionSnippetExclusion, ...]
    closure: ActionSetSupportClosureReceipt
    bindings: tuple[EvidenceHandleBinding, ...]
    parsed: ParsedTypedItems
    contribution: TypedEvidenceContribution
    provider_payload_token_proxy: int
    payload_token_cap: int
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.demand) is ActionLinkedSetDemand, "action-set demand changed")
        require_sha256(self.selection_receipt_sha256, "action-set compression selection")
        for values, expected, label in (
            (self.bound_candidates, BoundActionSetCandidate, "bound candidates"),
            (self.facts, CompressedActionSetFact, "compressed facts"),
            (
                self.post_selection_exclusions,
                PostSelectionSnippetExclusion,
                "post-selection exclusions",
            ),
            (self.bindings, EvidenceHandleBinding, "typed bindings"),
        ):
            _require(
                type(values) is tuple and all(type(row) is expected for row in values),
                f"action-set {label} changed type",
            )
        _ordered_unique(
            self.selected_snippet_receipt_sha256s,
            "action-set selected snippet receipts",
        )
        _require(
            type(self.closure) is ActionSetSupportClosureReceipt
            and type(self.parsed) is ParsedTypedItems
            and type(self.contribution) is TypedEvidenceContribution,
            "action-set typed output changed type",
        )
        _require(
            self.closure.demand_receipt_sha256 == self.demand.receipt_sha256
            and self.closure.selection_receipt_sha256
            == self.selection_receipt_sha256
            and self.closure.selected_snippet_count
            == len(self.selected_snippet_receipt_sha256s)
            and self.closure.bound_candidate_receipt_sha256s
            == tuple(row.receipt_sha256 for row in self.bound_candidates)
            and self.closure.compressed_fact_receipt_sha256s
            == tuple(row.receipt_sha256 for row in self.facts)
            and self.closure.post_selection_exclusion_receipt_sha256s
            == tuple(row.receipt_sha256 for row in self.post_selection_exclusions),
            "action-set closure escaped its compression population",
        )
        _require(
            self.contribution.bindings == self.bindings
            and self.contribution.parsed == self.parsed
            and self.contribution.frontier_mode is FrontierMode.BOUNDED,
            "selected-scope action-set witness upgraded the generic frontier",
        )
        _require(
            type(self.provider_payload_token_proxy) is int
            and type(self.payload_token_cap) is int
            and 0 < self.provider_payload_token_proxy <= self.payload_token_cap
            and self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "action-set payload cap/provider/state changed",
        )
        rendered = json.dumps(
            self.provider_projection(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        _require(
            count_tokens(rendered) == self.provider_payload_token_proxy,
            "action-set provider payload token accounting changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "action-set compression receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="action_linked_set_compression")
        assert_gold_blind(self.provider_projection(), path="action_set_provider")

    def provider_projection(self) -> dict[str, Any]:
        by_receipt = {row.receipt_sha256: row for row in self.bound_candidates}
        return {
            "cardinality": self.demand.cardinality,
            "facts": [
                {
                    "action": fact.action_concept,
                    "member": fact.member_text,
                    "relation_anchor_terms": list(fact.relation_anchor_terms),
                    "support": [
                        {
                            "evidence_handle": handle,
                            "quote": by_receipt[receipt].support_quote,
                            "quote_sha256": by_receipt[receipt].support_quote_sha256,
                        }
                        for handle, receipt in zip(
                            fact.handle_ids,
                            fact.bound_candidate_receipt_sha256s,
                            strict=True,
                        )
                    ],
                }
                for fact in self.facts
            ],
            "format": PROVIDER_FORMAT,
            "support_frontier": {
                "closed": self.closure.support_frontier_closed,
                "closure_basis": self.closure.closure_basis,
                "generic_frontier_closed": False,
                "receipt_sha256": self.closure.receipt_sha256,
                "scope": "selected_action_linked_members_only",
                "semantic_absence_may_be_inferred": False,
            },
        }

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
            "bound_candidates": [row.projection() for row in self.bound_candidates],
            "closure": self.closure.projection(),
            "contribution_receipt_sha256": self.contribution.receipt_sha256,
            "dedup_after_selection": True,
            "demand": self.demand.projection(),
            "facts": [row.projection() for row in self.facts],
            "format": FORMAT,
            "parsed_item_receipt_sha256s": [
                row.receipt_sha256 for row in self.parsed.accepted_items
            ],
            "payload_token_cap": self.payload_token_cap,
            "post_selection_exclusions": [
                row.projection() for row in self.post_selection_exclusions
            ],
            "provider_payload_token_proxy": self.provider_payload_token_proxy,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "selected_snippet_receipt_sha256s": list(
                self.selected_snippet_receipt_sha256s
            ),
            "selection_precedes_fact_compression": True,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "typed_binding_receipt_sha256s": [
                row.receipt_sha256 for row in self.bindings
            ],
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _bind_candidates(
    demand: ActionLinkedSetDemand,
    snippets: tuple[SelectedExactSnippet, ...],
    proposals: tuple[ExactActionSetFactProposal, ...],
) -> tuple[BoundActionSetCandidate, ...]:
    by_id = {row.candidate_id: row for row in snippets}
    bound: list[BoundActionSetCandidate] = []
    for ordinal, proposal in enumerate(proposals):
        snippet = by_id.get(proposal.selected_candidate_id)
        _require(snippet is not None, "action-set proposal escaped selected snippets")
        assert snippet is not None
        _require(
            proposal.action_concept in demand.action_concepts,
            "action-set proposal action escaped question demand",
        )
        _require(
            proposal.support_end_char <= len(snippet.quote)
            and proposal.member_end_char <= len(snippet.quote),
            "action-set member coordinates changed selected exact bytes",
        )
        member_surface = snippet.quote[
            proposal.member_start_char : proposal.member_end_char
        ]
        if proposal.member_derivation == "exact_span":
            _require(
                member_surface.casefold() == proposal.member_text.casefold(),
                "action-set exact member changed selected bytes",
            )
        else:
            _require(
                _lexically_normalized_activity_member(member_surface)
                == proposal.member_text,
                "action-set lexical member normalization is unsupported",
            )
        support = snippet.quote[
            proposal.support_start_char : proposal.support_end_char
        ]
        support_terms = set(normalized_terms(support))
        _require(bool(support), "action-set support quote became empty")
        _require(
            proposal.action_concept in completed_action_concepts(support),
            "action-set support does not prove the completed action",
        )
        _require(
            set(demand.relation_anchor_terms) <= support_terms,
            "action-set support does not prove the relation object",
        )
        member_terms = normalized_terms(proposal.member_text)
        _require(bool(member_terms), "action-set member has no semantic identity")
        bound.append(
            BoundActionSetCandidate(
                pre_dedup_ordinal=ordinal,
                selected_candidate_id=snippet.candidate_id,
                selected_source_group_handle=snippet.source_group_handle,
                selected_quote_sha256=snippet.quote_sha256,
                selected_evidence_receipt_sha256=(
                    snippet.evidence_receipt_sha256
                ),
                selected_local_binding_receipt_sha256=(
                    snippet.local_binding_receipt_sha256
                ),
                proposal_receipt_sha256=proposal.receipt_sha256,
                member_text=proposal.member_text,
                member_surface_text=member_surface,
                member_derivation=proposal.member_derivation,
                member_key=" ".join(member_terms),
                member_start_char=proposal.member_start_char,
                member_end_char=proposal.member_end_char,
                action_concept=proposal.action_concept,
                relation_anchor_terms=demand.relation_anchor_terms,
                support_quote=support,
                support_quote_sha256=quote_sha256(support),
                support_start_char=proposal.support_start_char,
                support_end_char=proposal.support_end_char,
            )
        )
    _require(
        len({row.receipt_sha256 for row in bound}) == len(bound),
        "action-set exact proposals repeat",
    )
    return tuple(bound)


def _allocate_bindings(
    bound: tuple[BoundActionSetCandidate, ...],
    snippets: tuple[SelectedExactSnippet, ...],
    *,
    sealed_selection_artifact_sha256: str,
    selection_receipt_sha256: str,
    handle_start: int,
    group_start: int,
    preserved_bindings_by_candidate_id: (
        Mapping[str, EvidenceHandleBinding] | None
    ) = None,
) -> tuple[tuple[EvidenceHandleBinding, ...], Mapping[str, str]]:
    if preserved_bindings_by_candidate_id is not None:
        _require(
            isinstance(preserved_bindings_by_candidate_id, Mapping),
            "action-set preserved binding map changed type",
        )
        selected_by_id = {row.candidate_id: row for row in snippets}
        bindings: list[EvidenceHandleBinding] = []
        handles: dict[str, str] = {}
        for candidate in bound:
            binding = preserved_bindings_by_candidate_id.get(
                candidate.selected_candidate_id
            )
            snippet = selected_by_id[candidate.selected_candidate_id]
            _require(
                type(binding) is EvidenceHandleBinding
                and binding.receipt_sha256
                == snippet.local_binding_receipt_sha256
                and binding.source_group_handle
                == candidate.selected_source_group_handle
                and binding.citation_sha256
                == candidate.support_quote_sha256
                and binding.sealed_artifact_sha256
                == sealed_selection_artifact_sha256,
                "action-set preserved binding escaped exact support provenance",
            )
            assert binding is not None
            bindings.append(binding)
            handles[candidate.receipt_sha256] = binding.handle_id
        _require(
            len({row.handle_id for row in bindings}) == len(bindings),
            "action-set preserved handles repeat",
        )
        return tuple(bindings), handles

    _require(
        type(handle_start) is int
        and type(group_start) is int
        and 1 <= handle_start <= 999_999
        and 1 <= group_start <= 999_999,
        "action-set opaque allocation start changed",
    )
    source_groups = tuple(
        dict.fromkeys(row.source_group_handle for row in snippets)
    )
    _require(
        group_start + len(source_groups) - 1 <= 999_999
        and handle_start + len(bound) - 1 <= 999_999,
        "action-set opaque allocation overflow",
    )
    remapped = {
        source: f"G{group_start + index:03d}"
        for index, source in enumerate(source_groups)
    }
    handles: dict[str, str] = {}
    bindings: list[EvidenceHandleBinding] = []
    for index, candidate in enumerate(bound):
        handle = f"H{handle_start + index:03d}"
        handles[candidate.receipt_sha256] = handle
        bindings.append(
            EvidenceHandleBinding(
                handle_id=handle,
                origin=EvidenceOrigin.DIRECT_POINTER,
                provenance_grade=ProvenanceGrade.DIRECT_POINTER,
                source_group_handle=remapped[
                    candidate.selected_source_group_handle
                ],
                sealed_artifact_sha256=sealed_selection_artifact_sha256,
                parent_receipt_sha256=selection_receipt_sha256,
                evidence_receipt_sha256=candidate.receipt_sha256,
                payload_sha256=quote_sha256(candidate.member_text),
                citation_sha256=candidate.support_quote_sha256,
                citation_char_count=len(candidate.support_quote),
                local_source_locator_sha256=identity_sha256(
                    {
                        "member_end_char": candidate.member_end_char,
                        "member_start_char": candidate.member_start_char,
                        "selected_local_binding_receipt_sha256": (
                            candidate.selected_local_binding_receipt_sha256
                        ),
                        "selected_quote_sha256": candidate.selected_quote_sha256,
                        "support_end_char": candidate.support_end_char,
                        "support_quote_sha256": candidate.support_quote_sha256,
                        "support_start_char": candidate.support_start_char,
                    }
                ),
            )
        )
    return tuple(bindings), handles


def _compress_facts(
    bound: tuple[BoundActionSetCandidate, ...],
    handles: Mapping[str, str],
) -> tuple[CompressedActionSetFact, ...]:
    grouped: dict[str, list[BoundActionSetCandidate]] = {}
    for candidate in bound:
        grouped.setdefault(candidate.member_key, []).append(candidate)
    return tuple(
        CompressedActionSetFact(
            member_text=rows[0].member_text,
            member_key=member_key,
            action_concept=rows[0].action_concept,
            relation_anchor_terms=rows[0].relation_anchor_terms,
            handle_ids=tuple(handles[row.receipt_sha256] for row in rows),
            bound_candidate_receipt_sha256s=tuple(
                row.receipt_sha256 for row in rows
            ),
        )
        for member_key, rows in grouped.items()
    )


def _raw_typed_items(facts: Sequence[CompressedActionSetFact]) -> list[dict[str, Any]]:
    return [
        {
            "entity_key": fact.member_text,
            "handle_ids": list(fact.handle_ids),
            "included": True,
            "kind": TypedItemKind.MEMBER.value,
            "relation": (
                f"{fact.action_concept} {' '.join(fact.relation_anchor_terms)}"
            ),
            "specificity_terms": [fact.member_text],
            "status": "completed",
            "summary": fact.summary,
            "value_authority": ValueAuthority.EXPLICIT.value,
        }
        for fact in facts
    ]


def compress_action_linked_set_evidence(
    demand: ActionLinkedSetDemand,
    selected_snippets: tuple[SelectedExactSnippet, ...],
    proposals: tuple[ExactActionSetFactProposal, ...],
    operator_spec: TypedOperatorSpec,
    *,
    sealed_selection_artifact_sha256: str,
    payload_token_cap: int = DEFAULT_PAYLOAD_TOKEN_CAP,
    handle_start: int = 1,
    group_start: int = 1,
    preserved_bindings_by_candidate_id: (
        Mapping[str, EvidenceHandleBinding] | None
    ) = None,
) -> ActionLinkedSetCompression:
    """Compress exact action-linked members after immutable selection."""

    if type(demand) is not ActionLinkedSetDemand:
        raise TypeError("demand must be an exact ActionLinkedSetDemand")
    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be an exact TypedOperatorSpec")
    _require(
        demand.question_sha256 == operator_spec.question_sha256
        and demand.operator_spec_receipt_sha256 == operator_spec.receipt_sha256,
        "action-set compressor escaped its operator demand",
    )
    _require(
        type(selected_snippets) is tuple
        and bool(selected_snippets)
        and all(type(row) is SelectedExactSnippet for row in selected_snippets),
        "action-set compressor requires selected exact snippets",
    )
    _require(
        type(proposals) is tuple
        and all(type(row) is ExactActionSetFactProposal for row in proposals),
        "action-set proposals changed immutable type",
    )
    require_sha256(sealed_selection_artifact_sha256, "action-set selection artifact")
    _require(
        type(payload_token_cap) is int and payload_token_cap > 0,
        "action-set payload token cap changed",
    )
    _require(
        tuple(row.selection_ordinal for row in selected_snippets)
        == tuple(sorted(row.selection_ordinal for row in selected_snippets))
        and len({row.selection_ordinal for row in selected_snippets})
        == len(selected_snippets)
        and len({row.candidate_id for row in selected_snippets})
        == len(selected_snippets),
        "action-set selected order or candidate identity changed",
    )
    selection_receipts = {row.selection_receipt_sha256 for row in selected_snippets}
    _require(
        len(selection_receipts) == 1,
        "action-set snippets escaped one upstream selection",
    )
    selection_receipt = next(iter(selection_receipts))
    if operator_spec.required_evidence_role is not None:
        _require(
            all(
                row.role == operator_spec.required_evidence_role
                for row in selected_snippets
            ),
            "action-set selected snippet role escaped operator requirement",
        )

    bound = _bind_candidates(demand, selected_snippets, proposals)
    bindings, handles = _allocate_bindings(
        bound,
        selected_snippets,
        sealed_selection_artifact_sha256=sealed_selection_artifact_sha256,
        selection_receipt_sha256=selection_receipt,
        handle_start=handle_start,
        group_start=group_start,
        preserved_bindings_by_candidate_id=preserved_bindings_by_candidate_id,
    )
    facts = _compress_facts(bound, handles)
    proposed_ids = {row.selected_candidate_id for row in bound}
    exclusions = tuple(
        PostSelectionSnippetExclusion(
            selected_candidate_id=row.candidate_id,
            selected_quote_sha256=row.quote_sha256,
            selected_local_binding_receipt_sha256=(
                row.local_binding_receipt_sha256
            ),
        )
        for row in selected_snippets
        if row.candidate_id not in proposed_ids
    )
    satisfied = len(facts) == demand.cardinality
    closure = ActionSetSupportClosureReceipt(
        demand_receipt_sha256=demand.receipt_sha256,
        selection_receipt_sha256=selection_receipt,
        selected_snippet_count=len(selected_snippets),
        candidate_count_before_dedup=len(bound),
        distinct_supported_member_count=len(facts),
        explicit_cardinality=demand.cardinality,
        bound_candidate_receipt_sha256s=tuple(
            row.receipt_sha256 for row in bound
        ),
        compressed_fact_receipt_sha256s=tuple(
            row.receipt_sha256 for row in facts
        ),
        post_selection_exclusion_receipt_sha256s=tuple(
            row.receipt_sha256 for row in exclusions
        ),
        explicit_cardinality_satisfied=satisfied,
        support_frontier_closed=satisfied,
        closure_basis=(
            "explicit_cardinality_satisfied_by_exact_action_linked_members"
            if satisfied
            else "explicit_cardinality_not_satisfied"
        ),
    )
    parsed = parse_typed_items(
        _raw_typed_items(facts),
        operator_spec=operator_spec,
        bindings=bindings,
    )
    _require(
        not parsed.rejected_items
        and len(parsed.accepted_items) == len(facts)
        and all(row.supported_slot_ids for row in parsed.accepted_items),
        "action-set facts failed typed item/slot validation",
    )
    contribution = TypedEvidenceContribution(
        mechanism_id=MECHANISM_ID,
        bindings=bindings,
        parsed=parsed,
        sealed_artifact_sha256=sealed_selection_artifact_sha256,
        # Exact question cardinality proves only that this selected-scope
        # witness contains the requested number of distinct supported members.
        # The upstream selector may still be truncated or semantically open.
        frontier_mode=FrontierMode.BOUNDED,
        truncated=False,
    )

    # Compute the compact payload before constructing the immutable result.
    by_receipt = {row.receipt_sha256: row for row in bound}
    provider = {
        "cardinality": demand.cardinality,
        "facts": [
            {
                "action": fact.action_concept,
                "member": fact.member_text,
                "relation_anchor_terms": list(fact.relation_anchor_terms),
                "support": [
                    {
                        "evidence_handle": handle,
                        "quote": by_receipt[receipt].support_quote,
                        "quote_sha256": by_receipt[receipt].support_quote_sha256,
                    }
                    for handle, receipt in zip(
                        fact.handle_ids,
                        fact.bound_candidate_receipt_sha256s,
                        strict=True,
                    )
                ],
            }
            for fact in facts
        ],
        "format": PROVIDER_FORMAT,
        "support_frontier": {
            "closed": closure.support_frontier_closed,
            "closure_basis": closure.closure_basis,
            "generic_frontier_closed": False,
            "receipt_sha256": closure.receipt_sha256,
            "scope": "selected_action_linked_members_only",
            "semantic_absence_may_be_inferred": False,
        },
    }
    rendered = json.dumps(
        provider,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    provider_tokens = count_tokens(rendered)
    _require(
        provider_tokens <= payload_token_cap,
        "action-set exact facts exceed their compact payload cap",
    )
    return ActionLinkedSetCompression(
        demand=demand,
        selection_receipt_sha256=selection_receipt,
        selected_snippet_receipt_sha256s=tuple(
            row.receipt_sha256 for row in selected_snippets
        ),
        bound_candidates=bound,
        facts=facts,
        post_selection_exclusions=exclusions,
        closure=closure,
        bindings=bindings,
        parsed=parsed,
        contribution=contribution,
        provider_payload_token_proxy=provider_tokens,
        payload_token_cap=payload_token_cap,
    )


def _typed_item_role(item: TypedEvidenceItem) -> str:
    relation = (item.relation or "").casefold()
    if "memory_role:user" in relation or "authored_by_user" in relation:
        return "user"
    if "memory_role:assistant" in relation or "authored_by_assistant" in relation:
        return "assistant"
    if re.match(r"^(?:i\b|i['’]ve\b|i['’]m\b|my\b)", item.summary, re.I):
        return "user"
    return "unknown"


def compress_selected_typed_action_set_evidence(
    demand: ActionLinkedSetDemand,
    selected_items: tuple[TypedEvidenceItem, ...],
    selected_bindings: tuple[EvidenceHandleBinding, ...],
    operator_spec: TypedOperatorSpec,
    *,
    selection_receipt_sha256: str,
    payload_token_cap: int = DEFAULT_PAYLOAD_TOKEN_CAP,
) -> ActionLinkedSetCompression:
    """Select and compress exact action members from an existing typed packet.

    Every input item must be a one-handle exact snippet: its full summary hash
    must equal both the binding payload and citation hashes.  The resulting
    contribution reuses those binding objects byte-for-byte, so the original
    evidence/source receipts remain authoritative.
    """

    if type(demand) is not ActionLinkedSetDemand:
        raise TypeError("demand must be an exact ActionLinkedSetDemand")
    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be an exact TypedOperatorSpec")
    _require(
        type(selected_items) is tuple
        and bool(selected_items)
        and all(type(row) is TypedEvidenceItem for row in selected_items),
        "action-set selected typed items changed immutable type",
    )
    _require(
        type(selected_bindings) is tuple
        and bool(selected_bindings)
        and all(type(row) is EvidenceHandleBinding for row in selected_bindings),
        "action-set selected typed bindings changed immutable type",
    )
    require_sha256(selection_receipt_sha256, "action-set typed selection")
    binding_by_handle = {row.handle_id: row for row in selected_bindings}
    _require(
        len(binding_by_handle) == len(selected_bindings),
        "action-set selected typed bindings repeat",
    )
    snippets: list[SelectedExactSnippet] = []
    preserved: dict[str, EvidenceHandleBinding] = {}
    for ordinal, item in enumerate(selected_items):
        _require(
            len(item.handle_ids) == 1,
            "action-set selected typed item must have one exact handle",
        )
        binding = binding_by_handle.get(item.handle_ids[0])
        digest = quote_sha256(item.summary)
        _require(
            binding is not None
            and binding.payload_sha256 == digest
            and binding.citation_sha256 == digest
            and binding.citation_char_count == len(item.summary),
            "action-set typed item summary is not its exact bound citation",
        )
        assert binding is not None
        snippet = SelectedExactSnippet(
            selection_ordinal=ordinal,
            candidate_id=item.item_id,
            source_group_handle=binding.source_group_handle,
            quote=item.summary,
            quote_sha256=digest,
            role=_typed_item_role(item),
            created_at=item.date or "undated",
            evidence_receipt_sha256=binding.evidence_receipt_sha256,
            local_binding_receipt_sha256=binding.receipt_sha256,
            local_source_locator_sha256=binding.local_source_locator_sha256,
            selection_receipt_sha256=selection_receipt_sha256,
            token_count=count_tokens(item.summary),
        )
        snippets.append(snippet)
        preserved[item.item_id] = binding
    frozen_snippets = tuple(snippets)
    proposals = infer_exact_action_set_fact_proposals(demand, frozen_snippets)
    supporting_artifacts = {
        preserved[row.selected_candidate_id].sealed_artifact_sha256
        for row in proposals
    }
    _require(
        len(supporting_artifacts) == 1,
        "action-set exact typed support spans multiple sealed artifacts",
    )
    return compress_action_linked_set_evidence(
        demand,
        frozen_snippets,
        proposals,
        operator_spec,
        sealed_selection_artifact_sha256=next(iter(supporting_artifacts)),
        payload_token_cap=payload_token_cap,
        preserved_bindings_by_candidate_id=preserved,
    )


def typed_packet_from_action_linked_set_compression(
    operator_spec: TypedOperatorSpec,
    compression: ActionLinkedSetCompression,
    *,
    output_token_reserve: int = 768,
) -> TypedEvidencePacket:
    """Build the standalone specialist packet used for deterministic SET execution."""

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be an exact TypedOperatorSpec")
    if type(compression) is not ActionLinkedSetCompression:
        raise TypeError("compression must be an exact ActionLinkedSetCompression")
    _require(
        compression.demand.question_sha256 == operator_spec.question_sha256
        and compression.demand.operator_spec_receipt_sha256
        == operator_spec.receipt_sha256,
        "action-set packet escaped its operator demand",
    )
    return build_typed_evidence_packet(
        operator_spec,
        compression.bindings,
        compression.parsed,
        sealed_input_artifact_sha256s=(
            compression.contribution.sealed_artifact_sha256,
        ),
        frontier_mode=compression.contribution.frontier_mode,
        conflict_policy=ConflictPolicy.QUARANTINE,
        output_token_reserve=output_token_reserve,
        truncated=False,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
    )


__all__ = [
    "ActionLinkedSetCompression",
    "ActionLinkedSetDemand",
    "ActionSetCompressionError",
    "ActionSetSupportClosureReceipt",
    "BoundActionSetCandidate",
    "CompressedActionSetFact",
    "DEFAULT_PAYLOAD_TOKEN_CAP",
    "ExactActionSetFactProposal",
    "FORMAT",
    "MECHANISM_ID",
    "PostSelectionSnippetExclusion",
    "SelectedExactSnippet",
    "compile_action_linked_set_demand",
    "compress_action_linked_set_evidence",
    "compress_selected_typed_action_set_evidence",
    "infer_exact_action_set_fact_proposals",
    "locate_exact_action_set_fact",
    "typed_packet_from_action_linked_set_compression",
]
