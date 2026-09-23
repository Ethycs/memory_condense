"""Strict ordered-story replacement for an ambiguous hot typed lane.

The fast typed overlay is deliberately broad: for an ordered question such as
"the three trips", every completed travel assertion is a plausible candidate.
When that deterministic frontier is demonstrably ambiguous, this module may
replace (never append to) the typed selection with the temporal specialist's
content-linked story bundle.

The public resolver is fail-open.  It returns the *same* baseline result object
unless the question-only ambiguity policy fires and the specialist proves an
exact-cardinality, chronological bundle of distinct dated, first-person user
events.  It does not accept question IDs, references, predictions, labels, or
provider clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256

from .contracts import assert_gold_blind, identity_sha256, require_sha256, require_text
from .full_store_slot_closure import FullStoreWindowIndex
from .hot_typed_witness import (
    HotTypedWitnessResult,
    assess_ordered_list_ambiguity,
)
from .temporal_insufficiency_specialist import (
    BundleRole,
    SpecialistRoute,
    TemporalInsufficiencyResult,
    TemporalInsufficiencySpecialistError,
    scan_temporal_insufficiency_specialist,
)


MECHANISM_ID = "hot_ambiguous_ordered_story_replacement_v1"
WITNESS_FORMAT = "memory-condense-hot-ordered-story-witness-v1"
RECEIPT_FORMAT = "memory-condense-hot-ordered-story-replacement-receipt-v1"


class HotOrderedStoryReplacementError(ValueError):
    """Raised when a successfully selected replacement loses its contract."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise HotOrderedStoryReplacementError(message)


@dataclass(frozen=True, slots=True)
class HotOrderedStoryWitness:
    """One exact specialist quote exposed through the hot typed-lane seam."""

    candidate_id: str
    source_id: str
    span: EvidenceSpan
    quote: str
    quote_sha256: str
    token_count: int
    event_date: str
    selection_axes: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.candidate_id, "ordered-story candidate")
        require_text(self.source_id, "ordered-story source")
        require_text(self.quote, "ordered-story quote")
        require_sha256(self.quote_sha256, "ordered-story quote")
        require_text(self.event_date, "ordered-story event date")
        _require(
            self.span.source_id == self.source_id
            and self.span.role == "user"
            and self.span.quote_sha256 == self.quote_sha256
            and quote_sha256(self.quote) == self.quote_sha256
            and count_tokens(self.quote) == self.token_count,
            "ordered-story witness lost exact user provenance",
        )
        _require(
            type(self.selection_axes) is tuple
            and len(self.selection_axes) == len(set(self.selection_axes))
            and all(type(value) is str and value for value in self.selection_axes),
            "ordered-story selection axes changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "ordered-story witness receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "candidate_id": self.candidate_id,
            "event_date": self.event_date,
            "format": WITNESS_FORMAT,
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "selection_axes": list(self.selection_axes),
            "source_id": self.source_id,
            "span": self.span.identity_payload(),
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotOrderedStoryReplacementReceipt:
    """Causal binding from broad typed selection to its narrow substitute."""

    question_sha256: str
    baseline_query_receipt_sha256: str
    ambiguity_decision_receipt_sha256: str
    specialist_receipt_sha256: str
    temporal_bundle_receipt_sha256: str
    requested_cardinality: int
    ordered_candidate_ids: tuple[str, ...]
    ordered_event_dates: tuple[str, ...]
    selected_evidence_tokens: int
    status: Literal["applicable_ordered_story_replaced"] = (
        "applicable_ordered_story_replaced"
    )
    baseline_appended: Literal[False] = False
    replacement_validated_before_emission: Literal[True] = True
    new_provider_calls: Literal[0] = 0
    model_calls: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.question_sha256, "ordered-story question"),
            (self.baseline_query_receipt_sha256, "ordered-story baseline"),
            (self.ambiguity_decision_receipt_sha256, "ordered-story decision"),
            (self.specialist_receipt_sha256, "ordered-story specialist"),
            (self.temporal_bundle_receipt_sha256, "ordered-story bundle"),
        ):
            require_sha256(value, label)
        _require(
            type(self.requested_cardinality) is int
            and self.requested_cardinality > 0,
            "ordered-story cardinality changed",
        )
        _require(
            len(self.ordered_candidate_ids) == self.requested_cardinality
            and len(set(self.ordered_candidate_ids)) == self.requested_cardinality
            and all(
                type(value) is str and len(value) == 64
                for value in self.ordered_candidate_ids
            ),
            "ordered-story candidate cardinality changed",
        )
        _require(
            len(self.ordered_event_dates) == self.requested_cardinality
            and len(set(self.ordered_event_dates)) == self.requested_cardinality
            and tuple(sorted(self.ordered_event_dates)) == self.ordered_event_dates,
            "ordered-story dates are not distinct chronological operands",
        )
        _require(
            type(self.selected_evidence_tokens) is int
            and self.selected_evidence_tokens > 0,
            "ordered-story token accounting changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "ordered-story replacement receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_ordered_story_replacement")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "ambiguity_decision_receipt_sha256": (
                self.ambiguity_decision_receipt_sha256
            ),
            "baseline_appended": False,
            "baseline_query_receipt_sha256": self.baseline_query_receipt_sha256,
            "format": RECEIPT_FORMAT,
            "gold_loaded": False,
            "model_calls": 0,
            "new_provider_calls": 0,
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "ordered_event_dates": list(self.ordered_event_dates),
            "question_sha256": self.question_sha256,
            "replacement_validated_before_emission": True,
            "requested_cardinality": self.requested_cardinality,
            "selected_evidence_tokens": self.selected_evidence_tokens,
            "specialist_receipt_sha256": self.specialist_receipt_sha256,
            "status": self.status,
            "temporal_bundle_receipt_sha256": (
                self.temporal_bundle_receipt_sha256
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotOrderedStoryReplacementResult:
    """Typed-lane compatible result containing only the accepted story."""

    dated_question: str
    selected_before_dedup: tuple[HotOrderedStoryWitness, ...]
    witnesses: tuple[HotOrderedStoryWitness, ...]
    receipt: HotOrderedStoryReplacementReceipt

    def __post_init__(self) -> None:
        require_text(self.dated_question, "ordered-story dated question")
        _require(
            self.selected_before_dedup == self.witnesses
            and tuple(row.candidate_id for row in self.witnesses)
            == self.receipt.ordered_candidate_ids
            and tuple(row.event_date for row in self.witnesses)
            == self.receipt.ordered_event_dates
            and sum(row.token_count for row in self.witnesses)
            == self.receipt.selected_evidence_tokens,
            "ordered-story result differs from its receipt",
        )

    @property
    def status(self) -> str:
        return self.receipt.status

    @property
    def origin_chunk_ids(self) -> tuple[str, ...]:
        return tuple(row.span.chunk_id for row in self.witnesses)


SpecialistScanner = Callable[
    [FullStoreWindowIndex, str], TemporalInsufficiencyResult
]


def _validated_replacement(
    baseline: HotTypedWitnessResult,
    decision: object,
    specialist: TemporalInsufficiencyResult,
) -> HotOrderedStoryReplacementResult | None:
    """Return a strict replacement, or ``None`` without weakening baseline."""

    cardinality = getattr(decision, "requested_cardinality", None)
    bundle = specialist.temporal_bundle
    if (
        type(cardinality) is not int
        or cardinality <= 0
        or SpecialistRoute.TEMPORAL_ORDER not in specialist.routes
        or bundle is None
        or bundle.route != SpecialistRoute.TEMPORAL_ORDER.value
        or bundle.requested_cardinality != cardinality
        or len(bundle.ordered_candidate_ids) != cardinality
    ):
        return None

    candidates = {row.candidate_id: row for row in specialist.candidates}
    bindings = {row.candidate_id: row for row in specialist.local_bindings}
    ordered_candidates = tuple(
        candidates.get(candidate_id)
        for candidate_id in bundle.ordered_candidate_ids
    )
    ordered_bindings = tuple(
        bindings.get(candidate_id)
        for candidate_id in bundle.ordered_candidate_ids
    )
    if any(row is None for row in (*ordered_candidates, *ordered_bindings)):
        return None

    # The concrete types are recovered by the membership check above.  Keep
    # the validation as data predicates so any incomplete specialist result
    # simply leaves the original typed result untouched.
    dates = tuple(str(row.event_date) for row in ordered_candidates)
    if (
        any(row.event_date is None for row in ordered_candidates)
        or len(set(dates)) != cardinality
        or tuple(sorted(dates)) != dates
        or len({row.source_id for row in ordered_bindings}) != cardinality
        or any(
            not (
                candidate.first_person_assertion
                and candidate.role == "user"
                and candidate.bundle_role is BundleRole.ORDERED_OPERAND
                and "concrete_completed_event_surface" in candidate.selection_axes
                and binding.span.role == "user"
                and binding.span.source_id == binding.source_id
                and binding.source_id
                and candidate.quote_sha256 == binding.quote_sha256
                and candidate.citation_binding_receipt_sha256
                == binding.receipt_sha256
            )
            for candidate, binding in zip(
                ordered_candidates, ordered_bindings, strict=True
            )
        )
    ):
        return None

    witnesses = tuple(
        HotOrderedStoryWitness(
            candidate_id=candidate.candidate_id,
            source_id=binding.source_id,
            span=binding.span,
            quote=candidate.quote,
            quote_sha256=candidate.quote_sha256,
            token_count=candidate.token_count,
            event_date=str(candidate.event_date),
            selection_axes=tuple(candidate.selection_axes),
        )
        for candidate, binding in zip(
            ordered_candidates, ordered_bindings, strict=True
        )
    )
    receipt = HotOrderedStoryReplacementReceipt(
        question_sha256=quote_sha256(baseline.dated_question),
        baseline_query_receipt_sha256=baseline.receipt.receipt_sha256,
        ambiguity_decision_receipt_sha256=getattr(decision, "receipt_sha256"),
        specialist_receipt_sha256=specialist.receipt.receipt_sha256,
        temporal_bundle_receipt_sha256=bundle.receipt_sha256,
        requested_cardinality=cardinality,
        ordered_candidate_ids=tuple(row.candidate_id for row in witnesses),
        ordered_event_dates=tuple(row.event_date for row in witnesses),
        selected_evidence_tokens=sum(row.token_count for row in witnesses),
    )
    return HotOrderedStoryReplacementResult(
        dated_question=baseline.dated_question,
        selected_before_dedup=witnesses,
        witnesses=witnesses,
        receipt=receipt,
    )


def replace_ambiguous_ordered_typed_witnesses(
    index: FullStoreWindowIndex,
    dated_question: str,
    baseline: HotTypedWitnessResult,
    /,
    *,
    specialist_scanner: SpecialistScanner = scan_temporal_insufficiency_specialist,
) -> HotTypedWitnessResult | HotOrderedStoryReplacementResult:
    """Replace only a validated high-ambiguity ordered temporal selection.

    A false gate, specialist failure, or incomplete bundle returns ``baseline``
    by object identity.  This makes the provider-bound v2 lane byte-semantic in
    every non-escalated/fail-open case.
    """

    _require(type(index) is FullStoreWindowIndex, "replacement requires exact index")
    require_text(dated_question, "ordered-story dated question")
    _require(
        type(baseline) is HotTypedWitnessResult
        and baseline.dated_question == dated_question,
        "replacement baseline differs from the dated question",
    )
    decision = assess_ordered_list_ambiguity(baseline)
    if not decision.escalate:
        return baseline
    try:
        specialist = specialist_scanner(index, dated_question)
    except TemporalInsufficiencySpecialistError:
        return baseline
    replacement = _validated_replacement(baseline, decision, specialist)
    return baseline if replacement is None else replacement


__all__ = [
    "HotOrderedStoryReplacementError",
    "HotOrderedStoryReplacementReceipt",
    "HotOrderedStoryReplacementResult",
    "HotOrderedStoryWitness",
    "MECHANISM_ID",
    "replace_ambiguous_ordered_typed_witnesses",
]
