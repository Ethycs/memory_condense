"""Minimal local scanner for :mod:`typed_active_reconstruction` requests.

The scanner reads only the immutable index and exact operator/temporal objects
already carried by the request.  It never reconstructs a dated question or
calls an operator compiler.  Candidate discovery is the union of cue postings,
a small audited action-equivalence table, and optional opaque component/source
affinity derived upstream from already selected evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from .full_store_slot_closure import (
    FullStoreSlotCandidate,
    LocalCitationBinding,
)
from .typed_action_semantics import (
    canonical_action_concepts,
    canonical_action_proof_terms,
)
from .typed_active_reconstruction import (
    ActiveReconstructionCandidateMatch,
    ActiveReconstructionDecision,
    ActiveReconstructionScanBatch,
    ActiveReconstructionScanRequest,
    ActiveReconstructionSupportKind,
    ActiveIndexLookup,
    TypedActiveReconstructionResult,
    active_candidate_id_for_window,
    active_cue_posting_fanout_cap,
    active_history_obligation_supported,
    active_index_lookup,
    active_selective_cue_terms,
    active_supported_slot_ids,
    active_temporal_support,
    candidate_projection_receipt_sha256,
    citation_span_receipt_sha256,
)


SCANNER_FORMAT = "memory-condense-typed-active-full-store-scanner-v1"
AUDIT_FORMAT = f"{SCANNER_FORMAT}-audit"
PRIORITY_FORMAT = f"{SCANNER_FORMAT}-cue-support-priority"
SUBCHANNEL_RECEIPT_FORMAT = f"{SCANNER_FORMAT}-subchannel-receipt-v1"
SELECTION_RECEIPT_FORMAT = f"{SCANNER_FORMAT}-selection-receipt-v1"


class TypedActiveFullStoreScannerError(MatchedEvalContractError):
    """Raised when a local scan escapes its exact request or budget."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedActiveFullStoreScannerError(message)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class ActiveFullStoreScanSubchannel(str, Enum):
    """Independent bounded reads over one immutable full-store index."""

    EXACT_SOURCE = "exact_source"
    ENCLOSING_HISTORY = "enclosing_history"
    GLOBAL_FACT_CUE = "global_fact_cue"


@dataclass(frozen=True, slots=True)
class _Draft:
    window_index: int
    subchannel: ActiveFullStoreScanSubchannel
    support_kind: ActiveReconstructionSupportKind
    source_affinity: bool
    component_affinity: bool
    supported_slot_ids: tuple[str, ...]
    matched_cue_terms: tuple[str, ...]
    matched_child_terms: tuple[str, ...]
    action_concept: str | None
    supporting_cue_receipt_sha256s: tuple[str, ...]
    temporal_distance_days: int | None
    temporal_support: bool
    direct_support_count: int
    fact_cue_support: bool
    best_cue_term_rank: int | None
    best_posting_fanout: int | None
    token_count: int
    identity_sha256: str

    @property
    def support_key(self) -> tuple[int, ...]:
        return (
            int(self.source_affinity),
            int(self.component_affinity),
            len(self.supported_slot_ids),
            int(
                self.support_kind
                is ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
            ),
            int(self.temporal_support),
            len(self.supporting_cue_receipt_sha256s),
            self.direct_support_count,
        )

    @property
    def fact_priority_key(self) -> tuple[object, ...]:
        """Prefer obligations, ranked cues, and low posting fanout."""

        return (
            -len(self.supported_slot_ids),
            -int(
                self.support_kind
                is ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
            ),
            -int(self.temporal_support),
            -int(self.fact_cue_support),
            -len(self.supporting_cue_receipt_sha256s),
            -self.direct_support_count,
            (
                1_000_000
                if self.best_cue_term_rank is None
                else self.best_cue_term_rank
            ),
            (
                1_000_000_000
                if self.best_posting_fanout is None
                else self.best_posting_fanout
            ),
        )


@dataclass(frozen=True, slots=True)
class _CoverageFeatures:
    required_slots: frozenset[str]
    cue_parents: frozenset[str]
    cue_receipts: frozenset[str]
    direct_support: frozenset[str]
    temporal: frozenset[str]
    actions: frozenset[str]
    personalization: frozenset[str]
    roles: frozenset[str]
    source_key_sha256: str
    history_key_sha256: str
    turn_key_sha256: str
    direct_quality: int
    protocol_only_ultrashort: bool
    density_numerator: int


@dataclass(slots=True)
class _CoverageState:
    required_slots: set[str]
    cue_parents: set[str]
    cue_receipts: set[str]
    direct_support: set[str]
    temporal: set[str]
    actions: set[str]
    personalization: set[str]
    roles: set[str]
    sources: set[str]
    histories: set[str]
    turns: set[str]
    features_by_draft_receipt: dict[str, _CoverageFeatures]


_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|i'm|i've|me|my|mine|we|we've|our|ours)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ActiveFullStoreScanSubchannelReceipt:
    """Sealed population/reservation/selection receipt for one subchannel."""

    subchannel: ActiveFullStoreScanSubchannel
    request_receipt_sha256: str
    reserved_candidate_cap: int
    reserved_token_cap: int
    candidate_population_count: int
    candidate_population_receipt_sha256: str
    reserved_selected_draft_receipt_sha256s: tuple[str, ...]
    spillover_selected_draft_receipt_sha256s: tuple[str, ...]
    selected_token_count: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.subchannel) is ActiveFullStoreScanSubchannel,
            "scan subchannel changed",
        )
        require_sha256(self.request_receipt_sha256, "subchannel request")
        require_sha256(
            self.candidate_population_receipt_sha256,
            "subchannel population",
        )
        for value, label in (
            (self.reserved_candidate_cap, "subchannel candidate cap"),
            (self.reserved_token_cap, "subchannel token cap"),
            (self.candidate_population_count, "subchannel population count"),
            (self.selected_token_count, "subchannel selected tokens"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        selected = (
            *self.reserved_selected_draft_receipt_sha256s,
            *self.spillover_selected_draft_receipt_sha256s,
        )
        _require(
            len(set(selected)) == len(selected)
            and len(selected) <= self.candidate_population_count,
            "subchannel selected drafts changed",
        )
        for value in selected:
            require_sha256(value, "subchannel selected draft")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "subchannel receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="active_scan_subchannel")

    @property
    def selected_candidate_count(self) -> int:
        return len(self.reserved_selected_draft_receipt_sha256s) + len(
            self.spillover_selected_draft_receipt_sha256s
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "candidate_population_count": self.candidate_population_count,
            "candidate_population_receipt_sha256": (
                self.candidate_population_receipt_sha256
            ),
            "format": SUBCHANNEL_RECEIPT_FORMAT,
            "request_receipt_sha256": self.request_receipt_sha256,
            "reserved_candidate_cap": self.reserved_candidate_cap,
            "reserved_selected_draft_receipt_sha256s": list(
                self.reserved_selected_draft_receipt_sha256s
            ),
            "reserved_token_cap": self.reserved_token_cap,
            "selected_candidate_count": self.selected_candidate_count,
            "selected_token_count": self.selected_token_count,
            "spillover_selected_draft_receipt_sha256s": list(
                self.spillover_selected_draft_receipt_sha256s
            ),
            "subchannel": self.subchannel.value,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class _FixedSubchannelSelection:
    selected: tuple[_Draft, ...]
    subchannel_receipts: tuple[ActiveFullStoreScanSubchannelReceipt, ...]
    candidate_population_count: int
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _ScanPlan:
    lookup: ActiveIndexLookup
    candidate_indices: tuple[int, ...]
    direct_candidate_indices: frozenset[int]
    action_candidate_indices: frozenset[int]
    history_relevance_indices: frozenset[int]
    selective_cue_terms: frozenset[str]
    direct_terms_by_window: Mapping[int, Mapping[str, tuple[str, ...]]]
    source_affinity_cues: Mapping[str, tuple[str, ...]]
    history_affinity_cues: Mapping[str, tuple[str, ...]]
    action_cues: Mapping[str, tuple[str, ...]]
    cue_term_rank: Mapping[tuple[str, str], int]


def _append_receipt(
    values: dict[str, list[str]], key: str, receipt_sha256: str
) -> None:
    target = values.setdefault(key, [])
    if receipt_sha256 not in target:
        target.append(receipt_sha256)


def _slot_relevance_indices(
    request: ActiveReconstructionScanRequest,
    lookup: ActiveIndexLookup,
) -> set[int]:
    """Return a postings-only superset of exact original-slot support."""

    numeric = set(request.index.numeric_window_indices)
    relevant: set[int] = set()
    for slot in request.operator_spec.required_slots:
        if slot.relation_constraint is not None:
            continue
        counts: dict[int, int] = {}
        for term in slot.match_terms:
            for window_index in request.index.term_postings.get(term, ()):
                counts[window_index] = counts.get(window_index, 0) + 1
        supported = {
            window_index
            for window_index, count in counts.items()
            if count >= slot.minimum_match_term_count
        }
        slot_actions = canonical_action_concepts(
            " ".join((slot.label, *slot.match_terms))
        )
        if slot.minimum_match_term_count <= 1:
            for concept in slot_actions:
                supported.update(
                    lookup.window_indices_by_action_concept.get(concept, ())
                )
        if slot.requires_numeric:
            supported.intersection_update(numeric)
        relevant.update(supported)
    return relevant


def _temporal_relevance_indices(
    request: ActiveReconstructionScanRequest,
) -> set[int]:
    relevant: set[int] = set()
    for day, postings in request.index.date_postings.items():
        _distance, supported = active_temporal_support(
            day, request.temporal_target
        )
        if supported:
            relevant.update(postings)
    return relevant


def _scan_plan(request: ActiveReconstructionScanRequest) -> _ScanPlan:
    lookup = active_index_lookup(request.index)
    selective_terms = active_selective_cue_terms(request)
    direct_indices: set[int] = set()
    action_indices: set[int] = set()
    source_indices: set[int] = set()
    history_indices: set[int] = set()
    source_cues: dict[str, list[str]] = {}
    history_cues: dict[str, list[str]] = {}
    action_cues: dict[str, list[str]] = {}
    direct_terms_by_window: dict[int, dict[str, list[str]]] = {}
    cue_term_rank: dict[tuple[str, str], int] = {}
    for cue in request.cues:
        for term_rank, term in enumerate(cue.terms):
            cue_term_rank[(cue.receipt_sha256, term)] = term_rank
            if term in selective_terms:
                postings = request.index.term_postings.get(term, ())
                direct_indices.update(postings)
                for window_index in postings:
                    by_cue = direct_terms_by_window.setdefault(window_index, {})
                    by_cue.setdefault(cue.receipt_sha256, []).append(term)
        for concept in cue.action_concepts:
            _append_receipt(action_cues, concept, cue.receipt_sha256)
            action_indices.update(
                lookup.window_indices_by_action_concept.get(concept, ())
            )
        affinity = cue.selected_evidence_affinity
        if affinity is None:
            continue
        _append_receipt(
            source_cues, affinity.source_key_sha256, cue.receipt_sha256
        )
        _append_receipt(
            history_cues, affinity.component_key_sha256, cue.receipt_sha256
        )
    for key in source_cues:
        source_indices.update(lookup.window_indices_by_source_key.get(key, ()))
    for key in history_cues:
        history_indices.update(lookup.window_indices_by_history_key.get(key, ()))

    relevance = set(direct_indices) | set(action_indices)
    relevance.update(_slot_relevance_indices(request, lookup))
    relevance.update(_temporal_relevance_indices(request))
    if (
        request.operator_spec.answer_shape.value == "number"
        and not request.operator_spec.required_slots
    ):
        relevance.update(request.index.numeric_window_indices)
    qualified_history = history_indices & relevance
    indices = direct_indices | action_indices | source_indices | qualified_history
    required_role = request.operator_spec.required_evidence_role
    if required_role is not None:
        indices.intersection_update(
            request.index.role_postings.get(required_role, ())
        )
    return _ScanPlan(
        lookup=lookup,
        candidate_indices=tuple(sorted(indices)),
        direct_candidate_indices=frozenset(direct_indices),
        action_candidate_indices=frozenset(action_indices),
        history_relevance_indices=frozenset(relevance),
        selective_cue_terms=selective_terms,
        direct_terms_by_window={
            window_index: {
                receipt: tuple(terms) for receipt, terms in by_cue.items()
            }
            for window_index, by_cue in direct_terms_by_window.items()
        },
        source_affinity_cues={
            key: tuple(values) for key, values in source_cues.items()
        },
        history_affinity_cues={
            key: tuple(values) for key, values in history_cues.items()
        },
        action_cues={key: tuple(values) for key, values in action_cues.items()},
        cue_term_rank=cue_term_rank,
    )


def _candidate_indices(request: ActiveReconstructionScanRequest) -> tuple[int, ...]:
    return _scan_plan(request).candidate_indices


def _draft(
    request: ActiveReconstructionScanRequest,
    window_index: int,
    plan: _ScanPlan,
    *,
    subchannel: ActiveFullStoreScanSubchannel | None = None,
) -> _Draft | None:
    window = request.index.windows[window_index]
    row = window.row
    if (
        request.operator_spec.required_evidence_role is not None
        and row.role != request.operator_spec.required_evidence_role
    ):
        return None

    history_key = plan.lookup.history_key_by_window[window_index]
    source_key = plan.lookup.source_key_by_window[window_index]
    direct_by_cue = plan.direct_terms_by_window.get(window_index, {})
    quote = row.text[window.start_char : window.end_char]
    child_action_concepts = set(
        plan.lookup.action_concepts_by_window[window_index]
    )
    action_by_concept = {
        concept: plan.action_cues[concept]
        for concept in sorted(child_action_concepts & set(plan.action_cues))
    }
    source_affinity_cues = plan.source_affinity_cues.get(source_key, ())
    component_affinity_cues = plan.history_affinity_cues.get(history_key, ())
    slots = active_supported_slot_ids(request.operator_spec, quote)
    temporal_distance, temporal = active_temporal_support(
        window.event_date, request.temporal_target
    )
    numeric = bool(
        request.operator_spec.answer_shape.value == "number"
        and not request.operator_spec.required_slots
        and window.contains_numeric_value
    )
    if component_affinity_cues and not (
        direct_by_cue or action_by_concept or slots or temporal or numeric
    ):
        component_affinity_cues = ()

    action_concept: str | None = None
    if (
        subchannel is ActiveFullStoreScanSubchannel.EXACT_SOURCE
        and not source_affinity_cues
    ):
        return None
    if (
        subchannel is ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY
        and not component_affinity_cues
    ):
        return None
    if (
        subchannel is ActiveFullStoreScanSubchannel.GLOBAL_FACT_CUE
        and not (action_by_concept or direct_by_cue)
    ):
        return None

    if (
        subchannel is ActiveFullStoreScanSubchannel.EXACT_SOURCE
        or (subchannel is None and source_affinity_cues)
    ):
        support_kind = ActiveReconstructionSupportKind.SELECTED_SOURCE_AFFINITY
        supporting_cues = _ordered_unique(source_affinity_cues)
        matched_cues: tuple[str, ...] = ()
        matched_children: tuple[str, ...] = ()
        actual_subchannel = ActiveFullStoreScanSubchannel.EXACT_SOURCE
    elif (
        subchannel is ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY
        or (subchannel is None and component_affinity_cues)
    ):
        support_kind = ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY
        supporting_cues = _ordered_unique(component_affinity_cues)
        matched_cues = ()
        matched_children = ()
        actual_subchannel = ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY
    elif direct_by_cue and request.use_coverage_aware_callback_selection:
        # In the treatment, an exact low-fanout lexical edge is the stronger
        # proof when the same window also happens to share a broad action
        # concept.  The legacy branch below intentionally remains action-first.
        support_kind = ActiveReconstructionSupportKind.DIRECT_LEXICAL
        supporting_cues = tuple(sorted(direct_by_cue))
        matched_cues = _ordered_unique(
            term
            for receipt in supporting_cues
            for term in direct_by_cue[receipt]
        )
        matched_children = matched_cues
        actual_subchannel = ActiveFullStoreScanSubchannel.GLOBAL_FACT_CUE
    elif action_by_concept:
        support_kind = ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
        action_concept = sorted(action_by_concept)[0]
        supporting_cues = _ordered_unique(action_by_concept[action_concept])
        matched_cues = ()
        matched_children = canonical_action_proof_terms(
            quote, action_concept
        )
        _require(matched_children, "action support lost its exact child term")
        actual_subchannel = ActiveFullStoreScanSubchannel.GLOBAL_FACT_CUE
    elif direct_by_cue:
        support_kind = ActiveReconstructionSupportKind.DIRECT_LEXICAL
        supporting_cues = tuple(sorted(direct_by_cue))
        matched_cues = _ordered_unique(
            term
            for receipt in supporting_cues
            for term in direct_by_cue[receipt]
        )
        matched_children = matched_cues
        actual_subchannel = ActiveFullStoreScanSubchannel.GLOBAL_FACT_CUE
    else:
        return None

    body = {
        "format": f"{SCANNER_FORMAT}-draft",
        "index_receipt_sha256": request.index.receipt_sha256,
        "window_index": window_index,
        "window_text_sha256": window.text_sha256,
    }
    if subchannel is not None:
        body.update(
            {
                "action_concept": action_concept,
                "matched_cue_terms": list(matched_cues),
                "subchannel": actual_subchannel.value,
                "support_kind": support_kind.value,
                "supporting_cue_receipt_sha256s": list(supporting_cues),
            }
        )
    ranked_term_support = tuple(
        (
            plan.cue_term_rank.get((receipt, term), 1_000_000),
            len(request.index.term_postings.get(term, ())),
        )
        for receipt, terms in direct_by_cue.items()
        for term in terms
    )
    action_fanouts = tuple(
        len(plan.lookup.window_indices_by_action_concept.get(concept, ()))
        for concept in action_by_concept
    )
    return _Draft(
        window_index=window_index,
        subchannel=actual_subchannel,
        support_kind=support_kind,
        source_affinity=bool(source_affinity_cues),
        component_affinity=bool(component_affinity_cues),
        supported_slot_ids=slots,
        matched_cue_terms=matched_cues,
        matched_child_terms=matched_children,
        action_concept=action_concept,
        supporting_cue_receipt_sha256s=supporting_cues,
        temporal_distance_days=temporal_distance,
        temporal_support=temporal,
        direct_support_count=len(matched_cues),
        fact_cue_support=bool(action_by_concept or direct_by_cue),
        best_cue_term_rank=(
            min(rank for rank, _fanout in ranked_term_support)
            if ranked_term_support
            else (0 if action_by_concept else None)
        ),
        best_posting_fanout=(
            min(
                (
                    *(fanout for _rank, fanout in ranked_term_support),
                    *action_fanouts,
                )
            )
            if ranked_term_support or action_fanouts
            else None
        ),
        token_count=window.token_count,
        identity_sha256=identity_sha256(body),
    )


def _select(
    request: ActiveReconstructionScanRequest, drafts: Sequence[_Draft]
) -> tuple[_Draft, ...]:
    remaining = list(drafts)
    selected: list[_Draft] = []
    selected_tokens = 0
    source_counts: dict[str, int] = {}
    coverage_state = (
        _new_coverage_state()
        if request.use_coverage_aware_callback_selection
        else None
    )
    while remaining and len(selected) < request.max_selected_candidates:
        draft = min(
            remaining,
            key=(
                (
                    lambda row: _coverage_priority(
                        request, row, coverage_state
                    )
                )
                if coverage_state is not None
                else (
                    lambda row: (
                        *(-value for value in row.support_key),
                        source_counts.get(
                            request.index.windows[row.window_index].row.source_id,
                            0,
                        ),
                        row.token_count,
                        row.identity_sha256,
                    )
                )
            ),
        )
        remaining.remove(draft)
        if selected_tokens + draft.token_count > request.max_selected_tokens:
            continue
        selected.append(draft)
        selected_tokens += draft.token_count
        if coverage_state is not None:
            _coverage_admit(request, draft, coverage_state)
        source = request.index.windows[draft.window_index].row.source_id
        source_counts[source] = source_counts.get(source, 0) + 1
    return tuple(selected)


_RESERVATION_ORDER = (
    ActiveFullStoreScanSubchannel.GLOBAL_FACT_CUE,
    ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY,
    ActiveFullStoreScanSubchannel.EXACT_SOURCE,
)
_SPILLOVER_CHANNEL_ORDER = {
    ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY: 0,
    ActiveFullStoreScanSubchannel.GLOBAL_FACT_CUE: 1,
    ActiveFullStoreScanSubchannel.EXACT_SOURCE: 2,
}


def _fixed_subchannel_cap_split(
    cap: int,
) -> Mapping[ActiveFullStoreScanSubchannel, int]:
    """Reserve 1:1:2 source/history/global capacity where possible."""

    _require(type(cap) is int and cap > 0, "subchannel cap changed")
    source = max(1, cap // 4) if cap >= 3 else 0
    history = max(1, cap // 4) if cap >= 2 else 0
    global_fact = cap - source - history
    return {
        ActiveFullStoreScanSubchannel.EXACT_SOURCE: source,
        ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY: history,
        ActiveFullStoreScanSubchannel.GLOBAL_FACT_CUE: global_fact,
    }


def _subchannel_drafts(
    request: ActiveReconstructionScanRequest,
    plan: _ScanPlan,
) -> Mapping[ActiveFullStoreScanSubchannel, tuple[_Draft, ...]]:
    return {
        channel: tuple(
            draft
            for window_index in plan.candidate_indices
            if (
                draft := _draft(
                    request,
                    window_index,
                    plan,
                    subchannel=channel,
                )
            )
            is not None
        )
        for channel in ActiveFullStoreScanSubchannel
    }


def _draft_priority(
    request: ActiveReconstructionScanRequest,
    draft: _Draft,
    source_counts: Mapping[str, int],
    /,
) -> tuple[object, ...]:
    return (
        *draft.fact_priority_key,
        source_counts.get(
            request.index.windows[draft.window_index].row.source_id, 0
        ),
        draft.token_count,
    )


def _coverage_features(
    request: ActiveReconstructionScanRequest,
    draft: _Draft,
) -> _CoverageFeatures:
    """Return only question/cue/provenance-derived callback obligations."""

    window = request.index.windows[draft.window_index]
    quote = window.row.text[window.start_char : window.end_char]
    cue_by_receipt = {row.receipt_sha256: row for row in request.cues}
    supporting = tuple(
        cue_by_receipt[value]
        for value in draft.supporting_cue_receipt_sha256s
    )
    required_slot_ids = {
        row.slot_id for row in request.operator_spec.required_slots
    }
    direct_support: set[str] = set()
    if draft.support_kind is ActiveReconstructionSupportKind.DIRECT_LEXICAL:
        direct_support.add("direct:lexical")
        direct_support.update(
            f"direct_term:{value}" for value in draft.matched_cue_terms
        )
        direct_quality = 3
    elif draft.supported_slot_ids:
        direct_quality = 2
    elif draft.subchannel is ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY:
        direct_quality = 1
    else:
        direct_quality = 0

    temporal: set[str] = set()
    if draft.temporal_support:
        temporal.add("temporal:evidence")
        if window.event_date is not None:
            temporal.add(f"event_date:{window.event_date}")

    requested_actions = {
        value for cue in supporting for value in cue.action_concepts
    }
    actions = {
        f"action:{value}"
        for value in set(canonical_action_concepts(quote)) & requested_actions
    }
    personalization: set[str] = set()
    if (
        request.operator_spec.personalization_required
        and _FIRST_PERSON_RE.search(quote)
    ):
        personalization.add("personalization:first_person")
    roles = {"exact_role:user"} if window.row.role == "user" else set()
    if window.row.role == request.operator_spec.required_evidence_role:
        roles.add(f"required_role:{window.row.role}")

    lookup = active_index_lookup(request.index)
    protocol_only = bool(
        draft.support_kind
        is ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
        and draft.token_count <= 4
        and not draft.supported_slot_ids
        and not draft.matched_cue_terms
        and not draft.temporal_support
    )
    slots = frozenset(set(draft.supported_slot_ids) & required_slot_ids)
    parents = frozenset(row.parent_receipt_sha256 for row in supporting)
    cues = frozenset(draft.supporting_cue_receipt_sha256s)
    density = (
        4 * len(slots)
        + 3 * len(direct_support)
        + 2 * len(parents)
        + len(cues)
        + 2 * len(temporal)
        + 2 * len(actions)
        + 2 * len(personalization)
        + 2 * len(roles)
        + int(window.contains_numeric_value)
    )
    if protocol_only:
        density = max(0, density - 8)
    return _CoverageFeatures(
        required_slots=slots,
        cue_parents=parents,
        cue_receipts=cues,
        direct_support=frozenset(direct_support),
        temporal=frozenset(temporal),
        actions=frozenset(actions),
        personalization=frozenset(personalization),
        roles=frozenset(roles),
        source_key_sha256=lookup.source_key_by_window[draft.window_index],
        history_key_sha256=lookup.history_key_by_window[draft.window_index],
        turn_key_sha256=identity_sha256(
            {
                "chunk_id": window.row.chunk_id,
                "format": f"{SCANNER_FORMAT}-turn-key-v1",
                "namespace_id": window.row.namespace_id,
                "ordinal": window.row.ordinal,
                "source_id": window.row.source_id,
                "turn_id": window.row.turn_id,
            }
        ),
        direct_quality=direct_quality,
        protocol_only_ultrashort=protocol_only,
        density_numerator=density,
    )


def _new_count(values: frozenset[str], covered: set[str]) -> int:
    return len(set(values) - covered)


def _coverage_priority(
    request: ActiveReconstructionScanRequest,
    draft: _Draft,
    covered: _CoverageState,
) -> tuple[object, ...]:
    features = covered.features_by_draft_receipt.get(draft.identity_sha256)
    if features is None:
        features = _coverage_features(request, draft)
        covered.features_by_draft_receipt[draft.identity_sha256] = features
    # Rare/direct evidence is ahead of broad action equivalence.  The feature
    # is marginal: after one direct surface is protected, a distinct action or
    # provenance obligation can still win the next bounded position.
    return (
        -_new_count(features.required_slots, covered.required_slots),
        -_new_count(features.direct_support, covered.direct_support),
        -_new_count(features.cue_parents, covered.cue_parents),
        -_new_count(features.cue_receipts, covered.cue_receipts),
        -_new_count(features.temporal, covered.temporal),
        -_new_count(features.personalization, covered.personalization),
        -_new_count(features.roles, covered.roles),
        -int(features.source_key_sha256 not in covered.sources),
        -int(features.history_key_sha256 not in covered.histories),
        -int(features.turn_key_sha256 not in covered.turns),
        -_new_count(features.actions, covered.actions),
        -features.direct_quality,
        features.protocol_only_ultrashort,
        (
            1_000_000_000
            if draft.best_posting_fanout is None
            else draft.best_posting_fanout
        ),
        -Fraction(features.density_numerator, max(1, draft.token_count)),
        draft.token_count,
        draft.identity_sha256,
    )


def _coverage_admit(
    request: ActiveReconstructionScanRequest,
    draft: _Draft,
    covered: _CoverageState,
) -> None:
    features = covered.features_by_draft_receipt.get(draft.identity_sha256)
    if features is None:
        features = _coverage_features(request, draft)
        covered.features_by_draft_receipt[draft.identity_sha256] = features
    covered.required_slots.update(features.required_slots)
    covered.cue_parents.update(features.cue_parents)
    covered.cue_receipts.update(features.cue_receipts)
    covered.direct_support.update(features.direct_support)
    covered.temporal.update(features.temporal)
    covered.actions.update(features.actions)
    covered.personalization.update(features.personalization)
    covered.roles.update(features.roles)
    covered.sources.add(features.source_key_sha256)
    covered.histories.add(features.history_key_sha256)
    covered.turns.add(features.turn_key_sha256)


def _new_coverage_state() -> _CoverageState:
    return _CoverageState(
        required_slots=set(),
        cue_parents=set(),
        cue_receipts=set(),
        direct_support=set(),
        temporal=set(),
        actions=set(),
        personalization=set(),
        roles=set(),
        sources=set(),
        histories=set(),
        turns=set(),
        features_by_draft_receipt={},
    )


def _take_bounded_drafts(
    request: ActiveReconstructionScanRequest,
    drafts: Sequence[_Draft],
    /,
    *,
    candidate_cap: int,
    token_cap: int,
    excluded_window_indices: set[int],
    coverage_state: _CoverageState | None = None,
) -> tuple[_Draft, ...]:
    if candidate_cap <= 0 or token_cap <= 0:
        return ()
    remaining = [
        row for row in drafts if row.window_index not in excluded_window_indices
    ]
    selected: list[_Draft] = []
    selected_tokens = 0
    source_counts: dict[str, int] = {}
    while remaining and len(selected) < candidate_cap:
        draft = min(
            remaining,
            key=(
                (lambda row: _coverage_priority(request, row, coverage_state))
                if coverage_state is not None
                else (
                    lambda row: _draft_priority(
                        request, row, source_counts
                    )
                    + (row.identity_sha256,)
                )
            ),
        )
        remaining.remove(draft)
        if selected_tokens + draft.token_count > token_cap:
            continue
        selected.append(draft)
        selected_tokens += draft.token_count
        excluded_window_indices.add(draft.window_index)
        if coverage_state is not None:
            _coverage_admit(request, draft, coverage_state)
        source = request.index.windows[draft.window_index].row.source_id
        source_counts[source] = source_counts.get(source, 0) + 1
    return tuple(selected)


def _fixed_subchannel_selection(
    request: ActiveReconstructionScanRequest,
    plan: _ScanPlan,
) -> _FixedSubchannelSelection:
    """Select reserved fact reads, then deterministically spend unused budget."""

    drafts_by_channel = _subchannel_drafts(request, plan)
    count_caps = _fixed_subchannel_cap_split(request.max_selected_candidates)
    token_caps = _fixed_subchannel_cap_split(request.max_selected_tokens)
    selected: list[_Draft] = []
    selected_windows: set[int] = set()
    reserved: dict[ActiveFullStoreScanSubchannel, tuple[_Draft, ...]] = {}
    spillover: dict[ActiveFullStoreScanSubchannel, list[_Draft]] = {
        channel: [] for channel in ActiveFullStoreScanSubchannel
    }
    coverage_state = (
        _new_coverage_state()
        if request.use_coverage_aware_callback_selection
        else None
    )
    # Keep the coverage-only ablation on the exact legacy channel order.  A
    # source/history-first order is justified only when the separate upstream
    # CAV treatment actually supplied receipt-bound affinity cues.
    has_selected_affinity = any(
        cue.selected_evidence_affinity is not None for cue in request.cues
    )
    reservation_order = (
        (
            ActiveFullStoreScanSubchannel.EXACT_SOURCE,
            ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY,
            ActiveFullStoreScanSubchannel.GLOBAL_FACT_CUE,
        )
        if coverage_state is not None and has_selected_affinity
        else _RESERVATION_ORDER
    )
    for channel in reservation_order:
        rows = _take_bounded_drafts(
            request,
            drafts_by_channel[channel],
            candidate_cap=count_caps[channel],
            token_cap=token_caps[channel],
            excluded_window_indices=selected_windows,
            coverage_state=coverage_state,
        )
        reserved[channel] = rows
        selected.extend(rows)

    remaining_count = request.max_selected_candidates - len(selected)
    remaining_tokens = request.max_selected_tokens - sum(
        row.token_count for row in selected
    )
    if remaining_count > 0 and remaining_tokens > 0:
        # A window may be reachable through multiple subchannels.  Keep its
        # strongest fact proof; history wins a complete tie so local context is
        # hydrated before cross-history lexical noise.
        best_by_window: dict[int, _Draft] = {}
        for channel, rows in drafts_by_channel.items():
            for row in rows:
                if row.window_index in selected_windows:
                    continue
                previous = best_by_window.get(row.window_index)
                if coverage_state is not None:
                    row_key = (
                        0
                        if row.support_kind
                        is ActiveReconstructionSupportKind.DIRECT_LEXICAL
                        else 1
                        if channel
                        is ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY
                        else 2
                        if channel is ActiveFullStoreScanSubchannel.EXACT_SOURCE
                        else 3,
                        row.best_posting_fanout or 1_000_000_000,
                        row.identity_sha256,
                    )
                    previous_key = (
                        0
                        if previous is not None
                        and previous.support_kind
                        is ActiveReconstructionSupportKind.DIRECT_LEXICAL
                        else 1
                        if previous is not None
                        and previous.subchannel
                        is ActiveFullStoreScanSubchannel.ENCLOSING_HISTORY
                        else 2
                        if previous is not None
                        and previous.subchannel
                        is ActiveFullStoreScanSubchannel.EXACT_SOURCE
                        else 3,
                        (
                            1_000_000_000
                            if previous is None
                            or previous.best_posting_fanout is None
                            else previous.best_posting_fanout
                        ),
                        "" if previous is None else previous.identity_sha256,
                    )
                    replace_previous = previous is None or row_key < previous_key
                else:
                    replace_previous = previous is None or (
                        *row.fact_priority_key,
                        _SPILLOVER_CHANNEL_ORDER[channel],
                        row.identity_sha256,
                    ) < (
                        *previous.fact_priority_key,
                        _SPILLOVER_CHANNEL_ORDER[previous.subchannel],
                        previous.identity_sha256,
                    )
                if replace_previous:
                    best_by_window[row.window_index] = row
        remaining = list(best_by_window.values())
        source_counts: dict[str, int] = {}
        while remaining and remaining_count > 0:
            draft = min(
                remaining,
                key=(
                    (
                        lambda row: _coverage_priority(
                            request, row, coverage_state
                        )
                    )
                    if coverage_state is not None
                    else (
                        lambda row: (
                            *_draft_priority(request, row, source_counts),
                            _SPILLOVER_CHANNEL_ORDER[row.subchannel],
                            row.identity_sha256,
                        )
                    )
                ),
            )
            remaining.remove(draft)
            if draft.token_count > remaining_tokens:
                continue
            selected.append(draft)
            selected_windows.add(draft.window_index)
            spillover[draft.subchannel].append(draft)
            if coverage_state is not None:
                _coverage_admit(request, draft, coverage_state)
            remaining_count -= 1
            remaining_tokens -= draft.token_count
            source = request.index.windows[draft.window_index].row.source_id
            source_counts[source] = source_counts.get(source, 0) + 1

    population_windows = {
        row.window_index
        for rows in drafts_by_channel.values()
        for row in rows
    }
    receipts: list[ActiveFullStoreScanSubchannelReceipt] = []
    for channel in ActiveFullStoreScanSubchannel:
        population = drafts_by_channel[channel]
        selected_rows = (*reserved.get(channel, ()), *spillover[channel])
        receipts.append(
            ActiveFullStoreScanSubchannelReceipt(
                subchannel=channel,
                request_receipt_sha256=request.receipt_sha256,
                reserved_candidate_cap=count_caps[channel],
                reserved_token_cap=token_caps[channel],
                candidate_population_count=len(population),
                candidate_population_receipt_sha256=identity_sha256(
                    {
                        "draft_receipt_sha256s": [
                            row.identity_sha256 for row in population
                        ],
                        "format": f"{SUBCHANNEL_RECEIPT_FORMAT}-population",
                        "index_receipt_sha256": request.index.receipt_sha256,
                        "request_receipt_sha256": request.receipt_sha256,
                        "subchannel": channel.value,
                    }
                ),
                reserved_selected_draft_receipt_sha256s=tuple(
                    row.identity_sha256 for row in reserved.get(channel, ())
                ),
                spillover_selected_draft_receipt_sha256s=tuple(
                    row.identity_sha256 for row in spillover[channel]
                ),
                selected_token_count=sum(
                    row.token_count for row in selected_rows
                ),
            )
        )
    selection_body = {
        "candidate_population_count": len(population_windows),
        "format": SELECTION_RECEIPT_FORMAT,
        "request_receipt_sha256": request.receipt_sha256,
        "selected_draft_receipt_sha256s": [
            row.identity_sha256 for row in selected
        ],
        "subchannel_receipt_sha256s": [
            row.receipt_sha256 for row in receipts
        ],
    }
    return _FixedSubchannelSelection(
        selected=tuple(selected),
        subchannel_receipts=tuple(receipts),
        candidate_population_count=len(population_windows),
        receipt_sha256=identity_sha256(selection_body),
    )


def derive_active_full_store_scan_subchannel_receipts(
    request: ActiveReconstructionScanRequest,
    /,
) -> tuple[ActiveFullStoreScanSubchannelReceipt, ...]:
    """Return deterministic receipts for the explicitly enabled split scan."""

    _require(
        type(request) is ActiveReconstructionScanRequest
        and request.use_fixed_scan_subchannels,
        "subchannel receipts require an opted-in exact scan request",
    )
    return _fixed_subchannel_selection(
        request, _scan_plan(request)
    ).subchannel_receipts


def _candidate_id(
    request: ActiveReconstructionScanRequest, draft: _Draft
) -> str:
    return active_candidate_id_for_window(
        request.index, draft.window_index
    )


def _materialize(
    request: ActiveReconstructionScanRequest, selected: Sequence[_Draft]
) -> tuple[ActiveReconstructionCandidateMatch, ...]:
    selected_sources = tuple(
        sorted(
            {
                request.index.windows[draft.window_index].row.source_id
                for draft in selected
            }
        )
    )
    _require(len(selected_sources) <= 999_999, "too many selected source groups")
    groups = {
        source: f"G{index:04d}" for index, source in enumerate(selected_sources, 1)
    }
    matches: list[ActiveReconstructionCandidateMatch] = []
    for draft in selected:
        window = request.index.windows[draft.window_index]
        row = window.row
        quote = row.text[window.start_char : window.end_char]
        candidate_id = _candidate_id(request, draft)
        group = groups[row.source_id]
        span = EvidenceSpan(
            chunk_id=row.chunk_id,
            start_char=window.start_char,
            end_char=window.end_char,
            quote_sha256=quote_sha256(quote),
            ordinal=row.ordinal,
            source_id=row.source_id,
            turn_start_char=row.turn_start_char,
            turn_id=row.turn_id,
            role=row.role,
            created_at=row.created_at,
        )
        binding = LocalCitationBinding(
            candidate_id=candidate_id,
            source_group_handle=group,
            namespace_id=request.index.cache.namespace_id,
            cache_receipt_sha256=request.index.cache.cache_receipt_sha256,
            source_database_sha256=request.index.cache.source_database_sha256,
            source_store_receipt_sha256=(
                request.index.cache.source_store_receipt_sha256
            ),
            source_id=row.source_id,
            partition_id=row.partition_id,
            span=span,
            quote_sha256=quote_sha256(quote),
        )
        axes = [f"active_support:{draft.support_kind.value}"]
        if draft.supported_slot_ids:
            axes.append("original_operator_slot_support")
        if draft.temporal_support:
            axes.append("original_temporal_target_support")
        candidate = FullStoreSlotCandidate(
            candidate_id=candidate_id,
            source_group_handle=group,
            quote=quote,
            quote_sha256=quote_sha256(quote),
            token_count=count_tokens(quote),
            role=row.role,
            created_at=row.created_at,
            event_date=window.event_date,
            event_date_basis=window.event_date_basis,
            supported_slot_ids=draft.supported_slot_ids,
            matched_query_terms=draft.matched_cue_terms,
            contains_numeric_value=window.contains_numeric_value,
            temporal_distance_days=draft.temporal_distance_days,
            selection_axes=tuple(axes),
            citation_binding_receipt_sha256=binding.receipt_sha256,
        )
        matches.append(
            ActiveReconstructionCandidateMatch(
                candidate=candidate,
                local_binding=binding,
                support_kind=draft.support_kind,
                supporting_cue_receipt_sha256s=(
                    draft.supporting_cue_receipt_sha256s
                ),
                matched_cue_terms=draft.matched_cue_terms,
                matched_child_terms=draft.matched_child_terms,
                action_concept=draft.action_concept,
            )
        )
    return tuple(matches)


def scan_typed_active_full_store(
    request: ActiveReconstructionScanRequest, /
) -> ActiveReconstructionScanBatch:
    """Satisfy ``ActiveReconstructionCandidateScanner`` without model calls."""

    _require(
        type(request) is ActiveReconstructionScanRequest,
        "active full-store scanner requires an exact request",
    )
    plan = _scan_plan(request)
    if request.use_fixed_scan_subchannels:
        split = _fixed_subchannel_selection(request, plan)
        selected = split.selected
        population_count = split.candidate_population_count
    else:
        drafts = tuple(
            draft
            for index in plan.candidate_indices
            if (draft := _draft(request, index, plan)) is not None
        )
        selected = _select(request, drafts)
        population_count = len(drafts)
    matches = _materialize(request, selected)
    return ActiveReconstructionScanBatch(
        request_receipt_sha256=request.receipt_sha256,
        matches=matches,
        candidate_population_count=population_count,
        selection_truncated=len(matches) < population_count,
    )


def active_full_store_scan_audit_projection(
    request: ActiveReconstructionScanRequest,
    batch: ActiveReconstructionScanBatch,
) -> dict[str, Any]:
    """Return the prompt-external no-completeness scanner audit surface."""

    _require(
        type(request) is ActiveReconstructionScanRequest
        and type(batch) is ActiveReconstructionScanBatch
        and batch.request_receipt_sha256 == request.receipt_sha256,
        "scanner audit request/batch lineage changed",
    )
    plan = _scan_plan(request)
    drafts = tuple(
        draft
        for index in plan.candidate_indices
        if (draft := _draft(request, index, plan)) is not None
    )
    support_kind_counts = {
        kind.value: sum(draft.support_kind is kind for draft in drafts)
        for kind in ActiveReconstructionSupportKind
    }
    split: _FixedSubchannelSelection | None = None
    if request.use_fixed_scan_subchannels:
        split = _fixed_subchannel_selection(request, plan)
        expected_matches = _materialize(request, split.selected)
        _require(
            {row.receipt_sha256 for row in expected_matches}
            == {row.receipt_sha256 for row in batch.matches}
            and len(expected_matches) == len(batch.matches),
            "scanner audit batch differs from its subchannel selection",
        )
    value = {
        "active_index_lookup_receipt_sha256": plan.lookup.receipt_sha256,
        "affinity_only_callback_status": (
            "exact_source_only_or_obligation_supported_history"
        ),
        "batch_receipt_sha256": batch.receipt_sha256,
        "candidate_scope": (
            "cue_action_exact_source_postings_plus_"
            "obligation_supported_selected_history_affinity"
        ),
        "cue_posting_fanout_cap": active_cue_posting_fanout_cap(request.index),
        "deterministic_priority_order": (
            "source_affinity,component_affinity,slot_support,action_support,"
            "temporal_support,cue_support,source_diversity,stable_identity"
        ),
        "format": AUDIT_FORMAT,
        "history_affinity_requires_obligation_support": True,
        "new_provider_calls": 0,
        "operator_spec_receipt_sha256": request.operator_spec.receipt_sha256,
        "raw_partition_ids_exposed": False,
        "raw_source_ids_exposed": False,
        "request_receipt_sha256": request.receipt_sha256,
        "retained_transformer_token_state_bytes": 0,
        "semantic_completeness_status": "not_claimed",
        "selective_cue_term_count": len(plan.selective_cue_terms),
        "support_kind_counts": support_kind_counts,
        "temporal_target_receipt_sha256": request.temporal_target.receipt_sha256,
    }
    if split is not None:
        value.update(
            {
                "candidate_scope": (
                    "fixed_exact_source_enclosing_history_and_global_fact_cue"
                ),
                "deterministic_priority_order": (
                    "fixed_global_history_source_reservations_then_"
                    "obligation_specificity_inverse_fanout_spillover"
                ),
                "scan_selection_receipt_sha256": split.receipt_sha256,
                "scan_subchannel_receipts": [
                    row.projection() for row in split.subchannel_receipts
                ],
                "selection_policy": "fixed_subchannels_with_bounded_spillover",
            }
        )
    if request.use_coverage_aware_callback_selection:
        affinity_reinjected = any(
            cue.selected_evidence_affinity is not None for cue in request.cues
        )
        value.update(
            {
                "coverage_reservation_order": (
                    "exact_source_enclosing_history_global_fact"
                    if affinity_reinjected
                    else "legacy_global_fact_enclosing_history_exact_source"
                ),
                "deterministic_priority_order": (
                    "required_slots,direct_rare_lexical,cue_parents,cues,"
                    "temporal,personalization,user_role,source_history_turn,"
                    "distinct_action,density_per_token,stable_identity"
                ),
                "selection_policy": (
                    "coverage_aware_fixed_subchannels_with_bounded_spillover"
                    if split is not None
                    else "coverage_aware_bounded_callback"
                ),
                "use_coverage_aware_callback_selection": True,
            }
        )
    assert_gold_blind(value, path="typed_active_full_store_scanner_audit")
    return value


@dataclass(frozen=True, slots=True)
class CandidateCueSupportPriority:
    """Cue support across every callback match, including parent duplicates."""

    span_receipt_sha256: str
    callback_match_receipt_sha256s: tuple[str, ...]
    callback_candidate_receipt_sha256s: tuple[str, ...]
    decision_receipt_sha256s: tuple[str, ...]
    decision_statuses: tuple[str, ...]
    supporting_cue_receipt_sha256s: tuple[str, ...]
    matched_original_cue_terms: tuple[str, ...]
    support_kinds: tuple[str, ...]
    parent_candidate_receipt_sha256: str | None
    parent_handle_id: str | None
    already_parent_selected: bool
    newly_admitted: bool
    recommended_parent_promotion: bool
    source_affinity: bool
    component_affinity: bool
    slot_support_count: int
    action_support: bool
    temporal_support: bool
    first_hop: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.span_receipt_sha256, "priority span")
        for values, label in (
            (self.callback_match_receipt_sha256s, "priority matches"),
            (self.callback_candidate_receipt_sha256s, "priority candidates"),
            (self.decision_receipt_sha256s, "priority decisions"),
            (self.supporting_cue_receipt_sha256s, "priority cues"),
        ):
            _require(values and len(set(values)) == len(values), f"{label} changed")
            for value in values:
                require_sha256(value, label)
        _require(
            self.decision_statuses
            and len(self.decision_statuses) == len(self.decision_receipt_sha256s),
            "priority decision statuses changed",
        )
        _require(
            len(set(self.matched_original_cue_terms))
            == len(self.matched_original_cue_terms),
            "priority matched terms changed",
        )
        _require(
            self.support_kinds
            and len(set(self.support_kinds)) == len(self.support_kinds)
            and set(self.support_kinds)
            <= {kind.value for kind in ActiveReconstructionSupportKind},
            "priority support kinds changed",
        )
        if self.parent_candidate_receipt_sha256 is not None:
            require_sha256(
                self.parent_candidate_receipt_sha256, "priority parent candidate"
            )
        if self.parent_handle_id is not None:
            _require(
                re.fullmatch(r"H[0-9]{3,6}", self.parent_handle_id) is not None,
                "priority parent handle must be opaque",
            )
        for value, label in (
            (self.already_parent_selected, "parent-selected flag"),
            (self.newly_admitted, "new-admission flag"),
            (self.recommended_parent_promotion, "promotion flag"),
            (self.source_affinity, "source-affinity flag"),
            (self.component_affinity, "component-affinity flag"),
            (self.action_support, "action-support flag"),
            (self.temporal_support, "temporal-support flag"),
        ):
            _require(type(value) is bool, f"{label} changed")
        _require(
            self.recommended_parent_promotion
            == bool(self.already_parent_selected and self.support_kinds)
            and not (self.already_parent_selected and self.newly_admitted),
            "parent promotion was confused with new admission",
        )
        _require(
            type(self.slot_support_count) is int and self.slot_support_count >= 0,
            "priority slot support changed",
        )
        _require(type(self.first_hop) is int and 1 <= self.first_hop <= 2, "priority hop changed")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "priority receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def cue_support_count(self) -> int:
        return len(self.supporting_cue_receipt_sha256s)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "action_support": self.action_support,
            "already_parent_selected": self.already_parent_selected,
            "callback_candidate_receipt_sha256s": list(
                self.callback_candidate_receipt_sha256s
            ),
            "callback_match_receipt_sha256s": list(
                self.callback_match_receipt_sha256s
            ),
            "component_affinity": self.component_affinity,
            "cue_support_count": self.cue_support_count,
            "decision_receipt_sha256s": list(self.decision_receipt_sha256s),
            "decision_statuses": list(self.decision_statuses),
            "first_hop": self.first_hop,
            "format": PRIORITY_FORMAT,
            "matched_original_cue_terms": list(self.matched_original_cue_terms),
            "newly_admitted": self.newly_admitted,
            "parent_candidate_receipt_sha256": (
                self.parent_candidate_receipt_sha256
            ),
            "parent_handle_id": self.parent_handle_id,
            "recommended_parent_promotion": self.recommended_parent_promotion,
            "slot_support_count": self.slot_support_count,
            "source_affinity": self.source_affinity,
            "span_receipt_sha256": self.span_receipt_sha256,
            "supporting_cue_receipt_sha256s": list(
                self.supporting_cue_receipt_sha256s
            ),
            "support_kinds": list(self.support_kinds),
            "temporal_support": self.temporal_support,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def derive_candidate_cue_support_priorities(
    result: TypedActiveReconstructionResult,
    /,
    *,
    parent_handle_ids: tuple[str, ...] | None = None,
) -> tuple[CandidateCueSupportPriority, ...]:
    """Rank all callback-selected spans, including deduplicated parent rows.

    ``parent_handle_ids`` is an optional prompt-external alignment with the
    first-pass candidate order.  It lets a runner promote an existing handle
    (for example ``H500017``) without representing that duplicate span as a new
    active-reconstruction admission.
    """

    _require(
        type(result) is TypedActiveReconstructionResult,
        "cue priority requires an exact reconstruction result",
    )
    if parent_handle_ids is not None:
        _require(
            type(parent_handle_ids) is tuple
            and len(parent_handle_ids) == len(result.parent_result.candidates)
            and len(set(parent_handle_ids)) == len(parent_handle_ids)
            and all(
                re.fullmatch(r"H[0-9]{3,6}", value) is not None
                for value in parent_handle_ids
            ),
            "parent handles must align exactly with first-pass candidates",
        )
    parent_by_span: dict[str, tuple[str, str | None]] = {}
    for index, (candidate, binding) in enumerate(
        zip(
            result.parent_result.candidates,
            result.parent_result.local_bindings,
            strict=True,
        )
    ):
        parent_by_span[citation_span_receipt_sha256(binding)] = (
            candidate_projection_receipt_sha256(candidate),
            None if parent_handle_ids is None else parent_handle_ids[index],
        )

    grouped: dict[str, list[tuple[int, ActiveReconstructionCandidateMatch, ActiveReconstructionDecision]]] = {}
    for hop in result.hops:
        for match, decision in zip(hop.batch.matches, hop.decisions, strict=True):
            span = citation_span_receipt_sha256(match.local_binding)
            grouped.setdefault(span, []).append((hop.hop, match, decision))

    rows: list[CandidateCueSupportPriority] = []
    for span, occurrences in grouped.items():
        parent = parent_by_span.get(span)
        candidate_receipts = _ordered_unique(
            candidate_projection_receipt_sha256(match.candidate)
            for _hop, match, _decision in occurrences
        )
        match_receipts = _ordered_unique(
            match.receipt_sha256 for _hop, match, _decision in occurrences
        )
        decision_status_by_receipt = dict.fromkeys(
            decision.receipt_sha256
            for _hop, _match, decision in occurrences
        )
        for _hop, _match, decision in occurrences:
            decision_status_by_receipt[decision.receipt_sha256] = decision.status
        decisions = tuple(decision_status_by_receipt)
        statuses = tuple(decision_status_by_receipt.values())
        cues = _ordered_unique(
            cue
            for _hop, match, _decision in occurrences
            for cue in match.supporting_cue_receipt_sha256s
        )
        terms = _ordered_unique(
            term
            for _hop, match, _decision in occurrences
            for term in (
                *match.matched_cue_terms,
                *((match.action_concept,) if match.action_concept else ()),
            )
        )
        support_kinds = _ordered_unique(
            match.support_kind.value
            for _hop, match, _decision in occurrences
        )
        slot_count = max(
            len(match.candidate.supported_slot_ids)
            for _hop, match, _decision in occurrences
        )
        parent_selected = parent is not None
        newly_admitted = bool(
            not parent_selected
            and any(
                decision.status == "admitted"
                for _hop, _match, decision in occurrences
            )
        )
        rows.append(
            CandidateCueSupportPriority(
                span_receipt_sha256=span,
                callback_match_receipt_sha256s=match_receipts,
                callback_candidate_receipt_sha256s=candidate_receipts,
                decision_receipt_sha256s=decisions,
                decision_statuses=statuses,
                supporting_cue_receipt_sha256s=cues,
                matched_original_cue_terms=terms,
                support_kinds=support_kinds,
                parent_candidate_receipt_sha256=None if parent is None else parent[0],
                parent_handle_id=None if parent is None else parent[1],
                already_parent_selected=parent_selected,
                newly_admitted=newly_admitted,
                recommended_parent_promotion=parent_selected,
                source_affinity=(
                    ActiveReconstructionSupportKind.SELECTED_SOURCE_AFFINITY.value
                    in support_kinds
                ),
                component_affinity=(
                    ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY.value
                    in support_kinds
                ),
                slot_support_count=slot_count,
                action_support=(
                    ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE.value
                    in support_kinds
                ),
                temporal_support=any(
                    "original_temporal_target_support"
                    in match.candidate.selection_axes
                    for _hop, match, _decision in occurrences
                ),
                first_hop=min(hop for hop, _match, _decision in occurrences),
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -int(row.source_affinity),
                -int(row.component_affinity),
                -row.slot_support_count,
                -int(row.action_support),
                -int(row.temporal_support),
                -row.cue_support_count,
                -int(row.recommended_parent_promotion),
                row.first_hop,
                row.span_receipt_sha256,
            ),
        )
    )


__all__ = [
    "ActiveFullStoreScanSubchannel",
    "ActiveFullStoreScanSubchannelReceipt",
    "CandidateCueSupportPriority",
    "TypedActiveFullStoreScannerError",
    "active_full_store_scan_audit_projection",
    "derive_candidate_cue_support_priorities",
    "derive_active_full_store_scan_subchannel_receipts",
    "scan_typed_active_full_store",
]
