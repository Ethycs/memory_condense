"""Provider-free global-to-local completion for unresolved memory obligations.

This is the bounded G plane that follows protected retrieval and source-local
reinjection.  It does not rerank the rows already present in those planes.
Instead it searches the complete immutable :class:`SemanticResidualIndex`,
keeps every unexpanded branch explicitly unresolved, hydrates exact segments
from selected cells and their source-local neighbours, and selects candidates
through four independent lanes before any protected-evidence deduplication.

The implementation deliberately separates *priority* from *authority*.
Dense, sparse, personal/temporal, and diversity signals decide which branch is
expanded first, but a low score can never prove that a branch is irrelevant.
Only a complete structural contradiction (an unavailable required author role
or a missing explicitly requested literal) may produce ``definitely_no``.
All other budget-limited branches remain visible in the open frontier.

No model client is owned or called here.  Query vectors, when present, are
sealed inputs produced outside this module.  The result persists receipts and
exact citations, never transformer tokens, activations, K/V cache, or hidden
state.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Any, Literal

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import LocalCitationBinding, indexed_surface_terms
from .semantic_binary_search import SemanticSearchNode
from .semantic_residual_search import (
    ExactCellSegment,
    SemanticResidualCell,
    SemanticResidualIndex,
    SemanticResidualQuery,
    semantic_residual_protected_evidence_population_receipt,
    semantic_residual_source_group_map,
)
from .typed_action_semantics import (
    canonical_action_concepts,
    completed_action_concepts,
    linked_action_concepts,
    planned_action_concepts,
)
from .typed_operator_spec import AnswerShape, RequiredSlot, TemporalMode


MECHANISM_ID = "semantic_global_to_local_completion_v1_1"
POLICY_FORMAT = "memory-condense-semantic-global-completion-policy-v1_1"
REQUEST_FORMAT = "memory-condense-semantic-global-completion-request-v1"
OBLIGATION_FORMAT = "memory-condense-semantic-global-obligation-v1"
TREE_VISIT_FORMAT = "memory-condense-semantic-global-tree-visit-v1"
TREE_FRONTIER_FORMAT = "memory-condense-semantic-global-tree-frontier-v1"
CANDIDATE_FORMAT = "memory-condense-semantic-global-candidate-v1"
LANE_FORMAT = "memory-condense-semantic-global-lane-receipt-v1"
ATTEMPT_FORMAT = "memory-condense-semantic-global-selection-attempt-v1"
EVIDENCE_FORMAT = "memory-condense-semantic-global-evidence-v1"
DUPLICATE_FORMAT = "memory-condense-semantic-global-protected-duplicate-v1"
CLOSURE_FORMAT = "memory-condense-semantic-global-closure-v1"
RESULT_FORMAT = "memory-condense-semantic-global-completion-result-v1"

DEFAULT_GLOBAL_PAYLOAD_TOKEN_CAP = 4_200
TARGET_GLOBAL_PAYLOAD_TOKEN_MIN = 3_600
TARGET_GLOBAL_PAYLOAD_TOKEN_MAX = 4_800
LANE_IDS = ("dense", "sparse", "personal_temporal", "source_date_diversity")
DATED_ACTION_WITNESS_BONUS = 48.0
_DATED_RE = re.compile(r"^\[Question asked at (?P<asked_at>.+?)\]\s*", re.I | re.S)
_EXACT_LITERAL_RE = re.compile(
    r"\bexact\s+(?:term|phrase)\s+[\"“](?P<literal>[^\"”]{1,160})[\"”]",
    re.I,
)
_INLINE_MONTH_DAY_RE = re.compile(
    r"(?<![\d/])(?P<month>0?[1-9]|1[0-2])/"
    r"(?P<day>0?[1-9]|[12]\d|3[01])(?![\d/])"
)
_FIRST_PERSON_RE = re.compile(r"\b(?:I|I'm|I've|I'd|my|mine|we|our)\b", re.I)
_GENERIC_ADVICE_RE = re.compile(
    r"\b(?:you (?:can|could|might|may|should)|consider|some options|"
    r"it depends|generally|typically|usually)\b",
    re.I,
)
_CAPITALIZED_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){0,3}\b"
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_ENTITY_STOP = frozenset(
    {
        "ask",
        "ago",
        "day",
        "different",
        "first",
        "last",
        "many",
        "month",
        "recent",
        "recently",
        "time",
        "today",
        "week",
        "year",
        *_MONTHS,
    }
)


class SemanticGlobalCompletionError(MatchedEvalContractError):
    """A global request, search frontier, lane, budget, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SemanticGlobalCompletionError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _receipt(value: Mapping[str, object], declared: str, label: str) -> str:
    expected = identity_sha256(value)
    if declared:
        _require(require_sha256(declared, label) == expected, f"{label} changed")
    return expected


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(value) is str and value for value in values)
        and len(set(values)) == len(values),
        f"{label} must be ordered unique exact text",
    )
    return values


def _finite(value: object, label: str) -> float:
    _require(type(value) in {int, float}, f"{label} must be exact numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _span_identity(span: EvidenceSpan) -> str:
    return identity_sha256(span.identity_payload())


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _parse_asked_at(question: str) -> datetime | None:
    match = _DATED_RE.match(question)
    if match is None:
        return None
    value = match.group("asked_at").strip()
    # Production LongMemEval dates include an informational weekday between
    # the calendar date and clock (for example, ``2023/03/25 (Sat) 18:04``).
    # Strip only that parenthesized token before parsing; the exact dated
    # question remains unchanged and receipt-bound everywhere else.
    without_weekday = re.sub(r"\s+\([^()\r\n]+\)\s+", " ", value).strip()
    candidates = tuple(
        dict.fromkeys(
            (
                value,
                value.replace("/", "-"),
                without_weekday,
                without_weekday.replace("/", "-"),
            )
        )
    )
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            pass
    for candidate in candidates:
        for pattern in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(candidate, pattern)
            except ValueError:
                pass
    return None


@dataclass(frozen=True, slots=True)
class GlobalLaneBudget:
    lane_id: str
    max_selected_segments: int
    pre_dedup_token_cap: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(self.lane_id in LANE_IDS, "global lane ID changed")
        _require(
            type(self.max_selected_segments) is int
            and self.max_selected_segments > 0
            and type(self.pre_dedup_token_cap) is int
            and self.pre_dedup_token_cap > 0,
            "global lane budget must be positive",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global lane budget receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": f"{POLICY_FORMAT}-lane-budget",
            "lane_id": self.lane_id,
            "max_selected_segments": self.max_selected_segments,
            "pre_dedup_token_cap": self.pre_dedup_token_cap,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _default_lane_budgets() -> tuple[GlobalLaneBudget, ...]:
    return (
        GlobalLaneBudget("dense", 28, 2_400),
        GlobalLaneBudget("sparse", 40, 2_800),
        GlobalLaneBudget("personal_temporal", 36, 2_400),
        GlobalLaneBudget("source_date_diversity", 28, 1_800),
    )


@dataclass(frozen=True, slots=True)
class SemanticGlobalCompletionPolicy:
    """Fixed, independent search/selection/packing limits for the G plane."""

    global_payload_token_cap: int = DEFAULT_GLOBAL_PAYLOAD_TOKEN_CAP
    max_node_visits: int = 768
    max_retained_leaf_cells: int = 192
    source_neighbor_radius: int = 1
    max_hydrated_segments: int = 768
    # Bound the primary entity-priority prefix only.  Additional entities are
    # conserved after structural obligations so closure can never silently
    # forget a query target merely because the question named more than four.
    max_entity_obligations: int = 4
    lane_budgets: tuple[GlobalLaneBudget, ...] = _default_lane_budgets()
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "global_payload_token_cap",
            "max_node_visits",
            "max_retained_leaf_cells",
            "max_hydrated_segments",
            "max_entity_obligations",
        ):
            value = getattr(self, name)
            _require(type(value) is int and value > 0, f"{name} must be positive")
        _require(
            self.global_payload_token_cap <= TARGET_GLOBAL_PAYLOAD_TOKEN_MAX,
            "G-plane payload exceeds the hard 4.8k ceiling",
        )
        _require(
            type(self.source_neighbor_radius) is int
            and self.source_neighbor_radius >= 0,
            "source neighbour radius must be nonnegative",
        )
        _require(
            type(self.lane_budgets) is tuple
            and tuple(row.lane_id for row in self.lane_budgets) == LANE_IDS
            and all(type(row) is GlobalLaneBudget for row in self.lane_budgets),
            "global lane budgets changed exact independent order",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global completion policy receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_global_policy")

    @property
    def target_payload_range_satisfied(self) -> bool:
        return (
            TARGET_GLOBAL_PAYLOAD_TOKEN_MIN
            <= self.global_payload_token_cap
            <= TARGET_GLOBAL_PAYLOAD_TOKEN_MAX
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": POLICY_FORMAT,
            "global_payload_budget_non_borrowable": True,
            "global_payload_token_cap": self.global_payload_token_cap,
            "lane_budgets": [row.projection() for row in self.lane_budgets],
            "max_entity_obligations": self.max_entity_obligations,
            "max_hydrated_segments": self.max_hydrated_segments,
            "max_node_visits": self.max_node_visits,
            "max_retained_leaf_cells": self.max_retained_leaf_cells,
            "new_provider_calls": 0,
            "priority_algorithm": (
                "best-first-four-lane+joint-date-role-completed-or-proposed-"
                "action-witness-v3"
            ),
            "retained_transformer_token_state_bytes": 0,
            "source_neighbor_radius": self.source_neighbor_radius,
            "target_payload_range_satisfied": self.target_payload_range_satisfied,
            "target_payload_token_max": TARGET_GLOBAL_PAYLOAD_TOKEN_MAX,
            "target_payload_token_min": TARGET_GLOBAL_PAYLOAD_TOKEN_MIN,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SemanticGlobalCompletionRequest:
    """Generic receipt-bound routing state; it has no benchmark identifiers."""

    query_receipt_sha256: str
    unresolved_slot_ids: tuple[str, ...]
    route_reasons: tuple[str, ...]
    routed: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.query_receipt_sha256, "global request query")
        _ordered_unique(self.unresolved_slot_ids, "global unresolved slots")
        _ordered_unique(self.route_reasons, "global route reasons")
        _require(type(self.routed) is bool, "global routed flag changed")
        _require(
            self.routed == bool(self.route_reasons),
            "global route reasons and decision diverged",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global request receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_global_request")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": REQUEST_FORMAT,
            "generic_question_only_policy": True,
            "known_source_rule_used": False,
            "new_provider_calls": 0,
            "query_receipt_sha256": self.query_receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "route_reasons": list(self.route_reasons),
            "routed": self.routed,
            "unresolved_slot_ids": list(self.unresolved_slot_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def compile_semantic_global_completion_request(
    query: SemanticResidualQuery,
    /,
    *,
    prior_needs_global_search: bool = False,
    unresolved_slot_ids: Sequence[str] = (),
    operand_closure_missing: bool = False,
    required_user_witness_missing: bool = False,
    required_temporal_witness_missing: bool = False,
    local_frontier_unresolved: bool = False,
) -> SemanticGlobalCompletionRequest:
    """Compile the generic V7 trigger from question/receipt-bound conditions."""

    _require(type(query) is SemanticResidualQuery, "global request query changed")
    supplied = tuple(unresolved_slot_ids)
    known = {row.slot_id for row in query.operator_spec.required_slots}
    _require(
        len(set(supplied)) == len(supplied) and set(supplied) <= known,
        "global request contains an unknown or repeated slot",
    )
    conditions = (
        (prior_needs_global_search, "prior_needs_global_search"),
        (bool(supplied), "required_typed_slot_unresolved"),
        (operand_closure_missing, "global_operand_closure_missing"),
        (required_user_witness_missing, "required_user_witness_missing"),
        (required_temporal_witness_missing, "required_temporal_witness_missing"),
        (local_frontier_unresolved, "source_local_frontier_unresolved"),
    )
    reasons = tuple(label for enabled, label in conditions if enabled)
    return SemanticGlobalCompletionRequest(
        query_receipt_sha256=query.receipt_sha256,
        unresolved_slot_ids=supplied,
        route_reasons=reasons,
        routed=bool(reasons),
    )


@dataclass(frozen=True, slots=True)
class GlobalEvidenceObligation:
    obligation_id: str
    kind: Literal["typed_slot", "entity", "action", "date", "role", "numeric"]
    typed_slot_id: str | None
    label: str
    match_terms: tuple[str, ...]
    minimum_match_term_count: int
    requires_numeric: bool
    required_role: str | None
    target_date_start: str | None
    target_date_end: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.obligation_id, "global obligation ID")
        _require(
            self.kind in {"typed_slot", "entity", "action", "date", "role", "numeric"},
            "global obligation kind changed",
        )
        _require(
            (self.kind == "typed_slot") == (self.typed_slot_id is not None),
            "global typed obligation lost its source slot",
        )
        if self.typed_slot_id is not None:
            require_sha256(self.typed_slot_id, "global obligation typed slot")
        require_text(self.label, "global obligation label")
        _require(type(self.match_terms) is tuple, "obligation terms changed type")
        if self.match_terms:
            _ordered_unique(self.match_terms, "global obligation terms")
        _require(
            type(self.minimum_match_term_count) is int
            and (
                0 <= self.minimum_match_term_count <= len(self.match_terms)
                if self.match_terms
                else self.minimum_match_term_count == 0
            ),
            "global obligation match threshold changed",
        )
        _require(type(self.requires_numeric) is bool, "obligation numeric flag changed")
        _require(
            self.required_role in {None, "user", "assistant"},
            "obligation role changed",
        )
        if self.target_date_start is None or self.target_date_end is None:
            _require(
                self.target_date_start is None and self.target_date_end is None,
                "obligation date interval lost one boundary",
            )
        else:
            require_text(self.target_date_start, "obligation date start")
            require_text(self.target_date_end, "obligation date end")
            _require(
                self.target_date_start <= self.target_date_end,
                "obligation date interval reversed",
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global obligation receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": OBLIGATION_FORMAT,
            "kind": self.kind,
            "label": self.label,
            "match_terms": list(self.match_terms),
            "minimum_match_term_count": self.minimum_match_term_count,
            "obligation_id": self.obligation_id,
            "required_role": self.required_role,
            "requires_numeric": self.requires_numeric,
            "target_date_end": self.target_date_end,
            "target_date_start": self.target_date_start,
            "typed_slot_id": self.typed_slot_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _obligation(
    kind: Literal["typed_slot", "entity", "action", "date", "role", "numeric"],
    label: str,
    *,
    typed_slot_id: str | None = None,
    match_terms: Sequence[str] = (),
    minimum_match_term_count: int | None = None,
    requires_numeric: bool = False,
    required_role: str | None = None,
    target_date_start: str | None = None,
    target_date_end: str | None = None,
) -> GlobalEvidenceObligation:
    terms = tuple(dict.fromkeys(value for value in match_terms if value))
    threshold = (
        (1 if terms else 0)
        if minimum_match_term_count is None
        else minimum_match_term_count
    )
    body = {
        "format": OBLIGATION_FORMAT,
        "kind": kind,
        "label": label,
        "match_terms": list(terms),
        "minimum_match_term_count": threshold,
        "required_role": required_role,
        "requires_numeric": requires_numeric,
        "target_date_end": target_date_end,
        "target_date_start": target_date_start,
        "typed_slot_id": typed_slot_id,
    }
    return GlobalEvidenceObligation(
        identity_sha256(body),
        kind,
        typed_slot_id,
        label,
        terms,
        threshold,
        requires_numeric,
        required_role,
        target_date_start,
        target_date_end,
    )


def _temporal_interval(question: str) -> tuple[str, str, str] | None:
    asked_at = _parse_asked_at(question)
    body = _DATED_RE.sub("", question).strip()
    if asked_at is not None:
        ago = re.search(r"\b(?P<days>[0-9]{1,4})\s+days?\s+ago\b", body, re.I)
        if ago is not None:
            target = (asked_at.date() - timedelta(days=int(ago.group("days")))).isoformat()
            return (target, target, f"relative_day:{ago.group('days')}")
        if re.search(r"\blast month\b", body, re.I):
            first_this = asked_at.date().replace(day=1)
            end = first_this - timedelta(days=1)
            start = end.replace(day=1)
            return (start.isoformat(), end.isoformat(), "previous_calendar_month")
    for name, month in _MONTHS.items():
        if re.search(rf"\b{re.escape(name)}\b", body, re.I):
            year = asked_at.year if asked_at is not None else datetime.now().year
            if asked_at is not None and month > asked_at.month:
                year -= 1
            start = date(year, month, 1)
            if month == 12:
                next_month = date(year + 1, 1, 1)
            else:
                next_month = date(year, month + 1, 1)
            return (start.isoformat(), (next_month - timedelta(days=1)).isoformat(), f"named_month:{name}")
    return None


def _slot_obligation(slot: RequiredSlot) -> GlobalEvidenceObligation:
    return _obligation(
        "typed_slot",
        slot.label,
        typed_slot_id=slot.slot_id,
        match_terms=slot.match_terms,
        minimum_match_term_count=slot.minimum_match_term_count,
        requires_numeric=slot.requires_numeric,
    )


def _compile_obligations(
    query: SemanticResidualQuery,
    request: SemanticGlobalCompletionRequest,
    policy: SemanticGlobalCompletionPolicy,
) -> tuple[GlobalEvidenceObligation, ...]:
    spec = query.operator_spec
    selected_slot_ids = set(request.unresolved_slot_ids)
    slots = tuple(
        slot
        for slot in spec.required_slots
        if not selected_slot_ids or slot.slot_id in selected_slot_ids
    )
    rows: list[GlobalEvidenceObligation] = [_slot_obligation(slot) for slot in slots]
    body = _DATED_RE.sub("", query.dated_question).strip()
    action_terms = {
        term
        for action in query.action_concepts
        for term in indexed_surface_terms(action)
    }
    for action in query.action_concepts:
        rows.append(_obligation("action", action, match_terms=(action,)))
    entity_candidates: list[tuple[str, ...]] = []
    for match in _CAPITALIZED_RE.finditer(body):
        terms = tuple(indexed_surface_terms(match.group(0)))
        if terms and not set(terms) <= {"what", "which", "how", "did", "do", "does"}:
            entity_candidates.append(terms)
    entity_candidates.extend(
        (term,)
        for term in query.query_terms
        if term not in _ENTITY_STOP
        and term not in action_terms
        and not canonical_action_concepts(term)
        and not term.isdigit()
    )
    seen_entities: set[tuple[str, ...]] = set()
    entity_rows: list[GlobalEvidenceObligation] = []
    for terms in entity_candidates:
        terms = tuple(value for value in terms if value not in _ENTITY_STOP)
        if not terms or terms in seen_entities:
            continue
        seen_entities.add(terms)
        entity_rows.append(
            _obligation("entity", " ".join(terms), match_terms=terms)
        )
    # Preserve the historical high-priority entity prefix, but never discard
    # overflow.  Appending overflow after date/role/numeric obligations keeps
    # those structural witnesses ahead of combinatorial query-term expansion
    # while retaining a receipt-bound unresolved obligation for every entity.
    rows.extend(entity_rows[: policy.max_entity_obligations])
    overflow_entity_rows = entity_rows[policy.max_entity_obligations :]
    interval = _temporal_interval(query.dated_question)
    if interval is not None:
        start, end, derivation = interval
        rows.append(
            _obligation(
                "date",
                derivation,
                target_date_start=start,
                target_date_end=end,
            )
        )
    personal_question = bool(_FIRST_PERSON_RE.search(body))
    required_role = spec.required_evidence_role or ("user" if personal_question else None)
    if required_role is not None:
        rows.append(
            _obligation(
                "role",
                f"authored_by_{required_role}",
                required_role=required_role,
            )
        )
    if any(slot.requires_numeric for slot in slots) or spec.answer_shape in {
        AnswerShape.BOOLEAN,
        AnswerShape.DURATION,
    }:
        rows.append(_obligation("numeric", "numeric operand", requires_numeric=True))
    rows.extend(overflow_entity_rows)
    unique: dict[str, GlobalEvidenceObligation] = {}
    for row in rows:
        unique.setdefault(row.obligation_id, row)
    return tuple(unique.values())


@dataclass(frozen=True, slots=True)
class GlobalTreeVisit:
    visit_ordinal: int
    node_receipt_sha256: str
    node_id: str
    covered_leaf_cell_ids: tuple[str, ...]
    action: Literal["expanded", "retained_leaf", "definitely_no"]
    definite_no_reason: str | None
    lane_upper_bounds: tuple[tuple[str, float], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.visit_ordinal) is int and self.visit_ordinal >= 0,
            "global visit ordinal changed",
        )
        require_sha256(self.node_receipt_sha256, "global visit node")
        require_text(self.node_id, "global visit node ID")
        _ordered_unique(self.covered_leaf_cell_ids, "global visit covered leaves")
        _require(bool(self.covered_leaf_cell_ids), "global visit lost leaf coverage")
        _require(
            self.action in {"expanded", "retained_leaf", "definitely_no"},
            "global visit action changed",
        )
        _require(
            (self.action == "definitely_no")
            == (self.definite_no_reason is not None),
            "global definite-no visit lost its proof reason",
        )
        if self.definite_no_reason is not None:
            _require(
                self.definite_no_reason
                in {"required_role_absent", "explicit_exact_literal_absent"},
                "global tree pruned without a proof-definite reason",
            )
        _require(
            type(self.lane_upper_bounds) is tuple
            and tuple(name for name, _value in self.lane_upper_bounds) == LANE_IDS,
            "global visit lane bounds changed order",
        )
        for _name, value in self.lane_upper_bounds:
            _finite(value, "global lane upper bound")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global tree visit receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "action": self.action,
            "covered_leaf_cell_ids": list(self.covered_leaf_cell_ids),
            "definite_no_reason": self.definite_no_reason,
            "format": TREE_VISIT_FORMAT,
            "lane_upper_bounds": [
                {"lane_id": lane_id, "upper_bound": upper}
                for lane_id, upper in self.lane_upper_bounds
            ],
            "node_id": self.node_id,
            "node_receipt_sha256": self.node_receipt_sha256,
            "visit_ordinal": self.visit_ordinal,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class GlobalTreeFrontier:
    tree_receipt_sha256: str
    policy_receipt_sha256: str
    query_receipt_sha256: str
    visits: tuple[GlobalTreeVisit, ...]
    retained_leaf_cell_ids: tuple[str, ...]
    definitely_no_leaf_cell_ids: tuple[str, ...]
    unresolved_leaf_cell_ids: tuple[str, ...]
    unexpanded_node_receipt_sha256s: tuple[str, ...]
    all_leaf_cell_ids: tuple[str, ...]
    search_closed: bool
    low_score_pruning_used: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.tree_receipt_sha256, "global frontier tree"),
            (self.policy_receipt_sha256, "global frontier policy"),
            (self.query_receipt_sha256, "global frontier query"),
        ):
            require_sha256(value, label)
        _require(
            type(self.visits) is tuple
            and all(type(row) is GlobalTreeVisit for row in self.visits)
            and tuple(row.visit_ordinal for row in self.visits)
            == tuple(range(len(self.visits))),
            "global frontier visits changed exact order",
        )
        for values, label in (
            (self.retained_leaf_cell_ids, "global retained leaves"),
            (self.definitely_no_leaf_cell_ids, "global definitely-no leaves"),
            (self.unresolved_leaf_cell_ids, "global unresolved leaves"),
            (self.unexpanded_node_receipt_sha256s, "global unexpanded nodes"),
            (self.all_leaf_cell_ids, "global all leaves"),
        ):
            _ordered_unique(values, label)
        retained = set(self.retained_leaf_cell_ids)
        pruned = set(self.definitely_no_leaf_cell_ids)
        unresolved = set(self.unresolved_leaf_cell_ids)
        _require(
            retained.isdisjoint(pruned)
            and retained.isdisjoint(unresolved)
            and pruned.isdisjoint(unresolved)
            and retained | pruned | unresolved == set(self.all_leaf_cell_ids),
            "global tree frontier lost its exact leaf partition",
        )
        _require(
            type(self.search_closed) is bool
            and self.search_closed == (not unresolved)
            and self.low_score_pruning_used is False,
            "global search closure flags changed",
        )
        _require(
            bool(self.unexpanded_node_receipt_sha256s) == bool(unresolved),
            "global unresolved leaves lost their unexpanded node frontier",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global tree frontier receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "all_leaf_cell_ids": list(self.all_leaf_cell_ids),
            "definitely_no_leaf_cell_ids": list(self.definitely_no_leaf_cell_ids),
            "format": TREE_FRONTIER_FORMAT,
            "low_score_pruning_used": False,
            "policy_receipt_sha256": self.policy_receipt_sha256,
            "query_receipt_sha256": self.query_receipt_sha256,
            "retained_leaf_cell_ids": list(self.retained_leaf_cell_ids),
            "search_closed": self.search_closed,
            "tree_receipt_sha256": self.tree_receipt_sha256,
            "unexpanded_node_receipt_sha256s": list(
                self.unexpanded_node_receipt_sha256s
            ),
            "unresolved_leaf_cell_ids": list(self.unresolved_leaf_cell_ids),
            "visits": [row.projection() for row in self.visits],
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class _CellSignals:
    dense: float
    sparse: float
    personal_temporal: float
    source_date_diversity: float

    def lane_value(self, lane_id: str) -> float:
        return float(getattr(self, lane_id))

    def ordered(self) -> tuple[tuple[str, float], ...]:
        return tuple((lane_id, self.lane_value(lane_id)) for lane_id in LANE_IDS)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right), "global semantic vector dimensions changed")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _date_in_obligation(segment: ExactCellSegment, obligation: GlobalEvidenceObligation) -> bool:
    start = obligation.target_date_start
    end = obligation.target_date_end
    if start is None or end is None:
        return False
    candidates = list(
        linked_event_dates(
            segment.quote,
            segment.created_at,
            segment.event_dates,
        )
    )
    try:
        candidates.append(datetime.fromisoformat(segment.created_at.replace("Z", "+00:00")).date().isoformat())
    except ValueError:
        pass
    return any(start <= value[:10] <= end for value in candidates)


def linked_event_dates(
    quote: str,
    created_at: str,
    existing_event_dates: Sequence[str] = (),
) -> tuple[str, ...]:
    """Overlay compact textual dates after exact segment hydration.

    A month/day surface such as ``2/8`` inherits only the immutable source-row
    year.  Relevance remains controlled by the compiled query interval.  This
    overlay deliberately leaves the authenticated R7 question-neutral index
    and its event-date inventory byte-identical.
    """

    _require(type(quote) is str, "linked event-date quote must be exact")
    _require(type(created_at) is str, "linked event-date timestamp must be exact")
    values = set(existing_event_dates)
    _require(
        all(type(value) is str and value for value in values),
        "linked event-date population changed",
    )
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        created = None
    if created is not None:
        for match in _INLINE_MONTH_DAY_RE.finditer(quote):
            try:
                values.add(
                    date(
                        created.year,
                        int(match.group("month")),
                        int(match.group("day")),
                    ).isoformat()
                )
            except ValueError:
                continue
    return tuple(sorted(values))


def _joint_completed_event_witness(
    query: SemanticResidualQuery,
    obligations: Sequence[GlobalEvidenceObligation],
    segment: ExactCellSegment,
) -> bool:
    """Return positive proof of one dated, user-owned completed query action.

    This is ranking/selection evidence only.  Its absence can never prune a
    branch or close an operand frontier.
    """

    date_obligations = tuple(row for row in obligations if row.kind == "date")
    required_roles = {
        row.required_role
        for row in obligations
        if row.kind == "role" and row.required_role is not None
    }
    query_actions = set(query.action_concepts)
    if not date_obligations or not required_roles or not query_actions:
        return False
    return (
        segment.role in required_roles
        and any(_date_in_obligation(segment, row) for row in date_obligations)
        and bool(query_actions & set(completed_action_concepts(segment.quote)))
    )


def _joint_planned_event_witness(
    query: SemanticResidualQuery,
    obligations: Sequence[GlobalEvidenceObligation],
    segment: ExactCellSegment,
) -> bool:
    """Return positive proof of one dated, user-owned proposed query action.

    A planned memory can enter this priority lane only when the question-only
    typed operator explicitly asks to include proposals.  As with completed
    events, failure to establish this witness has no pruning or closure power.
    """

    if not query.operator_spec.include_proposed:
        return False
    date_obligations = tuple(row for row in obligations if row.kind == "date")
    required_roles = {
        row.required_role
        for row in obligations
        if row.kind == "role" and row.required_role is not None
    }
    query_actions = set(query.action_concepts)
    if not date_obligations or not required_roles or not query_actions:
        return False
    return (
        segment.role in required_roles
        and any(_date_in_obligation(segment, row) for row in date_obligations)
        and bool(query_actions & set(planned_action_concepts(segment.quote)))
    )


def _joint_dated_action_witness(
    query: SemanticResidualQuery,
    obligations: Sequence[GlobalEvidenceObligation],
    segment: ExactCellSegment,
) -> bool:
    return _joint_completed_event_witness(
        query, obligations, segment
    ) or _joint_planned_event_witness(query, obligations, segment)


def _segment_supports(
    segment: ExactCellSegment,
    obligation: GlobalEvidenceObligation,
) -> bool:
    if obligation.required_role is not None and segment.role != obligation.required_role:
        return False
    if obligation.requires_numeric and not segment.contains_numeric_value:
        return False
    if obligation.kind == "role":
        return segment.role == obligation.required_role
    if obligation.kind == "numeric":
        return segment.contains_numeric_value
    if obligation.kind == "date":
        return _date_in_obligation(segment, obligation)
    if obligation.kind == "action":
        return bool(
            set(obligation.match_terms)
            & set(linked_action_concepts(segment.quote))
        )
    terms = set(segment.surface_terms)
    if not obligation.match_terms:
        return True
    return (
        len(set(obligation.match_terms) & terms)
        >= obligation.minimum_match_term_count
    )


def _cell_supports(
    cell: SemanticResidualCell,
    obligation: GlobalEvidenceObligation,
) -> bool:
    return any(_segment_supports(segment, obligation) for segment in cell.segments)


def _cell_signals(
    index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    obligations: Sequence[GlobalEvidenceObligation],
) -> Mapping[str, _CellSignals]:
    query_terms = set(query.query_terms) | set(query.slot_terms)
    query_actions = set(query.action_concepts)
    leaf_count = len(index.cells)
    idf = {
        term: math.log(
            (leaf_count + 1)
            / (sum(term in set(cell.surface_terms) for cell in index.cells) + 1)
        )
        for term in query_terms
    }
    needs_numeric = any(row.requires_numeric for row in obligations)
    needs_temporal = any(row.kind == "date" for row in obligations) or (
        query.operator_spec.temporal_mode is not TemporalMode.NONE
    )
    needs_user = any(row.required_role == "user" for row in obligations)
    result: dict[str, _CellSignals] = {}
    for cell in index.cells:
        if cell.normalized_source_centroid is None or not query.present_query_vectors:
            dense = -1.0
        else:
            dense = max(
                _dot(cell.normalized_source_centroid, vector)
                for vector in query.present_query_vectors
            )
        terms = set(cell.surface_terms)
        actions = {
            action
            for segment in cell.segments
            for action in linked_action_concepts(segment.quote)
        }
        term_score = math.fsum(idf[term] for term in query_terms & terms)
        action_score = 2.5 * len(query_actions & actions)
        supported = sum(_cell_supports(cell, row) for row in obligations)
        sparse = term_score + action_score + 3.0 * supported
        segments = cell.segments
        user = any(row.role == "user" for row in segments)
        numeric = any(row.contains_numeric_value for row in segments)
        temporal = any(
            row.event_dates or row.has_undated_window for row in segments
        )
        dated_action_witness = any(
            _joint_dated_action_witness(query, obligations, row)
            for row in segments
        )
        personal = (
            (3.0 if user else 0.0)
            + (2.0 if needs_user and user else 0.0)
            + (2.0 if needs_numeric and numeric else 0.0)
            + (2.0 if needs_temporal and temporal else 0.0)
            + action_score
            + min(term_score, 3.0)
            + (
                DATED_ACTION_WITNESS_BONUS
                if dated_action_witness
                else 0.0
            )
        )
        diversity = max(dense, 0.0) + min(sparse, 5.0) + (1.0 if user else 0.0)
        result[cell.cell_id] = _CellSignals(dense, sparse, personal, diversity)
    return MappingProxyType(result)


def _node_lane_bounds(
    node: SemanticSearchNode,
    signals: Mapping[str, _CellSignals],
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (
            lane_id,
            max(
                signals[cell.cell_id].lane_value(lane_id)
                for cell in node.iter_cells()
            ),
        )
        for lane_id in LANE_IDS
    )


def _priority_key(
    node: SemanticSearchNode,
    signals: Mapping[str, _CellSignals],
    *,
    lane_bounds: tuple[tuple[str, float], ...] | None = None,
) -> tuple[object, ...]:
    bounds = dict(
        _node_lane_bounds(node, signals) if lane_bounds is None else lane_bounds
    )
    # Normalize only for cross-lane queueing; exact lane scores remain sealed
    # separately.  Node size is deliberately not a pruning authority.
    combined = max(
        bounds["dense"],
        bounds["sparse"] / 12.0,
        bounds["personal_temporal"] / 12.0,
        bounds["source_date_diversity"] / 10.0,
    )
    return (
        -combined,
        -bounds["dense"],
        -bounds["sparse"],
        -bounds["personal_temporal"],
        -bounds["source_date_diversity"],
        node.span_start,
        node.span_end,
        node.receipt_sha256,
    )


def _definite_no_reason(
    query: SemanticResidualQuery,
    node: SemanticSearchNode,
) -> str | None:
    # Literal absence is not a no-support proof: another obligation's operand
    # may live in a branch that does not repeat the literal.
    return None


def _manifest_definite_no_reason(
    manifests: Mapping[str, SemanticNodeManifest],
    query: SemanticResidualQuery,
    node: SemanticSearchNode,
    obligations: Sequence[GlobalEvidenceObligation],
) -> str | None:
    # Role absence is likewise not compositional across obligations.  A user
    # witness and an assistant-authored factual expansion may be separate.
    return _definite_no_reason(query, node)


def _search_tree_best_first(
    index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    policy: SemanticGlobalCompletionPolicy,
    signals: Mapping[str, _CellSignals],
    obligations: Sequence[GlobalEvidenceObligation],
) -> GlobalTreeFrontier:
    root = index.core_tree.root
    manifests = index.manifest_by_node_receipt
    bounds_by_node_receipt: dict[str, tuple[tuple[str, float], ...]] = {}

    def bounds_for(node: SemanticSearchNode) -> tuple[tuple[str, float], ...]:
        existing = bounds_by_node_receipt.get(node.receipt_sha256)
        if existing is not None:
            return existing
        computed = _node_lane_bounds(node, signals)
        bounds_by_node_receipt[node.receipt_sha256] = computed
        return computed

    heap: list[tuple[tuple[object, ...], SemanticSearchNode]] = []
    root_bounds = bounds_for(root)
    heapq.heappush(
        heap,
        (_priority_key(root, signals, lane_bounds=root_bounds), root),
    )
    visits: list[GlobalTreeVisit] = []
    retained: list[str] = []
    pruned: list[str] = []
    while (
        heap
        and len(visits) < policy.max_node_visits
        and len(retained) < policy.max_retained_leaf_cells
    ):
        _key, node = heapq.heappop(heap)
        reason = _manifest_definite_no_reason(manifests, query, node, obligations)
        bounds = bounds_for(node)
        if reason is not None:
            action: Literal["expanded", "retained_leaf", "definitely_no"] = "definitely_no"
            pruned.extend(node.cell_ids)
        elif node.is_leaf:
            action = "retained_leaf"
            retained.extend(node.cell_ids)
        else:
            action = "expanded"
            for child in node.children:
                child_bounds = bounds_for(child)
                heapq.heappush(
                    heap,
                    (
                        _priority_key(
                            child,
                            signals,
                            lane_bounds=child_bounds,
                        ),
                        child,
                    ),
                )
        visits.append(
            GlobalTreeVisit(
                visit_ordinal=len(visits),
                node_receipt_sha256=node.receipt_sha256,
                node_id=node.node_id,
                covered_leaf_cell_ids=node.cell_ids,
                action=action,
                definite_no_reason=reason,
                lane_upper_bounds=bounds,
            )
        )
    unexpanded = tuple(node for _key, node in sorted(heap, key=lambda row: row[0]))
    unresolved_set = {
        cell_id for node in unexpanded for cell_id in node.cell_ids
    }
    all_ids = tuple(row.cell_id for row in index.cells)
    retained_set = set(retained)
    pruned_set = set(pruned)
    retained_ids = tuple(cell_id for cell_id in all_ids if cell_id in retained_set)
    pruned_ids = tuple(cell_id for cell_id in all_ids if cell_id in pruned_set)
    unresolved_ids = tuple(cell_id for cell_id in all_ids if cell_id in unresolved_set)
    _require(
        len(unresolved_set) == sum(len(node.cell_ids) for node in unexpanded),
        "global unexpanded frontier contains overlapping subtrees",
    )
    return GlobalTreeFrontier(
        tree_receipt_sha256=index.core_tree.receipt_sha256,
        policy_receipt_sha256=policy.receipt_sha256,
        query_receipt_sha256=query.receipt_sha256,
        visits=tuple(visits),
        retained_leaf_cell_ids=retained_ids,
        definitely_no_leaf_cell_ids=pruned_ids,
        unresolved_leaf_cell_ids=unresolved_ids,
        unexpanded_node_receipt_sha256s=tuple(
            node.receipt_sha256 for node in unexpanded
        ),
        all_leaf_cell_ids=all_ids,
        search_closed=not unresolved_ids,
    )


@dataclass(frozen=True, slots=True)
class GlobalCompletionCandidate:
    """One exact segment reached through G -> cell -> local source history."""

    candidate_id: str
    source_group_handle: str
    source_id: str
    source_history_receipt_sha256: str
    cell_id: str
    cell_receipt_sha256: str
    selected_origin_cell_ids: tuple[str, ...]
    segment_receipt_sha256: str
    span_identity_sha256: str
    partition_id: str
    quote: str
    quote_sha256: str
    token_count: int
    role: str
    created_at: str
    event_dates: tuple[str, ...]
    surface_terms: tuple[str, ...]
    action_concepts: tuple[str, ...]
    contains_numeric_value: bool
    supported_obligation_ids: tuple[str, ...]
    hydration_routes: tuple[str, ...]
    lane_scores: tuple[tuple[str, float], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "global candidate ID"),
            (self.source_history_receipt_sha256, "global candidate source history"),
            (self.cell_receipt_sha256, "global candidate cell"),
            (self.segment_receipt_sha256, "global candidate segment"),
            (self.span_identity_sha256, "global candidate span"),
            (self.quote_sha256, "global candidate quote"),
        ):
            require_sha256(value, label)
        _require(
            re.fullmatch(r"G[0-9]{6}", self.source_group_handle) is not None,
            "global candidate source group changed",
        )
        require_text(self.source_id, "global candidate source")
        require_text(self.cell_id, "global candidate cell ID")
        require_text(self.partition_id, "global candidate partition")
        require_text(self.quote, "global candidate quote")
        _require(
            self.quote_sha256 == quote_sha256(self.quote)
            and type(self.token_count) is int
            and self.token_count == count_tokens(self.quote),
            "global candidate exact quote changed",
        )
        require_text(self.role, "global candidate role")
        require_text(self.created_at, "global candidate created-at")
        for values, label in (
            (self.selected_origin_cell_ids, "candidate origin cells"),
            (self.event_dates, "candidate event dates"),
            (self.surface_terms, "candidate surface terms"),
            (self.action_concepts, "candidate actions"),
            (self.supported_obligation_ids, "candidate obligations"),
            (self.hydration_routes, "candidate hydration routes"),
        ):
            _ordered_unique(values, label)
        _require(
            type(self.contains_numeric_value) is bool,
            "global candidate numeric flag changed",
        )
        _require(
            type(self.lane_scores) is tuple
            and tuple(name for name, _value in self.lane_scores) == LANE_IDS,
            "candidate lane scores changed order",
        )
        for _name, value in self.lane_scores:
            _finite(value, "candidate lane score")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global candidate receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "action_concepts": list(self.action_concepts),
            "candidate_id": self.candidate_id,
            "cell_id": self.cell_id,
            "cell_receipt_sha256": self.cell_receipt_sha256,
            "contains_numeric_value": self.contains_numeric_value,
            "created_at": self.created_at,
            "event_dates": list(self.event_dates),
            "format": CANDIDATE_FORMAT,
            "hydration_routes": list(self.hydration_routes),
            "lane_scores": [
                {"lane_id": lane_id, "score": score}
                for lane_id, score in self.lane_scores
            ],
            "partition_id": self.partition_id,
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "selected_origin_cell_ids": list(self.selected_origin_cell_ids),
            "source_group_handle": self.source_group_handle,
            "source_history_receipt_sha256": self.source_history_receipt_sha256,
            "source_id": self.source_id,
            "span_identity_sha256": self.span_identity_sha256,
            "supported_obligation_ids": list(self.supported_obligation_ids),
            "surface_terms": list(self.surface_terms),
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _segment_lane_scores(
    index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    cell: SemanticResidualCell,
    segment: ExactCellSegment,
    obligations: Sequence[GlobalEvidenceObligation],
    idf: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    if cell.normalized_source_centroid is None or not query.present_query_vectors:
        dense = -1.0
    else:
        dense = max(
            _dot(cell.normalized_source_centroid, vector)
            for vector in query.present_query_vectors
        )
    query_terms = set(query.query_terms) | set(query.slot_terms)
    term_score = math.fsum(idf[term] for term in query_terms & set(segment.surface_terms))
    action_score = 3.0 * len(
        set(query.action_concepts) & set(linked_action_concepts(segment.quote))
    )
    supported = sum(_segment_supports(segment, row) for row in obligations)
    generic_penalty = 4.0 if _GENERIC_ADVICE_RE.search(segment.quote) else 0.0
    sparse = term_score + action_score + 4.0 * supported - generic_penalty
    user = segment.role == "user"
    first_person = bool(_FIRST_PERSON_RE.search(segment.quote))
    needs_numeric = any(row.requires_numeric for row in obligations)
    needs_temporal = any(row.kind == "date" for row in obligations) or (
        query.operator_spec.temporal_mode is not TemporalMode.NONE
    )
    date_support = any(
        row.kind == "date" and _segment_supports(segment, row)
        for row in obligations
    )
    dated_action_witness = _joint_dated_action_witness(
        query,
        obligations,
        segment,
    )
    personal = (
        (4.0 if user else 0.0)
        + (2.0 if first_person else 0.0)
        + (2.0 if needs_numeric and segment.contains_numeric_value else 0.0)
        + (3.0 if needs_temporal and date_support else 0.0)
        + action_score
        + min(term_score, 4.0)
        + (DATED_ACTION_WITNESS_BONUS if dated_action_witness else 0.0)
        - generic_penalty
    )
    diversity = (
        max(dense, 0.0)
        + min(max(sparse, 0.0), 8.0)
        + (2.0 if user else 0.0)
        + (
            1.0
            if linked_event_dates(
                segment.quote,
                segment.created_at,
                segment.event_dates,
            )
            else 0.0
        )
    )
    return (
        ("dense", dense),
        ("sparse", sparse),
        ("personal_temporal", personal),
        ("source_date_diversity", diversity),
    )


def _history_cells(
    index: SemanticResidualIndex,
) -> Mapping[str, tuple[SemanticResidualCell, ...]]:
    by_source: dict[str, list[SemanticResidualCell]] = defaultdict(list)
    for cell in index.cells:
        by_source[cell.source_id].append(cell)
    output: dict[str, tuple[SemanticResidualCell, ...]] = {}
    for source_id, rows in sorted(by_source.items()):
        ordered = tuple(sorted(rows, key=lambda row: row.source_cell_ordinal))
        _require(
            tuple(row.source_cell_ordinal for row in ordered)
            == tuple(range(len(ordered)))
            and all(row.source_cell_count == len(ordered) for row in ordered)
            and len({row.source_history_receipt_sha256 for row in ordered}) == 1,
            "global hydration lost source-local cell order",
        )
        output[source_id] = ordered
    return MappingProxyType(output)


def _candidate_combined_score(candidate: GlobalCompletionCandidate) -> float:
    values = dict(candidate.lane_scores)
    return max(
        values["dense"],
        values["sparse"] / 12.0,
        values["personal_temporal"] / 12.0,
        values["source_date_diversity"] / 10.0,
    )


def _hydrate_candidates(
    index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    obligations: Sequence[GlobalEvidenceObligation],
    tree_frontier: GlobalTreeFrontier,
    policy: SemanticGlobalCompletionPolicy,
) -> tuple[
    tuple[GlobalCompletionCandidate, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    histories = _history_cells(index)
    groups = semantic_residual_source_group_map(
        tuple(sorted(histories))
    )
    by_cell = index.cell_by_id
    retained_visit_order = tuple(
        row.covered_leaf_cell_ids[0]
        for row in tree_frontier.visits
        if row.action == "retained_leaf"
    )
    raw: dict[
        str,
        tuple[SemanticResidualCell, ExactCellSegment, list[str], list[str]],
    ] = {}
    for origin_id in retained_visit_order:
        origin = by_cell[origin_id]
        history = histories[origin.source_id]
        start = max(0, origin.source_cell_ordinal - policy.source_neighbor_radius)
        end = min(
            len(history),
            origin.source_cell_ordinal + policy.source_neighbor_radius + 1,
        )
        for cell in history[start:end]:
            offset = cell.source_cell_ordinal - origin.source_cell_ordinal
            for segment in cell.segments:
                # Preserve the complete local discourse neighborhood.  Role
                # is a ranking signal, not authority to discard linked facts.
                route = (
                    "selected_global_cell"
                    if offset == 0
                    else f"source_local_neighbor:offset={offset:+d}"
                )
                existing = raw.get(segment.receipt_sha256)
                if existing is None:
                    raw[segment.receipt_sha256] = (
                        cell,
                        segment,
                        [origin_id],
                        [route],
                    )
                else:
                    existing[2].append(origin_id)
                    existing[3].append(route)
    query_terms = set(query.query_terms) | set(query.slot_terms)
    leaf_count = len(index.cells)
    idf = {
        term: math.log(
            (leaf_count + 1)
            / (sum(term in set(cell.surface_terms) for cell in index.cells) + 1)
        )
        for term in query_terms
    }
    candidates: list[GlobalCompletionCandidate] = []
    for segment_receipt, (cell, segment, origin_ids, routes) in raw.items():
        supported = tuple(
            row.obligation_id
            for row in obligations
            if _segment_supports(segment, row)
        )
        candidate_id = identity_sha256(
            {
                "format": f"{CANDIDATE_FORMAT}-identity",
                "index_receipt_sha256": index.receipt_sha256,
                "segment_receipt_sha256": segment_receipt,
            }
        )
        candidates.append(
            GlobalCompletionCandidate(
                candidate_id=candidate_id,
                source_group_handle=groups[cell.source_id],
                source_id=cell.source_id,
                source_history_receipt_sha256=cell.source_history_receipt_sha256,
                cell_id=cell.cell_id,
                cell_receipt_sha256=cell.receipt_sha256,
                selected_origin_cell_ids=tuple(dict.fromkeys(origin_ids)),
                segment_receipt_sha256=segment.receipt_sha256,
                span_identity_sha256=_span_identity(segment.span),
                partition_id=segment.partition_id,
                quote=segment.quote,
                quote_sha256=segment.quote_sha256,
                token_count=segment.token_count,
                role=segment.role,
                created_at=segment.created_at,
                event_dates=linked_event_dates(
                    segment.quote,
                    segment.created_at,
                    segment.event_dates,
                ),
                surface_terms=segment.surface_terms,
                action_concepts=linked_action_concepts(segment.quote),
                contains_numeric_value=segment.contains_numeric_value,
                supported_obligation_ids=supported,
                hydration_routes=tuple(dict.fromkeys(routes)),
                lane_scores=_segment_lane_scores(
                    index,
                    query,
                    cell,
                    segment,
                    obligations,
                    idf,
                ),
            )
        )
    candidates.sort(
        key=lambda row: (
            -_candidate_combined_score(row),
            -len(row.supported_obligation_ids),
            row.source_id,
            -_timestamp(row.created_at),
            row.segment_receipt_sha256,
        )
    )
    protected: list[GlobalCompletionCandidate] = []
    for obligation in obligations:
        witness = next(
            (
                row
                for row in candidates
                if obligation.obligation_id in row.supported_obligation_ids
            ),
            None,
        )
        if witness is not None and witness not in protected:
            protected.append(witness)
    ordered = tuple((*protected, *(row for row in candidates if row not in protected)))
    # Obligation witnesses receive the front of the fixed hydration budget,
    # but they do not silently turn a named maximum into a soft limit.  When
    # distinct witnesses outnumber the cap, the remainder stay in the sealed
    # omitted partition so closure remains open for a successor search.
    admitted_count = policy.max_hydrated_segments
    admitted = ordered[:admitted_count]
    omitted = tuple(
        row.segment_receipt_sha256
        for row in ordered[admitted_count:]
    )
    return (
        admitted,
        tuple(row.receipt_sha256 for row in ordered),
        omitted,
    )


@dataclass(frozen=True, slots=True)
class GlobalLaneSelectionReceipt:
    lane_budget: GlobalLaneBudget
    candidate_population_receipt_sha256: str
    eligible_candidate_receipt_sha256s: tuple[str, ...]
    selected_candidate_receipt_sha256s: tuple[str, ...]
    selected_segment_receipt_sha256s: tuple[str, ...]
    protected_obligation_ids: tuple[str, ...]
    selected_pre_dedup_tokens: int
    budget_exhausted: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.lane_budget) is GlobalLaneBudget, "lane budget changed")
        require_sha256(
            self.candidate_population_receipt_sha256,
            "lane candidate population",
        )
        for values, label in (
            (self.eligible_candidate_receipt_sha256s, "lane eligible candidates"),
            (self.selected_candidate_receipt_sha256s, "lane selected candidates"),
            (self.selected_segment_receipt_sha256s, "lane selected segments"),
            (self.protected_obligation_ids, "lane protected obligations"),
        ):
            _ordered_unique(values, label)
        _require(
            len(self.selected_candidate_receipt_sha256s)
            == len(self.selected_segment_receipt_sha256s)
            <= self.lane_budget.max_selected_segments
            and set(self.selected_candidate_receipt_sha256s)
            <= set(self.eligible_candidate_receipt_sha256s),
            "lane selection escaped its independent population/count cap",
        )
        _require(
            type(self.selected_pre_dedup_tokens) is int
            and 0 <= self.selected_pre_dedup_tokens
            <= self.lane_budget.pre_dedup_token_cap
            and type(self.budget_exhausted) is bool,
            "lane token accounting changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global lane receipt",
            ),
        )

    @property
    def lane_id(self) -> str:
        return self.lane_budget.lane_id

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "budget_exhausted": self.budget_exhausted,
            "candidate_population_receipt_sha256": self.candidate_population_receipt_sha256,
            "eligible_candidate_receipt_sha256s": list(
                self.eligible_candidate_receipt_sha256s
            ),
            "format": LANE_FORMAT,
            "lane_budget": self.lane_budget.projection(),
            "protected_obligation_ids": list(self.protected_obligation_ids),
            "selected_candidate_receipt_sha256s": list(
                self.selected_candidate_receipt_sha256s
            ),
            "selected_pre_dedup_tokens": self.selected_pre_dedup_tokens,
            "selected_segment_receipt_sha256s": list(
                self.selected_segment_receipt_sha256s
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _candidate_date_bucket(candidate: GlobalCompletionCandidate) -> str:
    if candidate.event_dates:
        return candidate.event_dates[0][:7]
    try:
        return datetime.fromisoformat(
            candidate.created_at.replace("Z", "+00:00")
        ).date().isoformat()[:7]
    except ValueError:
        return "undated"


def _lane_eligible(
    lane_id: str,
    candidate: GlobalCompletionCandidate,
) -> bool:
    scores = dict(candidate.lane_scores)
    if lane_id == "dense":
        return scores[lane_id] > -1.0
    if lane_id == "sparse":
        return scores[lane_id] > 0.0
    if lane_id == "personal_temporal":
        return scores[lane_id] > 0.0 and (
            bool(candidate.supported_obligation_ids)
            or scores["sparse"] > 0.0
            or scores["dense"] >= 0.15
        )
    return (
        bool(candidate.supported_obligation_ids)
        or scores["sparse"] > 0.0
        or scores["dense"] >= 0.15
    )


def _rank_lane(
    lane_id: str,
    candidates: Sequence[GlobalCompletionCandidate],
) -> tuple[GlobalCompletionCandidate, ...]:
    eligible = tuple(row for row in candidates if _lane_eligible(lane_id, row))
    if lane_id != "source_date_diversity":
        return tuple(
            sorted(
                eligible,
                key=lambda row: (
                    -dict(row.lane_scores)[lane_id],
                    -len(row.supported_obligation_ids),
                    row.source_id,
                    -_timestamp(row.created_at),
                    row.segment_receipt_sha256,
                ),
            )
        )
    ranked = tuple(
        sorted(
            eligible,
            key=lambda row: (
                -dict(row.lane_scores)[lane_id],
                -len(row.supported_obligation_ids),
                row.source_id,
                _candidate_date_bucket(row),
                row.segment_receipt_sha256,
            ),
        )
    )
    chosen: list[GlobalCompletionCandidate] = []
    seen_source_dates: set[tuple[str, str]] = set()
    for row in ranked:
        key = (row.source_id, _candidate_date_bucket(row))
        if key not in seen_source_dates:
            chosen.append(row)
            seen_source_dates.add(key)
    chosen.extend(row for row in ranked if row not in chosen)
    return tuple(chosen)


def _protected_kinds_for_lane(lane_id: str) -> frozenset[str]:
    if lane_id == "sparse":
        return frozenset({"typed_slot", "entity", "action"})
    if lane_id == "personal_temporal":
        return frozenset({"date", "role", "numeric"})
    return frozenset()


def _select_lane(
    budget: GlobalLaneBudget,
    candidates: Sequence[GlobalCompletionCandidate],
    obligations: Sequence[GlobalEvidenceObligation],
    candidate_population_receipt_sha256: str,
) -> tuple[GlobalLaneSelectionReceipt, tuple[GlobalCompletionCandidate, ...]]:
    ranked = _rank_lane(budget.lane_id, candidates)
    protected_ids: list[str] = []
    protected_rows: list[GlobalCompletionCandidate] = []
    protected_kinds = _protected_kinds_for_lane(budget.lane_id)
    for obligation in obligations:
        if obligation.kind not in protected_kinds:
            continue
        witness = next(
            (
                row
                for row in ranked
                if obligation.obligation_id in row.supported_obligation_ids
            ),
            None,
        )
        if witness is not None:
            protected_ids.append(obligation.obligation_id)
            if witness not in protected_rows:
                protected_rows.append(witness)
    ordered = tuple((*protected_rows, *(row for row in ranked if row not in protected_rows)))
    selected: list[GlobalCompletionCandidate] = []
    used = 0
    omitted = False
    for row in ordered:
        if len(selected) >= budget.max_selected_segments:
            omitted = True
            continue
        if used + row.token_count > budget.pre_dedup_token_cap:
            omitted = True
            # Skip and continue so a later exact short witness can still fit.
            continue
        selected.append(row)
        used += row.token_count
    receipt = GlobalLaneSelectionReceipt(
        lane_budget=budget,
        candidate_population_receipt_sha256=candidate_population_receipt_sha256,
        eligible_candidate_receipt_sha256s=tuple(row.receipt_sha256 for row in ranked),
        selected_candidate_receipt_sha256s=tuple(row.receipt_sha256 for row in selected),
        selected_segment_receipt_sha256s=tuple(
            row.segment_receipt_sha256 for row in selected
        ),
        protected_obligation_ids=tuple(dict.fromkeys(protected_ids)),
        selected_pre_dedup_tokens=used,
        budget_exhausted=omitted,
    )
    return receipt, tuple(selected)


def _select_lanes(
    policy: SemanticGlobalCompletionPolicy,
    candidates: Sequence[GlobalCompletionCandidate],
    obligations: Sequence[GlobalEvidenceObligation],
) -> tuple[
    tuple[GlobalLaneSelectionReceipt, ...],
    tuple[GlobalCompletionCandidate, ...],
    Mapping[str, tuple[str, ...]],
]:
    population_receipt = identity_sha256(
        {
            "format": f"{LANE_FORMAT}-candidate-population",
            "ordered_candidate_receipt_sha256s": [
                row.receipt_sha256 for row in candidates
            ],
        }
    )
    lane_receipts: list[GlobalLaneSelectionReceipt] = []
    lane_rows: list[tuple[GlobalCompletionCandidate, ...]] = []
    for budget in policy.lane_budgets:
        lane_receipt, selected = _select_lane(
            budget,
            candidates,
            obligations,
            population_receipt,
        )
        lane_receipts.append(lane_receipt)
        lane_rows.append(selected)
    union: list[GlobalCompletionCandidate] = []
    owners: dict[str, list[str]] = defaultdict(list)
    width = max((len(rows) for rows in lane_rows), default=0)
    for position in range(width):
        for lane_receipt, rows in zip(lane_receipts, lane_rows, strict=True):
            if position >= len(rows):
                continue
            row = rows[position]
            if lane_receipt.lane_id not in owners[row.segment_receipt_sha256]:
                owners[row.segment_receipt_sha256].append(lane_receipt.lane_id)
            if row not in union:
                union.append(row)
    return (
        tuple(lane_receipts),
        tuple(union),
        MappingProxyType(
            {key: tuple(value) for key, value in owners.items()}
        ),
    )


@dataclass(frozen=True, slots=True)
class GlobalCompletionEvidence:
    candidate_id: str
    cell_id: str
    segment_receipt_sha256: str
    source_group_handle: str
    quote: str
    quote_sha256: str
    token_count: int
    role: str
    created_at: str
    event_dates: tuple[str, ...]
    contains_numeric_value: bool
    supported_obligation_ids: tuple[str, ...]
    selected_lane_ids: tuple[str, ...]
    citation_binding_receipt_sha256: str
    packing_protection: Literal["obligation_witness", "ordinary"]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "global evidence candidate"),
            (self.segment_receipt_sha256, "global evidence segment"),
            (self.quote_sha256, "global evidence quote"),
            (self.citation_binding_receipt_sha256, "global evidence citation"),
        ):
            require_sha256(value, label)
        require_text(self.cell_id, "global evidence cell")
        _require(
            re.fullmatch(r"G[0-9]{6}", self.source_group_handle) is not None,
            "global evidence source group changed",
        )
        require_text(self.quote, "global evidence quote")
        _require(
            self.quote_sha256 == quote_sha256(self.quote)
            and type(self.token_count) is int
            and self.token_count == count_tokens(self.quote),
            "global evidence exact quote changed",
        )
        require_text(self.role, "global evidence role")
        require_text(self.created_at, "global evidence created-at")
        for values, label in (
            (self.event_dates, "global evidence dates"),
            (self.supported_obligation_ids, "global evidence obligations"),
            (self.selected_lane_ids, "global evidence selected lanes"),
        ):
            _ordered_unique(values, label)
        _require(
            type(self.contains_numeric_value) is bool
            and self.packing_protection in {"obligation_witness", "ordinary"},
            "global evidence flags changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global evidence receipt",
            ),
        )

    def provider_row(self) -> dict[str, object]:
        return {
            "cell_handle": self.cell_id,
            "created_at": self.created_at,
            "event_dates": list(self.event_dates),
            "evidence_handle": self.candidate_id,
            "packing_protection": self.packing_protection,
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "selected_lane_ids": list(self.selected_lane_ids),
            "source_group_handle": self.source_group_handle,
            "supported_obligation_ids": list(self.supported_obligation_ids),
        }

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value = {
            **self.provider_row(),
            "candidate_id": self.candidate_id,
            "citation_binding_receipt_sha256": self.citation_binding_receipt_sha256,
            "contains_numeric_value": self.contains_numeric_value,
            "format": EVIDENCE_FORMAT,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class GlobalProtectedDuplicate:
    candidate_id: str
    segment_receipt_sha256: str
    span_identity_sha256: str
    protected_candidate_id: str
    protected_binding_receipt_sha256: str
    selected_lane_ids: tuple[str, ...]
    supported_obligation_ids: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "global duplicate candidate"),
            (self.segment_receipt_sha256, "global duplicate segment"),
            (self.span_identity_sha256, "global duplicate span"),
            (self.protected_candidate_id, "global duplicate owner"),
            (self.protected_binding_receipt_sha256, "global duplicate owner binding"),
        ):
            require_sha256(value, label)
        _ordered_unique(self.selected_lane_ids, "global duplicate lanes")
        _ordered_unique(self.supported_obligation_ids, "global duplicate obligations")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global duplicate receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "format": DUPLICATE_FORMAT,
            "protected_binding_receipt_sha256": self.protected_binding_receipt_sha256,
            "protected_candidate_id": self.protected_candidate_id,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "selected_lane_ids": list(self.selected_lane_ids),
            "span_identity_sha256": self.span_identity_sha256,
            "supported_obligation_ids": list(self.supported_obligation_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class GlobalSelectionAttempt:
    selection_rank: int
    candidate_id: str
    candidate_receipt_sha256: str
    segment_receipt_sha256: str
    selected_lane_ids: tuple[str, ...]
    supported_obligation_ids: tuple[str, ...]
    disposition: Literal["protected_exact_duplicate", "packed_novel", "budget_unpacked"]
    protected_duplicate_receipt_sha256: str | None
    evidence_receipt_sha256: str | None
    citation_binding_receipt_sha256: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.selection_rank) is int and self.selection_rank >= 0,
            "global selection rank changed",
        )
        for value, label in (
            (self.candidate_id, "global attempt candidate"),
            (self.candidate_receipt_sha256, "global attempt candidate receipt"),
            (self.segment_receipt_sha256, "global attempt segment"),
        ):
            require_sha256(value, label)
        _ordered_unique(self.selected_lane_ids, "global attempt lanes")
        _ordered_unique(self.supported_obligation_ids, "global attempt obligations")
        _require(
            self.disposition
            in {"protected_exact_duplicate", "packed_novel", "budget_unpacked"},
            "global attempt disposition changed",
        )
        presence = (
            self.protected_duplicate_receipt_sha256 is not None,
            self.evidence_receipt_sha256 is not None,
            self.citation_binding_receipt_sha256 is not None,
        )
        expected = {
            "protected_exact_duplicate": (True, False, False),
            "packed_novel": (False, True, True),
            "budget_unpacked": (False, False, True),
        }[self.disposition]
        _require(presence == expected, "global attempt lost exact ownership")
        for value, label in (
            (self.protected_duplicate_receipt_sha256, "attempt duplicate"),
            (self.evidence_receipt_sha256, "attempt evidence"),
            (self.citation_binding_receipt_sha256, "attempt citation"),
        ):
            if value is not None:
                require_sha256(value, label)
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global attempt receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "candidate_receipt_sha256": self.candidate_receipt_sha256,
            "citation_binding_receipt_sha256": self.citation_binding_receipt_sha256,
            "disposition": self.disposition,
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "format": ATTEMPT_FORMAT,
            "protected_duplicate_receipt_sha256": self.protected_duplicate_receipt_sha256,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "selected_lane_ids": list(self.selected_lane_ids),
            "selection_rank": self.selection_rank,
            "supported_obligation_ids": list(self.supported_obligation_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class GlobalCompletionClosure:
    required_obligation_ids: tuple[str, ...]
    covered_obligation_ids: tuple[str, ...]
    unresolved_obligation_ids: tuple[str, ...]
    required_typed_slot_ids: tuple[str, ...]
    covered_typed_slot_ids: tuple[str, ...]
    unresolved_typed_slot_ids: tuple[str, ...]
    selected_segment_receipt_sha256s: tuple[str, ...]
    packed_segment_receipt_sha256s: tuple[str, ...]
    protected_duplicate_segment_receipt_sha256s: tuple[str, ...]
    budget_unpacked_segment_receipt_sha256s: tuple[str, ...]
    tree_search_closed: bool
    hydration_complete: bool
    independent_lane_selection_complete: bool
    packing_closed: bool
    operand_closure_required: bool
    compiled_operand_closure_proven: bool
    support_closure_proven: Literal[False]
    needs_further_global_search: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for values, label in (
            (self.required_obligation_ids, "closure required obligations"),
            (self.covered_obligation_ids, "closure covered obligations"),
            (self.unresolved_obligation_ids, "closure unresolved obligations"),
            (self.required_typed_slot_ids, "closure required typed slots"),
            (self.covered_typed_slot_ids, "closure covered typed slots"),
            (self.unresolved_typed_slot_ids, "closure unresolved typed slots"),
            (self.selected_segment_receipt_sha256s, "closure selected segments"),
            (self.packed_segment_receipt_sha256s, "closure packed segments"),
            (
                self.protected_duplicate_segment_receipt_sha256s,
                "closure protected duplicate segments",
            ),
            (
                self.budget_unpacked_segment_receipt_sha256s,
                "closure budget-unpacked segments",
            ),
        ):
            _ordered_unique(values, label)
        required = set(self.required_obligation_ids)
        covered = set(self.covered_obligation_ids)
        unresolved = set(self.unresolved_obligation_ids)
        _require(
            covered.isdisjoint(unresolved)
            and covered | unresolved == required,
            "global closure lost its obligation partition",
        )
        required_slots = set(self.required_typed_slot_ids)
        covered_slots = set(self.covered_typed_slot_ids)
        unresolved_slots = set(self.unresolved_typed_slot_ids)
        _require(
            covered_slots.isdisjoint(unresolved_slots)
            and covered_slots | unresolved_slots == required_slots,
            "global closure lost its typed-slot partition",
        )
        selected = set(self.selected_segment_receipt_sha256s)
        packed = set(self.packed_segment_receipt_sha256s)
        duplicates = set(self.protected_duplicate_segment_receipt_sha256s)
        unpacked = set(self.budget_unpacked_segment_receipt_sha256s)
        _require(
            packed.isdisjoint(duplicates)
            and packed.isdisjoint(unpacked)
            and duplicates.isdisjoint(unpacked)
            and packed | duplicates | unpacked == selected,
            "global closure lost its post-selection partition",
        )
        for value, label in (
            (self.tree_search_closed, "tree closure"),
            (self.hydration_complete, "hydration closure"),
            (self.independent_lane_selection_complete, "lane closure"),
            (self.packing_closed, "packing closure"),
            (self.operand_closure_required, "operand closure requirement"),
            (self.compiled_operand_closure_proven, "compiled operand closure"),
            (self.needs_further_global_search, "further search flag"),
        ):
            _require(type(value) is bool, f"{label} changed type")
        expected_compiled = bool(
            self.operand_closure_required
            and self.tree_search_closed
            and self.hydration_complete
            and self.independent_lane_selection_complete
            and self.packing_closed
            and not self.unresolved_obligation_ids
        )
        _require(
            self.compiled_operand_closure_proven == expected_compiled,
            "compiled operand closure was overstated",
        )
        _require(
            self.support_closure_proven is False,
            "provider-free ranking cannot prove semantic support closure",
        )
        expected_needs_search = bool(
            self.unresolved_obligation_ids
            or not self.tree_search_closed
            or not self.hydration_complete
            or not self.independent_lane_selection_complete
            or not self.packing_closed
            or (
                self.operand_closure_required
                and not self.compiled_operand_closure_proven
            )
        )
        _require(
            self.needs_further_global_search == expected_needs_search,
            "global unresolved frontier changed routing",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global closure receipt",
            ),
        )

    def provider_projection(self) -> dict[str, object]:
        return {
            "compiled_operand_closure_proven": self.compiled_operand_closure_proven,
            "covered_obligation_ids": list(self.covered_obligation_ids),
            "needs_further_global_search": self.needs_further_global_search,
            "packing_closed": self.packing_closed,
            "support_closure_proven": False,
            "tree_search_closed": self.tree_search_closed,
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
        }

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "budget_unpacked_segment_receipt_sha256s": list(
                self.budget_unpacked_segment_receipt_sha256s
            ),
            "compiled_operand_closure_proven": self.compiled_operand_closure_proven,
            "covered_obligation_ids": list(self.covered_obligation_ids),
            "covered_typed_slot_ids": list(self.covered_typed_slot_ids),
            "format": CLOSURE_FORMAT,
            "hydration_complete": self.hydration_complete,
            "independent_lane_selection_complete": self.independent_lane_selection_complete,
            "needs_further_global_search": self.needs_further_global_search,
            "operand_closure_required": self.operand_closure_required,
            "packed_segment_receipt_sha256s": list(self.packed_segment_receipt_sha256s),
            "packing_closed": self.packing_closed,
            "protected_duplicate_segment_receipt_sha256s": list(
                self.protected_duplicate_segment_receipt_sha256s
            ),
            "required_obligation_ids": list(self.required_obligation_ids),
            "required_typed_slot_ids": list(self.required_typed_slot_ids),
            "selected_segment_receipt_sha256s": list(
                self.selected_segment_receipt_sha256s
            ),
            "support_closure_proven": False,
            "tree_search_closed": self.tree_search_closed,
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
            "unresolved_typed_slot_ids": list(self.unresolved_typed_slot_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _provider_payload(
    evidence: Sequence[GlobalCompletionEvidence],
    closure: GlobalCompletionClosure,
) -> dict[str, object]:
    return {
        "closure": closure.provider_projection(),
        "evidence": [row.provider_row() for row in evidence],
        "format": f"{RESULT_FORMAT}-provider-plane",
        "instructions": (
            "Use exact evidence only. Preserve unresolved status; do not infer "
            "global completeness when support_closure_proven is false."
        ),
    }


def _binding_for_candidate(
    index: SemanticResidualIndex,
    candidate: GlobalCompletionCandidate,
    segment: ExactCellSegment,
) -> LocalCitationBinding:
    return LocalCitationBinding(
        candidate_id=candidate.candidate_id,
        source_group_handle=candidate.source_group_handle,
        namespace_id=index.namespace_id,
        cache_receipt_sha256=index.cache_receipt_sha256,
        source_database_sha256=index.source_database_sha256,
        source_store_receipt_sha256=index.source_store_receipt_sha256,
        source_id=candidate.source_id,
        partition_id=candidate.partition_id,
        span=segment.span,
        quote_sha256=candidate.quote_sha256,
    )


def _evidence_for_candidate(
    candidate: GlobalCompletionCandidate,
    binding: LocalCitationBinding,
    lane_ids: tuple[str, ...],
    protected_obligation_ids: set[str],
) -> GlobalCompletionEvidence:
    return GlobalCompletionEvidence(
        candidate_id=candidate.candidate_id,
        cell_id=candidate.cell_id,
        segment_receipt_sha256=candidate.segment_receipt_sha256,
        source_group_handle=candidate.source_group_handle,
        quote=candidate.quote,
        quote_sha256=candidate.quote_sha256,
        token_count=candidate.token_count,
        role=candidate.role,
        created_at=candidate.created_at,
        event_dates=candidate.event_dates,
        contains_numeric_value=candidate.contains_numeric_value,
        supported_obligation_ids=candidate.supported_obligation_ids,
        selected_lane_ids=lane_ids,
        citation_binding_receipt_sha256=binding.receipt_sha256,
        packing_protection=(
            "obligation_witness"
            if protected_obligation_ids & set(candidate.supported_obligation_ids)
            else "ordinary"
        ),
    )


def _protected_inventory(
    index: SemanticResidualIndex,
    protected_evidence: Sequence[LocalCitationBinding],
) -> tuple[Mapping[str, LocalCitationBinding], str]:
    values = tuple(protected_evidence)
    all_segments = tuple(
        segment for cell in index.cells for segment in cell.segments
    )
    segment_by_span = {
        _span_identity(segment.span): segment for segment in all_segments
    }
    _require(
        len(segment_by_span) == sum(len(cell.segments) for cell in index.cells),
        "global index repeated an exact segment span",
    )
    segments_by_chunk: dict[str, list[ExactCellSegment]] = defaultdict(list)
    for segment in all_segments:
        segments_by_chunk[segment.span.chunk_id].append(segment)
    owners: dict[str, LocalCitationBinding] = {}
    for binding in values:
        _require(
            type(binding) is LocalCitationBinding
            and binding.namespace_id == index.namespace_id
            and binding.cache_receipt_sha256 == index.cache_receipt_sha256
            and binding.source_database_sha256 == index.source_database_sha256
            and binding.source_store_receipt_sha256
            == index.source_store_receipt_sha256,
            "protected evidence escaped the global immutable lineage",
        )
        span_sha = _span_identity(binding.span)
        segment = segment_by_span.get(span_sha)
        if segment is None:
            containing = tuple(
                row
                for row in segments_by_chunk.get(binding.span.chunk_id, ())
                if row.source_id == binding.source_id
                and row.partition_id == binding.partition_id
                and row.span.start_char <= binding.span.start_char
                and binding.span.end_char <= row.span.end_char
                and quote_sha256(
                    row.quote[
                        binding.span.start_char - row.span.start_char :
                        binding.span.end_char - row.span.start_char
                    ]
                )
                == binding.quote_sha256
            )
            _require(
                len(containing) == 1,
                "protected evidence does not resolve to one global exact segment",
            )
            segment = containing[0]
        _require(
            segment.source_id == binding.source_id
            and segment.partition_id == binding.partition_id
            and segment.span.chunk_id == binding.span.chunk_id,
            "protected evidence does not resolve to one global exact segment",
        )
        _require(span_sha not in owners, "protected planes repeated an exact owner span")
        owners[span_sha] = binding
    return (
        MappingProxyType(owners),
        semantic_residual_protected_evidence_population_receipt(index, values),
    )


def _protect_obligation_witness_order(
    selected: Sequence[GlobalCompletionCandidate],
    obligations: Sequence[GlobalEvidenceObligation],
) -> tuple[tuple[GlobalCompletionCandidate, ...], frozenset[str]]:
    protected_rows: list[GlobalCompletionCandidate] = []
    protected_ids: set[str] = set()
    for obligation in obligations:
        witness = next(
            (
                row
                for row in selected
                if obligation.obligation_id in row.supported_obligation_ids
            ),
            None,
        )
        if witness is not None:
            protected_ids.add(obligation.obligation_id)
            if witness not in protected_rows:
                protected_rows.append(witness)
    ordered = tuple(
        (*protected_rows, *(row for row in selected if row not in protected_rows))
    )
    return ordered, frozenset(protected_ids)


def _make_closure(
    query: SemanticResidualQuery,
    obligations: Sequence[GlobalEvidenceObligation],
    tree_frontier: GlobalTreeFrontier,
    omitted_hydrated_segment_receipts: Sequence[str],
    lane_receipts: Sequence[GlobalLaneSelectionReceipt],
    selected: Sequence[GlobalCompletionCandidate],
    evidence: Sequence[GlobalCompletionEvidence],
    duplicates: Sequence[GlobalProtectedDuplicate],
    unpacked_segment_receipts: Sequence[str],
) -> GlobalCompletionClosure:
    covered_set = {
        obligation_id
        for row in (*tuple(evidence), *tuple(duplicates))
        for obligation_id in row.supported_obligation_ids
    }
    required_ids = tuple(row.obligation_id for row in obligations)
    covered_ids = tuple(value for value in required_ids if value in covered_set)
    unresolved_ids = tuple(value for value in required_ids if value not in covered_set)
    typed = tuple(row for row in obligations if row.typed_slot_id is not None)
    required_slots = tuple(row.typed_slot_id for row in typed if row.typed_slot_id is not None)
    covered_slots = tuple(
        row.typed_slot_id
        for row in typed
        if row.typed_slot_id is not None and row.obligation_id in covered_set
    )
    unresolved_slots = tuple(
        value for value in required_slots if value not in set(covered_slots)
    )
    hydration_complete = not tuple(omitted_hydrated_segment_receipts)
    lanes_complete = not any(row.budget_exhausted for row in lane_receipts)
    packing_closed = not tuple(unpacked_segment_receipts)
    operand_required = query.operator_spec.requires_complete_frontier
    compiled = bool(
        operand_required
        and tree_frontier.search_closed
        and hydration_complete
        and lanes_complete
        and packing_closed
        and not unresolved_ids
    )
    needs_more = bool(
        unresolved_ids
        or not tree_frontier.search_closed
        or not hydration_complete
        or not lanes_complete
        or not packing_closed
        or (operand_required and not compiled)
    )
    return GlobalCompletionClosure(
        required_obligation_ids=required_ids,
        covered_obligation_ids=covered_ids,
        unresolved_obligation_ids=unresolved_ids,
        required_typed_slot_ids=required_slots,
        covered_typed_slot_ids=covered_slots,
        unresolved_typed_slot_ids=unresolved_slots,
        selected_segment_receipt_sha256s=tuple(
            row.segment_receipt_sha256 for row in selected
        ),
        packed_segment_receipt_sha256s=tuple(
            row.segment_receipt_sha256 for row in evidence
        ),
        protected_duplicate_segment_receipt_sha256s=tuple(
            row.segment_receipt_sha256 for row in duplicates
        ),
        budget_unpacked_segment_receipt_sha256s=tuple(unpacked_segment_receipts),
        tree_search_closed=tree_frontier.search_closed,
        hydration_complete=hydration_complete,
        independent_lane_selection_complete=lanes_complete,
        packing_closed=packing_closed,
        operand_closure_required=operand_required,
        compiled_operand_closure_proven=compiled,
        support_closure_proven=False,
        needs_further_global_search=needs_more,
    )


def _budget_probe_tokens(
    evidence: Sequence[GlobalCompletionEvidence],
    obligations: Sequence[GlobalEvidenceObligation],
    tree_frontier: GlobalTreeFrontier,
) -> int:
    required = [row.obligation_id for row in obligations]
    # This intentionally includes each obligation in both sides of the
    # partition, making it a conservative upper envelope for the final compact
    # closure row.  Audit receipts stay outside the provider plane.
    payload = {
        "closure": {
            "compiled_operand_closure_proven": False,
            "covered_obligation_ids": required,
            "needs_further_global_search": True,
            "packing_closed": False,
            "support_closure_proven": False,
            "tree_search_closed": tree_frontier.search_closed,
            "unresolved_obligation_ids": required,
        },
        "evidence": [row.provider_row() for row in evidence],
        "format": f"{RESULT_FORMAT}-provider-plane",
        "instructions": (
            "Use exact evidence only. Preserve unresolved status; do not infer "
            "global completeness when support_closure_proven is false."
        ),
    }
    return count_tokens(_canonical_json(payload))


@dataclass(frozen=True, slots=True)
class SemanticGlobalCompletionResult:
    residual_index_receipt_sha256: str
    query_receipt_sha256: str
    request: SemanticGlobalCompletionRequest
    policy: SemanticGlobalCompletionPolicy
    obligations: tuple[GlobalEvidenceObligation, ...]
    tree_frontier: GlobalTreeFrontier
    hydrated_candidate_population_receipt_sha256: str
    hydrated_candidate_receipt_sha256s: tuple[str, ...]
    omitted_hydrated_segment_receipt_sha256s: tuple[str, ...]
    candidates: tuple[GlobalCompletionCandidate, ...]
    lane_receipts: tuple[GlobalLaneSelectionReceipt, ...]
    attempted_selection: tuple[GlobalSelectionAttempt, ...]
    protected_evidence_population_receipt_sha256: str
    protected_duplicates: tuple[GlobalProtectedDuplicate, ...]
    evidence: tuple[GlobalCompletionEvidence, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    closure: GlobalCompletionClosure
    packed_global_evidence_tokens: int
    provider_payload_tokens: int
    packed_global_evidence_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.residual_index_receipt_sha256, "global result index"),
            (self.query_receipt_sha256, "global result query"),
            (
                self.hydrated_candidate_population_receipt_sha256,
                "global hydrated population",
            ),
            (
                self.protected_evidence_population_receipt_sha256,
                "global protected population",
            ),
            (self.packed_global_evidence_sha256, "global evidence plane"),
        ):
            require_sha256(value, label)
        _require(
            type(self.request) is SemanticGlobalCompletionRequest
            and self.request.query_receipt_sha256 == self.query_receipt_sha256
            and self.request.routed,
            "global result request changed or was not routed",
        )
        _require(
            type(self.policy) is SemanticGlobalCompletionPolicy,
            "global result policy changed",
        )
        for values, expected, label in (
            (self.obligations, GlobalEvidenceObligation, "global obligations"),
            (self.candidates, GlobalCompletionCandidate, "global candidates"),
            (self.lane_receipts, GlobalLaneSelectionReceipt, "global lane receipts"),
            (self.attempted_selection, GlobalSelectionAttempt, "global attempts"),
            (
                self.protected_duplicates,
                GlobalProtectedDuplicate,
                "global protected duplicates",
            ),
            (self.evidence, GlobalCompletionEvidence, "global evidence"),
            (self.local_bindings, LocalCitationBinding, "global local bindings"),
        ):
            _require(
                type(values) is tuple and all(type(row) is expected for row in values),
                f"{label} changed type",
            )
        _require(
            type(self.tree_frontier) is GlobalTreeFrontier
            and type(self.closure) is GlobalCompletionClosure,
            "global frontier/closure changed type",
        )
        for values, label in (
            (
                self.hydrated_candidate_receipt_sha256s,
                "global hydrated candidate receipts",
            ),
            (
                self.omitted_hydrated_segment_receipt_sha256s,
                "global omitted hydrated segments",
            ),
        ):
            _ordered_unique(values, label)
        _require(
            self.hydrated_candidate_population_receipt_sha256
            == identity_sha256(
                {
                    "format": f"{CANDIDATE_FORMAT}-hydrated-population",
                    "ordered_candidate_receipt_sha256s": list(
                        self.hydrated_candidate_receipt_sha256s
                    ),
                }
            )
            and tuple(row.receipt_sha256 for row in self.candidates)
            == self.hydrated_candidate_receipt_sha256s[: len(self.candidates)],
            "global hydrated candidate population changed",
        )
        _require(
            tuple(row.lane_id for row in self.lane_receipts) == LANE_IDS,
            "global result lost independent lane order",
        )
        _require(
            tuple(row.selection_rank for row in self.attempted_selection)
            == tuple(range(len(self.attempted_selection)))
            and tuple(row.segment_receipt_sha256 for row in self.attempted_selection)
            == self.closure.selected_segment_receipt_sha256s,
            "global attempts lost selected-before-dedup order",
        )
        _require(
            tuple(row.segment_receipt_sha256 for row in self.evidence)
            == self.closure.packed_segment_receipt_sha256s
            and tuple(row.segment_receipt_sha256 for row in self.protected_duplicates)
            == self.closure.protected_duplicate_segment_receipt_sha256s,
            "global evidence/owners escaped closure partition",
        )
        _require(
            tuple(row.candidate_id for row in self.evidence)
            == tuple(row.candidate_id for row in self.local_bindings)
            and all(
                evidence.citation_binding_receipt_sha256 == binding.receipt_sha256
                and evidence.quote_sha256 == binding.quote_sha256
                and evidence.source_group_handle == binding.source_group_handle
                for evidence, binding in zip(
                    self.evidence, self.local_bindings, strict=True
                )
            ),
            "global provider evidence lost exact local citations",
        )
        _require(
            type(self.packed_global_evidence_tokens) is int
            and self.packed_global_evidence_tokens
            == sum(row.token_count for row in self.evidence)
            and type(self.provider_payload_tokens) is int
            and self.provider_payload_tokens
            == count_tokens(_canonical_json(self.provider_projection()))
            <= self.policy.global_payload_token_cap,
            "global G-plane token accounting changed",
        )
        _require(
            self.packed_global_evidence_sha256
            == identity_sha256([row.provider_row() for row in self.evidence]),
            "global packed evidence plane changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "global completion result receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_global_result")
        assert_gold_blind(
            self.provider_projection(), path="semantic_global_provider_plane"
        )

    def provider_projection(self) -> dict[str, object]:
        return _provider_payload(self.evidence, self.closure)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "attempted_selection": [row.projection() for row in self.attempted_selection],
            "candidates": [row.projection() for row in self.candidates],
            "closure": self.closure.projection(),
            "dedup_after_all_independent_lane_selection": True,
            "evidence": [row.projection() for row in self.evidence],
            "format": RESULT_FORMAT,
            "global_payload_budget_non_borrowable": True,
            "hydrated_candidate_population_receipt_sha256": (
                self.hydrated_candidate_population_receipt_sha256
            ),
            "hydrated_candidate_receipt_sha256s": list(
                self.hydrated_candidate_receipt_sha256s
            ),
            "lane_receipts": [row.projection() for row in self.lane_receipts],
            "local_bindings": [row.projection() for row in self.local_bindings],
            "new_provider_calls": 0,
            "obligations": [row.projection() for row in self.obligations],
            "omitted_hydrated_segment_receipt_sha256s": list(
                self.omitted_hydrated_segment_receipt_sha256s
            ),
            "packed_global_evidence_sha256": self.packed_global_evidence_sha256,
            "packed_global_evidence_tokens": self.packed_global_evidence_tokens,
            "policy": self.policy.projection(),
            "protected_duplicates": [
                row.projection() for row in self.protected_duplicates
            ],
            "protected_evidence_population_receipt_sha256": (
                self.protected_evidence_population_receipt_sha256
            ),
            "provider_payload_tokens": self.provider_payload_tokens,
            "query_receipt_sha256": self.query_receipt_sha256,
            "request": self.request.projection(),
            "residual_index_receipt_sha256": self.residual_index_receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "selected_before_em_and_protected_dedup": True,
            "tree_frontier": self.tree_frontier.projection(),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def search_semantic_global_completion(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    request: SemanticGlobalCompletionRequest,
    /,
    *,
    policy: SemanticGlobalCompletionPolicy = SemanticGlobalCompletionPolicy(),
    protected_evidence: Sequence[LocalCitationBinding] = (),
) -> SemanticGlobalCompletionResult:
    """Search the immutable global store and pack one independent G plane."""

    _require(
        type(residual_index) is SemanticResidualIndex
        and type(query) is SemanticResidualQuery
        and query.residual_index_receipt_sha256 == residual_index.receipt_sha256,
        "global search query escaped its immutable residual index",
    )
    _require(
        type(request) is SemanticGlobalCompletionRequest
        and request.query_receipt_sha256 == query.receipt_sha256
        and request.routed,
        "global search requires an exact routed generic request",
    )
    _require(type(policy) is SemanticGlobalCompletionPolicy, "global search policy changed")
    obligations = _compile_obligations(query, request, policy)
    signals = _cell_signals(residual_index, query, obligations)
    tree_frontier = _search_tree_best_first(
        residual_index,
        query,
        policy,
        signals,
        obligations,
    )
    candidates, all_hydrated_receipts, omitted_hydrated = _hydrate_candidates(
        residual_index,
        query,
        obligations,
        tree_frontier,
        policy,
    )
    hydrated_population_receipt = identity_sha256(
        {
            "format": f"{CANDIDATE_FORMAT}-hydrated-population",
            "ordered_candidate_receipt_sha256s": list(all_hydrated_receipts),
        }
    )
    lane_receipts, lane_union, owner_lanes = _select_lanes(
        policy,
        candidates,
        obligations,
    )
    selected, protected_obligation_ids = _protect_obligation_witness_order(
        lane_union,
        obligations,
    )
    protected_by_span, protected_population_receipt = _protected_inventory(
        residual_index,
        protected_evidence,
    )
    segment_by_receipt = {
        segment.receipt_sha256: segment
        for cell in residual_index.cells
        for segment in cell.segments
    }
    _require(
        len(segment_by_receipt)
        == sum(len(cell.segments) for cell in residual_index.cells),
        "global residual index repeated a segment receipt",
    )
    duplicates: list[GlobalProtectedDuplicate] = []
    duplicate_by_segment: dict[str, GlobalProtectedDuplicate] = {}
    packed_evidence: list[GlobalCompletionEvidence] = []
    packed_bindings: list[LocalCitationBinding] = []
    binding_by_segment: dict[str, LocalCitationBinding] = {}
    evidence_by_segment: dict[str, GlobalCompletionEvidence] = {}
    packed_ids: set[str] = set()
    for candidate in selected:
        lane_ids = owner_lanes[candidate.segment_receipt_sha256]
        owner = protected_by_span.get(candidate.span_identity_sha256)
        if owner is not None:
            duplicate = GlobalProtectedDuplicate(
                candidate_id=candidate.candidate_id,
                segment_receipt_sha256=candidate.segment_receipt_sha256,
                span_identity_sha256=candidate.span_identity_sha256,
                protected_candidate_id=owner.candidate_id,
                protected_binding_receipt_sha256=owner.receipt_sha256,
                selected_lane_ids=lane_ids,
                supported_obligation_ids=candidate.supported_obligation_ids,
            )
            duplicates.append(duplicate)
            duplicate_by_segment[candidate.segment_receipt_sha256] = duplicate
            continue
        segment = segment_by_receipt[candidate.segment_receipt_sha256]
        binding = _binding_for_candidate(residual_index, candidate, segment)
        evidence = _evidence_for_candidate(
            candidate,
            binding,
            lane_ids,
            set(protected_obligation_ids),
        )
        binding_by_segment[candidate.segment_receipt_sha256] = binding
        evidence_by_segment[candidate.segment_receipt_sha256] = evidence
        if (
            _budget_probe_tokens(
                (*packed_evidence, evidence),
                obligations,
                tree_frontier,
            )
            <= policy.global_payload_token_cap
        ):
            packed_evidence.append(evidence)
            packed_bindings.append(binding)
            packed_ids.add(candidate.segment_receipt_sha256)
        # Else skip and continue; later shorter candidates remain eligible.
    unpacked = tuple(
        row.segment_receipt_sha256
        for row in selected
        if row.segment_receipt_sha256 not in duplicate_by_segment
        and row.segment_receipt_sha256 not in packed_ids
    )
    closure = _make_closure(
        query,
        obligations,
        tree_frontier,
        omitted_hydrated,
        lane_receipts,
        selected,
        packed_evidence,
        duplicates,
        unpacked,
    )
    # The probe is conservative, but verify the exact final compact payload.
    # If tokenizer boundary effects defeat that envelope, remove the latest
    # packed row and recompute; this never blocks a later short row because the
    # initial loop already used skip/continue.
    while (
        count_tokens(_canonical_json(_provider_payload(packed_evidence, closure)))
        > policy.global_payload_token_cap
        and packed_evidence
    ):
        removed = packed_evidence.pop()
        packed_bindings.pop()
        packed_ids.remove(removed.segment_receipt_sha256)
        unpacked = tuple(
            row.segment_receipt_sha256
            for row in selected
            if row.segment_receipt_sha256 not in duplicate_by_segment
            and row.segment_receipt_sha256 not in packed_ids
        )
        closure = _make_closure(
            query,
            obligations,
            tree_frontier,
            omitted_hydrated,
            lane_receipts,
            selected,
            packed_evidence,
            duplicates,
            unpacked,
        )
    provider_payload_tokens = count_tokens(
        _canonical_json(_provider_payload(packed_evidence, closure))
    )
    _require(
        provider_payload_tokens <= policy.global_payload_token_cap,
        "fixed G-plane cap cannot fit its compact closure metadata",
    )
    attempts: list[GlobalSelectionAttempt] = []
    for selection_rank, candidate in enumerate(selected):
        segment_receipt = candidate.segment_receipt_sha256
        duplicate = duplicate_by_segment.get(segment_receipt)
        if duplicate is not None:
            disposition: Literal[
                "protected_exact_duplicate", "packed_novel", "budget_unpacked"
            ] = "protected_exact_duplicate"
            duplicate_receipt = duplicate.receipt_sha256
            evidence_receipt = None
            binding_receipt = None
        elif segment_receipt in packed_ids:
            disposition = "packed_novel"
            duplicate_receipt = None
            evidence_receipt = evidence_by_segment[segment_receipt].receipt_sha256
            binding_receipt = binding_by_segment[segment_receipt].receipt_sha256
        else:
            disposition = "budget_unpacked"
            duplicate_receipt = None
            evidence_receipt = None
            binding_receipt = binding_by_segment[segment_receipt].receipt_sha256
        attempts.append(
            GlobalSelectionAttempt(
                selection_rank=selection_rank,
                candidate_id=candidate.candidate_id,
                candidate_receipt_sha256=candidate.receipt_sha256,
                segment_receipt_sha256=segment_receipt,
                selected_lane_ids=owner_lanes[segment_receipt],
                supported_obligation_ids=candidate.supported_obligation_ids,
                disposition=disposition,
                protected_duplicate_receipt_sha256=duplicate_receipt,
                evidence_receipt_sha256=evidence_receipt,
                citation_binding_receipt_sha256=binding_receipt,
            )
        )
    result = SemanticGlobalCompletionResult(
        residual_index_receipt_sha256=residual_index.receipt_sha256,
        query_receipt_sha256=query.receipt_sha256,
        request=request,
        policy=policy,
        obligations=obligations,
        tree_frontier=tree_frontier,
        hydrated_candidate_population_receipt_sha256=hydrated_population_receipt,
        hydrated_candidate_receipt_sha256s=all_hydrated_receipts,
        omitted_hydrated_segment_receipt_sha256s=omitted_hydrated,
        candidates=candidates,
        lane_receipts=lane_receipts,
        attempted_selection=tuple(attempts),
        protected_evidence_population_receipt_sha256=protected_population_receipt,
        protected_duplicates=tuple(duplicates),
        evidence=tuple(packed_evidence),
        local_bindings=tuple(packed_bindings),
        closure=closure,
        packed_global_evidence_tokens=sum(row.token_count for row in packed_evidence),
        provider_payload_tokens=provider_payload_tokens,
        packed_global_evidence_sha256=identity_sha256(
            [row.provider_row() for row in packed_evidence]
        ),
    )
    return result


def validate_semantic_global_completion(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    request: SemanticGlobalCompletionRequest,
    result: SemanticGlobalCompletionResult,
    /,
    *,
    protected_evidence: Sequence[LocalCitationBinding] = (),
) -> SemanticGlobalCompletionResult:
    """Return one fresh replay after requiring an exact sealed result."""

    _require(
        type(result) is SemanticGlobalCompletionResult,
        "global completion result changed type",
    )
    replayed = search_semantic_global_completion(
        residual_index,
        query,
        request,
        policy=result.policy,
        protected_evidence=protected_evidence,
    )
    _require(
        replayed.receipt_sha256 == result.receipt_sha256
        and replayed.projection() == result.projection(),
        "semantic global completion replay differs from sealed result",
    )
    return replayed


def replay_semantic_global_completion(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    request: SemanticGlobalCompletionRequest,
    sealed_result: SemanticGlobalCompletionResult,
    /,
    *,
    protected_evidence: Sequence[LocalCitationBinding] = (),
) -> SemanticGlobalCompletionResult:
    return validate_semantic_global_completion(
        residual_index,
        query,
        request,
        sealed_result,
        protected_evidence=protected_evidence,
    )


def validate_semantic_global_completion_projection(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    request: SemanticGlobalCompletionRequest,
    projection: Mapping[str, object],
    /,
    *,
    policy: SemanticGlobalCompletionPolicy = SemanticGlobalCompletionPolicy(),
    protected_evidence: Sequence[LocalCitationBinding] = (),
) -> SemanticGlobalCompletionResult:
    """Strict projection loader: reconstruct from authority, then compare."""

    _require(type(projection) is dict, "global completion projection changed schema")
    replayed = search_semantic_global_completion(
        residual_index,
        query,
        request,
        policy=policy,
        protected_evidence=protected_evidence,
    )
    _require(
        replayed.projection() == projection,
        "stored global completion projection differs from deterministic replay",
    )
    return replayed


load_semantic_global_completion_projection = (
    validate_semantic_global_completion_projection
)


__all__ = [
    "DEFAULT_GLOBAL_PAYLOAD_TOKEN_CAP",
    "GlobalCompletionCandidate",
    "GlobalCompletionClosure",
    "GlobalCompletionEvidence",
    "GlobalEvidenceObligation",
    "GlobalLaneBudget",
    "GlobalLaneSelectionReceipt",
    "GlobalProtectedDuplicate",
    "GlobalSelectionAttempt",
    "GlobalTreeFrontier",
    "GlobalTreeVisit",
    "MECHANISM_ID",
    "SemanticGlobalCompletionError",
    "SemanticGlobalCompletionPolicy",
    "SemanticGlobalCompletionRequest",
    "SemanticGlobalCompletionResult",
    "compile_semantic_global_completion_request",
    "linked_event_dates",
    "load_semantic_global_completion_projection",
    "replay_semantic_global_completion",
    "search_semantic_global_completion",
    "validate_semantic_global_completion",
    "validate_semantic_global_completion_projection",
]
