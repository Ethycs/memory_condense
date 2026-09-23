"""Provider-free typed witness retrieval over a resident full-store index.

The base :class:`FullStoreWindowIndex` amortizes the physical store scan.  This
module adds a second ingest-time compilation pass that splits those sentence
windows into clauses and freezes postings for normalized terms, evidence role,
event day, source, and completed/planned action concepts.  Query ticks then do
set operations over resident tuples; they do not rescan SQLite, invoke a model,
or retain transformer token state.

Selection owns an independent evidence budget.  It deliberately happens
*before* cross-arm/protected-evidence deduplication.  Overlap is removed only
from that frozen selection and is not backfilled, preserving the causal assay
contract used by the other memory arms.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from types import MappingProxyType
from typing import Any, Literal

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, make_atom_id, quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import (
    FullStoreWindowIndex,
    QuestionTemporalTarget,
    TemporalTargetMode,
    _event_date,
    _question_body,
    _question_terms,
    _slot_match,
    _temporal_fit,
    _temporal_target,
    indexed_surface_terms,
)
from .query_guided_scan import CachedContentRow
from .typed_action_semantics import (
    completed_action_concepts,
    linked_action_concepts,
    planned_action_concepts,
)
from .typed_operator_spec import (
    AnswerShape,
    TemporalMode,
    TypedOperatorSpec,
    canonicalize_question_text,
    compile_typed_operator_spec,
)


MECHANISM_ID = "hot_full_store_typed_witness_overlay_v3"
INDEX_FORMAT = "memory-condense-hot-typed-witness-index-v3"
WINDOW_FORMAT = "memory-condense-hot-typed-witness-window-v3"
WITNESS_FORMAT = "memory-condense-hot-typed-witness-v3"
RECEIPT_FORMAT = "memory-condense-hot-typed-witness-receipt-v3"
ESCALATION_FORMAT = "memory-condense-hot-typed-witness-escalation-v1"
DEFAULT_EVIDENCE_TOKEN_CAP = 1_200


class HotTypedWitnessError(MatchedEvalContractError):
    """Raised when a hot typed-witness invariant is violated."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise HotTypedWitnessError(message)


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    _require(
        all(type(value) is str and value and value.strip() == value for value in result),
        f"{label} must contain exact non-empty text",
    )
    _require(len(result) == len(set(result)), f"{label} must be ordered and unique")
    return result


@dataclass(frozen=True, slots=True)
class HotTypedWitnessBudget:
    """Budget owned by this overlay, before cross-arm deduplication."""

    evidence_token_cap: int = DEFAULT_EVIDENCE_TOKEN_CAP
    max_candidates: int = 24
    max_candidates_per_source: int = 6
    max_witness_tokens: int = 192
    candidates_per_action: int = 4

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            _require(type(value) is int and value > 0, f"{name} must be positive")
        _require(
            self.max_candidates_per_source <= self.max_candidates,
            "per-source cap exceeds candidate cap",
        )

    def projection(self) -> dict[str, int | str]:
        return {
            **asdict(self),
            "budget_scope": "independent_selection_before_protected_dedup",
        }

    @property
    def budget_id(self) -> str:
        return identity_sha256(
            {"mechanism_id": MECHANISM_ID, "budget": self.projection()}
        )


_CLAUSE_BOUNDARY_RE = re.compile(
    r";+|[\r\n]+|,\s*(?=(?:but|while|although|however)\b)", re.IGNORECASE
)
_PLANNING_CUE_RE = re.compile(
    r"\b(?:i|we)\s+(?:am\s+|are\s+|was\s+|were\s+)?"
    r"(?:plan(?:ned|ning)?(?:\s+to)?|intend(?:ed|ing)?(?:\s+to)?|"
    r"consider(?:ed|ing)?|hope(?:d|ing)?\s+to|(?:need|want|expect)\s+to|"
    r"going\s+to)"
    r"\b",
    re.IGNORECASE,
)
_PARTICIPATE_QUERY_RE = re.compile(
    r"\b(?:participat\w*|compet\w*)\b|"
    r"\b(?:take|took)\s+(?:part|place)\b|"
    r"\b(?:athletic|sporting)\s+events?\b",
    re.IGNORECASE,
)
_PARTICIPATE_COMPLETED_RE = re.compile(
    r"\b(?:participated|competed|took\s+part|joined)\b|"
    r"\b(?:ran|completed|finished)\b[^.!?;\r\n]{0,100}"
    r"\b(?:triathlon|5k|10k|run|race|marathon|tournament|meet|event)\b",
    re.IGNORECASE,
)
_TRAVEL_QUERY_RE = re.compile(
    r"\b(?:travel\w*|trips?|journeys?|countries|cities|destinations|"
    r"fly|flight|flew|went\s+to|hik\w*|camp\w*|road\s+trip)\b",
    re.IGNORECASE,
)
_TRAVEL_EVENT_NOUN = (
    r"(?:trips?|journeys?|hikes?|hiking|camp(?:ing)?|vacations?|tours?|"
    r"road\s+trips?)"
)
_TRAVEL_COMPLETED_RE = re.compile(
    rf"\b(?:got\s+back|returned)\s+from\b[^.!?;\r\n]{{0,80}}"
    rf"\b{_TRAVEL_EVENT_NOUN}\b|"
    rf"\b(?:went|took|started|completed|finished)\b"
    rf"[^.!?;\r\n]{{0,80}}\b{_TRAVEL_EVENT_NOUN}\b|"
    r"\b(?:traveled|travelled|flew|hiked|camped)\b",
    re.IGNORECASE,
)
_MILESTONE_QUERY_RE = re.compile(
    r"\b(?:business|company|shop|store|restaurant|studio|practice)\b"
    r"[^?]{0,120}\b(?:open\w*|launch\w*|start\w*|found\w*|"
    r"establish\w*|register\w*|milestone|begin|began)\b|"
    r"\b(?:open\w*|launch\w*|found\w*|establish\w*|milestone)\b"
    r"[^?]{0,120}\b(?:business|company|shop|store|restaurant|studio|practice)\b|"
    r"\b(?:sign\w*[^?]{0,60}contract|launch\w*[^?]{0,60}website)\b",
    re.IGNORECASE,
)
_MILESTONE_COMPLETED_RE = re.compile(
    r"\b(?:opened|launched|started|founded|established|incorporated|"
    r"registered|began\s+operations|made\s+(?:my|our|the)\s+first\s+sale)\b|"
    r"\bsigned\b[^.!?;\r\n]{0,60}\bcontract\b",
    re.IGNORECASE,
)
_STRICT_BUSINESS_MILESTONE_RE = re.compile(
    r"\b(?:first\s+(?:client|customer|sale|contract|employee)|"
    r"signed\s+(?:(?:a|the|my|our)\s+)?(?:company\s+)?contract|"
    r"launched\s+(?:(?:my|our|the|a)\s+)?"
    r"(?:website|business|company|product|service|store|shop)|"
    r"opened\s+(?:(?:my|our|the|a)\s+)?(?:business|company|store|shop)|"
    r"made\s+(?:my|our|the)\s+first\s+sale)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_BUSINESS_MILESTONE_RE = re.compile(
    r"\b(?:i|we)\s+"
    r"(?:(?:have|had)\s+)?"
    r"(?:(?:just|recently|finally|successfully|officially)\s+)?"
    r"(?:signed|launched|opened|started|founded|established|incorporated|"
    r"registered|made|began)\b|"
    r"\b(?:my|our)\s+(?:business|company|website|product|service|store|shop)\s+"
    r"(?:(?:has|had)\s+)?"
    r"(?:opened|launched|started|began|made)\b",
    re.IGNORECASE,
)
_RELATIVE_BUSINESS_DAY_RADIUS = 2
_RELATIVE_BUSINESS_PREDECESSOR_DAYS = 31
_GENERIC_COMPLETED_QUERY_RE = re.compile(
    r"\b(?:what|which|how\s+many)\b[^?]{0,100}"
    r"\b(?:did\s+i|did\s+we|have\s+i|have\s+we|completed|finished|done)\b",
    re.IGNORECASE,
)
_LATEST_QUERY_RE = re.compile(
    r"\b(?:last|latest|most\s+recent|most\s+recently)\b", re.IGNORECASE
)
_FIRST_PERSON_MEMORY_RE = re.compile(r"\b(?:i|me|my|we|us|our)\b", re.IGNORECASE)
_ASSISTANT_HISTORY_RE = re.compile(
    r"\b(?:you|assistant)\b[^?]{0,60}\b(?:said|told|recommended|suggested|"
    r"listed|reminded|answered|mentioned)\b",
    re.IGNORECASE,
)
_PARTICIPANT_TERMS = frozenset(
    {
        "friend",
        "family",
        "colleague",
        "coworker",
        "partner",
        "sister",
        "brother",
        "mother",
        "father",
        "mom",
        "dad",
        "wife",
        "husband",
    }
)
_CATEGORY_TERM_EXPANSIONS = (
    (
        re.compile(r"\bfurniture\b", re.IGNORECASE),
        "chair desk table cabinet sofa couch dresser shelf bookcase bed",
    ),
    (
        re.compile(r"\b(?:sporting|athletic)\s+events?\b", re.IGNORECASE),
        "triathlon 5k 10k run race marathon tournament soccer meet",
    ),
    (
        re.compile(r"\b(?:museum|museums|gallery|galleries)\b", re.IGNORECASE),
        "museum gallery exhibit exhibition",
    ),
    (
        re.compile(r"\b(?:business|company)\s+milestones?\b", re.IGNORECASE),
        "contract website launch open found establish register",
    ),
)


def _extra_action_concepts(text: str, *, completed: bool) -> tuple[str, ...]:
    patterns = (
        ("participate", _PARTICIPATE_COMPLETED_RE if completed else _PARTICIPATE_QUERY_RE),
        ("travel", _TRAVEL_COMPLETED_RE if completed else _TRAVEL_QUERY_RE),
        (
            "business_milestone",
            _MILESTONE_COMPLETED_RE if completed else _MILESTONE_QUERY_RE,
        ),
    )
    return tuple(name for name, pattern in patterns if pattern.search(text))


def _query_action_concepts(text: str) -> tuple[str, ...]:
    return tuple(
        sorted({*linked_action_concepts(text), *_extra_action_concepts(text, completed=False)})
    )


def _completed_concepts(text: str) -> tuple[str, ...]:
    return tuple(
        sorted({*completed_action_concepts(text), *_extra_action_concepts(text, completed=True)})
    )


def _planned_concepts(text: str) -> tuple[str, ...]:
    result = set(planned_action_concepts(text))
    if _PLANNING_CUE_RE.search(text):
        result.update(_extra_action_concepts(text, completed=False))
    return tuple(sorted(result))


def _expanded_question_terms(body: str) -> tuple[str, ...]:
    terms = list(_question_terms(body))
    for pattern, expansion in _CATEGORY_TERM_EXPANSIONS:
        if pattern.search(body):
            terms.extend(indexed_surface_terms(expansion))
    return tuple(dict.fromkeys(terms))


def _clause_coordinates(text: str, start: int, end: int) -> tuple[tuple[int, int], ...]:
    local = text[start:end]
    cursor = 0
    values: list[tuple[int, int]] = []
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(local):
        values.append((cursor, boundary.start()))
        cursor = boundary.end()
    values.append((cursor, len(local)))
    output: list[tuple[int, int]] = []
    for local_start, local_end in values:
        while local_start < local_end and local[local_start].isspace():
            local_start += 1
        while local_end > local_start and local[local_end - 1].isspace():
            local_end -= 1
        if local_start < local_end:
            output.append((start + local_start, start + local_end))
    return tuple(output) or ((start, end),)


@dataclass(frozen=True, slots=True)
class IndexedTypedWitnessWindow:
    """Question-neutral clause backed by an exact resident content row."""

    parent_window_index: int
    clause_index: int
    row: CachedContentRow
    start_char: int
    end_char: int
    quote_sha256: str
    token_count: int
    terms: frozenset[str]
    completed_actions: tuple[str, ...]
    planned_actions: tuple[str, ...]
    event_date: str | None
    event_date_basis: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.parent_window_index) is int
            and type(self.clause_index) is int
            and self.parent_window_index >= 0
            and self.clause_index >= 0,
            "typed witness coordinates changed",
        )
        _require(type(self.row) is CachedContentRow, "typed witness row changed")
        _require(
            0 <= self.start_char < self.end_char <= len(self.row.text),
            "typed witness span changed",
        )
        quote = self.row.text[self.start_char : self.end_char]
        require_sha256(self.quote_sha256, "typed witness quote")
        _require(
            self.quote_sha256 == quote_sha256(quote)
            and self.token_count == count_tokens(quote),
            "typed witness quote bytes changed",
        )
        _require(
            type(self.terms) is frozenset
            and all(type(term) is str and term for term in self.terms),
            "typed witness terms changed",
        )
        _ordered_unique(self.completed_actions, "completed actions")
        _ordered_unique(self.planned_actions, "planned actions")
        if self.event_date is None:
            _require(self.event_date_basis is None, "undated witness gained a date basis")
        else:
            require_text(self.event_date, "typed witness event day")
            require_text(self.event_date_basis or "", "typed witness event-day basis")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "typed witness window changed")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def quote(self) -> str:
        return self.row.text[self.start_char : self.end_char]

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "clause_index": self.clause_index,
            "completed_actions": list(self.completed_actions),
            "end_char": self.end_char,
            "event_date": self.event_date,
            "event_date_basis": self.event_date_basis,
            "format": WINDOW_FORMAT,
            "parent_window_index": self.parent_window_index,
            "quote_sha256": self.quote_sha256,
            "row_receipt": self.row.receipt_projection(),
            "start_char": self.start_char,
            "terms": sorted(self.terms),
            "token_count": self.token_count,
            "planned_actions": list(self.planned_actions),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotTypedWitnessIndex:
    """Immutable action/term/date/source postings over typed clauses."""

    parent: FullStoreWindowIndex
    windows: tuple[IndexedTypedWitnessWindow, ...]
    term_postings: Mapping[str, tuple[int, ...]]
    role_postings: Mapping[str, tuple[int, ...]]
    date_postings: Mapping[str, tuple[int, ...]]
    source_postings: Mapping[str, tuple[int, ...]]
    completed_action_postings: Mapping[str, tuple[int, ...]]
    planned_action_postings: Mapping[str, tuple[int, ...]]
    posting_inventory_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.parent) is FullStoreWindowIndex, "parent index changed")
        _require(
            type(self.windows) is tuple
            and all(type(row) is IndexedTypedWitnessWindow for row in self.windows),
            "typed witness inventory changed",
        )
        for postings, label in (
            (self.term_postings, "term postings"),
            (self.role_postings, "role postings"),
            (self.date_postings, "date postings"),
            (self.source_postings, "source postings"),
            (self.completed_action_postings, "completed-action postings"),
            (self.planned_action_postings, "planned-action postings"),
        ):
            _require(isinstance(postings, Mapping), f"{label} changed type")
            for key, indices in postings.items():
                require_text(key, label)
                _require(
                    type(indices) is tuple
                    and tuple(sorted(set(indices))) == indices
                    and all(0 <= index < len(self.windows) for index in indices),
                    f"{label} contains an invalid index",
                )
        require_sha256(self.posting_inventory_sha256, "posting inventory")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "typed witness index changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_typed_witness_index")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "all_parent_windows_compiled": True,
            "clause_window_count": len(self.windows),
            "completed_action_bucket_count": len(self.completed_action_postings),
            "date_bucket_count": len(self.date_postings),
            "format": INDEX_FORMAT,
            "gold_loaded": False,
            "model_calls": 0,
            "new_provider_calls": 0,
            "parent_window_index_receipt_sha256": self.parent.receipt_sha256,
            "planned_action_bucket_count": len(self.planned_action_postings),
            "posting_inventory_sha256": self.posting_inventory_sha256,
            "query_tick_full_physical_rescan": False,
            "retained_transformer_token_state_bytes": 0,
            "role_bucket_count": len(self.role_postings),
            "source_bucket_count": len(self.source_postings),
            "term_vocabulary_size": len(self.term_postings),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _freeze_postings(values: Mapping[str, list[int]]) -> Mapping[str, tuple[int, ...]]:
    return MappingProxyType(
        {key: tuple(indices) for key, indices in sorted(values.items())}
    )


def build_hot_typed_witness_index(
    parent: FullStoreWindowIndex, /
) -> HotTypedWitnessIndex:
    """Compile question-neutral typed clause postings once at ingestion."""

    _require(type(parent) is FullStoreWindowIndex, "hot overlay requires exact parent")
    windows: list[IndexedTypedWitnessWindow] = []
    postings: dict[str, dict[str, list[int]]] = {
        "terms": {},
        "roles": {},
        "dates": {},
        "sources": {},
        "completed": {},
        "planned": {},
    }
    for parent_index, base in enumerate(parent.windows):
        for clause_index, (start, end) in enumerate(
            _clause_coordinates(base.row.text, base.start_char, base.end_char)
        ):
            quote = base.row.text[start:end]
            event, basis = _event_date(quote, base.row)
            window = IndexedTypedWitnessWindow(
                parent_window_index=parent_index,
                clause_index=clause_index,
                row=base.row,
                start_char=start,
                end_char=end,
                quote_sha256=quote_sha256(quote),
                token_count=count_tokens(quote),
                terms=frozenset(indexed_surface_terms(quote)),
                completed_actions=_completed_concepts(quote),
                planned_actions=_planned_concepts(quote),
                event_date=event.isoformat() if event is not None else None,
                event_date_basis=basis,
            )
            index = len(windows)
            windows.append(window)
            for term in sorted(window.terms):
                postings["terms"].setdefault(term, []).append(index)
            postings["roles"].setdefault(base.row.role, []).append(index)
            postings["sources"].setdefault(base.row.source_id, []).append(index)
            if window.event_date is not None:
                postings["dates"].setdefault(window.event_date, []).append(index)
            for action in window.completed_actions:
                postings["completed"].setdefault(action, []).append(index)
            for action in window.planned_actions:
                postings["planned"].setdefault(action, []).append(index)
    frozen = {name: _freeze_postings(value) for name, value in postings.items()}
    inventory = {
        "completed": {key: list(value) for key, value in frozen["completed"].items()},
        "dates": {key: list(value) for key, value in frozen["dates"].items()},
        "planned": {key: list(value) for key, value in frozen["planned"].items()},
        "roles": {key: list(value) for key, value in frozen["roles"].items()},
        "sources": {key: list(value) for key, value in frozen["sources"].items()},
        "terms": {key: list(value) for key, value in frozen["terms"].items()},
        "window_receipt_sha256s": [row.receipt_sha256 for row in windows],
    }
    return HotTypedWitnessIndex(
        parent=parent,
        windows=tuple(windows),
        term_postings=frozen["terms"],
        role_postings=frozen["roles"],
        date_postings=frozen["dates"],
        source_postings=frozen["sources"],
        completed_action_postings=frozen["completed"],
        planned_action_postings=frozen["planned"],
        posting_inventory_sha256=identity_sha256(inventory),
    )


@dataclass(frozen=True, slots=True)
class _Draft:
    window_index: int
    completed_matches: tuple[str, ...]
    planned_matches: tuple[str, ...]
    supported_slot_ids: tuple[str, ...]
    matched_query_terms: tuple[str, ...]
    matched_participant_terms: tuple[str, ...]
    temporal_distance_days: int | None
    exact_temporal_match: bool
    within_temporal_window: bool
    lexical_score: float
    recency_key: int
    candidate_id: str
    temporal_frontier_priority: int = 0

    def score(
        self, temporal_mode: TemporalMode, *, latest_requested: bool
    ) -> tuple[Any, ...]:
        latest = (
            self.recency_key
            if temporal_mode is TemporalMode.LATEST_STATE or latest_requested
            else 0
        )
        return (
            self.temporal_frontier_priority,
            int(self.exact_temporal_match),
            int(self.within_temporal_window),
            len(self.completed_matches),
            len(self.matched_participant_terms),
            len(self.supported_slot_ids),
            len(self.planned_matches),
            self.lexical_score,
            len(self.matched_query_terms),
            latest,
            -(self.temporal_distance_days or 0),
            self.candidate_id,
        )


def _span(window: IndexedTypedWitnessWindow) -> EvidenceSpan:
    row = window.row
    return EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=window.start_char,
        end_char=window.end_char,
        quote_sha256=window.quote_sha256,
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )


def _candidate_id(window: IndexedTypedWitnessWindow) -> str:
    return identity_sha256(
        {"atom_id": make_atom_id(_span(window)), "mechanism_id": MECHANISM_ID}
    )


def _candidate_indices(
    index: HotTypedWitnessIndex,
    *,
    query_actions: Sequence[str],
    query_terms: Sequence[str],
    target: QuestionTemporalTarget,
    include_proposed: bool,
    generic_completed: bool,
) -> tuple[set[int], str]:
    action_values: set[int] = set()
    action_bucket_coverage = True
    for action in query_actions:
        values = set(index.completed_action_postings.get(action, ()))
        if include_proposed:
            values.update(index.planned_action_postings.get(action, ()))
        action_bucket_coverage = action_bucket_coverage and bool(values)
        action_values.update(values)
    if generic_completed and not query_actions:
        for values in index.completed_action_postings.values():
            action_values.update(values)

    if (
        target.mode is TemporalTargetMode.EXACT_DAY
        and target.target_date is not None
        and target.derivation == "unambiguous_relative_offset_expression"
        and tuple(query_actions) == ("business_milestone",)
    ):
        wanted = date.fromisoformat(target.target_date)
        user = set(index.role_postings.get("user", ()))
        milestone = set(index.completed_action_postings.get("business_milestone", ()))

        def proven(position: int) -> bool:
            quote = index.windows[position].quote
            return bool(
                _FIRST_PERSON_BUSINESS_MILESTONE_RE.search(quote)
                and _STRICT_BUSINESS_MILESTONE_RE.search(quote)
            )

        near: set[int] = set()
        for offset in range(-_RELATIVE_BUSINESS_DAY_RADIUS, _RELATIVE_BUSINESS_DAY_RADIUS + 1):
            near.update(
                index.date_postings.get((wanted + timedelta(days=offset)).isoformat(), ())
            )
        near = {position for position in near & user & milestone if proven(position)}
        if near:
            def winner_key(position: int) -> tuple[Any, ...]:
                window = index.windows[position]
                event = date.fromisoformat(window.event_date or "")
                return (
                    -abs((event - wanted).days),
                    len(_STRICT_BUSINESS_MILESTONE_RE.findall(window.quote)),
                    event.toordinal(),
                    -window.row.ordinal,
                    -position,
                )

            winner = max(near, key=winner_key)
            winner_day = date.fromisoformat(index.windows[winner].event_date or "")
            earliest = winner_day - timedelta(days=_RELATIVE_BUSINESS_PREDECESSOR_DAYS)
            predecessors = {
                position
                for position in milestone & user
                if position != winner
                and index.windows[position].event_date is not None
                and earliest <= date.fromisoformat(index.windows[position].event_date or "") < winner_day
                and proven(position)
            }
            frontier = {winner}
            if predecessors:
                frontier.add(
                    max(
                        predecessors,
                        key=lambda position: (
                            index.windows[position].event_date or "",
                            len(_STRICT_BUSINESS_MILESTONE_RE.findall(index.windows[position].quote)),
                            -index.windows[position].row.ordinal,
                            -position,
                        ),
                    )
                )
            return frontier, "relative_business_milestone_frontier"

    temporal_values: set[int] = set()
    if target.mode is TemporalTargetMode.EXACT_DAY and target.target_date is not None:
        temporal_values.update(index.date_postings.get(target.target_date, ()))
    elif target.mode is TemporalTargetMode.LOOKBACK_WINDOW:
        asked = date.fromisoformat((target.asked_at or "")[:10])
        earliest = asked.fromordinal(asked.toordinal() - (target.lookback_days or 0))
        for day, values in index.date_postings.items():
            if earliest <= date.fromisoformat(day) <= asked:
                temporal_values.update(values)

    if action_values and action_bucket_coverage:
        if temporal_values:
            constrained = action_values & temporal_values
            if constrained:
                return constrained, "action_temporal_intersection"
            return action_values | temporal_values, "action_temporal_fail_open_union"
        return action_values, "action_postings"

    result = set(action_values)
    for term in query_terms:
        result.update(index.term_postings.get(term, ()))
    result.update(temporal_values)
    return (
        result,
        "action_partial_fail_open_union" if action_values else "term_temporal_fallback",
    )


def _effective_evidence_role(
    spec: TypedOperatorSpec,
    body: str,
    *,
    has_action_constraint: bool,
) -> tuple[str | None, str]:
    if spec.required_evidence_role is not None:
        return spec.required_evidence_role, "typed_operator_spec"
    if (
        has_action_constraint
        and _FIRST_PERSON_MEMORY_RE.search(body)
        and not _ASSISTANT_HISTORY_RE.search(body)
    ):
        return "user", "first_person_completed_action_question"
    return None, "unconstrained"


def _eligible_indices(
    index: HotTypedWitnessIndex,
    eligible_source_ids: Collection[str] | None,
) -> tuple[set[int] | None, tuple[str, ...], int]:
    if eligible_source_ids is None:
        return None, (), len(index.source_postings)
    _require(
        not isinstance(eligible_source_ids, (str, bytes)),
        "eligible source IDs must be a collection, not text",
    )
    requested = tuple(sorted(set(eligible_source_ids)))
    _ordered_unique(requested, "eligible source IDs")
    allowed: set[int] = set()
    matched = 0
    for source_id in requested:
        values = index.source_postings.get(source_id)
        if values is not None:
            matched += 1
            allowed.update(values)
    return allowed, requested, matched


def _drafts(
    index: HotTypedWitnessIndex,
    dated_question: str,
    spec: TypedOperatorSpec,
    target: QuestionTemporalTarget,
    eligible_source_ids: Collection[str] | None,
) -> tuple[
    tuple[_Draft, ...],
    tuple[str, ...],
    int,
    int,
    tuple[str, ...],
    bool,
    bool,
    str,
    str | None,
    str,
]:
    body = _question_body(canonicalize_question_text(dated_question))
    query_actions = _query_action_concepts(body)
    action_constraints = query_actions
    query_terms = _expanded_question_terms(body)
    participant_terms = tuple(
        term for term in query_terms if term in _PARTICIPANT_TERMS
    )
    generic_completed = bool(
        _GENERIC_COMPLETED_QUERY_RE.search(body) and not spec.include_proposed
    )
    latest_requested = bool(_LATEST_QUERY_RE.search(body))
    if generic_completed and set(query_actions) <= {"complete"}:
        action_constraints = ()
    effective_role, role_derivation = _effective_evidence_role(
        spec,
        body,
        has_action_constraint=bool(action_constraints or generic_completed),
    )
    eligible, requested_sources, matched_source_count = _eligible_indices(
        index, eligible_source_ids
    )
    applicable = bool(
        query_actions
        or generic_completed
        or latest_requested
        or target.mode is not TemporalTargetMode.NONE
        or spec.temporal_mode is not TemporalMode.NONE
    )
    if not applicable:
        return (
            (),
            requested_sources,
            matched_source_count,
            0,
            query_actions,
            generic_completed,
            latest_requested,
            "not_applicable",
            effective_role,
            role_derivation,
        )
    candidate_indices, candidate_strategy = _candidate_indices(
        index,
        query_actions=action_constraints,
        query_terms=query_terms,
        target=target,
        include_proposed=spec.include_proposed,
        generic_completed=generic_completed,
    )
    if eligible is not None:
        candidate_indices.intersection_update(eligible)
    role_rejected = 0
    if effective_role is not None:
        accepted_role = set(index.role_postings.get(effective_role, ()))
        role_rejected = len(candidate_indices - accepted_role)
        candidate_indices.intersection_update(accepted_role)
    window_count = max(len(index.windows), 1)
    document_frequency = {
        term: len(index.term_postings.get(term, ())) for term in query_terms
    }
    drafts: list[_Draft] = []
    for window_index in sorted(candidate_indices):
        window = index.windows[window_index]
        if effective_role not in {None, window.row.role}:
            raise HotTypedWitnessError("role postings admitted a mismatched witness")
        completed = tuple(
            action
            for action in action_constraints
            if action in window.completed_actions
        )
        planned = tuple(
            action for action in action_constraints if action in window.planned_actions
        )
        if generic_completed and not action_constraints:
            completed = window.completed_actions
        # A planned-only clause cannot enter a completed-event query by sharing
        # nouns or a date.  Clause splitting prevents a proposal elsewhere in a
        # sentence from relabeling a completed witness.
        if planned and not completed and not spec.include_proposed:
            continue
        matched = tuple(term for term in query_terms if term in window.terms)
        participants = tuple(term for term in participant_terms if term in window.terms)
        supported = tuple(
            slot.slot_id
            for slot in spec.required_slots
            if _slot_match(slot, window.terms, bool(re.search(r"\d", window.quote)))
        )
        event = date.fromisoformat(window.event_date) if window.event_date else None
        distance, exact, within = _temporal_fit(event, target)
        frontier_priority = 0
        if candidate_strategy == "relative_business_milestone_frontier" and event:
            wanted = date.fromisoformat(target.target_date or "")
            frontier_priority = 2 if abs((event - wanted).days) <= _RELATIVE_BUSINESS_DAY_RADIUS else 1
        accepted_action = bool(completed or (spec.include_proposed and planned))
        relevant = bool(accepted_action or supported or matched or exact or within)
        if action_constraints and planned and not completed and not spec.include_proposed:
            relevant = False
        if not relevant:
            continue
        lexical = sum(
            math.log((window_count + 1) / (document_frequency[term] + 1)) + 1.0
            for term in matched
        )
        event_ordinal = event.toordinal() if event is not None else -1
        drafts.append(
            _Draft(
                window_index=window_index,
                completed_matches=completed,
                planned_matches=planned if spec.include_proposed else (),
                supported_slot_ids=supported,
                matched_query_terms=matched,
                matched_participant_terms=participants,
                temporal_distance_days=distance,
                exact_temporal_match=exact,
                within_temporal_window=within,
                lexical_score=round(lexical, 8),
                recency_key=event_ordinal,
                candidate_id=_candidate_id(window),
                temporal_frontier_priority=frontier_priority,
            )
        )
    ranked = tuple(
        sorted(
            drafts,
            key=lambda row: row.score(
                spec.temporal_mode, latest_requested=latest_requested
            ),
            reverse=True,
        )
    )
    return (
        ranked,
        requested_sources,
        matched_source_count,
        role_rejected,
        query_actions,
        generic_completed,
        latest_requested,
        candidate_strategy,
        effective_role,
        role_derivation,
    )


@dataclass(frozen=True, slots=True)
class HotTypedWitness:
    candidate_id: str
    parent_index_receipt_sha256: str
    indexed_window_receipt_sha256: str
    cache_receipt_sha256: str
    source_database_sha256: str
    source_store_receipt_sha256: str
    partition_id: str
    source_id: str
    span: EvidenceSpan
    quote: str
    quote_sha256: str
    token_count: int
    completed_action_concepts: tuple[str, ...]
    planned_action_concepts: tuple[str, ...]
    supported_slot_ids: tuple[str, ...]
    matched_query_terms: tuple[str, ...]
    matched_participant_terms: tuple[str, ...]
    event_date: str | None
    event_date_basis: str | None
    temporal_distance_days: int | None
    selection_axes: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "candidate"),
            (self.parent_index_receipt_sha256, "parent index"),
            (self.indexed_window_receipt_sha256, "indexed window"),
            (self.cache_receipt_sha256, "cache"),
            (self.source_database_sha256, "source database"),
            (self.source_store_receipt_sha256, "source store"),
            (self.quote_sha256, "quote"),
        ):
            require_sha256(value, label)
        require_text(self.partition_id, "partition ID")
        require_text(self.source_id, "source ID")
        require_text(self.quote, "witness quote")
        _require(
            self.span.source_id == self.source_id
            and self.span.chunk_id
            and self.span.quote_sha256 == self.quote_sha256
            and self.quote_sha256 == quote_sha256(self.quote)
            and self.token_count == count_tokens(self.quote),
            "witness provenance or quote changed",
        )
        for values, label in (
            (self.completed_action_concepts, "completed actions"),
            (self.planned_action_concepts, "planned actions"),
            (self.supported_slot_ids, "supported slots"),
            (self.matched_query_terms, "matched query terms"),
            (self.matched_participant_terms, "matched participant terms"),
            (self.selection_axes, "selection axes"),
        ):
            _ordered_unique(values, label)
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "witness receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "candidate_id": self.candidate_id,
            "completed_action_concepts": list(self.completed_action_concepts),
            "event_date": self.event_date,
            "event_date_basis": self.event_date_basis,
            "format": WITNESS_FORMAT,
            "indexed_window_receipt_sha256": self.indexed_window_receipt_sha256,
            "matched_participant_terms": list(self.matched_participant_terms),
            "matched_query_terms": list(self.matched_query_terms),
            "parent_index_receipt_sha256": self.parent_index_receipt_sha256,
            "partition_id": self.partition_id,
            "planned_action_concepts": list(self.planned_action_concepts),
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "selection_axes": list(self.selection_axes),
            "source_database_sha256": self.source_database_sha256,
            "source_id": self.source_id,
            "source_store_receipt_sha256": self.source_store_receipt_sha256,
            "span": self.span.identity_payload(),
            "supported_slot_ids": list(self.supported_slot_ids),
            "temporal_distance_days": self.temporal_distance_days,
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _materialize(
    index: HotTypedWitnessIndex, draft: _Draft
) -> HotTypedWitness:
    window = index.windows[draft.window_index]
    axes = tuple(
        dict.fromkeys(
            [
                *(f"completed_action:{value}" for value in draft.completed_matches),
                *(f"planned_action:{value}" for value in draft.planned_matches),
                *(f"required_slot:{value}" for value in draft.supported_slot_ids),
                *(("participant_constraint",) if draft.matched_participant_terms else ()),
                *(("exact_temporal_target",) if draft.exact_temporal_match else ()),
                *(("temporal_window",) if draft.within_temporal_window else ()),
                *(("normalized_term_match",) if draft.matched_query_terms else ()),
            ]
        )
    )
    cache = index.parent.cache
    return HotTypedWitness(
        candidate_id=draft.candidate_id,
        parent_index_receipt_sha256=index.parent.receipt_sha256,
        indexed_window_receipt_sha256=window.receipt_sha256,
        cache_receipt_sha256=cache.cache_receipt_sha256,
        source_database_sha256=cache.source_database_sha256,
        source_store_receipt_sha256=cache.source_store_receipt_sha256,
        partition_id=window.row.partition_id,
        source_id=window.row.source_id,
        span=_span(window),
        quote=window.quote,
        quote_sha256=window.quote_sha256,
        token_count=window.token_count,
        completed_action_concepts=draft.completed_matches,
        planned_action_concepts=draft.planned_matches,
        supported_slot_ids=draft.supported_slot_ids,
        matched_query_terms=draft.matched_query_terms,
        matched_participant_terms=draft.matched_participant_terms,
        event_date=window.event_date,
        event_date_basis=window.event_date_basis,
        temporal_distance_days=draft.temporal_distance_days,
        selection_axes=axes,
    )


@dataclass(frozen=True, slots=True)
class HotTypedWitnessReceipt:
    question_sha256: str
    operator_spec_receipt_sha256: str
    temporal_target_receipt_sha256: str
    index_receipt_sha256: str
    budget_id: str
    scope_mode: str
    eligible_source_ids_sha256: str
    eligible_source_count: int
    matched_eligible_source_count: int
    query_action_concepts: tuple[str, ...]
    generic_completed_action_query: bool
    latest_requested: bool
    candidate_strategy: str
    effective_evidence_role: str | None
    evidence_role_derivation: str
    applicable: bool
    status: str
    candidate_population_count: int
    candidate_population_sha256: str
    role_rejected_candidate_count: int
    selected_before_dedup_ids: tuple[str, ...]
    selected_before_dedup_tokens: int
    protected_chunk_ids_sha256: str
    dedup_excluded_ids: tuple[str, ...]
    admitted_ids: tuple[str, ...]
    admitted_tokens: int
    selection_truncated: bool
    selection_before_protected_dedup: Literal[True] = True
    refill_after_protected_dedup: Literal[False] = False
    query_tick_full_physical_rescan: Literal[False] = False
    new_provider_calls: Literal[0] = 0
    model_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.question_sha256, "question"),
            (self.operator_spec_receipt_sha256, "operator spec"),
            (self.temporal_target_receipt_sha256, "temporal target"),
            (self.index_receipt_sha256, "typed witness index"),
            (self.budget_id, "budget"),
            (self.eligible_source_ids_sha256, "eligible sources"),
            (self.candidate_population_sha256, "candidate population"),
            (self.protected_chunk_ids_sha256, "protected chunks"),
        ):
            require_sha256(value, label)
        _require(self.scope_mode in {"full_store", "eligible_sources"}, "scope changed")
        for value in (
            self.eligible_source_count,
            self.matched_eligible_source_count,
            self.candidate_population_count,
            self.role_rejected_candidate_count,
            self.selected_before_dedup_tokens,
            self.admitted_tokens,
        ):
            _require(type(value) is int and value >= 0, "receipt count changed")
        _require(
            self.matched_eligible_source_count <= self.eligible_source_count,
            "matched source count exceeds requested source scope",
        )
        _ordered_unique(self.query_action_concepts, "query actions")
        require_text(self.candidate_strategy, "candidate strategy")
        require_text(self.evidence_role_derivation, "evidence-role derivation")
        _require(
            self.effective_evidence_role in {None, "user", "assistant"},
            "effective evidence role changed",
        )
        _require(type(self.applicable) is bool, "applicability changed")
        _require(
            self.status
            in {
                "not_applicable",
                "applicable_witnesses_available",
                "applicable_selected_all_protected",
                "applicable_unresolved",
            },
            "routing status changed",
        )
        _require(
            self.applicable is (self.status != "not_applicable"),
            "routing status differs from applicability",
        )
        selected = _ordered_unique(self.selected_before_dedup_ids, "selected witnesses")
        excluded = _ordered_unique(self.dedup_excluded_ids, "dedup exclusions")
        admitted = _ordered_unique(self.admitted_ids, "admitted witnesses")
        _require(
            set(excluded).isdisjoint(admitted)
            and set(excluded) | set(admitted) == set(selected),
            "post-selection dedup does not partition the frozen selection",
        )
        _require(type(self.selection_truncated) is bool, "truncation flag changed")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "query receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_typed_witness_receipt")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "admitted_ids": list(self.admitted_ids),
            "admitted_tokens": self.admitted_tokens,
            "applicable": self.applicable,
            "budget_id": self.budget_id,
            "candidate_population_count": self.candidate_population_count,
            "candidate_population_sha256": self.candidate_population_sha256,
            "candidate_strategy": self.candidate_strategy,
            "dedup_excluded_ids": list(self.dedup_excluded_ids),
            "eligible_source_count": self.eligible_source_count,
            "eligible_source_ids_sha256": self.eligible_source_ids_sha256,
            "effective_evidence_role": self.effective_evidence_role,
            "evidence_role_derivation": self.evidence_role_derivation,
            "format": RECEIPT_FORMAT,
            "generic_completed_action_query": self.generic_completed_action_query,
            "gold_loaded": False,
            "index_receipt_sha256": self.index_receipt_sha256,
            "matched_eligible_source_count": self.matched_eligible_source_count,
            "latest_requested": self.latest_requested,
            "model_calls": 0,
            "new_provider_calls": 0,
            "operator_spec_receipt_sha256": self.operator_spec_receipt_sha256,
            "protected_chunk_ids_sha256": self.protected_chunk_ids_sha256,
            "query_action_concepts": list(self.query_action_concepts),
            "query_tick_full_physical_rescan": False,
            "question_sha256": self.question_sha256,
            "refill_after_protected_dedup": False,
            "retained_transformer_token_state_bytes": 0,
            "role_rejected_candidate_count": self.role_rejected_candidate_count,
            "scope_mode": self.scope_mode,
            "selected_before_dedup_ids": list(self.selected_before_dedup_ids),
            "selected_before_dedup_tokens": self.selected_before_dedup_tokens,
            "selection_before_protected_dedup": True,
            "selection_truncated": self.selection_truncated,
            "status": self.status,
            "temporal_target_receipt_sha256": self.temporal_target_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotTypedWitnessResult:
    dated_question: str
    operator_spec: TypedOperatorSpec
    temporal_target: QuestionTemporalTarget
    selected_before_dedup: tuple[HotTypedWitness, ...]
    witnesses: tuple[HotTypedWitness, ...]
    receipt: HotTypedWitnessReceipt
    budget: HotTypedWitnessBudget

    def __post_init__(self) -> None:
        require_text(self.dated_question, "dated question")
        _require(type(self.operator_spec) is TypedOperatorSpec, "operator spec changed")
        for values, label in (
            (self.selected_before_dedup, "pre-dedup witnesses"),
            (self.witnesses, "admitted witnesses"),
        ):
            _require(
                type(values) is tuple
                and all(type(row) is HotTypedWitness for row in values),
                f"{label} changed",
            )
        _require(
            tuple(row.candidate_id for row in self.selected_before_dedup)
            == self.receipt.selected_before_dedup_ids
            and tuple(row.candidate_id for row in self.witnesses)
            == self.receipt.admitted_ids,
            "result membership differs from its receipt",
        )
        _require(
            sum(row.token_count for row in self.selected_before_dedup)
            == self.receipt.selected_before_dedup_tokens
            <= self.budget.evidence_token_cap
            and sum(row.token_count for row in self.witnesses)
            == self.receipt.admitted_tokens,
            "result token accounting changed",
        )

    def audit_projection(self) -> dict[str, Any]:
        value = {
            "dated_question": self.dated_question,
            "operator_spec": self.operator_spec.projection(),
            "receipt": self.receipt.projection(),
            "selected_before_dedup": [
                row.projection() for row in self.selected_before_dedup
            ],
            "temporal_target": self.temporal_target.projection(),
            "witnesses": [row.projection() for row in self.witnesses],
        }
        assert_gold_blind(value, path="hot_typed_witness_result")
        return value

    @property
    def applicable(self) -> bool:
        return self.receipt.applicable

    @property
    def status(self) -> str:
        return self.receipt.status

    @property
    def origin_chunk_ids(self) -> tuple[str, ...]:
        return tuple(row.span.chunk_id for row in self.witnesses)


@dataclass(frozen=True, slots=True)
class HotTypedWitnessEscalationDecision:
    """Gold-blind policy decision for an ambiguous ordered-event frontier."""

    question_sha256: str
    query_receipt_sha256: str
    scope_mode: str
    requested_cardinality: int | None
    ambiguity_multiplier: Literal[2]
    ambiguity_threshold: int | None
    distinct_selected_source_count: int
    candidate_population_count: int
    policy_eligible: bool
    trigger_axes: tuple[str, ...]
    escalate: bool
    reason: str
    new_provider_calls: Literal[0] = 0
    model_calls: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.question_sha256, "escalation question")
        require_sha256(self.query_receipt_sha256, "escalation query receipt")
        _require(
            self.scope_mode in {"full_store", "eligible_sources"},
            "escalation scope changed",
        )
        if self.requested_cardinality is None:
            _require(
                self.ambiguity_threshold is None,
                "missing cardinality gained an ambiguity threshold",
            )
        else:
            _require(
                type(self.requested_cardinality) is int
                and self.requested_cardinality > 0,
                "escalation cardinality changed",
            )
            _require(
                self.ambiguity_threshold
                == self.ambiguity_multiplier * self.requested_cardinality,
                "ambiguity threshold changed",
            )
        _require(self.ambiguity_multiplier == 2, "ambiguity multiplier changed")
        for value in (
            self.distinct_selected_source_count,
            self.candidate_population_count,
        ):
            _require(type(value) is int and value >= 0, "ambiguity count changed")
        _require(type(self.policy_eligible) is bool, "policy eligibility changed")
        triggers = _ordered_unique(self.trigger_axes, "ambiguity trigger axes")
        _require(
            set(triggers)
            <= {"candidate_population", "distinct_selected_sources"},
            "unknown ambiguity trigger axis",
        )
        _require(type(self.escalate) is bool, "escalation decision changed")
        _require(
            self.escalate is bool(self.policy_eligible and triggers),
            "escalation does not follow its audited triggers",
        )
        _require(
            self.reason
            in {
                "ambiguity_threshold_not_exceeded",
                "operator_not_eligible",
                "ordered_list_ambiguity_exceeds_threshold",
                "query_not_applicable",
                "requested_cardinality_unavailable",
                "scoped_query_not_eligible",
            },
            "escalation reason changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "escalation receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_typed_witness_escalation")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "ambiguity_multiplier": self.ambiguity_multiplier,
            "ambiguity_threshold": self.ambiguity_threshold,
            "candidate_population_count": self.candidate_population_count,
            "distinct_selected_source_count": self.distinct_selected_source_count,
            "escalate": self.escalate,
            "format": ESCALATION_FORMAT,
            "gold_loaded": False,
            "model_calls": 0,
            "new_provider_calls": 0,
            "policy_eligible": self.policy_eligible,
            "query_receipt_sha256": self.query_receipt_sha256,
            "question_sha256": self.question_sha256,
            "reason": self.reason,
            "requested_cardinality": self.requested_cardinality,
            "scope_mode": self.scope_mode,
            "trigger_axes": list(self.trigger_axes),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def assess_ordered_list_ambiguity(
    result: HotTypedWitnessResult, /
) -> HotTypedWitnessEscalationDecision:
    """Assess whether an unrestricted ordered-list search needs source scoping.

    This policy looks only at the compiled question contract and deterministic
    retrieval population.  It deliberately uses the frozen pre-dedup selection
    so evidence already supplied by another arm cannot hide search ambiguity.
    """

    _require(
        type(result) is HotTypedWitnessResult,
        "ambiguity assessment requires exact typed result",
    )
    spec = result.operator_spec
    receipt = result.receipt
    cardinality = spec.cardinality
    threshold = 2 * cardinality if cardinality is not None else None
    distinct_sources = len(
        {row.source_id for row in result.selected_before_dedup}
    )
    operator_eligible = bool(
        spec.answer_shape is AnswerShape.ORDERED_LIST
        and spec.temporal_mode is TemporalMode.ORDER
        and spec.requires_complete_frontier
    )
    policy_eligible = bool(
        receipt.scope_mode == "full_store"
        and receipt.applicable
        and operator_eligible
        and cardinality is not None
    )
    triggers: list[str] = []
    if policy_eligible and threshold is not None:
        if distinct_sources > threshold:
            triggers.append("distinct_selected_sources")
        if receipt.candidate_population_count > threshold:
            triggers.append("candidate_population")
    if receipt.scope_mode != "full_store":
        reason = "scoped_query_not_eligible"
    elif not receipt.applicable:
        reason = "query_not_applicable"
    elif not operator_eligible:
        reason = "operator_not_eligible"
    elif cardinality is None:
        reason = "requested_cardinality_unavailable"
    elif triggers:
        reason = "ordered_list_ambiguity_exceeds_threshold"
    else:
        reason = "ambiguity_threshold_not_exceeded"
    return HotTypedWitnessEscalationDecision(
        question_sha256=receipt.question_sha256,
        query_receipt_sha256=receipt.receipt_sha256,
        scope_mode=receipt.scope_mode,
        requested_cardinality=cardinality,
        ambiguity_multiplier=2,
        ambiguity_threshold=threshold,
        distinct_selected_source_count=distinct_sources,
        candidate_population_count=receipt.candidate_population_count,
        policy_eligible=policy_eligible,
        trigger_axes=tuple(triggers),
        escalate=bool(triggers),
        reason=reason,
    )


def hot_typed_witness_applicable(dated_question: str, /) -> bool:
    """Return the question-only routing decision without touching an index."""

    require_text(dated_question, "dated question")
    spec = compile_typed_operator_spec(dated_question)
    target = _temporal_target(dated_question, spec)
    body = _question_body(canonicalize_question_text(dated_question))
    return bool(
        _query_action_concepts(body)
        or (
            _GENERIC_COMPLETED_QUERY_RE.search(body)
            and not spec.include_proposed
        )
        or _LATEST_QUERY_RE.search(body)
        or target.mode is not TemporalTargetMode.NONE
        or spec.temporal_mode is not TemporalMode.NONE
    )


def query_hot_typed_witnesses(
    index: HotTypedWitnessIndex,
    dated_question: str,
    /,
    *,
    eligible_source_ids: Collection[str] | None = None,
    protected_chunk_ids: Sequence[str] = (),
    budget: HotTypedWitnessBudget = HotTypedWitnessBudget(),
) -> HotTypedWitnessResult:
    """Select exact typed witnesses from resident postings.

    ``eligible_source_ids=None`` searches the complete resident store.  Passing
    a collection intersects the same postings with that activated source set;
    no second lexical index or store scan is needed.  Protected chunk IDs are
    intentionally invisible until after the independent selection is frozen.
    """

    _require(type(index) is HotTypedWitnessIndex, "query requires exact hot index")
    require_text(dated_question, "dated question")
    _require(type(budget) is HotTypedWitnessBudget, "query budget changed")
    protected = _ordered_unique(protected_chunk_ids, "protected chunk IDs")
    spec = compile_typed_operator_spec(dated_question)
    target = _temporal_target(dated_question, spec)
    (
        ranked,
        requested_sources,
        matched_source_count,
        role_rejected,
        query_actions,
        generic_completed,
        latest_requested,
        candidate_strategy,
        effective_role,
        role_derivation,
    ) = _drafts(index, dated_question, spec, target, eligible_source_ids)
    selected: list[HotTypedWitness] = []
    selected_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    used_tokens = 0

    def add(draft: _Draft) -> bool:
        nonlocal used_tokens
        if draft.candidate_id in selected_ids:
            return True
        window = index.windows[draft.window_index]
        if window.token_count > budget.max_witness_tokens:
            return False
        if len(selected) >= budget.max_candidates:
            return False
        if (
            source_counts.get(window.row.source_id, 0)
            >= budget.max_candidates_per_source
        ):
            return False
        if used_tokens + window.token_count > budget.evidence_token_cap:
            return False
        witness = _materialize(index, draft)
        selected.append(witness)
        selected_ids.add(witness.candidate_id)
        source_counts[window.row.source_id] = (
            source_counts.get(window.row.source_id, 0) + 1
        )
        used_tokens += window.token_count
        return True

    # Give every typed action its own small share before ranked fill.  This is
    # crucial for count/set questions such as buy+assemble+sell+fix: a frequent
    # action cannot consume the whole overlay before a rarer action is seen.
    for action in query_actions:
        retained = 0
        for draft in ranked:
            if action not in {*draft.completed_matches, *draft.planned_matches}:
                continue
            if add(draft):
                retained += 1
            if retained >= budget.candidates_per_action:
                break
    for draft in ranked:
        add(draft)

    # This is deliberately the first point at which protected evidence is
    # inspected.  Do not refill slots vacated here.
    protected_set = set(protected)
    excluded = tuple(
        row for row in selected if row.span.chunk_id in protected_set
    )
    admitted = tuple(
        row for row in selected if row.span.chunk_id not in protected_set
    )
    applicable = bool(
        query_actions
        or generic_completed
        or latest_requested
        or target.mode is not TemporalTargetMode.NONE
        or spec.temporal_mode is not TemporalMode.NONE
    )
    if not applicable:
        status = "not_applicable"
    elif admitted:
        status = "applicable_witnesses_available"
    elif selected and excluded:
        status = "applicable_selected_all_protected"
    else:
        status = "applicable_unresolved"
    receipt = HotTypedWitnessReceipt(
        question_sha256=quote_sha256(dated_question),
        operator_spec_receipt_sha256=spec.receipt_sha256,
        temporal_target_receipt_sha256=target.receipt_sha256,
        index_receipt_sha256=index.receipt_sha256,
        budget_id=budget.budget_id,
        scope_mode="full_store" if eligible_source_ids is None else "eligible_sources",
        eligible_source_ids_sha256=identity_sha256(list(requested_sources)),
        eligible_source_count=(
            len(index.source_postings)
            if eligible_source_ids is None
            else len(requested_sources)
        ),
        matched_eligible_source_count=matched_source_count,
        query_action_concepts=query_actions,
        generic_completed_action_query=generic_completed,
        latest_requested=latest_requested,
        candidate_strategy=candidate_strategy,
        effective_evidence_role=effective_role,
        evidence_role_derivation=role_derivation,
        applicable=applicable,
        status=status,
        candidate_population_count=len(ranked),
        candidate_population_sha256=identity_sha256(
            [row.candidate_id for row in ranked]
        ),
        role_rejected_candidate_count=role_rejected,
        selected_before_dedup_ids=tuple(row.candidate_id for row in selected),
        selected_before_dedup_tokens=used_tokens,
        protected_chunk_ids_sha256=identity_sha256(list(protected)),
        dedup_excluded_ids=tuple(row.candidate_id for row in excluded),
        admitted_ids=tuple(row.candidate_id for row in admitted),
        admitted_tokens=sum(row.token_count for row in admitted),
        selection_truncated=len(selected) < len(ranked),
    )
    return HotTypedWitnessResult(
        dated_question=dated_question,
        operator_spec=spec,
        temporal_target=target,
        selected_before_dedup=tuple(selected),
        witnesses=admitted,
        receipt=receipt,
        budget=budget,
    )


__all__ = [
    "DEFAULT_EVIDENCE_TOKEN_CAP",
    "ESCALATION_FORMAT",
    "HotTypedWitness",
    "HotTypedWitnessBudget",
    "HotTypedWitnessError",
    "HotTypedWitnessEscalationDecision",
    "HotTypedWitnessIndex",
    "HotTypedWitnessReceipt",
    "HotTypedWitnessResult",
    "IndexedTypedWitnessWindow",
    "assess_ordered_list_ambiguity",
    "build_hot_typed_witness_index",
    "hot_typed_witness_applicable",
    "query_hot_typed_witnesses",
]
