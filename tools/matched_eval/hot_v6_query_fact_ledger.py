"""Small provider-free fact ledger over exact retrieved evidence rows.

The ledger is an additive representation lane.  ``selected_rows`` remain the
authoritative raw retrieval result.  An adapter may additionally pass exact raw
``candidate_rows`` from the already activated sources (for example, rows found
through an ingest-time fact index or a source-neighborhood expansion).  This
module neither retrieves those rows nor replaces the protected raw selection.

Compilation is deliberately question-only and target-blind.  It uses the typed
operator grammar plus deterministic numeric and action semantics, retains exact
substrings as facts, and cites the raw evidence ID behind every derived fact.
No provider is called and no semantic/content deduplication is performed.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

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
    matched_action_concepts,
    planned_action_concepts,
)
from .typed_numeric_semantics import NumericMention, numeric_mentions
from .typed_operator_spec import (
    AnswerShape,
    ComparisonMode,
    RequiredSlot,
    TemporalMode,
    TypedOperatorSpec,
    compile_typed_operator_spec,
    normalized_terms,
)


FORMAT = "memory-condense-hot-v6-query-fact-ledger-v1"
SLICE_FORMAT = f"{FORMAT}-slice-v1"
RAW_AUTHORITY_POLICY = "external_exact_raw_rows_remain_authoritative"
DEDUP_POLICY = "post_lane_selection_exact_fact_and_evidence_ids_only"
DEFAULT_MAX_RENDERED_FACTS = 24
DEFAULT_MAX_RENDERED_TOKENS = 2_048


class QueryFactLedgerError(MatchedEvalContractError):
    """An exact-row, provenance, or ledger invariant changed."""


class FactStatus(str, Enum):
    ASSERTED = "asserted"
    MENTIONED = "mentioned"
    COMPLETED = "completed"
    PLANNED = "planned"
    FAILED = "failed"
    NEGATED = "negated"


class FactOrigin(str, Enum):
    SELECTED = "selected"
    CANDIDATE = "candidate"


_CLAUSE_RE = re.compile(r"[^.!?;\r\n]+(?:[.!?]+|;)?")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
_DATED_QUESTION_RE = re.compile(
    r"^\[Question asked at .+?\]\s*", re.IGNORECASE | re.DOTALL
)
_EXPLICIT_TIME_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?"
    r"(?:,?\s+(?:19|20)\d{2})?\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(?:19|20)\d{2}\b|"
    r"\b(?:today|yesterday|tomorrow|tonight|last\s+(?:Monday|Tuesday|"
    r"Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year)|"
    r"next\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"week|month|year)|(?:zero|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|\d+)\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
    re.IGNORECASE,
)
_NEGATED_RE = re.compile(
    r"\b(?:did\s+not|didn't|do\s+not|don't|never|not\s+(?:yet|currently)|"
    r"no\s+longer)\b",
    re.IGNORECASE,
)
_FAILED_RE = re.compile(
    r"\b(?:failed|could\s+not|couldn't|wasn't\s+able|were not able|"
    r"didn't\s+manage|did\s+not\s+manage)\b",
    re.IGNORECASE,
)
_PENDING_RE = re.compile(
    r"\b(?:pending|scheduled|planning|plan\s+to|intend\s+to|going\s+to|"
    r"hope\s+to|want\s+to)\b",
    re.IGNORECASE,
)
_ASSISTANT_MEMORY_LOOKUP_RE = re.compile(
    r"\b(?:what|which|where|when|why|how)\b.{0,50}"
    r"\b(?:you|assistant)\b.{0,35}"
    r"\b(?:said|answered|suggested|recommended|told|wrote)\b|"
    r"\b(?:did|had|have)\s+(?:you|the\s+assistant)\s+"
    r"(?:say|answer|suggest|recommend|tell|write)\b",
    re.IGNORECASE,
)
_LEXICAL_STOP = frozenset(
    {
        "asked",
        "question",
        "answer",
        "current",
        "latest",
        "most",
        "recent",
        "recently",
        "many",
        "much",
        "first",
        "last",
    }
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise QueryFactLedgerError(message)


def _seal(kind: str, body: Mapping[str, Any]) -> str:
    value = {"format": f"{FORMAT}-{kind}", **body}
    assert_gold_blind(value, path="hot_v6_query_fact_ledger")
    return identity_sha256(value)


@dataclass(frozen=True, slots=True)
class TypedNumericOperand:
    """One operator-compatible number copied from an exact fact quote."""

    operand_id: str
    surface: str
    value: float
    dimension: str
    qualifier: str
    unit: str | None
    quote_start: int
    quote_end: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.operand_id, "query-fact numeric operand ID")
        require_text(self.surface, "query-fact numeric surface")
        _require(
            type(self.value) in {int, float} and math.isfinite(self.value),
            "query-fact numeric value changed",
        )
        require_text(self.dimension, "query-fact numeric dimension")
        require_text(self.qualifier, "query-fact numeric qualifier")
        if self.unit is not None:
            require_text(self.unit, "query-fact numeric unit")
        _require(
            type(self.quote_start) is int
            and type(self.quote_end) is int
            and 0 <= self.quote_start < self.quote_end,
            "query-fact numeric quote coordinates changed",
        )
        expected = _seal("numeric-operand", self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "query-fact numeric operand receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "dimension": self.dimension,
            "operand_id": self.operand_id,
            "qualifier": self.qualifier,
            "quote_end": self.quote_end,
            "quote_start": self.quote_start,
            "surface": self.surface,
            "unit": self.unit,
            "value": self.value,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class TypedExactQuoteFact:
    """A derived, typed view that never claims authority over its raw row."""

    fact_id: str
    exact_quote: str
    exact_quote_sha256: str
    quote_start: int
    quote_end: int
    backing_evidence_id: str
    backing_source_id: str
    backing_row_sha256: str
    source_role: str
    exchange_id: str | None
    envelope_id: str | None
    user_lead_evidence_id: str | None
    selected_spine_affinity: bool
    source_created_at: str | None
    time_mentions: tuple[str, ...]
    time_basis: str
    status: FactStatus
    action_concepts: tuple[str, ...]
    completed_action_concepts: tuple[str, ...]
    planned_action_concepts: tuple[str, ...]
    matched_query_actions: tuple[str, ...]
    query_term_hits: tuple[str, ...]
    numeric_operands: tuple[TypedNumericOperand, ...]
    bound_slot_ids: tuple[str, ...]
    type_lanes: tuple[str, ...]
    origins: tuple[FactOrigin, ...]
    relevance_score: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.fact_id, "query fact ID")
        require_text(self.exact_quote, "query fact exact quote")
        require_sha256(self.exact_quote_sha256, "query fact quote SHA-256")
        _require(
            quote_sha256(self.exact_quote) == self.exact_quote_sha256,
            "query fact exact quote changed",
        )
        _require(
            type(self.quote_start) is int
            and type(self.quote_end) is int
            and 0 <= self.quote_start < self.quote_end,
            "query fact quote coordinates changed",
        )
        require_text(self.backing_evidence_id, "query fact backing evidence ID")
        require_text(self.backing_source_id, "query fact backing source ID")
        require_sha256(self.backing_row_sha256, "query fact backing row")
        _require(
            self.fact_id != self.backing_evidence_id,
            "derived query fact ID must differ from its backing evidence ID",
        )
        require_text(self.source_role, "query fact source role")
        for value, label in (
            (self.exchange_id, "exchange ID"),
            (self.envelope_id, "envelope ID"),
            (self.user_lead_evidence_id, "user-lead evidence ID"),
        ):
            if value is not None:
                require_text(value, f"query fact {label}")
        _require(
            type(self.selected_spine_affinity) is bool,
            "query fact selected-spine affinity changed",
        )
        if self.source_created_at is not None:
            require_text(self.source_created_at, "query fact source time")
        _require(
            type(self.time_mentions) is tuple
            and len(set(self.time_mentions)) == len(self.time_mentions),
            "query fact time mentions changed",
        )
        require_text(self.time_basis, "query fact time basis")
        _require(type(self.status) is FactStatus, "query fact status changed")
        for values, label in (
            (self.action_concepts, "actions"),
            (self.completed_action_concepts, "completed actions"),
            (self.planned_action_concepts, "planned actions"),
            (self.matched_query_actions, "matched query actions"),
            (self.query_term_hits, "query term hits"),
            (self.bound_slot_ids, "bound slots"),
            (self.type_lanes, "type lanes"),
        ):
            _require(
                type(values) is tuple and len(set(values)) == len(values),
                f"query fact {label} changed",
            )
        _require(
            type(self.numeric_operands) is tuple
            and all(type(row) is TypedNumericOperand for row in self.numeric_operands)
            and len({row.operand_id for row in self.numeric_operands})
            == len(self.numeric_operands),
            "query fact numeric operands changed",
        )
        _require(
            all(
                self.exact_quote[row.quote_start : row.quote_end] == row.surface
                for row in self.numeric_operands
            ),
            "query fact numeric operand escaped its exact quote",
        )
        _require(
            type(self.origins) is tuple
            and bool(self.origins)
            and all(type(row) is FactOrigin for row in self.origins)
            and len(set(self.origins)) == len(self.origins),
            "query fact origins changed",
        )
        _require(
            type(self.relevance_score) is int and self.relevance_score >= 0,
            "query fact relevance score changed",
        )
        expected = _seal("fact", self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "query fact receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v6_query_fact")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "action_concepts": list(self.action_concepts),
            "backing_evidence_id": self.backing_evidence_id,
            "backing_row_sha256": self.backing_row_sha256,
            "backing_source_id": self.backing_source_id,
            "bound_slot_ids": list(self.bound_slot_ids),
            "completed_action_concepts": list(self.completed_action_concepts),
            "exact_quote": self.exact_quote,
            "exact_quote_sha256": self.exact_quote_sha256,
            "envelope_id": self.envelope_id,
            "exchange_id": self.exchange_id,
            "fact_id": self.fact_id,
            "matched_query_actions": list(self.matched_query_actions),
            "numeric_operands": [row.projection() for row in self.numeric_operands],
            "origins": [row.value for row in self.origins],
            "planned_action_concepts": list(self.planned_action_concepts),
            "query_term_hits": list(self.query_term_hits),
            "quote_end": self.quote_end,
            "quote_start": self.quote_start,
            "relevance_score": self.relevance_score,
            "selected_spine_affinity": self.selected_spine_affinity,
            "source_created_at": self.source_created_at,
            "source_role": self.source_role,
            "status": self.status.value,
            "time_basis": self.time_basis,
            "time_mentions": list(self.time_mentions),
            "type_lanes": list(self.type_lanes),
            "user_lead_evidence_id": self.user_lead_evidence_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class RequiredSlotBinding:
    binding_id: str
    slot_id: str
    slot_label: str
    fact_id: str
    backing_evidence_id: str
    matched_terms: tuple[str, ...]
    numeric_operand_ids: tuple[str, ...]
    relation_constraint: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.binding_id, "query-fact slot binding ID")
        require_sha256(self.slot_id, "query-fact required slot ID")
        require_text(self.slot_label, "query-fact slot label")
        require_sha256(self.fact_id, "query-fact bound fact ID")
        require_text(self.backing_evidence_id, "query-fact bound evidence ID")
        _require(
            type(self.matched_terms) is tuple
            and bool(self.matched_terms)
            and len(set(self.matched_terms)) == len(self.matched_terms),
            "query-fact slot matched terms changed",
        )
        _require(
            type(self.numeric_operand_ids) is tuple
            and len(set(self.numeric_operand_ids)) == len(self.numeric_operand_ids),
            "query-fact slot numeric operands changed",
        )
        for operand_id in self.numeric_operand_ids:
            require_sha256(operand_id, "query-fact slot numeric operand ID")
        if self.relation_constraint is not None:
            require_text(self.relation_constraint, "query-fact slot relation")
        expected = _seal("slot-binding", self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "query-fact slot binding receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "backing_evidence_id": self.backing_evidence_id,
            "binding_id": self.binding_id,
            "fact_id": self.fact_id,
            "matched_terms": list(self.matched_terms),
            "numeric_operand_ids": list(self.numeric_operand_ids),
            "relation_constraint": self.relation_constraint,
            "slot_id": self.slot_id,
            "slot_label": self.slot_label,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class QueryFactLedger:
    question_sha256: str
    operator_spec: TypedOperatorSpec
    selected_population_sha256: str
    candidate_population_sha256: str | None
    selected_evidence_ids: tuple[str, ...]
    candidate_evidence_ids: tuple[str, ...]
    selected_lane_fact_count: int
    candidate_lane_fact_count: int
    facts: tuple[TypedExactQuoteFact, ...]
    slot_bindings: tuple[RequiredSlotBinding, ...]
    unresolved_slot_ids: tuple[str, ...]
    cited_backing_evidence_ids: tuple[str, ...]
    duplicate_fact_count: int
    raw_evidence_authority: Literal[
        "external_exact_raw_rows_remain_authoritative"
    ] = RAW_AUTHORITY_POLICY
    raw_rows_embedded: Literal[False] = False
    dedup_policy: Literal[
        "post_lane_selection_exact_fact_and_evidence_ids_only"
    ] = DEDUP_POLICY
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.question_sha256, "query-fact ledger question")
        _require(
            type(self.operator_spec) is TypedOperatorSpec
            and self.operator_spec.question_sha256 == self.question_sha256,
            "query-fact ledger operator spec changed",
        )
        require_sha256(
            self.selected_population_sha256,
            "query-fact selected population",
        )
        if self.candidate_population_sha256 is not None:
            require_sha256(
                self.candidate_population_sha256,
                "query-fact candidate population",
            )
        for values, label in (
            (self.selected_evidence_ids, "selected evidence IDs"),
            (self.candidate_evidence_ids, "candidate evidence IDs"),
        ):
            _require(type(values) is tuple, f"query-fact {label} changed")
            for value in values:
                require_text(value, f"query-fact {label}")
        _require(
            type(self.selected_lane_fact_count) is int
            and self.selected_lane_fact_count >= 0
            and type(self.candidate_lane_fact_count) is int
            and self.candidate_lane_fact_count >= 0,
            "query-fact lane counts changed",
        )
        _require(
            type(self.facts) is tuple
            and all(type(row) is TypedExactQuoteFact for row in self.facts)
            and len({row.fact_id for row in self.facts}) == len(self.facts),
            "query-fact rows changed or exact IDs repeat",
        )
        _require(
            type(self.slot_bindings) is tuple
            and all(type(row) is RequiredSlotBinding for row in self.slot_bindings)
            and len({row.binding_id for row in self.slot_bindings})
            == len(self.slot_bindings),
            "query-fact slot bindings changed",
        )
        fact_ids = {row.fact_id for row in self.facts}
        facts_by_id = {row.fact_id: row for row in self.facts}
        _require(
            all(row.fact_id in fact_ids for row in self.slot_bindings)
            and all(
                set(row.numeric_operand_ids)
                <= {
                    operand.operand_id
                    for operand in facts_by_id[row.fact_id].numeric_operands
                }
                for row in self.slot_bindings
            ),
            "query-fact slot binding escaped the fact population",
        )
        spec_slot_ids = {row.slot_id for row in self.operator_spec.required_slots}
        bound_slot_ids = {row.slot_id for row in self.slot_bindings}
        _require(
            bound_slot_ids <= spec_slot_ids
            and {
                (fact.fact_id, slot_id)
                for fact in self.facts
                for slot_id in fact.bound_slot_ids
            }
            == {(row.fact_id, row.slot_id) for row in self.slot_bindings},
            "query-fact slot bindings disagree with facts or operator spec",
        )
        _require(
            type(self.unresolved_slot_ids) is tuple
            and len(set(self.unresolved_slot_ids)) == len(self.unresolved_slot_ids)
            and set(self.unresolved_slot_ids) == spec_slot_ids - bound_slot_ids,
            "query-fact unresolved slot accounting changed",
        )
        _require(
            type(self.cited_backing_evidence_ids) is tuple
            and len(set(self.cited_backing_evidence_ids))
            == len(self.cited_backing_evidence_ids)
            and set(self.cited_backing_evidence_ids)
            == {row.backing_evidence_id for row in self.facts},
            "query-fact backing evidence accounting changed",
        )
        _require(
            type(self.duplicate_fact_count) is int
            and self.duplicate_fact_count
            == self.selected_lane_fact_count
            + self.candidate_lane_fact_count
            - len(self.facts),
            "query-fact exact-dedup accounting changed",
        )
        _require(
            self.raw_evidence_authority == RAW_AUTHORITY_POLICY
            and self.raw_rows_embedded is False
            and self.dedup_policy == DEDUP_POLICY,
            "query-fact raw authority or dedup policy changed",
        )
        _require(
            self.provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0,
            "query-fact ledger called a provider or retained token state",
        )
        expected = _seal("ledger", self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "query-fact ledger receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v6_query_fact_ledger")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidate_evidence_ids": list(self.candidate_evidence_ids),
            "candidate_lane_fact_count": self.candidate_lane_fact_count,
            "candidate_population_sha256": self.candidate_population_sha256,
            "cited_backing_evidence_ids": list(self.cited_backing_evidence_ids),
            "dedup_policy": self.dedup_policy,
            "duplicate_fact_count": self.duplicate_fact_count,
            "facts": [row.projection() for row in self.facts],
            "format": FORMAT,
            "operator_spec": self.operator_spec.projection(),
            "provider_calls": self.provider_calls,
            "question_sha256": self.question_sha256,
            "raw_evidence_authority": self.raw_evidence_authority,
            "raw_rows_embedded": self.raw_rows_embedded,
            "receipt_version": 1,
            "retained_transformer_token_state_bytes": (
                self.retained_transformer_token_state_bytes
            ),
            "selected_evidence_ids": list(self.selected_evidence_ids),
            "selected_lane_fact_count": self.selected_lane_fact_count,
            "selected_population_sha256": self.selected_population_sha256,
            "slot_bindings": [row.projection() for row in self.slot_bindings],
            "unresolved_slot_ids": list(self.unresolved_slot_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FactLedgerSlice:
    """One immutable, bounded selection and its exact provider-facing text."""

    question_sha256: str
    ledger_receipt_sha256: str
    compiled_fact_count: int
    ledger_fact_ids: tuple[str, ...]
    facts: tuple[TypedExactQuoteFact, ...]
    mandatory_fact_ids: tuple[str, ...]
    omitted_fact_ids: tuple[str, ...]
    max_facts: int
    max_tokens: int
    rendered_text: str
    rendered_text_sha256: str
    rendered_token_count: int
    provider_calls: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.question_sha256, "query-fact slice question")
        require_sha256(self.ledger_receipt_sha256, "query-fact slice ledger receipt")
        _require(
            type(self.compiled_fact_count) is int
            and self.compiled_fact_count >= 0,
            "query-fact slice compiled count changed",
        )
        _require(
            type(self.ledger_fact_ids) is tuple
            and len(set(self.ledger_fact_ids)) == len(self.ledger_fact_ids)
            and len(self.ledger_fact_ids) == self.compiled_fact_count,
            "query-fact slice ledger fact population changed",
        )
        for fact_id in self.ledger_fact_ids:
            require_sha256(fact_id, "query-fact slice ledger fact ID")
        _require(
            type(self.facts) is tuple
            and all(type(row) is TypedExactQuoteFact for row in self.facts),
            "query-fact slice facts changed",
        )
        fact_ids = self.fact_ids
        _require(
            len(set(fact_ids)) == len(fact_ids),
            "query-fact slice fact IDs repeat",
        )
        for values, label in (
            (self.mandatory_fact_ids, "mandatory fact IDs"),
            (self.omitted_fact_ids, "omitted fact IDs"),
        ):
            _require(type(values) is tuple, f"query-fact slice {label} changed")
            _require(
                len(set(values)) == len(values),
                f"query-fact slice {label} repeat",
            )
            for value in values:
                require_sha256(value, f"query-fact slice {label}")
        _require(
            set(self.mandatory_fact_ids) <= set(fact_ids),
            "query-fact slice lost mandatory coverage",
        )
        _require(
            self.mandatory_fact_ids
            == tuple(
                fact_id
                for fact_id in fact_ids
                if fact_id in set(self.mandatory_fact_ids)
            ),
            "query-fact slice mandatory order changed",
        )
        _require(
            bool(self.mandatory_fact_ids) == bool(fact_ids),
            "query-fact slice mandatory coverage became empty",
        )
        _require(
            not (set(fact_ids) & set(self.omitted_fact_ids))
            and fact_ids
            == tuple(
                fact_id
                for fact_id in self.ledger_fact_ids
                if fact_id in set(fact_ids)
            )
            and self.omitted_fact_ids
            == tuple(
                fact_id
                for fact_id in self.ledger_fact_ids
                if fact_id not in set(fact_ids)
            ),
            "query-fact slice selected/omitted accounting changed",
        )
        mandatory_facts = tuple(
            row for row in self.facts if row.fact_id in set(self.mandatory_fact_ids)
        )
        _require(
            {
                slot_id
                for row in self.facts
                for slot_id in row.bound_slot_ids
            }
            <= {
                slot_id
                for row in mandatory_facts
                for slot_id in row.bound_slot_ids
            }
            and {origin for row in self.facts for origin in row.origins}
            <= {origin for row in mandatory_facts for origin in row.origins}
            and {lane for row in self.facts for lane in row.type_lanes}
            <= {lane for row in mandatory_facts for lane in row.type_lanes},
            "query-fact slice mandatory witnesses lost represented coverage",
        )
        _require(
            type(self.max_facts) is int
            and self.max_facts >= 1
            and len(fact_ids) <= self.max_facts,
            "query-fact slice fact bound changed",
        )
        _require(
            type(self.max_tokens) is int and self.max_tokens >= 1,
            "query-fact slice token bound changed",
        )
        require_text(self.rendered_text, "query-fact slice rendered text")
        require_sha256(
            self.rendered_text_sha256,
            "query-fact slice rendered text",
        )
        _require(
            self.rendered_text_sha256 == quote_sha256(self.rendered_text),
            "query-fact slice rendered text digest changed",
        )
        _require(
            type(self.rendered_token_count) is int
            and self.rendered_token_count == count_tokens(self.rendered_text)
            and self.rendered_token_count <= self.max_tokens,
            "query-fact slice rendered token accounting changed",
        )
        rendered_lines = self.rendered_text.splitlines()
        rendered_fact_lines = tuple(
            line for line in rendered_lines if line.startswith("[F")
        )
        rendered_fact_ids = tuple(
            line.split(" fact=", 1)[1].split(" ", 1)[0]
            for line in rendered_fact_lines
            if line.startswith("[F") and " fact=" in line
        )
        _require(
            rendered_fact_ids == fact_ids,
            "query-fact slice facts disagree with rendered text",
        )
        _require(
            all(
                line.startswith(
                    f"[F{index} fact={fact.fact_id} "
                    f"backs={fact.backing_evidence_id} "
                    f"source={fact.backing_source_id}] "
                )
                for index, (line, fact) in enumerate(
                    zip(rendered_fact_lines, self.facts, strict=True),
                    1,
                )
            ),
            "query-fact slice rendered citation changed",
        )
        _require(
            (
                f"emitted_facts={len(fact_ids)}/{self.compiled_fact_count}; "
                f"bounds=max_facts:{self.max_facts},max_tokens:{self.max_tokens}"
            )
            in rendered_lines
            and f"ledger_receipt={self.ledger_receipt_sha256}" in rendered_lines,
            "query-fact slice rendered header changed",
        )
        _require(
            type(self.provider_calls) is int and self.provider_calls == 0,
            "query-fact slice called a provider",
        )
        expected = _seal("slice", self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "query-fact slice receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v6_query_fact_ledger_slice")

    @property
    def fact_ids(self) -> tuple[str, ...]:
        return tuple(row.fact_id for row in self.facts)

    @property
    def text(self) -> str:
        """Compatibility alias for the exact rendered provider text."""

        return self.rendered_text

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "compiled_fact_count": self.compiled_fact_count,
            "fact_ids": list(self.fact_ids),
            "facts": [row.projection() for row in self.facts],
            "format": SLICE_FORMAT,
            "ledger_receipt_sha256": self.ledger_receipt_sha256,
            "ledger_fact_ids": list(self.ledger_fact_ids),
            "mandatory_fact_ids": list(self.mandatory_fact_ids),
            "max_facts": self.max_facts,
            "max_tokens": self.max_tokens,
            "omitted_fact_ids": list(self.omitted_fact_ids),
            "provider_calls": self.provider_calls,
            "question_sha256": self.question_sha256,
            "rendered_text": self.rendered_text,
            "rendered_text_sha256": self.rendered_text_sha256,
            "rendered_token_count": self.rendered_token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class _ExactRow:
    evidence_id: str
    source_id: str
    role: str
    created_at: str | None
    text: str
    exchange_id: str | None
    envelope_id: str | None
    user_lead_evidence_id: str | None
    backing_row_sha256: str
    full_row_sha256: str


def _exact_row(raw: Mapping[str, Any], *, path: str) -> _ExactRow:
    _require(type(raw) is dict, f"{path} must contain exact objects")
    try:
        assert_gold_blind(raw, path=path)
    except MatchedEvalContractError as exc:
        raise QueryFactLedgerError(str(exc)) from exc
    _require("ordinal" not in raw, f"{path} must not carry benchmark ordinals")
    evidence_id = require_text(raw.get("evidence_id"), f"{path} evidence ID")
    source_id = require_text(raw.get("source_id"), f"{path} source ID")
    role = require_text(raw.get("role"), f"{path} role")
    text = require_text(raw.get("raw_text"), f"{path} raw text")
    created = raw.get("created_at")
    _require(
        created is None or (type(created) is str and bool(created.strip())),
        f"{path} source time changed",
    )
    exchange_id = raw.get("exchange_id")
    envelope_id = raw.get("envelope_id")
    user_lead_evidence_id = raw.get("user_lead_evidence_id")
    for value, label in (
        (exchange_id, "exchange ID"),
        (envelope_id, "envelope ID"),
        (user_lead_evidence_id, "user-lead evidence ID"),
    ):
        _require(
            value is None or (type(value) is str and bool(value.strip())),
            f"{path} {label} changed",
        )
    if "chunk_id" in raw:
        _require(raw["chunk_id"] == evidence_id, f"{path} chunk/evidence ID changed")
    if "raw_text_sha256" in raw:
        _require(
            raw["raw_text_sha256"] == quote_sha256(text),
            f"{path} raw text digest changed",
        )
    authority = {
        "created_at": created,
        "evidence_id": evidence_id,
        "envelope_id": envelope_id,
        "exchange_id": exchange_id,
        "raw_text": text,
        "raw_text_sha256": quote_sha256(text),
        "role": role,
        "source_id": source_id,
    }
    return _ExactRow(
        evidence_id,
        source_id,
        role,
        created,
        text,
        exchange_id,
        envelope_id,
        user_lead_evidence_id,
        identity_sha256(authority),
        identity_sha256(raw),
    )


def _rows(
    values: Sequence[Mapping[str, Any]], *, label: str
) -> tuple[_ExactRow, ...]:
    _require(
        not isinstance(values, (str, bytes)),
        f"query-fact {label} rows must be an exact sequence",
    )
    return tuple(
        _exact_row(raw, path=f"hot_v6_{label}_rows[{index}]")
        for index, raw in enumerate(values)
    )


def _spine_key(row: _ExactRow) -> tuple[str, str | None, str | None]:
    return row.source_id, row.exchange_id, row.envelope_id


def _attach_user_leads(
    selected: tuple[_ExactRow, ...],
    candidates: tuple[_ExactRow, ...],
) -> tuple[tuple[_ExactRow, ...], tuple[_ExactRow, ...]]:
    """Resolve only local, unambiguous user-lead ownership.

    A boundaryless assistant row is never attached merely because it shares a
    broad source.  Explicit lead addresses win; otherwise exactly one user row
    inside an exchange/envelope is the lead.  Authorship remains on ``role``.
    """

    groups: dict[tuple[str, str | None, str | None], list[_ExactRow]] = {}
    for row in (*selected, *candidates):
        if row.exchange_id is None and row.envelope_id is None:
            continue
        groups.setdefault(_spine_key(row), []).append(row)
    leads: dict[tuple[str, str | None, str | None], str] = {}
    for key, rows in groups.items():
        explicit = {
            row.user_lead_evidence_id
            for row in rows
            if row.user_lead_evidence_id is not None
        }
        _require(
            len(explicit) <= 1,
            "one source/exchange boundary names multiple user-lead addresses",
        )
        if explicit:
            leads[key] = next(iter(explicit))
            continue
        users = {row.evidence_id for row in rows if row.role == "user"}
        if len(users) == 1:
            leads[key] = next(iter(users))

    def attach(rows: tuple[_ExactRow, ...]) -> tuple[_ExactRow, ...]:
        output: list[_ExactRow] = []
        for row in rows:
            lead = row.user_lead_evidence_id
            if lead is None and (row.exchange_id is not None or row.envelope_id is not None):
                lead = leads.get(_spine_key(row))
            if lead == row.user_lead_evidence_id:
                output.append(row)
                continue
            output.append(
                replace(
                    row,
                    user_lead_evidence_id=lead,
                )
            )
        return tuple(output)

    return attach(selected), attach(candidates)


def _quote_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    output: list[tuple[int, int, str]] = []
    for match in _CLAUSE_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end and _WORD_RE.search(text[start:end]):
            output.append((start, end, text[start:end]))
    if not output:
        output.append((0, len(text), text))
    return tuple(output)


def _status(
    quote: str,
    actions: tuple[str, ...],
    completed: tuple[str, ...],
    planned: tuple[str, ...],
) -> FactStatus:
    if _FAILED_RE.search(quote):
        return FactStatus.FAILED
    if _NEGATED_RE.search(quote):
        return FactStatus.NEGATED
    if planned or _PENDING_RE.search(quote):
        return FactStatus.PLANNED
    if completed:
        return FactStatus.COMPLETED
    if actions:
        return FactStatus.MENTIONED
    return FactStatus.ASSERTED


def _preferred_role(question: str, spec: TypedOperatorSpec) -> str:
    if spec.required_evidence_role is not None:
        return spec.required_evidence_role
    if _ASSISTANT_MEMORY_LOOKUP_RE.search(question):
        return "assistant"
    # The memory corpus is user-led.  A present-tense request such as "Can you
    # recommend ...?" needs user preference/history evidence, not arbitrary old
    # assistant prose.
    return "user"


def _numeric_operand(
    row: _ExactRow,
    quote_start: int,
    quote: str,
    mention: NumericMention,
) -> TypedNumericOperand:
    body = {
        "backing_evidence_id": row.evidence_id,
        "dimension": mention.dimension.value,
        "quote_end": quote_start + mention.end,
        "quote_start": quote_start + mention.start,
        "qualifier": mention.qualifier.value,
        "surface": mention.surface,
        "unit": mention.unit,
        "value": mention.value,
    }
    return TypedNumericOperand(
        operand_id=_seal("numeric-operand-id", body),
        surface=mention.surface,
        value=float(mention.value),
        dimension=mention.dimension.value,
        qualifier=mention.qualifier.value,
        unit=mention.unit,
        quote_start=mention.start,
        quote_end=mention.end,
    )


def _slot_matches(
    slot: RequiredSlot,
    quote_terms: tuple[str, ...],
    numeric: tuple[TypedNumericOperand, ...],
) -> tuple[str, ...]:
    present = set(quote_terms)
    matched = tuple(term for term in slot.match_terms if term in present)
    if len(matched) < slot.minimum_match_term_count:
        return ()
    if slot.requires_numeric and not numeric:
        return ()
    return matched


def _compile_fact(
    *,
    question: str,
    question_terms: tuple[str, ...],
    spec: TypedOperatorSpec,
    row: _ExactRow,
    quote_start: int,
    quote_end: int,
    quote: str,
    origin: FactOrigin,
    selected_spines: frozenset[tuple[str, str | None, str | None]],
    selected_evidence_ids: frozenset[str],
) -> TypedExactQuoteFact:
    quote_terms = normalized_terms(quote)
    term_hits = tuple(term for term in question_terms if term in set(quote_terms))
    actions = canonical_action_concepts(quote)
    completed = completed_action_concepts(quote)
    planned = planned_action_concepts(quote)
    action_hits = matched_action_concepts(question, quote)
    operands = tuple(
        _numeric_operand(row, quote_start, quote, mention)
        for mention in numeric_mentions(
            quote,
            operator_spec=spec,
            question=question,
        )
    )
    slot_matches = {
        slot.slot_id: _slot_matches(slot, quote_terms, operands)
        for slot in spec.required_slots
    }
    bound_slots = tuple(
        slot.slot_id for slot in spec.required_slots if slot_matches[slot.slot_id]
    )
    times = tuple(dict.fromkeys(match.group(0) for match in _EXPLICIT_TIME_RE.finditer(quote)))
    time_basis = (
        "explicit_event_time_in_quote"
        if times
        else "source_created_at_fallback"
        if row.created_at is not None
        else "unknown"
    )
    status = _status(quote, actions, completed, planned)
    lanes: list[str] = []
    if bound_slots:
        lanes.append("required_slot")
    if operands:
        lanes.append("numeric_operand")
    if actions:
        lanes.append("action")
    if times:
        lanes.append("event_time")
    if term_hits:
        lanes.append("lexical")
    if not lanes:
        lanes.append("residual_exact_quote")
    role_match = row.role == _preferred_role(question, spec)
    spine_affinity = _spine_key(row) in selected_spines or (
        row.user_lead_evidence_id in selected_evidence_ids
        if row.user_lead_evidence_id is not None
        else False
    )
    numeric_weight = (
        28
        if spec.comparison_mode is not ComparisonMode.NONE
        or spec.answer_shape is AnswerShape.DURATION
        else 4
        if spec.operation == "count_or_aggregate"
        else 2
    )
    time_weight = 14 if spec.temporal_mode is not TemporalMode.NONE else 2
    score = max(
        0,
        100 * len(bound_slots)
        + numeric_weight * len(operands)
        + 28 * len(action_hits)
        + 10 * len(completed)
        + 32 * int(role_match)
        + 12 * int(spine_affinity)
        + time_weight * len(times)
        + 24 * len(term_hits)
        + int(status not in {FactStatus.FAILED, FactStatus.NEGATED})
        - min(12, len(quote) // 80),
    )
    fact_identity = {
        "backing_evidence_id": row.evidence_id,
        "backing_row_sha256": row.backing_row_sha256,
        "envelope_id": row.envelope_id,
        "exact_quote_sha256": quote_sha256(quote),
        "exchange_id": row.exchange_id,
        "quote_end": quote_end,
        "quote_start": quote_start,
        "user_lead_evidence_id": row.user_lead_evidence_id,
    }
    return TypedExactQuoteFact(
        fact_id=_seal("fact-id", fact_identity),
        exact_quote=quote,
        exact_quote_sha256=quote_sha256(quote),
        quote_start=quote_start,
        quote_end=quote_end,
        backing_evidence_id=row.evidence_id,
        backing_source_id=row.source_id,
        backing_row_sha256=row.backing_row_sha256,
        source_role=row.role,
        exchange_id=row.exchange_id,
        envelope_id=row.envelope_id,
        user_lead_evidence_id=row.user_lead_evidence_id,
        selected_spine_affinity=spine_affinity,
        source_created_at=row.created_at,
        time_mentions=times,
        time_basis=time_basis,
        status=status,
        action_concepts=actions,
        completed_action_concepts=completed,
        planned_action_concepts=planned,
        matched_query_actions=action_hits,
        query_term_hits=term_hits,
        numeric_operands=operands,
        bound_slot_ids=bound_slots,
        type_lanes=tuple(lanes),
        origins=(origin,),
        relevance_score=score,
    )


def _compile_lane(
    *,
    question: str,
    question_terms: tuple[str, ...],
    spec: TypedOperatorSpec,
    rows: tuple[_ExactRow, ...],
    origin: FactOrigin,
    selected_spines: frozenset[tuple[str, str | None, str | None]],
    selected_evidence_ids: frozenset[str],
) -> tuple[TypedExactQuoteFact, ...]:
    return tuple(
        _compile_fact(
            question=question,
            question_terms=question_terms,
            spec=spec,
            row=row,
            quote_start=start,
            quote_end=end,
            quote=quote,
            origin=origin,
            selected_spines=selected_spines,
            selected_evidence_ids=selected_evidence_ids,
        )
        for row in rows
        for start, end, quote in _quote_spans(row.text)
    )


def _merge_exact_facts(
    selected: tuple[TypedExactQuoteFact, ...],
    candidates: tuple[TypedExactQuoteFact, ...],
) -> tuple[TypedExactQuoteFact, ...]:
    merged: dict[str, TypedExactQuoteFact] = {}
    evidence_bindings: dict[str, str] = {}
    for fact in (*selected, *candidates):
        evidence_binding = identity_sha256(
            {
                "backing_row_sha256": fact.backing_row_sha256,
                "envelope_id": fact.envelope_id,
                "exchange_id": fact.exchange_id,
                "user_lead_evidence_id": fact.user_lead_evidence_id,
            }
        )
        previous_row = evidence_bindings.setdefault(
            fact.backing_evidence_id, evidence_binding
        )
        _require(
            previous_row == evidence_binding,
            "one exact evidence ID is bound to different authoritative raw bytes",
        )
        prior = merged.get(fact.fact_id)
        if prior is None:
            merged[fact.fact_id] = fact
            continue
        _require(
            prior.projection(include_receipt=False)
            == replace(
                fact,
                origins=prior.origins,
                receipt_sha256="",
            ).projection(include_receipt=False),
            "one exact fact ID is bound to different fact bytes",
        )
        origins = tuple(dict.fromkeys((*prior.origins, *fact.origins)))
        merged[fact.fact_id] = replace(prior, origins=origins, receipt_sha256="")
    return tuple(
        sorted(
            merged.values(),
            key=lambda row: (
                -row.relevance_score,
                row.backing_source_id,
                row.exchange_id or "",
                row.envelope_id or "",
                row.backing_evidence_id,
                row.quote_start,
                row.fact_id,
            ),
        )
    )


def _bindings(
    spec: TypedOperatorSpec,
    facts: tuple[TypedExactQuoteFact, ...],
) -> tuple[RequiredSlotBinding, ...]:
    slots = {slot.slot_id: slot for slot in spec.required_slots}
    output: list[RequiredSlotBinding] = []
    for fact in facts:
        quote_terms = set(normalized_terms(fact.exact_quote))
        for slot_id in fact.bound_slot_ids:
            slot = slots[slot_id]
            matched = tuple(term for term in slot.match_terms if term in quote_terms)
            operand_ids = (
                tuple(row.operand_id for row in fact.numeric_operands)
                if slot.requires_numeric
                else ()
            )
            body = {
                "backing_evidence_id": fact.backing_evidence_id,
                "fact_id": fact.fact_id,
                "matched_terms": list(matched),
                "numeric_operand_ids": list(operand_ids),
                "slot_id": slot_id,
            }
            output.append(
                RequiredSlotBinding(
                    binding_id=_seal("slot-binding-id", body),
                    slot_id=slot_id,
                    slot_label=slot.label,
                    fact_id=fact.fact_id,
                    backing_evidence_id=fact.backing_evidence_id,
                    matched_terms=matched,
                    numeric_operand_ids=operand_ids,
                    relation_constraint=slot.relation_constraint,
                )
            )
    return tuple(output)


def compile_query_fact_ledger(
    question: str,
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_rows: Sequence[Mapping[str, Any]] | None = None,
) -> QueryFactLedger:
    """Compile a ranked exact-quote ledger without changing raw selection.

    Candidate rows are an explicit activated-source seam, not a store scan.
    Every candidate source must already be represented by ``selected_rows``.
    Both lanes are compiled before exact-ID deduplication; identical text under
    distinct evidence IDs remains distinct.
    """

    if type(question) is not str or not question or question.strip() != question:
        raise QueryFactLedgerError("query-fact question must be non-empty exact text")
    spec = compile_typed_operator_spec(question)
    selected = _rows(selected_rows, label="selected")
    _require(selected, "query-fact selected rows must not be empty")
    candidates = (
        _rows(candidate_rows, label="candidate")
        if candidate_rows is not None
        else ()
    )
    selected_sources = {row.source_id for row in selected}
    _require(
        all(row.source_id in selected_sources for row in candidates),
        "candidate rows must stay inside sources activated by selected rows",
    )
    selected, candidates = _attach_user_leads(selected, candidates)
    selected_spines = frozenset(
        _spine_key(row)
        for row in selected
        if row.exchange_id is not None or row.envelope_id is not None
    )
    selected_evidence_ids = frozenset(row.evidence_id for row in selected)
    body = _DATED_QUESTION_RE.sub("", question).strip()
    question_terms = tuple(
        term
        for term in normalized_terms(body)
        if term not in _LEXICAL_STOP and not term.isdigit()
    )
    selected_facts = _compile_lane(
        question=question,
        question_terms=question_terms,
        spec=spec,
        rows=selected,
        origin=FactOrigin.SELECTED,
        selected_spines=selected_spines,
        selected_evidence_ids=selected_evidence_ids,
    )
    candidate_facts = _compile_lane(
        question=question,
        question_terms=question_terms,
        spec=spec,
        rows=candidates,
        origin=FactOrigin.CANDIDATE,
        selected_spines=selected_spines,
        selected_evidence_ids=selected_evidence_ids,
    )
    facts = _merge_exact_facts(selected_facts, candidate_facts)
    bindings = _bindings(spec, facts)
    bound_slot_ids = {row.slot_id for row in bindings}
    unresolved = tuple(
        slot.slot_id
        for slot in spec.required_slots
        if slot.slot_id not in bound_slot_ids
    )
    cited = tuple(dict.fromkeys(row.backing_evidence_id for row in facts))
    selected_population = identity_sha256(
        [row.full_row_sha256 for row in selected]
    )
    candidate_population = (
        identity_sha256([row.full_row_sha256 for row in candidates])
        if candidate_rows is not None
        else None
    )
    return QueryFactLedger(
        question_sha256=spec.question_sha256,
        operator_spec=spec,
        selected_population_sha256=selected_population,
        candidate_population_sha256=candidate_population,
        selected_evidence_ids=tuple(row.evidence_id for row in selected),
        candidate_evidence_ids=tuple(row.evidence_id for row in candidates),
        selected_lane_fact_count=len(selected_facts),
        candidate_lane_fact_count=len(candidate_facts),
        facts=facts,
        slot_bindings=bindings,
        unresolved_slot_ids=unresolved,
        cited_backing_evidence_ids=cited,
        duplicate_fact_count=len(selected_facts) + len(candidate_facts) - len(facts),
    )


def _coverage_fact_ids(ledger: QueryFactLedger) -> tuple[str, ...]:
    """Choose one highest-ranked witness for every obligation and active lane."""

    output: list[str] = []

    def keep(fact: TypedExactQuoteFact | None) -> None:
        if fact is not None and fact.fact_id not in output:
            output.append(fact.fact_id)

    bindings_by_slot: dict[str, set[str]] = {}
    for binding in ledger.slot_bindings:
        bindings_by_slot.setdefault(binding.slot_id, set()).add(binding.fact_id)
    for slot in ledger.operator_spec.required_slots:
        fact_ids = bindings_by_slot.get(slot.slot_id, set())
        keep(next((row for row in ledger.facts if row.fact_id in fact_ids), None))
    for origin in FactOrigin:
        keep(next((row for row in ledger.facts if origin in row.origins), None))
    type_lanes = tuple(
        dict.fromkeys(lane for row in ledger.facts for lane in row.type_lanes)
    )
    for lane in type_lanes:
        keep(next((row for row in ledger.facts if lane in row.type_lanes), None))
    return tuple(output)


def _render_lines(
    ledger: QueryFactLedger,
    facts: Sequence[TypedExactQuoteFact],
    *,
    max_facts: int,
    max_tokens: int,
) -> list[str]:
    spec = ledger.operator_spec
    lines = [
        "<QUERY_FACT_LEDGER>",
        (
            f"operation={spec.operation}; shape={spec.answer_shape.value}; "
            f"temporal={spec.temporal_mode.value}; comparison={spec.comparison_mode.value}"
        ),
        (
            "authority=raw backing rows remain authoritative and separately "
            "selectable; this selected/activated-source ledger cannot prove absence"
        ),
        (
            f"emitted_facts={len(facts)}/{len(ledger.facts)}; "
            f"bounds=max_facts:{max_facts},max_tokens:{max_tokens}"
        ),
    ]
    bindings_by_fact: dict[str, list[str]] = {}
    for binding in ledger.slot_bindings:
        bindings_by_fact.setdefault(binding.fact_id, []).append(binding.slot_label)
    if spec.required_slots:
        unresolved = {
            slot.slot_id: slot.label
            for slot in spec.required_slots
            if slot.slot_id in ledger.unresolved_slot_ids
        }
        lines.append(
            "required_slots="
            + (
                "; ".join(
                    f"{slot.label}:{'bound' if slot.slot_id not in unresolved else 'unresolved'}"
                    for slot in spec.required_slots
                )
            )
        )
    for index, fact in enumerate(facts, 1):
        metadata = [
            f"role={fact.source_role}",
            f"status={fact.status.value}",
            f"time_basis={fact.time_basis}",
        ]
        if fact.source_created_at is not None:
            metadata.append(f"source_time={fact.source_created_at}")
        if fact.exchange_id is not None:
            metadata.append(f"exchange={fact.exchange_id}")
        if fact.envelope_id is not None:
            metadata.append(f"envelope={fact.envelope_id}")
        if fact.user_lead_evidence_id is not None:
            metadata.append(f"user_lead={fact.user_lead_evidence_id}")
        if fact.time_mentions:
            metadata.append("event_time=" + "|".join(fact.time_mentions))
        if fact.action_concepts:
            metadata.append("actions=" + "|".join(fact.action_concepts))
        if fact.numeric_operands:
            metadata.append(
                "operands="
                + "|".join(
                    f"{row.surface}:{row.dimension}:{row.qualifier}"
                    for row in fact.numeric_operands
                )
            )
        if labels := bindings_by_fact.get(fact.fact_id):
            metadata.append("slots=" + "|".join(labels))
        lines.append(
            f"[F{index} fact={fact.fact_id} backs={fact.backing_evidence_id} "
            f"source={fact.backing_source_id}] "
            + "; ".join(metadata)
            + "; quote="
            + json.dumps(fact.exact_quote, ensure_ascii=False)
        )
    if not facts:
        lines.append("facts=none")
    lines.append(f"ledger_receipt={ledger.receipt_sha256}")
    lines.append("</QUERY_FACT_LEDGER>")
    return lines


def select_and_render_query_fact_ledger(
    ledger: QueryFactLedger,
    *,
    max_facts: int = DEFAULT_MAX_RENDERED_FACTS,
    max_tokens: int = DEFAULT_MAX_RENDERED_TOKENS,
) -> FactLedgerSlice:
    """Select and render one sealed overlay with slot/lane coverage intact.

    Raw rows are intentionally outside this renderer.  If the requested bounds
    cannot hold one witness for every bound required slot, origin lane, and
    represented fact-type lane, this fails closed instead of silently dropping
    coverage.
    """

    if type(ledger) is not QueryFactLedger:
        raise TypeError("fact selector requires an exact QueryFactLedger")
    _require(
        type(max_facts) is int and max_facts >= 1,
        "query-fact renderer max_facts must be a positive exact integer",
    )
    _require(
        type(max_tokens) is int and max_tokens >= 1,
        "query-fact renderer max_tokens must be a positive exact integer",
    )
    coverage = set(_coverage_fact_ids(ledger))
    _require(
        len(coverage) <= max_facts,
        "query-fact render fact bound cannot preserve slot/lane coverage",
    )
    optional = tuple(row for row in ledger.facts if row.fact_id not in coverage)

    def materialize(
        optional_count: int,
    ) -> tuple[tuple[TypedExactQuoteFact, ...], str]:
        selected_ids = coverage | {
            row.fact_id for row in optional[:optional_count]
        }
        facts = tuple(row for row in ledger.facts if row.fact_id in selected_ids)
        rendered = "\n".join(
            _render_lines(
                ledger,
                facts,
                max_facts=max_facts,
                max_tokens=max_tokens,
            )
        )
        return facts, rendered

    selected_facts, result = materialize(0)
    _require(
        count_tokens(result) <= max_tokens,
        "query-fact render token bound cannot preserve slot/lane coverage",
    )
    low = 0
    high = min(len(optional), max_facts - len(coverage))
    selected_optional_count = 0
    while low <= high:
        midpoint = (low + high) // 2
        candidate_facts, candidate_text = materialize(midpoint)
        if count_tokens(candidate_text) <= max_tokens:
            selected_facts = candidate_facts
            result = candidate_text
            selected_optional_count = midpoint
            low = midpoint + 1
        else:
            high = midpoint - 1
    _require(
        len(coverage) + selected_optional_count <= max_facts
        and count_tokens(result) <= max_tokens,
        "query-fact renderer exceeded its deterministic bounds",
    )
    assert_gold_blind({"rendered_ledger": result}, path="hot_v6_renderer")
    selected_ids = {row.fact_id for row in selected_facts}
    mandatory_fact_ids = tuple(
        row.fact_id for row in selected_facts if row.fact_id in coverage
    )
    omitted_fact_ids = tuple(
        row.fact_id for row in ledger.facts if row.fact_id not in selected_ids
    )
    return FactLedgerSlice(
        question_sha256=ledger.question_sha256,
        ledger_receipt_sha256=ledger.receipt_sha256,
        compiled_fact_count=len(ledger.facts),
        ledger_fact_ids=tuple(row.fact_id for row in ledger.facts),
        facts=selected_facts,
        mandatory_fact_ids=mandatory_fact_ids,
        omitted_fact_ids=omitted_fact_ids,
        max_facts=max_facts,
        max_tokens=max_tokens,
        rendered_text=result,
        rendered_text_sha256=quote_sha256(result),
        rendered_token_count=count_tokens(result),
    )


def render_query_fact_ledger(
    ledger: QueryFactLedger,
    *,
    max_facts: int = DEFAULT_MAX_RENDERED_FACTS,
    max_tokens: int = DEFAULT_MAX_RENDERED_TOKENS,
) -> str:
    """Return the compatible text view of a sealed bounded fact slice."""

    if type(ledger) is not QueryFactLedger:
        raise TypeError("renderer requires an exact QueryFactLedger")
    return select_and_render_query_fact_ledger(
        ledger,
        max_facts=max_facts,
        max_tokens=max_tokens,
    ).rendered_text


__all__ = [
    "DEDUP_POLICY",
    "DEFAULT_MAX_RENDERED_FACTS",
    "DEFAULT_MAX_RENDERED_TOKENS",
    "FORMAT",
    "RAW_AUTHORITY_POLICY",
    "SLICE_FORMAT",
    "FactLedgerSlice",
    "FactOrigin",
    "FactStatus",
    "QueryFactLedger",
    "QueryFactLedgerError",
    "RequiredSlotBinding",
    "TypedExactQuoteFact",
    "TypedNumericOperand",
    "compile_query_fact_ledger",
    "render_query_fact_ledger",
    "select_and_render_query_fact_ledger",
]
