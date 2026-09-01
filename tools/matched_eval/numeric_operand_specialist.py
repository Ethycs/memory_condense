"""Provider-free numeric operand closure over an immutable resident index.

The general full-store scanner ranks sentence evidence.  Numeric reductions
need a different unit of recall: every plausible first-person event operand.
This module performs that narrow job without a provider, a question ID, gold
labels, or persisted transformer state.

Selection is intentionally staged.  Numeric, event, source-diverse, seeded,
and per-action lanes select independently; exact spans are deduplicated only
after those selections.  A later event grouping step does *not* discard
citations.  It merely tells a downstream reducer that repeated mentions such
as "I got a new coffee table" are support for one purchase, while buying a
mattress, assembling a bookshelf, and fixing a kitchen table are separate
operands.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal, Sequence

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
    LocalCitationBinding,
    indexed_surface_terms,
)
from .query_guided_scan import CachedContentRow
from .typed_numeric_semantics import (
    NumericDimension,
    NumericMention,
    numeric_mentions,
)
from .typed_operator_spec import (
    ComparisonMode,
    TypedOperatorSpec,
    compile_typed_operator_spec,
)


MECHANISM_ID = "numeric_operand_closure_specialist_v1"
RESULT_FORMAT = "memory-condense-numeric-operand-closure-result-v1"
CANDIDATE_FORMAT = "memory-condense-numeric-operand-candidate-v1"
MENTION_FORMAT = "memory-condense-numeric-operand-mention-v1"
GROUP_FORMAT = "memory-condense-numeric-operand-group-v1"
RECEIPT_FORMAT = "memory-condense-numeric-operand-closure-receipt-v1"


class NumericOperandSpecialistError(MatchedEvalContractError):
    """Raised when numeric operand closure loses a sealed invariant."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise NumericOperandSpecialistError(message)


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(value) is str and value for value in values)
        and len(values) == len(set(values)),
        f"{label} must be an ordered unique exact tuple",
    )
    return values


@dataclass(frozen=True, slots=True)
class NumericOperandBudget:
    """Independent lane caps plus a complete provider-envelope bound."""

    evidence_token_cap: int = 1_800
    max_candidates: int = 36
    max_candidates_per_lane: int = 24
    lane_token_cap: int = 1_200
    max_candidates_per_source: int = 8
    max_window_sentences: int = 2
    max_excerpt_tokens: int = 192
    max_operand_groups: int = 32
    hard_prompt_token_cap: int = 8_000
    output_token_reserve: int = 768
    protocol_token_reserve: int = 512

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            _require(type(value) is int and value > 0, f"{name} must be positive")
        _require(
            self.lane_token_cap <= self.evidence_token_cap,
            "lane token cap exceeds the terminal evidence cap",
        )
        _require(
            self.max_candidates_per_lane <= self.max_candidates,
            "lane candidate cap exceeds the terminal candidate cap",
        )
        _require(
            self.max_candidates_per_source <= self.max_candidates,
            "per-source cap exceeds the terminal candidate cap",
        )
        _require(
            self.evidence_token_cap
            + self.output_token_reserve
            + self.protocol_token_reserve
            <= self.hard_prompt_token_cap,
            "numeric specialist reserves exceed the hard prompt cap",
        )

    @property
    def provider_payload_token_cap(self) -> int:
        return (
            self.hard_prompt_token_cap
            - self.output_token_reserve
            - self.protocol_token_reserve
        )

    def projection(self) -> dict[str, int]:
        return {
            **asdict(self),
            "provider_payload_token_cap": self.provider_payload_token_cap,
        }

    @property
    def budget_id(self) -> str:
        return identity_sha256(
            {"mechanism_id": MECHANISM_ID, "budget": self.projection()}
        )


@dataclass(frozen=True, slots=True)
class NumericOperandMention:
    value: float
    dimension: str
    qualifier: str
    unit: str | None
    start_char: int
    end_char: int
    surface: str

    def __post_init__(self) -> None:
        _require(type(self.value) in {int, float}, "numeric value changed type")
        require_text(self.dimension, "numeric dimension")
        require_text(self.qualifier, "numeric qualifier")
        if self.unit is not None:
            require_text(self.unit, "numeric unit")
        _require(
            type(self.start_char) is int
            and type(self.end_char) is int
            and 0 <= self.start_char < self.end_char,
            "numeric mention coordinates changed",
        )
        require_text(self.surface, "numeric surface")

    @classmethod
    def from_numeric_mention(cls, row: NumericMention) -> "NumericOperandMention":
        return cls(
            value=row.value,
            dimension=row.dimension.value,
            qualifier=row.qualifier.value,
            unit=row.unit,
            start_char=row.start,
            end_char=row.end,
            surface=row.surface,
        )

    def projection(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "end_char": self.end_char,
            "format": MENTION_FORMAT,
            "qualifier": self.qualifier,
            "start_char": self.start_char,
            "surface": self.surface,
            "unit": self.unit,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class NumericOperandCandidate:
    """Provider-visible quote with an opaque source and exact local binding."""

    candidate_id: str
    source_group_handle: str
    quote: str
    quote_sha256: str
    token_count: int
    role: Literal["user"]
    created_at: str
    action_classes: tuple[str, ...]
    entity_keys: tuple[str, ...]
    numeric_mentions: tuple[NumericOperandMention, ...]
    operand_group_ids: tuple[str, ...]
    selection_lanes: tuple[str, ...]
    temporal_distance_days: int | None
    temporal_window_match: bool | None
    citation_binding_receipt_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.candidate_id, "numeric candidate")
        _require(
            re.fullmatch(r"G[0-9]{4,6}", self.source_group_handle) is not None,
            "numeric source group handle must be opaque",
        )
        require_text(self.quote, "numeric candidate quote")
        require_sha256(self.quote_sha256, "numeric candidate quote")
        _require(
            self.quote_sha256 == quote_sha256(self.quote)
            and self.token_count == count_tokens(self.quote),
            "numeric candidate quote surface changed",
        )
        _require(self.role == "user", "numeric operand must be a user assertion")
        require_text(self.created_at, "numeric candidate timestamp")
        _ordered_unique(self.action_classes, "numeric action classes")
        _ordered_unique(self.entity_keys, "numeric entity keys")
        _ordered_unique(self.operand_group_ids, "numeric operand group IDs")
        _ordered_unique(self.selection_lanes, "numeric selection lanes")
        _require(
            self.action_classes and self.entity_keys and self.operand_group_ids,
            "numeric candidate lost its event operands",
        )
        _require(
            type(self.numeric_mentions) is tuple
            and all(type(row) is NumericOperandMention for row in self.numeric_mentions)
            and all(
                self.quote[row.start_char : row.end_char] == row.surface
                for row in self.numeric_mentions
            ),
            "numeric candidate mention surface changed",
        )
        if self.temporal_distance_days is not None:
            _require(
                type(self.temporal_distance_days) is int,
                "numeric temporal distance changed",
            )
        _require(
            self.temporal_window_match in {None, True, False},
            "numeric temporal match changed",
        )
        require_sha256(
            self.citation_binding_receipt_sha256,
            "numeric candidate citation binding",
        )

    def projection(self) -> dict[str, Any]:
        return {
            "action_classes": list(self.action_classes),
            "candidate_id": self.candidate_id,
            "citation_binding_receipt_sha256": (
                self.citation_binding_receipt_sha256
            ),
            "created_at": self.created_at,
            "entity_keys": list(self.entity_keys),
            "format": CANDIDATE_FORMAT,
            "numeric_mentions": [row.projection() for row in self.numeric_mentions],
            "operand_group_ids": list(self.operand_group_ids),
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "selection_lanes": list(self.selection_lanes),
            "source_group_handle": self.source_group_handle,
            "temporal_distance_days": self.temporal_distance_days,
            "temporal_window_match": self.temporal_window_match,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class NumericOperandGroup:
    """One distinct event operand with every retained supporting citation."""

    operand_group_id: str
    operation_mode: str
    action_class: str
    entity_key: str
    operand_values: tuple[float, ...]
    value_basis: str
    candidate_ids: tuple[str, ...]
    source_group_handles: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(self.operand_group_id, "numeric operand group")
        require_text(self.operation_mode, "numeric operation mode")
        require_text(self.action_class, "numeric group action")
        require_text(self.entity_key, "numeric group entity")
        _require(
            type(self.operand_values) is tuple
            and self.operand_values
            and all(type(value) in {int, float} for value in self.operand_values),
            "numeric operand values changed",
        )
        require_text(self.value_basis, "numeric group value basis")
        _ordered_unique(self.candidate_ids, "numeric group candidates")
        _ordered_unique(self.source_group_handles, "numeric group sources")
        _require(
            self.candidate_ids and self.source_group_handles,
            "numeric operand group lost all supporting citations",
        )

    def projection(self) -> dict[str, Any]:
        return {
            "action_class": self.action_class,
            "candidate_ids": list(self.candidate_ids),
            "entity_key": self.entity_key,
            "format": GROUP_FORMAT,
            "operand_group_id": self.operand_group_id,
            "operand_values": list(self.operand_values),
            "operation_mode": self.operation_mode,
            "source_group_handles": list(self.source_group_handles),
            "value_basis": self.value_basis,
        }


@dataclass(frozen=True, slots=True)
class NumericOperandClosureReceipt:
    question_sha256: str
    operator_spec_receipt_sha256: str
    window_index_receipt_sha256: str
    cache_receipt_sha256: str
    budget_id: str
    operation_mode: str
    expected_numeric_dimension: str
    question_action_classes: tuple[str, ...]
    question_entity_domain: str
    temporal_window_days: int | None
    seed_source_count: int
    seed_history_count: int
    seed_inventory_sha256: str
    physical_content_rows_scanned: int
    physical_sentence_windows_scanned: int
    user_sentence_windows_scanned: int
    first_person_window_variant_count: int
    candidate_population_count: int
    lane_selected_counts: tuple[tuple[str, int], ...]
    independent_lane_selected_occurrence_count: int
    post_selection_exact_span_count: int
    exact_span_duplicate_count: int
    plausible_operand_group_count: int
    selected_operand_group_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    selected_source_group_count: int
    selected_evidence_tokens: int
    provider_payload_tokens: int
    multi_mention_operand_group_count: int
    all_plausible_operand_groups_reserved: bool
    selection_truncated: bool
    semantic_completeness_status: Literal["not_claimed"] = "not_claimed"
    physical_scan_exhaustive: Literal[True] = True
    exact_span_dedup_stage: Literal["after_independent_lane_selection"] = (
        "after_independent_lane_selection"
    )
    semantic_grouping_discards_citations: Literal[False] = False
    question_id_filter_used: Literal[False] = False
    known_source_prefix_filter_used: Literal[False] = False
    partition_routing_used: Literal[False] = False
    raw_source_ids_provider_visible: Literal[False] = False
    new_provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.question_sha256, "numeric question"),
            (self.operator_spec_receipt_sha256, "numeric operator spec"),
            (self.window_index_receipt_sha256, "numeric window index"),
            (self.cache_receipt_sha256, "numeric cache"),
            (self.budget_id, "numeric budget"),
            (self.seed_inventory_sha256, "numeric seed inventory"),
        ):
            require_sha256(value, label)
        require_text(self.operation_mode, "numeric receipt operation")
        require_text(self.expected_numeric_dimension, "numeric receipt dimension")
        require_text(self.question_entity_domain, "numeric entity domain")
        _ordered_unique(self.question_action_classes, "question action classes")
        if self.temporal_window_days is not None:
            _require(
                type(self.temporal_window_days) is int
                and self.temporal_window_days > 0,
                "numeric temporal window changed",
            )
        for value, label in (
            (self.seed_source_count, "numeric seed sources"),
            (self.seed_history_count, "numeric seed histories"),
            (self.physical_content_rows_scanned, "numeric physical rows"),
            (self.physical_sentence_windows_scanned, "numeric physical windows"),
            (self.user_sentence_windows_scanned, "numeric user windows"),
            (self.first_person_window_variant_count, "numeric first-person variants"),
            (self.candidate_population_count, "numeric candidate population"),
            (
                self.independent_lane_selected_occurrence_count,
                "numeric independent lane occurrences",
            ),
            (self.post_selection_exact_span_count, "numeric post-dedup spans"),
            (self.exact_span_duplicate_count, "numeric exact duplicates"),
            (self.plausible_operand_group_count, "numeric plausible groups"),
            (self.selected_source_group_count, "numeric selected sources"),
            (self.selected_evidence_tokens, "numeric selected evidence tokens"),
            (self.provider_payload_tokens, "numeric provider payload tokens"),
            (
                self.multi_mention_operand_group_count,
                "numeric repeated-mention groups",
            ),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        _require(
            type(self.lane_selected_counts) is tuple
            and all(
                type(row) is tuple
                and len(row) == 2
                and type(row[0]) is str
                and row[0]
                and type(row[1]) is int
                and row[1] >= 0
                for row in self.lane_selected_counts
            )
            and len({row[0] for row in self.lane_selected_counts})
            == len(self.lane_selected_counts),
            "numeric lane counts changed",
        )
        _ordered_unique(self.selected_operand_group_ids, "selected numeric groups")
        _ordered_unique(self.selected_candidate_ids, "selected numeric candidates")
        _require(
            self.independent_lane_selected_occurrence_count
            == sum(row[1] for row in self.lane_selected_counts)
            and self.exact_span_duplicate_count
            == self.independent_lane_selected_occurrence_count
            - self.post_selection_exact_span_count,
            "numeric post-selection dedup accounting changed",
        )
        _require(
            len(self.selected_operand_group_ids)
            <= self.plausible_operand_group_count
            and self.all_plausible_operand_groups_reserved
            is (
                len(self.selected_operand_group_ids)
                == self.plausible_operand_group_count
            ),
            "numeric operand reservation accounting changed",
        )
        _require(
            type(self.selection_truncated) is bool
            and self.semantic_completeness_status == "not_claimed"
            and self.physical_scan_exhaustive is True
            and self.exact_span_dedup_stage
            == "after_independent_lane_selection"
            and self.semantic_grouping_discards_citations is False,
            "numeric scan semantics changed",
        )
        _require(
            self.question_id_filter_used is False
            and self.known_source_prefix_filter_used is False
            and self.partition_routing_used is False
            and self.raw_source_ids_provider_visible is False
            and self.new_provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "numeric specialist used a forbidden runtime path",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "numeric receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="numeric_operand_receipt")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "all_plausible_operand_groups_reserved": (
                self.all_plausible_operand_groups_reserved
            ),
            "budget_id": self.budget_id,
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "candidate_population_count": self.candidate_population_count,
            "exact_span_dedup_stage": self.exact_span_dedup_stage,
            "exact_span_duplicate_count": self.exact_span_duplicate_count,
            "expected_numeric_dimension": self.expected_numeric_dimension,
            "first_person_window_variant_count": (
                self.first_person_window_variant_count
            ),
            "format": RECEIPT_FORMAT,
            "gold_loaded": False,
            "independent_lane_selected_occurrence_count": (
                self.independent_lane_selected_occurrence_count
            ),
            "known_source_prefix_filter_used": False,
            "lane_selected_counts": [
                {"lane": lane, "selected_count": count}
                for lane, count in self.lane_selected_counts
            ],
            "multi_mention_operand_group_count": (
                self.multi_mention_operand_group_count
            ),
            "new_provider_calls": 0,
            "operation_mode": self.operation_mode,
            "operator_spec_receipt_sha256": self.operator_spec_receipt_sha256,
            "partition_routing_used": False,
            "physical_content_rows_scanned": self.physical_content_rows_scanned,
            "physical_scan_exhaustive": True,
            "physical_sentence_windows_scanned": (
                self.physical_sentence_windows_scanned
            ),
            "plausible_operand_group_count": self.plausible_operand_group_count,
            "post_selection_exact_span_count": (
                self.post_selection_exact_span_count
            ),
            "provider_payload_tokens": self.provider_payload_tokens,
            "question_action_classes": list(self.question_action_classes),
            "question_entity_domain": self.question_entity_domain,
            "question_id_filter_used": False,
            "question_sha256": self.question_sha256,
            "raw_source_ids_provider_visible": False,
            "retained_transformer_token_state_bytes": 0,
            "seed_history_count": self.seed_history_count,
            "seed_inventory_sha256": self.seed_inventory_sha256,
            "seed_source_count": self.seed_source_count,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_evidence_tokens": self.selected_evidence_tokens,
            "selected_operand_group_ids": list(self.selected_operand_group_ids),
            "selected_source_group_count": self.selected_source_group_count,
            "selection_truncated": self.selection_truncated,
            "semantic_completeness_status": "not_claimed",
            "semantic_grouping_discards_citations": False,
            "temporal_window_days": self.temporal_window_days,
            "user_sentence_windows_scanned": self.user_sentence_windows_scanned,
            "window_index_receipt_sha256": self.window_index_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class NumericOperandClosureResult:
    dated_question: str
    operator_spec: TypedOperatorSpec
    operation_mode: str
    expected_numeric_dimension: str
    candidates: tuple[NumericOperandCandidate, ...]
    operand_groups: tuple[NumericOperandGroup, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    receipt: NumericOperandClosureReceipt
    budget: NumericOperandBudget

    def __post_init__(self) -> None:
        require_text(self.dated_question, "numeric dated question")
        _require(
            type(self.operator_spec) is TypedOperatorSpec,
            "numeric result operator spec changed",
        )
        require_text(self.operation_mode, "numeric result operation")
        require_text(self.expected_numeric_dimension, "numeric result dimension")
        _require(
            type(self.candidates) is tuple
            and all(type(row) is NumericOperandCandidate for row in self.candidates)
            and type(self.operand_groups) is tuple
            and all(type(row) is NumericOperandGroup for row in self.operand_groups)
            and type(self.local_bindings) is tuple
            and all(type(row) is LocalCitationBinding for row in self.local_bindings),
            "numeric result collection types changed",
        )
        candidate_ids = tuple(row.candidate_id for row in self.candidates)
        binding_ids = tuple(row.candidate_id for row in self.local_bindings)
        group_ids = tuple(row.operand_group_id for row in self.operand_groups)
        _require(
            candidate_ids
            == binding_ids
            == self.receipt.selected_candidate_ids
            and group_ids == self.receipt.selected_operand_group_ids,
            "numeric result lost exact candidate/group bindings",
        )
        _require(
            all(
                candidate.citation_binding_receipt_sha256 == binding.receipt_sha256
                and candidate.source_group_handle == binding.source_group_handle
                for candidate, binding in zip(
                    self.candidates, self.local_bindings, strict=True
                )
            ),
            "numeric candidate citation changed",
        )
        known_candidates = set(candidate_ids)
        _require(
            all(set(row.candidate_ids) <= known_candidates for row in self.operand_groups),
            "numeric operand group cites an absent candidate",
        )
        provider_tokens = count_tokens(_canonical_json(self.provider_projection()))
        _require(
            provider_tokens == self.receipt.provider_payload_tokens
            and provider_tokens <= self.budget.provider_payload_token_cap,
            "numeric provider payload exceeds its hard envelope",
        )
        assert_gold_blind(
            self.provider_projection(), path="numeric_operand_provider_payload"
        )

    def provider_projection(self) -> dict[str, Any]:
        return {
            "candidates": [row.projection() for row in self.candidates],
            "dated_question": self.dated_question,
            "expected_numeric_dimension": self.expected_numeric_dimension,
            "format": RESULT_FORMAT,
            "operand_groups": [row.projection() for row in self.operand_groups],
            "operation_mode": self.operation_mode,
            "operator_spec": self.operator_spec.projection(),
        }

    def local_audit_projection(self) -> dict[str, Any]:
        return {
            "bindings": [row.projection() for row in self.local_bindings],
            "provider_payload_sha256": identity_sha256(self.provider_projection()),
            "receipt": self.receipt.projection(),
        }


@dataclass(frozen=True, slots=True)
class _ActionHit:
    action_class: str
    start: int
    end: int
    surface: str


@dataclass(frozen=True, slots=True)
class _EntityHit:
    entity_key: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Event:
    action_class: str
    entity_key: str
    mentions: tuple[NumericMention, ...]


@dataclass(frozen=True, slots=True)
class _Draft:
    candidate_id: str
    row: CachedContentRow
    start_char: int
    end_char: int
    quote: str
    events: tuple[_Event, ...]
    mentions: tuple[NumericMention, ...]
    anchor_overlap: int
    seeded: bool
    temporal_distance_days: int | None
    temporal_window_match: bool | None

    @property
    def token_count(self) -> int:
        return count_tokens(self.quote)

    @property
    def score(self) -> tuple[Any, ...]:
        numeric_events = sum(bool(row.mentions) for row in self.events)
        return (
            numeric_events,
            len(self.events),
            int(self.seeded),
            int(self.temporal_window_match is True),
            len(self.mentions),
            self.anchor_overlap,
            -self.token_count,
            self.row.ordinal,
            -self.start_char,
            self.candidate_id,
        )


@dataclass(frozen=True, slots=True)
class _SelectedDraft:
    draft: _Draft
    lanes: tuple[str, ...]


@dataclass(slots=True)
class _GroupDraft:
    operand_group_id: str
    operation_mode: str
    action_class: str
    entity_key: str
    operand_values: tuple[float, ...]
    value_basis: str
    candidate_ids: list[str]


_DATED_RE = re.compile(r"^\[Question asked at (?P<asked_at>.+?)\]\s*", re.I | re.S)
_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|I'm|I’ve|I've|I'd|I’d|my|mine|we|we've|our|ours)\b", re.I
)

_QUESTION_ACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "buy",
        re.compile(
            r"\b(?:buy|bought|purchase|purchased|order|ordered|"
            r"acquire|acquired|get|got)\b",
            re.I,
        ),
    ),
    ("assemble", re.compile(r"\b(?:assemble|assembled|build|built|put\s+together)\b", re.I)),
    ("sell", re.compile(r"\b(?:sell|sold|resell|resold)\b", re.I)),
    ("fix", re.compile(r"\b(?:fix|fixed|repair|repaired|mend|mended)\b", re.I)),
)
_EVIDENCE_ACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "buy",
        re.compile(
            r"\b(?:bought|purchased|ordered|acquired|picked\s+up|paid\s+for|"
            r"got(?!\s+around\s+to))\b",
            re.I,
        ),
    ),
    ("assemble", re.compile(r"\b(?:assembled|built|put\s+together)\b", re.I)),
    ("sell", re.compile(r"\b(?:sold|resold)\b", re.I)),
    (
        "fix",
        re.compile(r"\b(?:fixed|fixing|repaired|mended|tightened)\b", re.I),
    ),
)

_FURNITURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (key, re.compile(pattern, re.I))
    for key, pattern in (
        ("coffee_table", r"\bcoffee\s+table\b"),
        ("kitchen_table", r"\bkitchen\s+table\b"),
        ("dining_table", r"\bdining(?:\s+room)?\s+table\b"),
        ("bedside_table", r"\b(?:bedside\s+table|nightstand)\b"),
        ("bookshelf", r"\b(?:book\s*shelf|bookshelf|bookcase)\b"),
        ("mattress", r"\bmattress\b"),
        ("sofa", r"\b(?:sofa|couch)\b"),
        ("armchair", r"\barmchair\b"),
        ("chair", r"\bchair\b"),
        ("desk", r"\bdesk\b"),
        ("dresser", r"\bdresser\b"),
        ("cabinet", r"\bcabinet\b"),
        ("wardrobe", r"\bwardrobe\b"),
        ("ottoman", r"\b(?:ottoman|footstool)\b"),
        ("bench", r"\bbench\b"),
        ("stool", r"\bstool\b"),
        ("bed", r"\bbed\b"),
        ("table", r"\btable\b"),
    )
)
_FEED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (key, re.compile(pattern, re.I))
    for key, pattern in (
        ("layer_feed", r"\blayer\s+feed\b"),
        ("scratch_grain", r"\b(?:organic\s+)?scratch\s+grains?\b"),
        ("chicken_feed", r"\b(?:chicken|poultry)\s+feed\b"),
        ("grain_feed", r"\bgrains?\b"),
        ("feed", r"\bfeed\b"),
    )
)
_QUERY_FUNCTION_TERMS = frozenset(
    {
        "aggregate",
        "all",
        "altogether",
        "amount",
        "combined",
        "difference",
        "few",
        "many",
        "month",
        "new",
        "number",
        "past",
        "piece",
        "recent",
        "recently",
        "total",
        "two",
        "weight",
    }
)
_ACTION_TERMS = frozenset(
    term
    for value in (
        "buy bought purchase purchased order ordered acquire acquired get got "
        "assemble assembled build built sell sold fix fixed repair repaired"
    ).split()
    for term in (value, value.removesuffix("ed"))
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _question_body(dated_question: str) -> str:
    return _DATED_RE.sub("", dated_question).strip()


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    cleaned = value.strip().replace("/", "-")
    cleaned = re.sub(r"\s*\([A-Za-z]{3,9}\)\s*", " ", cleaned).strip()
    cleaned = cleaned.removesuffix("Z")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
    return None


def _operation_mode(body: str, spec: TypedOperatorSpec) -> str:
    if spec.comparison_mode is not ComparisonMode.NONE or re.search(
        r"\b(?:difference|how\s+much\s+(?:more|less)|compared\s+to)\b", body, re.I
    ):
        return "difference_or_compare"
    if re.search(r"\b(?:total|sum|combined|altogether|in\s+all)\b", body, re.I):
        return "sum"
    if re.search(r"\bhow\s+many\b|\bnumber\s+of\b", body, re.I):
        return "count"
    return "numeric_reduce"


def _expected_dimension(body: str, operation_mode: str) -> NumericDimension:
    if re.search(r"\b(?:weight|weigh|pounds?|lbs?|kilograms?|kgs?)\b", body, re.I):
        return NumericDimension.MEASURE
    if re.search(r"\b(?:cost|price|spend|spent|pay|paid|dollars?|usd)\b|\$", body, re.I):
        return NumericDimension.CURRENCY
    if re.search(r"\b(?:percent|percentage|rate)\b|%", body, re.I):
        return NumericDimension.PERCENTAGE
    if re.search(r"\bhow\s+long\b|\bduration\b", body, re.I):
        return NumericDimension.DURATION
    if operation_mode == "count":
        return NumericDimension.COUNT
    return NumericDimension.GENERIC


def _question_actions(body: str) -> tuple[str, ...]:
    actions = tuple(
        action for action, pattern in _QUESTION_ACTION_PATTERNS if pattern.search(body)
    )
    return tuple(dict.fromkeys(actions))


def _question_domain(body: str) -> str:
    if re.search(r"\bfurniture\b", body, re.I):
        return "furniture"
    if re.search(r"\b(?:feed|grain|grains|fodder)\b", body, re.I):
        return "feed"
    return "question_anchor"


def _question_anchors(body: str) -> tuple[str, ...]:
    return tuple(
        term
        for term in indexed_surface_terms(body)
        if term not in _QUERY_FUNCTION_TERMS and term not in _ACTION_TERMS
    )


def _temporal_window_days(body: str, spec: TypedOperatorSpec) -> int | None:
    if spec.temporal_window_days is not None:
        return spec.temporal_window_days
    match = re.search(r"\bpast\s+(\d+)\s+months?\b", body, re.I)
    if match is not None:
        return int(match.group(1)) * 31
    if re.search(r"\bpast\s+(?:few|couple(?:\s+of)?)\s+months?\b", body, re.I):
        return 124
    if re.search(r"\bpast\s+(?:several)\s+months?\b", body, re.I):
        return 186
    match = re.search(r"\bpast\s+(\d+)\s+weeks?\b", body, re.I)
    if match is not None:
        return int(match.group(1)) * 7
    if re.search(r"\bpast\s+month\b", body, re.I):
        return 31
    if re.search(r"\brecent(?:ly)?\b", body, re.I):
        return 93
    return None


def _temporal_fit(
    row: CachedContentRow, asked_at: datetime | None, window_days: int | None
) -> tuple[int | None, bool | None]:
    if asked_at is None or window_days is None:
        return None, None
    event_at = _parse_datetime(row.created_at)
    if event_at is None:
        return None, None
    distance = (asked_at.date() - event_at.date()).days
    return distance, bool(0 <= distance <= window_days)


def _evidence_actions(text: str, allowed: tuple[str, ...]) -> tuple[_ActionHit, ...]:
    permitted = set(allowed) if allowed else {
        action for action, _pattern in _EVIDENCE_ACTION_PATTERNS
    }
    output: list[_ActionHit] = []
    for action, pattern in _EVIDENCE_ACTION_PATTERNS:
        if action not in permitted:
            continue
        for match in pattern.finditer(text):
            surface = match.group(0).casefold()
            if surface == "got" and re.match(
                r"\s+(?:married|engaged|used\s+to|back|sick|tired|lost)\b",
                text[match.end() :],
                re.I,
            ):
                continue
            output.append(
                _ActionHit(action, match.start(), match.end(), surface)
            )
    return tuple(sorted(output, key=lambda row: (row.start, row.end, row.action_class)))


def _non_overlapping_entities(
    text: str, patterns: Sequence[tuple[str, re.Pattern[str]]]
) -> tuple[_EntityHit, ...]:
    output: list[_EntityHit] = []
    occupied: list[tuple[int, int]] = []
    for key, pattern in patterns:
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            output.append(_EntityHit(key, span[0], span[1]))
            occupied.append(span)
    return tuple(sorted(output, key=lambda row: (row.start, row.end, row.entity_key)))


def _entities(
    text: str, domain: str, anchors: tuple[str, ...]
) -> tuple[_EntityHit, ...]:
    if domain == "furniture":
        return _non_overlapping_entities(text, _FURNITURE_PATTERNS)
    if domain == "feed":
        return _non_overlapping_entities(text, _FEED_PATTERNS)
    output: list[_EntityHit] = []
    folded = text.casefold()
    for anchor in anchors:
        for match in re.finditer(rf"\b{re.escape(anchor)}\b", folded):
            output.append(_EntityHit(anchor.replace("'", ""), *match.span()))
    return tuple(sorted(output, key=lambda row: (row.start, row.end, row.entity_key)))


def _candidate_mentions(
    text: str, dimension: NumericDimension
) -> tuple[NumericMention, ...]:
    return numeric_mentions(text, expected_dimension=dimension)


def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0


def _events(
    actions: tuple[_ActionHit, ...],
    entities: tuple[_EntityHit, ...],
    mentions: tuple[NumericMention, ...],
) -> tuple[_Event, ...]:
    if not actions or not entities:
        return ()
    pairs: list[tuple[str, str, tuple[int, int]]] = []
    for entity in entities:
        nearest = min(
            actions,
            key=lambda row: (
                _distance((row.start, row.end), (entity.start, entity.end)),
                row.start,
                row.action_class,
            ),
        )
        key = (nearest.action_class, entity.entity_key)
        # ``got`` is highly polysemous.  Treat it as acquisition only when
        # the furniture/feed entity is its nearby following object.  Other
        # completed purchase verbs may follow their object ("feed I
        # purchased"), so this narrow grammar rule is deliberately verb-only.
        if nearest.surface == "got" and not (
            nearest.end <= entity.start <= nearest.end + 64
        ):
            continue
        if any((action, name) == key for action, name, _span in pairs):
            continue
        pairs.append(
            (
                nearest.action_class,
                entity.entity_key,
                (min(nearest.start, entity.start), max(nearest.end, entity.end)),
            )
        )
    if not pairs:
        return ()
    assigned: list[list[NumericMention]] = [[] for _row in pairs]
    for mention in mentions:
        index = min(
            range(len(pairs)),
            key=lambda value: (
                _distance(
                    pairs[value][2],
                    (mention.start, mention.end),
                ),
                value,
            ),
        )
        assigned[index].append(mention)
    return tuple(
        _Event(action, entity, tuple(values))
        for (action, entity, _span), values in zip(pairs, assigned, strict=True)
    )


def _is_seeded(source_id: str, seeds: frozenset[str]) -> bool:
    return any(
        source_id == seed or source_id.endswith(f"::{seed}") for seed in seeds
    )


def _span(row: CachedContentRow, start: int, end: int) -> EvidenceSpan:
    quote = row.text[start:end]
    return EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=start,
        end_char=end,
        quote_sha256=quote_sha256(quote),
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )


def _candidate_population(
    index: FullStoreWindowIndex,
    *,
    body: str,
    operation_mode: str,
    dimension: NumericDimension,
    allowed_actions: tuple[str, ...],
    domain: str,
    anchors: tuple[str, ...],
    asked_at: datetime | None,
    temporal_days: int | None,
    seeds: frozenset[str],
    budget: NumericOperandBudget,
) -> tuple[tuple[_Draft, ...], int, int]:
    drafts: dict[str, _Draft] = {}
    user_windows = 0
    first_person_variants = 0
    for row in index.rows:
        if row.role.casefold() != "user":
            continue
        user_windows += len(row.sentence_windows)
        distance, temporal_match = _temporal_fit(row, asked_at, temporal_days)
        if temporal_match is False:
            continue
        windows = row.sentence_windows
        for first in range(len(windows)):
            # Distinct-event counts already have their complete unit in one
            # assertion.  Crossing a sentence boundary can incorrectly bind
            # the next action to an entity mentioned only as background.
            # Numeric reductions retain the bounded adjacent-sentence join so
            # "purchased feed. I got a 50-pound batch" stays one operand.
            sentence_width = (
                1 if operation_mode == "count" else budget.max_window_sentences
            )
            last_limit = min(
                len(windows), first + sentence_width
            )
            for last in range(first, last_limit):
                start = windows[first].start_char
                end = windows[last].end_char
                quote = row.text[start:end]
                if count_tokens(quote) > budget.max_excerpt_tokens:
                    continue
                if _FIRST_PERSON_RE.search(quote) is None:
                    continue
                first_person_variants += 1
                actions = _evidence_actions(quote, allowed_actions)
                entities = _entities(quote, domain, anchors)
                mentions = _candidate_mentions(quote, dimension)
                events = _events(actions, entities, mentions)
                if not events:
                    continue
                if operation_mode in {"sum", "difference_or_compare", "numeric_reduce"}:
                    if not any(event.mentions for event in events):
                        continue
                span = _span(row, start, end)
                candidate_id = identity_sha256(
                    {"atom_id": make_atom_id(span), "mechanism_id": MECHANISM_ID}
                )
                terms = set(indexed_surface_terms(quote))
                draft = _Draft(
                    candidate_id=candidate_id,
                    row=row,
                    start_char=start,
                    end_char=end,
                    quote=quote,
                    events=events,
                    mentions=mentions,
                    anchor_overlap=len(terms & set(anchors)),
                    seeded=_is_seeded(row.source_id, seeds),
                    temporal_distance_days=distance,
                    temporal_window_match=temporal_match,
                )
                previous = drafts.get(candidate_id)
                if previous is None or draft.score > previous.score:
                    drafts[candidate_id] = draft
    return (
        tuple(sorted(drafts.values(), key=lambda row: row.score, reverse=True)),
        user_windows,
        first_person_variants,
    )


def _select_lane(
    drafts: Sequence[_Draft], budget: NumericOperandBudget
) -> tuple[_Draft, ...]:
    selected: list[_Draft] = []
    source_counts: dict[str, int] = {}
    tokens = 0
    for draft in sorted(drafts, key=lambda row: row.score, reverse=True):
        if (
            len(selected) >= budget.max_candidates_per_lane
            or source_counts.get(draft.row.source_id, 0)
            >= budget.max_candidates_per_source
            or tokens + draft.token_count > budget.lane_token_cap
        ):
            continue
        selected.append(draft)
        tokens += draft.token_count
        source_counts[draft.row.source_id] = (
            source_counts.get(draft.row.source_id, 0) + 1
        )
    return tuple(selected)


def _independent_lane_selections(
    drafts: tuple[_Draft, ...],
    allowed_actions: tuple[str, ...],
    budget: NumericOperandBudget,
) -> tuple[tuple[str, tuple[_Draft, ...]], ...]:
    by_source: dict[str, _Draft] = {}
    for draft in drafts:
        by_source.setdefault(draft.row.source_id, draft)
    lanes: list[tuple[str, tuple[_Draft, ...]]] = [
        (
            "numeric_operands",
            _select_lane(
                tuple(row for row in drafts if row.mentions), budget
            ),
        ),
        ("event_operands", _select_lane(drafts, budget)),
        (
            "source_diverse_operands",
            _select_lane(tuple(by_source.values()), budget),
        ),
    ]
    seeded = tuple(row for row in drafts if row.seeded)
    if seeded:
        lanes.append(("seeded_operand_closure", _select_lane(seeded, budget)))
    action_inventory = allowed_actions or tuple(
        dict.fromkeys(
            event.action_class for draft in drafts for event in draft.events
        )
    )
    for action in action_inventory:
        lane_rows = tuple(
            row
            for row in drafts
            if any(event.action_class == action for event in row.events)
        )
        lanes.append((f"action:{action}", _select_lane(lane_rows, budget)))
    return tuple(lanes)


def _post_selection_exact_dedup(
    lanes: Sequence[tuple[str, tuple[_Draft, ...]]]
) -> tuple[_SelectedDraft, ...]:
    selected: dict[
        tuple[str, str, int, int, str], tuple[_Draft, list[str]]
    ] = {}
    order: list[tuple[str, str, int, int, str]] = []
    for lane, values in lanes:
        for draft in values:
            key = (
                draft.row.source_id,
                draft.row.chunk_id,
                draft.start_char,
                draft.end_char,
                quote_sha256(draft.quote),
            )
            if key not in selected:
                selected[key] = (draft, [lane])
                order.append(key)
            elif lane not in selected[key][1]:
                selected[key][1].append(lane)
    return tuple(
        _SelectedDraft(selected[key][0], tuple(selected[key][1])) for key in order
    )


def _group_identity(
    operation_mode: str,
    event: _Event,
    mention: NumericMention | None,
) -> tuple[str, tuple[float, ...], str]:
    if mention is not None:
        body = {
            "action_class": event.action_class,
            "entity_key": event.entity_key,
            "numeric_dimension": mention.dimension.value,
            "numeric_qualifier": mention.qualifier.value,
            "numeric_unit": mention.unit,
            "numeric_value": mention.value,
            "operation_mode": operation_mode,
        }
        return identity_sha256(body), (mention.value,), "explicit_numeric_mention"
    body = {
        "action_class": event.action_class,
        "entity_key": event.entity_key,
        "operation_mode": operation_mode,
        "value_basis": "implicit_distinct_event_unit",
    }
    return identity_sha256(body), (1.0,), "implicit_distinct_event_unit"


def _build_groups(
    selected: Sequence[_SelectedDraft], operation_mode: str
) -> tuple[tuple[_GroupDraft, ...], dict[str, tuple[str, ...]]]:
    groups: dict[str, _GroupDraft] = {}
    group_order: list[str] = []
    candidate_groups: dict[str, list[str]] = {}
    for selected_row in selected:
        draft = selected_row.draft
        ids: list[str] = []
        for event in draft.events:
            mentions: tuple[NumericMention | None, ...]
            if operation_mode == "count":
                mentions = (None,)
            else:
                mentions = tuple(event.mentions)
            for mention in mentions:
                group_id, values, basis = _group_identity(
                    operation_mode, event, mention
                )
                if group_id not in groups:
                    groups[group_id] = _GroupDraft(
                        operand_group_id=group_id,
                        operation_mode=operation_mode,
                        action_class=event.action_class,
                        entity_key=event.entity_key,
                        operand_values=values,
                        value_basis=basis,
                        candidate_ids=[],
                    )
                    group_order.append(group_id)
                if draft.candidate_id not in groups[group_id].candidate_ids:
                    groups[group_id].candidate_ids.append(draft.candidate_id)
                if group_id not in ids:
                    ids.append(group_id)
        candidate_groups[draft.candidate_id] = ids
    return (
        tuple(groups[group_id] for group_id in group_order),
        {key: tuple(value) for key, value in candidate_groups.items()},
    )


def _terminal_select(
    selected: tuple[_SelectedDraft, ...],
    groups: tuple[_GroupDraft, ...],
    candidate_groups: dict[str, tuple[str, ...]],
    budget: NumericOperandBudget,
) -> tuple[_SelectedDraft, ...]:
    group_order = tuple(row.operand_group_id for row in groups)[
        : budget.max_operand_groups
    ]
    uncovered = set(group_order)
    ranked = tuple(
        sorted(selected, key=lambda row: row.draft.score, reverse=True)
    )
    output: list[_SelectedDraft] = []
    output_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    tokens = 0

    def fits(row: _SelectedDraft) -> bool:
        return bool(
            len(output) < budget.max_candidates
            and source_counts.get(row.draft.row.source_id, 0)
            < budget.max_candidates_per_source
            and tokens + row.draft.token_count <= budget.evidence_token_cap
        )

    def add(row: _SelectedDraft) -> bool:
        nonlocal tokens
        if row.draft.candidate_id in output_ids or not fits(row):
            return False
        output.append(row)
        output_ids.add(row.draft.candidate_id)
        tokens += row.draft.token_count
        source_counts[row.draft.row.source_id] = (
            source_counts.get(row.draft.row.source_id, 0) + 1
        )
        uncovered.difference_update(candidate_groups[row.draft.candidate_id])
        return True

    # Greedy set cover reserves one exact citation for every plausible group
    # before any repeated support consumes the common cap.
    while uncovered:
        choices = tuple(
            row
            for row in ranked
            if row.draft.candidate_id not in output_ids
            and set(candidate_groups[row.draft.candidate_id]) & uncovered
            and fits(row)
        )
        if not choices:
            break
        choice = max(
            choices,
            key=lambda row: (
                len(set(candidate_groups[row.draft.candidate_id]) & uncovered),
                row.draft.score,
            ),
        )
        add(choice)
    for row in ranked:
        add(row)
    return tuple(output)


def _materialize(
    selected: Sequence[_SelectedDraft],
    group_drafts: Sequence[_GroupDraft],
    candidate_groups: dict[str, tuple[str, ...]],
    index: FullStoreWindowIndex,
) -> tuple[
    tuple[NumericOperandCandidate, ...],
    tuple[NumericOperandGroup, ...],
    tuple[LocalCitationBinding, ...],
]:
    selected_ids = {row.draft.candidate_id for row in selected}
    selected_sources = tuple(
        sorted({row.draft.row.source_id for row in selected})
    )
    source_handles = {
        source: f"G{position:04d}"
        for position, source in enumerate(selected_sources, 1)
    }
    retained_group_ids = {
        group_id
        for candidate_id in selected_ids
        for group_id in candidate_groups[candidate_id]
    }
    bindings: list[LocalCitationBinding] = []
    candidates: list[NumericOperandCandidate] = []
    for selected_row in selected:
        draft = selected_row.draft
        row = draft.row
        binding = LocalCitationBinding(
            candidate_id=draft.candidate_id,
            source_group_handle=source_handles[row.source_id],
            namespace_id=index.cache.namespace_id,
            cache_receipt_sha256=index.cache.cache_receipt_sha256,
            source_database_sha256=index.cache.source_database_sha256,
            source_store_receipt_sha256=index.cache.source_store_receipt_sha256,
            source_id=row.source_id,
            partition_id=row.partition_id,
            span=_span(row, draft.start_char, draft.end_char),
            quote_sha256=quote_sha256(draft.quote),
        )
        candidates.append(
            NumericOperandCandidate(
                candidate_id=draft.candidate_id,
                source_group_handle=source_handles[row.source_id],
                quote=draft.quote,
                quote_sha256=quote_sha256(draft.quote),
                token_count=draft.token_count,
                role="user",
                created_at=row.created_at,
                action_classes=tuple(
                    dict.fromkeys(event.action_class for event in draft.events)
                ),
                entity_keys=tuple(
                    dict.fromkeys(event.entity_key for event in draft.events)
                ),
                numeric_mentions=tuple(
                    NumericOperandMention.from_numeric_mention(mention)
                    for mention in draft.mentions
                ),
                operand_group_ids=tuple(
                    group_id
                    for group_id in candidate_groups[draft.candidate_id]
                    if group_id in retained_group_ids
                ),
                selection_lanes=selected_row.lanes,
                temporal_distance_days=draft.temporal_distance_days,
                temporal_window_match=draft.temporal_window_match,
                citation_binding_receipt_sha256=binding.receipt_sha256,
            )
        )
        bindings.append(binding)
    candidate_order = {row.candidate_id: index for index, row in enumerate(candidates)}
    groups: list[NumericOperandGroup] = []
    for group in group_drafts:
        ids = tuple(
            sorted(
                (value for value in group.candidate_ids if value in selected_ids),
                key=candidate_order.__getitem__,
            )
        )
        if not ids:
            continue
        handles = tuple(
            dict.fromkeys(candidates[candidate_order[value]].source_group_handle for value in ids)
        )
        groups.append(
            NumericOperandGroup(
                operand_group_id=group.operand_group_id,
                operation_mode=group.operation_mode,
                action_class=group.action_class,
                entity_key=group.entity_key,
                operand_values=group.operand_values,
                value_basis=group.value_basis,
                candidate_ids=ids,
                source_group_handles=handles,
            )
        )
    return tuple(candidates), tuple(groups), tuple(bindings)


def _provider_projection(
    dated_question: str,
    spec: TypedOperatorSpec,
    operation_mode: str,
    dimension: NumericDimension,
    candidates: Sequence[NumericOperandCandidate],
    groups: Sequence[NumericOperandGroup],
) -> dict[str, Any]:
    return {
        "candidates": [row.projection() for row in candidates],
        "dated_question": dated_question,
        "expected_numeric_dimension": dimension.value,
        "format": RESULT_FORMAT,
        "operand_groups": [row.projection() for row in groups],
        "operation_mode": operation_mode,
        "operator_spec": spec.projection(),
    }


def scan_numeric_operand_closure(
    index: FullStoreWindowIndex,
    dated_question: str,
    /,
    *,
    operator_spec: TypedOperatorSpec | None = None,
    seed_source_ids: tuple[str, ...] = (),
    seed_history_ids: tuple[str, ...] = (),
    budget: NumericOperandBudget = NumericOperandBudget(),
) -> NumericOperandClosureResult:
    """Return bounded, exact numeric/event operands from a resident index.

    Seed IDs are optional upstream runtime hints.  They reserve a lane but are
    never used as a physical filter: every resident row is still scanned.
    The API deliberately accepts no question ID, reference, prediction,
    expected source, provider, or model state.
    """

    _require(
        type(index) is FullStoreWindowIndex,
        "numeric specialist requires an immutable resident window index",
    )
    require_text(dated_question, "numeric dated question")
    _require(type(budget) is NumericOperandBudget, "numeric budget changed type")
    _ordered_unique(seed_source_ids, "numeric seed source IDs")
    _ordered_unique(seed_history_ids, "numeric seed history IDs")
    spec = operator_spec or compile_typed_operator_spec(dated_question)
    _require(type(spec) is TypedOperatorSpec, "numeric operator spec changed type")
    _require(
        spec.question_sha256 == quote_sha256(dated_question),
        "numeric operator spec belongs to a different question",
    )
    body = _question_body(dated_question)
    operation_mode = _operation_mode(body, spec)
    dimension = _expected_dimension(body, operation_mode)
    allowed_actions = _question_actions(body)
    domain = _question_domain(body)
    anchors = _question_anchors(body)
    temporal_days = _temporal_window_days(body, spec)
    asked_at = _parse_datetime(spec.query_timestamp)
    seeds = frozenset((*seed_source_ids, *seed_history_ids))
    drafts, user_windows, first_person_variants = _candidate_population(
        index,
        body=body,
        operation_mode=operation_mode,
        dimension=dimension,
        allowed_actions=allowed_actions,
        domain=domain,
        anchors=anchors,
        asked_at=asked_at,
        temporal_days=temporal_days,
        seeds=seeds,
        budget=budget,
    )
    lanes = _independent_lane_selections(drafts, allowed_actions, budget)
    deduped = _post_selection_exact_dedup(lanes)
    # Group the complete gold-blind candidate population for audit.  Lane caps
    # may omit an operand; such an omission must make the reservation claim
    # false instead of silently shrinking the denominator to selected rows.
    population_for_grouping = tuple(
        _SelectedDraft(draft, ()) for draft in drafts
    )
    plausible_groups, candidate_groups = _build_groups(
        population_for_grouping, operation_mode
    )
    terminal = list(
        _terminal_select(deduped, plausible_groups, candidate_groups, budget)
    )

    # Candidate/evidence caps normally dominate.  This final deterministic
    # shrink proves the complete serialized provider payload also fits.
    while True:
        candidates, groups, bindings = _materialize(
            terminal, plausible_groups, candidate_groups, index
        )
        provider_tokens = count_tokens(
            _canonical_json(
                _provider_projection(
                    dated_question,
                    spec,
                    operation_mode,
                    dimension,
                    candidates,
                    groups,
                )
            )
        )
        if provider_tokens <= budget.provider_payload_token_cap:
            break
        if not terminal:
            raise NumericOperandSpecialistError(
                "numeric question/operator envelope exceeds the hard prompt budget"
            )
        terminal.pop()

    lane_counts = tuple((lane, len(values)) for lane, values in lanes)
    lane_occurrences = sum(count for _lane, count in lane_counts)
    selected_group_ids = tuple(row.operand_group_id for row in groups)
    all_groups_reserved = len(selected_group_ids) == len(plausible_groups)
    selection_truncated = bool(
        len(deduped) < len(drafts)
        or len(candidates) < len(deduped)
        or not all_groups_reserved
    )
    receipt = NumericOperandClosureReceipt(
        question_sha256=quote_sha256(dated_question),
        operator_spec_receipt_sha256=spec.receipt_sha256,
        window_index_receipt_sha256=index.receipt_sha256,
        cache_receipt_sha256=index.cache.cache_receipt_sha256,
        budget_id=budget.budget_id,
        operation_mode=operation_mode,
        expected_numeric_dimension=dimension.value,
        question_action_classes=allowed_actions,
        question_entity_domain=domain,
        temporal_window_days=temporal_days,
        seed_source_count=len(seed_source_ids),
        seed_history_count=len(seed_history_ids),
        seed_inventory_sha256=identity_sha256(
            {
                "seed_history_ids": list(seed_history_ids),
                "seed_source_ids": list(seed_source_ids),
            }
        ),
        physical_content_rows_scanned=len(index.rows),
        physical_sentence_windows_scanned=len(index.windows),
        user_sentence_windows_scanned=user_windows,
        first_person_window_variant_count=first_person_variants,
        candidate_population_count=len(drafts),
        lane_selected_counts=lane_counts,
        independent_lane_selected_occurrence_count=lane_occurrences,
        post_selection_exact_span_count=len(deduped),
        exact_span_duplicate_count=lane_occurrences - len(deduped),
        plausible_operand_group_count=len(plausible_groups),
        selected_operand_group_ids=selected_group_ids,
        selected_candidate_ids=tuple(row.candidate_id for row in candidates),
        selected_source_group_count=len(
            {row.source_group_handle for row in candidates}
        ),
        selected_evidence_tokens=sum(row.token_count for row in candidates),
        provider_payload_tokens=provider_tokens,
        multi_mention_operand_group_count=sum(
            len(row.candidate_ids) > 1 for row in groups
        ),
        all_plausible_operand_groups_reserved=all_groups_reserved,
        selection_truncated=selection_truncated,
    )
    return NumericOperandClosureResult(
        dated_question=dated_question,
        operator_spec=spec,
        operation_mode=operation_mode,
        expected_numeric_dimension=dimension.value,
        candidates=candidates,
        operand_groups=groups,
        local_bindings=bindings,
        receipt=receipt,
        budget=budget,
    )


def adapt_numeric_operand_closure_to_typed_contribution(
    result: NumericOperandClosureResult,
    /,
    *,
    handle_start: int,
    group_start: int,
) -> "TypedEvidenceContribution":
    """Convert distinct operand groups to one bounded typed contribution.

    Every selected quote receives its own direct-pointer handle and exact local
    citation receipt.  One typed item is then emitted per distinct operand
    group, citing *all* retained mention handles.  This keeps repeated support
    auditable without presenting repeated mentions as additional operands.
    """

    from .typed_operator_adapter import (
        EvidenceHandleBinding,
        EvidenceOrigin,
        FrontierMode,
        ProvenanceGrade,
        TypedEvidenceContribution,
        parse_typed_items,
    )

    _require(
        type(result) is NumericOperandClosureResult,
        "typed numeric contribution requires an exact closure result",
    )
    for value, label in (
        (handle_start, "numeric handle start"),
        (group_start, "numeric group start"),
    ):
        _require(type(value) is int and value >= 1, f"{label} must be positive")
    _require(
        handle_start + len(result.candidates) - 1 <= 999_999,
        "numeric handle range exceeds the opaque contract",
    )
    local_groups = tuple(
        dict.fromkeys(row.source_group_handle for row in result.candidates)
    )
    _require(
        group_start + len(local_groups) - 1 <= 999_999,
        "numeric group range exceeds the opaque contract",
    )
    global_groups = {
        local: f"G{group_start + offset:03d}"
        for offset, local in enumerate(local_groups)
    }
    sealed_artifact_sha256 = identity_sha256(result.local_audit_projection())
    typed_bindings: list[EvidenceHandleBinding] = []
    handle_by_candidate: dict[str, str] = {}
    candidate_by_id = {row.candidate_id: row for row in result.candidates}
    for offset, (candidate, local) in enumerate(
        zip(result.candidates, result.local_bindings, strict=True)
    ):
        handle_id = f"H{handle_start + offset:03d}"
        handle_by_candidate[candidate.candidate_id] = handle_id
        typed_bindings.append(
            EvidenceHandleBinding(
                handle_id=handle_id,
                origin=EvidenceOrigin.DIRECT_POINTER,
                provenance_grade=ProvenanceGrade.DIRECT_POINTER,
                source_group_handle=global_groups[candidate.source_group_handle],
                sealed_artifact_sha256=sealed_artifact_sha256,
                parent_receipt_sha256=result.receipt.receipt_sha256,
                evidence_receipt_sha256=local.receipt_sha256,
                payload_sha256=identity_sha256(candidate.projection()),
                citation_sha256=candidate.quote_sha256,
                citation_char_count=len(candidate.quote),
                local_source_locator_sha256=local.receipt_sha256,
            )
        )

    raw_items: list[dict[str, Any]] = []
    for group in result.operand_groups:
        primary = candidate_by_id[group.candidate_ids[0]]
        explicit_mentions = tuple(
            mention
            for candidate_id in group.candidate_ids
            for mention in candidate_by_id[candidate_id].numeric_mentions
            if mention.value in group.operand_values
        )
        mention = explicit_mentions[0] if explicit_mentions else None
        summary = primary.quote
        if group.value_basis == "implicit_distinct_event_unit":
            summary = (
                f"1 distinct completed {group.action_class} event for "
                f"{group.entity_key}: {primary.quote}"
            )
        raw: dict[str, Any] = {
            "entity_key": group.entity_key,
            "group_key": group.operand_group_id,
            "handle_ids": [
                handle_by_candidate[candidate_id]
                for candidate_id in group.candidate_ids
            ],
            "included": True,
            "kind": "operand",
            "numeric_qualifier": (
                mention.qualifier if mention is not None else "exact"
            ),
            "numeric_role": "operand",
            "numeric_value": group.operand_values[0],
            "personalization_anchors": ["first-person user event assertion"],
            "relation": (
                f"authored_by_user;event_action={group.action_class};"
                "distinct_operand_group"
            ),
            "specificity_terms": [],
            "status": "completed",
            "summary": summary,
            "value_authority": (
                "explicit"
                if group.value_basis == "explicit_numeric_mention"
                else "derived"
            ),
        }
        if mention is not None and mention.unit is not None:
            raw["unit"] = mention.unit
        raw_items.append(raw)

    frozen_bindings = tuple(typed_bindings)
    parsed = parse_typed_items(
        raw_items,
        operator_spec=result.operator_spec,
        bindings=frozen_bindings,
    )
    contribution = TypedEvidenceContribution(
        mechanism_id=MECHANISM_ID,
        bindings=frozen_bindings,
        parsed=parsed,
        sealed_artifact_sha256=sealed_artifact_sha256,
        frontier_mode=FrontierMode.BOUNDED,
        truncated=result.receipt.selection_truncated,
    )
    _require(
        not parsed.rejected_items
        and len(parsed.accepted_items) == len(result.operand_groups)
        and {
            row.group_key for row in parsed.accepted_items
        }
        == {row.operand_group_id for row in result.operand_groups}
        and all(
            binding.evidence_receipt_sha256 == local.receipt_sha256
            and binding.local_source_locator_sha256 == local.receipt_sha256
            for binding, local in zip(
                contribution.bindings, result.local_bindings, strict=True
            )
        ),
        "numeric typed adapter lost operand grouping or exact span receipts",
    )
    return contribution


__all__ = [
    "NumericOperandBudget",
    "NumericOperandCandidate",
    "NumericOperandClosureReceipt",
    "NumericOperandClosureResult",
    "NumericOperandGroup",
    "NumericOperandMention",
    "NumericOperandSpecialistError",
    "adapt_numeric_operand_closure_to_typed_contribution",
    "scan_numeric_operand_closure",
]
