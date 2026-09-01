"""Gold-blind slot and temporal closure over the complete cached store.

This mechanism is deliberately narrower than a semantic search claim.  It
walks every content row already frozen in :class:`NamespacePartitionCache`,
builds exact sentence citations against question-only typed obligations, and
then packs a bounded, source-diverse candidate set.  The receipt calls the
physical scan exhaustive while leaving semantic completeness unclaimed.

Only selected candidates receive opaque source-group handles.  Raw source and
partition identifiers live exclusively in the local citation bindings; they
are absent from the provider projection.  The public entry point accepts no
question ID, source prefix, partition route, reference, or prediction.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, make_atom_id, quote_sha256
from memory_condense.search.indexes.lexical import tokenize

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .partition_scan import _bounded_excerpt
from .query_guided_scan import CachedContentRow, NamespacePartitionCache
from .typed_operator_spec import (
    AnswerShape,
    RequiredSlot,
    SlotKind,
    TypedOperatorSpec,
    compile_typed_operator_spec,
    normalized_terms,
)
from .typed_numeric_semantics import (
    NumericMention,
    numeric_mentions,
    single_numeric_mention,
)

if TYPE_CHECKING:
    from .typed_operator_adapter import TypedEvidenceContribution


MECHANISM_ID = "full_combined_store_slot_temporal_closure_v1"
RESULT_FORMAT = "memory-condense-full-store-slot-closure-result-v1"
CANDIDATE_FORMAT = "memory-condense-full-store-slot-candidate-v1"
BINDING_FORMAT = "memory-condense-full-store-local-citation-v1"
RECEIPT_FORMAT = "memory-condense-full-store-slot-closure-receipt-v1"
ABSENCE_FORMAT = "memory-condense-exact-term-absence-witness-v1"
TEMPORAL_FORMAT = "memory-condense-question-temporal-target-v1"
WINDOW_INDEX_FORMAT = "memory-condense-full-store-window-index-v1"

DEFAULT_EVIDENCE_TOKEN_CAP = 2_400
HARD_PROMPT_TOKEN_CAP = 8_000
DEFAULT_OUTPUT_TOKEN_RESERVE = 768
DEFAULT_PROTOCOL_TOKEN_RESERVE = 512


class FullStoreSlotClosureError(MatchedEvalContractError):
    """Raised when a closure scan or its budget loses an invariant."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise FullStoreSlotClosureError(message)


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(value) is str and value for value in values)
        and len(set(values)) == len(values),
        f"{label} must be an ordered unique exact tuple",
    )
    return values


@dataclass(frozen=True, slots=True)
class FullStoreSlotClosureBudget:
    """Independent evidence and complete prompt-envelope bounds."""

    evidence_token_cap: int = DEFAULT_EVIDENCE_TOKEN_CAP
    hard_prompt_token_cap: int = HARD_PROMPT_TOKEN_CAP
    output_token_reserve: int = DEFAULT_OUTPUT_TOKEN_RESERVE
    protocol_token_reserve: int = DEFAULT_PROTOCOL_TOKEN_RESERVE
    max_candidates: int = 32
    max_excerpt_tokens: int = 128
    max_candidates_per_source: int = 6
    candidates_per_required_slot: int = 2
    temporal_candidate_reserve: int = 4
    source_coherence_candidate_reserve: int = 8

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            _require(type(value) is int and value > 0, f"{name} must be positive")
        _require(
            self.evidence_token_cap
            + self.output_token_reserve
            + self.protocol_token_reserve
            <= self.hard_prompt_token_cap,
            "independent reserves exceed the hard prompt cap",
        )
        _require(
            self.max_candidates_per_source <= self.max_candidates,
            "per-source cap exceeds the candidate cap",
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
class IndexedContentWindow:
    """Question-neutral local index row; no copied transformer state."""

    row: CachedContentRow
    start_char: int
    end_char: int
    text_sha256: str
    token_count: int
    terms: frozenset[str]
    contains_numeric_value: bool
    event_date: str | None
    event_date_basis: str | None

    def __post_init__(self) -> None:
        _require(
            type(self.row) is CachedContentRow
            and type(self.start_char) is int
            and type(self.end_char) is int
            and 0 <= self.start_char < self.end_char <= len(self.row.text),
            "indexed window coordinates changed",
        )
        text = self.row.text[self.start_char : self.end_char]
        require_sha256(self.text_sha256, "indexed window text")
        _require(
            self.text_sha256 == quote_sha256(text)
            and self.token_count == count_tokens(text)
            and type(self.terms) is frozenset
            and all(type(term) is str and term for term in self.terms),
            "indexed window surface changed",
        )
        _require(
            type(self.contains_numeric_value) is bool,
            "indexed window numeric flag changed",
        )
        if self.event_date is not None:
            require_text(self.event_date, "indexed window event date")
            require_text(self.event_date_basis or "", "indexed window date basis")
        else:
            _require(
                self.event_date_basis is None,
                "undated indexed window has a date basis",
            )


@dataclass(frozen=True, slots=True)
class FullStoreWindowIndex:
    """One-pass immutable postings shared by every question tick."""

    cache: NamespacePartitionCache
    rows: tuple[CachedContentRow, ...]
    windows: tuple[IndexedContentWindow, ...]
    term_postings: Mapping[str, tuple[int, ...]]
    role_postings: Mapping[str, tuple[int, ...]]
    date_postings: Mapping[str, tuple[int, ...]]
    numeric_window_indices: tuple[int, ...]
    physical_content_tokens_indexed: int
    posting_inventory_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.cache) is NamespacePartitionCache, "index cache changed")
        _require(
            type(self.rows) is tuple
            and all(type(row) is CachedContentRow for row in self.rows)
            and len(self.rows) == self.cache.content_row_count,
            "indexed content-row inventory changed",
        )
        _require(
            type(self.windows) is tuple
            and all(type(row) is IndexedContentWindow for row in self.windows),
            "indexed sentence-window inventory changed",
        )
        for postings, label in (
            (self.term_postings, "term postings"),
            (self.role_postings, "role postings"),
            (self.date_postings, "date postings"),
        ):
            _require(isinstance(postings, Mapping), f"{label} changed type")
            for key, values in postings.items():
                require_text(key, label)
                _require(
                    type(values) is tuple
                    and tuple(sorted(set(values))) == values
                    and all(0 <= value < len(self.windows) for value in values),
                    f"{label} contains an invalid window index",
                )
        _require(
            tuple(sorted(set(self.numeric_window_indices)))
            == self.numeric_window_indices
            and all(
                0 <= value < len(self.windows)
                for value in self.numeric_window_indices
            ),
            "numeric postings changed",
        )
        _require(
            self.physical_content_tokens_indexed
            == sum(row.token_count for row in self.rows),
            "indexed physical-token count changed",
        )
        require_sha256(self.posting_inventory_sha256, "posting inventory")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "window index receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="full_store_window_index")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "all_cache_partitions_indexed": True,
            "all_content_rows_indexed": True,
            "cache_receipt_sha256": self.cache.cache_receipt_sha256,
            "date_bucket_count": len(self.date_postings),
            "database_read_passes_bound_by_cache": 1,
            "format": WINDOW_INDEX_FORMAT,
            "gold_loaded": False,
            "known_source_prefix_filter_used": False,
            "new_provider_calls": 0,
            "numeric_window_count": len(self.numeric_window_indices),
            "partition_routing_used": False,
            "physical_content_row_count": len(self.rows),
            "physical_content_token_count": self.physical_content_tokens_indexed,
            "physical_partition_count": len(self.cache.rows_by_partition),
            "posting_inventory_sha256": self.posting_inventory_sha256,
            "question_id_filter_used": False,
            "retained_transformer_token_state_bytes": 0,
            "role_bucket_count": len(self.role_postings),
            "semantic_completeness_status": "not_claimed",
            "sentence_window_count": len(self.windows),
            "term_vocabulary_size": len(self.term_postings),
            "window_index_build_passes": 1,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


class TemporalTargetMode(str, Enum):
    NONE = "none"
    EXACT_DAY = "exact_day"
    LOOKBACK_WINDOW = "lookback_window"


@dataclass(frozen=True, slots=True)
class QuestionTemporalTarget:
    mode: TemporalTargetMode
    asked_at: str | None
    target_date: str | None
    lookback_days: int | None
    derivation: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.mode) is TemporalTargetMode, "temporal mode changed")
        for value, label in (
            (self.asked_at, "asked-at timestamp"),
            (self.target_date, "temporal target date"),
        ):
            if value is not None:
                require_text(value, label)
        if self.lookback_days is not None:
            _require(
                type(self.lookback_days) is int and self.lookback_days > 0,
                "lookback days changed",
            )
        require_text(self.derivation, "temporal target derivation")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "temporal target changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "asked_at": self.asked_at,
            "derivation": self.derivation,
            "format": TEMPORAL_FORMAT,
            "lookback_days": self.lookback_days,
            "mode": self.mode.value,
            "target_date": self.target_date,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


class ExactTermAbsenceStatus(str, Enum):
    UNRESOLVED = "unresolved"
    LITERAL_PRESENT = "exact_literal_present"
    LITERAL_ABSENT = "exact_literal_absent"


@dataclass(frozen=True, slots=True)
class ExactTermAbsenceWitness:
    """A narrow statement about one explicitly quoted literal, never meaning."""

    status: ExactTermAbsenceStatus
    literal: str | None
    physical_content_rows_scanned: int
    matching_content_row_count: int
    all_content_rows_scanned: bool
    may_assert_exact_literal_absence: bool
    semantic_absence_may_be_inferred: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.status) is ExactTermAbsenceStatus, "absence status changed")
        if self.literal is not None:
            require_text(self.literal, "exact absence literal")
        _require(
            type(self.physical_content_rows_scanned) is int
            and type(self.matching_content_row_count) is int
            and 0 <= self.matching_content_row_count <= self.physical_content_rows_scanned,
            "absence scan counts changed",
        )
        _require(
            type(self.all_content_rows_scanned) is bool
            and type(self.may_assert_exact_literal_absence) is bool,
            "absence flags changed",
        )
        expected_assertion = bool(
            self.status is ExactTermAbsenceStatus.LITERAL_ABSENT
            and self.literal is not None
            and self.all_content_rows_scanned
            and self.matching_content_row_count == 0
        )
        _require(
            self.may_assert_exact_literal_absence is expected_assertion,
            "exact-literal absence assertion is not justified",
        )
        _require(
            self.semantic_absence_may_be_inferred is False,
            "an exact-term witness cannot prove semantic absence",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "absence witness changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "all_content_rows_scanned": self.all_content_rows_scanned,
            "format": ABSENCE_FORMAT,
            "literal": self.literal,
            "matching_content_row_count": self.matching_content_row_count,
            "may_assert_exact_literal_absence": (
                self.may_assert_exact_literal_absence
            ),
            "physical_content_rows_scanned": self.physical_content_rows_scanned,
            "scope": "casefolded_exact_literal_over_all_cached_content_rows",
            "semantic_absence_may_be_inferred": False,
            "status": self.status.value,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FullStoreSlotCandidate:
    """Provider-visible exact quote with only opaque source co-membership."""

    candidate_id: str
    source_group_handle: str
    quote: str
    quote_sha256: str
    token_count: int
    role: str
    created_at: str
    event_date: str | None
    event_date_basis: str | None
    supported_slot_ids: tuple[str, ...]
    matched_query_terms: tuple[str, ...]
    contains_numeric_value: bool
    temporal_distance_days: int | None
    selection_axes: tuple[str, ...]
    citation_binding_receipt_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.candidate_id, "full-store candidate")
        _require(
            re.fullmatch(r"G[0-9]{4,6}", self.source_group_handle) is not None,
            "source group handle must be opaque",
        )
        require_text(self.quote, "full-store quote")
        require_sha256(self.quote_sha256, "full-store quote")
        _require(
            self.quote_sha256 == quote_sha256(self.quote)
            and self.token_count == count_tokens(self.quote),
            "full-store candidate quote changed",
        )
        require_text(self.role, "full-store evidence role")
        require_text(self.created_at, "full-store created-at")
        if self.event_date is not None:
            require_text(self.event_date, "full-store event date")
            require_text(self.event_date_basis or "", "full-store event-date basis")
        else:
            _require(self.event_date_basis is None, "undated evidence has a date basis")
        _ordered_unique(self.supported_slot_ids, "supported slot IDs")
        _ordered_unique(self.matched_query_terms, "matched query terms")
        _ordered_unique(self.selection_axes, "selection axes")
        _require(type(self.contains_numeric_value) is bool, "numeric flag changed")
        if self.temporal_distance_days is not None:
            _require(
                type(self.temporal_distance_days) is int
                and self.temporal_distance_days >= 0,
                "temporal distance changed",
            )
        require_sha256(
            self.citation_binding_receipt_sha256, "candidate citation binding"
        )

    def projection(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "citation_binding_receipt_sha256": (
                self.citation_binding_receipt_sha256
            ),
            "contains_numeric_value": self.contains_numeric_value,
            "created_at": self.created_at,
            "event_date": self.event_date,
            "event_date_basis": self.event_date_basis,
            "format": CANDIDATE_FORMAT,
            "matched_query_terms": list(self.matched_query_terms),
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "selection_axes": list(self.selection_axes),
            "source_group_handle": self.source_group_handle,
            "supported_slot_ids": list(self.supported_slot_ids),
            "temporal_distance_days": self.temporal_distance_days,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class LocalCitationBinding:
    """Prompt-external exact provenance for one opaque candidate."""

    candidate_id: str
    source_group_handle: str
    namespace_id: str
    cache_receipt_sha256: str
    source_database_sha256: str
    source_store_receipt_sha256: str
    source_id: str
    partition_id: str
    span: EvidenceSpan
    quote_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "local candidate"),
            (self.namespace_id, "local namespace"),
            (self.cache_receipt_sha256, "local cache"),
            (self.source_database_sha256, "local database"),
            (self.source_store_receipt_sha256, "local store"),
            (self.quote_sha256, "local quote"),
        ):
            require_sha256(value, label)
        _require(
            re.fullmatch(r"G[0-9]{4,6}", self.source_group_handle) is not None,
            "local source group must be opaque",
        )
        require_text(self.source_id, "local source ID")
        require_text(self.partition_id, "local partition ID")
        _require(self.span.source_id == self.source_id, "citation source changed")
        _require(
            self.span.quote_sha256 == self.quote_sha256,
            "citation quote digest changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "local citation changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "candidate_id": self.candidate_id,
            "format": BINDING_FORMAT,
            "namespace_id": self.namespace_id,
            "partition_id": self.partition_id,
            "quote_sha256": self.quote_sha256,
            "source_database_sha256": self.source_database_sha256,
            "source_group_handle": self.source_group_handle,
            "source_id": self.source_id,
            "source_store_receipt_sha256": self.source_store_receipt_sha256,
            "span": self.span.identity_payload(),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FullStoreSlotClosureReceipt:
    question_sha256: str
    typed_operator_spec_receipt_sha256: str
    temporal_target_receipt_sha256: str
    absence_witness_receipt_sha256: str
    cache_receipt_sha256: str
    window_index_receipt_sha256: str
    window_index_reuse_mode: str
    budget_id: str
    physical_content_rows_scanned: int
    physical_content_tokens_scanned: int
    physical_sentence_windows_scanned: int
    physical_partition_count: int
    query_candidate_windows_considered: int
    candidate_population_count: int
    selected_candidate_ids: tuple[str, ...]
    selected_source_group_count: int
    selected_evidence_tokens: int
    provider_payload_tokens: int
    required_slot_ids: tuple[str, ...]
    covered_slot_ids: tuple[str, ...]
    unresolved_slot_ids: tuple[str, ...]
    role_rejected_candidate_count: int
    selection_truncated: bool
    evidence_status: str
    semantic_completeness_status: Literal["not_claimed"] = "not_claimed"
    physical_scan_scope: Literal["all_cached_content_rows"] = (
        "all_cached_content_rows"
    )
    physical_scan_exhaustive: Literal[True] = True
    query_tick_full_physical_rescan: Literal[False] = False
    question_id_filter_used: Literal[False] = False
    known_source_prefix_filter_used: Literal[False] = False
    partition_routing_used: Literal[False] = False
    raw_partition_ids_provider_visible: Literal[False] = False
    new_provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.question_sha256, "closure question"),
            (self.typed_operator_spec_receipt_sha256, "closure typed spec"),
            (self.temporal_target_receipt_sha256, "closure temporal target"),
            (self.absence_witness_receipt_sha256, "closure absence witness"),
            (self.cache_receipt_sha256, "closure cache"),
            (self.window_index_receipt_sha256, "closure window index"),
            (self.budget_id, "closure budget"),
        ):
            require_sha256(value, label)
        for value, label in (
            (self.physical_content_rows_scanned, "physical rows"),
            (self.physical_content_tokens_scanned, "physical tokens"),
            (self.physical_sentence_windows_scanned, "physical sentence windows"),
            (self.physical_partition_count, "physical partitions"),
            (self.query_candidate_windows_considered, "query candidate windows"),
            (self.candidate_population_count, "candidate population"),
            (self.selected_source_group_count, "selected source groups"),
            (self.selected_evidence_tokens, "selected evidence tokens"),
            (self.provider_payload_tokens, "provider payload tokens"),
            (self.role_rejected_candidate_count, "role rejected candidates"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        selected = _ordered_unique(self.selected_candidate_ids, "selected candidates")
        required = _ordered_unique(self.required_slot_ids, "required slots")
        covered = _ordered_unique(self.covered_slot_ids, "covered slots")
        unresolved = _ordered_unique(self.unresolved_slot_ids, "unresolved slots")
        _require(
            set(covered) <= set(required)
            and set(unresolved) == set(required) - set(covered),
            "slot coverage partition changed",
        )
        _require(
            len(selected) <= self.candidate_population_count,
            "selection exceeds its candidate population",
        )
        _require(type(self.selection_truncated) is bool, "truncation flag changed")
        require_text(self.evidence_status, "closure evidence status")
        _require(
            self.window_index_reuse_mode
            in {"built_for_single_tick", "reused_prebuilt_index"},
            "window-index reuse mode changed",
        )
        _require(
            self.semantic_completeness_status == "not_claimed"
            and self.physical_scan_scope == "all_cached_content_rows"
            and self.physical_scan_exhaustive is True,
            "physical exhaustion was confused with semantic completeness",
        )
        _require(
            self.query_tick_full_physical_rescan is False,
            "query tick unexpectedly repeated the full physical scan",
        )
        _require(
            self.question_id_filter_used is False
            and self.known_source_prefix_filter_used is False
            and self.partition_routing_used is False
            and self.raw_partition_ids_provider_visible is False,
            "full-store scan used a forbidden route or exposed a partition",
        )
        _require(
            self.new_provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "closure scan must remain provider-free, zero-state, and gold-blind",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "closure receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="full_store_slot_closure_receipt")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "absence_witness_receipt_sha256": self.absence_witness_receipt_sha256,
            "budget_id": self.budget_id,
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "candidate_population_count": self.candidate_population_count,
            "covered_slot_ids": list(self.covered_slot_ids),
            "evidence_status": self.evidence_status,
            "format": RECEIPT_FORMAT,
            "gold_loaded": False,
            "known_source_prefix_filter_used": False,
            "new_provider_calls": 0,
            "partition_routing_used": False,
            "physical_content_rows_scanned": self.physical_content_rows_scanned,
            "physical_content_tokens_scanned": self.physical_content_tokens_scanned,
            "physical_partition_count": self.physical_partition_count,
            "physical_scan_exhaustive": True,
            "physical_scan_scope": "all_cached_content_rows",
            "physical_sentence_windows_scanned": self.physical_sentence_windows_scanned,
            "provider_payload_tokens": self.provider_payload_tokens,
            "query_candidate_windows_considered": self.query_candidate_windows_considered,
            "question_id_filter_used": False,
            "query_tick_full_physical_rescan": False,
            "question_sha256": self.question_sha256,
            "raw_partition_ids_provider_visible": False,
            "required_slot_ids": list(self.required_slot_ids),
            "retained_transformer_token_state_bytes": 0,
            "role_rejected_candidate_count": self.role_rejected_candidate_count,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_evidence_tokens": self.selected_evidence_tokens,
            "selected_source_group_count": self.selected_source_group_count,
            "selection_truncated": self.selection_truncated,
            "semantic_completeness_status": "not_claimed",
            "temporal_target_receipt_sha256": self.temporal_target_receipt_sha256,
            "typed_operator_spec_receipt_sha256": (
                self.typed_operator_spec_receipt_sha256
            ),
            "unresolved_slot_ids": list(self.unresolved_slot_ids),
            "window_index_receipt_sha256": self.window_index_receipt_sha256,
            "window_index_reuse_mode": self.window_index_reuse_mode,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FullStoreSlotClosureResult:
    dated_question: str
    operator_spec: TypedOperatorSpec
    temporal_target: QuestionTemporalTarget
    absence_witness: ExactTermAbsenceWitness
    candidates: tuple[FullStoreSlotCandidate, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    receipt: FullStoreSlotClosureReceipt
    budget: FullStoreSlotClosureBudget

    def __post_init__(self) -> None:
        require_text(self.dated_question, "full-store dated question")
        _require(
            type(self.operator_spec) is TypedOperatorSpec,
            "full-store result requires an exact typed operator spec",
        )
        _require(
            type(self.candidates) is tuple
            and all(type(row) is FullStoreSlotCandidate for row in self.candidates),
            "full-store candidates changed type",
        )
        _require(
            type(self.local_bindings) is tuple
            and all(type(row) is LocalCitationBinding for row in self.local_bindings),
            "full-store local bindings changed type",
        )
        candidate_ids = tuple(row.candidate_id for row in self.candidates)
        binding_ids = tuple(row.candidate_id for row in self.local_bindings)
        _require(
            candidate_ids == binding_ids == self.receipt.selected_candidate_ids,
            "opaque candidates lost exact local citations",
        )
        _require(
            all(
                candidate.citation_binding_receipt_sha256 == binding.receipt_sha256
                and candidate.source_group_handle == binding.source_group_handle
                for candidate, binding in zip(
                    self.candidates, self.local_bindings, strict=True
                )
            ),
            "candidate citation binding changed",
        )
        groups = {row.source_group_handle for row in self.candidates}
        _require(
            len(groups) == self.receipt.selected_source_group_count,
            "source group count changed",
        )
        payload_tokens = count_tokens(_canonical_payload_json(self.provider_projection()))
        _require(
            payload_tokens == self.receipt.provider_payload_tokens,
            "provider payload token accounting changed",
        )
        _require(
            payload_tokens
            + self.budget.output_token_reserve
            + self.budget.protocol_token_reserve
            <= self.budget.hard_prompt_token_cap,
            "provider payload exceeds the hard prompt envelope",
        )
        assert_gold_blind(self.provider_projection(), path="full_store_provider_payload")

    def provider_projection(self) -> dict[str, Any]:
        """Projection safe for a later model; no raw source/partition locators."""

        return {
            "absence_witness": self.absence_witness.projection(),
            "candidates": [row.projection() for row in self.candidates],
            "dated_question": self.dated_question,
            "format": RESULT_FORMAT,
            "operator_spec": self.operator_spec.projection(),
            "temporal_target": self.temporal_target.projection(),
        }

    def local_audit_projection(self) -> dict[str, Any]:
        """Exact provenance kept outside the provider-visible payload."""

        return {
            "bindings": [row.projection() for row in self.local_bindings],
            "provider_payload_sha256": identity_sha256(self.provider_projection()),
            "receipt": self.receipt.projection(),
        }


@dataclass(frozen=True, slots=True)
class _Draft:
    candidate_id: str
    row: CachedContentRow
    start_char: int
    end_char: int
    quote: str
    supported_slot_ids: tuple[str, ...]
    matched_query_terms: tuple[str, ...]
    contains_numeric_value: bool
    event_date: str | None
    event_date_basis: str | None
    temporal_distance_days: int | None
    exact_temporal_match: bool
    within_temporal_window: bool
    lexical_score: float

    @property
    def score(self) -> tuple[Any, ...]:
        return (
            len(self.supported_slot_ids),
            int(self.exact_temporal_match),
            int(self.within_temporal_window),
            self.lexical_score,
            len(self.matched_query_terms),
            int(self.contains_numeric_value),
            -(self.temporal_distance_days or 0),
            -self.row.ordinal,
            -self.start_char,
            self.candidate_id,
        )


_DATED_RE = re.compile(
    r"^\[Question asked at (?P<asked_at>.+?)\]\s*", re.I | re.S
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
_NUMBER_RE = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w.])"
)
_NUMBER_WORD_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b",
    re.I,
)
_EXACT_LITERAL_ABSENCE_RE = re.compile(
    r"\b(?:did|have)\s+(?:i|we)\s+(?:ever\s+)?"
    r"(?:say|mention|use|write)\s+(?:the\s+)?exact\s+"
    r"(?:term|phrase)\s+[\"“](?P<literal>[^\"”]{1,160})[\"”]",
    re.I,
)
_ISO_DATE_RE = re.compile(r"\b(?P<year>20\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b")
_MONTH_DATE_RE = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(?P<day>\d{1,2})"
    r"(?:st|nd|rd|th)?(?:,\s*(?P<year>20\d{2}))?\b",
    re.I,
)
_LAST_WEEKDAY_RE = re.compile(
    r"\blast\s+(?P<weekday>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.I,
)
_RELATIVE_OFFSET_RE = re.compile(
    r"\b(?P<count>an?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|\d+)\s+(?P<unit>days?|weeks?|months?|years?)\s+"
    r"(?P<direction>ago|earlier|before)\b",
    re.I,
)
_NUMBER_VALUES = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_WEEKDAY = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_QUERY_FUNCTION_TERMS = frozenset(
    {
        "answer",
        "anything",
        "detail",
        "did",
        "exact",
        "exactly",
        "happen",
        "kind",
        "many",
        "much",
        "name",
        "number",
        "please",
        "specific",
        "specifically",
        "tell",
        "type",
    }
)


def _canonical_payload_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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


def _question_body(dated_question: str) -> str:
    return _DATED_RE.sub("", dated_question).strip()


def indexed_surface_terms(value: str) -> tuple[str, ...]:
    """Return the exact term normal forms used by full-store postings.

    The conservative possessive alias is part of the index contract, not just
    an implementation detail: callback validators must replay this same
    surface relation when proving a lexical match against an exact quote.
    """

    output: list[str] = []
    for term in normalized_terms(value):
        output.append(term)
        if term.endswith("'s") and len(term) > 2:
            output.append(term[:-2])
        # The shared light stemmer removes the plural-looking ``s`` from an
        # English possessive and leaves the apostrophe (``Johnson's`` ->
        # ``johnson'``).  Retain that exact form but add its conservative base.
        elif term.endswith("'") and len(term) > 1:
            output.append(term[:-1])
    return tuple(dict.fromkeys(output))


def _question_terms(body: str) -> tuple[str, ...]:
    terms = tuple(
        term
        for term in indexed_surface_terms(body)
        if term not in _QUERY_FUNCTION_TERMS
    )
    return terms or indexed_surface_terms(body)


def _temporal_target(
    dated_question: str, spec: TypedOperatorSpec
) -> QuestionTemporalTarget:
    body = _question_body(dated_question)
    asked = _parse_datetime(spec.query_timestamp)
    asked_text = asked.isoformat() if asked is not None else None

    explicit = _ISO_DATE_RE.search(body)
    if explicit is not None:
        try:
            target = date(
                int(explicit.group("year")),
                int(explicit.group("month")),
                int(explicit.group("day")),
            )
        except ValueError:
            target = None
        if target is not None:
            return QuestionTemporalTarget(
                TemporalTargetMode.EXACT_DAY,
                asked_text,
                target.isoformat(),
                None,
                "explicit_iso_date_in_question",
            )

    named = _MONTH_DATE_RE.search(body)
    if named is not None:
        year = int(named.group("year")) if named.group("year") else (
            asked.year if asked is not None else None
        )
        if year is not None:
            try:
                target = datetime.strptime(
                    f"{named.group('month')} {named.group('day')} {year}",
                    "%B %d %Y",
                ).date()
            except ValueError:
                target = None
            if target is not None:
                return QuestionTemporalTarget(
                    TemporalTargetMode.EXACT_DAY,
                    asked_text,
                    target.isoformat(),
                    None,
                    "explicit_named_date_in_question",
                )

    weekday = _LAST_WEEKDAY_RE.search(body)
    if weekday is not None and asked is not None:
        target_weekday = _WEEKDAY[weekday.group("weekday").casefold()]
        delta = (asked.weekday() - target_weekday) % 7
        delta = delta or 7
        target = asked.date() - timedelta(days=delta)
        return QuestionTemporalTarget(
            TemporalTargetMode.EXACT_DAY,
            asked_text,
            target.isoformat(),
            None,
            "last_weekday_relative_to_question_timestamp",
        )

    relative = _RELATIVE_OFFSET_RE.search(body)
    if relative is not None and asked is not None:
        direction = relative.group("direction").casefold()
        tail = body[relative.end() :].strip()
        unambiguous = bool(
            direction == "ago"
            or (
                direction == "earlier"
                and not re.match(r"^(?:than|before|after|when)\b", tail, re.I)
            )
            or (
                direction == "before"
                and re.fullmatch(r"(?:(?:now|today)\s*)?[?.!,]?", tail, re.I)
                is not None
            )
        )
        if not unambiguous:
            relative = None
    if relative is not None and asked is not None:
        raw = relative.group("count").casefold()
        amount = int(raw) if raw.isdigit() else _NUMBER_VALUES[raw]
        unit = relative.group("unit").casefold()
        days = amount * (
            1
            if unit.startswith("day")
            else 7
            if unit.startswith("week")
            else 31
            if unit.startswith("month")
            else 366
        )
        target = asked.date() - timedelta(days=days)
        return QuestionTemporalTarget(
            TemporalTargetMode.EXACT_DAY,
            asked_text,
            target.isoformat(),
            None,
            "unambiguous_relative_offset_expression",
        )

    if asked is not None and spec.temporal_window_days is not None:
        return QuestionTemporalTarget(
            TemporalTargetMode.LOOKBACK_WINDOW,
            asked_text,
            None,
            spec.temporal_window_days,
            "typed_operator_lookback_window",
        )
    return QuestionTemporalTarget(
        TemporalTargetMode.NONE,
        asked_text,
        None,
        None,
        "no_question_derived_temporal_target",
    )


def _event_date(text: str, row: CachedContentRow) -> tuple[date | None, str | None]:
    explicit = _ISO_DATE_RE.search(text)
    if explicit is not None:
        try:
            return (
                date(
                    int(explicit.group("year")),
                    int(explicit.group("month")),
                    int(explicit.group("day")),
                ),
                "explicit_text_date",
            )
        except ValueError:
            pass
    named = _MONTH_DATE_RE.search(text)
    created = _parse_datetime(row.created_at)
    if named is not None:
        year = int(named.group("year")) if named.group("year") else (
            created.year if created is not None else None
        )
        if year is not None:
            try:
                parsed = datetime.strptime(
                    f"{named.group('month')} {named.group('day')} {year}",
                    "%B %d %Y",
                ).date()
                return parsed, "explicit_text_date"
            except ValueError:
                pass
    if created is not None:
        return created.date(), "row_created_at"
    return None, None


def _temporal_fit(
    event: date | None, target: QuestionTemporalTarget
) -> tuple[int | None, bool, bool]:
    if event is None or target.mode is TemporalTargetMode.NONE:
        return None, False, False
    asked = _parse_datetime(target.asked_at)
    if target.mode is TemporalTargetMode.EXACT_DAY:
        wanted = date.fromisoformat(target.target_date or "")
        distance = abs((event - wanted).days)
        return distance, distance == 0, distance == 0
    _require(asked is not None, "lookback target lost asked-at timestamp")
    distance = (asked.date() - event).days
    within = bool(0 <= distance <= (target.lookback_days or 0))
    return abs(distance), False, within


def _contains_numeric(
    text: str,
    *,
    spec: TypedOperatorSpec | None = None,
    question: str | None = None,
) -> bool:
    return bool(
        numeric_mentions(
            text,
            operator_spec=spec,
            question=question,
        )
    )


def _slot_match(slot: RequiredSlot, terms: frozenset[str], numeric: bool) -> bool:
    overlap = len(set(slot.match_terms) & terms)
    return bool(
        overlap >= slot.minimum_match_term_count
        and (not slot.requires_numeric or numeric)
    )


def _bounded_window(
    row: CachedContentRow,
    start: int,
    end: int,
    query_tokens: frozenset[str],
    *,
    max_tokens: int,
) -> tuple[int, int, str]:
    text = row.text[start:end]
    if count_tokens(text) <= max_tokens:
        return start, end, text
    local_start, local_end, excerpt = _bounded_excerpt(
        text, query_tokens, max_tokens=max_tokens
    )
    return start + local_start, start + local_end, excerpt


def _all_rows(cache: NamespacePartitionCache) -> tuple[CachedContentRow, ...]:
    """Flatten every partition without accepting a route or identifier filter."""

    rows = tuple(
        row
        for partition_rows in cache.rows_by_partition.values()
        for row in partition_rows
    )
    _require(
        len(rows) == cache.content_row_count
        and len({row.chunk_id for row in rows}) == len(rows),
        "full-store row inventory changed",
    )
    return rows


def build_full_store_window_index(
    cache: NamespacePartitionCache, /
) -> FullStoreWindowIndex:
    """Normalize every cached sentence once for reuse across prompt ticks."""

    _require(type(cache) is NamespacePartitionCache, "index requires exact cache")
    rows = _all_rows(cache)
    windows: list[IndexedContentWindow] = []
    mutable_terms: dict[str, list[int]] = {}
    mutable_roles: dict[str, list[int]] = {}
    mutable_dates: dict[str, list[int]] = {}
    numeric: list[int] = []
    for row in rows:
        for window in row.sentence_windows:
            text = row.text[window.start_char : window.end_char]
            event, event_basis = _event_date(text, row)
            indexed = IndexedContentWindow(
                row=row,
                start_char=window.start_char,
                end_char=window.end_char,
                text_sha256=quote_sha256(text),
                token_count=count_tokens(text),
                terms=frozenset(indexed_surface_terms(text)),
                contains_numeric_value=_contains_numeric(text),
                event_date=event.isoformat() if event is not None else None,
                event_date_basis=event_basis,
            )
            window_index = len(windows)
            windows.append(indexed)
            for term in sorted(indexed.terms):
                mutable_terms.setdefault(term, []).append(window_index)
            mutable_roles.setdefault(row.role, []).append(window_index)
            if indexed.event_date is not None:
                mutable_dates.setdefault(indexed.event_date, []).append(window_index)
            if indexed.contains_numeric_value:
                numeric.append(window_index)
    term_postings = MappingProxyType(
        {term: tuple(values) for term, values in sorted(mutable_terms.items())}
    )
    role_postings = MappingProxyType(
        {role: tuple(values) for role, values in sorted(mutable_roles.items())}
    )
    date_postings = MappingProxyType(
        {day: tuple(values) for day, values in sorted(mutable_dates.items())}
    )
    posting_inventory = {
        "dates": {key: list(value) for key, value in date_postings.items()},
        "numeric": numeric,
        "roles": {key: list(value) for key, value in role_postings.items()},
        "terms": {key: list(value) for key, value in term_postings.items()},
        "window_text_sha256s": [row.text_sha256 for row in windows],
    }
    return FullStoreWindowIndex(
        cache=cache,
        rows=rows,
        windows=tuple(windows),
        term_postings=term_postings,
        role_postings=role_postings,
        date_postings=date_postings,
        numeric_window_indices=tuple(numeric),
        physical_content_tokens_indexed=sum(row.token_count for row in rows),
        posting_inventory_sha256=identity_sha256(posting_inventory),
    )


def _absence_witness(
    body: str, rows: Sequence[CachedContentRow]
) -> ExactTermAbsenceWitness:
    match = _EXACT_LITERAL_ABSENCE_RE.search(body)
    if match is None:
        return ExactTermAbsenceWitness(
            ExactTermAbsenceStatus.UNRESOLVED,
            None,
            len(rows),
            0,
            True,
            False,
        )
    literal = match.group("literal").strip()
    folded = literal.casefold()
    matching = sum(folded in row.text.casefold() for row in rows)
    status = (
        ExactTermAbsenceStatus.LITERAL_PRESENT
        if matching
        else ExactTermAbsenceStatus.LITERAL_ABSENT
    )
    return ExactTermAbsenceWitness(
        status,
        literal,
        len(rows),
        matching,
        True,
        status is ExactTermAbsenceStatus.LITERAL_ABSENT,
    )


def _candidate_drafts(
    index: FullStoreWindowIndex,
    spec: TypedOperatorSpec,
    body: str,
    target: QuestionTemporalTarget,
    budget: FullStoreSlotClosureBudget,
) -> tuple[tuple[_Draft, ...], int, int]:
    query_terms = _question_terms(body)
    query_tokens = frozenset(tokenize(body))
    retrieval_terms = set(query_terms)
    for slot in spec.required_slots:
        retrieval_terms.update(slot.match_terms)
    candidate_indices: set[int] = set()
    for term in retrieval_terms:
        candidate_indices.update(index.term_postings.get(term, ()))
    if target.mode is TemporalTargetMode.EXACT_DAY and target.target_date is not None:
        candidate_indices.update(index.date_postings.get(target.target_date, ()))
    elif target.mode is TemporalTargetMode.LOOKBACK_WINDOW:
        asked = _parse_datetime(target.asked_at)
        _require(asked is not None, "lookback target lost asked-at timestamp")
        earliest = asked.date() - timedelta(days=target.lookback_days or 0)
        for day, postings in index.date_postings.items():
            parsed = date.fromisoformat(day)
            if earliest <= parsed <= asked.date():
                candidate_indices.update(postings)
    document_frequency = {
        term: len(index.term_postings.get(term, ()))
        for term in query_terms
    }
    window_count = max(len(index.windows), 1)
    drafts: list[_Draft] = []
    role_rejected = 0
    for window_index in sorted(candidate_indices):
        window = index.windows[window_index]
        row = window.row
        if window.token_count <= budget.max_excerpt_tokens:
            start, end = window.start_char, window.end_char
            excerpt = row.text[start:end]
            terms = window.terms
        else:
            start, end, excerpt = _bounded_window(
                row,
                window.start_char,
                window.end_char,
                query_tokens,
                max_tokens=budget.max_excerpt_tokens,
            )
            terms = frozenset(indexed_surface_terms(excerpt))
        numeric = _contains_numeric(excerpt, spec=spec, question=body)
        supported = tuple(
            slot.slot_id
            for slot in spec.required_slots
            if _slot_match(slot, terms, numeric)
        )
        matched = tuple(term for term in query_terms if term in terms)
        event = (
            date.fromisoformat(window.event_date)
            if window.event_date is not None
            else None
        )
        event_basis = window.event_date_basis
        distance, exact_temporal, within_window = _temporal_fit(event, target)
        role_ok = spec.required_evidence_role in {None, row.role}
        relevant = bool(supported or matched or exact_temporal or within_window)
        if relevant and not role_ok:
            role_rejected += 1
            continue
        if not role_ok or not relevant:
            continue
        lexical = sum(
            math.log((window_count + 1) / (document_frequency[term] + 1)) + 1.0
            for term in matched
        )
        if exact_temporal:
            lexical += 4.0
        elif within_window:
            lexical += 1.0
        span = _evidence_span(row, start, end, excerpt)
        candidate_id = identity_sha256(
            {"atom_id": make_atom_id(span), "mechanism_id": MECHANISM_ID}
        )
        drafts.append(
            _Draft(
                candidate_id=candidate_id,
                row=row,
                start_char=start,
                end_char=end,
                quote=excerpt,
                supported_slot_ids=supported,
                matched_query_terms=matched,
                contains_numeric_value=numeric,
                event_date=event.isoformat() if event is not None else None,
                event_date_basis=event_basis,
                temporal_distance_days=distance,
                exact_temporal_match=exact_temporal,
                within_temporal_window=within_window,
                lexical_score=round(lexical, 8),
            )
        )
    unique: dict[str, _Draft] = {}
    for draft in sorted(drafts, key=lambda row: row.score, reverse=True):
        unique.setdefault(draft.candidate_id, draft)
    return tuple(unique.values()), len(candidate_indices), role_rejected


def _evidence_span(
    row: CachedContentRow, start_char: int, end_char: int, quote: str
) -> EvidenceSpan:
    return EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=start_char,
        end_char=end_char,
        quote_sha256=quote_sha256(quote),
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )


def _span(draft: _Draft) -> EvidenceSpan:
    return _evidence_span(
        draft.row, draft.start_char, draft.end_char, draft.quote
    )


def _select_drafts(
    drafts: Sequence[_Draft],
    spec: TypedOperatorSpec,
    target: QuestionTemporalTarget,
    budget: FullStoreSlotClosureBudget,
) -> tuple[tuple[_Draft, tuple[str, ...]], ...]:
    ranked = tuple(sorted(drafts, key=lambda row: row.score, reverse=True))
    selected: list[tuple[_Draft, tuple[str, ...]]] = []
    selected_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    evidence_tokens = 0

    def add(draft: _Draft, axis: str) -> bool:
        nonlocal evidence_tokens
        for index, (current, axes) in enumerate(selected):
            if current.candidate_id == draft.candidate_id:
                if axis not in axes:
                    selected[index] = (current, (*axes, axis))
                return True
        token_count = count_tokens(draft.quote)
        if (
            len(selected) >= budget.max_candidates
            or source_counts.get(draft.row.source_id, 0)
            >= budget.max_candidates_per_source
            or evidence_tokens + token_count > budget.evidence_token_cap
        ):
            return False
        selected.append((draft, (axis,)))
        selected_ids.add(draft.candidate_id)
        source_counts[draft.row.source_id] = (
            source_counts.get(draft.row.source_id, 0) + 1
        )
        evidence_tokens += token_count
        return True

    # First reserve evidence independently for every question-derived slot.
    for slot in spec.required_slots:
        slot_candidates = tuple(
            draft for draft in ranked if slot.slot_id in draft.supported_slot_ids
        )
        slot_sources = {draft.row.source_id for draft in slot_candidates}
        used_sources: set[str] = set()
        retained = 0
        for draft in slot_candidates:
            if draft.row.source_id in used_sources and len(slot_sources) > 1:
                continue
            if add(draft, f"required_slot:{slot.slot_id}"):
                used_sources.add(draft.row.source_id)
                retained += 1
            if retained == budget.candidates_per_required_slot:
                break

    # Exact/range dates are an independent admission axis, including rows whose
    # wording shares no lexical term with a high-level phrase such as
    # "gardening activity".
    if target.mode is not TemporalTargetMode.NONE:
        used_sources: set[str] = set()
        retained = 0
        for draft in ranked:
            if not (draft.exact_temporal_match or draft.within_temporal_window):
                continue
            if draft.row.source_id in used_sources:
                continue
            if add(draft, "question_derived_temporal_target"):
                used_sources.add(draft.row.source_id)
                retained += 1
            if retained == budget.temporal_candidate_reserve:
                break

    # Content-only story coherence is independent of source diversity.  A
    # candidate group that collectively covers several question obligations is
    # retained before filler, but source IDs are used only for local
    # co-membership—not compared with a question ID or namespace prefix.
    coherent_sources: list[
        tuple[int, int, tuple[Any, ...], str, tuple[_Draft, ...]]
    ] = []
    mutable_source_values: dict[str, list[_Draft]] = {}
    for draft in ranked:
        mutable_source_values.setdefault(draft.row.source_id, []).append(draft)
    source_values = {
        source: tuple(values) for source, values in mutable_source_values.items()
    }
    for source, values in source_values.items():
        slot_union = {slot for row in values for slot in row.supported_slot_ids}
        term_union = {term for row in values for term in row.matched_query_terms}
        if len(slot_union) + len(term_union) < 2:
            continue
        coherent_sources.append(
            (len(slot_union), len(term_union), values[0].score, source, values)
        )
    retained_coherent = 0
    for _slots, _terms, _score, _source, values in sorted(
        coherent_sources, reverse=True
    ):
        covered_slots: set[str] = set()
        covered_terms: set[str] = set()
        for draft in values:
            expands = bool(
                set(draft.supported_slot_ids) - covered_slots
                or set(draft.matched_query_terms) - covered_terms
            )
            if not expands:
                continue
            if add(draft, "content_source_coherence"):
                covered_slots.update(draft.supported_slot_ids)
                covered_terms.update(draft.matched_query_terms)
                retained_coherent += 1
            if retained_coherent == budget.source_coherence_candidate_reserve:
                break
        if retained_coherent == budget.source_coherence_candidate_reserve:
            break

    # Finish in source round-robin order so one verbose history cannot consume
    # the independent evidence budget.
    by_source: dict[str, list[_Draft]] = {}
    for draft in ranked:
        by_source.setdefault(draft.row.source_id, []).append(draft)
    source_order = sorted(
        by_source,
        key=lambda source: (by_source[source][0].score, source),
        reverse=True,
    )
    width = max((len(values) for values in by_source.values()), default=0)
    for source_rank in range(width):
        for source in source_order:
            values = by_source[source]
            if source_rank < len(values):
                add(values[source_rank], "source_diverse_ranked_recall")
    return tuple(selected)


def _materialize_candidates(
    selected: Sequence[tuple[_Draft, tuple[str, ...]]],
    cache: NamespacePartitionCache,
) -> tuple[tuple[FullStoreSlotCandidate, ...], tuple[LocalCitationBinding, ...]]:
    selected_sources = tuple(
        sorted({draft.row.source_id for draft, _axes in selected})
    )
    groups = {
        source: f"G{index:04d}" for index, source in enumerate(selected_sources, 1)
    }
    candidates: list[FullStoreSlotCandidate] = []
    bindings: list[LocalCitationBinding] = []
    for draft, axes in selected:
        span = _span(draft)
        candidate_id = draft.candidate_id
        binding = LocalCitationBinding(
            candidate_id=candidate_id,
            source_group_handle=groups[draft.row.source_id],
            namespace_id=cache.namespace_id,
            cache_receipt_sha256=cache.cache_receipt_sha256,
            source_database_sha256=cache.source_database_sha256,
            source_store_receipt_sha256=cache.source_store_receipt_sha256,
            source_id=draft.row.source_id,
            partition_id=draft.row.partition_id,
            span=span,
            quote_sha256=quote_sha256(draft.quote),
        )
        candidate = FullStoreSlotCandidate(
            candidate_id=candidate_id,
            source_group_handle=groups[draft.row.source_id],
            quote=draft.quote,
            quote_sha256=quote_sha256(draft.quote),
            token_count=count_tokens(draft.quote),
            role=draft.row.role,
            created_at=draft.row.created_at,
            event_date=draft.event_date,
            event_date_basis=draft.event_date_basis,
            supported_slot_ids=draft.supported_slot_ids,
            matched_query_terms=draft.matched_query_terms,
            contains_numeric_value=draft.contains_numeric_value,
            temporal_distance_days=draft.temporal_distance_days,
            selection_axes=axes,
            citation_binding_receipt_sha256=binding.receipt_sha256,
        )
        bindings.append(binding)
        candidates.append(candidate)
    return tuple(candidates), tuple(bindings)


def _provider_projection(
    dated_question: str,
    spec: TypedOperatorSpec,
    target: QuestionTemporalTarget,
    absence: ExactTermAbsenceWitness,
    candidates: Sequence[FullStoreSlotCandidate],
) -> dict[str, Any]:
    return {
        "absence_witness": absence.projection(),
        "candidates": [row.projection() for row in candidates],
        "dated_question": dated_question,
        "format": RESULT_FORMAT,
        "operator_spec": spec.projection(),
        "temporal_target": target.projection(),
    }


def scan_full_store_slot_closure(
    cache: NamespacePartitionCache | FullStoreWindowIndex,
    dated_question: str,
    /,
    *,
    budget: FullStoreSlotClosureBudget = FullStoreSlotClosureBudget(),
) -> FullStoreSlotClosureResult:
    """Scan every cached content row and return bounded exact citations.

    ``dated_question`` is the only query-side input.  The signature
    intentionally has no question ID, known source, source prefix, partition
    list, reference, prediction, or provider client.
    """

    _require(
        type(cache) in {NamespacePartitionCache, FullStoreWindowIndex},
        "scan requires an exact cache or immutable window index",
    )
    require_text(dated_question, "full-store dated question")
    _require(
        type(budget) is FullStoreSlotClosureBudget,
        "scan requires an exact closure budget",
    )
    if type(cache) is NamespacePartitionCache:
        index = build_full_store_window_index(cache)
        reuse_mode = "built_for_single_tick"
    else:
        index = cache
        reuse_mode = "reused_prebuilt_index"
    source_cache = index.cache
    spec = compile_typed_operator_spec(dated_question)
    rows = index.rows
    body = _question_body(dated_question)
    target = _temporal_target(dated_question, spec)
    absence = _absence_witness(body, rows)
    drafts, candidate_windows_considered, role_rejected = _candidate_drafts(
        index, spec, body, target, budget
    )
    selected = list(_select_drafts(drafts, spec, target, budget))

    # Exact provider-payload accounting is applied after source handles exist.
    # Lowest-priority candidates are removed until the complete bounded payload
    # fits alongside explicit protocol and output reserves.
    while True:
        candidates, bindings = _materialize_candidates(selected, source_cache)
        provider_projection = _provider_projection(
            dated_question, spec, target, absence, candidates
        )
        provider_tokens = count_tokens(_canonical_payload_json(provider_projection))
        if provider_tokens <= budget.provider_payload_token_cap:
            break
        if not selected:
            raise FullStoreSlotClosureError(
                "question/operator envelope exceeds the hard prompt budget"
            )
        selected.pop()

    required = tuple(slot.slot_id for slot in spec.required_slots)
    covered_set = {
        slot_id for row in candidates for slot_id in row.supported_slot_ids
    }
    covered = tuple(slot_id for slot_id in required if slot_id in covered_set)
    unresolved = tuple(slot_id for slot_id in required if slot_id not in covered_set)
    if required and not unresolved:
        evidence_status = "all_required_slots_lexically_covered"
    elif covered:
        evidence_status = "partial_required_slot_coverage"
    elif candidates:
        evidence_status = "bounded_candidate_evidence_available"
    elif absence.may_assert_exact_literal_absence:
        evidence_status = "narrow_exact_literal_absence_only"
    else:
        evidence_status = "unresolved"

    receipt = FullStoreSlotClosureReceipt(
        question_sha256=quote_sha256(dated_question),
        typed_operator_spec_receipt_sha256=spec.receipt_sha256,
        temporal_target_receipt_sha256=target.receipt_sha256,
        absence_witness_receipt_sha256=absence.receipt_sha256,
        cache_receipt_sha256=source_cache.cache_receipt_sha256,
        window_index_receipt_sha256=index.receipt_sha256,
        window_index_reuse_mode=reuse_mode,
        budget_id=budget.budget_id,
        physical_content_rows_scanned=len(rows),
        physical_content_tokens_scanned=index.physical_content_tokens_indexed,
        physical_sentence_windows_scanned=len(index.windows),
        physical_partition_count=len(source_cache.rows_by_partition),
        query_candidate_windows_considered=candidate_windows_considered,
        candidate_population_count=len(drafts),
        selected_candidate_ids=tuple(row.candidate_id for row in candidates),
        selected_source_group_count=len(
            {row.source_group_handle for row in candidates}
        ),
        selected_evidence_tokens=sum(row.token_count for row in candidates),
        provider_payload_tokens=provider_tokens,
        required_slot_ids=required,
        covered_slot_ids=covered,
        unresolved_slot_ids=unresolved,
        role_rejected_candidate_count=role_rejected,
        selection_truncated=len(candidates) < len(drafts),
        evidence_status=evidence_status,
    )
    return FullStoreSlotClosureResult(
        dated_question=dated_question,
        operator_spec=spec,
        temporal_target=target,
        absence_witness=absence,
        candidates=candidates,
        local_bindings=bindings,
        receipt=receipt,
        budget=budget,
    )


def scan_full_store_slot_closures(
    cache: NamespacePartitionCache,
    dated_questions: Sequence[str],
    /,
    *,
    budget: FullStoreSlotClosureBudget = FullStoreSlotClosureBudget(),
) -> tuple[FullStoreSlotClosureResult, ...]:
    """Build one immutable index, then execute multiple prompt ticks."""

    _require(type(cache) is NamespacePartitionCache, "batch scan requires exact cache")
    questions = tuple(dated_questions)
    _require(
        questions
        and all(
            type(question) is str and question and question.strip() == question
            for question in questions
        ),
        "batch scan questions must be exact non-empty text",
    )
    index = build_full_store_window_index(cache)
    return tuple(
        scan_full_store_slot_closure(index, question, budget=budget)
        for question in questions
    )


def _safe_typed_mention(
    candidate: FullStoreSlotCandidate,
    spec: TypedOperatorSpec,
    *,
    dated_question: str,
) -> NumericMention | None:
    numeric_slots = {
        slot.slot_id for slot in spec.required_slots if slot.requires_numeric
    }
    if not (
        numeric_slots & set(candidate.supported_slot_ids)
        or spec.answer_shape is AnswerShape.NUMBER and not spec.required_slots
    ):
        return None
    return single_numeric_mention(
        candidate.quote,
        operator_spec=spec,
        question=dated_question,
    )


def _safe_typed_number(
    candidate: FullStoreSlotCandidate, spec: TypedOperatorSpec
) -> float | None:
    """Compatibility wrapper for callers that only need the scalar."""

    mention = _safe_typed_mention(
        candidate,
        spec,
        dated_question="",
    )
    return None if mention is None else mention.value


def _safe_typed_status(quote: str) -> str:
    if re.search(r"\b(?:cancelled|canceled|abandoned)\b", quote, re.I):
        return "cancelled"
    if re.search(r"\b(?:plan|planned|proposed|intend(?:ed)?)\b", quote, re.I):
        return "proposed"
    if re.search(r"\b(?:current|currently|right now|latest)\b", quote, re.I):
        return "current"
    if re.search(
        r"\b(?:completed|finished|bought|paid|spent|went|visited|planted|did)\b",
        quote,
        re.I,
    ):
        return "completed"
    return "unknown"


def adapt_full_store_slot_closure_to_typed_contribution(
    result: FullStoreSlotClosureResult,
    /,
    *,
    handle_start: int,
    group_start: int,
) -> "TypedEvidenceContribution":
    """Convert selected exact pointers into one bounded typed contribution.

    The caller allocates globally unique opaque H/G ranges.  The adapter uses
    ``DIRECT_POINTER`` provenance and binds each handle to the exact local
    citation receipt.  An exhaustive physical index never upgrades the common
    typed frontier beyond ``BOUNDED``.
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
        type(result) is FullStoreSlotClosureResult,
        "typed contribution requires an exact full-store result",
    )
    for value, label in (
        (handle_start, "global handle start"),
        (group_start, "global group start"),
    ):
        _require(type(value) is int and value >= 1, f"{label} must be positive")
    _require(
        handle_start + len(result.candidates) - 1 <= 999_999,
        "global handle range exceeds the opaque contract",
    )
    local_groups = tuple(
        dict.fromkeys(row.source_group_handle for row in result.candidates)
    )
    _require(
        group_start + len(local_groups) - 1 <= 999_999,
        "global group range exceeds the opaque contract",
    )
    global_groups = {
        local: f"G{group_start + index:03d}"
        for index, local in enumerate(local_groups)
    }
    sealed_artifact_sha256 = identity_sha256(result.local_audit_projection())
    bindings: list[Any] = []
    raw_items: list[dict[str, Any]] = []
    for index, (candidate, local) in enumerate(
        zip(result.candidates, result.local_bindings, strict=True)
    ):
        handle_id = f"H{handle_start + index:03d}"
        group_handle = global_groups[candidate.source_group_handle]
        binding = EvidenceHandleBinding(
            handle_id=handle_id,
            origin=EvidenceOrigin.DIRECT_POINTER,
            provenance_grade=ProvenanceGrade.DIRECT_POINTER,
            source_group_handle=group_handle,
            sealed_artifact_sha256=sealed_artifact_sha256,
            parent_receipt_sha256=result.receipt.receipt_sha256,
            evidence_receipt_sha256=local.receipt_sha256,
            payload_sha256=identity_sha256(candidate.projection()),
            citation_sha256=candidate.quote_sha256,
            citation_char_count=len(candidate.quote),
            local_source_locator_sha256=local.receipt_sha256,
        )
        mention = _safe_typed_mention(
            candidate,
            result.operator_spec,
            dated_question=result.dated_question,
        )
        numeric = None if mention is None else mention.value
        if numeric is not None:
            kind = "operand"
        elif result.operator_spec.temporal_mode.value != "none":
            kind = "event"
        elif result.operator_spec.answer_shape.value == "set_list":
            kind = "member"
        elif result.operator_spec.style.value == "state_chain":
            kind = "state"
        else:
            kind = "direct"
        relation = (
            f"authored_by_{candidate.role}"
            if candidate.role in {"user", "assistant"}
            else None
        )
        semantic_slots = tuple(
            slot
            for slot in result.operator_spec.required_slots
            if slot.slot_id in candidate.supported_slot_ids
            and slot.kind in {SlotKind.OPERAND, SlotKind.COMPARISON_SIDE}
        )
        entity_key = semantic_slots[0].label if len(semantic_slots) == 1 else None
        raw: dict[str, Any] = {
            "handle_ids": [handle_id],
            "included": True,
            "kind": kind,
            "numeric_role": "operand" if numeric is not None else "none",
            # Retrieval overlap is not an answer-value specificity certificate.
            "specificity_terms": [],
            "status": _safe_typed_status(candidate.quote),
            "summary": candidate.quote,
            "value_authority": "explicit",
        }
        if entity_key is not None:
            raw["entity_key"] = entity_key
        if candidate.event_date is not None:
            raw["date"] = candidate.event_date
        if relation is not None:
            raw["relation"] = relation
        if numeric is not None:
            raw["numeric_qualifier"] = mention.qualifier.value
            raw["numeric_value"] = numeric
            if mention.unit is not None:
                raw["unit"] = mention.unit
        bindings.append(binding)
        raw_items.append(raw)
    frozen_bindings = tuple(bindings)
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
        contribution.frontier_mode is FrontierMode.BOUNDED,
        "physical exhaustion upgraded the typed frontier",
    )
    return contribution


__all__ = [
    "ExactTermAbsenceStatus",
    "ExactTermAbsenceWitness",
    "FullStoreSlotCandidate",
    "FullStoreSlotClosureBudget",
    "FullStoreSlotClosureError",
    "FullStoreSlotClosureReceipt",
    "FullStoreSlotClosureResult",
    "FullStoreWindowIndex",
    "IndexedContentWindow",
    "LocalCitationBinding",
    "MECHANISM_ID",
    "QuestionTemporalTarget",
    "TemporalTargetMode",
    "build_full_store_window_index",
    "indexed_surface_terms",
    "adapt_full_store_slot_closure_to_typed_contribution",
    "scan_full_store_slot_closure",
    "scan_full_store_slot_closures",
]
