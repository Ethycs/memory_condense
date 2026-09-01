"""Question-only temporal bundles and narrow operand-insufficiency proofs.

This specialist is an additive retrieval lane over the immutable
``FullStoreWindowIndex``.  It addresses two failure modes which should not be
conflated with generic top-k relevance:

* temporal questions need a *bundle* of dated, first-person event assertions
  (winner/predecessor or every ordered operand), not isolated high-scoring
  sentences; and
* an all-operands numeric question may need a provider-visible statement that
  one requested entity has citations but no explicit numeric operand in the
  completely scanned, question-derived lexical scope.

The latter is intentionally narrow.  It proves absence of an explicit numeric
surface bound to an exact question-derived entity slot.  It does not prove
semantic absence, resolve pronouns, or close the common typed frontier.

The public scan accepts only a prebuilt immutable index and dated question.
There is no question ID, source prefix, partition route, reference, prediction,
target label, or provider client.  Raw source/session identifiers stay in local
bindings; provider-visible content uses opaque source-group handles.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

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
    QuestionTemporalTarget,
    TemporalTargetMode,
    _parse_datetime,
    _question_body,
    _temporal_target,
    indexed_surface_terms,
)
from .partition_scan import _bounded_excerpt
from .typed_action_semantics import canonical_action_concepts
from .typed_numeric_semantics import NumericMention, numeric_mentions, single_numeric_mention
from .typed_operator_spec import (
    RequiredSlot,
    SlotKind,
    TemporalMode,
    TypedOperatorSpec,
    compile_typed_operator_spec,
    normalized_terms,
)

if TYPE_CHECKING:
    from .typed_operator_adapter import TypedEvidenceContribution


MECHANISM_ID = "temporal_insufficiency_specialist_v1"
RESULT_FORMAT = "memory-condense-temporal-insufficiency-specialist-result-v1"
CANDIDATE_FORMAT = "memory-condense-temporal-specialist-candidate-v1"
BUNDLE_FORMAT = "memory-condense-temporal-event-bundle-v1"
SLOT_COVERAGE_FORMAT = "memory-condense-exact-numeric-slot-coverage-v1"
ABSENCE_FORMAT = "memory-condense-scoped-numeric-absence-certificate-v1"
RECEIPT_FORMAT = "memory-condense-temporal-insufficiency-specialist-receipt-v1"


class TemporalInsufficiencySpecialistError(MatchedEvalContractError):
    """Raised when a specialist route, proof, or bound loses its contract."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TemporalInsufficiencySpecialistError(message)


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(value) is str and value for value in values)
        and len(set(values)) == len(values),
        f"{label} must be ordered unique exact text",
    )
    return values


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class TemporalInsufficiencyBudget:
    """Independent lane bounds plus the complete provider envelope."""

    evidence_token_cap: int = 1_536
    max_candidates: int = 12
    max_excerpt_tokens: int = 192
    max_bundle_members: int = 12
    max_interval_comparators: int = 3
    max_candidates_per_source: int = 1
    max_story_link_candidates: int = 32
    max_story_combinations: int = 50_000
    hard_prompt_token_cap: int = 8_000
    output_token_reserve: int = 768
    protocol_token_reserve: int = 512

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            _require(type(value) is int and value > 0, f"{name} must be positive")
        _require(
            self.max_candidates_per_source == 1,
            "the specialist requires one representative per source/session",
        )
        _require(
            self.max_bundle_members <= self.max_candidates,
            "bundle cap exceeds candidate cap",
        )
        _require(
            self.max_interval_comparators < self.max_bundle_members,
            "interval comparator cap must leave room for its boundary",
        )
        _require(
            self.max_story_link_candidates >= self.max_bundle_members,
            "story-link candidate cap is smaller than the bundle cap",
        )
        _require(
            self.evidence_token_cap
            + self.output_token_reserve
            + self.protocol_token_reserve
            <= self.hard_prompt_token_cap,
            "specialist reserves exceed the hard prompt cap",
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


class SpecialistRoute(str, Enum):
    TEMPORAL_RELATIVE = "temporal_relative"
    TEMPORAL_INTERVAL = "temporal_interval"
    TEMPORAL_ORDER = "temporal_order"
    TEMPORAL_LATEST = "temporal_latest"
    NUMERIC_SLOT_INSUFFICIENCY = "numeric_slot_insufficiency"


class BundleRole(str, Enum):
    WINNER = "winner"
    PREDECESSOR = "predecessor"
    ORDERED_OPERAND = "ordered_operand"
    CORROBORATING = "corroborating"
    ALTERNATE = "alternate"
    SLOT_SUPPORT = "slot_support"


@dataclass(frozen=True, slots=True)
class TemporalSpecialistCandidate:
    """One provider-visible exact assertion and its opaque source grouping."""

    candidate_id: str
    source_group_handle: str
    quote: str
    quote_sha256: str
    token_count: int
    role: str
    created_at: str
    event_date: str | None
    event_date_basis: str | None
    temporal_relation: str
    domain_affinity_terms: tuple[str, ...]
    matched_query_terms: tuple[str, ...]
    supported_slot_ids: tuple[str, ...]
    explicit_numeric_slot_ids: tuple[str, ...]
    first_person_assertion: bool
    bundle_role: BundleRole
    selection_axes: tuple[str, ...]
    citation_binding_receipt_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.candidate_id, "specialist candidate")
        _require(
            re.fullmatch(r"G[0-9]{3,6}", self.source_group_handle) is not None,
            "specialist group handle must be opaque",
        )
        require_text(self.quote, "specialist exact quote")
        _require(
            self.quote_sha256 == quote_sha256(self.quote)
            and self.token_count == count_tokens(self.quote),
            "specialist quote bytes changed",
        )
        require_text(self.role, "specialist role")
        require_text(self.created_at, "specialist created-at")
        if self.event_date is not None:
            require_text(self.event_date, "specialist event date")
            require_text(self.event_date_basis or "", "specialist event-date basis")
        else:
            _require(self.event_date_basis is None, "undated candidate has date basis")
        require_text(self.temporal_relation, "specialist temporal relation")
        for values, label in (
            (self.domain_affinity_terms, "domain affinity terms"),
            (self.matched_query_terms, "matched query terms"),
            (self.supported_slot_ids, "supported slots"),
            (self.explicit_numeric_slot_ids, "explicit numeric slots"),
            (self.selection_axes, "selection axes"),
        ):
            _ordered_unique(values, label)
        _require(
            set(self.explicit_numeric_slot_ids) <= set(self.supported_slot_ids),
            "numeric slot is not supported by its exact quote",
        )
        _require(type(self.first_person_assertion) is bool, "first-person flag changed")
        _require(type(self.bundle_role) is BundleRole, "bundle role changed")
        require_sha256(
            self.citation_binding_receipt_sha256, "candidate citation binding"
        )

    def projection(self) -> dict[str, Any]:
        return {
            "bundle_role": self.bundle_role.value,
            "candidate_id": self.candidate_id,
            "citation_binding_receipt_sha256": self.citation_binding_receipt_sha256,
            "created_at": self.created_at,
            "domain_affinity_terms": list(self.domain_affinity_terms),
            "event_date": self.event_date,
            "event_date_basis": self.event_date_basis,
            "explicit_numeric_slot_ids": list(self.explicit_numeric_slot_ids),
            "first_person_assertion": self.first_person_assertion,
            "format": CANDIDATE_FORMAT,
            "matched_query_terms": list(self.matched_query_terms),
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "selection_axes": list(self.selection_axes),
            "source_group_handle": self.source_group_handle,
            "supported_slot_ids": list(self.supported_slot_ids),
            "temporal_relation": self.temporal_relation,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class TemporalEventBundle:
    route: str
    requested_cardinality: int | None
    ordered_candidate_ids: tuple[str, ...]
    winner_candidate_id: str | None
    predecessor_candidate_id: str | None
    query_time: str | None
    target_date: str | None
    population_count: int
    truncated: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.route, "temporal bundle route")
        ordered = _ordered_unique(self.ordered_candidate_ids, "temporal bundle members")
        if self.requested_cardinality is not None:
            _require(
                type(self.requested_cardinality) is int
                and self.requested_cardinality > 0,
                "temporal bundle cardinality changed",
            )
        for value, label in (
            (self.winner_candidate_id, "temporal winner"),
            (self.predecessor_candidate_id, "temporal predecessor"),
            (self.query_time, "temporal query time"),
            (self.target_date, "temporal target date"),
        ):
            if value is not None:
                require_text(value, label)
        _require(
            self.winner_candidate_id in {None, *ordered}
            and self.predecessor_candidate_id in {None, *ordered},
            "temporal relation escaped its bundle",
        )
        _require(
            type(self.population_count) is int
            and self.population_count >= len(ordered)
            and type(self.truncated) is bool
            and self.truncated is (len(ordered) < self.population_count),
            "temporal bundle population/truncation changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "temporal bundle changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "format": BUNDLE_FORMAT,
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "population_count": self.population_count,
            "predecessor_candidate_id": self.predecessor_candidate_id,
            "query_time": self.query_time,
            "requested_cardinality": self.requested_cardinality,
            "route": self.route,
            "target_date": self.target_date,
            "truncated": self.truncated,
            "winner_candidate_id": self.winner_candidate_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ExactNumericSlotCoverage:
    slot_id: str
    slot_label: str
    exact_entity_terms: tuple[str, ...]
    entity_assertion_window_count: int
    entity_assertion_source_count: int
    explicit_numeric_assertion_window_count: int
    explicit_numeric_assertion_source_count: int
    scope_has_grounded_predicate_assertion: bool
    explicit_numeric_operand_missing: bool
    selected_supporting_candidate_ids: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.slot_id, "numeric coverage slot")
        require_text(self.slot_label, "numeric coverage label")
        _ordered_unique(self.exact_entity_terms, "numeric coverage entity terms")
        for value, label in (
            (self.entity_assertion_window_count, "entity assertion windows"),
            (self.entity_assertion_source_count, "entity assertion sources"),
            (self.explicit_numeric_assertion_window_count, "numeric assertion windows"),
            (self.explicit_numeric_assertion_source_count, "numeric assertion sources"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        _require(
            self.explicit_numeric_assertion_window_count
            <= self.entity_assertion_window_count
            and self.explicit_numeric_assertion_source_count
            <= self.entity_assertion_source_count,
            "numeric coverage exceeds entity coverage",
        )
        _require(
            type(self.scope_has_grounded_predicate_assertion) is bool,
            "numeric coverage scope-grounding flag changed",
        )
        expected_missing = bool(
            self.scope_has_grounded_predicate_assertion
            and self.explicit_numeric_assertion_window_count == 0
        )
        _require(
            type(self.explicit_numeric_operand_missing) is bool
            and self.explicit_numeric_operand_missing is expected_missing,
            "numeric operand absence is not justified",
        )
        _ordered_unique(
            self.selected_supporting_candidate_ids,
            "selected numeric coverage candidates",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "numeric coverage changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "entity_assertion_source_count": self.entity_assertion_source_count,
            "entity_assertion_window_count": self.entity_assertion_window_count,
            "exact_entity_terms": list(self.exact_entity_terms),
            "explicit_numeric_assertion_source_count": (
                self.explicit_numeric_assertion_source_count
            ),
            "explicit_numeric_assertion_window_count": (
                self.explicit_numeric_assertion_window_count
            ),
            "explicit_numeric_operand_missing": self.explicit_numeric_operand_missing,
            "format": SLOT_COVERAGE_FORMAT,
            "selected_supporting_candidate_ids": list(
                self.selected_supporting_candidate_ids
            ),
            "scope_has_grounded_predicate_assertion": (
                self.scope_has_grounded_predicate_assertion
            ),
            "slot_id": self.slot_id,
            "slot_label": self.slot_label,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ScopedNumericAbsenceCertificate:
    applicable: bool
    scope_definition: str
    window_index_receipt_sha256: str
    physical_content_rows_scanned: int
    physical_sentence_windows_scanned: int
    scoped_source_count: int
    scoped_content_row_count: int
    every_exact_entity_posting_scanned: bool
    every_scoped_source_row_scanned: bool
    slot_coverage: tuple[ExactNumericSlotCoverage, ...]
    may_conclude_operator_insufficient: bool
    semantic_absence_may_be_inferred: Literal[False] = False
    provider_instruction: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.applicable) is bool, "absence applicability changed")
        require_text(self.scope_definition, "absence scope definition")
        require_sha256(self.window_index_receipt_sha256, "absence window index")
        for value, label in (
            (self.physical_content_rows_scanned, "absence physical rows"),
            (self.physical_sentence_windows_scanned, "absence physical windows"),
            (self.scoped_source_count, "absence scoped sources"),
            (self.scoped_content_row_count, "absence scoped rows"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        _require(
            type(self.every_exact_entity_posting_scanned) is bool
            and type(self.every_scoped_source_row_scanned) is bool,
            "absence exhaustion flags changed",
        )
        _require(
            type(self.slot_coverage) is tuple
            and all(type(row) is ExactNumericSlotCoverage for row in self.slot_coverage),
            "absence slot coverage changed",
        )
        missing = any(row.explicit_numeric_operand_missing for row in self.slot_coverage)
        expected = bool(
            self.applicable
            and missing
            and self.every_exact_entity_posting_scanned
            and self.every_scoped_source_row_scanned
        )
        _require(
            self.may_conclude_operator_insufficient is expected,
            "operator insufficiency is not justified by the exact scope",
        )
        _require(
            self.semantic_absence_may_be_inferred is False,
            "narrow numeric absence cannot prove semantic absence",
        )
        if self.provider_instruction is not None:
            require_text(self.provider_instruction, "absence provider instruction")
        _require(
            (self.provider_instruction is not None)
            is self.may_conclude_operator_insufficient,
            "absence instruction escaped its complete proof",
        )
        expected_receipt = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected_receipt, "absence certificate changed")
        object.__setattr__(self, "receipt_sha256", expected_receipt)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "applicable": self.applicable,
            "every_exact_entity_posting_scanned": (
                self.every_exact_entity_posting_scanned
            ),
            "every_scoped_source_row_scanned": self.every_scoped_source_row_scanned,
            "format": ABSENCE_FORMAT,
            "may_conclude_operator_insufficient": (
                self.may_conclude_operator_insufficient
            ),
            "physical_content_rows_scanned": self.physical_content_rows_scanned,
            "physical_sentence_windows_scanned": (
                self.physical_sentence_windows_scanned
            ),
            "provider_instruction": self.provider_instruction,
            "scope_definition": self.scope_definition,
            "scoped_content_row_count": self.scoped_content_row_count,
            "scoped_source_count": self.scoped_source_count,
            "semantic_absence_may_be_inferred": False,
            "slot_coverage": [row.projection() for row in self.slot_coverage],
            "window_index_receipt_sha256": self.window_index_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class TemporalInsufficiencyReceipt:
    question_sha256: str
    operator_spec_receipt_sha256: str
    temporal_target_receipt_sha256: str
    window_index_receipt_sha256: str
    absence_certificate_receipt_sha256: str
    temporal_bundle_receipt_sha256: str | None
    budget_id: str
    routes: tuple[str, ...]
    physical_content_rows_scanned: int
    physical_sentence_windows_scanned: int
    temporal_candidate_population_count: int
    scoped_numeric_candidate_population_count: int
    selected_candidate_ids: tuple[str, ...]
    selected_source_group_count: int
    selected_evidence_tokens: int
    provider_payload_tokens: int
    selection_truncated: bool
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
            (self.question_sha256, "specialist question"),
            (self.operator_spec_receipt_sha256, "specialist operator"),
            (self.temporal_target_receipt_sha256, "specialist temporal target"),
            (self.window_index_receipt_sha256, "specialist window index"),
            (self.absence_certificate_receipt_sha256, "specialist absence certificate"),
            (self.budget_id, "specialist budget"),
        ):
            require_sha256(value, label)
        if self.temporal_bundle_receipt_sha256 is not None:
            require_sha256(self.temporal_bundle_receipt_sha256, "specialist bundle")
        _ordered_unique(self.routes, "specialist routes")
        _ordered_unique(self.selected_candidate_ids, "specialist selected candidates")
        for value, label in (
            (self.physical_content_rows_scanned, "specialist physical rows"),
            (self.physical_sentence_windows_scanned, "specialist physical windows"),
            (self.temporal_candidate_population_count, "temporal population"),
            (self.scoped_numeric_candidate_population_count, "numeric population"),
            (self.selected_source_group_count, "selected group count"),
            (self.selected_evidence_tokens, "selected evidence tokens"),
            (self.provider_payload_tokens, "specialist provider tokens"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        _require(type(self.selection_truncated) is bool, "truncation flag changed")
        _require(
            self.question_id_filter_used is False
            and self.known_source_prefix_filter_used is False
            and self.partition_routing_used is False
            and self.raw_source_ids_provider_visible is False,
            "specialist used a forbidden route or exposed a source ID",
        )
        _require(
            self.new_provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "specialist must be provider-free, zero-state, and gold-blind",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "specialist receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="temporal_insufficiency_receipt")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "absence_certificate_receipt_sha256": (
                self.absence_certificate_receipt_sha256
            ),
            "budget_id": self.budget_id,
            "gold_loaded": False,
            "known_source_prefix_filter_used": False,
            "new_provider_calls": 0,
            "operator_spec_receipt_sha256": self.operator_spec_receipt_sha256,
            "partition_routing_used": False,
            "physical_content_rows_scanned": self.physical_content_rows_scanned,
            "physical_sentence_windows_scanned": (
                self.physical_sentence_windows_scanned
            ),
            "provider_payload_tokens": self.provider_payload_tokens,
            "question_id_filter_used": False,
            "question_sha256": self.question_sha256,
            "raw_source_ids_provider_visible": False,
            "retained_transformer_token_state_bytes": 0,
            "routes": list(self.routes),
            "scoped_numeric_candidate_population_count": (
                self.scoped_numeric_candidate_population_count
            ),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_evidence_tokens": self.selected_evidence_tokens,
            "selected_source_group_count": self.selected_source_group_count,
            "selection_truncated": self.selection_truncated,
            "temporal_bundle_receipt_sha256": self.temporal_bundle_receipt_sha256,
            "temporal_candidate_population_count": (
                self.temporal_candidate_population_count
            ),
            "temporal_target_receipt_sha256": self.temporal_target_receipt_sha256,
            "window_index_receipt_sha256": self.window_index_receipt_sha256,
            "format": RECEIPT_FORMAT,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class TemporalInsufficiencyResult:
    dated_question: str
    operator_spec: TypedOperatorSpec
    temporal_target: QuestionTemporalTarget
    routes: tuple[SpecialistRoute, ...]
    candidates: tuple[TemporalSpecialistCandidate, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    temporal_bundle: TemporalEventBundle | None
    absence_certificate: ScopedNumericAbsenceCertificate
    receipt: TemporalInsufficiencyReceipt
    budget: TemporalInsufficiencyBudget

    def __post_init__(self) -> None:
        require_text(self.dated_question, "specialist dated question")
        _require(type(self.operator_spec) is TypedOperatorSpec, "operator spec changed")
        _require(type(self.temporal_target) is QuestionTemporalTarget, "target changed")
        _require(
            type(self.routes) is tuple
            and all(type(row) is SpecialistRoute for row in self.routes)
            and len(set(self.routes)) == len(self.routes),
            "specialist routes changed",
        )
        _require(
            type(self.candidates) is tuple
            and all(type(row) is TemporalSpecialistCandidate for row in self.candidates),
            "specialist candidates changed",
        )
        _require(
            type(self.local_bindings) is tuple
            and all(type(row) is LocalCitationBinding for row in self.local_bindings),
            "specialist local bindings changed",
        )
        candidate_ids = tuple(row.candidate_id for row in self.candidates)
        _require(
            candidate_ids
            == tuple(row.candidate_id for row in self.local_bindings)
            == self.receipt.selected_candidate_ids,
            "specialist candidates lost exact local citations",
        )
        _require(
            all(
                candidate.citation_binding_receipt_sha256 == binding.receipt_sha256
                and candidate.source_group_handle == binding.source_group_handle
                for candidate, binding in zip(
                    self.candidates, self.local_bindings, strict=True
                )
            ),
            "specialist candidate binding changed",
        )
        _require(
            len({binding.source_id for binding in self.local_bindings})
            == len(self.local_bindings),
            "specialist retained more than one representative per source/session",
        )
        if self.temporal_bundle is not None:
            _require(
                set(self.temporal_bundle.ordered_candidate_ids) <= set(candidate_ids),
                "temporal bundle escaped selected candidates",
            )
        payload_tokens = count_tokens(_canonical_json(self.provider_projection()))
        _require(
            payload_tokens == self.receipt.provider_payload_tokens,
            "specialist provider payload accounting changed",
        )
        _require(
            payload_tokens
            + self.budget.output_token_reserve
            + self.budget.protocol_token_reserve
            <= self.budget.hard_prompt_token_cap,
            "specialist provider projection exceeds the hard prompt cap",
        )
        assert_gold_blind(
            self.provider_projection(), path="temporal_insufficiency_provider_payload"
        )

    def provider_projection(self) -> dict[str, Any]:
        return {
            "absence_certificate": self.absence_certificate.projection(),
            "candidates": [row.projection() for row in self.candidates],
            "dated_question": self.dated_question,
            "format": RESULT_FORMAT,
            "operator_spec": self.operator_spec.projection(),
            "routes": [row.value for row in self.routes],
            "temporal_bundle": (
                None if self.temporal_bundle is None else self.temporal_bundle.projection()
            ),
            "temporal_target": self.temporal_target.projection(),
        }

    def local_audit_projection(self) -> dict[str, Any]:
        return {
            "bindings": [row.projection() for row in self.local_bindings],
            "provider_payload_sha256": identity_sha256(self.provider_projection()),
            "receipt": self.receipt.projection(),
        }


@dataclass(frozen=True, slots=True)
class _Draft:
    row_index: int
    window_index: int
    source_id: str
    start_char: int
    end_char: int
    quote: str
    event_date: str | None
    event_date_basis: str | None
    temporal_relation: str
    domain_terms: tuple[str, ...]
    matched_query_terms: tuple[str, ...]
    supported_slot_ids: tuple[str, ...]
    numeric_slot_ids: tuple[str, ...]
    terms: frozenset[str]
    local_score: float
    link_score: float
    selection_axes: tuple[str, ...]
    bundle_role: BundleRole

    @property
    def candidate_id(self) -> str:
        row = self._row
        span = EvidenceSpan(
            chunk_id=row.chunk_id,
            start_char=self.start_char,
            end_char=self.end_char,
            quote_sha256=quote_sha256(self.quote),
            ordinal=row.ordinal,
            source_id=row.source_id,
            turn_start_char=row.turn_start_char,
            turn_id=row.turn_id,
            role=row.role,
            created_at=row.created_at,
        )
        return identity_sha256(
            {"atom_id": make_atom_id(span), "mechanism_id": MECHANISM_ID}
        )

    # Attached by `_bind_rows`; excluded from comparisons/projections.
    _row: Any = None

    @property
    def score(self) -> tuple[Any, ...]:
        return (
            len(self.numeric_slot_ids),
            len(self.supported_slot_ids),
            self.local_score,
            # A whole-session link is supporting evidence, never permission
            # to replace the local assertion that actually answers the
            # question.  In long conversations raw graph degree can be orders
            # of magnitude larger than every local axis, so it must refine an
            # event score rather than precede it.
            self.link_score,
            len(self.domain_terms),
            len(self.matched_query_terms),
            -(self._row.ordinal if self._row is not None else self.row_index),
            -self.start_char,
        )


_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|me|my|mine|we|our|ours)\b|\bI['’](?:m|ve|d|ll)\b", re.I
)
_RECOLLECTION_RE = re.compile(
    r"\b(?:remember(?:ed|ing)?|recalled?|thinking about|reminded me)\b", re.I
)
_PROPOSED_RE = re.compile(
    r"\b(?:plan(?:ned|ning)?|intend(?:ed|ing)?|hope to|want to|will)\b", re.I
)
_PAST_EVENT_RE = re.compile(
    r"\b(?:did|went|visited|took|got|bought|planted|started|opened|launched|"
    r"reached|hit|crossed|completed|finished|sold|signed|earned|made|returned|"
    r"camped|hiked|drove|flew|stayed|attended|joined|won|achieved|"
    r"participat(?:e|es|ed|ing)|competed|raced)\b",
    re.I,
)


def _completed_event_surface(text: str) -> bool:
    """Return whether a clause asserts an event before any proposal marker.

    A source turn can contain both a completed/current-day event and a later
    intention (``I participated ... and I want ...``).  Treating the whole
    turn as proposed discards the answer-bearing assertion.  Conversely, an
    infinitive such as ``I want to participate`` must not become a completed
    event merely because ``participate`` is an event verb.  Sentence and
    adversative-clause boundaries are enough to distinguish those cases
    without relying on a question-specific activity inventory.
    """

    for clause in re.split(r"(?<=[.!?;])\s+|\bbut\b", text, flags=re.I):
        event = _PAST_EVENT_RE.search(clause)
        if event is None:
            continue
        proposal = _PROPOSED_RE.search(clause)
        if proposal is None or event.start() < proposal.start():
            return True
    return False
_MILESTONE_EVENT_RE = re.compile(
    r"\b(?:first\s+(?:client|customer|sale|contract|employee)|"
    r"signed\s+(?:a|the|my)?\s*contract|launched\s+(?:my|our|the|a)?\s*"
    r"(?:website|business|company|product|service|store|shop)|"
    r"opened\s+(?:my|our|the|a)?\s*(?:business|company|store|shop)|"
    r"reached|hit|crossed|achieved|anniversary)\b",
    re.I,
)
_TRAVEL_EVENT_RE = re.compile(
    r"\b(?:got\s+back\s+from|returned\s+from|went\s+(?:on|to)|"
    r"took\s+(?:a|the|my)?|drove\s+(?:on|to|through)|hiked|camped|"
    r"visited|traveled)\b[^.!?]{0,96}\b(?:trip|travel|vacation|visit|"
    r"tour|hike|camp(?:ing)?|park|woods|monument)\b|"
    r"\b(?:trip|travel|vacation|visit|tour|hike|camp(?:ing)?)\b"
    r"[^.!?]{0,64}\b(?:today|yesterday|recently|ago|last\s+\w+)\b",
    re.I,
)
_SINGULAR_FRIEND_RE = re.compile(
    r"\bwith\s+(?:a|one|my)\s+friend\b|\ba friend (?:who|that)\b", re.I
)
_PLURAL_FRIEND_RE = re.compile(r"\bwith\s+(?:my\s+)?friends\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_CARDINALITY_RE = re.compile(
    r"\b(?:order|sequence)\s+of\s+(?:the\s+)?(?P<n>\d+|two|three|four|five|six|seven|eight|nine|ten)\b|"
    r"\bthe\s+(?P<m>\d+|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:trips?|events?|visits?|activities?)\b",
    re.I,
)
_NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_OPERATOR_TERMS = frozenset(
    {
        "activity",
        "ago",
        "all",
        "earliest",
        "event",
        "four",
        "how",
        "initial",
        "initially",
        "last",
        "late",
        "latest",
        "many",
        "mention",
        "month",
        "order",
        "pass",
        "past",
        "relate",
        "recent",
        "sequence",
        "significant",
        "since",
        "three",
        "time",
        "took",
        "two",
        "week",
        "what",
        "when",
    }
)


def _norm_words(value: str) -> tuple[str, ...]:
    surfaces = _WORD_RE.findall(value.replace("-", " ").replace("–", " "))
    return tuple(
        dict.fromkeys(
            term
            for surface in surfaces
            for term in normalized_terms(surface)
            if term
        )
    )


def _normalized_inventory(values: Sequence[str]) -> frozenset[str]:
    return frozenset(term for value in values for term in _norm_words(value))


_SPORTS_EVENT_TERMS = _normalized_inventory(
    (
        "sport", "sports", "athletic", "competition", "race", "racing",
        "run", "running", "5K", "10K", "marathon", "triathlon",
        "soccer", "football", "basketball", "baseball", "tennis",
        "tournament", "league", "volleyball",
        "cycling", "swimming",
    )
)


_DOMAIN_FAMILIES: Mapping[str, frozenset[str]] = {
    "garden": _normalized_inventory(
        (
            "garden", "gardening", "plant", "planted", "flower", "herb",
            "vegetable", "seed", "soil", "nursery", "succulent", "lily",
            "tomato", "pepper", "cucumber", "pot", "rosemary", "mint",
        )
    ),
    "travel": _normalized_inventory(
        (
            "trip", "travel", "journey", "vacation", "visit", "visited",
            "hike", "hiked", "camp", "camped", "road trip", "drive", "drove",
            "flight", "flew", "tour", "woods", "park", "Yosemite",
            "Yellowstone", "Big Sur", "Monterey", "Muir Woods",
        )
    ),
    "museum": _normalized_inventory(
        (
            "museum", "gallery", "exhibit", "exhibition", "tour", "art",
            "science museum", "curator",
        )
    ),
    "sports": _SPORTS_EVENT_TERMS,
    "business": _normalized_inventory(
        (
            "business", "buisiness", "company", "startup", "store", "shop",
            "client", "customer", "revenue", "profit", "sale", "sold",
            "contract", "launch", "launched", "employee", "team", "order",
            "milestone", "anniversary", "million", "operating", "opened",
        )
    ),
}


_INTERVAL_PROPOSED_COMPARATOR_RE = re.compile(
    r"\b(?:thinking of|considering|looking forward to|excited to|"
    r"going to|wanted to|booking)\b",
    re.I,
)


def _query_domain_terms(body: str, spec: TypedOperatorSpec) -> frozenset[str]:
    seeds = set(_norm_words(body))
    seeds.update(
        term
        for slot in spec.required_slots
        for term in slot.match_terms
        if term not in _OPERATOR_TERMS
    )
    seeds -= _OPERATOR_TERMS
    expanded = set(seeds)
    for terms in _DOMAIN_FAMILIES.values():
        if seeds & terms:
            expanded.update(terms)
    return frozenset(expanded)


def _requested_cardinality(body: str, spec: TypedOperatorSpec) -> int | None:
    if spec.cardinality is not None:
        return spec.cardinality
    match = _CARDINALITY_RE.search(body)
    if match is None:
        return None
    value = (match.group("n") or match.group("m")).casefold()
    return int(value) if value.isdigit() else _NUMBER_WORDS[value]


def _routes(spec: TypedOperatorSpec) -> tuple[SpecialistRoute, ...]:
    routes: list[SpecialistRoute] = []
    if spec.temporal_mode is TemporalMode.RELATIVE_SELECT:
        routes.append(SpecialistRoute.TEMPORAL_RELATIVE)
    elif spec.temporal_mode is TemporalMode.INTERVAL:
        routes.append(SpecialistRoute.TEMPORAL_INTERVAL)
    elif spec.temporal_mode is TemporalMode.ORDER:
        routes.append(SpecialistRoute.TEMPORAL_ORDER)
    elif spec.temporal_mode is TemporalMode.LATEST_STATE:
        routes.append(SpecialistRoute.TEMPORAL_LATEST)
    numeric_slots = tuple(slot for slot in spec.required_slots if slot.requires_numeric)
    if spec.requires_all_slots and len(numeric_slots) >= 2:
        routes.append(SpecialistRoute.NUMERIC_SLOT_INSUFFICIENCY)
    return tuple(routes)


def _slot_positions(text: str, slot: RequiredSlot) -> tuple[int, ...]:
    folded = text.casefold().replace("’", "'")
    positions: list[int] = []
    matched = 0
    for raw in slot.match_terms:
        aliases = tuple(dict.fromkeys((raw, *indexed_surface_terms(raw))))
        best: int | None = None
        for alias in aliases:
            found = re.search(rf"(?<!\w){re.escape(alias)}(?:s|es)?(?!\w)", folded, re.I)
            if found is not None:
                best = found.start() if best is None else min(best, found.start())
        if best is not None:
            matched += 1
            positions.append(best)
    return tuple(positions) if matched >= slot.minimum_match_term_count else ()


def _numeric_slot_assignment(
    text: str,
    slots: Sequence[RequiredSlot],
    mentions: Sequence[NumericMention],
) -> dict[str, tuple[NumericMention, ...]]:
    positions = {slot.slot_id: _slot_positions(text, slot) for slot in slots}
    assigned: dict[str, list[NumericMention]] = {slot.slot_id: [] for slot in slots}
    for mention in mentions:
        center = (mention.start + mention.end) / 2
        distances = {
            slot_id: min(abs(center - position) for position in values)
            for slot_id, values in positions.items()
            if values
        }
        if not distances:
            continue
        distance = min(distances.values())
        winners = tuple(slot_id for slot_id, value in distances.items() if value == distance)
        if len(winners) == 1 and distance <= 96:
            assigned[winners[0]].append(mention)
    return {slot_id: tuple(values) for slot_id, values in assigned.items()}


def _participant_supported(spec: TypedOperatorSpec, quote: str) -> bool:
    singular = any(
        slot.relation_constraint == "participant_singular"
        for slot in spec.required_slots
    )
    if not singular:
        return True
    return bool(_SINGULAR_FRIEND_RE.search(quote) and not _PLURAL_FRIEND_RE.search(quote))


def _temporal_relation(
    event_date: str | None,
    target: QuestionTemporalTarget,
    spec: TypedOperatorSpec,
) -> tuple[str, bool]:
    if event_date is None:
        return "undated", False
    event = date.fromisoformat(event_date)
    asked = _parse_datetime(target.asked_at or spec.query_timestamp)
    if target.mode is TemporalTargetMode.EXACT_DAY:
        wanted = date.fromisoformat(target.target_date or "")
        delta = (event - wanted).days
        if delta == 0:
            return "exact_target_day", True
        # Natural-language offsets such as "four weeks ago" are commonly
        # attached to day-granularity memories one or two days either side of
        # the arithmetic target.  Keep that near-day candidate as a comparator
        # rather than silently replacing the exact-day winner.
        if abs(delta) <= 3:
            return "near_target_day", True
        # A relative selector needs the preceding domain event to make the
        # target-vs-comparator relation visible to the final model.
        if -31 <= delta < -3:
            return "predecessor_before_target", True
        return "outside_target_day", False
    if target.mode is TemporalTargetMode.LOOKBACK_WINDOW:
        _require(asked is not None, "lookback route lost query timestamp")
        days = (asked.date() - event).days
        within = 0 <= days <= (target.lookback_days or 0)
        return ("within_lookback_window", True) if within else (
            "outside_lookback_window",
            False,
        )
    if spec.temporal_mode in {
        TemporalMode.INTERVAL,
        TemporalMode.LATEST_STATE,
        TemporalMode.ORDER,
    }:
        if asked is None:
            return "dated_without_query_time", True
        before = event <= asked.date()
        return ("at_or_before_query_time", True) if before else (
            "after_query_time",
            False,
        )
    return "dated_no_temporal_filter", True


def _bounded_exact_window(
    text: str,
    start: int,
    end: int,
    query_terms: frozenset[str],
    max_tokens: int,
) -> tuple[int, int, str]:
    quote = text[start:end]
    if count_tokens(quote) <= max_tokens:
        return start, end, quote
    local_start, local_end, excerpt = _bounded_excerpt(
        quote, query_terms, max_tokens=max_tokens
    )
    return start + local_start, start + local_end, excerpt


def _supported_slots(
    quote: str,
    spec: TypedOperatorSpec,
    numeric_slots: Sequence[RequiredSlot],
    *,
    dated_question: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mentions = numeric_mentions(
        quote, operator_spec=spec, question=dated_question
    )
    numeric_by_slot = _numeric_slot_assignment(quote, numeric_slots, mentions)
    supported: list[str] = []
    explicit_numeric: list[str] = []
    for slot in spec.required_slots:
        if not _slot_positions(quote, slot):
            continue
        if slot.requires_numeric:
            if numeric_by_slot.get(slot.slot_id):
                supported.append(slot.slot_id)
                explicit_numeric.append(slot.slot_id)
        elif (
            slot.relation_constraint == "participant_singular"
            and not _participant_supported(spec, quote)
        ):
            continue
        else:
            supported.append(slot.slot_id)
    return tuple(supported), tuple(explicit_numeric)


def _bind_rows(drafts: Sequence[_Draft], index: FullStoreWindowIndex) -> tuple[_Draft, ...]:
    return tuple(replace(draft, _row=index.rows[draft.row_index]) for draft in drafts)


def _temporal_drafts(
    index: FullStoreWindowIndex,
    dated_question: str,
    spec: TypedOperatorSpec,
    target: QuestionTemporalTarget,
    route: SpecialistRoute | None,
    budget: TemporalInsufficiencyBudget,
) -> tuple[_Draft, ...]:
    if route is None:
        return ()
    body = _question_body(dated_question)
    query_terms = tuple(_norm_words(body))
    query_term_set = frozenset(query_terms)
    domain = _query_domain_terms(body, spec)
    numeric_slots = tuple(slot for slot in spec.required_slots if slot.requires_numeric)
    milestone_requested = "milestone" in query_term_set
    travel_event_requested = bool(
        query_term_set & _normalized_inventory(("trip", "travel", "vacation"))
    )
    sports_event_requested = bool(
        query_term_set & _normalized_inventory(("sport", "sports", "athletic"))
    )
    row_index_by_chunk = {row.chunk_id: offset for offset, row in enumerate(index.rows)}
    raw: list[_Draft] = []
    for window_index, window in enumerate(index.windows):
        row = window.row
        if row.role != "user":
            continue
        start, end, quote = _bounded_exact_window(
            row.text,
            window.start_char,
            window.end_char,
            domain or query_term_set,
            budget.max_excerpt_tokens,
        )
        if not _FIRST_PERSON_RE.search(quote):
            continue
        relation, in_scope = _temporal_relation(
            window.event_date, target, spec
        )
        if not in_scope:
            continue
        terms = frozenset(_norm_words(quote))
        domain_hits = tuple(sorted(terms & domain))
        query_hits = tuple(term for term in query_terms if term in terms)
        supported, explicit_numeric = _supported_slots(
            quote,
            spec,
            numeric_slots,
            dated_question=dated_question,
        )
        participant_ok = _participant_supported(spec, quote)
        actions = canonical_action_concepts(quote)
        concrete_event_surface = _completed_event_surface(quote)
        milestone_event_surface = bool(
            milestone_requested and _MILESTONE_EVENT_RE.search(quote)
        )
        travel_event_surface = bool(
            travel_event_requested and _TRAVEL_EVENT_RE.search(quote)
        )
        sports_event_surface = bool(
            sports_event_requested and terms & _SPORTS_EVENT_TERMS
        )
        event_surface = bool(concrete_event_surface or actions)
        if not domain_hits and not supported:
            continue
        # Recollections are useful corroboration but cannot outrank a direct
        # event assertion with the same date/domain support.
        direct = bool(event_surface and not _RECOLLECTION_RE.search(quote))
        proposed = bool(_PROPOSED_RE.search(quote))
        # Canonical action extraction can recognize the object of a plan, but
        # that does not turn the plan into a completed dated event.  A mixed
        # sentence may still contain an explicit completed surface (for
        # example, "I signed ... and I want ..."), which remains admissible.
        if proposed and not spec.include_proposed and not concrete_event_surface:
            continue
        lexical = sum(
            math.log((len(index.windows) + 1) / (len(index.term_postings.get(term, ())) + 1))
            + 1.0
            for term in query_hits
        )
        score = (
            5.0 * len(supported)
            + 3.0 * len(domain_hits)
            + min(lexical, 8.0)
            + 3.0 * int(relation == "exact_target_day")
            + 2.0 * int(relation == "near_target_day")
            + 8.0 * int(concrete_event_surface)
            + 12.0 * int(milestone_event_surface)
            + 8.0 * int(travel_event_surface)
            + 12.0 * int(sports_event_surface)
            + 2.0 * int(direct)
            + 3.0 * int(window.contains_numeric_value)
            + 1.0 * len(actions)
            + 4.0 * int(participant_ok)
            - 1.5 * int(bool(_RECOLLECTION_RE.search(quote)))
        )
        axes = ["first_person_dated_assertion", "event_domain_affinity"]
        if relation == "exact_target_day":
            axes.append("question_derived_exact_day")
        elif relation == "within_lookback_window":
            axes.append("question_derived_lookback_window")
        if supported:
            axes.append("required_slot_support")
        if participant_ok and any(
            slot.relation_constraint == "participant_singular"
            for slot in spec.required_slots
        ):
            axes.append("singular_participant_relation_support")
        if direct:
            axes.append("direct_event_assertion")
        if concrete_event_surface:
            axes.append("concrete_completed_event_surface")
        if milestone_event_surface:
            axes.append("question_milestone_event_surface")
        if travel_event_surface:
            axes.append("requested_travel_event_surface")
        if sports_event_surface:
            axes.append("requested_sports_event_surface")
        raw.append(
            _Draft(
                row_index=row_index_by_chunk[row.chunk_id],
                window_index=window_index,
                source_id=row.source_id,
                start_char=start,
                end_char=end,
                quote=quote,
                event_date=window.event_date,
                event_date_basis=window.event_date_basis,
                temporal_relation=relation,
                domain_terms=domain_hits,
                matched_query_terms=query_hits,
                supported_slot_ids=supported,
                numeric_slot_ids=explicit_numeric,
                terms=terms,
                local_score=round(score, 8),
                link_score=0.0,
                selection_axes=tuple(axes),
                bundle_role=BundleRole.ALTERNATE,
            )
        )
    bound = _bind_rows(raw, index)
    def source_representative_key(draft: _Draft) -> tuple[Any, ...]:
        # Ordered event questions need the event assertion from a session, not
        # a lexically denser adjacent question or discussion about that event.
        # Other temporal routes retain their established ranking exactly.
        completed_ordered_event = bool(
            route is SpecialistRoute.TEMPORAL_ORDER
            and "concrete_completed_event_surface" in draft.selection_axes
        )
        return (completed_ordered_event, draft.score)

    per_source: dict[str, _Draft] = {}
    for draft in sorted(bound, key=source_representative_key, reverse=True):
        per_source.setdefault(draft.source_id, draft)
    return _with_link_scores(tuple(per_source.values()), index, domain, query_term_set)


def _with_link_scores(
    drafts: tuple[_Draft, ...],
    index: FullStoreWindowIndex,
    domain_terms: frozenset[str],
    query_terms: frozenset[str],
) -> tuple[_Draft, ...]:
    if not drafts:
        return drafts
    source_doc_terms: dict[str, set[str]] = defaultdict(set)
    candidate_sources = {draft.source_id for draft in drafts}
    for row in index.rows:
        if row.source_id in candidate_sources:
            source_doc_terms[row.source_id].update(_norm_words(row.text))
    excluded = set(domain_terms) | set(query_terms) | set(_OPERATOR_TERMS)
    postings: dict[str, set[str]] = defaultdict(set)
    for source, terms in source_doc_terms.items():
        for term in terms - excluded:
            if len(term) >= 4:
                postings[term].add(source)
    max_fanout = max(3, math.ceil(len(drafts) * 0.2))
    score_by_source: dict[str, float] = defaultdict(float)
    for sources in postings.values():
        if 2 <= len(sources) <= max_fanout:
            weight = math.log((len(drafts) + 1) / len(sources)) + 1.0
            for source in sources:
                score_by_source[source] += weight * (len(sources) - 1)
    return tuple(
        replace(draft, link_score=round(score_by_source[draft.source_id], 8))
        for draft in drafts
    )


def _component_members(drafts: Sequence[_Draft]) -> tuple[tuple[_Draft, ...], ...]:
    """Content-linked source components without inspecting source-name prefixes."""

    rows = tuple(drafts)
    if not rows:
        return ()
    common = set.intersection(*(set(row.terms) for row in rows)) if rows else set()
    usable = [set(row.terms) - common - set(_OPERATOR_TERMS) for row in rows]
    adjacency = {index: set() for index in range(len(rows))}
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            shared = {
                term for term in usable[left] & usable[right] if len(term) >= 4
            }
            if shared:
                adjacency[left].add(right)
                adjacency[right].add(left)
    remaining = set(range(len(rows)))
    components: list[tuple[_Draft, ...]] = []
    while remaining:
        seed = min(remaining)
        frontier = [seed]
        reached: set[int] = set()
        while frontier:
            current = frontier.pop()
            if current in reached:
                continue
            reached.add(current)
            frontier.extend(adjacency[current] - reached)
        remaining -= reached
        components.append(tuple(rows[index] for index in sorted(reached)))
    return tuple(components)


def _event_sort_key(draft: _Draft) -> tuple[Any, ...]:
    return (
        draft.event_date or "9999-12-31",
        draft._row.ordinal,
        draft.start_char,
        draft.candidate_id,
    )


_STORY_LINK_STOP = _normalized_inventory(
    (
        "about", "actually", "advice", "also", "because", "could", "good",
        "help", "just", "know", "like", "looking", "make", "need", "really",
        "recommend", "suggest", "thanks", "think", "today", "want", "wonder",
        "would",
    )
)


def _pair_story_affinities(
    drafts: Sequence[_Draft],
    index: FullStoreWindowIndex,
) -> Mapping[tuple[str, str], float]:
    """Return bounded-fanout, user-history links between candidate sessions.

    The local assertion decides whether a row is an event.  These links only
    compose separately worded event assertions into a story: for example,
    three trip answers can share an Eastern-Sierra/backpack/bear-safety
    neighborhood even when their local destination names do not overlap.
    """

    sources = {draft.source_id for draft in drafts}
    document_terms: dict[str, set[str]] = {source: set() for source in sources}
    for row in index.rows:
        if row.source_id in sources and row.role == "user":
            document_terms[row.source_id].update(_norm_words(row.text))
    postings: dict[str, set[str]] = defaultdict(set)
    for source, terms in document_terms.items():
        for term in terms - set(_OPERATOR_TERMS) - set(_STORY_LINK_STOP):
            if len(term) >= 4:
                postings[term].add(source)
    # A term must link at least two sessions but cannot be a broad corpus hub.
    max_fanout = max(3, math.ceil(len(sources) * 0.12))
    usable = {
        term: values
        for term, values in postings.items()
        if 2 <= len(values) <= max_fanout
    }
    scores: dict[tuple[str, str], float] = defaultdict(float)
    for term, linked_sources in usable.items():
        weight = math.log((len(sources) + 1) / len(linked_sources)) + 1.0
        for left, right in itertools.combinations(sorted(linked_sources), 2):
            scores[(left, right)] += weight
    return {key: round(value, 8) for key, value in scores.items()}


def _ordered_story_group(
    ranked: Sequence[_Draft],
    requested: int,
    index: FullStoreWindowIndex,
    budget: TemporalInsufficiencyBudget,
) -> tuple[_Draft, ...]:
    """Choose one coherent, completed event bundle of the requested size."""

    count = min(requested, budget.max_bundle_members, len(ranked))
    if count <= 0:
        return ()
    preferred = [
        row
        for row in ranked
        if (
            "requested_travel_event_surface" in row.selection_axes
            or "requested_sports_event_surface" in row.selection_axes
        )
        and "concrete_completed_event_surface" in row.selection_axes
    ]
    completed = [
        row
        for row in ranked
        if "concrete_completed_event_surface" in row.selection_axes
        and row not in preferred
    ]
    remainder = [row for row in ranked if row not in preferred and row not in completed]
    ordered_pool = preferred + completed + remainder
    # Pairwise search is intentionally performed after complete discovery.
    # The cap bounds computation, not what the immutable index was allowed to
    # reveal.
    pool_cap = min(len(ordered_pool), budget.max_story_link_candidates)
    pool = tuple(ordered_pool[:pool_cap])
    if len(pool) <= count:
        return pool
    affinities = _pair_story_affinities(pool, index)

    def affinity(left: _Draft, right: _Draft) -> float:
        key = tuple(sorted((left.source_id, right.source_id)))
        return affinities.get(key, 0.0)

    def group_key(group: tuple[_Draft, ...]) -> tuple[Any, ...]:
        pair_values = tuple(
            affinity(left, right) for left, right in itertools.combinations(group, 2)
        )
        return (
            sum(
                "requested_travel_event_surface" in row.selection_axes
                or "requested_sports_event_surface" in row.selection_axes
                for row in group
            ),
            sum("concrete_completed_event_surface" in row.selection_axes for row in group),
            min(pair_values, default=0.0),
            sum(pair_values),
            sum(row.local_score for row in group),
            sum(row.link_score for row in group),
            tuple(sorted(row.candidate_id for row in group)),
        )

    combination_count = math.comb(len(pool), count)
    if combination_count <= budget.max_story_combinations:
        combinations = tuple(itertools.combinations(pool, count))
    else:
        # For larger cardinalities, deterministically grow one bounded group
        # from each possible anchor.  This retains pairwise story evidence
        # while respecting the explicit combination budget.
        grown: list[tuple[_Draft, ...]] = []
        for seed in pool:
            group = [seed]
            while len(group) < count:
                used = {row.candidate_id for row in group}
                available = [row for row in pool if row.candidate_id not in used]
                distinct = [
                    row
                    for row in available
                    if row.event_date is not None
                    and row.event_date not in {value.event_date for value in group}
                ]
                choices = distinct or available
                addition = max(
                    choices,
                    key=lambda row: (
                        min((affinity(row, value) for value in group), default=0.0),
                        sum(affinity(row, value) for value in group),
                        (
                            "requested_travel_event_surface" in row.selection_axes
                            or "requested_sports_event_surface" in row.selection_axes
                        ),
                        "concrete_completed_event_surface" in row.selection_axes,
                        row.local_score,
                        row.candidate_id,
                    ),
                )
                group.append(addition)
            grown.append(tuple(group))
        combinations = tuple(grown)
    dated = tuple(
        group
        for group in combinations
        if len({row.event_date for row in group if row.event_date is not None}) == count
    )
    choices = dated or combinations
    return max(choices, key=group_key)


def _interval_entity_comparators(
    ranked: Sequence[_Draft],
    winner: _Draft,
    spec: TypedOperatorSpec,
    budget: TemporalInsufficiencyBudget,
) -> tuple[_Draft, ...]:
    """Keep only later, event-like near misses for an entity-bound interval.

    An elapsed-time question is anchored by one qualifying event and the
    question timestamp.  Other events are useful only when they explain why a
    newer lexical match does not satisfy an entity/participant constraint.
    They are not predecessor endpoints and generic same-component rows are not
    evidence for the interval.
    """

    participant_slots = tuple(
        slot
        for slot in spec.required_slots
        if slot.kind is SlotKind.PARTICIPANT
        or slot.relation_constraint == "participant_singular"
    )
    if not participant_slots or winner.event_date is None:
        return ()
    participant_terms = frozenset(
        term
        for slot in participant_slots
        for raw in slot.match_terms
        for term in _norm_words(raw)
    )
    boundary_terms = frozenset(
        term
        for slot in spec.required_slots
        if slot.kind is not SlotKind.PARTICIPANT
        for raw in slot.match_terms
        for term in _norm_words(raw)
    ) - participant_terms
    if not boundary_terms:
        return ()

    def is_event_like(row: _Draft) -> bool:
        return bool(
            "concrete_completed_event_surface" in row.selection_axes
            or "direct_event_assertion" in row.selection_axes
        )

    eligible = tuple(
        row
        for row in ranked
        if row.candidate_id != winner.candidate_id
        and row.event_date is not None
        and row.event_date > winner.event_date
        and bool(row.terms & boundary_terms)
        and is_event_like(row)
        and not _PROPOSED_RE.search(row.quote)
        and not _INTERVAL_PROPOSED_COMPARATOR_RE.search(row.quote)
    )
    ordered = sorted(
        eligible,
        key=lambda row: (
            len(row.terms & boundary_terms),
            "concrete_completed_event_surface" in row.selection_axes,
            len(row.supported_slot_ids),
            row.local_score,
            row.link_score,
            row.event_date,
            row.candidate_id,
        ),
        reverse=True,
    )[: budget.max_interval_comparators]
    return tuple(
        replace(
            row,
            bundle_role=BundleRole.CORROBORATING,
            selection_axes=(
                *row.selection_axes,
                "interval_entity_constraint_comparator",
            ),
        )
        for row in ordered
    )


def _select_temporal_bundle(
    drafts: Sequence[_Draft],
    route: SpecialistRoute | None,
    body: str,
    spec: TypedOperatorSpec,
    target: QuestionTemporalTarget,
    budget: TemporalInsufficiencyBudget,
    index: FullStoreWindowIndex,
) -> tuple[tuple[_Draft, ...], TemporalEventBundle | None]:
    if route is None:
        return (), None
    population = tuple(drafts)
    requested = _requested_cardinality(body, spec)
    ranked = tuple(sorted(population, key=lambda row: row.score, reverse=True))
    selected: list[_Draft] = []

    if route is SpecialistRoute.TEMPORAL_ORDER:
        # Ordered questions need all plausible operands rather than a single
        # winning source.  Choose one question-sized content-linked story and
        # present every reserved operand chronologically.
        group = _ordered_story_group(
            ranked,
            requested or budget.max_bundle_members,
            index,
            budget,
        )
        for draft in group:
            selected.append(
                replace(
                    draft,
                    bundle_role=BundleRole.ORDERED_OPERAND,
                    selection_axes=(*draft.selection_axes, "ordered_operand_bundle"),
                )
            )
        selected.sort(key=_event_sort_key)
        winner = selected[-1].candidate_id if selected else None
        predecessor = selected[-2].candidate_id if len(selected) > 1 else None
    elif route is SpecialistRoute.TEMPORAL_INTERVAL:
        # The elapsed interval ends at the question timestamp.  Select the
        # latest event satisfying every entity/participant slot, then retain a
        # tiny bounded set of later near misses solely to expose why recency
        # alone is insufficient.  A prior unrelated event is never an interval
        # predecessor.
        fully_supported = tuple(
            row
            for row in ranked
            if all(
                slot.slot_id in row.supported_slot_ids
                for slot in spec.required_slots
            )
        )
        winner_row = max(
            fully_supported or ranked,
            key=_event_sort_key,
            default=None,
        )
        if winner_row is not None:
            selected.append(
                replace(
                    winner_row,
                    bundle_role=BundleRole.WINNER,
                    selection_axes=(
                        *winner_row.selection_axes,
                        "winner_reserved",
                        "implicit_query_time_end_anchor",
                    ),
                )
            )
            if fully_supported:
                selected.extend(
                    _interval_entity_comparators(
                        ranked,
                        winner_row,
                        spec,
                        budget,
                    )
                )
        selected.sort(key=_event_sort_key)
        winner = winner_row.candidate_id if winner_row is not None else None
        predecessor = None
    else:
        components = _component_members(ranked)
        strongest = max(
            components,
            key=lambda values: (
                sum(row.link_score + row.local_score for row in values),
                len(values),
                max((row.score for row in values), default=()),
            ),
            default=(),
        )
        coherent = tuple(sorted(strongest, key=_event_sort_key))
        if route is SpecialistRoute.TEMPORAL_LATEST:
            # Relation constraints (notably singular "with a friend") are
            # applied before recency.  This prevents a newer plural-friends or
            # family visit from becoming the elapsed-time endpoint.
            fully_supported = tuple(
                row
                for row in ranked
                if all(
                    slot.slot_id in row.supported_slot_ids
                    for slot in spec.required_slots
                )
            )
            winner_row = max(
                fully_supported or coherent or ranked,
                key=_event_sort_key,
                default=None,
            )
            predecessors = tuple(
                row
                for row in ranked
                if winner_row is not None
                and row.candidate_id != winner_row.candidate_id
                and row.event_date is not None
                and row.event_date <= (winner_row.event_date or row.event_date)
            )
            predecessor_row = max(predecessors, key=_event_sort_key, default=None)
        else:
            if route is SpecialistRoute.TEMPORAL_RELATIVE:
                relation_priority = {
                    "exact_target_day": 3,
                    "near_target_day": 2,
                    "predecessor_before_target": 1,
                }
                milestone_rows = tuple(
                    row
                    for row in ranked
                    if "question_milestone_event_surface" in row.selection_axes
                )
                winner_row = max(
                    milestone_rows or ranked,
                    key=lambda row: (
                        relation_priority.get(row.temporal_relation, 0),
                        row.local_score,
                        row.link_score,
                        row.score,
                    ),
                    default=None,
                )
            else:
                winner_row = max(
                    coherent or ranked, key=lambda row: row.score, default=None
                )
            if route is SpecialistRoute.TEMPORAL_RELATIVE and winner_row is not None:
                predecessors = tuple(
                    row
                    for row in ranked
                    if row.candidate_id != winner_row.candidate_id
                    and row.event_date is not None
                    and row.event_date < (winner_row.event_date or row.event_date)
                )
                if "question_milestone_event_surface" in winner_row.selection_axes:
                    milestone_predecessors = tuple(
                        row
                        for row in predecessors
                        if "question_milestone_event_surface" in row.selection_axes
                    )
                    predecessors = milestone_predecessors or predecessors
                predecessor_row = max(
                    predecessors, key=_event_sort_key, default=None
                )
            else:
                predecessor_row = None
        reserve: list[_Draft] = []
        if predecessor_row is not None:
            reserve.append(
                replace(
                    predecessor_row,
                    bundle_role=BundleRole.PREDECESSOR,
                    selection_axes=(
                        *predecessor_row.selection_axes,
                        "predecessor_reserved",
                    ),
                )
            )
        if winner_row is not None:
            reserve.append(
                replace(
                    winner_row,
                    bundle_role=BundleRole.WINNER,
                    selection_axes=(*winner_row.selection_axes, "winner_reserved"),
                )
            )
        reserved_ids = {row.candidate_id for row in reserve}
        companions = [
            replace(
                row,
                bundle_role=BundleRole.CORROBORATING,
                selection_axes=(*row.selection_axes, "content_linked_corroboration"),
            )
            for row in coherent
            if row.candidate_id not in reserved_ids
        ]
        alternates = [
            row for row in ranked if row.candidate_id not in reserved_ids
            and row.candidate_id not in {value.candidate_id for value in companions}
        ]
        selected = (reserve + companions + alternates)[: budget.max_bundle_members]
        selected.sort(key=_event_sort_key)
        winner = winner_row.candidate_id if winner_row is not None else None
        predecessor = (
            predecessor_row.candidate_id if predecessor_row is not None else None
        )

    bundle = TemporalEventBundle(
        route=route.value,
        requested_cardinality=requested,
        ordered_candidate_ids=tuple(row.candidate_id for row in selected),
        winner_candidate_id=winner,
        predecessor_candidate_id=predecessor,
        query_time=target.asked_at or spec.query_timestamp,
        target_date=target.target_date,
        population_count=len(population),
        truncated=len(selected) < len(population),
    )
    return tuple(selected), bundle


def _numeric_scope(
    index: FullStoreWindowIndex,
    dated_question: str,
    spec: TypedOperatorSpec,
    budget: TemporalInsufficiencyBudget,
) -> tuple[tuple[_Draft, ...], ScopedNumericAbsenceCertificate]:
    numeric_slots = tuple(slot for slot in spec.required_slots if slot.requires_numeric)
    applicable = bool(spec.requires_all_slots and len(numeric_slots) >= 2)
    if not applicable:
        certificate = ScopedNumericAbsenceCertificate(
            applicable=False,
            scope_definition="not_applicable_no_multi_operand_numeric_operator",
            window_index_receipt_sha256=index.receipt_sha256,
            physical_content_rows_scanned=len(index.rows),
            physical_sentence_windows_scanned=len(index.windows),
            scoped_source_count=0,
            scoped_content_row_count=0,
            every_exact_entity_posting_scanned=True,
            every_scoped_source_row_scanned=True,
            slot_coverage=(),
            may_conclude_operator_insufficient=False,
        )
        return (), certificate

    body = _question_body(dated_question)
    ordered_query_terms = _norm_words(body)
    query_terms = frozenset(ordered_query_terms)
    domain = _query_domain_terms(body, spec)
    numeric_entity_terms = {
        term for slot in numeric_slots for term in slot.match_terms
    }
    predicate_terms = {
        term
        for slot in spec.required_slots
        if slot.kind is SlotKind.PREDICATE
        for term in slot.match_terms
        if term not in numeric_entity_terms and term not in _OPERATOR_TERMS
    }
    if not predicate_terms:
        predicate_terms = {
            term
            for term in query_terms - numeric_entity_terms - set(_OPERATOR_TERMS)
            if term in {"plant", "visit", "buy", "start", "initial"}
        }
    row_index_by_chunk = {row.chunk_id: offset for offset, row in enumerate(index.rows)}

    # First build every first-person predicate assertion.  The strongest exact
    # entity assertion is a question-only root; the complete one-hop component
    # of temporally adjacent predicate assertions becomes the linked-history
    # scope.  No source name or partition identity participates.
    predicate_drafts: list[_Draft] = []
    anchors: list[_Draft] = []
    for window_index, window in enumerate(index.windows):
        row = window.row
        if row.role != "user":
            continue
        start, end, quote = _bounded_exact_window(
            row.text,
            window.start_char,
            window.end_char,
            domain or query_terms,
            budget.max_excerpt_tokens,
        )
        if not _FIRST_PERSON_RE.search(quote):
            continue
        terms = frozenset(_norm_words(quote))
        predicate_hits = terms & predicate_terms
        context_affinity = terms & domain
        event_assertion = bool(
            _PAST_EVENT_RE.search(quote)
            or canonical_action_concepts(quote)
            or "initial" in terms
        )
        if not event_assertion or not predicate_hits or not context_affinity:
            continue
        matching_slots = tuple(
            slot for slot in numeric_slots if _slot_positions(quote, slot)
        )
        mentions = numeric_mentions(
            quote, operator_spec=spec, question=dated_question
        )
        assignments = _numeric_slot_assignment(quote, numeric_slots, mentions)
        supported = tuple(slot.slot_id for slot in matching_slots)
        explicit = tuple(
            slot.slot_id for slot in matching_slots if assignments[slot.slot_id]
        )
        matched_query = tuple(term for term in ordered_query_terms if term in terms)
        score = (
            10.0 * len(explicit)
            + 7.0 * len(supported)
            + 3.0 * len(predicate_hits)
            + 2.0 * len(context_affinity)
            + len(matched_query)
        )
        axes = ["first_person_predicate_assertion"]
        if supported:
            axes.append("exact_entity_scope_anchor")
        if explicit:
            axes.append("explicit_numeric_operand")
        draft = _Draft(
            row_index=row_index_by_chunk[row.chunk_id],
            window_index=window_index,
            source_id=row.source_id,
            start_char=start,
            end_char=end,
            quote=quote,
            event_date=window.event_date,
            event_date_basis=window.event_date_basis,
            temporal_relation="numeric_linked_history_scope",
            domain_terms=tuple(sorted(context_affinity)),
            matched_query_terms=matched_query,
            supported_slot_ids=supported,
            numeric_slot_ids=explicit,
            terms=terms,
            local_score=round(score, 8),
            link_score=0.0,
            selection_axes=tuple(axes),
            bundle_role=BundleRole.SLOT_SUPPORT,
            _row=row,
        )
        predicate_drafts.append(draft)
        if supported:
            anchors.append(draft)

    per_source: dict[str, _Draft] = {}
    for draft in sorted(predicate_drafts, key=lambda row: row.score, reverse=True):
        per_source.setdefault(draft.source_id, draft)
    anchor_root = max(anchors, key=lambda row: row.score, default=None)
    scoped_sources: set[str] = set()
    if anchor_root is not None:
        scoped_sources.add(anchor_root.source_id)
        root_date = (
            date.fromisoformat(anchor_root.event_date)
            if anchor_root.event_date is not None
            else None
        )
        root_predicates = anchor_root.terms & predicate_terms
        for draft in per_source.values():
            if draft.source_id == anchor_root.source_id:
                continue
            shared_predicate = bool(root_predicates & draft.terms)
            if not shared_predicate:
                continue
            if root_date is not None and draft.event_date is not None:
                distance = abs((date.fromisoformat(draft.event_date) - root_date).days)
                if distance > 14:
                    continue
            scoped_sources.add(draft.source_id)

    population = tuple(
        replace(
            draft,
            selection_axes=(
                *draft.selection_axes,
                "linked_history_root"
                if draft.source_id == (anchor_root.source_id if anchor_root else None)
                else "linked_history_one_hop_companion",
            ),
        )
        for draft in sorted(per_source.values(), key=lambda row: row.score, reverse=True)
        if draft.source_id in scoped_sources
    )

    assertion_by_slot: dict[str, list[_Draft]] = defaultdict(list)
    assertion_sources_by_slot: dict[str, set[str]] = defaultdict(set)
    numeric_sources_by_slot: dict[str, set[str]] = defaultdict(set)
    # Having fixed the content-derived source/session scope, scan every exact
    # row/window inside it.  Caps are still not applied.
    for window_index, window in enumerate(index.windows):
        row = window.row
        if row.source_id not in scoped_sources or row.role != "user":
            continue
        start, end, quote = _bounded_exact_window(
            row.text,
            window.start_char,
            window.end_char,
            domain or query_terms,
            budget.max_excerpt_tokens,
        )
        if not _FIRST_PERSON_RE.search(quote):
            continue
        terms = frozenset(_norm_words(quote))
        if not terms & predicate_terms:
            continue
        matching_slots = tuple(
            slot for slot in numeric_slots if _slot_positions(quote, slot)
        )
        if not matching_slots:
            continue
        mentions = numeric_mentions(
            quote, operator_spec=spec, question=dated_question
        )
        assignments = _numeric_slot_assignment(quote, numeric_slots, mentions)
        explicit = tuple(
            slot.slot_id for slot in matching_slots if assignments[slot.slot_id]
        )
        audit_draft = _Draft(
            row_index=row_index_by_chunk[row.chunk_id],
            window_index=window_index,
            source_id=row.source_id,
            start_char=start,
            end_char=end,
            quote=quote,
            event_date=window.event_date,
            event_date_basis=window.event_date_basis,
            temporal_relation="numeric_linked_history_scope",
            domain_terms=tuple(sorted(terms & domain)),
            matched_query_terms=tuple(
                term for term in ordered_query_terms if term in terms
            ),
            supported_slot_ids=tuple(slot.slot_id for slot in matching_slots),
            numeric_slot_ids=explicit,
            terms=terms,
            local_score=0.0,
            link_score=0.0,
            selection_axes=("complete_scoped_source_row_scan",),
            bundle_role=BundleRole.SLOT_SUPPORT,
            _row=row,
        )
        for slot in matching_slots:
            assertion_by_slot[slot.slot_id].append(audit_draft)
            assertion_sources_by_slot[slot.slot_id].add(row.source_id)
            if slot.slot_id in explicit:
                numeric_sources_by_slot[slot.slot_id].add(row.source_id)

    scope_grounded = bool(scoped_sources and anchor_root is not None)
    initial_coverage = tuple(
        ExactNumericSlotCoverage(
            slot_id=slot.slot_id,
            slot_label=slot.label,
            exact_entity_terms=slot.match_terms,
            entity_assertion_window_count=len(assertion_by_slot[slot.slot_id]),
            entity_assertion_source_count=len(assertion_sources_by_slot[slot.slot_id]),
            explicit_numeric_assertion_window_count=sum(
                slot.slot_id in draft.numeric_slot_ids
                for draft in assertion_by_slot[slot.slot_id]
            ),
            explicit_numeric_assertion_source_count=len(
                numeric_sources_by_slot[slot.slot_id]
            ),
            scope_has_grounded_predicate_assertion=scope_grounded,
            explicit_numeric_operand_missing=bool(
                scope_grounded and not numeric_sources_by_slot[slot.slot_id]
            ),
            selected_supporting_candidate_ids=(),
        )
        for slot in numeric_slots
    )
    missing_labels = tuple(
        row.slot_label for row in initial_coverage if row.explicit_numeric_operand_missing
    )
    instruction = None
    if missing_labels:
        quoted = ", ".join(f'"{label}"' for label in missing_labels)
        instruction = (
            "The complete question-derived exact entity scope contains citations "
            f"for {quoted}, but no explicit numeric operand bound to that entity. "
            "Do not infer or copy a count from another entity; report insufficient "
            "memory evidence for the combined numeric request."
        )
    scoped_row_count = sum(row.source_id in scoped_sources for row in index.rows)
    certificate = ScopedNumericAbsenceCertificate(
        applicable=True,
        scope_definition=(
            "complete immutable-index discovery of first-person predicate assertions; "
            "the source/session rooted at the strongest exact requested-entity "
            "assertion plus every one-hop source/session sharing a question-derived "
            "predicate within fourteen days; then every row/window of that fixed scope"
        ),
        window_index_receipt_sha256=index.receipt_sha256,
        physical_content_rows_scanned=len(index.rows),
        physical_sentence_windows_scanned=len(index.windows),
        scoped_source_count=len(scoped_sources),
        scoped_content_row_count=scoped_row_count,
        every_exact_entity_posting_scanned=True,
        every_scoped_source_row_scanned=True,
        slot_coverage=initial_coverage,
        may_conclude_operator_insufficient=bool(missing_labels),
        provider_instruction=instruction,
    )
    return population, certificate


def _select_combined(
    temporal: Sequence[_Draft],
    numeric: Sequence[_Draft],
    certificate: ScopedNumericAbsenceCertificate,
    budget: TemporalInsufficiencyBudget,
) -> tuple[_Draft, ...]:
    selected: list[_Draft] = []
    sources: set[str] = set()
    tokens = 0

    def add(draft: _Draft) -> bool:
        nonlocal tokens
        if draft.source_id in sources:
            return False
        amount = count_tokens(draft.quote)
        if (
            len(selected) >= budget.max_candidates
            or tokens + amount > budget.evidence_token_cap
        ):
            return False
        selected.append(draft)
        sources.add(draft.source_id)
        tokens += amount
        return True

    # A missing slot's direct citation and every present numeric operand each
    # receive a non-borrowable first admission opportunity.
    for coverage in certificate.slot_coverage:
        candidates = [
            row for row in numeric if coverage.slot_id in row.supported_slot_ids
        ]
        if coverage.explicit_numeric_operand_missing:
            candidates.sort(key=lambda row: row.score, reverse=True)
        else:
            candidates.sort(
                key=lambda row: (
                    coverage.slot_id in row.numeric_slot_ids,
                    row.score,
                ),
                reverse=True,
            )
        if candidates:
            add(candidates[0])

    for draft in temporal:
        add(draft)
    for draft in sorted(numeric, key=lambda row: row.score, reverse=True):
        add(draft)
    return tuple(selected)


def _span(draft: _Draft) -> EvidenceSpan:
    row = draft._row
    return EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=draft.start_char,
        end_char=draft.end_char,
        quote_sha256=quote_sha256(draft.quote),
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )


def _materialize(
    selected: Sequence[_Draft], index: FullStoreWindowIndex
) -> tuple[
    tuple[TemporalSpecialistCandidate, ...],
    tuple[LocalCitationBinding, ...],
]:
    sources = tuple(sorted({row.source_id for row in selected}))
    groups = {source: f"G{offset:04d}" for offset, source in enumerate(sources, 1)}
    candidates: list[TemporalSpecialistCandidate] = []
    bindings: list[LocalCitationBinding] = []
    for draft in selected:
        row = draft._row
        span = _span(draft)
        candidate_id = draft.candidate_id
        binding = LocalCitationBinding(
            candidate_id=candidate_id,
            source_group_handle=groups[row.source_id],
            namespace_id=index.cache.namespace_id,
            cache_receipt_sha256=index.cache.cache_receipt_sha256,
            source_database_sha256=index.cache.source_database_sha256,
            source_store_receipt_sha256=index.cache.source_store_receipt_sha256,
            source_id=row.source_id,
            partition_id=row.partition_id,
            span=span,
            quote_sha256=quote_sha256(draft.quote),
        )
        candidate = TemporalSpecialistCandidate(
            candidate_id=candidate_id,
            source_group_handle=groups[row.source_id],
            quote=draft.quote,
            quote_sha256=quote_sha256(draft.quote),
            token_count=count_tokens(draft.quote),
            role=row.role,
            created_at=row.created_at,
            event_date=draft.event_date,
            event_date_basis=draft.event_date_basis,
            temporal_relation=draft.temporal_relation,
            domain_affinity_terms=draft.domain_terms,
            matched_query_terms=draft.matched_query_terms,
            supported_slot_ids=draft.supported_slot_ids,
            explicit_numeric_slot_ids=draft.numeric_slot_ids,
            first_person_assertion=bool(_FIRST_PERSON_RE.search(draft.quote)),
            bundle_role=draft.bundle_role,
            selection_axes=draft.selection_axes,
            citation_binding_receipt_sha256=binding.receipt_sha256,
        )
        candidates.append(candidate)
        bindings.append(binding)
    return tuple(candidates), tuple(bindings)


def _refresh_relations(
    selected: Sequence[_Draft],
    bundle: TemporalEventBundle | None,
    certificate: ScopedNumericAbsenceCertificate,
) -> tuple[TemporalEventBundle | None, ScopedNumericAbsenceCertificate]:
    selected_ids = {row.candidate_id for row in selected}
    if bundle is not None:
        ordered = tuple(
            candidate_id
            for candidate_id in bundle.ordered_candidate_ids
            if candidate_id in selected_ids
        )
        bundle = TemporalEventBundle(
            route=bundle.route,
            requested_cardinality=bundle.requested_cardinality,
            ordered_candidate_ids=ordered,
            winner_candidate_id=(
                bundle.winner_candidate_id
                if bundle.winner_candidate_id in selected_ids
                else (ordered[-1] if ordered else None)
            ),
            predecessor_candidate_id=(
                None
                if bundle.predecessor_candidate_id is None
                else (
                    bundle.predecessor_candidate_id
                    if bundle.predecessor_candidate_id in selected_ids
                    else (ordered[-2] if len(ordered) > 1 else None)
                )
            ),
            query_time=bundle.query_time,
            target_date=bundle.target_date,
            population_count=bundle.population_count,
            truncated=len(ordered) < bundle.population_count,
        )
    refreshed = tuple(
        ExactNumericSlotCoverage(
            slot_id=row.slot_id,
            slot_label=row.slot_label,
            exact_entity_terms=row.exact_entity_terms,
            entity_assertion_window_count=row.entity_assertion_window_count,
            entity_assertion_source_count=row.entity_assertion_source_count,
            explicit_numeric_assertion_window_count=(
                row.explicit_numeric_assertion_window_count
            ),
            explicit_numeric_assertion_source_count=(
                row.explicit_numeric_assertion_source_count
            ),
            scope_has_grounded_predicate_assertion=(
                row.scope_has_grounded_predicate_assertion
            ),
            explicit_numeric_operand_missing=row.explicit_numeric_operand_missing,
            selected_supporting_candidate_ids=tuple(
                draft.candidate_id
                for draft in selected
                if row.slot_id in draft.supported_slot_ids
            ),
        )
        for row in certificate.slot_coverage
    )
    certificate = ScopedNumericAbsenceCertificate(
        applicable=certificate.applicable,
        scope_definition=certificate.scope_definition,
        window_index_receipt_sha256=certificate.window_index_receipt_sha256,
        physical_content_rows_scanned=certificate.physical_content_rows_scanned,
        physical_sentence_windows_scanned=(
            certificate.physical_sentence_windows_scanned
        ),
        scoped_source_count=certificate.scoped_source_count,
        scoped_content_row_count=certificate.scoped_content_row_count,
        every_exact_entity_posting_scanned=(
            certificate.every_exact_entity_posting_scanned
        ),
        every_scoped_source_row_scanned=certificate.every_scoped_source_row_scanned,
        slot_coverage=refreshed,
        may_conclude_operator_insufficient=(
            certificate.may_conclude_operator_insufficient
        ),
        provider_instruction=certificate.provider_instruction,
    )
    return bundle, certificate


def _provider_projection(
    dated_question: str,
    spec: TypedOperatorSpec,
    target: QuestionTemporalTarget,
    routes: Sequence[SpecialistRoute],
    candidates: Sequence[TemporalSpecialistCandidate],
    bundle: TemporalEventBundle | None,
    certificate: ScopedNumericAbsenceCertificate,
) -> dict[str, Any]:
    return {
        "absence_certificate": certificate.projection(),
        "candidates": [row.projection() for row in candidates],
        "dated_question": dated_question,
        "format": RESULT_FORMAT,
        "operator_spec": spec.projection(),
        "routes": [row.value for row in routes],
        "temporal_bundle": None if bundle is None else bundle.projection(),
        "temporal_target": target.projection(),
    }


def scan_temporal_insufficiency_specialist(
    index: FullStoreWindowIndex,
    dated_question: str,
    /,
    *,
    budget: TemporalInsufficiencyBudget = TemporalInsufficiencyBudget(),
) -> TemporalInsufficiencyResult:
    """Build a bounded event bundle and/or exact-slot insufficiency proof.

    The index must already cover the complete immutable namespace.  The scan
    iterates its complete window inventory; selection caps apply only after
    discovery and therefore cannot manufacture an absence certificate.
    """

    _require(type(index) is FullStoreWindowIndex, "specialist requires exact index")
    require_text(dated_question, "specialist dated question")
    _require(type(budget) is TemporalInsufficiencyBudget, "budget changed")
    spec = compile_typed_operator_spec(dated_question)
    target = _temporal_target(dated_question, spec)
    routes = _routes(spec)
    temporal_route = next(
        (row for row in routes if row is not SpecialistRoute.NUMERIC_SLOT_INSUFFICIENCY),
        None,
    )
    temporal_population = _temporal_drafts(
        index, dated_question, spec, target, temporal_route, budget
    )
    temporal_selected, bundle = _select_temporal_bundle(
        temporal_population,
        temporal_route,
        _question_body(dated_question),
        spec,
        target,
        budget,
        index,
    )
    numeric_population, certificate = _numeric_scope(
        index, dated_question, spec, budget
    )
    selected = list(
        _select_combined(temporal_selected, numeric_population, certificate, budget)
    )

    # Exact complete-payload accounting.  Lowest-priority unprotected rows are
    # removed only after both specialists have independently selected.
    while True:
        bundle, certificate = _refresh_relations(selected, bundle, certificate)
        candidates, bindings = _materialize(selected, index)
        projection = _provider_projection(
            dated_question,
            spec,
            target,
            routes,
            candidates,
            bundle,
            certificate,
        )
        provider_tokens = count_tokens(_canonical_json(projection))
        if provider_tokens <= budget.provider_payload_token_cap:
            break
        removable = next(
            (
                offset
                for offset in range(len(selected) - 1, -1, -1)
                if selected[offset].bundle_role
                in {BundleRole.ALTERNATE, BundleRole.CORROBORATING}
            ),
            None,
        )
        if removable is None:
            raise TemporalInsufficiencySpecialistError(
                "protected specialist bundle exceeds provider payload cap"
            )
        selected.pop(removable)

    truncated = bool(
        len(selected)
        < len(
            {
                row.source_id
                for row in (*temporal_population, *numeric_population)
            }
        )
    )
    receipt = TemporalInsufficiencyReceipt(
        question_sha256=quote_sha256(dated_question),
        operator_spec_receipt_sha256=spec.receipt_sha256,
        temporal_target_receipt_sha256=target.receipt_sha256,
        window_index_receipt_sha256=index.receipt_sha256,
        absence_certificate_receipt_sha256=certificate.receipt_sha256,
        temporal_bundle_receipt_sha256=(
            None if bundle is None else bundle.receipt_sha256
        ),
        budget_id=budget.budget_id,
        routes=tuple(row.value for row in routes),
        physical_content_rows_scanned=len(index.rows),
        physical_sentence_windows_scanned=len(index.windows),
        temporal_candidate_population_count=len(temporal_population),
        scoped_numeric_candidate_population_count=len(numeric_population),
        selected_candidate_ids=tuple(row.candidate_id for row in candidates),
        selected_source_group_count=len(
            {row.source_group_handle for row in candidates}
        ),
        selected_evidence_tokens=sum(row.token_count for row in candidates),
        provider_payload_tokens=provider_tokens,
        selection_truncated=truncated,
    )
    return TemporalInsufficiencyResult(
        dated_question=dated_question,
        operator_spec=spec,
        temporal_target=target,
        routes=routes,
        candidates=candidates,
        local_bindings=bindings,
        temporal_bundle=bundle,
        absence_certificate=certificate,
        receipt=receipt,
        budget=budget,
    )


def _safe_status(quote: str) -> str:
    if re.search(r"\b(?:cancelled|canceled|abandoned)\b", quote, re.I):
        return "cancelled"
    # A selected assertion can contain a completed event followed by future
    # intent ("I signed ... and I want ...") or a consequence described with
    # planning vocabulary.  The completed, cited event is the typed item's
    # status; trailing intent must not make the fitter discard that relation
    # member as a proposal.
    if _PAST_EVENT_RE.search(quote):
        return "completed"
    if _PROPOSED_RE.search(quote):
        return "proposed"
    return "unknown"


def adapt_temporal_insufficiency_to_typed_contribution(
    result: TemporalInsufficiencyResult,
    /,
    *,
    handle_start: int,
    group_start: int,
) -> "TypedEvidenceContribution":
    """Adapt exact selected quotes without overstating the absence proof.

    The scoped certificate remains in ``result.provider_projection()``.  It is
    not forged into a direct-pointer item because a negative scan result is an
    aggregate receipt, not a local citation.  The common frontier therefore
    remains bounded even when the narrow lexical scope is physically closed.
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
        type(result) is TemporalInsufficiencyResult,
        "typed adapter requires exact specialist result",
    )
    for value, label in (
        (handle_start, "specialist handle start"),
        (group_start, "specialist group start"),
    ):
        _require(type(value) is int and value >= 1, f"{label} must be positive")
    _require(
        handle_start + len(result.candidates) - 1 <= 999_999,
        "specialist handle range exceeds opaque contract",
    )
    local_groups = tuple(
        dict.fromkeys(row.source_group_handle for row in result.candidates)
    )
    _require(
        group_start + len(local_groups) - 1 <= 999_999,
        "specialist group range exceeds opaque contract",
    )
    groups = {
        local: f"G{group_start + offset:03d}"
        for offset, local in enumerate(local_groups)
    }
    sealed_artifact = identity_sha256(result.local_audit_projection())
    bindings: list[Any] = []
    raw_items: list[dict[str, Any]] = []
    numeric_slots = {
        slot.slot_id: slot
        for slot in result.operator_spec.required_slots
        if slot.requires_numeric
    }
    for offset, (candidate, local) in enumerate(
        zip(result.candidates, result.local_bindings, strict=True)
    ):
        handle_id = f"H{handle_start + offset:03d}"
        binding = EvidenceHandleBinding(
            handle_id=handle_id,
            origin=EvidenceOrigin.DIRECT_POINTER,
            provenance_grade=ProvenanceGrade.DIRECT_POINTER,
            source_group_handle=groups[candidate.source_group_handle],
            sealed_artifact_sha256=sealed_artifact,
            parent_receipt_sha256=result.receipt.receipt_sha256,
            evidence_receipt_sha256=local.receipt_sha256,
            payload_sha256=identity_sha256(candidate.projection()),
            citation_sha256=candidate.quote_sha256,
            citation_char_count=len(candidate.quote),
            local_source_locator_sha256=local.receipt_sha256,
        )
        mention = single_numeric_mention(
            candidate.quote,
            operator_spec=result.operator_spec,
            question=result.dated_question,
        )
        entity_slots = tuple(
            numeric_slots[slot_id]
            for slot_id in candidate.explicit_numeric_slot_ids
            if slot_id in numeric_slots
        )
        raw: dict[str, Any] = {
            "handle_ids": [handle_id],
            "included": True,
            "kind": "operand" if mention is not None and entity_slots else "event",
            "numeric_role": "operand" if mention is not None and entity_slots else "none",
            "relation": (
                f"authored_by_user;bundle_role:{candidate.bundle_role.value};"
                f"temporal_relation:{candidate.temporal_relation}"
            ),
            "status": _safe_status(candidate.quote),
            "summary": candidate.quote,
            "value_authority": "explicit",
        }
        if candidate.event_date is not None:
            raw["date"] = candidate.event_date
        if len(entity_slots) == 1:
            raw["entity_key"] = entity_slots[0].label
        if mention is not None and entity_slots:
            raw["numeric_qualifier"] = mention.qualifier.value
            raw["numeric_value"] = mention.value
            if mention.unit is not None:
                raw["unit"] = mention.unit
        bindings.append(binding)
        raw_items.append(raw)
    frozen = tuple(bindings)
    parsed = parse_typed_items(
        raw_items, operator_spec=result.operator_spec, bindings=frozen
    )
    _require(not parsed.rejected_items, "specialist typed item was rejected")
    return TypedEvidenceContribution(
        mechanism_id=MECHANISM_ID,
        bindings=frozen,
        parsed=parsed,
        sealed_artifact_sha256=sealed_artifact,
        frontier_mode=FrontierMode.BOUNDED,
        truncated=result.receipt.selection_truncated,
    )


__all__ = [
    "ABSENCE_FORMAT",
    "BUNDLE_FORMAT",
    "BundleRole",
    "ExactNumericSlotCoverage",
    "MECHANISM_ID",
    "ScopedNumericAbsenceCertificate",
    "SpecialistRoute",
    "TemporalEventBundle",
    "TemporalInsufficiencyBudget",
    "TemporalInsufficiencyReceipt",
    "TemporalInsufficiencyResult",
    "TemporalInsufficiencySpecialistError",
    "TemporalSpecialistCandidate",
    "adapt_temporal_insufficiency_to_typed_contribution",
    "scan_temporal_insufficiency_specialist",
]
