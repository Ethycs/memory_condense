"""Provider-free coverage packing for fact-seeded reconstruction evidence.

Fact-seeded reconstruction deliberately discovers exact evidence before it
knows how much of that evidence will fit in the final prompt.  A plain prefix
cut can therefore spend the whole allowance on several windows that repeat
one fact, cue, source, or turn while omitting another operand or the exact user
turn.  This module is the separate, opt-in post-discovery selector for that
problem.

The selector is gold blind and deterministic.  It first removes exact span
duplicates from the complete discovered population, then greedily maximizes
new coverage in this order: unresolved operator slots; fact and cue lineage;
temporal, action, and personalization obligations; exact role evidence; source
and turn diversity; and finally evidence density per token.  No provider call,
embedding, mutable state, question ID, answer, or target label enters the
decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Any, Literal, Sequence

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import FullStoreSlotCandidate, LocalCitationBinding
from .typed_action_semantics import (
    canonical_action_concepts,
    completed_action_concepts,
    matched_action_concepts,
)
from .typed_active_reconstruction import (
    candidate_projection_receipt_sha256,
    citation_span_receipt_sha256,
)
from .typed_fact_seeded_reconstruction import (
    FactSeededRecoveryLineage,
    TypedFactSeededReconstructionResult,
)
from .typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    TypedEvidenceContribution,
    TypedEvidenceItem,
    conservative_numeric_value,
    parse_typed_items,
)
from .typed_operator_spec import SlotKind, TemporalMode, TypedOperatorSpec, normalized_terms


MECHANISM_ID = "typed_fact_seeded_coverage_pack_v1"
BUDGET_FORMAT = "memory-condense-typed-fact-seeded-coverage-budget-v1"
INVENTORY_FORMAT = "memory-condense-typed-fact-seeded-packing-inventory-v1"
DECISION_FORMAT = "memory-condense-typed-fact-seeded-coverage-decision-v1"
COVERAGE_FORMAT = "memory-condense-typed-fact-seeded-coverage-receipt-v1"
RESULT_FORMAT = "memory-condense-typed-fact-seeded-coverage-pack-v1"


class TypedFactSeededCoveragePackingError(MatchedEvalContractError):
    """Raised when exact candidate lineage, caps, or sealed packing changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedFactSeededCoveragePackingError(message)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


@dataclass(frozen=True, slots=True)
class FactSeededCoverageBudget:
    """Exact final candidate and evidence-text allowances."""

    max_candidates: int = 8
    max_tokens: int = 2_048
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.max_candidates) is int
            and 0 <= self.max_candidates <= 128,
            "fact coverage candidate cap changed",
        )
        _require(
            type(self.max_tokens) is int and 0 <= self.max_tokens <= 65_536,
            "fact coverage token cap changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact coverage budget changed")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def budget_id(self) -> str:
        return self.receipt_sha256

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": BUDGET_FORMAT,
            "max_candidates": self.max_candidates,
            "max_tokens": self.max_tokens,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FactSeededPackingInventory:
    """Canonical post-discovery triples presented to the coverage selector.

    ``build_fact_seeded_packing_inventory`` is the production constructor.
    Keeping this exact inventory contract public also permits small synthetic
    mechanism tests without constructing a scanner or a resident index.
    """

    dated_question: str
    operator_spec: TypedOperatorSpec
    source_result_receipt_sha256: str
    source_result_status: Literal["scanned", "packet_invalid", "no_fact_cues"]
    source_result_truncated: bool
    seed_items: tuple[TypedEvidenceItem, ...]
    candidates: tuple[FullStoreSlotCandidate, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    lineages: tuple[FactSeededRecoveryLineage, ...]
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.dated_question, "fact coverage dated question")
        _require(
            type(self.operator_spec) is TypedOperatorSpec,
            "fact coverage operator changed",
        )
        require_sha256(
            self.source_result_receipt_sha256,
            "fact coverage source result",
        )
        _require(
            self.source_result_status in {
                "scanned",
                "packet_invalid",
                "no_fact_cues",
            },
            "fact coverage source status changed",
        )
        _require(
            type(self.source_result_truncated) is bool,
            "fact coverage source truncation changed",
        )
        _require(
            type(self.seed_items) is tuple
            and all(type(row) is TypedEvidenceItem for row in self.seed_items),
            "fact coverage seed items changed",
        )
        _require(
            type(self.candidates) is tuple
            and all(type(row) is FullStoreSlotCandidate for row in self.candidates)
            and type(self.local_bindings) is tuple
            and all(type(row) is LocalCitationBinding for row in self.local_bindings)
            and type(self.lineages) is tuple
            and all(type(row) is FactSeededRecoveryLineage for row in self.lineages)
            and len(self.candidates)
            == len(self.local_bindings)
            == len(self.lineages),
            "fact coverage discovered triples changed",
        )

        triples = tuple(
            sorted(
                zip(
                    self.candidates,
                    self.local_bindings,
                    self.lineages,
                    strict=True,
                ),
                key=lambda row: (
                    candidate_projection_receipt_sha256(row[0]),
                    row[1].receipt_sha256,
                    row[2].receipt_sha256,
                ),
            )
        )
        if triples:
            object.__setattr__(self, "candidates", tuple(row[0] for row in triples))
            object.__setattr__(
                self, "local_bindings", tuple(row[1] for row in triples)
            )
            object.__setattr__(self, "lineages", tuple(row[2] for row in triples))

        required_slots = {row.slot_id for row in self.operator_spec.required_slots}
        for candidate, binding, lineage in zip(
            self.candidates, self.local_bindings, self.lineages, strict=True
        ):
            candidate_receipt = candidate_projection_receipt_sha256(candidate)
            _require(
                candidate.candidate_id == binding.candidate_id
                and candidate.source_group_handle == binding.source_group_handle
                and candidate.citation_binding_receipt_sha256
                == binding.receipt_sha256
                and candidate_receipt
                == lineage.recovered_candidate_receipt_sha256
                and binding.receipt_sha256
                == lineage.recovered_local_binding_receipt_sha256
                and citation_span_receipt_sha256(binding)
                == lineage.recovered_span_receipt_sha256,
                "fact coverage candidate/local/lineage triple changed",
            )
            _require(
                set(candidate.supported_slot_ids) <= required_slots,
                "fact coverage candidate escaped operator slots",
            )
        _require(
            self.provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0,
            "fact coverage inventory retained provider state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact coverage inventory changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="fact_seeded_packing_inventory")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidate_receipt_sha256s": [
                candidate_projection_receipt_sha256(row) for row in self.candidates
            ],
            "format": INVENTORY_FORMAT,
            "lineage_receipt_sha256s": [row.receipt_sha256 for row in self.lineages],
            "local_binding_receipt_sha256s": [
                row.receipt_sha256 for row in self.local_bindings
            ],
            "new_provider_calls": 0,
            "operator_spec_receipt_sha256": self.operator_spec.receipt_sha256,
            "question_sha256": self.operator_spec.question_sha256,
            "retained_transformer_token_state_bytes": 0,
            "seed_item_receipt_sha256s": [
                row.receipt_sha256 for row in self.seed_items
            ],
            "source_result_receipt_sha256": self.source_result_receipt_sha256,
            "source_result_status": self.source_result_status,
            "source_result_truncated": self.source_result_truncated,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


PackingDecisionStatus = Literal[
    "selected",
    "duplicate_after_discovery",
    "candidate_cap_excluded",
    "token_cap_excluded",
]


@dataclass(frozen=True, slots=True)
class FactSeededCoverageDecision:
    occurrence_receipt_sha256: str
    candidate_receipt_sha256: str
    local_binding_receipt_sha256: str
    lineage_receipt_sha256: str
    status: PackingDecisionStatus
    selection_rank: int | None
    token_count: int
    cumulative_selected_tokens: int
    marginal_slot_ids: tuple[str, ...]
    marginal_fact_receipt_sha256s: tuple[str, ...]
    marginal_cue_receipt_sha256s: tuple[str, ...]
    marginal_support_features: tuple[str, ...]
    marginal_temporal_features: tuple[str, ...]
    marginal_action_features: tuple[str, ...]
    marginal_personalization_features: tuple[str, ...]
    marginal_role_features: tuple[str, ...]
    source_diversity_added: bool
    turn_diversity_added: bool
    density_numerator: int
    support_quality: int
    protocol_only_ultrashort: bool
    reason: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.occurrence_receipt_sha256, "fact coverage occurrence"),
            (self.candidate_receipt_sha256, "fact coverage candidate"),
            (self.local_binding_receipt_sha256, "fact coverage binding"),
            (self.lineage_receipt_sha256, "fact coverage lineage"),
        ):
            require_sha256(value, label)
        _require(
            self.status
            in {
                "selected",
                "duplicate_after_discovery",
                "candidate_cap_excluded",
                "token_cap_excluded",
            },
            "fact coverage decision status changed",
        )
        _require(
            (self.status == "selected")
            == (type(self.selection_rank) is int and (self.selection_rank or 0) >= 1),
            "fact coverage selection rank changed",
        )
        _require(
            type(self.token_count) is int
            and self.token_count >= 0
            and type(self.cumulative_selected_tokens) is int
            and self.cumulative_selected_tokens >= 0
            and type(self.density_numerator) is int
            and self.density_numerator >= 0,
            "fact coverage decision accounting changed",
        )
        _require(
            type(self.support_quality) is int
            and 0 <= self.support_quality <= 4
            and type(self.protocol_only_ultrashort) is bool,
            "fact coverage support quality changed",
        )
        for values, label, sha_values in (
            (self.marginal_slot_ids, "fact coverage marginal slots", True),
            (self.marginal_fact_receipt_sha256s, "fact coverage marginal facts", True),
            (self.marginal_cue_receipt_sha256s, "fact coverage marginal cues", True),
            (self.marginal_support_features, "fact coverage support features", False),
            (self.marginal_temporal_features, "fact coverage temporal features", False),
            (self.marginal_action_features, "fact coverage action features", False),
            (
                self.marginal_personalization_features,
                "fact coverage personalization features",
                False,
            ),
            (self.marginal_role_features, "fact coverage role features", False),
        ):
            _require(
                type(values) is tuple
                and len(values) == len(set(values))
                and tuple(sorted(values)) == values,
                f"{label} changed",
            )
            for value in values:
                if sha_values:
                    require_sha256(value, label)
                else:
                    require_text(value, label)
        _require(
            type(self.source_diversity_added) is bool
            and type(self.turn_diversity_added) is bool,
            "fact coverage diversity flags changed",
        )
        if self.status != "selected":
            _require(
                not self.marginal_slot_ids
                and not self.marginal_fact_receipt_sha256s
                and not self.marginal_cue_receipt_sha256s
                and not self.marginal_support_features
                and not self.marginal_temporal_features
                and not self.marginal_action_features
                and not self.marginal_personalization_features
                and not self.marginal_role_features
                and not self.source_diversity_added
                and not self.turn_diversity_added,
                "excluded fact coverage decision claimed marginal coverage",
            )
        require_text(self.reason, "fact coverage decision reason")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact coverage decision changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidate_receipt_sha256": self.candidate_receipt_sha256,
            "cumulative_selected_tokens": self.cumulative_selected_tokens,
            "density_numerator": self.density_numerator,
            "format": DECISION_FORMAT,
            "lineage_receipt_sha256": self.lineage_receipt_sha256,
            "local_binding_receipt_sha256": self.local_binding_receipt_sha256,
            "marginal_action_features": list(self.marginal_action_features),
            "marginal_cue_receipt_sha256s": list(
                self.marginal_cue_receipt_sha256s
            ),
            "marginal_fact_receipt_sha256s": list(
                self.marginal_fact_receipt_sha256s
            ),
            "marginal_personalization_features": list(
                self.marginal_personalization_features
            ),
            "marginal_role_features": list(self.marginal_role_features),
            "marginal_slot_ids": list(self.marginal_slot_ids),
            "marginal_support_features": list(self.marginal_support_features),
            "marginal_temporal_features": list(self.marginal_temporal_features),
            "occurrence_receipt_sha256": self.occurrence_receipt_sha256,
            "protocol_only_ultrashort": self.protocol_only_ultrashort,
            "reason": self.reason,
            "selection_rank": self.selection_rank,
            "source_diversity_added": self.source_diversity_added,
            "status": self.status,
            "support_quality": self.support_quality,
            "token_count": self.token_count,
            "turn_diversity_added": self.turn_diversity_added,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FactSeededCoverageReceipt:
    population_candidate_count: int
    unique_candidate_count: int
    duplicate_candidate_count: int
    selected_candidate_count: int
    selected_candidate_tokens: int
    unresolved_slot_ids_before: tuple[str, ...]
    unresolved_slot_ids_after: tuple[str, ...]
    selected_fact_receipt_sha256s: tuple[str, ...]
    selected_cue_receipt_sha256s: tuple[str, ...]
    selected_support_features: tuple[str, ...]
    selected_temporal_features: tuple[str, ...]
    selected_action_features: tuple[str, ...]
    selected_personalization_features: tuple[str, ...]
    selected_role_features: tuple[str, ...]
    selected_source_key_sha256s: tuple[str, ...]
    selected_turn_key_sha256s: tuple[str, ...]
    selection_order_occurrence_receipt_sha256s: tuple[str, ...]
    budget_receipt_sha256: str
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.population_candidate_count, "fact coverage population"),
            (self.unique_candidate_count, "fact coverage unique population"),
            (self.duplicate_candidate_count, "fact coverage duplicates"),
            (self.selected_candidate_count, "fact coverage selected count"),
            (self.selected_candidate_tokens, "fact coverage selected tokens"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        _require(
            self.population_candidate_count
            == self.unique_candidate_count + self.duplicate_candidate_count
            and self.selected_candidate_count <= self.unique_candidate_count,
            "fact coverage population accounting changed",
        )
        sha_groups = (
            (self.unresolved_slot_ids_before, "fact coverage unresolved-before"),
            (self.unresolved_slot_ids_after, "fact coverage unresolved-after"),
            (self.selected_fact_receipt_sha256s, "fact coverage selected facts"),
            (self.selected_cue_receipt_sha256s, "fact coverage selected cues"),
            (self.selected_source_key_sha256s, "fact coverage selected sources"),
            (self.selected_turn_key_sha256s, "fact coverage selected turns"),
            (
                self.selection_order_occurrence_receipt_sha256s,
                "fact coverage selection order",
            ),
        )
        for values, label in sha_groups:
            _require(
                type(values) is tuple and len(values) == len(set(values)),
                f"{label} changed",
            )
            for value in values:
                require_sha256(value, label)
        for values, label in (
            (self.selected_support_features, "fact coverage support features"),
            (self.selected_temporal_features, "fact coverage temporal"),
            (self.selected_action_features, "fact coverage actions"),
            (
                self.selected_personalization_features,
                "fact coverage personalization",
            ),
            (self.selected_role_features, "fact coverage roles"),
        ):
            _require(
                type(values) is tuple
                and tuple(sorted(values)) == values
                and len(values) == len(set(values)),
                f"{label} changed",
            )
            for value in values:
                require_text(value, label)
        _require(
            set(self.unresolved_slot_ids_after)
            <= set(self.unresolved_slot_ids_before),
            "fact coverage introduced an unresolved slot",
        )
        _require(
            len(self.selection_order_occurrence_receipt_sha256s)
            == self.selected_candidate_count,
            "fact coverage selection order changed",
        )
        require_sha256(self.budget_receipt_sha256, "fact coverage receipt budget")
        _require(
            self.provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0,
            "fact coverage receipt retained provider state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact coverage receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "budget_receipt_sha256": self.budget_receipt_sha256,
            "duplicate_candidate_count": self.duplicate_candidate_count,
            "format": COVERAGE_FORMAT,
            "new_provider_calls": 0,
            "population_candidate_count": self.population_candidate_count,
            "retained_transformer_token_state_bytes": 0,
            "selected_action_features": list(self.selected_action_features),
            "selected_candidate_count": self.selected_candidate_count,
            "selected_candidate_tokens": self.selected_candidate_tokens,
            "selected_cue_receipt_sha256s": list(
                self.selected_cue_receipt_sha256s
            ),
            "selected_fact_receipt_sha256s": list(
                self.selected_fact_receipt_sha256s
            ),
            "selected_personalization_features": list(
                self.selected_personalization_features
            ),
            "selected_role_features": list(self.selected_role_features),
            "selected_source_key_sha256s": list(self.selected_source_key_sha256s),
            "selected_support_features": list(self.selected_support_features),
            "selected_temporal_features": list(self.selected_temporal_features),
            "selected_turn_key_sha256s": list(self.selected_turn_key_sha256s),
            "selection_order_occurrence_receipt_sha256s": list(
                self.selection_order_occurrence_receipt_sha256s
            ),
            "unique_candidate_count": self.unique_candidate_count,
            "unresolved_slot_ids_after": list(self.unresolved_slot_ids_after),
            "unresolved_slot_ids_before": list(self.unresolved_slot_ids_before),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class TypedFactSeededCoveragePackResult:
    inventory: FactSeededPackingInventory
    budget: FactSeededCoverageBudget
    decisions: tuple[FactSeededCoverageDecision, ...]
    coverage: FactSeededCoverageReceipt
    candidates: tuple[FullStoreSlotCandidate, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    lineages: tuple[FactSeededRecoveryLineage, ...]
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.inventory) is FactSeededPackingInventory
            and type(self.budget) is FactSeededCoverageBudget
            and type(self.coverage) is FactSeededCoverageReceipt,
            "fact coverage result parents changed",
        )
        _require(
            type(self.decisions) is tuple
            and all(type(row) is FactSeededCoverageDecision for row in self.decisions)
            and len(self.decisions) == len(self.inventory.candidates),
            "fact coverage result decisions changed",
        )
        _require(
            type(self.candidates) is tuple
            and all(type(row) is FullStoreSlotCandidate for row in self.candidates)
            and type(self.local_bindings) is tuple
            and all(type(row) is LocalCitationBinding for row in self.local_bindings)
            and type(self.lineages) is tuple
            and all(type(row) is FactSeededRecoveryLineage for row in self.lineages)
            and len(self.candidates)
            == len(self.local_bindings)
            == len(self.lineages)
            == self.coverage.selected_candidate_count,
            "fact coverage selected triples changed",
        )
        _require(
            len(self.candidates) <= self.budget.max_candidates
            and sum(row.token_count for row in self.candidates)
            == self.coverage.selected_candidate_tokens
            <= self.budget.max_tokens
            and self.coverage.budget_receipt_sha256
            == self.budget.receipt_sha256,
            "fact coverage result escaped exact caps",
        )
        selected_decisions = tuple(
            sorted(
                (row for row in self.decisions if row.status == "selected"),
                key=lambda row: row.selection_rank or 0,
            )
        )
        _require(
            tuple(row.selection_rank for row in selected_decisions)
            == tuple(range(1, len(selected_decisions) + 1))
            and tuple(row.occurrence_receipt_sha256 for row in selected_decisions)
            == self.coverage.selection_order_occurrence_receipt_sha256s,
            "fact coverage selected decision order changed",
        )
        for candidate, binding, lineage, decision in zip(
            self.candidates,
            self.local_bindings,
            self.lineages,
            selected_decisions,
            strict=True,
        ):
            _require(
                candidate_projection_receipt_sha256(candidate)
                == decision.candidate_receipt_sha256
                == lineage.recovered_candidate_receipt_sha256
                and binding.receipt_sha256
                == decision.local_binding_receipt_sha256
                == lineage.recovered_local_binding_receipt_sha256
                and decision.lineage_receipt_sha256 == lineage.receipt_sha256,
                "fact coverage selected lineage changed",
            )
        _require(
            self.coverage.population_candidate_count == len(self.inventory.candidates),
            "fact coverage result population changed",
        )
        _require(
            self.provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0,
            "fact coverage result retained provider state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact coverage result changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_fact_seeded_coverage_pack")
        assert_gold_blind(
            self.provider_projection(),
            path="typed_fact_seeded_coverage_pack_provider",
        )

    @property
    def truncated(self) -> bool:
        return bool(
            self.inventory.source_result_truncated
            or self.inventory.source_result_status != "scanned"
            or self.coverage.selected_candidate_count
            < self.coverage.unique_candidate_count
        )

    def provider_projection(self) -> dict[str, Any]:
        return {
            "candidates": [row.projection() for row in self.candidates],
            "dated_question": self.inventory.dated_question,
            "format": RESULT_FORMAT,
            "frontier_mode": FrontierMode.BOUNDED.value,
            "new_provider_calls": 0,
            "operator_spec": self.inventory.operator_spec.projection(),
            "retained_transformer_token_state_bytes": 0,
            "semantic_completeness_status": "not_claimed",
            "truncated": self.truncated,
        }

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "budget_receipt_sha256": self.budget.receipt_sha256,
            "coverage_receipt_sha256": self.coverage.receipt_sha256,
            "decision_receipt_sha256s": [
                row.receipt_sha256 for row in self.decisions
            ],
            "format": RESULT_FORMAT,
            "frontier_mode": FrontierMode.BOUNDED.value,
            "inventory_receipt_sha256": self.inventory.receipt_sha256,
            "new_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
            "selected_candidate_receipt_sha256s": [
                candidate_projection_receipt_sha256(row) for row in self.candidates
            ],
            "selected_lineage_receipt_sha256s": [
                row.receipt_sha256 for row in self.lineages
            ],
            "selected_local_binding_receipt_sha256s": [
                row.receipt_sha256 for row in self.local_bindings
            ],
            "semantic_completeness_status": "not_claimed",
            "truncated": self.truncated,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def local_audit_projection(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage.projection(),
            "decisions": [row.projection() for row in self.decisions],
            "inventory": self.inventory.projection(),
            "lineages": [row.projection() for row in self.lineages],
            "local_bindings": [row.projection() for row in self.local_bindings],
            "provider_projection_sha256": identity_sha256(
                self.provider_projection()
            ),
            "receipt": self.projection(),
        }


@dataclass(frozen=True, slots=True)
class _Record:
    occurrence_receipt_sha256: str
    candidate: FullStoreSlotCandidate
    local: LocalCitationBinding
    lineage: FactSeededRecoveryLineage
    candidate_receipt_sha256: str
    exact_span_receipt_sha256: str
    dedup_key_sha256: str
    facts: frozenset[str]
    cues: frozenset[str]
    support: frozenset[str]
    temporal: frozenset[str]
    actions: frozenset[str]
    personalization: frozenset[str]
    roles: frozenset[str]
    source_key_sha256: str
    turn_key_sha256: str
    density_numerator: int
    support_quality: int
    protocol_only_ultrashort: bool


@dataclass(slots=True)
class _Covered:
    slots: set[str]
    facts: set[str]
    cues: set[str]
    support: set[str]
    temporal: set[str]
    actions: set[str]
    personalization: set[str]
    roles: set[str]
    sources: set[str]
    turns: set[str]


@dataclass(frozen=True, slots=True)
class _Marginal:
    slots: tuple[str, ...]
    facts: tuple[str, ...]
    cues: tuple[str, ...]
    support: tuple[str, ...]
    temporal: tuple[str, ...]
    actions: tuple[str, ...]
    personalization: tuple[str, ...]
    roles: tuple[str, ...]
    source: bool
    turn: bool

    @property
    def rank_counts(self) -> tuple[int, ...]:
        return (
            len(self.slots),
            len(self.facts),
            len(self.cues),
            len(self.temporal),
            len(self.personalization),
            len(self.roles),
            # A direct lexical proof is stronger than a broad action-only
            # equivalence, but does not displace temporal/personal/role
            # obligations.  Because support is marginal, one direct row does
            # not permanently crowd out a genuinely new action relation.
            len(self.support),
            len(self.actions),
            int(self.source),
            int(self.turn),
        )


def build_fact_seeded_packing_inventory(
    result: TypedFactSeededReconstructionResult, /
) -> FactSeededPackingInventory:
    """Freeze the exact discovered triples of one reconstruction result."""

    _require(
        type(result) is TypedFactSeededReconstructionResult,
        "fact coverage inventory requires an exact reconstruction result",
    )
    return FactSeededPackingInventory(
        dated_question=result.parent_result.dated_question,
        operator_spec=result.parent_result.operator_spec,
        source_result_receipt_sha256=result.receipt_sha256,
        source_result_status=result.status,
        source_result_truncated=result.truncated,
        seed_items=result.seed_items,
        candidates=result.candidates,
        local_bindings=result.local_bindings,
        lineages=result.lineages,
    )


_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|i've|me|my|mine|we|our|ours)\b", re.I)


def _records(inventory: FactSeededPackingInventory) -> tuple[_Record, ...]:
    spec = inventory.operator_spec
    required_slots = {row.slot_id for row in spec.required_slots}
    question_actions = set(canonical_action_concepts(inventory.dated_question))
    personal_anchors = {
        term
        for item in inventory.seed_items
        for value in item.personalization_anchors
        for term in normalized_terms(value)
    }
    occurrence_counts: dict[str, int] = {}
    result: list[_Record] = []
    for candidate, local, lineage in zip(
        inventory.candidates,
        inventory.local_bindings,
        inventory.lineages,
        strict=True,
    ):
        candidate_receipt = candidate_projection_receipt_sha256(candidate)
        exact_span_receipt = citation_span_receipt_sha256(local)
        triple_receipt = identity_sha256(
            {
                "candidate_receipt_sha256": candidate_receipt,
                "lineage_receipt_sha256": lineage.receipt_sha256,
                "local_binding_receipt_sha256": local.receipt_sha256,
            }
        )
        occurrence = occurrence_counts.get(triple_receipt, 0)
        occurrence_counts[triple_receipt] = occurrence + 1
        occurrence_receipt = identity_sha256(
            {
                "format": f"{INVENTORY_FORMAT}-occurrence",
                "occurrence": occurrence,
                "triple_receipt_sha256": triple_receipt,
            }
        )
        temporal: set[str] = set()
        if spec.temporal_mode is not TemporalMode.NONE:
            if candidate.event_date is not None:
                temporal.add(f"event_date:{candidate.event_date}")
            temporal.update(
                f"axis:{value}"
                for value in candidate.selection_axes
                if "temporal" in value.casefold()
                or "date" in value.casefold()
                or "latest" in value.casefold()
                or "order" in value.casefold()
            )
            if temporal:
                temporal.add("temporal:evidence")

        actions = {
            f"action:{value}"
            for value in matched_action_concepts(
                inventory.dated_question, candidate.quote
            )
        }
        actions.update(
            f"completed_action:{value}"
            for value in completed_action_concepts(candidate.quote)
            if value in question_actions
        )

        axes = set(candidate.selection_axes)
        direct_lexical = bool(
            "fact_seed_support:direct_lexical" in axes
            or (
                candidate.matched_query_terms
                and "fact_seed_support:sealed_action_equivalence" not in axes
            )
        )
        action_equivalence = "fact_seed_support:sealed_action_equivalence" in axes
        support: set[str] = set()
        if direct_lexical:
            support.add("support:direct_lexical")
            support.update(
                f"direct_term:{value}" for value in candidate.matched_query_terms
            )
            support_quality = 4
        elif "original_operator_slot_support" in axes:
            support.add("support:operator_slot")
            support_quality = 3
        elif "fact_seed_support:selected_history_affinity" in axes:
            support.add("support:selected_history_affinity")
            support_quality = 2
        elif "fact_seed_support:selected_source_affinity" in axes:
            support.add("support:selected_source_affinity")
            support_quality = 1
        else:
            support_quality = 0
        protocol_only_ultrashort = bool(
            action_equivalence
            and candidate.token_count <= 4
            and not candidate.supported_slot_ids
            and not candidate.matched_query_terms
        )

        candidate_terms = set(normalized_terms(candidate.quote))
        personalization = {
            f"anchor:{value}" for value in personal_anchors & candidate_terms
        }
        if spec.personalization_required and _FIRST_PERSON_RE.search(candidate.quote):
            personalization.add("personalization:first_person")

        roles: set[str] = set()
        if candidate.role == "user":
            roles.add("exact_role:user")
        if candidate.role == spec.required_evidence_role:
            roles.add(f"required_role:{candidate.role}")

        source_key = identity_sha256(
            {
                "format": f"{INVENTORY_FORMAT}-source-key",
                "namespace_id": local.namespace_id,
                "source_id": local.source_id,
            }
        )
        turn_key = identity_sha256(
            {
                "chunk_id": local.span.chunk_id,
                "format": f"{INVENTORY_FORMAT}-turn-key",
                "namespace_id": local.namespace_id,
                "ordinal": local.span.ordinal,
                "source_id": local.source_id,
                "turn_id": local.span.turn_id,
            }
        )
        facts = frozenset(lineage.supporting_fact_receipt_sha256s)
        cues = frozenset(lineage.cue_receipt_sha256s)
        supported = frozenset(set(candidate.supported_slot_ids) & required_slots)
        density_numerator = (
            4 * len(supported)
            + 2 * len(facts)
            + len(cues)
            + len(candidate.matched_query_terms)
            + 2 * len(temporal)
            + 2 * len(actions)
            + 2 * len(personalization)
            + 2 * len(roles)
            + 2 * int(candidate.contains_numeric_value)
        )
        if protocol_only_ultrashort:
            density_numerator = max(0, density_numerator - 8)
        result.append(
            _Record(
                occurrence_receipt_sha256=occurrence_receipt,
                candidate=candidate,
                local=local,
                lineage=lineage,
                candidate_receipt_sha256=candidate_receipt,
                exact_span_receipt_sha256=exact_span_receipt,
                dedup_key_sha256=identity_sha256(
                    {
                        "format": f"{INVENTORY_FORMAT}-exact-dedup-key",
                        "quote_sha256": candidate.quote_sha256,
                        "span_receipt_sha256": exact_span_receipt,
                    }
                ),
                facts=facts,
                cues=cues,
                support=frozenset(support),
                temporal=frozenset(temporal),
                actions=frozenset(actions),
                personalization=frozenset(personalization),
                roles=frozenset(roles),
                source_key_sha256=source_key,
                turn_key_sha256=turn_key,
                density_numerator=density_numerator,
                support_quality=support_quality,
                protocol_only_ultrashort=protocol_only_ultrashort,
            )
        )
    return tuple(sorted(result, key=lambda row: row.occurrence_receipt_sha256))


def _marginal(
    record: _Record,
    covered: _Covered,
    required_slots: frozenset[str],
) -> _Marginal:
    return _Marginal(
        slots=_sorted_unique(
            (set(record.candidate.supported_slot_ids) & set(required_slots))
            - covered.slots
        ),
        facts=_sorted_unique(set(record.facts) - covered.facts),
        cues=_sorted_unique(set(record.cues) - covered.cues),
        support=_sorted_unique(set(record.support) - covered.support),
        temporal=_sorted_unique(set(record.temporal) - covered.temporal),
        actions=_sorted_unique(set(record.actions) - covered.actions),
        personalization=_sorted_unique(
            set(record.personalization) - covered.personalization
        ),
        roles=_sorted_unique(set(record.roles) - covered.roles),
        source=record.source_key_sha256 not in covered.sources,
        turn=record.turn_key_sha256 not in covered.turns,
    )


def _rank_key(
    record: _Record,
    covered: _Covered,
    required_slots: frozenset[str],
) -> tuple[Any, ...]:
    marginal = _marginal(record, covered, required_slots)
    return (
        *(-value for value in marginal.rank_counts),
        -record.support_quality,
        int(record.protocol_only_ultrashort),
        -Fraction(record.density_numerator, max(1, record.candidate.token_count)),
        record.candidate.token_count,
        record.occurrence_receipt_sha256,
    )


def _admit(
    record: _Record,
    marginal: _Marginal,
    covered: _Covered,
    required_slots: frozenset[str],
) -> None:
    covered.slots.update(set(record.candidate.supported_slot_ids) & required_slots)
    covered.facts.update(record.facts)
    covered.cues.update(record.cues)
    covered.support.update(record.support)
    covered.temporal.update(record.temporal)
    covered.actions.update(record.actions)
    covered.personalization.update(record.personalization)
    covered.roles.update(record.roles)
    covered.sources.add(record.source_key_sha256)
    covered.turns.add(record.turn_key_sha256)


def pack_fact_seeded_inventory(
    inventory: FactSeededPackingInventory,
    /,
    *,
    budget: FactSeededCoverageBudget | None = None,
) -> TypedFactSeededCoveragePackResult:
    """Select exact triples under both caps using only gold-blind coverage."""

    _require(
        type(inventory) is FactSeededPackingInventory,
        "fact coverage pack requires an exact inventory",
    )
    exact_budget = budget or FactSeededCoverageBudget()
    _require(
        type(exact_budget) is FactSeededCoverageBudget,
        "fact coverage pack requires an exact budget",
    )
    records = _records(inventory)

    unique: list[_Record] = []
    duplicates: dict[str, _Record] = {}
    dedup_seen: set[str] = set()
    for record in records:
        if record.dedup_key_sha256 in dedup_seen:
            duplicates[record.occurrence_receipt_sha256] = record
        else:
            dedup_seen.add(record.dedup_key_sha256)
            unique.append(record)

    # Compiled facts seed discovery; they do not replace an exact recovered
    # source candidate in the final pack.  Candidate slot coverage therefore
    # starts empty even when a cue fact already names the slot.
    required_slots = frozenset(
        row.slot_id for row in inventory.operator_spec.required_slots
    )
    unresolved_before = _sorted_unique(required_slots)
    covered = _Covered(
        slots=set(),
        facts=set(),
        cues=set(),
        support=set(),
        temporal=set(),
        actions=set(),
        personalization=set(),
        roles=set(),
        sources=set(),
        turns=set(),
    )
    remaining = list(unique)
    selected: list[_Record] = []
    selection_marginals: dict[str, _Marginal] = {}
    cumulative_tokens: dict[str, int] = {}
    selected_tokens = 0
    while remaining and len(selected) < exact_budget.max_candidates:
        feasible = tuple(
            row
            for row in remaining
            if selected_tokens + row.candidate.token_count <= exact_budget.max_tokens
        )
        if not feasible:
            break
        winner = min(
            feasible,
            key=lambda row: _rank_key(row, covered, required_slots),
        )
        marginal = _marginal(winner, covered, required_slots)
        selected.append(winner)
        selected_tokens += winner.candidate.token_count
        selection_marginals[winner.occurrence_receipt_sha256] = marginal
        cumulative_tokens[winner.occurrence_receipt_sha256] = selected_tokens
        _admit(winner, marginal, covered, required_slots)
        remaining.remove(winner)

    selected_ids = {row.occurrence_receipt_sha256 for row in selected}
    decisions: list[FactSeededCoverageDecision] = []
    rank_by_occurrence = {
        row.occurrence_receipt_sha256: index
        for index, row in enumerate(selected, start=1)
    }
    for record in records:
        occurrence = record.occurrence_receipt_sha256
        if occurrence in selected_ids:
            marginal = selection_marginals[occurrence]
            status: PackingDecisionStatus = "selected"
            rank = rank_by_occurrence[occurrence]
            cumulative = cumulative_tokens[occurrence]
            reason = "selected_by_lexicographic_marginal_coverage_then_density"
        elif occurrence in duplicates:
            marginal = _Marginal((), (), (), (), (), (), (), (), False, False)
            status = "duplicate_after_discovery"
            rank = None
            cumulative = selected_tokens
            reason = "exact_quote_and_span_deduplicated_after_candidate_discovery"
        else:
            marginal = _Marginal((), (), (), (), (), (), (), (), False, False)
            rank = None
            cumulative = selected_tokens
            if len(selected) >= exact_budget.max_candidates:
                status = "candidate_cap_excluded"
                reason = "unique_candidate_excluded_by_exact_candidate_cap"
            else:
                status = "token_cap_excluded"
                reason = "unique_candidate_excluded_by_exact_token_cap"
        decisions.append(
            FactSeededCoverageDecision(
                occurrence_receipt_sha256=occurrence,
                candidate_receipt_sha256=record.candidate_receipt_sha256,
                local_binding_receipt_sha256=record.local.receipt_sha256,
                lineage_receipt_sha256=record.lineage.receipt_sha256,
                status=status,
                selection_rank=rank,
                token_count=record.candidate.token_count,
                cumulative_selected_tokens=cumulative,
                marginal_slot_ids=marginal.slots,
                marginal_fact_receipt_sha256s=marginal.facts,
                marginal_cue_receipt_sha256s=marginal.cues,
                marginal_support_features=marginal.support,
                marginal_temporal_features=marginal.temporal,
                marginal_action_features=marginal.actions,
                marginal_personalization_features=marginal.personalization,
                marginal_role_features=marginal.roles,
                source_diversity_added=marginal.source,
                turn_diversity_added=marginal.turn,
                density_numerator=record.density_numerator,
                support_quality=record.support_quality,
                protocol_only_ultrashort=record.protocol_only_ultrashort,
                reason=reason,
            )
        )

    coverage = FactSeededCoverageReceipt(
        population_candidate_count=len(records),
        unique_candidate_count=len(unique),
        duplicate_candidate_count=len(duplicates),
        selected_candidate_count=len(selected),
        selected_candidate_tokens=selected_tokens,
        unresolved_slot_ids_before=unresolved_before,
        unresolved_slot_ids_after=_sorted_unique(required_slots - covered.slots),
        selected_fact_receipt_sha256s=_sorted_unique(covered.facts),
        selected_cue_receipt_sha256s=_sorted_unique(covered.cues),
        selected_support_features=_sorted_unique(covered.support),
        selected_temporal_features=_sorted_unique(covered.temporal),
        selected_action_features=_sorted_unique(covered.actions),
        selected_personalization_features=_sorted_unique(covered.personalization),
        selected_role_features=_sorted_unique(covered.roles),
        selected_source_key_sha256s=_sorted_unique(covered.sources),
        selected_turn_key_sha256s=_sorted_unique(covered.turns),
        selection_order_occurrence_receipt_sha256s=tuple(
            row.occurrence_receipt_sha256 for row in selected
        ),
        budget_receipt_sha256=exact_budget.receipt_sha256,
    )
    return TypedFactSeededCoveragePackResult(
        inventory=inventory,
        budget=exact_budget,
        decisions=tuple(sorted(decisions, key=lambda row: row.occurrence_receipt_sha256)),
        coverage=coverage,
        candidates=tuple(row.candidate for row in selected),
        local_bindings=tuple(row.local for row in selected),
        lineages=tuple(row.lineage for row in selected),
    )


def pack_typed_fact_seeded_reconstruction(
    result: TypedFactSeededReconstructionResult,
    /,
    *,
    budget: FactSeededCoverageBudget | None = None,
) -> TypedFactSeededCoveragePackResult:
    """Convenience entry point from the existing reconstruction adapter."""

    return pack_fact_seeded_inventory(
        build_fact_seeded_packing_inventory(result), budget=budget
    )


def adapt_fact_seeded_coverage_pack_to_contribution(
    result: TypedFactSeededCoveragePackResult,
    /,
    *,
    handle_start: int,
    group_start: int,
) -> TypedEvidenceContribution:
    """Expose the selected exact pairs as a bounded direct-pointer lane."""

    _require(
        type(result) is TypedFactSeededCoveragePackResult,
        "fact coverage contribution requires an exact pack result",
    )
    for value, label in (
        (handle_start, "fact coverage handle start"),
        (group_start, "fact coverage group start"),
    ):
        _require(type(value) is int and 1 <= value <= 999_999, f"{label} changed")
    _require(
        not result.candidates
        or handle_start + len(result.candidates) - 1 <= 999_999,
        "fact coverage handle range overflowed",
    )
    source_keys = _ordered_unique(
        tuple(
            f"{local.namespace_id}\0{local.source_id}"
            for local in result.local_bindings
        )
    )
    _require(
        not source_keys or group_start + len(source_keys) - 1 <= 999_999,
        "fact coverage group range overflowed",
    )
    groups = {
        key: f"G{group_start + index:03d}"
        for index, key in enumerate(source_keys)
    }
    sealed = identity_sha256(result.local_audit_projection())
    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, Any]] = []
    numeric_slots = {
        slot.slot_id
        for slot in result.inventory.operator_spec.required_slots
        if slot.requires_numeric
    }
    for index, (candidate, local, lineage) in enumerate(
        zip(result.candidates, result.local_bindings, result.lineages, strict=True)
    ):
        handle = f"H{handle_start + index:03d}"
        group = groups[f"{local.namespace_id}\0{local.source_id}"]
        binding = EvidenceHandleBinding(
            handle_id=handle,
            origin=EvidenceOrigin.DIRECT_POINTER,
            provenance_grade=ProvenanceGrade.DIRECT_POINTER,
            source_group_handle=group,
            sealed_artifact_sha256=sealed,
            parent_receipt_sha256=lineage.receipt_sha256,
            evidence_receipt_sha256=local.receipt_sha256,
            payload_sha256=candidate_projection_receipt_sha256(candidate),
            citation_sha256=candidate.quote_sha256,
            citation_char_count=len(candidate.quote),
            local_source_locator_sha256=local.receipt_sha256,
        )
        numeric = (
            conservative_numeric_value(candidate.quote)
            if numeric_slots & set(candidate.supported_slot_ids)
            else None
        )
        if numeric is not None:
            kind = "operand"
        elif result.inventory.operator_spec.temporal_mode is not TemporalMode.NONE:
            kind = "event"
        elif result.inventory.operator_spec.answer_shape.value == "set_list":
            kind = "member"
        elif result.inventory.operator_spec.style.value == "state_chain":
            kind = "state"
        else:
            kind = "direct"
        semantic_slots = tuple(
            slot
            for slot in result.inventory.operator_spec.required_slots
            if slot.slot_id in candidate.supported_slot_ids
            and slot.kind in {SlotKind.OPERAND, SlotKind.COMPARISON_SIDE}
        )
        raw: dict[str, Any] = {
            "handle_ids": [handle],
            "included": True,
            "kind": kind,
            "numeric_role": "operand" if numeric is not None else "none",
            "specificity_terms": [],
            "summary": candidate.quote,
            "value_authority": "explicit",
        }
        if len(semantic_slots) == 1:
            raw["entity_key"] = semantic_slots[0].label
        if candidate.event_date is not None:
            raw["date"] = candidate.event_date
        if candidate.role in {"user", "assistant"}:
            raw["relation"] = f"authored_by_{candidate.role}"
        if numeric is not None:
            raw["numeric_value"] = numeric
        bindings.append(binding)
        raw_items.append(raw)
    exact_bindings = tuple(bindings)
    parsed = parse_typed_items(
        raw_items,
        operator_spec=result.inventory.operator_spec,
        bindings=exact_bindings,
    )
    return TypedEvidenceContribution(
        mechanism_id=MECHANISM_ID,
        bindings=exact_bindings,
        parsed=parsed,
        sealed_artifact_sha256=sealed,
        frontier_mode=FrontierMode.BOUNDED,
        truncated=result.truncated,
    )


__all__ = [
    "FactSeededCoverageBudget",
    "FactSeededCoverageDecision",
    "FactSeededCoverageReceipt",
    "FactSeededPackingInventory",
    "MECHANISM_ID",
    "TypedFactSeededCoveragePackResult",
    "TypedFactSeededCoveragePackingError",
    "adapt_fact_seeded_coverage_pack_to_contribution",
    "build_fact_seeded_packing_inventory",
    "pack_fact_seeded_inventory",
    "pack_typed_fact_seeded_reconstruction",
]
