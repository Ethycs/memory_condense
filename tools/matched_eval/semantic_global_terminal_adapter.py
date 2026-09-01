"""Gold-blind cumulative P/R/L/G evidence compiler for one terminal prompt.

The four retrieval planes select against independent, non-borrowable budgets.
Only after every plane has selected do we resolve protected duplicates and
deduplicate exact spans.  Selected duplicates are replaced with the exact
provider-visible owner bytes, backed by a local containment proof.  Raw source
locators stay in the local audit plane; the provider sees only compact typed
evidence with disjoint opaque H/G ranges and the V3 prediction as fallback.

This module deliberately does not own a provider or a runner.  It consumes the
live, already authenticated R7/V6/V7 objects before their exact evidence bytes
are discarded and returns a replayable, provider-free compilation receipt.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
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
from .full_store_slot_closure import LocalCitationBinding
from .semantic_global_completion import (
    GlobalCompletionCandidate,
    GlobalCompletionEvidence,
    GlobalEvidenceObligation,
    GlobalProtectedDuplicate,
    GlobalSelectionAttempt,
    SemanticGlobalCompletionResult,
    _segment_supports as _global_segment_supports,
    linked_event_dates,
)
from .semantic_residual_search import (
    ExactCellSegment,
    SemanticResidualEvidence,
    SemanticResidualIndex,
    SemanticResidualProtectedDuplicate,
    SemanticResidualQuery,
    SemanticResidualSearchResult,
    semantic_residual_protected_evidence_population_receipt,
    semantic_residual_source_group_map,
)
from .source_group_reinjection import (
    LocalReinjectionEvidence,
    ProtectedLocalDuplicate,
    SourceGroupReinjectionResult,
)
from .selected_evidence_discourse_links import (
    SelectedEvidenceLinkInput,
    link_selected_evidence,
)
from .typed_action_semantics import (
    canonical_action_concepts,
    completed_action_concepts,
    linked_action_concepts,
    planned_action_concepts,
)
from .typed_memory_final_arm import (
    FittedTypedFinalPrompt,
    LOCAL_RETENTION_PRIORITY_WIDTH,
    fit_typed_final_prompt,
)
from .typed_numeric_semantics import single_numeric_mention
from .typed_operator_adapter import (
    COMPACT_FINAL_PROVIDER_FORMAT,
    ITEM_FORMAT,
    ConflictPolicy,
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    NumericRole,
    ParsedTypedItems,
    ProvenanceGrade,
    ProviderPayloadMode,
    TypedEvidenceItem,
    TypedEvidencePacket,
    TypedItemKind,
    build_typed_evidence_packet,
    parse_typed_items,
)
from .typed_operator_spec import AnswerShape, TypedOperatorSpec, normalized_terms


FORMAT = "memory-condense-semantic-global-terminal-compilation-v2"
LINKED_FORMAT = "memory-condense-semantic-global-terminal-compilation-v3"
POLICY_FORMAT = f"{FORMAT}-policy-v1"
SOURCE_FORMAT = f"{FORMAT}-sealed-sources-v1"
OWNER_FORMAT = f"{FORMAT}-protected-owner-v1"
SELECTION_FORMAT = f"{FORMAT}-plane-selection-v1"
DEDUP_FORMAT = f"{FORMAT}-post-selection-dedup-v2"
ROW_FORMAT = f"{FORMAT}-selected-row-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"
EXACT_SPAN_SUPPORT_FORMAT = f"{FORMAT}-exact-span-support-authority-v1"
EXACT_SPAN_SUPPORT_POPULATION_FORMAT = (
    f"{FORMAT}-exact-span-support-population-v1"
)

Plane = Literal["P", "R", "L", "G"]
ClosureClass = Literal[
    "none",
    "semantic_source_head",
    "selected_cluster_anchor",
    "selected_outside_anchor",
]
Disposition = Literal[
    "protected_owner",
    "packed_novel",
    "budget_unpacked",
    "protected_exact_duplicate",
    "completed_event_lane",
    "proposed_action_lane",
    "source_group_closure_lane",
    "selected_anchor_closure_lane",
]

PLANE_ORDER: tuple[Plane, ...] = ("P", "R", "L", "G")
MECHANISM_BY_PLANE: Mapping[Plane, str] = MappingProxyType(
    {
        "P": "terminal_protected_owner_v2",
        "R": "terminal_residual_retrieval_v2",
        "L": "terminal_local_reinjection_v2",
        "G": "terminal_global_completion_v2",
    }
)
HANDLE_RANGE_START: Mapping[Plane, int] = MappingProxyType(
    {"P": 100_000, "R": 200_000, "L": 300_000, "G": 400_000}
)
GROUP_RANGE_START: Mapping[Plane, int] = MappingProxyType(
    {"P": 500_000, "R": 600_000, "L": 700_000, "G": 800_000}
)
HARD_PROMPT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
CONSIDERATION_PRIORITY_WIDTH = 16
MAX_DIRECT_OPERAND_LANE_ITEMS = 3
SOURCE_ORDER_CONSIDERATION_POLICY = (
    "source-population-order-post-selection-skip-continue-v2"
)
R_CONSIDERATION_POLICY = (
    "query-event-temporal-role-action-relevance_then-source-group-round-robin_"
    "post-selection-skip-continue-v2"
)
L_CONSIDERATION_POLICY = (
    "packed-novel_then_protected-owner-substitute_then_budget-unpacked_"
    "completed-proposed-user-date-action-relevance_then-source-rank-v2"
)
G_CONSIDERATION_POLICY = (
    "bounded-direct-counted-operand_then-base-status-refill_then-"
    "top-four-evidence-partitions-source-group-round-robin-v3"
)
FINAL_PROTECTED_TRANCHE_POLICY = (
    "plane-minima_plus-all-l-packed-novel_plus-bounded-g-core_"
    "partition0-depth5_top4-round0_cluster-anchors_qualified-escape_"
    "plus-authenticated-dedup-authority-v2"
)
RETENTION_AUTHORITY_TRANSFER_POLICY = (
    "exact-span-provider-first-provenance_with-lexicographic-max-priority_"
    "and-or-hard-protection-v1"
)
EXACT_SPAN_SUPPORT_RANKING_POLICY = (
    "all-plane-candidate_exact-span_support-union_then-legacy-authority_"
    "fixed-width-24-v1"
)
CONSIDERATION_POLICY_BY_PLANE: Mapping[Plane, str] = MappingProxyType(
    {
        "P": SOURCE_ORDER_CONSIDERATION_POLICY,
        "R": R_CONSIDERATION_POLICY,
        "L": L_CONSIDERATION_POLICY,
        "G": G_CONSIDERATION_POLICY,
    }
)


class SemanticGlobalTerminalError(MatchedEvalContractError):
    """A terminal evidence, budget, ownership, or replay invariant changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SemanticGlobalTerminalError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    _require(
        all(type(value) is str and bool(value) for value in result)
        and len(set(result)) == len(result),
        f"{label} must be ordered unique text",
    )
    return result


@dataclass(frozen=True, slots=True)
class PlaneBudget:
    plane: Plane
    max_items: int
    evidence_token_cap: int
    minimum_items: int = 1

    def __post_init__(self) -> None:
        _require(self.plane in PLANE_ORDER, "terminal plane changed")
        _require(
            type(self.max_items) is int
            and self.max_items > 0
            and type(self.evidence_token_cap) is int
            and self.evidence_token_cap > 0
            and type(self.minimum_items) is int
            and 0 <= self.minimum_items <= self.max_items,
            "terminal plane budget changed",
        )

    def projection(self) -> dict[str, object]:
        return {
            "evidence_token_cap": self.evidence_token_cap,
            "max_items": self.max_items,
            "minimum_items": self.minimum_items,
            "plane": self.plane,
        }


def _default_budgets() -> tuple[PlaneBudget, ...]:
    return (
        PlaneBudget("P", 16, 1_400),
        PlaneBudget("R", 16, 1_600),
        PlaneBudget("L", 16, 1_600),
        PlaneBudget("G", 24, 2_400),
    )


@dataclass(frozen=True, slots=True)
class SemanticGlobalTerminalPolicy:
    plane_budgets: tuple[PlaneBudget, ...] = field(
        default_factory=_default_budgets
    )
    max_completed_event_lane_items: int = 8
    max_direct_operand_lane_items: Literal[3] = MAX_DIRECT_OPERAND_LANE_ITEMS
    max_partition_clusters: int = 4
    max_source_group_rounds_per_partition: int = 5
    max_anchor_partition_clusters: int = 3
    max_outside_cluster_anchors: int = 1
    minimum_usable_items_per_nonempty_mechanism: int = 1
    output_token_reserve: Literal[768] = OUTPUT_TOKEN_RESERVE
    hard_prompt_token_cap: Literal[8000] = HARD_PROMPT_TOKEN_CAP
    frontier_mode: Literal["open"] = "open"
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.plane_budgets) is tuple
            and tuple(row.plane for row in self.plane_budgets) == PLANE_ORDER
            and all(type(row) is PlaneBudget for row in self.plane_budgets),
            "terminal plane budgets lost exact P/R/L/G order",
        )
        _require(
            type(self.max_completed_event_lane_items) is int
            and self.max_completed_event_lane_items > 0
            and self.max_direct_operand_lane_items
            == MAX_DIRECT_OPERAND_LANE_ITEMS
            and self.max_partition_clusters == 4
            and self.max_source_group_rounds_per_partition == 5
            and self.max_anchor_partition_clusters == 3
            and self.max_outside_cluster_anchors == 1
            and type(self.minimum_usable_items_per_nonempty_mechanism) is int
            and self.minimum_usable_items_per_nonempty_mechanism >= 1,
            "terminal lane/minimum policy changed",
        )
        _require(
            self.output_token_reserve == OUTPUT_TOKEN_RESERVE
            and self.hard_prompt_token_cap == HARD_PROMPT_TOKEN_CAP
            and self.frontier_mode == "open",
            "terminal hard-cap/frontier policy changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "terminal policy changed")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def budget_by_plane(self) -> Mapping[Plane, PlaneBudget]:
        return MappingProxyType({row.plane: row for row in self.plane_budgets})

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": POLICY_FORMAT,
            "frontier_mode": "open",
            "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
            "max_completed_event_lane_items": self.max_completed_event_lane_items,
            "max_direct_operand_lane_items": self.max_direct_operand_lane_items,
            "max_partition_clusters": self.max_partition_clusters,
            "max_source_group_rounds_per_partition": (
                self.max_source_group_rounds_per_partition
            ),
            "max_anchor_partition_clusters": self.max_anchor_partition_clusters,
            "max_outside_cluster_anchors": self.max_outside_cluster_anchors,
            "minimum_usable_items_per_nonempty_mechanism": (
                self.minimum_usable_items_per_nonempty_mechanism
            ),
            "output_token_reserve": OUTPUT_TOKEN_RESERVE,
            "plane_consideration_policy": dict(CONSIDERATION_POLICY_BY_PLANE),
            "final_protected_tranche_policy": FINAL_PROTECTED_TRANCHE_POLICY,
            "post_selection_dedup_retention_authority_policy": (
                RETENTION_AUTHORITY_TRANSFER_POLICY
            ),
            "plane_budgets": [row.projection() for row in self.plane_budgets],
            "selection_budgets_non_borrowable": True,
            "skip_oversized_and_continue": True,
            "terminal_provider_order_may_reprioritize_upstream_within_plane": True,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class TerminalSealedSources:
    protected_owner_artifact_sha256: str
    residual_artifact_sha256: str
    parent_artifact_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.protected_owner_artifact_sha256, "protected-owner artifact"),
            (self.residual_artifact_sha256, "residual artifact"),
            (self.parent_artifact_sha256, "parent artifact"),
        ):
            require_sha256(value, label)
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "terminal sources changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def artifact_for_plane(self, plane: Plane) -> str:
        # L/G are live results over the exact same sealed residual/index
        # ancestor.  Calling their result receipts artifacts would create a
        # circular pre-publication claim, so their identities are bound by the
        # compilation fields while their citations retain this real ancestor.
        return (
            self.protected_owner_artifact_sha256
            if plane == "P"
            else self.residual_artifact_sha256
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": SOURCE_FORMAT,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "protected_owner_artifact_sha256": (
                self.protected_owner_artifact_sha256
            ),
            "residual_artifact_sha256": self.residual_artifact_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ProtectedOwnerEvidence:
    """One selected provider-visible P row bound to an owner citation.

    This is intentionally the small selected-owner inventory emitted by the
    R7 terminal construction, not the full protected binding universe used to
    authenticate duplicate ownership.
    """

    owner_binding_receipt_sha256: str
    protected_duplicate_receipt_sha256: str
    segment_receipt_sha256: str
    owner_candidate_id: str
    source_group_handle: str
    quote: str
    quote_sha256: str
    role: str
    created_at: str
    event_dates: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.owner_binding_receipt_sha256, "protected owner binding"),
            (self.protected_duplicate_receipt_sha256, "protected duplicate"),
            (self.segment_receipt_sha256, "protected owner segment"),
            (self.owner_candidate_id, "protected owner candidate"),
        ):
            require_sha256(value, label)
        _require(
            type(self.source_group_handle) is str
            and self.source_group_handle.startswith("G")
            and self.source_group_handle[1:].isdigit(),
            "protected owner source group changed",
        )
        for value, label in (
            (self.quote, "protected owner quote"),
            (self.role, "protected owner role"),
            (self.created_at, "protected owner created-at"),
        ):
            require_text(value, label)
        require_sha256(self.quote_sha256, "protected owner quote")
        _require(
            quote_sha256(self.quote) == self.quote_sha256,
            "protected owner quote bytes changed",
        )
        _ordered_unique(self.event_dates, "protected owner dates")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "protected owner changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "created_at": self.created_at,
            "event_dates": list(self.event_dates),
            "format": OWNER_FORMAT,
            "owner_binding_receipt_sha256": self.owner_binding_receipt_sha256,
            "owner_candidate_id": self.owner_candidate_id,
            "protected_duplicate_receipt_sha256": (
                self.protected_duplicate_receipt_sha256
            ),
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "source_group_handle": self.source_group_handle,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    @classmethod
    def from_provider_row(cls, row: Mapping[str, Any], /) -> "ProtectedOwnerEvidence":
        """Load the exact R7 selected-owner row without retaining its old H ID."""

        expected = {
            "created_at",
            "event_dates",
            "evidence_handle",
            "owner_binding_receipt_sha256",
            "owner_candidate_id",
            "protected_duplicate_receipt_sha256",
            "quote",
            "quote_sha256",
            "role",
            "segment_receipt_sha256",
            "source_group_handle",
        }
        _require(type(row) is dict and set(row) == expected, "protected owner row schema changed")
        handle = row["evidence_handle"]
        dates = row["event_dates"]
        _require(
            type(handle) is str and handle.startswith("P") and handle[1:].isdigit()
            and type(dates) is list and all(type(value) is str for value in dates),
            "protected owner provider row changed",
        )
        return cls(
            owner_binding_receipt_sha256=row["owner_binding_receipt_sha256"],
            protected_duplicate_receipt_sha256=row[
                "protected_duplicate_receipt_sha256"
            ],
            segment_receipt_sha256=row["segment_receipt_sha256"],
            owner_candidate_id=row["owner_candidate_id"],
            source_group_handle=row["source_group_handle"],
            quote=row["quote"],
            quote_sha256=row["quote_sha256"],
            role=row["role"],
            created_at=row["created_at"],
            event_dates=tuple(dates),
        )


@dataclass(frozen=True, slots=True)
class PlaneSelectionReceipt:
    plane: Plane
    candidate_receipt_sha256s: tuple[str, ...]
    consideration_policy_id: str
    consideration_candidate_receipt_sha256s: tuple[str, ...]
    consideration_priority_vectors: tuple[tuple[int, ...], ...]
    upstream_attempt_receipt_sha256s: tuple[str, ...]
    selected_candidate_receipt_sha256s: tuple[str, ...]
    skipped_candidate_receipt_sha256s: tuple[str, ...]
    selected_evidence_tokens: int
    evidence_token_cap: int
    max_items: int
    minimum_items: int
    upstream_budget_unpacked_selected: int
    completed_event_lane_selected: int
    proposed_action_lane_selected: int
    source_group_closure_lane_selected: int
    selected_anchor_closure_lane_selected: int
    direct_operand_population_candidate_receipt_sha256s: tuple[str, ...] = ()
    direct_operand_reserved_candidate_receipt_sha256s: tuple[str, ...] = ()
    base_status_refill_candidate_receipt_sha256s: tuple[str, ...] = ()
    direct_operand_lane_selected: int = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(self.plane in PLANE_ORDER, "selection plane changed")
        for values, label in (
            (self.candidate_receipt_sha256s, "selection candidates"),
            (
                self.consideration_candidate_receipt_sha256s,
                "selection consideration candidates",
            ),
            (self.upstream_attempt_receipt_sha256s, "upstream attempts"),
            (self.selected_candidate_receipt_sha256s, "selected candidates"),
            (self.skipped_candidate_receipt_sha256s, "skipped candidates"),
            (
                self.direct_operand_population_candidate_receipt_sha256s,
                "direct operand population",
            ),
            (
                self.direct_operand_reserved_candidate_receipt_sha256s,
                "direct operand reservations",
            ),
            (
                self.base_status_refill_candidate_receipt_sha256s,
                "base status refill",
            ),
        ):
            _ordered_unique(values, label)
            for value in values:
                require_sha256(value, label)
        candidates = set(self.candidate_receipt_sha256s)
        consideration = set(self.consideration_candidate_receipt_sha256s)
        selected = set(self.selected_candidate_receipt_sha256s)
        skipped = set(self.skipped_candidate_receipt_sha256s)
        direct_population = set(
            self.direct_operand_population_candidate_receipt_sha256s
        )
        direct_reserved = set(
            self.direct_operand_reserved_candidate_receipt_sha256s
        )
        base_status_refill = set(
            self.base_status_refill_candidate_receipt_sha256s
        )
        _require(
            consideration == candidates
            and selected.isdisjoint(skipped)
            and selected | skipped == candidates,
            "plane selection lost skip/retain partition",
        )
        _require(
            direct_population <= candidates
            and direct_reserved <= direct_population
            and self.direct_operand_reserved_candidate_receipt_sha256s
            == self.direct_operand_population_candidate_receipt_sha256s[
                : len(self.direct_operand_reserved_candidate_receipt_sha256s)
            ]
            and base_status_refill <= candidates
            and len(direct_reserved) <= MAX_DIRECT_OPERAND_LANE_ITEMS
            and (
                self.plane == "G"
                or not direct_population
                and not direct_reserved
                and not base_status_refill
            ),
            "direct operand/base-status refill audit changed",
        )
        _require(
            self.consideration_policy_id
            == CONSIDERATION_POLICY_BY_PLANE[self.plane]
            and type(self.consideration_priority_vectors) is tuple
            and len(self.consideration_priority_vectors)
            == len(self.consideration_candidate_receipt_sha256s)
            and all(
                type(vector) is tuple
                and len(vector) == CONSIDERATION_PRIORITY_WIDTH
                and all(type(value) is int for value in vector)
                for vector in self.consideration_priority_vectors
            ),
            "plane consideration policy or priority vectors changed",
        )
        _require(
            self.selected_candidate_receipt_sha256s
            == tuple(
                receipt
                for receipt in self.consideration_candidate_receipt_sha256s
                if receipt in selected
            )
            and self.skipped_candidate_receipt_sha256s
            == tuple(
                receipt
                for receipt in self.candidate_receipt_sha256s
                if receipt not in selected
            ),
            "plane selection order escaped its receipt-bound consideration order",
        )
        for value, label in (
            (self.selected_evidence_tokens, "selected tokens"),
            (self.evidence_token_cap, "selection token cap"),
            (self.max_items, "selection item cap"),
            (self.minimum_items, "selection minimum"),
            (
                self.upstream_budget_unpacked_selected,
                "selected upstream-unpacked count",
            ),
            (self.completed_event_lane_selected, "selected completed lane count"),
            (self.proposed_action_lane_selected, "selected proposed lane count"),
            (
                self.source_group_closure_lane_selected,
                "selected source-group closure lane count",
            ),
            (
                self.selected_anchor_closure_lane_selected,
                "selected anchor closure lane count",
            ),
            (self.direct_operand_lane_selected, "selected direct operand count"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        _require(
            self.direct_operand_lane_selected
            == len(direct_reserved & selected),
            "selected direct operand count changed",
        )
        _require(
            self.selected_evidence_tokens <= self.evidence_token_cap
            and len(selected) <= self.max_items
            and (
                not candidates or len(selected) >= self.minimum_items
            ),
            "plane selection violated its non-borrowable budget/minimum",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "plane selection changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "candidate_receipt_sha256s": list(self.candidate_receipt_sha256s),
            "base_status_refill_candidate_receipt_sha256s": list(
                self.base_status_refill_candidate_receipt_sha256s
            ),
            "completed_event_lane_selected": self.completed_event_lane_selected,
            "consideration_order": [
                {
                    "candidate_receipt_sha256": receipt,
                    "priority": list(priority),
                }
                for receipt, priority in zip(
                    self.consideration_candidate_receipt_sha256s,
                    self.consideration_priority_vectors,
                    strict=True,
                )
            ],
            "consideration_policy_id": self.consideration_policy_id,
            "dedup_applied": False,
            "direct_operand_lane_selected": self.direct_operand_lane_selected,
            "direct_operand_population_candidate_receipt_sha256s": list(
                self.direct_operand_population_candidate_receipt_sha256s
            ),
            "direct_operand_reserved_candidate_receipt_sha256s": list(
                self.direct_operand_reserved_candidate_receipt_sha256s
            ),
            "evidence_token_cap": self.evidence_token_cap,
            "format": SELECTION_FORMAT,
            "max_items": self.max_items,
            "minimum_items": self.minimum_items,
            "plane": self.plane,
            "proposed_action_lane_selected": self.proposed_action_lane_selected,
            "source_group_closure_lane_selected": (
                self.source_group_closure_lane_selected
            ),
            "selected_anchor_closure_lane_selected": (
                self.selected_anchor_closure_lane_selected
            ),
            "selected_candidate_receipt_sha256s": list(
                self.selected_candidate_receipt_sha256s
            ),
            "selected_evidence_tokens": self.selected_evidence_tokens,
            "skipped_candidate_receipt_sha256s": list(
                self.skipped_candidate_receipt_sha256s
            ),
            "upstream_budget_unpacked_selected": (
                self.upstream_budget_unpacked_selected
            ),
            "upstream_attempt_population_receipt_sha256": identity_sha256(
                list(self.upstream_attempt_receipt_sha256s)
            ),
            "upstream_attempt_receipt_sha256s": list(
                self.upstream_attempt_receipt_sha256s
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class DeduplicationReceipt:
    selected_before_dedup_receipt_sha256s: tuple[str, ...]
    retained_after_dedup_receipt_sha256s: tuple[str, ...]
    exclusions: tuple[Mapping[str, Any], ...]
    substitutions: tuple[Mapping[str, Any], ...]
    retention_authority_transfers: tuple[Mapping[str, Any], ...]
    substitution_count: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for values, label in (
            (self.selected_before_dedup_receipt_sha256s, "dedup selected"),
            (self.retained_after_dedup_receipt_sha256s, "dedup retained"),
        ):
            _ordered_unique(values, label)
            for value in values:
                require_sha256(value, label)
        selected = set(self.selected_before_dedup_receipt_sha256s)
        retained = set(self.retained_after_dedup_receipt_sha256s)
        _require(
            retained <= selected
            and type(self.exclusions) is tuple
            and all(type(row) is MappingProxyType for row in self.exclusions)
            and type(self.substitutions) is tuple
            and all(type(row) is MappingProxyType for row in self.substitutions)
            and type(self.retention_authority_transfers) is tuple
            and all(
                type(row) is MappingProxyType
                for row in self.retention_authority_transfers
            )
            and type(self.substitution_count) is int
            and self.substitution_count == len(self.substitutions),
            "post-selection dedup receipt changed",
        )
        exclusion_keys = {
            "dropped_candidate_receipt_sha256",
            "exact_span_identity_sha256",
            "kept_candidate_receipt_sha256",
            "policy",
        }
        exclusion_triples: list[tuple[str, str, str]] = []
        dropped_receipts: list[str] = []
        for row in self.exclusions:
            _require(set(row) == exclusion_keys, "dedup exclusion schema changed")
            kept = row["kept_candidate_receipt_sha256"]
            dropped = row["dropped_candidate_receipt_sha256"]
            span = row["exact_span_identity_sha256"]
            for value, label in (
                (kept, "dedup kept candidate"),
                (dropped, "dedup dropped candidate"),
                (span, "dedup exact span"),
            ):
                require_sha256(value, label)
            _require(
                kept in retained
                and dropped in selected - retained
                and kept != dropped
                and row["policy"] == "exact_span_after_independent_plane_selection",
                "dedup exclusion escaped its selected/retained populations",
            )
            exclusion_triples.append((kept, dropped, span))
            dropped_receipts.append(dropped)
        _ordered_unique(tuple(dropped_receipts), "dedup exclusions")
        _require(
            set(dropped_receipts) == selected - retained,
            "dedup exclusions do not exactly explain the dropped population",
        )
        transfer_keys = {
            "authority_candidate_receipt_sha256",
            "authority_source_plane",
            "exact_span_identity_sha256",
            "hard_protected",
            "kept_candidate_receipt_sha256",
            "policy",
            "retention_priority",
        }
        transfer_triples: list[tuple[str, str, str]] = []
        for row in self.retention_authority_transfers:
            _require(
                set(row) == transfer_keys,
                "dedup retention-authority transfer schema changed",
            )
            kept = row["kept_candidate_receipt_sha256"]
            authority = row["authority_candidate_receipt_sha256"]
            span = row["exact_span_identity_sha256"]
            for value, label in (
                (kept, "dedup authority kept candidate"),
                (authority, "dedup authority candidate"),
                (span, "dedup authority exact span"),
            ):
                require_sha256(value, label)
            priority = row["retention_priority"]
            _require(
                kept in retained
                and authority in selected - retained
                and row["authority_source_plane"] in PLANE_ORDER
                and type(row["hard_protected"]) is bool
                and type(priority) is tuple
                and len(priority) == LOCAL_RETENTION_PRIORITY_WIDTH
                and all(type(value) is int for value in priority)
                and row["policy"] == RETENTION_AUTHORITY_TRANSFER_POLICY,
                "dedup retention authority escaped its exact selected span",
            )
            transfer_triples.append((kept, authority, span))
        _require(
            transfer_triples == exclusion_triples,
            "dedup retention authority does not exactly cover every exclusion",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "post-selection dedup changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "dedup_after_all_plane_selection": True,
            "exclusions": [dict(row) for row in self.exclusions],
            "format": DEDUP_FORMAT,
            "retained_after_dedup_receipt_sha256s": list(
                self.retained_after_dedup_receipt_sha256s
            ),
            "retention_authority_transfers": [
                {
                    **dict(row),
                    "retention_priority": list(row["retention_priority"]),
                }
                for row in self.retention_authority_transfers
            ],
            "retention_authority_transfer_policy": (
                RETENTION_AUTHORITY_TRANSFER_POLICY
            ),
            "selected_before_dedup_receipt_sha256s": list(
                self.selected_before_dedup_receipt_sha256s
            ),
            "substitutions": [dict(row) for row in self.substitutions],
            "substitution_count": self.substitution_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class _Candidate:
    plane: Plane
    candidate_id: str
    segment_receipt_sha256: str
    binding: LocalCitationBinding
    selection_quote: str
    quote: str
    role: str
    created_at: str
    event_dates: tuple[str, ...]
    action_concepts: tuple[str, ...]
    upstream_receipt_sha256: str
    selection_receipt_sha256: str
    disposition: Disposition
    upstream_disposition: str
    source_rank: int
    matched_completed_actions: tuple[str, ...]
    matched_planned_actions: tuple[str, ...]
    matched_query_actions: tuple[str, ...]
    supported_obligation_ids: tuple[str, ...] = ()
    source_group_supported_obligation_ids: tuple[str, ...] = ()
    source_group_supported_kinds: tuple[str, ...] = ()
    query_temporal_support: bool = False
    explicit_temporal_conflict: bool = False
    past_event_witness: bool = False
    exact_relation_support: bool = False
    closure_class: ClosureClass = "none"
    partition_cluster_rank: int = -1
    source_group_round: int = -1
    partition_joint_source_group_count: int = 0
    partition_supported_source_group_count: int = 0
    duplicate_owner_binding_receipt_sha256: str | None = None
    duplicate_span_identity_sha256: str | None = None
    owner_source_plane: Plane | None = None
    upstream_attempt_receipt_sha256: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(self.plane in PLANE_ORDER, "candidate plane changed")
        for value, label in (
            (self.candidate_id, "terminal candidate"),
            (self.segment_receipt_sha256, "terminal segment"),
            (self.upstream_receipt_sha256, "terminal upstream row"),
            (self.selection_receipt_sha256, "terminal selection row"),
        ):
            require_sha256(value, label)
        _require(type(self.binding) is LocalCitationBinding, "candidate binding changed")
        for value, label in (
            (self.selection_quote, "candidate selection quote"),
            (self.quote, "candidate exact quote"),
            (self.role, "candidate role"),
            (self.created_at, "candidate created-at"),
        ):
            require_text(value, label)
        _ordered_unique(self.event_dates, "candidate event dates")
        _ordered_unique(self.action_concepts, "candidate actions")
        _ordered_unique(self.matched_completed_actions, "candidate completed actions")
        _ordered_unique(self.matched_planned_actions, "candidate planned actions")
        _ordered_unique(self.matched_query_actions, "candidate query actions")
        _ordered_unique(self.supported_obligation_ids, "candidate obligations")
        _ordered_unique(
            self.source_group_supported_obligation_ids,
            "candidate source-group obligations",
        )
        _ordered_unique(
            self.source_group_supported_kinds,
            "candidate source-group obligation kinds",
        )
        _require(
            type(self.source_rank) is int
            and self.source_rank >= 0
            and self.disposition
            in {
                "protected_owner",
                "packed_novel",
                "budget_unpacked",
                "protected_exact_duplicate",
                "completed_event_lane",
                "proposed_action_lane",
                "source_group_closure_lane",
                "selected_anchor_closure_lane",
            },
            "candidate rank/disposition changed",
        )
        _require(
            type(self.query_temporal_support) is bool
            and type(self.explicit_temporal_conflict) is bool
            and type(self.past_event_witness) is bool
            and type(self.exact_relation_support) is bool
            and self.closure_class
            in {
                "none",
                "semantic_source_head",
                "selected_cluster_anchor",
                "selected_outside_anchor",
            }
            and type(self.partition_cluster_rank) is int
            and self.partition_cluster_rank >= -1
            and type(self.source_group_round) is int
            and self.source_group_round >= -1
            and type(self.partition_joint_source_group_count) is int
            and self.partition_joint_source_group_count >= 0
            and type(self.partition_supported_source_group_count) is int
            and self.partition_supported_source_group_count >= 0
            and (
                self.closure_class != "none"
            )
            == (
                self.partition_cluster_rank >= 0
                and self.source_group_round >= 0
            ),
            "candidate query/group closure metadata changed",
        )
        require_text(self.upstream_disposition, "candidate upstream disposition")
        if self.upstream_attempt_receipt_sha256 is not None:
            require_sha256(
                self.upstream_attempt_receipt_sha256,
                "candidate upstream attempt",
            )
        duplicate = self.upstream_disposition == "protected_exact_duplicate"
        _require(
            duplicate
            == (
                self.duplicate_owner_binding_receipt_sha256 is not None
                and self.duplicate_span_identity_sha256 is not None
            ),
            "duplicate candidate lost owner proof",
        )
        if duplicate:
            require_sha256(
                self.duplicate_owner_binding_receipt_sha256,
                "duplicate owner binding",
            )
            require_sha256(
                self.duplicate_span_identity_sha256,
                "duplicate selected span",
            )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "terminal candidate changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "action_concepts": list(self.action_concepts),
            "binding_receipt_sha256": self.binding.receipt_sha256,
            "candidate_id": self.candidate_id,
            "created_at": self.created_at,
            "disposition": self.disposition,
            "duplicate_owner_binding_receipt_sha256": (
                self.duplicate_owner_binding_receipt_sha256
            ),
            "duplicate_span_identity_sha256": self.duplicate_span_identity_sha256,
            "event_dates": list(self.event_dates),
            "format": ROW_FORMAT,
            "matched_completed_actions": list(self.matched_completed_actions),
            "matched_planned_actions": list(self.matched_planned_actions),
            "matched_query_actions": list(self.matched_query_actions),
            "supported_obligation_ids": list(self.supported_obligation_ids),
            "source_group_supported_obligation_ids": list(
                self.source_group_supported_obligation_ids
            ),
            "source_group_supported_kinds": list(
                self.source_group_supported_kinds
            ),
            "query_temporal_support": self.query_temporal_support,
            "explicit_temporal_conflict": self.explicit_temporal_conflict,
            "past_event_witness": self.past_event_witness,
            "exact_relation_support": self.exact_relation_support,
            "closure_class": self.closure_class,
            "partition_cluster_rank": self.partition_cluster_rank,
            "source_group_round": self.source_group_round,
            "partition_joint_source_group_count": (
                self.partition_joint_source_group_count
            ),
            "partition_supported_source_group_count": (
                self.partition_supported_source_group_count
            ),
            "owner_source_plane": self.owner_source_plane,
            "plane": self.plane,
            "quote_sha256": quote_sha256(self.quote),
            "role": self.role,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "selection_quote_sha256": quote_sha256(self.selection_quote),
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "source_rank": self.source_rank,
            "upstream_receipt_sha256": self.upstream_receipt_sha256,
            "upstream_disposition": self.upstream_disposition,
            "upstream_attempt_receipt_sha256": (
                self.upstream_attempt_receipt_sha256
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ExactSpanSupportAuthority:
    """Local-only support signals carried by every copy of one exact span."""

    exact_span_identity_sha256: str
    authority_candidate_receipt_sha256s: tuple[str, ...]
    authority_source_planes: tuple[Plane, ...]
    supported_obligation_ids: tuple[str, ...]
    source_group_supported_obligation_ids: tuple[str, ...]
    source_group_supported_kinds: tuple[str, ...]
    matched_query_actions: tuple[str, ...]
    exact_relation_support: bool
    query_temporal_support: bool
    past_event_witness: bool
    role: str
    priority_prefix: tuple[int, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(
            self.exact_span_identity_sha256,
            "exact-span support identity",
        )
        _ordered_unique(
            self.authority_candidate_receipt_sha256s,
            "exact-span support candidates",
        )
        for receipt in self.authority_candidate_receipt_sha256s:
            require_sha256(receipt, "exact-span support candidate")
        _require(
            bool(self.authority_candidate_receipt_sha256s)
            and type(self.authority_source_planes) is tuple
            and len(self.authority_source_planes)
            == len(self.authority_candidate_receipt_sha256s)
            and all(plane in PLANE_ORDER for plane in self.authority_source_planes),
            "exact-span support candidate planes changed",
        )
        for values, label in (
            (self.supported_obligation_ids, "exact-span direct obligations"),
            (
                self.source_group_supported_obligation_ids,
                "exact-span source-group obligations",
            ),
            (self.source_group_supported_kinds, "exact-span support kinds"),
            (self.matched_query_actions, "exact-span query actions"),
        ):
            _ordered_unique(values, label)
        _require(
            type(self.exact_relation_support) is bool
            and type(self.query_temporal_support) is bool
            and type(self.past_event_witness) is bool,
            "exact-span support flags changed",
        )
        require_text(self.role, "exact-span support role")
        expected_prefix = (
            len(self.supported_obligation_ids),
            len(self.source_group_supported_obligation_ids),
            int("typed_slot" in self.source_group_supported_kinds),
            int("entity" in self.source_group_supported_kinds),
            int("action" in self.source_group_supported_kinds),
            int(self.exact_relation_support),
            int(self.query_temporal_support),
            int(self.past_event_witness),
            len(self.matched_query_actions),
            int(self.role == "user"),
        )
        _require(
            type(self.priority_prefix) is tuple
            and len(self.priority_prefix) == 10
            and all(type(value) is int for value in self.priority_prefix)
            and self.priority_prefix == expected_prefix,
            "exact-span support priority prefix changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "exact-span support authority changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "authority_candidate_receipt_sha256s": list(
                self.authority_candidate_receipt_sha256s
            ),
            "authority_source_planes": list(self.authority_source_planes),
            "exact_relation_support": self.exact_relation_support,
            "exact_span_identity_sha256": self.exact_span_identity_sha256,
            "format": EXACT_SPAN_SUPPORT_FORMAT,
            "matched_query_actions": list(self.matched_query_actions),
            "past_event_witness": self.past_event_witness,
            "policy": EXACT_SPAN_SUPPORT_RANKING_POLICY,
            "priority_prefix": list(self.priority_prefix),
            "query_temporal_support": self.query_temporal_support,
            "role": self.role,
            "source_group_supported_kinds": list(
                self.source_group_supported_kinds
            ),
            "source_group_supported_obligation_ids": list(
                self.source_group_supported_obligation_ids
            ),
            "supported_obligation_ids": list(self.supported_obligation_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ExactSpanSupportPopulationReceipt:
    """Authenticate the complete pre-selection population and its span unions."""

    plane_candidate_receipt_sha256s: tuple[
        tuple[Plane, tuple[str, ...]], ...
    ]
    plane_selection_receipt_sha256s: tuple[str, ...]
    authorities: tuple[ExactSpanSupportAuthority, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.plane_candidate_receipt_sha256s) is tuple
            and tuple(plane for plane, _receipts in self.plane_candidate_receipt_sha256s)
            == PLANE_ORDER,
            "exact-span support population lost P/R/L/G order",
        )
        candidate_plane: dict[str, Plane] = {}
        for plane, receipts in self.plane_candidate_receipt_sha256s:
            _ordered_unique(receipts, f"exact-span support {plane} candidates")
            for receipt in receipts:
                require_sha256(receipt, "exact-span support population candidate")
                _require(
                    receipt not in candidate_plane,
                    "exact-span support population repeated a candidate",
                )
                candidate_plane[receipt] = plane
        _require(
            type(self.plane_selection_receipt_sha256s) is tuple
            and len(self.plane_selection_receipt_sha256s) in {0, len(PLANE_ORDER)},
            "exact-span support selection receipt population changed",
        )
        for receipt in self.plane_selection_receipt_sha256s:
            require_sha256(receipt, "exact-span support plane selection")
        _require(
            type(self.authorities) is tuple
            and all(type(row) is ExactSpanSupportAuthority for row in self.authorities)
            and len({row.exact_span_identity_sha256 for row in self.authorities})
            == len(self.authorities),
            "exact-span support authorities changed",
        )
        represented: list[str] = []
        for authority in self.authorities:
            represented.extend(authority.authority_candidate_receipt_sha256s)
            _require(
                all(
                    candidate_plane.get(receipt) == plane
                    for receipt, plane in zip(
                        authority.authority_candidate_receipt_sha256s,
                        authority.authority_source_planes,
                        strict=True,
                    )
                ),
                "exact-span support authority escaped its candidate plane",
            )
        _require(
            len(represented) == len(set(represented))
            and set(represented) == set(candidate_plane),
            "exact-span support authorities do not partition the candidate population",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "exact-span support population changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def candidate_receipt_sha256s(self) -> tuple[str, ...]:
        return tuple(
            receipt
            for _plane, receipts in self.plane_candidate_receipt_sha256s
            for receipt in receipts
        )

    @property
    def authority_by_span(self) -> Mapping[str, ExactSpanSupportAuthority]:
        return MappingProxyType(
            {row.exact_span_identity_sha256: row for row in self.authorities}
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "authorities": [row.projection() for row in self.authorities],
            "format": EXACT_SPAN_SUPPORT_POPULATION_FORMAT,
            "plane_candidate_receipt_sha256s": {
                plane: list(receipts)
                for plane, receipts in self.plane_candidate_receipt_sha256s
            },
            "plane_selection_receipt_sha256s": list(
                self.plane_selection_receipt_sha256s
            ),
            "policy": EXACT_SPAN_SUPPORT_RANKING_POLICY,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _exact_span_support_population(
    *,
    candidates_by_plane: Mapping[Plane, Sequence[_Candidate]],
    plane_selections: Sequence[PlaneSelectionReceipt] = (),
) -> ExactSpanSupportPopulationReceipt:
    """Aggregate support from all candidates without granting donor ownership."""

    _require(
        set(candidates_by_plane) == set(PLANE_ORDER),
        "exact-span support population changed planes",
    )
    plane_rows = tuple(
        (plane, tuple(candidates_by_plane[plane])) for plane in PLANE_ORDER
    )
    flattened = tuple(row for _plane, rows in plane_rows for row in rows)
    _require(
        all(
            type(row) is _Candidate and row.plane == plane
            for plane, rows in plane_rows
            for row in rows
        )
        and len({row.receipt_sha256 for row in flattened}) == len(flattened),
        "exact-span support population contains invalid candidates",
    )
    selections = tuple(plane_selections)
    if selections:
        _require(
            tuple(row.plane for row in selections) == PLANE_ORDER
            and all(type(row) is PlaneSelectionReceipt for row in selections)
            and all(
                selection.candidate_receipt_sha256s
                == tuple(row.receipt_sha256 for row in rows)
                for selection, (_plane, rows) in zip(
                    selections,
                    plane_rows,
                    strict=True,
                )
            ),
            "exact-span support population differs from plane selection candidates",
        )

    grouped: dict[str, list[_Candidate]] = {}
    for row in flattened:
        span_sha = identity_sha256(row.binding.span.identity_payload())
        grouped.setdefault(span_sha, []).append(row)

    authorities: list[ExactSpanSupportAuthority] = []
    for span_sha, rows in grouped.items():
        quote_hashes = {quote_sha256(row.quote) for row in rows}
        roles = {row.role for row in rows}
        _require(
            len(quote_hashes) == 1
            and len(roles) == 1
            and all(
                identity_sha256(row.binding.span.identity_payload()) == span_sha
                for row in rows
            ),
            "exact-span support copies disagree on exact content or role",
        )

        def ordered_union(attribute: str) -> tuple[str, ...]:
            return tuple(
                dict.fromkeys(
                    value
                    for row in rows
                    for value in getattr(row, attribute)
                )
            )

        direct = ordered_union("supported_obligation_ids")
        grouped_obligations = ordered_union(
            "source_group_supported_obligation_ids"
        )
        kinds = ordered_union("source_group_supported_kinds")
        query_actions = ordered_union("matched_query_actions")
        exact_relation = any(row.exact_relation_support for row in rows)
        temporal = any(row.query_temporal_support for row in rows)
        past_event = any(row.past_event_witness for row in rows)
        role = rows[0].role
        authorities.append(
            ExactSpanSupportAuthority(
                exact_span_identity_sha256=span_sha,
                authority_candidate_receipt_sha256s=tuple(
                    row.receipt_sha256 for row in rows
                ),
                authority_source_planes=tuple(row.plane for row in rows),
                supported_obligation_ids=direct,
                source_group_supported_obligation_ids=grouped_obligations,
                source_group_supported_kinds=kinds,
                matched_query_actions=query_actions,
                exact_relation_support=exact_relation,
                query_temporal_support=temporal,
                past_event_witness=past_event,
                role=role,
                priority_prefix=(
                    len(direct),
                    len(grouped_obligations),
                    int("typed_slot" in kinds),
                    int("entity" in kinds),
                    int("action" in kinds),
                    int(exact_relation),
                    int(temporal),
                    int(past_event),
                    len(query_actions),
                    int(role == "user"),
                ),
            )
        )
    return ExactSpanSupportPopulationReceipt(
        plane_candidate_receipt_sha256s=tuple(
            (plane, tuple(row.receipt_sha256 for row in rows))
            for plane, rows in plane_rows
        ),
        plane_selection_receipt_sha256s=tuple(
            row.receipt_sha256 for row in selections
        ),
        authorities=tuple(authorities),
    )


_RETENTION_AUTHORITY_SEQUENCE_KEYS = (
    "authority_source_planes",
    "authority_source_receipt_sha256s",
    "effective_retention_priority",
    "own_retention_priority",
)
_EXACT_SPAN_SUPPORT_SEQUENCE_KEYS = (
    "authority_candidate_receipt_sha256s",
    "authority_source_planes",
    "matched_query_actions",
    "priority_prefix",
    "source_group_supported_kinds",
    "source_group_supported_obligation_ids",
    "supported_obligation_ids",
)


def _project_local_audit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return JSON-native local audit rows without changing ranking types."""

    projected: list[dict[str, Any]] = []
    for row in rows:
        public_row = dict(row)
        typed_terminal = public_row.get("typed_terminal")
        if typed_terminal is not None:
            _require(
                type(typed_terminal) is dict,
                "terminal typed local audit changed schema",
            )
            public_typed = dict(typed_terminal)
            authority = public_typed.get("retention_authority")
            _require(
                type(authority) is dict,
                "terminal retention authority changed schema",
            )
            public_authority = dict(authority)
            for key in _RETENTION_AUTHORITY_SEQUENCE_KEYS:
                values = public_authority.get(key)
                _require(
                    type(values) is tuple,
                    "terminal retention authority sequence changed type",
                )
                public_authority[key] = list(values)
            public_typed["retention_authority"] = public_authority
            support = public_typed.get("exact_span_support_authority")
            _require(
                type(support) is dict,
                "terminal exact-span support authority changed schema",
            )
            public_support = dict(support)
            for key in _EXACT_SPAN_SUPPORT_SEQUENCE_KEYS:
                values = public_support.get(key)
                _require(
                    type(values) is tuple,
                    "terminal exact-span support authority sequence changed type",
                )
                public_support[key] = list(values)
            public_typed["exact_span_support_authority"] = public_support
            public_row["typed_terminal"] = public_typed
        projected.append(public_row)
    return projected


@dataclass(frozen=True, slots=True)
class SemanticGlobalTerminalCompilation:
    policy: SemanticGlobalTerminalPolicy
    sealed_sources: TerminalSealedSources
    residual_index_receipt_sha256: str
    query_receipt_sha256: str
    residual_result_receipt_sha256: str
    local_result_receipt_sha256: str
    global_result_receipt_sha256: str
    protected_owner_population_receipt_sha256: str
    exact_span_support_population: ExactSpanSupportPopulationReceipt
    plane_selections: tuple[PlaneSelectionReceipt, ...]
    post_selection_dedup: DeduplicationReceipt
    packet: TypedEvidencePacket
    fitted: FittedTypedFinalPrompt
    mechanism_by_handle: Mapping[str, str]
    retained_row_receipt_sha256s: tuple[str, ...]
    local_rows: tuple[Mapping[str, Any], ...]
    format_id: str = FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            self.format_id in {FORMAT, LINKED_FORMAT},
            "terminal compilation format changed",
        )
        _require(type(self.policy) is SemanticGlobalTerminalPolicy, "terminal policy type changed")
        _require(type(self.sealed_sources) is TerminalSealedSources, "terminal sources type changed")
        for value, label in (
            (self.residual_index_receipt_sha256, "terminal index"),
            (self.query_receipt_sha256, "terminal query"),
            (self.residual_result_receipt_sha256, "terminal residual result"),
            (self.local_result_receipt_sha256, "terminal local result"),
            (self.global_result_receipt_sha256, "terminal global result"),
            (
                self.protected_owner_population_receipt_sha256,
                "terminal protected-owner population",
            ),
        ):
            require_sha256(value, label)
        _require(
            type(self.plane_selections) is tuple
            and tuple(row.plane for row in self.plane_selections) == PLANE_ORDER
            and all(type(row) is PlaneSelectionReceipt for row in self.plane_selections),
            "terminal plane selection order changed",
        )
        _require(
            type(self.exact_span_support_population)
            is ExactSpanSupportPopulationReceipt
            and self.exact_span_support_population.plane_selection_receipt_sha256s
            == tuple(row.receipt_sha256 for row in self.plane_selections)
            and self.exact_span_support_population.candidate_receipt_sha256s
            == tuple(
                receipt
                for selection in self.plane_selections
                for receipt in selection.candidate_receipt_sha256s
            ),
            "terminal exact-span support escaped plane selection populations",
        )
        _require(
            type(self.post_selection_dedup) is DeduplicationReceipt
            and type(self.packet) is TypedEvidencePacket
            and type(self.fitted) is FittedTypedFinalPrompt
            and self.fitted.packet.receipt_sha256 == self.packet.receipt_sha256,
            "terminal packet/fitted proof changed",
        )
        _require(
            self.packet.frontier.mode is FrontierMode.OPEN
            and self.packet.frontier.closed is False
            and self.packet.provider_payload_mode is ProviderPayloadMode.COMPACT_FINAL
            and self.fitted.prompt_token_proxy + OUTPUT_TOKEN_RESERVE
            <= HARD_PROMPT_TOKEN_CAP,
            "terminal prompt overstated closure or exceeded 8k",
        )
        _ordered_unique(self.retained_row_receipt_sha256s, "terminal retained rows")
        _require(
            type(self.mechanism_by_handle) in {dict, MappingProxyType}
            and set(self.mechanism_by_handle)
            == {row.handle_id for row in self.packet.handles}
            and type(self.local_rows) is tuple
            and all(type(row) is MappingProxyType for row in self.local_rows),
            "terminal local audit changed",
        )
        expected = identity_sha256(self.projection(include_local=False, include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "terminal compilation changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="semantic_global_terminal")
        assert_gold_blind(self.provider_projection(), path="semantic_global_terminal_provider")

    def provider_projection(self) -> dict[str, Any]:
        return dict(self.fitted.provider_input)

    @property
    def local_audit_receipt_sha256(self) -> str:
        return identity_sha256(
            {
                "format": f"{FORMAT}-local-audit-v1",
                "exact_span_support_population": (
                    self.exact_span_support_population.projection()
                ),
                "local_rows": _project_local_audit_rows(self.local_rows),
                "mechanism_by_handle": dict(self.mechanism_by_handle),
            }
        )

    def projection(
        self,
        *,
        include_local: bool = False,
        include_receipt: bool = True,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": self.format_id,
            "frontier_support_closure_proven": False,
            "exact_span_support_population_receipt_sha256": (
                self.exact_span_support_population.receipt_sha256
            ),
            "global_result_receipt_sha256": self.global_result_receipt_sha256,
            "gold_loaded": False,
            "local_result_receipt_sha256": self.local_result_receipt_sha256,
            "local_audit_receipt_sha256": self.local_audit_receipt_sha256,
            "new_provider_calls": 0,
            "packet": self.packet.projection(),
            "plane_selections": [row.projection() for row in self.plane_selections],
            "policy": self.policy.projection(),
            "post_selection_dedup": self.post_selection_dedup.projection(),
            "protected_owner_population_receipt_sha256": (
                self.protected_owner_population_receipt_sha256
            ),
            "query_receipt_sha256": self.query_receipt_sha256,
            "residual_index_receipt_sha256": self.residual_index_receipt_sha256,
            "residual_result_receipt_sha256": self.residual_result_receipt_sha256,
            "retained_row_receipt_sha256s": list(
                self.retained_row_receipt_sha256s
            ),
            "retained_transformer_token_state_bytes": 0,
            "sealed_sources": self.sealed_sources.projection(),
            "terminal_prompt": self.fitted.projection(include_local=False),
        }
        if include_local:
            value["local_audit"] = {
                "exact_span_support_population": (
                    self.exact_span_support_population.projection()
                ),
                "local_rows": _project_local_audit_rows(self.local_rows),
                "mechanism_by_handle": dict(self.mechanism_by_handle),
                "packet": self.packet.projection(),
                "terminal_prompt": self.fitted.projection(include_local=True),
            }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _segment_inventory(
    residual_index: SemanticResidualIndex,
) -> tuple[
    Mapping[str, tuple[str, ExactCellSegment]],
    Mapping[str, tuple[ExactCellSegment, ...]],
]:
    by_receipt: dict[str, tuple[str, ExactCellSegment]] = {}
    by_chunk: dict[str, list[ExactCellSegment]] = {}
    for cell in residual_index.cells:
        for segment in cell.segments:
            _require(
                segment.receipt_sha256 not in by_receipt,
                "terminal index repeats an exact segment receipt",
            )
            by_receipt[segment.receipt_sha256] = (cell.cell_id, segment)
            by_chunk.setdefault(segment.span.chunk_id, []).append(segment)
    return (
        MappingProxyType(by_receipt),
        MappingProxyType({key: tuple(value) for key, value in by_chunk.items()}),
    )


def _span_identity_sha256(segment: ExactCellSegment) -> str:
    return identity_sha256(segment.span.identity_payload())


def _binding_for_segment(
    residual_index: SemanticResidualIndex,
    *,
    candidate_id: str,
    source_group_handle: str,
    segment: ExactCellSegment,
) -> LocalCitationBinding:
    return LocalCitationBinding(
        candidate_id=candidate_id,
        source_group_handle=source_group_handle,
        namespace_id=residual_index.namespace_id,
        cache_receipt_sha256=residual_index.cache_receipt_sha256,
        source_database_sha256=residual_index.source_database_sha256,
        source_store_receipt_sha256=residual_index.source_store_receipt_sha256,
        source_id=segment.source_id,
        partition_id=segment.partition_id,
        span=segment.span,
        quote_sha256=segment.quote_sha256,
    )


def _resolve_binding_quote(
    residual_index: SemanticResidualIndex,
    binding: LocalCitationBinding,
    by_chunk: Mapping[str, tuple[ExactCellSegment, ...]],
) -> tuple[ExactCellSegment, str]:
    _require(
        type(binding) is LocalCitationBinding
        and binding.namespace_id == residual_index.namespace_id
        and binding.cache_receipt_sha256 == residual_index.cache_receipt_sha256
        and binding.source_database_sha256
        == residual_index.source_database_sha256
        and binding.source_store_receipt_sha256
        == residual_index.source_store_receipt_sha256,
        "terminal binding escaped its exact residual index",
    )
    matches = tuple(
        segment
        for segment in by_chunk.get(binding.span.chunk_id, ())
        if segment.source_id == binding.source_id
        and segment.partition_id == binding.partition_id
        and segment.span.start_char <= binding.span.start_char
        and binding.span.end_char <= segment.span.end_char
    )
    _require(
        len(matches) == 1,
        "terminal binding does not have one exact containing segment",
    )
    segment = matches[0]
    start = binding.span.start_char - segment.span.start_char
    end = binding.span.end_char - segment.span.start_char
    quote = segment.quote[start:end]
    _require(
        bool(quote)
        and quote_sha256(quote) == binding.quote_sha256
        == binding.span.quote_sha256,
        "terminal binding subspan does not reproduce its exact quote",
    )
    return segment, quote


def _validated_segment(
    by_receipt: Mapping[str, tuple[str, ExactCellSegment]],
    receipt_sha256: str,
    *,
    cell_id: str | None = None,
) -> ExactCellSegment:
    value = by_receipt.get(receipt_sha256)
    _require(value is not None, "terminal candidate lost its exact segment")
    assert value is not None
    actual_cell_id, segment = value
    _require(
        cell_id is None or actual_cell_id == cell_id,
        "terminal candidate escaped its exact cell",
    )
    return segment


def _matched_completed_actions(question: str, quote: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(canonical_action_concepts(question))
            & set(completed_action_concepts(quote))
        )
    )


def _matched_planned_actions(question: str, quote: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(canonical_action_concepts(question))
            & set(planned_action_concepts(quote))
        )
    )


def _matched_query_actions(question: str, quote: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(canonical_action_concepts(question))
            & set(linked_action_concepts(quote))
        )
    )


_MONTH_NUMBER_BY_NAME: Mapping[str, int] = MappingProxyType(
    {
        name: number
        for number, name in enumerate(
            (
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ),
            start=1,
        )
    }
)
_MONTH_NAME_PATTERN = "|".join(_MONTH_NUMBER_BY_NAME)
_MONTH_NAME_RE = re.compile(
    rf"\b(?P<month>{_MONTH_NAME_PATTERN})\b", re.IGNORECASE
)
_ADJACENT_EXCLUSIVE_MONTH_BOUNDARY_RE = re.compile(
    rf"\bbefore\s+(?:the\s+)?(?P<month>{_MONTH_NAME_PATTERN})\b"
    r"(?:\s+(?:comes|starts|begins|arrives))?",
    re.IGNORECASE,
)
_PAST_EVENT_RE = re.compile(
    r"\b(?:attended|visited|went|tried|learned|bought|got|acquired|met|"
    r"came|returned|brought|recently|last\s+(?:week|month)|"
    r"\w+\s+weeks?\s+ago|opening\s+night)\b",
    re.IGNORECASE,
)
_NEGATIVE_OR_IRREALIS_RE = re.compile(
    r"\b(?:won['’]?t|wouldn['’]?t|didn['’]?t|never|not\s+going\s+to|"
    r"might\s+not|may\s+not|could\s+have|would\s+have)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _QueryRelevance:
    temporal_support: bool
    explicit_temporal_conflict: bool
    past_event_witness: bool
    role_support: bool


def _iso_day(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(value[:10].replace("/", "-"), "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _adjacent_exclusive_month_boundary_spans(
    *,
    quote: str,
    date_obligations: Sequence[GlobalEvidenceObligation],
) -> tuple[tuple[int, int], ...]:
    """Locate ``before <next month>`` boundary labels, not event months.

    A label is excluded only when it is the calendar month immediately after
    an obligation's inclusive end date.  Other month mentions remain ordinary
    evidence and can still establish an explicit temporal conflict.
    """

    adjacent_months: set[int] = set()
    for obligation in date_obligations:
        if obligation.target_date_end is None:
            continue
        end = _iso_day(obligation.target_date_end)
        if end is None:
            continue
        following = end.date() + timedelta(days=1)
        if following.day == 1:
            adjacent_months.add(following.month)
    return tuple(
        match.span("month")
        for match in _ADJACENT_EXCLUSIVE_MONTH_BOUNDARY_RE.finditer(quote)
        if _MONTH_NUMBER_BY_NAME[match.group("month").casefold()]
        in adjacent_months
    )


def _query_relevance(
    *,
    query: SemanticResidualQuery,
    obligations: Sequence[Any],
    quote: str,
    role: str,
    created_at: str,
    event_dates: Sequence[str],
) -> _QueryRelevance:
    date_obligations = tuple(
        row
        for row in obligations
        if row.kind == "date"
        and row.target_date_start is not None
        and row.target_date_end is not None
    )
    required_roles = {
        row.required_role
        for row in obligations
        if row.kind == "role" and row.required_role is not None
    }
    role_support = not required_roles or role in required_roles
    target_months = {
        int(row.target_date_start[5:7])
        for row in date_obligations
        if len(row.target_date_start) >= 7
    }
    boundary_spans = set(
        _adjacent_exclusive_month_boundary_spans(
            quote=quote,
            date_obligations=date_obligations,
        )
    )
    quoted_months = {
        _MONTH_NUMBER_BY_NAME[match.group("month").casefold()]
        for match in _MONTH_NAME_RE.finditer(quote)
        if match.span("month") not in boundary_spans
    }
    explicit_conflict = bool(target_months and quoted_months - target_months)
    parsed_dates = tuple(
        parsed
        for value in event_dates
        if (parsed := _iso_day(value)) is not None
    )
    created = _iso_day(created_at)
    exact_date = any(
        row.target_date_start <= parsed.date().isoformat() <= row.target_date_end
        for row in date_obligations
        for parsed in parsed_dates
    )
    created_in_interval = bool(
        created is not None
        and any(
            row.target_date_start
            <= created.date().isoformat()
            <= row.target_date_end
            for row in date_obligations
        )
    )
    rolling = False
    if (
        created is not None
        and query.operator_spec.query_timestamp is not None
        and query.operator_spec.temporal_window_days is not None
    ):
        asked = _iso_day(
            query.operator_spec.query_timestamp.replace("/", "-").split(" (")[0]
        )
        rolling = bool(
            asked is not None
            and asked - timedelta(days=query.operator_spec.temporal_window_days)
            <= created
            <= asked
        )
    matched_completed = _matched_completed_actions(query.dated_question, quote)
    matched_planned = _matched_planned_actions(query.dated_question, quote)
    negative = bool(_NEGATIVE_OR_IRREALIS_RE.search(quote))
    past_event = bool(
        role == "user"
        and not negative
        and (
            matched_completed
            or (_PAST_EVENT_RE.search(quote) and not matched_planned)
        )
    )
    near_interval = bool(
        created is not None
        and any(
            datetime.fromisoformat(row.target_date_start).date()
            <= created.date()
            <= datetime.fromisoformat(row.target_date_end).date()
            + timedelta(days=7)
            for row in date_obligations
        )
    )
    return _QueryRelevance(
        temporal_support=bool(
            exact_date
            or created_in_interval
            or rolling
            or (past_event and near_interval and not explicit_conflict)
        ),
        explicit_temporal_conflict=explicit_conflict,
        past_event_witness=past_event,
        role_support=role_support,
    )


def _ordinary_candidate(
    *,
    plane: Plane,
    candidate_id: str,
    segment: ExactCellSegment,
    binding: LocalCitationBinding,
    upstream_receipt_sha256: str,
    selection_receipt_sha256: str,
    disposition: Disposition,
    upstream_disposition: str,
    source_rank: int,
    dated_question: str,
    upstream_attempt_receipt_sha256: str | None = None,
    event_dates: tuple[str, ...] | None = None,
    action_concepts: tuple[str, ...] | None = None,
    supported_obligation_ids: tuple[str, ...] = (),
    source_group_supported_obligation_ids: tuple[str, ...] = (),
    source_group_supported_kinds: tuple[str, ...] = (),
    query_relevance: _QueryRelevance | None = None,
    partition_cluster_rank: int = -1,
    source_group_round: int = -1,
    partition_joint_source_group_count: int = 0,
    partition_supported_source_group_count: int = 0,
    exact_relation_support: bool = False,
    closure_class: ClosureClass = "none",
) -> _Candidate:
    return _Candidate(
        plane=plane,
        candidate_id=candidate_id,
        segment_receipt_sha256=segment.receipt_sha256,
        binding=binding,
        selection_quote=segment.quote,
        quote=segment.quote,
        role=segment.role,
        created_at=segment.created_at,
        event_dates=segment.event_dates if event_dates is None else event_dates,
        action_concepts=(
            canonical_action_concepts(segment.quote)
            if action_concepts is None
            else action_concepts
        ),
        upstream_receipt_sha256=upstream_receipt_sha256,
        selection_receipt_sha256=selection_receipt_sha256,
        disposition=disposition,
        upstream_disposition=upstream_disposition,
        source_rank=source_rank,
        matched_completed_actions=_matched_completed_actions(
            dated_question, segment.quote
        ),
        matched_planned_actions=_matched_planned_actions(
            dated_question, segment.quote
        ),
        matched_query_actions=_matched_query_actions(dated_question, segment.quote),
        supported_obligation_ids=supported_obligation_ids,
        source_group_supported_obligation_ids=source_group_supported_obligation_ids,
        source_group_supported_kinds=source_group_supported_kinds,
        query_temporal_support=(
            False if query_relevance is None else query_relevance.temporal_support
        ),
        explicit_temporal_conflict=(
            False
            if query_relevance is None
            else query_relevance.explicit_temporal_conflict
        ),
        past_event_witness=(
            False if query_relevance is None else query_relevance.past_event_witness
        ),
        exact_relation_support=exact_relation_support,
        closure_class=closure_class,
        partition_cluster_rank=partition_cluster_rank,
        source_group_round=source_group_round,
        partition_joint_source_group_count=partition_joint_source_group_count,
        partition_supported_source_group_count=(
            partition_supported_source_group_count
        ),
        upstream_attempt_receipt_sha256=upstream_attempt_receipt_sha256,
    )


def _duplicate_candidate(
    *,
    plane: Plane,
    candidate_id: str,
    selected_segment: ExactCellSegment,
    owner_binding: LocalCitationBinding,
    owner_plane: Plane,
    owner_segment: ExactCellSegment,
    owner_quote: str,
    duplicate_span_identity_sha256: str,
    upstream_receipt_sha256: str,
    selection_receipt_sha256: str,
    disposition: Disposition,
    source_rank: int,
    dated_question: str,
    upstream_attempt_receipt_sha256: str | None = None,
    event_dates: tuple[str, ...] | None = None,
    action_concepts: tuple[str, ...] | None = None,
    supported_obligation_ids: tuple[str, ...] = (),
    source_group_supported_obligation_ids: tuple[str, ...] = (),
    source_group_supported_kinds: tuple[str, ...] = (),
    query_relevance: _QueryRelevance | None = None,
    partition_cluster_rank: int = -1,
    source_group_round: int = -1,
    partition_joint_source_group_count: int = 0,
    partition_supported_source_group_count: int = 0,
    exact_relation_support: bool = False,
    closure_class: ClosureClass = "none",
) -> _Candidate:
    _require(
        _span_identity_sha256(selected_segment)
        == duplicate_span_identity_sha256
        and selected_segment.span.chunk_id == owner_binding.span.chunk_id
        and selected_segment.span.start_char <= owner_binding.span.start_char
        and owner_binding.span.end_char <= selected_segment.span.end_char,
        "selected duplicate does not contain its exact protected owner",
    )
    relative_start = owner_binding.span.start_char - selected_segment.span.start_char
    relative_end = owner_binding.span.end_char - selected_segment.span.start_char
    contained = selected_segment.quote[relative_start:relative_end]
    _require(
        contained == owner_quote
        and quote_sha256(contained) == owner_binding.quote_sha256,
        "selected duplicate containment proof changed owner bytes",
    )
    return _Candidate(
        plane=plane,
        candidate_id=candidate_id,
        segment_receipt_sha256=selected_segment.receipt_sha256,
        binding=owner_binding,
        selection_quote=selected_segment.quote,
        quote=owner_quote,
        role=owner_segment.role,
        created_at=owner_segment.created_at,
        event_dates=(
            owner_segment.event_dates if event_dates is None else event_dates
        ),
        action_concepts=(
            canonical_action_concepts(owner_quote)
            if action_concepts is None
            else action_concepts
        ),
        upstream_receipt_sha256=upstream_receipt_sha256,
        selection_receipt_sha256=selection_receipt_sha256,
        disposition=disposition,
        upstream_disposition="protected_exact_duplicate",
        source_rank=source_rank,
        matched_completed_actions=_matched_completed_actions(
            dated_question, owner_quote
        ),
        matched_planned_actions=_matched_planned_actions(dated_question, owner_quote),
        matched_query_actions=_matched_query_actions(dated_question, owner_quote),
        supported_obligation_ids=supported_obligation_ids,
        source_group_supported_obligation_ids=source_group_supported_obligation_ids,
        source_group_supported_kinds=source_group_supported_kinds,
        query_temporal_support=(
            False if query_relevance is None else query_relevance.temporal_support
        ),
        explicit_temporal_conflict=(
            False
            if query_relevance is None
            else query_relevance.explicit_temporal_conflict
        ),
        past_event_witness=(
            False if query_relevance is None else query_relevance.past_event_witness
        ),
        exact_relation_support=exact_relation_support,
        closure_class=closure_class,
        partition_cluster_rank=partition_cluster_rank,
        source_group_round=source_group_round,
        partition_joint_source_group_count=partition_joint_source_group_count,
        partition_supported_source_group_count=(
            partition_supported_source_group_count
        ),
        duplicate_owner_binding_receipt_sha256=owner_binding.receipt_sha256,
        duplicate_span_identity_sha256=duplicate_span_identity_sha256,
        owner_source_plane=owner_plane,
        upstream_attempt_receipt_sha256=upstream_attempt_receipt_sha256,
    )


def _owner_inventory(
    protected_owner_universe_bindings: Sequence[LocalCitationBinding],
    residual_result: SemanticResidualSearchResult,
    local_result: SourceGroupReinjectionResult,
) -> tuple[Mapping[str, LocalCitationBinding], Mapping[str, Plane]]:
    owners: dict[str, LocalCitationBinding] = {}
    planes: dict[str, Plane] = {}
    for plane, rows in (
        ("P", tuple(protected_owner_universe_bindings)),
        ("R", residual_result.local_bindings),
        ("L", local_result.local_bindings),
    ):
        for row in rows:
            _require(type(row) is LocalCitationBinding, "terminal owner binding changed")
            prior = owners.get(row.receipt_sha256)
            _require(
                prior is None or prior == row,
                "terminal owner receipt identifies different bindings",
            )
            owners[row.receipt_sha256] = row
            planes.setdefault(row.receipt_sha256, plane)  # exact repeats keep earliest owner
    return MappingProxyType(owners), MappingProxyType(planes)


def _selected_protected_owner_candidates(
    *,
    residual_index: SemanticResidualIndex,
    dated_question: str,
    selected_protected_owner_evidence: Sequence[ProtectedOwnerEvidence],
    residual_result: SemanticResidualSearchResult,
    owner_by_receipt: Mapping[str, LocalCitationBinding],
    owner_plane_by_receipt: Mapping[str, Plane],
    by_receipt: Mapping[str, tuple[str, ExactCellSegment]],
    by_chunk: Mapping[str, tuple[ExactCellSegment, ...]],
) -> tuple[_Candidate, ...]:
    duplicate_by_receipt = {
        row.receipt_sha256: row for row in residual_result.protected_duplicates
    }
    _require(
        len(duplicate_by_receipt) == len(residual_result.protected_duplicates),
        "residual duplicates repeat receipts",
    )
    output: list[_Candidate] = []
    for rank, row in enumerate(selected_protected_owner_evidence):
        _require(type(row) is ProtectedOwnerEvidence, "selected P evidence changed")
        binding = owner_by_receipt.get(row.owner_binding_receipt_sha256)
        duplicate = duplicate_by_receipt.get(
            row.protected_duplicate_receipt_sha256
        )
        _require(
            binding is not None
            and owner_plane_by_receipt.get(binding.receipt_sha256) == "P"
            and duplicate is not None
            and duplicate.protected_binding_receipt_sha256
            == binding.receipt_sha256
            and duplicate.protected_candidate_id == binding.candidate_id
            == row.owner_candidate_id
            and duplicate.segment_receipt_sha256 == row.segment_receipt_sha256,
            "selected P row lost its exact R duplicate/owner binding",
        )
        assert binding is not None and duplicate is not None
        owner_segment, owner_quote = _resolve_binding_quote(
            residual_index, binding, by_chunk
        )
        selected_segment = _validated_segment(
            by_receipt, duplicate.segment_receipt_sha256, cell_id=duplicate.cell_id
        )
        _require(
            owner_quote == row.quote
            and row.quote_sha256 == binding.quote_sha256
            and row.role == owner_segment.role
            and row.created_at == owner_segment.created_at
            and row.event_dates == owner_segment.event_dates,
            "selected P provider row differs from its exact owner bytes",
        )
        # P itself is the exact owner.  Its selection cost is the exact owner
        # quote; later duplicate planes use the same containment proof.
        output.append(
            _ordinary_candidate(
                plane="P",
                candidate_id=binding.candidate_id,
                segment=owner_segment,
                binding=binding,
                upstream_receipt_sha256=row.receipt_sha256,
                selection_receipt_sha256=row.receipt_sha256,
                disposition="protected_owner",
                upstream_disposition="protected_owner",
                source_rank=rank,
                dated_question=dated_question,
            )
        )
        _require(
            _span_identity_sha256(selected_segment)
            == duplicate.span_identity_sha256
            and selected_segment.span.chunk_id == binding.span.chunk_id
            and selected_segment.span.start_char <= binding.span.start_char
            and binding.span.end_char <= selected_segment.span.end_char,
            "selected P owner lost duplicate containment proof",
        )
    return tuple(output)


def _residual_candidates(
    *,
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    obligations: Sequence[Any],
    result: SemanticResidualSearchResult,
    owner_by_receipt: Mapping[str, LocalCitationBinding],
    owner_plane_by_receipt: Mapping[str, Plane],
    by_receipt: Mapping[str, tuple[str, ExactCellSegment]],
    by_chunk: Mapping[str, tuple[ExactCellSegment, ...]],
) -> tuple[_Candidate, ...]:
    sources = tuple(sorted({row.source_id for row in result.attempted_selection}))
    group_by_source = semantic_residual_source_group_map(sources)
    evidence_by_receipt = {row.receipt_sha256: row for row in result.evidence}
    binding_by_receipt = {row.receipt_sha256: row for row in result.local_bindings}
    duplicate_by_receipt = {
        row.receipt_sha256: row for row in result.protected_duplicates
    }
    output: list[_Candidate] = []
    for rank, attempt in enumerate(result.attempted_selection):
        segment = _validated_segment(
            by_receipt, attempt.segment_receipt_sha256, cell_id=attempt.cell_id
        )
        _require(segment.source_id == attempt.source_id, "R attempt source changed")
        if attempt.disposition == "protected_exact_duplicate":
            duplicate = duplicate_by_receipt.get(
                attempt.protected_duplicate_receipt_sha256 or ""
            )
            _require(
                duplicate is not None
                and duplicate.segment_receipt_sha256 == segment.receipt_sha256
                and duplicate.protected_binding_receipt_sha256
                == attempt.local_binding_receipt_sha256,
                "R duplicate attempt lost its exact audit",
            )
            assert duplicate is not None
            owner = owner_by_receipt.get(duplicate.protected_binding_receipt_sha256)
            owner_plane = owner_plane_by_receipt.get(
                duplicate.protected_binding_receipt_sha256
            )
            _require(owner is not None and owner_plane is not None, "R owner missing")
            assert owner is not None and owner_plane is not None
            owner_segment, owner_quote = _resolve_binding_quote(
                residual_index, owner, by_chunk
            )
            output.append(
                _duplicate_candidate(
                    plane="R",
                    candidate_id=attempt.candidate_id,
                    selected_segment=segment,
                    owner_binding=owner,
                    owner_plane=owner_plane,
                    owner_segment=owner_segment,
                    owner_quote=owner_quote,
                    duplicate_span_identity_sha256=duplicate.span_identity_sha256,
                    upstream_receipt_sha256=duplicate.receipt_sha256,
                    selection_receipt_sha256=attempt.receipt_sha256,
                    disposition="protected_exact_duplicate",
                    source_rank=rank,
                    dated_question=query.dated_question,
                    upstream_attempt_receipt_sha256=attempt.receipt_sha256,
                    query_relevance=_query_relevance(
                        query=query,
                        obligations=obligations,
                        quote=owner_quote,
                        role=owner_segment.role,
                        created_at=owner_segment.created_at,
                        event_dates=owner_segment.event_dates,
                    ),
                )
            )
            continue
        binding = _binding_for_segment(
            residual_index,
            candidate_id=attempt.candidate_id,
            source_group_handle=group_by_source[attempt.source_id],
            segment=segment,
        )
        _require(
            binding.receipt_sha256 == attempt.local_binding_receipt_sha256,
            "R attempted binding cannot be reconstructed from the exact segment",
        )
        evidence = evidence_by_receipt.get(attempt.evidence_receipt_sha256 or "")
        if evidence is not None:
            _require(
                evidence.segment_receipt_sha256 == segment.receipt_sha256
                and evidence.citation_binding_receipt_sha256
                == binding.receipt_sha256
                and evidence.quote == segment.quote
                and binding_by_receipt.get(binding.receipt_sha256) == binding,
                "packed R evidence changed exact bytes or ownership",
            )
        output.append(
            _ordinary_candidate(
                plane="R",
                candidate_id=attempt.candidate_id,
                segment=segment,
                binding=binding,
                upstream_receipt_sha256=(
                    evidence.receipt_sha256
                    if evidence is not None
                    else attempt.evidence_receipt_sha256 or attempt.receipt_sha256
                ),
                selection_receipt_sha256=attempt.receipt_sha256,
                disposition="packed_novel" if evidence is not None else "budget_unpacked",
                upstream_disposition=(
                    "packed_novel" if evidence is not None else "budget_unpacked"
                ),
                source_rank=rank,
                dated_question=query.dated_question,
                upstream_attempt_receipt_sha256=attempt.receipt_sha256,
                query_relevance=_query_relevance(
                    query=query,
                    obligations=obligations,
                    quote=segment.quote,
                    role=segment.role,
                    created_at=segment.created_at,
                    event_dates=segment.event_dates,
                ),
            )
        )
    return tuple(output)


def _local_candidates(
    *,
    residual_index: SemanticResidualIndex,
    dated_question: str,
    result: SourceGroupReinjectionResult,
    owner_by_receipt: Mapping[str, LocalCitationBinding],
    owner_plane_by_receipt: Mapping[str, Plane],
    by_receipt: Mapping[str, tuple[str, ExactCellSegment]],
    by_chunk: Mapping[str, tuple[ExactCellSegment, ...]],
) -> tuple[_Candidate, ...]:
    evidence_by_receipt = {row.receipt_sha256: row for row in result.evidence}
    binding_by_receipt = {row.receipt_sha256: row for row in result.local_bindings}
    duplicate_by_receipt = {
        row.receipt_sha256: row for row in result.protected_duplicates
    }
    output: list[_Candidate] = []
    for attempt in result.attempted_selection:
        segment = _validated_segment(
            by_receipt, attempt.segment_receipt_sha256, cell_id=attempt.cell_id
        )
        _require(segment.source_id == attempt.source_id, "L attempt source changed")
        if attempt.disposition == "protected_exact_duplicate":
            duplicate = duplicate_by_receipt.get(
                attempt.protected_duplicate_receipt_sha256 or ""
            )
            _require(
                duplicate is not None
                and duplicate.segment_receipt_sha256 == segment.receipt_sha256
                and duplicate.span_identity_sha256 == attempt.span_identity_sha256,
                "L duplicate attempt lost its exact audit",
            )
            assert duplicate is not None
            owner = owner_by_receipt.get(duplicate.protected_binding_receipt_sha256)
            owner_plane = owner_plane_by_receipt.get(
                duplicate.protected_binding_receipt_sha256
            )
            _require(owner is not None and owner_plane is not None, "L owner missing")
            assert owner is not None and owner_plane is not None
            owner_segment, owner_quote = _resolve_binding_quote(
                residual_index, owner, by_chunk
            )
            output.append(
                _duplicate_candidate(
                    plane="L",
                    candidate_id=attempt.candidate_id,
                    selected_segment=segment,
                    owner_binding=owner,
                    owner_plane=owner_plane,
                    owner_segment=owner_segment,
                    owner_quote=owner_quote,
                    duplicate_span_identity_sha256=duplicate.span_identity_sha256,
                    upstream_receipt_sha256=duplicate.receipt_sha256,
                    selection_receipt_sha256=attempt.receipt_sha256,
                    disposition="protected_exact_duplicate",
                    source_rank=attempt.selection_rank,
                    dated_question=dated_question,
                    upstream_attempt_receipt_sha256=attempt.receipt_sha256,
                )
            )
            continue
        binding = _binding_for_segment(
            residual_index,
            candidate_id=attempt.candidate_id,
            source_group_handle=attempt.source_group_handle,
            segment=segment,
        )
        _require(
            binding.receipt_sha256 == attempt.citation_binding_receipt_sha256,
            "L citation cannot be reconstructed from the exact segment",
        )
        evidence = evidence_by_receipt.get(attempt.evidence_receipt_sha256 or "")
        if evidence is not None:
            _require(
                evidence.segment_receipt_sha256 == segment.receipt_sha256
                and evidence.citation_binding_receipt_sha256
                == binding.receipt_sha256
                and evidence.quote == segment.quote
                and binding_by_receipt.get(binding.receipt_sha256) == binding,
                "packed L evidence changed exact bytes or ownership",
            )
        output.append(
            _ordinary_candidate(
                plane="L",
                candidate_id=attempt.candidate_id,
                segment=segment,
                binding=binding,
                upstream_receipt_sha256=(
                    evidence.receipt_sha256
                    if evidence is not None
                    else attempt.receipt_sha256
                ),
                selection_receipt_sha256=attempt.receipt_sha256,
                disposition=attempt.disposition,
                upstream_disposition=attempt.disposition,
                source_rank=attempt.selection_rank,
                dated_question=dated_question,
                upstream_attempt_receipt_sha256=attempt.receipt_sha256,
            )
        )
    return tuple(output)


def _validate_global_candidate(
    candidate: GlobalCompletionCandidate,
    segment: ExactCellSegment,
    cell_id: str,
) -> None:
    _require(
        candidate.cell_id == cell_id
        and candidate.segment_receipt_sha256 == segment.receipt_sha256
        and candidate.span_identity_sha256 == _span_identity_sha256(segment)
        and candidate.source_id == segment.source_id
        and candidate.partition_id == segment.partition_id
        and candidate.quote == segment.quote
        and candidate.quote_sha256 == segment.quote_sha256
        and candidate.role == segment.role
        and candidate.created_at == segment.created_at
        and candidate.event_dates
        == linked_event_dates(
            segment.quote,
            segment.created_at,
            segment.event_dates,
        )
        and candidate.action_concepts == linked_action_concepts(segment.quote),
        "hydrated G candidate differs from its exact index segment",
    )


def _global_candidates(
    *,
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    result: SemanticGlobalCompletionResult,
    policy: SemanticGlobalTerminalPolicy,
    owner_by_receipt: Mapping[str, LocalCitationBinding],
    owner_plane_by_receipt: Mapping[str, Plane],
    by_receipt: Mapping[str, tuple[str, ExactCellSegment]],
    by_chunk: Mapping[str, tuple[ExactCellSegment, ...]],
) -> tuple[_Candidate, ...]:
    cell_by_id = {row.cell_id: row for row in residual_index.cells}
    _require(
        len(cell_by_id) == len(residual_index.cells),
        "terminal index repeats G cell IDs",
    )
    seed_by_group: dict[str, list[GlobalCompletionCandidate]] = defaultdict(list)
    for row in result.candidates:
        seed_by_group[row.source_group_handle].append(row)
    existing_segment_receipts = {
        row.segment_receipt_sha256 for row in result.candidates
    }
    closure_candidates: list[GlobalCompletionCandidate] = []
    for source_group_handle, seeds in seed_by_group.items():
        source_ids = {row.source_id for row in seeds}
        partitions = {row.partition_id for row in seeds}
        _require(
            len(source_ids) == 1 and len(partitions) == 1,
            "G source-group seed crossed source or partition identity",
        )
        source_id = next(iter(source_ids))
        partition_id = next(iter(partitions))
        origin_ids = tuple(
            dict.fromkeys(
                cell_id
                for row in seeds
                for cell_id in row.selected_origin_cell_ids
            )
        )
        lane_names = tuple(name for name, _value in seeds[0].lane_scores)
        for cell in residual_index.cells:
            if cell.source_id != source_id:
                continue
            for segment in cell.segments:
                if (
                    segment.partition_id != partition_id
                    or segment.receipt_sha256 in existing_segment_receipts
                ):
                    continue
                candidate_id = identity_sha256(
                    {
                        "format": f"{ROW_FORMAT}-partition-source-group-head-v2",
                        "query_receipt_sha256": query.receipt_sha256,
                        "segment_receipt_sha256": segment.receipt_sha256,
                        "source_history_receipt_sha256": (
                            cell.source_history_receipt_sha256
                        ),
                    }
                )
                closure_candidates.append(
                    GlobalCompletionCandidate(
                        candidate_id=candidate_id,
                        source_group_handle=source_group_handle,
                        source_id=source_id,
                        source_history_receipt_sha256=(
                            cell.source_history_receipt_sha256
                        ),
                        cell_id=cell.cell_id,
                        cell_receipt_sha256=cell.receipt_sha256,
                        selected_origin_cell_ids=origin_ids,
                        segment_receipt_sha256=segment.receipt_sha256,
                        span_identity_sha256=_span_identity_sha256(segment),
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
                        supported_obligation_ids=tuple(
                            row.obligation_id
                            for row in result.obligations
                            if _global_segment_supports(segment, row)
                        ),
                        hydration_routes=(
                            "terminal_partition_source_group_closure",
                        ),
                        lane_scores=tuple((name, 0.0) for name in lane_names),
                    )
                )
                existing_segment_receipts.add(segment.receipt_sha256)
    all_candidates = (*result.candidates, *closure_candidates)
    candidates_by_id = {row.candidate_id: row for row in all_candidates}
    _require(
        len(candidates_by_id) == len(all_candidates),
        "G hydrated/closure candidates repeat IDs",
    )
    attempts_by_id = {row.candidate_id: row for row in result.attempted_selection}
    _require(
        len(attempts_by_id) == len(result.attempted_selection),
        "G attempts repeat candidates",
    )
    evidence_by_receipt = {row.receipt_sha256: row for row in result.evidence}
    binding_by_receipt = {row.receipt_sha256: row for row in result.local_bindings}
    duplicate_by_receipt = {
        row.receipt_sha256: row for row in result.protected_duplicates
    }
    question_actions = set(canonical_action_concepts(query.dated_question))
    obligation_by_id = {row.obligation_id: row for row in result.obligations}
    _require(
        len(obligation_by_id) == len(result.obligations),
        "G obligations repeat identities",
    )
    hydrated_rank_by_id = {
        candidate.candidate_id: rank
        for rank, candidate in enumerate(all_candidates)
    }
    relevance_by_id = {
        candidate.candidate_id: _query_relevance(
            query=query,
            obligations=result.obligations,
            quote=candidate.quote,
            role=candidate.role,
            created_at=candidate.created_at,
            event_dates=candidate.event_dates,
        )
        for candidate in all_candidates
    }

    def supported_kinds(candidate: GlobalCompletionCandidate) -> frozenset[str]:
        return frozenset(
            obligation_by_id[value].kind
            for value in candidate.supported_obligation_ids
        )

    def supported_entity_ids(
        candidate: GlobalCompletionCandidate,
    ) -> frozenset[str]:
        return frozenset(
            value
            for value in candidate.supported_obligation_ids
            if obligation_by_id[value].kind in {"entity", "typed_slot"}
        )

    def upstream_tier(candidate: GlobalCompletionCandidate) -> int:
        attempt = attempts_by_id.get(candidate.candidate_id)
        return {
            "packed_novel": 3,
            "protected_exact_duplicate": 2,
            "budget_unpacked": 1,
        }.get(None if attempt is None else attempt.disposition, 0)

    def candidate_score(candidate: GlobalCompletionCandidate) -> tuple[int, ...]:
        relevance = relevance_by_id[candidate.candidate_id]
        return (
            int(not relevance.explicit_temporal_conflict),
            int(relevance.past_event_witness),
            int(relevance.temporal_support),
            int(relevance.role_support),
            len(
                question_actions
                & set(linked_action_concepts(candidate.quote))
            ),
            len(supported_entity_ids(candidate)),
            len(candidate.supported_obligation_ids),
            upstream_tier(candidate),
            -candidate.token_count,
            -hydrated_rank_by_id[candidate.candidate_id],
        )

    def semantic_lane(candidate: GlobalCompletionCandidate) -> Disposition | None:
        completed = bool(
            question_actions & set(completed_action_concepts(candidate.quote))
        )
        if completed:
            return "completed_event_lane"
        relevance = relevance_by_id[candidate.candidate_id]
        if (
            query.operator_spec.include_proposed
            and question_actions & set(planned_action_concepts(candidate.quote))
            and candidate.role == "user"
            and relevance.temporal_support
        ):
            return "proposed_action_lane"
        return None

    # Treat an opaque partition equality as a linking edge, never as a query
    # identifier.  Each source group contributes its strongest user/event
    # segment; partitions are ranked by how many distinct groups jointly
    # support the question's role/entity/action obligations.  Round-robin over
    # the top clusters prevents a long noisy source from consuming G's fixed
    # 24-item/2.4k budget before a linked neighbour receives one slot.
    grouped: dict[str, list[GlobalCompletionCandidate]] = defaultdict(list)
    for candidate in all_candidates:
        grouped[candidate.source_group_handle].append(candidate)
    group_rows: list[dict[str, Any]] = []
    for source_group_handle, values in grouped.items():
        partitions = {row.partition_id for row in values}
        _require(
            len(partitions) == 1,
            "one G source group crossed opaque episode partitions",
        )
        union_ids = tuple(
            sorted(
                {
                    obligation_id
                    for row in values
                    for obligation_id in row.supported_obligation_ids
                }
            )
        )
        union_kinds = tuple(
            sorted({obligation_by_id[value].kind for value in union_ids})
        )
        entity_ids = {
            value
            for value in union_ids
            if obligation_by_id[value].kind in {"entity", "typed_slot"}
        }
        ordered_values = tuple(
            sorted(
                values,
                key=lambda row: (
                    tuple(-value for value in candidate_score(row)),
                    hydrated_rank_by_id[row.candidate_id],
                    row.receipt_sha256,
                ),
            )
        )
        has_event = any(
            relevance_by_id[row.candidate_id].past_event_witness
            for row in values
        )
        has_role = any(
            relevance_by_id[row.candidate_id].role_support for row in values
        )
        has_temporal = any(
            relevance_by_id[row.candidate_id].temporal_support for row in values
        )
        has_entity = bool(entity_ids)
        has_action = "action" in union_kinds or any(
            question_actions & set(linked_action_concepts(row.quote))
            for row in values
        )
        group_rows.append(
            {
                "best": ordered_values[0],
                "entity_ids": frozenset(entity_ids),
                "has_event": has_event,
                "has_role": has_role,
                "has_temporal": has_temporal,
                "joint": bool(has_role and has_entity and (has_action or has_event)),
                "partition_id": next(iter(partitions)),
                "score": (
                    int(has_entity),
                    int(has_event),
                    int(has_role),
                    int(has_temporal),
                    len(entity_ids),
                    len(union_ids),
                    *candidate_score(ordered_values[0]),
                ),
                "source_group_handle": source_group_handle,
                "supported_ids": union_ids,
                "supported_kinds": union_kinds,
            }
        )
    groups_by_partition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        groups_by_partition[row["partition_id"]].append(row)
    partition_rows: list[dict[str, Any]] = []
    for first_seen, (partition_id, values) in enumerate(groups_by_partition.items()):
        ordered_groups = tuple(
            sorted(
                values,
                key=lambda row: (
                    tuple(-value for value in row["score"]),
                    hydrated_rank_by_id[row["best"].candidate_id],
                    row["best"].receipt_sha256,
                ),
            )
        )
        partition_rows.append(
            {
                "first_seen": first_seen,
                "joint_count": sum(row["joint"] for row in values),
                "partition_id": partition_id,
                "supported_count": sum(
                    bool(row["entity_ids"] and row["has_role"])
                    for row in values
                ),
                "groups": ordered_groups,
                "score": (
                    sum(row["joint"] for row in values),
                    sum(
                        bool(row["entity_ids"] and row["has_role"])
                        for row in values
                    ),
                    sum(row["has_event"] for row in values),
                    len(
                        set().union(*(row["entity_ids"] for row in values))
                    ),
                    len(
                        {
                            value
                            for row in values
                            for value in row["supported_ids"]
                        }
                    ),
                    *ordered_groups[0]["score"],
                ),
            }
        )
    ranked_partitions = tuple(
        sorted(
            partition_rows,
            key=lambda row: (
                tuple(-value for value in row["score"]),
                row["first_seen"],
                identity_sha256({"partition_id": row["partition_id"]}),
            ),
        )
    )
    top_partitions = ranked_partitions[: policy.max_partition_clusters]
    remaining_partitions = ranked_partitions[policy.max_partition_clusters :]

    ordered: list[
        tuple[
            GlobalCompletionCandidate,
            Disposition | None,
            int,
            int,
            tuple[str, ...],
            tuple[str, ...],
            int,
            int,
            ClosureClass,
        ]
    ] = []
    seen_candidates: set[str] = set()

    group_row_by_handle = {
        row["source_group_handle"]: row for row in group_rows
    }
    partition_rank_by_id = {
        row["partition_id"]: rank for rank, row in enumerate(ranked_partitions)
    }
    group_round_by_handle = {
        group["source_group_handle"]: source_group_round
        for partition in ranked_partitions
        for source_group_round, group in enumerate(partition["groups"])
    }

    def append_partition_rounds(
        partitions: Sequence[dict[str, Any]],
        offset: int,
        *,
        start_round: int = 0,
        stop_round: int | None = None,
    ) -> None:
        width = max((len(row["groups"]) for row in partitions), default=0)
        stop = width if stop_round is None else min(width, stop_round)
        for source_group_round in range(start_round, stop):
            for relative_cluster_rank, partition in enumerate(partitions):
                if source_group_round >= len(partition["groups"]):
                    continue
                group = partition["groups"][source_group_round]
                candidate = group["best"]
                if candidate.candidate_id in seen_candidates:
                    continue
                ordered.append(
                    (
                        candidate,
                        semantic_lane(candidate) or "source_group_closure_lane",
                        offset + relative_cluster_rank,
                        source_group_round,
                        group["supported_ids"],
                        group["supported_kinds"],
                        partition["joint_count"],
                        partition["supported_count"],
                        "semantic_source_head",
                    )
                )
                seen_candidates.add(candidate.candidate_id)

    append_partition_rounds(
        top_partitions,
        0,
        stop_round=policy.max_source_group_rounds_per_partition,
    )

    def append_anchor(candidate: GlobalCompletionCandidate) -> None:
        group = group_row_by_handle[candidate.source_group_handle]
        partition_rank = partition_rank_by_id[candidate.partition_id]
        partition = ranked_partitions[partition_rank]
        ordered.append(
            (
                candidate,
                semantic_lane(candidate) or "selected_anchor_closure_lane",
                partition_rank,
                group_round_by_handle[candidate.source_group_handle],
                group["supported_ids"],
                group["supported_kinds"],
                partition["joint_count"],
                partition["supported_count"],
                (
                    "selected_cluster_anchor"
                    if partition_rank < policy.max_partition_clusters
                    else "selected_outside_anchor"
                ),
            )
        )
        seen_candidates.add(candidate.candidate_id)

    def anchor_score(candidate: GlobalCompletionCandidate) -> tuple[int, ...]:
        kinds = supported_kinds(candidate)
        exact_relation = bool(
            {"entity", "typed_slot"} & kinds
            and (
                "action" in kinds
                or question_actions
                & set(linked_action_concepts(candidate.quote))
            )
        )
        return (int(exact_relation), *candidate_score(candidate))

    for partition in top_partitions[: policy.max_anchor_partition_clusters]:
        anchor_population = tuple(
            candidate
            for candidate in all_candidates
            if candidate.partition_id == partition["partition_id"]
            and candidate.candidate_id in attempts_by_id
            and candidate.candidate_id not in seen_candidates
        )
        if anchor_population:
            append_anchor(
                max(
                    anchor_population,
                    key=lambda row: (
                        anchor_score(row),
                        -hydrated_rank_by_id[row.candidate_id],
                        row.receipt_sha256,
                    ),
                )
            )

    outside_top = {
        row["partition_id"] for row in remaining_partitions
    }
    escape_population = tuple(
        candidate
        for candidate in all_candidates
        if candidate.partition_id in outside_top
        and candidate.candidate_id in attempts_by_id
        and candidate.candidate_id not in seen_candidates
    )
    if escape_population and policy.max_outside_cluster_anchors:
        append_anchor(
            max(
                escape_population,
                key=lambda row: (
                    anchor_score(row),
                    -hydrated_rank_by_id[row.candidate_id],
                    row.receipt_sha256,
                ),
            )
        )

    append_partition_rounds(
        top_partitions,
        0,
        start_round=policy.max_source_group_rounds_per_partition,
    )
    append_partition_rounds(remaining_partitions, len(top_partitions))

    supplemental_unranked: list[
        tuple[int, GlobalCompletionCandidate, Disposition, tuple[int, ...]]
    ] = []
    for hydrated_rank, candidate in enumerate(all_candidates):
        completed = tuple(
            sorted(
                question_actions & set(completed_action_concepts(candidate.quote))
            )
        )
        planned_match = bool(
            question_actions & set(planned_action_concepts(candidate.quote))
        )
        relevance = relevance_by_id[candidate.candidate_id]
        lane: Disposition | None = None
        if completed:
            lane = "completed_event_lane"
        elif (
            query.operator_spec.include_proposed
            and planned_match
            and candidate.role == "user"
            and relevance.temporal_support
        ):
            lane = "proposed_action_lane"
        if lane is not None:
            supplemental_unranked.append(
                (hydrated_rank, candidate, lane, candidate_score(candidate))
            )
    supplemental = tuple(
        (candidate, lane)
        for _rank, candidate, lane, _score in sorted(
            supplemental_unranked,
            key=lambda row: (
                tuple(-value for value in row[3]),
                row[0],
                row[1].receipt_sha256,
            ),
        )[: policy.max_completed_event_lane_items]
    )
    for candidate, lane in supplemental:
        if candidate.candidate_id in seen_candidates:
            continue
        ordered.append((candidate, lane, -1, -1, (), (), 0, 0, "none"))
        seen_candidates.add(candidate.candidate_id)
    for attempt in result.attempted_selection:
        candidate = candidates_by_id.get(attempt.candidate_id)
        _require(
            candidate is not None
            and candidate.receipt_sha256 == attempt.candidate_receipt_sha256,
            "G attempt lost its hydrated candidate",
        )
        assert candidate is not None
        if candidate.candidate_id not in seen_candidates:
            ordered.append((candidate, None, -1, -1, (), (), 0, 0, "none"))
            seen_candidates.add(candidate.candidate_id)

    output: list[_Candidate] = []
    for fallback_rank, (
        candidate,
        lane,
        partition_cluster_rank,
        source_group_round,
        source_group_supported_ids,
        source_group_supported_kinds,
        partition_joint_count,
        partition_supported_count,
        closure_class,
    ) in enumerate(ordered):
        location = by_receipt.get(candidate.segment_receipt_sha256)
        _require(location is not None, "G candidate segment is absent")
        assert location is not None
        cell_id, segment = location
        _validate_global_candidate(candidate, segment, cell_id)
        attempt = attempts_by_id.get(candidate.candidate_id)
        upstream_disposition = (
            attempt.disposition
            if attempt is not None
            else "hydrated_not_upstream_selected"
        )
        selection_receipt = (
            identity_sha256(
                {
                    "candidate_receipt_sha256": candidate.receipt_sha256,
                    "format": f"{SELECTION_FORMAT}-supplemental-g-v1",
                    "lane": lane,
                    "partition_cluster_receipt_sha256": (
                        identity_sha256({"partition_id": candidate.partition_id})
                        if partition_cluster_rank >= 0
                        else None
                    ),
                    "partition_cluster_rank": partition_cluster_rank,
                    "query_receipt_sha256": query.receipt_sha256,
                    "source_group_round": source_group_round,
                    "source_group_supported_obligation_ids": list(
                        source_group_supported_ids
                    ),
                }
            )
            if lane is not None
            else attempt.receipt_sha256  # type: ignore[union-attr]
        )
        rank = attempt.selection_rank if attempt is not None else fallback_rank
        disposition: Disposition = (
            lane if lane is not None else attempt.disposition  # type: ignore[assignment,union-attr]
        )
        relevance = relevance_by_id[candidate.candidate_id]
        candidate_kinds = supported_kinds(candidate)
        exact_relation_support = bool(
            {"entity", "typed_slot"} & candidate_kinds
            and (
                "action" in candidate_kinds
                or question_actions
                & set(linked_action_concepts(candidate.quote))
            )
        )
        if attempt is not None and attempt.disposition == "protected_exact_duplicate":
            duplicate = duplicate_by_receipt.get(
                attempt.protected_duplicate_receipt_sha256 or ""
            )
            _require(
                duplicate is not None
                and duplicate.candidate_id == candidate.candidate_id
                and duplicate.segment_receipt_sha256 == segment.receipt_sha256
                and duplicate.span_identity_sha256
                == candidate.span_identity_sha256,
                "G duplicate attempt lost its exact audit",
            )
            assert duplicate is not None
            owner = owner_by_receipt.get(duplicate.protected_binding_receipt_sha256)
            owner_plane = owner_plane_by_receipt.get(
                duplicate.protected_binding_receipt_sha256
            )
            _require(owner is not None and owner_plane is not None, "G owner missing")
            assert owner is not None and owner_plane is not None
            owner_segment, owner_quote = _resolve_binding_quote(
                residual_index, owner, by_chunk
            )
            output.append(
                _duplicate_candidate(
                    plane="G",
                    candidate_id=candidate.candidate_id,
                    selected_segment=segment,
                    owner_binding=owner,
                    owner_plane=owner_plane,
                    owner_segment=owner_segment,
                    owner_quote=owner_quote,
                    duplicate_span_identity_sha256=duplicate.span_identity_sha256,
                    upstream_receipt_sha256=duplicate.receipt_sha256,
                    selection_receipt_sha256=selection_receipt,
                    disposition=disposition,
                    source_rank=rank,
                    dated_question=query.dated_question,
                    upstream_attempt_receipt_sha256=(
                        None if attempt is None else attempt.receipt_sha256
                    ),
                    event_dates=linked_event_dates(
                        owner_quote,
                        owner_segment.created_at,
                        owner_segment.event_dates,
                    ),
                    action_concepts=linked_action_concepts(owner_quote),
                    supported_obligation_ids=candidate.supported_obligation_ids,
                    source_group_supported_obligation_ids=(
                        source_group_supported_ids
                    ),
                    source_group_supported_kinds=source_group_supported_kinds,
                    query_relevance=_query_relevance(
                        query=query,
                        obligations=result.obligations,
                        quote=owner_quote,
                        role=owner_segment.role,
                        created_at=owner_segment.created_at,
                        event_dates=linked_event_dates(
                            owner_quote,
                            owner_segment.created_at,
                            owner_segment.event_dates,
                        ),
                    ),
                    partition_cluster_rank=partition_cluster_rank,
                    source_group_round=source_group_round,
                    partition_joint_source_group_count=partition_joint_count,
                    partition_supported_source_group_count=(
                        partition_supported_count
                    ),
                    exact_relation_support=exact_relation_support,
                    closure_class=closure_class,
                )
            )
            continue
        binding = _binding_for_segment(
            residual_index,
            candidate_id=candidate.candidate_id,
            source_group_handle=candidate.source_group_handle,
            segment=segment,
        )
        if attempt is not None:
            _require(
                attempt.disposition != "protected_exact_duplicate"
                and binding.receipt_sha256
                == attempt.citation_binding_receipt_sha256,
                "G citation cannot be reconstructed from hydrated bytes",
            )
            evidence = evidence_by_receipt.get(
                attempt.evidence_receipt_sha256 or ""
            )
            if evidence is not None:
                _require(
                    evidence.segment_receipt_sha256 == segment.receipt_sha256
                    and evidence.citation_binding_receipt_sha256
                    == binding.receipt_sha256
                    and evidence.quote == segment.quote
                    and binding_by_receipt.get(binding.receipt_sha256) == binding,
                    "packed G evidence changed exact bytes or ownership",
                )
        output.append(
            _ordinary_candidate(
                plane="G",
                candidate_id=candidate.candidate_id,
                segment=segment,
                binding=binding,
                upstream_receipt_sha256=(
                    candidate.receipt_sha256
                    if attempt is None
                    else attempt.evidence_receipt_sha256 or attempt.receipt_sha256
                ),
                selection_receipt_sha256=selection_receipt,
                disposition=disposition,
                upstream_disposition=upstream_disposition,
                source_rank=rank,
                dated_question=query.dated_question,
                upstream_attempt_receipt_sha256=(
                    None if attempt is None else attempt.receipt_sha256
                ),
                event_dates=candidate.event_dates,
                action_concepts=candidate.action_concepts,
                supported_obligation_ids=candidate.supported_obligation_ids,
                source_group_supported_obligation_ids=(
                    source_group_supported_ids
                ),
                source_group_supported_kinds=source_group_supported_kinds,
                query_relevance=relevance,
                partition_cluster_rank=partition_cluster_rank,
                source_group_round=source_group_round,
                partition_joint_source_group_count=partition_joint_count,
                partition_supported_source_group_count=partition_supported_count,
                exact_relation_support=exact_relation_support,
                closure_class=closure_class,
            )
        )
    return tuple(output)


_COUNTED_SUBJECT_RE = re.compile(
    r"\bhow\s+many\s+(?P<subject>[A-Za-z0-9'’-]+(?:\s+[A-Za-z0-9'’-]+){0,5}?)"
    r"\s+(?:did|do|does|have|has|had|am|is|are|was|were|will|would|"
    r"can|could|should)\b",
    re.IGNORECASE,
)
_COUNTED_SUBJECT_MODIFIERS = frozenset({"different", "distinct", "total"})
_COUNTED_SUBJECT_GENERIC_HEADS = frozenset(
    {"piece", "item", "kind", "type", "number", "amount", "total"}
)


def _counted_subject_obligation_ids(
    *,
    dated_question: str,
    spec: TypedOperatorSpec,
    obligations: Sequence[GlobalEvidenceObligation],
) -> tuple[str, ...]:
    """Map a numeric question's counted noun phrase to direct entity duties."""

    if not (
        spec.operation == "count_or_aggregate"
        and spec.answer_shape is AnswerShape.NUMBER
        and spec.requires_all_slots
        and spec.requires_complete_frontier
    ):
        return ()
    match = _COUNTED_SUBJECT_RE.search(dated_question)
    if match is None:
        return ()
    subject = match.group("subject")
    subject_terms = set(normalized_terms(subject)) - _COUNTED_SUBJECT_MODIFIERS
    if re.search(r"\bof\b", subject, re.IGNORECASE):
        subject_terms -= _COUNTED_SUBJECT_GENERIC_HEADS
    if not subject_terms:
        return ()
    return tuple(
        obligation.obligation_id
        for obligation in obligations
        if obligation.kind == "entity"
        and bool(
            subject_terms
            & (
                set(obligation.match_terms)
                | set(normalized_terms(obligation.label))
            )
        )
    )


def _direct_operand_priority(
    row: _Candidate,
    *,
    obligations_by_id: Mapping[str, GlobalEvidenceObligation],
    counted_obligation_ids: frozenset[str],
) -> tuple[int, ...]:
    closure_tier = {
        "semantic_source_head": 3,
        "selected_cluster_anchor": 2,
        "selected_outside_anchor": 1,
        "none": 0,
    }[row.closure_class]
    direct_ids = set(row.supported_obligation_ids)
    upstream_tier = {
        "packed_novel": 3,
        "protected_exact_duplicate": 2,
        "budget_unpacked": 1,
    }.get(row.upstream_disposition, 0)
    return (
        closure_tier,
        -row.partition_cluster_rank if row.partition_cluster_rank >= 0 else -10_000,
        -row.source_group_round if row.source_group_round >= 0 else -10_000,
        int(row.query_temporal_support),
        int(bool(row.matched_completed_actions)),
        int(row.past_event_witness),
        len(row.matched_query_actions),
        sum(
            obligations_by_id[obligation_id].kind == "action"
            for obligation_id in direct_ids
            if obligation_id in obligations_by_id
        ),
        len(direct_ids & counted_obligation_ids),
        len(direct_ids),
        upstream_tier,
        -count_tokens(row.selection_quote),
        -row.source_rank,
    )


def _direct_operand_lane(
    candidates: Sequence[_Candidate],
    *,
    dated_question: str,
    spec: TypedOperatorSpec,
    obligations: Sequence[GlobalEvidenceObligation],
    max_items: int,
) -> tuple[tuple[_Candidate, ...], tuple[_Candidate, ...]]:
    """Return the eligible direct population and its bounded reservations."""

    _require(
        type(max_items) is int and max_items == MAX_DIRECT_OPERAND_LANE_ITEMS,
        "direct operand lane cap changed",
    )
    counted_ids = frozenset(
        _counted_subject_obligation_ids(
            dated_question=dated_question,
            spec=spec,
            obligations=obligations,
        )
    )
    if not counted_ids:
        return (), ()
    obligations_by_id = {row.obligation_id: row for row in obligations}
    has_date_obligations = any(row.kind == "date" for row in obligations)
    indexed = tuple(
        (population_index, row)
        for population_index, row in enumerate(candidates)
        if row.plane == "G"
        and row.role == "user"
        and row.exact_relation_support
        and not row.explicit_temporal_conflict
        and (not has_date_obligations or row.query_temporal_support)
        and bool(counted_ids & set(row.supported_obligation_ids))
        and bool(
            row.matched_completed_actions
            or spec.include_proposed
            and row.matched_planned_actions
        )
    )
    ordered = tuple(
        row
        for _index, row in sorted(
            indexed,
            key=lambda value: (
                tuple(
                    -component
                    for component in _direct_operand_priority(
                        value[1],
                        obligations_by_id=obligations_by_id,
                        counted_obligation_ids=counted_ids,
                    )
                ),
                value[0],
                value[1].receipt_sha256,
            ),
        )
    )
    return ordered, ordered[:max_items]


def _consideration_priority(
    row: _Candidate,
    *,
    population_index: int,
) -> tuple[int, ...]:
    tier = {
        "packed_novel": 3,
        "protected_exact_duplicate": 2,
        "budget_unpacked": 1,
    }.get(row.upstream_disposition, 0)
    if row.plane == "P":
        values = (-population_index,)
    elif row.plane == "R":
        values = (
            int(not row.explicit_temporal_conflict),
            int(row.past_event_witness),
            int(row.query_temporal_support),
            int(row.role == "user"),
            int(bool(row.matched_completed_actions)),
            int(bool(row.matched_planned_actions)),
            len(row.matched_query_actions),
            tier,
            -count_tokens(row.selection_quote),
            -row.source_rank,
        )
    elif row.plane == "L":
        values = (
            tier,
            int(bool(row.matched_completed_actions)),
            int(bool(row.matched_planned_actions)),
            int(row.role == "user"),
            int(bool(row.event_dates)),
            len(row.matched_query_actions),
            -row.source_rank,
        )
    else:
        top_cluster_head = bool(
            row.closure_class == "semantic_source_head"
            and 0 <= row.partition_cluster_rank < 4
            and 0 <= row.source_group_round < 5
        )
        top_anchor = bool(
            row.closure_class == "selected_cluster_anchor"
            and 0 <= row.partition_cluster_rank < 3
        )
        escape_anchor = bool(
            row.closure_class == "selected_outside_anchor"
            and row.partition_cluster_rank >= 4
        )
        lane_tier = (
            4
            if top_cluster_head
            else 3
            if top_anchor
            else 2
            if escape_anchor
            else 1
        )
        values = (
            lane_tier,
            -row.source_group_round if top_cluster_head else -10_000,
            -row.partition_cluster_rank if top_cluster_head else -10_000,
            row.partition_joint_source_group_count,
            row.partition_supported_source_group_count,
            int(not row.explicit_temporal_conflict),
            int(row.past_event_witness),
            int(row.query_temporal_support),
            int(row.role == "user"),
            int(
                bool(
                    {"entity", "typed_slot"}
                    & set(row.source_group_supported_kinds)
                )
            ),
            int("action" in row.source_group_supported_kinds),
            int("role" in row.source_group_supported_kinds),
            len(row.source_group_supported_obligation_ids),
            len(row.matched_query_actions),
            tier,
            -row.source_rank,
        )
    return (*values, *((0,) * (CONSIDERATION_PRIORITY_WIDTH - len(values))))


def _ordered_plane_consideration(
    candidates: Sequence[_Candidate],
    budget: PlaneBudget,
) -> tuple[tuple[_Candidate, ...], tuple[tuple[int, ...], ...]]:
    indexed = tuple(enumerate(candidates))
    if budget.plane in {"L", "G"}:
        indexed = tuple(
            sorted(
                indexed,
                key=lambda value: (
                    tuple(
                        -component
                        for component in _consideration_priority(
                            value[1], population_index=value[0]
                        )
                    ),
                    value[0],
                    value[1].receipt_sha256,
                ),
            )
        )
    elif budget.plane == "R":
        grouped: dict[str, list[tuple[int, _Candidate]]] = defaultdict(list)
        for indexed_row in indexed:
            grouped[indexed_row[1].binding.source_id].append(indexed_row)
        for values in grouped.values():
            values.sort(
                key=lambda value: (
                    tuple(
                        -component
                        for component in _consideration_priority(
                            value[1], population_index=value[0]
                        )
                    ),
                    value[0],
                    value[1].receipt_sha256,
                )
            )
        ordered_groups = sorted(
            grouped.values(),
            key=lambda values: (
                tuple(
                    -component
                    for component in _consideration_priority(
                        values[0][1], population_index=values[0][0]
                    )
                ),
                values[0][0],
                values[0][1].receipt_sha256,
            ),
        )
        fair: list[tuple[int, _Candidate]] = []
        for round_index in range(max(map(len, ordered_groups), default=0)):
            fair.extend(
                values[round_index]
                for values in ordered_groups
                if round_index < len(values)
            )
        indexed = tuple(fair)
    ordered = tuple(row for _index, row in indexed)
    priorities = tuple(
        _consideration_priority(row, population_index=index)
        for index, row in indexed
    )
    return ordered, priorities


def _select_plane(
    candidates: Sequence[_Candidate],
    budget: PlaneBudget,
    *,
    direct_operand_population: Sequence[_Candidate] = (),
    direct_operand_reserved: Sequence[_Candidate] = (),
    include_proposed: bool = False,
    has_date_obligations: bool = False,
) -> tuple[tuple[_Candidate, ...], PlaneSelectionReceipt]:
    _require(
        all(type(row) is _Candidate and row.plane == budget.plane for row in candidates),
        "plane candidate inventory changed",
    )
    candidate_receipts = tuple(row.receipt_sha256 for row in candidates)
    _ordered_unique(candidate_receipts, f"{budget.plane} candidate receipts")
    direct_population = tuple(direct_operand_population)
    direct_reserved = tuple(direct_operand_reserved)
    direct_population_receipts = tuple(
        row.receipt_sha256 for row in direct_population
    )
    direct_reserved_receipts = tuple(row.receipt_sha256 for row in direct_reserved)
    _require(
        type(include_proposed) is bool
        and type(has_date_obligations) is bool
        and all(row in candidates for row in direct_population)
        and direct_reserved == direct_population[: len(direct_reserved)]
        and len(direct_reserved) <= MAX_DIRECT_OPERAND_LANE_ITEMS
        and (budget.plane == "G" or not direct_population and not direct_reserved),
        "direct operand plane inputs changed",
    )
    _ordered_unique(direct_population_receipts, "direct operand population")
    _ordered_unique(direct_reserved_receipts, "direct operand reservations")
    legacy_consideration, legacy_priorities = _ordered_plane_consideration(
        candidates, budget
    )

    def pack(rows: Sequence[_Candidate]) -> tuple[tuple[_Candidate, ...], int]:
        packed: list[_Candidate] = []
        packed_tokens = 0
        for candidate in rows:
            candidate_tokens = count_tokens(candidate.selection_quote)
            if (
                len(packed) >= budget.max_items
                or packed_tokens + candidate_tokens > budget.evidence_token_cap
            ):
                continue
            packed.append(candidate)
            packed_tokens += candidate_tokens
        return tuple(packed), packed_tokens

    base_status_refill: tuple[_Candidate, ...] = ()
    if direct_reserved:
        base_selected, _base_tokens = pack(legacy_consideration)
        direct_receipts = set(direct_reserved_receipts)
        base_status_refill = tuple(
            row
            for row in base_selected
            if row.receipt_sha256 not in direct_receipts
            and row.role == "user"
            and not row.explicit_temporal_conflict
            and (not has_date_obligations or row.query_temporal_support)
            and bool(
                row.matched_completed_actions
                or include_proposed
                and row.matched_planned_actions
            )
        )
        staged: list[_Candidate] = []
        seen: set[str] = set()
        for row in (*direct_reserved, *base_status_refill, *legacy_consideration):
            if row.receipt_sha256 in seen:
                continue
            staged.append(row)
            seen.add(row.receipt_sha256)
        consideration = tuple(staged)
        legacy_priority_by_receipt = {
            row.receipt_sha256: priority
            for row, priority in zip(
                legacy_consideration, legacy_priorities, strict=True
            )
        }
        direct_rank_by_receipt = {
            row.receipt_sha256: rank for rank, row in enumerate(direct_reserved)
        }
        base_rank_by_receipt = {
            row.receipt_sha256: rank for rank, row in enumerate(base_status_refill)
        }
        consideration_priorities = tuple(
            (
                (6, -direct_rank_by_receipt[row.receipt_sha256])
                + (0,) * (CONSIDERATION_PRIORITY_WIDTH - 2)
                if row.receipt_sha256 in direct_rank_by_receipt
                else (5, -base_rank_by_receipt[row.receipt_sha256])
                + (0,) * (CONSIDERATION_PRIORITY_WIDTH - 2)
                if row.receipt_sha256 in base_rank_by_receipt
                else legacy_priority_by_receipt[row.receipt_sha256]
            )
            for row in consideration
        )
    else:
        consideration = legacy_consideration
        consideration_priorities = legacy_priorities

    selected_tuple, token_count = pack(consideration)
    selected = list(selected_tuple)
    _require(
        not candidates or len(selected) >= budget.minimum_items,
        f"{budget.plane} minimum cannot fit its non-borrowable budget",
    )
    selected_receipts = {row.receipt_sha256 for row in selected}
    ordered_skipped = tuple(
        row.receipt_sha256
        for row in candidates
        if row.receipt_sha256 not in selected_receipts
    )
    receipt = PlaneSelectionReceipt(
        plane=budget.plane,
        candidate_receipt_sha256s=candidate_receipts,
        consideration_policy_id=CONSIDERATION_POLICY_BY_PLANE[budget.plane],
        consideration_candidate_receipt_sha256s=tuple(
            row.receipt_sha256 for row in consideration
        ),
        consideration_priority_vectors=consideration_priorities,
        upstream_attempt_receipt_sha256s=tuple(
            row.upstream_attempt_receipt_sha256
            for row in candidates
            if row.upstream_attempt_receipt_sha256 is not None
        ),
        selected_candidate_receipt_sha256s=tuple(
            row.receipt_sha256 for row in selected
        ),
        skipped_candidate_receipt_sha256s=ordered_skipped,
        selected_evidence_tokens=token_count,
        evidence_token_cap=budget.evidence_token_cap,
        max_items=budget.max_items,
        minimum_items=budget.minimum_items,
        upstream_budget_unpacked_selected=sum(
            row.upstream_disposition == "budget_unpacked" for row in selected
        ),
        completed_event_lane_selected=sum(
            row.disposition == "completed_event_lane" for row in selected
        ),
        proposed_action_lane_selected=sum(
            row.disposition == "proposed_action_lane" for row in selected
        ),
        source_group_closure_lane_selected=sum(
            row.closure_class == "semantic_source_head" for row in selected
        ),
        selected_anchor_closure_lane_selected=sum(
            row.closure_class
            in {"selected_cluster_anchor", "selected_outside_anchor"}
            for row in selected
        ),
        direct_operand_population_candidate_receipt_sha256s=(
            direct_population_receipts
        ),
        direct_operand_reserved_candidate_receipt_sha256s=(
            direct_reserved_receipts
        ),
        base_status_refill_candidate_receipt_sha256s=tuple(
            row.receipt_sha256 for row in base_status_refill
        ),
        direct_operand_lane_selected=sum(
            row.receipt_sha256 in set(direct_reserved_receipts)
            for row in selected
        ),
    )
    return tuple(selected), receipt


def _post_selection_dedup(
    selected_by_plane: Mapping[Plane, tuple[_Candidate, ...]],
    *,
    by_receipt: Mapping[str, tuple[str, ExactCellSegment]],
) -> tuple[tuple[_Candidate, ...], DeduplicationReceipt]:
    before = tuple(
        row
        for plane in PLANE_ORDER
        for row in selected_by_plane.get(plane, ())
    )
    seen_spans: dict[str, _Candidate] = {}
    retained: list[_Candidate] = []
    exclusions: list[Mapping[str, Any]] = []
    substitutions: list[Mapping[str, Any]] = []
    retention_authority_transfers: list[Mapping[str, Any]] = []
    protected_g_span_sha256s = {
        identity_sha256(row.binding.span.identity_payload())
        for row in before
        if _hard_protected_global_core(row)
    }
    _require(
        len(protected_g_span_sha256s) <= 12,
        "selected G hard-protection authority exceeded its bounded 12-span core",
    )
    for row in before:
        span_sha = identity_sha256(row.binding.span.identity_payload())
        if row.upstream_disposition == "protected_exact_duplicate":
            selected_segment = _validated_segment(
                by_receipt, row.segment_receipt_sha256
            )
            substitutions.append(
                MappingProxyType(
                    {
                        "candidate_receipt_sha256": row.receipt_sha256,
                        "containment_proven": True,
                        "exact_owner_quote_sha256": quote_sha256(row.quote),
                        "owner_binding_receipt_sha256": (
                            row.duplicate_owner_binding_receipt_sha256
                        ),
                        "owner_source_plane": row.owner_source_plane,
                        "selected_segment_receipt_sha256": (
                            selected_segment.receipt_sha256
                        ),
                        "selected_span_identity_sha256": (
                            row.duplicate_span_identity_sha256
                        ),
                    }
                )
            )
        prior = seen_spans.get(span_sha)
        if prior is not None:
            exclusions.append(
                MappingProxyType(
                    {
                        "dropped_candidate_receipt_sha256": row.receipt_sha256,
                        "exact_span_identity_sha256": span_sha,
                        "kept_candidate_receipt_sha256": prior.receipt_sha256,
                        "policy": "exact_span_after_independent_plane_selection",
                    }
                )
            )
            retention_authority_transfers.append(
                MappingProxyType(
                    {
                        "authority_candidate_receipt_sha256": row.receipt_sha256,
                        "authority_source_plane": row.plane,
                        "exact_span_identity_sha256": span_sha,
                        "hard_protected": _has_hard_retention_authority(row),
                        "kept_candidate_receipt_sha256": prior.receipt_sha256,
                        "policy": RETENTION_AUTHORITY_TRANSFER_POLICY,
                        "retention_priority": _local_priority(row),
                    }
                )
            )
            continue
        seen_spans[span_sha] = row
        retained.append(row)
    receipt = DeduplicationReceipt(
        selected_before_dedup_receipt_sha256s=tuple(
            row.receipt_sha256 for row in before
        ),
        retained_after_dedup_receipt_sha256s=tuple(
            row.receipt_sha256 for row in retained
        ),
        exclusions=tuple(exclusions),
        substitutions=tuple(substitutions),
        retention_authority_transfers=tuple(retention_authority_transfers),
        substitution_count=len(substitutions),
    )
    return tuple(retained), receipt


def _typed_kind(
    spec: TypedOperatorSpec,
    quote: str,
    dated_question: str,
) -> tuple[str, object | None]:
    mention = single_numeric_mention(
        quote,
        operator_spec=spec,
        question=dated_question,
    )
    if mention is not None:
        kind = TypedItemKind.OPERAND.value
    elif spec.temporal_mode.value != "none":
        kind = TypedItemKind.EVENT.value
    elif spec.answer_shape is AnswerShape.SET_LIST:
        kind = TypedItemKind.MEMBER.value
    elif spec.answer_shape is AnswerShape.SYNTHESIS:
        kind = TypedItemKind.CLAIM.value
    elif spec.style.value == "state_chain":
        kind = TypedItemKind.STATE.value
    else:
        kind = TypedItemKind.DIRECT.value
    return kind, mention


def _local_priority(row: _Candidate) -> tuple[int, ...]:
    closure_class = (
        4
        if (
            (
                row.closure_class == "selected_cluster_anchor"
                and 0 <= row.partition_cluster_rank < 3
            )
            or (
                row.closure_class == "selected_outside_anchor"
                and row.partition_cluster_rank >= 4
                and row.exact_relation_support
                and row.query_temporal_support
            )
        )
        else 3
        if row.closure_class == "semantic_source_head"
        and row.partition_cluster_rank == 0
        and 0 <= row.source_group_round < 5
        else 2
        if row.closure_class == "semantic_source_head"
        and 0 <= row.partition_cluster_rank < 4
        and row.source_group_round == 0
        else 1
        if row.closure_class != "none"
        else 0
    )
    values = (
        closure_class,
        int(not row.explicit_temporal_conflict),
        int(row.query_temporal_support),
        int(row.past_event_witness),
        int(row.role == "user"),
        int(
            bool(
                {"entity", "typed_slot"}
                & set(row.source_group_supported_kinds)
            )
        ),
        int("action" in row.source_group_supported_kinds),
        int("role" in row.source_group_supported_kinds),
        len(row.source_group_supported_obligation_ids),
        row.partition_joint_source_group_count,
        row.partition_supported_source_group_count,
        -row.source_group_round,
        -row.partition_cluster_rank,
        int(bool(row.matched_completed_actions)),
        int(row.disposition == "completed_event_lane"),
        int(row.disposition == "proposed_action_lane"),
        len(row.matched_completed_actions),
        int(row.plane == "P"),
        int(row.plane == "R"),
        int(row.plane == "L"),
        int(row.plane == "G"),
        int(row.upstream_disposition == "packed_novel"),
        int(row.upstream_disposition == "budget_unpacked"),
        -row.source_rank,
    )
    return (*values, *((0,) * (LOCAL_RETENTION_PRIORITY_WIDTH - len(values))))


def _hard_protected_global_core(row: _Candidate) -> bool:
    """Return whether one selected G row is non-evictable at final fit.

    The plane still contributes its complete independently selected 24-row
    candidate set.  Hard protection is deliberately narrower so a dense G
    neighborhood cannot consume the final 8k envelope before the cumulative
    L plane.  The protected core mirrors the authenticated 20+3+1 strata:
    deep heads from the strongest partition, one diversity head from every
    other top-four partition, the three selected cluster anchors, and the
    qualified exact-relation escape anchor.
    """

    if row.plane != "G":
        return False
    if (
        row.closure_class == "selected_cluster_anchor"
        and 0 <= row.partition_cluster_rank < 3
    ):
        return True
    if (
        row.closure_class == "selected_outside_anchor"
        and row.partition_cluster_rank >= 4
        and row.exact_relation_support
        and row.query_temporal_support
    ):
        return True
    return bool(
        row.closure_class == "semantic_source_head"
        and (
            (
                row.partition_cluster_rank == 0
                and 0 <= row.source_group_round < 5
            )
            or (
                0 <= row.partition_cluster_rank < 4
                and row.source_group_round == 0
            )
        )
    )


def _has_hard_retention_authority(row: _Candidate) -> bool:
    return bool(
        (
            row.plane == "L"
            and row.upstream_disposition == "packed_novel"
        )
        or _hard_protected_global_core(row)
    )


def _support_aware_retention_priority(
    legacy_priority: tuple[int, ...],
    support: ExactSpanSupportAuthority,
) -> tuple[int, ...]:
    """Prepend aggregate support while preserving the common width of 24."""

    _require(
        type(legacy_priority) is tuple
        and len(legacy_priority) == LOCAL_RETENTION_PRIORITY_WIDTH
        and all(type(value) is int for value in legacy_priority),
        "legacy terminal retention priority changed width",
    )
    plane_rank = (
        (4 * legacy_priority[17])
        + (3 * legacy_priority[18])
        + (2 * legacy_priority[19])
        + legacy_priority[20]
    )
    priority = (
        *support.priority_prefix,
        legacy_priority[0],
        legacy_priority[1],
        legacy_priority[9],
        legacy_priority[10],
        legacy_priority[11],
        legacy_priority[12],
        legacy_priority[13],
        legacy_priority[14],
        legacy_priority[15],
        legacy_priority[16],
        plane_rank,
        legacy_priority[21],
        legacy_priority[22],
        legacy_priority[23],
    )
    _require(
        len(priority) == LOCAL_RETENTION_PRIORITY_WIDTH
        and all(type(value) is int for value in priority),
        "exact-span support changed the common retention-priority width",
    )
    return priority


def _retention_authority_overlay(
    *,
    rows: Sequence[_Candidate],
    dedup_receipt: DeduplicationReceipt | None,
    exact_span_support_population: ExactSpanSupportPopulationReceipt | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Bind dropped exact-span authority to the provider-first retained row.

    The retained candidate is never rewritten: its plane, mechanism, citation,
    binding, and provider provenance remain byte-identical.  Only local final-fit
    priority/protection authority is inherited from independently selected exact
    duplicates, and that overlay is authenticated by ``DeduplicationReceipt``.
    """

    retained_receipts = tuple(row.receipt_sha256 for row in rows)
    if exact_span_support_population is None:
        exact_span_support_population = _exact_span_support_population(
            candidates_by_plane={
                plane: tuple(row for row in rows if row.plane == plane)
                for plane in PLANE_ORDER
            }
        )
    _require(
        type(exact_span_support_population)
        is ExactSpanSupportPopulationReceipt
        and set(retained_receipts)
        <= set(exact_span_support_population.candidate_receipt_sha256s),
        "terminal final fit received rows outside exact-span support population",
    )
    support_by_span = exact_span_support_population.authority_by_span
    if dedup_receipt is not None:
        _require(
            type(dedup_receipt) is DeduplicationReceipt
            and retained_receipts
            == dedup_receipt.retained_after_dedup_receipt_sha256s,
            "terminal final fit received rows outside its authenticated dedup population",
        )
    transfers_by_kept: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if dedup_receipt is not None:
        for transfer in dedup_receipt.retention_authority_transfers:
            transfers_by_kept[transfer["kept_candidate_receipt_sha256"]].append(
                transfer
            )

    overlay: dict[str, Mapping[str, Any]] = {}
    protected_g_spans: set[str] = set()
    for row in rows:
        span_sha = identity_sha256(row.binding.span.identity_payload())
        inherited = tuple(transfers_by_kept.get(row.receipt_sha256, ()))
        _require(
            all(
                transfer["exact_span_identity_sha256"] == span_sha
                for transfer in inherited
            ),
            "dedup retention authority was not bound to the retained exact span",
        )
        support = support_by_span.get(span_sha)
        _require(
            support is not None
            and row.receipt_sha256
            in support.authority_candidate_receipt_sha256s,
            "retained row lacks authenticated exact-span support authority",
        )
        own_priority = _local_priority(row)
        legacy_effective_priority = max(
            (own_priority, *(transfer["retention_priority"] for transfer in inherited))
        )
        effective_priority = _support_aware_retention_priority(
            legacy_effective_priority,
            support,
        )
        own_hard = _has_hard_retention_authority(row)
        inherited_hard = any(transfer["hard_protected"] for transfer in inherited)
        if row.plane == "G" and _hard_protected_global_core(row):
            protected_g_spans.add(span_sha)
        protected_g_spans.update(
            transfer["exact_span_identity_sha256"]
            for transfer in inherited
            if transfer["authority_source_plane"] == "G"
            and transfer["hard_protected"]
        )
        overlay[row.receipt_sha256] = MappingProxyType(
            {
                "authority_source_planes": tuple(
                    transfer["authority_source_plane"] for transfer in inherited
                ),
                "authority_source_receipt_sha256s": tuple(
                    transfer["authority_candidate_receipt_sha256"]
                    for transfer in inherited
                ),
                "effective_hard_protection": own_hard or inherited_hard,
                "effective_retention_priority": effective_priority,
                "exact_span_identity_sha256": span_sha,
                "inherited_hard_protection": inherited_hard,
                "own_hard_protection": own_hard,
                "own_retention_priority": own_priority,
                "policy": RETENTION_AUTHORITY_TRANSFER_POLICY,
            }
        )
    _require(
        len(protected_g_spans) <= 12,
        "final-fit G hard-protection authority exceeded its bounded 12-span core",
    )
    return MappingProxyType(overlay)


def _typed_rows(
    *,
    rows: Sequence[_Candidate],
    spec: TypedOperatorSpec,
    dated_question: str,
    sealed_sources: TerminalSealedSources,
    parent_receipt_by_plane: Mapping[Plane, str],
) -> tuple[
    tuple[EvidenceHandleBinding, ...],
    ParsedTypedItems,
    Mapping[str, str],
    Mapping[str, tuple[int, ...]],
    Mapping[str, tuple[str, ...]],
    Mapping[str, _Candidate],
]:
    groups_by_plane: dict[Plane, dict[str, str]] = {}
    for plane in PLANE_ORDER:
        source_receipts = tuple(
            dict.fromkeys(
                identity_sha256(
                    {
                        "namespace_id": row.binding.namespace_id,
                        "source_id": row.binding.source_id,
                    }
                )
                for row in rows
                if row.plane == plane
            )
        )
        groups_by_plane[plane] = {
            source_receipt: f"G{GROUP_RANGE_START[plane] + ordinal:03d}"
            for ordinal, source_receipt in enumerate(source_receipts)
        }

    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, object]] = []
    mechanism: dict[str, str] = {}
    priorities: dict[str, tuple[int, ...]] = {}
    story_keys: dict[str, tuple[str, ...]] = {}
    candidate_by_handle: dict[str, _Candidate] = {}
    plane_ordinals: dict[Plane, int] = {plane: 0 for plane in PLANE_ORDER}
    for row in rows:
        ordinal = plane_ordinals[row.plane]
        plane_ordinals[row.plane] += 1
        handle_id = f"H{HANDLE_RANGE_START[row.plane] + ordinal:03d}"
        source_receipt = identity_sha256(
            {
                "namespace_id": row.binding.namespace_id,
                "source_id": row.binding.source_id,
            }
        )
        group_handle = groups_by_plane[row.plane][source_receipt]
        binding = EvidenceHandleBinding(
            handle_id=handle_id,
            origin=EvidenceOrigin.MAP,
            provenance_grade=ProvenanceGrade.EXACT_CITATION,
            source_group_handle=group_handle,
            sealed_artifact_sha256=sealed_sources.artifact_for_plane(row.plane),
            parent_receipt_sha256=parent_receipt_by_plane[row.plane],
            evidence_receipt_sha256=row.receipt_sha256,
            payload_sha256=identity_sha256(row.projection()),
            citation_sha256=quote_sha256(row.quote),
            citation_char_count=len(row.quote),
            local_source_locator_sha256=row.binding.receipt_sha256,
        )
        kind, mention = _typed_kind(spec, row.quote, dated_question)
        raw: dict[str, object] = {
            "handle_ids": [handle_id],
            "included": True,
            "kind": kind,
            "numeric_role": (
                NumericRole.OPERAND.value
                if mention is not None
                else NumericRole.NONE.value
            ),
            "specificity_terms": [],
            "status": (
                "completed"
                if row.matched_completed_actions
                else "proposed"
                if row.disposition == "proposed_action_lane"
                else "unknown"
            ),
            "summary": row.quote,
            "value_authority": "explicit",
        }
        if len(row.event_dates) == 1:
            raw["date"] = row.event_dates[0]
            date_basis = "textual_event_date"
        else:
            raw["date"] = row.created_at
            raw["value_authority"] = "derived"
            date_basis = "source_created_at"
        raw["relation"] = (
            f"authored_by_{row.role};date_basis={date_basis}"
            if row.role in {"user", "assistant"}
            else f"date_basis={date_basis}"
        )
        if mention is not None:
            raw["numeric_value"] = mention.value
            raw["numeric_qualifier"] = mention.qualifier.value
            if mention.unit is not None:
                raw["unit"] = mention.unit
        bindings.append(binding)
        raw_items.append(raw)
        mechanism[handle_id] = MECHANISM_BY_PLANE[row.plane]
        priorities[handle_id] = _local_priority(row)
        story_keys.setdefault(group_handle, (source_receipt,))
        candidate_by_handle[handle_id] = row

    frozen_bindings = tuple(bindings)
    parser_items = [
        {**raw, "summary": str(raw["summary"]).strip()} for raw in raw_items
    ]
    parsed = parse_typed_items(
        parser_items,
        operator_spec=spec,
        bindings=frozen_bindings,
    )
    _require(
        not parsed.rejected_items
        and len(parsed.accepted_items) == len(rows),
        "terminal exact evidence failed common typed parsing",
    )
    exact_items: list[TypedEvidenceItem] = []
    for source_index, (item, row) in enumerate(
        zip(parsed.accepted_items, rows, strict=True)
    ):
        item_id = identity_sha256(
            {
                "format": ITEM_FORMAT,
                "handle_ids": list(item.handle_ids),
                "source_index": source_index,
                "summary_sha256": quote_sha256(row.quote),
                "supported_slot_ids": list(item.supported_slot_ids),
            }
        )
        exact_items.append(
            replace(item, item_id=item_id, summary=row.quote, receipt_sha256="")
        )
    parsed = ParsedTypedItems(
        accepted_items=tuple(exact_items),
        rejected_items=(),
        parse_receipt_sha256=identity_sha256(
            {
                "accepted_item_receipt_sha256s": [
                    row.receipt_sha256 for row in exact_items
                ],
                "format": f"{ITEM_FORMAT}-parse",
                "rejected_item_receipt_sha256s": [],
            }
        ),
    )
    return (
        frozen_bindings,
        parsed,
        MappingProxyType(mechanism),
        MappingProxyType(priorities),
        MappingProxyType(story_keys),
        MappingProxyType(candidate_by_handle),
    )


def _packet_for_entries(
    *,
    spec: TypedOperatorSpec,
    entries: Sequence[tuple[EvidenceHandleBinding, TypedEvidenceItem]],
    sealed_sources: TerminalSealedSources,
) -> TypedEvidencePacket:
    bindings = tuple(binding for binding, _item in entries)
    items = tuple(item for _binding, item in entries)
    parsed = ParsedTypedItems(
        accepted_items=items,
        rejected_items=(),
        parse_receipt_sha256=identity_sha256(
            {
                "accepted_item_receipt_sha256s": [
                    row.receipt_sha256 for row in items
                ],
                "format": f"{ITEM_FORMAT}-terminal-fair-subset-v1",
                "rejected_item_receipt_sha256s": [],
            }
        ),
    )
    return build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=tuple(
            dict.fromkeys(
                (
                    sealed_sources.protected_owner_artifact_sha256,
                    sealed_sources.residual_artifact_sha256,
                )
            )
        ),
        frontier_mode=FrontierMode.OPEN,
        conflict_policy=ConflictPolicy.QUARANTINE,
        output_token_reserve=OUTPUT_TOKEN_RESERVE,
        truncated=True,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
    )


def _compile_typed_prompt(
    *,
    rows: tuple[_Candidate, ...],
    spec: TypedOperatorSpec,
    dated_question: str,
    parent_prediction: str,
    sealed_sources: TerminalSealedSources,
    parent_receipt_by_plane: Mapping[Plane, str],
    policy: SemanticGlobalTerminalPolicy,
    dedup_receipt: DeduplicationReceipt | None = None,
    exact_span_support_population: ExactSpanSupportPopulationReceipt | None = None,
    enable_selected_evidence_discourse_links: bool = False,
) -> tuple[
    TypedEvidencePacket,
    FittedTypedFinalPrompt,
    Mapping[str, str],
    tuple[Mapping[str, Any], ...],
    tuple[str, ...],
]:
    _require(
        type(enable_selected_evidence_discourse_links) is bool,
        "terminal selected-evidence discourse-link feature flag changed type",
    )
    (
        bindings,
        parsed,
        mechanism,
        priorities,
        story_keys,
        candidate_by_handle,
    ) = _typed_rows(
        rows=rows,
        spec=spec,
        dated_question=dated_question,
        sealed_sources=sealed_sources,
        parent_receipt_by_plane=parent_receipt_by_plane,
    )
    if exact_span_support_population is None:
        exact_span_support_population = _exact_span_support_population(
            candidates_by_plane={
                plane: tuple(row for row in rows if row.plane == plane)
                for plane in PLANE_ORDER
            }
        )
    authority_by_candidate = _retention_authority_overlay(
        rows=rows,
        dedup_receipt=dedup_receipt,
        exact_span_support_population=exact_span_support_population,
    )
    support_by_span = exact_span_support_population.authority_by_span
    priorities = MappingProxyType(
        {
            handle: authority_by_candidate[candidate.receipt_sha256][
                "effective_retention_priority"
            ]
            for handle, candidate in candidate_by_handle.items()
        }
    )
    entries = tuple(zip(bindings, parsed.accepted_items, strict=True))
    by_plane: dict[Plane, list[tuple[EvidenceHandleBinding, TypedEvidenceItem]]] = {
        plane: [] for plane in PLANE_ORDER
    }
    for entry in entries:
        plane = candidate_by_handle[entry[0].handle_id].plane
        by_plane[plane].append(entry)
    # The non-empty-plane floor must use the same authenticated retention
    # authority as the final fitter.  Protecting the first upstream row can
    # otherwise freeze a shorter but weaker candidate and force a stronger
    # exact witness from that plane out of the wrapped 8k prompt.
    minima = [
        max(
            values,
            key=lambda entry: priorities[entry[0].handle_id],
        )
        for values in by_plane.values()
        if values
    ]
    minimum_handles = {binding.handle_id for binding, _item in minima}
    protected_tranche = list(minima)
    protected_handles = set(minimum_handles)
    for entry in entries:
        handle = entry[0].handle_id
        candidate = candidate_by_handle[handle]
        if handle in protected_handles:
            continue
        if authority_by_candidate[candidate.receipt_sha256][
            "effective_hard_protection"
        ]:
            protected_tranche.append(entry)
            protected_handles.add(handle)
    protected_receipts = tuple(
        item.receipt_sha256 for _binding, item in protected_tranche
    )
    selected: list[tuple[EvidenceHandleBinding, TypedEvidenceItem]] = list(
        protected_tranche
    )
    if selected:
        packet = _packet_for_entries(
            spec=spec,
            entries=selected,
            sealed_sources=sealed_sources,
        )
        _require(
            {row.receipt_sha256 for row in packet.items}
            == set(protected_receipts),
            "terminal L-packed/bounded-G-core protected tranche exceeds the compact 8k evidence envelope",
        )
    else:
        packet = _packet_for_entries(
            spec=spec,
            entries=(),
            sealed_sources=sealed_sources,
        )
    entries_by_source_group: dict[
        tuple[Plane, str],
        list[tuple[EvidenceHandleBinding, TypedEvidenceItem]],
    ] = defaultdict(list)
    for entry in entries:
        candidate = candidate_by_handle[entry[0].handle_id]
        entries_by_source_group[(candidate.plane, candidate.binding.source_id)].append(
            entry
        )
    compact_group_round_by_handle: dict[str, int] = {}
    for values in entries_by_source_group.values():
        values.sort(
            key=lambda entry: (
                priorities[entry[0].handle_id],
                entry[0].handle_id,
            ),
            reverse=True,
        )
        for source_group_round, entry in enumerate(values):
            compact_group_round_by_handle[entry[0].handle_id] = source_group_round
    remaining = sorted(
        (
            entry
            for entry in entries
            if entry[0].handle_id not in protected_handles
        ),
        key=lambda entry: (
            *priorities[entry[0].handle_id][:12],
            -compact_group_round_by_handle[entry[0].handle_id],
            *priorities[entry[0].handle_id][12:],
            entry[0].handle_id,
        ),
        reverse=True,
    )
    compact_consideration_rank = {
        entry[0].handle_id: rank for rank, entry in enumerate(remaining)
    }
    for entry in remaining:
        trial = (*selected, entry)
        candidate_packet = _packet_for_entries(
            spec=spec,
            entries=trial,
            sealed_sources=sealed_sources,
        )
        if len(candidate_packet.items) == len(trial):
            selected.append(entry)
            packet = candidate_packet

    selected_handles = {binding.handle_id for binding, _item in selected}
    selected_discourse_links = (
        link_selected_evidence(
            tuple(
                SelectedEvidenceLinkInput(
                    handle_id=binding.handle_id,
                    span=candidate_by_handle[binding.handle_id].binding.span,
                    quote=candidate_by_handle[binding.handle_id].quote,
                    source_binding_receipt_sha256=(
                        candidate_by_handle[
                            binding.handle_id
                        ].binding.receipt_sha256
                    ),
                    selected_evidence_receipt_sha256=(
                        candidate_by_handle[binding.handle_id].receipt_sha256
                    ),
                )
                for binding, _item in selected
            )
        )
        if enable_selected_evidence_discourse_links
        else None
    )
    current_mechanism = MappingProxyType(
        {handle: value for handle, value in mechanism.items() if handle in selected_handles}
    )
    current_priorities = MappingProxyType(
        {handle: value for handle, value in priorities.items() if handle in selected_handles}
    )
    current_story_keys = MappingProxyType(
        {
            group: values
            for group, values in story_keys.items()
            if any(
                binding.source_group_handle == group
                for binding, _item in selected
            )
        }
    )
    def opaque_locator_literal(value: str) -> bool:
        # Ordinary semantic source labels can legitimately occur in exact
        # citations (for example a source called ``computer``).  The typed
        # projection structurally omits every locator; this supplemental
        # substring tripwire is restricted to production-opaque/path-like
        # values so it cannot reject their byte-identical citation content.
        return bool(
            len(value) >= 12
            and (
                re.fullmatch(r"[0-9a-fA-F]{16,}", value) is not None
                or any(character in value for character in ("/", "\\", ":", "@"))
                or (
                    any(character.isdigit() for character in value)
                    and any(not character.isalnum() for character in value)
                )
            )
        )

    forbidden = tuple(
        dict.fromkeys(
            value
            for row in rows
            for value in (
                row.binding.namespace_id,
                row.binding.source_id,
                row.binding.partition_id,
            )
            if opaque_locator_literal(value)
        )
    )
    fitted = fit_typed_final_prompt(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        packet=packet,
        mechanism_by_handle=current_mechanism,
        local_story_keys_by_group=current_story_keys,
        selected_evidence_discourse_links=selected_discourse_links,
        local_retention_priority_by_handle=current_priorities,
        forbidden_provider_literals=forbidden,
        minimum_usable_items_per_mechanism=(
            policy.minimum_usable_items_per_nonempty_mechanism
        ),
        protected_item_receipt_sha256s=protected_receipts,
        protection_source_receipt_sha256=(
            identity_sha256(
                {
                    "dedup_receipt_sha256": (
                        None if dedup_receipt is None else dedup_receipt.receipt_sha256
                    ),
                    "format": f"{FORMAT}-final-protection-authority-v1",
                    "policy_receipt_sha256": policy.receipt_sha256,
                    "retention_authority_transfer_policy": (
                        RETENTION_AUTHORITY_TRANSFER_POLICY
                    ),
                }
            )
            if protected_receipts
            else None
        ),
    )
    final_handles = {row.handle_id for row in fitted.packet.handles}
    final_mechanism = MappingProxyType(
        {
            handle: value
            for handle, value in current_mechanism.items()
            if handle in final_handles
        }
    )
    retained_candidates = tuple(
        candidate_by_handle[handle]
        for handle in candidate_by_handle
        if handle in final_handles
    )
    local_rows = tuple(
        MappingProxyType(
            {
                "admitted_to_compact_packet": handle in selected_handles,
                "binding": candidate.binding.projection(),
                "candidate": candidate.projection(),
                "compact_consideration_rank": compact_consideration_rank.get(handle),
                "compact_source_group_round": compact_group_round_by_handle[handle],
                "exact_span_support_authority": {
                    key: (
                        tuple(value)
                        if key in _EXACT_SPAN_SUPPORT_SEQUENCE_KEYS
                        else value
                    )
                    for key, value in support_by_span[
                        identity_sha256(candidate.binding.span.identity_payload())
                    ].projection().items()
                },
                "final_handle_id": handle,
                "mechanism_id": mechanism[handle],
                "protected_in_final_fit": handle in protected_handles,
                "retention_authority": dict(
                    authority_by_candidate[candidate.receipt_sha256]
                ),
                "retained_in_final_prompt": handle in final_handles,
            }
        )
        for handle, candidate in candidate_by_handle.items()
    )
    return (
        fitted.packet,
        fitted,
        final_mechanism,
        local_rows,
        tuple(row.receipt_sha256 for row in retained_candidates),
    )


def load_selected_protected_owner_evidence(
    rows: Sequence[Mapping[str, Any]],
    /,
) -> tuple[ProtectedOwnerEvidence, ...]:
    """Strictly load the selected P rows from the existing R7 provider input."""

    _require(
        type(rows) in {list, tuple}
        and all(type(row) is dict for row in rows),
        "selected protected-owner inventory changed schema",
    )
    loaded = tuple(ProtectedOwnerEvidence.from_provider_row(row) for row in rows)
    _require(
        len({row.receipt_sha256 for row in loaded}) == len(loaded),
        "selected protected-owner inventory repeats exact rows",
    )
    return loaded


def compile_semantic_global_terminal(
    *,
    dated_question: str,
    parent_prediction: str,
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    protected_owner_universe_bindings: Sequence[LocalCitationBinding],
    selected_protected_owner_evidence: Sequence[ProtectedOwnerEvidence],
    residual_result: SemanticResidualSearchResult,
    local_result: SourceGroupReinjectionResult,
    global_result: SemanticGlobalCompletionResult,
    sealed_sources: TerminalSealedSources,
    policy: SemanticGlobalTerminalPolicy | None = None,
    enable_selected_evidence_discourse_links: bool = False,
) -> SemanticGlobalTerminalCompilation:
    """Compile one cumulative, bounded, provider-free terminal prompt.

    ``protected_owner_universe_bindings`` is the full R7 owner universe used
    for authentication.  ``selected_protected_owner_evidence`` is only the P
    row population already selected for provider visibility by R7.
    """

    require_text(dated_question, "terminal dated question")
    require_text(parent_prediction, "terminal parent prediction")
    _require(
        type(enable_selected_evidence_discourse_links) is bool,
        "terminal selected-evidence discourse-link feature flag changed type",
    )
    _require(
        type(residual_index) is SemanticResidualIndex
        and type(query) is SemanticResidualQuery
        and type(residual_result) is SemanticResidualSearchResult
        and type(local_result) is SourceGroupReinjectionResult
        and type(global_result) is SemanticGlobalCompletionResult
        and type(sealed_sources) is TerminalSealedSources,
        "terminal live input types changed",
    )
    active_policy = policy or SemanticGlobalTerminalPolicy()
    _require(
        type(active_policy) is SemanticGlobalTerminalPolicy,
        "terminal policy type changed",
    )
    protected = tuple(protected_owner_universe_bindings)
    selected_owners = tuple(selected_protected_owner_evidence)
    _require(
        all(type(row) is LocalCitationBinding for row in protected)
        and len({row.receipt_sha256 for row in protected}) == len(protected)
        and all(type(row) is ProtectedOwnerEvidence for row in selected_owners),
        "terminal protected owner inputs changed",
    )
    _require(
        dated_question == query.dated_question
        and query.residual_index_receipt_sha256 == residual_index.receipt_sha256
        and residual_result.residual_index_receipt_sha256
        == residual_index.receipt_sha256
        and residual_result.query.receipt_sha256 == query.receipt_sha256
        and local_result.residual_index_receipt_sha256
        == residual_index.receipt_sha256
        and local_result.query_receipt_sha256 == query.receipt_sha256
        and global_result.residual_index_receipt_sha256
        == residual_index.receipt_sha256
        and global_result.query_receipt_sha256 == query.receipt_sha256,
        "terminal inputs escaped their exact index/question",
    )
    expected_p_population = semantic_residual_protected_evidence_population_receipt(
        residual_index, protected
    )
    _require(
        residual_result.protected_evidence_population_receipt_sha256
        == expected_p_population,
        "terminal P universe differs from residual search-time owners",
    )
    expected_global_population = (
        semantic_residual_protected_evidence_population_receipt(
            residual_index,
            (*protected, *residual_result.local_bindings, *local_result.local_bindings),
        )
    )
    _require(
        global_result.protected_evidence_population_receipt_sha256
        == expected_global_population,
        "terminal P/R/L union differs from global search-time owners",
    )
    _require(
        {row.protected_duplicate_receipt_sha256 for row in selected_owners}
        == {row.receipt_sha256 for row in residual_result.protected_duplicates},
        "selected P rows do not cover every removed residual duplicate exactly",
    )

    by_receipt, by_chunk = _segment_inventory(residual_index)
    owner_by_receipt, owner_plane_by_receipt = _owner_inventory(
        protected, residual_result, local_result
    )
    candidates_by_plane: dict[Plane, tuple[_Candidate, ...]] = {
        "P": _selected_protected_owner_candidates(
            residual_index=residual_index,
            dated_question=dated_question,
            selected_protected_owner_evidence=selected_owners,
            residual_result=residual_result,
            owner_by_receipt=owner_by_receipt,
            owner_plane_by_receipt=owner_plane_by_receipt,
            by_receipt=by_receipt,
            by_chunk=by_chunk,
        ),
        "R": _residual_candidates(
            residual_index=residual_index,
            query=query,
            obligations=global_result.obligations,
            result=residual_result,
            owner_by_receipt=owner_by_receipt,
            owner_plane_by_receipt=owner_plane_by_receipt,
            by_receipt=by_receipt,
            by_chunk=by_chunk,
        ),
        "L": _local_candidates(
            residual_index=residual_index,
            dated_question=dated_question,
            result=local_result,
            owner_by_receipt=owner_by_receipt,
            owner_plane_by_receipt=owner_plane_by_receipt,
            by_receipt=by_receipt,
            by_chunk=by_chunk,
        ),
        "G": _global_candidates(
            residual_index=residual_index,
            query=query,
            result=global_result,
            policy=active_policy,
            owner_by_receipt=owner_by_receipt,
            owner_plane_by_receipt=owner_plane_by_receipt,
            by_receipt=by_receipt,
            by_chunk=by_chunk,
        ),
    }
    direct_operand_population, direct_operand_reserved = _direct_operand_lane(
        candidates_by_plane["G"],
        dated_question=dated_question,
        spec=query.operator_spec,
        obligations=global_result.obligations,
        max_items=active_policy.max_direct_operand_lane_items,
    )
    has_date_obligations = any(
        row.kind == "date" for row in global_result.obligations
    )
    selected_by_plane: dict[Plane, tuple[_Candidate, ...]] = {}
    selection_receipts: list[PlaneSelectionReceipt] = []
    for plane in PLANE_ORDER:
        selected, receipt = _select_plane(
            candidates_by_plane[plane],
            active_policy.budget_by_plane[plane],
            direct_operand_population=(
                direct_operand_population if plane == "G" else ()
            ),
            direct_operand_reserved=(
                direct_operand_reserved if plane == "G" else ()
            ),
            include_proposed=query.operator_spec.include_proposed,
            has_date_obligations=has_date_obligations,
        )
        selected_by_plane[plane] = selected
        selection_receipts.append(receipt)
    selection_receipt_tuple = tuple(selection_receipts)
    exact_span_support_population = _exact_span_support_population(
        candidates_by_plane=candidates_by_plane,
        plane_selections=selection_receipt_tuple,
    )
    post_dedup_rows, dedup_receipt = _post_selection_dedup(
        MappingProxyType(selected_by_plane), by_receipt=by_receipt
    )
    protected_population_receipt = identity_sha256(
        {
            "format": f"{OWNER_FORMAT}-selected-population-v1",
            "full_owner_binding_receipt_sha256s": [
                row.receipt_sha256 for row in protected
            ],
            "selected_owner_row_receipt_sha256s": [
                row.receipt_sha256 for row in selected_owners
            ],
        }
    )
    packet, fitted, mechanism, typed_local_rows, retained_receipts = (
        _compile_typed_prompt(
            rows=post_dedup_rows,
            spec=query.operator_spec,
            dated_question=dated_question,
            parent_prediction=parent_prediction,
            sealed_sources=sealed_sources,
            parent_receipt_by_plane=MappingProxyType(
                {
                    "P": protected_population_receipt,
                    "R": residual_result.receipt_sha256,
                    "L": local_result.receipt_sha256,
                    "G": global_result.receipt_sha256,
                }
            ),
            policy=active_policy,
            dedup_receipt=dedup_receipt,
            exact_span_support_population=exact_span_support_population,
            enable_selected_evidence_discourse_links=(
                enable_selected_evidence_discourse_links
            ),
        )
    )
    typed_by_candidate = {
        row["candidate"]["receipt_sha256"]: row for row in typed_local_rows
    }
    selected_receipt_set = {
        row.receipt_sha256
        for values in selected_by_plane.values()
        for row in values
    }
    post_dedup_receipt_set = {
        row.receipt_sha256 for row in post_dedup_rows
    }
    local_rows = tuple(
        MappingProxyType(
            {
                "binding": candidate.binding.projection(),
                "candidate": candidate.projection(),
                "retained_after_post_selection_dedup": (
                    candidate.receipt_sha256 in post_dedup_receipt_set
                ),
                "selected_by_independent_plane_budget": (
                    candidate.receipt_sha256 in selected_receipt_set
                ),
                "typed_terminal": (
                    None
                    if candidate.receipt_sha256 not in typed_by_candidate
                    else dict(typed_by_candidate[candidate.receipt_sha256])
                ),
            }
        )
        for plane in PLANE_ORDER
        for candidate in candidates_by_plane[plane]
    )
    compilation = SemanticGlobalTerminalCompilation(
        policy=active_policy,
        sealed_sources=sealed_sources,
        residual_index_receipt_sha256=residual_index.receipt_sha256,
        query_receipt_sha256=query.receipt_sha256,
        residual_result_receipt_sha256=residual_result.receipt_sha256,
        local_result_receipt_sha256=local_result.receipt_sha256,
        global_result_receipt_sha256=global_result.receipt_sha256,
        protected_owner_population_receipt_sha256=protected_population_receipt,
        exact_span_support_population=exact_span_support_population,
        plane_selections=selection_receipt_tuple,
        post_selection_dedup=dedup_receipt,
        packet=packet,
        fitted=fitted,
        mechanism_by_handle=mechanism,
        retained_row_receipt_sha256s=retained_receipts,
        local_rows=local_rows,
        format_id=(
            LINKED_FORMAT
            if enable_selected_evidence_discourse_links
            else FORMAT
        ),
    )
    _require(
        sealed_sources.parent_artifact_sha256
        not in compilation.packet.sealed_input_artifact_sha256s
        and all(
            row.parent_receipt_sha256 != sealed_sources.parent_artifact_sha256
            for row in compilation.packet.local_bindings
        ),
        "parent fallback was mislabeled as terminal evidence",
    )
    return compilation


def replay_semantic_global_terminal(
    *,
    dated_question: str,
    parent_prediction: str,
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    protected_owner_universe_bindings: Sequence[LocalCitationBinding],
    selected_protected_owner_evidence: Sequence[ProtectedOwnerEvidence],
    residual_result: SemanticResidualSearchResult,
    local_result: SourceGroupReinjectionResult,
    global_result: SemanticGlobalCompletionResult,
    sealed_sources: TerminalSealedSources,
    sealed_compilation: SemanticGlobalTerminalCompilation,
    policy: SemanticGlobalTerminalPolicy | None = None,
) -> SemanticGlobalTerminalCompilation:
    """Recompile from authenticated live inputs and require byte identity."""

    _require(
        type(sealed_compilation) is SemanticGlobalTerminalCompilation,
        "sealed terminal compilation type changed",
    )
    replayed = compile_semantic_global_terminal(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        residual_index=residual_index,
        query=query,
        protected_owner_universe_bindings=protected_owner_universe_bindings,
        selected_protected_owner_evidence=selected_protected_owner_evidence,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
        policy=policy or sealed_compilation.policy,
        enable_selected_evidence_discourse_links=(
            sealed_compilation.format_id == LINKED_FORMAT
        ),
    )
    _require(
        replayed.receipt_sha256 == sealed_compilation.receipt_sha256
        and replayed.projection(include_local=True)
        == sealed_compilation.projection(include_local=True),
        "sealed terminal compilation differs from deterministic replay",
    )
    return replayed


__all__ = [
    "DeduplicationReceipt",
    "PlaneBudget",
    "PlaneSelectionReceipt",
    "ProtectedOwnerEvidence",
    "SemanticGlobalTerminalCompilation",
    "SemanticGlobalTerminalError",
    "SemanticGlobalTerminalPolicy",
    "LINKED_FORMAT",
    "TerminalSealedSources",
    "compile_semantic_global_terminal",
    "load_selected_protected_owner_evidence",
    "replay_semantic_global_terminal",
]
