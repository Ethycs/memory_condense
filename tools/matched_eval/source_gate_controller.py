"""Small, provider-free fallback gate for exact source-history mapping.

The gate runs only after the existing packet leaves sealed obligations
unresolved.  It preserves each method's source-selection credit, hydrates exact
namespaced memberships through :mod:`source_history_fact_union`, and groups
identical cross-method windows into one question-bound physical map call.  One
completion is validated once per logical window alias, so post-map union keeps
all lane provenance without paying for duplicate calls.

This module opens no files or stores, calls no provider, and accepts no gold,
target, prediction, or judge input.  Heavy artifact loading and provider
execution deliberately live outside this controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256
from tools._routed_repair_routing import (
    RoutedRepairReceipt,
    RoutedRepairStyle,
    route_question,
)

from .contracts import (
    ArtifactRef,
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .source_history_fact_union import (
    EventTuple,
    FactLane,
    FrozenHistoryChunk,
    MappedFactBatch,
    ParentIdentity,
    PostMapFactUnion,
    SourceHistoryHydrationPlan,
    SourceHistoryWindow,
    SourceSelection,
    validate_mapper_completion,
)


FORMAT = "memory-condense-source-gate-controller-v1"
SOURCE_GATE_LANES = (FactLane.DIRECT, FactLane.PARTITION, FactLane.GUIDED)


class SourceGateControllerError(MatchedEvalContractError):
    """A gate identity, prefix, provenance binding, or cap changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SourceGateControllerError(message)


def _int(value: object, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be >= {minimum}")
    return value  # type: ignore[return-value]


def _typed(values: object, cls: type, label: str) -> tuple[Any, ...]:
    _require(
        type(values) is tuple and all(type(row) is cls for row in values),
        f"{label} must be an immutable exact-{cls.__name__} tuple",
    )
    return values  # type: ignore[return-value]


def _unique_text(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(type(values) is tuple, f"{label} must be an immutable tuple")
    for value in values:
        require_text(value, label)
    _require(len(values) == len(set(values)), f"{label} must be ordered and unique")
    return values


def _unique_sha(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(type(values) is tuple, f"{label} must be an immutable tuple")
    for value in values:
        require_sha256(value, label)
    _require(len(values) == len(set(values)), f"{label} must be ordered and unique")
    return values


def _seal(kind: str, body: Mapping[str, Any]) -> str:
    value = {"format": f"{FORMAT}-{kind}", **body}
    assert_gold_blind(value, path="source_gate_controller")
    return identity_sha256(value)


def _record(kind: str, body: Mapping[str, Any], **identity: str) -> dict[str, Any]:
    value = {"format": f"{FORMAT}-{kind}", **body, **identity}
    assert_gold_blind(value, path="source_gate_controller")
    return value


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


class _SealedRecord:
    __slots__ = ()

    _kind = ""

    def _body(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def receipt_sha256(self) -> str:
        return _seal(self._kind, self._body())

    def projection(self) -> dict[str, Any]:
        body = self._body()
        return _record(
            self._kind,
            body,
            receipt_sha256=_seal(self._kind, body),
        )


@dataclass(frozen=True, slots=True)
class NamespacedSourceKey:
    namespace_id: str
    source_id: str

    def __post_init__(self) -> None:
        require_sha256(self.namespace_id, "source-key namespace")
        require_text(self.source_id, "source-key source ID")

    def projection(self) -> dict[str, str]:
        return {"namespace_id": self.namespace_id, "source_id": self.source_id}


@dataclass(frozen=True, slots=True)
class SourceGateCandidate:
    """A method-local distinct source, ranked by its first selected span."""

    lane: FactLane
    namespace_id: str
    source_id: str
    rank: int
    membership_projection_sha256: str
    stream_sha256: str
    source_stream_receipt_sha256: str

    def __post_init__(self) -> None:
        _require(self.lane in SOURCE_GATE_LANES, "candidate lane is not source-gateable")
        require_sha256(self.namespace_id, "candidate namespace")
        require_text(self.source_id, "candidate source ID")
        _int(self.rank, "candidate rank")
        require_sha256(self.membership_projection_sha256, "candidate membership")
        require_sha256(self.stream_sha256, "candidate source stream")
        require_sha256(self.source_stream_receipt_sha256, "candidate stream receipt")

    def _body(self) -> dict[str, Any]:
        return {
            "lane": self.lane.value,
            "membership_projection_sha256": self.membership_projection_sha256,
            "namespace_id": self.namespace_id,
            "rank": self.rank,
            "source_id": self.source_id,
            "source_stream_receipt_sha256": self.source_stream_receipt_sha256,
            "stream_sha256": self.stream_sha256,
        }

    @property
    def candidate_id(self) -> str:
        return _seal("candidate-id", self._body())

    @property
    def source_key(self) -> NamespacedSourceKey:
        return NamespacedSourceKey(self.namespace_id, self.source_id)

    def projection(self) -> dict[str, Any]:
        return _record("candidate", self._body(), candidate_id=self.candidate_id)


@dataclass(frozen=True, slots=True)
class LaneSourceBudget:
    lane: FactLane
    base_source_cap: int
    hard_source_cap: int
    tail_step_source_cap: int

    def __post_init__(self) -> None:
        _require(self.lane in SOURCE_GATE_LANES, "lane budget is not source-gateable")
        base, hard = _int(self.base_source_cap, "base cap"), _int(self.hard_source_cap, "hard cap")
        step = _int(self.tail_step_source_cap, "tail step", 1)
        _require(base <= hard, "base source cap exceeds hard source cap")
        _require(step <= max(1, hard), "tail step exceeds usable hard source cap")

    def projection(self) -> dict[str, Any]:
        return {
            "base_source_cap": self.base_source_cap,
            "hard_source_cap": self.hard_source_cap,
            "lane": self.lane.value,
            "tail_step_source_cap": self.tail_step_source_cap,
        }


@dataclass(frozen=True, slots=True)
class SourceGatePolicy(_SealedRecord):
    _kind = "policy"
    policy_id: str
    lane_budgets: tuple[LaneSourceBudget, ...]
    global_unique_source_cap: int
    max_physical_map_calls: int
    max_rounds: int

    def __post_init__(self) -> None:
        require_text(self.policy_id, "source-gate policy ID")
        _typed(self.lane_budgets, LaneSourceBudget, "lane budgets")
        _require(tuple(row.lane for row in self.lane_budgets) == SOURCE_GATE_LANES, "lane budgets changed canonical order")
        _int(self.global_unique_source_cap, "global unique-source cap", 1)
        _int(self.max_physical_map_calls, "physical map-call cap")
        _int(self.max_rounds, "round cap", 1)
        _require(sum(row.base_source_cap for row in self.lane_budgets) <= self.global_unique_source_cap, "worst-case base exceeds global source cap")

    def _body(self) -> dict[str, Any]:
        return {
            "global_unique_source_cap": self.global_unique_source_cap,
            "lane_budgets": [row.projection() for row in self.lane_budgets],
            "max_physical_map_calls": self.max_physical_map_calls,
            "max_rounds": self.max_rounds,
            "policy_id": self.policy_id,
        }

    def budget_for(self, lane: FactLane) -> LaneSourceBudget:
        rows = tuple(row for row in self.lane_budgets if row.lane is lane)
        _require(len(rows) == 1, "policy lost a lane budget")
        return rows[0]


def default_source_gate_policy() -> SourceGatePolicy:
    return SourceGatePolicy(
        "direct5-guided2-adaptive-tail-v1",
        (
            LaneSourceBudget(FactLane.DIRECT, 5, 12, 2),
            LaneSourceBudget(FactLane.PARTITION, 0, 10, 2),
            LaneSourceBudget(FactLane.GUIDED, 2, 8, 2),
        ),
        global_unique_source_cap=24,
        max_physical_map_calls=48,
        max_rounds=16,
    )


class ObligationKind(str, Enum):
    SUPPORT = "support"
    TEMPORAL = "temporal"
    FRONTIER = "frontier"
    CARDINALITY = "cardinality"


@dataclass(frozen=True, slots=True)
class QuestionObligation:
    kind: ObligationKind
    match_terms: tuple[str, ...]
    required_match_term_count: int
    minimum_fact_count: int = 1
    minimum_source_count: int = 1
    requires_temporal_metadata: bool = False
    requires_complete_frontier: bool = False

    def __post_init__(self) -> None:
        _require(type(self.kind) is ObligationKind, "obligation kind changed")
        _unique_text(self.match_terms, "obligation terms")
        _require(bool(self.match_terms), "obligation requires a match term")
        _require(len({_norm(row) for row in self.match_terms}) == len(self.match_terms), "normalized obligation terms repeat")
        count = _int(self.required_match_term_count, "required term count", 1)
        _require(count <= len(self.match_terms), "required term count exceeds terms")
        _int(self.minimum_fact_count, "minimum fact count", 1)
        _int(self.minimum_source_count, "minimum source count", 1)
        _require(type(self.requires_temporal_metadata) is bool and type(self.requires_complete_frontier) is bool, "obligation flags must be bools")

    def _body(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "match_terms": list(self.match_terms),
            "minimum_fact_count": self.minimum_fact_count,
            "minimum_source_count": self.minimum_source_count,
            "required_match_term_count": self.required_match_term_count,
            "requires_complete_frontier": self.requires_complete_frontier,
            "requires_temporal_metadata": self.requires_temporal_metadata,
        }

    @property
    def obligation_id(self) -> str:
        return _seal("obligation-id", self._body())

    def projection(self) -> dict[str, Any]:
        return _record("obligation", self._body(), obligation_id=self.obligation_id)


@dataclass(frozen=True, slots=True)
class SourceGateActivationReceipt(_SealedRecord):
    """Proof that the ordinary evidence/fact packet still needs this fallback."""

    _kind = "activation"

    question_id: str
    question_sha256: str
    dated_question_sha256: str
    parent_packet_id: str
    upstream_question_plan_receipt_sha256: str
    upstream_fact_frontier_receipt_sha256: str
    obligation_ids: tuple[str, ...]
    unresolved_obligation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text(self.question_id, "activation question ID")
        for value, label in (
            (self.question_sha256, "activation question"),
            (self.dated_question_sha256, "activation dated question"),
            (self.parent_packet_id, "activation parent packet"),
            (self.upstream_question_plan_receipt_sha256, "upstream question plan"),
            (self.upstream_fact_frontier_receipt_sha256, "upstream fact frontier"),
        ):
            require_sha256(value, label)
        _unique_sha(self.obligation_ids, "activation obligations")
        _unique_sha(self.unresolved_obligation_ids, "activation unresolved obligations")
        _require(bool(self.obligation_ids), "activation requires upstream obligations")
        _require(bool(self.unresolved_obligation_ids), "source gate cannot activate when upstream obligations are satisfied")
        _require(set(self.unresolved_obligation_ids) <= set(self.obligation_ids), "unresolved obligations escaped upstream plan")

    def _body(self) -> dict[str, Any]:
        return {
            "dated_question_sha256": self.dated_question_sha256,
            "fallback_required": True,
            "obligation_ids": list(self.obligation_ids),
            "parent_packet_id": self.parent_packet_id,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
            "upstream_fact_frontier_receipt_sha256": self.upstream_fact_frontier_receipt_sha256,
            "upstream_question_plan_receipt_sha256": self.upstream_question_plan_receipt_sha256,
        }

@dataclass(frozen=True, slots=True)
class EligibleFrontierScope(_SealedRecord):
    """Sealed eligible candidate scope; exhaustiveness is never inferred."""

    _kind = "eligible-frontier"

    eligible_candidate_ids: tuple[str, ...]
    exhaustive: bool
    basis_receipt_sha256: str

    def __post_init__(self) -> None:
        _unique_sha(self.eligible_candidate_ids, "eligible frontier candidates")
        _require(type(self.exhaustive) is bool, "frontier exhaustive flag changed")
        _require(not self.exhaustive or bool(self.eligible_candidate_ids), "exhaustive frontier cannot be empty")
        require_sha256(self.basis_receipt_sha256, "frontier basis receipt")

    def _body(self) -> dict[str, Any]:
        return {
            "basis_receipt_sha256": self.basis_receipt_sha256,
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "exhaustive": self.exhaustive,
        }

@dataclass(frozen=True, slots=True)
class SourceGatePlan(_SealedRecord):
    _kind = "plan"
    parent: ParentIdentity
    question_id: str
    question_sha256: str
    dated_question: str
    dated_question_sha256: str
    as_of_turn: int
    route: RoutedRepairReceipt
    sealed_input_artifacts: tuple[ArtifactRef, ...]
    candidates: tuple[SourceGateCandidate, ...]
    obligations: tuple[QuestionObligation, ...]
    activation: SourceGateActivationReceipt
    eligible_frontier: EligibleFrontierScope
    policy: SourceGatePolicy

    def __post_init__(self) -> None:
        _require(type(self.parent) is ParentIdentity, "gate parent must be exact")
        require_text(self.question_id, "gate question ID")
        require_sha256(self.question_sha256, "gate question")
        require_text(self.dated_question, "gate dated question")
        require_sha256(self.dated_question_sha256, "gate dated question")
        _require(quote_sha256(self.dated_question) == self.dated_question_sha256, "dated question text changed")
        _int(self.as_of_turn, "gate as-of turn")
        _require(type(self.route) is RoutedRepairReceipt and route_question(self.dated_question).receipt_sha256 == self.route.receipt_sha256, "gate route changed its dated question")
        _require(self.route.question_sha256 == self.dated_question_sha256, "route question hash changed")
        _typed(self.sealed_input_artifacts, ArtifactRef, "sealed inputs")
        _require(bool(self.sealed_input_artifacts) and len({row.role for row in self.sealed_input_artifacts}) == len(self.sealed_input_artifacts), "sealed inputs are empty or repeat roles")
        _typed(self.candidates, SourceGateCandidate, "gate candidates")
        _typed(self.obligations, QuestionObligation, "gate obligations")
        _require(bool(self.obligations), "gate requires unresolved obligations")
        _require(type(self.activation) is SourceGateActivationReceipt and type(self.eligible_frontier) is EligibleFrontierScope and type(self.policy) is SourceGatePolicy, "gate activation/frontier/policy must be exact")
        _require(
            self.activation.question_id == self.question_id
            and self.activation.question_sha256 == self.question_sha256
            and self.activation.dated_question_sha256 == self.dated_question_sha256
            and self.activation.parent_packet_id == self.parent.parent_packet_id,
            "activation changed question/parent binding",
        )
        _require(tuple(row.obligation_id for row in self.obligations) == self.activation.unresolved_obligation_ids, "gate must carry exactly unresolved upstream obligations")
        _require(all(row.namespace_id == self.parent.namespace_id for row in self.candidates), "candidate escaped parent namespace")
        _require(len({row.candidate_id for row in self.candidates}) == len(self.candidates), "candidate IDs repeat")
        expected = tuple(row for lane in SOURCE_GATE_LANES for row in self.candidates if row.lane is lane)
        _require(expected == self.candidates, "candidates changed canonical lane order")
        provenance: dict[NamespacedSourceKey, tuple[str, str]] = {}
        for lane in SOURCE_GATE_LANES:
            rows = self.candidates_for(lane)
            _require(tuple(row.rank for row in rows) == tuple(range(len(rows))), "method-local ranks must be contiguous")
            _require(len({row.source_key for row in rows}) == len(rows), "method-local stream repeats a source")
            for row in rows:
                value = (row.membership_projection_sha256, row.stream_sha256)
                _require(row.source_key not in provenance or provenance[row.source_key] == value, "cross-method source provenance disagrees")
                provenance[row.source_key] = value
        _require(len({row.obligation_id for row in self.obligations}) == len(self.obligations), "obligation IDs repeat")
        candidate_ids = tuple(row.candidate_id for row in self.candidates)
        eligible = self.eligible_frontier.eligible_candidate_ids
        iterator = iter(candidate_ids)
        _require(set(eligible) <= set(candidate_ids) and all(any(candidate == value for candidate in iterator) for value in eligible), "eligible frontier escaped/reordered candidates")
        if self.route.modifiers.requires_temporal_metadata:
            _require(any(row.requires_temporal_metadata for row in self.obligations), "temporal route lacks temporal obligation")
        if self.route.modifiers.requires_complete_frontier:
            _require(any(row.requires_complete_frontier for row in self.obligations), "complete route lacks frontier obligation")
        if self.route.modifiers.cardinality is not None:
            _require(max(row.minimum_fact_count for row in self.obligations) >= self.route.modifiers.cardinality, "cardinality route lost fact count")
        base = sum(min(self.policy.budget_for(lane).base_source_cap, len(self.candidates_for(lane))) for lane in SOURCE_GATE_LANES)
        _require(base <= self.policy.global_unique_source_cap, "plan base exceeds global source cap")
        assert_gold_blind(self.projection(), path="source_gate_plan")

    def _body(self) -> dict[str, Any]:
        return {
            "activation_receipt_sha256": self.activation.receipt_sha256,
            "as_of_turn": self.as_of_turn,
            "candidates": [row.projection() for row in self.candidates],
            "dated_question": self.dated_question,
            "dated_question_sha256": self.dated_question_sha256,
            "eligible_frontier_receipt_sha256": self.eligible_frontier.receipt_sha256,
            "gold_loaded": False,
            "obligations": [row.projection() for row in self.obligations],
            "parent_identity_sha256": self.parent.identity_sha256,
            "policy_receipt_sha256": self.policy.receipt_sha256,
            "provider_calls": 0,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "retained_transformer_token_state_bytes": 0,
            "route_receipt_sha256": self.route.receipt_sha256,
            "sealed_input_artifacts": [row.projection() for row in self.sealed_input_artifacts],
        }

    @property
    def question_plan_receipt_sha256(self) -> str:
        """Bind this digest in ``PromptTickPlan.question_plan_receipt_sha256``."""
        return self.receipt_sha256

    def candidates_for(self, lane: FactLane) -> tuple[SourceGateCandidate, ...]:
        return tuple(row for row in self.candidates if row.lane is lane)

    def candidate_by_id(self, candidate_id: str) -> SourceGateCandidate:
        rows = tuple(row for row in self.candidates if row.candidate_id == candidate_id)
        _require(len(rows) == 1, "candidate ID escaped gate plan")
        return rows[0]


class GateRoundKind(str, Enum):
    BASE = "base"
    TAIL = "tail"


@dataclass(frozen=True, slots=True)
class SourceGateRound(_SealedRecord):
    _kind = "round"
    gate_plan_receipt_sha256: str
    round_index: int
    kind: GateRoundKind
    tail_lane: FactLane | None
    prior_round_receipt_sha256: str | None
    selected_candidates: tuple[SourceGateCandidate, ...]
    cumulative_selected_candidate_ids: tuple[str, ...]
    cumulative_unique_source_count: int

    def __post_init__(self) -> None:
        require_sha256(self.gate_plan_receipt_sha256, "round gate plan")
        _int(self.round_index, "round index")
        _require(type(self.kind) is GateRoundKind, "round kind changed")
        if self.kind is GateRoundKind.BASE:
            _require(self.round_index == 0 and self.tail_lane is None and self.prior_round_receipt_sha256 is None, "base round coordinates changed")
        else:
            _require(self.round_index > 0 and self.tail_lane in SOURCE_GATE_LANES, "tail round coordinates changed")
            require_sha256(self.prior_round_receipt_sha256 or "", "tail prior round")
        _typed(self.selected_candidates, SourceGateCandidate, "round candidates")
        _unique_sha(self.cumulative_selected_candidate_ids, "cumulative candidate IDs")
        _require(not self.selected_candidate_ids or self.cumulative_selected_candidate_ids[-len(self.selected_candidate_ids):] == self.selected_candidate_ids, "round is not cumulative suffix")
        _int(self.cumulative_unique_source_count, "cumulative unique-source count")

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(row.candidate_id for row in self.selected_candidates)

    def _decision(self) -> dict[str, Any]:
        return {
            "cumulative_selected_candidate_ids": list(self.cumulative_selected_candidate_ids),
            "gate_plan_receipt_sha256": self.gate_plan_receipt_sha256,
            "kind": self.kind.value,
            "prior_round_receipt_sha256": self.prior_round_receipt_sha256,
            "round_index": self.round_index,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "tail_lane": None if self.tail_lane is None else self.tail_lane.value,
        }

    @property
    def selector_receipt_sha256(self) -> str:
        return _seal("selector", self._decision())

    @property
    def selections(self) -> tuple[SourceSelection, ...]:
        return tuple(
            SourceSelection(
                selection_id=_seal("source-selection-id", {
                    "candidate_id": row.candidate_id,
                    "gate_plan_receipt_sha256": self.gate_plan_receipt_sha256,
                    "selector_receipt_sha256": self.selector_receipt_sha256,
                }),
                lane=row.lane,
                namespace_id=row.namespace_id,
                source_id=row.source_id,
                rank=row.rank,
                selector_receipt_sha256=self.selector_receipt_sha256,
            )
            for row in self.selected_candidates
        )

    def _body(self) -> dict[str, Any]:
        return {
            **self._decision(),
            "cumulative_unique_source_count": self.cumulative_unique_source_count,
            "selections": [row.projection() for row in self.selections],
            "selector_receipt_sha256": self.selector_receipt_sha256,
        }

def _make_round(plan: SourceGatePlan, prior: SourceGateRound | None, kind: GateRoundKind, lane: FactLane | None, selected: tuple[SourceGateCandidate, ...]) -> SourceGateRound:
    cumulative = (() if prior is None else prior.cumulative_selected_candidate_ids) + tuple(row.candidate_id for row in selected)
    keys = {plan.candidate_by_id(candidate_id).source_key for candidate_id in cumulative}
    return SourceGateRound(
        plan.receipt_sha256,
        0 if prior is None else prior.round_index + 1,
        kind,
        lane,
        None if prior is None else prior.receipt_sha256,
        selected,
        cumulative,
        len(keys),
    )


def start_source_gate(plan: SourceGatePlan) -> SourceGateRound:
    """Select the base only for a plan carrying an unresolved activation proof."""
    if type(plan) is not SourceGatePlan:
        raise TypeError("plan must be an exact SourceGatePlan")
    selected = tuple(candidate for lane in SOURCE_GATE_LANES for candidate in plan.candidates_for(lane)[: plan.policy.budget_for(lane).base_source_cap])
    return _make_round(plan, None, GateRoundKind.BASE, None, selected)


@dataclass(frozen=True, slots=True)
class QuestionBoundMapWork:
    gate_plan_receipt_sha256: str
    parent_identity_sha256: str
    question_id: str
    question_sha256: str
    dated_question: str
    dated_question_sha256: str
    route_receipt_sha256: str
    obligations: tuple[QuestionObligation, ...]
    namespace_id: str
    source_id: str
    membership_projection_sha256: str
    stream_sha256: str
    source_history_receipt_sha256: str
    history_window_ordinal: int
    history_window_token_cap: int
    content_token_proxy: int
    chunks: tuple[FrozenHistoryChunk, ...]
    mapper_contract_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.gate_plan_receipt_sha256, "work gate plan"), (self.parent_identity_sha256, "work parent"),
            (self.question_sha256, "work question"), (self.dated_question_sha256, "work dated question"),
            (self.route_receipt_sha256, "work route"), (self.namespace_id, "work namespace"),
            (self.membership_projection_sha256, "work membership"), (self.stream_sha256, "work stream"),
            (self.source_history_receipt_sha256, "work history"), (self.mapper_contract_sha256, "work mapper contract"),
        ):
            require_sha256(value, label)
        require_text(self.question_id, "work question ID")
        require_text(self.dated_question, "work dated question")
        _require(quote_sha256(self.dated_question) == self.dated_question_sha256, "work dated question text changed")
        require_text(self.source_id, "work source ID")
        _typed(self.obligations, QuestionObligation, "work obligations")
        _require(bool(self.obligations), "work requires obligations")
        _int(self.history_window_ordinal, "work window ordinal")
        _int(self.history_window_token_cap, "work window cap", 1)
        _int(self.content_token_proxy, "work content tokens")
        _typed(self.chunks, FrozenHistoryChunk, "work chunks")
        _require(bool(self.chunks) and all(row.source_id == self.source_id for row in self.chunks), "work chunks escaped source")
        _require(sum(row.token_count for row in self.chunks) == self.content_token_proxy <= self.history_window_token_cap, "work token accounting changed")

    def mapping_payload(self) -> dict[str, Any]:
        value = {
            "chunks": [row.projection(include_text=True) for row in self.chunks],
            "content_token_proxy": self.content_token_proxy,
            "dated_question": self.dated_question,
            "dated_question_sha256": self.dated_question_sha256,
            "format": f"{FORMAT}-question-bound-map-work",
            "frozen_chunk_boundaries": True,
            "gate_plan_receipt_sha256": self.gate_plan_receipt_sha256,
            "history_window_ordinal": self.history_window_ordinal,
            "history_window_token_cap": self.history_window_token_cap,
            "mapper_contract_sha256": self.mapper_contract_sha256,
            "membership_projection_sha256": self.membership_projection_sha256,
            "namespace_id": self.namespace_id,
            "obligations": [row.projection() for row in self.obligations],
            "parent_identity_sha256": self.parent_identity_sha256,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "route_receipt_sha256": self.route_receipt_sha256,
            "source_history_receipt_sha256": self.source_history_receipt_sha256,
            "source_id": self.source_id,
            "stream_sha256": self.stream_sha256,
        }
        assert_gold_blind(value, path="question_bound_map_work")
        return value

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.mapping_payload())

    @property
    def work_id(self) -> str:
        return self.receipt_sha256


@dataclass(frozen=True, slots=True)
class MapWorkAlias:
    physical_work_id: str
    hydration_plan_receipt_sha256: str
    window_id: str
    window_receipt_sha256: str
    mapping_payload_sha256: str
    selection_id: str
    lane: FactLane

    def __post_init__(self) -> None:
        for value, label in (
            (self.physical_work_id, "alias work"), (self.hydration_plan_receipt_sha256, "alias hydration"),
            (self.window_id, "alias window"), (self.window_receipt_sha256, "alias window receipt"),
            (self.mapping_payload_sha256, "alias core payload"),
        ):
            require_sha256(value, label)
        require_text(self.selection_id, "alias selection")
        _require(type(self.lane) is FactLane, "alias lane changed")

    def _body(self) -> dict[str, Any]:
        return {
            "hydration_plan_receipt_sha256": self.hydration_plan_receipt_sha256,
            "lane": self.lane.value,
            "mapping_payload_sha256": self.mapping_payload_sha256,
            "physical_work_id": self.physical_work_id,
            "selection_id": self.selection_id,
            "window_id": self.window_id,
            "window_receipt_sha256": self.window_receipt_sha256,
        }

    @property
    def alias_receipt_sha256(self) -> str:
        return _seal("map-alias", self._body())

    def projection(self) -> dict[str, Any]:
        return _record("map-alias", self._body(), alias_receipt_sha256=self.alias_receipt_sha256)


@dataclass(frozen=True, slots=True)
class QuestionBoundMappingPlan(_SealedRecord):
    _kind = "mapping-plan"
    gate_plan_receipt_sha256: str
    gate_round_receipt_sha256: str
    hydration_plan_receipt_sha256: str
    work_items: tuple[QuestionBoundMapWork, ...]
    aliases: tuple[MapWorkAlias, ...]
    reused_work_ids: tuple[str, ...]
    new_call_work_ids: tuple[str, ...]
    deferred_work_ids: tuple[str, ...]
    prior_call_work_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in ((self.gate_plan_receipt_sha256, "mapping gate plan"), (self.gate_round_receipt_sha256, "mapping round"), (self.hydration_plan_receipt_sha256, "mapping hydration")):
            require_sha256(value, label)
        _typed(self.work_items, QuestionBoundMapWork, "map work")
        _typed(self.aliases, MapWorkAlias, "map aliases")
        work_ids = tuple(row.work_id for row in self.work_items)
        _require(len(work_ids) == len(set(work_ids)), "physical work IDs repeat")
        for values, label in ((self.reused_work_ids, "reused work"), (self.new_call_work_ids, "new-call work"), (self.deferred_work_ids, "deferred work"), (self.prior_call_work_ids, "prior-call work")):
            _unique_sha(values, label)
        groups = (set(self.reused_work_ids), set(self.new_call_work_ids), set(self.deferred_work_ids))
        _require(set.union(*groups) == set(work_ids) and not any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))), "physical work lifecycle changed")
        _require(all(row.physical_work_id in set(work_ids) and row.hydration_plan_receipt_sha256 == self.hydration_plan_receipt_sha256 for row in self.aliases), "alias escaped work/hydration plan")

    @property
    def planned_provider_calls(self) -> int:
        return len(self.new_call_work_ids)

    def _body(self) -> dict[str, Any]:
        return {
            "aliases": [row.projection() for row in self.aliases],
            "deferred_work_ids": list(self.deferred_work_ids),
            "gate_plan_receipt_sha256": self.gate_plan_receipt_sha256,
            "gate_round_receipt_sha256": self.gate_round_receipt_sha256,
            "hydration_plan_receipt_sha256": self.hydration_plan_receipt_sha256,
            "new_call_work_ids": list(self.new_call_work_ids),
            "planned_provider_calls": self.planned_provider_calls,
            "prior_call_work_ids": list(self.prior_call_work_ids),
            "provider_calls": 0,
            "reused_work_ids": list(self.reused_work_ids),
            "work_item_receipt_sha256s": [row.work_id for row in self.work_items],
        }

def _work(gate: SourceGatePlan, candidate: SourceGateCandidate, window: SourceHistoryWindow, mapper_contract_sha256: str) -> QuestionBoundMapWork:
    return QuestionBoundMapWork(
        gate.receipt_sha256, gate.parent.identity_sha256, gate.question_id, gate.question_sha256,
        gate.dated_question, gate.dated_question_sha256, gate.route.receipt_sha256, gate.obligations,
        window.selection.namespace_id, window.selection.source_id, candidate.membership_projection_sha256,
        candidate.stream_sha256, window.history_receipt_sha256, window.window_ordinal, window.token_cap,
        window.content_token_proxy, window.chunks, mapper_contract_sha256,
    )


def build_question_bound_mapping_plan(
    gate: SourceGatePlan,
    round_plan: SourceGateRound,
    hydration_plan: SourceHistoryHydrationPlan,
    *,
    mapper_contract_sha256: str,
    cached_work_ids: tuple[str, ...] = (),
    prior_call_work_ids: tuple[str, ...] = (),
) -> QuestionBoundMappingPlan:
    """Seal one round's exact physical work and lane-specific aliases."""
    if type(gate) is not SourceGatePlan or type(round_plan) is not SourceGateRound or type(hydration_plan) is not SourceHistoryHydrationPlan:
        raise TypeError("gate, round_plan, and hydration_plan must be exact types")
    require_sha256(mapper_contract_sha256, "mapper contract")
    _unique_sha(cached_work_ids, "cached work IDs")
    _unique_sha(prior_call_work_ids, "prior call work IDs")
    _require(round_plan.gate_plan_receipt_sha256 == gate.receipt_sha256, "mapping round escaped gate")
    _require(hydration_plan.parent == gate.parent and hydration_plan.selections == round_plan.selections, "hydration changed parent or round selections")
    candidate_by_selection = {selection.selection_id: candidate for selection, candidate in zip(round_plan.selections, round_plan.selected_candidates, strict=True)}
    histories = {NamespacedSourceKey(row.namespace_id, row.source_id): row for row in hydration_plan.histories}
    for selection in round_plan.selections:
        candidate = candidate_by_selection[selection.selection_id]
        history = histories.get(NamespacedSourceKey(selection.namespace_id, selection.source_id))
        _require(history is not None and history.membership_projection_sha256 == candidate.membership_projection_sha256 and history.stream_sha256 == candidate.stream_sha256, "mapping history differs from sealed membership provenance")
    work_by_id: dict[str, QuestionBoundMapWork] = {}
    aliases: list[MapWorkAlias] = []
    for window in hydration_plan.windows:
        candidate = candidate_by_selection.get(window.selection.selection_id)
        _require(candidate is not None, "window escaped selected candidates")
        item = _work(gate, candidate, window, mapper_contract_sha256)
        _require(work_by_id.setdefault(item.work_id, item) == item, "physical work identity collided")
        aliases.append(MapWorkAlias(item.work_id, hydration_plan.receipt_sha256, window.window_id, window.receipt_sha256, window.mapping_payload_sha256, window.selection.selection_id, window.selection.lane))
    work_items = tuple(work_by_id.values())
    cache = set(cached_work_ids) | set(prior_call_work_ids)
    reused = tuple(row.work_id for row in work_items if row.work_id in cache)
    pending = tuple(row.work_id for row in work_items if row.work_id not in cache)
    remaining = max(0, gate.policy.max_physical_map_calls - len(set(prior_call_work_ids)))
    return QuestionBoundMappingPlan(gate.receipt_sha256, round_plan.receipt_sha256, hydration_plan.receipt_sha256, work_items, tuple(aliases), reused, pending[:remaining], pending[remaining:], prior_call_work_ids)


def validate_question_bound_completion(
    hydration_plan: SourceHistoryHydrationPlan,
    mapping_plan: QuestionBoundMappingPlan,
    *,
    physical_work_id: str,
    completion: str,
) -> tuple[MappedFactBatch, ...]:
    """Fan one called/cached completion out to every logical window alias."""
    if type(hydration_plan) is not SourceHistoryHydrationPlan or type(mapping_plan) is not QuestionBoundMappingPlan:
        raise TypeError("hydration_plan and mapping_plan must be exact types")
    require_sha256(physical_work_id, "physical work ID")
    _require(type(completion) is str, "completion must be exact text")
    _require(mapping_plan.hydration_plan_receipt_sha256 == hydration_plan.receipt_sha256, "mapping changed hydration binding")
    _require(physical_work_id in set(mapping_plan.reused_work_ids) | set(mapping_plan.new_call_work_ids), "completion is not called or cached work")
    aliases = tuple(row for row in mapping_plan.aliases if row.physical_work_id == physical_work_id)
    _require(bool(aliases), "completion lacks logical aliases")
    windows = {row.window_id: row for row in hydration_plan.windows}
    result: list[MappedFactBatch] = []
    for alias in aliases:
        window = windows.get(alias.window_id)
        _require(window is not None and window.receipt_sha256 == alias.window_receipt_sha256 and window.mapping_payload_sha256 == alias.mapping_payload_sha256, "alias changed core window")
        result.append(validate_mapper_completion(hydration_plan, window, completion))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CoverageFact(_SealedRecord):
    _kind = "coverage-fact"
    fact_id: str
    fact_variants: tuple[str, ...]
    source_keys: tuple[NamespacedSourceKey, ...]
    event_tuple: EventTuple | None
    provenance_receipt_sha256: str

    def __post_init__(self) -> None:
        require_text(self.fact_id, "coverage fact ID")
        _unique_text(self.fact_variants, "coverage fact variants")
        _require(bool(self.fact_variants), "coverage fact requires text")
        _typed(self.source_keys, NamespacedSourceKey, "coverage sources")
        _require(bool(self.source_keys) and len(set(self.source_keys)) == len(self.source_keys), "coverage sources empty or repeated")
        _require(self.event_tuple is None or type(self.event_tuple) is EventTuple, "coverage event changed type")
        require_sha256(self.provenance_receipt_sha256, "coverage provenance")

    def _body(self) -> dict[str, Any]:
        return {
            "event_tuple": None if self.event_tuple is None else self.event_tuple.projection(),
            "fact_id": self.fact_id,
            "fact_variants": list(self.fact_variants),
            "provenance_receipt_sha256": self.provenance_receipt_sha256,
            "source_keys": [row.projection() for row in self.source_keys],
        }

def coverage_facts_from_fact_union(fact_union: PostMapFactUnion) -> tuple[CoverageFact, ...]:
    if type(fact_union) is not PostMapFactUnion:
        raise TypeError("fact_union must be an exact PostMapFactUnion")
    result: list[CoverageFact] = []
    for fact in fact_union.retained_facts:
        keys = tuple(dict.fromkeys(NamespacedSourceKey(row.namespace_id, row.source_id) for row in fact.origins))
        result.append(CoverageFact(fact.union_fact_id, fact.fact_variants, keys, fact.event_tuple, fact.receipt_sha256))
    return tuple(result)


class ObligationState(str, Enum):
    UNRESOLVED = "unresolved"
    PARTIAL = "partial"
    CONFLICTED = "conflicted"
    SATISFIED = "satisfied"


@dataclass(frozen=True, slots=True)
class ObligationCoverage(_SealedRecord):
    _kind = "obligation-coverage"
    obligation_id: str
    state: ObligationState
    supporting_fact_ids: tuple[str, ...]
    supporting_source_keys: tuple[NamespacedSourceKey, ...]
    matched_term_count: int
    reason: str

    def __post_init__(self) -> None:
        require_sha256(self.obligation_id, "coverage obligation")
        _require(type(self.state) is ObligationState, "coverage state changed")
        _unique_text(self.supporting_fact_ids, "supporting facts")
        _typed(self.supporting_source_keys, NamespacedSourceKey, "supporting sources")
        _require(len(set(self.supporting_source_keys)) == len(self.supporting_source_keys), "supporting sources repeat")
        _int(self.matched_term_count, "matched term count")
        require_text(self.reason, "coverage reason")

    def _body(self) -> dict[str, Any]:
        return {
            "matched_term_count": self.matched_term_count,
            "obligation_id": self.obligation_id,
            "reason": self.reason,
            "state": self.state.value,
            "supporting_fact_ids": list(self.supporting_fact_ids),
            "supporting_source_keys": [row.projection() for row in self.supporting_source_keys],
        }

@dataclass(frozen=True, slots=True)
class ObligationCoverageReceipt(_SealedRecord):
    _kind = "coverage"
    gate_plan_receipt_sha256: str
    gate_round_receipt_sha256: str
    prior_coverage_receipt_sha256: str | None
    coverage_fact_receipt_sha256s: tuple[str, ...]
    mapping_plan_receipt_sha256s: tuple[str, ...]
    cumulative_physical_work_call_ids: tuple[str, ...]
    pending_physical_work_ids: tuple[str, ...]
    frontier_closed: bool
    obligations: tuple[ObligationCoverage, ...]
    made_progress: bool

    def __post_init__(self) -> None:
        require_sha256(self.gate_plan_receipt_sha256, "coverage gate plan")
        require_sha256(self.gate_round_receipt_sha256, "coverage gate round")
        if self.prior_coverage_receipt_sha256 is not None:
            require_sha256(self.prior_coverage_receipt_sha256, "prior coverage")
        for values, label in (
            (self.coverage_fact_receipt_sha256s, "coverage facts"), (self.mapping_plan_receipt_sha256s, "coverage mapping plans"),
            (self.cumulative_physical_work_call_ids, "coverage physical calls"), (self.pending_physical_work_ids, "coverage pending work"),
        ):
            _unique_sha(values, label)
        _require(type(self.frontier_closed) is bool and type(self.made_progress) is bool, "coverage flags changed")
        _typed(self.obligations, ObligationCoverage, "obligation coverage")
        _require(bool(self.obligations), "coverage requires obligations")

    @property
    def all_satisfied(self) -> bool:
        return all(row.state is ObligationState.SATISFIED for row in self.obligations)

    @property
    def unresolved_obligation_ids(self) -> tuple[str, ...]:
        return tuple(row.obligation_id for row in self.obligations if row.state is not ObligationState.SATISFIED)

    def _body(self) -> dict[str, Any]:
        return {
            "coverage_fact_receipt_sha256s": list(self.coverage_fact_receipt_sha256s),
            "cumulative_physical_work_call_ids": list(self.cumulative_physical_work_call_ids),
            "frontier_closed": self.frontier_closed,
            "gate_plan_receipt_sha256": self.gate_plan_receipt_sha256,
            "gate_round_receipt_sha256": self.gate_round_receipt_sha256,
            "made_progress": self.made_progress,
            "mapping_plan_receipt_sha256s": list(self.mapping_plan_receipt_sha256s),
            "obligations": [row.projection() for row in self.obligations],
            "pending_physical_work_ids": list(self.pending_physical_work_ids),
            "prior_coverage_receipt_sha256": self.prior_coverage_receipt_sha256,
        }

def _search_text(fact: CoverageFact) -> str:
    values = list(fact.fact_variants)
    if fact.event_tuple is not None:
        values.extend(fact.event_tuple.projection().values())
    return _norm(" ".join(values))


def _current_conflict(facts: Sequence[CoverageFact]) -> bool:
    values: dict[tuple[str, str], set[str]] = {}
    for fact in facts:
        event = fact.event_tuple
        if event is not None and _norm(event.status) == "current":
            values.setdefault((_norm(event.subject), _norm(event.predicate)), set()).add(_norm(event.object_value))
    return any(len(objects) > 1 for objects in values.values())


def _frontier_closed(plan: SourceGatePlan, round_plan: SourceGateRound, pending: tuple[str, ...]) -> bool:
    return plan.eligible_frontier.exhaustive and not pending and set(plan.eligible_frontier.eligible_candidate_ids) <= set(round_plan.cumulative_selected_candidate_ids)


def assess_obligation_coverage(
    plan: SourceGatePlan,
    round_plan: SourceGateRound,
    facts: tuple[CoverageFact, ...],
    *,
    mapping_plan_receipt_sha256s: tuple[str, ...] = (),
    cumulative_physical_work_call_ids: tuple[str, ...] = (),
    pending_physical_work_ids: tuple[str, ...] = (),
    previous: ObligationCoverageReceipt | None = None,
) -> ObligationCoverageReceipt:
    """Assess exact grounded support; capped absence never proves completeness."""
    if type(plan) is not SourceGatePlan or type(round_plan) is not SourceGateRound:
        raise TypeError("plan and round_plan must be exact types")
    _typed(facts, CoverageFact, "coverage facts")
    _require(len({row.fact_id for row in facts}) == len(facts), "coverage fact IDs repeat")
    _require(round_plan.gate_plan_receipt_sha256 == plan.receipt_sha256, "coverage round escaped gate")
    for values, label in ((mapping_plan_receipt_sha256s, "mapping receipts"), (cumulative_physical_work_call_ids, "physical calls"), (pending_physical_work_ids, "pending work")):
        _unique_sha(values, label)
    _require(len(cumulative_physical_work_call_ids) <= plan.policy.max_physical_map_calls, "coverage exceeds call cap")
    if previous is not None:
        _require(type(previous) is ObligationCoverageReceipt and previous.gate_plan_receipt_sha256 == plan.receipt_sha256, "prior coverage escaped gate")
        prior_support = {fact_id for row in previous.obligations for fact_id in row.supporting_fact_ids}
        _require(prior_support <= {row.fact_id for row in facts}, "coverage dropped prior support")
    frontier_closed = _frontier_closed(plan, round_plan, pending_physical_work_ids)
    results: list[ObligationCoverage] = []
    for obligation in plan.obligations:
        terms = tuple(_norm(row) for row in obligation.match_terms)
        matched, best = [], 0
        for fact in facts:
            count = sum(term in _search_text(fact) for term in terms)
            best = max(best, count)
            if count >= obligation.required_match_term_count:
                matched.append(fact)
        sources = tuple(dict.fromkeys(key for fact in matched for key in fact.source_keys))
        temporal = not obligation.requires_temporal_metadata or any(row.event_tuple is not None and row.event_tuple.event_time for row in matched)
        counts = len(matched) >= obligation.minimum_fact_count and len(sources) >= obligation.minimum_source_count
        complete = not obligation.requires_complete_frontier or frontier_closed
        if _current_conflict(matched):
            state, reason = ObligationState.CONFLICTED, "current_event_conflict"
        elif counts and temporal and complete:
            state, reason = ObligationState.SATISFIED, "grounded_thresholds_met"
        elif matched:
            state = ObligationState.PARTIAL
            reason = "grounded_count_incomplete" if not counts else "temporal_coordinate_missing" if not temporal else "frontier_scope_not_exhaustive" if not plan.eligible_frontier.exhaustive else "frontier_not_closed"
        else:
            state, reason = ObligationState.UNRESOLVED, "no_grounded_match"
        results.append(ObligationCoverage(obligation.obligation_id, state, tuple(row.fact_id for row in matched), sources, best, reason))
    if previous is None:
        progress = any(row.supporting_fact_ids for row in results)
    else:
        prior = {row.obligation_id: row for row in previous.obligations}
        rank = {ObligationState.UNRESOLVED: 0, ObligationState.PARTIAL: 1, ObligationState.CONFLICTED: 1, ObligationState.SATISFIED: 2}
        progress = any(set(row.supporting_fact_ids) - set(prior[row.obligation_id].supporting_fact_ids) or rank[row.state] > rank[prior[row.obligation_id].state] for row in results)
    return ObligationCoverageReceipt(
        plan.receipt_sha256, round_plan.receipt_sha256, None if previous is None else previous.receipt_sha256,
        tuple(row.receipt_sha256 for row in facts), mapping_plan_receipt_sha256s,
        cumulative_physical_work_call_ids, pending_physical_work_ids, frontier_closed, tuple(results), progress,
    )


_TAIL_ORDER: Mapping[RoutedRepairStyle, tuple[FactLane, ...]] = {
    RoutedRepairStyle.EXTRACT: (FactLane.DIRECT, FactLane.GUIDED, FactLane.PARTITION),
    RoutedRepairStyle.STATE_CHAIN: (FactLane.GUIDED, FactLane.DIRECT, FactLane.PARTITION),
    RoutedRepairStyle.TIMELINE: (FactLane.GUIDED, FactLane.DIRECT, FactLane.PARTITION),
    RoutedRepairStyle.NUMERIC_REDUCE: (FactLane.PARTITION, FactLane.GUIDED, FactLane.DIRECT),
    RoutedRepairStyle.SET_JOIN: (FactLane.PARTITION, FactLane.GUIDED, FactLane.DIRECT),
    RoutedRepairStyle.SYNTHESIZE: (FactLane.DIRECT, FactLane.GUIDED, FactLane.PARTITION),
}


class GateStopReason(str, Enum):
    SATISFIED = "all_obligations_satisfied"
    NO_PROGRESS = "no_progress_lane_pass"
    CANDIDATES_EXHAUSTED = "candidate_prefixes_exhausted"
    UNIQUE_SOURCE_CAP = "global_unique_source_cap"
    PHYSICAL_CALL_CAP = "physical_map_call_cap"
    ROUND_CAP = "round_cap"


@dataclass(frozen=True, slots=True)
class SourceGateStopReceipt(_SealedRecord):
    _kind = "stop"
    gate_plan_receipt_sha256: str
    final_round_receipt_sha256: str
    final_coverage_receipt_sha256: str
    reason: GateStopReason
    unresolved_obligation_ids: tuple[str, ...]
    cumulative_selected_candidate_ids: tuple[str, ...]
    cumulative_unique_source_count: int
    cumulative_physical_map_call_count: int

    def __post_init__(self) -> None:
        for value, label in ((self.gate_plan_receipt_sha256, "stop gate"), (self.final_round_receipt_sha256, "stop round"), (self.final_coverage_receipt_sha256, "stop coverage")):
            require_sha256(value, label)
        _require(type(self.reason) is GateStopReason, "stop reason changed")
        _unique_sha(self.unresolved_obligation_ids, "unresolved obligations")
        _unique_sha(self.cumulative_selected_candidate_ids, "stop candidates")
        _int(self.cumulative_unique_source_count, "stop unique sources")
        _int(self.cumulative_physical_map_call_count, "stop map calls")
        _require((self.reason is GateStopReason.SATISFIED) == (not self.unresolved_obligation_ids), "satisfied stop disagrees with unresolved obligations")

    def _body(self) -> dict[str, Any]:
        return {
            "cumulative_physical_map_call_count": self.cumulative_physical_map_call_count,
            "cumulative_selected_candidate_ids": list(self.cumulative_selected_candidate_ids),
            "cumulative_unique_source_count": self.cumulative_unique_source_count,
            "final_coverage_receipt_sha256": self.final_coverage_receipt_sha256,
            "final_round_receipt_sha256": self.final_round_receipt_sha256,
            "gate_plan_receipt_sha256": self.gate_plan_receipt_sha256,
            "reason": self.reason.value,
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
        }

def _stop(plan: SourceGatePlan, round_plan: SourceGateRound, coverage: ObligationCoverageReceipt, reason: GateStopReason) -> SourceGateStopReceipt:
    return SourceGateStopReceipt(plan.receipt_sha256, round_plan.receipt_sha256, coverage.receipt_sha256, reason, coverage.unresolved_obligation_ids, round_plan.cumulative_selected_candidate_ids, round_plan.cumulative_unique_source_count, len(coverage.cumulative_physical_work_call_ids))


def _next(plan: SourceGatePlan, lane: FactLane, selected: set[str]) -> tuple[SourceGateCandidate, ...]:
    budget = plan.policy.budget_for(lane)
    return tuple(row for row in plan.candidates_for(lane)[: budget.hard_source_cap] if row.candidate_id not in selected)


def _validate_lifecycle(plan: SourceGatePlan, rounds: tuple[SourceGateRound, ...], coverages: tuple[ObligationCoverageReceipt, ...]) -> None:
    _typed(rounds, SourceGateRound, "gate rounds")
    _typed(coverages, ObligationCoverageReceipt, "coverage receipts")
    _require(bool(rounds) and len(rounds) == len(coverages), "gate lifecycle requires one coverage per round")
    _require(rounds[0] == start_source_gate(plan), "base replay changed")
    for index, (round_plan, coverage) in enumerate(zip(rounds, coverages, strict=True)):
        _require(round_plan.gate_plan_receipt_sha256 == plan.receipt_sha256 and round_plan.round_index == index, "round order changed")
        _require(round_plan.prior_round_receipt_sha256 == (None if index == 0 else rounds[index - 1].receipt_sha256), "round chain changed")
        _require(coverage.gate_plan_receipt_sha256 == plan.receipt_sha256 and coverage.gate_round_receipt_sha256 == round_plan.receipt_sha256 and coverage.prior_coverage_receipt_sha256 == (None if index == 0 else coverages[index - 1].receipt_sha256), "coverage chain changed")


def advance_source_gate(plan: SourceGatePlan, rounds: tuple[SourceGateRound, ...], coverage: tuple[ObligationCoverageReceipt, ...]) -> SourceGateRound | SourceGateStopReceipt:
    """Select one route-specific tail prefix or seal a fail-closed stop."""
    if type(plan) is not SourceGatePlan:
        raise TypeError("plan must be an exact SourceGatePlan")
    _validate_lifecycle(plan, rounds, coverage)
    current, observed = rounds[-1], coverage[-1]
    if observed.all_satisfied:
        return _stop(plan, current, observed, GateStopReason.SATISFIED)
    if len(observed.cumulative_physical_work_call_ids) >= plan.policy.max_physical_map_calls:
        return _stop(plan, current, observed, GateStopReason.PHYSICAL_CALL_CAP)
    if len(rounds) >= plan.policy.max_rounds:
        return _stop(plan, current, observed, GateStopReason.ROUND_CAP)
    selected_ids = set(current.cumulative_selected_candidate_ids)
    available = {lane: _next(plan, lane, selected_ids) for lane in SOURCE_GATE_LANES}
    if not any(available.values()):
        return _stop(plan, current, observed, GateStopReason.CANDIDATES_EXHAUSTED)
    order = _TAIL_ORDER[plan.route.style]
    tail = tuple((row, result) for row, result in zip(rounds, coverage, strict=True) if row.kind is GateRoundKind.TAIL)
    if tail:
        last = tail[-1][0].tail_lane
        assert last is not None
        start = (order.index(last) + 1) % len(order)
        lane_order = order[start:] + order[:start]
    else:
        lane_order = order
    # A no-progress prefix does not exhaust its lane.  Rotate fairly, then
    # revisit the lane while later candidates remain inside its hard cap.
    lanes = tuple(lane for lane in lane_order if available[lane])
    if not lanes:
        return _stop(plan, current, observed, GateStopReason.CANDIDATES_EXHAUSTED)
    initial_known = {
        plan.candidate_by_id(value).source_key
        for value in current.cumulative_selected_candidate_ids
    }
    for lane in lanes:
        known = set(initial_known)
        remaining = plan.policy.global_unique_source_cap - len(known)
        chosen: list[SourceGateCandidate] = []
        for candidate in available[lane]:
            new = candidate.source_key not in known
            if new and remaining == 0:
                # A later candidate in this lane, or a later lane in the fair
                # rotation, may already be known and cost no unique capacity.
                continue
            chosen.append(candidate)
            if new:
                known.add(candidate.source_key)
                remaining -= 1
            if len(chosen) == plan.policy.budget_for(lane).tail_step_source_cap:
                break
        if chosen:
            return _make_round(
                plan,
                current,
                GateRoundKind.TAIL,
                lane,
                tuple(chosen),
            )
    return _stop(plan, current, observed, GateStopReason.UNIQUE_SOURCE_CAP)


@dataclass(frozen=True, slots=True)
class SourceGateReplayReceipt(_SealedRecord):
    _kind = "replay"
    gate_plan_receipt_sha256: str
    round_receipt_sha256s: tuple[str, ...]
    coverage_receipt_sha256s: tuple[str, ...]
    stop_receipt_sha256: str
    byte_identical: bool = True

    def __post_init__(self) -> None:
        require_sha256(self.gate_plan_receipt_sha256, "replay gate")
        _unique_sha(self.round_receipt_sha256s, "replay rounds")
        _unique_sha(self.coverage_receipt_sha256s, "replay coverage")
        require_sha256(self.stop_receipt_sha256, "replay stop")
        _require(self.byte_identical is True, "replay must be byte-identical")

    def _body(self) -> dict[str, Any]:
        return {
            "byte_identical": self.byte_identical,
            "coverage_receipt_sha256s": list(self.coverage_receipt_sha256s),
            "gate_plan_receipt_sha256": self.gate_plan_receipt_sha256,
            "round_receipt_sha256s": list(self.round_receipt_sha256s),
            "stop_receipt_sha256": self.stop_receipt_sha256,
        }

def replay_source_gate(plan: SourceGatePlan, rounds: tuple[SourceGateRound, ...], coverage: tuple[ObligationCoverageReceipt, ...], expected_stop: SourceGateStopReceipt) -> SourceGateReplayReceipt:
    """Recompute every selection and stop from sealed gold-blind coverage."""
    if type(expected_stop) is not SourceGateStopReceipt:
        raise TypeError("expected_stop must be exact")
    _validate_lifecycle(plan, rounds, coverage)
    observed = [start_source_gate(plan)]
    _require(observed[0] == rounds[0], "base replay differs")
    stop: SourceGateStopReceipt | None = None
    for index in range(len(rounds)):
        decision = advance_source_gate(plan, tuple(observed), coverage[: index + 1])
        if type(decision) is SourceGateStopReceipt:
            _require(index == len(rounds) - 1, "replay stopped early")
            stop = decision
            break
        _require(index + 1 < len(rounds) and decision == rounds[index + 1], "tail replay differs")
        observed.append(decision)
    _require(stop == expected_stop, "stop replay differs")
    return SourceGateReplayReceipt(plan.receipt_sha256, tuple(row.receipt_sha256 for row in rounds), tuple(row.receipt_sha256 for row in coverage), expected_stop.receipt_sha256)


__all__ = [
    "FORMAT", "SOURCE_GATE_LANES", "CoverageFact", "EligibleFrontierScope",
    "GateRoundKind", "GateStopReason", "LaneSourceBudget", "MapWorkAlias",
    "NamespacedSourceKey", "ObligationCoverage", "ObligationCoverageReceipt",
    "ObligationKind", "ObligationState", "QuestionBoundMapWork",
    "QuestionBoundMappingPlan", "QuestionObligation", "SourceGateActivationReceipt",
    "SourceGateCandidate", "SourceGateControllerError", "SourceGatePlan",
    "SourceGatePolicy", "SourceGateReplayReceipt", "SourceGateRound",
    "SourceGateStopReceipt", "advance_source_gate", "assess_obligation_coverage",
    "build_question_bound_mapping_plan", "coverage_facts_from_fact_union",
    "default_source_gate_policy", "replay_source_gate", "start_source_gate",
    "validate_question_bound_completion",
]
