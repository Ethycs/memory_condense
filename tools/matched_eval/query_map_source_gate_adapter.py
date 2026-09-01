"""Provider-free query-plan/map coverage adapter for the adaptive source gate.

The sealed query-expansion row is the question-only obligation authority. A
terminal V2 evidence-map plane is only a bounded grounded frontier: accepted
items have exact citations, but it has neither structured event coordinates
nor an exhaustive-source witness. This adapter checks the two parent seals,
compiles compact obligations, and activates source hydration exactly when the
validated map cannot mechanically discharge every obligation.

The final V2 solver is deliberately absent. Source hydration belongs after
the terminal map and before the answer solver; judge, gold, reference, and
prediction values are neither accepted nor read here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256
from tools._routed_repair_routing import route_question

from .artifacts import SealedArtifact
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .query_evidence_map_solver_v2_live import (
    MAP_ITEM_FORMAT,
    MAP_PARSE_FORMAT,
    MAP_REJECT_FORMAT,
    EvidenceMapPlan,
    RejectedMapItem,
    ValidatedMapItem,
    VerifiedEvidenceMapPlane,
)
from .query_expansion import (
    RUN_FORMAT as QUERY_RUN_FORMAT,
    QueryExpansionBudget,
    QueryPlan,
    parse_query_plan,
)
from .source_gate_controller import (
    ObligationKind,
    QuestionObligation,
    SourceGateActivationReceipt,
)


FORMAT = "memory-condense-query-map-source-gate-adapter-v1"
LEGACY_OBLIGATION_MODE = "entity_per_support_v1"
CONSOLIDATED_OBLIGATION_MODE = "consolidated_any_anchor_parent_verified_v2"
PARENT_VERIFICATION_RULE_ID = "normalized_bidirectional_containment_v1"
OBLIGATION_MODES = frozenset(
    {LEGACY_OBLIGATION_MODE, CONSOLIDATED_OBLIGATION_MODE}
)
STRICT_STATE_CHAIN_PROFILE = "strict_state_chain_source_verification_v1"
STATE_CHAIN_DIRECT_AUTHORITY_PROFILE = "state_chain_direct_authority_v1"
STATE_CHAIN_PROFILES = frozenset(
    {STRICT_STATE_CHAIN_PROFILE, STATE_CHAIN_DIRECT_AUTHORITY_PROFILE}
)
_TEMPORAL_OPERATORS = frozenset(
    {"timeline", "earliest", "latest", "before_after", "state_transition"}
)
_FRONTIER_OPERATORS = frozenset(
    {"enumerate_repeated_events", "count_distinct", "earliest", "latest"}
)
_PAIR_FACT_OPERATORS = frozenset({"before_after", "state_transition"})


class QueryMapSourceGateAdapterError(MatchedEvalContractError):
    """A sealed query-plan/map parent or mechanical coverage proof changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise QueryMapSourceGateAdapterError(message)


def _zero(value: object, label: str) -> None:
    _require(type(value) is int and value == 0, f"{label} must be exact zero")


def _typed(values: object, cls: type, label: str) -> tuple[Any, ...]:
    _require(
        type(values) is tuple and all(type(row) is cls for row in values),
        f"{label} must be an immutable exact-{cls.__name__} tuple",
    )
    return values  # type: ignore[return-value]


def _seal(kind: str, body: Mapping[str, Any]) -> str:
    value = {"format": f"{FORMAT}-{kind}", **body}
    assert_gold_blind(value, path="query_map_source_gate_adapter")
    return identity_sha256(value)


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _unique_terms(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        require_text(value, "query-plan obligation term")
        key = _norm(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ParentPredictionVerification:
    rule_id: str
    parent_prediction_sha256: str
    accepted_item_sha256s: tuple[str, ...]
    agreeing_item_ids: tuple[str, ...]
    agreement_required_for_support: bool
    mechanically_agrees: bool

    def __post_init__(self) -> None:
        _require(
            self.rule_id == PARENT_VERIFICATION_RULE_ID,
            "parent-verification rule changed",
        )
        require_sha256(self.parent_prediction_sha256, "parent prediction")
        for value in self.accepted_item_sha256s:
            require_sha256(value, "parent-verification accepted item")
        _require(
            len(self.accepted_item_sha256s)
            == len(set(self.accepted_item_sha256s)),
            "parent-verification accepted items repeat",
        )
        for value in self.agreeing_item_ids:
            require_text(value, "parent-verification agreeing item")
        _require(
            len(self.agreeing_item_ids) == len(set(self.agreeing_item_ids)),
            "parent-verification agreeing items repeat",
        )
        _require(
            type(self.agreement_required_for_support) is bool
            and self.agreement_required_for_support,
            "consolidated support must require parent verification",
        )
        _require(
            type(self.mechanically_agrees) is bool
            and self.mechanically_agrees == bool(self.agreeing_item_ids),
            "parent-verification disposition changed",
        )

    def projection(self) -> dict[str, Any]:
        return {
            "accepted_item_sha256s": list(self.accepted_item_sha256s),
            "agreement_required_for_support": self.agreement_required_for_support,
            "agreeing_item_ids": list(self.agreeing_item_ids),
            "format": f"{FORMAT}-parent-prediction-verification",
            "mechanically_agrees": self.mechanically_agrees,
            "parent_prediction_sha256": self.parent_prediction_sha256,
            "rule_id": self.rule_id,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class QueryMapSourceGateAdapterRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    source_packet_id: str
    map_packet_id: str
    query_row_receipt_sha256: str
    map_plan_row_receipt_sha256: str
    map_source_row_sha256: str
    map_parse_receipt_sha256: str
    upstream_question_plan_receipt_sha256: str
    upstream_fact_frontier_receipt_sha256: str
    obligations: tuple[QuestionObligation, ...]
    satisfied_obligation_ids: tuple[str, ...]
    unresolved_obligation_ids: tuple[str, ...]
    disposition: StageDisposition
    reason: str
    activation: SourceGateActivationReceipt | None
    obligation_compilation_mode: str = LEGACY_OBLIGATION_MODE
    parent_prediction_verification: ParentPredictionVerification | None = None
    state_chain_profile: str = STRICT_STATE_CHAIN_PROFILE
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        _require(type(self.ordinal) is int and self.ordinal >= 0, "adapter ordinal changed")
        require_text(self.question_id, "adapter question ID")
        for value, label in (
            (self.question_sha256, "adapter question"),
            (self.dated_question_sha256, "adapter dated question"),
            (self.source_packet_id, "adapter source packet"),
            (self.map_packet_id, "adapter map packet"),
            (self.query_row_receipt_sha256, "adapter query row"),
            (self.map_plan_row_receipt_sha256, "adapter map plan row"),
            (self.map_source_row_sha256, "adapter map source row"),
            (self.map_parse_receipt_sha256, "adapter map parse"),
            (self.upstream_question_plan_receipt_sha256, "adapter question plan"),
            (self.upstream_fact_frontier_receipt_sha256, "adapter fact frontier"),
        ):
            require_sha256(value, label)
        _require(
            self.source_packet_id != self.map_packet_id,
            "adapter collapsed source and post-map packet identities",
        )
        obligations = _typed(self.obligations, QuestionObligation, "adapter obligations")
        _require(bool(obligations), "adapter requires query-plan obligations")
        all_ids = tuple(row.obligation_id for row in obligations)
        _require(len(set(all_ids)) == len(all_ids), "adapter obligations repeat")
        for values, label in (
            (self.satisfied_obligation_ids, "satisfied obligations"),
            (self.unresolved_obligation_ids, "unresolved obligations"),
        ):
            _require(type(values) is tuple, f"{label} must be an immutable tuple")
            for value in values:
                require_sha256(value, label)
            _require(len(set(values)) == len(values), f"{label} repeat")
        _require(
            self.satisfied_obligation_ids
            == tuple(value for value in all_ids if value not in self.unresolved_obligation_ids)
            and self.unresolved_obligation_ids
            == tuple(value for value in all_ids if value not in self.satisfied_obligation_ids),
            "adapter obligation states do not partition the question plan",
        )
        _require(type(self.disposition) is StageDisposition, "adapter disposition changed")
        require_text(self.reason, "adapter reason")
        _require(
            self.obligation_compilation_mode in OBLIGATION_MODES,
            "adapter obligation compilation mode changed",
        )
        if self.obligation_compilation_mode == LEGACY_OBLIGATION_MODE:
            _require(
                self.parent_prediction_verification is None,
                "legacy obligation mode cannot carry parent verification",
            )
        else:
            _require(
                type(self.parent_prediction_verification)
                is ParentPredictionVerification,
                "consolidated obligation mode requires parent verification",
            )
        _require(
            self.state_chain_profile in STATE_CHAIN_PROFILES,
            "adapter state-chain profile changed",
        )
        _zero(self.provider_calls, "adapter provider calls")
        _zero(self.retained_transformer_token_state_bytes, "adapter retained token state")
        if self.unresolved_obligation_ids:
            _require(
                self.disposition is StageDisposition.ADDED
                and type(self.activation) is SourceGateActivationReceipt,
                "unresolved adapter row must activate",
            )
            assert self.activation is not None
            _require(
                self.activation.question_id == self.question_id
                and self.activation.question_sha256 == self.question_sha256
                and self.activation.dated_question_sha256 == self.dated_question_sha256
                and self.activation.parent_packet_id == self.map_packet_id
                and self.activation.upstream_question_plan_receipt_sha256
                == self.upstream_question_plan_receipt_sha256
                and self.activation.upstream_fact_frontier_receipt_sha256
                == self.upstream_fact_frontier_receipt_sha256
                and self.activation.obligation_ids == all_ids
                and self.activation.unresolved_obligation_ids
                == self.unresolved_obligation_ids,
                "activation changed its query-plan/map binding",
            )
        else:
            _require(
                self.disposition is StageDisposition.NO_OP and self.activation is None,
                "satisfied adapter row must be a no-op",
            )
        assert_gold_blind(self.projection(), path="query_map_source_gate_adapter_row")

    @property
    def activated(self) -> bool:
        return self.activation is not None

    @property
    def parent_packet_id(self) -> str:
        """The post-map packet is the source-fact/solver parent identity."""

        return self.map_packet_id

    def projection(self) -> dict[str, Any]:
        value = {
            "activation": None if self.activation is None else self.activation.projection(),
            "dated_question_sha256": self.dated_question_sha256,
            "disposition": self.disposition.value,
            "format": f"{FORMAT}-row",
            "gold_loaded": False,
            "map_parse_receipt_sha256": self.map_parse_receipt_sha256,
            "map_plan_row_receipt_sha256": self.map_plan_row_receipt_sha256,
            "map_source_row_sha256": self.map_source_row_sha256,
            "obligations": [row.projection() for row in self.obligations],
            "ordinal": self.ordinal,
            "map_packet_id": self.map_packet_id,
            "parent_packet_id": self.parent_packet_id,
            "provider_calls": self.provider_calls,
            "query_row_receipt_sha256": self.query_row_receipt_sha256,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "reason": self.reason,
            "retained_transformer_token_state_bytes": self.retained_transformer_token_state_bytes,
            "satisfied_obligation_ids": list(self.satisfied_obligation_ids),
            "source_packet_id": self.source_packet_id,
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
            "upstream_fact_frontier_receipt_sha256": self.upstream_fact_frontier_receipt_sha256,
            "upstream_question_plan_receipt_sha256": self.upstream_question_plan_receipt_sha256,
        }
        if self.obligation_compilation_mode != LEGACY_OBLIGATION_MODE:
            assert self.parent_prediction_verification is not None
            value.update(
                {
                    "obligation_compilation_mode": self.obligation_compilation_mode,
                    "parent_prediction_verification": (
                        self.parent_prediction_verification.projection()
                    ),
                    "parent_prediction_verification_receipt_sha256": (
                        self.parent_prediction_verification.receipt_sha256
                    ),
                }
            )
        if self.state_chain_profile != STRICT_STATE_CHAIN_PROFILE:
            value["state_chain_profile"] = self.state_chain_profile
            value["state_chain_direct_authority_applied"] = (
                self.reason == "state_chain_direct_authority_no_source_hydration"
            )
        return value

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class QueryMapSourceGateAdapterPlane:
    query_run_sha256: str
    map_plan_identity_sha256: str
    map_run_sha256: str
    map_runtime_ledger_sha256: str
    snapshot_id: str
    rows: tuple[QueryMapSourceGateAdapterRow, ...]
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        for value, label in (
            (self.query_run_sha256, "adapter query run"),
            (self.map_plan_identity_sha256, "adapter map plan"),
            (self.map_run_sha256, "adapter map run"),
            (self.map_runtime_ledger_sha256, "adapter map runtime"),
            (self.snapshot_id, "adapter snapshot"),
        ):
            require_sha256(value, label)
        rows = _typed(self.rows, QueryMapSourceGateAdapterRow, "adapter rows")
        _require(bool(rows), "adapter plane requires rows")
        _require(
            tuple(row.ordinal for row in rows) == tuple(sorted({row.ordinal for row in rows}))
            and len({row.question_id for row in rows}) == len(rows),
            "adapter rows repeat or changed order",
        )
        _require(
            len({row.obligation_compilation_mode for row in rows}) == 1,
            "adapter plane mixed obligation compilation modes",
        )
        _require(
            len({row.state_chain_profile for row in rows}) == 1,
            "adapter plane mixed state-chain profiles",
        )
        _zero(self.provider_calls, "adapter plane provider calls")
        _zero(self.retained_transformer_token_state_bytes, "adapter plane retained token state")
        assert_gold_blind(self.projection(), path="query_map_source_gate_adapter_plane")

    @property
    def activated_rows(self) -> tuple[QueryMapSourceGateAdapterRow, ...]:
        return tuple(row for row in self.rows if row.activated)

    @property
    def no_op_rows(self) -> tuple[QueryMapSourceGateAdapterRow, ...]:
        return tuple(row for row in self.rows if not row.activated)

    @property
    def obligation_compilation_mode(self) -> str:
        return self.rows[0].obligation_compilation_mode

    @property
    def state_chain_profile(self) -> str:
        return self.rows[0].state_chain_profile

    def projection(self) -> dict[str, Any]:
        value = {
            "format": f"{FORMAT}-plane",
            "gold_loaded": False,
            "map_plan_identity_sha256": self.map_plan_identity_sha256,
            "map_run_sha256": self.map_run_sha256,
            "map_runtime_ledger_sha256": self.map_runtime_ledger_sha256,
            "provider_calls": self.provider_calls,
            "query_run_sha256": self.query_run_sha256,
            "retained_transformer_token_state_bytes": self.retained_transformer_token_state_bytes,
            "row_receipt_sha256s": [row.receipt_sha256 for row in self.rows],
            "snapshot_id": self.snapshot_id,
        }
        if self.obligation_compilation_mode != LEGACY_OBLIGATION_MODE:
            value["obligation_compilation_mode"] = self.obligation_compilation_mode
            value["parent_prediction_verification_rule_id"] = (
                PARENT_VERIFICATION_RULE_ID
            )
        if self.state_chain_profile != STRICT_STATE_CHAIN_PROFILE:
            value["state_chain_profile"] = self.state_chain_profile
            value["state_chain_direct_authority_row_receipt_sha256s"] = [
                row.receipt_sha256
                for row in self.rows
                if row.reason == "state_chain_direct_authority_no_source_hydration"
            ]
        return value

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


def _query_plans(
    query_run: SealedArtifact,
    map_plan: EvidenceMapPlan,
) -> tuple[tuple[QueryPlan, str], ...]:
    _require(
        sha256(canonical_json_bytes(query_run.payload)).hexdigest() == query_run.sha256,
        "query-run artifact seal changed",
    )
    payload = query_run.payload
    assert_gold_blind(payload, path="query_map_source_gate_adapter.query_run")
    population = map_plan.direct_plan.adapter_population
    _require(
        query_run.sha256 == population.query_run_sha256
        and payload.get("format") == QUERY_RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("query_population_id") == population.query_population_id
        and payload.get("source_population_id") == population.source_population.population_id,
        "query run changed its V2 adapter parent",
    )
    _zero(payload.get("retained_transformer_token_state_bytes"), "query-run retained token state")
    budget_raw = payload.get("budget")
    _require(type(budget_raw) is dict, "query-run budget changed")
    try:
        budget = QueryExpansionBudget(**budget_raw)
    except (TypeError, MatchedEvalContractError) as exc:
        raise QueryMapSourceGateAdapterError("query-run budget is invalid") from exc
    _require(budget.projection() == budget_raw, "query-run budget projection changed")
    raw_rows = payload.get("questions")
    _require(
        type(raw_rows) is list
        and payload.get("question_count") == len(raw_rows) == len(map_plan.rows),
        "query/map populations differ",
    )
    result: list[tuple[QueryPlan, str]] = []
    for planned, raw in zip(map_plan.rows, raw_rows, strict=True):
        _require(type(raw) is dict, "query-run row changed type")
        assert type(raw) is dict
        receipt = require_sha256(raw.get("receipt_sha256"), "query-run row receipt")
        unsigned = dict(raw)
        unsigned.pop("receipt_sha256", None)
        packet = planned.direct_plan_row.adapter.source.packet
        _require(
            receipt == planned.direct_plan_row.adapter.query_row_receipt_sha256
            and identity_sha256(unsigned) == receipt
            and raw.get("ordinal") == planned.ordinal
            and raw.get("question_id") == packet.question_id
            and raw.get("question_sha256") == packet.question_sha256
            and raw.get("dated_question_sha256") == packet.dated_question_sha256
            and raw.get("parent_packet_id") == packet.packet_id,
            "query-plan row changed its sealed V2 binding",
        )
        projection = raw.get("query_plan")
        _require(type(projection) is dict, "sealed query plan is missing or invalid")
        try:
            plan = parse_query_plan(
                json.dumps(
                    projection,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                budget=budget,
            )
        except (TypeError, ValueError, MatchedEvalContractError) as exc:
            raise QueryMapSourceGateAdapterError("sealed query plan is missing or invalid") from exc
        _require(plan.projection() == projection, "sealed query-plan projection changed")
        result.append((plan, receipt))
    return tuple(result)


def _validate_item(
    item: ValidatedMapItem,
    *,
    answer_kind: str,
    evidence_by_alias: Mapping[str, tuple[str, str]],
) -> str:
    _require(item.kind == answer_kind, "grounded map item changed answer kind")
    for value, label in (
        (item.item_id, "grounded map item ID"),
        (item.alias, "grounded map item alias"),
        (item.candidate, "grounded map item candidate"),
    ):
        require_text(value, label)
    _require(
        type(item.citation) is str and bool(_norm(item.citation)),
        "grounded map item citation changed",
    )
    _require(type(item.source_index) is int and item.source_index >= 0, "map item source index changed")
    evidence = evidence_by_alias.get(item.alias)
    _require(evidence is not None, "grounded map item escaped its sealed alias inventory")
    assert evidence is not None
    evidence_text, source_id = evidence
    expected_match = (
        "full_evidence"
        if item.citation == evidence_text
        else "normalized_contiguous_substring"
        if _norm(item.citation) in _norm(evidence_text)
        else None
    )
    _require(item.citation_match == expected_match, "grounded map citation changed")
    expected_sha = identity_sha256(
        {
            "alias": item.alias,
            "candidate": item.candidate,
            "citation": item.citation,
            "citation_match": item.citation_match,
            "format": MAP_ITEM_FORMAT,
            "item_id": item.item_id,
            "kind": item.kind,
            "source_index": item.source_index,
        }
    )
    _require(item.item_sha256 == expected_sha, "grounded map item seal changed")
    return source_id


def _validate_rejected(item: RejectedMapItem) -> None:
    _require(type(item.source_index) is int, "rejected map item source index changed")
    require_text(item.reason, "rejected map item reason")
    require_sha256(item.raw_item_sha256, "rejected raw map item")
    _require(
        item.rejection_sha256
        == identity_sha256(
            {
                "format": MAP_REJECT_FORMAT,
                "raw_item_sha256": item.raw_item_sha256,
                "reason": item.reason,
                "source_index": item.source_index,
            }
        ),
        "rejected map item seal changed",
    )


def _obligations(
    plan: QueryPlan,
    route: Any,
    *,
    mode: str,
) -> tuple[QuestionObligation, ...]:
    _require(mode in OBLIGATION_MODES, "obligation compilation mode changed")
    anchors = _unique_terms(plan.entities or plan.queries)
    _require(bool(anchors), "query plan cannot compile a grounded obligation")
    rows = (
        [QuestionObligation(ObligationKind.SUPPORT, anchors, 1)]
        if mode == CONSOLIDATED_OBLIGATION_MODE
        else [
            QuestionObligation(ObligationKind.SUPPORT, (term,), 1)
            for term in anchors
        ]
    )
    operators = frozenset(plan.operators)
    temporal = route.modifiers.requires_temporal_metadata or bool(
        operators & _TEMPORAL_OPERATORS
    )
    frontier = route.modifiers.requires_complete_frontier or bool(
        operators & _FRONTIER_OPERATORS
    )
    minimum = route.modifiers.cardinality or (
        2 if operators & _PAIR_FACT_OPERATORS else 1
    )
    if temporal or frontier or minimum > 1:
        operation_terms = _unique_terms((*anchors, *plan.dates))
        kind = (
            ObligationKind.TEMPORAL
            if temporal
            else ObligationKind.FRONTIER
            if frontier
            else ObligationKind.CARDINALITY
        )
        rows.append(
            QuestionObligation(
                kind,
                operation_terms,
                1,
                minimum_fact_count=minimum,
                minimum_source_count=1,
                requires_temporal_metadata=temporal,
                requires_complete_frontier=frontier,
            )
        )
    return tuple(rows)


def _parent_prediction_verification(
    parent_prediction: str,
    accepted: tuple[ValidatedMapItem, ...],
) -> ParentPredictionVerification:
    require_text(parent_prediction, "direct parent prediction")
    parent = _norm(parent_prediction)
    agreeing = tuple(
        item.item_id
        for item in accepted
        if (candidate := _norm(item.candidate))
        and (candidate in parent or parent in candidate)
    )
    return ParentPredictionVerification(
        PARENT_VERIFICATION_RULE_ID,
        quote_sha256(parent_prediction),
        tuple(item.item_sha256 for item in accepted),
        agreeing,
        True,
        bool(agreeing),
    )


def _satisfied(
    obligation: QuestionObligation,
    items: tuple[ValidatedMapItem, ...],
    source_by_item_id: Mapping[str, str],
) -> bool:
    terms = tuple(_norm(value) for value in obligation.match_terms)
    matched = tuple(
        item
        for item in items
        if sum(term in _norm(f"{item.candidate} {item.citation}") for term in terms)
        >= obligation.required_match_term_count
    )
    sources = {source_by_item_id[item.item_id] for item in matched}
    # V2 map items contain no structured event tuple and the map frontier is
    # bounded. Text that merely mentions a date cannot forge either proof.
    return (
        len(matched) >= obligation.minimum_fact_count
        and len(sources) >= obligation.minimum_source_count
        and not obligation.requires_temporal_metadata
        and not obligation.requires_complete_frontier
    )


def _uses_state_chain_direct_authority(
    *,
    route_style: str,
    map_submitted: bool,
    map_status: str,
    profile: str,
) -> bool:
    _require(profile in STATE_CHAIN_PROFILES, "state-chain profile changed")
    return (
        profile == STATE_CHAIN_DIRECT_AUTHORITY_PROFILE
        and route_style == "state_chain"
        and map_submitted is False
        and map_status == "not_submitted_state_chain"
    )


def adapt_query_map_solver_v2(
    query_run: SealedArtifact,
    map_plan: EvidenceMapPlan,
    map_plane: VerifiedEvidenceMapPlane,
    *,
    obligation_mode: str = LEGACY_OBLIGATION_MODE,
    state_chain_profile: str = STRICT_STATE_CHAIN_PROFILE,
) -> QueryMapSourceGateAdapterPlane:
    """Adapt sealed query plans plus the terminal map, before the final solver."""

    if type(query_run) is not SealedArtifact:
        raise TypeError("query_run must be an exact SealedArtifact")
    if type(map_plan) is not EvidenceMapPlan:
        raise TypeError("map_plan must be an exact EvidenceMapPlan")
    if type(map_plane) is not VerifiedEvidenceMapPlane:
        raise TypeError("map_plane must be an exact VerifiedEvidenceMapPlane")
    _require(obligation_mode in OBLIGATION_MODES, "obligation compilation mode changed")
    _require(state_chain_profile in STATE_CHAIN_PROFILES, "state-chain profile changed")
    plans = _query_plans(query_run, map_plan)
    _require(
        map_plane.run_sha256 == map_plane.replay_sha256
        and map_plane.parent_plane is map_plan.direct_plane
        and map_plane.parent_answer_run_sha256 == map_plan.direct_plane.run_sha256
        and map_plane.adapter_population_id
        == map_plan.direct_plan.adapter_population.population_id
        and map_plane.retrieval_sha256
        == map_plan.direct_plan.adapter_population.source_population.retrieval_sha256
        and map_plane.snapshot_id == map_plan.snapshot.snapshot_id
        and len(map_plane.rows) == len(map_plan.rows),
        "terminal map changed its V2 parent chain",
    )
    rows: list[QueryMapSourceGateAdapterRow] = []
    for planned, mapped, (query_plan, query_receipt) in zip(
        map_plan.rows, map_plane.rows, plans, strict=True
    ):
        packet = planned.direct_plan_row.adapter.source.packet
        route = route_question(packet.dated_question)
        _require(
            mapped.ordinal == planned.ordinal
            and mapped.question_id == packet.question_id
            and mapped.question_sha256 == packet.question_sha256
            and mapped.dated_question_sha256 == packet.dated_question_sha256
            and quote_sha256(packet.dated_question) == packet.dated_question_sha256
            and mapped.route_id == planned.route.style.value
            and route.receipt_sha256 == planned.route.receipt_sha256
            and mapped.map_plan_row_receipt_sha256 == planned.receipt_sha256,
            "map row changed its sealed question/route/plan binding",
        )
        require_sha256(planned.packet_id, "V2 map packet")
        require_sha256(mapped.source_row_sha256, "V2 map source row")
        aliases = planned.aliases
        evidence = (
            *planned.direct_plan_row.adapter.source.packet.protected_evidence,
            *planned.retained_query_delta,
        )
        _require(len(aliases) == len(evidence), "V2 map alias inventory changed")
        evidence_by_alias: dict[str, tuple[str, str]] = {}
        for alias, source in zip(aliases, evidence, strict=True):
            _require(
                alias.alias not in evidence_by_alias
                and alias.evidence_id == source.evidence_id
                and alias.source_id == source.source_id
                and alias.text_sha256 == quote_sha256(source.text),
                "V2 map alias/evidence binding changed",
            )
            evidence_by_alias[alias.alias] = (source.text, source.source_id)
        accepted = _typed(mapped.accepted_items, ValidatedMapItem, "V2 map items")
        rejected = _typed(mapped.rejected_items, RejectedMapItem, "V2 rejected map items")
        _require(
            tuple(item.item_id for item in accepted)
            == tuple(f"M{index:03d}" for index in range(1, len(accepted) + 1)),
            "V2 accepted map item order changed",
        )
        source_by_item_id: dict[str, str] = {}
        if accepted:
            _require(
                mapped.map_status == "validated_items" and mapped.answer_kind is not None,
                "grounded V2 items changed map status",
            )
            for item in accepted:
                source_by_item_id[item.item_id] = _validate_item(
                    item,
                    answer_kind=mapped.answer_kind,
                    evidence_by_alias=evidence_by_alias,
                )
        else:
            _require(
                mapped.map_status
                == ("no_valid_items" if planned.submitted else "not_submitted_state_chain"),
                "empty V2 map changed status",
            )
        for item in rejected:
            _validate_rejected(item)
        expected_parse = identity_sha256(
            {
                "accepted_item_sha256s": [item.item_sha256 for item in accepted],
                "format": MAP_PARSE_FORMAT,
                "rejected_item_sha256s": [item.rejection_sha256 for item in rejected],
            }
        )
        _require(mapped.map_parse_receipt_sha256 == expected_parse, "V2 map parse seal changed")

        obligations = _obligations(
            query_plan,
            planned.route,
            mode=obligation_mode,
        )
        parent_verification = (
            _parent_prediction_verification(
                planned.direct_answer_row.prediction,
                accepted,
            )
            if obligation_mode == CONSOLIDATED_OBLIGATION_MODE
            else None
        )
        state_chain_authority = _uses_state_chain_direct_authority(
            route_style=planned.route.style.value,
            map_submitted=planned.submitted,
            map_status=mapped.map_status,
            profile=state_chain_profile,
        )
        satisfied = (
            tuple(row.obligation_id for row in obligations)
            if state_chain_authority
            else tuple(
                row.obligation_id
                for row in obligations
                if _satisfied(row, accepted, source_by_item_id)
                and (
                    row.kind is not ObligationKind.SUPPORT
                    or parent_verification is None
                    or parent_verification.mechanically_agrees
                )
            )
        )
        unresolved = tuple(
            row.obligation_id for row in obligations if row.obligation_id not in satisfied
        )
        question_plan_receipt = _seal(
            "question-plan",
            {
                "query_plan": query_plan.projection(),
                "query_row_receipt_sha256": query_receipt,
                "route_receipt_sha256": planned.route.receipt_sha256,
                **(
                    {}
                    if obligation_mode == LEGACY_OBLIGATION_MODE
                    else {
                        "obligation_compilation_mode": obligation_mode,
                        "parent_prediction_verification_rule_id": (
                            PARENT_VERIFICATION_RULE_ID
                        ),
                    }
                ),
                **(
                    {}
                    if state_chain_profile == STRICT_STATE_CHAIN_PROFILE
                    else {"state_chain_profile": state_chain_profile}
                ),
            },
        )
        fact_frontier_receipt = _seal(
            "map-fact-frontier",
            {
                "accepted_item_sha256s": [item.item_sha256 for item in accepted],
                "bounded_frontier_exhaustive": False,
                "map_parse_receipt_sha256": mapped.map_parse_receipt_sha256,
                "map_packet_id": planned.packet_id,
                "map_plan_row_receipt_sha256": planned.receipt_sha256,
                "map_source_row_sha256": mapped.source_row_sha256,
                "structured_temporal_metadata_available": False,
                **(
                    {}
                    if parent_verification is None
                    else {
                        "parent_prediction_verification_receipt_sha256": (
                            parent_verification.receipt_sha256
                        )
                    }
                ),
                **(
                    {
                        "state_chain_direct_authority_applied": True,
                        "state_chain_direct_prediction_sha256": (
                            planned.direct_answer_row.prediction_sha256
                        ),
                    }
                    if state_chain_authority
                    else {}
                ),
            },
        )
        activation = None
        if unresolved:
            activation = SourceGateActivationReceipt(
                packet.question_id,
                packet.question_sha256,
                packet.dated_question_sha256,
                planned.packet_id,
                question_plan_receipt,
                fact_frontier_receipt,
                tuple(row.obligation_id for row in obligations),
                unresolved,
            )
        rows.append(
            QueryMapSourceGateAdapterRow(
                planned.ordinal,
                packet.question_id,
                packet.question_sha256,
                packet.dated_question_sha256,
                packet.packet_id,
                planned.packet_id,
                query_receipt,
                planned.receipt_sha256,
                mapped.source_row_sha256,
                mapped.map_parse_receipt_sha256,
                question_plan_receipt,
                fact_frontier_receipt,
                obligations,
                satisfied,
                unresolved,
                StageDisposition.ADDED if unresolved else StageDisposition.NO_OP,
                (
                    "state_chain_direct_authority_no_source_hydration"
                    if state_chain_authority
                    else "parent_prediction_verification_disagreed"
                    if unresolved
                    and parent_verification is not None
                    and not parent_verification.mechanically_agrees
                    else "mechanically_unresolved_query_plan_obligations"
                    if unresolved
                    else "all_query_plan_obligations_grounded"
                ),
                activation,
                obligation_mode,
                parent_verification,
                state_chain_profile,
            )
        )
    return QueryMapSourceGateAdapterPlane(
        query_run.sha256,
        map_plan.plan_identity_sha256,
        map_plane.run_sha256,
        map_plane.runtime_ledger_sha256,
        map_plane.snapshot_id,
        tuple(rows),
    )


__all__ = [
    "CONSOLIDATED_OBLIGATION_MODE",
    "FORMAT",
    "LEGACY_OBLIGATION_MODE",
    "OBLIGATION_MODES",
    "PARENT_VERIFICATION_RULE_ID",
    "ParentPredictionVerification",
    "QueryMapSourceGateAdapterError",
    "QueryMapSourceGateAdapterPlane",
    "QueryMapSourceGateAdapterRow",
    "STATE_CHAIN_DIRECT_AUTHORITY_PROFILE",
    "STATE_CHAIN_PROFILES",
    "STRICT_STATE_CHAIN_PROFILE",
    "adapt_query_map_solver_v2",
]
