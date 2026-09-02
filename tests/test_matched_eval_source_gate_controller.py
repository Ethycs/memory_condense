from __future__ import annotations

import json
from dataclasses import replace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256

from tools._routed_repair_routing import RoutedRepairStyle, route_question
from tools.matched_eval.contracts import ArtifactRef, assert_gold_blind, identity_sha256
from tools.matched_eval.source_gate_controller import (
    CoverageFact,
    EligibleFrontierScope,
    GateRoundKind,
    GateStopReason,
    LaneSourceBudget,
    NamespacedSourceKey,
    ObligationKind,
    ObligationState,
    QuestionObligation,
    SourceGateCandidate,
    SourceGateActivationReceipt,
    SourceGateControllerError,
    SourceGatePlan,
    SourceGatePolicy,
    SourceGateRound,
    SourceGateStopReceipt,
    advance_source_gate,
    assess_obligation_coverage,
    build_question_bound_mapping_plan,
    coverage_facts_from_fact_union,
    replay_source_gate,
    start_source_gate,
    validate_question_bound_completion,
)
from tools.matched_eval.source_history_fact_union import (
    FactLane,
    FrozenHistoryChunk,
    HydratedSourceHistory,
    ParentIdentity,
    build_post_map_fact_union,
    direct_evidence_projection_sha256,
    plan_source_history_hydration,
)


def _sha(value: str) -> str:
    return quote_sha256(value)


_NAMESPACE = _sha("namespace")


def _parent() -> ParentIdentity:
    return ParentIdentity(
        population_identity_sha256=_sha("population"),
        question_order_sha256=_sha("question-order"),
        snapshot_id=_sha("snapshot"),
        namespace_id=_NAMESPACE,
        parent_packet_id=_sha("parent-packet"),
        parent_stage_receipt_sha256=_sha("parent-stage"),
        direct_evidence_projection_sha256=direct_evidence_projection_sha256(()),
    )


def _membership_sha(source_id: str, chunk_id: str) -> str:
    return identity_sha256(
        {
            "content_chunk_ids": [chunk_id],
            "metadata_chunk_ids": [],
            "source_id": source_id,
            "stream_sha256": _sha(f"stream:{source_id}"),
        }
    )


def _history(source_id: str, text: str | None = None) -> HydratedSourceHistory:
    body = text or f"{source_id} contains an exact memory."
    chunk_id = _sha(f"chunk:{source_id}")
    chunk = FrozenHistoryChunk(
        source_id=source_id,
        chunk_id=chunk_id,
        turn_id=_sha(f"turn:{source_id}"),
        turn_ordinal=1,
        role="user",
        created_at="2026-08-01T00:00:00+00:00",
        start_char=0,
        end_char=len(body),
        text=body,
        token_count=count_tokens(body),
        turn_text_sha256=quote_sha256(body),
        metadata_chunk=False,
    )
    return HydratedSourceHistory(
        namespace_id=_NAMESPACE,
        source_id=source_id,
        content_chunk_ids=(chunk_id,),
        metadata_chunk_ids=(),
        stream_sha256=_sha(f"stream:{source_id}"),
        membership_projection_sha256=_membership_sha(source_id, chunk_id),
        chunks=(chunk,),
        store_bytes_revalidated=True,
        receipt_sha256=_sha(f"history:{source_id}:{body}"),
    )


def _candidate(
    lane: FactLane,
    source_id: str,
    rank: int,
    *,
    stream_receipt: str | None = None,
) -> SourceGateCandidate:
    chunk_id = _sha(f"chunk:{source_id}")
    return SourceGateCandidate(
        lane=lane,
        namespace_id=_NAMESPACE,
        source_id=source_id,
        rank=rank,
        membership_projection_sha256=_membership_sha(source_id, chunk_id),
        stream_sha256=_sha(f"stream:{source_id}"),
        source_stream_receipt_sha256=stream_receipt
        or _sha(f"stream-receipt:{lane.value}"),
    )


def _policy(
    *,
    direct: tuple[int, int, int] = (1, 3, 1),
    partition: tuple[int, int, int] = (0, 2, 1),
    guided: tuple[int, int, int] = (1, 3, 1),
    unique_cap: int = 8,
    call_cap: int = 12,
    rounds: int = 8,
) -> SourceGatePolicy:
    return SourceGatePolicy(
        policy_id="test-source-gate-v1",
        lane_budgets=(
            LaneSourceBudget(FactLane.DIRECT, *direct),
            LaneSourceBudget(FactLane.PARTITION, *partition),
            LaneSourceBudget(FactLane.GUIDED, *guided),
        ),
        global_unique_source_cap=unique_cap,
        max_physical_map_calls=call_cap,
        max_rounds=rounds,
    )


def _candidates(*, shared_base: bool = True) -> tuple[SourceGateCandidate, ...]:
    direct0 = "history-shared" if shared_base else "history-direct-0"
    guided0 = "history-shared" if shared_base else "history-guided-0"
    return (
        _candidate(FactLane.DIRECT, direct0, 0),
        _candidate(FactLane.DIRECT, "history-direct-1", 1),
        _candidate(FactLane.DIRECT, "history-direct-2", 2),
        _candidate(FactLane.PARTITION, "history-partition-0", 0),
        _candidate(FactLane.PARTITION, "history-partition-1", 1),
        _candidate(FactLane.GUIDED, guided0, 0),
        _candidate(FactLane.GUIDED, "history-guided-1", 1),
        _candidate(FactLane.GUIDED, "history-guided-2", 2),
    )


def _obligation_for(question: str) -> QuestionObligation:
    route = route_question(question)
    return QuestionObligation(
        kind=(
            ObligationKind.TEMPORAL
            if route.modifiers.requires_temporal_metadata
            else ObligationKind.FRONTIER
            if route.modifiers.requires_complete_frontier
            else ObligationKind.SUPPORT
        ),
        match_terms=("alpha", "blue"),
        required_match_term_count=2,
        minimum_fact_count=route.modifiers.cardinality or 1,
        minimum_source_count=1,
        requires_temporal_metadata=route.modifiers.requires_temporal_metadata,
        requires_complete_frontier=route.modifiers.requires_complete_frontier,
    )


def _plan(
    question: str = "What color did Alpha choose?",
    *,
    candidates: tuple[SourceGateCandidate, ...] | None = None,
    policy: SourceGatePolicy | None = None,
    obligation: QuestionObligation | None = None,
) -> SourceGatePlan:
    candidate_rows = _candidates() if candidates is None else candidates
    obligation_row = obligation or _obligation_for(question)
    activation = SourceGateActivationReceipt(
        question_id="question-1",
        question_sha256=_sha(question),
        dated_question_sha256=_sha(question),
        parent_packet_id=_parent().parent_packet_id,
        upstream_question_plan_receipt_sha256=_sha("upstream-question-plan"),
        upstream_fact_frontier_receipt_sha256=_sha("upstream-fact-frontier"),
        obligation_ids=(obligation_row.obligation_id,),
        unresolved_obligation_ids=(obligation_row.obligation_id,),
    )
    return SourceGatePlan(
        parent=_parent(),
        question_id="question-1",
        question_sha256=_sha(question),
        dated_question=question,
        dated_question_sha256=_sha(question),
        as_of_turn=37,
        route=route_question(question),
        sealed_input_artifacts=(
            ArtifactRef("direct_query_run", _sha("direct-run"), "direct/run.json"),
            ArtifactRef(
                "partition_r96_generation",
                _sha("partition-r96"),
                "partition-scan-v2-r96/retrieval-generation.json",
            ),
            ArtifactRef("guided_run", _sha("guided-run"), "guided/run.json"),
        ),
        candidates=candidate_rows,
        obligations=(obligation_row,),
        activation=activation,
        eligible_frontier=EligibleFrontierScope(
            eligible_candidate_ids=tuple(row.candidate_id for row in candidate_rows),
            exhaustive=False,
            basis_receipt_sha256=_sha("non-exhaustive-ranked-frontier"),
        ),
        policy=policy or _policy(),
    )


def _empty_coverage(
    plan: SourceGatePlan,
    round_plan: SourceGateRound,
    *,
    previous=None,
    calls: tuple[str, ...] = (),
):
    return assess_obligation_coverage(
        plan,
        round_plan,
        (),
        cumulative_physical_work_call_ids=calls,
        previous=previous,
    )


def test_question_bound_work_maps_cross_method_source_once_and_keeps_lane_credit() -> None:
    plan = _plan()
    base = start_source_gate(plan)
    assert base.kind is GateRoundKind.BASE
    assert tuple(row.lane for row in base.selections) == (
        FactLane.DIRECT,
        FactLane.GUIDED,
    )
    assert tuple(row.source_id for row in base.selections) == (
        "history-shared",
        "history-shared",
    )
    assert base.cumulative_unique_source_count == 1

    history = _history("history-shared", "Alpha chose blue yesterday.")
    hydration = plan_source_history_hydration(
        plan.parent,
        selections=base.selections,
        histories=(history,),
    )
    mapping = build_question_bound_mapping_plan(
        plan,
        base,
        hydration,
        mapper_contract_sha256=_sha("mapper-contract-v1"),
    )

    assert len(hydration.windows) == 2
    assert len({row.mapping_payload_sha256 for row in hydration.windows}) == 2
    assert len(mapping.work_items) == 1
    assert mapping.planned_provider_calls == 1
    assert len(mapping.aliases) == 2
    assert {row.lane for row in mapping.aliases} == {
        FactLane.DIRECT,
        FactLane.GUIDED,
    }
    payload = mapping.work_items[0].mapping_payload()
    assert payload["dated_question"] == plan.dated_question
    assert payload["dated_question_sha256"] == plan.dated_question_sha256
    assert payload["obligations"] == [plan.obligations[0].projection()]
    assert "lane" not in payload and "selection_id" not in payload

    chunk = history.chunks[0]
    quote = "Alpha chose blue"
    start = chunk.text.index(quote)
    completion = json.dumps(
        {
            "facts": [
                {
                    "chunk_id": chunk.chunk_id,
                    "event_tuple": None,
                    "fact": "Alpha chose blue.",
                    "mapper_item_id": "mapped-1",
                    "quote": quote,
                    "quote_end_char": start + len(quote),
                    "quote_sha256": quote_sha256(quote),
                    "quote_start_char": start,
                    "source_id": "history-shared",
                }
            ]
        }
    )
    batches = validate_question_bound_completion(
        hydration,
        mapping,
        physical_work_id=mapping.new_call_work_ids[0],
        completion=completion,
    )
    assert len(batches) == 2
    assert tuple(row.accepted[0].lane for row in batches) == (
        FactLane.DIRECT,
        FactLane.GUIDED,
    )
    fact_union = build_post_map_fact_union(hydration, batches=batches)
    assert len(fact_union.retained_facts) == 1
    assert {row.lane for row in fact_union.retained_facts[0].origins} == {
        FactLane.DIRECT,
        FactLane.GUIDED,
    }

    facts = coverage_facts_from_fact_union(fact_union)
    coverage = assess_obligation_coverage(
        plan,
        base,
        facts,
        mapping_plan_receipt_sha256s=(mapping.receipt_sha256,),
        cumulative_physical_work_call_ids=mapping.new_call_work_ids,
    )
    assert coverage.all_satisfied
    stop = advance_source_gate(plan, (base,), (coverage,))
    assert type(stop) is SourceGateStopReceipt
    assert stop.reason is GateStopReason.SATISFIED
    replay = replay_source_gate(plan, (base,), (coverage,), stop)
    assert replay.byte_identical is True


def test_mapping_cache_is_question_bound_and_reuses_only_the_same_work_identity() -> None:
    plan = _plan()
    base = start_source_gate(plan)
    history = _history("history-shared", "Alpha chose blue yesterday.")
    hydration = plan_source_history_hydration(
        plan.parent,
        selections=base.selections,
        histories=(history,),
    )
    first = build_question_bound_mapping_plan(
        plan,
        base,
        hydration,
        mapper_contract_sha256=_sha("mapper-contract-v1"),
    )
    cached = build_question_bound_mapping_plan(
        plan,
        base,
        hydration,
        mapper_contract_sha256=_sha("mapper-contract-v1"),
        cached_work_ids=(first.work_items[0].work_id,),
    )
    assert cached.reused_work_ids == (first.work_items[0].work_id,)
    assert cached.new_call_work_ids == ()

    changed = _plan("What shade did Alpha choose?")
    changed_base = start_source_gate(changed)
    changed_hydration = plan_source_history_hydration(
        changed.parent,
        selections=changed_base.selections,
        histories=(history,),
    )
    changed_mapping = build_question_bound_mapping_plan(
        changed,
        changed_base,
        changed_hydration,
        mapper_contract_sha256=_sha("mapper-contract-v1"),
        cached_work_ids=(first.work_items[0].work_id,),
    )
    assert changed_mapping.reused_work_ids == ()
    assert changed_mapping.work_items[0].work_id != first.work_items[0].work_id


def test_temporal_and_numeric_routes_slew_to_their_specialized_tail_lane() -> None:
    temporal_question = "Which happened first, Alpha or Beta?"
    temporal = _plan(temporal_question)
    assert temporal.route.style is RoutedRepairStyle.TIMELINE
    temporal_base = start_source_gate(temporal)
    temporal_coverage = _empty_coverage(temporal, temporal_base)
    temporal_tail = advance_source_gate(
        temporal,
        (temporal_base,),
        (temporal_coverage,),
    )
    assert type(temporal_tail) is SourceGateRound
    assert temporal_tail.tail_lane is FactLane.GUIDED
    assert tuple(row.rank for row in temporal_tail.selections) == (1,)

    numeric_question = "How many blue items did Alpha have in total?"
    numeric = _plan(numeric_question)
    assert numeric.route.style is RoutedRepairStyle.NUMERIC_REDUCE
    numeric_base = start_source_gate(numeric)
    numeric_coverage = _empty_coverage(numeric, numeric_base)
    numeric_tail = advance_source_gate(numeric, (numeric_base,), (numeric_coverage,))
    assert type(numeric_tail) is SourceGateRound
    assert numeric_tail.tail_lane is FactLane.PARTITION
    assert tuple(row.rank for row in numeric_tail.selections) == (0,)


def test_no_progress_rotates_through_later_batches_before_exhaustion() -> None:
    plan = _plan()
    rounds: list[SourceGateRound] = [start_source_gate(plan)]
    coverages = [_empty_coverage(plan, rounds[0])]
    expected_lanes = (
        FactLane.DIRECT,
        FactLane.GUIDED,
        FactLane.PARTITION,
        FactLane.DIRECT,
        FactLane.GUIDED,
        FactLane.PARTITION,
    )
    for lane in expected_lanes:
        decision = advance_source_gate(plan, tuple(rounds), tuple(coverages))
        assert type(decision) is SourceGateRound
        assert decision.tail_lane is lane
        rounds.append(decision)
        coverages.append(
            _empty_coverage(
                plan,
                decision,
                previous=coverages[-1],
            )
        )
    stop = advance_source_gate(plan, tuple(rounds), tuple(coverages))
    assert type(stop) is SourceGateStopReceipt
    assert stop.reason is GateStopReason.CANDIDATES_EXHAUSTED
    assert set(stop.cumulative_selected_candidate_ids) == {
        row.candidate_id for row in plan.candidates
    }
    assert stop.unresolved_obligation_ids == (plan.obligations[0].obligation_id,)
    replay = replay_source_gate(
        plan,
        tuple(rounds),
        tuple(coverages),
        stop,
    )
    assert replay.byte_identical is True
    assert replay.round_receipt_sha256s == tuple(row.receipt_sha256 for row in rounds)


def test_progress_does_not_grant_a_lane_a_second_turn_before_others() -> None:
    obligation = replace(_obligation_for("What color did Alpha choose?"), minimum_fact_count=99)
    plan = _plan(obligation=obligation)
    base = start_source_gate(plan)

    def fact(index: int) -> CoverageFact:
        return CoverageFact(
            fact_id=_sha(f"round-robin-fact:{index}"),
            fact_variants=(f"Alpha chose blue in observation {index}.",),
            source_keys=(NamespacedSourceKey(_NAMESPACE, f"fact-source-{index}"),),
            event_tuple=None,
            provenance_receipt_sha256=_sha(f"round-robin-provenance:{index}"),
        )

    facts = (fact(0),)
    base_coverage = assess_obligation_coverage(plan, base, facts)
    direct = advance_source_gate(plan, (base,), (base_coverage,))
    assert type(direct) is SourceGateRound
    assert direct.tail_lane is FactLane.DIRECT

    facts = (*facts, fact(1))
    direct_coverage = assess_obligation_coverage(
        plan, direct, facts, previous=base_coverage
    )
    assert direct_coverage.made_progress is True
    guided = advance_source_gate(
        plan,
        (base, direct),
        (base_coverage, direct_coverage),
    )
    assert type(guided) is SourceGateRound
    assert guided.tail_lane is FactLane.GUIDED

    facts = (*facts, fact(2))
    guided_coverage = assess_obligation_coverage(
        plan, guided, facts, previous=direct_coverage
    )
    assert guided_coverage.made_progress is True
    partition = advance_source_gate(
        plan,
        (base, direct, guided),
        (base_coverage, direct_coverage, guided_coverage),
    )
    assert type(partition) is SourceGateRound
    assert partition.tail_lane is FactLane.PARTITION


def test_unique_source_cap_skips_new_source_to_select_later_known_alias() -> None:
    candidates = (
        _candidate(FactLane.DIRECT, "known-a", 0),
        _candidate(FactLane.PARTITION, "known-b", 0),
        _candidate(FactLane.GUIDED, "new-c", 0),
        _candidate(FactLane.GUIDED, "known-a", 1),
    )
    plan = _plan(
        candidates=candidates,
        policy=_policy(
            direct=(1, 1, 1),
            partition=(1, 1, 1),
            guided=(0, 2, 1),
            unique_cap=2,
        ),
    )
    base = start_source_gate(plan)
    assert base.cumulative_unique_source_count == 2

    tail = advance_source_gate(plan, (base,), (_empty_coverage(plan, base),))

    assert type(tail) is SourceGateRound
    assert tail.tail_lane is FactLane.GUIDED
    assert tuple(row.source_id for row in tail.selected_candidates) == ("known-a",)
    assert tail.cumulative_unique_source_count == 2


def test_unique_source_cap_skips_unaffordable_lane_for_known_alias_lane() -> None:
    candidates = (
        _candidate(FactLane.DIRECT, "known-a", 0),
        _candidate(FactLane.PARTITION, "known-b", 0),
        _candidate(FactLane.PARTITION, "known-a", 1),
        _candidate(FactLane.GUIDED, "new-c", 0),
    )
    plan = _plan(
        candidates=candidates,
        policy=_policy(
            direct=(1, 1, 1),
            guided=(0, 1, 1),
            partition=(1, 2, 1),
            unique_cap=2,
        ),
    )
    base = start_source_gate(plan)
    assert base.cumulative_unique_source_count == 2

    tail = advance_source_gate(plan, (base,), (_empty_coverage(plan, base),))

    assert type(tail) is SourceGateRound
    assert tail.tail_lane is FactLane.PARTITION
    assert tuple(row.source_id for row in tail.selected_candidates) == ("known-a",)
    assert tail.cumulative_unique_source_count == 2


def test_grounded_thresholds_conflicts_frontier_and_call_cap_are_fail_closed() -> None:
    question = "How many blue items did Alpha have in total?"
    plan = _plan(question)
    base = start_source_gate(plan)
    key = NamespacedSourceKey(_NAMESPACE, "history-shared")
    fact = CoverageFact(
        fact_id=_sha("fact-one"),
        fact_variants=("Alpha had one blue item.",),
        source_keys=(key,),
        event_tuple=None,
        provenance_receipt_sha256=_sha("fact-one-provenance"),
    )
    partial = assess_obligation_coverage(plan, base, (fact,))
    assert partial.obligations[0].state is ObligationState.PARTIAL
    assert partial.obligations[0].reason == "frontier_scope_not_exhaustive"

    explicit_scope = EligibleFrontierScope(
        eligible_candidate_ids=base.selected_candidate_ids,
        exhaustive=True,
        basis_receipt_sha256=_sha("explicit-exhaustive-scope"),
    )
    exhaustive_plan = replace(plan, eligible_frontier=explicit_scope)
    exhaustive_base = start_source_gate(exhaustive_plan)
    exhaustive = assess_obligation_coverage(exhaustive_plan, exhaustive_base, (fact,))
    assert exhaustive.frontier_closed is True
    assert exhaustive.obligations[0].state is ObligationState.SATISFIED

    one_candidate = (_candidate(FactLane.DIRECT, "history-shared", 0),)
    exhausted_plan = _plan(
        question,
        candidates=one_candidate,
        policy=_policy(
            direct=(1, 1, 1),
            partition=(0, 0, 1),
            guided=(0, 0, 1),
            unique_cap=2,
        ),
    )
    exhausted_base = start_source_gate(exhausted_plan)
    exhausted_coverage = assess_obligation_coverage(
        exhausted_plan,
        exhausted_base,
        (fact,),
    )
    exhausted_stop = advance_source_gate(
        exhausted_plan,
        (exhausted_base,),
        (exhausted_coverage,),
    )
    assert type(exhausted_stop) is SourceGateStopReceipt
    assert exhausted_stop.reason is GateStopReason.CANDIDATES_EXHAUSTED
    assert exhausted_stop.unresolved_obligation_ids == (
        exhausted_plan.obligations[0].obligation_id,
    )

    capped_plan = _plan(
        policy=_policy(call_cap=0),
    )
    capped_base = start_source_gate(capped_plan)
    capped_history = _history("history-shared", "Alpha chose blue yesterday.")
    capped_hydration = plan_source_history_hydration(
        capped_plan.parent,
        selections=capped_base.selections,
        histories=(capped_history,),
    )
    capped_mapping = build_question_bound_mapping_plan(
        capped_plan,
        capped_base,
        capped_hydration,
        mapper_contract_sha256=_sha("mapper-contract-v1"),
    )
    assert capped_mapping.new_call_work_ids == ()
    assert capped_mapping.deferred_work_ids == tuple(
        row.work_id for row in capped_mapping.work_items
    )
    capped_coverage = _empty_coverage(capped_plan, capped_base)
    capped = advance_source_gate(capped_plan, (capped_base,), (capped_coverage,))
    assert type(capped) is SourceGateStopReceipt
    assert capped.reason is GateStopReason.PHYSICAL_CALL_CAP


def test_contracts_reject_route_candidate_provenance_and_replay_mutations() -> None:
    plan = _plan()
    assert_gold_blind(plan.projection())
    assert plan.projection()["gold_loaded"] is False
    assert plan.activation.projection()["fallback_required"] is True
    assert plan.question_plan_receipt_sha256 == plan.receipt_sha256
    with pytest.raises(SourceGateControllerError, match="cannot activate"):
        SourceGateActivationReceipt(
            question_id=plan.question_id,
            question_sha256=plan.question_sha256,
            dated_question_sha256=plan.dated_question_sha256,
            parent_packet_id=plan.parent.parent_packet_id,
            upstream_question_plan_receipt_sha256=_sha("upstream-question-plan"),
            upstream_fact_frontier_receipt_sha256=_sha("upstream-fact-frontier"),
            obligation_ids=(plan.obligations[0].obligation_id,),
            unresolved_obligation_ids=(),
        )
    with pytest.raises(SourceGateControllerError, match="dated question text changed"):
        replace(plan, dated_question="What color did Beta choose?")

    bad_order = (_candidates()[1], _candidates()[0], *_candidates()[2:])
    with pytest.raises(SourceGateControllerError, match="ranks must be contiguous"):
        _plan(candidates=bad_order)

    base = start_source_gate(plan)
    history = _history("history-shared", "Alpha chose blue yesterday.")
    tampered = replace(history, stream_sha256=_sha("changed-stream"))
    hydration = plan_source_history_hydration(
        plan.parent,
        selections=base.selections,
        histories=(tampered,),
    )
    with pytest.raises(SourceGateControllerError, match="membership provenance"):
        build_question_bound_mapping_plan(
            plan,
            base,
            hydration,
            mapper_contract_sha256=_sha("mapper-contract-v1"),
        )

    coverage = _empty_coverage(plan, base)
    tail = advance_source_gate(plan, (base,), (coverage,))
    assert type(tail) is SourceGateRound
    tail_coverage = _empty_coverage(plan, tail, previous=coverage)
    with pytest.raises(SourceGateControllerError, match="coverage chain changed"):
        advance_source_gate(plan, (base, tail), (tail_coverage, coverage))
