from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_adaptive_source_map as base_cli
from tools import run_locked_adaptive_source_tail_wave as tail_cli
from tools._routed_repair_routing import RoutedRepairStyle, route_question
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import ArtifactRef, identity_sha256
from tools.matched_eval.source_gate_controller import (
    EligibleFrontierScope,
    LaneSourceBudget,
    ObligationKind,
    QuestionObligation,
    SourceGateActivationReceipt,
    SourceGateCandidate,
    SourceGatePlan,
    SourceGatePolicy,
    build_question_bound_mapping_plan,
    start_source_gate,
)
from tools.matched_eval.source_history_fact_union import (
    FactLane,
    FrozenHistoryChunk,
    HydratedSourceHistory,
    ParentIdentity,
    direct_evidence_projection_sha256,
    plan_source_history_hydration,
)
from tools.matched_eval.source_history_mapper_live import (
    SourceMapperCachedCompletion,
)


def _sha(value: str) -> str:
    return quote_sha256(value)


_NAMESPACE = _sha("tail-namespace")


def _membership(source_id: str) -> str:
    return identity_sha256(
        {
            "content_chunk_ids": [_sha(f"chunk:{source_id}")],
            "metadata_chunk_ids": [],
            "source_id": source_id,
            "stream_sha256": _sha(f"stream:{source_id}"),
        }
    )


def _candidate(lane: FactLane, source_id: str, rank: int) -> SourceGateCandidate:
    return SourceGateCandidate(
        lane,
        _NAMESPACE,
        source_id,
        rank,
        _membership(source_id),
        _sha(f"stream:{source_id}"),
        _sha(f"stream-receipt:{lane.value}"),
    )


def _parent() -> ParentIdentity:
    return ParentIdentity(
        _sha("population"),
        _sha("order"),
        _sha("snapshot"),
        _NAMESPACE,
        _sha("parent-packet"),
        _sha("parent-stage"),
        direct_evidence_projection_sha256(()),
    )


def _plan(question: str) -> SourceGatePlan:
    candidates = (
        _candidate(FactLane.DIRECT, "direct-0", 0),
        _candidate(FactLane.DIRECT, "direct-1", 1),
        _candidate(FactLane.PARTITION, "partition-0", 0),
        _candidate(FactLane.GUIDED, "guided-0", 0),
        _candidate(FactLane.GUIDED, "guided-1", 1),
    )
    policy = SourceGatePolicy(
        "tail-test-d1-g1",
        (
            LaneSourceBudget(FactLane.DIRECT, 1, 12, 2),
            LaneSourceBudget(FactLane.PARTITION, 0, 10, 2),
            LaneSourceBudget(FactLane.GUIDED, 1, 8, 2),
        ),
        24,
        48,
        16,
    )
    route = route_question(question)
    obligation = QuestionObligation(
        ObligationKind.TEMPORAL
        if route.modifiers.requires_temporal_metadata
        else ObligationKind.FRONTIER
        if route.modifiers.requires_complete_frontier
        else ObligationKind.SUPPORT,
        ("alpha",),
        1,
        route.modifiers.cardinality or 1,
        1,
        route.modifiers.requires_temporal_metadata,
        route.modifiers.requires_complete_frontier,
    )
    activation = SourceGateActivationReceipt(
        "q1",
        _sha(question),
        _sha(question),
        _parent().parent_packet_id,
        _sha("upstream-plan"),
        _sha("upstream-frontier"),
        (obligation.obligation_id,),
        (obligation.obligation_id,),
    )
    return SourceGatePlan(
        _parent(),
        "q1",
        _sha(question),
        question,
        _sha(question),
        0,
        route,
        (ArtifactRef("source", _sha("source"), "source.json"),),
        candidates,
        (obligation,),
        activation,
        EligibleFrontierScope(
            tuple(row.candidate_id for row in candidates),
            False,
            _sha("frontier"),
        ),
        policy,
    )


def _history(source_id: str) -> HydratedSourceHistory:
    text = f"{source_id} has exact tail evidence."
    chunk_id = _sha(f"chunk:{source_id}")
    chunk = FrozenHistoryChunk(
        source_id,
        chunk_id,
        _sha(f"turn:{source_id}"),
        1,
        "user",
        "2026-08-01T00:00:00+00:00",
        0,
        len(text),
        text,
        count_tokens(text),
        quote_sha256(text),
        False,
    )
    return HydratedSourceHistory(
        _NAMESPACE,
        source_id,
        (chunk_id,),
        (),
        _sha(f"stream:{source_id}"),
        _membership(source_id),
        (chunk,),
        True,
        _sha(f"history:{source_id}"),
    )


def test_route_policy_is_specialized_and_one_source_per_question() -> None:
    assert tail_cli.route_lane_order(RoutedRepairStyle.NUMERIC_REDUCE) == (
        FactLane.PARTITION,
        FactLane.GUIDED,
        FactLane.DIRECT,
    )
    assert tail_cli.route_lane_order(RoutedRepairStyle.SET_JOIN)[0] is FactLane.PARTITION
    assert tail_cli.route_lane_order(RoutedRepairStyle.TIMELINE)[0] is FactLane.GUIDED
    assert tail_cli.route_lane_order(RoutedRepairStyle.STATE_CHAIN)[0] is FactLane.GUIDED
    assert tail_cli.route_lane_order(RoutedRepairStyle.EXTRACT)[0] is FactLane.DIRECT
    assert tail_cli.route_lane_order(RoutedRepairStyle.SYNTHESIZE)[0] is FactLane.DIRECT

    numeric = _plan("How many blue items did Alpha have in total?")
    numeric_base = start_source_gate(numeric)
    selected_plan, lane, candidate, profile = tail_cli.select_one_tail_candidate(  # type: ignore[misc]
        numeric, numeric_base
    )
    assert lane is FactLane.PARTITION
    assert candidate.source_id == "partition-0"
    assert profile == tail_cli.DIRECT_STREAM_PROFILE_V1
    tail = tail_cli._tail_round(selected_plan, numeric_base, lane, candidate)
    assert len(tail.selected_candidates) == 1
    assert tail.cumulative_selected_candidate_ids[:-1] == numeric_base.cumulative_selected_candidate_ids

    extract = _plan("What color did Alpha choose?")
    selected_plan, lane, candidate, profile = tail_cli.select_one_tail_candidate(
        extract, start_source_gate(extract)
    )  # type: ignore[misc]
    assert (lane, candidate.rank, candidate.source_id) == (
        FactLane.DIRECT,
        1,
        "direct-1",
    )
    assert selected_plan is extract
    assert profile == tail_cli.DIRECT_STREAM_PROFILE_V1


def test_repack_v2_is_opt_in_only_at_deep_direct_rank() -> None:
    assert (
        tail_cli.direct_stream_profile_for_rank(FactLane.DIRECT, 3)
        == tail_cli.DIRECT_STREAM_PROFILE_V1
    )
    assert (
        tail_cli.direct_stream_profile_for_rank(FactLane.DIRECT, 4)
        == tail_cli.DIRECT_STREAM_PROFILE_REPACK_V2
    )
    assert (
        tail_cli.direct_stream_profile_for_rank(FactLane.GUIDED, 99)
        == tail_cli.DIRECT_STREAM_PROFILE_V1
    )


def test_repack_selector_skips_base_sources_and_rejects_direct_ranks_below_four() -> None:
    base = _plan("What color did Alpha choose?")
    base_round = start_source_gate(base)
    base_partition = base.candidates_for(FactLane.PARTITION)
    base_guided = base.candidates_for(FactLane.GUIDED)
    repack_direct = (
        _candidate(FactLane.DIRECT, "direct-0", 0),
        _candidate(FactLane.DIRECT, "guided-0", 1),
        _candidate(FactLane.DIRECT, "repack-shallow-2", 2),
        _candidate(FactLane.DIRECT, "repack-shallow-3", 3),
        _candidate(FactLane.DIRECT, "repack-deep-4", 4),
    )
    repack_candidates = (*repack_direct, *base_partition, *base_guided)
    repack = replace(
        base,
        candidates=repack_candidates,
        eligible_frontier=EligibleFrontierScope(
            tuple(row.candidate_id for row in repack_candidates),
            False,
            _sha("repack-frontier"),
        ),
    )
    selected_plan, lane, candidate, profile = tail_cli.select_one_tail_candidate(  # type: ignore[misc]
        base,
        base_round,
        direct_plan=repack,
    )
    assert selected_plan is repack
    assert (lane, candidate.rank, candidate.source_id) == (
        FactLane.DIRECT,
        4,
        "repack-deep-4",
    )
    assert profile == tail_cli.DIRECT_STREAM_PROFILE_REPACK_V2

    shallow_candidates = (*repack_direct[:4], *base_partition, *base_guided)
    shallow_only = replace(
        base,
        candidates=shallow_candidates,
        eligible_frontier=EligibleFrontierScope(
            tuple(row.candidate_id for row in shallow_candidates),
            False,
            _sha("shallow-frontier"),
        ),
    )
    _selected_plan, lane, candidate, profile = tail_cli.select_one_tail_candidate(  # type: ignore[misc]
        base,
        base_round,
        direct_plan=shallow_only,
    )
    assert (lane, candidate.source_id) == (FactLane.GUIDED, "guided-1")
    assert profile == tail_cli.DIRECT_STREAM_PROFILE_V1


def test_global_call_cap_defers_only_the_new_call_suffix() -> None:
    plan = _plan("What color did Alpha choose?")
    base = start_source_gate(plan)
    selected_plan, lane, candidate, _profile = tail_cli.select_one_tail_candidate(  # type: ignore[misc]
        plan, base
    )
    tail = tail_cli._tail_round(selected_plan, base, lane, candidate)
    history = _history(candidate.source_id)
    hydration = plan_source_history_hydration(
        plan.parent,
        selections=tail.selections,
        histories=(history,),
        max_window_tokens=800,
    )
    mapping = build_question_bound_mapping_plan(
        plan,
        tail,
        hydration,
        mapper_contract_sha256=_sha("mapper-contract"),
    )
    assert len(mapping.new_call_work_ids) == 1
    capped, remaining = tail_cli.cap_mapping_plan_new_calls(mapping, 0)
    assert remaining == 0
    assert capped.new_call_work_ids == ()
    assert capped.deferred_work_ids == mapping.new_call_work_ids
    assert capped.work_items == mapping.work_items
    assert capped.aliases == mapping.aliases


def test_tail_cache_parser_round_trips_exact_prompt_external_completion() -> None:
    cached = SourceMapperCachedCompletion(
        _sha("work"),
        _sha("prompt"),
        _sha("messages"),
        '{"facts":[]}',
        quote_sha256('{"facts":[]}'),
        _sha("original-result"),
        0,
    )
    payload = {
        "base_materialization_sha256": tail_cli.EXPECTED_BASE_MATERIALIZATION_SHA256,
        "base_preflight_sha256": tail_cli.EXPECTED_BASE_PREFLIGHT_SHA256,
        "campaign_id": tail_cli.CAMPAIGN_ID,
        "cached_completion_count": 1,
        "cached_completions": [cached.projection()],
        "format": tail_cli.CACHE_FORMAT,
        "gold_loaded": False,
        "invalid_wave1_checkpoint_reads": 0,
        "invalid_wave1_checkpoint_reuse": False,
        "provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    artifact = SealedArtifact(
        Path("cache.json"), identity_sha256(payload), payload
    )
    assert tail_cli._parse_cache(artifact) == {cached.physical_work_id: cached}


def test_public_base_loader_keeps_legacy_five_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = (1, 2, 3, 4, 5, 6)
    monkeypatch.setattr(
        base_cli,
        "load_typed_materialization_root_with_batch",
        lambda *args, **kwargs: sentinel,
    )
    observed = base_cli.load_typed_materialization_root(
        "unused",
        expected_preflight_sha256=_sha("preflight"),
        expected_materialization_sha256=_sha("materialization"),
        model="model",
        gateway_url="http://gateway",
        max_concurrency=1,
        direct_base_cap=1,
        partition_base_cap=0,
        guided_base_cap=1,
    )
    assert observed == sentinel[:-1]


def test_protocol_invalid_output_root_can_never_be_reused(tmp_path: Path) -> None:
    sentinel = tmp_path / tail_cli.PROTOCOL_INVALID_SENTINEL_NAME
    sentinel.write_text("terminal request uncertainty\n", encoding="utf-8")

    with pytest.raises(
        tail_cli.LockedAdaptiveSourceTailError,
        match="permanently protocol-invalid",
    ):
        tail_cli._reject_protocol_invalid_root(tmp_path)


def test_recovery_incident_is_exact_and_tamper_evident() -> None:
    payload = tail_cli.recovery_incident_projection()
    artifact = SealedArtifact(
        Path("incident.json"), identity_sha256(payload), payload
    )
    tail_cli._validate_incident_artifact(artifact)
    assert payload["invalid_wave1_status"] == "abandoned_terminal_uncertainty"
    assert payload["request_journal_count_observed_before_deletion"] == 4
    assert payload["response_journal_count_observed"] == 0
    assert len(payload["terminal_calls"]) == 4

    changed = {**payload, "wave1_checkpoint_reuse_permitted": True}
    with pytest.raises(
        tail_cli.LockedAdaptiveSourceTailError,
        match="incident receipt changed",
    ):
        tail_cli._validate_incident_artifact(
            SealedArtifact(
                Path("changed.json"), identity_sha256(changed), changed
            )
        )


def test_recovery_selector_advances_within_same_lane() -> None:
    plan = _plan("What color did Alpha choose?")
    candidates = (
        _candidate(FactLane.DIRECT, "direct-0", 0),
        _candidate(FactLane.DIRECT, "terminal-direct-1", 1),
        _candidate(FactLane.DIRECT, "recovery-direct-2", 2),
        *plan.candidates_for(FactLane.PARTITION),
        *plan.candidates_for(FactLane.GUIDED),
    )
    plan = replace(
        plan,
        candidates=candidates,
        eligible_frontier=EligibleFrontierScope(
            tuple(row.candidate_id for row in candidates),
            False,
            _sha("recovery-frontier"),
        ),
    )
    selected_plan, lane, candidate, _profile = tail_cli.select_one_tail_candidate(  # type: ignore[misc]
        plan,
        start_source_gate(plan),
        excluded_source_ids=frozenset({"terminal-direct-1"}),
        required_lane=FactLane.DIRECT,
    )
    assert selected_plan is plan
    assert (lane, candidate.rank, candidate.source_id) == (
        FactLane.DIRECT,
        2,
        "recovery-direct-2",
    )


def test_three_recovery_deny_stages_fail_closed() -> None:
    terminal = tail_cli.TERMINAL_CALL_IDENTITIES[0]
    clean = _sha("clean")
    question = SimpleNamespace(
        gate_round=SimpleNamespace(
            selections=(SimpleNamespace(selection_id=terminal.selection_id),)
        ),
        hydration_plan=SimpleNamespace(
            windows=(SimpleNamespace(window_id=terminal.window_id),)
        ),
        mapping_plan=SimpleNamespace(
            work_items=(SimpleNamespace(work_id=terminal.physical_work_id),)
        ),
    )
    with pytest.raises(
        tail_cli.LockedAdaptiveSourceTailError,
        match="structural population reused",
    ):
        tail_cli.enforce_structural_recovery_denylist(
            (SimpleNamespace(selected_source_id=terminal.source_id),),
            (question,),
        )

    with pytest.raises(
        tail_cli.LockedAdaptiveSourceTailError,
        match="prompt or message",
    ):
        tail_cli.enforce_prompt_recovery_denylist(
            (
                SimpleNamespace(
                    prompt_id=terminal.prompt_id,
                    messages_sha256=clean,
                ),
            )
        )

    with pytest.raises(
        tail_cli.LockedAdaptiveSourceTailError,
        match="runtime reused",
    ):
        tail_cli.enforce_runtime_recovery_denylist(
            SimpleNamespace(
                runtime_identity_sha256=clean,
                _call_keys={clean: terminal.call_key_sha256},
            )
        )


def test_recovery_root_and_checkpoint_are_isolated(tmp_path: Path) -> None:
    base = tmp_path / "base"
    invalid = base / tail_cli.INVALID_WAVE1_DIR_NAME
    recovery = base / tail_cli.CAMPAIGN_ID
    tail_cli._assert_recovery_root_isolated(recovery, base)

    for unsafe in (base, invalid, invalid / "nested"):
        with pytest.raises(tail_cli.LockedAdaptiveSourceTailError):
            tail_cli._assert_recovery_root_isolated(unsafe, base)

    assert tail_cli.CHECKPOINT_DIR_NAME != tail_cli.INVALID_WAVE1_CHECKPOINT_DIR_NAME
