from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests.test_matched_eval_locked_source_gate_adapter import _fixture, _policy
from tests.test_matched_eval_query_map_source_gate_adapter import (
    _map_plan,
    _map_plane,
    _plan,
)
from tools.matched_eval.locked_source_gate_adapter import (
    build_locked_source_gate_adapter,
)
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.query_map_source_gate_adapter import (
    adapt_query_map_solver_v2,
)
from tools.matched_eval.source_history_mapper_live import (
    HARD_CONTEXT_TOKEN_CAP,
    OUTPUT_TOKEN_RESERVE,
    WorkDisposition,
)
from tools.run_locked_adaptive_source_map import (
    _pareto_preflight,
    _source_target_hit,
    activation_inputs_from_query_map,
    build_locked_base_round,
    load_fast_materialization_manifest,
    preflight_projection,
    repolicy_source_population,
    source_gate_policy,
    work_manifest_projection,
)


def test_posthoc_source_target_matching_is_exact_or_question_qualified() -> None:
    assert _source_target_hit("q0", "answer_1", {"answer_1"}) is True
    assert _source_target_hit("q0", "answer_1", {"q0::answer_1"}) is True
    assert _source_target_hit("q0", "answer_1", {"q1::answer_1"}) is False


def test_query_map_conversion_activates_only_unresolved_rows(tmp_path: Path) -> None:
    query_run, map_plan = _map_plan(tmp_path, _plan())
    adapter = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _map_plane(map_plan, ()),
    )

    activations = activation_inputs_from_query_map(adapter, as_of_turn=19)

    assert len(activations) == 1
    source = activations[0]
    adapted = adapter.activated_rows[0]
    assert source.question_id == adapted.question_id
    assert source.source_packet_id == adapted.source_packet_id
    assert source.map_packet_id == adapted.map_packet_id
    assert source.source_packet_id != source.map_packet_id
    assert source.as_of_turn == 19
    assert source.obligation_ids == adapted.activation.obligation_ids
    assert tuple(row.obligation_id for row in source.unresolved_obligations) == (
        adapted.unresolved_obligation_ids
    )
    assert source.upstream_question_plan_receipt_sha256 == (
        adapted.upstream_question_plan_receipt_sha256
    )
    assert source.upstream_fact_frontier_receipt_sha256 == (
        adapted.upstream_fact_frontier_receipt_sha256
    )


def test_base_round_batches_one_namespace_and_reuses_cross_lane_source(
    tmp_path: Path,
) -> None:
    row, activation, artifacts = _fixture(tmp_path)
    population = build_locked_source_gate_adapter(
        (row,),
        (activation,),
        source_artifacts=artifacts,
        policy=_policy(),
    )

    plan = build_locked_base_round(population)

    assert len(plan.questions) == 1
    assert len(plan.hydration_batches) == 1
    batch = plan.hydration_batches[0]
    assert batch.source_ids == (
        "shared",
        "direct-only",
        "partition-only",
        "guided-only",
    )
    question = plan.questions[0]
    assert len(question.gate_round.selections) == 6
    assert len(question.hydration_plan.histories) == 4
    assert len(question.hydration_plan.windows) == 6
    assert len(question.mapping_plan.aliases) == 6
    assert len(question.mapping_plan.work_items) == 4
    assert len(question.mapper_preflight.prompt_rows) == 4
    assert question.mapper_preflight.required_provider_calls == 4
    assert all(
        row.disposition is WorkDisposition.NEW_CALL
        for row in question.mapper_preflight.prompt_rows
    )
    assert plan.required_provider_calls == 4
    assert plan.provider_population.logical_prompt_count == 4
    assert plan.provider_population.unique_prompt_count == 4
    assert question.mapper_preflight.maximum_combined_token_proxy <= (
        HARD_CONTEXT_TOKEN_CAP
    )

    projection = preflight_projection(
        plan,
        gateway_url="https://controlled.invalid/v1",
        model="controlled/test-model",
        max_concurrency=2,
    )
    assert projection["logical_selection_count"] == 6
    assert projection["logical_window_count"] == 6
    assert projection["unique_physical_window_count"] == 4
    assert projection["physical_prompt_count"] == 4
    assert projection["required_authorized_provider_calls"] == 4
    assert projection["namespace_batch_count"] == 1
    assert projection["unique_namespaced_source_count"] == 4
    assert projection["logical_selection_dedup_performed"] is False
    assert projection["post_map_dedup_performed"] is False
    assert projection["direct_evidence_exclusion_performed"] is False
    assert projection["physical_work_reuse_across_logical_aliases"] is True
    assert projection["provider_calls"] == 0
    assert projection["retained_transformer_token_state_bytes"] == 0
    assert projection["maximum_prompt_and_output_token_envelope"] <= 8_000
    assert projection["output_token_reserve"] == OUTPUT_TOKEN_RESERVE

    manifest_payload = work_manifest_projection(plan)
    restored = load_fast_materialization_manifest(
        SealedArtifact(tmp_path / "manifest.json", "a" * 64, manifest_payload),
        expected_source_population_receipt_sha256=(
            plan.source_population.receipt_sha256
        ),
    )
    assert len(restored) == 1
    assert restored[0].hydration_plan == question.hydration_plan
    assert restored[0].mapping_plan == question.mapping_plan
    assert restored[0].mapper_preflight == question.mapper_preflight


def test_satisfied_query_map_row_creates_no_source_activation(tmp_path: Path) -> None:
    query_run, map_plan = _map_plan(tmp_path, _plan())
    adapter = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _map_plane(map_plan, ("rosemary and mint",)),
    )

    assert activation_inputs_from_query_map(adapter) == ()
    assert len(adapter.no_op_rows) == 1


def test_explicit_dpg_policy_is_sealed_and_changes_only_base_selection(
    tmp_path: Path,
) -> None:
    row, activation, artifacts = _fixture(tmp_path)
    original = build_locked_source_gate_adapter(
        (row,),
        (activation,),
        source_artifacts=artifacts,
        policy=_policy(),
    )
    policy = source_gate_policy(1, 1, 0)
    population = repolicy_source_population(original, policy)

    assert population.questions[0].plan.policy.receipt_sha256 == (
        policy.receipt_sha256
    )
    assert population.questions[0].source_packet_id == (
        original.questions[0].source_packet_id
    )
    assert population.questions[0].plan.parent == original.questions[0].plan.parent
    assert population.questions[0].plan.activation == (
        original.questions[0].plan.activation
    )

    plan = build_locked_base_round(population)
    question = plan.questions[0]
    assert tuple(
        (selection.lane.value, selection.source_id)
        for selection in question.gate_round.selections
    ) == (
        ("direct", "shared"),
        ("partition", "partition-only"),
    )
    projection = preflight_projection(
        plan,
        gateway_url="https://controlled.invalid/v1",
        model="controlled/test-model",
        max_concurrency=2,
    )
    assert projection["source_gate_policy_receipt_sha256"] == policy.receipt_sha256
    assert projection["source_gate_policy"]["policy_id"].endswith(
        "d1-p1-g0-v1"
    )
    assert projection["logical_selection_count"] == 2
    assert projection["unique_namespaced_source_count"] == 2


def test_pareto_preflight_reuses_one_source_plane_and_shared_hydration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    row, activation, artifacts = _fixture(tmp_path / "store")
    original = build_locked_source_gate_adapter(
        (row,),
        (activation,),
        source_artifacts=artifacts,
        policy=_policy(),
    )
    query_run, map_plan = _map_plan(tmp_path / "query-map", _plan())
    adapter = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _map_plane(map_plan, ()),
    )
    loads = {"map": 0, "source": 0}

    def fake_map(**_kwargs):
        loads["map"] += 1
        return query_run, map_plan, object(), adapter

    def fake_source(_activations, *, pins, policy):
        del pins
        loads["source"] += 1
        return repolicy_source_population(original, policy)

    monkeypatch.setattr(
        "tools.run_locked_adaptive_source_map.load_locked_query_map", fake_map
    )
    monkeypatch.setattr(
        "tools.run_locked_adaptive_source_map.load_locked_source_gate_adapter",
        fake_source,
    )
    output = tmp_path / "pareto"
    result = _pareto_preflight(
        SimpleNamespace(
            policy=[(1, 0, 1), (1, 1, 0)],
            max_concurrency=2,
            gateway_url="https://controlled.invalid/v1",
            obligation_mode="entity_per_support_v1",
            model="controlled/test-model",
            output_root=output,
        )
    )

    assert loads == {"map": 1, "source": 1}
    assert result["source_plane_loaded_once"] is True
    assert result["namespace_batch_count"] == 1
    assert len(result["points"]) == 2
    assert tuple(
        row["required_authorized_provider_calls"] for row in result["points"]
    ) == (1, 2)
    assert (output / "d1-p0-g1" / "adaptive-source-map-base-preflight-v2.json").is_file()
    assert (output / "d1-p1-g0" / "adaptive-source-map-base-preflight-v2.json").is_file()
