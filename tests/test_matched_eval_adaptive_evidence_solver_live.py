from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_matched_eval_query_evidence_map_solver_v2_live import _terminal_map
from tests.test_matched_eval_source_history_fact_union import (
    _NAMESPACE_ID,
    _hydrate_all,
    _mapped_item,
    _selection,
    _write_store,
)
from tools.matched_eval.adaptive_evidence_solver_live import (
    ARM_LABEL,
    FORMAT,
    PLAN_ID,
    RENDERER_ID,
    AdaptiveEvidenceSolverError,
    _operator_policy_projection,
    build_adaptive_evidence_solver_plan,
    capture_adaptive_solver_completions,
    materialize_adaptive_evidence_solver,
    parse_adaptive_solver_completion,
    preflight_adaptive_evidence_solver,
    replay_adaptive_evidence_solver,
)
from tools._routed_repair_routing import RoutedRepairStyle
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    ARM_LABEL as BASELINE_V2_ARM_LABEL,
    MAX_PROMPT_TOKENS,
    SOLVER_OUTPUT_TOKEN_RESERVE,
)
from tools.matched_eval.source_history_fact_union import (
    FactLane,
    ParentIdentity,
    build_post_map_fact_union,
    direct_evidence_projection_sha256,
    plan_source_history_hydration,
    validate_mapped_facts,
)


def _parents(tmp_path: Path):
    map_plan, output, preflight, provider, journals, result, map_plane = (
        _terminal_map(tmp_path)
    )
    del output, preflight, provider, journals, result
    return map_plan, map_plane


def _source_union(
    tmp_path: Path,
    map_plan,
    map_plane,
    *,
    include_mapped_batch: bool = True,
):
    store = tmp_path / "source-store"
    memberships = _write_store(
        store,
        {"source-alpha": ["Alpha stored the blue token on Tuesday."]},
    )
    histories = _hydrate_all(store, memberships)
    mapped = map_plane.rows[0]
    planned = map_plan.rows[0]
    source_snapshot = (
        map_plan.direct_plan.adapter_population.source_population.snapshot
    )
    parent = ParentIdentity(
        source_snapshot.population_identity_sha256,
        source_snapshot.question_order_sha256,
        source_snapshot.snapshot_id,
        _NAMESPACE_ID,
        planned.packet_id,
        mapped.map_parse_receipt_sha256,
        direct_evidence_projection_sha256(()),
    )
    selection = _selection(
        "direct-source-alpha",
        FactLane.DIRECT,
        "source-alpha",
    )
    hydration = plan_source_history_hydration(
        parent,
        selections=(selection,),
        histories=(histories["source-alpha"],),
    )
    window = hydration.windows[0]
    batch = validate_mapped_facts(
        hydration,
        window,
        (
            _mapped_item(
                window,
                mapper_item_id="fact-alpha-blue",
                fact="Alpha stored the blue token on Tuesday.",
                quote="blue token",
            ),
        ),
    )
    return build_post_map_fact_union(
        hydration,
        batches=(batch,) if include_mapped_batch else (),
    )


def test_map_only_plan_is_provider_free_exact_parent_fallback(
    tmp_path: Path,
) -> None:
    map_plan, map_plane = _parents(tmp_path)
    plan = build_adaptive_evidence_solver_plan(map_plan, map_plane)
    row = plan.rows[0]

    assert ARM_LABEL != BASELINE_V2_ARM_LABEL
    assert FORMAT == "memory-condense-adaptive-evidence-solver-v3"
    assert PLAN_ID == "matched_adaptive_evidence_solver_v3"
    assert RENDERER_ID == "matched_adaptive_evidence_solver_v3"
    assert plan.required_calls == 0
    assert plan.submitted_rows == ()
    assert row.messages is None
    assert row.prompt_token_proxy is None
    assert row.reason == "no_admitted_source_fact"
    assert row.retained_transformer_token_state_bytes == 0

    first = preflight_adaptive_evidence_solver(plan)
    second = preflight_adaptive_evidence_solver(plan)
    assert first == second
    assert first.required_authorized_provider_calls == 0
    assert first.provider_calls_executed == 0
    assert first.retained_transformer_token_state_bytes == 0
    assert first.map_run_sha256 == first.map_replay_sha256


def test_source_facts_are_post_map_packed_with_distinct_ids_and_exact_provenance(
    tmp_path: Path,
) -> None:
    map_plan, map_plane = _parents(tmp_path / "parents")
    union = _source_union(tmp_path, map_plan, map_plane)
    question_id = map_plane.rows[0].question_id
    plan = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        source_fact_unions={question_id: union},
    )
    row = plan.rows[0]
    joined = "\n".join(message.content for message in row.messages or ())

    assert row.activated is True
    assert row.source_fact_validation_mode == "post_map_union_repacked"
    assert row.allowed_map_item_ids == ("M001",)
    assert row.allowed_source_fact_ids == ("D001",)
    assert not set(row.allowed_map_item_ids) & set(row.allowed_source_fact_ids)
    assert row.fact_envelope is not None
    assert row.fact_envelope.fact_union_receipt_sha256 == union.receipt_sha256
    assert plan.required_calls == 1
    assert row.prompt_token_proxy is not None
    assert row.prompt_token_proxy + SOLVER_OUTPUT_TOKEN_RESERVE <= MAX_PROMPT_TOKENS
    assert union.retained_facts[0].union_fact_id not in joined
    assert union.retained_facts[0].receipt_sha256 not in joined
    assert union.retained_facts[0].origins[0].mapped_item_receipt_sha256 not in joined
    assert "Alpha stored the blue token on Tuesday." in joined
    assert '"contexts":[{"date":"2026-08-01T00:00:00+00:00","role":"user"}]' in joined
    assert '"evidence_id":"D001"' in joined
    assert '"decision":"replace"' in joined
    assert '"used_evidence_ids":["D001"]' in joined
    assert '"origins"' not in joined
    assert '"namespace_id"' not in joined
    assert '"quote"' not in joined
    assert '"window_id"' not in joined
    assert '"activated":true' in joined
    admission = row.fact_envelope.lane_packs[0].admissions[0]
    assert admission.union_fact.origins[0].quote == "blue token"
    assert admission.union_fact.origins[0].namespace_id == _NAMESPACE_ID
    assert admission.union_fact.receipt_sha256 == union.retained_facts[0].receipt_sha256

    same = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        source_fact_unions={question_id: union},
        packed_fact_envelopes={question_id: row.fact_envelope},
    )
    assert same.plan_identity_sha256 == plan.plan_identity_sha256
    assert same.rows[0].receipt_sha256 == row.receipt_sha256
    packed_only = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        packed_fact_envelopes={question_id: row.fact_envelope},
    )
    assert packed_only.rows[0].source_fact_validation_mode == (
        "sealed_fact_envelope_revalidated"
    )
    assert packed_only.rows[0].allowed_source_fact_ids == ("D001",)


def test_source_activation_without_an_admitted_fact_preserves_provenance_and_skips_call(
    tmp_path: Path,
) -> None:
    map_plan, map_plane = _parents(tmp_path / "parents")
    empty_union = _source_union(
        tmp_path,
        map_plan,
        map_plane,
        include_mapped_batch=False,
    )
    question_id = map_plane.rows[0].question_id

    plan = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        source_fact_unions={question_id: empty_union},
    )
    row = plan.rows[0]

    assert row.activated is True
    assert row.fact_union is empty_union
    assert row.fact_envelope is not None
    assert row.allowed_source_fact_ids == ()
    assert row.submitted is False
    assert row.reason == "source_gate_activated_without_admitted_source_fact"
    assert plan.required_calls == 0


@pytest.mark.parametrize(
    ("style", "route_specific_term"),
    [
        (RoutedRepairStyle.EXTRACT, "candidate value"),
        (RoutedRepairStyle.NUMERIC_REDUCE, "operand"),
        (RoutedRepairStyle.SET_JOIN, "candidate member"),
        (RoutedRepairStyle.SYNTHESIZE, "claim"),
        (RoutedRepairStyle.TIMELINE, "event"),
        (RoutedRepairStyle.STATE_CHAIN, "state"),
    ],
)
def test_question_only_operator_policy_is_deterministic_and_route_specific(
    style: RoutedRepairStyle,
    route_specific_term: str,
) -> None:
    first = _operator_policy_projection(style)
    second = _operator_policy_projection(style)

    assert first == second
    assert first is not second
    assert first["question_only"] is True
    assert first["operator"] == style.value
    assert "requires_complete_frontier" in first["complete_frontier_rule"]
    assert "preserve the parent" in first["complete_frontier_rule"]
    assert route_specific_term in " ".join(first["silent_work"])


def test_all_route_operator_prompts_are_distinct_and_cover_every_style() -> None:
    rendered = {
        style: json.dumps(
            _operator_policy_projection(style),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for style in RoutedRepairStyle
    }

    assert set(rendered) == set(RoutedRepairStyle)
    assert len(set(rendered.values())) == len(RoutedRepairStyle)
    assert "partial operand list" in rendered[RoutedRepairStyle.NUMERIC_REDUCE]
    assert "subset of the member frontier" in rendered[RoutedRepairStyle.SET_JOIN]
    assert "partial timeline" in rendered[RoutedRepairStyle.TIMELINE]
    assert "existing parent may already be" in rendered[RoutedRepairStyle.STATE_CHAIN]


@pytest.mark.parametrize(
    ("payload", "valid", "decision", "error"),
    [
        (
            {"decision": "replace", "prediction": "blue", "used_evidence_ids": ["D001"]},
            True,
            "replace",
            "none",
        ),
        (
            {"decision": "keep_parent", "prediction": "parent", "used_evidence_ids": []},
            True,
            "keep_parent",
            "none",
        ),
        (
            {"decision": "insufficient", "prediction": "", "used_evidence_ids": []},
            True,
            "insufficient",
            "none",
        ),
        (
            {"decision": "replace", "prediction": "blue", "used_evidence_ids": ["X999"]},
            False,
            "invalid",
            "unknown_evidence_id",
        ),
        (
            {"decision": "replace", "prediction": "blue", "used_evidence_ids": ["M001"]},
            False,
            "invalid",
            "replace_requires_source_fact",
        ),
        (
            {"decision": "keep_parent", "prediction": "changed", "used_evidence_ids": []},
            False,
            "invalid",
            "keep_parent_contract",
        ),
    ],
)
def test_strict_decision_parser_enforces_ids_and_lifecycle(
    payload: dict[str, object],
    valid: bool,
    decision: str,
    error: str,
) -> None:
    parsed = parse_adaptive_solver_completion(
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        allowed_evidence_ids=("M001", "D001"),
        replacement_evidence_ids=("D001",),
        parent_prediction="parent",
    )
    assert parsed.valid is valid
    assert parsed.decision == decision
    assert parsed.error_code == error


def test_completion_materialize_and_replay_use_source_fact_without_hidden_calls(
    tmp_path: Path,
) -> None:
    map_plan, map_plane = _parents(tmp_path / "parents")
    union = _source_union(tmp_path, map_plan, map_plane)
    question_id = map_plane.rows[0].question_id
    plan = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        source_fact_unions={question_id: union},
    )
    preflight = preflight_adaptive_evidence_solver(plan)
    completion = json.dumps(
        {
            "decision": "replace",
            "prediction": "blue token",
            "used_evidence_ids": ["D001"],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    completions = capture_adaptive_solver_completions(
        plan,
        preflight,
        {question_id: completion},
    )
    run = materialize_adaptive_evidence_solver(plan, preflight, completions)
    verified = replay_adaptive_evidence_solver(
        plan,
        preflight,
        completions,
        run,
    )
    row = run.rows[0]

    assert completions.provider_calls_executed_by_ingest == 0
    assert completions.retained_transformer_token_state_bytes == 0
    assert run.provider_calls_executed_by_materializer == 0
    assert run.retained_transformer_token_state_bytes == 0
    assert row.prediction == "blue token"
    assert row.solver_decision == "replace"
    assert row.solver_used_evidence_ids == ("D001",)
    assert row.solver_used_map_item_ids == ()
    assert row.solver_used_source_fact_ids == ("D001",)
    assert verified.run_receipt_sha256 == verified.replay_receipt_sha256
    assert verified.provider_calls_executed_by_replay == 0
    assert verified.retained_transformer_token_state_bytes == 0


@pytest.mark.parametrize(
    "completion",
    [
        json.dumps(
            {"decision": "insufficient", "prediction": "", "used_evidence_ids": []}
        ),
        json.dumps(
            {"decision": "replace", "prediction": "wrong", "used_evidence_ids": ["X999"]}
        ),
        json.dumps(
            {"decision": "replace", "prediction": "wrong", "used_evidence_ids": ["M001"]}
        ),
    ],
)
def test_insufficient_or_invalid_decision_preserves_parent_bytes(
    tmp_path: Path,
    completion: str,
) -> None:
    map_plan, map_plane = _parents(tmp_path / "parents")
    union = _source_union(tmp_path, map_plan, map_plane)
    plan = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        source_fact_unions={map_plane.rows[0].question_id: union},
    )
    preflight = preflight_adaptive_evidence_solver(plan)
    question_id = map_plane.rows[0].question_id
    completions = capture_adaptive_solver_completions(
        plan, preflight, {question_id: completion}
    )
    run = materialize_adaptive_evidence_solver(plan, preflight, completions)
    parent = map_plan.rows[0].direct_answer_row.prediction
    row = run.rows[0]

    assert row.prediction.encode("utf-8") == parent.encode("utf-8")
    assert row.changed_from_parent is False
    assert row.prediction_source == "sealed_direct_query_fallback"


def test_keep_parent_decision_preserves_exact_bytes_and_validated_map_id(
    tmp_path: Path,
) -> None:
    map_plan, map_plane = _parents(tmp_path / "parents")
    union = _source_union(tmp_path, map_plan, map_plane)
    plan = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        source_fact_unions={map_plane.rows[0].question_id: union},
    )
    preflight = preflight_adaptive_evidence_solver(plan)
    question_id = map_plane.rows[0].question_id
    parent = map_plan.rows[0].direct_answer_row.prediction
    completions = capture_adaptive_solver_completions(
        plan,
        preflight,
        {
            question_id: json.dumps(
                {
                    "decision": "keep_parent",
                    "prediction": parent,
                    "used_evidence_ids": ["M001"],
                },
                ensure_ascii=False,
            )
        },
    )
    row = materialize_adaptive_evidence_solver(
        plan, preflight, completions
    ).rows[0]

    assert row.prediction.encode("utf-8") == parent.encode("utf-8")
    assert row.changed_from_parent is False
    assert row.solver_valid is True
    assert row.solver_decision == "keep_parent"
    assert row.solver_used_map_item_ids == ("M001",)
    assert row.prediction_source == "adaptive_validated_evidence_keep_parent_v3"


def test_completion_population_is_exact_and_fact_parent_cannot_cross_questions(
    tmp_path: Path,
) -> None:
    map_plan, map_plane = _parents(tmp_path / "parents")
    union = _source_union(tmp_path, map_plan, map_plane)
    question_id = map_plane.rows[0].question_id
    plan = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        source_fact_unions={question_id: union},
    )
    preflight = preflight_adaptive_evidence_solver(plan)

    with pytest.raises(AdaptiveEvidenceSolverError, match="exactly equal"):
        capture_adaptive_solver_completions(plan, preflight, {})
    with pytest.raises(AdaptiveEvidenceSolverError, match="exactly equal"):
        capture_adaptive_solver_completions(
            plan,
            preflight,
            {question_id: "{}", "extra-question": "{}"},
        )

    crossed = replace(
        union,
        parent=replace(union.parent, parent_packet_id="0" * 64),
    )
    with pytest.raises(AdaptiveEvidenceSolverError, match="exact question/map parent"):
        build_adaptive_evidence_solver_plan(
            map_plan,
            map_plane,
            source_fact_unions={question_id: crossed},
        )


def test_fixed_solver_output_reserve_cannot_be_relaxed(tmp_path: Path) -> None:
    map_plan, map_plane = _parents(tmp_path)
    with pytest.raises(AdaptiveEvidenceSolverError, match="token envelope changed"):
        build_adaptive_evidence_solver_plan(
            map_plan,
            map_plane,
            output_token_reserve=SOLVER_OUTPUT_TOKEN_RESERVE - 1,
        )
