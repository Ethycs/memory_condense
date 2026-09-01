from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from tools.matched_eval.adaptive_memory_preflight import (
    HARD_PROMPT_TOKEN_CAP,
    SOURCE_GATE_INSERTION_POINT,
    AdaptiveMemoryPreflightError,
    ConditionalSourceGateContract,
    compose_adaptive_memory_preflight,
)
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import ArtifactRef, canonical_json_bytes
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    ANSWER_PLAN_ID,
    ARM_LABEL,
    MAP_OUTPUT_TOKEN_RESERVE,
    MAP_PLAN_ID,
    MAP_PREFLIGHT_FORMAT,
    MAP_RENDERER_ID,
    SOLVER_OUTPUT_TOKEN_RESERVE,
    SOLVER_PREFLIGHT_FORMAT,
    SOLVER_RENDERER_ID,
)
from tools.matched_eval.source_gate_controller import (
    LaneSourceBudget,
    SourceGatePolicy,
)
from tools.matched_eval.source_history_fact_union import FactLane


def _sha(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


_QUESTIONS = (
    (0, "q-a", True, 5_000),
    (1, "q-b", False, None),
    (2, "q-c", True, 4_500),
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "dated_question_sha256": _sha(f"dated:{question_id}"),
            "ordinal": ordinal,
            "prompt_token_proxy": prompt_tokens,
            "provider_call_planned": planned,
            "question_id": question_id,
            "question_sha256": _sha(f"question:{question_id}"),
        }
        for ordinal, question_id, planned, prompt_tokens in _QUESTIONS
    ]


def _population(stage: str) -> tuple[str, dict[str, object]]:
    digest = _sha(f"{stage}:population")
    return digest, {
        "logical_prompt_count": 2,
        "prompt_population_sha256": digest,
        "unique_prompt_count": 2,
    }


def _map_payload() -> dict[str, object]:
    population_sha, population = _population("map")
    return {
        "adapter_population_id": _sha("adapter-population"),
        "arm_label": ARM_LABEL,
        "direct_answer_run_sha256": _sha("direct-answer-run"),
        "format": MAP_PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "logical_prompt_count": 2,
        "map_plan_id": MAP_PLAN_ID,
        "map_renderer_id": MAP_RENDERER_ID,
        "observed_max_prompt_token_proxy": 5_000,
        "ordered_rows": _rows(),
        "output_token_reserve": MAP_OUTPUT_TOKEN_RESERVE,
        "plan_identity_sha256": _sha("map-plan"),
        "prompt_and_output_token_envelope": HARD_PROMPT_TOKEN_CAP,
        "prompt_population": population,
        "prompt_population_sha256": population_sha,
        "provider_calls": 0,
        "provider_prompts": [
            [{"role": "user", "content": "map a"}],
            [{"role": "user", "content": "map c"}],
        ],
        "question_count": 3,
        "required_authorized_provider_calls": 2,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": _sha("retrieval"),
        "snapshot_id": _sha("snapshot"),
        "stage": "map",
        "unique_prompt_count": 2,
    }


def _solver_payload() -> dict[str, object]:
    population_sha, population = _population("solver")
    return {
        "adapter_population_id": _sha("adapter-population"),
        "answer_plan_id": ANSWER_PLAN_ID,
        "arm_label": ARM_LABEL,
        "direct_answer_run_sha256": _sha("direct-answer-run"),
        "format": SOLVER_PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "logical_prompt_count": 2,
        "map_replay_sha256": _sha("terminal-map"),
        "map_run_sha256": _sha("terminal-map"),
        "observed_max_prompt_token_proxy": 5_500,
        "ordered_rows": _rows(),
        "output_token_reserve": SOLVER_OUTPUT_TOKEN_RESERVE,
        "plan_identity_sha256": _sha("solver-plan"),
        "prompt_and_output_token_envelope": HARD_PROMPT_TOKEN_CAP,
        "prompt_population": population,
        "prompt_population_sha256": population_sha,
        "provider_calls": 0,
        "provider_prompts": [
            [{"role": "user", "content": "solve a"}],
            [{"role": "user", "content": "solve c"}],
        ],
        "question_count": 3,
        "required_authorized_provider_calls": 2,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": _sha("retrieval"),
        "snapshot_id": _sha("snapshot"),
        "solver_renderer_id": SOLVER_RENDERER_ID,
        "stage": "solver",
        "unique_prompt_count": 2,
    }


def _artifact(name: str, payload: dict[str, object]) -> SealedArtifact:
    return SealedArtifact(
        Path(name),
        sha256(canonical_json_bytes(payload)).hexdigest(),
        payload,
    )


def _policy() -> SourceGatePolicy:
    return SourceGatePolicy(
        "adaptive-preflight-test-v1",
        (
            LaneSourceBudget(FactLane.DIRECT, 1, 2, 1),
            LaneSourceBudget(FactLane.PARTITION, 0, 2, 1),
            LaneSourceBudget(FactLane.GUIDED, 1, 2, 1),
        ),
        global_unique_source_cap=6,
        max_physical_map_calls=3,
        max_rounds=4,
    )


def _gate(
    *,
    mapper_envelope: bool = True,
    combined_solver: bool = True,
    activation_question_ids: tuple[str, ...] = ("q-a", "q-b", "q-c"),
) -> ConditionalSourceGateContract:
    return ConditionalSourceGateContract(
        controller_contract_sha256=_sha("source-gate-controller"),
        mapper_contract_sha256=_sha("source-gate-mapper"),
        upstream_fact_frontier_contract_sha256=_sha("fact-frontier"),
        policy=_policy(),
        candidate_artifacts=(
            ArtifactRef("direct_query_run", _sha("direct")),
            ArtifactRef("partition_r96_generation", _sha("partition-r96")),
            ArtifactRef("guided_run", _sha("guided")),
        ),
        activation_question_ids=activation_question_ids,
        mapper_max_prompt_token_proxy=5_800 if mapper_envelope else None,
        mapper_output_token_reserve=1_024 if mapper_envelope else None,
        augmented_solver_contract_sha256=(
            _sha("combined-solver") if combined_solver else None
        ),
        augmented_solver_max_prompt_token_proxy=(
            5_700 if combined_solver else None
        ),
        augmented_solver_output_token_reserve=(
            SOLVER_OUTPUT_TOKEN_RESERVE if combined_solver else None
        ),
    )


def test_composes_exact_two_pass_population_and_conditional_bounds() -> None:
    result = compose_adaptive_memory_preflight(
        _artifact("map-preflight.json", _map_payload()),
        solver_preflight=_artifact("solver-preflight.json", _solver_payload()),
        source_gate=_gate(),
    )

    assert result.map_stage.exact_required_calls == 2
    assert result.solver_stage.exact_required_calls == 2
    assert result.sealed_two_pass_exact_calls == 4
    assert result.baseline_two_pass_call_upper_bound == 4
    # q-b is normally a no-op, but an activated fallback needs one combined
    # final-solver call so the post-gate facts can actually affect the answer.
    assert result.combined_solver_call_upper_bound == 3
    assert result.conditional_source_map_call_upper_bound == 9
    assert result.conditional_total_provider_call_upper_bound == 14
    assert result.contract_complete is True
    assert result.missing_inputs == ()
    assert result.provider_calls_executed == 0
    assert result.retained_transformer_token_state_bytes == 0
    assert result.projection()["solver_insertion_point"] == (
        SOURCE_GATE_INSERTION_POINT
    )
    assert result.projection()["solver_population_role"] == (
        "direct_map_baseline_only"
    )


def test_current_map_only_state_is_useful_and_names_exact_missing_inputs() -> None:
    result = compose_adaptive_memory_preflight(
        _artifact("map-preflight.json", _map_payload()),
        source_gate=_gate(mapper_envelope=False, combined_solver=False),
    )

    assert result.solver_stage.availability == "deferred"
    assert result.solver_stage.prerequisite == "terminal_v2_map_replay"
    assert result.sealed_two_pass_exact_calls is None
    assert result.baseline_two_pass_call_upper_bound == 4
    assert result.combined_solver_call_upper_bound == 3
    assert result.conditional_total_provider_call_upper_bound == 14
    assert result.contract_complete is False
    assert result.missing_inputs == (
        "terminal_v2_map_replay_and_solver_preflight",
        "source_gate_mapper_render_envelope",
        "source_gate_combined_solver_adapter",
        "source_gate_combined_solver_render_envelope",
    )


@pytest.mark.parametrize("stage", ["map", "solver"])
def test_rejects_any_stage_that_exceeds_8k(stage: str) -> None:
    map_payload = _map_payload()
    solver_payload = _solver_payload()
    target = map_payload if stage == "map" else solver_payload
    target["observed_max_prompt_token_proxy"] = 7_500

    with pytest.raises(AdaptiveMemoryPreflightError, match="8k envelope"):
        compose_adaptive_memory_preflight(
            _artifact("map-preflight.json", map_payload),
            solver_preflight=(
                None
                if stage == "map"
                else _artifact("solver-preflight.json", solver_payload)
            ),
            source_gate=_gate(),
        )


def test_rejects_bool_as_zero_state_and_forged_artifact_identity() -> None:
    false_state = _map_payload()
    false_state["retained_request_token_state_bytes"] = False
    with pytest.raises(AdaptiveMemoryPreflightError, match="retained"):
        compose_adaptive_memory_preflight(
            _artifact("map-preflight.json", false_state),
            source_gate=_gate(),
        )

    forged = _artifact("map-preflight.json", _map_payload())
    forged = replace(forged, sha256=_sha("forged"))
    with pytest.raises(AdaptiveMemoryPreflightError, match="artifact SHA-256"):
        compose_adaptive_memory_preflight(forged, source_gate=_gate())


def test_solver_must_preserve_map_population_and_terminal_replay() -> None:
    reordered = _solver_payload()
    reordered["ordered_rows"] = list(reversed(_rows()))
    with pytest.raises(AdaptiveMemoryPreflightError, match="population differs"):
        compose_adaptive_memory_preflight(
            _artifact("map-preflight.json", _map_payload()),
            solver_preflight=_artifact("solver-preflight.json", reordered),
            source_gate=_gate(),
        )

    unsealed = _solver_payload()
    unsealed["map_replay_sha256"] = _sha("different-replay")
    with pytest.raises(AdaptiveMemoryPreflightError, match="terminal map replay"):
        compose_adaptive_memory_preflight(
            _artifact("map-preflight.json", _map_payload()),
            solver_preflight=_artifact("solver-preflight.json", unsealed),
            source_gate=_gate(),
        )


def test_source_gate_population_and_mapper_envelope_fail_closed() -> None:
    with pytest.raises(AdaptiveMemoryPreflightError, match="escaped or reordered"):
        compose_adaptive_memory_preflight(
            _artifact("map-preflight.json", _map_payload()),
            source_gate=_gate(
                activation_question_ids=("q-c", "q-a"),
            ),
        )

    with pytest.raises(AdaptiveMemoryPreflightError, match="exceeds 8k"):
        replace(
            _gate(),
            mapper_max_prompt_token_proxy=7_500,
            mapper_output_token_reserve=1_024,
        )

    with pytest.raises(AdaptiveMemoryPreflightError, match="augmented solver"):
        replace(
            _gate(),
            augmented_solver_max_prompt_token_proxy=7_000,
        )


def test_receipt_is_deterministic_gold_blind_and_provider_free() -> None:
    first = compose_adaptive_memory_preflight(
        _artifact("map-preflight.json", _map_payload()),
        source_gate=_gate(),
    )
    second = compose_adaptive_memory_preflight(
        _artifact("elsewhere/map-preflight.json", _map_payload()),
        source_gate=_gate(),
    )

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.projection()["provider_calls_executed"] == 0
    assert first.hard_prompt_token_cap == 8_000
