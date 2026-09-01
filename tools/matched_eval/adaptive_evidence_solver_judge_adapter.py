"""Gold-blind judge adapter for native adaptive evidence-solver V3 runs.

The adaptive solver deliberately has a small, provider-free native result
plane rather than the older answer/runtime artifact pair.  Changed-only
judging consumes the latter surface, so this module deterministically projects
an already replayed native run into that common surface.  The projection makes
no calls and does not open benchmark gold or judge state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

from memory_condense.domain.discourse import quote_sha256

from . import adaptive_evidence_solver_live as adaptive
from . import live, query_payload_live
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from .ledger import RuntimeLedgerEntry, build_runtime_ledger
from .source_history_fact_union import FactLane


FORMAT = "memory-condense-adaptive-evidence-solver-v3-judge-adapter-v1"
RENDERER_ID = adaptive.RENDERER_ID
SOLVER_STAGE_ID = "adaptive_evidence_solver_v3"
ANSWER_STAGE_ID = "adaptive_evidence_solver_v3_answer"
MAP_PARENT_STAGE_ID = "query_evidence_map_v2"


class AdaptiveEvidenceSolverJudgeAdapterError(MatchedEvalContractError):
    """A native solver run or judge-facing projection changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise AdaptiveEvidenceSolverJudgeAdapterError(message)


@dataclass(frozen=True, slots=True)
class AdaptiveJudgeProfile:
    profile_id: str
    lanes: tuple[FactLane, ...]
    kind: str
    arm_label: str
    arm_plan_id: str
    answer_plan_id: str


DIRECT_PROFILE = AdaptiveJudgeProfile(
    profile_id="d_only",
    lanes=(FactLane.DIRECT,),
    kind="adaptive_evidence_solver_v3_d",
    arm_label="S0_PLUS_ADAPTIVE_EVIDENCE_SOLVER_V3_D_ONLY",
    arm_plan_id="matched_adaptive_evidence_solver_v3_d_only",
    answer_plan_id="matched_adaptive_evidence_solver_v3_d_only_answer",
)
PARTITION_PROFILE = AdaptiveJudgeProfile(
    profile_id="p_only",
    lanes=(FactLane.PARTITION,),
    kind="adaptive_evidence_solver_v3_p",
    arm_label="S0_PLUS_ADAPTIVE_EVIDENCE_SOLVER_V3_P_ONLY",
    arm_plan_id="matched_adaptive_evidence_solver_v3_p_only",
    answer_plan_id="matched_adaptive_evidence_solver_v3_p_only_answer",
)
GUIDED_PROFILE = AdaptiveJudgeProfile(
    profile_id="g_only",
    lanes=(FactLane.GUIDED,),
    kind="adaptive_evidence_solver_v3_g",
    arm_label="S0_PLUS_ADAPTIVE_EVIDENCE_SOLVER_V3_G_ONLY",
    arm_plan_id="matched_adaptive_evidence_solver_v3_g_only",
    answer_plan_id="matched_adaptive_evidence_solver_v3_g_only_answer",
)
DIRECT_GUIDED_PROFILE = AdaptiveJudgeProfile(
    profile_id="d_g",
    lanes=(FactLane.DIRECT, FactLane.GUIDED),
    kind="adaptive_evidence_solver_v3_dg",
    arm_label="S0_PLUS_ADAPTIVE_EVIDENCE_SOLVER_V3_D_G",
    arm_plan_id="matched_adaptive_evidence_solver_v3_d_g",
    answer_plan_id="matched_adaptive_evidence_solver_v3_d_g_answer",
)
PROFILES = (
    DIRECT_PROFILE,
    PARTITION_PROFILE,
    GUIDED_PROFILE,
    DIRECT_GUIDED_PROFILE,
)
PROFILE_BY_ID = {profile.profile_id: profile for profile in PROFILES}
PROFILE_BY_LANES = {profile.lanes: profile for profile in PROFILES}


SOURCE_ROLES = (
    "sealed_retrieval",
    "query_preflight",
    "query_run",
    "query_adapter",
    "direct_answer_run",
    "direct_runtime_ledger",
    "map_run",
    "map_runtime_ledger",
    "source_preflight",
    "source_work_manifest",
    "source_materialization",
    "lane_filter",
    "adaptive_plan",
    "adaptive_preflight",
    "adaptive_completion_plane",
    "adaptive_run_receipt",
    "adaptive_verified_plane",
    "solver_preflight",
    "answer_run",
)


@dataclass(frozen=True, slots=True)
class VerifiedAdaptiveEvidenceSolverJudgeRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    prediction: str
    prediction_sha256: str
    prediction_source: str
    parent_prediction_sha256: str
    changed_from_parent: bool
    route_id: str
    solver_valid: bool
    solver_decision: str
    solver_used_evidence_ids: tuple[str, ...]
    solver_used_map_item_ids: tuple[str, ...]
    solver_used_source_fact_ids: tuple[str, ...]
    solver_parse_receipt_sha256: str | None
    solver_plan_row_receipt_sha256: str
    completion_receipt_sha256: str | None
    native_result_row_receipt_sha256: str
    source_row_sha256: str
    runtime_row_id: str


@dataclass(frozen=True, slots=True)
class _VerifiedAdaptiveEvidenceSolverJudgePlane:
    run_sha256: str
    replay_sha256: str
    runtime_ledger_sha256: str
    runtime_ledger: Mapping[str, Any]
    parent_answer_run_sha256: str
    adapter_population_id: str
    retrieval_sha256: str
    snapshot_id: str
    rows: tuple[VerifiedAdaptiveEvidenceSolverJudgeRow, ...]
    parent_plane: query_payload_live.VerifiedQueryPayloadAnswerPlane
    map_plane: Any
    profile_id: str
    lanes: tuple[FactLane, ...]
    source_preflight_sha256: str
    source_work_manifest_sha256: str
    source_materialization_sha256: str
    lane_filter_receipt_sha256: str
    solver_preflight_artifact_sha256: str
    native_plan: adaptive.AdaptiveEvidenceSolverPlan
    native_preflight: adaptive.AdaptiveEvidenceSolverPreflight
    native_completion_plane: adaptive.AdaptiveSolverCompletionPlane
    native_run: adaptive.AdaptiveEvidenceSolverRun
    native_verified_plane: adaptive.VerifiedAdaptiveEvidenceSolverPlane

    @property
    def ordered_rows(self) -> tuple[VerifiedAdaptiveEvidenceSolverJudgeRow, ...]:
        return self.rows

    @property
    def changed_rows(self) -> tuple[VerifiedAdaptiveEvidenceSolverJudgeRow, ...]:
        return tuple(row for row in self.rows if row.changed_from_parent)


@dataclass(frozen=True, slots=True)
class VerifiedAdaptiveEvidenceSolverDirectJudgePlane(
    _VerifiedAdaptiveEvidenceSolverJudgePlane
):
    """Exact D-only adaptive solver population."""


@dataclass(frozen=True, slots=True)
class VerifiedAdaptiveEvidenceSolverGuidedJudgePlane(
    _VerifiedAdaptiveEvidenceSolverJudgePlane
):
    """Exact G-only adaptive solver population."""


@dataclass(frozen=True, slots=True)
class VerifiedAdaptiveEvidenceSolverPartitionJudgePlane(
    _VerifiedAdaptiveEvidenceSolverJudgePlane
):
    """Exact P-only adaptive solver population."""


@dataclass(frozen=True, slots=True)
class VerifiedAdaptiveEvidenceSolverDirectGuidedJudgePlane(
    _VerifiedAdaptiveEvidenceSolverJudgePlane
):
    """Exact combined D+G adaptive solver population."""


PLANE_TYPE_BY_PROFILE = {
    DIRECT_PROFILE.profile_id: VerifiedAdaptiveEvidenceSolverDirectJudgePlane,
    PARTITION_PROFILE.profile_id: VerifiedAdaptiveEvidenceSolverPartitionJudgePlane,
    GUIDED_PROFILE.profile_id: VerifiedAdaptiveEvidenceSolverGuidedJudgePlane,
    DIRECT_GUIDED_PROFILE.profile_id: (
        VerifiedAdaptiveEvidenceSolverDirectGuidedJudgePlane
    ),
}
PROFILE_BY_PLANE_TYPE = {
    plane_type: PROFILE_BY_ID[profile_id]
    for profile_id, plane_type in PLANE_TYPE_BY_PROFILE.items()
}


def profile_for_lanes(lanes: tuple[FactLane, ...]) -> AdaptiveJudgeProfile:
    try:
        return PROFILE_BY_LANES[lanes]
    except KeyError as exc:
        raise AdaptiveEvidenceSolverJudgeAdapterError(
            "adaptive judge supports only the exact D, P, G, or D+G lane profiles"
        ) from exc


def profile_for_plane(answer_plane: object) -> AdaptiveJudgeProfile:
    try:
        return PROFILE_BY_PLANE_TYPE[type(answer_plane)]
    except KeyError as exc:
        raise TypeError("answer plane is not an exact adaptive V3 judge plane") from exc


def _source_bindings(
    *,
    profile: AdaptiveJudgeProfile,
    plan: adaptive.AdaptiveEvidenceSolverPlan,
    verified: adaptive.VerifiedAdaptiveEvidenceSolverPlane,
    terminal_run_sha256: str,
    solver_preflight_artifact_sha256: str,
    source_preflight_sha256: str,
    source_work_manifest_sha256: str,
    source_materialization_sha256: str,
    lane_filter_receipt_sha256: str,
) -> dict[str, str]:
    direct_plan = plan.map_plan.direct_plan
    parent = plan.map_plan.direct_plane
    values = {
        "sealed_retrieval": (
            direct_plan.adapter_population.source_population.retrieval_sha256
        ),
        "query_preflight": direct_plan.adapter_population.query_preflight_sha256,
        "query_run": direct_plan.adapter_population.query_run_sha256,
        "query_adapter": direct_plan.adapter_population.population_id,
        "direct_answer_run": parent.run_sha256,
        "direct_runtime_ledger": parent.runtime_ledger_sha256,
        "map_run": plan.map_plane.run_sha256,
        "map_runtime_ledger": plan.map_plane.runtime_ledger_sha256,
        "source_preflight": source_preflight_sha256,
        "source_work_manifest": source_work_manifest_sha256,
        "source_materialization": source_materialization_sha256,
        "lane_filter": lane_filter_receipt_sha256,
        "adaptive_plan": plan.plan_identity_sha256,
        "adaptive_preflight": verified.preflight_receipt_sha256,
        "adaptive_completion_plane": verified.completion_plane_receipt_sha256,
        "adaptive_run_receipt": verified.run_receipt_sha256,
        "adaptive_verified_plane": verified.receipt_sha256,
        "solver_preflight": solver_preflight_artifact_sha256,
        "answer_run": terminal_run_sha256,
    }
    _require(tuple(values) == SOURCE_ROLES, "adaptive judge source roles changed")
    for role, value in values.items():
        require_sha256(value, f"adaptive judge {role}")
    return values


def expected_source_bindings(
    plane: _VerifiedAdaptiveEvidenceSolverJudgePlane,
) -> dict[str, str]:
    """Return the exact source envelope independently of the runtime ledger."""

    profile = profile_for_plane(plane)
    return _source_bindings(
        profile=profile,
        plan=plane.native_plan,
        verified=plane.native_verified_plane,
        terminal_run_sha256=plane.run_sha256,
        solver_preflight_artifact_sha256=(
            plane.solver_preflight_artifact_sha256
        ),
        source_preflight_sha256=plane.source_preflight_sha256,
        source_work_manifest_sha256=plane.source_work_manifest_sha256,
        source_materialization_sha256=plane.source_materialization_sha256,
        lane_filter_receipt_sha256=plane.lane_filter_receipt_sha256,
    )


def _mechanism(prediction_source: str) -> str:
    if prediction_source == "adaptive_validated_evidence_replacement_v3":
        return "terra_adaptive_validated_evidence_replacement_v3"
    if prediction_source == "adaptive_validated_evidence_keep_parent_v3":
        return "terra_adaptive_validated_evidence_keep_parent_v3"
    if prediction_source == "sealed_direct_query_fallback":
        return "sealed_direct_query_prediction_reuse"
    raise AdaptiveEvidenceSolverJudgeAdapterError(
        f"unknown adaptive prediction source: {prediction_source!r}"
    )


def _runtime_entries(
    profile: AdaptiveJudgeProfile,
    plan: adaptive.AdaptiveEvidenceSolverPlan,
    verified: adaptive.VerifiedAdaptiveEvidenceSolverPlane,
) -> tuple[tuple[RuntimeLedgerEntry, ...], tuple[RuntimeLedgerEntry, ...]]:
    _require(
        len(plan.rows) == len(verified.rows),
        "adaptive judge plan/result populations differ",
    )
    all_entries: list[RuntimeLedgerEntry] = []
    answers: list[RuntimeLedgerEntry] = []
    for planned, result in zip(plan.rows, verified.rows, strict=True):
        _require(
            result.ordinal == planned.ordinal
            and result.question_id == planned.question_id
            and result.question_sha256 == planned.map_row.question_sha256
            and result.dated_question_sha256
            == planned.map_row.dated_question_sha256
            and result.plan_row_receipt_sha256 == planned.receipt_sha256,
            f"adaptive judge native row binding changed at {planned.ordinal}",
        )
        operation_id = identity_sha256(
            {
                "format": f"{FORMAT}-operation-id",
                "profile_id": profile.profile_id,
                "solver_plan_row_receipt_sha256": planned.receipt_sha256,
                "stage_id": SOLVER_STAGE_ID,
            }
        )
        operation_ids = (operation_id,) if planned.submitted else ()
        stage = RuntimeLedgerEntry(
            event_type="stage",
            ordinal=planned.ordinal,
            question_id=planned.question_id,
            question_sha256=planned.map_row.question_sha256,
            arm_label=profile.arm_label,
            parent_arm_label=query_payload_live.ARM_LABEL,
            stage_id=SOLVER_STAGE_ID,
            parent_stage_id=MAP_PARENT_STAGE_ID,
            mechanism_id="adaptive_map_plus_source_evidence_solver_v3",
            delta_kind="answer_operator",
            renderer_id=RENDERER_ID,
            legacy_renderer=False,
            disposition=planned.disposition,
            candidate_ids=operation_ids,
            selected_before_dedup_ids=operation_ids,
            admitted_ids=operation_ids,
            provider_calls=0,
            global_provider_prompt_cap=plan.required_calls,
            max_final_prompt_tokens=plan.max_prompt_tokens,
            prompt_token_proxy=planned.prompt_token_proxy,
            parent_packet_sha256=planned.map_plan_row.packet_id,
            packet_sha256=planned.packet_id,
            prompt_id=planned.prompt_id,
            prompt_messages_sha256=planned.messages_sha256,
            delta_sha256=planned.receipt_sha256,
            stage_receipt_sha256=planned.receipt_sha256,
            reason=planned.reason,
        )
        provider_calls = int(planned.submitted)
        answer = RuntimeLedgerEntry(
            event_type="answer_observation",
            ordinal=planned.ordinal,
            question_id=planned.question_id,
            question_sha256=planned.map_row.question_sha256,
            arm_label=profile.arm_label,
            parent_arm_label=query_payload_live.ARM_LABEL,
            stage_id=ANSWER_STAGE_ID,
            parent_stage_id=SOLVER_STAGE_ID,
            mechanism_id=_mechanism(result.prediction_source),
            delta_kind="observation",
            renderer_id=RENDERER_ID,
            legacy_renderer=False,
            disposition=StageDisposition.NO_OP,
            provider_calls=provider_calls,
            provider_prompt_cap=provider_calls,
            provider_prompt_reserved=provider_calls,
            global_provider_prompt_cap=plan.required_calls,
            max_final_prompt_tokens=plan.max_prompt_tokens,
            prompt_token_proxy=planned.prompt_token_proxy,
            parent_packet_sha256=planned.map_plan_row.packet_id,
            packet_sha256=planned.packet_id,
            prompt_id=planned.prompt_id,
            prompt_messages_sha256=planned.messages_sha256,
            prediction=result.prediction,
            prediction_sha256=result.prediction_sha256,
            changed_from_parent=result.changed_from_parent,
            source_row_sha256=result.receipt_sha256,
            reason=f"adaptive_solver_v3_{result.solver_decision}",
        )
        all_entries.extend((stage, answer))
        answers.append(answer)
    return tuple(all_entries), tuple(answers)


def adapt_verified_adaptive_evidence_solver(
    *,
    lanes: tuple[FactLane, ...],
    plan: adaptive.AdaptiveEvidenceSolverPlan,
    preflight: adaptive.AdaptiveEvidenceSolverPreflight,
    completion_plane: adaptive.AdaptiveSolverCompletionPlane,
    run: adaptive.AdaptiveEvidenceSolverRun,
    verified_plane: adaptive.VerifiedAdaptiveEvidenceSolverPlane,
    terminal_run_sha256: str,
    solver_preflight_artifact_sha256: str,
    source_preflight_sha256: str,
    source_work_manifest_sha256: str,
    source_materialization_sha256: str,
    lane_filter_receipt_sha256: str,
) -> _VerifiedAdaptiveEvidenceSolverJudgePlane:
    """Build the exact common judge plane from a native replayed V3 run."""

    profile = profile_for_lanes(lanes)
    if type(plan) is not adaptive.AdaptiveEvidenceSolverPlan:
        raise TypeError("plan must be an exact AdaptiveEvidenceSolverPlan")
    _require(
        type(preflight) is adaptive.AdaptiveEvidenceSolverPreflight
        and type(completion_plane) is adaptive.AdaptiveSolverCompletionPlane
        and type(run) is adaptive.AdaptiveEvidenceSolverRun
        and type(verified_plane) is adaptive.VerifiedAdaptiveEvidenceSolverPlane,
        "adaptive judge requires exact native lifecycle types",
    )
    replayed = adaptive.replay_adaptive_evidence_solver(
        plan, preflight, completion_plane, run
    )
    _require(
        replayed == verified_plane
        and verified_plane.run_receipt_sha256
        == verified_plane.replay_receipt_sha256
        == run.receipt_sha256,
        "adaptive judge native replay changed",
    )
    for value, label in (
        (terminal_run_sha256, "adaptive terminal run"),
        (solver_preflight_artifact_sha256, "adaptive solver preflight artifact"),
        (source_preflight_sha256, "adaptive source preflight"),
        (source_work_manifest_sha256, "adaptive source work manifest"),
        (source_materialization_sha256, "adaptive source materialization"),
        (lane_filter_receipt_sha256, "adaptive lane-filter receipt"),
    ):
        require_sha256(value, label)
    parent = plan.map_plan.direct_plane
    _require(
        type(parent) is query_payload_live.VerifiedQueryPayloadAnswerPlane
        and len(plan.rows) == len(parent.rows),
        "adaptive judge lost its exact direct parent population",
    )
    selected_lanes = set(lanes)
    for planned in plan.rows:
        union = planned.fact_union
        if union is None:
            continue
        observed_lanes = {
            origin.lane
            for fact in union.union_facts_before_direct_exclusion
            for origin in fact.origins
        }
        _require(
            observed_lanes <= selected_lanes,
            f"adaptive judge source fact escaped lane profile at {planned.ordinal}",
        )
    bindings = _source_bindings(
        profile=profile,
        plan=plan,
        verified=verified_plane,
        terminal_run_sha256=terminal_run_sha256,
        solver_preflight_artifact_sha256=solver_preflight_artifact_sha256,
        source_preflight_sha256=source_preflight_sha256,
        source_work_manifest_sha256=source_work_manifest_sha256,
        source_materialization_sha256=source_materialization_sha256,
        lane_filter_receipt_sha256=lane_filter_receipt_sha256,
    )
    entries, answer_entries = _runtime_entries(profile, plan, verified_plane)
    runtime = build_runtime_ledger(
        snapshot_id=plan.map_plan.direct_plan.adapter_population.source_population.snapshot.snapshot_id,
        plan_id=profile.answer_plan_id,
        entries=entries,
        source_artifacts=tuple(
            {
                "role": f"{profile.arm_label}:{role}",
                "sha256": bindings[role],
            }
            for role in SOURCE_ROLES
        ),
    )
    rows = tuple(
        VerifiedAdaptiveEvidenceSolverJudgeRow(
            ordinal=result.ordinal,
            question_id=result.question_id,
            question_sha256=result.question_sha256,
            dated_question_sha256=result.dated_question_sha256,
            prediction=result.prediction,
            prediction_sha256=result.prediction_sha256,
            prediction_source=result.prediction_source,
            parent_prediction_sha256=result.parent_prediction_sha256,
            changed_from_parent=result.changed_from_parent,
            route_id=planned.map_row.route_id,
            solver_valid=result.solver_valid,
            solver_decision=result.solver_decision,
            solver_used_evidence_ids=result.solver_used_evidence_ids,
            solver_used_map_item_ids=result.solver_used_map_item_ids,
            solver_used_source_fact_ids=result.solver_used_source_fact_ids,
            solver_parse_receipt_sha256=result.solver_parse_receipt_sha256,
            solver_plan_row_receipt_sha256=result.plan_row_receipt_sha256,
            completion_receipt_sha256=result.completion_receipt_sha256,
            native_result_row_receipt_sha256=result.receipt_sha256,
            source_row_sha256=result.receipt_sha256,
            runtime_row_id=answer.row_id,
        )
        for planned, result, answer in zip(
            plan.rows, verified_plane.rows, answer_entries, strict=True
        )
    )
    plane_type = PLANE_TYPE_BY_PROFILE[profile.profile_id]
    return plane_type(
        run_sha256=terminal_run_sha256,
        replay_sha256=terminal_run_sha256,
        runtime_ledger_sha256=sha256(canonical_json_bytes(runtime)).hexdigest(),
        runtime_ledger=live._freeze_json(runtime),
        parent_answer_run_sha256=parent.run_sha256,
        adapter_population_id=(
            plan.map_plan.direct_plan.adapter_population.population_id
        ),
        retrieval_sha256=(
            plan.map_plan.direct_plan.adapter_population.source_population.retrieval_sha256
        ),
        snapshot_id=(
            plan.map_plan.direct_plan.adapter_population.source_population.snapshot.snapshot_id
        ),
        rows=rows,
        parent_plane=parent,
        map_plane=plan.map_plane,
        profile_id=profile.profile_id,
        lanes=lanes,
        source_preflight_sha256=source_preflight_sha256,
        source_work_manifest_sha256=source_work_manifest_sha256,
        source_materialization_sha256=source_materialization_sha256,
        lane_filter_receipt_sha256=lane_filter_receipt_sha256,
        solver_preflight_artifact_sha256=solver_preflight_artifact_sha256,
        native_plan=plan,
        native_preflight=preflight,
        native_completion_plane=completion_plane,
        native_run=run,
        native_verified_plane=verified_plane,
    )


def validate_adaptive_judge_plane(
    answer_plane: object,
) -> AdaptiveJudgeProfile:
    """Rebuild the adapter projection and require exact byte-level equality."""

    profile = profile_for_plane(answer_plane)
    plane = answer_plane
    assert isinstance(plane, _VerifiedAdaptiveEvidenceSolverJudgePlane)
    _require(
        plane.profile_id == profile.profile_id and plane.lanes == profile.lanes,
        "adaptive judge lane profile changed",
    )
    rebuilt = adapt_verified_adaptive_evidence_solver(
        lanes=plane.lanes,
        plan=plane.native_plan,
        preflight=plane.native_preflight,
        completion_plane=plane.native_completion_plane,
        run=plane.native_run,
        verified_plane=plane.native_verified_plane,
        terminal_run_sha256=plane.run_sha256,
        solver_preflight_artifact_sha256=(
            plane.solver_preflight_artifact_sha256
        ),
        source_preflight_sha256=plane.source_preflight_sha256,
        source_work_manifest_sha256=plane.source_work_manifest_sha256,
        source_materialization_sha256=plane.source_materialization_sha256,
        lane_filter_receipt_sha256=plane.lane_filter_receipt_sha256,
    )
    _require(rebuilt == plane, "adaptive judge projection changed")
    return profile


__all__ = [
    "ANSWER_STAGE_ID",
    "DIRECT_GUIDED_PROFILE",
    "DIRECT_PROFILE",
    "GUIDED_PROFILE",
    "PARTITION_PROFILE",
    "PROFILES",
    "RENDERER_ID",
    "SOURCE_ROLES",
    "SOLVER_STAGE_ID",
    "AdaptiveEvidenceSolverJudgeAdapterError",
    "AdaptiveJudgeProfile",
    "VerifiedAdaptiveEvidenceSolverDirectGuidedJudgePlane",
    "VerifiedAdaptiveEvidenceSolverDirectJudgePlane",
    "VerifiedAdaptiveEvidenceSolverGuidedJudgePlane",
    "VerifiedAdaptiveEvidenceSolverPartitionJudgePlane",
    "VerifiedAdaptiveEvidenceSolverJudgeRow",
    "adapt_verified_adaptive_evidence_solver",
    "expected_source_bindings",
    "profile_for_lanes",
    "profile_for_plane",
    "validate_adaptive_judge_plane",
]
