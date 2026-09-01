#!/usr/bin/env python3
"""Provider-free posthoc target analysis for the two sealed closure arms.

The gold-blind v9 generation, its eligibility manifest, and the underlying
locked S0 population are fully verified before this tool opens the immutable
gold-bearing target plan.  The resulting artifact is evaluation-only and may
never be consumed by a retrieval or answer runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from tools import score_locked_retrieval_target_ownership as target_scorer
from tools.build_locked_retrieval_target_registry import _validate_plan
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.closure import (
    ARM_LABELS,
    GLOBAL_ARM,
    REPRESENTATIVE_ARM,
    IndependentClosureGeneration,
    IndependentClosureStructuralProjection,
    build_structural_target_projection,
    load_independent_closure_generation,
)
from tools.matched_eval.contracts import identity_sha256, require_sha256
from tools.matched_eval.population import load_s0_population


FORMAT = "memory-condense-independent-closure-target-analysis-v2"
STAGES = (
    "raw_candidate_reach",
    "selected_before_dedup",
    "post_s0_admission",
)
REPRESENTATIVE_ROUTE = "representative"
GLOBAL_ROUTE = "global"
UNION_ROUTE = "union"
ROUTES = (REPRESENTATIVE_ROUTE, GLOBAL_ROUTE, UNION_ROUTE)
ROUTE_TO_ARM = {
    REPRESENTATIVE_ROUTE: REPRESENTATIVE_ARM,
    GLOBAL_ROUTE: GLOBAL_ARM,
}
ARM_TO_ROUTE = {value: key for key, value in ROUTE_TO_ARM.items()}

DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
DEFAULT_CLOSURE_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/independent-closure-v9"
)
DEFAULT_GENERATION = DEFAULT_CLOSURE_ROOT / "retrieval-generation.json"
DEFAULT_ELIGIBILITY_MANIFEST = DEFAULT_CLOSURE_ROOT / "eligibility-manifest.json"
DEFAULT_TARGET_PLAN = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/target-owner-plan-v1/target-plan.json"
)
EXPECTED_RETRIEVAL_SHA256 = (
    "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
)
EXPECTED_ELIGIBILITY_MANIFEST_SHA256 = (
    "748bd56a7efb8fd70d36bc96f099a53fc506469565577de9635908f6773bdee1"
)
PINNED_TARGET_PLAN_SHA256 = (
    "b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff"
)


class ClosureTargetAnalysisError(ValueError):
    """Raised when a sealed input or posthoc join invariant changes."""


def _require(ok: Any, message: str) -> None:
    if not ok:
        raise ClosureTargetAnalysisError(message)


def _normalized_event(
    *,
    target_id: str,
    source_id: str,
    discovering_method: str,
) -> dict[str, Any]:
    """Return the exact event surface consumed by the common target scorer."""

    return {
        "target_id": target_id,
        "target_kind": "evidence_atom",
        "discovering_method": discovering_method,
        "_source_target_ids": [source_id],
    }


def _project_runtime_stages(
    generation: IndependentClosureGeneration,
) -> tuple[
    dict[str, IndependentClosureStructuralProjection],
    dict[str, dict[str, dict[int, tuple[dict[str, Any], ...]]]],
]:
    """Project raw, selected, and admitted events without opening target tags."""

    _require(
        type(generation) is IndependentClosureGeneration,
        "target analysis requires an exact independent-closure generation",
    )
    projections = {
        arm_label: build_structural_target_projection(generation, arm_label)
        for arm_label in ARM_LABELS
    }
    events: dict[str, dict[str, dict[int, tuple[dict[str, Any], ...]]]] = {
        route: {stage: {} for stage in STAGES}
        for route in (REPRESENTATIVE_ROUTE, GLOBAL_ROUTE)
    }
    for question in generation.questions:
        for route, arm_label in ROUTE_TO_ARM.items():
            arm = question.arm(arm_label)
            structural = projections[arm_label].questions[question.ordinal]
            _require(
                structural.ordinal == question.ordinal
                and structural.question_id == question.question_id,
                f"closure structural projection order changed at {question.ordinal}",
            )
            raw = (
                ()
                if arm is None
                else tuple(
                    _normalized_event(
                        target_id=row.atom_id,
                        source_id=row.source_id,
                        discovering_method=arm_label,
                    )
                    for row in arm.targets
                )
            )
            selected = tuple(
                _normalized_event(
                    target_id=row.target_id,
                    source_id=row.source_target_ids[0],
                    discovering_method=arm_label,
                )
                for row in structural.selected_targets_before_dedup
            )
            admitted = tuple(
                _normalized_event(
                    target_id=row.target_id,
                    source_id=row.source_target_ids[0],
                    discovering_method=arm_label,
                )
                for row in structural.admitted_targets_after_dedup
            )
            raw_ids = {row["target_id"] for row in raw}
            selected_ids = {row["target_id"] for row in selected}
            admitted_ids = {row["target_id"] for row in admitted}
            _require(
                admitted_ids <= selected_ids <= raw_ids,
                f"closure stage partition changed at {question.ordinal} for {arm_label}",
            )
            events[route]["raw_candidate_reach"][question.ordinal] = raw
            events[route]["selected_before_dedup"][question.ordinal] = selected
            events[route]["post_s0_admission"][question.ordinal] = admitted
    return projections, events


def _events_for(
    events: Mapping[str, Mapping[str, Mapping[int, tuple[dict[str, Any], ...]]]],
    *,
    route: str,
    stage: str,
    ordinal: int,
) -> tuple[dict[str, Any], ...]:
    if route == UNION_ROUTE:
        return tuple(events[REPRESENTATIVE_ROUTE][stage][ordinal]) + tuple(
            events[GLOBAL_ROUTE][stage][ordinal]
        )
    return tuple(events[route][stage][ordinal])


def _source_aliases(
    target: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> set[str]:
    question_id = str(target["question_id"])
    aliases: set[str] = set()
    for event in events:
        aliases.update(target_scorer._source_aliases(event, question_id))
    return aliases


def _relation_operand_complete(
    target: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> bool:
    if target.get("target_kind") != "relation":
        return False
    basis = target.get("assignment_basis")
    expected = basis.get("expected_source_ids") if isinstance(basis, Mapping) else None
    _require(
        isinstance(expected, Sequence)
        and not isinstance(expected, (str, bytes))
        and bool(expected)
        and all(isinstance(row, str) and row for row in expected),
        "relation target lacks exact expected source operands",
    )
    return set(expected) <= _source_aliases(target, events)


def _ratio(hit_count: int, target_count: int) -> float:
    return hit_count / target_count if target_count else 0.0


def _metric(targets: Sequence[Mapping[str, Any]], hits: Sequence[bool]) -> dict[str, Any]:
    hit_count = sum(hits)
    return {
        "target_count": len(targets),
        "hit_count": hit_count,
        "recall": _ratio(hit_count, len(targets)),
    }


def _route_stage_summary(
    *,
    route: str,
    targets: Sequence[Mapping[str, Any]],
    route_hits: Sequence[bool],
    relation_complete: Sequence[bool],
    primary_hits: Mapping[str, Sequence[bool]],
    events_by_ordinal: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    sources = [row for row in targets if row["target_kind"] == "source_id"]
    source_hits = [
        hit
        for row, hit in zip(targets, route_hits, strict=True)
        if row["target_kind"] == "source_id"
    ]
    relations = [row for row in targets if row["target_kind"] == "relation"]
    relation_hits = [
        hit
        for row, hit in zip(targets, route_hits, strict=True)
        if row["target_kind"] == "relation"
    ]
    relation_operand_hits = [
        hit
        for row, hit in zip(targets, relation_complete, strict=True)
        if row["target_kind"] == "relation"
    ]
    coverage = [
        row for row in targets if row["target_kind"] == "coverage_check"
    ]
    coverage_hits = [
        hit
        for row, hit in zip(targets, route_hits, strict=True)
        if row["target_kind"] == "coverage_check"
    ]
    owner_scope = {
        REPRESENTATIVE_ROUTE: (REPRESENTATIVE_ROUTE,),
        GLOBAL_ROUTE: (GLOBAL_ROUTE,),
        UNION_ROUTE: (REPRESENTATIVE_ROUTE, GLOBAL_ROUTE),
    }[route]
    primary_targets = [row for row in targets if row["primary_owner"] in owner_scope]
    primary_route_hits = [
        hit
        for row, hit in zip(targets, route_hits, strict=True)
        if row["primary_owner"] in owner_scope
    ]
    owner_metrics: dict[str, Any] = {}
    for owner in sorted({str(row["primary_owner"]) for row in targets}):
        owner_targets = [row for row in targets if row["primary_owner"] == owner]
        owner_hits = [
            hit
            for row, hit in zip(targets, route_hits, strict=True)
            if row["primary_owner"] == owner
        ]
        owner_source_targets = [
            row for row in owner_targets if row["target_kind"] == "source_id"
        ]
        owner_source_hits = [
            hit
            for row, hit in zip(targets, route_hits, strict=True)
            if row["primary_owner"] == owner and row["target_kind"] == "source_id"
        ]
        owner_metrics[owner] = _metric(owner_targets, owner_hits) | {
            "source_targets": _metric(owner_source_targets, owner_source_hits)
        }

    if route == REPRESENTATIVE_ROUTE:
        rescued_owner = GLOBAL_ROUTE
        alternate_route = REPRESENTATIVE_ROUTE
    elif route == GLOBAL_ROUTE:
        rescued_owner = REPRESENTATIVE_ROUTE
        alternate_route = GLOBAL_ROUTE
    else:
        rescued_owner = None
        alternate_route = None
    rescue_hashes: list[str] = []
    if rescued_owner is not None and alternate_route is not None:
        primary = primary_hits[rescued_owner]
        alternate = primary_hits[alternate_route]
        rescue_hashes = [
            str(row["target_sha256"])
            for row, primary_hit, alternate_hit in zip(
                targets, primary, alternate, strict=True
            )
            if row["primary_owner"] == rescued_owner
            and not primary_hit
            and alternate_hit
        ]
    else:
        for owner, alternate in (
            (REPRESENTATIVE_ROUTE, GLOBAL_ROUTE),
            (GLOBAL_ROUTE, REPRESENTATIVE_ROUTE),
        ):
            rescue_hashes.extend(
                str(row["target_sha256"])
                for row, primary_hit, alternate_hit in zip(
                    targets,
                    primary_hits[owner],
                    primary_hits[alternate],
                    strict=True,
                )
                if row["primary_owner"] == owner
                and not primary_hit
                and alternate_hit
            )

    event_count = sum(len(rows) for rows in events_by_ordinal.values())
    exact_sources = {
        (ordinal, source_id)
        for ordinal, rows in events_by_ordinal.items()
        for event in rows
        for source_id in event["_source_target_ids"]
    }
    return {
        "event_count": event_count,
        "question_with_event_count": sum(bool(rows) for rows in events_by_ordinal.values()),
        "unique_question_source_count": len(exact_sources),
        "all_formal_targets": _metric(targets, route_hits),
        "source_targets": _metric(sources, source_hits),
        "formal_relation_targets": _metric(relations, relation_hits),
        "relation_operand_completeness": _metric(
            relations, relation_operand_hits
        ),
        "coverage_check_targets": _metric(coverage, coverage_hits),
        "primary_owner_scope": list(owner_scope),
        "primary_owner_targets": _metric(primary_targets, primary_route_hits),
        "by_primary_owner": owner_metrics,
        "alternate_method_rescue_count": len(rescue_hashes),
        "alternate_method_rescue_target_sha256s": rescue_hashes,
    }


def score_target_reach(
    plan: Mapping[str, Any],
    events: Mapping[
        str,
        Mapping[str, Mapping[int, tuple[dict[str, Any], ...]]],
    ],
) -> dict[str, Any]:
    """Score closure-only reach after the immutable plan has been opened."""

    targets_raw = plan.get("desired_targets")
    questions_raw = plan.get("ordered_question_keys")
    _require(isinstance(targets_raw, list), "target plan lacks desired targets")
    _require(isinstance(questions_raw, list), "target plan lacks question order")
    targets = [dict(row) for row in targets_raw if isinstance(row, Mapping)]
    _require(len(targets) == len(targets_raw), "target plan contains an invalid target")
    question_count = len(questions_raw)
    for route in (REPRESENTATIVE_ROUTE, GLOBAL_ROUTE):
        _require(route in events, f"missing closure events for {route}")
        for stage in STAGES:
            rows = events[route].get(stage)
            _require(
                isinstance(rows, Mapping)
                and set(rows) == set(range(question_count)),
                f"closure event population changed for {route}/{stage}",
            )

    target_rows: list[dict[str, Any]] = []
    route_summaries: dict[str, dict[str, Any]] = {
        route: {"stages": {}} for route in ROUTES
    }
    rescue_summary: dict[str, Any] = {}
    closure_owner_outcomes: dict[str, Any] = {}

    for stage in STAGES:
        hits: dict[str, list[bool]] = {route: [] for route in ROUTES}
        operand_complete: dict[str, list[bool]] = {route: [] for route in ROUTES}
        for target in targets:
            ordinal = int(target["ordinal"])
            for route in ROUTES:
                route_events = _events_for(
                    events, route=route, stage=stage, ordinal=ordinal
                )
                hits[route].append(target_scorer._target_reached(target, route_events))
                operand_complete[route].append(
                    _relation_operand_complete(target, route_events)
                    if target["target_kind"] == "relation"
                    else False
                )

        for route in ROUTES:
            route_events_by_ordinal = {
                ordinal: _events_for(
                    events, route=route, stage=stage, ordinal=ordinal
                )
                for ordinal in range(question_count)
            }
            route_summaries[route]["stages"][stage] = _route_stage_summary(
                route=route,
                targets=targets,
                route_hits=hits[route],
                relation_complete=operand_complete[route],
                primary_hits={
                    REPRESENTATIVE_ROUTE: hits[REPRESENTATIVE_ROUTE],
                    GLOBAL_ROUTE: hits[GLOBAL_ROUTE],
                },
                events_by_ordinal=route_events_by_ordinal,
            )

        stage_rescues: dict[str, Any] = {}
        stage_outcomes: dict[str, Any] = {}
        for owner, primary_route, alternate_route in (
            (
                REPRESENTATIVE_ROUTE,
                REPRESENTATIVE_ROUTE,
                GLOBAL_ROUTE,
            ),
            (GLOBAL_ROUTE, GLOBAL_ROUTE, REPRESENTATIVE_ROUTE),
        ):
            owner_indexes = [
                index
                for index, target in enumerate(targets)
                if target["primary_owner"] == owner
            ]
            rescued = [
                index
                for index in owner_indexes
                if not hits[primary_route][index] and hits[alternate_route][index]
            ]
            union_misses = [
                index for index in owner_indexes if not hits[UNION_ROUTE][index]
            ]
            both = [
                index
                for index in owner_indexes
                if hits[primary_route][index] and hits[alternate_route][index]
            ]
            stage_rescues[f"{owner}_owner_rescued_by_{alternate_route}"] = {
                "count": len(rescued),
                "target_sha256s": [targets[index]["target_sha256"] for index in rescued],
            }
            primary_hit_count = sum(hits[primary_route][index] for index in owner_indexes)
            union_hit_count = sum(hits[UNION_ROUTE][index] for index in owner_indexes)
            stage_outcomes[owner] = {
                "target_count": len(owner_indexes),
                "primary_hit_count": primary_hit_count,
                "primary_miss_count": len(owner_indexes) - primary_hit_count,
                "alternate_method_rescue_count": len(rescued),
                "both_arms_hit_count": len(both),
                "union_hit_count": union_hit_count,
                "union_miss_count": len(union_misses),
                "union_miss_target_sha256s": [
                    targets[index]["target_sha256"] for index in union_misses
                ],
            }
        stage_rescues["total_count"] = sum(
            row["count"] for row in stage_rescues.values()
        )
        rescue_summary[stage] = stage_rescues
        closure_owner_outcomes[stage] = stage_outcomes

        for index, target in enumerate(targets):
            if stage == STAGES[0]:
                target_rows.append(
                    {
                        "ordinal": target["ordinal"],
                        "question_id": target["question_id"],
                        "target_kind": target["target_kind"],
                        "target_id": target["target_id"],
                        "target_sha256": target["target_sha256"],
                        "primary_owner": target["primary_owner"],
                        "stages": {},
                    }
                )
            owner = str(target["primary_owner"])
            primary_route = owner if owner in ROUTE_TO_ARM else None
            alternate_route = (
                GLOBAL_ROUTE
                if primary_route == REPRESENTATIVE_ROUTE
                else REPRESENTATIVE_ROUTE
                if primary_route == GLOBAL_ROUTE
                else None
            )
            target_rows[index]["stages"][stage] = {
                "formal_hit_by_route": {
                    route: hits[route][index] for route in ROUTES
                },
                "relation_operand_complete_by_route": (
                    {
                        route: operand_complete[route][index] for route in ROUTES
                    }
                    if target["target_kind"] == "relation"
                    else None
                ),
                "primary_route": primary_route,
                "primary_formal_hit": (
                    hits[primary_route][index] if primary_route is not None else None
                ),
                "alternate_route": alternate_route,
                "alternate_method_rescue": (
                    not hits[primary_route][index] and hits[alternate_route][index]
                    if primary_route is not None and alternate_route is not None
                    else False
                ),
            }

    return {
        "routes": route_summaries,
        "alternate_method_rescues_by_stage": rescue_summary,
        "closure_primary_owner_outcomes_by_stage": closure_owner_outcomes,
        "targets": target_rows,
    }


def eligible_incremental_source_funnel(
    plan: Mapping[str, Any],
    generation: IndependentClosureGeneration,
    events: Mapping[
        str,
        Mapping[str, Mapping[int, tuple[dict[str, Any], ...]]],
    ],
) -> dict[str, Any]:
    """Expose marginal closure reach over S0 on the eligible denominator.

    The full-plan 188-source accounting remains useful, but closure ran for
    only 79 questions.  This projection prevents the 26 ineligible source
    targets and the closure pool's S0 redundancy from being mistaken for
    incremental mechanism recall.
    """

    eligible_ordinals = {
        row.ordinal for row in generation.questions if row.eligible
    }
    source_targets = [
        row
        for row in plan["desired_targets"]
        if row["target_kind"] == "source_id"
        and int(row["ordinal"]) in eligible_ordinals
    ]
    s0_events: dict[int, tuple[dict[str, Any], ...]] = {
        row.ordinal: tuple(
            _normalized_event(
                target_id=evidence.evidence_id,
                source_id=evidence.source_id,
                discovering_method="causal_graph_coverage_predecessor",
            )
            for evidence in row.root_protected_evidence
        )
        for row in generation.questions
    }
    s0_hits = [
        target_scorer._target_reached(target, s0_events[int(target["ordinal"])])
        for target in source_targets
    ]
    missing_indexes = [index for index, hit in enumerate(s0_hits) if not hit]
    route_rows: dict[str, Any] = {}
    for route in ROUTES:
        route_hits = [
            target_scorer._target_reached(
                target,
                _events_for(
                    events,
                    route=route,
                    stage="raw_candidate_reach",
                    ordinal=int(target["ordinal"]),
                ),
            )
            for target in source_targets
        ]
        novel_hits = [route_hits[index] for index in missing_indexes]
        route_rows[route] = {
            "all_eligible_source_targets": _metric(source_targets, route_hits),
            "novel_over_s0_missing_sources": _metric(
                [source_targets[index] for index in missing_indexes],
                novel_hits,
            ),
            "overlap_with_s0_hit_count": sum(
                route_hit and s0_hit
                for route_hit, s0_hit in zip(route_hits, s0_hits, strict=True)
            ),
        }
    return {
        "eligible_question_count": len(eligible_ordinals),
        "eligible_source_target_count": len(source_targets),
        "ineligible_source_target_count": sum(
            row["target_kind"] == "source_id"
            and int(row["ordinal"]) not in eligible_ordinals
            for row in plan["desired_targets"]
        ),
        "s0": _metric(source_targets, s0_hits),
        "s0_missing": {
            "target_count": len(missing_indexes),
            "target_sha256s": [
                source_targets[index]["target_sha256"] for index in missing_indexes
            ],
        },
        "raw_closure_routes": route_rows,
    }


def _load_pinned_target_plan(path: Path) -> tuple[dict[str, Any], str]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == PINNED_TARGET_PLAN_SHA256,
        "target-plan file differs from the immutable pinned checkpoint",
    )
    plan = _validate_plan(artifact.payload)
    _require(
        plan.get("gold_target_tags_posthoc_only") is True
        and plan.get("runtime_use_forbidden") is True
        and plan.get("provider_calls") == 0,
        "target plan lost its posthoc-only firewall",
    )
    return plan, artifact.sha256


def build_analysis_payload(
    *,
    generation: IndependentClosureGeneration,
    projections: Mapping[str, IndependentClosureStructuralProjection],
    events: Mapping[
        str,
        Mapping[str, Mapping[int, tuple[dict[str, Any], ...]]],
    ],
    plan: Mapping[str, Any],
    target_plan_file_sha256: str,
) -> dict[str, Any]:
    """Join a verified closure runtime to a validated immutable target plan."""

    _require(
        plan.get("population_identity_sha256")
        == generation.population_identity_sha256,
        "target plan belongs to another population",
    )
    expected_questions = [
        {"ordinal": row.ordinal, "question_id": row.question_id}
        for row in generation.questions
    ]
    _require(
        plan.get("ordered_question_keys") == expected_questions,
        "target plan question order differs from closure generation",
    )
    _require(
        set(projections) == set(ARM_LABELS)
        and all(
            projection.source_retrieval_generation_sha256
            == generation.source_retrieval_generation_sha256
            and projection.population_identity_sha256
            == generation.population_identity_sha256
            for projection in projections.values()
        ),
        "closure structural projection binding changed",
    )
    scored = score_target_reach(plan, events)
    incremental = eligible_incremental_source_funnel(plan, generation, events)
    payload: dict[str, Any] = {
        "format": FORMAT,
        "runtime_use_forbidden": True,
        "gold_target_tags_posthoc_only": True,
        "provider_calls": 0,
        "runtime_artifacts_verified_before_target_plan_load": True,
        "question_count": len(generation.questions),
        "eligible_question_count": sum(row.eligible for row in generation.questions),
        "desired_target_count": plan["desired_target_count"],
        "population_identity_sha256": generation.population_identity_sha256,
        "bindings": {
            "retrieval_sha256": generation.retrieval_sha256,
            "retrieval_generation_file_sha256": (
                generation.source_retrieval_generation_sha256
            ),
            "eligibility_manifest_file_sha256": (
                generation.source_eligibility_manifest_sha256
            ),
            "preflight_sha256": generation.preflight_sha256,
            "policy_receipt_sha256": generation.policy_receipt_sha256,
            "target_plan_file_sha256": target_plan_file_sha256,
            "target_plan_identity_sha256": plan["plan_sha256"],
            "structural_projection_identity_sha256s": {
                ARM_TO_ROUTE[arm_label]: projections[arm_label].projection_sha256
                for arm_label in ARM_LABELS
            },
        },
        "arm_bindings": {
            REPRESENTATIVE_ROUTE: REPRESENTATIVE_ARM,
            GLOBAL_ROUTE: GLOBAL_ARM,
            UNION_ROUTE: list(ARM_LABELS),
        },
        "stage_definitions": {
            "raw_candidate_reach": "all structurally reachable candidate atoms",
            "selected_before_dedup": (
                "selected atoms before exact protected-S0 deduplication"
            ),
            "post_s0_admission": (
                "novel selected atoms admitted after protected-S0 deduplication"
            ),
            "formal_relation_hit": (
                "requires a relation-kind structural event under the common target scorer"
            ),
            "relation_operand_complete": (
                "all expected source operands are present; diagnostic only and never "
                "counted as a formal relation hit"
            ),
            "alternate_method_rescue": (
                "the primary closure arm missed its owned target and the other closure "
                "arm hit it at the same stage"
            ),
        },
        "eligible_incremental_source_funnel": incremental,
        **scored,
    }
    payload["analysis_sha256"] = identity_sha256(payload)
    return payload


def analyze_paths(
    *,
    retrieval_path: Path,
    expected_retrieval_sha256: str,
    generation_path: Path,
    expected_generation_sha256: str,
    eligibility_manifest_path: Path,
    expected_eligibility_manifest_sha256: str,
    target_plan_path: Path,
) -> dict[str, Any]:
    """Verify the complete runtime boundary before opening the target plan."""

    require_sha256(expected_retrieval_sha256, "expected retrieval SHA-256")
    require_sha256(expected_generation_sha256, "expected generation SHA-256")
    require_sha256(
        expected_eligibility_manifest_sha256,
        "expected eligibility-manifest SHA-256",
    )
    population = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
    )
    generation = load_independent_closure_generation(
        generation_path,
        expected_generation_sha256=expected_generation_sha256,
        eligibility_manifest_path=eligibility_manifest_path,
        expected_eligibility_manifest_sha256=(
            expected_eligibility_manifest_sha256
        ),
        population=population,
    )
    projections, events = _project_runtime_stages(generation)

    # Gold-bearing data is intentionally not opened above this line.
    plan, target_plan_file_sha256 = _load_pinned_target_plan(target_plan_path)
    return build_analysis_payload(
        generation=generation,
        projections=projections,
        events=events,
        plan=plan,
        target_plan_file_sha256=target_plan_file_sha256,
    )


def run_analysis(
    *,
    retrieval_path: Path,
    expected_retrieval_sha256: str,
    generation_path: Path,
    expected_generation_sha256: str,
    eligibility_manifest_path: Path,
    expected_eligibility_manifest_sha256: str,
    target_plan_path: Path,
    output_path: Path,
) -> tuple[SealedArtifact, bool]:
    payload = analyze_paths(
        retrieval_path=retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        generation_path=generation_path,
        expected_generation_sha256=expected_generation_sha256,
        eligibility_manifest_path=eligibility_manifest_path,
        expected_eligibility_manifest_sha256=(
            expected_eligibility_manifest_sha256
        ),
        target_plan_path=target_plan_path,
    )
    return publish_sealed_json(output_path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256", default=EXPECTED_RETRIEVAL_SHA256
    )
    parser.add_argument("--generation", type=Path, default=DEFAULT_GENERATION)
    parser.add_argument("--expected-generation-sha256", required=True)
    parser.add_argument(
        "--eligibility-manifest",
        type=Path,
        default=DEFAULT_ELIGIBILITY_MANIFEST,
    )
    parser.add_argument(
        "--expected-eligibility-manifest-sha256",
        default=EXPECTED_ELIGIBILITY_MANIFEST_SHA256,
    )
    parser.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact, created = run_analysis(
        retrieval_path=args.retrieval,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        generation_path=args.generation,
        expected_generation_sha256=args.expected_generation_sha256,
        eligibility_manifest_path=args.eligibility_manifest,
        expected_eligibility_manifest_sha256=(
            args.expected_eligibility_manifest_sha256
        ),
        target_plan_path=args.target_plan,
        output_path=args.output,
    )
    union = artifact.payload["routes"][UNION_ROUTE]["stages"]
    concise = {
        stage: {
            "source_hits": values["source_targets"]["hit_count"],
            "relation_hits": values["formal_relation_targets"]["hit_count"],
            "relation_operands_complete": values[
                "relation_operand_completeness"
            ]["hit_count"],
        }
        for stage, values in union.items()
    }
    print(
        f"Closure target analysis {artifact.sha256} "
        f"({'created' if created else 'reused'}): "
        f"{json.dumps(concise, sort_keys=True, separators=(',', ':'))}; "
        "provider_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ELIGIBILITY_MANIFEST_SHA256",
    "EXPECTED_RETRIEVAL_SHA256",
    "FORMAT",
    "PINNED_TARGET_PLAN_SHA256",
    "STAGES",
    "ClosureTargetAnalysisError",
    "analyze_paths",
    "build_analysis_payload",
    "build_parser",
    "eligible_incremental_source_funnel",
    "main",
    "run_analysis",
    "score_target_reach",
]
