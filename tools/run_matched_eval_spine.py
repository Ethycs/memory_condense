#!/usr/bin/env python3
"""Entry point for the matched retrieval evaluation spine.

The first command migrates sealed historical observations without making model
calls.  The versioned S0 commands build a common-renderer prompt population,
execute an explicitly authorized Terra answer batch, verify it without calls,
and only then allow an explicitly authorized Sol judge batch to load post-hoc
gold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import (
    StageDisposition,
    assert_gold_blind,
    identity_sha256,
)
from tools.matched_eval.ledger import (
    RuntimeLedgerEntry,
    ScoreLedgerEntry,
    build_runtime_ledger,
    build_score_ledger,
)


DEFAULT_CAMPAIGN_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_CAMPAIGN_ROOT / "matched-eval-spine-v2"
DEFAULT_S0_V2_ROOT = DEFAULT_OUTPUT_ROOT / "s0-control-v2"
DEFAULT_V3_OUTPUT_ROOT = DEFAULT_CAMPAIGN_ROOT / "matched-eval-spine-v3"
DEFAULT_S0_V3_ROOT = DEFAULT_V3_OUTPUT_ROOT / "s0-control-v3"
DEFAULT_V4_OUTPUT_ROOT = DEFAULT_CAMPAIGN_ROOT / "matched-eval-spine-v4"
DEFAULT_S0_V4_ROOT = DEFAULT_V4_OUTPUT_ROOT / "s0-control-v4"
DEFAULT_S0_V4_DIAGNOSTIC_ROOT = DEFAULT_V4_OUTPUT_ROOT / "s0-control-v4-flip10"
DEFAULT_CLOSURE_V9_ROOT = DEFAULT_CAMPAIGN_ROOT / "independent-closure-v9"
DEFAULT_CLOSURE_GENERATION = DEFAULT_CLOSURE_V9_ROOT / "retrieval-generation.json"
DEFAULT_CLOSURE_ELIGIBILITY = DEFAULT_CLOSURE_V9_ROOT / "eligibility-manifest.json"
DEFAULT_CLOSURE_REPRESENTATIVE_ANSWER_ROOT = (
    DEFAULT_OUTPUT_ROOT / "s0-plus-representative-bridge"
)
DEFAULT_CLOSURE_GLOBAL_ANSWER_ROOT = (
    DEFAULT_OUTPUT_ROOT / "s0-plus-artifact-global"
)
DEFAULT_FACT_GATE_ANSWER_ROOT = (
    DEFAULT_OUTPUT_ROOT / "s0-plus-routed-em-fact-gate-v1"
)
DEFAULT_FACT_GATE_EM_RUN = DEFAULT_CAMPAIGN_ROOT / "s0-plus-em-facts-v1/run.json"
DEFAULT_FIXED_S1_BASELINE_ANSWERS = Path(
    "eval_results/longmemeval-1m-fixed-s1-validation-20260826"
    "/final-answers.json"
)
DEFAULT_FIXED_S1_BASELINE_ANSWERS_SHA256 = (
    "d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd"
)
CLOSURE_REPRESENTATIVE_ARM = "S0_PLUS_REPRESENTATIVE_BRIDGE"
CLOSURE_GLOBAL_ARM = "S0_PLUS_ARTIFACT_GLOBAL"
CLOSURE_ARM_LABELS = (CLOSURE_REPRESENTATIVE_ARM, CLOSURE_GLOBAL_ARM)
DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
V3_DIAGNOSTIC_GATE_ORDINALS = (5, 16, 29, 34, 50, 52, 65, 79, 83, 97)
V3_DIAGNOSTIC_RESCUE_ORDINALS = (29, 34, 50)
V3_DIAGNOSTIC_REGRESSION_ORDINALS = (5, 16, 52, 65, 79, 83, 97)
V4_DIAGNOSTIC_GATE_ORDINALS = V3_DIAGNOSTIC_GATE_ORDINALS
V4_DIAGNOSTIC_RESCUE_ORDINALS = V3_DIAGNOSTIC_RESCUE_ORDINALS
V4_DIAGNOSTIC_REGRESSION_ORDINALS = V3_DIAGNOSTIC_REGRESSION_ORDINALS


def _closure_answer_output_root(
    arm_label: str,
    explicit_root: Path | None,
) -> Path:
    if explicit_root is not None:
        return explicit_root
    if arm_label == CLOSURE_REPRESENTATIVE_ARM:
        return DEFAULT_CLOSURE_REPRESENTATIVE_ANSWER_ROOT
    if arm_label == CLOSURE_GLOBAL_ARM:
        return DEFAULT_CLOSURE_GLOBAL_ANSWER_ROOT
    raise ValueError(f"unknown independent closure arm: {arm_label!r}")


def _closure_answer_request(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    root = _closure_answer_output_root(
        str(args.arm),
        None if args.output_root is None else Path(args.output_root),
    )
    return root, {
        "arm_label": str(args.arm),
        "retrieval_path": Path(args.retrieval),
        "generation_path": Path(args.generation),
        "expected_generation_sha256": str(args.expected_generation_sha256),
        "eligibility_manifest_path": Path(args.eligibility_manifest),
        "expected_eligibility_manifest_sha256": str(
            args.expected_eligibility_manifest_sha256
        ),
        "parent_root": Path(args.parent_root),
        "expected_parent_answer_run_sha256": str(
            args.expected_parent_answer_run_sha256
        ),
        "output_root": root,
        "max_concurrency": int(args.max_concurrency),
    }


def _fact_gate_answer_request(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    root = Path(args.output_root)
    return root, {
        "retrieval_path": Path(args.retrieval),
        "baseline_answers_path": Path(args.baseline_answers),
        "expected_baseline_answers_sha256": str(
            args.expected_baseline_answers_sha256
        ),
        "em_run_path": Path(args.em_run),
        "expected_em_run_sha256": str(args.expected_em_run_sha256),
        "parent_root": Path(args.parent_root),
        "expected_parent_answer_run_sha256": str(
            args.expected_parent_answer_run_sha256
        ),
        "output_root": root,
        "max_concurrency": int(args.max_concurrency),
    }


def _fact_gate_judge_request(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    """Load the replayed fact-gate answer plane without provider access."""

    from tools.matched_eval.fact_gate_live import (
        ANSWER_REPLAY_NAME,
        RUNTIME_LEDGER_NAME,
        RUNTIME_LEDGER_REPLAY_NAME,
        replay_fact_gate_answers,
    )

    root, answer_request = _fact_gate_answer_request(args)
    expected_answer_run_sha256 = str(args.expected_answer_run_sha256)
    answer_replay = read_sealed_json(root / ANSWER_REPLAY_NAME)
    if answer_replay.sha256 != expected_answer_run_sha256:
        raise RuntimeError("fact-gate answer replay SHA-256 changed")
    runtime_ledger = read_sealed_json(root / RUNTIME_LEDGER_NAME)
    runtime_ledger_replay = read_sealed_json(root / RUNTIME_LEDGER_REPLAY_NAME)
    if (
        runtime_ledger.sha256 != runtime_ledger_replay.sha256
        or runtime_ledger.payload != runtime_ledger_replay.payload
    ):
        raise RuntimeError("fact-gate runtime ledger and replay differ")
    answer_plane = replay_fact_gate_answers(
        **answer_request,
        expected_run_sha256=expected_answer_run_sha256,
    )
    return root, {
        "answer_plane": answer_plane,
        "dataset_path": Path(args.dataset),
        "split_path": Path(args.split),
        "parent_judge_root": Path(args.parent_judge_root),
        "expected_parent_judge_sha256": str(
            args.expected_parent_judge_sha256
        ),
        "expected_parent_score_ledger_sha256": str(
            args.expected_parent_score_ledger_sha256
        ),
        "output_root": root,
    }


def _closure_judge_request(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    """Load the already replayed closure answer plane without provider access."""

    from tools.matched_eval.closure_live import (
        ANSWER_REPLAY_NAME,
        RUNTIME_LEDGER_NAME,
        RUNTIME_LEDGER_REPLAY_NAME,
        replay_closure_answers,
    )

    root, answer_request = _closure_answer_request(args)
    expected_answer_run_sha256 = str(args.expected_answer_run_sha256)

    # A judge command must consume an answer plane that was explicitly replayed
    # first.  Requiring both replay artifacts here keeps the subsequent replay
    # call byte-idempotent, so a bad judge authorization cannot create answer
    # artifacts before the judge's own fail-closed budget check.
    answer_replay = read_sealed_json(root / ANSWER_REPLAY_NAME)
    if answer_replay.sha256 != expected_answer_run_sha256:
        raise RuntimeError("closure answer replay SHA-256 changed")
    runtime_ledger = read_sealed_json(root / RUNTIME_LEDGER_NAME)
    runtime_ledger_replay = read_sealed_json(root / RUNTIME_LEDGER_REPLAY_NAME)
    if (
        runtime_ledger.sha256 != runtime_ledger_replay.sha256
        or runtime_ledger.payload != runtime_ledger_replay.payload
    ):
        raise RuntimeError("closure runtime ledger and replay differ")

    answer_plane = replay_closure_answers(
        **answer_request,
        expected_run_sha256=expected_answer_run_sha256,
    )
    return root, {
        "answer_plane": answer_plane,
        "dataset_path": Path(args.dataset),
        "split_path": Path(args.split),
        "parent_judge_root": Path(args.parent_judge_root),
        "expected_parent_judge_sha256": str(
            args.expected_parent_judge_sha256
        ),
        "expected_parent_score_ledger_sha256": str(
            args.expected_parent_score_ledger_sha256
        ),
        "output_root": root,
    }


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} projection must be a mapping")
    return dict(value)


def _parse_ordinals(value: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in value.split(","))
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "ordinals must be a comma-separated list of integers"
        )
    try:
        ordinals = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ordinals must be a comma-separated list of integers"
        ) from exc
    if any(ordinal < 0 for ordinal in ordinals):
        raise argparse.ArgumentTypeError("ordinals must be non-negative")
    if ordinals != tuple(sorted(set(ordinals))):
        raise argparse.ArgumentTypeError("ordinals must be sorted and unique")
    return ordinals


def _seal_v3_diagnostic_gate(
    *,
    output_root: Path,
    selected_ordinals: tuple[int, ...] | None,
    preflight: Any,
) -> Any | None:
    if selected_ordinals != V3_DIAGNOSTIC_GATE_ORDINALS:
        return None
    if output_root.resolve() == DEFAULT_S0_V3_ROOT.resolve():
        raise RuntimeError(
            "the flip diagnostic requires a dedicated --output-root"
        )
    full_preflight = read_sealed_json(
        DEFAULT_S0_V3_ROOT / "s0-v3-preflight.json"
    )
    full_rows = full_preflight.payload.get("ordered_rows")
    diagnostic_rows = preflight.payload.get("ordered_rows")
    if (
        type(full_rows) is not list
        or len(full_rows) != 100
        or type(diagnostic_rows) is not list
        or len(diagnostic_rows) != len(V3_DIAGNOSTIC_GATE_ORDINALS)
        or diagnostic_rows
        != [full_rows[ordinal] for ordinal in V3_DIAGNOSTIC_GATE_ORDINALS]
    ):
        raise RuntimeError(
            "diagnostic prompts are not an exact view of the sealed full population"
        )
    from tools.matched_eval.judging import DEFAULT_SOL_CALLER_MODEL
    from tools.matched_eval.live import DEFAULT_TERRA_CALLER_MODEL
    from tools.matched_eval.renderer import V3_RAW_SYSTEM_POLICY_SHA256

    gate = {
        "format": "memory-condense-matched-s0-v3-diagnostic-gate-v1",
        "baseline_bindings": {
            "legacy_s0_correct": 57,
            "legacy_s0_judge_sha256": (
                "1c9ea03121478edd053c666bfffb8eaf1db508f001df76367ce14adc8f5022cb"
            ),
            "matched_v2_answer_run_sha256": (
                "1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a"
            ),
            "matched_v2_correct": 53,
            "matched_v2_judge_sha256": (
                "05fec9a7f284bb4e95d286f44e7378a8bbc1737a03e7c2ed60aefd50e6ddc689"
            ),
            "matched_v2_score_ledger_sha256": (
                "3422ce2825bdcdc347c8307bd3fed5a46de3dff6d33510c8bc3a3ba1c31c56e1"
            ),
        },
        "diagnostic_preflight_sha256": preflight.sha256,
        "full_100_promotion_gate": {
            "minimum_correct": 57,
            "minimum_paired_net_improvement_vs_v2": 4,
            "question_count": 100,
        },
        "full_preflight_sha256": full_preflight.sha256,
        "gold_loaded": False,
        "matched_population_id": preflight.payload["matched_population_id"],
        "minimum_proceed_gate": {
            "minimum_correct": 7,
            "minimum_regressions_recovered": 4,
            "minimum_rescues_retained": 3,
            "question_count": 10,
            "retain_all_rescues": True,
        },
        "regression_ordinals": list(V3_DIAGNOSTIC_REGRESSION_ORDINALS),
        "rescue_ordinals": list(V3_DIAGNOSTIC_RESCUE_ORDINALS),
        "selected_ordinals": list(V3_DIAGNOSTIC_GATE_ORDINALS),
        "selection_is_posthoc_outcome_conditioned": True,
        "strong_recovery_goal": {
            "minimum_correct": 10,
            "recover_all_regressions": True,
            "retain_all_rescues": True,
        },
        "treatment_bindings": {
            "full_prompt_population_sha256": full_preflight.payload[
                "prompt_population_sha256"
            ],
            "policy_id": full_preflight.payload["snapshot"]["policy_id"],
            "raw_system_policy_sha256": V3_RAW_SYSTEM_POLICY_SHA256,
            "renderer_id": full_preflight.payload["renderer_id"],
            "sol_judge_model": DEFAULT_SOL_CALLER_MODEL,
            "terra_answer_model": DEFAULT_TERRA_CALLER_MODEL,
        },
    }
    artifact, _created = publish_sealed_json(
        output_root / "diagnostic-gate.json",
        gate,
    )
    return artifact


def _require_v4_diagnostic_root(
    *,
    output_root: Path,
    selected_ordinals: tuple[int, ...] | None,
) -> None:
    if selected_ordinals is None:
        return
    if output_root.resolve() == DEFAULT_S0_V4_ROOT.resolve():
        raise RuntimeError("a v4 subset requires a dedicated --output-root")
    if (
        selected_ordinals == V4_DIAGNOSTIC_GATE_ORDINALS
        and output_root.resolve() != DEFAULT_S0_V4_DIAGNOSTIC_ROOT.resolve()
    ):
        raise RuntimeError(
            "the v4 flip diagnostic requires --output-root "
            f"{DEFAULT_S0_V4_DIAGNOSTIC_ROOT}"
        )


def _seal_v4_diagnostic_gate(
    *,
    output_root: Path,
    selected_ordinals: tuple[int, ...] | None,
    preflight: Any,
) -> Any | None:
    if selected_ordinals != V4_DIAGNOSTIC_GATE_ORDINALS:
        return None
    _require_v4_diagnostic_root(
        output_root=output_root,
        selected_ordinals=selected_ordinals,
    )

    from tools.matched_eval.judging import DEFAULT_SOL_CALLER_MODEL
    from tools.matched_eval.live import (
        DEFAULT_TERRA_CALLER_MODEL,
        V4_PREFLIGHT_NAME,
    )
    from tools.matched_eval.renderer import (
        V4_RAW_SYSTEM_POLICY_SHA256,
        V4_RENDERER_ID,
    )

    full_preflight = read_sealed_json(DEFAULT_S0_V4_ROOT / V4_PREFLIGHT_NAME)
    full_rows = full_preflight.payload.get("ordered_rows")
    diagnostic_rows = preflight.payload.get("ordered_rows")
    if (
        type(full_rows) is not list
        or len(full_rows) != 100
        or type(diagnostic_rows) is not list
        or len(diagnostic_rows) != len(V4_DIAGNOSTIC_GATE_ORDINALS)
        or diagnostic_rows
        != [full_rows[ordinal] for ordinal in V4_DIAGNOSTIC_GATE_ORDINALS]
    ):
        raise RuntimeError(
            "v4 diagnostic prompts are not an exact view of the sealed full "
            "population"
        )
    if (
        full_preflight.payload.get("renderer_id") != V4_RENDERER_ID
        or preflight.payload.get("renderer_id") != V4_RENDERER_ID
    ):
        raise RuntimeError("v4 diagnostic preflights changed renderer identity")

    gate = {
        "format": "memory-condense-matched-s0-v4-diagnostic-gate-v1",
        "baseline_bindings": {
            "legacy_s0_correct": 57,
            "legacy_s0_judge_sha256": (
                "1c9ea03121478edd053c666bfffb8eaf1db508f001df76367ce14adc8f5022cb"
            ),
            "legacy_s0_run_sha256": (
                "a713328485ebef452a0dd30626a7ffc20126999162723cb543da4f94a87b8e68"
            ),
            "matched_v2_answer_run_sha256": (
                "1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a"
            ),
            "matched_v2_correct": 53,
            "matched_v2_judge_sha256": (
                "05fec9a7f284bb4e95d286f44e7378a8bbc1737a03e7c2ed60aefd50e6ddc689"
            ),
            "matched_v2_score_ledger_sha256": (
                "3422ce2825bdcdc347c8307bd3fed5a46de3dff6d33510c8bc3a3ba1c31c56e1"
            ),
        },
        "diagnostic_preflight_sha256": preflight.sha256,
        "full_100_promotion_gate": {
            "minimum_correct": 57,
            "minimum_paired_net_improvement_vs_v2": 4,
            "question_count": 100,
        },
        "full_preflight_sha256": full_preflight.sha256,
        "gold_loaded": False,
        "matched_population_id": preflight.payload["matched_population_id"],
        "minimum_proceed_gate": {
            "minimum_correct": 7,
            "minimum_regressions_recovered": 4,
            "minimum_rescues_retained": 3,
            "question_count": 10,
            "retain_all_rescues": True,
        },
        "regression_ordinals": list(V4_DIAGNOSTIC_REGRESSION_ORDINALS),
        "rescue_ordinals": list(V4_DIAGNOSTIC_RESCUE_ORDINALS),
        "selected_ordinals": list(V4_DIAGNOSTIC_GATE_ORDINALS),
        "selection_is_posthoc_outcome_conditioned": True,
        "strong_recovery_goal": {
            "minimum_correct": 10,
            "recover_all_regressions": True,
            "retain_all_rescues": True,
        },
        "treatment_bindings": {
            "diagnostic_prompt_population_sha256": preflight.payload[
                "prompt_population_sha256"
            ],
            "diagnostic_rows_are_exact_full_view": True,
            "full_matched_population_id": full_preflight.payload[
                "matched_population_id"
            ],
            "full_prompt_population_sha256": full_preflight.payload[
                "prompt_population_sha256"
            ],
            "policy_id": full_preflight.payload["snapshot"]["policy_id"],
            "raw_system_policy_sha256": V4_RAW_SYSTEM_POLICY_SHA256,
            "renderer_id": full_preflight.payload["renderer_id"],
            "retrieval_sha256": full_preflight.payload["retrieval_sha256"],
            "sol_judge_model": DEFAULT_SOL_CALLER_MODEL,
            "terra_answer_model": DEFAULT_TERRA_CALLER_MODEL,
        },
    }
    artifact, _created = publish_sealed_json(
        output_root / "diagnostic-gate.json",
        gate,
    )
    return artifact


def _require_v4_diagnostic_gate(
    *,
    output_root: Path,
    selected_ordinals: tuple[int, ...] | None,
) -> None:
    _require_v4_diagnostic_root(
        output_root=output_root,
        selected_ordinals=selected_ordinals,
    )
    if selected_ordinals != V4_DIAGNOSTIC_GATE_ORDINALS:
        return
    from tools.matched_eval.live import V4_PREFLIGHT_NAME

    preflight = read_sealed_json(output_root / V4_PREFLIGHT_NAME)
    full_preflight = read_sealed_json(DEFAULT_S0_V4_ROOT / V4_PREFLIGHT_NAME)
    gate = read_sealed_json(output_root / "diagnostic-gate.json")
    if (
        gate.payload.get("format")
        != "memory-condense-matched-s0-v4-diagnostic-gate-v1"
        or gate.payload.get("diagnostic_preflight_sha256") != preflight.sha256
        or gate.payload.get("full_preflight_sha256") != full_preflight.sha256
        or gate.payload.get("selected_ordinals")
        != list(V4_DIAGNOSTIC_GATE_ORDINALS)
    ):
        raise RuntimeError("v4 diagnostic gate is not bound to this preflight")


def _legacy_score_summary(score: Mapping[str, Any]) -> dict[str, int]:
    arms = score.get("arms")
    if not isinstance(arms, list):
        raise RuntimeError("legacy score projection is missing its arms")
    result: dict[str, int] = {}
    for raw in arms:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("aggregate"), Mapping):
            raise RuntimeError("legacy score arm is malformed")
        label = raw.get("arm_label")
        candidate = raw["aggregate"].get("candidate_correct")
        if not isinstance(label, str) or type(candidate) is not int:
            raise RuntimeError("legacy score arm is missing its reproduced score")
        result[label] = candidate
    return result


def _common_legacy_ledgers(migration: object) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = getattr(migration, "arms", None)
    if not isinstance(arms, tuple) or not arms:
        raise RuntimeError("legacy migration is missing its typed arms")
    population_sha = arms[0].population_identity_sha256
    retrieval_sha = arms[0].retrieval_sha256
    snapshot_id = identity_sha256(
        {
            "format": "memory-condense-legacy-evaluation-snapshot-import-v2",
            "population_identity_sha256": population_sha,
            "retrieval_sha256": retrieval_sha,
        }
    )
    runtime_rows: list[RuntimeLedgerEntry] = []
    paired_scores: list[tuple[object, object]] = []
    runtime_source_artifacts: list[dict[str, str]] = []
    score_source_artifacts: list[dict[str, str]] = []
    for arm in arms:
        runtime_source_artifacts.extend(
            (
                {"role": f"{arm.spec.arm_label}:run", "sha256": arm.run_artifact.sha256},
                {
                    "role": f"{arm.spec.arm_label}:run_replay",
                    "sha256": arm.run_replay_artifact.sha256,
                },
            )
        )
        score_source_artifacts.extend(
            (
                {
                    "role": f"{arm.spec.arm_label}:judge",
                    "sha256": arm.judge_artifact.sha256,
                },
                {
                    "role": f"{arm.spec.arm_label}:judge_replay",
                    "sha256": arm.judge_replay_artifact.sha256,
                },
            )
        )
        for answer, score in zip(
            arm.runtime_observations, arm.score_observations, strict=True
        ):
            entry = RuntimeLedgerEntry(
                event_type="answer_observation",
                ordinal=answer.ordinal,
                question_id=answer.question_id,
                question_sha256=answer.question_sha256,
                arm_label=arm.spec.arm_label,
                parent_arm_label=arm.spec.parent_arm_label,
                stage_id=arm.spec.arm_label,
                parent_stage_id=arm.spec.parent_arm_label,
                mechanism_id=f"legacy_import/{arm.spec.delta_kind}",
                delta_kind=arm.spec.delta_kind,
                renderer_id=arm.spec.renderer_identity,
                legacy_renderer=True,
                disposition=StageDisposition.NO_OP,
                provider_calls=0,
                historical_provider_calls=(
                    (1 if answer.call_key_sha256 else 0)
                    + (1 if arm.spec.arm_label == "S0_PLUS_EM_FACTS" else 0)
                ),
                prompt_messages_sha256=answer.prompt_messages_sha256,
                prediction=answer.prediction_text,
                prediction_sha256=answer.prediction_sha256,
                changed_from_parent=answer.changed_from_parent,
                source_row_sha256=answer.source_row_sha256,
                reason="sealed_legacy_answer_observation",
            )
            runtime_rows.append(entry)
            paired_scores.append((entry, score))
    runtime = build_runtime_ledger(
        snapshot_id=snapshot_id,
        plan_id="legacy_s0_em_cav_import_v2",
        entries=runtime_rows,
        source_artifacts=runtime_source_artifacts,
        historical_shared_local_model_calls=4,
    )
    score_rows = tuple(
        ScoreLedgerEntry(
            runtime_row_id=entry.row_id,
            correct=score.correct,
            baseline_correct=score.baseline_correct,
            changed_from_baseline=score.changed_from_baseline,
            rescued=score.rescued,
            regressed=score.regressed,
            question_only_demand_class=score.question_only_demand_class,
            evidence_topology_class=score.evidence_topology_class,
            judge_row_sha256=score.judge_row_sha256,
            judge_verdict_sha256=score.judge_verdict_sha256,
            baseline_judge_row_sha256=score.baseline_judge_row_sha256,
            historical_provider_calls=(1 if score.judge_verdict_sha256 else 0),
        )
        for entry, score in paired_scores
    )
    scores = build_score_ledger(
        runtime_ledger=runtime,
        entries=score_rows,
        source_artifacts=score_source_artifacts,
    )
    return runtime, scores


def migrate_legacy(*, campaign_root: Path, output_root: Path) -> dict[str, Any]:
    """Validate and normalize S0/EM/CAV without provider access."""

    from tools.matched_eval.legacy import load_legacy_campaign

    migration = load_legacy_campaign(campaign_root)
    legacy_runtime = _mapping(migration.runtime_projection(), "runtime")
    legacy_score = _mapping(migration.score_projection(), "score")
    manifest = _mapping(migration.manifest_projection(), "manifest")
    score_summary = _legacy_score_summary(legacy_score)
    runtime, score = _common_legacy_ledgers(migration)
    assert_gold_blind(legacy_runtime, path="legacy_runtime_projection")
    assert_gold_blind(runtime, path="common_runtime_ledger")
    if runtime.get("provider_calls") not in (0, None) or runtime.get(
        "total_provider_calls"
    ) not in (0, None):
        raise RuntimeError("legacy migration attempted to report provider calls")

    runtime_artifact, runtime_created = publish_sealed_json(
        output_root / "legacy-runtime-ledger.json", runtime
    )
    score_artifact, score_created = publish_sealed_json(
        output_root / "legacy-score-ledger.json", score
    )
    migration_output = {
        "format": "memory-condense-matched-legacy-migration-output-v2",
        "historical_renderer_comparability": "legacy_renderer_only",
        "legacy_behavior_projection_identities": {
            "runtime": identity_sha256(legacy_runtime),
            "score": identity_sha256(legacy_score),
        },
        "manifest": manifest,
        "new_provider_calls": 0,
        "outputs": {
            "runtime_ledger_sha256": runtime_artifact.sha256,
            "score_ledger_sha256": score_artifact.sha256,
        },
        "reproduced_scores": score_summary,
    }
    migration_artifact, migration_created = publish_sealed_json(
        output_root / "legacy-migration.json", migration_output
    )
    return {
        "migration_created": migration_created,
        "migration_sha256": migration_artifact.sha256,
        "output_root": str(output_root),
        "runtime_created": runtime_created,
        "runtime_sha256": runtime_artifact.sha256,
        "score_created": score_created,
        "score_sha256": score_artifact.sha256,
        "scores": score_summary,
    }


def preflight_s0_v2(*, retrieval_path: Path, output_root: Path) -> dict[str, Any]:
    """Build and seal the common-renderer S0 population with zero calls."""

    from tools.matched_eval.population import (
        EXPECTED_RETRIEVAL_SHA256,
        load_s0_population,
    )

    expected_sha256 = (
        EXPECTED_RETRIEVAL_SHA256
        if retrieval_path.resolve() == DEFAULT_RETRIEVAL.resolve()
        else None
    )
    population = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_sha256,
    )
    preflight = _mapping(population.preflight_projection(), "S0 v2 preflight")
    assert_gold_blind(preflight, path="s0_v2_preflight")
    if preflight.get("provider_calls") != 0:
        raise RuntimeError("S0 v2 preflight must report zero provider calls")
    artifact, created = publish_sealed_json(output_root / "s0-v2-preflight.json", preflight)
    return {
        "created": created,
        "logical_prompt_count": preflight.get("logical_prompt_count"),
        "hard_prompt_token_cap": population.max_prompt_tokens,
        "observed_max_prompt_token_proxy": preflight.get(
            "observed_max_prompt_token_proxy"
        ),
        "output": str(artifact.path),
        "preflight_identity_sha256": population.preflight_sha256,
        "prompt_population_sha256": (
            population.prompt_population.prompt_population_sha256
        ),
        "provider_calls": 0,
        "sha256": artifact.sha256,
        "unique_prompt_count": preflight.get("unique_prompt_count"),
    }


def run_s0_v4_pipeline(
    *,
    retrieval_path: Path,
    dataset_path: Path,
    split_path: Path,
    output_root: Path,
    enable_answer_provider: bool,
    authorized_answer_provider_calls: int,
    enable_judge_provider: bool,
    authorized_judge_provider_calls: int,
    api_key_env: str,
    max_concurrency: int,
    selected_ordinals: Sequence[int] | None,
    expected_retrieval_sha256: str | None,
    expected_question_count: int,
) -> dict[str, Any]:
    """Run the complete v4 answer/judge lineage with one in-memory snapshot."""

    from tools.matched_eval import judging, live
    from tools.matched_eval.renderer import V4_RENDERER_ID

    root = Path(output_root)
    ordinals = None if selected_ordinals is None else tuple(selected_ordinals)
    _require_v4_diagnostic_root(
        output_root=root,
        selected_ordinals=ordinals,
    )
    profile = live.execution_profile(V4_RENDERER_ID)
    population = live._load_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=profile.renderer_id,
        selected_ordinals=ordinals,
    )
    answer_calls = population.prompt_population.unique_prompt_count
    judge_calls = population.question_count
    if not enable_answer_provider:
        raise ValueError("v4 pipeline requires answer-provider enablement")
    if (
        type(authorized_answer_provider_calls) is not int
        or authorized_answer_provider_calls != answer_calls
    ):
        raise ValueError(
            "authorized answer-provider calls must exactly equal "
            f"{answer_calls}"
        )
    if not enable_judge_provider:
        raise ValueError("v4 pipeline requires judge-provider enablement")
    if (
        type(authorized_judge_provider_calls) is not int
        or authorized_judge_provider_calls != judge_calls
    ):
        raise ValueError(
            "authorized judge-provider calls must exactly equal "
            f"{judge_calls}"
        )

    answer_preflight = live._preflight_s0_v2_answers_for_population(
        population=population,
        output_root=root,
        profile=profile,
    )
    diagnostic_gate = _seal_v4_diagnostic_gate(
        output_root=root,
        selected_ordinals=ordinals,
        preflight=answer_preflight,
    )
    if diagnostic_gate is not None:
        _require_v4_diagnostic_gate(
            output_root=root,
            selected_ordinals=ordinals,
        )
    answer = live._run_s0_v2_answers_for_population(
        population=population,
        output_root=root,
        enable_provider=enable_answer_provider,
        authorized_provider_calls=authorized_answer_provider_calls,
        api_key_env=api_key_env,
        max_concurrency=max_concurrency,
        profile=profile,
    )
    answer_plane = live._replay_s0_v2_answers_for_population(
        population=population,
        source=answer.answer_artifact,
        output_root=root,
        expected_run_sha256=answer.answer_artifact.sha256,
        max_concurrency=max_concurrency,
        profile=profile,
    )

    judge_profile = judging._judge_profile(V4_RENDERER_ID)
    judge_plan = judging._build_plan_from_answer_plane(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        profile=judge_profile,
    )
    if judge_plan.required_calls != authorized_judge_provider_calls:
        raise RuntimeError("judge prompt population changed after authorization")
    judge_preflight, _created = publish_sealed_json(
        root / judging.JUDGE_PREFLIGHT_NAME,
        judging._preflight_artifact(judge_plan),
    )
    judged = judging._run_prebuilt_judge_plan(
        judge_plan,
        output_root=root,
        enable_provider=enable_judge_provider,
        authorized_provider_calls=authorized_judge_provider_calls,
        api_key_env=api_key_env,
        max_concurrency=max_concurrency,
    )
    judge_replay = judging._replay_prebuilt_judge_plan(
        judge_plan,
        expected_judge_sha256=judged.judge_artifact.sha256,
        output_root=root,
        max_concurrency=max_concurrency,
    )

    final_retrieval = read_sealed_json(retrieval_path)
    if final_retrieval.sha256 != population.retrieval_sha256:
        raise RuntimeError("sealed retrieval changed during the v4 pipeline")
    return {
        "answer_checkpoint_hits": answer.checkpoint_hits,
        "answer_physical_provider_calls": answer.physical_provider_calls,
        "answer_preflight_sha256": answer_preflight.sha256,
        "answer_replay_sha256": answer_plane.replay_sha256,
        "answer_run_sha256": answer_plane.run_sha256,
        "answer_runtime_ledger_sha256": answer_plane.runtime_ledger_sha256,
        "correct": judge_replay.correct,
        "format": "memory-condense-matched-s0-v4-single-process-pipeline-v1",
        "judge_checkpoint_hits": judged.checkpoint_hits,
        "judge_physical_provider_calls": judged.physical_provider_calls,
        "judge_preflight_sha256": judge_preflight.sha256,
        "judge_replay_sha256": judge_replay.judge_artifact.sha256,
        "judge_run_sha256": judged.judge_artifact.sha256,
        "matched_population_id": population.population_id,
        "question_count": population.question_count,
        "retrieval_reverified_after_judge_replay": True,
        "retrieval_sha256": final_retrieval.sha256,
        "score_ledger_replay_sha256": (
            judge_replay.score_ledger_artifact.sha256
        ),
        "score_ledger_sha256": judged.score_ledger_artifact.sha256,
        "total_physical_provider_calls": (
            answer.physical_provider_calls + judged.physical_provider_calls
        ),
    }


def inspect_outputs(*, output_root: Path) -> dict[str, Any]:
    migration = read_sealed_json(output_root / "legacy-migration.json")
    payload = migration.payload
    return {
        "migration_sha256": migration.sha256,
        "new_provider_calls": payload.get("new_provider_calls"),
        "outputs": payload.get("outputs"),
        "scores": payload.get("reproduced_scores"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser(
        "migrate-legacy", help="normalize sealed S0/EM/CAV observations"
    )
    migrate.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    migrate.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "legacy-import"
    )

    s0 = subparsers.add_parser(
        "s0-v2-preflight", help="seal the fresh matched-renderer S0 prompts"
    )
    s0.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    s0.add_argument(
        "--output-root", type=Path, default=DEFAULT_S0_V2_ROOT
    )

    answer = subparsers.add_parser(
        "s0-v2-answer", help="execute the authorized Terra S0-v2 answer batch"
    )
    answer.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    answer.add_argument("--output-root", type=Path, default=DEFAULT_S0_V2_ROOT)
    answer.add_argument("--api-key-env", default="LITELLM_KEY")
    answer.add_argument("--max-concurrency", type=int, default=4)
    answer.add_argument("--enable-provider", action="store_true")
    answer.add_argument("--authorized-provider-calls", type=int, default=0)

    answer_replay = subparsers.add_parser(
        "s0-v2-answer-replay", help="verify Terra journals with zero calls"
    )
    answer_replay.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    answer_replay.add_argument(
        "--output-root", type=Path, default=DEFAULT_S0_V2_ROOT
    )
    answer_replay.add_argument("--expected-run-sha256", required=True)
    answer_replay.add_argument("--max-concurrency", type=int, default=4)

    def add_ordinals(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--ordinals",
            type=_parse_ordinals,
            default=None,
            metavar="N[,N...]",
            help="run a sorted unique subset of zero-based question ordinals",
        )

    s0_v3 = subparsers.add_parser(
        "s0-v3-preflight", help="seal the compact question-last S0-v3 prompts"
    )
    s0_v3.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    s0_v3.add_argument("--output-root", type=Path, default=DEFAULT_S0_V3_ROOT)
    add_ordinals(s0_v3)

    answer_v3 = subparsers.add_parser(
        "s0-v3-answer", help="execute the authorized Terra S0-v3 answer batch"
    )
    answer_v3.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    answer_v3.add_argument("--output-root", type=Path, default=DEFAULT_S0_V3_ROOT)
    answer_v3.add_argument("--api-key-env", default="LITELLM_KEY")
    answer_v3.add_argument("--max-concurrency", type=int, default=4)
    answer_v3.add_argument("--enable-provider", action="store_true")
    answer_v3.add_argument("--authorized-provider-calls", type=int, default=0)
    add_ordinals(answer_v3)

    answer_replay_v3 = subparsers.add_parser(
        "s0-v3-answer-replay", help="verify S0-v3 Terra journals with zero calls"
    )
    answer_replay_v3.add_argument(
        "--retrieval", type=Path, default=DEFAULT_RETRIEVAL
    )
    answer_replay_v3.add_argument(
        "--output-root", type=Path, default=DEFAULT_S0_V3_ROOT
    )
    answer_replay_v3.add_argument("--expected-run-sha256", required=True)
    answer_replay_v3.add_argument("--max-concurrency", type=int, default=4)
    add_ordinals(answer_replay_v3)

    s0_v4 = subparsers.add_parser(
        "s0-v4-preflight", help="seal the legacy-shaped typed S0-v4 prompts"
    )
    s0_v4.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    s0_v4.add_argument("--output-root", type=Path, default=DEFAULT_S0_V4_ROOT)
    add_ordinals(s0_v4)

    answer_v4 = subparsers.add_parser(
        "s0-v4-answer", help="execute the authorized Terra S0-v4 answer batch"
    )
    answer_v4.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    answer_v4.add_argument("--output-root", type=Path, default=DEFAULT_S0_V4_ROOT)
    answer_v4.add_argument("--api-key-env", default="LITELLM_KEY")
    answer_v4.add_argument("--max-concurrency", type=int, default=4)
    answer_v4.add_argument("--enable-provider", action="store_true")
    answer_v4.add_argument("--authorized-provider-calls", type=int, default=0)
    add_ordinals(answer_v4)

    answer_replay_v4 = subparsers.add_parser(
        "s0-v4-answer-replay", help="verify S0-v4 Terra journals with zero calls"
    )
    answer_replay_v4.add_argument(
        "--retrieval", type=Path, default=DEFAULT_RETRIEVAL
    )
    answer_replay_v4.add_argument(
        "--output-root", type=Path, default=DEFAULT_S0_V4_ROOT
    )
    answer_replay_v4.add_argument("--expected-run-sha256", required=True)
    answer_replay_v4.add_argument("--max-concurrency", type=int, default=4)
    add_ordinals(answer_replay_v4)

    pipeline_v4 = subparsers.add_parser(
        "s0-v4-pipeline",
        help=(
            "run and replay v4 Terra answers and Sol judgments in one "
            "snapshot-bound process"
        ),
    )
    pipeline_v4.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    pipeline_v4.add_argument("--dataset", type=Path, required=True)
    pipeline_v4.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    pipeline_v4.add_argument("--output-root", type=Path, default=DEFAULT_S0_V4_ROOT)
    pipeline_v4.add_argument("--api-key-env", default="LITELLM_KEY")
    pipeline_v4.add_argument("--max-concurrency", type=int, default=4)
    pipeline_v4.add_argument("--enable-answer-provider", action="store_true")
    pipeline_v4.add_argument("--authorized-answer-provider-calls", type=int, default=0)
    pipeline_v4.add_argument("--enable-judge-provider", action="store_true")
    pipeline_v4.add_argument("--authorized-judge-provider-calls", type=int, default=0)
    add_ordinals(pipeline_v4)

    def add_fact_gate_answer_sources(command: argparse.ArgumentParser) -> None:
        command.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
        command.add_argument(
            "--baseline-answers",
            type=Path,
            default=DEFAULT_FIXED_S1_BASELINE_ANSWERS,
        )
        command.add_argument(
            "--expected-baseline-answers-sha256",
            default=DEFAULT_FIXED_S1_BASELINE_ANSWERS_SHA256,
        )
        command.add_argument(
            "--em-run",
            type=Path,
            default=DEFAULT_FACT_GATE_EM_RUN,
        )
        command.add_argument("--expected-em-run-sha256", required=True)
        command.add_argument(
            "--parent-root",
            type=Path,
            default=DEFAULT_S0_V2_ROOT,
        )
        command.add_argument(
            "--expected-parent-answer-run-sha256",
            required=True,
        )
        command.add_argument(
            "--output-root",
            type=Path,
            default=DEFAULT_FACT_GATE_ANSWER_ROOT,
        )
        command.add_argument("--max-concurrency", type=int, default=4)

    fact_gate_preflight = subparsers.add_parser(
        "fact-gate-answer-preflight",
        help=(
            "verify sealed fixed-S1 EM facts and S0-v2 parents, then seal "
            "the routed Terra prompt population"
        ),
    )
    add_fact_gate_answer_sources(fact_gate_preflight)

    fact_gate_answer = subparsers.add_parser(
        "fact-gate-answer",
        help="execute Terra only for routed valid fixed-S1 fact packets",
    )
    add_fact_gate_answer_sources(fact_gate_answer)
    fact_gate_answer.add_argument("--api-key-env", default="LITELLM_KEY")
    fact_gate_answer.add_argument("--enable-provider", action="store_true")
    fact_gate_answer.add_argument(
        "--authorized-provider-calls",
        type=int,
        default=0,
    )

    fact_gate_replay = subparsers.add_parser(
        "fact-gate-answer-replay",
        help="reconstruct routed fact-gate predictions from sealed Terra journals",
    )
    add_fact_gate_answer_sources(fact_gate_replay)
    fact_gate_replay.add_argument("--expected-run-sha256", required=True)

    def add_fact_gate_judge_sources(command: argparse.ArgumentParser) -> None:
        add_fact_gate_answer_sources(command)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
        command.add_argument("--expected-answer-run-sha256", required=True)
        command.add_argument(
            "--parent-judge-root",
            type=Path,
            default=DEFAULT_S0_V2_ROOT,
        )
        command.add_argument("--expected-parent-judge-sha256", required=True)
        command.add_argument(
            "--expected-parent-score-ledger-sha256",
            required=True,
        )

    fact_gate_judge_preflight = subparsers.add_parser(
        "fact-gate-judge-preflight",
        help=(
            "verify replayed fact-gate predictions and the sealed S0 judge, "
            "then seal changed-only Sol prompts"
        ),
    )
    add_fact_gate_judge_sources(fact_gate_judge_preflight)

    fact_gate_judge = subparsers.add_parser(
        "fact-gate-judge",
        help="execute Sol only for fact-gate predictions changed from S0",
    )
    add_fact_gate_judge_sources(fact_gate_judge)
    fact_gate_judge.add_argument("--api-key-env", default="LITELLM_KEY")
    fact_gate_judge.add_argument("--enable-provider", action="store_true")
    fact_gate_judge.add_argument(
        "--authorized-provider-calls",
        type=int,
        default=0,
    )

    fact_gate_judge_replay = subparsers.add_parser(
        "fact-gate-judge-replay",
        help="verify changed-only Sol journals and the 100-row score ledger",
    )
    add_fact_gate_judge_sources(fact_gate_judge_replay)
    fact_gate_judge_replay.add_argument(
        "--expected-judge-sha256",
        required=True,
    )

    def add_closure_answer_sources(command: argparse.ArgumentParser) -> None:
        command.add_argument("--arm", choices=CLOSURE_ARM_LABELS, required=True)
        command.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
        command.add_argument(
            "--generation",
            type=Path,
            default=DEFAULT_CLOSURE_GENERATION,
        )
        command.add_argument(
            "--expected-generation-sha256",
            required=True,
        )
        command.add_argument(
            "--eligibility-manifest",
            type=Path,
            default=DEFAULT_CLOSURE_ELIGIBILITY,
        )
        command.add_argument(
            "--expected-eligibility-manifest-sha256",
            required=True,
        )
        command.add_argument(
            "--parent-root",
            type=Path,
            default=DEFAULT_S0_V2_ROOT,
        )
        command.add_argument(
            "--expected-parent-answer-run-sha256",
            required=True,
        )
        command.add_argument(
            "--output-root",
            type=Path,
            default=None,
            help="defaults to a distinct root selected by --arm",
        )
        command.add_argument("--max-concurrency", type=int, default=4)

    closure_preflight = subparsers.add_parser(
        "closure-answer-preflight",
        help=(
            "verify a sealed closure arm and parent predictions, then seal "
            "the changed-only Terra prompt population"
        ),
    )
    add_closure_answer_sources(closure_preflight)

    closure_answer = subparsers.add_parser(
        "closure-answer",
        help="execute Terra only for valid closure descendants",
    )
    add_closure_answer_sources(closure_answer)
    closure_answer.add_argument("--api-key-env", default="LITELLM_KEY")
    closure_answer.add_argument("--enable-provider", action="store_true")
    closure_answer.add_argument(
        "--authorized-provider-calls",
        type=int,
        default=0,
    )

    closure_replay = subparsers.add_parser(
        "closure-answer-replay",
        help="reconstruct closure predictions from sealed Terra journals",
    )
    add_closure_answer_sources(closure_replay)
    closure_replay.add_argument("--expected-run-sha256", required=True)

    def add_closure_judge_sources(command: argparse.ArgumentParser) -> None:
        add_closure_answer_sources(command)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
        command.add_argument(
            "--expected-answer-run-sha256",
            required=True,
            help="pinned closure answer run/replay SHA-256",
        )
        command.add_argument(
            "--parent-judge-root",
            type=Path,
            default=DEFAULT_S0_V2_ROOT,
        )
        command.add_argument(
            "--expected-parent-judge-sha256",
            required=True,
        )
        command.add_argument(
            "--expected-parent-score-ledger-sha256",
            required=True,
        )

    closure_judge_preflight = subparsers.add_parser(
        "closure-judge-preflight",
        help=(
            "verify replayed closure predictions and the sealed S0 judge, "
            "then seal changed-only Sol prompts"
        ),
    )
    add_closure_judge_sources(closure_judge_preflight)

    closure_judge = subparsers.add_parser(
        "closure-judge",
        help="execute Sol only for closure predictions changed from S0",
    )
    add_closure_judge_sources(closure_judge)
    closure_judge.add_argument("--api-key-env", default="LITELLM_KEY")
    closure_judge.add_argument("--enable-provider", action="store_true")
    closure_judge.add_argument(
        "--authorized-provider-calls",
        type=int,
        default=0,
    )

    closure_judge_replay = subparsers.add_parser(
        "closure-judge-replay",
        help="verify changed-only Sol journals and the closure score ledger",
    )
    add_closure_judge_sources(closure_judge_replay)
    closure_judge_replay.add_argument(
        "--expected-judge-sha256",
        required=True,
    )

    def add_judge_sources(
        command: argparse.ArgumentParser,
        *,
        output_root: Path = DEFAULT_S0_V2_ROOT,
        with_ordinals: bool = False,
    ) -> None:
        command.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
        command.add_argument("--output-root", type=Path, default=output_root)
        command.add_argument("--expected-answer-run-sha256", required=True)
        command.add_argument("--max-concurrency", type=int, default=4)
        if with_ordinals:
            add_ordinals(command)

    judge_preflight = subparsers.add_parser(
        "s0-v2-judge-preflight",
        help="verify sealed predictions, then seal the Sol prompt population",
    )
    add_judge_sources(judge_preflight)

    judge = subparsers.add_parser(
        "s0-v2-judge", help="execute the authorized post-hoc Sol judge batch"
    )
    add_judge_sources(judge)
    judge.add_argument("--api-key-env", default="LITELLM_KEY")
    judge.add_argument("--enable-provider", action="store_true")
    judge.add_argument("--authorized-provider-calls", type=int, default=0)

    judge_replay = subparsers.add_parser(
        "s0-v2-judge-replay", help="verify Sol journals and score ledger"
    )
    add_judge_sources(judge_replay)
    judge_replay.add_argument("--expected-judge-sha256", required=True)

    judge_preflight_v3 = subparsers.add_parser(
        "s0-v3-judge-preflight",
        help="verify sealed S0-v3 predictions, then seal the Sol prompts",
    )
    add_judge_sources(
        judge_preflight_v3,
        output_root=DEFAULT_S0_V3_ROOT,
        with_ordinals=True,
    )

    judge_v3 = subparsers.add_parser(
        "s0-v3-judge", help="execute the authorized post-hoc S0-v3 Sol judge batch"
    )
    add_judge_sources(
        judge_v3,
        output_root=DEFAULT_S0_V3_ROOT,
        with_ordinals=True,
    )
    judge_v3.add_argument("--api-key-env", default="LITELLM_KEY")
    judge_v3.add_argument("--enable-provider", action="store_true")
    judge_v3.add_argument("--authorized-provider-calls", type=int, default=0)

    judge_replay_v3 = subparsers.add_parser(
        "s0-v3-judge-replay", help="verify S0-v3 Sol journals and score ledger"
    )
    add_judge_sources(
        judge_replay_v3,
        output_root=DEFAULT_S0_V3_ROOT,
        with_ordinals=True,
    )
    judge_replay_v3.add_argument("--expected-judge-sha256", required=True)

    judge_preflight_v4 = subparsers.add_parser(
        "s0-v4-judge-preflight",
        help="verify sealed S0-v4 predictions, then seal the Sol prompts",
    )
    add_judge_sources(
        judge_preflight_v4,
        output_root=DEFAULT_S0_V4_ROOT,
        with_ordinals=True,
    )

    judge_v4 = subparsers.add_parser(
        "s0-v4-judge", help="execute the authorized post-hoc S0-v4 Sol judge batch"
    )
    add_judge_sources(
        judge_v4,
        output_root=DEFAULT_S0_V4_ROOT,
        with_ordinals=True,
    )
    judge_v4.add_argument("--api-key-env", default="LITELLM_KEY")
    judge_v4.add_argument("--enable-provider", action="store_true")
    judge_v4.add_argument("--authorized-provider-calls", type=int, default=0)

    judge_replay_v4 = subparsers.add_parser(
        "s0-v4-judge-replay", help="verify S0-v4 Sol journals and score ledger"
    )
    add_judge_sources(
        judge_replay_v4,
        output_root=DEFAULT_S0_V4_ROOT,
        with_ordinals=True,
    )
    judge_replay_v4.add_argument("--expected-judge-sha256", required=True)

    inspect = subparsers.add_parser(
        "inspect", help="verify and summarize a sealed legacy migration"
    )
    inspect.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "legacy-import"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "migrate-legacy":
        result = migrate_legacy(
            campaign_root=Path(args.campaign_root),
            output_root=Path(args.output_root),
        )
    elif args.command == "s0-v2-preflight":
        result = preflight_s0_v2(
            retrieval_path=Path(args.retrieval),
            output_root=Path(args.output_root),
        )
    elif args.command == "s0-v2-answer":
        from tools.matched_eval.live import run_s0_v2_answers

        run = run_s0_v2_answers(
            retrieval_path=Path(args.retrieval),
            output_root=Path(args.output_root),
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
        )
        result = {
            "answer_run_sha256": run.answer_artifact.sha256,
            "checkpoint_hits": run.checkpoint_hits,
            "physical_provider_calls": run.physical_provider_calls,
            "runtime_ledger_sha256": run.runtime_ledger_artifact.sha256,
        }
    elif args.command == "s0-v2-answer-replay":
        from tools.matched_eval.live import replay_s0_v2_answers

        replay = replay_s0_v2_answers(
            retrieval_path=Path(args.retrieval),
            output_root=Path(args.output_root),
            expected_run_sha256=str(args.expected_run_sha256),
            max_concurrency=int(args.max_concurrency),
        )
        result = {
            "answer_run_sha256": replay.run_sha256,
            "answer_replay_sha256": replay.replay_sha256,
            "physical_provider_calls": 0,
            "runtime_ledger_sha256": replay.runtime_ledger_sha256,
        }
    elif args.command == "s0-v3-preflight":
        from tools.matched_eval.live import preflight_s0_v3_answers

        preflight = preflight_s0_v3_answers(
            retrieval_path=Path(args.retrieval),
            output_root=Path(args.output_root),
            selected_ordinals=args.ordinals,
        )
        result = {
            "hard_prompt_token_cap": preflight.payload.get(
                "hard_prompt_token_cap"
            ),
            "logical_prompt_count": preflight.payload.get("logical_prompt_count"),
            "observed_max_prompt_token_proxy": preflight.payload.get(
                "observed_max_prompt_token_proxy"
            ),
            "output": str(preflight.path),
            "preflight_identity_sha256": identity_sha256(preflight.payload),
            "prompt_population_sha256": preflight.payload.get(
                "prompt_population_sha256"
            ),
            "provider_calls": 0,
            "sha256": preflight.sha256,
            "unique_prompt_count": preflight.payload.get("unique_prompt_count"),
        }
        diagnostic_gate = _seal_v3_diagnostic_gate(
            output_root=Path(args.output_root),
            selected_ordinals=args.ordinals,
            preflight=preflight,
        )
        if diagnostic_gate is not None:
            result["diagnostic_gate"] = str(diagnostic_gate.path)
            result["diagnostic_gate_sha256"] = diagnostic_gate.sha256
    elif args.command == "s0-v3-answer":
        from tools.matched_eval.live import run_s0_v3_answers

        run = run_s0_v3_answers(
            retrieval_path=Path(args.retrieval),
            output_root=Path(args.output_root),
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "answer_run_sha256": run.answer_artifact.sha256,
            "checkpoint_hits": run.checkpoint_hits,
            "physical_provider_calls": run.physical_provider_calls,
            "runtime_ledger_sha256": run.runtime_ledger_artifact.sha256,
        }
    elif args.command == "s0-v3-answer-replay":
        from tools.matched_eval.live import replay_s0_v3_answers

        replay = replay_s0_v3_answers(
            retrieval_path=Path(args.retrieval),
            output_root=Path(args.output_root),
            expected_run_sha256=str(args.expected_run_sha256),
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "answer_run_sha256": replay.run_sha256,
            "answer_replay_sha256": replay.replay_sha256,
            "physical_provider_calls": 0,
            "runtime_ledger_sha256": replay.runtime_ledger_sha256,
        }
    elif args.command == "s0-v4-preflight":
        from tools.matched_eval.live import preflight_s0_v4_answers

        root = Path(args.output_root)
        _require_v4_diagnostic_root(
            output_root=root,
            selected_ordinals=args.ordinals,
        )
        preflight = preflight_s0_v4_answers(
            retrieval_path=Path(args.retrieval),
            output_root=root,
            selected_ordinals=args.ordinals,
        )
        result = {
            "hard_prompt_token_cap": preflight.payload.get(
                "hard_prompt_token_cap"
            ),
            "logical_prompt_count": preflight.payload.get("logical_prompt_count"),
            "observed_max_prompt_token_proxy": preflight.payload.get(
                "observed_max_prompt_token_proxy"
            ),
            "output": str(preflight.path),
            "preflight_identity_sha256": identity_sha256(preflight.payload),
            "prompt_population_sha256": preflight.payload.get(
                "prompt_population_sha256"
            ),
            "provider_calls": 0,
            "sha256": preflight.sha256,
            "unique_prompt_count": preflight.payload.get("unique_prompt_count"),
        }
        diagnostic_gate = _seal_v4_diagnostic_gate(
            output_root=root,
            selected_ordinals=args.ordinals,
            preflight=preflight,
        )
        if diagnostic_gate is not None:
            result["diagnostic_gate"] = str(diagnostic_gate.path)
            result["diagnostic_gate_sha256"] = diagnostic_gate.sha256
    elif args.command == "s0-v4-answer":
        from tools.matched_eval.live import run_s0_v4_answers

        root = Path(args.output_root)
        _require_v4_diagnostic_gate(
            output_root=root,
            selected_ordinals=args.ordinals,
        )
        run = run_s0_v4_answers(
            retrieval_path=Path(args.retrieval),
            output_root=root,
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "answer_run_sha256": run.answer_artifact.sha256,
            "checkpoint_hits": run.checkpoint_hits,
            "physical_provider_calls": run.physical_provider_calls,
            "runtime_ledger_sha256": run.runtime_ledger_artifact.sha256,
        }
    elif args.command == "s0-v4-answer-replay":
        from tools.matched_eval.live import replay_s0_v4_answers

        root = Path(args.output_root)
        _require_v4_diagnostic_gate(
            output_root=root,
            selected_ordinals=args.ordinals,
        )
        replay = replay_s0_v4_answers(
            retrieval_path=Path(args.retrieval),
            output_root=root,
            expected_run_sha256=str(args.expected_run_sha256),
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "answer_run_sha256": replay.run_sha256,
            "answer_replay_sha256": replay.replay_sha256,
            "physical_provider_calls": 0,
            "runtime_ledger_sha256": replay.runtime_ledger_sha256,
        }
    elif args.command == "s0-v4-pipeline":
        from tools.matched_eval.population import (
            EXPECTED_QUESTION_COUNT,
            EXPECTED_RETRIEVAL_SHA256,
        )

        result = run_s0_v4_pipeline(
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=Path(args.output_root),
            enable_answer_provider=bool(args.enable_answer_provider),
            authorized_answer_provider_calls=int(
                args.authorized_answer_provider_calls
            ),
            enable_judge_provider=bool(args.enable_judge_provider),
            authorized_judge_provider_calls=int(
                args.authorized_judge_provider_calls
            ),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
            expected_retrieval_sha256=EXPECTED_RETRIEVAL_SHA256,
            expected_question_count=EXPECTED_QUESTION_COUNT,
        )
    elif args.command == "fact-gate-answer-preflight":
        from tools.matched_eval.fact_gate_live import preflight_fact_gate_answers

        root, request = _fact_gate_answer_request(args)
        preflight = preflight_fact_gate_answers(**request)
        result = {
            "answer_preflight_sha256": preflight.sha256,
            "arm_label": preflight.payload["arm_label"],
            "output_root": str(root),
            "physical_provider_calls": 0,
            "required_authorized_provider_calls": preflight.payload[
                "required_authorized_provider_calls"
            ],
        }
    elif args.command == "fact-gate-answer":
        from tools.matched_eval.fact_gate_live import run_fact_gate_answers

        root, request = _fact_gate_answer_request(args)
        run = run_fact_gate_answers(
            **request,
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
        )
        result = {
            "answer_run_sha256": run.answer_artifact.sha256,
            "arm_label": run.answer_artifact.payload["arm_label"],
            "checkpoint_hits": run.checkpoint_hits,
            "output_root": str(root),
            "physical_provider_calls": run.physical_provider_calls,
            "runtime_ledger_sha256": run.runtime_ledger_artifact.sha256,
        }
    elif args.command == "fact-gate-answer-replay":
        from tools.matched_eval.fact_gate_live import replay_fact_gate_answers

        root, request = _fact_gate_answer_request(args)
        replay = replay_fact_gate_answers(
            **request,
            expected_run_sha256=str(args.expected_run_sha256),
        )
        result = {
            "answer_replay_sha256": replay.replay_sha256,
            "answer_run_sha256": replay.run_sha256,
            "arm_label": replay.arm_label,
            "changed_from_parent_count": len(replay.changed_rows),
            "output_root": str(root),
            "physical_provider_calls": 0,
            "runtime_ledger_sha256": replay.runtime_ledger_sha256,
        }
    elif args.command == "fact-gate-judge-preflight":
        from tools.matched_eval.fact_gate_judging import (
            preflight_fact_gate_changed_only_judge,
        )

        root, request = _fact_gate_judge_request(args)
        preflight = preflight_fact_gate_changed_only_judge(**request)
        result = {
            "arm_label": preflight.payload["arm_label"],
            "judge_preflight_sha256": preflight.sha256,
            "output_root": str(root),
            "physical_provider_calls": 0,
            "required_authorized_provider_calls": preflight.payload[
                "required_authorized_provider_calls"
            ],
        }
    elif args.command == "fact-gate-judge":
        from tools.matched_eval.fact_gate_judging import (
            run_fact_gate_changed_only_judge,
        )

        root, request = _fact_gate_judge_request(args)
        judged = run_fact_gate_changed_only_judge(
            **request,
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
        )
        result = {
            "arm_label": judged.judge_artifact.payload["arm_label"],
            "checkpoint_hits": judged.checkpoint_hits,
            "correct": judged.correct,
            "judge_sha256": judged.judge_artifact.sha256,
            "output_root": str(root),
            "physical_provider_calls": judged.physical_provider_calls,
            "score_ledger_sha256": judged.score_ledger_artifact.sha256,
        }
    elif args.command == "fact-gate-judge-replay":
        from tools.matched_eval.fact_gate_judging import (
            replay_fact_gate_changed_only_judge,
        )

        root, request = _fact_gate_judge_request(args)
        replay = replay_fact_gate_changed_only_judge(
            **request,
            expected_judge_sha256=str(args.expected_judge_sha256),
            max_concurrency=int(args.max_concurrency),
        )
        result = {
            "arm_label": replay.judge_artifact.payload["arm_label"],
            "checkpoint_hits": replay.checkpoint_hits,
            "correct": replay.correct,
            "judge_replay_sha256": replay.judge_artifact.sha256,
            "output_root": str(root),
            "physical_provider_calls": 0,
            "score_ledger_replay_sha256": (
                replay.score_ledger_artifact.sha256
            ),
        }
    elif args.command == "closure-answer-preflight":
        from tools.matched_eval.closure_live import preflight_closure_answers

        root, request = _closure_answer_request(args)
        preflight = preflight_closure_answers(**request)
        result = {
            "answer_preflight_sha256": preflight.sha256,
            "arm_label": preflight.payload["arm_label"],
            "output_root": str(root),
            "physical_provider_calls": 0,
            "required_authorized_provider_calls": preflight.payload[
                "required_authorized_provider_calls"
            ],
        }
    elif args.command == "closure-answer":
        from tools.matched_eval.closure_live import run_closure_answers

        root, request = _closure_answer_request(args)
        run = run_closure_answers(
            **request,
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
        )
        result = {
            "answer_run_sha256": run.answer_artifact.sha256,
            "arm_label": run.answer_artifact.payload["arm_label"],
            "checkpoint_hits": run.checkpoint_hits,
            "output_root": str(root),
            "physical_provider_calls": run.physical_provider_calls,
            "runtime_ledger_sha256": run.runtime_ledger_artifact.sha256,
        }
    elif args.command == "closure-answer-replay":
        from tools.matched_eval.closure_live import replay_closure_answers

        root, request = _closure_answer_request(args)
        replay = replay_closure_answers(
            **request,
            expected_run_sha256=str(args.expected_run_sha256),
        )
        result = {
            "answer_replay_sha256": replay.replay_sha256,
            "answer_run_sha256": replay.run_sha256,
            "arm_label": replay.arm_label,
            "output_root": str(root),
            "physical_provider_calls": 0,
            "runtime_ledger_sha256": replay.runtime_ledger_sha256,
        }
    elif args.command == "closure-judge-preflight":
        from tools.matched_eval.closure_judging import (
            preflight_closure_changed_only_judge,
        )

        root, request = _closure_judge_request(args)
        preflight = preflight_closure_changed_only_judge(**request)
        result = {
            "arm_label": preflight.payload["arm_label"],
            "judge_preflight_sha256": preflight.sha256,
            "output_root": str(root),
            "physical_provider_calls": 0,
            "required_authorized_provider_calls": preflight.payload[
                "required_authorized_provider_calls"
            ],
        }
    elif args.command == "closure-judge":
        from tools.matched_eval.closure_judging import (
            run_closure_changed_only_judge,
        )

        root, request = _closure_judge_request(args)
        judged = run_closure_changed_only_judge(
            **request,
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
        )
        result = {
            "arm_label": judged.judge_artifact.payload["arm_label"],
            "checkpoint_hits": judged.checkpoint_hits,
            "correct": judged.correct,
            "judge_sha256": judged.judge_artifact.sha256,
            "output_root": str(root),
            "physical_provider_calls": judged.physical_provider_calls,
            "score_ledger_sha256": judged.score_ledger_artifact.sha256,
        }
    elif args.command == "closure-judge-replay":
        from tools.matched_eval.closure_judging import (
            replay_closure_changed_only_judge,
        )

        root, request = _closure_judge_request(args)
        replay = replay_closure_changed_only_judge(
            **request,
            expected_judge_sha256=str(args.expected_judge_sha256),
            max_concurrency=int(args.max_concurrency),
        )
        result = {
            "arm_label": replay.judge_artifact.payload["arm_label"],
            "checkpoint_hits": replay.checkpoint_hits,
            "correct": replay.correct,
            "judge_replay_sha256": replay.judge_artifact.sha256,
            "output_root": str(root),
            "physical_provider_calls": 0,
            "score_ledger_replay_sha256": (
                replay.score_ledger_artifact.sha256
            ),
        }
    elif args.command == "s0-v2-judge-preflight":
        from tools.matched_eval.judging import preflight_s0_v2_judge
        from tools.matched_eval.live import ANSWER_REPLAY_NAME, ANSWER_RUN_NAME

        root = Path(args.output_root)
        preflight = preflight_s0_v2_judge(
            answer_run_path=root / ANSWER_RUN_NAME,
            answer_replay_path=root / ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=str(args.expected_answer_run_sha256),
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=root,
            max_concurrency=int(args.max_concurrency),
        )
        result = {
            "judge_preflight_sha256": preflight.sha256,
            "physical_provider_calls": 0,
            "required_authorized_provider_calls": preflight.payload[
                "required_authorized_provider_calls"
            ],
        }
    elif args.command == "s0-v2-judge":
        from tools.matched_eval.judging import run_s0_v2_judge
        from tools.matched_eval.live import ANSWER_REPLAY_NAME, ANSWER_RUN_NAME

        root = Path(args.output_root)
        judged = run_s0_v2_judge(
            answer_run_path=root / ANSWER_RUN_NAME,
            answer_replay_path=root / ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=str(args.expected_answer_run_sha256),
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=root,
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
        )
        result = {
            "correct": judged.correct,
            "judge_sha256": judged.judge_artifact.sha256,
            "checkpoint_hits": judged.checkpoint_hits,
            "physical_provider_calls": judged.physical_provider_calls,
            "score_ledger_sha256": judged.score_ledger_artifact.sha256,
        }
    elif args.command == "s0-v2-judge-replay":
        from tools.matched_eval.judging import replay_s0_v2_judge
        from tools.matched_eval.live import ANSWER_REPLAY_NAME, ANSWER_RUN_NAME

        root = Path(args.output_root)
        replay = replay_s0_v2_judge(
            answer_run_path=root / ANSWER_RUN_NAME,
            answer_replay_path=root / ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=str(args.expected_answer_run_sha256),
            expected_judge_sha256=str(args.expected_judge_sha256),
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=root,
            max_concurrency=int(args.max_concurrency),
        )
        result = {
            "correct": replay.correct,
            "judge_replay_sha256": replay.judge_artifact.sha256,
            "physical_provider_calls": 0,
            "score_ledger_replay_sha256": replay.score_ledger_artifact.sha256,
        }
    elif args.command == "s0-v3-judge-preflight":
        from tools.matched_eval.judging import preflight_s0_v3_judge
        from tools.matched_eval.live import ANSWER_REPLAY_NAME, ANSWER_RUN_NAME

        root = Path(args.output_root)
        preflight = preflight_s0_v3_judge(
            answer_run_path=root / ANSWER_RUN_NAME,
            answer_replay_path=root / ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=str(args.expected_answer_run_sha256),
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=root,
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "judge_preflight_sha256": preflight.sha256,
            "physical_provider_calls": 0,
            "required_authorized_provider_calls": preflight.payload[
                "required_authorized_provider_calls"
            ],
        }
    elif args.command == "s0-v3-judge":
        from tools.matched_eval.judging import run_s0_v3_judge
        from tools.matched_eval.live import ANSWER_REPLAY_NAME, ANSWER_RUN_NAME

        root = Path(args.output_root)
        judged = run_s0_v3_judge(
            answer_run_path=root / ANSWER_RUN_NAME,
            answer_replay_path=root / ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=str(args.expected_answer_run_sha256),
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=root,
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "correct": judged.correct,
            "judge_sha256": judged.judge_artifact.sha256,
            "checkpoint_hits": judged.checkpoint_hits,
            "physical_provider_calls": judged.physical_provider_calls,
            "score_ledger_sha256": judged.score_ledger_artifact.sha256,
        }
    elif args.command == "s0-v3-judge-replay":
        from tools.matched_eval.judging import replay_s0_v3_judge
        from tools.matched_eval.live import ANSWER_REPLAY_NAME, ANSWER_RUN_NAME

        root = Path(args.output_root)
        replay = replay_s0_v3_judge(
            answer_run_path=root / ANSWER_RUN_NAME,
            answer_replay_path=root / ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=str(args.expected_answer_run_sha256),
            expected_judge_sha256=str(args.expected_judge_sha256),
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=root,
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "correct": replay.correct,
            "judge_replay_sha256": replay.judge_artifact.sha256,
            "physical_provider_calls": 0,
            "score_ledger_replay_sha256": replay.score_ledger_artifact.sha256,
        }
    elif args.command == "s0-v4-judge-preflight":
        from tools.matched_eval.judging import preflight_s0_v4_judge
        from tools.matched_eval.live import ANSWER_REPLAY_NAME, ANSWER_RUN_NAME

        root = Path(args.output_root)
        _require_v4_diagnostic_gate(
            output_root=root,
            selected_ordinals=args.ordinals,
        )
        preflight = preflight_s0_v4_judge(
            answer_run_path=root / ANSWER_RUN_NAME,
            answer_replay_path=root / ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=str(args.expected_answer_run_sha256),
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=root,
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "judge_preflight_sha256": preflight.sha256,
            "physical_provider_calls": 0,
            "required_authorized_provider_calls": preflight.payload[
                "required_authorized_provider_calls"
            ],
        }
    elif args.command == "s0-v4-judge":
        from tools.matched_eval.judging import run_s0_v4_judge
        from tools.matched_eval.live import ANSWER_REPLAY_NAME, ANSWER_RUN_NAME

        root = Path(args.output_root)
        _require_v4_diagnostic_gate(
            output_root=root,
            selected_ordinals=args.ordinals,
        )
        judged = run_s0_v4_judge(
            answer_run_path=root / ANSWER_RUN_NAME,
            answer_replay_path=root / ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=str(args.expected_answer_run_sha256),
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=root,
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "correct": judged.correct,
            "judge_sha256": judged.judge_artifact.sha256,
            "checkpoint_hits": judged.checkpoint_hits,
            "physical_provider_calls": judged.physical_provider_calls,
            "score_ledger_sha256": judged.score_ledger_artifact.sha256,
        }
    elif args.command == "s0-v4-judge-replay":
        from tools.matched_eval.judging import replay_s0_v4_judge
        from tools.matched_eval.live import ANSWER_REPLAY_NAME, ANSWER_RUN_NAME

        root = Path(args.output_root)
        _require_v4_diagnostic_gate(
            output_root=root,
            selected_ordinals=args.ordinals,
        )
        replay = replay_s0_v4_judge(
            answer_run_path=root / ANSWER_RUN_NAME,
            answer_replay_path=root / ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=str(args.expected_answer_run_sha256),
            expected_judge_sha256=str(args.expected_judge_sha256),
            retrieval_path=Path(args.retrieval),
            dataset_path=Path(args.dataset),
            split_path=Path(args.split),
            output_root=root,
            max_concurrency=int(args.max_concurrency),
            selected_ordinals=args.ordinals,
        )
        result = {
            "correct": replay.correct,
            "judge_replay_sha256": replay.judge_artifact.sha256,
            "physical_provider_calls": 0,
            "score_ledger_replay_sha256": replay.score_ledger_artifact.sha256,
        }
    else:
        result = inspect_outputs(output_root=Path(args.output_root))
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
