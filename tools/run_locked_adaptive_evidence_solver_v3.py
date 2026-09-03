#!/usr/bin/env python3
"""Run the locked adaptive map-plus-source evidence solver V3 arm.

The source mapper and final solver remain two separately authorized planes.
This runner consumes only an already sealed source-map materialization, derives
post-map fact unions (optionally filtering logical source lanes), and seals one
actionable final prompt per question with at least one admitted source fact.
Map-only and empty-lane rows preserve the exact direct parent without a call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import run_locked_adaptive_source_map as source_cli  # noqa: E402
from tools import run_locked_query_evidence_map_solver_v2 as map_cli  # noqa: E402
from tools import run_locked_query_payload_answers as payload_cli  # noqa: E402
from tools.matched_eval import provider_runtime  # noqa: E402
from tools.matched_eval.adaptive_evidence_solver_live import (  # noqa: E402
    ARM_LABEL,
    FORMAT as SOLVER_FORMAT,
    AdaptiveEvidenceSolverPlan,
    AdaptiveEvidenceSolverPreflight,
    AdaptiveEvidenceSolverRun,
    AdaptiveSolverCompletionPlane,
    VerifiedAdaptiveEvidenceSolverPlane,
    build_adaptive_evidence_solver_plan,
    capture_adaptive_solver_completions,
    materialize_adaptive_evidence_solver,
    preflight_adaptive_evidence_solver,
    replay_adaptive_evidence_solver,
)
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.query_evidence_map_solver_v2_live import (  # noqa: E402
    MAX_PROMPT_TOKENS,
    SOLVER_OUTPUT_TOKEN_RESERVE,
    EvidenceMapPlan,
    VerifiedEvidenceMapPlane,
)
from tools.matched_eval.source_history_fact_union import (  # noqa: E402
    LANE_ORDER,
    FactLane,
    PostMapFactUnion,
    build_post_map_fact_union,
)
from tools.matched_eval.source_history_mapper_live import (  # noqa: E402
    SourceMapperMaterialization,
)


FORMAT = "memory-condense-locked-adaptive-evidence-solver-v3"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RUN_FORMAT = f"{FORMAT}-run-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"
PREFLIGHT_NAME = "adaptive-evidence-solver-v3-preflight.json"
RUN_NAME = "adaptive-evidence-solver-v3-run.json"
REPLAY_NAME = "adaptive-evidence-solver-v3-replay.json"
CHECKPOINT_DIR_NAME = "terra-adaptive-evidence-solver-v3-calls"

DEFAULT_SOURCE_ROOT = source_cli.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    map_cli.DEFAULT_OUTPUT.parent / "s0-plus-adaptive-evidence-solver-v3"
)
EXPECTED_QUESTION_COUNT = 100

_LANE_ALIASES = {
    "d": FactLane.DIRECT,
    "direct": FactLane.DIRECT,
    "p": FactLane.PARTITION,
    "partition": FactLane.PARTITION,
    "g": FactLane.GUIDED,
    "guided": FactLane.GUIDED,
    "e": FactLane.EM,
    "em": FactLane.EM,
}


class LockedAdaptiveEvidenceSolverError(MatchedEvalContractError):
    """A parent, source materialization, lane profile, prompt, or run changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedAdaptiveEvidenceSolverError(message)


def _plain_messages(messages: Sequence[Any]) -> tuple[dict[str, str], ...]:
    return tuple({"role": row.role, "content": row.content} for row in messages)


def parse_lane_filter(value: str) -> tuple[FactLane, ...]:
    """Parse ``all`` or a comma-separated D/P/G/EM profile canonically."""

    if type(value) is not str or not value.strip():
        raise argparse.ArgumentTypeError("lane filter must be nonempty")
    raw = tuple(part.strip().casefold() for part in value.split(","))
    if raw == ("all",):
        return LANE_ORDER
    if "all" in raw:
        raise argparse.ArgumentTypeError("all cannot be combined with lane names")
    if any(part not in _LANE_ALIASES for part in raw):
        raise argparse.ArgumentTypeError(
            "lanes must be all or a comma-separated subset of D,P,G,EM"
        )
    selected = {_LANE_ALIASES[part] for part in raw}
    if not selected:
        raise argparse.ArgumentTypeError("lane filter cannot be empty")
    return tuple(lane for lane in LANE_ORDER if lane in selected)


def _lane_filter_receipt(
    lanes: tuple[FactLane, ...],
    *,
    source_preflight_sha256: str,
    source_materialization_sha256: str,
) -> str:
    return identity_sha256(
        {
            "format": f"{FORMAT}-lane-filter-v1",
            "lanes": [row.value for row in lanes],
            "operation_position": "after_mapper_validation_before_post_map_union",
            "preserve_logical_lane_origins": True,
            "source_materialization_sha256": source_materialization_sha256,
            "source_preflight_sha256": source_preflight_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class LoadedAdaptiveSolverPlan:
    source_preflight: SealedArtifact
    source_work_manifest: SealedArtifact
    source_materialization: SealedArtifact
    map_plan: EvidenceMapPlan
    map_plane: VerifiedEvidenceMapPlane
    fact_unions: Mapping[str, PostMapFactUnion]
    lanes: tuple[FactLane, ...]
    lane_filter_receipt_sha256: str
    plan: AdaptiveEvidenceSolverPlan
    preflight: AdaptiveEvidenceSolverPreflight


@dataclass(frozen=True, slots=True)
class LoadedAdaptiveSolverRun:
    """Complete provider-free replay of one exact terminal solver artifact."""

    loaded: LoadedAdaptiveSolverPlan
    provider_preflight: SealedArtifact
    completion_batch: FastCompletionBatch | None
    completion_plane: AdaptiveSolverCompletionPlane
    run: AdaptiveEvidenceSolverRun
    verified_plane: VerifiedAdaptiveEvidenceSolverPlane
    terminal: SealedArtifact


def _source_fact_unions(
    questions: Sequence[source_cli.FastMaterializationQuestionPlan],
    materializations: Sequence[SourceMapperMaterialization],
    *,
    lanes: tuple[FactLane, ...],
) -> dict[str, PostMapFactUnion]:
    """Derive one exact post-map union per activated source question."""

    _require(
        type(lanes) is tuple
        and bool(lanes)
        and tuple(row for row in LANE_ORDER if row in set(lanes)) == lanes,
        "lane profile changed canonical order",
    )
    _require(
        len(questions) == len(materializations),
        "source question/materialization populations differ",
    )
    selected = set(lanes)
    result: dict[str, PostMapFactUnion] = {}
    for question, materialization in zip(
        questions, materializations, strict=True
    ):
        _require(
            type(question) is source_cli.FastMaterializationQuestionPlan
            and type(materialization) is SourceMapperMaterialization,
            "typed source materialization input changed",
        )
        hydration = question.hydration_plan
        _require(
            materialization.hydration_plan_receipt_sha256
            == hydration.receipt_sha256
            and materialization.mapping_plan_receipt_sha256
            == question.mapping_plan.receipt_sha256
            and materialization.preflight_receipt_sha256
            == question.mapper_preflight.receipt_sha256
            and materialization.provider_calls_during_materialization == 0
            and materialization.retained_transformer_token_state_bytes == 0,
            "typed mapper materialization escaped its question plans",
        )
        windows = {row.window_id: row for row in hydration.windows}
        batches = []
        for batch in materialization.batches:
            window = windows.get(batch.window_id)
            _require(window is not None, "mapper batch escaped hydration windows")
            assert window is not None
            if window.selection.lane in selected:
                batches.append(batch)
        union = build_post_map_fact_union(
            hydration,
            batches=tuple(batches),
            direct_evidence=question.direct_evidence,
        )
        _require(
            question.question_id not in result,
            "source materialization repeated a question ID",
        )
        result[question.question_id] = union
    return result


def _source_profile_from_preflight(
    artifact: SealedArtifact,
) -> tuple[str, str]:
    obligation_mode = require_text(
        artifact.payload.get("obligation_compilation_mode"),
        "source-map obligation compilation mode",
    )
    state_chain_profile = require_text(
        artifact.payload.get("state_chain_profile"),
        "source-map state-chain profile",
    )
    return obligation_mode, state_chain_profile


def _guard_output_roots(output_root: Path, source_root: Path) -> None:
    output = output_root.resolve()
    immutable = (
        source_root.resolve(),
        Path(source_cli.DEFAULT_MAP_ROOT).resolve(),
        Path(map_cli.DEFAULT_DIRECT_ANSWER_ROOT).resolve(),
        Path(payload_cli.DEFAULT_PARENT_ROOT).resolve(),
    )
    _require(output not in immutable, "solver output must be a separate artifact root")


def _load_plan(args: argparse.Namespace) -> LoadedAdaptiveSolverPlan:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    _guard_output_roots(output_root, source_root)
    (
        source_preflight,
        source_work_manifest,
        source_materialization,
        source_questions,
        source_results,
    ) = source_cli.load_typed_materialization_root(
        source_root,
        expected_preflight_sha256=args.expected_source_preflight_sha256,
        expected_materialization_sha256=(
            args.expected_source_materialization_sha256
        ),
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
        direct_base_cap=int(args.direct_base_cap),
        partition_base_cap=int(args.partition_base_cap),
        guided_base_cap=int(args.guided_base_cap),
    )
    obligation_mode, state_chain_profile = _source_profile_from_preflight(
        source_preflight
    )
    _query_run, map_plan, map_plane, adapter = source_cli.load_locked_query_map(
        max_concurrency=int(args.max_concurrency),
        gateway_url=str(args.gateway_url),
        obligation_mode=obligation_mode,
        state_chain_profile=state_chain_profile,
    )
    source_payload = source_preflight.payload
    _require(
        source_payload.get("map_preflight_sha256")
        == source_cli.EXPECTED_MAP_PREFLIGHT_SHA256
        and source_payload.get("map_run_sha256")
        == source_cli.EXPECTED_MAP_RUN_SHA256
        and source_payload.get("map_runtime_ledger_sha256")
        == source_cli.EXPECTED_MAP_RUNTIME_SHA256
        and source_payload.get("query_map_adapter_receipt_sha256")
        == adapter.receipt_sha256,
        "source-map artifacts escaped their terminal map/adapter parent",
    )
    _require(
        len(map_plan.rows) == len(map_plane.rows) == EXPECTED_QUESTION_COUNT,
        "locked adaptive solver requires exactly 100 terminal map rows",
    )
    map_by_id = {row.question_id: row for row in map_plane.rows}
    _require(
        len(map_by_id) == EXPECTED_QUESTION_COUNT
        and all(
            question.question_id in map_by_id
            and question.ordinal == map_by_id[question.question_id].ordinal
            and question.hydration_plan.parent.parent_packet_id
            == map_plan.rows[question.ordinal].packet_id
            for question in source_questions
        ),
        "typed source questions escaped terminal map order/packet IDs",
    )
    lanes = tuple(args.lanes)
    unions = _source_fact_unions(source_questions, source_results, lanes=lanes)
    plan = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        source_fact_unions=unions,
    )
    preflight = preflight_adaptive_evidence_solver(plan)
    lane_receipt = _lane_filter_receipt(
        lanes,
        source_preflight_sha256=source_preflight.sha256,
        source_materialization_sha256=source_materialization.sha256,
    )
    return LoadedAdaptiveSolverPlan(
        source_preflight,
        source_work_manifest,
        source_materialization,
        map_plan,
        map_plane,
        unions,
        lanes,
        lane_receipt,
        plan,
        preflight,
    )


def _prompt_rows(plan: AdaptiveEvidenceSolverPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in plan.submitted_rows:
        assert row.messages is not None
        assert row.messages_sha256 is not None
        assert row.prompt_id is not None
        assert row.prompt_token_proxy is not None
        messages = _plain_messages(row.messages)
        _require(
            identity_sha256(list(messages)) == row.messages_sha256,
            "adaptive solver messages changed before sealing",
        )
        rows.append(
            {
                "messages": list(messages),
                "messages_sha256": row.messages_sha256,
                "ordinal": row.ordinal,
                "prompt_id": row.prompt_id,
                "prompt_token_proxy": row.prompt_token_proxy,
                "question_id": row.question_id,
            }
        )
    return rows


def _preflight_projection(
    loaded: LoadedAdaptiveSolverPlan,
    *,
    gateway_url: str,
    model: str,
    max_concurrency: int,
) -> dict[str, Any]:
    plan, preflight = loaded.plan, loaded.preflight
    prompt_rows = _prompt_rows(plan)
    source_rows = []
    for row in plan.rows:
        union = row.fact_union
        envelope = row.fact_envelope
        source_rows.append(
            {
                "admitted_source_fact_count": len(row.allowed_source_fact_ids),
                "fact_envelope_receipt_sha256": (
                    None if envelope is None else envelope.receipt_sha256
                ),
                "fact_union_receipt_sha256": (
                    None if union is None else union.receipt_sha256
                ),
                "ordinal": row.ordinal,
                "plan_row_receipt_sha256": row.receipt_sha256,
                "question_id": row.question_id,
                "submitted": row.submitted,
            }
        )
    payload: dict[str, Any] = {
        "actionable_submission_rule": "at_least_one_admitted_source_fact_alias",
        "adaptive_solver_preflight": preflight.projection(),
        "adaptive_solver_preflight_receipt_sha256": preflight.receipt_sha256,
        "arm_label": ARM_LABEL,
        "fact_union_question_count": len(loaded.fact_unions),
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_prompt_token_cap": MAX_PROMPT_TOKENS,
        "lane_filter": [row.value for row in loaded.lanes],
        "lane_filter_receipt_sha256": loaded.lane_filter_receipt_sha256,
        "map_plan_identity_sha256": loaded.map_plan.plan_identity_sha256,
        "map_replay_sha256": loaded.map_plane.replay_sha256,
        "map_run_sha256": loaded.map_plane.run_sha256,
        "map_runtime_ledger_sha256": loaded.map_plane.runtime_ledger_sha256,
        "max_concurrency": max_concurrency,
        "model": model,
        "output_token_reserve": SOLVER_OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": prompt_rows,
        "provider_calls": 0,
        "question_count": len(plan.rows),
        "required_authorized_provider_calls": plan.required_calls,
        "retained_transformer_token_state_bytes": 0,
        "source_materialization_sha256": loaded.source_materialization.sha256,
        "source_preflight_sha256": loaded.source_preflight.sha256,
        "source_question_rows": source_rows,
        "source_work_manifest_sha256": loaded.source_work_manifest.sha256,
    }
    assert_gold_blind(payload, path="locked_adaptive_solver_preflight")
    return payload


def _adaptive_preflight_receipt(value: Mapping[str, Any]) -> str:
    body = {
        "map_replay_sha256": value.get("map_replay_sha256"),
        "map_run_sha256": value.get("map_run_sha256"),
        "map_runtime_ledger_sha256": value.get("map_runtime_ledger_sha256"),
        "observed_max_prompt_token_proxy": value.get(
            "observed_max_prompt_token_proxy"
        ),
        "ordered_row_receipt_sha256s": value.get(
            "ordered_row_receipt_sha256s"
        ),
        "output_token_reserve": value.get("output_token_reserve"),
        "plan_identity_sha256": value.get("plan_identity_sha256"),
        "prompt_population_sha256": value.get("prompt_population_sha256"),
        "required_authorized_provider_calls": value.get(
            "required_authorized_provider_calls"
        ),
        "submitted_prompt_ids": value.get("submitted_prompt_ids"),
    }
    return identity_sha256(
        {"format": f"{SOLVER_FORMAT}-preflight", **body}
    )


def _validate_provider_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[str, ...]]:
    payload = artifact.payload
    assert_gold_blind(payload, path="locked_adaptive_solver_provider")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("arm_label") == ARM_LABEL
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_prompt_token_cap") == MAX_PROMPT_TOKENS
        and payload.get("output_token_reserve") == SOLVER_OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT,
        "adaptive solver provider preflight changed its firewall/envelope",
    )
    adaptive = payload.get("adaptive_solver_preflight")
    _require(type(adaptive) is dict, "adaptive solver preflight projection missing")
    assert type(adaptive) is dict
    expected_receipt = require_sha256(
        payload.get("adaptive_solver_preflight_receipt_sha256"),
        "adaptive solver preflight receipt",
    )
    _require(
        _adaptive_preflight_receipt(adaptive) == expected_receipt,
        "adaptive solver preflight receipt changed",
    )
    raw_rows = payload.get("physical_prompt_rows")
    _require(type(raw_rows) is list, "physical adaptive prompt rows changed type")
    prompts: list[tuple[dict[str, str], ...]] = []
    question_ids: list[str] = []
    prompt_ids: list[str] = []
    for index, raw in enumerate(raw_rows):
        _require(type(raw) is dict, "physical adaptive prompt row changed type")
        assert type(raw) is dict
        _require(
            set(raw)
            == {
                "messages",
                "messages_sha256",
                "ordinal",
                "prompt_id",
                "prompt_token_proxy",
                "question_id",
            },
            "physical adaptive prompt row schema changed",
        )
        messages = raw.get("messages")
        _require(
            type(messages) is list
            and bool(messages)
            and all(
                type(row) is dict
                and set(row) == {"role", "content"}
                and row.get("role") in {"system", "user", "assistant"}
                and type(row.get("content")) is str
                for row in messages
            ),
            "sealed adaptive prompt messages changed",
        )
        plain = tuple(
            {"role": str(row["role"]), "content": str(row["content"])}
            for row in messages
        )
        message_sha = require_sha256(
            raw.get("messages_sha256"), "adaptive prompt messages"
        )
        prompt_id = require_sha256(raw.get("prompt_id"), "adaptive prompt ID")
        question_id = require_text(raw.get("question_id"), "adaptive question ID")
        _require(
            identity_sha256(list(plain)) == message_sha
            and type(raw.get("ordinal")) is int
            and int(raw["ordinal"]) >= 0
            and type(raw.get("prompt_token_proxy")) is int
            and int(raw["prompt_token_proxy"]) >= 1
            and int(raw["prompt_token_proxy"])
            + SOLVER_OUTPUT_TOKEN_RESERVE
            <= MAX_PROMPT_TOKENS,
            f"sealed adaptive prompt binding changed at submitted row {index}",
        )
        prompts.append(plain)
        prompt_ids.append(prompt_id)
        question_ids.append(question_id)
    required = payload.get("required_authorized_provider_calls")
    _require(
        type(required) is int
        and required == len(prompts)
        and adaptive.get("required_authorized_provider_calls") == required
        and adaptive.get("submitted_prompt_ids") == prompt_ids
        and len(set(prompt_ids)) == len(prompt_ids)
        and len(set(question_ids)) == len(question_ids),
        "adaptive provider prompt/call population changed",
    )
    if prompts:
        population = preflight_fast_completion_prompts(
            prompts, max_prompt_tokens=MAX_PROMPT_TOKENS
        )
        _require(
            population.logical_prompt_count
            == population.unique_prompt_count
            == required
            and population.prompt_population_sha256
            == adaptive.get("prompt_population_sha256")
            and all(
                fast.messages_sha256 == raw["messages_sha256"]
                and fast.prompt_token_proxy == raw["prompt_token_proxy"]
                for fast, raw in zip(
                    population.ordered_rows, raw_rows, strict=True
                )
            ),
            "adaptive sealed prompt population changed",
        )
    else:
        empty_sha = identity_sha256(
            {
                "format": f"{SOLVER_FORMAT}-empty-prompt-population",
                "max_prompt_tokens": MAX_PROMPT_TOKENS,
            }
        )
        _require(
            adaptive.get("prompt_population_sha256") == empty_sha,
            "empty adaptive prompt population changed",
        )
    return tuple(prompts), tuple(question_ids)


def _read_preflight(
    output_root: Path,
    expected_sha256: str,
) -> tuple[SealedArtifact, tuple[tuple[dict[str, str], ...], ...], tuple[str, ...]]:
    expected = require_sha256(expected_sha256, "expected adaptive solver preflight")
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(artifact.sha256 == expected, "adaptive solver preflight changed")
    prompts, question_ids = _validate_provider_preflight(artifact)
    return artifact, prompts, question_ids


def _runtime(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionRuntime:
    payload = artifact.payload
    _require(
        payload.get("model") == model
        and payload.get("gateway_url") == gateway_url
        and payload.get("max_concurrency") == max_concurrency,
        "adaptive solver runtime settings differ from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=output_root / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=SOLVER_OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "arm_label": ARM_LABEL,
            "experiment_format": RUN_FORMAT,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "lane_filter_receipt_sha256": payload[
                "lane_filter_receipt_sha256"
            ],
            "solver_preflight_artifact_sha256": artifact.sha256,
            "source_materialization_sha256": payload[
                "source_materialization_sha256"
            ],
        },
    )


def _checkpoint_batch(
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    _require(bool(prompts), "empty solver population has no completion batch")
    runtime = _runtime(
        artifact,
        prompts,
        output_root=Path(args.output_root),
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def _completion_plane(
    loaded: LoadedAdaptiveSolverPlan,
    question_ids: tuple[str, ...],
    batch: FastCompletionBatch | None,
) -> AdaptiveSolverCompletionPlane:
    plan = loaded.plan
    expected_ids = tuple(row.question_id for row in plan.submitted_rows)
    _require(
        question_ids == expected_ids,
        "sealed provider prompt order differs from rebuilt adaptive plan",
    )
    if batch is None:
        _require(not expected_ids, "submitted solver rows lost completion batch")
        completions: dict[str, str] = {}
    else:
        _require(
            batch.usage.physical_calls == 0
            and batch.usage.checkpoint_hits == len(expected_ids)
            and tuple(
                row.messages_sha256 for row in batch.prompt_population.ordered_rows
            )
            == tuple(
                require_sha256(row.messages_sha256, "rebuilt solver messages")
                for row in plan.submitted_rows
            ),
            "adaptive materialization requires a complete checkpoint-only population",
        )
        completions = dict(
            zip(expected_ids, batch.logical_completions, strict=True)
        )
    return capture_adaptive_solver_completions(
        plan,
        loaded.preflight,
        completions,
    )


def _run_projection(
    loaded: LoadedAdaptiveSolverPlan,
    completion_plane: AdaptiveSolverCompletionPlane,
    run: AdaptiveEvidenceSolverRun,
    batch: FastCompletionBatch | None,
    *,
    solver_preflight_artifact_sha256: str,
) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    planned_by_id = {row.question_id: row for row in loaded.plan.rows}
    for row in run.rows:
        planned = planned_by_id[row.question_id]
        questions.append(
            {
                "allowed_map_item_ids": list(planned.allowed_map_item_ids),
                "allowed_source_fact_ids": list(
                    planned.allowed_source_fact_ids
                ),
                "changed_from_parent": row.changed_from_parent,
                "completion_receipt_sha256": row.completion_receipt_sha256,
                "dated_question_sha256": row.dated_question_sha256,
                "fact_envelope_receipt_sha256": (
                    None
                    if planned.fact_envelope is None
                    else planned.fact_envelope.receipt_sha256
                ),
                "fact_union_receipt_sha256": (
                    None
                    if planned.fact_union is None
                    else planned.fact_union.receipt_sha256
                ),
                "ordinal": row.ordinal,
                "parent_prediction_sha256": row.parent_prediction_sha256,
                "plan_row_receipt_sha256": row.plan_row_receipt_sha256,
                "prediction": {
                    "sha256": row.prediction_sha256,
                    "text": row.prediction,
                },
                "prediction_source": row.prediction_source,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
                "solver_decision": row.solver_decision,
                "solver_parse_receipt_sha256": (
                    row.solver_parse_receipt_sha256
                ),
                "solver_used_evidence_ids": list(
                    row.solver_used_evidence_ids
                ),
                "solver_used_map_item_ids": list(
                    row.solver_used_map_item_ids
                ),
                "solver_used_source_fact_ids": list(
                    row.solver_used_source_fact_ids
                ),
                "solver_valid": row.solver_valid,
            }
        )
    payload: dict[str, Any] = {
        "adaptive_solver_completion_plane_receipt_sha256": (
            completion_plane.receipt_sha256
        ),
        "adaptive_solver_preflight_receipt_sha256": (
            loaded.preflight.receipt_sha256
        ),
        "adaptive_solver_run_receipt_sha256": run.receipt_sha256,
        "arm_label": ARM_LABEL,
        "completion_batch": None if batch is None else batch.model_dump(),
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "lane_filter": [row.value for row in loaded.lanes],
        "lane_filter_receipt_sha256": loaded.lane_filter_receipt_sha256,
        "map_run_sha256": loaded.map_plane.run_sha256,
        "physical_provider_calls_during_materialization": 0,
        "plan_identity_sha256": loaded.plan.plan_identity_sha256,
        "question_count": len(run.rows),
        "questions": questions,
        "required_authorized_provider_calls": loaded.plan.required_calls,
        "retained_transformer_token_state_bytes": 0,
        "solver_preflight_artifact_sha256": solver_preflight_artifact_sha256,
        "source_materialization_sha256": loaded.source_materialization.sha256,
        "source_preflight_sha256": loaded.source_preflight.sha256,
        "source_work_manifest_sha256": loaded.source_work_manifest.sha256,
    }
    assert_gold_blind(payload, path="locked_adaptive_solver_run")
    return payload


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    loaded = _load_plan(args)
    payload = _preflight_projection(
        loaded,
        gateway_url=str(args.gateway_url),
        model=str(args.model),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / PREFLIGHT_NAME,
        payload,
    )
    return {
        "actionable_question_count": loaded.plan.required_calls,
        "artifact": artifact.path.as_posix(),
        "created": created,
        "gold_loaded": False,
        "lane_filter": [row.value for row in loaded.lanes],
        "max_combined_prompt_and_reserve_tokens": max(
            (
                int(row.prompt_token_proxy or 0)
                + SOLVER_OUTPUT_TOKEN_RESERVE
                for row in loaded.plan.submitted_rows
            ),
            default=0,
        ),
        "preflight_sha256": artifact.sha256,
        "provider_calls": 0,
        "question_count": len(loaded.plan.rows),
        "required_authorized_provider_calls": loaded.plan.required_calls,
        "retained_transformer_token_state_bytes": 0,
    }


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, _question_ids = _read_preflight(
        Path(args.output_root), args.expected_preflight_sha256
    )
    required = len(prompts)
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == required,
        f"provider-run requires exact authorization for {required} calls",
    )
    if required == 0:
        return {
            "checkpoint_hits": 0,
            "gold_loaded": False,
            "physical_provider_calls": 0,
            "preflight_sha256": artifact.sha256,
            "required_authorized_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
        }
    # The exact sealed gold-blind population and authorization are verified
    # before environment loading, client creation, or checkpoint I/O.
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = provider_runtime.make_provider_client(api_key, str(args.gateway_url))
    try:
        batch = _checkpoint_batch(artifact, prompts, args=args, client=client)
    except BaseException:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        raise
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
    }


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    loaded = _load_plan(args)
    artifact, prompts, question_ids = _read_preflight(
        Path(args.output_root), args.expected_preflight_sha256
    )
    rebuilt = _preflight_projection(
        loaded,
        gateway_url=str(args.gateway_url),
        model=str(args.model),
        max_concurrency=int(args.max_concurrency),
    )
    _require(
        artifact.payload == rebuilt,
        "adaptive solver preflight differs from rebuilt exact parents",
    )
    batch = (
        None
        if not prompts
        else _checkpoint_batch(artifact, prompts, args=args, client=None)
    )
    completions = _completion_plane(loaded, question_ids, batch)
    run = materialize_adaptive_evidence_solver(
        loaded.plan,
        loaded.preflight,
        completions,
    )
    payload = _run_projection(
        loaded,
        completions,
        run,
        batch,
        solver_preflight_artifact_sha256=artifact.sha256,
    )
    terminal, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    return {
        "changed_prediction_count": sum(row.changed_from_parent for row in run.rows),
        "checkpoint_hits": 0 if batch is None else batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "run_sha256": terminal.sha256,
        "terminal_run_replayed": not created,
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    loaded_run = load_verified_adaptive_solver_run(args)
    loaded = loaded_run.loaded
    artifact = loaded_run.provider_preflight
    terminal = loaded_run.terminal
    verified = loaded_run.verified_plane
    expected_run = terminal.sha256
    replay_payload = {
        "adaptive_verified_plane_receipt_sha256": verified.receipt_sha256,
        "byte_identical": True,
        "expected_run_sha256": expected_run,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "replayed_run_sha256": terminal.sha256,
        "retained_transformer_token_state_bytes": 0,
        "solver_preflight_artifact_sha256": artifact.sha256,
    }
    assert_gold_blind(replay_payload, path="locked_adaptive_solver_replay")
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME,
        replay_payload,
    )
    return {
        "byte_identical": True,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "run_sha256": terminal.sha256,
    }


def load_verified_adaptive_solver_run(
    args: argparse.Namespace,
) -> LoadedAdaptiveSolverRun:
    """Load and replay one exact terminal run with checkpoint hits only.

    This is the public judge seam.  It performs the same reconstruction as the
    ``replay`` command but publishes nothing and cannot create a provider
    client.
    """

    expected_run = require_sha256(args.expected_run_sha256, "expected adaptive run")
    loaded = _load_plan(args)
    artifact, prompts, question_ids = _read_preflight(
        Path(args.output_root), args.expected_preflight_sha256
    )
    rebuilt = _preflight_projection(
        loaded,
        gateway_url=str(args.gateway_url),
        model=str(args.model),
        max_concurrency=int(args.max_concurrency),
    )
    _require(artifact.payload == rebuilt, "adaptive replay preflight changed")
    batch = (
        None
        if not prompts
        else _checkpoint_batch(artifact, prompts, args=args, client=None)
    )
    completions = _completion_plane(loaded, question_ids, batch)
    run = materialize_adaptive_evidence_solver(
        loaded.plan,
        loaded.preflight,
        completions,
    )
    verified = replay_adaptive_evidence_solver(
        loaded.plan,
        loaded.preflight,
        completions,
        run,
    )
    expected_payload = _run_projection(
        loaded,
        completions,
        run,
        batch,
        solver_preflight_artifact_sha256=artifact.sha256,
    )
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        terminal.sha256 == expected_run and terminal.payload == expected_payload,
        "adaptive solver terminal run differs from deterministic replay",
    )
    return LoadedAdaptiveSolverRun(
        loaded,
        artifact,
        batch,
        completions,
        run,
        verified,
        terminal,
    )


def _add_plan_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--expected-source-preflight-sha256", required=True)
    parser.add_argument("--expected-source-materialization-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lanes", type=parse_lane_filter, default=LANE_ORDER)
    parser.add_argument(
        "--direct-base-cap", type=int, default=source_cli.DEFAULT_DIRECT_BASE_CAP
    )
    parser.add_argument(
        "--partition-base-cap",
        type=int,
        default=source_cli.DEFAULT_PARTITION_BASE_CAP,
    )
    parser.add_argument(
        "--guided-base-cap", type=int, default=source_cli.DEFAULT_GUIDED_BASE_CAP
    )
    parser.add_argument("--model", default=provider_runtime.DEFAULT_TERRA_GATEWAY_MODEL)
    parser.add_argument("--gateway-url", default=provider_runtime.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def _add_provider_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    parser.add_argument("--api-key-env", default=provider_runtime.DEFAULT_API_KEY_ENV)
    parser.add_argument("--model", default=provider_runtime.DEFAULT_TERRA_GATEWAY_MODEL)
    parser.add_argument("--gateway-url", default=provider_runtime.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _add_plan_inputs(preflight)
    provider = commands.add_parser("provider-run")
    _add_provider_inputs(provider)
    materialize = commands.add_parser("materialize")
    _add_plan_inputs(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)
    replay = commands.add_parser("replay")
    _add_plan_inputs(replay)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = _preflight(args)
    elif args.command == "provider-run":
        result = _provider(args)
    elif args.command == "materialize":
        result = _materialize(args)
    elif args.command == "replay":
        result = _replay(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE_ROOT",
    "FORMAT",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_FORMAT",
    "REPLAY_NAME",
    "RUN_FORMAT",
    "RUN_NAME",
    "LoadedAdaptiveSolverPlan",
    "LoadedAdaptiveSolverRun",
    "LockedAdaptiveEvidenceSolverError",
    "main",
    "load_verified_adaptive_solver_run",
    "parse_lane_filter",
]
