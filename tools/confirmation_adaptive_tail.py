#!/usr/bin/env python3
"""Confirmation adaptive-evidence solver and zero-fact source-tail stage.

The frozen validation lineage contains two independent Terra planes after the
D1/P0/G1 source map:

* the adaptive evidence solver consumes the validated V2 map plus the base
  post-map fact unions; and
* the source tail advances exactly one method-local source only for unresolved
  rows whose base mapping retained no facts.

This module ports those planes without validation ordinals, fixed hashes, or a
fixed population size.  Planning is typed, gold-blind, and provider-free.
Only ``run_*_provider`` may construct a client.  Both provider lifecycles use
the repository's native ``FastCompletionRuntime`` request/response journals,
seal exact remaining-call releases, refuse response-less requests and foreign
checkpoint state, and replay with ``client=None``.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools.confirmation_adaptive_source_map import (
    VerifiedConfirmationAdaptiveSourceMapPlane,
)
from tools.confirmation_source_streams import ConfirmationSourceStreamsResult
from tools import run_locked_adaptive_source_map as source_cli
from tools import run_locked_adaptive_source_tail_wave as tail_core
from tools.matched_eval import provider_runtime
from tools.matched_eval.adaptive_evidence_solver_live import (
    ARM_LABEL as SOLVER_ARM_LABEL,
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
from tools.matched_eval.adaptive_source_tail_typed import (
    TailFactUnionRow,
    build_tail_post_map_fact_unions,
)
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.locked_source_gate_adapter import (
    DIRECT_STREAM_PROFILE_REPACK_V2,
    DIRECT_STREAM_PROFILE_V1,
    LockedSourceGateAdapterPopulation,
)
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    MAX_PROMPT_TOKENS as SOLVER_MAX_PROMPT_TOKENS,
    SOLVER_OUTPUT_TOKEN_RESERVE,
    EvidenceMapPlan,
    VerifiedEvidenceMapPlane,
)
from tools.matched_eval.query_map_source_gate_adapter import (
    CONSOLIDATED_OBLIGATION_MODE,
    STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
)
from tools.matched_eval.source_gate_controller import (
    ObligationCoverageReceipt,
    SourceGateRound,
    assess_obligation_coverage,
    build_question_bound_mapping_plan,
    coverage_facts_from_fact_union,
    start_source_gate,
)
from tools.matched_eval.source_history_fact_union import (
    FactLane,
    HydratedSourceHistory,
    PostMapFactUnion,
    build_post_map_fact_union,
    plan_source_history_hydration,
)
from tools.matched_eval.source_history_mapper_live import (
    HARD_CONTEXT_TOKEN_CAP,
    MAPPER_CONTRACT_SHA256,
    MAX_PROMPT_TOKENS as MAPPER_MAX_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE as MAPPER_OUTPUT_TOKEN_RESERVE,
    SourceHistoryMapperError,
    SourceMapperCachedCompletion,
    SourceMapperMaterialization,
    SourceMapperProviderJournal,
    WorkDisposition,
    build_source_history_mapper_preflight,
    materialize_source_history_mapper,
)


FORMAT = "memory-condense-confirmation-adaptive-evidence-tail-v1"
SOLVER_PREFLIGHT_FORMAT = f"{FORMAT}-solver-preflight-v1"
TAIL_PREFLIGHT_FORMAT = f"{FORMAT}-tail-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
SOLVER_RUN_FORMAT = f"{FORMAT}-solver-run-v1"
SOLVER_REPLAY_FORMAT = f"{FORMAT}-solver-replay-v1"
TAIL_RUN_FORMAT = f"{FORMAT}-tail-run-v1"
TAIL_REPLAY_FORMAT = f"{FORMAT}-tail-replay-v1"

SOLVER_PREFLIGHT_NAME = "confirmation-adaptive-evidence-preflight-v1.json"
SOLVER_RELEASE_NAME = "confirmation-adaptive-evidence-provider-release-v1.json"
SOLVER_RUN_NAME = "confirmation-adaptive-evidence-run-v1.json"
SOLVER_REPLAY_NAME = "confirmation-adaptive-evidence-replay-v1.json"
TAIL_WORK_MANIFEST_NAME = "confirmation-adaptive-tail-work-manifest-v1.json"
TAIL_PREFLIGHT_NAME = "confirmation-adaptive-tail-preflight-v1.json"
TAIL_RELEASE_NAME = "confirmation-adaptive-tail-provider-release-v1.json"
TAIL_RUN_NAME = "confirmation-adaptive-tail-run-v1.json"
TAIL_REPLAY_NAME = "confirmation-adaptive-tail-replay-v1.json"

SOLVER_CHECKPOINT_DIR_NAME = "terra-adaptive-evidence-solver-v3-calls"
TAIL_CHECKPOINT_DIR_NAME = "terra-source-history-tail-calls"
DEFAULT_MAX_NEW_TAIL_CALLS = tail_core.MAX_NEW_PROVIDER_CALLS

_JOURNAL_NAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_RELEASE_KEYS = frozenset(
    {
        "approval_opt_in",
        "checkpoint_namespace",
        "checkpoint_snapshot",
        "format",
        "gold_loaded",
        "output_root",
        "output_root_sha256",
        "physical_provider_calls",
        "preflight_sha256",
        "release_identity_sha256",
        "release_status",
        "required_authorized_provider_calls",
        "stage",
        "unsafe_retry_policy",
    }
)
_SNAPSHOT_KEYS = frozenset(
    {"authenticated_complete_count", "ordered_records", "ordered_records_sha256"}
)
_RECORD_KEYS = frozenset(
    {
        "call_key_sha256",
        "messages_sha256",
        "request_journal_sha256",
        "response_journal_sha256",
    }
)


class ConfirmationAdaptiveTailError(MatchedEvalContractError):
    """An adaptive solver, tail, release, or replay invariant failed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationAdaptiveTailError(message)


def _plain_messages(messages: Sequence[Any]) -> tuple[dict[str, str], ...]:
    return tuple({"role": row.role, "content": row.content} for row in messages)


@dataclass(frozen=True, slots=True)
class ConfirmationAdaptiveUpstream:
    """Authenticated exact objects emitted by map, source-stream, and base stages."""

    base_preflight_artifact: SealedArtifact
    base_work_manifest_artifact: SealedArtifact
    base_materialization_artifact: SealedArtifact
    base_replay_artifact: SealedArtifact
    base_questions: tuple[source_cli.FastMaterializationQuestionPlan, ...]
    base_materializations: tuple[SourceMapperMaterialization, ...]
    base_completion_batch: FastCompletionBatch | None
    map_plan: EvidenceMapPlan
    map_plane: VerifiedEvidenceMapPlane
    source_population: LockedSourceGateAdapterPopulation
    repack_source_population: LockedSourceGateAdapterPopulation
    source_stream_plane_artifact: SealedArtifact

    def __post_init__(self) -> None:
        for artifact, label in (
            (self.base_preflight_artifact, "base preflight"),
            (self.base_work_manifest_artifact, "base work manifest"),
            (self.base_materialization_artifact, "base materialization"),
            (self.base_replay_artifact, "base replay"),
            (self.source_stream_plane_artifact, "source-stream plane"),
        ):
            _require(type(artifact) is SealedArtifact, f"{label} is not exact")
            require_sha256(artifact.sha256, f"{label} SHA-256")
        _require(type(self.map_plan) is EvidenceMapPlan, "map plan is not exact")
        _require(
            type(self.map_plane) is VerifiedEvidenceMapPlane,
            "map plane is not exact",
        )
        _require(
            type(self.source_population) is LockedSourceGateAdapterPopulation
            and type(self.repack_source_population)
            is LockedSourceGateAdapterPopulation,
            "source populations are not exact",
        )
        _require(
            type(self.base_questions) is tuple
            and type(self.base_materializations) is tuple
            and len(self.base_questions) == len(self.base_materializations)
            and all(
                type(row) is source_cli.FastMaterializationQuestionPlan
                for row in self.base_questions
            )
            and all(
                type(row) is SourceMapperMaterialization
                for row in self.base_materializations
            ),
            "base typed materialization population changed",
        )
        _require(
            self.base_completion_batch is None
            or type(self.base_completion_batch) is FastCompletionBatch,
            "base completion batch is not exact",
        )
        _require(
            self.base_replay_artifact.payload.get("byte_identical") is True,
            "base source map is not replay-verified",
        )
        materialized_preflight = self.base_materialization_artifact.payload.get(
            "preflight_artifact_sha256"
        )
        _require(
            materialized_preflight in {None, self.base_preflight_artifact.sha256},
            "base materialization escaped its preflight",
        )
        replay_expected = self.base_replay_artifact.payload.get(
            "expected_materialization_sha256"
        )
        _require(
            replay_expected in {None, self.base_materialization_artifact.sha256},
            "base replay escaped its materialization",
        )
        _validate_source_populations(
            self.source_population, self.repack_source_population
        )
        base_receipt = self.base_preflight_artifact.payload.get(
            "source_gate_population_receipt_sha256"
        )
        _require(
            base_receipt in {None, self.source_population.receipt_sha256},
            "base preflight escaped the D1/P0/G1 source population",
        )


@dataclass(frozen=True, slots=True)
class ConfirmationAdaptiveEvidencePlan:
    upstream: ConfirmationAdaptiveUpstream
    fact_unions: tuple[tuple[str, PostMapFactUnion], ...]
    plan: AdaptiveEvidenceSolverPlan
    preflight: AdaptiveEvidenceSolverPreflight

    @property
    def required_calls(self) -> int:
        return self.plan.required_calls

    @property
    def fact_union_map(self) -> dict[str, PostMapFactUnion]:
        return dict(self.fact_unions)


@dataclass(frozen=True, slots=True)
class ConfirmationAdaptiveTailPlan:
    upstream: ConfirmationAdaptiveUpstream
    base_fact_unions: tuple[tuple[str, PostMapFactUnion], ...]
    hydration_batches: tuple[source_cli.NamespaceHydrationBatch, ...]
    decisions: tuple[tail_core.TailQuestionDecision, ...]
    questions: tuple[tail_core.TailQuestionWork, ...]
    provider_population: FastPromptPopulation | None
    max_new_provider_calls: int

    @property
    def required_calls(self) -> int:
        return 0 if self.provider_population is None else self.provider_population.unique_prompt_count

    @property
    def all_prompt_rows(self) -> tuple[Any, ...]:
        return tuple(
            prompt
            for question in self.questions
            for prompt in question.mapper_preflight.prompt_rows
        )

    @property
    def submitted_prompt_rows(self) -> tuple[Any, ...]:
        return tuple(
            row
            for row in self.all_prompt_rows
            if row.disposition is WorkDisposition.NEW_CALL
        )


@dataclass(frozen=True, slots=True)
class ConfirmationAdaptiveEvidencePreflight:
    plan: ConfirmationAdaptiveEvidencePlan
    artifact: SealedArtifact


@dataclass(frozen=True, slots=True)
class ConfirmationAdaptiveTailPreflight:
    plan: ConfirmationAdaptiveTailPlan
    work_manifest_artifact: SealedArtifact
    artifact: SealedArtifact


@dataclass(frozen=True, slots=True)
class ConfirmationProviderExecution:
    stage: Literal["solver", "tail"]
    batch: FastCompletionBatch | None
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class VerifiedConfirmationAdaptiveEvidencePlane:
    plan: ConfirmationAdaptiveEvidencePlan
    preflight_artifact: SealedArtifact
    release_artifact: SealedArtifact
    run_artifact: SealedArtifact
    replay_artifact: SealedArtifact
    completion_batch: FastCompletionBatch | None
    completion_plane: AdaptiveSolverCompletionPlane
    run: AdaptiveEvidenceSolverRun
    plane: VerifiedAdaptiveEvidenceSolverPlane


@dataclass(frozen=True, slots=True)
class VerifiedConfirmationAdaptiveTailPlane:
    plan: ConfirmationAdaptiveTailPlan
    preflight_artifact: SealedArtifact
    work_manifest_artifact: SealedArtifact
    release_artifact: SealedArtifact
    run_artifact: SealedArtifact
    replay_artifact: SealedArtifact
    completion_batch: FastCompletionBatch | None
    questions: tuple[source_cli.FastMaterializationQuestionPlan, ...]
    materializations: tuple[SourceMapperMaterialization, ...]
    fact_union_rows: tuple[TailFactUnionRow, ...]
    decisions: tuple[tail_core.TailQuestionDecision, ...]


TailHydrator = Callable[
    [Sequence[tuple[Any, SourceGateRound]]],
    tuple[
        tuple[source_cli.NamespaceHydrationBatch, ...],
        Mapping[tuple[str, str], HydratedSourceHistory],
    ],
]
ClientFactory = Callable[[str, str], Any]


def confirmation_adaptive_upstream(
    source_streams: ConfirmationSourceStreamsResult,
    source_map: VerifiedConfirmationAdaptiveSourceMapPlane,
    map_plan: EvidenceMapPlan,
    map_plane: VerifiedEvidenceMapPlane,
) -> ConfirmationAdaptiveUpstream:
    """Join exact replayed upstream carriers without reopening any dataset.

    This is the normal in-process confirmation pipeline seam.  The lower-level
    dataclass remains public for focused synthetic tests and authenticated
    orchestration that already owns the same exact objects.
    """

    if type(source_streams) is not ConfirmationSourceStreamsResult:
        raise TypeError("source_streams must be an exact ConfirmationSourceStreamsResult")
    if type(source_map) is not VerifiedConfirmationAdaptiveSourceMapPlane:
        raise TypeError(
            "source_map must be an exact VerifiedConfirmationAdaptiveSourceMapPlane"
        )
    if type(map_plan) is not EvidenceMapPlan:
        raise TypeError("map_plan must be an exact EvidenceMapPlan")
    if type(map_plane) is not VerifiedEvidenceMapPlane:
        raise TypeError("map_plane must be an exact VerifiedEvidenceMapPlane")
    _require(
        source_map.source_population is source_streams.base_population
        or source_map.source_population == source_streams.base_population,
        "source map escaped the authenticated base source-stream population",
    )
    _require(
        source_map.query_adapter is source_streams.query_map_adapter
        or source_map.query_adapter == source_streams.query_map_adapter,
        "source map escaped the authenticated query-map adapter",
    )
    _require(
        tuple(row.question_id for row in map_plane.rows)
        == tuple(
            row.direct_plan_row.adapter.source.packet.question_id
            for row in map_plan.rows
        )
        and tuple(row.map_plan_row_receipt_sha256 for row in map_plane.rows)
        == tuple(row.receipt_sha256 for row in map_plan.rows),
        "evidence-map replay escaped its exact plan/question order",
    )
    return ConfirmationAdaptiveUpstream(
        source_map.preflight_artifact,
        source_map.work_manifest_artifact,
        source_map.materialization_artifact,
        source_map.replay_artifact,
        source_map.questions,
        source_map.materializations,
        source_map.completion_batch,
        map_plan,
        map_plane,
        source_streams.base_population,
        source_streams.repack_population,
        source_streams.plane_artifact,
    )


def _validate_source_populations(
    source: LockedSourceGateAdapterPopulation,
    repack: LockedSourceGateAdapterPopulation,
) -> None:
    expected = source_cli.source_gate_policy(1, 0, 1)
    _require(
        source.direct_stream_profile == DIRECT_STREAM_PROFILE_V1
        and repack.direct_stream_profile == DIRECT_STREAM_PROFILE_REPACK_V2,
        "confirmation tail requires V1 base and repack-V2 direct streams",
    )
    source_ids = tuple(row.plan.question_id for row in source.questions)
    repack_ids = tuple(row.plan.question_id for row in repack.questions)
    _require(
        source_ids == repack_ids and len(set(source_ids)) == len(source_ids),
        "base and repack source populations changed question order",
    )
    for base, deep in zip(source.questions, repack.questions, strict=True):
        _require(
            base.ordinal == deep.ordinal
            and base.plan.parent == deep.plan.parent
            and base.plan.activation == deep.plan.activation
            and base.plan.policy == deep.plan.policy == expected,
            "source populations escaped frozen D1/P0/G1 parent/policy",
        )


def _base_fact_unions(
    upstream: ConfirmationAdaptiveUpstream,
) -> tuple[tuple[str, PostMapFactUnion], ...]:
    result: list[tuple[str, PostMapFactUnion]] = []
    for question, materialization in zip(
        upstream.base_questions, upstream.base_materializations, strict=True
    ):
        _require(
            materialization.hydration_plan_receipt_sha256
            == question.hydration_plan.receipt_sha256
            and materialization.mapping_plan_receipt_sha256
            == question.mapping_plan.receipt_sha256
            and materialization.preflight_receipt_sha256
            == question.mapper_preflight.receipt_sha256,
            "base mapper materialization escaped its exact question plan",
        )
        result.append(
            (
                question.question_id,
                build_post_map_fact_union(
                    question.hydration_plan,
                    batches=materialization.batches,
                    direct_evidence=question.direct_evidence,
                ),
            )
        )
    _require(
        len({key for key, _value in result}) == len(result),
        "base fact-union question IDs repeat",
    )
    return tuple(result)


def build_confirmation_adaptive_evidence_plan(
    upstream: ConfirmationAdaptiveUpstream,
) -> ConfirmationAdaptiveEvidencePlan:
    """Build the exact historical adaptive solver over base post-map facts."""

    if type(upstream) is not ConfirmationAdaptiveUpstream:
        raise TypeError("upstream must be an exact ConfirmationAdaptiveUpstream")
    unions = _base_fact_unions(upstream)
    plan = build_adaptive_evidence_solver_plan(
        upstream.map_plan,
        upstream.map_plane,
        source_fact_unions=dict(unions),
    )
    preflight = preflight_adaptive_evidence_solver(plan)
    return ConfirmationAdaptiveEvidencePlan(upstream, unions, plan, preflight)


def _base_cache(
    upstream: ConfirmationAdaptiveUpstream,
) -> dict[str, SourceMapperCachedCompletion]:
    if not upstream.base_questions:
        return {}
    _require(
        type(upstream.base_completion_batch) is FastCompletionBatch,
        "base mapper cache requires its exact completion batch",
    )
    assert upstream.base_completion_batch is not None
    return tail_core._base_cache(  # noqa: SLF001 - authoritative cache seam
        upstream.base_questions,
        upstream.base_materializations,
        upstream.base_completion_batch,
    )


def _tail_decision(
    question: Any,
    base_round: SourceGateRound,
    coverage: ObligationCoverageReceipt,
    union: PostMapFactUnion,
    disposition: tail_core.TailDisposition,
    *,
    tail_round: SourceGateRound | None = None,
    tail_plan: Any | None = None,
    direct_stream_profile: str | None = None,
) -> tail_core.TailQuestionDecision:
    return tail_core._decision(  # noqa: SLF001 - preserves frozen decision receipt
        question,
        base_round,
        coverage,
        union,
        disposition,
        tail_round=tail_round,
        tail_plan=tail_plan,
        direct_stream_profile=direct_stream_profile,
    )


def build_confirmation_adaptive_tail_plan(
    upstream: ConfirmationAdaptiveUpstream,
    *,
    max_new_provider_calls: int = DEFAULT_MAX_NEW_TAIL_CALLS,
    hydrator: TailHydrator = source_cli.hydrate_namespace_batches,
) -> ConfirmationAdaptiveTailPlan:
    """Select one deeper source for each unresolved zero-fact base row.

    Selection is performed on logical method-local candidates before physical
    mapper work is deduplicated.  The exact frozen lane order, D1/P0/G1 base,
    consolidated obligations, state-chain direct authority, and deep direct
    repack-V2 rule are inherited from the historical cores.
    """

    if type(upstream) is not ConfirmationAdaptiveUpstream:
        raise TypeError("upstream must be an exact ConfirmationAdaptiveUpstream")
    _require(
        type(max_new_provider_calls) is int
        and 0 <= max_new_provider_calls <= tail_core.MAX_NEW_PROVIDER_CALLS,
        "tail provider budget must be within 0..128",
    )
    _require(callable(hydrator), "tail hydrator is not callable")
    unions = _base_fact_unions(upstream)
    union_by_id = dict(unions)
    base_by_id = {row.question_id: row for row in upstream.base_questions}
    materialization_by_id = {
        question.question_id: result
        for question, result in zip(
            upstream.base_questions,
            upstream.base_materializations,
            strict=True,
        )
    }
    repack_by_id = {
        row.plan.question_id: row
        for row in upstream.repack_source_population.questions
    }
    cache_by_work = _base_cache(upstream)
    decisions: list[tail_core.TailQuestionDecision] = []
    selected: list[tuple[Any, SourceGateRound, tail_core.TailQuestionDecision]] = []

    for question in upstream.source_population.questions:
        gate = question.plan
        base_question = base_by_id.get(gate.question_id)
        base_result = materialization_by_id.get(gate.question_id)
        union = union_by_id.get(gate.question_id)
        _require(
            base_question is not None and base_result is not None and union is not None,
            "tail source row escaped the replayed base materialization",
        )
        assert base_question is not None and base_result is not None and union is not None
        base_round = start_source_gate(gate)
        _require(
            base_round.receipt_sha256
            == base_question.mapping_plan.gate_round_receipt_sha256
            and base_round.selections == base_question.hydration_plan.selections,
            "tail base round changed its source-map selection binding",
        )
        coverage = assess_obligation_coverage(
            gate,
            base_round,
            coverage_facts_from_fact_union(union),
            mapping_plan_receipt_sha256s=(base_question.mapping_plan.receipt_sha256,),
            cumulative_physical_work_call_ids=tuple(
                row.physical_work_id for row in base_result.work_results
            ),
            pending_physical_work_ids=base_result.deferred_work_ids,
        )
        if union.retained_facts:
            decisions.append(
                _tail_decision(
                    question,
                    base_round,
                    coverage,
                    union,
                    tail_core.TailDisposition.PENDING_SOLVER,
                )
            )
            continue
        if coverage.all_satisfied:
            decisions.append(
                _tail_decision(
                    question,
                    base_round,
                    coverage,
                    union,
                    tail_core.TailDisposition.SATISFIED,
                )
            )
            continue
        repack_question = repack_by_id.get(gate.question_id)
        _require(repack_question is not None, "tail lost its repack-V2 row")
        assert repack_question is not None
        choice = tail_core.select_one_tail_candidate(
            gate,
            base_round,
            direct_plan=repack_question.plan,
        )
        if choice is None:
            decisions.append(
                _tail_decision(
                    question,
                    base_round,
                    coverage,
                    union,
                    tail_core.TailDisposition.EXHAUSTED,
                )
            )
            continue
        tail_plan, lane, candidate, profile = choice
        tail_round = tail_core._tail_round(  # noqa: SLF001 - frozen round semantics
            tail_plan, base_round, lane, candidate
        )
        hydration_question = repack_question if lane is FactLane.DIRECT else question
        decision = _tail_decision(
            question,
            base_round,
            coverage,
            union,
            tail_core.TailDisposition.SELECTED,
            tail_round=tail_round,
            tail_plan=tail_plan,
            direct_stream_profile=profile,
        )
        decisions.append(decision)
        selected.append((hydration_question, tail_round, decision))

    hydration_batches, supplied_histories = (
        hydrator(tuple((question, round_) for question, round_, _ in selected))
        if selected
        else ((), {})
    )
    histories_by_key = dict(supplied_histories)
    remaining = max_new_provider_calls
    work_rows: list[tail_core.TailQuestionWork] = []
    for question, tail_round, decision in selected:
        base_question = base_by_id[question.plan.question_id]
        hydration_input = question.hydration_input(tail_round)
        histories = tuple(
            histories_by_key[(hydration_input.namespace_id, row.source_id)]
            for row in hydration_input.memberships
        )
        _require(bool(histories), "tail selected source has no hydrated history")
        base_work_ids = tuple(row.work_id for row in base_question.mapping_plan.work_items)
        base_cap = base_question.hydration_plan.max_window_tokens
        largest_chunk = max(
            chunk.token_count
            for history in histories
            for chunk in history.chunks
            if not chunk.metadata_chunk
        )
        candidate_caps = tuple(
            dict.fromkeys(
                value
                for value in (
                    base_cap,
                    min(base_cap, 4_800),
                    min(base_cap, 4_000),
                    min(base_cap, 3_200),
                    min(base_cap, 2_400),
                    min(base_cap, 1_600),
                    min(base_cap, 800),
                    largest_chunk,
                )
                if value >= largest_chunk
            )
        )
        last_overflow: SourceHistoryMapperError | None = None
        for cap in candidate_caps:
            hydration = plan_source_history_hydration(
                question.plan.parent,
                selections=tail_round.selections,
                histories=histories,
                max_window_tokens=cap,
            )
            raw_mapping = build_question_bound_mapping_plan(
                question.plan,
                tail_round,
                hydration,
                mapper_contract_sha256=MAPPER_CONTRACT_SHA256,
                cached_work_ids=base_work_ids,
                prior_call_work_ids=tuple(
                    row.physical_work_id
                    for row in materialization_by_id[question.plan.question_id].work_results
                ),
            )
            mapping, next_remaining = tail_core.cap_mapping_plan_new_calls(
                raw_mapping, remaining
            )
            try:
                mapper_preflight = build_source_history_mapper_preflight(
                    hydration, mapping
                )
            except SourceHistoryMapperError as exc:
                if "envelope overflow" not in str(exc):
                    raise
                last_overflow = exc
                continue
            remaining = next_remaining
            break
        else:
            assert last_overflow is not None
            raise last_overflow
        _require(
            mapper_preflight.maximum_combined_token_proxy <= HARD_CONTEXT_TOKEN_CAP,
            "tail mapper prompt escaped the hard context envelope",
        )
        caches = tuple(cache_by_work[work_id] for work_id in mapping.reused_work_ids)
        work_rows.append(
            tail_core.TailQuestionWork(
                question.ordinal,
                question.plan.question_id,
                decision,
                tail_round,
                hydration,
                mapping,
                mapper_preflight,
                caches,
                base_cap,
            )
        )

    submitted = tuple(
        _plain_messages(row.messages)
        for question in work_rows
        for row in question.mapper_preflight.prompt_rows
        if row.disposition is WorkDisposition.NEW_CALL
    )
    population = (
        preflight_fast_completion_prompts(
            submitted, max_prompt_tokens=MAPPER_MAX_PROMPT_TOKENS
        )
        if submitted
        else None
    )
    if population is not None:
        _require(
            population.logical_prompt_count
            == population.unique_prompt_count
            == sum(row.mapping_plan.planned_provider_calls for row in work_rows)
            <= max_new_provider_calls,
            "tail prompt population escaped exact physical work/call budget",
        )
    _require(
        len(decisions) == len(upstream.source_population.questions)
        and tuple(row.question_id for row in decisions)
        == tuple(row.plan.question_id for row in upstream.source_population.questions),
        "tail decisions did not preserve every activated source row",
    )
    return ConfirmationAdaptiveTailPlan(
        upstream,
        unions,
        tuple(hydration_batches),
        tuple(decisions),
        tuple(work_rows),
        population,
        max_new_provider_calls,
    )


def _solver_prompt_rows(plan: ConfirmationAdaptiveEvidencePlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in plan.plan.submitted_rows:
        assert row.messages is not None
        assert row.messages_sha256 is not None
        assert row.prompt_id is not None
        assert row.prompt_token_proxy is not None
        messages = list(_plain_messages(row.messages))
        _require(
            identity_sha256(messages) == row.messages_sha256,
            "adaptive solver messages changed",
        )
        rows.append(
            {
                "messages": messages,
                "messages_sha256": row.messages_sha256,
                "ordinal": row.ordinal,
                "prompt_id": row.prompt_id,
                "prompt_token_proxy": row.prompt_token_proxy,
                "question_id": row.question_id,
            }
        )
    return rows


def _solver_preflight_payload(
    plan: ConfirmationAdaptiveEvidencePlan,
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    rows = _solver_prompt_rows(plan)
    payload = {
        "adaptive_solver_preflight": plan.preflight.projection(),
        "adaptive_solver_preflight_receipt_sha256": plan.preflight.receipt_sha256,
        "arm_label": SOLVER_ARM_LABEL,
        "base_materialization_sha256": plan.upstream.base_materialization_artifact.sha256,
        "base_replay_sha256": plan.upstream.base_replay_artifact.sha256,
        "format": SOLVER_PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "map_plan_identity_sha256": plan.upstream.map_plan.plan_identity_sha256,
        "map_replay_sha256": plan.upstream.map_plane.replay_sha256,
        "map_run_sha256": plan.upstream.map_plane.run_sha256,
        "max_concurrency": max_concurrency,
        "model": model,
        "no_op_question_count": len(plan.plan.rows) - len(rows),
        "ordered_plan_row_receipt_sha256s": [
            row.receipt_sha256 for row in plan.plan.rows
        ],
        "output_token_reserve": SOLVER_OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": rows,
        "provider_calls": 0,
        "question_count": len(plan.plan.rows),
        "required_authorized_provider_calls": plan.required_calls,
        "retained_transformer_token_state_bytes": 0,
        "source_fact_union_receipt_sha256s": [
            value.receipt_sha256 for _key, value in plan.fact_unions
        ],
        "source_stream_plane_sha256": plan.upstream.source_stream_plane_artifact.sha256,
        "status": "preflighted",
    }
    assert_gold_blind(payload, path="confirmation_adaptive_evidence_preflight")
    return payload


def publish_confirmation_adaptive_evidence_preflight(
    plan: ConfirmationAdaptiveEvidencePlan,
    *,
    output_root: str | Path,
    model: str = provider_runtime.DEFAULT_TERRA_GATEWAY_MODEL,
    gateway_url: str = provider_runtime.DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> ConfirmationAdaptiveEvidencePreflight:
    if type(plan) is not ConfirmationAdaptiveEvidencePlan:
        raise TypeError("plan must be an exact ConfirmationAdaptiveEvidencePlan")
    _require(type(max_concurrency) is int and max_concurrency > 0, "bad concurrency")
    payload = _solver_preflight_payload(
        plan,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    artifact, _created = publish_sealed_json(
        Path(output_root) / SOLVER_PREFLIGHT_NAME, payload
    )
    return ConfirmationAdaptiveEvidencePreflight(plan, artifact)


def _tail_as_base_plan(
    plan: ConfirmationAdaptiveTailPlan,
) -> source_cli.LockedAdaptiveBasePlan:
    questions = tuple(
        source_cli.BaseQuestionSourceMap(
            row.ordinal,
            row.question_id,
            row.gate_round,
            row.hydration_plan,
            row.mapping_plan,
            row.mapper_preflight,
        )
        for row in plan.questions
    )
    routes = Counter(row.decision.route for row in plan.questions)
    # Work-manifest serialization does not inspect provider_population.  The
    # annotation is non-optional historically, but an empty confirmation tail
    # is a legitimate typed no-op and must remain representable.
    return source_cli.LockedAdaptiveBasePlan(
        None,
        plan.upstream.source_population,
        plan.hydration_batches,
        questions,
        tuple(sorted(routes.items())),
        plan.provider_population,  # type: ignore[arg-type]
    )


def _tail_preflight_payload(
    plan: ConfirmationAdaptiveTailPlan,
    *,
    work_manifest_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    prompts = [row.projection(include_messages=True) for row in plan.all_prompt_rows]
    selected = tuple(
        row
        for row in plan.decisions
        if row.disposition is tail_core.TailDisposition.SELECTED
    )
    payload = {
        "base_materialization_sha256": plan.upstream.base_materialization_artifact.sha256,
        "base_preflight_sha256": plan.upstream.base_preflight_artifact.sha256,
        "base_replay_sha256": plan.upstream.base_replay_artifact.sha256,
        "base_work_manifest_sha256": plan.upstream.base_work_manifest_artifact.sha256,
        "direct_repack_min_rank": tail_core.DIRECT_REPACK_MIN_RANK,
        "direct_stream_profile": DIRECT_STREAM_PROFILE_V1,
        "format": TAIL_PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "logical_selection_dedup_performed": False,
        "logical_source_selected_before_physical_dedup": True,
        "mapper_contract_sha256": MAPPER_CONTRACT_SHA256,
        "max_concurrency": max_concurrency,
        "max_new_provider_calls": plan.max_new_provider_calls,
        "model": model,
        "obligation_compilation_mode": CONSOLIDATED_OBLIGATION_MODE,
        "ordered_question_decisions": [row.projection() for row in plan.decisions],
        "output_token_reserve": MAPPER_OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": prompts,
        "post_map_dedup_performed": False,
        "provider_calls": 0,
        "provider_population": (
            None if plan.provider_population is None else plan.provider_population.model_dump()
        ),
        "question_count": len(plan.decisions),
        "repack_source_gate_population_receipt_sha256": (
            plan.upstream.repack_source_population.receipt_sha256
        ),
        "repack_v2_selected_question_count": sum(
            row.selected_direct_stream_profile == DIRECT_STREAM_PROFILE_REPACK_V2
            for row in selected
        ),
        "required_authorized_provider_calls": plan.required_calls,
        "retained_transformer_token_state_bytes": 0,
        "selection_rule": "unresolved_and_zero_retained_base_facts_only",
        "source_gate_policy_receipt_sha256": source_cli.source_gate_policy(1, 0, 1).receipt_sha256,
        "source_gate_population_receipt_sha256": plan.upstream.source_population.receipt_sha256,
        "source_stream_plane_sha256": plan.upstream.source_stream_plane_artifact.sha256,
        "state_chain_profile": STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
        "status": "preflighted",
        "work_manifest_sha256": require_sha256(
            work_manifest_sha256, "tail work manifest"
        ),
        "zero_new_fact_selected_question_count": len(selected),
    }
    assert_gold_blind(payload, path="confirmation_adaptive_tail_preflight")
    return payload


def publish_confirmation_adaptive_tail_preflight(
    plan: ConfirmationAdaptiveTailPlan,
    *,
    output_root: str | Path,
    model: str = provider_runtime.DEFAULT_TERRA_GATEWAY_MODEL,
    gateway_url: str = provider_runtime.DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> ConfirmationAdaptiveTailPreflight:
    if type(plan) is not ConfirmationAdaptiveTailPlan:
        raise TypeError("plan must be an exact ConfirmationAdaptiveTailPlan")
    root = Path(output_root)
    work, _created = publish_sealed_json(
        root / TAIL_WORK_MANIFEST_NAME,
        source_cli.work_manifest_projection(_tail_as_base_plan(plan)),
    )
    payload = _tail_preflight_payload(
        plan,
        work_manifest_sha256=work.sha256,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    artifact, _created = publish_sealed_json(root / TAIL_PREFLIGHT_NAME, payload)
    return ConfirmationAdaptiveTailPreflight(plan, work, artifact)


def _stage_config(
    preflight: ConfirmationAdaptiveEvidencePreflight | ConfirmationAdaptiveTailPreflight,
) -> tuple[str, str, str, int, int, list[dict[str, Any]]]:
    if type(preflight) is ConfirmationAdaptiveEvidencePreflight:
        return (
            "solver",
            SOLVER_CHECKPOINT_DIR_NAME,
            SOLVER_RELEASE_NAME,
            SOLVER_MAX_PROMPT_TOKENS,
            SOLVER_OUTPUT_TOKEN_RESERVE,
            _solver_prompt_rows(preflight.plan),
        )
    if type(preflight) is ConfirmationAdaptiveTailPreflight:
        rows = [
            {
                "messages": list(_plain_messages(row.messages)),
                "messages_sha256": row.messages_sha256,
            }
            for row in preflight.plan.submitted_prompt_rows
        ]
        return (
            "tail",
            TAIL_CHECKPOINT_DIR_NAME,
            TAIL_RELEASE_NAME,
            MAPPER_MAX_PROMPT_TOKENS,
            MAPPER_OUTPUT_TOKEN_RESERVE,
            rows,
        )
    raise TypeError("preflight has an unsupported exact type")


def _verify_preflight(
    preflight: ConfirmationAdaptiveEvidencePreflight | ConfirmationAdaptiveTailPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
) -> SealedArtifact:
    expected = require_sha256(expected_preflight_sha256, "adaptive preflight")
    if type(preflight) is ConfirmationAdaptiveEvidencePreflight:
        name = SOLVER_PREFLIGHT_NAME
        rebuilt = _solver_preflight_payload(
            preflight.plan,
            model=str(preflight.artifact.payload["model"]),
            gateway_url=str(preflight.artifact.payload["gateway_url"]),
            max_concurrency=int(preflight.artifact.payload["max_concurrency"]),
        )
    elif type(preflight) is ConfirmationAdaptiveTailPreflight:
        name = TAIL_PREFLIGHT_NAME
        work = read_sealed_json(Path(output_root) / TAIL_WORK_MANIFEST_NAME)
        _require(
            work.sha256 == preflight.work_manifest_artifact.sha256,
            "tail work manifest changed",
        )
        rebuilt = _tail_preflight_payload(
            preflight.plan,
            work_manifest_sha256=work.sha256,
            model=str(preflight.artifact.payload["model"]),
            gateway_url=str(preflight.artifact.payload["gateway_url"]),
            max_concurrency=int(preflight.artifact.payload["max_concurrency"]),
        )
    else:
        raise TypeError("preflight has an unsupported exact type")
    artifact = read_sealed_json(Path(output_root) / name)
    _require(
        artifact.sha256 == preflight.artifact.sha256 == expected
        and artifact.payload == rebuilt,
        "adaptive preflight differs from exact typed parents",
    )
    return artifact


def _runtime(
    preflight_artifact: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
    stage: str,
    checkpoint_name: str,
    max_prompt_tokens: int,
    output_token_reserve: int,
    client: Any | None,
) -> FastCompletionRuntime:
    prompts = tuple(tuple(dict(message) for message in row["messages"]) for row in prompt_rows)
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / checkpoint_name,
        prompt_population=prompts,
        model=str(preflight_artifact.payload["model"]),
        client=client,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=output_token_reserve,
        max_concurrency=int(preflight_artifact.payload["max_concurrency"]),
        retries=0,
        benchmark_provenance={
            "arm": f"confirmation_adaptive_{stage}_v1",
            "gateway_url": preflight_artifact.payload["gateway_url"],
            "gold_loaded": False,
            "preflight_artifact_sha256": preflight_artifact.sha256,
            "source_stream_plane_sha256": preflight_artifact.payload[
                "source_stream_plane_sha256"
            ],
        },
    )


def _checkpoint_records(
    preflight: ConfirmationAdaptiveEvidencePreflight | ConfirmationAdaptiveTailPreflight,
    *,
    output_root: str | Path,
    preflight_artifact: SealedArtifact,
) -> tuple[dict[str, str], ...]:
    stage, checkpoint_name, _release, max_tokens, reserve, rows = _stage_config(preflight)
    checkpoint = Path(output_root) / checkpoint_name
    if not checkpoint.exists():
        return ()
    _require(
        checkpoint.is_dir() and not checkpoint.is_symlink(),
        "adaptive checkpoint root is unsafe",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in checkpoint.iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            "adaptive checkpoint contains unsafe state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_NAME.fullmatch(path.name)
        _require(match is not None, "adaptive checkpoint contains foreign state")
        assert match is not None
        (requests if match.group("kind") == "request" else responses).add(
            match.group("key")
        )
    _require(
        requests == responses,
        "adaptive request/response pair is incomplete; unsafe retry forbidden",
    )
    if not requests:
        return ()
    runtime = _runtime(
        preflight_artifact,
        rows,
        output_root=output_root,
        stage=stage,
        checkpoint_name=checkpoint_name,
        max_prompt_tokens=max_tokens,
        output_token_reserve=reserve,
        client=None,
    )
    try:
        with runtime._journal_guard():  # noqa: SLF001 - read-only authentication
            records = runtime._load_all_records()  # noqa: SLF001
        call_keys = dict(runtime._call_keys)  # noqa: SLF001
    finally:
        runtime.close()
    _require(len(records) == len(requests), "adaptive checkpoint population changed")
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        messages_sha = require_sha256(row["messages_sha256"], "adaptive messages")
        if messages_sha in seen:
            continue
        record = records.get(messages_sha)
        if record is None:
            continue
        _require(
            record.call_key_sha256 == call_keys[messages_sha],
            "adaptive checkpoint call key changed",
        )
        ordered.append(
            {
                "call_key_sha256": record.call_key_sha256,
                "messages_sha256": record.messages_sha256,
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
            }
        )
        seen.add(messages_sha)
    _require(len(ordered) == len(requests), "adaptive checkpoint order changed")
    return tuple(ordered)


def approve_confirmation_adaptive_release(
    preflight: ConfirmationAdaptiveEvidencePreflight | ConfirmationAdaptiveTailPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    approve_provider_release: bool,
    authorized_provider_calls: int,
) -> SealedArtifact:
    """Seal approval for exactly the currently missing native journals."""

    _require(approve_provider_release is True, "provider release requires approval")
    artifact = _verify_preflight(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
    )
    stage, checkpoint_name, release_name, _cap, _reserve, rows = _stage_config(preflight)
    records = _checkpoint_records(
        preflight, output_root=output_root, preflight_artifact=artifact
    )
    remaining = len(rows) - len(records)
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "adaptive release authorization must equal exact remaining calls",
    )
    root = Path(output_root).resolve().as_posix()
    body = {
        "approval_opt_in": True,
        "checkpoint_namespace": checkpoint_name,
        "checkpoint_snapshot": {
            "authenticated_complete_count": len(records),
            "ordered_records": list(records),
            "ordered_records_sha256": identity_sha256(list(records)),
        },
        "format": RELEASE_FORMAT,
        "gold_loaded": False,
        "output_root": root,
        "output_root_sha256": identity_sha256({"canonical_root": root}),
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": remaining,
        "stage": stage,
        "unsafe_retry_policy": "refuse-incomplete-request-response-pair-v1",
    }
    assert_gold_blind(body, path="confirmation_adaptive_provider_release")
    payload = {**body, "release_identity_sha256": identity_sha256(body)}
    release, _created = publish_sealed_json(Path(output_root) / release_name, payload)
    return release


def _verify_release(
    preflight: ConfirmationAdaptiveEvidencePreflight | ConfirmationAdaptiveTailPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, str], ...]]:
    artifact = _verify_preflight(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
    )
    stage, checkpoint_name, release_name, _cap, _reserve, rows = _stage_config(preflight)
    release = read_sealed_json(Path(output_root) / release_name)
    _require(
        release.sha256 == require_sha256(expected_release_sha256, "adaptive release")
        and set(release.payload) == _RELEASE_KEYS,
        "adaptive provider release changed hash or schema",
    )
    body = dict(release.payload)
    declared = body.pop("release_identity_sha256", None)
    _require(declared == identity_sha256(body), "adaptive release self-seal changed")
    snapshot = release.payload.get("checkpoint_snapshot")
    _require(
        type(snapshot) is dict and set(snapshot) == _SNAPSHOT_KEYS,
        "adaptive release snapshot schema changed",
    )
    raw_records = snapshot.get("ordered_records")
    _require(
        type(raw_records) is list
        and all(type(row) is dict and set(row) == _RECORD_KEYS for row in raw_records),
        "adaptive release record schema changed",
    )
    released = tuple(dict(row) for row in raw_records)
    for index, row in enumerate(released):
        for key, value in row.items():
            require_sha256(value, f"adaptive release record {index} {key}")
    root = Path(output_root).resolve().as_posix()
    _require(
        release.payload.get("format") == RELEASE_FORMAT
        and release.payload.get("approval_opt_in") is True
        and release.payload.get("release_status") == "approved_for_provider_execution"
        and release.payload.get("stage") == stage
        and release.payload.get("checkpoint_namespace") == checkpoint_name
        and release.payload.get("preflight_sha256") == artifact.sha256
        and release.payload.get("gold_loaded") is False
        and release.payload.get("physical_provider_calls") == 0
        and release.payload.get("output_root") == root
        and release.payload.get("output_root_sha256")
        == identity_sha256({"canonical_root": root})
        and release.payload.get("unsafe_retry_policy")
        == "refuse-incomplete-request-response-pair-v1"
        and snapshot.get("authenticated_complete_count") == len(released)
        and snapshot.get("ordered_records_sha256") == identity_sha256(list(released))
        and release.payload.get("required_authorized_provider_calls")
        == len(rows) - len(released),
        "adaptive release bindings changed",
    )
    current = _checkpoint_records(
        preflight, output_root=output_root, preflight_artifact=artifact
    )
    current_by_message = {row["messages_sha256"]: row for row in current}
    _require(
        all(current_by_message.get(row["messages_sha256"]) == row for row in released),
        "adaptive released checkpoint snapshot is not present",
    )
    assert_gold_blind(release.payload, path="confirmation_adaptive_provider_release")
    return artifact, release, current


def _default_client_factory(gateway_url: str, api_key_env: str) -> Any:
    api_key = os.environ.get(api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {api_key_env}")
    return provider_runtime.make_provider_client(api_key, gateway_url)


def run_confirmation_adaptive_provider(
    preflight: ConfirmationAdaptiveEvidencePreflight | ConfirmationAdaptiveTailPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = provider_runtime.DEFAULT_API_KEY_ENV,
    client_factory: ClientFactory = _default_client_factory,
) -> ConfirmationProviderExecution:
    """Execute only the missing calls for one independently sealed plane."""

    artifact, release, current = _verify_release(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    stage, checkpoint_name, _release, cap, reserve, rows = _stage_config(preflight)
    remaining = len(rows) - len(current)
    _require(
        enable_provider == bool(rows),
        "adaptive provider opt-in must match its prompt population",
    )
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "adaptive provider authorization must equal exact remaining calls",
    )
    _require(
        remaining <= release.payload["required_authorized_provider_calls"],
        "adaptive current state exceeds its sealed release budget",
    )
    if not rows:
        return ConfirmationProviderExecution(stage, None, 0, 0)
    client = (
        client_factory(str(artifact.payload["gateway_url"]), api_key_env)
        if remaining
        else None
    )
    runtime = _runtime(
        artifact,
        rows,
        output_root=output_root,
        stage=stage,
        checkpoint_name=checkpoint_name,
        max_prompt_tokens=cap,
        output_token_reserve=reserve,
        client=client,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.physical_calls == remaining
        and batch.usage.checkpoint_hits == len(current),
        "adaptive native provider accounting differs from exact authorization",
    )
    return ConfirmationProviderExecution(
        stage, batch, batch.usage.physical_calls, batch.usage.checkpoint_hits
    )


def _client_free_batch(
    preflight: ConfirmationAdaptiveEvidencePreflight | ConfirmationAdaptiveTailPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, FastCompletionBatch | None]:
    artifact, release, current = _verify_release(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    stage, checkpoint_name, _release, cap, reserve, rows = _stage_config(preflight)
    _require(
        len(current) == len(rows),
        "adaptive materialization requires a complete checkpoint population",
    )
    if not rows:
        return artifact, release, None
    runtime = _runtime(
        artifact,
        rows,
        output_root=output_root,
        stage=stage,
        checkpoint_name=checkpoint_name,
        max_prompt_tokens=cap,
        output_token_reserve=reserve,
        client=None,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.physical_calls == 0
        and batch.usage.checkpoint_hits == len(rows),
        "adaptive client-free replay entered provider work",
    )
    return artifact, release, batch


def _solver_completion_plane(
    plan: ConfirmationAdaptiveEvidencePlan,
    batch: FastCompletionBatch | None,
) -> AdaptiveSolverCompletionPlane:
    ids = tuple(row.question_id for row in plan.plan.submitted_rows)
    if batch is None:
        _require(not ids, "adaptive solver submitted rows lost their completion batch")
        completions: dict[str, str] = {}
    else:
        _require(
            tuple(row.messages_sha256 for row in batch.prompt_population.ordered_rows)
            == tuple(
                require_sha256(row.messages_sha256, "solver messages")
                for row in plan.plan.submitted_rows
            ),
            "adaptive solver checkpoint order changed",
        )
        completions = dict(zip(ids, batch.logical_completions, strict=True))
    return capture_adaptive_solver_completions(
        plan.plan, plan.preflight, completions
    )


def _solver_run_payload(
    plan: ConfirmationAdaptiveEvidencePlan,
    completion_plane: AdaptiveSolverCompletionPlane,
    run: AdaptiveEvidenceSolverRun,
    *,
    preflight_sha256: str,
) -> dict[str, Any]:
    rows = [
        {
            "changed_from_parent": row.changed_from_parent,
            "completion_receipt_sha256": row.completion_receipt_sha256,
            "dated_question_sha256": row.dated_question_sha256,
            "ordinal": row.ordinal,
            "parent_prediction_sha256": row.parent_prediction_sha256,
            "plan_row_receipt_sha256": row.plan_row_receipt_sha256,
            "prediction": row.prediction,
            "prediction_sha256": row.prediction_sha256,
            "prediction_source": row.prediction_source,
            "question_id": row.question_id,
            "question_sha256": row.question_sha256,
            "solver_decision": row.solver_decision,
            "solver_parse_receipt_sha256": row.solver_parse_receipt_sha256,
            "solver_used_evidence_ids": list(row.solver_used_evidence_ids),
            "solver_used_map_item_ids": list(row.solver_used_map_item_ids),
            "solver_used_source_fact_ids": list(row.solver_used_source_fact_ids),
            "solver_valid": row.solver_valid,
        }
        for row in run.rows
    ]
    payload = {
        "completion_plane_receipt_sha256": completion_plane.receipt_sha256,
        "format": SOLVER_RUN_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls_during_materialization": 0,
        "plan_identity_sha256": plan.plan.plan_identity_sha256,
        "preflight_sha256": require_sha256(preflight_sha256, "solver preflight"),
        "question_count": len(rows),
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "run_receipt_sha256": run.receipt_sha256,
    }
    assert_gold_blind(payload, path="confirmation_adaptive_evidence_run")
    return payload


def materialize_confirmation_adaptive_evidence(
    preflight: ConfirmationAdaptiveEvidencePreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
) -> VerifiedConfirmationAdaptiveEvidencePlane:
    artifact, release, batch = _client_free_batch(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    completion_plane = _solver_completion_plane(preflight.plan, batch)
    run = materialize_adaptive_evidence_solver(
        preflight.plan.plan, preflight.plan.preflight, completion_plane
    )
    plane = replay_adaptive_evidence_solver(
        preflight.plan.plan, preflight.plan.preflight, completion_plane, run
    )
    payload = _solver_run_payload(
        preflight.plan,
        completion_plane,
        run,
        preflight_sha256=artifact.sha256,
    )
    terminal, _created = publish_sealed_json(Path(output_root) / SOLVER_RUN_NAME, payload)
    replay_payload = {
        "byte_identical": True,
        "expected_run_sha256": terminal.sha256,
        "format": SOLVER_REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "replayed_run_sha256": terminal.sha256,
        "retained_transformer_token_state_bytes": 0,
        "verified_plane_receipt_sha256": plane.receipt_sha256,
    }
    replay, _created = publish_sealed_json(
        Path(output_root) / SOLVER_REPLAY_NAME, replay_payload
    )
    return VerifiedConfirmationAdaptiveEvidencePlane(
        preflight.plan,
        artifact,
        release,
        terminal,
        replay,
        batch,
        completion_plane,
        run,
        plane,
    )


def replay_confirmation_adaptive_evidence(
    preflight: ConfirmationAdaptiveEvidencePreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> VerifiedConfirmationAdaptiveEvidencePlane:
    result = materialize_confirmation_adaptive_evidence(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    _require(
        result.run_artifact.sha256
        == require_sha256(expected_run_sha256, "adaptive solver run")
        and result.replay_artifact.sha256
        == require_sha256(expected_replay_sha256, "adaptive solver replay"),
        "adaptive solver run/replay bytes changed",
    )
    return result


def _tail_materializations(
    plan: ConfirmationAdaptiveTailPlan,
    batch: FastCompletionBatch | None,
) -> tuple[
    tuple[source_cli.FastMaterializationQuestionPlan, ...],
    tuple[SourceMapperMaterialization, ...],
]:
    typed_questions: list[source_cli.FastMaterializationQuestionPlan] = []
    results: list[SourceMapperMaterialization] = []
    for question in plan.questions:
        journals: tuple[SourceMapperProviderJournal, ...]
        if batch is None:
            _require(
                not any(
                    row.disposition is WorkDisposition.NEW_CALL
                    for row in question.mapper_preflight.prompt_rows
                ),
                "tail new-call work lost its completion batch",
            )
            journals = ()
        else:
            journals = source_cli.provider_journals_for_question(
                question.mapper_preflight, batch
            )
        result = materialize_source_history_mapper(
            question.mapper_preflight,
            question.hydration_plan,
            question.mapping_plan,
            provider_journals=journals,
            cached_completions=question.cached_completions,
        )
        source_question = next(
            row
            for row in plan.upstream.source_population.questions
            if row.plan.question_id == question.question_id
        )
        typed_questions.append(
            source_cli.FastMaterializationQuestionPlan(
                question.ordinal,
                question.question_id,
                source_question.direct_evidence,
                question.hydration_plan,
                question.mapping_plan,
                question.mapper_preflight,
            )
        )
        results.append(result)
    return tuple(typed_questions), tuple(results)


def _tail_run_payload(
    plan: ConfirmationAdaptiveTailPlan,
    questions: tuple[source_cli.FastMaterializationQuestionPlan, ...],
    materializations: tuple[SourceMapperMaterialization, ...],
    fact_rows: tuple[TailFactUnionRow, ...],
    *,
    preflight_sha256: str,
    work_manifest_sha256: str,
) -> dict[str, Any]:
    payload = {
        "decision_receipt_sha256s": [row.receipt_sha256 for row in plan.decisions],
        "fact_union_rows": [row.projection() for row in fact_rows],
        "format": TAIL_RUN_FORMAT,
        "gold_loaded": False,
        "mapped_question_count": len(questions),
        "materializations": [row.projection() for row in materializations],
        "physical_provider_calls_during_materialization": 0,
        "post_map_dedup_performed": True,
        "preflight_sha256": require_sha256(preflight_sha256, "tail preflight"),
        "question_count": len(plan.decisions),
        "retained_transformer_token_state_bytes": 0,
        "selection_before_physical_dedup": True,
        "work_manifest_sha256": require_sha256(work_manifest_sha256, "tail work"),
    }
    assert_gold_blind(payload, path="confirmation_adaptive_tail_run")
    return payload


def materialize_confirmation_adaptive_tail(
    preflight: ConfirmationAdaptiveTailPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
) -> VerifiedConfirmationAdaptiveTailPlane:
    artifact, release, batch = _client_free_batch(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    questions, materializations = _tail_materializations(preflight.plan, batch)
    fact_rows = build_tail_post_map_fact_unions(questions, materializations)
    payload = _tail_run_payload(
        preflight.plan,
        questions,
        materializations,
        fact_rows,
        preflight_sha256=artifact.sha256,
        work_manifest_sha256=preflight.work_manifest_artifact.sha256,
    )
    terminal, _created = publish_sealed_json(Path(output_root) / TAIL_RUN_NAME, payload)
    replay_payload = {
        "byte_identical": True,
        "expected_run_sha256": terminal.sha256,
        "format": TAIL_REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "replayed_run_sha256": terminal.sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    replay, _created = publish_sealed_json(
        Path(output_root) / TAIL_REPLAY_NAME, replay_payload
    )
    return VerifiedConfirmationAdaptiveTailPlane(
        preflight.plan,
        artifact,
        preflight.work_manifest_artifact,
        release,
        terminal,
        replay,
        batch,
        questions,
        materializations,
        fact_rows,
        preflight.plan.decisions,
    )


def replay_confirmation_adaptive_tail(
    preflight: ConfirmationAdaptiveTailPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> VerifiedConfirmationAdaptiveTailPlane:
    result = materialize_confirmation_adaptive_tail(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    _require(
        result.run_artifact.sha256
        == require_sha256(expected_run_sha256, "adaptive tail run")
        and result.replay_artifact.sha256
        == require_sha256(expected_replay_sha256, "adaptive tail replay"),
        "adaptive tail run/replay bytes changed",
    )
    return result


__all__ = [
    "ConfirmationAdaptiveEvidencePlan",
    "ConfirmationAdaptiveEvidencePreflight",
    "ConfirmationAdaptiveTailError",
    "ConfirmationAdaptiveTailPlan",
    "ConfirmationAdaptiveTailPreflight",
    "ConfirmationAdaptiveUpstream",
    "ConfirmationProviderExecution",
    "SOLVER_CHECKPOINT_DIR_NAME",
    "SOLVER_PREFLIGHT_NAME",
    "SOLVER_RELEASE_NAME",
    "SOLVER_REPLAY_NAME",
    "SOLVER_RUN_NAME",
    "TAIL_CHECKPOINT_DIR_NAME",
    "TAIL_PREFLIGHT_NAME",
    "TAIL_RELEASE_NAME",
    "TAIL_REPLAY_NAME",
    "TAIL_RUN_NAME",
    "TAIL_WORK_MANIFEST_NAME",
    "VerifiedConfirmationAdaptiveEvidencePlane",
    "VerifiedConfirmationAdaptiveTailPlane",
    "approve_confirmation_adaptive_release",
    "build_confirmation_adaptive_evidence_plan",
    "build_confirmation_adaptive_tail_plan",
    "confirmation_adaptive_upstream",
    "materialize_confirmation_adaptive_evidence",
    "materialize_confirmation_adaptive_tail",
    "publish_confirmation_adaptive_evidence_preflight",
    "publish_confirmation_adaptive_tail_preflight",
    "replay_confirmation_adaptive_evidence",
    "replay_confirmation_adaptive_tail",
    "run_confirmation_adaptive_provider",
]
