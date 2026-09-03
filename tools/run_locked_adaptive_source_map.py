#!/usr/bin/env python3
"""Preflight and run the locked adaptive source-history mapper base round.

The command deliberately separates provider-free planning from provider I/O.
``preflight`` is the only phase which discovers source work.  ``provider-run``
reads only that sealed prompt population and exact authorization; it never
opens a memory store or a gold-bearing artifact.  ``materialize`` reuses the
sealed prompt-external work/alias manifest without opening stores. ``replay``
rebuilds the locked source plan, revalidates the stores, and consumes only
immutable FastCompletion request/response journals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools import run_locked_query_evidence_map_solver_v2 as map_cli  # noqa: E402
from tools import run_locked_query_payload_answers as payload_cli  # noqa: E402
from tools.matched_eval import provider_runtime  # noqa: E402
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
from tools.matched_eval.locked_source_gate_adapter import (  # noqa: E402
    LockedSourceGateActivationInput,
    LockedSourceGateAdapterPopulation,
    LockedSourceGateQuestion,
    LockedSourceGatePins,
    LockedSourceHydrationInput,
    load_locked_source_gate_adapter,
    locked_activation_input_from_query_map_adapter,
)
from tools.matched_eval.query_evidence_map_solver_v2_live import (  # noqa: E402
    EvidenceMapPlan,
    VerifiedEvidenceMapPlane,
    replay_evidence_map,
)
from tools.matched_eval.query_map_source_gate_adapter import (  # noqa: E402
    CONSOLIDATED_OBLIGATION_MODE,
    LEGACY_OBLIGATION_MODE,
    OBLIGATION_MODES,
    QueryMapSourceGateAdapterPlane,
    STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    STATE_CHAIN_PROFILES,
    STRICT_STATE_CHAIN_PROFILE,
    adapt_query_map_solver_v2,
)
from tools.matched_eval.source_gate_controller import (  # noqa: E402
    LaneSourceBudget,
    MapWorkAlias,
    ObligationKind,
    QuestionBoundMapWork,
    QuestionBoundMappingPlan,
    QuestionObligation,
    SourceGateRound,
    SourceGatePolicy,
    build_question_bound_mapping_plan,
    start_source_gate,
)
from tools.matched_eval.source_history_fact_union import (  # noqa: E402
    DEFAULT_HISTORY_WINDOW_TOKEN_CAP,
    DirectEvidenceRef,
    EventTuple,
    FrozenHistoryChunk,
    HydratedSourceHistory,
    ParentIdentity,
    SourceHistoryHydrationPlan,
    SourceHistoryWindow,
    SourceSelection,
    direct_evidence_projection_sha256,
    FactLane,
    hydrate_source_histories,
    plan_source_history_hydration,
)
from tools.matched_eval.source_history_mapper_live import (  # noqa: E402
    HARD_CONTEXT_TOKEN_CAP,
    MAPPER_CONTRACT_SHA256,
    MAX_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    SourceMapperMaterialization,
    SourceMapperPreflight,
    SourceMapperProviderJournal,
    SourceHistoryMapperError,
    WorkDisposition,
    build_source_history_mapper_preflight,
    materialize_source_history_mapper,
)


FORMAT = "memory-condense-locked-adaptive-source-map-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"
MATERIALIZATION_FORMAT = f"{FORMAT}-materialization"
REPLAY_FORMAT = f"{FORMAT}-replay"
PREFLIGHT_NAME = "adaptive-source-map-base-preflight-v2.json"
MATERIALIZATION_NAME = "adaptive-source-map-base-materialization-v2.json"
REPLAY_NAME = "adaptive-source-map-base-replay-v2.json"
PARETO_NAME = "adaptive-source-map-pareto-preflight-v1.json"
PARETO_COVERAGE_NAME = "adaptive-source-map-pareto-posthoc-coverage-v2.json"
WORK_MANIFEST_NAME = "adaptive-source-map-base-work-manifest-v1.json"
WORK_MANIFEST_FORMAT = f"{FORMAT}-work-manifest-v1"
CHECKPOINT_DIR_NAME = "terra-source-history-map-calls"

# The optional post-hoc coverage command historically imported the whole
# joint-failure analysis CLI merely to obtain these two immutable pins.  That
# CLI opens the locked validation split and benchmark, which made the pure
# source-map implementation import-reachable from prediction code.  Keep the
# frozen values beside the legacy command that consumes them instead.
DEFAULT_POSTHOC_TARGET_PLAN = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "target-owner-plan-v1/target-plan.json"
)
EXPECTED_POSTHOC_TARGET_PLAN_SHA256 = (
    "b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff"
)

DEFAULT_OUTPUT = (
    payload_cli.DEFAULT_CAMPAIGN_ROOT
    / "s0-plus-query-evidence-map-adaptive-source-v1"
)
DEFAULT_MAP_ROOT = map_cli.DEFAULT_OUTPUT

EXPECTED_MAP_PREFLIGHT_SHA256 = (
    "5bffdac293c064b13bbc8580a0453ac1413bb0b5f3b8465690a66e945e8b8afe"
)
EXPECTED_MAP_RUN_SHA256 = (
    "f658d41b02bb764f85443af055530b0177e715d4e426e77061c4b6e975fce7bd"
)
EXPECTED_MAP_RUNTIME_SHA256 = (
    "0967c0206a0e6a5ee02eaa5995b33b6d7d5dd258c67dffbb763d9e55f83a974c"
)
EXPECTED_LEGACY_QUERY_MAP_ADAPTER_SHA256 = (
    "229c86490a32f9654a6cb12646734c67aa0718e822e014ab0076b850f0b29ea0"
)
EXPECTED_CONSOLIDATED_QUERY_MAP_ADAPTER_SHA256 = (
    "1dc5ff379d04d153d9ea5bcdffd363ca5e337b74d9fff7144cb892fdd6179c55"
)
# Filled from the same provider-free V2 replay as the strict pin.  The
# authority profile is deliberately opt-in and changes only the nine
# question-only state-chain rows whose map stage was intentionally skipped.
EXPECTED_CONSOLIDATED_STATE_CHAIN_AUTHORITY_ADAPTER_SHA256 = (
    "e9ee30e30913ede1daa116f0ec9513906ee55dfa22ed367d32dbd12809bb0a5d"
)
EXPECTED_QUERY_MAP_ADAPTER_SHA256 = EXPECTED_CONSOLIDATED_QUERY_MAP_ADAPTER_SHA256
EXPECTED_QUESTION_COUNT = 100
EXPECTED_LEGACY_ACTIVATION_COUNT = 97
EXPECTED_CONSOLIDATED_ACTIVATION_COUNT = 95
EXPECTED_CONSOLIDATED_STATE_CHAIN_AUTHORITY_ACTIVATION_COUNT = 86
EXPECTED_ACTIVATION_COUNT = EXPECTED_CONSOLIDATED_ACTIVATION_COUNT
TERMINAL_BENCHMARK_AS_OF_TURN = 0
DEFAULT_DIRECT_BASE_CAP = 5
DEFAULT_PARTITION_BASE_CAP = 0
DEFAULT_GUIDED_BASE_CAP = 2


class LockedAdaptiveSourceMapError(MatchedEvalContractError):
    """A locked parent, activation, hydration, prompt, or journal changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedAdaptiveSourceMapError(message)


def _plain_messages(value: Sequence[Any]) -> tuple[dict[str, str], ...]:
    return tuple({"role": row.role, "content": row.content} for row in value)


def source_gate_policy(
    direct_base_cap: int = DEFAULT_DIRECT_BASE_CAP,
    partition_base_cap: int = DEFAULT_PARTITION_BASE_CAP,
    guided_base_cap: int = DEFAULT_GUIDED_BASE_CAP,
) -> SourceGatePolicy:
    """Build the sealed D/P/G base-budget policy used by one Pareto point."""

    caps = (direct_base_cap, partition_base_cap, guided_base_cap)
    _require(
        all(type(value) is int and value >= 0 for value in caps)
        and sum(caps) > 0,
        "D/P/G base caps must be nonnegative with a nonempty union",
    )
    hard_caps = (12, 10, 8)
    _require(
        all(value <= hard for value, hard in zip(caps, hard_caps, strict=True))
        and sum(caps) <= 24,
        "D/P/G base caps exceed the locked lane/global hard caps",
    )
    policy_id = (
        "locked-adaptive-source-map-"
        f"d{direct_base_cap}-p{partition_base_cap}-g{guided_base_cap}-v1"
    )
    return SourceGatePolicy(
        policy_id,
        (
            LaneSourceBudget(FactLane.DIRECT, direct_base_cap, 12, 2),
            LaneSourceBudget(FactLane.PARTITION, partition_base_cap, 10, 2),
            LaneSourceBudget(FactLane.GUIDED, guided_base_cap, 8, 2),
        ),
        global_unique_source_cap=24,
        max_physical_map_calls=48,
        max_rounds=16,
    )


def repolicy_source_population(
    population: LockedSourceGateAdapterPopulation,
    policy: SourceGatePolicy,
) -> LockedSourceGateAdapterPopulation:
    """Reuse one verified in-process source plane under another sealed policy."""

    if type(population) is not LockedSourceGateAdapterPopulation:
        raise TypeError("population must be an exact locked source population")
    if type(policy) is not SourceGatePolicy:
        raise TypeError("policy must be an exact SourceGatePolicy")
    questions = tuple(
        LockedSourceGateQuestion(
            row.ordinal,
            replace(row.plan, policy=policy),
            row.source_packet_id,
            row.activation_input_receipt_sha256,
            row.namespace,
            row.direct_evidence,
            row.store_dir,
            row.database_sha256,
            row.index_sha256,
        )
        for row in population.questions
    )
    return LockedSourceGateAdapterPopulation(
        population.source_artifacts,
        questions,
        population.direct_stream_profile,
    )


def _sealed_parent_args(*, max_concurrency: int, gateway_url: str) -> argparse.Namespace:
    """Construct the exact existing V2 map-loader namespace."""

    return argparse.Namespace(
        retrieval=payload_cli.DEFAULT_RETRIEVAL,
        query_preflight=payload_cli.DEFAULT_QUERY_PREFLIGHT,
        query_run=payload_cli.DEFAULT_QUERY_RUN,
        parent_root=payload_cli.DEFAULT_PARENT_ROOT,
        direct_answer_root=map_cli.DEFAULT_DIRECT_ANSWER_ROOT,
        output_root=DEFAULT_MAP_ROOT,
        expected_retrieval_sha256=payload_cli.EXPECTED_RETRIEVAL_SHA256,
        expected_source_population_id=payload_cli.EXPECTED_SOURCE_POPULATION_ID,
        expected_query_preflight_sha256=payload_cli.EXPECTED_QUERY_PREFLIGHT_SHA256,
        expected_query_run_sha256=payload_cli.EXPECTED_QUERY_RUN_SHA256,
        expected_query_population_id=payload_cli.EXPECTED_QUERY_POPULATION_ID,
        expected_query_prompt_population_sha256=(
            payload_cli.EXPECTED_QUERY_PROMPT_POPULATION_SHA256
        ),
        expected_parent_answer_run_sha256=(
            payload_cli.EXPECTED_PARENT_ANSWER_RUN_SHA256
        ),
        expected_direct_answer_preflight_sha256=(
            map_cli.EXPECTED_DIRECT_ANSWER_PREFLIGHT_SHA256
        ),
        expected_direct_answer_run_sha256=(
            map_cli.EXPECTED_DIRECT_ANSWER_RUN_SHA256
        ),
        expected_direct_semantic_binding_sha256=(
            map_cli.EXPECTED_DIRECT_SEMANTIC_BINDING_SHA256
        ),
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )


def load_locked_query_map(
    *,
    max_concurrency: int,
    gateway_url: str,
    obligation_mode: str = CONSOLIDATED_OBLIGATION_MODE,
    state_chain_profile: str = STRICT_STATE_CHAIN_PROFILE,
) -> tuple[SealedArtifact, EvidenceMapPlan, VerifiedEvidenceMapPlane, QueryMapSourceGateAdapterPlane]:
    """Replay the exact V2 map and mechanically derive source activations."""

    preflight = read_sealed_json(DEFAULT_MAP_ROOT / "map-preflight.json")
    _require(
        preflight.sha256 == EXPECTED_MAP_PREFLIGHT_SHA256,
        "locked V2 evidence-map preflight changed",
    )
    plan = map_cli._load_map_plan(  # noqa: SLF001 - this is the pinned CLI loader
        _sealed_parent_args(
            max_concurrency=max_concurrency,
            gateway_url=gateway_url,
        )
    )
    plane = replay_evidence_map(
        plan,
        output_root=DEFAULT_MAP_ROOT,
        expected_preflight_sha256=EXPECTED_MAP_PREFLIGHT_SHA256,
        expected_run_sha256=EXPECTED_MAP_RUN_SHA256,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    _require(
        plane.run_sha256 == EXPECTED_MAP_RUN_SHA256
        and plane.replay_sha256 == EXPECTED_MAP_RUN_SHA256
        and plane.runtime_ledger_sha256 == EXPECTED_MAP_RUNTIME_SHA256,
        "locked V2 evidence-map replay changed",
    )
    query_run = read_sealed_json(payload_cli.DEFAULT_QUERY_RUN)
    _require(
        query_run.sha256 == payload_cli.EXPECTED_QUERY_RUN_SHA256,
        "sealed query-expansion run changed",
    )
    _require(obligation_mode in OBLIGATION_MODES, "obligation mode changed")
    _require(state_chain_profile in STATE_CHAIN_PROFILES, "state-chain profile changed")
    adapter = adapt_query_map_solver_v2(
        query_run,
        plan,
        plane,
        obligation_mode=obligation_mode,
        state_chain_profile=state_chain_profile,
    )
    expected_by_profile = {
        (LEGACY_OBLIGATION_MODE, STRICT_STATE_CHAIN_PROFILE): (
            EXPECTED_LEGACY_QUERY_MAP_ADAPTER_SHA256,
            EXPECTED_LEGACY_ACTIVATION_COUNT,
        ),
        (CONSOLIDATED_OBLIGATION_MODE, STRICT_STATE_CHAIN_PROFILE): (
            EXPECTED_CONSOLIDATED_QUERY_MAP_ADAPTER_SHA256,
            EXPECTED_CONSOLIDATED_ACTIVATION_COUNT,
        ),
        (
            CONSOLIDATED_OBLIGATION_MODE,
            STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
        ): (
            EXPECTED_CONSOLIDATED_STATE_CHAIN_AUTHORITY_ADAPTER_SHA256,
            EXPECTED_CONSOLIDATED_STATE_CHAIN_AUTHORITY_ACTIVATION_COUNT,
        ),
    }
    _require(
        (obligation_mode, state_chain_profile) in expected_by_profile,
        "requested obligation/state-chain profile has no locked adapter pin",
    )
    expected_adapter, expected_activations = expected_by_profile[
        (obligation_mode, state_chain_profile)
    ]
    _require(
        len(adapter.rows) == EXPECTED_QUESTION_COUNT
        and len(adapter.activated_rows) == expected_activations
        and adapter.receipt_sha256 == expected_adapter,
        "locked query-map activation population changed: "
        f"rows={len(adapter.rows)} activated={len(adapter.activated_rows)} "
        f"receipt={adapter.receipt_sha256}",
    )
    return query_run, plan, plane, adapter


def activation_inputs_from_query_map(
    adapter: QueryMapSourceGateAdapterPlane,
    *,
    as_of_turn: int = TERMINAL_BENCHMARK_AS_OF_TURN,
) -> tuple[LockedSourceGateActivationInput, ...]:
    """Convert only mechanically unresolved adapter rows to gate inputs."""

    if type(adapter) is not QueryMapSourceGateAdapterPlane:
        raise TypeError("adapter must be an exact QueryMapSourceGateAdapterPlane")
    _require(type(as_of_turn) is int and as_of_turn >= 0, "as-of turn changed")
    result: list[LockedSourceGateActivationInput] = []
    for row in adapter.rows:
        if row.activation is None:
            continue
        result.append(
            locked_activation_input_from_query_map_adapter(
                row,
                as_of_turn=as_of_turn,
            )
        )
    _require(
        len(result) == len(adapter.activated_rows),
        "activation conversion changed unresolved row count",
    )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class NamespaceHydrationBatch:
    namespace_id: str
    store_dir: Path
    database_sha256: str
    index_sha256: str
    source_ids: tuple[str, ...]
    history_receipt_sha256s: tuple[str, ...]
    receipt_sha256: str

    def projection(self) -> dict[str, Any]:
        return {
            "database_read_only": True,
            "database_sha256": self.database_sha256,
            "history_receipt_sha256s": list(self.history_receipt_sha256s),
            "index_sha256": self.index_sha256,
            "namespace_id": self.namespace_id,
            "namespace_read_count": 1,
            "namespace_scan_count": 1,
            "receipt_sha256": self.receipt_sha256,
            "source_ids": list(self.source_ids),
            "store_bytes_revalidated_before_after": True,
            "store_dir": str(self.store_dir),
        }


@dataclass(frozen=True, slots=True)
class BaseQuestionSourceMap:
    ordinal: int
    question_id: str
    gate_round: SourceGateRound
    hydration_plan: SourceHistoryHydrationPlan
    mapping_plan: QuestionBoundMappingPlan
    mapper_preflight: SourceMapperPreflight

    def projection(self) -> dict[str, Any]:
        return {
            "base_round_receipt_sha256": self.gate_round.receipt_sha256,
            "deferred_physical_work_count": len(
                self.mapping_plan.deferred_work_ids
            ),
            "gate_plan_receipt_sha256": (
                self.gate_round.gate_plan_receipt_sha256
            ),
            "hydration_plan_receipt_sha256": self.hydration_plan.receipt_sha256,
            "history_window_token_cap": self.hydration_plan.max_window_tokens,
            "logical_selection_count": len(self.gate_round.selections),
            "logical_window_count": len(self.hydration_plan.windows),
            "mapper_preflight_receipt_sha256": self.mapper_preflight.receipt_sha256,
            "mapping_plan_receipt_sha256": self.mapping_plan.receipt_sha256,
            "maximum_prompt_and_output_token_envelope": (
                self.mapper_preflight.maximum_combined_token_proxy
            ),
            "new_provider_call_count": self.mapping_plan.planned_provider_calls,
            "ordinal": self.ordinal,
            "physical_prompt_count": len(self.mapper_preflight.prompt_rows),
            "physical_work_count": len(self.mapping_plan.work_items),
            "question_id": self.question_id,
            "reused_physical_work_count": len(self.mapping_plan.reused_work_ids),
            "unique_selected_source_count": self.gate_round.cumulative_unique_source_count,
        }


@dataclass(frozen=True, slots=True)
class LockedAdaptiveBasePlan:
    query_adapter: QueryMapSourceGateAdapterPlane | None
    source_population: LockedSourceGateAdapterPopulation
    hydration_batches: tuple[NamespaceHydrationBatch, ...]
    questions: tuple[BaseQuestionSourceMap, ...]
    route_counts: tuple[tuple[str, int], ...]
    provider_population: FastPromptPopulation

    @property
    def required_provider_calls(self) -> int:
        return self.provider_population.unique_prompt_count

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
class FastMaterializationQuestionPlan:
    """Store-free exact-quote validation objects reconstructed from one seal."""

    ordinal: int
    question_id: str
    direct_evidence: tuple[DirectEvidenceRef, ...]
    hydration_plan: SourceHistoryHydrationPlan
    mapping_plan: QuestionBoundMappingPlan
    mapper_preflight: SourceMapperPreflight


def _fact_union_seal(kind: str, body: Mapping[str, Any]) -> str:
    return identity_sha256(
        {
            "format": f"memory-condense-source-history-fact-union-v2-{kind}",
            **body,
        }
    )


def _exact_int(value: object, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} changed")
    return value  # type: ignore[return-value]


def _exact_bool(value: object, label: str) -> bool:
    _require(type(value) is bool, f"{label} changed")
    return value  # type: ignore[return-value]


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    _require(
        type(value) is list
        and all(type(item) is str and bool(item) for item in value),
        f"{label} changed",
    )
    return tuple(value)


def _chunk_manifest(chunk: FrozenHistoryChunk) -> dict[str, Any]:
    return {
        **chunk.projection(include_text=True),
        "chunk_receipt_sha256": chunk.chunk_receipt_sha256,
    }


def _direct_ref_manifest(row: DirectEvidenceRef) -> dict[str, Any]:
    return {**row.projection(), "text": row.text}


def _history_manifest(history: HydratedSourceHistory) -> dict[str, Any]:
    return {
        "chunk_receipt_sha256s": [
            row.chunk_receipt_sha256 for row in history.chunks
        ],
        "content_chunk_ids": list(history.content_chunk_ids),
        "membership_projection_sha256": history.membership_projection_sha256,
        "metadata_chunk_ids": list(history.metadata_chunk_ids),
        "namespace_id": history.namespace_id,
        "receipt_sha256": history.receipt_sha256,
        "source_id": history.source_id,
        "store_bytes_revalidated_before_after": history.store_bytes_revalidated,
        "stream_sha256": history.stream_sha256,
    }


def _window_manifest(window: SourceHistoryWindow) -> dict[str, Any]:
    return {
        "chunk_receipt_sha256s": [
            row.chunk_receipt_sha256 for row in window.chunks
        ],
        "content_token_proxy": window.content_token_proxy,
        "history_receipt_sha256": window.history_receipt_sha256,
        "parent_identity_sha256": window.parent_identity_sha256,
        "receipt_sha256": window.receipt_sha256,
        "selection_id": window.selection.selection_id,
        "token_cap": window.token_cap,
        "window_id": window.window_id,
        "window_ordinal": window.window_ordinal,
    }


def _hydration_manifest(plan: SourceHistoryHydrationPlan) -> dict[str, Any]:
    return {
        "histories": [_history_manifest(row) for row in plan.histories],
        "max_window_tokens": plan.max_window_tokens,
        "parent": plan.parent.projection(),
        "receipt_sha256": plan.receipt_sha256,
        "selections": [row.projection() for row in plan.selections],
        "windows": [_window_manifest(row) for row in plan.windows],
    }


def _work_manifest(work: QuestionBoundMapWork) -> dict[str, Any]:
    value = work.mapping_payload()
    value.pop("chunks")
    value["chunk_receipt_sha256s"] = [
        row.chunk_receipt_sha256 for row in work.chunks
    ]
    value["work_id"] = work.work_id
    return value


def _mapping_manifest(plan: QuestionBoundMappingPlan) -> dict[str, Any]:
    return {
        "aliases": [row.projection() for row in plan.aliases],
        "deferred_work_ids": list(plan.deferred_work_ids),
        "gate_plan_receipt_sha256": plan.gate_plan_receipt_sha256,
        "gate_round_receipt_sha256": plan.gate_round_receipt_sha256,
        "hydration_plan_receipt_sha256": plan.hydration_plan_receipt_sha256,
        "new_call_work_ids": list(plan.new_call_work_ids),
        "prior_call_work_ids": list(plan.prior_call_work_ids),
        "receipt_sha256": plan.receipt_sha256,
        "reused_work_ids": list(plan.reused_work_ids),
        "work_items": [_work_manifest(row) for row in plan.work_items],
    }


def work_manifest_projection(plan: LockedAdaptiveBasePlan) -> dict[str, Any]:
    """Seal exact validation work once so dev materialization needs no store."""

    chunks: "OrderedDict[str, FrozenHistoryChunk]" = OrderedDict()
    for question in plan.questions:
        for history in question.hydration_plan.histories:
            for chunk in history.chunks:
                previous = chunks.setdefault(chunk.chunk_receipt_sha256, chunk)
                _require(previous == chunk, "chunk receipt collided in work manifest")
    source_question_by_id = {
        row.plan.question_id: row for row in plan.source_population.questions
    }
    questions = [
        {
            "direct_evidence": [
                _direct_ref_manifest(item)
                for item in source_question_by_id[row.question_id].direct_evidence
            ],
            "hydration_plan": _hydration_manifest(row.hydration_plan),
            "mapper_preflight_receipt_sha256": row.mapper_preflight.receipt_sha256,
            "mapping_plan": _mapping_manifest(row.mapping_plan),
            "obligations": [
                item.projection() for item in row.mapping_plan.work_items[0].obligations
            ],
            "ordinal": row.ordinal,
            "question_id": row.question_id,
        }
        for row in plan.questions
    ]
    payload: dict[str, Any] = {
        "chunk_pool": [_chunk_manifest(row) for row in chunks.values()],
        "format": WORK_MANIFEST_FORMAT,
        "gold_loaded": False,
        "mapper_contract_sha256": MAPPER_CONTRACT_SHA256,
        "provider_calls": 0,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
        "source_gate_population_receipt_sha256": (
            plan.source_population.receipt_sha256
        ),
        "store_reads_during_reuse": 0,
    }
    assert_gold_blind(payload, path="locked_adaptive_source_map_work_manifest")
    return payload


def _parse_chunk(raw: object) -> FrozenHistoryChunk:
    _require(type(raw) is dict, "work-manifest chunk changed type")
    assert type(raw) is dict
    chunk = FrozenHistoryChunk(
        require_text(raw.get("source_id"), "manifest chunk source"),
        require_text(raw.get("chunk_id"), "manifest chunk ID"),
        require_text(raw.get("turn_id"), "manifest chunk turn"),
        _exact_int(raw.get("turn_ordinal"), "manifest chunk turn ordinal"),
        require_text(raw.get("role"), "manifest chunk role"),
        require_text(raw.get("created_at"), "manifest chunk creation"),
        _exact_int(raw.get("start_char"), "manifest chunk start"),
        _exact_int(raw.get("end_char"), "manifest chunk end", 1),
        require_text(raw.get("text"), "manifest chunk text"),
        _exact_int(raw.get("token_count"), "manifest chunk tokens", 1),
        require_sha256(raw.get("turn_text_sha256"), "manifest chunk turn text"),
        _exact_bool(raw.get("metadata_chunk"), "manifest metadata flag"),
    )
    _require(raw == _chunk_manifest(chunk), "work-manifest chunk seal changed")
    return chunk


def _parse_event(raw: object) -> EventTuple | None:
    if raw is None:
        return None
    _require(type(raw) is dict, "manifest direct event changed type")
    assert type(raw) is dict
    event = EventTuple(
        require_text(raw.get("subject"), "manifest event subject"),
        require_text(raw.get("predicate"), "manifest event predicate"),
        require_text(raw.get("object"), "manifest event object"),
        require_text(raw.get("event_time"), "manifest event time"),
        require_text(raw.get("polarity"), "manifest event polarity"),
        require_text(raw.get("status"), "manifest event status"),
    )
    _require(raw == event.projection(), "manifest direct event changed")
    return event


def _parse_direct_ref(raw: object) -> DirectEvidenceRef:
    _require(type(raw) is dict, "manifest direct evidence changed type")
    assert type(raw) is dict
    text = raw.get("text")
    _require(text is None or type(text) is str, "manifest direct text changed type")
    row = DirectEvidenceRef(
        require_text(raw.get("evidence_id"), "manifest direct evidence ID"),
        require_sha256(raw.get("namespace_id"), "manifest direct namespace"),
        require_text(raw.get("source_id"), "manifest direct source"),
        require_sha256(raw.get("quote_sha256"), "manifest direct quote"),
        require_sha256(
            raw.get("evidence_receipt_sha256"), "manifest direct receipt"
        ),
        _parse_event(raw.get("event_tuple")),
        text,
    )
    _require(raw == _direct_ref_manifest(row), "manifest direct evidence seal changed")
    return row


def _parse_parent(raw: object) -> ParentIdentity:
    _require(type(raw) is dict, "work-manifest parent changed type")
    assert type(raw) is dict
    parent = ParentIdentity(
        require_sha256(raw.get("population_identity_sha256"), "manifest population"),
        require_sha256(raw.get("question_order_sha256"), "manifest question order"),
        require_sha256(raw.get("snapshot_id"), "manifest snapshot"),
        require_sha256(raw.get("namespace_id"), "manifest namespace"),
        require_sha256(raw.get("parent_packet_id"), "manifest parent packet"),
        require_sha256(raw.get("parent_stage_receipt_sha256"), "manifest parent stage"),
        require_sha256(
            raw.get("direct_evidence_projection_sha256"),
            "manifest direct evidence",
        ),
    )
    _require(raw == parent.projection(), "work-manifest parent seal changed")
    return parent


def _parse_selection(raw: object) -> SourceSelection:
    _require(type(raw) is dict, "work-manifest selection changed type")
    assert type(raw) is dict
    selection = SourceSelection(
        require_text(raw.get("selection_id"), "manifest selection ID"),
        FactLane(require_text(raw.get("lane"), "manifest selection lane")),
        require_sha256(raw.get("namespace_id"), "manifest selection namespace"),
        require_text(raw.get("source_id"), "manifest selection source"),
        _exact_int(raw.get("rank"), "manifest selection rank"),
        require_sha256(
            raw.get("selector_receipt_sha256"), "manifest selector receipt"
        ),
    )
    _require(raw == selection.projection(), "work-manifest selection seal changed")
    return selection


def _parse_obligation(raw: object) -> QuestionObligation:
    _require(type(raw) is dict, "work-manifest obligation changed type")
    assert type(raw) is dict
    obligation = QuestionObligation(
        ObligationKind(require_text(raw.get("kind"), "manifest obligation kind")),
        _string_tuple(raw.get("match_terms"), "manifest obligation terms"),
        _exact_int(
            raw.get("required_match_term_count"), "manifest required terms", 1
        ),
        _exact_int(raw.get("minimum_fact_count"), "manifest minimum facts", 1),
        _exact_int(
            raw.get("minimum_source_count"), "manifest minimum sources", 1
        ),
        _exact_bool(
            raw.get("requires_temporal_metadata"), "manifest temporal flag"
        ),
        _exact_bool(
            raw.get("requires_complete_frontier"), "manifest frontier flag"
        ),
    )
    _require(raw == obligation.projection(), "work-manifest obligation seal changed")
    return obligation


def _parse_history(
    raw: object,
    chunks: Mapping[str, FrozenHistoryChunk],
) -> HydratedSourceHistory:
    _require(type(raw) is dict, "work-manifest history changed type")
    assert type(raw) is dict
    chunk_ids = _string_tuple(
        raw.get("chunk_receipt_sha256s"), "manifest history chunks"
    )
    _require(all(value in chunks for value in chunk_ids), "manifest history lost chunk")
    history = HydratedSourceHistory(
        require_sha256(raw.get("namespace_id"), "manifest history namespace"),
        require_text(raw.get("source_id"), "manifest history source"),
        _string_tuple(raw.get("content_chunk_ids"), "manifest content chunks"),
        _string_tuple(raw.get("metadata_chunk_ids"), "manifest metadata chunks"),
        require_sha256(raw.get("stream_sha256"), "manifest history stream"),
        require_sha256(
            raw.get("membership_projection_sha256"), "manifest history membership"
        ),
        tuple(chunks[value] for value in chunk_ids),
        _exact_bool(
            raw.get("store_bytes_revalidated_before_after"),
            "manifest store revalidation",
        ),
        require_sha256(raw.get("receipt_sha256"), "manifest history receipt"),
    )
    membership = {
        "content_chunk_ids": list(history.content_chunk_ids),
        "metadata_chunk_ids": list(history.metadata_chunk_ids),
        "source_id": history.source_id,
        "stream_sha256": history.stream_sha256,
    }
    _require(
        identity_sha256(membership) == history.membership_projection_sha256,
        "work-manifest membership projection changed",
    )
    body = {
        "chunk_receipt_sha256s": [
            row.chunk_receipt_sha256 for row in history.chunks
        ],
        "content_chunk_ids": list(history.content_chunk_ids),
        "database_read_only": True,
        "membership_projection_sha256": history.membership_projection_sha256,
        "metadata_chunk_ids": list(history.metadata_chunk_ids),
        "namespace_id": history.namespace_id,
        "source_id": history.source_id,
        "store_bytes_revalidated_before_after": True,
        "stream_sha256": history.stream_sha256,
        "validated_against_scan_discourse_source_chunks": True,
    }
    _require(
        history.store_bytes_revalidated
        and history.receipt_sha256 == _fact_union_seal("hydrated-source", body)
        and raw == _history_manifest(history),
        "work-manifest history seal changed",
    )
    return history


def _parse_hydration(
    raw: object,
    chunks: Mapping[str, FrozenHistoryChunk],
) -> SourceHistoryHydrationPlan:
    _require(type(raw) is dict, "work-manifest hydration changed type")
    assert type(raw) is dict
    parent = _parse_parent(raw.get("parent"))
    selections = tuple(
        _parse_selection(value)
        for value in (
            raw.get("selections")
            if type(raw.get("selections")) is list
            else ()
        )
    )
    histories = tuple(
        _parse_history(value, chunks)
        for value in (
            raw.get("histories") if type(raw.get("histories")) is list else ()
        )
    )
    _require(bool(selections) and bool(histories), "manifest hydration is empty")
    selection_by_id = {row.selection_id: row for row in selections}
    _require(
        len(selection_by_id) == len(selections), "manifest selections repeat"
    )
    raw_windows = raw.get("windows")
    _require(type(raw_windows) is list and bool(raw_windows), "manifest windows missing")
    windows: list[SourceHistoryWindow] = []
    for value in raw_windows:
        _require(type(value) is dict, "manifest window changed type")
        assert type(value) is dict
        selection_id = require_text(
            value.get("selection_id"), "manifest window selection"
        )
        _require(selection_id in selection_by_id, "manifest window lost selection")
        chunk_ids = _string_tuple(
            value.get("chunk_receipt_sha256s"), "manifest window chunks"
        )
        _require(all(item in chunks for item in chunk_ids), "manifest window lost chunk")
        window = SourceHistoryWindow(
            require_sha256(
                value.get("parent_identity_sha256"), "manifest window parent"
            ),
            selection_by_id[selection_id],
            require_sha256(
                value.get("history_receipt_sha256"), "manifest window history"
            ),
            _exact_int(value.get("window_ordinal"), "manifest window ordinal"),
            tuple(chunks[item] for item in chunk_ids),
            _exact_int(
                value.get("content_token_proxy"), "manifest window tokens", 1
            ),
            _exact_int(value.get("token_cap"), "manifest window cap", 1),
            require_sha256(value.get("window_id"), "manifest window ID"),
            require_sha256(value.get("receipt_sha256"), "manifest window receipt"),
        )
        body = {
            "chunk_receipt_sha256s": [
                item.chunk_receipt_sha256 for item in window.chunks
            ],
            "content_token_proxy": window.content_token_proxy,
            "frozen_chunk_boundaries": True,
            "history_receipt_sha256": window.history_receipt_sha256,
            "parent_identity_sha256": window.parent_identity_sha256,
            "selection": window.selection.projection(),
            "token_cap": window.token_cap,
            "window_ordinal": window.window_ordinal,
        }
        _require(
            window.parent_identity_sha256 == parent.identity_sha256
            and window.window_id == _fact_union_seal("window-id", body)
            and window.receipt_sha256
            == _fact_union_seal("window", {**body, "window_id": window.window_id})
            and value == _window_manifest(window),
            "work-manifest window seal changed",
        )
        windows.append(window)
    cap = _exact_int(raw.get("max_window_tokens"), "manifest hydration cap", 1)
    body = {
        "history_receipt_sha256s": [row.receipt_sha256 for row in histories],
        "max_window_tokens": cap,
        "no_dedup_before_mapping": True,
        "parent_identity_sha256": parent.identity_sha256,
        "selections": [row.projection() for row in selections],
        "window_receipt_sha256s": [row.receipt_sha256 for row in windows],
    }
    receipt = require_sha256(raw.get("receipt_sha256"), "manifest hydration receipt")
    _require(
        receipt == _fact_union_seal("hydration-plan", body),
        "work-manifest hydration seal changed",
    )
    plan = SourceHistoryHydrationPlan(
        parent, selections, histories, tuple(windows), cap, receipt
    )
    _require(raw == _hydration_manifest(plan), "work-manifest hydration bytes changed")
    return plan


def _parse_work(
    raw: object,
    *,
    chunks: Mapping[str, FrozenHistoryChunk],
    obligations: tuple[QuestionObligation, ...],
) -> QuestionBoundMapWork:
    _require(type(raw) is dict, "work-manifest mapping work changed type")
    assert type(raw) is dict
    chunk_ids = _string_tuple(
        raw.get("chunk_receipt_sha256s"), "manifest mapping work chunks"
    )
    _require(all(value in chunks for value in chunk_ids), "mapping work lost chunk")
    work = QuestionBoundMapWork(
        require_sha256(raw.get("gate_plan_receipt_sha256"), "manifest work gate"),
        require_sha256(raw.get("parent_identity_sha256"), "manifest work parent"),
        require_text(raw.get("question_id"), "manifest work question ID"),
        require_sha256(raw.get("question_sha256"), "manifest work question"),
        require_text(raw.get("dated_question"), "manifest dated question"),
        require_sha256(
            raw.get("dated_question_sha256"), "manifest dated question"
        ),
        require_sha256(raw.get("route_receipt_sha256"), "manifest work route"),
        obligations,
        require_sha256(raw.get("namespace_id"), "manifest work namespace"),
        require_text(raw.get("source_id"), "manifest work source"),
        require_sha256(
            raw.get("membership_projection_sha256"), "manifest work membership"
        ),
        require_sha256(raw.get("stream_sha256"), "manifest work stream"),
        require_sha256(
            raw.get("source_history_receipt_sha256"), "manifest work history"
        ),
        _exact_int(
            raw.get("history_window_ordinal"), "manifest work window ordinal"
        ),
        _exact_int(
            raw.get("history_window_token_cap"), "manifest work window cap", 1
        ),
        _exact_int(raw.get("content_token_proxy"), "manifest work tokens", 1),
        tuple(chunks[value] for value in chunk_ids),
        require_sha256(
            raw.get("mapper_contract_sha256"), "manifest mapper contract"
        ),
    )
    _require(raw == _work_manifest(work), "work-manifest mapping work seal changed")
    return work


def _parse_mapping(
    raw: object,
    *,
    chunks: Mapping[str, FrozenHistoryChunk],
    obligations: tuple[QuestionObligation, ...],
) -> QuestionBoundMappingPlan:
    _require(type(raw) is dict, "work-manifest mapping plan changed type")
    assert type(raw) is dict
    raw_work = raw.get("work_items")
    raw_aliases = raw.get("aliases")
    _require(
        type(raw_work) is list
        and bool(raw_work)
        and type(raw_aliases) is list
        and bool(raw_aliases),
        "manifest mapping plan is empty",
    )
    work = tuple(
        _parse_work(value, chunks=chunks, obligations=obligations)
        for value in raw_work
    )
    aliases: list[MapWorkAlias] = []
    for value in raw_aliases:
        _require(type(value) is dict, "manifest mapping alias changed type")
        assert type(value) is dict
        alias = MapWorkAlias(
            require_sha256(value.get("physical_work_id"), "manifest alias work"),
            require_sha256(
                value.get("hydration_plan_receipt_sha256"),
                "manifest alias hydration",
            ),
            require_sha256(value.get("window_id"), "manifest alias window"),
            require_sha256(
                value.get("window_receipt_sha256"), "manifest alias window receipt"
            ),
            require_sha256(
                value.get("mapping_payload_sha256"), "manifest alias payload"
            ),
            require_text(value.get("selection_id"), "manifest alias selection"),
            FactLane(require_text(value.get("lane"), "manifest alias lane")),
        )
        _require(value == alias.projection(), "work-manifest alias seal changed")
        aliases.append(alias)
    plan = QuestionBoundMappingPlan(
        require_sha256(raw.get("gate_plan_receipt_sha256"), "manifest mapping gate"),
        require_sha256(
            raw.get("gate_round_receipt_sha256"), "manifest mapping round"
        ),
        require_sha256(
            raw.get("hydration_plan_receipt_sha256"), "manifest mapping hydration"
        ),
        work,
        tuple(aliases),
        _string_tuple(raw.get("reused_work_ids"), "manifest reused work"),
        _string_tuple(raw.get("new_call_work_ids"), "manifest new-call work"),
        _string_tuple(raw.get("deferred_work_ids"), "manifest deferred work"),
        _string_tuple(raw.get("prior_call_work_ids"), "manifest prior work"),
    )
    _require(
        raw.get("receipt_sha256") == plan.receipt_sha256
        and raw == _mapping_manifest(plan),
        "work-manifest mapping plan seal changed",
    )
    return plan


def load_fast_materialization_manifest(
    artifact: SealedArtifact,
    *,
    expected_source_population_receipt_sha256: str,
) -> tuple[FastMaterializationQuestionPlan, ...]:
    """Rehydrate only sealed validation objects; never open a memory store."""

    require_sha256(
        expected_source_population_receipt_sha256, "expected source population"
    )
    payload = artifact.payload
    assert_gold_blind(payload, path="locked_adaptive_source_map_fast_materialize")
    _require(
        payload.get("format") == WORK_MANIFEST_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("store_reads_during_reuse") == 0
        and payload.get("mapper_contract_sha256") == MAPPER_CONTRACT_SHA256
        and payload.get("source_gate_population_receipt_sha256")
        == expected_source_population_receipt_sha256,
        "fast materialization manifest changed its seal/firewall",
    )
    raw_chunks = payload.get("chunk_pool")
    _require(type(raw_chunks) is list and bool(raw_chunks), "manifest chunk pool empty")
    parsed_chunks = tuple(_parse_chunk(value) for value in raw_chunks)
    chunks = {row.chunk_receipt_sha256: row for row in parsed_chunks}
    _require(len(chunks) == len(parsed_chunks), "manifest chunk pool repeats")
    raw_questions = payload.get("questions")
    _require(
        type(raw_questions) is list and bool(raw_questions),
        "manifest questions are empty",
    )
    result: list[FastMaterializationQuestionPlan] = []
    for value in raw_questions:
        _require(type(value) is dict, "manifest question changed type")
        assert type(value) is dict
        raw_direct = value.get("direct_evidence")
        _require(type(raw_direct) is list, "manifest direct evidence changed type")
        direct_evidence = tuple(_parse_direct_ref(row) for row in raw_direct)
        raw_obligations = value.get("obligations")
        _require(
            type(raw_obligations) is list and bool(raw_obligations),
            "manifest question obligations are empty",
        )
        obligations = tuple(_parse_obligation(row) for row in raw_obligations)
        hydration = _parse_hydration(value.get("hydration_plan"), chunks)
        _require(
            direct_evidence_projection_sha256(direct_evidence)
            == hydration.parent.direct_evidence_projection_sha256,
            "manifest direct evidence changed hydration-parent binding",
        )
        mapping = _parse_mapping(
            value.get("mapping_plan"),
            chunks=chunks,
            obligations=obligations,
        )
        _require(
            mapping.hydration_plan_receipt_sha256 == hydration.receipt_sha256,
            "manifest mapping/hydration binding changed",
        )
        mapper_preflight = build_source_history_mapper_preflight(
            hydration, mapping
        )
        _require(
            mapper_preflight.receipt_sha256
            == value.get("mapper_preflight_receipt_sha256"),
            "manifest mapper preflight changed",
        )
        ordinal = _exact_int(value.get("ordinal"), "manifest question ordinal")
        question_id = require_text(value.get("question_id"), "manifest question ID")
        _require(
            all(row.question_id == question_id for row in mapping.work_items),
            "manifest work escaped question",
        )
        expected_question = {
            "direct_evidence": [
                _direct_ref_manifest(row) for row in direct_evidence
            ],
            "hydration_plan": _hydration_manifest(hydration),
            "mapper_preflight_receipt_sha256": mapper_preflight.receipt_sha256,
            "mapping_plan": _mapping_manifest(mapping),
            "obligations": [row.projection() for row in obligations],
            "ordinal": ordinal,
            "question_id": question_id,
        }
        _require(value == expected_question, "work-manifest question bytes changed")
        result.append(
            FastMaterializationQuestionPlan(
                ordinal,
                question_id,
                direct_evidence,
                hydration,
                mapping,
                mapper_preflight,
            )
        )
    _require(
        tuple(row.ordinal for row in result)
        == tuple(sorted({row.ordinal for row in result}))
        and len({row.question_id for row in result}) == len(result),
        "work-manifest question order changed",
    )
    return tuple(result)


def _membership_identity(value: Any) -> str:
    return identity_sha256(value.projection())


def _hydrate_namespace_batches(
    question_rounds: Sequence[tuple[Any, SourceGateRound]],
) -> tuple[
    tuple[NamespaceHydrationBatch, ...],
    dict[tuple[str, str], HydratedSourceHistory],
]:
    grouped: "OrderedDict[str, list[LockedSourceHydrationInput]]" = OrderedDict()
    for question, round_plan in question_rounds:
        if not round_plan.selected_candidates:
            continue
        hydration = question.hydration_input(round_plan)
        grouped.setdefault(hydration.namespace_id, []).append(hydration)

    batches: list[NamespaceHydrationBatch] = []
    histories_by_key: dict[tuple[str, str], HydratedSourceHistory] = {}
    for namespace_id, inputs in grouped.items():
        first = inputs[0]
        _require(
            all(
                (row.store_dir, row.database_sha256, row.index_sha256)
                == (first.store_dir, first.database_sha256, first.index_sha256)
                for row in inputs
            ),
            "one namespace changed immutable store coordinates",
        )
        memberships: list[Any] = []
        observed: dict[str, str] = {}
        for row in inputs:
            for membership in row.memberships:
                digest = _membership_identity(membership)
                previous = observed.setdefault(membership.source_id, digest)
                _require(
                    previous == digest,
                    "one namespaced source changed frozen membership",
                )
                if previous == digest and not any(
                    value.source_id == membership.source_id
                    for value in memberships
                ):
                    memberships.append(membership)
        merged = LockedSourceHydrationInput(
            namespace_id,
            first.store_dir,
            first.database_sha256,
            first.index_sha256,
            tuple(memberships),
        )
        merged.revalidate_store_bytes()
        database = merged.open_read_only_database()
        try:
            histories = hydrate_source_histories(
                database,
                merged.memberships,
                namespace_id=namespace_id,
                revalidate_store_bytes=merged.revalidate_store_bytes,
            )
        finally:
            database.close()
        merged.revalidate_store_bytes()
        for history in histories:
            key = (namespace_id, history.source_id)
            _require(key not in histories_by_key, "hydrated source repeated globally")
            histories_by_key[key] = history
        body = {
            "database_read_only": True,
            "database_sha256": merged.database_sha256,
            "history_receipt_sha256s": [row.receipt_sha256 for row in histories],
            "index_sha256": merged.index_sha256,
            "namespace_id": namespace_id,
            "namespace_read_count": 1,
            "namespace_scan_count": 1,
            "source_ids": [row.source_id for row in histories],
            "store_bytes_revalidated_before_after": True,
            "store_dir": str(merged.store_dir),
        }
        batches.append(
            NamespaceHydrationBatch(
                namespace_id,
                merged.store_dir,
                merged.database_sha256,
                merged.index_sha256,
                tuple(row.source_id for row in histories),
                tuple(row.receipt_sha256 for row in histories),
                identity_sha256({"format": f"{FORMAT}-namespace-batch", **body}),
            )
        )
    return tuple(batches), histories_by_key


def hydrate_namespace_batches(
    question_rounds: Sequence[tuple[Any, SourceGateRound]],
) -> tuple[
    tuple[NamespaceHydrationBatch, ...],
    dict[tuple[str, str], HydratedSourceHistory],
]:
    """Public locked namespace-batched hydration seam for later rounds."""

    return _hydrate_namespace_batches(question_rounds)


def build_locked_base_round(
    source_population: LockedSourceGateAdapterPopulation,
    *,
    query_adapter: QueryMapSourceGateAdapterPlane | None = None,
    max_window_tokens: int = DEFAULT_HISTORY_WINDOW_TOKEN_CAP,
    prehydrated: tuple[
        tuple[NamespaceHydrationBatch, ...],
        Mapping[tuple[str, str], HydratedSourceHistory],
    ]
    | None = None,
) -> LockedAdaptiveBasePlan:
    """Start every gate, batch hydrate, window, bind, and preflight prompts."""

    if type(source_population) is not LockedSourceGateAdapterPopulation:
        raise TypeError(
            "source_population must be an exact LockedSourceGateAdapterPopulation"
        )
    if query_adapter is not None and type(query_adapter) is not QueryMapSourceGateAdapterPlane:
        raise TypeError("query_adapter must be an exact adapter plane or None")
    _require(
        type(max_window_tokens) is int and max_window_tokens > 0,
        "history window cap changed",
    )
    rounds = {
        question.plan.question_id: start_source_gate(question.plan)
        for question in source_population.questions
    }
    if prehydrated is None:
        hydration_batches, histories_by_key = _hydrate_namespace_batches(
            tuple(
                (question, rounds[question.plan.question_id])
                for question in source_population.questions
            )
        )
    else:
        hydration_batches, supplied = prehydrated
        _require(
            type(hydration_batches) is tuple
            and all(type(row) is NamespaceHydrationBatch for row in hydration_batches),
            "prehydrated namespace batches changed type",
        )
        histories_by_key = dict(supplied)
    rows: list[BaseQuestionSourceMap] = []
    routes: Counter[str] = Counter()
    for question in source_population.questions:
        gate = question.plan
        round_plan = rounds[gate.question_id]
        if not round_plan.selected_candidates:
            routes[gate.route.style.value] += 1
            continue
        hydration_input = question.hydration_input(round_plan)
        histories = tuple(
            histories_by_key[(hydration_input.namespace_id, row.source_id)]
            for row in hydration_input.memberships
        )
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
                    max_window_tokens,
                    min(max_window_tokens, 4_800),
                    min(max_window_tokens, 4_000),
                    min(max_window_tokens, 3_200),
                    min(max_window_tokens, 2_400),
                    min(max_window_tokens, 1_600),
                    min(max_window_tokens, 800),
                    largest_chunk,
                )
                if value >= largest_chunk
            )
        )
        last_overflow: SourceHistoryMapperError | None = None
        for cap in candidate_caps:
            hydration_plan = plan_source_history_hydration(
                gate.parent,
                selections=round_plan.selections,
                histories=histories,
                max_window_tokens=cap,
            )
            mapping_plan = build_question_bound_mapping_plan(
                gate,
                round_plan,
                hydration_plan,
                mapper_contract_sha256=MAPPER_CONTRACT_SHA256,
            )
            try:
                mapper_preflight = build_source_history_mapper_preflight(
                    hydration_plan, mapping_plan
                )
            except SourceHistoryMapperError as exc:
                if "envelope overflow" not in str(exc):
                    raise
                last_overflow = exc
                continue
            break
        else:
            assert last_overflow is not None
            raise last_overflow
        _require(
            mapper_preflight.maximum_combined_token_proxy
            <= HARD_CONTEXT_TOKEN_CAP,
            "mapper prompt and output reserve escaped the hard envelope",
        )
        routes[gate.route.style.value] += 1
        rows.append(
            BaseQuestionSourceMap(
                question.ordinal,
                gate.question_id,
                round_plan,
                hydration_plan,
                mapping_plan,
                mapper_preflight,
            )
        )
    submitted = tuple(
        _plain_messages(prompt.messages)
        for row in rows
        for prompt in row.mapper_preflight.prompt_rows
        if prompt.disposition is WorkDisposition.NEW_CALL
    )
    _require(bool(submitted), "locked base round has no provider work")
    provider_population = preflight_fast_completion_prompts(
        submitted, max_prompt_tokens=MAX_PROMPT_TOKENS
    )
    _require(
        provider_population.logical_prompt_count
        == provider_population.unique_prompt_count
        == sum(row.mapping_plan.planned_provider_calls for row in rows),
        "global mapper prompts are not one-to-one with authorized physical work",
    )
    result = LockedAdaptiveBasePlan(
        query_adapter,
        source_population,
        hydration_batches,
        tuple(rows),
        tuple(sorted(routes.items())),
        provider_population,
    )
    return result


def _source_pins() -> LockedSourceGatePins:
    return LockedSourceGatePins()


def load_locked_base_round(
    *,
    max_concurrency: int,
    gateway_url: str,
    policy: SourceGatePolicy | None = None,
    obligation_mode: str = CONSOLIDATED_OBLIGATION_MODE,
    state_chain_profile: str = STRICT_STATE_CHAIN_PROFILE,
) -> LockedAdaptiveBasePlan:
    _query_run, _map_plan, _map_plane, adapter = load_locked_query_map(
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
        obligation_mode=obligation_mode,
        state_chain_profile=state_chain_profile,
    )
    activations = activation_inputs_from_query_map(adapter)
    source_population = load_locked_source_gate_adapter(
        activations,
        pins=_source_pins(),
        policy=policy,
    )
    _require(
        len(source_population.questions) == len(activations),
        "locked source-gate adapter changed activation coverage",
    )
    return build_locked_base_round(
        source_population,
        query_adapter=adapter,
    )


def _physical_prompt_rows(plan: LockedAdaptiveBasePlan) -> list[dict[str, Any]]:
    return [row.projection(include_messages=True) for row in plan.all_prompt_rows]


def preflight_projection(
    plan: LockedAdaptiveBasePlan,
    *,
    gateway_url: str,
    model: str,
    max_concurrency: int,
) -> dict[str, Any]:
    adapter = plan.query_adapter
    activated_rows = () if adapter is None else adapter.activated_rows
    no_op_rows = () if adapter is None else adapter.no_op_rows
    question_by_id = {
        question.plan.question_id: question
        for question in plan.source_population.questions
    }
    policies = {row.plan.policy.receipt_sha256: row.plan.policy for row in question_by_id.values()}
    _require(len(policies) == 1, "one preflight mixed source-gate policies")
    policy = next(iter(policies.values()))
    selected_source_keys = {
        (selection.namespace_id, selection.source_id)
        for question in plan.questions
        for selection in question.gate_round.selections
    }
    questions: list[dict[str, Any]] = []
    for row in plan.questions:
        gate = question_by_id[row.question_id].plan
        value = row.projection()
        value["activation_receipt_sha256"] = gate.activation.receipt_sha256
        value["parent_identity_sha256"] = gate.parent.identity_sha256
        value["route"] = gate.route.style.value
        value["source_candidate_count"] = len(gate.candidates)
        questions.append(value)
    payload: dict[str, Any] = {
        "activated_question_count": (
            len(activated_rows)
            if adapter is not None
            else len(plan.source_population.questions)
        ),
        "activation_row_receipt_sha256s": [
            row.receipt_sha256 for row in activated_rows
        ],
        "as_of_turn_semantics": "terminal_benchmark_tick_zero",
        "deferred_physical_prompt_count": sum(
            row.disposition is WorkDisposition.DEFERRED
            for row in plan.all_prompt_rows
        ),
        "direct_evidence_exclusion_performed": False,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_context_token_cap": HARD_CONTEXT_TOKEN_CAP,
        "logical_selection_count": sum(
            len(row.gate_round.selections) for row in plan.questions
        ),
        "logical_selection_dedup_performed": False,
        "logical_window_count": sum(
            len(row.hydration_plan.windows) for row in plan.questions
        ),
        "map_preflight_sha256": EXPECTED_MAP_PREFLIGHT_SHA256,
        "map_run_sha256": EXPECTED_MAP_RUN_SHA256,
        "map_runtime_ledger_sha256": EXPECTED_MAP_RUNTIME_SHA256,
        "mapped_activated_question_count": len(plan.questions),
        "mapper_contract_sha256": MAPPER_CONTRACT_SHA256,
        "max_concurrency": max_concurrency,
        "maximum_prompt_and_output_token_envelope": max(
            row.mapper_preflight.maximum_combined_token_proxy
            for row in plan.questions
        ),
        "model": model,
        "namespace_batch_count": len(plan.hydration_batches),
        "namespace_hydration_batches": [
            row.projection() for row in plan.hydration_batches
        ],
        "no_op_question_count": len(no_op_rows),
        "no_op_row_receipt_sha256s": [row.receipt_sha256 for row in no_op_rows],
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "obligation_compilation_mode": (
            None if adapter is None else adapter.obligation_compilation_mode
        ),
        "state_chain_profile": (
            None if adapter is None else adapter.state_chain_profile
        ),
        "physical_prompt_count": len(plan.all_prompt_rows),
        "physical_prompt_rows": _physical_prompt_rows(plan),
        "physical_work_reuse_across_logical_aliases": True,
        "post_map_dedup_performed": False,
        "provider_calls": 0,
        "provider_population": plan.provider_population.model_dump(),
        "query_map_adapter_receipt_sha256": (
            None if adapter is None else adapter.receipt_sha256
        ),
        "query_run_sha256": payload_cli.EXPECTED_QUERY_RUN_SHA256,
        "question_count": EXPECTED_QUESTION_COUNT if adapter is not None else len(plan.questions),
        "question_plans": questions,
        "required_authorized_provider_calls": plan.required_provider_calls,
        "retained_transformer_token_state_bytes": 0,
        "route_counts": dict(plan.route_counts),
        "source_gate_population_receipt_sha256": plan.source_population.receipt_sha256,
        "source_gate_policy": policy.projection(),
        "source_gate_policy_receipt_sha256": policy.receipt_sha256,
        "source_input_artifacts": [
            row.projection() for row in plan.source_population.source_artifacts
        ],
        "source_mapper_preflight_receipt_sha256s": [
            row.mapper_preflight.receipt_sha256 for row in plan.questions
        ],
        "stage": "base_round_preflight",
        "hydration_cache_unique_namespaced_source_count": len(
            {
                (batch.namespace_id, source_id)
                for batch in plan.hydration_batches
                for source_id in batch.source_ids
            }
        ),
        "unique_namespaced_source_count": len(selected_source_keys),
        "unique_physical_window_count": len(plan.all_prompt_rows),
        "zero_selection_activated_question_count": (
            len(plan.source_population.questions) - len(plan.questions)
        ),
        "zero_selection_activated_question_ids": [
            row.plan.question_id
            for row in plan.source_population.questions
            if row.plan.question_id not in {item.question_id for item in plan.questions}
        ],
    }
    _require(
        payload["maximum_prompt_and_output_token_envelope"]
        <= HARD_CONTEXT_TOKEN_CAP,
        "global mapper envelope exceeds 8K",
    )
    assert_gold_blind(payload, path="locked_adaptive_source_map_preflight")
    return payload


def _validate_provider_preflight(
    artifact: SealedArtifact,
) -> tuple[FastPromptPopulation, tuple[tuple[dict[str, str], ...], ...]]:
    payload = artifact.payload
    assert_gold_blind(payload, path="locked_adaptive_source_map_provider")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("stage") == "base_round_preflight"
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("map_preflight_sha256") == EXPECTED_MAP_PREFLIGHT_SHA256
        and payload.get("map_run_sha256") == EXPECTED_MAP_RUN_SHA256
        and payload.get("map_runtime_ledger_sha256") == EXPECTED_MAP_RUNTIME_SHA256
        and payload.get("query_run_sha256") == payload_cli.EXPECTED_QUERY_RUN_SHA256
        and payload.get("mapper_contract_sha256") == MAPPER_CONTRACT_SHA256
        and payload.get("work_manifest_name") == WORK_MANIFEST_NAME
        and type(payload.get("work_manifest_sha256")) is str
        and payload.get("fast_materialization_store_reads") == 0
        and payload.get("full_replay_revalidates_stores") is True,
        "source-map provider preflight changed locked parent or firewall",
    )
    adapter_by_profile = {
        (LEGACY_OBLIGATION_MODE, STRICT_STATE_CHAIN_PROFILE): (
            EXPECTED_LEGACY_QUERY_MAP_ADAPTER_SHA256
        ),
        (CONSOLIDATED_OBLIGATION_MODE, STRICT_STATE_CHAIN_PROFILE): (
            EXPECTED_CONSOLIDATED_QUERY_MAP_ADAPTER_SHA256
        ),
        (
            CONSOLIDATED_OBLIGATION_MODE,
            STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
        ): EXPECTED_CONSOLIDATED_STATE_CHAIN_AUTHORITY_ADAPTER_SHA256,
    }
    _require(
        payload.get("query_map_adapter_receipt_sha256")
        == adapter_by_profile.get(
            (
                payload.get("obligation_compilation_mode"),
                payload.get("state_chain_profile"),
            )
        ),
        "source-map obligation/state-chain profile adapter seal changed",
    )
    rows = payload.get("physical_prompt_rows")
    _require(type(rows) is list and bool(rows), "sealed source-map prompts are missing")
    prompts: list[tuple[dict[str, str], ...]] = []
    seen_work: set[str] = set()
    for raw in rows:
        _require(type(raw) is dict, "sealed source-map prompt row changed type")
        work_id = require_sha256(raw.get("physical_work_id"), "sealed physical work")
        _require(work_id not in seen_work, "sealed physical work repeats")
        seen_work.add(work_id)
        disposition = raw.get("disposition")
        _require(
            disposition in {value.value for value in WorkDisposition},
            "sealed source-map disposition changed",
        )
        messages = raw.get("messages")
        _require(
            type(messages) is list
            and len(messages) == 2
            and all(
                type(value) is dict
                and set(value) == {"role", "content"}
                and type(value.get("role")) is str
                and type(value.get("content")) is str
                for value in messages
            ),
            "sealed source-map messages changed",
        )
        _require(
            identity_sha256(messages) == raw.get("messages_sha256"),
            "sealed source-map messages hash changed",
        )
        if disposition == WorkDisposition.NEW_CALL.value:
            prompts.append(tuple(dict(value) for value in messages))
    _require(bool(prompts), "sealed source-map provider population is empty")
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_PROMPT_TOKENS
    )
    _require(
        population.model_dump() == payload.get("provider_population")
        and population.logical_prompt_count
        == population.unique_prompt_count
        == payload.get("required_authorized_provider_calls"),
        "sealed source-map provider population changed",
    )
    return population, tuple(prompts)


def _runtime(
    *,
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    checkpoint_dir: Path,
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
        "runtime configuration differs from sealed source-map preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "arm": "locked_adaptive_source_map_base_v1",
            "authorized_unique_calls": len(prompts),
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "mapper_contract_sha256": MAPPER_CONTRACT_SHA256,
            "preflight_artifact_sha256": artifact.sha256,
            "query_map_adapter_receipt_sha256": artifact.payload.get(
                "query_map_adapter_receipt_sha256"
            ),
        },
    )


def _journals_for_question(
    preflight: SourceMapperPreflight,
    batch: FastCompletionBatch,
) -> tuple[SourceMapperProviderJournal, ...]:
    _require(
        batch.provenance.retained_transformer_token_state_bytes == 0
        and not batch.provenance.persisted_transformer_token_state
        and batch.provenance.max_new_tokens == OUTPUT_TOKEN_RESERVE
        and batch.provenance.max_prompt_token_proxy == MAX_PROMPT_TOKENS,
        "global mapper completion batch retained transformer state",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    result: list[SourceMapperProviderJournal] = []
    for prompt in preflight.prompt_rows:
        if prompt.disposition is not WorkDisposition.NEW_CALL:
            continue
        record = records.get(prompt.messages_sha256)
        _require(record is not None, "global mapper journal lacks question prompt")
        assert record is not None
        result.append(
            SourceMapperProviderJournal(
                prompt.physical_work_id,
                prompt.prompt_id,
                prompt.messages_sha256,
                record.call_key_sha256,
                record.request_journal_sha256,
                record.response_journal_sha256,
                record.completion,
                record.completion_sha256,
                record.physical_call,
                record.checkpoint_hit,
                0,
            )
        )
    _require(
        tuple(row.physical_work_id for row in result)
        == tuple(
            row.physical_work_id
            for row in preflight.prompt_rows
            if row.disposition is WorkDisposition.NEW_CALL
        ),
        "question mapper journal order changed",
    )
    return tuple(result)


def provider_journals_for_question(
    preflight: SourceMapperPreflight,
    batch: FastCompletionBatch,
) -> tuple[SourceMapperProviderJournal, ...]:
    """Public exact adapter for one question's sealed mapper journals."""

    return _journals_for_question(preflight, batch)


def materialize_fast_question_plans(
    questions: tuple[FastMaterializationQuestionPlan, ...],
    batch: FastCompletionBatch,
) -> tuple[SourceMapperMaterialization, ...]:
    """Return typed mapper results from sealed plans and immutable journals."""

    _require(
        type(questions) is tuple
        and bool(questions)
        and all(type(row) is FastMaterializationQuestionPlan for row in questions),
        "fast materialization questions changed type",
    )
    _require(type(batch) is FastCompletionBatch, "fast completion batch changed type")
    return tuple(
        materialize_source_history_mapper(
            question.mapper_preflight,
            question.hydration_plan,
            question.mapping_plan,
            provider_journals=_journals_for_question(
                question.mapper_preflight, batch
            ),
        )
        for question in questions
    )


def materialization_projection(
    plan: LockedAdaptiveBasePlan,
    batch: FastCompletionBatch,
    *,
    preflight_artifact_sha256: str,
) -> dict[str, Any]:
    source_question_by_id = {
        row.plan.question_id: row for row in plan.source_population.questions
    }
    questions = tuple(
        FastMaterializationQuestionPlan(
            row.ordinal,
            row.question_id,
            source_question_by_id[row.question_id].direct_evidence,
            row.hydration_plan,
            row.mapping_plan,
            row.mapper_preflight,
        )
        for row in plan.questions
    )
    return _materialization_projection_from_questions(
        questions,
        batch,
        preflight_artifact_sha256=preflight_artifact_sha256,
        source_gate_population_receipt_sha256=(
            plan.source_population.receipt_sha256
        ),
    )


def _materialization_projection_from_questions(
    questions: tuple[FastMaterializationQuestionPlan, ...],
    batch: FastCompletionBatch,
    *,
    preflight_artifact_sha256: str,
    source_gate_population_receipt_sha256: str,
) -> dict[str, Any]:
    results = materialize_fast_question_plans(questions, batch)
    return _materialization_projection_from_results(
        questions,
        results,
        batch,
        preflight_artifact_sha256=preflight_artifact_sha256,
        source_gate_population_receipt_sha256=(
            source_gate_population_receipt_sha256
        ),
    )


def _materialization_projection_from_results(
    questions: tuple[FastMaterializationQuestionPlan, ...],
    results: tuple[SourceMapperMaterialization, ...],
    batch: FastCompletionBatch,
    *,
    preflight_artifact_sha256: str,
    source_gate_population_receipt_sha256: str,
) -> dict[str, Any]:
    require_sha256(preflight_artifact_sha256, "adaptive source-map preflight")
    require_sha256(
        source_gate_population_receipt_sha256, "source-gate population"
    )
    _require(
        type(results) is tuple
        and len(results) == len(questions)
        and all(type(row) is SourceMapperMaterialization for row in results),
        "typed materialization results changed population",
    )
    payload: dict[str, Any] = {
        "accepted_before_post_map_dedup_count": sum(
            row.accepted_before_post_map_dedup_count
            for result in results
            for row in result.work_results
        ),
        "format": MATERIALIZATION_FORMAT,
        "gold_loaded": False,
        "historical_checkpoint_hits": batch.usage.checkpoint_hits,
        "historical_physical_provider_calls": batch.usage.physical_calls,
        "materializations": [row.projection() for row in results],
        "post_map_dedup_performed": False,
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "provider_calls_during_materialization": 0,
        "provider_journal_receipt_sha256s": [
            journal.receipt_sha256
            for question in questions
            for journal in _journals_for_question(question.mapper_preflight, batch)
        ],
        "question_count": len(results),
        "rejected_item_count": sum(
            row.rejected_item_count
            for result in results
            for row in result.work_results
        ),
        "retained_transformer_token_state_bytes": 0,
        "source_gate_population_receipt_sha256": (
            source_gate_population_receipt_sha256
        ),
        "source_mapper_materialization_receipt_sha256s": [
            row.receipt_sha256 for row in results
        ],
    }
    assert_gold_blind(payload, path="locked_adaptive_source_map_materialization")
    return payload


def replay_typed_fast_materialization(
    preflight: SealedArtifact,
    work_manifest: SealedArtifact,
    terminal_materialization: SealedArtifact,
    batch: FastCompletionBatch,
    *,
    questions: tuple[FastMaterializationQuestionPlan, ...] | None = None,
) -> tuple[SourceMapperMaterialization, ...]:
    """Strict store-free loader for downstream typed post-map fact unions."""

    for value, label in (
        (preflight, "preflight"),
        (work_manifest, "work manifest"),
        (terminal_materialization, "terminal materialization"),
    ):
        _require(type(value) is SealedArtifact, f"typed replay {label} changed type")
    _require(
        preflight.payload.get("work_manifest_sha256") == work_manifest.sha256,
        "typed replay changed work-manifest binding",
    )
    source_receipt = require_sha256(
        preflight.payload.get("source_gate_population_receipt_sha256"),
        "typed replay source population",
    )
    if questions is None:
        questions = load_fast_materialization_manifest(
            work_manifest,
            expected_source_population_receipt_sha256=source_receipt,
        )
    else:
        _require(
            type(questions) is tuple
            and bool(questions)
            and all(
                type(row) is FastMaterializationQuestionPlan
                for row in questions
            ),
            "typed replay supplied questions changed type",
        )
    results = materialize_fast_question_plans(questions, batch)
    expected = _materialization_projection_from_results(
        questions,
        results,
        batch,
        preflight_artifact_sha256=preflight.sha256,
        source_gate_population_receipt_sha256=source_receipt,
    )
    _require(
        terminal_materialization.payload == expected,
        "typed fast materialization differs from terminal seal",
    )
    return results


def _parse_policy_spec(value: str) -> tuple[int, int, int]:
    try:
        fields = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "policy must be three comma-separated integer caps: D,P,G"
        ) from exc
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(
            "policy must be three comma-separated integer caps: D,P,G"
        )
    try:
        source_gate_policy(*fields)
    except LockedAdaptiveSourceMapError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return fields  # type: ignore[return-value]


def _policy_slug(caps: tuple[int, int, int]) -> str:
    return f"d{caps[0]}-p{caps[1]}-g{caps[2]}"


def _source_target_hit(
    question_id: str,
    target_id: str,
    source_ids: set[str],
) -> bool:
    """Match the registry's local source ID to its exact question-qualified ID."""

    return target_id in source_ids or f"{question_id}::{target_id}" in source_ids


def _publish_preflight_with_work_manifest(
    plan: LockedAdaptiveBasePlan,
    *,
    output_root: Path,
    gateway_url: str,
    model: str,
    max_concurrency: int,
) -> tuple[SealedArtifact, SealedArtifact]:
    work_manifest, _created = publish_sealed_json(
        output_root / WORK_MANIFEST_NAME,
        work_manifest_projection(plan),
    )
    payload = preflight_projection(
        plan,
        gateway_url=gateway_url,
        model=model,
        max_concurrency=max_concurrency,
    )
    payload.update(
        {
            "fast_materialization_store_reads": 0,
            "full_replay_revalidates_stores": True,
            "work_manifest_name": WORK_MANIFEST_NAME,
            "work_manifest_sha256": work_manifest.sha256,
        }
    )
    assert_gold_blind(payload, path="locked_adaptive_source_map_preflight")
    preflight, _created = publish_sealed_json(
        output_root / PREFLIGHT_NAME,
        payload,
    )
    return preflight, work_manifest


def _pareto_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Load/verify once, hydrate the policy union once, then seal each point."""

    caps_population = tuple(
        dict.fromkeys(
            args.policy
            or (
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (2, 0, 1),
            )
        )
    )
    _require(bool(caps_population), "Pareto preflight requires policies")
    policies = tuple(source_gate_policy(*caps) for caps in caps_population)
    _query_run, _map_plan, _map_plane, adapter = load_locked_query_map(
        max_concurrency=args.max_concurrency,
        gateway_url=args.gateway_url,
        obligation_mode=args.obligation_mode,
        state_chain_profile=getattr(
            args, "state_chain_profile", STRICT_STATE_CHAIN_PROFILE
        ),
    )
    activations = activation_inputs_from_query_map(adapter)
    # The expensive pinned-artifact and store verification happens once.
    source_plane = load_locked_source_gate_adapter(
        activations,
        pins=_source_pins(),
        policy=policies[0],
    )
    populations = tuple(
        source_plane
        if index == 0
        else repolicy_source_population(source_plane, policy)
        for index, policy in enumerate(policies)
    )
    question_rounds = tuple(
        (question, start_source_gate(question.plan))
        for population in populations
        for question in population.questions
    )
    shared_batches, shared_histories = _hydrate_namespace_batches(question_rounds)
    points: list[dict[str, Any]] = []
    for caps, population in zip(caps_population, populations, strict=True):
        plan = build_locked_base_round(
            population,
            query_adapter=adapter,
            prehydrated=(shared_batches, shared_histories),
        )
        point_root = Path(args.output_root) / _policy_slug(caps)
        artifact, work_manifest = _publish_preflight_with_work_manifest(
            plan,
            output_root=point_root,
            gateway_url=args.gateway_url,
            model=args.model,
            max_concurrency=args.max_concurrency,
        )
        points.append(
            {
                "activated_question_count": len(adapter.activated_rows),
                "caps": {
                    "direct": caps[0],
                    "guided": caps[2],
                    "partition": caps[1],
                },
                "logical_selection_count": sum(
                    len(row.gate_round.selections) for row in plan.questions
                ),
                "logical_window_count": sum(
                    len(row.hydration_plan.windows) for row in plan.questions
                ),
                "maximum_prompt_and_output_token_envelope": max(
                    row.mapper_preflight.maximum_combined_token_proxy
                    for row in plan.questions
                ),
                "mapped_activated_question_count": len(plan.questions),
                "physical_prompt_count": len(plan.all_prompt_rows),
                "preflight_path": artifact.path.as_posix(),
                "preflight_sha256": artifact.sha256,
                "required_authorized_provider_calls": plan.required_provider_calls,
                "source_gate_policy_receipt_sha256": (
                    population.questions[0].plan.policy.receipt_sha256
                ),
                "unique_namespaced_source_count": len(
                    {
                        (selection.namespace_id, selection.source_id)
                        for row in plan.questions
                        for selection in row.gate_round.selections
                    }
                ),
                "work_manifest_sha256": work_manifest.sha256,
                "zero_selection_activated_question_count": (
                    len(adapter.activated_rows) - len(plan.questions)
                ),
            }
        )
    manifest_payload = {
        "format": f"{FORMAT}-pareto-preflight",
        "gold_loaded": False,
        "map_preflight_sha256": EXPECTED_MAP_PREFLIGHT_SHA256,
        "map_run_sha256": EXPECTED_MAP_RUN_SHA256,
        "namespace_batch_count": len(shared_batches),
        "obligation_compilation_mode": adapter.obligation_compilation_mode,
        "state_chain_profile": adapter.state_chain_profile,
        "points": points,
        "provider_calls": 0,
        "query_map_adapter_receipt_sha256": adapter.receipt_sha256,
        "retained_transformer_token_state_bytes": 0,
        "shared_hydration_batch_receipt_sha256s": [
            row.receipt_sha256 for row in shared_batches
        ],
        "shared_hydration_unique_namespaced_source_count": len(shared_histories),
        "source_plane_loaded_once": True,
        "store_namespace_scan_count": len(shared_batches),
    }
    assert_gold_blind(
        manifest_payload, path="locked_adaptive_source_map_pareto"
    )
    manifest, _created = publish_sealed_json(
        Path(args.output_root) / PARETO_NAME, manifest_payload
    )
    return {
        "gold_loaded": False,
        "manifest": manifest.path.as_posix(),
        "manifest_sha256": manifest.sha256,
        "namespace_batch_count": len(shared_batches),
        "points": points,
        "provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "source_plane_loaded_once": True,
    }


def _pareto_posthoc_coverage(args: argparse.Namespace) -> dict[str, Any]:
    """Attach registered source-target coverage after structural seals exist."""

    root = Path(args.output_root)
    expected_pareto = require_sha256(
        args.expected_pareto_sha256, "expected source-map Pareto manifest"
    )
    structural = read_sealed_json(root / PARETO_NAME)
    _require(structural.sha256 == expected_pareto, "Pareto manifest changed")
    structural_payload = structural.payload
    assert_gold_blind(
        structural_payload, path="source_map_pareto_posthoc.structural"
    )
    _require(
        structural_payload.get("format") == f"{FORMAT}-pareto-preflight"
        and structural_payload.get("provider_calls") == 0,
        "posthoc coverage requires a sealed provider-free Pareto manifest",
    )
    target_path = (
        DEFAULT_POSTHOC_TARGET_PLAN
        if args.target_plan is None
        else Path(args.target_plan)
    )
    target = read_sealed_json(target_path)
    expected_target = (
        EXPECTED_POSTHOC_TARGET_PLAN_SHA256
        if args.expected_target_plan_sha256 is None
        else require_sha256(
            args.expected_target_plan_sha256, "expected target-plan artifact"
        )
    )
    _require(target.sha256 == expected_target, "posthoc target plan changed")
    target_payload = target.payload
    _require(
        target_payload.get("format")
        == "memory-condense-retrieval-target-owner-plan-v1"
        and target_payload.get("gold_target_tags_posthoc_only") is True
        and target_payload.get("runtime_use_forbidden") is True
        and target_payload.get("provider_calls") == 0,
        "target plan is not sealed posthoc-only input",
    )
    unsigned_target = dict(target_payload)
    target_self_sha = unsigned_target.pop("plan_sha256", None)
    _require(
        require_sha256(target_self_sha, "target-plan self seal")
        == identity_sha256(unsigned_target),
        "target-plan self seal changed",
    )
    map_plan = map_cli._load_map_plan(  # noqa: SLF001 - pinned structural replay
        _sealed_parent_args(
            max_concurrency=args.max_concurrency,
            gateway_url=args.gateway_url,
        )
    )
    ordered_questions = tuple(
        (
            row.ordinal,
            row.direct_plan_row.adapter.source.packet.question_id,
        )
        for row in map_plan.rows
    )
    _require(
        tuple(
            (row.get("ordinal"), row.get("question_id"))
            for row in target_payload.get("ordered_question_keys", ())
            if type(row) is dict
        )
        == ordered_questions,
        "target-plan question order changed",
    )
    targets: list[tuple[int, str, str]] = []
    raw_targets = target_payload.get("desired_targets")
    _require(type(raw_targets) is list, "target-plan targets changed type")
    for raw in raw_targets:
        _require(type(raw) is dict, "target-plan target changed type")
        if raw.get("target_kind") != "source_id":
            continue
        ordinal = _exact_int(raw.get("ordinal"), "source target ordinal")
        source_id = require_text(raw.get("target_id"), "source target ID")
        _require(
            ordinal < len(ordered_questions)
            and raw.get("question_id") == ordered_questions[ordinal][1],
            "source target escaped question order",
        )
        require_sha256(raw.get("target_sha256"), "source target seal")
        targets.append((ordinal, ordered_questions[ordinal][1], source_id))
    _require(
        len(targets) == target_payload.get("desired_source_target_count")
        and len(targets) == len(set(targets)),
        "source-target count changed or repeats",
    )
    baseline: dict[int, set[str]] = {}
    for row in map_plan.rows:
        packet = row.direct_plan_row.adapter.source.packet
        evidence = (*packet.protected_evidence, *row.retained_query_delta)
        baseline[row.ordinal] = {item.source_id for item in evidence}
    baseline_covered = sum(
        _source_target_hit(question_id, source_id, baseline[ordinal])
        for ordinal, question_id, source_id in targets
    )
    raw_points = structural_payload.get("points")
    _require(type(raw_points) is list and bool(raw_points), "Pareto points missing")
    coverage_rows: list[dict[str, Any]] = []
    for point in raw_points:
        _require(type(point) is dict, "Pareto point changed type")
        caps = point.get("caps")
        _require(type(caps) is dict, "Pareto caps changed type")
        slug = _policy_slug(
            (
                _exact_int(caps.get("direct"), "Pareto direct cap"),
                _exact_int(caps.get("partition"), "Pareto partition cap"),
                _exact_int(caps.get("guided"), "Pareto guided cap"),
            )
        )
        point_root = root / slug
        point_preflight = read_sealed_json(point_root / PREFLIGHT_NAME)
        work_manifest = read_sealed_json(point_root / WORK_MANIFEST_NAME)
        _require(
            point_preflight.sha256 == point.get("preflight_sha256")
            and work_manifest.sha256 == point.get("work_manifest_sha256")
            and point_preflight.payload.get("work_manifest_sha256")
            == work_manifest.sha256,
            "posthoc point artifact binding changed",
        )
        selected = {ordinal: set(values) for ordinal, values in baseline.items()}
        raw_questions = work_manifest.payload.get("questions")
        _require(type(raw_questions) is list, "point work questions changed type")
        for question in raw_questions:
            _require(type(question) is dict, "point work question changed type")
            ordinal = _exact_int(question.get("ordinal"), "point work ordinal")
            hydration = question.get("hydration_plan")
            _require(type(hydration) is dict, "point hydration changed type")
            selections = hydration.get("selections")
            _require(type(selections) is list, "point selections changed type")
            for selection in selections:
                _require(type(selection) is dict, "point selection changed type")
                selected[ordinal].add(
                    require_text(selection.get("source_id"), "selected source")
                )
        covered = sum(
            _source_target_hit(question_id, source_id, selected[ordinal])
            for ordinal, question_id, source_id in targets
        )
        coverage_rows.append(
            {
                "baseline_covered_source_target_count": baseline_covered,
                "caps": dict(caps),
                "covered_source_target_count": covered,
                "incremental_covered_source_target_count": (
                    covered - baseline_covered
                ),
                "preflight_sha256": point_preflight.sha256,
                "required_authorized_provider_calls": point.get(
                    "required_authorized_provider_calls"
                ),
                "source_target_coverage": {
                    "denominator": len(targets),
                    "numerator": covered,
                },
                "uncovered_source_target_count": len(targets) - covered,
                "work_manifest_sha256": work_manifest.sha256,
            }
        )
    for row in coverage_rows:
        row["pareto_on_provider_calls"] = not any(
            other["covered_source_target_count"]
            >= row["covered_source_target_count"]
            and other["required_authorized_provider_calls"]
            <= row["required_authorized_provider_calls"]
            and (
                other["covered_source_target_count"]
                > row["covered_source_target_count"]
                or other["required_authorized_provider_calls"]
                < row["required_authorized_provider_calls"]
            )
            for other in coverage_rows
        )
    payload = {
        "coverage_call_pareto": coverage_rows,
        "format": f"{FORMAT}-pareto-posthoc-coverage-v2",
        "source_target_match_rule": "exact_or_question_id_double_colon_qualified_v1",
        "posthoc_analysis_only": True,
        "provider_calls": 0,
        "registered_source_target_count": len(targets),
        "runtime_use_forbidden": True,
        "selection_and_routing_frozen_before_target_plan_load": True,
        "structural_pareto_manifest_sha256": structural.sha256,
        "target_plan_artifact_sha256": target.sha256,
        "target_plan_loaded_after_structural_selection": True,
        "target_plan_self_sha256": target_self_sha,
    }
    result, _created = publish_sealed_json(root / PARETO_COVERAGE_NAME, payload)
    return {
        "artifact": result.path.as_posix(),
        "artifact_sha256": result.sha256,
        "coverage_call_pareto": coverage_rows,
        "posthoc_analysis_only": True,
        "provider_calls": 0,
    }


def _build_and_publish_preflight(args: argparse.Namespace) -> tuple[LockedAdaptiveBasePlan, SealedArtifact]:
    policy = source_gate_policy(
        args.direct_base_cap,
        args.partition_base_cap,
        args.guided_base_cap,
    )
    plan = load_locked_base_round(
        max_concurrency=args.max_concurrency,
        gateway_url=args.gateway_url,
        policy=policy,
        obligation_mode=args.obligation_mode,
        state_chain_profile=getattr(
            args, "state_chain_profile", STRICT_STATE_CHAIN_PROFILE
        ),
    )
    artifact, _work_manifest = _publish_preflight_with_work_manifest(
        plan,
        output_root=Path(args.output_root),
        gateway_url=args.gateway_url,
        model=args.model,
        max_concurrency=args.max_concurrency,
    )
    return plan, artifact


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    plan, artifact = _build_and_publish_preflight(args)
    return {
        "activated_question_count": artifact.payload["activated_question_count"],
        "artifact": artifact.path.as_posix(),
        "gold_loaded": False,
        "logical_selection_count": sum(
            len(row.gate_round.selections) for row in plan.questions
        ),
        "logical_window_count": sum(
            len(row.hydration_plan.windows) for row in plan.questions
        ),
        "maximum_prompt_and_output_token_envelope": max(
            row.mapper_preflight.maximum_combined_token_proxy
            for row in plan.questions
        ),
        "mapped_activated_question_count": len(plan.questions),
        "namespace_batch_count": len(plan.hydration_batches),
        "physical_prompt_count": len(plan.all_prompt_rows),
        "preflight_sha256": artifact.sha256,
        "provider_calls": 0,
        "required_authorized_provider_calls": plan.required_provider_calls,
        "retained_transformer_token_state_bytes": 0,
        "route_counts": dict(plan.route_counts),
        "unique_namespaced_source_count": sum(
            len(row.source_ids) for row in plan.hydration_batches
        ),
    }


def _read_expected_preflight(args: argparse.Namespace) -> tuple[SealedArtifact, tuple[tuple[dict[str, str], ...], ...]]:
    expected = require_sha256(
        args.expected_preflight_sha256, "expected adaptive source-map preflight"
    )
    artifact = read_sealed_json(Path(args.output_root) / PREFLIGHT_NAME)
    _require(artifact.sha256 == expected, "adaptive source-map preflight changed")
    expected_policy = source_gate_policy(
        args.direct_base_cap,
        args.partition_base_cap,
        args.guided_base_cap,
    )
    _require(
        artifact.payload.get("source_gate_policy_receipt_sha256")
        == expected_policy.receipt_sha256,
        "CLI D/P/G base caps differ from sealed source-map policy",
    )
    _population, prompts = _validate_provider_preflight(artifact)
    return artifact, prompts


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts = _read_expected_preflight(args)
    required = len(prompts)
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == required,
        f"provider-run requires exact authorization for {required} calls",
    )
    # Authorization and the complete sealed gold-blind prompt population are
    # checked before environment loading, client creation, or checkpoint I/O.
    load_dotenv()
    api_key = os.environ.get(args.api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = provider_runtime.make_provider_client(api_key, args.gateway_url)
    runtime = _runtime(
        artifact=artifact,
        prompts=prompts,
        checkpoint_dir=Path(args.output_root) / CHECKPOINT_DIR_NAME,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
        client=client,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
    }


def _journal_batch(
    args: argparse.Namespace, artifact: SealedArtifact, prompts: tuple[tuple[dict[str, str], ...], ...]
) -> FastCompletionBatch:
    runtime = _runtime(
        artifact=artifact,
        prompts=prompts,
        checkpoint_dir=Path(args.output_root) / CHECKPOINT_DIR_NAME,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
        client=None,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def load_typed_materialization_root(
    output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_materialization_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    direct_base_cap: int,
    partition_base_cap: int,
    guided_base_cap: int,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[FastMaterializationQuestionPlan, ...],
    tuple[SourceMapperMaterialization, ...],
]:
    """Load a complete typed source-map run without stores or provider I/O."""

    loaded = load_typed_materialization_root_with_batch(
        output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_materialization_sha256=expected_materialization_sha256,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        direct_base_cap=direct_base_cap,
        partition_base_cap=partition_base_cap,
        guided_base_cap=guided_base_cap,
    )
    return loaded[:-1]


def load_typed_materialization_root_with_batch(
    output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_materialization_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    direct_base_cap: int,
    partition_base_cap: int,
    guided_base_cap: int,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[FastMaterializationQuestionPlan, ...],
    tuple[SourceMapperMaterialization, ...],
    FastCompletionBatch,
]:
    """Load typed results plus their checkpoint-only completion population.

    This lets a later sealed round reuse an already validated physical mapper
    result without reopening source stores or privately parsing journals.
    """

    root = Path(output_root)
    expected_preflight = require_sha256(
        expected_preflight_sha256, "expected adaptive source-map preflight"
    )
    expected_terminal = require_sha256(
        expected_materialization_sha256,
        "expected adaptive source-map materialization",
    )
    preflight = read_sealed_json(root / PREFLIGHT_NAME)
    _require(preflight.sha256 == expected_preflight, "typed preflight changed")
    policy = source_gate_policy(
        direct_base_cap, partition_base_cap, guided_base_cap
    )
    _require(
        preflight.payload.get("source_gate_policy_receipt_sha256")
        == policy.receipt_sha256,
        "typed loader D/P/G caps differ from sealed policy",
    )
    _population, prompts = _validate_provider_preflight(preflight)
    work_manifest = read_sealed_json(root / WORK_MANIFEST_NAME)
    _require(
        work_manifest.sha256 == preflight.payload.get("work_manifest_sha256"),
        "typed loader work manifest changed",
    )
    source_receipt = require_sha256(
        preflight.payload.get("source_gate_population_receipt_sha256"),
        "typed loader source population",
    )
    questions = load_fast_materialization_manifest(
        work_manifest,
        expected_source_population_receipt_sha256=source_receipt,
    )
    terminal = read_sealed_json(root / MATERIALIZATION_NAME)
    _require(terminal.sha256 == expected_terminal, "typed materialization changed")
    runtime = _runtime(
        artifact=preflight,
        prompts=prompts,
        checkpoint_dir=root / CHECKPOINT_DIR_NAME,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        client=None,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.checkpoint_hits == len(prompts)
        and batch.usage.physical_calls == 0,
        "typed loader requires a complete checkpoint-only population",
    )
    materializations = replay_typed_fast_materialization(
        preflight,
        work_manifest,
        terminal,
        batch,
        questions=questions,
    )
    return (
        preflight,
        work_manifest,
        terminal,
        questions,
        materializations,
        batch,
    )


def _read_fast_work_manifest(
    args: argparse.Namespace,
    preflight: SealedArtifact,
) -> tuple[SealedArtifact, tuple[FastMaterializationQuestionPlan, ...]]:
    expected = require_sha256(
        preflight.payload.get("work_manifest_sha256"),
        "expected source-map work manifest",
    )
    manifest = read_sealed_json(Path(args.output_root) / WORK_MANIFEST_NAME)
    _require(manifest.sha256 == expected, "source-map work manifest changed")
    source_population = require_sha256(
        preflight.payload.get("source_gate_population_receipt_sha256"),
        "preflight source population",
    )
    questions = load_fast_materialization_manifest(
        manifest,
        expected_source_population_receipt_sha256=source_population,
    )
    _require(
        len(questions)
        == preflight.payload.get("mapped_activated_question_count")
        and [row.mapper_preflight.receipt_sha256 for row in questions]
        == preflight.payload.get("source_mapper_preflight_receipt_sha256s")
        and [
            prompt.projection(include_messages=True)
            for row in questions
            for prompt in row.mapper_preflight.prompt_rows
        ]
        == preflight.payload.get("physical_prompt_rows"),
        "fast work manifest differs from sealed provider preflight",
    )
    sealed_questions = preflight.payload.get("question_plans")
    _require(
        type(sealed_questions) is list
        and tuple(
            (row.ordinal, row.question_id) for row in questions
        )
        == tuple(
            (value.get("ordinal"), value.get("question_id"))
            for value in sealed_questions
            if type(value) is dict
        ),
        "fast work manifest changed question order",
    )
    return manifest, questions


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts = _read_expected_preflight(args)
    manifest, questions = _read_fast_work_manifest(args, artifact)
    batch = _journal_batch(args, artifact, prompts)
    _require(
        batch.usage.checkpoint_hits == len(prompts)
        and batch.usage.physical_calls == 0,
        "materialization requires a complete checkpoint-only population",
    )
    payload = _materialization_projection_from_questions(
        questions,
        batch,
        preflight_artifact_sha256=artifact.sha256,
        source_gate_population_receipt_sha256=require_sha256(
            artifact.payload.get("source_gate_population_receipt_sha256"),
            "preflight source population",
        ),
    )
    result, created = publish_sealed_json(
        Path(args.output_root) / MATERIALIZATION_NAME, payload
    )
    return {
        "accepted_before_post_map_dedup_count": payload[
            "accepted_before_post_map_dedup_count"
        ],
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "materialization_sha256": result.sha256,
        "physical_provider_calls": 0,
        "store_reads_during_materialization": 0,
        "terminal_materialization_replayed": not created,
        "work_manifest_sha256": manifest.sha256,
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    expected_materialization = require_sha256(
        args.expected_materialization_sha256,
        "expected adaptive source-map materialization",
    )
    artifact, prompts = _read_expected_preflight(args)
    plan, rebuilt = _build_and_publish_preflight(args)
    _require(rebuilt.sha256 == artifact.sha256, "replay preflight changed")
    batch = _journal_batch(args, artifact, prompts)
    _require(
        batch.usage.physical_calls == 0
        and batch.usage.checkpoint_hits == len(prompts),
        "replay requires a checkpoint-only population",
    )
    payload = materialization_projection(
        plan, batch, preflight_artifact_sha256=artifact.sha256
    )
    terminal = read_sealed_json(Path(args.output_root) / MATERIALIZATION_NAME)
    _require(
        terminal.sha256 == expected_materialization
        and terminal.payload == payload,
        "adaptive source-map materialization replay changed bytes",
    )
    replay_payload = {
        "byte_identical": True,
        "expected_materialization_sha256": expected_materialization,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "preflight_artifact_sha256": artifact.sha256,
        "provider_calls_during_replay": 0,
        "replayed_materialization_sha256": terminal.sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, replay_payload
    )
    return {
        "byte_identical": True,
        "gold_loaded": False,
        "materialization_sha256": terminal.sha256,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gateway-url", default=provider_runtime.DEFAULT_GATEWAY_URL)
    parser.add_argument("--model", default=provider_runtime.DEFAULT_TERRA_GATEWAY_MODEL)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--direct-base-cap", type=int, default=DEFAULT_DIRECT_BASE_CAP
    )
    parser.add_argument(
        "--partition-base-cap", type=int, default=DEFAULT_PARTITION_BASE_CAP
    )
    parser.add_argument(
        "--guided-base-cap", type=int, default=DEFAULT_GUIDED_BASE_CAP
    )
    parser.add_argument(
        "--obligation-mode",
        choices=sorted(OBLIGATION_MODES),
        default=CONSOLIDATED_OBLIGATION_MODE,
    )
    parser.add_argument(
        "--state-chain-profile",
        choices=sorted(STATE_CHAIN_PROFILES),
        default=STRICT_STATE_CHAIN_PROFILE,
        help=(
            "sealed handling for intentionally unsubmitted state-chain maps; "
            "direct authority is opt-in"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _common(preflight)
    pareto = commands.add_parser("pareto-preflight")
    _common(pareto)
    pareto.add_argument(
        "--policy",
        action="append",
        type=_parse_policy_spec,
        default=[],
        metavar="D,P,G",
        help=(
            "repeatable source base-cap point; defaults to isolated D/P/G, "
            "D+G, D+P+G, and D2+G1"
        ),
    )
    coverage = commands.add_parser("pareto-posthoc-coverage")
    coverage.add_argument("--output-root", type=Path, required=True)
    coverage.add_argument("--expected-pareto-sha256", required=True)
    coverage.add_argument("--target-plan", type=Path)
    coverage.add_argument("--expected-target-plan-sha256")
    coverage.add_argument("--gateway-url", default=provider_runtime.DEFAULT_GATEWAY_URL)
    coverage.add_argument("--max-concurrency", type=int, default=4)
    provider = commands.add_parser("provider-run")
    _common(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=provider_runtime.DEFAULT_API_KEY_ENV)
    materialize = commands.add_parser("materialize")
    _common(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)
    replay = commands.add_parser("replay")
    _common(replay)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-materialization-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = _preflight(args)
    elif args.command == "pareto-preflight":
        result = _pareto_preflight(args)
    elif args.command == "pareto-posthoc-coverage":
        result = _pareto_posthoc_coverage(args)
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
    "EXPECTED_ACTIVATION_COUNT",
    "EXPECTED_MAP_PREFLIGHT_SHA256",
    "EXPECTED_MAP_RUN_SHA256",
    "EXPECTED_MAP_RUNTIME_SHA256",
    "EXPECTED_QUERY_MAP_ADAPTER_SHA256",
    "FastMaterializationQuestionPlan",
    "LockedAdaptiveBasePlan",
    "LockedAdaptiveSourceMapError",
    "NamespaceHydrationBatch",
    "activation_inputs_from_query_map",
    "build_locked_base_round",
    "hydrate_namespace_batches",
    "load_locked_base_round",
    "load_fast_materialization_manifest",
    "main",
    "load_typed_materialization_root",
    "load_typed_materialization_root_with_batch",
    "materialize_fast_question_plans",
    "materialization_projection",
    "preflight_projection",
    "provider_journals_for_question",
    "replay_typed_fast_materialization",
    "work_manifest_projection",
]
