#!/usr/bin/env python3
"""Provider-free confirmation query/partition/guided source plane.

This is the executable bridge between the authenticated confirmation query
artifacts and the adaptive source-map stages.  It deliberately reuses the
matched-eval query-guided, query-repack, partition-scan-v2, query-map, and
locked source-gate implementations.  It accepts no gold, target, prediction,
judge, provider client, or validation-population coordinate.

The partition eligibility decision is compiled from the dated question only:
``requires_temporal_metadata OR requires_complete_frontier``.  Every store is
verified before and after construction and is opened read-only once per
namespace for the partition generation.  All source collapse happens after
the underlying span selection lifecycle has been sealed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from memory_condense.persistence.db import Database
from tools._routed_repair_routing import RoutedRepairReceipt, route_question
from tools.confirmation_query_expansion_adapter import (
    ConfirmationQueryExpansionContext,
)
from tools.confirmation_query_artifacts import VerifiedQueryExpansionArtifacts
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    ArtifactRef,
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.locked_source_gate_adapter import (
    DIRECT_STREAM_PROFILE_REPACK_V2,
    DIRECT_STREAM_PROFILE_V1,
    LockedSourceGateActivationInput,
    LockedSourceGateAdapterPopulation,
    VerifiedLockedSourceGateRow,
    _candidate_sources,
    _direct_refs,
    _repack_direct_frontier,
    build_locked_source_gate_adapter,
    locked_activation_input_from_query_map_adapter,
    project_locked_lane_source_stream,
)
from tools.matched_eval.partition_scan_v2 import (
    PartitionScanV2Generation,
    construct_partition_scan_v2_question,
    project_partition_scan_v2_generation,
)
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    EvidenceMapPlan,
    VerifiedEvidenceMapPlane,
)
from tools.matched_eval.query_expansion_repack_v2 import (
    QueryExpansionRepackResult,
    VerifiedQueryExpansionParent,
    materialize_query_expansion_repack_v2,
    replay_query_expansion_repack_v2,
    verify_query_expansion_parent,
)
from tools.matched_eval.query_guided_payload_adapter import (
    VerifiedQueryGuidedConstruction,
    build_query_guided_payload_adapter,
    verify_query_guided_construction,
)
from tools.matched_eval.query_fact_adapter import QueryFactAdapterPopulation
from tools.matched_eval.query_guided_scan import (
    QueryGuidedScanResult,
    materialize_query_guided_scan,
    replay_query_guided_scan,
)
from tools.matched_eval.query_map_source_gate_adapter import (
    CONSOLIDATED_OBLIGATION_MODE,
    STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    QueryMapSourceGateAdapterPlane,
    adapt_query_map_solver_v2,
)
from tools.matched_eval.source_gate_controller import (
    LaneSourceBudget,
    SourceGatePolicy,
)
from tools.matched_eval.source_history_fact_union import FactLane


FORMAT = "memory-condense-confirmation-source-streams-v1"
ELIGIBILITY_FORMAT = "memory-condense-independent-closure-eligibility-manifest-v9"
PLANE_NAME = "confirmation-source-streams-v1.json"
PLANE_REPLAY_NAME = "confirmation-source-streams-v1-replay.json"
ELIGIBILITY_NAME = "partition-eligibility-v9.json"
ELIGIBILITY_REPLAY_NAME = "partition-eligibility-v9-replay.json"
PARTITION_NAME = "partition-scan-v2-generation.json"
PARTITION_REPLAY_NAME = "partition-scan-v2-generation-replay.json"
GUIDED_DIR_NAME = "query-guided-scan-v1"
REPACK_DIR_NAME = "query-expansion-repack-v2"
TERMINAL_BENCHMARK_AS_OF_TURN = 0


class ConfirmationSourceStreamsError(MatchedEvalContractError):
    """An authenticated source-stream input or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationSourceStreamsError(message)


def question_only_partition_eligible(route: RoutedRepairReceipt) -> bool:
    """Return the frozen question-only temporal/complete-frontier decision."""

    if type(route) is not RoutedRepairReceipt:
        raise TypeError("route must be an exact RoutedRepairReceipt")
    return bool(
        route.modifiers.requires_temporal_metadata
        or route.modifiers.requires_complete_frontier
    )


def confirmation_source_gate_policy() -> SourceGatePolicy:
    """Return the frozen d1-p0-g1 base source budget."""

    return SourceGatePolicy(
        "locked-adaptive-source-map-d1-p0-g1-v1",
        (
            LaneSourceBudget(FactLane.DIRECT, 1, 12, 2),
            LaneSourceBudget(FactLane.PARTITION, 0, 10, 2),
            LaneSourceBudget(FactLane.GUIDED, 1, 8, 2),
        ),
        global_unique_source_cap=24,
        max_physical_map_calls=48,
        max_rounds=16,
    )


@dataclass(frozen=True, slots=True)
class ConfirmationSourceStreamsResult:
    """Exact in-process source plane consumed by adaptive map/tail stages."""

    plane_artifact: SealedArtifact
    query_parent: VerifiedQueryExpansionParent
    guided: QueryGuidedScanResult
    repack: QueryExpansionRepackResult
    eligibility_artifact: SealedArtifact
    partition_artifact: SealedArtifact
    partition_generation: PartitionScanV2Generation
    query_map_adapter: QueryMapSourceGateAdapterPlane
    verified_base_rows: tuple[VerifiedLockedSourceGateRow, ...]
    verified_repack_rows: tuple[VerifiedLockedSourceGateRow, ...]
    base_population: LockedSourceGateAdapterPopulation
    repack_population: LockedSourceGateAdapterPopulation
    physical_provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        _require(
            self.guided.physical_provider_calls
            == self.repack.physical_provider_calls
            == self.physical_provider_calls
            == 0,
            "confirmation source streams gained a provider call",
        )
        _require(
            self.guided.retained_transformer_token_state_bytes
            == self.repack.retained_transformer_token_state_bytes
            == self.retained_transformer_token_state_bytes
            == 0,
            "confirmation source streams retained transformer token state",
        )
        _require(
            len(self.verified_base_rows) == len(self.verified_repack_rows)
            == len(self.query_map_adapter.rows),
            "confirmation source-stream population changed",
        )
        _require(
            self.base_population.direct_stream_profile
            == DIRECT_STREAM_PROFILE_V1
            and self.repack_population.direct_stream_profile
            == DIRECT_STREAM_PROFILE_REPACK_V2,
            "confirmation source-stream profiles changed",
        )


def _verified_query_parent(
    context: ConfirmationQueryExpansionContext,
    artifacts: VerifiedQueryExpansionArtifacts,
) -> VerifiedQueryExpansionParent:
    if type(context) is not ConfirmationQueryExpansionContext:
        raise TypeError("context must be an exact ConfirmationQueryExpansionContext")
    if type(artifacts) is not VerifiedQueryExpansionArtifacts:
        raise TypeError("artifacts must be exact VerifiedQueryExpansionArtifacts")
    roots = {
        artifact.path.parent.resolve()
        for artifact in (
            artifacts.preflight,
            artifacts.run,
            artifacts.run_replay,
            artifacts.runtime_ledger,
            artifacts.runtime_ledger_replay,
        )
    }
    _require(len(roots) == 1, "query-expansion artifacts do not share one root")
    parent = verify_query_expansion_parent(
        context.population,
        parent_output_root=next(iter(roots)),
        expected_preflight_sha256=artifacts.preflight.sha256,
        expected_run_sha256=artifacts.run.sha256,
        expected_runtime_ledger_sha256=artifacts.runtime_ledger.sha256,
    )
    _require(
        parent.preflight.sha256 == artifacts.preflight.sha256
        and parent.run.sha256
        == artifacts.run.sha256
        == artifacts.run_replay.sha256
        and parent.runtime_ledger.sha256
        == artifacts.runtime_ledger.sha256
        == artifacts.runtime_ledger_replay.sha256,
        "query-expansion artifact object differs from native replay",
    )
    return parent


def _verify_map_parents(
    context: ConfirmationQueryExpansionContext,
    parent: VerifiedQueryExpansionParent,
    map_plan: EvidenceMapPlan,
    map_plane: VerifiedEvidenceMapPlane,
) -> None:
    if type(map_plan) is not EvidenceMapPlan:
        raise TypeError("map_plan must be an exact EvidenceMapPlan")
    if type(map_plane) is not VerifiedEvidenceMapPlane:
        raise TypeError("map_plane must be an exact VerifiedEvidenceMapPlane")
    direct = map_plan.direct_plan.adapter_population
    source = context.population.source_population
    _require(
        direct.source_population.population_id == source.population_id
        and direct.source_population.snapshot.snapshot_id
        == source.snapshot.snapshot_id
        and direct.query_preflight_sha256 == parent.preflight.sha256
        and direct.query_run_sha256 == parent.run.sha256
        and map_plan.direct_plane is map_plane.parent_plane,
        "direct/map parents differ from the authenticated query population",
    )


def _eligibility_payload(context: ConfirmationQueryExpansionContext) -> dict[str, Any]:
    source = context.population.source_population
    rows: list[dict[str, Any]] = []
    for row in source.rows:
        route = route_question(row.packet.dated_question)
        eligible = question_only_partition_eligible(route)
        body = {
            "ordinal": row.ordinal,
            "question_id": row.packet.question_id,
            "question_sha256": row.packet.question_sha256,
            "dated_question_sha256": row.packet.dated_question_sha256,
            "dated_question": row.packet.dated_question,
            "route_receipt": route.identity_payload(),
            "eligible": eligible,
            "eligibility_basis": (
                "temporal_metadata_or_complete_frontier_demand"
                if eligible
                else "question_requests_neither_temporal_metadata_nor_complete_frontier"
            ),
        }
        rows.append({**body, "row_identity_sha256": identity_sha256(body)})
    body = {
        "format": ELIGIBILITY_FORMAT,
        "selection_input": "dated_question_text_only",
        "selection_policy": {
            "eligible_when": (
                "route.modifiers.requires_temporal_metadata == true OR "
                "route.modifiers.requires_complete_frontier == true"
            ),
            "focus": "temporal_or_dispersed_complete_frontier_demand",
            "source_labels_used": False,
            "gold_topology_used": False,
        },
        "retrieval_sha256": source.retrieval_sha256,
        "population_identity_sha256": source.snapshot.population_identity_sha256,
        "question_count": len(rows),
        "eligible_question_count": sum(row["eligible"] is True for row in rows),
        "questions": rows,
        "provider_calls": 0,
        "gold_loaded": False,
    }
    payload = {**body, "manifest_identity_sha256": identity_sha256(body)}
    assert_gold_blind(payload, path="confirmation_source_streams.eligibility")
    return payload


def _verify_eligibility_payload(
    artifact: SealedArtifact,
    context: ConfirmationQueryExpansionContext,
) -> tuple[bool, ...]:
    expected = _eligibility_payload(context)
    _require(
        artifact.payload == expected,
        "partition eligibility differs from question-only reconstruction",
    )
    return tuple(bool(row["eligible"]) for row in expected["questions"])


def _build_partition_generation(
    context: ConfirmationQueryExpansionContext,
    *,
    eligibility_artifact: SealedArtifact,
) -> PartitionScanV2Generation:
    eligible = _verify_eligibility_payload(eligibility_artifact, context)
    context.revalidate_store_bytes()
    questions: list[Any | None] = [None] * context.question_count
    grouped: dict[str, list[Any]] = {}
    for prompt in context.population.rows:
        grouped.setdefault(prompt.namespace.namespace_id, []).append(prompt)
    for namespace in context.population.namespaces:
        namespace_id = namespace.namespace_id
        store = context.store_dirs_by_namespace[namespace_id]
        with Database(store / "memory.db", read_only=True) as database:
            for prompt in grouped.get(namespace_id, ()):
                ordinal = prompt.source.ordinal
                packet = prompt.source.packet
                questions[ordinal] = construct_partition_scan_v2_question(
                    database,
                    ordinal=ordinal,
                    shard_offset=context.shard_offsets_by_question[
                        packet.question_id
                    ],
                    packet=packet,
                    eligible=eligible[ordinal],
                    source_database_sha256=context.database_sha256_by_namespace[
                        namespace_id
                    ],
                    source_store_receipt_sha256=(
                        namespace.combined_store_receipt_sha256
                    ),
                )
    _require(all(row is not None for row in questions), "partition scan omitted rows")
    context.revalidate_store_bytes()
    return PartitionScanV2Generation(
        retrieval_sha256=context.population.source_population.retrieval_sha256,
        eligibility_manifest_sha256=eligibility_artifact.sha256,
        population_identity_sha256=(
            context.population.source_population.snapshot.population_identity_sha256
        ),
        questions=tuple(row for row in questions if row is not None),
    )


def _artifact_rows(
    artifact: SealedArtifact,
    *,
    expected_count: int,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    raw = artifact.payload.get("questions")
    _require(
        type(raw) is list
        and len(raw) == expected_count
        and all(type(row) is dict for row in raw),
        f"{label} question population changed",
    )
    return tuple(raw)  # type: ignore[return-value]


def _source_artifacts(
    context: ConfirmationQueryExpansionContext,
    parent: VerifiedQueryExpansionParent,
    guided: QueryGuidedScanResult,
    eligibility: SealedArtifact,
    partition: SealedArtifact,
    repack: QueryExpansionRepackResult | None = None,
) -> tuple[ArtifactRef, ...]:
    rows = (
        ArtifactRef(
            "sealed_retrieval",
            context.population.source_population.retrieval_sha256,
            str(context.cumulative_artifact.path),
        ),
        ArtifactRef("query_preflight", parent.preflight.sha256, str(parent.preflight.path)),
        ArtifactRef("query_run", parent.run.sha256, str(parent.run.path)),
        ArtifactRef(
            "query_runtime", parent.runtime_ledger.sha256, str(parent.runtime_ledger.path)
        ),
        ArtifactRef("partition_eligibility", eligibility.sha256, str(eligibility.path)),
        ArtifactRef("partition_r96_generation", partition.sha256, str(partition.path)),
        ArtifactRef("guided_run", guided.run_artifact.sha256, str(guided.run_artifact.path)),
        ArtifactRef(
            "guided_runtime",
            guided.runtime_ledger_artifact.sha256,
            str(guided.runtime_ledger_artifact.path),
        ),
    )
    if repack is None:
        return rows
    return rows + (
        ArtifactRef(
            "query_repack_v2_run", repack.run_artifact.sha256, str(repack.run_artifact.path)
        ),
        ArtifactRef(
            "query_repack_v2_runtime",
            repack.runtime_ledger_artifact.sha256,
            str(repack.runtime_ledger_artifact.path),
        ),
    )


def _verified_rows(
    context: ConfirmationQueryExpansionContext,
    *,
    map_plan: EvidenceMapPlan,
    partition: PartitionScanV2Generation,
    partition_artifact: SealedArtifact,
    guided: QueryGuidedScanResult,
    guided_population: QueryFactAdapterPopulation,
    repack: QueryExpansionRepackResult,
    direct_stream_profile: str,
) -> tuple[VerifiedLockedSourceGateRow, ...]:
    count = context.question_count
    guided_rows = _artifact_rows(guided.run_artifact, expected_count=count, label="guided")
    repack_rows = _artifact_rows(repack.run_artifact, expected_count=count, label="repack")
    direct = map_plan.direct_plan.adapter_population
    source = context.population.source_population
    _require(
        len(direct.rows)
        == len(partition.questions)
        == len(guided_population.rows)
        == count,
        "source-stream parents changed population size",
    )
    rows: list[VerifiedLockedSourceGateRow] = []
    for ordinal, (prompt, direct_row, partition_row, guided_row, guided_adapter_row) in enumerate(
        zip(
            context.population.rows,
            direct.rows,
            partition.questions,
            guided_rows,
            guided_population.rows,
            strict=True,
        )
    ):
        _require(
            guided_row.get("receipt_sha256")
            == guided_adapter_row.query_row_receipt_sha256,
            "guided verified row receipt changed",
        )
        namespace = prompt.namespace
        namespace_id = namespace.namespace_id
        _require(
            partition_row.source_database_sha256
            == context.database_sha256_by_namespace[namespace_id]
            and partition_row.source_store_receipt_sha256
            == namespace.combined_store_receipt_sha256,
            "partition row changed its immutable store binding",
        )
        if direct_stream_profile == DIRECT_STREAM_PROFILE_V1:
            direct_ids = tuple(row.evidence_id for row in direct_row.admitted_delta)
            direct_sources = tuple(row.source_id for row in direct_row.admitted_delta)
            direct_receipt = direct_row.query_row_receipt_sha256
            direct_artifact = (
                map_plan.direct_plan.adapter_population.query_run_sha256
            )
        elif direct_stream_profile == DIRECT_STREAM_PROFILE_REPACK_V2:
            direct_ids, direct_sources, direct_receipt = _repack_direct_frontier(
                repack_rows[ordinal],
                ordinal=ordinal,
                prompt=prompt,
                parent_row_receipt_sha256=direct_row.query_row_receipt_sha256,
            )
            direct_artifact = repack.run_artifact.sha256
        else:
            raise ConfirmationSourceStreamsError("direct source profile changed")
        direct_stream = project_locked_lane_source_stream(
            FactLane.DIRECT,
            direct_sources,
            row_receipt=direct_receipt,
            selected_ids=direct_ids,
            artifact_sha256=direct_artifact,
        )
        partition_ids = partition_row.trace.selected_before_dedup_ids
        partition_sources = {
            row.evidence_id: row.source_id for row in partition_row.candidates
        }
        _require(
            all(value in partition_sources for value in partition_ids),
            "partition selection escaped candidate catalog",
        )
        partition_stream = project_locked_lane_source_stream(
            FactLane.PARTITION,
            tuple(partition_sources[value] for value in partition_ids),
            row_receipt=partition_row.question_identity_sha256,
            selected_ids=partition_ids,
            artifact_sha256=partition_artifact.sha256,
        )
        guided_ids = tuple(
            guided_row.get("selected_before_dedup_candidate_ids", ())
        )
        _require(
            guided_ids == guided_adapter_row.selected_before_dedup_ids,
            "guided selected-before-dedup IDs changed",
        )
        guided_stream = project_locked_lane_source_stream(
            FactLane.GUIDED,
            _candidate_sources(guided_row, guided_ids, f"guided row {ordinal}"),
            row_receipt=guided_adapter_row.query_row_receipt_sha256,
            selected_ids=guided_ids,
            artifact_sha256=guided.run_artifact.sha256,
        )
        rows.append(
            VerifiedLockedSourceGateRow(
                ordinal,
                prompt.source.packet.question_id,
                prompt.source.packet.question_sha256,
                prompt.source.packet.dated_question,
                prompt.source.packet.dated_question_sha256,
                prompt.source.packet.packet_id,
                source.snapshot.population_identity_sha256,
                source.snapshot.question_order_sha256,
                source.snapshot.snapshot_id,
                namespace,
                _direct_refs(namespace, direct_row),
                (direct_stream, partition_stream, guided_stream),
                context.store_dirs_by_namespace[namespace_id],
                context.database_sha256_by_namespace[namespace_id],
                context.index_sha256_by_namespace[namespace_id],
            )
        )
    return tuple(rows)


def _plane_payload(
    *,
    context: ConfirmationQueryExpansionContext,
    parent: VerifiedQueryExpansionParent,
    guided: QueryGuidedScanResult,
    repack: QueryExpansionRepackResult,
    eligibility: SealedArtifact,
    partition: SealedArtifact,
    map_adapter: QueryMapSourceGateAdapterPlane,
    base: LockedSourceGateAdapterPopulation,
    repack_population: LockedSourceGateAdapterPopulation,
) -> dict[str, Any]:
    body = {
        "format": FORMAT,
        "gold_loaded": False,
        "provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "question_count": context.question_count,
        "activated_question_count": len(map_adapter.activated_rows),
        "question_only_partition_eligible_count": int(
            eligibility.payload["eligible_question_count"]
        ),
        "source_population_id": context.population.source_population.population_id,
        "query_population_id": context.population.population_id,
        "query_preflight_sha256": parent.preflight.sha256,
        "query_run_sha256": parent.run.sha256,
        "query_runtime_ledger_sha256": parent.runtime_ledger.sha256,
        "guided_run_sha256": guided.run_artifact.sha256,
        "guided_runtime_ledger_sha256": guided.runtime_ledger_artifact.sha256,
        "query_repack_v2_run_sha256": repack.run_artifact.sha256,
        "query_repack_v2_runtime_ledger_sha256": (
            repack.runtime_ledger_artifact.sha256
        ),
        "partition_eligibility_sha256": eligibility.sha256,
        "partition_generation_sha256": partition.sha256,
        "query_map_adapter_receipt_sha256": map_adapter.receipt_sha256,
        "obligation_compilation_mode": CONSOLIDATED_OBLIGATION_MODE,
        "state_chain_profile": STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
        "base_source_profile": DIRECT_STREAM_PROFILE_V1,
        "base_source_population_receipt_sha256": base.receipt_sha256,
        "tail_source_profile": DIRECT_STREAM_PROFILE_REPACK_V2,
        "tail_source_population_receipt_sha256": repack_population.receipt_sha256,
        "source_gate_policy_receipt_sha256": confirmation_source_gate_policy().receipt_sha256,
    }
    assert_gold_blind(body, path="confirmation_source_streams.plane")
    return {**body, "plane_identity_sha256": identity_sha256(body)}


def _build_result(
    *,
    plane_artifact: SealedArtifact | None,
    context: ConfirmationQueryExpansionContext,
    parent: VerifiedQueryExpansionParent,
    map_plan: EvidenceMapPlan,
    map_plane: VerifiedEvidenceMapPlane,
    guided: QueryGuidedScanResult,
    guided_construction: VerifiedQueryGuidedConstruction,
    repack: QueryExpansionRepackResult,
    eligibility: SealedArtifact,
    partition_artifact: SealedArtifact,
    partition: PartitionScanV2Generation,
    output_root: Path,
) -> ConfirmationSourceStreamsResult:
    adapter = adapt_query_map_solver_v2(
        parent.run,
        map_plan,
        map_plane,
        obligation_mode=CONSOLIDATED_OBLIGATION_MODE,
        state_chain_profile=STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    )
    activations = tuple(
        locked_activation_input_from_query_map_adapter(
            row,
            as_of_turn=TERMINAL_BENCHMARK_AS_OF_TURN,
        )
        for row in adapter.activated_rows
    )
    _require(bool(activations), "source gate has no unresolved activation")
    guided_population = build_query_guided_payload_adapter(
        context.population,
        guided_construction,
    )
    base_rows = _verified_rows(
        context,
        map_plan=map_plan,
        partition=partition,
        partition_artifact=partition_artifact,
        guided=guided,
        guided_population=guided_population,
        repack=repack,
        direct_stream_profile=DIRECT_STREAM_PROFILE_V1,
    )
    repack_rows = _verified_rows(
        context,
        map_plan=map_plan,
        partition=partition,
        partition_artifact=partition_artifact,
        guided=guided,
        guided_population=guided_population,
        repack=repack,
        direct_stream_profile=DIRECT_STREAM_PROFILE_REPACK_V2,
    )
    policy = confirmation_source_gate_policy()
    base = build_locked_source_gate_adapter(
        base_rows,
        activations,
        source_artifacts=_source_artifacts(
            context, parent, guided, eligibility, partition_artifact
        ),
        policy=policy,
        direct_stream_profile=DIRECT_STREAM_PROFILE_V1,
    )
    tail = build_locked_source_gate_adapter(
        repack_rows,
        activations,
        source_artifacts=_source_artifacts(
            context, parent, guided, eligibility, partition_artifact, repack
        ),
        policy=policy,
        direct_stream_profile=DIRECT_STREAM_PROFILE_REPACK_V2,
    )
    payload = _plane_payload(
        context=context,
        parent=parent,
        guided=guided,
        repack=repack,
        eligibility=eligibility,
        partition=partition_artifact,
        map_adapter=adapter,
        base=base,
        repack_population=tail,
    )
    if plane_artifact is None:
        plane_artifact, _created = publish_sealed_json(
            output_root / PLANE_NAME, payload
        )
    else:
        _require(
            plane_artifact.payload == payload,
            "source plane differs from authenticated reconstruction",
        )
    return ConfirmationSourceStreamsResult(
        plane_artifact,
        parent,
        guided,
        repack,
        eligibility,
        partition_artifact,
        partition,
        adapter,
        base_rows,
        repack_rows,
        base,
        tail,
    )


def materialize_confirmation_source_streams(
    context: ConfirmationQueryExpansionContext,
    query_artifacts: VerifiedQueryExpansionArtifacts,
    map_plan: EvidenceMapPlan,
    map_plane: VerifiedEvidenceMapPlane,
    *,
    output_root: str | Path,
) -> ConfirmationSourceStreamsResult:
    """Materialize the complete provider-free source plane for arbitrary N."""

    output = Path(output_root)
    _require(not (output / PLANE_NAME).exists(), "source plane exists; use replay")
    parent = _verified_query_parent(context, query_artifacts)
    _verify_map_parents(context, parent, map_plan, map_plane)
    context.revalidate_store_bytes()
    query_root = parent.preflight.path.parent
    guided = materialize_query_guided_scan(
        context,  # structural superset of LockedQueryExpansionContext
        parent_output_root=query_root,
        output_root=output / GUIDED_DIR_NAME,
        expected_parent_preflight_sha256=parent.preflight.sha256,
        expected_parent_run_sha256=parent.run.sha256,
        expected_parent_runtime_ledger_sha256=parent.runtime_ledger.sha256,
    )
    guided_construction = VerifiedQueryGuidedConstruction(
        guided.run_artifact.path,
        guided.run_artifact.sha256,
        guided.runtime_ledger_artifact,
        parent,
    )
    repack = materialize_query_expansion_repack_v2(
        context,  # structural superset of LockedQueryExpansionContext
        parent_output_root=query_root,
        output_root=output / REPACK_DIR_NAME,
        expected_parent_preflight_sha256=parent.preflight.sha256,
        expected_parent_run_sha256=parent.run.sha256,
        expected_parent_runtime_ledger_sha256=parent.runtime_ledger.sha256,
    )
    eligibility, _created = publish_sealed_json(
        output / ELIGIBILITY_NAME,
        _eligibility_payload(context),
    )
    partition_unsealed = _build_partition_generation(
        context,
        eligibility_artifact=eligibility,
    )
    partition_artifact, _created = publish_sealed_json(
        output / PARTITION_NAME,
        partition_unsealed.projection(),
    )
    partition = project_partition_scan_v2_generation(
        partition_artifact.payload,
        generation_sha256=partition_artifact.sha256,
        population=context.population.source_population,
        expected_eligibility_manifest_sha256=eligibility.sha256,
    )
    context.revalidate_store_bytes()
    return _build_result(
        plane_artifact=None,
        context=context,
        parent=parent,
        map_plan=map_plan,
        map_plane=map_plane,
        guided=guided,
        guided_construction=guided_construction,
        repack=repack,
        eligibility=eligibility,
        partition_artifact=partition_artifact,
        partition=partition,
        output_root=output,
    )


def replay_confirmation_source_streams(
    context: ConfirmationQueryExpansionContext,
    query_artifacts: VerifiedQueryExpansionArtifacts,
    map_plan: EvidenceMapPlan,
    map_plane: VerifiedEvidenceMapPlane,
    *,
    output_root: str | Path,
    expected_plane_sha256: str,
) -> ConfirmationSourceStreamsResult:
    """Rebuild every provider-free stage and require byte-identical outputs."""

    output = Path(output_root)
    expected = require_sha256(expected_plane_sha256, "expected source-plane SHA-256")
    plane = read_sealed_json(output / PLANE_NAME)
    _require(plane.sha256 == expected, "source-plane artifact changed")
    raw_identity = plane.payload.get("plane_identity_sha256")
    body = dict(plane.payload)
    body.pop("plane_identity_sha256", None)
    _require(
        raw_identity == identity_sha256(body),
        "source-plane self-seal changed",
    )
    parent = _verified_query_parent(context, query_artifacts)
    _verify_map_parents(context, parent, map_plan, map_plane)
    query_root = parent.preflight.path.parent
    guided = replay_query_guided_scan(
        context,
        parent_output_root=query_root,
        output_root=output / GUIDED_DIR_NAME,
        expected_parent_preflight_sha256=parent.preflight.sha256,
        expected_parent_run_sha256=parent.run.sha256,
        expected_parent_runtime_ledger_sha256=parent.runtime_ledger.sha256,
        expected_run_sha256=require_sha256(
            plane.payload.get("guided_run_sha256"), "source-plane guided run"
        ),
    )
    guided_construction = verify_query_guided_construction(
        context.population,
        query_parent_root=query_root,
        guided_root=output / GUIDED_DIR_NAME,
        expected_query_preflight_sha256=parent.preflight.sha256,
        expected_query_run_sha256=parent.run.sha256,
        expected_query_runtime_ledger_sha256=parent.runtime_ledger.sha256,
        expected_guided_run_sha256=guided.run_artifact.sha256,
        expected_guided_runtime_ledger_sha256=require_sha256(
            plane.payload.get("guided_runtime_ledger_sha256"),
            "source-plane guided runtime",
        ),
    )
    repack = replay_query_expansion_repack_v2(
        context,
        parent_output_root=query_root,
        output_root=output / REPACK_DIR_NAME,
        expected_parent_preflight_sha256=parent.preflight.sha256,
        expected_parent_run_sha256=parent.run.sha256,
        expected_parent_runtime_ledger_sha256=parent.runtime_ledger.sha256,
        expected_run_sha256=require_sha256(
            plane.payload.get("query_repack_v2_run_sha256"),
            "source-plane repack run",
        ),
    )
    _require(
        repack.runtime_ledger_artifact.sha256
        == require_sha256(
            plane.payload.get("query_repack_v2_runtime_ledger_sha256"),
            "source-plane repack runtime",
        ),
        "source-plane repack runtime changed",
    )
    eligibility = read_sealed_json(output / ELIGIBILITY_NAME)
    _require(
        eligibility.sha256
        == require_sha256(
            plane.payload.get("partition_eligibility_sha256"),
            "source-plane eligibility",
        ),
        "source-plane eligibility artifact changed",
    )
    rebuilt_eligibility = _eligibility_payload(context)
    _require(
        eligibility.payload == rebuilt_eligibility,
        "source-plane eligibility reconstruction changed",
    )
    eligibility_replay, _created = publish_sealed_json(
        output / ELIGIBILITY_REPLAY_NAME,
        rebuilt_eligibility,
    )
    _require(
        eligibility_replay.sha256 == eligibility.sha256,
        "eligibility replay seal changed",
    )
    partition_artifact = read_sealed_json(output / PARTITION_NAME)
    _require(
        partition_artifact.sha256
        == require_sha256(
            plane.payload.get("partition_generation_sha256"),
            "source-plane partition generation",
        ),
        "source-plane partition artifact changed",
    )
    rebuilt_partition = _build_partition_generation(
        context,
        eligibility_artifact=eligibility,
    )
    rebuilt_payload = rebuilt_partition.projection()
    _require(
        canonical_json_bytes(rebuilt_payload)
        == canonical_json_bytes(partition_artifact.payload),
        "partition generation differs from store reconstruction",
    )
    partition_replay, _created = publish_sealed_json(
        output / PARTITION_REPLAY_NAME,
        rebuilt_payload,
    )
    _require(
        partition_replay.sha256 == partition_artifact.sha256,
        "partition generation replay seal changed",
    )
    partition = project_partition_scan_v2_generation(
        partition_artifact.payload,
        generation_sha256=partition_artifact.sha256,
        population=context.population.source_population,
        expected_eligibility_manifest_sha256=eligibility.sha256,
    )
    result = _build_result(
        plane_artifact=plane,
        context=context,
        parent=parent,
        map_plan=map_plan,
        map_plane=map_plane,
        guided=guided,
        guided_construction=guided_construction,
        repack=repack,
        eligibility=eligibility,
        partition_artifact=partition_artifact,
        partition=partition,
        output_root=output,
    )
    replay, _created = publish_sealed_json(
        output / PLANE_REPLAY_NAME,
        result.plane_artifact.payload,
    )
    _require(replay.sha256 == plane.sha256, "source-plane replay seal changed")
    context.revalidate_store_bytes()
    return result


__all__ = [
    "ELIGIBILITY_NAME",
    "ELIGIBILITY_REPLAY_NAME",
    "FORMAT",
    "GUIDED_DIR_NAME",
    "PARTITION_NAME",
    "PARTITION_REPLAY_NAME",
    "PLANE_NAME",
    "PLANE_REPLAY_NAME",
    "REPACK_DIR_NAME",
    "ConfirmationSourceStreamsError",
    "ConfirmationSourceStreamsResult",
    "confirmation_source_gate_policy",
    "materialize_confirmation_source_streams",
    "question_only_partition_eligible",
    "replay_confirmation_source_streams",
]
