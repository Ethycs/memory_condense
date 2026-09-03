#!/usr/bin/env python3
"""Arbitrary-population confirmation typed composition and Terra final plane.

This is the executable confirmation port of the frozen compact typed-memory
arm.  Composition is gold blind and provider free: it authenticates the exact
adaptive solver, base-map, and tail objects; scans every immutable namespace
once; lets each mechanism select independently; and only then performs
identity-proven cross-mechanism deduplication.  The final provider sees the
compact typed projection and opaque story links, never local source locators.

Provider execution is a separate exact-remaining release over native
``FastCompletionRuntime`` journals with ``retries=0``.  Answer
materialization is store free and invalid output preserves the protected
adaptive parent byte for byte.  Replay revalidates the source stores, rebuilds
the composition, and requires byte-identical terminal output.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from memory_condense.persistence.db import Database
from tools import run_locked_typed_memory_final_arm as frozen
from tools.confirmation_adaptive_source_map import (
    VerifiedConfirmationAdaptiveSourceMapPlane,
)
from tools.confirmation_adaptive_tail import (
    VerifiedConfirmationAdaptiveEvidencePlane,
    VerifiedConfirmationAdaptiveTailPlane,
)
from tools.confirmation_query_expansion_adapter import (
    ConfirmationQueryExpansionContext,
)
from tools.matched_eval import provider_runtime
from tools.matched_eval.adaptive_source_tail_typed import (
    TailFactUnionRow,
    adapt_tail_question_contributions,
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
    require_text,
)
from tools.matched_eval.full_store_slot_closure import (
    FullStoreSlotClosureResult,
    FullStoreWindowIndex,
    build_full_store_window_index,
    scan_full_store_slot_closure,
)
from tools.matched_eval.full_store_typed_adapter import (
    adapt_full_store_slot_closure,
)
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_connectivity_ledger import (
    build_typed_connectivity_ledger,
)
from tools.matched_eval.typed_memory_final_arm import (
    COMPOSITION_FORMAT,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    VALIDATOR_POLICY_FORMAT,
    fit_typed_final_prompt,
    materialize_typed_final_result_row,
    render_final_messages,
)
from tools.matched_eval.prediction_row_projection import (
    prediction_row_projection,
)
from tools.matched_eval.typed_operator_adapter import (
    COMPACT_FINAL_PROVIDER_FORMAT,
    FrontierMode,
    ProviderPayloadMode,
    adapt_verified_evidence,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


FORMAT = "memory-condense-confirmation-typed-final-v1"
CLOSURE_INPUT_FORMAT = f"{FORMAT}-full-store-input-v1"
COMPOSITION_ARTIFACT_FORMAT = f"{FORMAT}-composition-v1"
COMPOSITION_REPLAY_FORMAT = f"{FORMAT}-composition-replay-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
RUN_FORMAT = f"{FORMAT}-run-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"

CLOSURE_INPUT_NAME = "confirmation-typed-final-full-store-input-v1.json"
COMPOSITION_NAME = "confirmation-typed-final-composition-v1.json"
COMPOSITION_REPLAY_NAME = "confirmation-typed-final-composition-replay-v1.json"
PREFLIGHT_NAME = "confirmation-typed-final-preflight-v1.json"
RELEASE_NAME = "confirmation-typed-final-provider-release-v1.json"
RUN_NAME = "confirmation-typed-final-run-v1.json"
REPLAY_NAME = "confirmation-typed-final-replay-v1.json"
CHECKPOINT_DIR_NAME = "terra-confirmation-typed-final-v1-calls"

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
        "total_provider_call_budget",
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


class ConfirmationTypedFinalError(MatchedEvalContractError):
    """A typed parent, source, composition, prompt, or journal changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationTypedFinalError(message)


ClientFactory = Callable[[str, str], Any]


@dataclass(frozen=True, slots=True)
class ConfirmationTypedFinalInputs:
    """Exact in-process parents needed to rebuild typed composition."""

    context: ConfirmationQueryExpansionContext
    adaptive_plane: VerifiedConfirmationAdaptiveEvidencePlane
    base_plane: VerifiedConfirmationAdaptiveSourceMapPlane
    tail_plane: VerifiedConfirmationAdaptiveTailPlane

    def __post_init__(self) -> None:
        _require(
            type(self.context) is ConfirmationQueryExpansionContext,
            "typed final context is not exact",
        )
        _require(
            type(self.adaptive_plane) is VerifiedConfirmationAdaptiveEvidencePlane,
            "typed final adaptive parent is not exact",
        )
        _require(
            type(self.base_plane) is VerifiedConfirmationAdaptiveSourceMapPlane,
            "typed final base source-map parent is not exact",
        )
        _require(
            type(self.tail_plane) is VerifiedConfirmationAdaptiveTailPlane,
            "typed final tail parent is not exact",
        )
        _validate_inputs(self)


@dataclass(frozen=True, slots=True)
class ConfirmationTypedComposition:
    inputs: ConfirmationTypedFinalInputs
    closure_input_artifact: SealedArtifact
    composition_artifact: SealedArtifact

    def __post_init__(self) -> None:
        _require(
            type(self.inputs) is ConfirmationTypedFinalInputs
            and type(self.closure_input_artifact) is SealedArtifact
            and type(self.composition_artifact) is SealedArtifact,
            "typed composition changed exact types",
        )
        _require(
            self.composition_artifact.payload.get("closure_input_artifact_sha256")
            == self.closure_input_artifact.sha256,
            "typed composition lost its full-store input",
        )
        rows = self.composition_artifact.payload.get("questions")
        _require(
            self.composition_artifact.payload.get("format")
            == COMPOSITION_ARTIFACT_FORMAT
            and type(rows) is list
            and bool(rows)
            and self.composition_artifact.payload.get("question_count") == len(rows)
            and all(
                row.get("provider_projection", {})
                .get("provider_input", {})
                .get("typed_evidence", {})
                .get("format")
                == COMPACT_FINAL_PROVIDER_FORMAT
                for row in rows
                if type(row) is dict
            )
            and all(type(row) is dict for row in rows),
            "typed composition escaped the compact-final boundary",
        )


@dataclass(frozen=True, slots=True)
class ConfirmationTypedFinalPreflight:
    composition: ConfirmationTypedComposition
    artifact: SealedArtifact

    @property
    def required_provider_calls(self) -> int:
        return int(self.artifact.payload["required_authorized_provider_calls"])


@dataclass(frozen=True, slots=True)
class ConfirmationTypedProviderExecution:
    batch: FastCompletionBatch | None
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class ConfirmationTypedFinalMaterialization:
    composition_artifact: SealedArtifact
    closure_input_artifact: SealedArtifact
    preflight_artifact: SealedArtifact
    release_artifact: SealedArtifact
    run_artifact: SealedArtifact
    completion_batch: FastCompletionBatch
    predictions: tuple[str, ...]
    result_rows: tuple[dict[str, Any], ...]
    judge_rows: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        _require(
            type(self.completion_batch) is FastCompletionBatch
            and len(self.predictions) == len(self.result_rows) == len(self.judge_rows)
            and tuple(row["prediction"] for row in self.result_rows)
            == self.predictions,
            "typed final materialization changed ordered predictions",
        )


@dataclass(frozen=True, slots=True)
class VerifiedConfirmationTypedFinalPlane:
    composition_artifact: SealedArtifact
    closure_input_artifact: SealedArtifact
    preflight_artifact: SealedArtifact
    release_artifact: SealedArtifact
    run_artifact: SealedArtifact
    replay_artifact: SealedArtifact
    completion_batch: FastCompletionBatch
    predictions: tuple[str, ...]
    result_rows: tuple[dict[str, Any], ...]
    judge_rows: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        _require(
            all(
                type(value) is SealedArtifact
                for value in (
                    self.composition_artifact,
                    self.closure_input_artifact,
                    self.preflight_artifact,
                    self.release_artifact,
                    self.run_artifact,
                    self.replay_artifact,
                )
            )
            and type(self.completion_batch) is FastCompletionBatch
            and type(self.predictions) is tuple
            and type(self.result_rows) is tuple
            and type(self.judge_rows) is tuple,
            "verified typed final plane changed result types",
        )
        _require(
            len(self.predictions) == len(self.result_rows) == len(self.judge_rows)
            and tuple(row["prediction"] for row in self.result_rows)
            == self.predictions
            and tuple(row["question_id"] for row in self.result_rows)
            == tuple(row["question_id"] for row in self.judge_rows),
            "verified typed final prediction order changed",
        )
        _require(
            self.replay_artifact.payload.get("replayed_run_sha256")
            == self.run_artifact.sha256,
            "verified typed final replay lost its run",
        )


def _ordered_context_rows(
    context: ConfirmationQueryExpansionContext,
) -> tuple[Any, ...]:
    rows = tuple(context.population.rows)
    _require(bool(rows), "typed final population is empty")
    ids = tuple(row.source.packet.question_id for row in rows)
    _require(len(set(ids)) == len(ids), "typed final question IDs repeat")
    return rows


def _validate_inputs(inputs: ConfirmationTypedFinalInputs) -> None:
    context_rows = _ordered_context_rows(inputs.context)
    planned = inputs.adaptive_plane.plan.plan.rows
    adaptive_rows = inputs.adaptive_plane.run.rows
    ids = tuple(row.source.packet.question_id for row in context_rows)
    _require(
        inputs.adaptive_plane.run_artifact.payload.get("run_receipt_sha256")
        == inputs.adaptive_plane.run.receipt_sha256
        and inputs.adaptive_plane.replay_artifact.payload.get("replayed_run_sha256")
        == inputs.adaptive_plane.run_artifact.sha256,
        "typed final adaptive run/preflight/replay binding changed",
    )
    _require(
        len(planned) == len(adaptive_rows) == len(ids)
        and tuple(row.question_id for row in planned) == ids
        and tuple(row.question_id for row in adaptive_rows) == ids
        and tuple(row.ordinal for row in planned) == tuple(range(len(ids)))
        and tuple(row.ordinal for row in adaptive_rows) == tuple(range(len(ids))),
        "typed final adaptive population/order changed",
    )
    _require(
        inputs.base_plane.replay_artifact.payload.get("materialization_sha256")
        == inputs.base_plane.materialization_artifact.sha256
        and inputs.tail_plane.replay_artifact.payload.get("replayed_run_sha256")
        == inputs.tail_plane.run_artifact.sha256,
        "typed final base/tail replay binding changed",
    )
    base_ids = tuple(row.question_id for row in inputs.base_plane.questions)
    tail_ids = tuple(row.question_id for row in inputs.tail_plane.questions)
    _require(
        len(set(base_ids)) == len(base_ids)
        and len(set(tail_ids)) == len(tail_ids)
        and set(base_ids) <= set(ids)
        and set(tail_ids) <= set(ids),
        "typed final source-map rows escaped the adaptive population",
    )
    adaptive_upstream = inputs.adaptive_plane.plan.upstream
    tail_upstream = inputs.tail_plane.plan.upstream
    _require(
        adaptive_upstream == tail_upstream
        and adaptive_upstream.base_preflight_artifact.sha256
        == inputs.base_plane.preflight_artifact.sha256
        and adaptive_upstream.base_work_manifest_artifact.sha256
        == inputs.base_plane.work_manifest_artifact.sha256
        and adaptive_upstream.base_materialization_artifact.sha256
        == inputs.base_plane.materialization_artifact.sha256
        and adaptive_upstream.base_replay_artifact.sha256
        == inputs.base_plane.replay_artifact.sha256
        and adaptive_upstream.base_questions == inputs.base_plane.questions
        and adaptive_upstream.base_materializations
        == inputs.base_plane.materializations
        and adaptive_upstream.source_population
        == inputs.base_plane.source_population,
        "typed final adaptive/base/tail carriers do not share one upstream",
    )
    source_population = inputs.context.population.source_population
    _require(
        adaptive_upstream.map_plan is inputs.adaptive_plane.plan.plan.map_plan
        and adaptive_upstream.map_plane is inputs.adaptive_plane.plan.plan.map_plane
        and adaptive_upstream.map_plan.direct_plan.adapter_population.source_population
        == source_population,
        "typed final context escaped the exact adaptive evidence-map lineage",
    )


def _evidence_items_belong_to_namespace(
    evidence_items: Sequence[Any], namespace: Any
) -> bool:
    source_ids = {row.source_id for row in namespace.sources}
    return all(
        hasattr(row, "source_id") and row.source_id in source_ids
        for row in evidence_items
    )


def _build_full_store_results(
    inputs: ConfirmationTypedFinalInputs,
    *,
    namespace: Any,
    context_rows: Sequence[Any],
) -> tuple[
    dict[str, FullStoreSlotClosureResult],
    FullStoreWindowIndex,
    dict[str, Any],
]:
    """Read and index exactly one namespace; the caller drops it immediately."""

    namespace_id = namespace.namespace_id
    _require(
        bool(context_rows)
        and all(row.namespace.namespace_id == namespace_id for row in context_rows),
        "full-store namespace batch changed membership",
    )
    database_path = inputs.context.store_dirs_by_namespace[namespace_id] / "memory.db"
    with Database(database_path, read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            namespace,
            source_database_sha256=inputs.context.database_sha256_by_namespace[
                namespace_id
            ],
            source_store_receipt_sha256=namespace.combined_store_receipt_sha256,
        )
    index = build_full_store_window_index(cache)
    results: dict[str, FullStoreSlotClosureResult] = {}
    for row in context_rows:
        packet = row.source.packet
        results[packet.question_id] = scan_full_store_slot_closure(
            index, packet.dated_question
        )
    _require(
        len(results) == len(context_rows),
        "full-store namespace result population changed",
    )
    receipt = {
        "cache_receipt_sha256": cache.cache_receipt_sha256,
        "content_row_count": cache.content_row_count,
        "database_read_passes": 1,
        "namespace_id": namespace_id,
        "physical_store_row_count": cache.physical_store_row_count,
        "window_index_receipt_sha256": index.receipt_sha256,
    }
    return results, index, receipt


def _closure_input_projection(
    results: Mapping[str, FullStoreSlotClosureResult],
    cache_receipts: Sequence[Mapping[str, Any]],
    ordered_ids: Sequence[str],
    *,
    ordinal_by_question: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ordinal_map = (
        {question_id: ordinal for ordinal, question_id in enumerate(ordered_ids)}
        if ordinal_by_question is None
        else dict(ordinal_by_question)
    )
    for question_id in ordered_ids:
        ordinal = ordinal_map[question_id]
        result = results[question_id]
        body = {
            "local_audit": result.local_audit_projection(),
            "ordinal": ordinal,
            "provider_projection": result.provider_projection(),
            "question_id": question_id,
            "result_receipt_sha256": result.receipt.receipt_sha256,
        }
        rows.append({**body, "row_receipt_sha256": identity_sha256(body)})
    payload = {
        "cache_receipts": [dict(row) for row in cache_receipts],
        "database_read_passes_per_unique_namespace": 1,
        "format": CLOSURE_INPUT_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "question_count": len(rows),
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "unique_namespace_count": len(cache_receipts),
    }
    assert_gold_blind(payload, path="confirmation_typed_final_full_store_input")
    return payload


def _compose_rows(
    inputs: ConfirmationTypedFinalInputs,
    *,
    closure_by_question: Mapping[str, FullStoreSlotClosureResult],
    index_by_namespace: Mapping[str, FullStoreWindowIndex],
    closure_artifact_sha256: str,
    ordinals: Sequence[int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Apply the frozen selection, linking, dedup, and compact-fit policy."""

    planned_rows = inputs.adaptive_plane.plan.plan.rows
    parent_rows = inputs.adaptive_plane.run.rows
    context_rows = _ordered_context_rows(inputs.context)
    map_artifact_sha256 = inputs.adaptive_plane.plan.upstream.map_plane.run_sha256
    base_rows = build_tail_post_map_fact_unions(
        inputs.base_plane.questions, inputs.base_plane.materializations
    )
    rebuilt_tail_rows = build_tail_post_map_fact_unions(
        inputs.tail_plane.questions, inputs.tail_plane.materializations
    )
    _require(
        rebuilt_tail_rows == inputs.tail_plane.fact_union_rows,
        "typed final tail fact unions differ from deterministic rebuild",
    )
    base_by_question = {row.question_id: row for row in base_rows}
    tail_by_question = {row.question_id: row for row in rebuilt_tail_rows}
    context_by_question = {
        row.source.packet.question_id: row for row in context_rows
    }
    composition_rows: list[dict[str, Any]] = []
    selected_ordinals = (
        tuple(range(len(planned_rows)))
        if ordinals is None
        else tuple(ordinals)
    )
    _require(
        len(set(selected_ordinals)) == len(selected_ordinals)
        and all(0 <= value < len(planned_rows) for value in selected_ordinals),
        "typed namespace ordinals changed",
    )
    for ordinal in selected_ordinals:
        planned = planned_rows[ordinal]
        parent_row = parent_rows[ordinal]
        question_id = planned.question_id
        source_packet = planned.map_plan_row.direct_plan_row.adapter.source.packet
        context_row = context_by_question[question_id]
        _require(
            ordinal == planned.ordinal == parent_row.ordinal
            and parent_row.question_id == question_id
            and context_row.source.packet.dated_question == source_packet.dated_question
            and context_row.source.packet.question_sha256 == source_packet.question_sha256
            and _evidence_items_belong_to_namespace(
                tuple(source_packet.protected_evidence)
                + tuple(source_packet.admitted_evidence),
                context_row.namespace,
            ),
            "typed final question/store/adaptive binding changed",
        )
        spec = compile_typed_operator_spec(source_packet.dated_question)
        _require(
            spec.question_sha256 == source_packet.dated_question_sha256,
            "typed final operator escaped its dated question",
        )

        map_packet = adapt_verified_evidence(
            spec,
            planned.map_plan_row,
            planned.map_row,
            map_artifact_sha256=map_artifact_sha256,
            fact_envelope=None,
            source_artifact_sha256=None,
            frontier_mode=FrontierMode.BOUNDED,
            handle_start=frozen.PARENT_MAP_RANGE,
            group_start=frozen.PARENT_MAP_RANGE,
        )
        parent_map = frozen._packet_contribution(  # noqa: SLF001
            map_packet,
            mechanism_id=frozen.PARENT_MAP_MECHANISM,
            sealed_artifact_sha256=map_artifact_sha256,
        )
        parent_prompt_proxy = frozen._adaptive_parent_prompt_proxy(planned)  # noqa: SLF001

        original = [parent_map]
        base_original = ()
        base_row = base_by_question.get(question_id)
        if base_row is not None:
            base_original = adapt_tail_question_contributions(
                spec,
                base_row,
                materialization_artifact_sha256=(
                    inputs.base_plane.materialization_artifact.sha256
                ),
                parent_prompt_token_proxy=parent_prompt_proxy,
                source_handle_start=frozen.PARENT_SOURCE_RANGE,
                source_group_start=frozen.PARENT_SOURCE_RANGE,
                pointer_handle_start=frozen.PARENT_POINTER_RANGE,
                pointer_group_start=frozen.PARENT_POINTER_RANGE,
                source_mechanism_id=frozen.PARENT_SOURCE_MECHANISM,
                pointer_mechanism_id=frozen.PARENT_POINTER_MECHANISM,
            )
            original.extend(base_original)

        tail_original = ()
        tail_row = tail_by_question.get(question_id)
        if tail_row is not None:
            tail_original = adapt_tail_question_contributions(
                spec,
                tail_row,
                materialization_artifact_sha256=inputs.tail_plane.run_artifact.sha256,
                parent_prompt_token_proxy=parent_prompt_proxy,
                source_handle_start=frozen.TAIL_SOURCE_RANGE,
                source_group_start=frozen.TAIL_SOURCE_RANGE,
                pointer_handle_start=frozen.TAIL_POINTER_RANGE,
                pointer_group_start=frozen.TAIL_POINTER_RANGE,
                source_mechanism_id=frozen.TAIL_SOURCE_MECHANISM,
                pointer_mechanism_id=frozen.TAIL_POINTER_MECHANISM,
            )
            original.extend(tail_original)

        closure = closure_by_question[question_id]
        full_contribution, full_audit = adapt_full_store_slot_closure(
            spec,
            closure,
            closure_artifact_sha256=closure_artifact_sha256,
            handle_start=frozen.FULL_STORE_RANGE,
            group_start=frozen.FULL_STORE_RANGE,
            mechanism_id=frozen.FULL_STORE_MECHANISM,
        )
        full_priorities, full_priority_audit = frozen._full_store_selection_priorities(  # noqa: SLF001
            full_contribution, closure
        )
        original.append(full_contribution)
        namespace_index = index_by_namespace[context_row.namespace.namespace_id]
        _require(
            closure.receipt.window_index_receipt_sha256
            == namespace_index.receipt_sha256,
            "typed active reconstruction did not reuse its namespace index",
        )
        active_result, active_contribution, active_alignment = frozen._build_active_reconstruction(  # noqa: SLF001
            namespace_index, closure, full_contribution
        )
        active_full_priorities, active_priority_audit = frozen._active_selection_priorities(  # noqa: SLF001
            full_contribution,
            full_priorities,
            active_contribution,
            active_result,
        )
        # Each method has completed selection before cross-method exclusions.
        original.append(active_contribution)

        exact_span_keys: dict[str, tuple[str, ...]] = {}
        exact_span_keys.update(
            frozen._map_exact_span_keys(  # noqa: SLF001
                parent_map, planned, context_row.namespace.namespace_id
            )
        )
        if base_row is not None:
            for contribution in base_original:
                exact_span_keys.update(
                    frozen._union_exact_span_keys(  # noqa: SLF001
                        contribution,
                        base_row,
                        parent_prompt_token_proxy=parent_prompt_proxy,
                    )
                )
        if tail_row is not None:
            for contribution in tail_original:
                exact_span_keys.update(
                    frozen._union_exact_span_keys(  # noqa: SLF001
                        contribution,
                        tail_row,
                        parent_prompt_token_proxy=parent_prompt_proxy,
                    )
                )
        exact_span_keys.update(frozen._full_store_exact_span_keys(full_audit))  # noqa: SLF001
        exact_span_keys.update(
            frozen._active_exact_span_keys(active_contribution, active_result)  # noqa: SLF001
        )
        _require(
            set(exact_span_keys)
            == {
                binding.handle_id
                for contribution in original
                for binding in contribution.bindings
            },
            "typed post-selection exact-span lineage changed",
        )
        deduped, postselection_exclusions = frozen._dedup_selected_contributions(  # noqa: SLF001
            tuple(original), exact_span_keys_by_handle=exact_span_keys
        )
        deduped_handles = {
            binding.handle_id
            for contribution in deduped
            for binding in contribution.bindings
        }
        retained_priorities = {
            handle: priority
            for handle, priority in active_full_priorities.items()
            if handle in deduped_handles
        }
        minimum, lane_audit = frozen._allocate_non_borrowable_lanes(  # noqa: SLF001
            deduped,
            operator_spec=spec,
            local_selection_priority_by_handle=retained_priorities,
        )
        protected_item_receipts = tuple(
            receipt
            for lane in minimum.receipts
            for receipt in lane.selected_item_receipt_sha256s
        )
        allocated, surplus_audit = frozen._fill_shared_lane_surplus(  # noqa: SLF001
            deduped,
            minimum,
            operator_spec=spec,
            local_selection_priority_by_handle=retained_priorities,
        )
        allocated_owner = {
            binding.handle_id: contribution.mechanism_id
            for contribution in allocated
            for binding in contribution.bindings
        }
        allocated_priorities = {
            handle: priority
            for handle, priority in retained_priorities.items()
            if handle in allocated_owner
        }
        expected_priority_handles = {
            handle
            for handle, owner in allocated_owner.items()
            if owner
            in {
                frozen.FULL_STORE_MECHANISM,
                frozen.ACTIVE_RECONSTRUCTION_MECHANISM,
            }
        }
        _require(
            set(allocated_priorities) == expected_priority_handles,
            "typed allocated active/full priorities changed coverage",
        )
        packet, fair_audit = frozen._fair_merge_contributions(  # noqa: SLF001
            spec,
            allocated,
            local_selection_priority_by_handle=allocated_priorities,
            protected_item_receipt_sha256s=protected_item_receipts,
            minimum_allocation_receipt_sha256=minimum.receipt_sha256,
            surplus_fill_audit=surplus_audit,
        )
        _require(
            packet.provider_payload_mode is ProviderPayloadMode.COMPACT_FINAL,
            "typed fair merge left the frozen COMPACT_FINAL payload mode",
        )
        mechanism_by_handle, dropped_allocated = frozen._retained_mechanism_bindings(  # noqa: SLF001
            allocated, packet
        )
        final_priorities = {
            handle: priority
            for handle, priority in retained_priorities.items()
            if handle in mechanism_by_handle
        }
        expected_final_priority_handles = {
            handle
            for handle, owner in mechanism_by_handle.items()
            if owner
            in {
                frozen.FULL_STORE_MECHANISM,
                frozen.ACTIVE_RECONSTRUCTION_MECHANISM,
            }
        }
        _require(
            set(final_priorities) == expected_final_priority_handles,
            "typed retained active/full priorities changed coverage",
        )

        dedup_by_mechanism = {row.mechanism_id: row for row in allocated}
        base_dedup = tuple(
            dedup_by_mechanism[row.mechanism_id] for row in base_original
        )
        tail_dedup = tuple(
            dedup_by_mechanism[row.mechanism_id] for row in tail_original
        )
        retained_handles = frozenset(
            binding.handle_id for binding in packet.local_bindings
        )
        retained_groups = frozenset(
            binding.source_group_handle for binding in packet.local_bindings
        )
        story_keys, prefit_story_audit = frozen._local_story_keys(  # noqa: SLF001
            parent_map=dedup_by_mechanism[frozen.PARENT_MAP_MECHANISM],
            planned=planned,
            namespace_id=context_row.namespace.namespace_id,
            base_row=base_row,
            base_contributions=base_dedup,
            base_parent_prompt_token_proxy=parent_prompt_proxy,
            tail_row=tail_row,
            tail_contributions=tail_dedup,
            tail_parent_prompt_token_proxy=parent_prompt_proxy,
            full_audit=full_audit,
            active_contribution=active_contribution,
            active_result=active_result,
            retained_handle_ids=retained_handles,
            retained_group_handles=retained_groups,
        )
        forbidden = list(frozen._map_forbidden_literals(planned))  # noqa: SLF001
        if base_row is not None:
            forbidden.extend(frozen._union_forbidden_literals(base_row))  # noqa: SLF001
        if tail_row is not None:
            forbidden.extend(frozen._union_forbidden_literals(tail_row))  # noqa: SLF001
        forbidden.extend(frozen._full_store_forbidden_literals(closure))  # noqa: SLF001
        forbidden.extend(frozen._active_forbidden_literals(active_result))  # noqa: SLF001
        fitted = fit_typed_final_prompt(
            dated_question=source_packet.dated_question,
            parent_prediction=parent_row.prediction,
            packet=packet,
            mechanism_by_handle=mechanism_by_handle,
            local_story_keys_by_group=story_keys,
            local_retention_priority_by_handle=final_priorities,
            forbidden_provider_literals=tuple(dict.fromkeys(forbidden)),
            minimum_usable_items_per_mechanism=1,
            protected_item_receipt_sha256s=protected_item_receipts,
            protection_source_receipt_sha256=fair_audit["receipt_sha256"],
        )
        _require(
            fitted.packet.provider_payload_mode is ProviderPayloadMode.COMPACT_FINAL
            and fitted.prompt_token_proxy + OUTPUT_TOKEN_RESERVE <= 8_000,
            "typed compact final chat escaped its hard envelope",
        )
        final_story_keys, final_story_audit = frozen._local_story_keys(  # noqa: SLF001
            parent_map=dedup_by_mechanism[frozen.PARENT_MAP_MECHANISM],
            planned=planned,
            namespace_id=context_row.namespace.namespace_id,
            base_row=base_row,
            base_contributions=base_dedup,
            base_parent_prompt_token_proxy=parent_prompt_proxy,
            tail_row=tail_row,
            tail_contributions=tail_dedup,
            tail_parent_prompt_token_proxy=parent_prompt_proxy,
            full_audit=full_audit,
            active_contribution=active_contribution,
            active_result=active_result,
            retained_handle_ids=frozenset(fitted.allowed_handle_ids),
            retained_group_handles=frozenset(fitted.handle_group_by_id.values()),
        )
        _require(
            all(set(keys) <= set(story_keys[group]) for group, keys in final_story_keys.items()),
            "typed fitted story provenance escaped the fair packet",
        )
        active_quote_by_handle = {
            binding.handle_id: candidate.quote
            for binding, candidate in zip(
                active_contribution.bindings, active_result.candidates, strict=True
            )
        }
        _require(
            all(
                item.summary == active_quote_by_handle[item.handle_ids[0]]
                for item in fitted.packet.items
                if len(item.handle_ids) == 1
                and item.handle_ids[0] in active_quote_by_handle
            ),
            "typed final fitting rewrote an active exact chunk",
        )
        connectivity = build_typed_connectivity_ledger(
            tuple(original),
            fitted,
            post_selection_dedup_exclusions=postselection_exclusions,
        )
        local_audit = {
            "adaptive_parent_map": frozen._map_local_audit(parent_map, planned),  # noqa: SLF001
            "adaptive_parent_source": (
                None
                if base_row is None
                else frozen._union_local_audit(  # noqa: SLF001
                    base_row,
                    base_original,
                    parent_prompt_token_proxy=parent_prompt_proxy,
                )
            ),
            "adaptive_tail_source": (
                None
                if tail_row is None
                else frozen._union_local_audit(  # noqa: SLF001
                    tail_row,
                    tail_original,
                    parent_prompt_token_proxy=parent_prompt_proxy,
                )
            ),
            "full_store_slot_closure": full_audit.projection(),
            "full_store_selection_priority": full_priority_audit,
            "active_reconstruction": {
                "contribution": active_contribution.projection(),
                "local_result": active_result.local_audit_projection(),
                "parent_alignment": active_alignment,
                "provider_projection_sha256": identity_sha256(
                    active_result.provider_projection()
                ),
                "scanner_batches_reused_without_rescan": True,
            },
            "active_full_selection_priority": active_priority_audit,
            "fair_premerge": fair_audit,
            "fair_premerge_dropped_allocated_bindings": dropped_allocated,
            "non_borrowable_lane_allocation": lane_audit,
            "shared_lane_surplus_fill": surplus_audit,
            "local_to_global_connectivity": connectivity,
            "post_selection_dedup_exclusions": list(postselection_exclusions),
            "retained_fitted_bindings": [
                row.projection() for row in fitted.packet.local_bindings
            ],
            "story_link_local_bindings": [
                dict(row) for row in fitted.story_link_local_bindings
            ],
            "story_source_history_keys_pre_fit": prefit_story_audit,
            "story_source_history_keys": final_story_audit,
            "selection_completed_before_cross_method_dedup": True,
        }
        final_mechanism_counts: dict[str, int] = {}
        for item in fitted.packet.items:
            if not item.included or item.content_conflict:
                continue
            for owner in {
                fitted.mechanism_by_handle[handle] for handle in item.handle_ids
            }:
                final_mechanism_counts[owner] = final_mechanism_counts.get(owner, 0) + 1
        required_mechanisms = {
            row["mechanism_id"]
            for row in fair_audit["mechanisms"]
            if row["usable_candidate_count"] > 0
        }
        _require(
            all(final_mechanism_counts.get(row, 0) >= 1 for row in required_mechanisms),
            "typed final prompt starved a nonempty mechanism",
        )
        local_audit["final_usable_item_count_by_mechanism"] = final_mechanism_counts
        provider_projection = fitted.projection(include_local=False)
        body = {
            "allowed_handle_ids": list(fitted.allowed_handle_ids),
            "dated_question_sha256": source_packet.dated_question_sha256,
            "format": COMPOSITION_FORMAT,
            "handle_group_by_id": dict(fitted.handle_group_by_id),
            "local_audit": local_audit,
            "mechanism_by_handle": dict(fitted.mechanism_by_handle),
            "ordinal": ordinal,
            "parent_prediction": parent_row.prediction,
            "parent_prediction_sha256": parent_row.prediction_sha256,
            "preservation_requirements": dict(fitted.preservation_requirements),
            "provider_projection": provider_projection,
            "question_id": question_id,
            "question_sha256": source_packet.question_sha256,
            "route_id": spec.style.value,
            "story_coherence": dict(fitted.story_coherence),
            "typed_composition_receipt_sha256": fitted.receipt_sha256,
            "validation_contract": dict(fitted.validation_contract),
        }
        composition_rows.append(
            {**body, "composition_row_sha256": identity_sha256(body)}
        )
    _require(
        len(composition_rows) == len(selected_ordinals)
        and tuple(row["ordinal"] for row in composition_rows) == selected_ordinals
        and len({row["question_id"] for row in composition_rows})
        == len(composition_rows),
        "typed final composition population changed",
    )
    return tuple(composition_rows)


def _composition_payload(
    inputs: ConfirmationTypedFinalInputs,
    *,
    output_root: Path,
) -> tuple[dict[str, Any], SealedArtifact]:
    _validate_inputs(inputs)
    inputs.context.revalidate_store_bytes()
    context_rows = _ordered_context_rows(inputs.context)
    ordinal_by_question = {
        row.source.packet.question_id: ordinal
        for ordinal, row in enumerate(context_rows)
    }
    rows_by_namespace: dict[str, list[Any]] = {}
    for row in context_rows:
        rows_by_namespace.setdefault(row.namespace.namespace_id, []).append(row)

    cache_receipts: list[dict[str, Any]] = []
    namespace_artifacts: list[dict[str, Any]] = []
    closure_question_rows: list[dict[str, Any]] = []
    composition_by_ordinal: dict[int, dict[str, Any]] = {}
    live_index_count = 0
    maximum_live_index_count = 0
    for namespace_ordinal, namespace in enumerate(inputs.context.population.namespaces):
        namespace_id = namespace.namespace_id
        namespace_rows = tuple(rows_by_namespace.get(namespace_id, ()))
        _require(bool(namespace_rows), "typed namespace has no bound questions")
        closures, index, cache_receipt = _build_full_store_results(
            inputs,
            namespace=namespace,
            context_rows=namespace_rows,
        )
        live_index_count += 1
        maximum_live_index_count = max(maximum_live_index_count, live_index_count)
        _require(
            live_index_count == 1,
            "typed composition retained more than one namespace index",
        )
        local_ids = tuple(row.source.packet.question_id for row in namespace_rows)
        local_payload = _closure_input_projection(
            closures,
            (cache_receipt,),
            local_ids,
            ordinal_by_question=ordinal_by_question,
        )
        local_path = (
            output_root
            / "typed-final-full-store-namespaces-v1"
            / f"{namespace_ordinal:04d}-{namespace_id}.json"
        )
        local_artifact, _created = publish_sealed_json(local_path, local_payload)
        local_ordinals = tuple(ordinal_by_question[value] for value in local_ids)
        local_composition = _compose_rows(
            inputs,
            closure_by_question=closures,
            index_by_namespace={namespace_id: index},
            closure_artifact_sha256=local_artifact.sha256,
            ordinals=local_ordinals,
        )
        for row in local_composition:
            _require(
                row["ordinal"] not in composition_by_ordinal,
                "typed namespace composition ordinal repeated",
            )
            composition_by_ordinal[row["ordinal"]] = row
        cache_receipts.append(cache_receipt)
        closure_question_rows.extend(local_payload["questions"])
        namespace_artifacts.append(
            {
                "cache_receipt_sha256": cache_receipt["cache_receipt_sha256"],
                "closure_input_sha256": local_artifact.sha256,
                "namespace_id": namespace_id,
                "ordered_question_ids": list(local_ids),
                "window_index_receipt_sha256": index.receipt_sha256,
            }
        )
        # The only large object in this loop is intentionally released before
        # the next immutable 1M namespace is opened.
        del index
        del closures
        live_index_count -= 1

    _require(
        live_index_count == 0
        and maximum_live_index_count == 1
        and len(composition_by_ordinal) == len(context_rows),
        "typed namespace streaming bound or population changed",
    )
    rows = tuple(composition_by_ordinal[index] for index in range(len(context_rows)))
    closure_question_rows.sort(key=lambda row: row["ordinal"])
    closure_payload = {
        "cache_receipts": cache_receipts,
        "database_read_passes_per_unique_namespace": 1,
        "format": CLOSURE_INPUT_FORMAT,
        "gold_loaded": False,
        "maximum_simultaneous_namespace_indexes": maximum_live_index_count,
        "namespace_closure_artifacts": namespace_artifacts,
        "new_provider_calls": 0,
        "question_count": len(closure_question_rows),
        "questions": closure_question_rows,
        "retained_transformer_token_state_bytes": 0,
        "unique_namespace_count": len(cache_receipts),
    }
    assert_gold_blind(closure_payload, path="confirmation_typed_final_full_store_manifest")
    closure, _created = publish_sealed_json(
        output_root / CLOSURE_INPUT_NAME, closure_payload
    )
    payload = {
        "cache_receipts": list(cache_receipts),
        "closure_input_artifact_sha256": closure.sha256,
        "context_binding_identity_sha256": inputs.context.binding_identity_sha256,
        "database_read_passes_per_unique_namespace": 1,
        "format": COMPOSITION_ARTIFACT_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "parent_adaptive_run_sha256": inputs.adaptive_plane.run_artifact.sha256,
        "parent_map_run_sha256": (
            inputs.adaptive_plane.plan.upstream.map_plane.run_sha256
        ),
        "parent_source_materialization_sha256": (
            inputs.base_plane.materialization_artifact.sha256
        ),
        "maximum_simultaneous_namespace_indexes": maximum_live_index_count,
        "namespace_closure_artifact_sha256s": [
            row["closure_input_sha256"] for row in namespace_artifacts
        ],
        "question_count": len(rows),
        "questions": list(rows),
        "retained_transformer_token_state_bytes": 0,
        "tail_materialization_sha256": inputs.tail_plane.run_artifact.sha256,
        "unique_namespace_count": len(cache_receipts),
    }
    assert_gold_blind(payload, path="confirmation_typed_final_composition")
    return payload, closure


def materialize_confirmation_typed_composition(
    inputs: ConfirmationTypedFinalInputs,
    *,
    output_root: str | Path,
) -> ConfirmationTypedComposition:
    """Build and seal the provider-free arbitrary-N typed composition."""

    if type(inputs) is not ConfirmationTypedFinalInputs:
        raise TypeError("inputs must be an exact ConfirmationTypedFinalInputs")
    root = Path(output_root)
    payload, closure = _composition_payload(inputs, output_root=root)
    composition, _created = publish_sealed_json(root / COMPOSITION_NAME, payload)
    return ConfirmationTypedComposition(inputs, closure, composition)


def replay_confirmation_typed_composition(
    inputs: ConfirmationTypedFinalInputs,
    *,
    output_root: str | Path,
    expected_closure_input_sha256: str,
    expected_composition_sha256: str,
) -> ConfirmationTypedComposition:
    """Revalidate stores and require a byte-identical provider-free rebuild."""

    result = materialize_confirmation_typed_composition(
        inputs, output_root=output_root
    )
    _require(
        result.closure_input_artifact.sha256
        == require_sha256(expected_closure_input_sha256, "typed closure input")
        and result.composition_artifact.sha256
        == require_sha256(expected_composition_sha256, "typed composition"),
        "typed composition replay bytes changed",
    )
    payload = {
        "byte_identical": True,
        "closure_input_artifact_sha256": result.closure_input_artifact.sha256,
        "composition_artifact_sha256": result.composition_artifact.sha256,
        "format": COMPOSITION_REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "stores_revalidated_during_replay": True,
    }
    assert_gold_blind(payload, path="confirmation_typed_composition_replay")
    publish_sealed_json(Path(output_root) / COMPOSITION_REPLAY_NAME, payload)
    return result


def _prompt_plan_row(composition_row: Mapping[str, Any]) -> dict[str, Any]:
    """Render and reseal one compact provider row without local locators."""

    row = frozen._prompt_plan_row(composition_row)  # noqa: SLF001
    _require(
        type(row) is dict
        and type(row.get("messages")) is list
        and row.get("prompt_token_proxy", 0) + OUTPUT_TOKEN_RESERVE <= 8_000,
        "typed prompt row changed compact envelope",
    )
    serialized = str(row["messages"])
    local = composition_row.get("local_audit")
    _require(type(local) is dict, "typed composition lost its local audit")
    # The core fitter already checked every concrete locator.  Also keep the
    # structural firewall explicit at this final prompt boundary.
    for forbidden_key in (
        "namespace_id",
        "partition_id",
        "question_id",
        "source_id",
        "source_prefix",
        "store_path",
    ):
        _require(
            f'"{forbidden_key}"' not in serialized,
            f"typed provider prompt leaked local key {forbidden_key}",
        )
    return row


def _preflight_payload(
    composition: SealedArtifact,
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[
    dict[str, Any],
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    require_text(model, "typed final model")
    require_text(gateway_url, "typed final gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "typed final concurrency changed",
    )
    raw_rows = composition.payload.get("questions")
    _require(type(raw_rows) is list and bool(raw_rows), "typed composition rows changed")
    rows = tuple(_prompt_plan_row(row) for row in raw_rows)
    prompts = tuple(
        tuple({"role": item["role"], "content": item["content"]} for item in row["messages"])
        for row in rows
    )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    count = len(rows)
    _require(
        population.logical_prompt_count == population.unique_prompt_count == count,
        "typed final requires one distinct physical prompt per ordered row",
    )
    source_keys = (
        "closure_input_artifact_sha256",
        "context_binding_identity_sha256",
        "parent_adaptive_run_sha256",
        "parent_map_run_sha256",
        "parent_source_materialization_sha256",
        "tail_materialization_sha256",
    )
    payload = {
        "composition_artifact_sha256": composition.sha256,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_prompt_token_cap": 8_000,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in rows
        ),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": list(rows),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": count,
        "required_authorized_provider_calls": count,
        "retained_transformer_token_state_bytes": 0,
        "source_hash_bindings": {
            key: composition.payload[key] for key in source_keys
        },
    }
    assert_gold_blind(payload, path="confirmation_typed_final_preflight")
    return payload, prompts, rows


def publish_confirmation_typed_final_preflight(
    composition: ConfirmationTypedComposition,
    *,
    output_root: str | Path,
    model: str = provider_runtime.DEFAULT_TERRA_GATEWAY_MODEL,
    gateway_url: str = provider_runtime.DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> ConfirmationTypedFinalPreflight:
    """Seal the exact all-row Terra prompt population."""

    if type(composition) is not ConfirmationTypedComposition:
        raise TypeError("composition must be an exact ConfirmationTypedComposition")
    payload, _prompts, _rows = _preflight_payload(
        composition.composition_artifact,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    artifact, _created = publish_sealed_json(
        Path(output_root) / PREFLIGHT_NAME, payload
    )
    return ConfirmationTypedFinalPreflight(composition, artifact)


def _verified_preflight(
    preflight: ConfirmationTypedFinalPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    if type(preflight) is not ConfirmationTypedFinalPreflight:
        raise TypeError("preflight must be an exact ConfirmationTypedFinalPreflight")
    artifact = read_sealed_json(Path(output_root) / PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == preflight.artifact.sha256
        == require_sha256(expected_preflight_sha256, "typed final preflight"),
        "typed final preflight changed hash",
    )
    payload, prompts, rows = _preflight_payload(
        preflight.composition.composition_artifact,
        model=str(artifact.payload.get("model")),
        gateway_url=str(artifact.payload.get("gateway_url")),
        max_concurrency=int(artifact.payload.get("max_concurrency", 0)),
    )
    _require(
        artifact.payload == payload,
        "typed final preflight differs from exact composition",
    )
    return artifact, prompts, rows


def _runtime(
    preflight: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: str | Path,
    client: Any | None,
) -> FastCompletionRuntime:
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=str(preflight.payload["model"]),
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=int(preflight.payload["max_concurrency"]),
        retries=0,
        benchmark_provenance={
            "arm": "confirmation_typed_final_v1",
            "authorized_unique_calls": preflight.payload[
                "required_authorized_provider_calls"
            ],
            "composition_artifact_sha256": preflight.payload[
                "composition_artifact_sha256"
            ],
            "experiment_format": RUN_FORMAT,
            "gateway_url": preflight.payload["gateway_url"],
            "gold_loaded": False,
            "preflight_artifact_sha256": preflight.sha256,
            "source_hash_bindings": preflight.payload["source_hash_bindings"],
        },
    )


def _checkpoint_records(
    preflight: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: str | Path,
) -> tuple[dict[str, str], ...]:
    checkpoint = Path(output_root) / CHECKPOINT_DIR_NAME
    if not checkpoint.exists():
        return ()
    _require(
        checkpoint.is_dir() and not checkpoint.is_symlink(),
        "typed checkpoint root is unsafe",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in checkpoint.iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            "typed checkpoint contains unsafe state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_NAME.fullmatch(path.name)
        _require(match is not None, "typed checkpoint contains foreign state")
        assert match is not None
        (requests if match.group("kind") == "request" else responses).add(
            match.group("key")
        )
    _require(
        requests == responses,
        "typed request/response pair is incomplete; unsafe retry forbidden",
    )
    if not requests:
        return ()
    runtime = _runtime(preflight, prompts, output_root=output_root, client=None)
    try:
        with runtime._journal_guard():  # noqa: SLF001 - authentication only
            records = runtime._load_all_records()  # noqa: SLF001
        call_keys = dict(runtime._call_keys)  # noqa: SLF001
    finally:
        runtime.close()
    _require(len(records) == len(requests), "typed checkpoint population changed")
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for prompt in prompts:
        messages_sha = identity_sha256(
            [{"role": row["role"], "content": row["content"]} for row in prompt]
        )
        if messages_sha in seen:
            continue
        record = records.get(messages_sha)
        if record is None:
            continue
        _require(
            record.call_key_sha256 == call_keys[messages_sha],
            "typed checkpoint call key changed",
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
    _require(len(ordered) == len(requests), "typed checkpoint order changed")
    return tuple(ordered)


def approve_confirmation_typed_final_release(
    preflight: ConfirmationTypedFinalPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    approve_provider_release: bool,
    authorized_provider_calls: int,
) -> SealedArtifact:
    """Seal approval for exactly the currently absent response journals."""

    _require(approve_provider_release is True, "typed provider release requires approval")
    artifact, prompts, _rows = _verified_preflight(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
    )
    records = _checkpoint_records(artifact, prompts, output_root=output_root)
    remaining = len(prompts) - len(records)
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "typed release authorization must equal exact remaining calls",
    )
    root = Path(output_root).resolve().as_posix()
    body = {
        "approval_opt_in": True,
        "checkpoint_namespace": CHECKPOINT_DIR_NAME,
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
        "total_provider_call_budget": len(prompts),
        "unsafe_retry_policy": "refuse-incomplete-request-response-pair-v1",
    }
    assert_gold_blind(body, path="confirmation_typed_final_release")
    payload = {**body, "release_identity_sha256": identity_sha256(body)}
    release, _created = publish_sealed_json(Path(output_root) / RELEASE_NAME, payload)
    return release


def _verified_release(
    preflight: ConfirmationTypedFinalPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, str], ...],
    tuple[dict[str, str], ...],
]:
    artifact, prompts, rows = _verified_preflight(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
    )
    release = read_sealed_json(Path(output_root) / RELEASE_NAME)
    _require(
        release.sha256 == require_sha256(expected_release_sha256, "typed release")
        and set(release.payload) == _RELEASE_KEYS,
        "typed provider release changed hash or schema",
    )
    body = dict(release.payload)
    declared = body.pop("release_identity_sha256", None)
    _require(declared == identity_sha256(body), "typed release self-seal changed")
    snapshot = release.payload.get("checkpoint_snapshot")
    _require(
        type(snapshot) is dict and set(snapshot) == _SNAPSHOT_KEYS,
        "typed release snapshot schema changed",
    )
    raw_records = snapshot.get("ordered_records")
    _require(
        type(raw_records) is list
        and all(type(row) is dict and set(row) == _RECORD_KEYS for row in raw_records),
        "typed release record schema changed",
    )
    released = tuple(dict(row) for row in raw_records)
    for index, row in enumerate(released):
        for key, value in row.items():
            require_sha256(value, f"typed release record {index} {key}")
    root = Path(output_root).resolve().as_posix()
    _require(
        release.payload.get("format") == RELEASE_FORMAT
        and release.payload.get("approval_opt_in") is True
        and release.payload.get("release_status") == "approved_for_provider_execution"
        and release.payload.get("checkpoint_namespace") == CHECKPOINT_DIR_NAME
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
        == len(prompts) - len(released)
        and release.payload.get("total_provider_call_budget") == len(prompts),
        "typed release bindings changed",
    )
    current = _checkpoint_records(artifact, prompts, output_root=output_root)
    current_by_messages = {row["messages_sha256"]: row for row in current}
    _require(
        all(current_by_messages.get(row["messages_sha256"]) == row for row in released),
        "typed released checkpoint snapshot is not present",
    )
    assert_gold_blind(release.payload, path="confirmation_typed_final_release")
    return artifact, release, prompts, rows, released, current


def _default_client_factory(gateway_url: str, api_key_env: str) -> Any:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError(
            "python-dotenv is required for provider execution"
        ) from exc
    load_dotenv(override=False)
    api_key = os.environ.get(api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {api_key_env}")
    return provider_runtime.make_provider_client(api_key, gateway_url)


def run_confirmation_typed_final_provider(
    preflight: ConfirmationTypedFinalPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = provider_runtime.DEFAULT_API_KEY_ENV,
    client_factory: ClientFactory = _default_client_factory,
) -> ConfirmationTypedProviderExecution:
    """Execute exactly the missing native checkpoints; never retry an orphan."""

    artifact, release, prompts, _rows, released, current = _verified_release(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    remaining = len(prompts) - len(current)
    completed_after_release = len(current) - len(released)
    _require(
        enable_provider is (remaining > 0)
        and type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining
        and completed_after_release >= 0
        and completed_after_release + remaining
        == release.payload["required_authorized_provider_calls"]
        and len(current) + remaining
        == release.payload["total_provider_call_budget"],
        "typed provider authorization must equal exact remaining calls",
    )
    client = (
        client_factory(str(artifact.payload["gateway_url"]), api_key_env)
        if remaining
        else None
    )
    runtime = _runtime(
        artifact, prompts, output_root=output_root, client=client
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == len(prompts)
        and batch.usage.physical_calls == remaining
        and batch.usage.checkpoint_hits == len(current),
        "typed provider execution changed released call accounting",
    )
    return ConfirmationTypedProviderExecution(
        batch, batch.usage.physical_calls, batch.usage.checkpoint_hits
    )


def _client_free_batch(
    preflight: ConfirmationTypedFinalPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    FastCompletionBatch,
]:
    artifact, release, prompts, rows, _released, current = _verified_release(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    _require(
        len(current) == len(prompts),
        "typed materialization requires every released completion",
    )
    runtime = _runtime(artifact, prompts, output_root=output_root, client=None)
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == len(prompts)
        and batch.usage.checkpoint_hits == len(prompts)
        and batch.usage.physical_calls == 0,
        "typed client-free materialization entered provider work",
    )
    return artifact, release, rows, batch


def _materialization_payload(
    preflight: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    count = len(prompt_rows)
    _require(
        count > 0
        and batch.usage.logical_calls == batch.usage.unique_calls == count
        and batch.usage.checkpoint_hits == count
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == count
        and len(batch.unique_records) == count,
        "typed materialization requires complete checkpoint-only completions",
    )
    record_by_messages = {row.messages_sha256: row for row in batch.unique_records}
    _require(len(record_by_messages) == count, "typed completion records repeat")
    results: list[dict[str, Any]] = []
    for plan, completion in zip(
        prompt_rows, batch.logical_completions, strict=True
    ):
        record = record_by_messages.get(plan["messages_sha256"])
        _require(record is not None, "typed completion lost its prompt")
        assert record is not None
        _require(
            record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            "typed checkpoint record changed during materialization",
        )
        results.append(
            materialize_typed_final_result_row(
                plan,
                completion,
                completion_receipt_sha256=record.completion_sha256,
                call_key_sha256=record.call_key_sha256,
                request_journal_sha256=record.request_journal_sha256,
                response_journal_sha256=record.response_journal_sha256,
            )
        )
    judge_rows = [prediction_row_projection(row) for row in results]
    _require(
        tuple(row["ordinal"] for row in results) == tuple(range(count))
        and tuple(row["question_id"] for row in results)
        == tuple(row["question_id"] for row in judge_rows),
        "typed final judge seam changed ordered identities",
    )
    payload = {
        "changed_prediction_count": sum(
            bool(row["changed_from_parent"]) for row in results
        ),
        "completion_batch": batch.model_dump(),
        "composition_artifact_sha256": preflight.payload[
            "composition_artifact_sha256"
        ],
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"] == "typed_final_invalid_keep_parent_v1"
            for row in results
        ),
        "judge_rows": judge_rows,
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": count,
        "questions": results,
        "required_authorized_provider_calls": count,
        "retained_transformer_token_state_bytes": 0,
        "source_hash_bindings": preflight.payload["source_hash_bindings"],
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="confirmation_typed_final_run")
    return payload, tuple(results), tuple(judge_rows)


def materialize_confirmation_typed_final(
    preflight: ConfirmationTypedFinalPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
) -> ConfirmationTypedFinalMaterialization:
    """Materialize typed answers from journals without opening a store."""

    artifact, release, rows, batch = _client_free_batch(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    payload, results, judge_rows = _materialization_payload(artifact, rows, batch)
    terminal, _created = publish_sealed_json(Path(output_root) / RUN_NAME, payload)
    return ConfirmationTypedFinalMaterialization(
        preflight.composition.composition_artifact,
        preflight.composition.closure_input_artifact,
        artifact,
        release,
        terminal,
        batch,
        tuple(row["prediction"] for row in results),
        results,
        judge_rows,
    )


def replay_confirmation_typed_final(
    preflight: ConfirmationTypedFinalPreflight,
    *,
    output_root: str | Path,
    expected_closure_input_sha256: str,
    expected_composition_sha256: str,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    expected_run_sha256: str,
) -> VerifiedConfirmationTypedFinalPlane:
    """Revalidate stores, rebuild composition, and replay exact final bytes."""

    rebuilt = replay_confirmation_typed_composition(
        preflight.composition.inputs,
        output_root=output_root,
        expected_closure_input_sha256=expected_closure_input_sha256,
        expected_composition_sha256=expected_composition_sha256,
    )
    _require(
        rebuilt.composition_artifact.sha256
        == preflight.composition.composition_artifact.sha256,
        "typed final replay changed preflight composition parent",
    )
    materialized = materialize_confirmation_typed_final(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    _require(
        materialized.run_artifact.sha256
        == require_sha256(expected_run_sha256, "typed final run"),
        "typed final run differs from checkpoint-only replay",
    )
    payload = {
        "byte_identical": True,
        "composition_artifact_sha256": rebuilt.composition_artifact.sha256,
        "expected_run_sha256": materialized.run_artifact.sha256,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "preflight_artifact_sha256": materialized.preflight_artifact.sha256,
        "replayed_run_sha256": materialized.run_artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
        "stores_revalidated_during_replay": True,
    }
    assert_gold_blind(payload, path="confirmation_typed_final_replay")
    replay, _created = publish_sealed_json(Path(output_root) / REPLAY_NAME, payload)
    return VerifiedConfirmationTypedFinalPlane(
        rebuilt.composition_artifact,
        rebuilt.closure_input_artifact,
        materialized.preflight_artifact,
        materialized.release_artifact,
        materialized.run_artifact,
        replay,
        materialized.completion_batch,
        materialized.predictions,
        materialized.result_rows,
        materialized.judge_rows,
    )


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "CLOSURE_INPUT_NAME",
    "COMPOSITION_NAME",
    "PREFLIGHT_NAME",
    "RELEASE_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "ConfirmationTypedComposition",
    "ConfirmationTypedFinalError",
    "ConfirmationTypedFinalInputs",
    "ConfirmationTypedFinalMaterialization",
    "ConfirmationTypedFinalPreflight",
    "ConfirmationTypedProviderExecution",
    "VerifiedConfirmationTypedFinalPlane",
    "approve_confirmation_typed_final_release",
    "materialize_confirmation_typed_composition",
    "materialize_confirmation_typed_final",
    "publish_confirmation_typed_final_preflight",
    "replay_confirmation_typed_composition",
    "replay_confirmation_typed_final",
    "run_confirmation_typed_final_provider",
]
