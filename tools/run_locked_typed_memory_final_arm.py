#!/usr/bin/env python3
"""Run the locked common typed-memory final-answer arm.

Lifecycle boundaries are explicit:

* ``compose`` is provider-free and gold-blind.  It replays the locked adaptive
  D+G parent and wave-2 tail, reads every unique 1M namespace exactly once into
  a reusable full-store window index, and seals both the closure inputs and a
  full 100-row typed composition with prompt-external provenance.
* ``preflight`` reads only the sealed composition, renders exactly one complete
  Terra chat prompt per locked question, counts the full wrapper, and seals the
  immutable prompt population.
* ``provider-run`` reads only that sealed prompt population and requires exact
  authorization for its physical call count.
* ``materialize`` reads only immutable completion checkpoints plus the sealed
  preflight; invalid completions retain the exact adaptive parent.
* ``replay`` reads the sealed composition/preflight and requires byte-identical
  terminal materialization using checkpoint hits only.

No phase loads benchmark gold.  Raw source/partition identities are retained
only in the composition's local audit projection and never reach provider
messages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy  # noqa: E402
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_adaptive_evidence_solver_v3 as adaptive_cli  # noqa: E402
from tools import run_locked_adaptive_source_map as source_cli  # noqa: E402
from tools import run_locked_adaptive_source_tail_wave as tail_cli  # noqa: E402
from tools import run_locked_query_guided_scan as guided_scan_cli  # noqa: E402
from tools.matched_eval import live  # noqa: E402
from tools.matched_eval import adaptive_evidence_solver_live as adaptive_live  # noqa: E402
from tools.matched_eval.adaptive_source_tail_typed import (  # noqa: E402
    TailFactUnionRow,
    adapt_tail_question_contributions,
    build_tail_post_map_fact_unions,
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
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    FullStoreSlotClosureResult,
    FullStoreWindowIndex,
    adapt_full_store_slot_closure_to_typed_contribution,
    build_full_store_window_index,
    scan_full_store_slot_closure,
)
from tools.matched_eval.full_store_typed_adapter import (  # noqa: E402
    adapt_full_store_slot_closure,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    cache_namespace_partitions,
)
from tools.matched_eval.source_history_fact_union import (  # noqa: E402
    FactLane,
    pack_fact_union_envelope,
)
from tools.matched_eval.prompt_tick_contracts import (  # noqa: E402
    CallBudget,
    LaneBudget,
)
from tools.matched_eval.typed_lane_allocator import (  # noqa: E402
    TypedLaneAllocation,
    allocate_typed_contribution_lanes,
    fill_typed_lane_surplus,
)
from tools.matched_eval.typed_action_semantics import (  # noqa: E402
    completed_action_concepts,
    matched_action_concepts,
)
from tools.matched_eval.typed_active_full_store_scanner import (  # noqa: E402
    derive_candidate_cue_support_priorities,
    scan_typed_active_full_store,
)
from tools.matched_eval.typed_active_reconstruction import (  # noqa: E402
    MECHANISM_ID as CORE_ACTIVE_RECONSTRUCTION_MECHANISM,
    ActiveReconstructionBudget,
    TypedActiveReconstructionResult,
    adapt_typed_active_reconstruction_to_contribution,
    citation_span_receipt_sha256,
    run_typed_active_reconstruction,
)
from tools.matched_eval.typed_connectivity_ledger import (  # noqa: E402
    build_typed_connectivity_ledger,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    COMPOSITION_FORMAT,
    EXPECTED_QUESTION_COUNT,
    FORMAT as CORE_FORMAT,
    LOCAL_RETENTION_PRIORITY_WIDTH,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    VALIDATOR_POLICY_FORMAT,
    fit_typed_final_prompt,
    judge_row_projection,
    materialize_typed_final_result_row,
    render_final_messages,
)
from tools.matched_eval.typed_operator_adapter import (  # noqa: E402
    ConflictPolicy,
    ContentCoherence,
    EvidenceOrigin,
    EvidenceStatus,
    FrontierMode,
    ParsedTypedItems,
    ProviderPayloadMode,
    TypedEvidenceContribution,
    TypedEvidenceItem,
    TypedEvidencePacket,
    adapt_verified_evidence,
    build_typed_evidence_packet,
)
from tools.matched_eval.typed_operator_spec import (  # noqa: E402
    compile_typed_operator_spec,
    normalized_terms,
)
from tools.matched_eval.typed_story_affinity import (  # noqa: E402
    derive_evidence_story_affinity,
)


FORMAT = "memory-condense-locked-typed-memory-final-arm-v1"
CLOSURE_INPUT_FORMAT = f"{FORMAT}-full-store-closure-input-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RUN_FORMAT = f"{FORMAT}-run-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"
CLOSURE_INPUT_NAME = "typed-memory-final-full-store-input-v1.json"
COMPOSITION_NAME = "typed-memory-final-composition-v1.json"
PREFLIGHT_NAME = "typed-memory-final-preflight-v1.json"
RUN_NAME = "typed-memory-final-run-v1.json"
REPLAY_NAME = "typed-memory-final-replay-v1.json"
CHECKPOINT_DIR_NAME = "terra-typed-memory-final-v1-calls"

DEFAULT_OUTPUT = Path("eval_results/matched_eval_100/typed-memory-final-v1")
DEFAULT_PARENT_SOURCE_ROOT = Path(
    "eval_results/matched_eval_100/"
    "adaptive-source-pareto-consolidated-authority-v1/d1-p0-g1"
)
DEFAULT_PARENT_ROOT = Path("eval_results/matched_eval_100/adaptive-solver-v3-dg")
DEFAULT_TAIL_ROOT = DEFAULT_PARENT_SOURCE_ROOT / "tail-wave-2-recovery-v1"

EXPECTED_PARENT_SOURCE_PREFLIGHT_SHA256 = (
    "216be985c901e47b2bc8ae21917f7417e1443704051f150aba7a4b40dec1a3e6"
)
EXPECTED_PARENT_SOURCE_MATERIALIZATION_SHA256 = (
    "21f4c79c1c0d4d663bca8fffbfb3f38933ae5ab72492434b2af860babfdd03e6"
)
EXPECTED_PARENT_PREFLIGHT_SHA256 = (
    "ba5419cb94c1431ed61b3b519fd8eea0b8aeeb716b433d99c55692920613222e"
)
EXPECTED_PARENT_RUN_SHA256 = (
    "bf1f5238feb67c1ffc2044192f946dcf755d0e14f235cc7252a61c5236c552ca"
)
EXPECTED_TAIL_PREFLIGHT_SHA256 = (
    "c6618f8c1050ec64d0f744c1666484b43e4ac814331dedce7138a3e47d3ea335"
)
EXPECTED_TAIL_MATERIALIZATION_SHA256 = (
    "e482c600ae89b85381d0d9b842ed5bb053770c1d544633241e6da7769c5d52ee"
)
EXPECTED_TAIL_REPLAY_SHA256 = (
    "ae2f7d4ffe1a89f790c12e9256e472a38209717523569ac44cecfc849b338e64"
)

# Per-question contribution ranges.  H/G ranges remain globally disjoint while
# opaque L overlays restore exact cross-method co-membership.
PARENT_MAP_RANGE = 1
PARENT_SOURCE_RANGE = 100_001
PARENT_POINTER_RANGE = 200_001
TAIL_SOURCE_RANGE = 300_001
TAIL_POINTER_RANGE = 400_001
FULL_STORE_RANGE = 500_001
ACTIVE_RECONSTRUCTION_RANGE = 600_001

PARENT_MAP_MECHANISM = "adaptive_parent_map_v1"
PARENT_SOURCE_MECHANISM = "adaptive_parent_source_dg_v1"
PARENT_POINTER_MECHANISM = "adaptive_parent_direct_pointer_v1"
TAIL_SOURCE_MECHANISM = "adaptive_source_tail_wave_2_recovery_v1"
TAIL_POINTER_MECHANISM = "adaptive_source_tail_wave_2_recovery_direct_pointer_v1"
FULL_STORE_MECHANISM = "full_store_slot_closure_v1"
ACTIVE_RECONSTRUCTION_MECHANISM = "active_reconstruction_v1"

# Non-borrowable compact evidence allowances.  The 6,144-token content total
# plus the independent 256-token CAV allowance and 768-token output reserve is
# 7,168; the remaining 832 tokens cover the complete chat/question/parent/
# schema wrappers before exact final fitting.
LANE_CONTENT_TOKEN_CAPS = {
    "protected_parent": 3_072,
    "base_source": 768,
    "tail_source": 512,
    "full_store": 768,
    "active_reconstruction": 1_024,
}
LANE_BY_MECHANISM = {
    PARENT_MAP_MECHANISM: "protected_parent",
    PARENT_SOURCE_MECHANISM: "base_source",
    PARENT_POINTER_MECHANISM: "base_source",
    TAIL_SOURCE_MECHANISM: "tail_source",
    TAIL_POINTER_MECHANISM: "tail_source",
    FULL_STORE_MECHANISM: "full_store",
    ACTIVE_RECONSTRUCTION_MECHANISM: "active_reconstruction",
}


class LockedTypedMemoryFinalError(MatchedEvalContractError):
    """A locked parent, contribution, prompt, checkpoint, or result changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedTypedMemoryFinalError(message)


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _plain_messages(messages: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    return tuple({"role": row["role"], "content": row["content"]} for row in messages)


def _parent_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=Path(args.parent_source_root),
        expected_source_preflight_sha256=args.expected_parent_source_preflight_sha256,
        expected_source_materialization_sha256=(
            args.expected_parent_source_materialization_sha256
        ),
        output_root=Path(args.parent_root),
        lanes=(FactLane.DIRECT, FactLane.GUIDED),
        direct_base_cap=1,
        partition_base_cap=0,
        guided_base_cap=1,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
        expected_preflight_sha256=args.expected_parent_preflight_sha256,
        expected_run_sha256=args.expected_parent_run_sha256,
    )


def _tail_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=Path(args.tail_root),
        expected_preflight_sha256=args.expected_tail_preflight_sha256,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )


def _load_verified_tail(
    args: argparse.Namespace,
) -> tuple[
    SealedArtifact,
    tuple[Any, ...],
    tuple[Any, ...],
]:
    tail_args = _tail_args(args)
    preflight, prompts = tail_cli._read_preflight(tail_args)  # noqa: SLF001
    work, cache_artifact, questions, cache = tail_cli._read_store_free_inputs(  # noqa: SLF001
        tail_args, preflight
    )
    batch = tail_cli._journal_batch(tail_args, preflight, prompts)  # noqa: SLF001
    _require(
        batch.usage.physical_calls == 0
        and batch.usage.checkpoint_hits == len(prompts),
        "tail typed input requires complete checkpoint-only replay",
    )
    results = tail_cli._materialize_results(questions, batch, cache)  # noqa: SLF001
    expected_payload = tail_cli.materialization_projection(
        preflight, work, cache_artifact, results, batch
    )
    terminal = read_sealed_json(Path(args.tail_root) / tail_cli.MATERIALIZATION_NAME)
    _require(
        terminal.sha256 == require_sha256(
            args.expected_tail_materialization_sha256,
            "expected tail materialization",
        )
        and terminal.payload == expected_payload,
        "tail typed terminal changed from checkpoint-only reconstruction",
    )
    replay = read_sealed_json(Path(args.tail_root) / tail_cli.REPLAY_NAME)
    _require(
        replay.sha256
        == require_sha256(args.expected_tail_replay_sha256, "expected tail replay")
        and replay.payload.get("byte_identical") is True
        and replay.payload.get("expected_materialization_sha256") == terminal.sha256
        and replay.payload.get("provider_calls_during_replay") == 0,
        "tail typed input is not the replay-verified wave-2 terminal",
    )
    return terminal, questions, results


def _load_parent_source_rows(
    args: argparse.Namespace,
) -> tuple[SealedArtifact, tuple[TailFactUnionRow, ...]]:
    source_preflight, _work, terminal, questions, results = (
        source_cli.load_typed_materialization_root(
            args.parent_source_root,
            expected_preflight_sha256=args.expected_parent_source_preflight_sha256,
            expected_materialization_sha256=(
                args.expected_parent_source_materialization_sha256
            ),
            model=str(args.model),
            gateway_url=str(args.gateway_url),
            max_concurrency=int(args.max_concurrency),
            direct_base_cap=1,
            partition_base_cap=0,
            guided_base_cap=1,
        )
    )
    _require(
        source_preflight.sha256 == args.expected_parent_source_preflight_sha256,
        "base typed source preflight changed",
    )
    return terminal, build_tail_post_map_fact_unions(questions, results)


def _guided_context(args: argparse.Namespace) -> Any:
    return guided_scan_cli._load_context(  # noqa: SLF001
        argparse.Namespace(
            retrieval=Path(args.retrieval),
            store_root=Path(args.store_root),
            parent_output_root=Path(args.query_parent_output_root),
            expected_retrieval_sha256=args.expected_retrieval_sha256,
            expected_parent_preflight_sha256=(
                args.expected_query_parent_preflight_sha256
            ),
        )
    )


def _evidence_items_belong_to_namespace(
    evidence_items: Sequence[Any],
    namespace: Any,
) -> bool:
    """Join EvidenceItem.source_id to the frozen namespace membership."""

    source_ids = {row.source_id for row in namespace.sources}
    return all(
        hasattr(row, "source_id") and row.source_id in source_ids
        for row in evidence_items
    )


def _build_full_store_results(
    args: argparse.Namespace,
    dated_by_question: Mapping[str, str],
) -> tuple[
    Any,
    dict[str, FullStoreSlotClosureResult],
    dict[str, FullStoreWindowIndex],
    tuple[dict[str, Any], ...],
]:
    context = _guided_context(args)
    index_by_namespace: dict[str, FullStoreWindowIndex] = {}
    cache_receipts: list[dict[str, Any]] = []
    for namespace in context.population.namespaces:
        namespace_id = namespace.namespace_id
        _require(namespace_id not in index_by_namespace, "full-store namespace repeated")
        database_path = context.store_dirs_by_namespace[namespace_id] / "memory.db"
        with Database(database_path, read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                namespace,
                source_database_sha256=(
                    context.database_sha256_by_namespace[namespace_id]
                ),
                source_store_receipt_sha256=(
                    namespace.combined_store_receipt_sha256
                ),
            )
        index = build_full_store_window_index(cache)
        index_by_namespace[namespace_id] = index
        cache_receipts.append(
            {
                "cache_receipt_sha256": cache.cache_receipt_sha256,
                "content_row_count": cache.content_row_count,
                "database_read_passes": 1,
                "namespace_id": namespace_id,
                "physical_store_row_count": cache.physical_store_row_count,
                "window_index_receipt_sha256": index.receipt_sha256,
            }
        )
    result: dict[str, FullStoreSlotClosureResult] = {}
    for prompt in context.population.rows:
        question_id = prompt.source.packet.question_id
        _require(question_id in dated_by_question, "closure question escaped parent")
        dated_question = dated_by_question[question_id]
        _require(
            prompt.source.packet.dated_question == dated_question,
            "closure dated question differs from adaptive parent",
        )
        result[question_id] = scan_full_store_slot_closure(
            index_by_namespace[prompt.namespace.namespace_id], dated_question
        )
    _require(
        len(index_by_namespace) == len(context.population.namespaces)
        and len(result) == EXPECTED_QUESTION_COUNT,
        "full-store cache/result population changed",
    )
    return context, result, index_by_namespace, tuple(cache_receipts)


def _closure_input_projection(
    results: Mapping[str, FullStoreSlotClosureResult],
    cache_receipts: tuple[dict[str, Any], ...],
    ordered_question_ids: Sequence[str],
) -> dict[str, Any]:
    rows = []
    for ordinal, question_id in enumerate(ordered_question_ids):
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
        "cache_receipts": list(cache_receipts),
        "database_read_passes_per_unique_namespace": 1,
        "format": CLOSURE_INPUT_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "question_count": len(rows),
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "unique_namespace_count": len(cache_receipts),
    }
    assert_gold_blind(payload, path="typed_final_full_store_input")
    return payload


def _packet_contribution(
    packet: TypedEvidencePacket,
    *,
    mechanism_id: str,
    sealed_artifact_sha256: str,
) -> TypedEvidenceContribution:
    _require(
        all(
            row.sealed_artifact_sha256 == sealed_artifact_sha256
            for row in packet.local_bindings
        ),
        "packet contribution crossed its sealed artifact",
    )
    parsed = ParsedTypedItems(
        packet.items,
        packet.rejected_items,
        identity_sha256(
            {
                "format": f"{FORMAT}-packet-contribution-parse-v1",
                "mechanism_id": mechanism_id,
                "packet_receipt_sha256": packet.receipt_sha256,
            }
        ),
    )
    return TypedEvidenceContribution(
        mechanism_id,
        packet.local_bindings,
        parsed,
        sealed_artifact_sha256,
        packet.frontier.mode,
        packet.frontier.truncated,
    )


def _canonical_story_keys(
    namespace_id: str,
    source_id: str,
) -> tuple[tuple[str, ...], str | None]:
    """Return source/history keys and identify a distinct history overlay."""

    affinity = derive_evidence_story_affinity(namespace_id, source_id)
    return (
        affinity.story_keys,
        (
            affinity.history_story_key_sha256
            if affinity.history_key_distinct_from_source
            else None
        ),
    )


def _canonical_coordinate_span_key(
    namespace_id: str,
    source_id: str,
    chunk_id: str,
    start_char: int,
    end_char: int,
    quote_receipt_sha256: str,
) -> str:
    require_sha256(namespace_id, "span namespace")
    require_text(source_id, "span source")
    require_text(chunk_id, "span chunk")
    _require(
        type(start_char) is int
        and type(end_char) is int
        and 0 <= start_char < end_char,
        "span coordinates changed",
    )
    require_sha256(quote_receipt_sha256, "span quote")
    return identity_sha256(
        {
            "chunk_id": chunk_id,
            "end_char": end_char,
            "format": f"{FORMAT}-canonical-coordinate-span-v1",
            "namespace_id": namespace_id,
            "quote_sha256": quote_receipt_sha256,
            "source_id": source_id,
            "start_char": start_char,
        }
    )


def _canonical_native_evidence_key(
    namespace_id: str,
    source_id: str,
    evidence_id: str,
    quote_receipt_sha256: str,
) -> str:
    """Seal a native immutable evidence identity without guessing coordinates.

    Native IDs and coordinate keys deliberately occupy different domains.  A
    caller that cannot prove a common coordinate therefore retains both
    selected items instead of coalescing same-text occurrences speculatively.
    """

    require_sha256(namespace_id, "evidence namespace")
    require_text(source_id, "evidence source")
    require_text(evidence_id, "native evidence ID")
    require_sha256(quote_receipt_sha256, "evidence quote")
    return identity_sha256(
        {
            "evidence_id": evidence_id,
            "format": f"{FORMAT}-canonical-native-evidence-v1",
            "namespace_id": namespace_id,
            "quote_sha256": quote_receipt_sha256,
            "source_id": source_id,
        }
    )


def _map_exact_span_keys(
    contribution: TypedEvidenceContribution,
    planned: Any,
    namespace_id: str,
) -> dict[str, tuple[str, ...]]:
    aliases = {row.alias: row for row in planned.map_plan_row.aliases}
    mapped = {row.item_sha256: row for row in planned.map_row.accepted_items}
    result: dict[str, tuple[str, ...]] = {}
    for binding in contribution.bindings:
        item = mapped.get(binding.evidence_receipt_sha256)
        _require(item is not None and item.alias in aliases, "map span binding changed")
        assert item is not None
        alias = aliases[item.alias]
        result[binding.handle_id] = (
            _canonical_native_evidence_key(
                namespace_id,
                alias.source_id,
                alias.evidence_id,
                binding.citation_sha256,
            ),
        )
    return result


def _union_exact_span_keys(
    contribution: TypedEvidenceContribution,
    row: TailFactUnionRow,
    *,
    parent_prompt_token_proxy: int,
) -> dict[str, tuple[str, ...]]:
    envelope = pack_fact_union_envelope(
        row.fact_union,
        parent_prompt_token_proxy=parent_prompt_token_proxy,
    )
    admissions = {
        admission.receipt_sha256: admission
        for lane_pack in envelope.lane_packs
        for admission in lane_pack.admissions
    }
    exclusions = {
        exclusion.receipt_sha256: exclusion
        for exclusion in row.fact_union.direct_exclusions
    }
    direct = {ref.evidence_id: ref for ref in row.question_plan.direct_evidence}
    result: dict[str, tuple[str, ...]] = {}
    for binding in contribution.bindings:
        if binding.origin is EvidenceOrigin.SOURCE_FACT:
            admission = admissions.get(binding.evidence_receipt_sha256)
            _require(admission is not None, "source-fact span admission changed")
            assert admission is not None
            keys = tuple(
                _canonical_coordinate_span_key(
                    origin.namespace_id,
                    origin.source_id,
                    origin.chunk_id,
                    origin.quote_start_char,
                    origin.quote_end_char,
                    origin.quote_sha256,
                )
                for origin in admission.union_fact.origins
            )
        elif binding.origin is EvidenceOrigin.DIRECT_POINTER:
            exclusion = exclusions.get(binding.evidence_receipt_sha256)
            _require(exclusion is not None, "direct-pointer span exclusion changed")
            assert exclusion is not None
            refs = tuple(direct.get(row) for row in exclusion.matching_direct_evidence_ids)
            _require(
                bool(refs) and all(ref is not None for ref in refs),
                "direct-pointer span evidence changed",
            )
            keys = tuple(
                _canonical_native_evidence_key(
                    ref.namespace_id,
                    ref.source_id,
                    ref.evidence_id,
                    ref.quote_sha256,
                )
                for ref in refs
                if ref is not None
            )
        else:  # pragma: no cover - typed contribution seal guards this
            raise LockedTypedMemoryFinalError("union span origin changed")
        result[binding.handle_id] = tuple(dict.fromkeys(keys))
    return result


def _full_store_exact_span_keys(
    audit: Any,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for row in audit.local_citation_bindings:
        local = row["local_citation_binding"]
        span = local["span"]
        result[row["handle_id"]] = (
            _canonical_coordinate_span_key(
                local["namespace_id"],
                local["source_id"],
                span["chunk_id"],
                span["start_char"],
                span["end_char"],
                local["quote_sha256"],
            ),
        )
    return result


def _active_exact_span_keys(
    contribution: TypedEvidenceContribution,
    result: TypedActiveReconstructionResult,
) -> dict[str, tuple[str, ...]]:
    """Bind every active H handle to its exact hydrated local span."""

    _require(
        len(contribution.bindings)
        == len(result.candidates)
        == len(result.local_bindings),
        "active exact-span contribution alignment changed",
    )
    exact: dict[str, tuple[str, ...]] = {}
    for binding, candidate, local in zip(
        contribution.bindings,
        result.candidates,
        result.local_bindings,
        strict=True,
    ):
        _require(
            binding.citation_sha256 == candidate.quote_sha256
            and binding.citation_char_count == len(candidate.quote),
            "active exact chunk binding changed",
        )
        exact[binding.handle_id] = (
            _canonical_coordinate_span_key(
                local.namespace_id,
                local.source_id,
                local.span.chunk_id,
                local.span.start_char,
                local.span.end_char,
                local.quote_sha256,
            ),
        )
    return exact


def _full_store_story_keys(
    audit: Any,
    *,
    retained_handle_ids: frozenset[str] | None = None,
) -> tuple[dict[str, tuple[str, ...]], frozenset[str]]:
    """Rebuild common source/history keys from exact local citations."""

    merged: dict[str, list[str]] = {}
    history_keys: set[str] = set()
    for row in audit.local_citation_bindings:
        if (
            retained_handle_ids is not None
            and row["handle_id"] not in retained_handle_ids
        ):
            continue
        local = row["local_citation_binding"]
        keys, history_key = _canonical_story_keys(
            local["namespace_id"], local["source_id"]
        )
        merged.setdefault(row["opaque_group_handle"], []).extend(keys)
        if history_key is not None:
            history_keys.add(history_key)
    return (
        {
            group: tuple(dict.fromkeys(values))
            for group, values in merged.items()
        },
        frozenset(history_keys),
    )


def _active_story_keys(
    contribution: TypedEvidenceContribution,
    result: TypedActiveReconstructionResult,
    *,
    retained_handle_ids: frozenset[str] | None = None,
) -> tuple[dict[str, tuple[str, ...]], frozenset[str]]:
    """Derive active source/history links only from admitted exact spans."""

    _require(
        len(contribution.bindings) == len(result.local_bindings),
        "active story binding alignment changed",
    )
    merged: dict[str, list[str]] = {}
    history_keys: set[str] = set()
    for binding, local in zip(
        contribution.bindings, result.local_bindings, strict=True
    ):
        if (
            retained_handle_ids is not None
            and binding.handle_id not in retained_handle_ids
        ):
            continue
        keys, history_key = _canonical_story_keys(
            local.namespace_id, local.source_id
        )
        merged.setdefault(binding.source_group_handle, []).extend(keys)
        if history_key is not None:
            history_keys.add(history_key)
    return (
        {
            group: tuple(dict.fromkeys(values))
            for group, values in merged.items()
        },
        frozenset(history_keys),
    )


def _map_story_keys(
    contribution: TypedEvidenceContribution,
    planned: Any,
    namespace_id: str,
    *,
    retained_handle_ids: frozenset[str] | None = None,
) -> tuple[dict[str, tuple[str, ...]], frozenset[str]]:
    aliases = {row.alias: row for row in planned.map_plan_row.aliases}
    mapped = {
        row.item_sha256: row for row in planned.map_row.accepted_items
    }
    result: dict[str, list[str]] = {}
    history_keys: set[str] = set()
    for binding in contribution.bindings:
        if (
            retained_handle_ids is not None
            and binding.handle_id not in retained_handle_ids
        ):
            continue
        item = mapped.get(binding.evidence_receipt_sha256)
        _require(item is not None and item.alias in aliases, "map story binding changed")
        keys, history_key = _canonical_story_keys(
            namespace_id, aliases[item.alias].source_id
        )
        result.setdefault(binding.source_group_handle, []).extend(keys)
        if history_key is not None:
            history_keys.add(history_key)
    return (
        {
            group: tuple(dict.fromkeys(values))
            for group, values in result.items()
        },
        frozenset(history_keys),
    )


def _union_story_keys(
    contribution: TypedEvidenceContribution,
    row: TailFactUnionRow,
    *,
    parent_prompt_token_proxy: int,
    retained_handle_ids: frozenset[str] | None = None,
) -> tuple[dict[str, tuple[str, ...]], frozenset[str]]:
    envelope = pack_fact_union_envelope(
        row.fact_union,
        parent_prompt_token_proxy=parent_prompt_token_proxy,
    )
    admissions = {
        admission.receipt_sha256: admission
        for pack in envelope.lane_packs
        for admission in pack.admissions
    }
    exclusions = {
        exclusion.receipt_sha256: exclusion
        for exclusion in row.fact_union.direct_exclusions
    }
    direct = {
        ref.evidence_id: ref for ref in row.question_plan.direct_evidence
    }
    result: dict[str, list[str]] = {}
    history_keys: set[str] = set()
    for binding in contribution.bindings:
        if (
            retained_handle_ids is not None
            and binding.handle_id not in retained_handle_ids
        ):
            continue
        keys: list[str] = []
        if binding.origin is EvidenceOrigin.SOURCE_FACT:
            admission = admissions.get(binding.evidence_receipt_sha256)
            _require(admission is not None, "source-fact story admission changed")
            for origin in admission.union_fact.origins:
                origin_keys, history_key = _canonical_story_keys(
                    origin.namespace_id, origin.source_id
                )
                keys.extend(origin_keys)
                if history_key is not None:
                    history_keys.add(history_key)
        elif binding.origin is EvidenceOrigin.DIRECT_POINTER:
            exclusion = exclusions.get(binding.evidence_receipt_sha256)
            _require(exclusion is not None, "direct-pointer exclusion changed")
            for evidence_id in exclusion.matching_direct_evidence_ids:
                ref = direct.get(evidence_id)
                _require(ref is not None, "direct-pointer protected evidence changed")
                ref_keys, history_key = _canonical_story_keys(
                    ref.namespace_id, ref.source_id
                )
                keys.extend(ref_keys)
                if history_key is not None:
                    history_keys.add(history_key)
        else:  # pragma: no cover - adapter type seal guards this
            raise LockedTypedMemoryFinalError("union contribution origin changed")
        result.setdefault(binding.source_group_handle, []).extend(keys)
    return (
        {
            group: tuple(dict.fromkeys(values))
            for group, values in result.items()
        },
        frozenset(history_keys),
    )


def _union_local_audit(
    row: TailFactUnionRow,
    contributions: Sequence[TypedEvidenceContribution],
    *,
    parent_prompt_token_proxy: int,
) -> dict[str, Any]:
    facts = []
    for fact in row.fact_union.union_facts_before_direct_exclusion:
        facts.append(
            {
                "dedup_key_sha256": fact.dedup_key_sha256,
                "event": None if fact.event_tuple is None else fact.event_tuple.projection(),
                "fact_variants": list(fact.fact_variants),
                "origins": [origin.projection() for origin in fact.origins],
                "owner_lane": fact.owner_lane.value,
                "receipt_sha256": fact.receipt_sha256,
                "union_fact_id": fact.union_fact_id,
            }
        )
    envelope = pack_fact_union_envelope(
        row.fact_union,
        parent_prompt_token_proxy=parent_prompt_token_proxy,
    )
    admission_by_receipt = {
        admission.receipt_sha256: admission
        for lane_pack in envelope.lane_packs
        for admission in lane_pack.admissions
    }
    source_fact_admission_bindings = []
    for contribution in contributions:
        for binding in contribution.bindings:
            if binding.origin is not EvidenceOrigin.SOURCE_FACT:
                continue
            admission = admission_by_receipt.get(binding.evidence_receipt_sha256)
            _require(
                admission is not None,
                "source-fact local audit lost its admission",
            )
            assert admission is not None
            body = {
                "admission_receipt_sha256": binding.evidence_receipt_sha256,
                "binding_receipt_sha256": binding.receipt_sha256,
                "exact_origins": [
                    origin.projection() for origin in admission.union_fact.origins
                ],
                "handle_id": binding.handle_id,
                "union_fact_id": admission.union_fact.union_fact_id,
                "union_fact_receipt_sha256": (
                    admission.union_fact.receipt_sha256
                ),
            }
            source_fact_admission_bindings.append(
                {**body, "receipt_sha256": identity_sha256(body)}
            )
    semantic_direct_pointers = [
        binding.projection()
        for contribution in contributions
        for binding in contribution.bindings
        if binding.origin is EvidenceOrigin.DIRECT_POINTER
    ]
    return {
        "contributions": [
            {
                "bindings": [binding.projection() for binding in contribution.bindings],
                "contribution": contribution.projection(),
            }
            for contribution in contributions
        ],
        "direct_evidence": [
            ref.projection() for ref in row.question_plan.direct_evidence
        ],
        "direct_exclusions": [
            {
                "match_modes": list(exclusion.match_modes),
                "matching_direct_evidence_ids": list(
                    exclusion.matching_direct_evidence_ids
                ),
                "receipt_sha256": exclusion.receipt_sha256,
                "union_fact_id": exclusion.union_fact_id,
            }
            for exclusion in row.fact_union.direct_exclusions
        ],
        "fact_union_receipt_sha256": row.fact_union.receipt_sha256,
        "facts_before_direct_exclusion": facts,
        "retained_union_fact_ids": [
            fact.union_fact_id for fact in row.fact_union.retained_facts
        ],
        "semantic_direct_pointers": semantic_direct_pointers,
        "source_fact_admission_bindings": source_fact_admission_bindings,
        "typed_row": row.projection(),
    }


def _dedup_selected_contributions(
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    exact_span_keys_by_handle: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[TypedEvidenceContribution, ...], tuple[dict[str, Any], ...]]:
    """Identity-proven semantic dedup after every mechanism has selected.

    Same text or the same typed projection is insufficient: every cross-method
    exclusion also requires a shared immutable evidence ID or exact chunk
    coordinates.  DIRECT_POINTER items always survive the within-mechanism
    raw-citation exclusion and are coalesced only when that identity plus
    compatible action/entity/status semantics proves equivalence.
    """

    span_keys = {
        handle: tuple(values)
        for handle, values in (exact_span_keys_by_handle or {}).items()
    }
    for handle, values in span_keys.items():
        require_text(handle, "dedup span handle")
        _require(bool(values), "dedup span binding is empty")
        for value in values:
            require_sha256(value, "dedup exact span receipt")

    def provider_semantic_projection(item: TypedEvidenceItem) -> dict[str, Any]:
        """Return all typed semantics that can affect final provider context."""

        value = item.projection(include_receipt=False)
        # These are representation identities rather than provider-visible
        # semantics.  Everything else, including the exact summary bytes,
        # qualifier, authority, relation, slots, and personalization anchors,
        # must agree before one selected representation may subsume another.
        value.pop("handle_ids")
        value.pop("item_id")
        return value

    semantic_records: dict[
        str,
        list[tuple[str, TypedEvidenceItem, frozenset[str], tuple[str, ...]]],
    ] = {}
    rebuilt_by_index: dict[int, TypedEvidenceContribution] = {}
    exclusions: list[dict[str, Any]] = []
    # A richer hydrated representation must never be discarded merely because
    # a compressed mechanism appeared first in the stack.  Strongest-method
    # ownership is deterministic, while the returned contribution order stays
    # identical to the caller's declared lane order.
    processing_order = tuple(
        sorted(
            enumerate(contributions),
            key=lambda row: (-_mechanism_strength(row[1].mechanism_id), row[0]),
        )
    )
    for contribution_index, contribution in processing_order:
        binding_by_handle = {row.handle_id: row for row in contribution.bindings}
        accepted = []
        for item in contribution.parsed.accepted_items:
            key = identity_sha256(provider_semantic_projection(item))
            item_spans = frozenset(
                span
                for handle in item.handle_ids
                for span in span_keys.get(handle, ())
            )
            semantic_duplicate = next(
                (
                    record
                    for record in semantic_records.get(key, ())
                    if record[0] != contribution.mechanism_id
                    and bool(item_spans)
                    and record[2] == item_spans
                ),
                None,
            )
            if semantic_duplicate is not None:
                (
                    owner_mechanism,
                    owner_item,
                    owner_spans,
                    owner_bindings,
                ) = semantic_duplicate
                shared_spans = tuple(sorted(owner_spans & item_spans))
                exclusions.append(
                    {
                        "duplicate_binding_receipt_sha256s": [
                            binding_by_handle[handle].receipt_sha256
                            for handle in item.handle_ids
                        ],
                        "duplicate_item_receipt_sha256": item.receipt_sha256,
                        "duplicate_mechanism_id": contribution.mechanism_id,
                        "operation_position": "after_each_mechanism_selection",
                        "owner_binding_receipt_sha256s": list(owner_bindings),
                        "owner_item_receipt_sha256": owner_item.receipt_sha256,
                        "owner_mechanism_id": owner_mechanism,
                        "semantic_dedup_key_sha256": key,
                        "shared_exact_span_receipt_sha256s": list(shared_spans),
                        "dedup_proof": (
                            "shared_immutable_evidence_identity_plus_exact_"
                            "provider_semantic_projection"
                        ),
                    }
                )
                continue
            accepted.append(item)
            semantic_records.setdefault(key, []).append(
                (
                    contribution.mechanism_id,
                    item,
                    item_spans,
                    tuple(
                        binding_by_handle[handle].receipt_sha256
                        for handle in item.handle_ids
                    ),
                )
            )
        represented_handles = {
            handle for item in accepted for handle in item.handle_ids
        }
        retained_bindings = tuple(
            binding
            for binding in contribution.bindings
            if binding.handle_id in represented_handles
        )
        _require(
            {binding.handle_id for binding in retained_bindings}
            == represented_handles,
            "post-selection dedup left an unrepresented or unbound handle",
        )
        parsed = ParsedTypedItems(
            tuple(accepted),
            contribution.parsed.rejected_items,
            identity_sha256(
                {
                    "accepted_item_receipt_sha256s": [
                        item.receipt_sha256 for item in accepted
                    ],
                    "format": f"{FORMAT}-post-selection-dedup-parse-v1",
                    "original_parse_receipt_sha256": (
                        contribution.parsed.parse_receipt_sha256
                    ),
                    "post_selection_duplicate_receipt_sha256s": [
                        row["duplicate_item_receipt_sha256"]
                        for row in exclusions
                        if row["duplicate_mechanism_id"] == contribution.mechanism_id
                    ],
                }
            ),
        )
        rebuilt_by_index[contribution_index] = TypedEvidenceContribution(
            contribution.mechanism_id,
            retained_bindings,
            parsed,
            contribution.sealed_artifact_sha256,
            contribution.frontier_mode,
            contribution.truncated,
        )
    _require(
        set(rebuilt_by_index) == set(range(len(contributions))),
        "post-selection dedup contribution order changed",
    )
    return (
        tuple(rebuilt_by_index[index] for index in range(len(contributions))),
        tuple(exclusions),
    )


def _mechanism_strength(mechanism_id: str) -> int:
    value = mechanism_id.casefold()
    if "active_reconstruction" in value:
        return 60
    if "full_store" in value or "slot_closure" in value:
        return 50
    if "tail" in value:
        return 40
    if "source" in value or "guided" in value or "direct" in value:
        return 30
    if "map" in value:
        return 20
    return 10


def _active_lane_contract(
    contributions: tuple[TypedEvidenceContribution, ...],
) -> tuple[tuple[LaneBudget, ...], dict[str, str]]:
    active_lanes = tuple(
        lane_id
        for lane_id in LANE_CONTENT_TOKEN_CAPS
        if any(
            LANE_BY_MECHANISM.get(row.mechanism_id) == lane_id
            for row in contributions
        )
    )
    mapping = {
        row.mechanism_id: LANE_BY_MECHANISM[row.mechanism_id]
        for row in contributions
    }
    _require(
        len(mapping) == len(contributions)
        and set(mapping.values()) == set(active_lanes),
        "typed lane mechanism declaration changed",
    )
    zero_call_budget = CallBudget(8_000, OUTPUT_TOKEN_RESERVE, 0)
    return (
        tuple(
            LaneBudget(
                lane_id,
                LANE_CONTENT_TOKEN_CAPS[lane_id],
                zero_call_budget,
            )
            for lane_id in active_lanes
        ),
        mapping,
    )


def _allocate_non_borrowable_lanes(
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    operator_spec: Any,
    local_selection_priority_by_handle: Mapping[str, Sequence[int]] | None = None,
) -> tuple[TypedLaneAllocation, dict[str, Any]]:
    """Apply the declared per-method content caps before generic union."""

    lane_budgets, mapping = _active_lane_contract(contributions)
    allocation = allocate_typed_contribution_lanes(
        contributions,
        lane_budgets=lane_budgets,
        lane_by_mechanism=mapping,
        operator_spec=operator_spec,
        local_selection_priority_by_handle=local_selection_priority_by_handle,
    )
    by_mechanism = {
        row.mechanism_id: row for row in allocation.contributions
    }

    def usable_count(contribution: TypedEvidenceContribution) -> int:
        return sum(
            bool(
                item.included
                and not item.content_conflict
                and item.status is not EvidenceStatus.CANCELLED
                and (
                    item.status is not EvidenceStatus.PROPOSED
                    or operator_spec.include_proposed
                )
            )
            for item in contribution.parsed.accepted_items
        )

    # A declared lane may leave capacity unused, but it cannot silently erase
    # a nonempty specialist.  Oversize protected minima stop composition.
    _require(
        all(
            usable_count(original) == 0
            or usable_count(by_mechanism[original.mechanism_id]) >= 1
            for original in contributions
        ),
        "non-borrowable lane cap starved a nonempty typed mechanism",
    )
    projection = allocation.projection()
    projection["allocation_receipt_sha256"] = allocation.receipt_sha256
    projection["declared_lane_content_token_caps"] = dict(
        LANE_CONTENT_TOKEN_CAPS
    )
    projection["inactive_declared_lanes"] = [
        row
        for row in LANE_CONTENT_TOKEN_CAPS
        if row not in {budget.lane_id for budget in lane_budgets}
    ]
    projection["receipt_sha256"] = identity_sha256(
        {key: value for key, value in projection.items() if key != "receipt_sha256"}
    )
    return allocation, projection


def _fill_shared_lane_surplus(
    original_contributions: tuple[TypedEvidenceContribution, ...],
    minimum_allocation: TypedLaneAllocation,
    *,
    operator_spec: Any,
    local_selection_priority_by_handle: Mapping[str, Sequence[int]] | None = None,
) -> tuple[tuple[TypedEvidenceContribution, ...], dict[str, Any]]:
    """Keep every lane minimum, then spend only aggregate unused lane tokens."""

    lane_budgets, mapping = _active_lane_contract(original_contributions)
    return fill_typed_lane_surplus(
        original_contributions,
        minimum_allocation,
        lane_budgets=lane_budgets,
        lane_by_mechanism=mapping,
        operator_spec=operator_spec,
        local_selection_priority_by_handle=local_selection_priority_by_handle,
    )


def _fair_merge_contributions(
    operator_spec: Any,
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    local_selection_priority_by_handle: Mapping[str, Sequence[int]] | None = None,
    protected_item_receipt_sha256s: Sequence[str] = (),
    minimum_allocation_receipt_sha256: str | None = None,
    surplus_fill_audit: Mapping[str, Any] | None = None,
) -> tuple[TypedEvidencePacket, dict[str, Any]]:
    """Build a bounded union without first-in mechanism starvation.

    Exact items selected by the non-borrowable first phase are admitted first.
    Remaining usable items compete globally by the same strength dimensions
    used by final prompt fitting.  The temporary packet uses a one-token
    reserve only to satisfy the typed packet invariant; the exact full-chat
    768-token envelope is enforced later by ``fit_typed_final_prompt``.
    """

    _require(bool(contributions), "fair typed merge requires contributions")
    _require(
        len({row.mechanism_id for row in contributions}) == len(contributions),
        "fair typed merge mechanisms repeat",
    )
    binding_by_handle = {
        binding.handle_id: binding
        for contribution in contributions
        for binding in contribution.bindings
    }
    _require(
        len(binding_by_handle)
        == sum(len(row.bindings) for row in contributions),
        "fair typed merge handles collide",
    )
    local_priorities = {
        handle: tuple(values)
        for handle, values in (
            local_selection_priority_by_handle or {}
        ).items()
    }
    _require(
        set(local_priorities) <= set(binding_by_handle)
        and all(
            len(priority) == LOCAL_RETENTION_PRIORITY_WIDTH
            and all(type(value) is int for value in priority)
            for priority in local_priorities.values()
        ),
        "fair typed merge local priorities changed coverage or width",
    )
    zero_local_priority = (0,) * LOCAL_RETENTION_PRIORITY_WIDTH
    owner_by_receipt: dict[str, str] = {}
    ordered_items: list[Any] = []
    ordinal_by_receipt: dict[str, int] = {}
    for contribution in contributions:
        local_handles = {row.handle_id for row in contribution.bindings}
        for item in contribution.parsed.accepted_items:
            _require(
                set(item.handle_ids) <= local_handles
                and item.receipt_sha256 not in owner_by_receipt,
                "fair typed merge item escaped or collided",
            )
            owner_by_receipt[item.receipt_sha256] = contribution.mechanism_id
            ordinal_by_receipt[item.receipt_sha256] = len(ordered_items)
            ordered_items.append(item)
    item_by_receipt = {
        item.receipt_sha256: item for item in ordered_items
    }

    def usable(item: Any) -> bool:
        return bool(
            item.included
            and not item.content_conflict
            and item.status is not EvidenceStatus.CANCELLED
            and (
                item.status is not EvidenceStatus.PROPOSED
                or operator_spec.include_proposed
            )
        )

    def strength(
        item: Any,
    ) -> tuple[int, int, tuple[int, ...], int, int, int, str]:
        local_priority = max(
            (
                local_priorities.get(handle, zero_local_priority)
                for handle in item.handle_ids
            ),
            default=zero_local_priority,
        )
        return (
            int(item.included),
            int(item.content_coherence is not ContentCoherence.CONFLICT),
            local_priority,
            len(item.supported_slot_ids),
            _mechanism_strength(owner_by_receipt[item.receipt_sha256]),
            -ordinal_by_receipt[item.receipt_sha256],
            item.receipt_sha256,
        )

    declared_protected_receipts = tuple(protected_item_receipt_sha256s)
    _require(
        len(set(declared_protected_receipts))
        == len(declared_protected_receipts)
        and all(type(value) is str for value in declared_protected_receipts),
        "fair typed merge protected receipts must be ordered unique text",
    )
    for value in declared_protected_receipts:
        require_sha256(value, "fair typed merge protected item receipt")

    protected: list[Any]
    protected_binding_receipts: tuple[str, ...]
    surplus_receipt: str | None = None
    surplus_partitions: dict[str, list[str]] = {
        "added_binding_receipt_sha256s": [],
        "added_item_receipt_sha256s": [],
        "budget_omitted_item_receipt_sha256s": [],
        "ineligible_item_receipt_sha256s": [],
    }
    if declared_protected_receipts:
        _require(
            minimum_allocation_receipt_sha256 is not None
            and surplus_fill_audit is not None,
            "exact fair minima require their lane and surplus receipts",
        )
        minimum_receipt = require_sha256(
            minimum_allocation_receipt_sha256,
            "fair minimum lane allocation receipt",
        )
        _require(
            type(surplus_fill_audit) is dict,
            "fair surplus audit must be an exact dict",
        )
        sealed_surplus = dict(surplus_fill_audit)
        surplus_receipt = require_sha256(
            sealed_surplus.pop("receipt_sha256", None),
            "fair surplus fill receipt",
        )
        _require(
            identity_sha256(sealed_surplus) == surplus_receipt
            and sealed_surplus.get("minimum_allocation_receipt_sha256")
            == minimum_receipt
            and sealed_surplus.get("minimum_item_receipt_sha256s")
            == list(declared_protected_receipts)
            and sealed_surplus.get("contribution_receipt_sha256s")
            == [row.receipt_sha256 for row in contributions],
            "fair merge inputs do not match the sealed surplus fill",
        )
        for key in surplus_partitions:
            values = sealed_surplus.get(key)
            _require(
                type(values) is list
                and len(values) == len(set(values))
                and all(type(value) is str for value in values),
                f"fair surplus {key} partition changed",
            )
            for value in values:
                require_sha256(value, f"fair surplus {key}")
            surplus_partitions[key] = list(values)
        _require(
            set(declared_protected_receipts) <= set(item_by_receipt)
            and all(
                usable(item_by_receipt[receipt])
                for receipt in declared_protected_receipts
            ),
            "fair protected lane minimum is missing or unusable",
        )
        protected = [
            item_by_receipt[receipt] for receipt in declared_protected_receipts
        ]
        protected_owner_ids = {
            owner_by_receipt[item.receipt_sha256] for item in protected
        }
        _require(
            all(
                not any(usable(item) for item in row.parsed.accepted_items)
                or row.mechanism_id in protected_owner_ids
                for row in contributions
            ),
            "fair protected lane minima lost a nonempty mechanism",
        )
        protected_handles = {
            handle for item in protected for handle in item.handle_ids
        }
        computed_protected_binding_receipts = tuple(
            binding.receipt_sha256
            for contribution in contributions
            for binding in contribution.bindings
            if binding.handle_id in protected_handles
        )
        declared_protected_binding_receipts = sealed_surplus.get(
            "minimum_binding_receipt_sha256s"
        )
        _require(
            type(declared_protected_binding_receipts) is list
            and len(declared_protected_binding_receipts)
            == len(set(declared_protected_binding_receipts))
            and set(declared_protected_binding_receipts)
            == set(computed_protected_binding_receipts),
            "fair protected binding partition changed",
        )
        protected_binding_receipts = tuple(
            declared_protected_binding_receipts
        )
    else:
        _require(
            minimum_allocation_receipt_sha256 is None
            and surplus_fill_audit is None,
            "fair lane receipts were supplied without protected minima",
        )
        protected = []
        for contribution in contributions:
            candidates = [
                item
                for item in contribution.parsed.accepted_items
                if usable(item)
            ]
            if candidates:
                protected.append(max(candidates, key=strength))
        protected_binding_receipts = tuple(
            binding.receipt_sha256
            for contribution in contributions
            for binding in contribution.bindings
            if binding.handle_id
            in {handle for item in protected for handle in item.handle_ids}
        )
        minimum_receipt = None

    selected: list[Any] = list(dict.fromkeys(protected))
    selected_receipts = {item.receipt_sha256 for item in selected}
    remaining = sorted(
        (
            item
            for item in ordered_items
            if usable(item) and item.receipt_sha256 not in selected_receipts
        ),
        key=strength,
        reverse=True,
    )
    rejected = tuple(
        item
        for contribution in contributions
        for item in contribution.parsed.rejected_items
    )
    modes = {row.frontier_mode for row in contributions}
    mode = (
        FrontierMode.OPEN
        if FrontierMode.OPEN in modes
        else FrontierMode.BOUNDED
        if FrontierMode.BOUNDED in modes
        else FrontierMode.EXHAUSTIVE
    )
    artifacts = tuple(
        dict.fromkeys(row.sealed_artifact_sha256 for row in contributions)
    )

    def build(items: Sequence[Any]) -> TypedEvidencePacket:
        handles = {handle for item in items for handle in item.handle_ids}
        bindings = tuple(
            binding
            for contribution in contributions
            for binding in contribution.bindings
            if binding.handle_id in handles
        )
        parsed = ParsedTypedItems(
            tuple(items),
            rejected,
            identity_sha256(
                {
                    "accepted_item_receipt_sha256s": [
                        item.receipt_sha256 for item in items
                    ],
                    "contribution_receipt_sha256s": [
                        row.receipt_sha256 for row in contributions
                    ],
                    "format": f"{FORMAT}-fair-premerge-parse-v1",
                    "rejected_item_receipt_sha256s": [
                        row.rejection_sha256 for row in rejected
                    ],
                }
            ),
        )
        return build_typed_evidence_packet(
            operator_spec,
            bindings,
            parsed,
            sealed_input_artifact_sha256s=artifacts,
            frontier_mode=mode,
            conflict_policy=ConflictPolicy.QUARANTINE,
            output_token_reserve=1,
            truncated=(
                any(row.truncated for row in contributions)
                or len(items) < len(ordered_items)
            ),
            provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
        )

    packet = build(selected)
    _require(
        {row.receipt_sha256 for row in packet.items}
        == {row.receipt_sha256 for row in selected},
        "per-mechanism protected minima exceed the typed packet envelope",
    )
    for item in remaining:
        trial = tuple((*selected, item))
        candidate = build(trial)
        if {row.receipt_sha256 for row in candidate.items} == {
            row.receipt_sha256 for row in trial
        }:
            selected.append(item)
            packet = candidate

    final_receipts = {row.receipt_sha256 for row in packet.items}
    rows = []
    for contribution in contributions:
        candidates = tuple(contribution.parsed.accepted_items)
        usable_candidates = tuple(item for item in candidates if usable(item))
        admitted = tuple(
            item for item in candidates if item.receipt_sha256 in final_receipts
        )
        protected_items = tuple(
            item
            for item in protected
            if owner_by_receipt[item.receipt_sha256]
            == contribution.mechanism_id
        )
        rows.append(
            {
                "accepted_candidate_count": len(candidates),
                "admitted_item_receipt_sha256s": [
                    item.receipt_sha256 for item in admitted
                ],
                "dropped_item_receipt_sha256s": [
                    item.receipt_sha256
                    for item in candidates
                    if item.receipt_sha256 not in final_receipts
                ],
                "mechanism_id": contribution.mechanism_id,
                "parser_rejected_count": len(
                    contribution.parsed.rejected_items
                ),
                "protected_minimum_item_receipt_sha256s": [
                    item.receipt_sha256 for item in protected_items
                ],
                "usable_candidate_count": len(usable_candidates),
            }
        )
    audit = {
        "format": f"{FORMAT}-fair-premerge-audit-v3",
        "gold_loaded": False,
        "input_contribution_receipt_sha256s": [
            row.receipt_sha256 for row in contributions
        ],
        "local_selection_priority_receipt_sha256": identity_sha256(
            {
                "fixed_width": LOCAL_RETENTION_PRIORITY_WIDTH,
                "format": f"{FORMAT}-fair-premerge-local-priority-v1",
                "rows": [
                    {"handle_id": handle, "priority": list(priority)}
                    for handle, priority in sorted(local_priorities.items())
                ],
            }
        ),
        "mechanisms": rows,
        "minimum_allocation_receipt_sha256": minimum_receipt,
        "packet_receipt_sha256": packet.receipt_sha256,
        "policy": (
            "all_exact_non_borrowable_lane_minima_then_"
            "local_retrieval_priority_then_global_strength_fill_against_"
            "compact_final_provider_projection"
        ),
        "protected_minimum_binding_receipt_sha256s": list(
            protected_binding_receipts
        ),
        "protected_minimum_item_receipt_sha256s": [
            item.receipt_sha256 for item in protected
        ],
        "provider_prompt_count": 0,
        "retained_transformer_token_state_bytes": 0,
        "shared_lane_surplus_fill_receipt_sha256": surplus_receipt,
        "surplus_partitions": surplus_partitions,
    }
    audit["receipt_sha256"] = identity_sha256(audit)
    assert_gold_blind(audit, path="typed_final_fair_premerge")
    return packet, audit


def _retained_mechanism_bindings(
    contributions: tuple[TypedEvidenceContribution, ...],
    packet: TypedEvidencePacket,
) -> tuple[dict[str, str], tuple[dict[str, Any], ...]]:
    """Project post-merge mechanism ownership to only retained packet H IDs."""

    allocated: dict[str, tuple[str, Any]] = {}
    for contribution in contributions:
        for binding in contribution.bindings:
            _require(
                binding.handle_id not in allocated,
                "allocated mechanism bindings collide",
            )
            allocated[binding.handle_id] = (contribution.mechanism_id, binding)
    packet_handle_ids = {row.handle_id for row in packet.handles}
    _require(
        packet_handle_ids <= set(allocated),
        "fair-merge packet handle escaped allocated contributions",
    )
    retained = {
        handle: allocated[handle][0]
        for handle in allocated
        if handle in packet_handle_ids
    }
    dropped = tuple(
        binding.projection()
        for handle, (_mechanism, binding) in allocated.items()
        if handle not in packet_handle_ids
    )
    _require(
        set(retained) == packet_handle_ids,
        "typed contribution mechanism bindings changed",
    )
    return retained, dropped


def _adaptive_parent_prompt_proxy(planned: Any) -> int:
    if planned.fact_envelope is not None:
        value = planned.fact_envelope.parent_prompt_token_proxy
        _require(
            type(value) is int and value > 0,
            "adaptive parent fact envelope lost its prompt-token binding",
        )
        return value
    messages = tuple(
        {"role": row.role, "content": row.content}
        for row in adaptive_live._render_messages(  # noqa: SLF001
            planned.map_plan_row,
            planned.map_row,
            None,
        )
    )
    return count_chat_prompt_token_proxy(messages)


def _map_local_audit(
    contribution: TypedEvidenceContribution,
    planned: Any,
) -> dict[str, Any]:
    alias_by_name = {row.alias: row for row in planned.map_plan_row.aliases}
    item_by_receipt = {
        row.item_sha256: row for row in planned.map_row.accepted_items
    }
    rows: list[dict[str, Any]] = []
    for binding in contribution.bindings:
        item = item_by_receipt.get(binding.evidence_receipt_sha256)
        _require(item is not None, "map local binding lost its accepted item")
        alias = alias_by_name.get(item.alias)
        _require(alias is not None, "map local binding lost its exact alias")
        rows.append(
            {
                "accepted_map_item": item.projection(),
                "binding": binding.projection(),
                "payload_alias": alias.projection(),
            }
        )
    return {
        "contribution": contribution.projection(),
        "exact_item_bindings": rows,
        "map_plan_row_receipt_sha256": planned.map_plan_row.receipt_sha256,
        "map_row_source_sha256": planned.map_row.source_row_sha256,
    }


def _union_forbidden_literals(row: TailFactUnionRow) -> tuple[str, ...]:
    values: list[str] = [row.fact_union.parent.namespace_id]
    for fact in row.fact_union.union_facts_before_direct_exclusion:
        for origin in fact.origins:
            values.extend((origin.namespace_id, origin.source_id))
    for ref in row.question_plan.direct_evidence:
        values.extend((ref.namespace_id, ref.source_id))
    return tuple(dict.fromkeys(values))


def _map_forbidden_literals(planned: Any) -> tuple[str, ...]:
    aliases = {row.alias: row for row in planned.map_plan_row.aliases}
    values: list[str] = []
    packet = planned.map_plan_row.direct_plan_row.adapter.source.packet
    for item in planned.map_row.accepted_items:
        alias = aliases.get(item.alias)
        _require(alias is not None, "map forbidden-literal alias changed")
        values.append(alias.source_id)
    values.extend(
        evidence.source_id
        for evidence in (
            tuple(packet.protected_evidence) + tuple(packet.admitted_evidence)
        )
    )
    return tuple(dict.fromkeys(values))


def _full_store_forbidden_literals(result: FullStoreSlotClosureResult) -> tuple[str, ...]:
    values: list[str] = []
    for binding in result.local_bindings:
        values.extend(
            (binding.namespace_id, binding.source_id, binding.partition_id)
        )
    return tuple(dict.fromkeys(values))


def _candidate_intrinsic_selection_priority(
    dated_question: str,
    candidate: Any,
) -> tuple[int, ...]:
    """Return the common fixed-width question/candidate priority suffix."""

    axes = tuple(candidate.selection_axes)
    candidate_terms = set(normalized_terms(candidate.quote))
    matched_actions = matched_action_concepts(
        dated_question,
        candidate.quote,
    )
    completed_actions = completed_action_concepts(candidate.quote)
    action_obligation_term_count = len(
        candidate_terms
        & {
            "acquire",
            "assist",
            "buy",
            "clean",
            "collect",
            "donate",
            "need",
            "pick",
            "pickup",
            "purchase",
            "return",
            "service",
            "visit",
        }
    )
    required_axis_count = sum(
        value.startswith("required_slot:") for value in axes
    )
    return (
        int(bool(candidate.supported_slot_ids)),
        len(candidate.supported_slot_ids),
        int(bool(matched_actions)),
        len(matched_actions),
        int(bool(set(matched_actions) & set(completed_actions))),
        int(candidate.role.casefold() == "user"),
        action_obligation_term_count,
        len(candidate.matched_query_terms),
        int(candidate.contains_numeric_value),
        int(required_axis_count > 0),
        int("question_derived_temporal_target" in axes),
        int("content_source_coherence" in axes),
        required_axis_count,
        int(candidate.event_date is not None),
        -(
            candidate.temporal_distance_days
            if candidate.temporal_distance_days is not None
            else 1_000_000_000
        ),
    )


def _full_store_selection_priorities(
    contribution: TypedEvidenceContribution,
    result: FullStoreSlotClosureResult,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    """Keep scanner selection strength local while ranking opaque handles."""

    _require(
        len(contribution.bindings) == len(result.candidates),
        "full-store selection-priority candidate binding changed",
    )
    priority_by_handle: dict[str, tuple[int, ...]] = {}
    rows: list[dict[str, Any]] = []
    for binding, candidate in zip(
        contribution.bindings,
        result.candidates,
        strict=True,
    ):
        axes = tuple(candidate.selection_axes)
        priority = _candidate_intrinsic_selection_priority(
            result.dated_question,
            candidate,
        )
        priority_by_handle[binding.handle_id] = priority
        body = {
            "candidate_id": candidate.candidate_id,
            "candidate_receipt_sha256": identity_sha256(candidate.projection()),
            "handle_id": binding.handle_id,
            "local_selection_priority": list(priority),
            "selection_axes": list(axes),
            "source_group_handle": binding.source_group_handle,
        }
        rows.append({**body, "receipt_sha256": identity_sha256(body)})
    audit: dict[str, Any] = {
        "format": f"{FORMAT}-full-store-local-selection-priority-v1",
        "policy": (
            "source_group_round_robin_then_intrinsic_slot_action_query_"
            "support_and_scanner_temporal_coherence_axes"
        ),
        "rows": rows,
    }
    audit["receipt_sha256"] = identity_sha256(audit)
    return priority_by_handle, audit


def _build_active_reconstruction(
    index: FullStoreWindowIndex,
    parent_result: FullStoreSlotClosureResult,
    full_contribution: TypedEvidenceContribution,
) -> tuple[
    TypedActiveReconstructionResult,
    TypedEvidenceContribution,
    dict[str, Any],
]:
    """Run the provider-free active layer against the already built index."""

    expected_full, _expected_full_audit = adapt_full_store_slot_closure(
        parent_result.operator_spec,
        parent_result,
        closure_artifact_sha256=full_contribution.sealed_artifact_sha256,
        handle_start=FULL_STORE_RANGE,
        group_start=FULL_STORE_RANGE,
        mechanism_id=FULL_STORE_MECHANISM,
    )
    _require(
        expected_full.projection() == full_contribution.projection(),
        "active seed received a noncanonical audited full-store contribution",
    )
    canonical_seed = adapt_full_store_slot_closure_to_typed_contribution(
        parent_result,
        handle_start=FULL_STORE_RANGE,
        group_start=FULL_STORE_RANGE,
    )
    _require(
        tuple(row.handle_id for row in canonical_seed.bindings)
        == tuple(row.handle_id for row in full_contribution.bindings)
        and tuple(row.source_group_handle for row in canonical_seed.bindings)
        == tuple(row.source_group_handle for row in full_contribution.bindings),
        "active seed/full-store opaque alignment changed",
    )
    # The active core requires its canonical DIRECT_POINTER bindings, while the
    # final lane's audited adapter owns the richer typed interpretation.  Reuse
    # those exact parsed items so reconstruction cannot silently reason from a
    # different relation, date authority, qualifier, or slot projection than
    # the chunk that is ultimately packed for the answer LLM.
    seed = TypedEvidenceContribution(
        canonical_seed.mechanism_id,
        canonical_seed.bindings,
        full_contribution.parsed,
        canonical_seed.sealed_artifact_sha256,
        canonical_seed.frontier_mode,
        canonical_seed.truncated,
    )
    _require(
        tuple(
            item.projection()
            for item in seed.parsed.accepted_items
        )
        == tuple(
            item.projection()
            for item in full_contribution.parsed.accepted_items
        ),
        "active seed semantic items differ from the audited full-store lane",
    )
    expected_quote_by_handle = {
        f"H{FULL_STORE_RANGE + offset:03d}": candidate.quote
        for offset, candidate in enumerate(parent_result.candidates)
    }
    for contribution, label in (
        (seed, "active seed"),
        (full_contribution, "audited full-store"),
    ):
        _require(
            {
                item.handle_ids[0]: item.summary
                for item in contribution.parsed.accepted_items
                if len(item.handle_ids) == 1
            }
            == expected_quote_by_handle,
            f"{label} did not preserve every exact parent chunk",
        )

    result = run_typed_active_reconstruction(
        index,
        parent_result,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=seed,
        budget=ActiveReconstructionBudget(
            use_selected_provenance_affinity=True,
        ),
    )
    adapted = adapt_typed_active_reconstruction_to_contribution(
        result,
        handle_start=ACTIVE_RECONSTRUCTION_RANGE,
        group_start=ACTIVE_RECONSTRUCTION_RANGE,
    )
    _require(
        adapted.mechanism_id == CORE_ACTIVE_RECONSTRUCTION_MECHANISM,
        "active adapter mechanism changed",
    )
    contribution = TypedEvidenceContribution(
        ACTIVE_RECONSTRUCTION_MECHANISM,
        adapted.bindings,
        adapted.parsed,
        adapted.sealed_artifact_sha256,
        adapted.frontier_mode,
        adapted.truncated,
    )
    exact_quote_by_handle = {
        binding.handle_id: candidate.quote
        for binding, candidate in zip(
            contribution.bindings, result.candidates, strict=True
        )
    }
    represented = {
        item.handle_ids[0]: item.summary
        for item in contribution.parsed.accepted_items
        if len(item.handle_ids) == 1
    }
    _require(
        represented == exact_quote_by_handle
        and all(
            binding.citation_sha256 == candidate.quote_sha256
            and binding.citation_char_count == len(candidate.quote)
            for binding, candidate in zip(
                contribution.bindings, result.candidates, strict=True
            )
        ),
        "active adapter compressed or rewrote an exact hydrated chunk",
    )
    alignment_body: dict[str, Any] = {
        "active_contribution_receipt_sha256": contribution.receipt_sha256,
        "active_result_receipt_sha256": result.receipt_sha256,
        "active_seed_contribution_receipt_sha256": seed.receipt_sha256,
        "canonical_seed_binding_receipt_sha256s": [
            row.receipt_sha256 for row in canonical_seed.bindings
        ],
        "seed_item_semantics_source": "audited_full_store_contribution",
        "audited_full_store_contribution_receipt_sha256": (
            full_contribution.receipt_sha256
        ),
        "exact_chunk_count": len(exact_quote_by_handle),
        "exact_chunk_payload_policy": "byte_for_byte_or_whole_item_drop",
        "format": f"{FORMAT}-active-parent-alignment-v1",
        "gold_loaded": False,
        "new_provider_calls": 0,
        "opaque_handle_ids": list(expected_quote_by_handle),
        "retained_transformer_token_state_bytes": 0,
    }
    alignment = {
        **alignment_body,
        "receipt_sha256": identity_sha256(alignment_body),
    }
    assert_gold_blind(alignment, path="typed_final_active_alignment")
    return result, contribution, alignment


def _active_support_prefix(priority: Any | None) -> tuple[int, ...]:
    """Project only scanner-proven cue support into a fixed-width prefix."""

    if priority is None:
        return (0, 0, 0, 0, 0, 0, 0, 0, -3)
    return (
        int(priority.source_affinity),
        int(priority.component_affinity),
        priority.slot_support_count,
        int(priority.action_support),
        int(priority.temporal_support),
        priority.cue_support_count,
        int(priority.recommended_parent_promotion),
        int(priority.newly_admitted),
        -priority.first_hop,
    )


def _active_selection_priorities(
    full_contribution: TypedEvidenceContribution,
    full_intrinsic_priority_by_handle: Mapping[str, Sequence[int]],
    active_contribution: TypedEvidenceContribution,
    result: TypedActiveReconstructionResult,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    """Merge exact callback support with a common intrinsic priority suffix."""

    parent_handles = tuple(row.handle_id for row in full_contribution.bindings)
    _require(
        set(full_intrinsic_priority_by_handle) == set(parent_handles),
        "active/full intrinsic priority alignment changed",
    )
    cue_rows = derive_candidate_cue_support_priorities(
        result,
        parent_handle_ids=parent_handles,
    )
    cue_by_span = {row.span_receipt_sha256: row for row in cue_rows}
    _require(
        len(cue_by_span) == len(cue_rows),
        "active cue-priority spans repeated",
    )
    combined: dict[str, tuple[int, ...]] = {}
    rows: list[dict[str, Any]] = []

    def add(
        handle_id: str,
        span_receipt: str,
        intrinsic: Sequence[int],
        *,
        owner: str,
    ) -> None:
        cue = cue_by_span.get(span_receipt)
        priority = (*_active_support_prefix(cue), *tuple(intrinsic))
        _require(
            len(priority) == 24 and all(type(value) is int for value in priority),
            "active/full local selection priority changed width",
        )
        combined[handle_id] = priority
        body = {
            "candidate_cue_support": (
                None if cue is None else cue.projection()
            ),
            "handle_id": handle_id,
            "local_selection_priority": list(priority),
            "mechanism_id": owner,
            "span_receipt_sha256": span_receipt,
        }
        rows.append({**body, "receipt_sha256": identity_sha256(body)})

    for binding, local in zip(
        full_contribution.bindings,
        result.parent_result.local_bindings,
        strict=True,
    ):
        add(
            binding.handle_id,
            citation_span_receipt_sha256(local),
            full_intrinsic_priority_by_handle[binding.handle_id],
            owner=FULL_STORE_MECHANISM,
        )
    for binding, candidate, local in zip(
        active_contribution.bindings,
        result.candidates,
        result.local_bindings,
        strict=True,
    ):
        add(
            binding.handle_id,
            citation_span_receipt_sha256(local),
            _candidate_intrinsic_selection_priority(
                result.parent_result.dated_question,
                candidate,
            ),
            owner=ACTIVE_RECONSTRUCTION_MECHANISM,
        )
    _require(
        len(combined)
        == len(full_contribution.bindings) + len(active_contribution.bindings),
        "active/full priority handles collided",
    )
    audit: dict[str, Any] = {
        "candidate_cue_support_rows": [row.projection() for row in cue_rows],
        "fixed_priority_width": 24,
        "format": f"{FORMAT}-active-full-local-selection-priority-v1",
        "policy": (
            "explicit_callback_source_component_slot_action_temporal_cue_"
            "support_then_common_intrinsic_candidate_strength"
        ),
        "rows": rows,
    }
    audit["receipt_sha256"] = identity_sha256(audit)
    assert_gold_blind(audit, path="typed_final_active_selection_priority")
    return combined, audit


def _active_forbidden_literals(
    result: TypedActiveReconstructionResult,
) -> tuple[str, ...]:
    values: list[str] = []
    for binding in result.local_bindings:
        values.extend(
            (binding.namespace_id, binding.source_id, binding.partition_id)
        )
    return tuple(dict.fromkeys(values))


def _local_story_keys(
    *,
    parent_map: TypedEvidenceContribution,
    planned: Any,
    namespace_id: str,
    base_row: TailFactUnionRow | None,
    base_contributions: tuple[TypedEvidenceContribution, ...],
    base_parent_prompt_token_proxy: int,
    tail_row: TailFactUnionRow | None,
    tail_contributions: tuple[TypedEvidenceContribution, ...],
    tail_parent_prompt_token_proxy: int,
    full_audit: Any,
    active_contribution: TypedEvidenceContribution,
    active_result: TypedActiveReconstructionResult,
    retained_handle_ids: frozenset[str],
    retained_group_handles: frozenset[str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    _require(
        bool(retained_handle_ids) == bool(retained_group_handles),
        "story projection retained handle/group emptiness differs",
    )
    merged: dict[str, list[str]] = {}
    history_keys: set[str] = set()

    def extend(
        plane: tuple[Mapping[str, Sequence[str]], frozenset[str]],
    ) -> None:
        values, histories = plane
        for group, receipts in values.items():
            target = merged.setdefault(group, [])
            target.extend(receipts)
        history_keys.update(histories)

    extend(
        _map_story_keys(
            parent_map,
            planned,
            namespace_id,
            retained_handle_ids=retained_handle_ids,
        )
    )
    if base_row is not None:
        for contribution in base_contributions:
            extend(
                _union_story_keys(
                    contribution,
                    base_row,
                    parent_prompt_token_proxy=base_parent_prompt_token_proxy,
                    retained_handle_ids=retained_handle_ids,
                )
            )
    if tail_row is not None:
        for contribution in tail_contributions:
            extend(
                _union_story_keys(
                    contribution,
                    tail_row,
                    parent_prompt_token_proxy=tail_parent_prompt_token_proxy,
                    retained_handle_ids=retained_handle_ids,
                )
            )
    extend(
        _full_store_story_keys(
            full_audit,
            retained_handle_ids=retained_handle_ids,
        )
    )
    extend(
        _active_story_keys(
            active_contribution,
            active_result,
            retained_handle_ids=retained_handle_ids,
        )
    )
    keys = {
        group: tuple(dict.fromkeys(receipts))
        for group, receipts in merged.items()
    }
    _require(
        set(keys) == set(retained_group_handles),
        "story groups differ from the retained packet groups",
    )
    shared = {
        key: sorted(
            group for group, group_keys in keys.items() if key in group_keys
        )
        for key in {key for group_keys in keys.values() for key in group_keys}
    }
    shared = {
        key: groups for key, groups in shared.items() if len(groups) >= 2
    }
    audit_body: dict[str, Any] = {
        "format": f"{FORMAT}-common-source-history-story-keys-v1",
        "group_count": len(keys),
        "retained_packet_group_count": len(retained_group_handles),
        "retained_packet_handle_count": len(retained_handle_ids),
        "history_shared_key_count": sum(
            key in history_keys for key in shared
        ),
        "policy": (
            "post_selection_exact_provenance_common_source_and_history_keys"
        ),
        "shared_keys": [
            {
                "group_handles": groups,
                "history_key": key in history_keys,
                "member_group_count": len(groups),
                "story_key_sha256": key,
            }
            for key, groups in sorted(
                shared.items(),
                key=lambda row: (
                    -len(row[1]),
                    -int(row[0] in history_keys),
                    row[0],
                ),
            )
        ],
    }
    audit = {**audit_body, "receipt_sha256": identity_sha256(audit_body)}
    assert_gold_blind(audit, path="typed_final_common_story_keys")
    return keys, audit


def _composition_projection(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], SealedArtifact]:
    """Rebuild the complete provider-free evidence and receipt plane."""

    parent = adaptive_cli.load_verified_adaptive_solver_run(_parent_args(args))
    _require(
        parent.terminal.sha256
        == require_sha256(args.expected_parent_run_sha256, "expected parent run"),
        "adaptive parent terminal changed",
    )
    base_terminal, base_rows = _load_parent_source_rows(args)
    tail_terminal, tail_questions, tail_results = _load_verified_tail(args)
    tail_rows = build_tail_post_map_fact_unions(tail_questions, tail_results)

    planned_rows = parent.loaded.plan.rows
    parent_rows = parent.run.rows
    _require(
        len(planned_rows) == len(parent_rows) == EXPECTED_QUESTION_COUNT,
        "typed final arm requires the full locked 100-row parent",
    )
    _require(
        tuple(row.question_id for row in planned_rows)
        == tuple(row.question_id for row in parent_rows),
        "adaptive parent plan/run order changed",
    )
    dated_by_question = {
        row.question_id: (
            row.map_plan_row.direct_plan_row.adapter.source.packet.dated_question
        )
        for row in planned_rows
    }
    (
        context,
        closure_by_question,
        index_by_namespace,
        cache_receipts,
    ) = _build_full_store_results(args, dated_by_question)
    ordered_ids = tuple(row.question_id for row in planned_rows)
    closure_payload = _closure_input_projection(
        closure_by_question, cache_receipts, ordered_ids
    )
    closure_artifact, _closure_created = publish_sealed_json(
        Path(args.output_root) / CLOSURE_INPUT_NAME,
        closure_payload,
    )

    context_by_question = {
        row.source.packet.question_id: row for row in context.population.rows
    }
    base_by_question = {row.question_id: row for row in base_rows}
    tail_by_question = {row.question_id: row for row in tail_rows}
    _require(
        len(context_by_question) == EXPECTED_QUESTION_COUNT
        and len(base_by_question) == len(base_rows)
        and len(tail_by_question) == len(tail_rows)
        and set(base_by_question) <= set(ordered_ids)
        and set(tail_by_question) <= set(ordered_ids),
        "typed contribution question identities changed",
    )

    composition_rows: list[dict[str, Any]] = []
    for ordinal, (planned, parent_row) in enumerate(
        zip(planned_rows, parent_rows, strict=True)
    ):
        question_id = planned.question_id
        source_packet = (
            planned.map_plan_row.direct_plan_row.adapter.source.packet
        )
        context_row = context_by_question[question_id]
        _require(
            ordinal == planned.ordinal == parent_row.ordinal
            and parent_row.question_id == question_id
            and context_row.source.packet.question_id == question_id
            and context_row.source.packet.dated_question
            == source_packet.dated_question
            and context_row.source.packet.question_sha256
            == source_packet.question_sha256
            and _evidence_items_belong_to_namespace(
                (
                    tuple(source_packet.protected_evidence)
                    + tuple(source_packet.admitted_evidence)
                ),
                context_row.namespace,
            ),
            "typed final question/store/parent binding changed",
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
            map_artifact_sha256=parent.loaded.map_plane.run_sha256,
            fact_envelope=None,
            source_artifact_sha256=None,
            frontier_mode=FrontierMode.BOUNDED,
            handle_start=PARENT_MAP_RANGE,
            group_start=PARENT_MAP_RANGE,
        )
        parent_map = _packet_contribution(
            map_packet,
            mechanism_id=PARENT_MAP_MECHANISM,
            sealed_artifact_sha256=parent.loaded.map_plane.run_sha256,
        )
        parent_prompt_proxy = _adaptive_parent_prompt_proxy(planned)

        original: list[TypedEvidenceContribution] = [parent_map]
        base_original: tuple[TypedEvidenceContribution, ...] = ()
        base_row = base_by_question.get(question_id)
        if base_row is not None:
            base_original = adapt_tail_question_contributions(
                spec,
                base_row,
                materialization_artifact_sha256=base_terminal.sha256,
                parent_prompt_token_proxy=parent_prompt_proxy,
                source_handle_start=PARENT_SOURCE_RANGE,
                source_group_start=PARENT_SOURCE_RANGE,
                pointer_handle_start=PARENT_POINTER_RANGE,
                pointer_group_start=PARENT_POINTER_RANGE,
                source_mechanism_id=PARENT_SOURCE_MECHANISM,
                pointer_mechanism_id=PARENT_POINTER_MECHANISM,
            )
            original.extend(base_original)

        tail_original: tuple[TypedEvidenceContribution, ...] = ()
        tail_row = tail_by_question.get(question_id)
        if tail_row is not None:
            tail_original = adapt_tail_question_contributions(
                spec,
                tail_row,
                materialization_artifact_sha256=tail_terminal.sha256,
                parent_prompt_token_proxy=parent_prompt_proxy,
                source_handle_start=TAIL_SOURCE_RANGE,
                source_group_start=TAIL_SOURCE_RANGE,
                pointer_handle_start=TAIL_POINTER_RANGE,
                pointer_group_start=TAIL_POINTER_RANGE,
                source_mechanism_id=TAIL_SOURCE_MECHANISM,
                pointer_mechanism_id=TAIL_POINTER_MECHANISM,
            )
            original.extend(tail_original)

        full_contribution, full_audit = adapt_full_store_slot_closure(
            spec,
            closure_by_question[question_id],
            closure_artifact_sha256=closure_artifact.sha256,
            handle_start=FULL_STORE_RANGE,
            group_start=FULL_STORE_RANGE,
            mechanism_id=FULL_STORE_MECHANISM,
        )
        full_selection_priorities, full_selection_priority_audit = (
            _full_store_selection_priorities(
                full_contribution,
                closure_by_question[question_id],
            )
        )
        original.append(full_contribution)
        namespace_index = index_by_namespace[
            context_row.namespace.namespace_id
        ]
        _require(
            closure_by_question[question_id].receipt.window_index_receipt_sha256
            == namespace_index.receipt_sha256,
            "active reconstruction did not reuse the question's prebuilt index",
        )
        (
            active_result,
            active_contribution,
            active_parent_alignment_audit,
        ) = _build_active_reconstruction(
            namespace_index,
            closure_by_question[question_id],
            full_contribution,
        )
        (
            active_full_selection_priorities,
            active_selection_priority_audit,
        ) = _active_selection_priorities(
            full_contribution,
            full_selection_priorities,
            active_contribution,
            active_result,
        )
        # All mechanisms select independently.  Exact-span dedup happens only
        # after the active layer has hydrated and selected its own chunks.
        original.append(active_contribution)
        exact_span_keys_by_handle: dict[str, tuple[str, ...]] = {}
        exact_span_keys_by_handle.update(
            _map_exact_span_keys(
                parent_map,
                planned,
                context_row.namespace.namespace_id,
            )
        )
        if base_row is not None:
            for contribution in base_original:
                exact_span_keys_by_handle.update(
                    _union_exact_span_keys(
                        contribution,
                        base_row,
                        parent_prompt_token_proxy=parent_prompt_proxy,
                    )
                )
        if tail_row is not None:
            for contribution in tail_original:
                exact_span_keys_by_handle.update(
                    _union_exact_span_keys(
                        contribution,
                        tail_row,
                        parent_prompt_token_proxy=parent_prompt_proxy,
                    )
                )
        exact_span_keys_by_handle.update(
            _full_store_exact_span_keys(full_audit)
        )
        exact_span_keys_by_handle.update(
            _active_exact_span_keys(active_contribution, active_result)
        )
        _require(
            set(exact_span_keys_by_handle)
            == {
                binding.handle_id
                for contribution in original
                for binding in contribution.bindings
            },
            "post-selection exact span lineage changed",
        )
        deduped, postselection_exclusions = _dedup_selected_contributions(
            tuple(original),
            exact_span_keys_by_handle=exact_span_keys_by_handle,
        )
        deduped_handle_ids = {
            binding.handle_id
            for contribution in deduped
            for binding in contribution.bindings
        }
        retained_active_full_selection_priorities = {
            handle: priority
            for handle, priority in active_full_selection_priorities.items()
            if handle in deduped_handle_ids
        }
        minimum_allocation, lane_allocation_audit = _allocate_non_borrowable_lanes(
            deduped,
            operator_spec=spec,
            local_selection_priority_by_handle=(
                retained_active_full_selection_priorities
            ),
        )
        protected_lane_item_receipts = tuple(
            item_receipt
            for lane_receipt in minimum_allocation.receipts
            for item_receipt in lane_receipt.selected_item_receipt_sha256s
        )
        allocated, lane_surplus_audit = _fill_shared_lane_surplus(
            deduped,
            minimum_allocation,
            operator_spec=spec,
            local_selection_priority_by_handle=(
                retained_active_full_selection_priorities
            ),
        )
        allocated_handle_by_mechanism = {
            binding.handle_id: contribution.mechanism_id
            for contribution in allocated
            for binding in contribution.bindings
        }
        fair_active_full_selection_priorities = {
            handle: priority
            for handle, priority in retained_active_full_selection_priorities.items()
            if handle in allocated_handle_by_mechanism
        }
        expected_allocated_active_full_priority_handles = {
            handle
            for handle, mechanism_id in allocated_handle_by_mechanism.items()
            if mechanism_id
            in {FULL_STORE_MECHANISM, ACTIVE_RECONSTRUCTION_MECHANISM}
        }
        _require(
            set(fair_active_full_selection_priorities)
            == expected_allocated_active_full_priority_handles,
            "allocated active/full fair-merge priorities changed coverage",
        )
        packet, fair_merge_audit = _fair_merge_contributions(
            spec,
            allocated,
            local_selection_priority_by_handle=(
                fair_active_full_selection_priorities
            ),
            protected_item_receipt_sha256s=(
                protected_lane_item_receipts
            ),
            minimum_allocation_receipt_sha256=(
                minimum_allocation.receipt_sha256
            ),
            surplus_fill_audit=lane_surplus_audit,
        )
        (
            mechanism_by_handle,
            fair_merge_dropped_allocated_bindings,
        ) = _retained_mechanism_bindings(
            allocated,
            packet,
        )
        final_active_full_selection_priorities = {
            handle: priority
            for handle, priority in retained_active_full_selection_priorities.items()
            if handle in mechanism_by_handle
        }
        expected_active_full_priority_handles = {
            handle
            for handle, mechanism_id in mechanism_by_handle.items()
            if mechanism_id
            in {FULL_STORE_MECHANISM, ACTIVE_RECONSTRUCTION_MECHANISM}
        }
        _require(
            set(final_active_full_selection_priorities)
            == expected_active_full_priority_handles,
            "retained active/full hard-fit priorities changed coverage",
        )

        dedup_by_mechanism = {row.mechanism_id: row for row in allocated}
        base_dedup = tuple(
            dedup_by_mechanism[row.mechanism_id] for row in base_original
        )
        tail_dedup = tuple(
            dedup_by_mechanism[row.mechanism_id] for row in tail_original
        )
        fair_retained_handle_ids = frozenset(
            binding.handle_id for binding in packet.local_bindings
        )
        fair_retained_group_handles = frozenset(
            binding.source_group_handle for binding in packet.local_bindings
        )
        story_keys, prefit_story_key_audit = _local_story_keys(
            parent_map=dedup_by_mechanism[PARENT_MAP_MECHANISM],
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
            retained_handle_ids=fair_retained_handle_ids,
            retained_group_handles=fair_retained_group_handles,
        )
        forbidden = list(_map_forbidden_literals(planned))
        if base_row is not None:
            forbidden.extend(_union_forbidden_literals(base_row))
        if tail_row is not None:
            forbidden.extend(_union_forbidden_literals(tail_row))
        forbidden.extend(
            _full_store_forbidden_literals(closure_by_question[question_id])
        )
        forbidden.extend(_active_forbidden_literals(active_result))
        fitted = fit_typed_final_prompt(
            dated_question=source_packet.dated_question,
            parent_prediction=parent_row.prediction,
            packet=packet,
            mechanism_by_handle=mechanism_by_handle,
            local_story_keys_by_group=story_keys,
            local_retention_priority_by_handle=(
                final_active_full_selection_priorities
            ),
            forbidden_provider_literals=tuple(dict.fromkeys(forbidden)),
            minimum_usable_items_per_mechanism=1,
            protected_item_receipt_sha256s=protected_lane_item_receipts,
            protection_source_receipt_sha256=(
                fair_merge_audit["receipt_sha256"]
            ),
        )
        _require(
            fitted.prompt_token_proxy + OUTPUT_TOKEN_RESERVE
            <= MAX_CHAT_PROMPT_TOKENS + OUTPUT_TOKEN_RESERVE,
            "complete typed final chat escaped its hard envelope",
        )
        final_story_keys, story_key_audit = _local_story_keys(
            parent_map=dedup_by_mechanism[PARENT_MAP_MECHANISM],
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
            retained_group_handles=frozenset(
                fitted.handle_group_by_id.values()
            ),
        )
        _require(
            all(
                set(group_keys) <= set(story_keys[group])
                for group, group_keys in final_story_keys.items()
            ),
            "final fitted story provenance escaped the fair packet",
        )
        active_quote_by_handle = {
            binding.handle_id: candidate.quote
            for binding, candidate in zip(
                active_contribution.bindings,
                active_result.candidates,
                strict=True,
            )
        }
        _require(
            all(
                item.summary
                == active_quote_by_handle[item.handle_ids[0]]
                for item in fitted.packet.items
                if len(item.handle_ids) == 1
                and item.handle_ids[0] in active_quote_by_handle
            ),
            "final fitting rewrote an active exact chunk",
        )
        connectivity_ledger = build_typed_connectivity_ledger(
            tuple(original),
            fitted,
            post_selection_dedup_exclusions=postselection_exclusions,
        )

        local_audit = {
            "adaptive_parent_map": _map_local_audit(parent_map, planned),
            "adaptive_parent_source": (
                None
                if base_row is None
                else _union_local_audit(
                    base_row,
                    base_original,
                    parent_prompt_token_proxy=parent_prompt_proxy,
                )
            ),
            "adaptive_tail_source": (
                None
                if tail_row is None
                else _union_local_audit(
                    tail_row,
                    tail_original,
                    parent_prompt_token_proxy=parent_prompt_proxy,
                )
            ),
            "full_store_slot_closure": full_audit.projection(),
            "full_store_selection_priority": (
                full_selection_priority_audit
            ),
            "active_reconstruction": {
                "contribution": active_contribution.projection(),
                "local_result": active_result.local_audit_projection(),
                "parent_alignment": active_parent_alignment_audit,
                "provider_projection_sha256": identity_sha256(
                    active_result.provider_projection()
                ),
                "scanner_batches_reused_without_rescan": True,
            },
            "active_full_selection_priority": (
                active_selection_priority_audit
            ),
            "fair_premerge": fair_merge_audit,
            "fair_premerge_dropped_allocated_bindings": (
                fair_merge_dropped_allocated_bindings
            ),
            "non_borrowable_lane_allocation": lane_allocation_audit,
            "shared_lane_surplus_fill": lane_surplus_audit,
            "local_to_global_connectivity": connectivity_ledger,
            "post_selection_dedup_exclusions": list(postselection_exclusions),
            "retained_fitted_bindings": [
                row.projection() for row in fitted.packet.local_bindings
            ],
            "story_link_local_bindings": [
                dict(row) for row in fitted.story_link_local_bindings
            ],
            "story_source_history_keys_pre_fit": (
                prefit_story_key_audit
            ),
            "story_source_history_keys": story_key_audit,
        }
        final_mechanism_counts: dict[str, int] = {}
        for item in fitted.packet.items:
            if not item.included or item.content_conflict:
                continue
            owners = {
                fitted.mechanism_by_handle[handle]
                for handle in item.handle_ids
            }
            for owner in owners:
                final_mechanism_counts[owner] = (
                    final_mechanism_counts.get(owner, 0) + 1
                )
        required_mechanisms = {
            row["mechanism_id"]
            for row in fair_merge_audit["mechanisms"]
            if row["usable_candidate_count"] > 0
        }
        _require(
            all(final_mechanism_counts.get(row, 0) >= 1 for row in required_mechanisms),
            "final prompt starved a nonempty typed mechanism",
        )
        local_audit["final_usable_item_count_by_mechanism"] = (
            final_mechanism_counts
        )
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
            "preservation_requirements": dict(
                fitted.preservation_requirements
            ),
            "validation_contract": dict(fitted.validation_contract),
            "provider_projection": provider_projection,
            "question_id": question_id,
            "question_sha256": source_packet.question_sha256,
            "route_id": spec.style.value,
            "story_coherence": dict(fitted.story_coherence),
            "typed_composition_receipt_sha256": fitted.receipt_sha256,
        }
        composition_rows.append(
            {**body, "composition_row_sha256": identity_sha256(body)}
        )

    payload = {
        "cache_receipts": list(cache_receipts),
        "closure_input_artifact_sha256": closure_artifact.sha256,
        "database_read_passes_per_unique_namespace": 1,
        "format": COMPOSITION_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "parent_adaptive_run_sha256": parent.terminal.sha256,
        "parent_map_run_sha256": parent.loaded.map_plane.run_sha256,
        "parent_source_materialization_sha256": base_terminal.sha256,
        "question_count": len(composition_rows),
        "questions": composition_rows,
        "retained_transformer_token_state_bytes": 0,
        "tail_materialization_sha256": tail_terminal.sha256,
        "unique_namespace_count": len(cache_receipts),
    }
    _require(
        len(composition_rows) == EXPECTED_QUESTION_COUNT
        and len({row["question_id"] for row in composition_rows})
        == EXPECTED_QUESTION_COUNT
        and len({row["ordinal"] for row in composition_rows})
        == EXPECTED_QUESTION_COUNT,
        "typed final composition population changed",
    )
    assert_gold_blind(payload, path="locked_typed_final_composition")
    return payload, closure_artifact


def _compose(args: argparse.Namespace) -> dict[str, Any]:
    payload, closure = _composition_projection(args)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / COMPOSITION_NAME,
        payload,
    )
    return {
        "artifact": artifact.path.as_posix(),
        "closure_input_sha256": closure.sha256,
        "composition_sha256": artifact.sha256,
        "created": created,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "question_count": payload["question_count"],
        "retained_transformer_token_state_bytes": 0,
    }


def _read_composition(
    output_root: Path,
    expected_sha256: str,
) -> SealedArtifact:
    expected = require_sha256(expected_sha256, "expected typed composition")
    artifact = read_sealed_json(output_root / COMPOSITION_NAME)
    _require(artifact.sha256 == expected, "typed composition artifact changed")
    payload = artifact.payload
    assert_gold_blind(payload, path="typed_final_read_composition")
    rows = payload.get("questions")
    _require(
        payload.get("format") == COMPOSITION_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and type(rows) is list
        and len(rows) == EXPECTED_QUESTION_COUNT,
        "typed composition firewall or population changed",
    )
    observed_ids: list[str] = []
    for ordinal, row in enumerate(rows):
        _require(type(row) is dict, "typed composition row changed type")
        assert type(row) is dict
        declared = require_sha256(
            row.get("composition_row_sha256"), "typed composition row"
        )
        body = dict(row)
        body.pop("composition_row_sha256")
        _require(
            identity_sha256(body) == declared
            and row.get("ordinal") == ordinal
            and row.get("format") == COMPOSITION_FORMAT,
            "typed composition row seal/order changed",
        )
        observed_ids.append(require_text(row.get("question_id"), "composition question"))
    _require(
        len(set(observed_ids)) == EXPECTED_QUESTION_COUNT,
        "typed composition question identities repeat",
    )
    return artifact


def _prompt_plan_row(composition_row: Mapping[str, Any]) -> dict[str, Any]:
    provider = composition_row.get("provider_projection")
    _require(type(provider) is dict, "composition provider projection missing")
    assert type(provider) is dict
    provider_input = provider.get("provider_input")
    _require(type(provider_input) is dict, "composition provider input missing")
    assert type(provider_input) is dict
    messages = render_final_messages(provider_input)
    messages_sha = identity_sha256(list(messages))
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        messages_sha
        == require_sha256(provider.get("messages_sha256"), "composition messages")
        and prompt_tokens == provider.get("prompt_token_proxy")
        and prompt_tokens + OUTPUT_TOKEN_RESERVE <= 8_000
        and provider.get("full_chat_plus_output_tokens")
        == prompt_tokens + OUTPUT_TOKEN_RESERVE,
        "composition complete chat projection changed or exceeds 8k",
    )
    body = {
        "allowed_handle_ids": list(composition_row.get("allowed_handle_ids", [])),
        "composition_row_sha256": require_sha256(
            composition_row.get("composition_row_sha256"), "composition row"
        ),
        "dated_question_sha256": require_sha256(
            composition_row.get("dated_question_sha256"), "dated question"
        ),
        "handle_group_by_id": dict(
            composition_row.get("handle_group_by_id", {})
        ),
        "messages": list(messages),
        "messages_sha256": messages_sha,
        "ordinal": composition_row.get("ordinal"),
        "parent_prediction": require_text(
            composition_row.get("parent_prediction"), "parent prediction"
        ),
        "preservation_requirements": dict(
            composition_row.get("preservation_requirements", {})
        ),
        "validation_contract": dict(
            composition_row.get("validation_contract", {})
        ),
        "prompt_token_proxy": prompt_tokens,
        "question_id": require_text(
            composition_row.get("question_id"), "prompt question"
        ),
        "question_sha256": require_sha256(
            composition_row.get("question_sha256"), "prompt question"
        ),
        "route_id": require_text(
            composition_row.get("route_id"), "prompt route"
        ),
        "story_coherence": dict(
            composition_row.get("story_coherence", {})
        ),
        "typed_composition_receipt_sha256": require_sha256(
            composition_row.get("typed_composition_receipt_sha256"),
            "typed composition receipt",
        ),
    }
    _require(
        type(body["ordinal"]) is int
        and body["ordinal"] >= 0
        and set(body["handle_group_by_id"]) == set(body["allowed_handle_ids"]),
        "typed prompt parser bindings changed",
    )
    body["prompt_row_receipt_sha256"] = identity_sha256(body)
    assert_gold_blind(body, path="typed_final_prompt_plan_row")
    return body


def _preflight_projection(
    composition: SealedArtifact,
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    require_text(model, "typed final model")
    require_text(gateway_url, "typed final gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "typed final concurrency changed",
    )
    raw_rows = composition.payload["questions"]
    rows = tuple(_prompt_plan_row(row) for row in raw_rows)
    prompts = tuple(tuple(row for row in plan["messages"]) for plan in rows)
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == EXPECTED_QUESTION_COUNT,
        "typed final requires 100 distinct physical prompt identities",
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
        "question_count": EXPECTED_QUESTION_COUNT,
        "required_authorized_provider_calls": EXPECTED_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "source_hash_bindings": {
            key: composition.payload[key]
            for key in (
                "closure_input_artifact_sha256",
                "parent_adaptive_run_sha256",
                "parent_map_run_sha256",
                "parent_source_materialization_sha256",
                "tail_materialization_sha256",
            )
        },
    }
    assert_gold_blind(payload, path="locked_typed_final_preflight")
    return payload, prompts


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    composition = _read_composition(
        Path(args.output_root), args.expected_composition_sha256
    )
    payload, _prompts = _preflight_projection(
        composition,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / PREFLIGHT_NAME,
        payload,
    )
    return {
        "artifact": artifact.path.as_posix(),
        "composition_sha256": composition.sha256,
        "created": created,
        "gold_loaded": False,
        "maximum_complete_prompt_envelope": payload[
            "observed_max_complete_envelope_tokens"
        ],
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "question_count": EXPECTED_QUESTION_COUNT,
        "required_authorized_provider_calls": EXPECTED_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    assert_gold_blind(payload, path="typed_final_provider_preflight")
    rows = payload.get("physical_prompt_rows")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_prompt_token_cap") == 8_000
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and payload.get("required_authorized_provider_calls")
        == EXPECTED_QUESTION_COUNT
        and type(rows) is list
        and len(rows) == EXPECTED_QUESTION_COUNT,
        "typed final sealed preflight firewall/population changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    validated_rows: list[dict[str, Any]] = []
    question_ids: list[str] = []
    for ordinal, raw in enumerate(rows):
        _require(type(raw) is dict, "typed final prompt row changed type")
        assert type(raw) is dict
        declared = require_sha256(
            raw.get("prompt_row_receipt_sha256"), "typed final prompt row"
        )
        body = dict(raw)
        body.pop("prompt_row_receipt_sha256")
        messages = raw.get("messages")
        _require(
            identity_sha256(body) == declared
            and raw.get("ordinal") == ordinal
            and type(messages) is list,
            "typed final prompt row seal/order changed",
        )
        plain = tuple(
            {"role": row["role"], "content": row["content"]}
            for row in messages
            if type(row) is dict
            and set(row) == {"role", "content"}
            and type(row.get("role")) is str
            and type(row.get("content")) is str
        )
        _require(
            len(plain) == len(messages)
            and identity_sha256(list(plain)) == raw.get("messages_sha256")
            and count_chat_prompt_token_proxy(plain)
            == raw.get("prompt_token_proxy")
            and int(raw["prompt_token_proxy"]) + OUTPUT_TOKEN_RESERVE <= 8_000,
            "typed final prompt messages/budget changed",
        )
        prompts.append(plain)
        validated_rows.append(dict(raw))
        question_ids.append(require_text(raw.get("question_id"), "prompt question"))
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == EXPECTED_QUESTION_COUNT
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.model_dump() == payload.get("prompt_population")
        and len(set(question_ids)) == EXPECTED_QUESTION_COUNT,
        "typed final sealed physical population changed",
    )
    return tuple(prompts), tuple(validated_rows)


def _read_preflight(
    output_root: Path,
    expected_sha256: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    expected = require_sha256(expected_sha256, "expected typed final preflight")
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(artifact.sha256 == expected, "typed final preflight changed")
    prompts, rows = _validate_preflight(artifact)
    return artifact, prompts, rows


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
        "typed final runtime settings differ from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=output_root / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "arm": "locked_common_typed_memory_final_answer_v1",
            "authorized_unique_calls": EXPECTED_QUESTION_COUNT,
            "composition_artifact_sha256": payload[
                "composition_artifact_sha256"
            ],
            "experiment_format": RUN_FORMAT,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "preflight_artifact_sha256": artifact.sha256,
            "source_hash_bindings": payload["source_hash_bindings"],
        },
    )


def _checkpoint_batch(
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
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


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, _rows = _read_preflight(
        Path(args.output_root), args.expected_preflight_sha256
    )
    required = len(prompts)
    _require(
        required == EXPECTED_QUESTION_COUNT
        and args.enable_provider is True
        and args.authorized_provider_calls == required,
        f"provider-run requires exact authorization for {required} calls",
    )
    # The entire immutable gold-blind population and exact authorization are
    # verified before environment access, client construction, or checkpoint
    # mutation.
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))  # noqa: SLF001
    try:
        batch = _checkpoint_batch(
            artifact,
            prompts,
            args=args,
            client=client,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == EXPECTED_QUESTION_COUNT,
        "typed final provider population changed",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
    }


def _materialization_projection(
    preflight: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == EXPECTED_QUESTION_COUNT
        and batch.usage.checkpoint_hits == EXPECTED_QUESTION_COUNT
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == EXPECTED_QUESTION_COUNT
        and len(batch.unique_records) == EXPECTED_QUESTION_COUNT,
        "typed final materialization requires 100 checkpoint-only completions",
    )
    record_by_messages = {
        row.messages_sha256: row for row in batch.unique_records
    }
    _require(
        len(record_by_messages) == EXPECTED_QUESTION_COUNT,
        "typed final completion record identities repeat",
    )
    results: list[dict[str, Any]] = []
    for plan, completion in zip(
        prompt_rows, batch.logical_completions, strict=True
    ):
        record = record_by_messages.get(plan["messages_sha256"])
        _require(record is not None, "typed final completion lost its prompt")
        assert record is not None
        _require(
            record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            "typed final checkpoint record changed",
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
    judge_rows = [judge_row_projection(row) for row in results]
    _require(
        len(results) == len(judge_rows) == EXPECTED_QUESTION_COUNT
        and tuple(row["question_id"] for row in results)
        == tuple(row["question_id"] for row in judge_rows),
        "typed final judge seam changed prediction/source identities",
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
            row["prediction_source"]
            == "typed_final_invalid_keep_parent_v1"
            for row in results
        ),
        "judge_rows": judge_rows,
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": EXPECTED_QUESTION_COUNT,
        "questions": results,
        "required_authorized_provider_calls": EXPECTED_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "source_hash_bindings": preflight.payload["source_hash_bindings"],
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="locked_typed_final_run")
    return payload


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, rows = _read_preflight(
        Path(args.output_root), args.expected_preflight_sha256
    )
    batch = _checkpoint_batch(
        preflight,
        prompts,
        args=args,
        client=None,
    )
    payload = _materialization_projection(preflight, rows, batch)
    terminal, created = publish_sealed_json(
        Path(args.output_root) / RUN_NAME,
        payload,
    )
    return {
        "changed_prediction_count": payload["changed_prediction_count"],
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "run_sha256": terminal.sha256,
        "terminal_run_replayed": not created,
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    composition = _read_composition(
        Path(args.output_root), args.expected_composition_sha256
    )
    preflight, prompts, rows = _read_preflight(
        Path(args.output_root), args.expected_preflight_sha256
    )
    _require(
        preflight.payload.get("composition_artifact_sha256")
        == composition.sha256,
        "typed replay composition/preflight binding changed",
    )
    batch = _checkpoint_batch(
        preflight,
        prompts,
        args=args,
        client=None,
    )
    rebuilt = _materialization_projection(preflight, rows, batch)
    expected_run = require_sha256(args.expected_run_sha256, "expected typed run")
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        terminal.sha256 == expected_run and terminal.payload == rebuilt,
        "typed final terminal differs from checkpoint-only replay",
    )
    replay_payload = {
        "byte_identical": True,
        "composition_artifact_sha256": composition.sha256,
        "expected_run_sha256": expected_run,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "replayed_run_sha256": terminal.sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(replay_payload, path="locked_typed_final_replay")
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


def _add_runtime_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=live.DEFAULT_TERRA_GATEWAY_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def _add_compose_inputs(parser: argparse.ArgumentParser) -> None:
    _add_runtime_settings(parser)
    parser.add_argument(
        "--parent-source-root", type=Path, default=DEFAULT_PARENT_SOURCE_ROOT
    )
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--tail-root", type=Path, default=DEFAULT_TAIL_ROOT)
    parser.add_argument(
        "--expected-parent-source-preflight-sha256",
        default=EXPECTED_PARENT_SOURCE_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-parent-source-materialization-sha256",
        default=EXPECTED_PARENT_SOURCE_MATERIALIZATION_SHA256,
    )
    parser.add_argument(
        "--expected-parent-preflight-sha256",
        default=EXPECTED_PARENT_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-parent-run-sha256", default=EXPECTED_PARENT_RUN_SHA256
    )
    parser.add_argument(
        "--expected-tail-preflight-sha256",
        default=EXPECTED_TAIL_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-tail-materialization-sha256",
        default=EXPECTED_TAIL_MATERIALIZATION_SHA256,
    )
    parser.add_argument(
        "--expected-tail-replay-sha256", default=EXPECTED_TAIL_REPLAY_SHA256
    )
    parser.add_argument(
        "--retrieval", type=Path, default=guided_scan_cli.DEFAULT_RETRIEVAL
    )
    parser.add_argument(
        "--store-root", type=Path, default=guided_scan_cli.DEFAULT_STORE_ROOT
    )
    parser.add_argument(
        "--query-parent-output-root",
        type=Path,
        default=guided_scan_cli.DEFAULT_PARENT_OUTPUT,
    )
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=guided_scan_cli.EXPECTED_RETRIEVAL_SHA256,
    )
    parser.add_argument(
        "--expected-query-parent-preflight-sha256",
        default=guided_scan_cli.EXPECTED_PARENT_PREFLIGHT_SHA256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    compose = commands.add_parser(
        "compose", help="seal provider-free typed evidence and local receipts"
    )
    _add_compose_inputs(compose)

    preflight = commands.add_parser(
        "preflight", help="seal one exact full chat prompt per locked question"
    )
    _add_runtime_settings(preflight)
    preflight.add_argument("--expected-composition-sha256", required=True)

    provider = commands.add_parser(
        "provider-run", help="execute only the sealed 100-prompt population"
    )
    _add_runtime_settings(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser(
        "materialize", help="consume immutable completion checkpoints only"
    )
    _add_runtime_settings(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)

    replay = commands.add_parser(
        "replay", help="replay the terminal run from sealed checkpoints only"
    )
    _add_runtime_settings(replay)
    replay.add_argument("--expected-composition-sha256", required=True)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compose":
        result = _compose(args)
    elif args.command == "preflight":
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
    "CLOSURE_INPUT_NAME",
    "COMPOSITION_NAME",
    "DEFAULT_OUTPUT",
    "FORMAT",
    "LockedTypedMemoryFinalError",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "main",
]
