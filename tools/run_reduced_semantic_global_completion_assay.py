#!/usr/bin/env python3
"""Rebuild sealed R7 namespaces and assay cumulative provider-free V6+V7.

The explicit ordinal list is a post-score diagnostic population only.  The V7
mechanism receives the dated question, sealed query vectors, generic unresolved
frontier flags, and the immutable common store; it never receives the ordinal,
question ID, reference, outcome, or target source IDs.  Every namespace is
opened once, then R7, V6 local reinjection, and V7 global completion execute in
one pass over the shared exact residual index.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_semantic_residual_construction_v4 as r7_cli  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as reduced_cli  # noqa: E402
from tools import run_reduced_source_group_reinjection_assay as v6_cli  # noqa: E402
from tools.matched_eval import semantic_residual_search as residual  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    LocalCitationBinding,
    build_full_store_window_index,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    cache_namespace_partitions,
)
from tools.matched_eval.semantic_global_completion import (  # noqa: E402
    GlobalLaneBudget,
    SemanticGlobalCompletionPolicy,
    compile_semantic_global_completion_request,
    replay_semantic_global_completion,
    search_semantic_global_completion,
)
from tools.matched_eval.source_group_reinjection import (  # noqa: E402
    SourceGroupReinjectionPolicy,
    authenticate_source_group_selection,
    replay_source_group_reinjection,
    search_source_group_reinjection,
)


FORMAT = "memory-condense-reduced-semantic-global-completion-assay-v1"
CONSTRUCTION_NAME = "reduced-semantic-global-completion-assay-v1.json"
REPLAY_NAME = "reduced-semantic-global-completion-assay-replay-v1.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-global-completion-v7-r1"
)
DEFAULT_ORDINALS = (14, 28, 40, 49, 53, 54, 67, 69, 82, 94, 97)


class ReducedSemanticGlobalCompletionAssayError(MatchedEvalContractError):
    """An R7/V6 binding, shared index, V7 result, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSemanticGlobalCompletionAssayError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _with_receipt(
    body: Mapping[str, Any], key: str = "receipt_sha256"
) -> dict[str, Any]:
    return {**dict(body), key: identity_sha256(body)}


def _sealed_r7_semantic_policy(
    r7_payload: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    payload_token_cap: int,
) -> residual.SemanticResidualPolicy:
    """Reconstruct classifier semantics from the authenticated R7 policy."""

    try:
        policy = residual.semantic_residual_policy_from_projection(
            r7_payload.get("residual_search_policy")
        )
    except residual.SemanticResidualSearchError as exc:
        raise ReducedSemanticGlobalCompletionAssayError(
            "R7 semantic policy authentication failed"
        ) from exc
    _require(
        policy.max_cell_tokens == int(args.max_cell_tokens)
        and policy.payload_token_cap == payload_token_cap
        and policy.cosine_upper_bound_floor
        == float(args.cosine_upper_bound_floor)
        and policy.specificity_upper_bound_ratio
        == float(args.specificity_upper_bound_ratio)
        and policy.dual_gate_enabled is bool(args.dual_gate_enabled),
        "R7 semantic policy parameters changed",
    )
    return policy


def _require_r7_replay_component(
    *,
    ordinal: int,
    component: str,
    actual: object,
    expected: object,
) -> None:
    _require(
        actual == expected,
        f"R7 question {ordinal} {component} replay changed",
    )


def _ordered_protected_union(
    r7_protected: Sequence[LocalCitationBinding],
    r7_global: Sequence[LocalCitationBinding],
    v6_local: Sequence[LocalCitationBinding],
) -> tuple[LocalCitationBinding, ...]:
    rows = tuple((*r7_protected, *r7_global, *v6_local))
    span_receipts = tuple(
        identity_sha256(row.span.identity_payload()) for row in rows
    )
    _require(
        len(set(span_receipts)) == len(span_receipts),
        "cumulative R/P/L protected union repeated an exact span",
    )
    return rows


def run_global_completion_question_adapter(
    *,
    ordinal: int,
    r7_question: Mapping[str, Any],
    gate_row: Mapping[str, Any],
    source_construction_row: Mapping[str, Any],
    composition_row: Mapping[str, Any],
    composition_sha256: str,
    semantic_index: residual.SemanticResidualIndex,
    vectors: Sequence[Sequence[float]],
    vector_artifact_sha256: str,
    r7_policy: Any,
    protected_owner_token_cap: int,
    local_policy: SourceGroupReinjectionPolicy,
    global_policy: SemanticGlobalCompletionPolicy,
    episode_lookup: Any | None = None,
    episode_policy: Any | None = None,
    episode_artifact_binding_receipt_sha256: str | None = None,
    terminal_compiler: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one exact question while the namespace index is resident.

    ``ordinal`` and ``gate_row`` identity fields are used only to authenticate
    the sealed diagnostic row around the mechanism call.  V6/V7 and the
    optional terminal compiler receive no ordinal or target-bearing value.
    """

    dated_question = str(gate_row["dated_question"])
    query = residual.compile_semantic_residual_query(
        semantic_index,
        dated_question,
        query_vectors=vectors,
        query_vector_artifact_sha256=vector_artifact_sha256,
    )
    r7_protected = r7_cli._protected_evidence(  # noqa: SLF001
        construction_row=source_construction_row,
        composition_row=composition_row,
        composition_sha256=composition_sha256,
        namespace_id=semantic_index.namespace_id,
    )
    r7_search = residual.search_semantic_residual(
        semantic_index,
        query,
        protected_evidence=r7_protected,
    )
    terminal, terminal_reason = r7_cli.build_separate_terminal_prompt(
        dated_question=dated_question,
        current_prediction=str(gate_row["current_prediction"]),
        result=r7_search,
        residual_index=semantic_index,
        protected_evidence=r7_protected,
        policy=r7_policy,
        protected_owner_token_cap=protected_owner_token_cap,
    )
    _require(
        terminal_reason == "none" and terminal is not None,
        f"R7 question {ordinal} terminal reconstruction was unavailable",
    )
    _require_r7_replay_component(
        ordinal=ordinal,
        component="terminal prompt",
        actual=terminal,
        expected=r7_question.get("terminal_prompt"),
    )
    _require_r7_replay_component(
        ordinal=ordinal,
        component="semantic query",
        actual=r7_cli._compact_query_commitment(query),  # noqa: SLF001
        expected=r7_question.get("semantic_query_commitment"),
    )
    _require_r7_replay_component(
        ordinal=ordinal,
        component="semantic search",
        actual=r7_cli._compact_search_commitment(r7_search),  # noqa: SLF001
        expected=r7_question.get("semantic_search_commitment"),
    )
    _require_r7_replay_component(
        ordinal=ordinal,
        component="selected provenance",
        actual=r7_cli._selected_provenance(  # noqa: SLF001
            semantic_index, r7_search
        ),
        expected=r7_question.get("selected_exact_provenance"),
    )
    handle_bindings, handle_groups, retained_sources = (
        v6_cli._selected_handle_bindings(  # noqa: SLF001
            semantic_index,
            r7_search,
            r7_protected,
            terminal,
        )
    )
    selection = authenticate_source_group_selection(
        semantic_index,
        handle_bindings,
        group_universe_source_ids=retained_sources,
        selected_handle_groups=handle_groups,
    )
    local = search_source_group_reinjection(
        semantic_index,
        query,
        selection,
        protected_handle_bindings=handle_bindings,
        policy=local_policy,
        episode_lookup=episode_lookup,
        episode_policy=episode_policy,
    )
    local_replay = replay_source_group_reinjection(
        semantic_index,
        query,
        selection,
        local,
        protected_handle_bindings=handle_bindings,
        episode_lookup=episode_lookup,
        episode_policy=episode_policy,
    )
    _require(
        local_replay.receipt_sha256 == local.receipt_sha256
        and local_replay.projection() == local.projection(),
        f"V6 question {ordinal} replay changed bytes",
    )
    protected_union = _ordered_protected_union(
        r7_protected,
        r7_search.local_bindings,
        local.local_bindings,
    )
    request = compile_semantic_global_completion_request(
        query,
        prior_needs_global_search=True,
        operand_closure_missing=query.operator_spec.requires_complete_frontier,
        local_frontier_unresolved=local.frontier.needs_global_search,
    )
    global_result = search_semantic_global_completion(
        semantic_index,
        query,
        request,
        policy=global_policy,
        protected_evidence=protected_union,
    )
    global_replay = replay_semantic_global_completion(
        semantic_index,
        query,
        request,
        global_result,
        protected_evidence=protected_union,
    )
    _require(
        global_replay.receipt_sha256 == global_result.receipt_sha256
        and global_replay.projection() == global_result.projection(),
        f"V7 question {ordinal} replay changed bytes",
    )
    terminal_answer_plan: dict[str, Any] | None = None
    if terminal_compiler is not None:
        terminal_provider = _exact_dict(
            terminal.get("provider_input"), "R7 terminal provider input"
        )
        selected_owner_rows = _exact_list(
            terminal_provider.get("protected_owner_evidence"),
            "R7 selected protected-owner evidence",
        )
        core = terminal_compiler(
            dated_question=dated_question,
            parent_prediction=str(gate_row["current_prediction"]),
            residual_index=semantic_index,
            query=query,
            protected_owner_universe_bindings=r7_protected,
            selected_protected_owner_evidence_rows=selected_owner_rows,
            residual_result=r7_search,
            local_result=local,
            global_result=global_result,
        )
        _require(
            type(core) is dict
            and not {
                "answer_plan_receipt_sha256",
                "dated_question_sha256",
                "ordinal",
                "question_id",
                "question_sha256",
            }
            & set(core)
            and core.get("dated_question") == dated_question
            and core.get("parent_prediction") == gate_row["current_prediction"],
            "terminal compiler returned an invalid gold-blind answer-plan core",
        )
        plan_body = {
            **core,
            "dated_question_sha256": gate_row["dated_question_sha256"],
            "ordinal": ordinal,
            "question_id": gate_row["question_id"],
            "question_sha256": gate_row["question_sha256"],
        }
        assert_gold_blind(plan_body, path="semantic_global_terminal_answer_plan")
        terminal_answer_plan = _with_receipt(
            plan_body, "answer_plan_receipt_sha256"
        )
    body = {
        "dated_question_sha256": gate_row["dated_question_sha256"],
        "episode_artifact_binding_receipt_sha256": (
            episode_artifact_binding_receipt_sha256
        ),
        "global_completion": global_result.projection(),
        "namespace_id": semantic_index.namespace_id,
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "protected_union_binding_receipt_sha256s": [
            row.receipt_sha256 for row in protected_union
        ],
        "protected_union_count": len(protected_union),
        "question_id": gate_row["question_id"],
        "question_sha256": gate_row["question_sha256"],
        "r7_exact_question_rebuilt": True,
        "r7_question_receipt_sha256": r7_question["question_receipt_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "v6_exact_replay_identical": True,
        "v6_frontier": local.frontier.projection(),
        "v6_local_binding_receipt_sha256s": [
            row.receipt_sha256 for row in local.local_bindings
        ],
        "v6_result_receipt_sha256": local.receipt_sha256,
        "v7_exact_replay_identical": True,
    }
    if terminal_answer_plan is not None:
        body["terminal_answer_plan"] = terminal_answer_plan
    return _with_receipt(body, "question_assay_receipt_sha256")


def _global_policy(args: argparse.Namespace) -> SemanticGlobalCompletionPolicy:
    return SemanticGlobalCompletionPolicy(
        global_payload_token_cap=int(args.global_payload_token_cap),
        max_node_visits=int(args.global_max_node_visits),
        max_retained_leaf_cells=int(args.global_max_retained_leaf_cells),
        source_neighbor_radius=int(args.global_source_neighbor_radius),
        max_hydrated_segments=int(args.global_max_hydrated_segments),
        max_entity_obligations=int(args.global_max_entity_obligations),
        lane_budgets=(
            GlobalLaneBudget(
                "dense",
                int(args.dense_max_segments),
                int(args.dense_token_cap),
            ),
            GlobalLaneBudget(
                "sparse",
                int(args.sparse_max_segments),
                int(args.sparse_token_cap),
            ),
            GlobalLaneBudget(
                "personal_temporal",
                int(args.personal_temporal_max_segments),
                int(args.personal_temporal_token_cap),
            ),
            GlobalLaneBudget(
                "source_date_diversity",
                int(args.diversity_max_segments),
                int(args.diversity_token_cap),
            ),
        ),
    )


def build_assay(
    args: argparse.Namespace,
    *,
    terminal_compiler: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    ordinals = tuple(int(value) for value in args.ordinals)
    _require(
        ordinals == tuple(sorted(set(ordinals)))
        and ordinals
        and all(0 <= value < r7_cli.QUESTION_COUNT for value in ordinals),
        "diagnostic ordinals must be sorted unique values in the locked population",
    )
    r7_artifact = v6_cli._verified_r7_construction(args)  # noqa: SLF001
    gate, sources = r7_cli._load_verified_gate(args)  # noqa: SLF001
    vector_artifact, vectors_by_ordinal = r7_cli._load_vectors(  # noqa: SLF001
        Path(args.vectors), str(args.expected_vector_sha256), gate
    )
    vector_replay, replay_vectors = r7_cli._load_vectors(  # noqa: SLF001
        Path(args.vector_replay), str(args.expected_vector_sha256), gate
    )
    _require(
        vector_replay.sha256 == vector_artifact.sha256
        and replay_vectors == vectors_by_ordinal,
        "R7 query-vector replay changed",
    )
    r7_payload = _exact_dict(r7_artifact.payload, "R7 payload")
    bindings = _exact_dict(r7_payload.get("bindings"), "R7 bindings")
    _require(
        bindings.get("gate_artifact_sha256") == gate.sha256
        and bindings.get("query_vector_artifact_sha256") == vector_artifact.sha256
        and r7_payload.get("question_count") == r7_cli.QUESTION_COUNT,
        "R7 construction escaped its exact gate/vector population",
    )
    gate_rows = _exact_list(gate.payload.get("questions"), "R7 gate questions")
    r7_questions = _exact_list(r7_payload.get("questions"), "R7 questions")
    _require(
        all(
            gate_rows[ordinal]["eligibility"]["eligible"]
            and r7_questions[ordinal]["mode"] == "residual_synthesis"
            and ordinal in vectors_by_ordinal
            for ordinal in ordinals
        ),
        "explicit V7 assay population contains a non-residual R7 row",
    )

    guided_args = reduced_cli._guided_args(args)  # noqa: SLF001
    population, query_preflight = (
        reduced_cli.load_preflighted_query_expansion_population(
            Path(guided_args.retrieval),
            output_root=Path(guided_args.query_parent_output_root),
            expected_retrieval_sha256=guided_args.expected_retrieval_sha256,
            expected_question_count=r7_cli.QUESTION_COUNT,
        )
    )
    namespace_by_id = {row.namespace_id: row for row in population.namespaces}
    sealed_cache = r7_cli.parent_cli._cache_receipts_by_namespace(  # noqa: SLF001
        sources[9][1]
    )
    r7_lifecycle = {
        row["namespace_id"]: row
        for row in _exact_list(
            _exact_dict(
                r7_payload.get("resident_index_lifecycle"), "R7 lifecycle"
            ).get("receipts"),
            "R7 lifecycle rows",
        )
    }
    semantic_policy = _sealed_r7_semantic_policy(
        r7_payload,
        args,
        payload_token_cap=sources[8].residual_payload_token_cap,
    )
    _require(
        query_preflight.sha256
        == _exact_dict(
            r7_payload.get("resident_index_lifecycle"), "R7 lifecycle"
        ).get("query_parent_preflight_sha256"),
        "R7 semantic policy/query population changed",
    )
    local_policy = SourceGroupReinjectionPolicy(
        local_payload_token_cap=int(args.local_payload_token_cap),
        max_selected_segments=int(args.max_selected_segments),
        base_segments_per_group=int(args.base_segments_per_group),
        max_query_term_obligations=int(args.max_query_term_obligations),
        source_neighbor_radius=int(args.source_neighbor_radius),
        max_source_neighbors_per_anchor=int(args.max_source_neighbors_per_anchor),
        max_episode_segments_per_seed=int(args.max_episode_segments_per_seed),
    )
    global_policy = _global_policy(args)
    by_namespace: dict[str, list[int]] = defaultdict(list)
    for ordinal in ordinals:
        by_namespace[str(gate_rows[ordinal]["namespace_id"])].append(ordinal)
    question_by_ordinal: dict[int, dict[str, Any]] = {}
    namespace_receipts: list[dict[str, Any]] = []
    for namespace_position, namespace_id in enumerate(sorted(by_namespace), start=1):
        print(
            json.dumps(
                {
                    "event": "v7_namespace_start",
                    "namespace_count": len(by_namespace),
                    "namespace_position": namespace_position,
                    "question_count": len(by_namespace[namespace_id]),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        scoped = reduced_cli._scoped_guided_context(  # noqa: SLF001
            guided_args, namespace_id
        )
        _require(
            scoped.namespace == namespace_by_id.get(namespace_id),
            "V7 scoped namespace escaped the verified R7 population",
        )
        with Database(scoped.store_dir / "memory.db", read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                scoped.namespace,
                source_database_sha256=scoped.database_sha256,
                source_store_receipt_sha256=(
                    scoped.namespace.combined_store_receipt_sha256
                ),
            )
            window_index = build_full_store_window_index(cache)
            source_vectors = residual.load_stored_chunk_vectors(database, window_index)
            embedding_binding = r7_cli._verified_source_embedding_binding(  # noqa: SLF001
                scoped,
                source_vectors,
                _exact_dict(
                    vector_artifact.payload.get("embedding"), "R7 query embedding"
                ),
            )
            semantic_index = residual.build_semantic_residual_index(
                window_index,
                source_vectors,
                policy=semantic_policy,
            )
            sealed = sealed_cache[namespace_id]
            lifecycle = r7_lifecycle[namespace_id]
            _require(
                sealed.get("cache_receipt_sha256") == cache.cache_receipt_sha256
                and sealed.get("window_index_receipt_sha256")
                == window_index.receipt_sha256
                and r7_cli._compact_index_commitment(semantic_index)  # noqa: SLF001
                == lifecycle.get("question_neutral_semantic_index_commitment")
                and embedding_binding
                == lifecycle.get("source_query_embedding_binding"),
                "V7 namespace differs from the exact R7 index lifecycle",
            )
            episode_lookup, episode_policy, episode_binding = (
                v6_cli._resolve_episode_artifact(database, args)  # noqa: SLF001
            )
            for ordinal in by_namespace[namespace_id]:
                question_by_ordinal[ordinal] = run_global_completion_question_adapter(
                    ordinal=ordinal,
                    r7_question=r7_questions[ordinal],
                    gate_row=gate_rows[ordinal],
                    source_construction_row=sources[1][ordinal],
                    composition_row=sources[5][ordinal],
                    composition_sha256=sources[4].sha256,
                    semantic_index=semantic_index,
                    vectors=vectors_by_ordinal[ordinal],
                    vector_artifact_sha256=vector_artifact.sha256,
                    r7_policy=sources[8],
                    protected_owner_token_cap=int(args.protected_owner_token_cap),
                    local_policy=local_policy,
                    global_policy=global_policy,
                    episode_lookup=episode_lookup,
                    episode_policy=episode_policy,
                    episode_artifact_binding_receipt_sha256=str(
                        episode_binding[
                            "episode_artifact_binding_receipt_sha256"
                        ]
                    ),
                    terminal_compiler=terminal_compiler,
                )
        namespace_body = {
            "cache_receipt_sha256": cache.cache_receipt_sha256,
            "episode_artifact_binding": episode_binding,
            "namespace_id": namespace_id,
            "question_assay_receipt_sha256s": [
                question_by_ordinal[ordinal]["question_assay_receipt_sha256"]
                for ordinal in by_namespace[namespace_id]
            ],
            "semantic_residual_index_receipt_sha256": semantic_index.receipt_sha256,
            "source_vector_set_receipt_sha256": source_vectors.receipt_sha256,
            "window_index_receipt_sha256": window_index.receipt_sha256,
        }
        namespace_receipts.append(
            _with_receipt(namespace_body, "namespace_assay_receipt_sha256")
        )
        print(
            json.dumps(
                {
                    "event": "v7_namespace_complete",
                    "namespace_count": len(by_namespace),
                    "namespace_position": namespace_position,
                    "question_count": len(by_namespace[namespace_id]),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        del semantic_index, source_vectors, window_index, cache
        gc.collect()

    questions = [question_by_ordinal[ordinal] for ordinal in ordinals]
    body: dict[str, Any] = {
        "diagnostic_population_explicitly_supplied": True,
        "format": FORMAT,
        "global_policy": global_policy.projection(),
        "gold_loaded": False,
        "local_policy": local_policy.projection(),
        "namespace_receipts": namespace_receipts,
        "new_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "question_count": len(questions),
        "questions": questions,
        "r7_bindings": {
            "construction_artifact_sha256": r7_artifact.sha256,
            "gate_artifact_sha256": gate.sha256,
            "query_vector_artifact_sha256": vector_artifact.sha256,
            "query_vector_replay_artifact_sha256": vector_replay.sha256,
        },
        "retained_transformer_token_state_bytes": 0,
        "source_indexes_rebuilt_not_serialized": True,
        "v6_v7_single_resident_index_pass": True,
        "v7_replay_count": len(questions),
    }
    assert_gold_blind(body, path="reduced_semantic_global_completion_assay")
    return {**body, "construction_identity_sha256": identity_sha256(body)}


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_assay(args)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / CONSTRUCTION_NAME,
        payload,
    )
    return {
        "assay_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": payload["question_count"],
        "retained_transformer_token_state_bytes": 0,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    rebuilt = build_assay(args)
    artifact = read_sealed_json(Path(args.output_root) / CONSTRUCTION_NAME)
    _require(
        artifact.sha256
        == require_sha256(str(args.expected_assay_sha256), "V7 assay")
        and artifact.payload == rebuilt,
        "V7 assay differs from exact store replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME,
        rebuilt,
    )
    _require(replay.sha256 == artifact.sha256, "V7 assay replay changed bytes")
    return {
        "assay_sha256": artifact.sha256,
        "byte_identical": True,
        "new_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def _add_global_args(parser: argparse.ArgumentParser) -> None:
    default = SemanticGlobalCompletionPolicy()
    parser.add_argument(
        "--global-payload-token-cap",
        type=int,
        default=default.global_payload_token_cap,
    )
    parser.add_argument(
        "--global-max-node-visits",
        type=int,
        default=default.max_node_visits,
    )
    parser.add_argument(
        "--global-max-retained-leaf-cells",
        type=int,
        default=default.max_retained_leaf_cells,
    )
    parser.add_argument(
        "--global-source-neighbor-radius",
        type=int,
        default=default.source_neighbor_radius,
    )
    parser.add_argument(
        "--global-max-hydrated-segments",
        type=int,
        default=default.max_hydrated_segments,
    )
    parser.add_argument(
        "--global-max-entity-obligations",
        type=int,
        default=default.max_entity_obligations,
    )
    lane = {row.lane_id: row for row in default.lane_budgets}
    for prefix, lane_id in (
        ("dense", "dense"),
        ("sparse", "sparse"),
        ("personal-temporal", "personal_temporal"),
        ("diversity", "source_date_diversity"),
    ):
        parser.add_argument(
            f"--{prefix}-max-segments",
            type=int,
            default=lane[lane_id].max_selected_segments,
        )
        parser.add_argument(
            f"--{prefix}-token-cap",
            type=int,
            default=lane[lane_id].pre_dedup_token_cap,
        )


def _add_args(parser: argparse.ArgumentParser) -> None:
    v6_cli._add_args(parser)  # noqa: SLF001
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)
    _add_global_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    construct = commands.add_parser("construct")
    _add_args(construct)
    replay = commands.add_parser("replay")
    _add_args(replay)
    replay.add_argument("--expected-assay-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_construct(args) if args.command == "construct" else run_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONSTRUCTION_NAME",
    "DEFAULT_ORDINALS",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "REPLAY_NAME",
    "ReducedSemanticGlobalCompletionAssayError",
    "build_assay",
    "build_parser",
    "main",
    "run_construct",
    "run_global_completion_question_adapter",
    "run_replay",
]
