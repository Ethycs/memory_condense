#!/usr/bin/env python3
"""Rebuild sealed R7 namespaces and assay provider-free V6 local linking.

This diagnostic accepts an explicit ordinal population, but the V6 mechanism
it invokes contains no ordinal, reference, outcome, or source-ID route.  Each
selected R/P handle is reconstructed from the byte-authenticated R7 terminal
mapping, joined back to the immutable store/index, and passed through the
generic source-group reinjection primitive.  Episode metadata can be resolved
independently inside each immutable namespace.  No residual index is
serialized; the exact R7 loader seam is reused one namespace at a time.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.persistence.db import Database  # noqa: E402
from memory_condense.persistence.discourse_store import DiscourseStore  # noqa: E402
from memory_condense.search.episodes.retrieval import (  # noqa: E402
    EpisodeRetrievalPolicy,
)
from tools import run_locked_semantic_residual_construction_v4 as r7_cli  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as reduced_cli  # noqa: E402
from tools import run_reduced_semantic_binary_search_assay as semantic_cli  # noqa: E402
from tools.matched_eval import semantic_residual_search as residual  # noqa: E402
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
)
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    LocalCitationBinding,
    build_full_store_window_index,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    cache_namespace_partitions,
)
from tools.matched_eval.source_group_reinjection import (  # noqa: E402
    SourceGroupReinjectionPolicy,
    authenticate_source_group_selection,
    replay_source_group_reinjection,
    search_source_group_reinjection,
)


FORMAT = "memory-condense-reduced-source-group-reinjection-assay-v2"
CONSTRUCTION_NAME = "reduced-source-group-reinjection-assay-v2.json"
REPLAY_NAME = "reduced-source-group-reinjection-assay-replay-v2.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R7_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-residual-v4-r7"
)
DEFAULT_R7_CONSTRUCTION = DEFAULT_R7_ROOT / r7_cli.CONSTRUCTION_NAME
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-source-group-reinjection-v6-r2"
)
FIXED_INTERVAL_EPISODE_KINDS = (
    "fixed_interval",
    "longmemeval-diffuse-fixed_interval",
)
EXPECTED_R7_CONSTRUCTION_SHA256 = (
    "d0f226b1577a6bf40c54758d2fdc477ab98483613ca7c4fc77ef93383a651f6a"
)
EXPECTED_R7_GATE_SHA256 = (
    "779c711e090ecb9faad92d9845158d939411dfa3a965669a26cfe8a8062fb912"
)
EXPECTED_R7_VECTOR_SHA256 = (
    "ce9b10803146a70ec18d9c907aceb2fa469fa5491818bc72721e7f5cefbcc8e2"
)


class ReducedSourceGroupReinjectionAssayError(MatchedEvalContractError):
    """An R7 artifact, reconstructed namespace, handle, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSourceGroupReinjectionAssayError(message)


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


def _resolve_episode_artifact(
    database: Database,
    args: argparse.Namespace,
) -> tuple[
    DiscourseStore | None,
    EpisodeRetrievalPolicy | None,
    dict[str, Any],
]:
    """Resolve one namespace-local episode artifact from authenticated metadata."""

    explicit_id = getattr(args, "episode_artifact_id", None)
    auto = bool(getattr(args, "auto_resolve_episode_artifact", False))
    _require(
        not (auto and explicit_id is not None),
        "episode artifact auto-resolution and explicit ID are mutually exclusive",
    )
    if not auto and explicit_id is None:
        binding = _with_receipt(
            {
                "artifact_id": None,
                "artifact_kind": None,
                "artifact_receipt_sha256": None,
                "episode_count": 0,
                "resolution_mode": "disabled",
            },
            "episode_artifact_binding_receipt_sha256",
        )
        return None, None, binding

    store = DiscourseStore(database)
    if auto:
        placeholders = ", ".join("?" for _ in FIXED_INTERVAL_EPISODE_KINDS)
        rows = database.execute(
            "SELECT DISTINCT a.artifact_id "
            "FROM discourse_artifacts AS a "
            "JOIN episodes AS e ON e.artifact_id = a.artifact_id "
            f"WHERE a.kind IN ({placeholders}) ORDER BY a.artifact_id",
            tuple(FIXED_INTERVAL_EPISODE_KINDS),
        ).fetchall()
        _require(
            len(rows) == 1,
            "episode auto-resolution requires exactly one populated fixed-interval artifact",
        )
        artifact_id = str(rows[0][0])
        mode = "authenticated_namespace_fixed_interval_auto"
    else:
        _require(
            type(explicit_id) is str and bool(explicit_id),
            "explicit episode artifact ID must be non-empty text",
        )
        artifact_id = str(explicit_id)
        mode = "explicit_artifact_id"
    artifact = store.get_artifact(artifact_id)
    _require(artifact is not None, "episode artifact is absent from immutable namespace")
    assert artifact is not None
    episode_count = int(
        database.execute(
            "SELECT COUNT(*) FROM episodes WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()[0]
    )
    _require(episode_count > 0, "episode artifact contains no published episodes")
    binding = _with_receipt(
        {
            "artifact_id": artifact.artifact_id,
            "artifact_kind": artifact.kind,
            "artifact_receipt_sha256": identity_sha256(
                artifact.identity_payload()
            ),
            "episode_count": episode_count,
            "resolution_mode": mode,
        },
        "episode_artifact_binding_receipt_sha256",
    )
    policy = EpisodeRetrievalPolicy(
        artifact_id=artifact.artifact_id,
        max_anchor_episodes=int(args.max_episode_anchors),
        previous_episodes=int(args.previous_episodes),
        next_episodes=int(args.next_episodes),
        max_episode_seeds=int(args.max_episode_seeds),
        max_direct_fallbacks=int(args.max_episode_direct_fallbacks),
    )
    return store, policy, binding


def _verified_r7_construction(args: argparse.Namespace) -> SealedArtifact:
    artifact = read_sealed_json(Path(args.r7_construction))
    _require(
        artifact.sha256
        == require_sha256(
            str(args.expected_r7_construction_sha256), "R7 construction"
        ),
        "R7 construction artifact changed",
    )
    payload = _exact_dict(artifact.payload, "R7 construction")
    unsigned = dict(payload)
    declared = require_sha256(
        unsigned.pop("construction_identity_sha256", None),
        "R7 construction identity",
    )
    _require(
        payload.get("format") == r7_cli.CONSTRUCTION_FORMAT
        and identity_sha256(unsigned) == declared
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0,
        "R7 construction internal identity changed",
    )
    assert_gold_blind(payload, path="source_group_assay.r7_construction")
    return artifact


def _selected_handle_bindings(
    semantic_index: residual.SemanticResidualIndex,
    search: residual.SemanticResidualSearchResult,
    protected: Sequence[LocalCitationBinding],
    terminal: Mapping[str, Any],
) -> tuple[
    dict[str, LocalCitationBinding],
    dict[str, str],
    tuple[str, ...],
]:
    protected_by_receipt = {row.receipt_sha256: row for row in protected}
    _require(
        len(protected_by_receipt) == len(protected),
        "R7 protected binding population repeated a receipt",
    )
    handle_bindings: dict[str, LocalCitationBinding] = {}
    for index, binding in enumerate(search.local_bindings, start=1):
        handle_bindings[f"R{index:04d}"] = binding
    for index, duplicate in enumerate(search.protected_duplicates, start=1):
        binding = protected_by_receipt.get(
            duplicate.protected_binding_receipt_sha256
        )
        _require(
            binding is not None
            and binding.candidate_id == duplicate.protected_candidate_id,
            "R7 protected duplicate lost its immutable provider owner",
        )
        assert binding is not None
        handle_bindings[f"P{index:04d}"] = binding

    mapping = _exact_dict(
        terminal.get("prompt_external_unified_group_mapping"),
        "R7 terminal group mapping",
    )
    mapping_body = dict(mapping)
    declared_mapping = require_sha256(
        mapping_body.pop("receipt_sha256", None), "R7 terminal group mapping"
    )
    _require(
        identity_sha256(mapping_body) == declared_mapping,
        "R7 terminal group mapping receipt changed",
    )
    group_by_handle: dict[str, str] = {}
    source_by_handle: dict[str, str] = {}
    for raw in _exact_list(mapping.get("rows"), "R7 terminal group rows"):
        row = _exact_dict(raw, "R7 terminal group row")
        for handle in _exact_list(
            row.get("evidence_handle_ids"), "R7 mapped handles"
        ):
            _require(handle not in group_by_handle, "R7 handle repeated a G mapping")
            group_by_handle[str(handle)] = str(row["source_group_handle"])
            source_by_handle[str(handle)] = str(row["source_id"])
    _require(
        set(group_by_handle) == set(handle_bindings)
        and all(
            source_by_handle[handle] == binding.source_id
            for handle, binding in handle_bindings.items()
        ),
        "R7 terminal handle population differs from reconstructed exact owners",
    )
    retained_source_ids = tuple(
        sorted({row.source_id for row in search.attempted_selection})
    )
    expected_groups = residual.semantic_residual_source_group_map(retained_source_ids)
    _require(
        all(
            group_by_handle[handle] == expected_groups[binding.source_id]
            for handle, binding in handle_bindings.items()
        )
        and mapping.get("retained_source_count") == len(retained_source_ids),
        "R7 terminal G mapping differs from retained-source allocation",
    )
    return handle_bindings, group_by_handle, retained_source_ids


def _compact_selection(selection: Any) -> dict[str, Any]:
    body = {
        "group_mapping_receipt_sha256": selection.group_mapping_receipt_sha256,
        "group_row_population_sha256": identity_sha256(
            [row.receipt_sha256 for row in selection.group_rows]
        ),
        "group_universe_source_identity_population_sha256": identity_sha256(
            [
                residual.semantic_residual_source_identity_receipt(source_id)
                for source_id in selection.group_universe_source_ids
            ]
        ),
        "residual_index_receipt_sha256": (
            selection.residual_index_receipt_sha256
        ),
        "selected_handle_reverse_maps": [
            row.projection() for row in selection.selected_handles
        ],
        "selection_receipt_sha256": selection.receipt_sha256,
    }
    return _with_receipt(body)


def _question_assay(
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
    episode_lookup: DiscourseStore | None,
    episode_policy: EpisodeRetrievalPolicy | None,
    episode_artifact_binding_receipt_sha256: str,
) -> dict[str, Any]:
    dated_question = str(gate_row["dated_question"])
    query = residual.compile_semantic_residual_query(
        semantic_index,
        dated_question,
        query_vectors=vectors,
        query_vector_artifact_sha256=vector_artifact_sha256,
    )
    protected = r7_cli._protected_evidence(  # noqa: SLF001
        construction_row=source_construction_row,
        composition_row=composition_row,
        composition_sha256=composition_sha256,
        namespace_id=semantic_index.namespace_id,
    )
    search = residual.search_semantic_residual(
        semantic_index, query, protected_evidence=protected
    )
    terminal, terminal_reason = r7_cli.build_separate_terminal_prompt(
        dated_question=dated_question,
        current_prediction=str(gate_row["current_prediction"]),
        result=search,
        residual_index=semantic_index,
        protected_evidence=protected,
        policy=r7_policy,
        protected_owner_token_cap=protected_owner_token_cap,
    )
    _require(
        terminal_reason == "none"
        and terminal is not None
        and terminal == r7_question.get("terminal_prompt")
        and r7_cli._compact_query_commitment(query)  # noqa: SLF001
        == r7_question.get("semantic_query_commitment")
        and r7_cli._compact_search_commitment(search)  # noqa: SLF001
        == r7_question.get("semantic_search_commitment")
        and r7_cli._selected_provenance(semantic_index, search)  # noqa: SLF001
        == r7_question.get("selected_exact_provenance"),
        f"R7 question {ordinal} differs from exact store/search replay",
    )
    handle_bindings, handle_groups, retained_sources = _selected_handle_bindings(
        semantic_index, search, protected, terminal
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
    replayed = replay_source_group_reinjection(
        semantic_index,
        query,
        selection,
        local,
        protected_handle_bindings=handle_bindings,
        episode_lookup=episode_lookup,
        episode_policy=episode_policy,
    )
    _require(
        replayed.receipt_sha256 == local.receipt_sha256
        and replayed.projection() == local.projection(),
        f"V6 question {ordinal} replay changed bytes",
    )
    body = {
        "compact_authenticated_selection": _compact_selection(selection),
        "dated_question_sha256": gate_row["dated_question_sha256"],
        "episode_artifact_binding_receipt_sha256": (
            episode_artifact_binding_receipt_sha256
        ),
        "local_reinjection": local.projection(),
        "namespace_id": semantic_index.namespace_id,
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "question_id": gate_row["question_id"],
        "question_sha256": gate_row["question_sha256"],
        "r7_exact_question_rebuilt": True,
        "r7_question_receipt_sha256": r7_question["question_receipt_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "v6_exact_replay_identical": True,
    }
    return _with_receipt(body, "question_assay_receipt_sha256")


def build_assay(args: argparse.Namespace) -> dict[str, Any]:
    ordinals = tuple(int(value) for value in args.ordinals)
    _require(
        ordinals == tuple(sorted(set(ordinals)))
        and ordinals
        and all(0 <= value < r7_cli.QUESTION_COUNT for value in ordinals),
        "diagnostic ordinals must be sorted unique values in the locked population",
    )
    r7_artifact = _verified_r7_construction(args)
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
        "the explicit assay population contains a non-residual R7 row",
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
    semantic_policy = residual.SemanticResidualPolicy(
        max_cell_tokens=int(args.max_cell_tokens),
        payload_token_cap=sources[8].residual_payload_token_cap,
        cosine_upper_bound_floor=float(args.cosine_upper_bound_floor),
        specificity_upper_bound_ratio=float(args.specificity_upper_bound_ratio),
        dual_gate_enabled=bool(args.dual_gate_enabled),
    )
    _require(
        semantic_policy.projection() == r7_payload.get("residual_search_policy")
        and query_preflight.sha256
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
    by_namespace: dict[str, list[int]] = defaultdict(list)
    for ordinal in ordinals:
        by_namespace[str(gate_rows[ordinal]["namespace_id"])].append(ordinal)
    question_by_ordinal: dict[int, dict[str, Any]] = {}
    namespace_receipts: list[dict[str, Any]] = []
    for namespace_position, namespace_id in enumerate(sorted(by_namespace), start=1):
        print(
            json.dumps(
                {
                    "event": "v6_namespace_start",
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
            "V6 scoped namespace escaped the verified R7 population",
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
                window_index, source_vectors, policy=semantic_policy
            )
            sealed = sealed_cache[namespace_id]
            lifecycle = r7_lifecycle[namespace_id]
            _require(
                sealed.get("cache_receipt_sha256") == cache.cache_receipt_sha256
                and sealed.get("window_index_receipt_sha256")
                == window_index.receipt_sha256
                and r7_cli._compact_index_commitment(semantic_index)  # noqa: SLF001
                == lifecycle.get("question_neutral_semantic_index_commitment")
                and embedding_binding == lifecycle.get("source_query_embedding_binding"),
                "V6 namespace differs from the exact R7 index lifecycle",
            )
            episode_lookup, episode_policy, episode_binding = (
                _resolve_episode_artifact(database, args)
            )
            for ordinal in by_namespace[namespace_id]:
                question_by_ordinal[ordinal] = _question_assay(
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
                    episode_lookup=episode_lookup,
                    episode_policy=episode_policy,
                    episode_artifact_binding_receipt_sha256=str(
                        episode_binding[
                            "episode_artifact_binding_receipt_sha256"
                        ]
                    ),
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
                    "event": "v6_namespace_complete",
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
        "episode_artifact_resolution_mode": (
            "authenticated_namespace_fixed_interval_auto"
            if bool(args.auto_resolve_episode_artifact)
            else (
                "explicit_artifact_id"
                if args.episode_artifact_id is not None
                else "disabled"
            )
        ),
        "format": FORMAT,
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
        "v6_replay_count": len(questions),
    }
    assert_gold_blind(body, path="reduced_source_group_reinjection_assay")
    return {**body, "construction_identity_sha256": identity_sha256(body)}


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_assay(args)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / CONSTRUCTION_NAME, payload
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
        == require_sha256(str(args.expected_assay_sha256), "V6 assay")
        and artifact.payload == rebuilt,
        "V6 assay differs from exact store replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, rebuilt
    )
    _require(replay.sha256 == artifact.sha256, "V6 assay replay changed bytes")
    return {
        "assay_sha256": artifact.sha256,
        "byte_identical": True,
        "new_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def _add_args(parser: argparse.ArgumentParser) -> None:
    r7_cli._add_sources(parser)  # noqa: SLF001
    r7_cli._add_budget(parser)  # noqa: SLF001
    reduced_cli._add_store_args(parser)  # noqa: SLF001
    semantic_cli._add_policy_args(parser)  # noqa: SLF001
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--r7-construction", type=Path, default=DEFAULT_R7_CONSTRUCTION)
    parser.add_argument(
        "--expected-r7-construction-sha256",
        default=EXPECTED_R7_CONSTRUCTION_SHA256,
    )
    parser.add_argument("--vectors", type=Path, default=r7_cli.DEFAULT_VECTORS)
    parser.add_argument(
        "--vector-replay", type=Path, default=r7_cli.DEFAULT_VECTOR_REPLAY
    )
    parser.add_argument(
        "--expected-vector-sha256", default=EXPECTED_R7_VECTOR_SHA256
    )
    parser.set_defaults(expected_gate_sha256=EXPECTED_R7_GATE_SHA256)
    parser.add_argument(
        "--protected-owner-token-cap",
        type=int,
        default=r7_cli.DEFAULT_PROTECTED_OWNER_TOKEN_CAP,
    )
    parser.add_argument("--ordinals", type=int, nargs="+", required=True)
    local = SourceGroupReinjectionPolicy()
    parser.add_argument(
        "--local-payload-token-cap",
        type=int,
        default=local.local_payload_token_cap,
    )
    parser.add_argument(
        "--max-selected-segments", type=int, default=local.max_selected_segments
    )
    parser.add_argument(
        "--base-segments-per-group",
        type=int,
        default=local.base_segments_per_group,
    )
    parser.add_argument(
        "--max-query-term-obligations",
        type=int,
        default=local.max_query_term_obligations,
    )
    parser.add_argument(
        "--source-neighbor-radius", type=int, default=local.source_neighbor_radius
    )
    parser.add_argument(
        "--max-source-neighbors-per-anchor",
        type=int,
        default=local.max_source_neighbors_per_anchor,
    )
    parser.add_argument(
        "--max-episode-segments-per-seed",
        type=int,
        default=local.max_episode_segments_per_seed,
    )
    parser.add_argument("--episode-artifact-id")
    parser.add_argument(
        "--auto-resolve-episode-artifact",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-episode-anchors", type=int, default=8)
    parser.add_argument("--previous-episodes", type=int, default=1)
    parser.add_argument("--next-episodes", type=int, default=1)
    parser.add_argument("--max-episode-seeds", type=int, default=24)
    parser.add_argument("--max-episode-direct-fallbacks", type=int, default=16)


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
    "DEFAULT_OUTPUT_ROOT",
    "EXPECTED_R7_CONSTRUCTION_SHA256",
    "FORMAT",
    "REPLAY_NAME",
    "ReducedSourceGroupReinjectionAssayError",
    "build_assay",
    "build_parser",
    "main",
    "run_construct",
    "run_replay",
]
