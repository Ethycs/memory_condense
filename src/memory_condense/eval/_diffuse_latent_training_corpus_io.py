"""Private no-clobber publication and disk verification for route-v2 corpora."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.domain.discourse import ClosureScopeWitness
from memory_condense.eval._diffuse_latent_training_corpus_codec import (
    decode_latent_training_payload,
)
from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
    CorpusTreeSnapshot,
    owned_staging,
    publish_staging,
    remove_owned,
    require_plain_parent,
    write_new,
)
from memory_condense.eval._diffuse_latent_training_corpus_models import (
    LATENT_TRAINING_CORPUS_FORMAT,
    LATENT_TRAINING_PARTITION_FORMAT,
    LATENT_TRAINING_ROW_FORMAT,
    MAX_METADATA_FILE_BYTES,
    MAX_PAYLOAD_SHARD_BYTES,
    ROOT_MANIFEST_NAME,
    AnalysisPopulationProjection,
    DecodedLatentTrainingCorpusRow,
    LatentTrainingCorpusError,
    LatentTrainingCorpusManifest,
    LatentTrainingCorpusPartitionManifest,
    LatentTrainingCorpusPublicationReceipt,
    LatentTrainingCorpusRowManifest,
    LatentTrainingFileIdentity,
    LatentTrainingPopulationExpectation,
    LatentTrainingRouteEvidence,
    VerifiedLatentTrainingFitCorpus,
    VerifiedLatentTrainingFullCorpus,
    VerifiedLatentTrainingValidationCorpus,
    _ATOM_REF_KIND,
    _HYPEREDGE_KIND,
    _canonical_bytes,
    _file_identity,
    _ids_sha256,
    _integer,
    _list,
    _mapping,
    _plain,
    _reconstruct_target,
    _safe_relative,
    _sha,
    _target_body,
    _text,
)
from memory_condense.eval._diffuse_latent_training_corpus_route import (
    live_route_v2_implementation_sha256,
    validate_persisted_route,
)
from memory_condense.eval.diffuse_latent_training_corpus import (
    RouteV2PopulationMapper,
    _build_row,
    _validate_projection,
    latent_training_corpus_implementation_sha256,
)
from memory_condense.eval.diffuse_longmemeval_route_v2 import (
    EPISODE_PRIMARY_ANALYSIS_QUERY_V2_FORMAT,
    EpisodePrimaryRouteReceiptV2,
)
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalAnalysisQueryReceipt,
)
from memory_condense.eval.diffuse_longmemeval import LongMemEvalDiffuseQueryReceipt
from memory_condense.eval.diffuse_longmemeval_inputs import LegacyDiffuseInputReceipt
from memory_condense.eval.benchmark import build_qa_prompt
from memory_condense.search.fusion.models import FusionCaps
from memory_condense.search.fusion.planner import (
    _atom_refs,
    _authoritative_hyperedges,
)
from memory_condense.search.fusion.resident_models import resident_values_sha256
from memory_condense.search.fusion.training_targets import (
    AtomPositionPairTarget,
    DirectCoBundleNeighborhood,
    LatentRouterStructuralTargetReceipt,
    LatentRouterStructuralTargets,
    build_latent_router_structural_targets,
)


_ROOT_KEYS = {
    "format",
    "population_projection_sha256",
    "implementation_sha256",
    "treatment_file_sha256",
    "sanitized_projection_sha256",
    "dataset_sha256",
    "split_manifest_sha256",
    "analysis_ordered_question_ids_sha256",
    "fit_partition_sha256",
    "validation_partition_sha256",
    "excluded_confirmation_count",
    "excluded_confirmation_ordered_question_ids_sha256",
    "inventory",
    "inventory_sha256",
    "population_status",
    "episodic_route",
    "closure_routing_scope",
    "scorer_labels_present",
    "evaluator_label_schema_present",
    "tensor_or_embedding_payload_present",
    "source_treatment_exact_type_verified",
    "production_authorized",
    "d1_eligible",
    "corpus_sha256",
}
_PARTITION_KEYS = {
    "format",
    "partition",
    "start_ordinal",
    "row_count",
    "ordered_question_ids_sha256",
    "row_relative_paths",
    "row_sha256s",
    "production_authorized",
    "d1_eligible",
    "partition_sha256",
}
_ROW_KEYS = {
    "format",
    "ordinal",
    "partition",
    "partition_ordinal",
    "question_id",
    "question_id_sha256",
    "question_probe_sha256",
    "retrieval_query_sha256",
    "prompt_question_sha256",
    "route_record_sha256",
    "route_evidence",
    "payload_relative_path",
    "payload_sha256",
    "payload_bytes",
    "packet_receipt_sha256",
    "closure_plan_sha256",
    "ordered_atom_refs_sha256",
    "authoritative_hyperedges_sha256",
    "structural_target",
    "row_sha256",
}
_ROUTE_EVIDENCE_KEYS = {
    "analysis_arm_v2_body",
    "analysis_base_arm_body",
    "episode_policy_body",
    "closure_policy_body",
    "compilation_receipt_body",
    "compilation_snapshot_body",
    "representative_policy_body",
    "anchor_projection_body",
    "inner_analysis_query_receipt_body",
    "inner_diffuse_query_receipt_body",
    "legacy_input_receipt_body",
    "source_scope_body",
    "direct_expansion_body",
    "representative_expansion_body",
    "route_receipt",
    "evidence_sha256",
}
_ANALYSIS_BASE_ARM_KEYS = {
    "arm_id", "compilation", "episode_policy_sha256",
    "closure_policy_sha256", "max_context_tokens",
    "responder_output_token_reserve", "require_owned_representative_runtime",
}
_COMPILATION_POLICY_KEYS = {
    "boundary_mode", "min_episode_size", "max_episode_size", "fixed_interval",
    "surprise_window", "surprise_gamma", "surprise_min_history",
    "refinement_window", "refinement_max_nodes", "refinement_max_degree",
    "lexical_weight", "embedding_weight", "representative_limit",
}
_EPISODE_POLICY_KEYS = {
    "artifact_id", "max_anchor_episodes", "previous_episodes", "next_episodes",
    "max_episode_seeds", "max_direct_fallbacks", "neighbor_decay",
}
_CLOSURE_POLICY_KEYS = {
    "max_hops", "max_units", "max_relations", "max_degree",
    "max_episode_neighbors", "max_frontier", "max_bundles", "beam_width",
    "min_relation_confidence",
}
_COMPILATION_RECEIPT_KEYS = {
    "format", "artifact", "compilation_policy_sha256", "policy_sha256",
    "source_receipts", "episode_coverage_receipt_sha256",
    "discourse_coverage_receipt_sha256", "final_snapshot_sha256",
    "persisted_request_token_state_bytes", "receipt_sha256",
}
_SNAPSHOT_KEYS = {
    "max_turn_ordinal", "chunk_count", "graph_revision", "schema_version",
    "artifact_ids", "source_revision", "graph_content_revision",
    "source_content_sha256", "graph_content_sha256", "snapshot_sha256",
}
_REPRESENTATIVE_POLICY_KEYS = {
    "artifact_id", "max_input_sources", "max_source_groups",
    "max_episodes_per_source", "max_total_episodes",
    "max_representatives_per_episode", "group_size", "beam_per_group", "top_k",
    "representative_tokens", "query_tokens", "score_mode",
}
_ANCHOR_PROJECTION_KEYS = {"legacy", "diffuse"}
_ARM_V2_KEYS = {
    "format", "base_arm_sha256", "episodic_route", "closure_routing_scope",
    "arm_sha256",
}
_ANALYSIS_RECEIPT_KEYS = {
    "corpus_sha256", "question_probe_sha256", "analysis_arm_sha256",
    "matched_controls_sha256", "evaluation_policy_sha256",
    "legacy_input_provider_identity_sha256", "representative_linker_identity_sha256",
    "representative_policy_factory_identity_sha256", "representative_policy_sha256",
    "representative_policy_controls_sha256", "compilation_receipt_sha256",
    "legacy_input_receipt_sha256", "diffuse_query_receipt_sha256", "artifact_id",
    "snapshot_sha256", "format", "receipt_sha256",
}
_DIFFUSE_RECEIPT_KEYS = {
    "artifact_id", "snapshot_sha256", "anchor_sequence_sha256",
    "input_anchor_chunk_ids", "episode_policy_sha256", "expansion_receipt_sha256",
    "representative_receipt_sha256", "representative_scope_exhaustive",
    "representative_runtime_binding_certified",
    "representative_returned_plan_transformer_state_bytes",
    "combined_expansion_sha256", "representative_seed_episode_ids",
    "truncated_episode_ids", "truncated_direct_chunk_ids", "expansion_exhaustive",
    "query_program_sha256", "retrieval_query_sha256", "prompt_question_sha256",
    "closure_policy_sha256", "closure_plan_sha256", "closure_stopping_reason",
    "closure_complete_claimed", "scope_witness_sha256s", "closure_scope_exhaustive",
    "packet_receipt_sha256", "context_sha256", "evidence_coordinates_sha256",
    "prompt_messages_sha256", "prompt_token_proxy", "max_input_prompt_token_proxy",
    "responder_output_token_reserve", "prompt_workspace_token_proxy",
    "max_prompt_workspace_token_proxy", "packet_retained_request_token_state_bytes",
    "store_retained_request_token_state_bytes", "format", "receipt_sha256",
}
_LEGACY_RECEIPT_KEYS = {
    "artifact_id", "query_sha256", "retrieval_policy_sha256",
    "anchor_sequence_sha256", "source_candidate_sequence_sha256",
    "source_candidate_scope_receipt_sha256", "anchor_chunk_ids",
    "source_candidate_ids", "format", "receipt_sha256",
}
_ROUTE_RECEIPT_KEYS = {
    "format", "analysis_arm_v2_sha256", "route_v2_implementation_sha256",
    "inner_analysis_query_receipt_sha256", "inner_diffuse_query_receipt_sha256",
    "legacy_input_receipt_sha256", "artifact_id", "snapshot_sha256",
    "retrieval_query_sha256", "source_candidate_scope_receipt_sha256",
    "source_scope_exhaustive", "episodic_route", "closure_routing_scope",
    "direct_expansion_receipt_sha256", "direct_seed_count",
    "direct_fallback_count", "direct_truncated_episode_count",
    "direct_truncated_chunk_count", "representative_expansion_receipt_sha256",
    "representative_linker_identity_sha256", "representative_policy_sha256",
    "representative_runtime_binding_certified",
    "representative_returned_plan_transformer_state_bytes",
    "representative_scope_exhaustive", "representative_seed_count",
    "representative_seed_projection_sha256", "closure_seed_projection_sha256",
    "closure_direct_chunk_count", "combined_expansion_sha256",
    "expansion_exhaustive", "closure_policy_sha256", "closure_max_frontier",
    "closure_plan_sha256", "closure_routing_scope_witness_sha256",
    "closure_routing_scope_witness_count", "episode_expansion_witness_sha256",
    "episode_expansion_witness_count", "artifact_global_routes_admitted",
    "artifact_unit_scan_witness_count", "closure_scope_exhaustive",
    "query_program_sha256", "packet_receipt_sha256", "context_sha256",
    "prompt_messages_sha256", "packet_retained_request_token_state_bytes",
    "store_retained_request_token_state_bytes", "receipt_sha256",
}
_SOURCE_SCOPE_KEYS = {
    "artifact_id", "snapshot_sha256", "source_revision", "source_content_sha256",
    "query_sha256", "router_policy_sha256", "universe_source_ids", "candidates",
    "truncated_source_ids", "universe_enumerated", "receipt_sha256",
}
_DIRECT_KEYS = {
    "policy_sha256", "seeds", "direct_fallbacks", "truncated_episode_ids",
    "truncated_direct_chunk_ids", "receipt_sha256",
}
_REPRESENTATIVE_KEYS = {
    "artifact_id", "policy_sha256", "query_sha256", "query_input_sha256",
    "linker_identity_sha256", "runtime_binding_certified",
    "source_scope_receipt_sha256", "source_universe_exhaustive", "source_scans",
    "candidate_witnesses", "seeds", "truncated_source_ids",
    "truncated_episode_ids", "unavailable_episode_ids", "passes",
    "max_workspace_candidates", "max_workspace_tokens", "total_candidate_inspections",
    "returned_plan_transformer_state_bytes", "receipt_sha256",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _loads(payload: bytes, label: str, *, limit: int) -> dict[str, Any]:
    if type(payload) is not bytes or len(payload) > limit:
        raise LatentTrainingCorpusError(f"{label} exceeds its byte cap")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LatentTrainingCorpusError(f"cannot parse {label}") from exc
    if type(value) is not dict:
        raise LatentTrainingCorpusError(f"{label} must be a JSON object")
    if canonical_json(value).encode("utf-8") != payload:
        raise LatentTrainingCorpusError(f"{label} is not canonical UTF-8 JSON")
    return value


def _body_seal(body: Mapping[str, Any], field: str, label: str) -> None:
    current = _sha(body[field], f"{label}.{field}")
    unsigned = {key: _plain(value) for key, value in body.items() if key != field}
    if identity_sha256(unsigned) != current:
        raise LatentTrainingCorpusError(f"{label} self-hash changed")


def _decode_pair(value: Any, label: str) -> AtomPositionPairTarget:
    row = _mapping(
        value,
        {"left_position", "right_position", "direct_co_bundle_target", "pair_sha256"},
        label,
    )
    return AtomPositionPairTarget(
        left_position=row["left_position"],
        right_position=row["right_position"],
        direct_co_bundle_target=row["direct_co_bundle_target"],
        pair_sha256=row["pair_sha256"],
    )


def _decode_neighborhood(value: Any, label: str) -> DirectCoBundleNeighborhood:
    row = _mapping(
        value,
        {"atom_position", "member_positions", "neighborhood_sha256"},
        label,
    )
    members = tuple(_integer(item, f"{label}.member") for item in _list(row["member_positions"], label))
    return DirectCoBundleNeighborhood(
        atom_position=row["atom_position"],
        member_positions=members,
        neighborhood_sha256=row["neighborhood_sha256"],
    )


def _decode_target(value: Any, label: str) -> LatentRouterStructuralTargetReceipt:
    row = _mapping(
        value,
        {
            "packet_receipt_sha256", "closure_plan_sha256", "fusion_caps_sha256",
            "ordered_atom_refs_sha256", "authoritative_hyperedges_sha256",
            "structural_targets", "target_receipt_sha256",
        },
        label,
    )
    nested = _mapping(
        row["structural_targets"],
        {
            "atom_count", "positive_pairs", "negative_pairs", "neighborhoods",
            "positive_pair_count", "negative_pair_count",
            "positive_pair_sequence_sha256", "negative_pair_sequence_sha256",
            "target_sha256",
        },
        f"{label}.structural_targets",
    )
    positives = tuple(
        _decode_pair(item, f"{label}.positive_pairs[{index}]")
        for index, item in enumerate(_list(nested["positive_pairs"], label))
    )
    negatives = tuple(
        _decode_pair(item, f"{label}.negative_pairs[{index}]")
        for index, item in enumerate(_list(nested["negative_pairs"], label))
    )
    neighborhoods = tuple(
        _decode_neighborhood(item, f"{label}.neighborhoods[{index}]")
        for index, item in enumerate(_list(nested["neighborhoods"], label))
    )
    targets = LatentRouterStructuralTargets(
        atom_count=nested["atom_count"],
        positive_pairs=positives,
        negative_pairs=negatives,
        neighborhoods=neighborhoods,
        positive_pair_count=nested["positive_pair_count"],
        negative_pair_count=nested["negative_pair_count"],
        positive_pair_sequence_sha256=nested["positive_pair_sequence_sha256"],
        negative_pair_sequence_sha256=nested["negative_pair_sequence_sha256"],
        target_sha256=nested["target_sha256"],
    )
    return LatentRouterStructuralTargetReceipt(
        packet_receipt_sha256=row["packet_receipt_sha256"],
        closure_plan_sha256=row["closure_plan_sha256"],
        fusion_caps_sha256=row["fusion_caps_sha256"],
        ordered_atom_refs_sha256=row["ordered_atom_refs_sha256"],
        authoritative_hyperedges_sha256=row["authoritative_hyperedges_sha256"],
        structural_targets=targets,
        target_receipt_sha256=row["target_receipt_sha256"],
    )


def _decode_route_evidence(value: Any) -> LatentTrainingRouteEvidence:
    row = _mapping(value, _ROUTE_EVIDENCE_KEYS, "route_evidence")
    route = _mapping(row["route_receipt"], _ROUTE_RECEIPT_KEYS, "route_receipt")
    return LatentTrainingRouteEvidence(
        analysis_arm_v2_body=_mapping(
            row["analysis_arm_v2_body"], _ARM_V2_KEYS, "analysis_arm_v2_body"
        ),
        analysis_base_arm_body=_mapping(
            row["analysis_base_arm_body"],
            _ANALYSIS_BASE_ARM_KEYS,
            "analysis_base_arm_body",
        ),
        episode_policy_body=_mapping(
            row["episode_policy_body"], _EPISODE_POLICY_KEYS, "episode_policy_body"
        ),
        closure_policy_body=_mapping(
            row["closure_policy_body"], _CLOSURE_POLICY_KEYS, "closure_policy_body"
        ),
        compilation_receipt_body=_mapping(
            row["compilation_receipt_body"],
            _COMPILATION_RECEIPT_KEYS,
            "compilation_receipt_body",
        ),
        compilation_snapshot_body=_mapping(
            row["compilation_snapshot_body"],
            _SNAPSHOT_KEYS,
            "compilation_snapshot_body",
        ),
        representative_policy_body=_mapping(
            row["representative_policy_body"],
            _REPRESENTATIVE_POLICY_KEYS,
            "representative_policy_body",
        ),
        anchor_projection_body=_mapping(
            row["anchor_projection_body"],
            _ANCHOR_PROJECTION_KEYS,
            "anchor_projection_body",
        ),
        inner_analysis_query_receipt_body=_mapping(
            row["inner_analysis_query_receipt_body"],
            _ANALYSIS_RECEIPT_KEYS,
            "inner_analysis_query_receipt_body",
        ),
        inner_diffuse_query_receipt_body=_mapping(
            row["inner_diffuse_query_receipt_body"],
            _DIFFUSE_RECEIPT_KEYS,
            "inner_diffuse_query_receipt_body",
        ),
        legacy_input_receipt_body=_mapping(
            row["legacy_input_receipt_body"],
            _LEGACY_RECEIPT_KEYS,
            "legacy_input_receipt_body",
        ),
        source_scope_body=_mapping(
            row["source_scope_body"], _SOURCE_SCOPE_KEYS, "source_scope_body"
        ),
        direct_expansion_body=_mapping(
            row["direct_expansion_body"], _DIRECT_KEYS, "direct_expansion_body"
        ),
        representative_expansion_body=_mapping(
            row["representative_expansion_body"],
            _REPRESENTATIVE_KEYS,
            "representative_expansion_body",
        ),
        route_receipt=EpisodePrimaryRouteReceiptV2(**route),
        evidence_sha256=row["evidence_sha256"],
    )


def _decode_row(value: Any) -> LatentTrainingCorpusRowManifest:
    row = _mapping(value, _ROW_KEYS, "row manifest")
    return LatentTrainingCorpusRowManifest(
        ordinal=row["ordinal"],
        partition=row["partition"],
        partition_ordinal=row["partition_ordinal"],
        question_id=row["question_id"],
        question_id_sha256=row["question_id_sha256"],
        question_probe_sha256=row["question_probe_sha256"],
        retrieval_query_sha256=row["retrieval_query_sha256"],
        prompt_question_sha256=row["prompt_question_sha256"],
        route_record_sha256=row["route_record_sha256"],
        route_evidence=_decode_route_evidence(row["route_evidence"]),
        payload_relative_path=row["payload_relative_path"],
        payload_sha256=row["payload_sha256"],
        payload_bytes=row["payload_bytes"],
        packet_receipt_sha256=row["packet_receipt_sha256"],
        closure_plan_sha256=row["closure_plan_sha256"],
        ordered_atom_refs_sha256=row["ordered_atom_refs_sha256"],
        authoritative_hyperedges_sha256=row["authoritative_hyperedges_sha256"],
        structural_target=_decode_target(row["structural_target"], "structural_target"),
        format=row["format"],
        row_sha256=row["row_sha256"],
    )


def _decode_partition(value: Any) -> LatentTrainingCorpusPartitionManifest:
    row = _mapping(value, _PARTITION_KEYS, "partition manifest")
    return LatentTrainingCorpusPartitionManifest(
        partition=row["partition"],
        start_ordinal=row["start_ordinal"],
        row_count=row["row_count"],
        ordered_question_ids_sha256=row["ordered_question_ids_sha256"],
        row_relative_paths=tuple(_list(row["row_relative_paths"], "row paths")),
        row_sha256s=tuple(_list(row["row_sha256s"], "row hashes")),
        production_authorized=row["production_authorized"],
        d1_eligible=row["d1_eligible"],
        format=row["format"],
        partition_sha256=row["partition_sha256"],
    )


def _decode_manifest(value: Any) -> LatentTrainingCorpusManifest:
    row = _mapping(value, _ROOT_KEYS, "root manifest")
    inventory = tuple(
        LatentTrainingFileIdentity(
            relative_path=_mapping(
                item, {"relative_path", "sha256", "bytes"}, "inventory row"
            )["relative_path"],
            sha256=item["sha256"],
            bytes=item["bytes"],
        )
        for item in _list(row["inventory"], "inventory")
    )
    return LatentTrainingCorpusManifest(
        population_projection_sha256=row["population_projection_sha256"],
        implementation_sha256=row["implementation_sha256"],
        treatment_file_sha256=row["treatment_file_sha256"],
        sanitized_projection_sha256=row["sanitized_projection_sha256"],
        dataset_sha256=row["dataset_sha256"],
        split_manifest_sha256=row["split_manifest_sha256"],
        analysis_ordered_question_ids_sha256=row["analysis_ordered_question_ids_sha256"],
        fit_partition_sha256=row["fit_partition_sha256"],
        validation_partition_sha256=row["validation_partition_sha256"],
        excluded_confirmation_count=row["excluded_confirmation_count"],
        excluded_confirmation_ordered_question_ids_sha256=row[
            "excluded_confirmation_ordered_question_ids_sha256"
        ],
        inventory=inventory,
        inventory_sha256=row["inventory_sha256"],
        population_status=row["population_status"],
        episodic_route=row["episodic_route"],
        closure_routing_scope=row["closure_routing_scope"],
        scorer_labels_present=row["scorer_labels_present"],
        evaluator_label_schema_present=row["evaluator_label_schema_present"],
        tensor_or_embedding_payload_present=row["tensor_or_embedding_payload_present"],
        source_treatment_exact_type_verified=row["source_treatment_exact_type_verified"],
        production_authorized=row["production_authorized"],
        d1_eligible=row["d1_eligible"],
        format=row["format"],
        corpus_sha256=row["corpus_sha256"],
    )


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be an exact finite float")
    return value


def _string_list(value: object, label: str) -> tuple[str, ...]:
    rows = _list(value, label)
    return tuple(_text(item, f"{label} item") for item in rows)


def _seed_rows(value: object, label: str) -> tuple[dict[str, Any], ...]:
    result = []
    for index, item in enumerate(_list(value, label)):
        row = _mapping(
            item,
            {"episode_id", "anchor_chunk_id", "score", "route", "path"},
            f"{label}[{index}]",
        )
        _text(row["episode_id"], f"{label}.episode_id")
        _text(row["anchor_chunk_id"], f"{label}.anchor_chunk_id")
        _finite_float(row["score"], f"{label}.score")
        _text(row["route"], f"{label}.route")
        _string_list(row["path"], f"{label}.path")
        result.append(row)
    return tuple(result)


def _validate_source_scope(body: Mapping[str, Any], route: Any) -> bool:
    _body_seal(body, "receipt_sha256", "source scope")
    universe = _string_list(body["universe_source_ids"], "source universe")
    truncated = _string_list(body["truncated_source_ids"], "truncated sources")
    candidates = []
    for index, value in enumerate(_list(body["candidates"], "source candidates")):
        row = _mapping(value, {"source_id", "score", "route"}, f"source candidate {index}")
        _text(row["source_id"], "source candidate ID")
        _finite_float(row["score"], "source candidate score")
        _text(row["route"], "source candidate route")
        candidates.append(row["source_id"])
    if len(set(universe)) != len(universe) or len(set(candidates)) != len(candidates):
        raise LatentTrainingCorpusError("source scope repeats an opaque source ID")
    if set(candidates) & set(truncated) or not set((*candidates, *truncated)) <= set(universe):
        raise LatentTrainingCorpusError("source scope membership is inconsistent")
    exhaustive_flag = body["universe_enumerated"]
    if type(exhaustive_flag) is not bool:
        raise TypeError("source universe flag must be an exact boolean")
    if exhaustive_flag and set((*candidates, *truncated)) != set(universe):
        raise LatentTrainingCorpusError("exhaustive source scope is incomplete")
    selected_exhaustive = exhaustive_flag and not truncated
    if (
        body["receipt_sha256"] != route.source_candidate_scope_receipt_sha256
        or body["artifact_id"] != route.artifact_id
        or body["snapshot_sha256"] != route.snapshot_sha256
        or body["query_sha256"] != route.retrieval_query_sha256
        or selected_exhaustive is not route.source_scope_exhaustive
    ):
        raise LatentTrainingCorpusError("source scope does not join the route receipt")
    return selected_exhaustive


def _decode_upstream_receipts(evidence: Any) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    arm = evidence.analysis_arm_v2_body
    _body_seal(arm, "arm_sha256", "analysis arm v2")
    if (
        arm["episodic_route"] != "episode_primary"
        or arm["closure_routing_scope"] != "seeded_graph"
    ):
        raise LatentTrainingCorpusError("persisted analysis arm is not episode_primary")
    analysis_body = dict(_plain(evidence.inner_analysis_query_receipt_body))
    analysis = DiffuseLongMemEvalAnalysisQueryReceipt(**analysis_body)
    diffuse_body = dict(_plain(evidence.inner_diffuse_query_receipt_body))
    for name in (
        "input_anchor_chunk_ids", "representative_seed_episode_ids",
        "truncated_episode_ids", "truncated_direct_chunk_ids",
        "scope_witness_sha256s",
    ):
        diffuse_body[name] = tuple(
            _string_list(diffuse_body[name], f"diffuse receipt {name}")
        )
    diffuse = LongMemEvalDiffuseQueryReceipt(**diffuse_body)
    legacy_body = dict(_plain(evidence.legacy_input_receipt_body))
    legacy_body["anchor_chunk_ids"] = tuple(
        _string_list(legacy_body["anchor_chunk_ids"], "legacy anchor IDs")
    )
    legacy_body["source_candidate_ids"] = tuple(
        _string_list(legacy_body["source_candidate_ids"], "legacy source IDs")
    )
    legacy = LegacyDiffuseInputReceipt(**legacy_body)
    return analysis, diffuse, legacy, arm


def _validate_intermediate_route(
    row: LatentTrainingCorpusRowManifest,
    payload: Any,
) -> None:
    evidence = row.route_evidence
    route = evidence.route_receipt
    analysis, diffuse, legacy, arm = _decode_upstream_receipts(evidence)
    source = evidence.source_scope_body
    direct = evidence.direct_expansion_body
    representative = evidence.representative_expansion_body
    _validate_source_scope(source, route)
    _body_seal(direct, "receipt_sha256", "direct expansion")
    _body_seal(representative, "receipt_sha256", "representative expansion")
    direct_seeds = _seed_rows(direct["seeds"], "direct seeds")
    direct_fallbacks = _list(direct["direct_fallbacks"], "direct fallbacks")
    direct_truncated = _string_list(direct["truncated_episode_ids"], "direct truncation")
    direct_chunks = _string_list(
        direct["truncated_direct_chunk_ids"], "direct chunk truncation"
    )
    if direct_seeds or direct_fallbacks or direct_truncated or direct_chunks:
        raise LatentTrainingCorpusError("episode_primary persisted a non-empty direct route")
    if (
        direct["receipt_sha256"] != route.direct_expansion_receipt_sha256
        or route.direct_seed_count != 0
        or route.direct_fallback_count != 0
        or route.direct_truncated_episode_count != 0
        or route.direct_truncated_chunk_count != 0
    ):
        raise LatentTrainingCorpusError("empty direct expansion does not join the route")

    rep_seeds = _seed_rows(representative["seeds"], "representative seeds")
    if not rep_seeds:
        raise LatentTrainingCorpusError("episode_primary representative route is empty")
    scans = []
    for index, item in enumerate(_list(representative["source_scans"], "source scans")):
        scan = _mapping(
            item,
            {"source_id", "requested_limit", "observed_count", "candidate_count", "exhaustive", "status"},
            f"source scan {index}",
        )
        _text(scan["source_id"], "source scan ID")
        for name in ("requested_limit", "observed_count", "candidate_count"):
            _integer(scan[name], f"source scan {name}")
        if type(scan["exhaustive"]) is not bool:
            raise TypeError("source scan exhaustive must be exact bool")
        scans.append(scan)
    witnessed = []
    for index, item in enumerate(
        _list(representative["candidate_witnesses"], "candidate witnesses")
    ):
        witness = _mapping(
            item,
            {
                "episode_id", "source_id", "anchor_chunk_id",
                "representative_chunk_ids", "representative_identity_sha256s",
                "candidate_text_sha256", "source_score", "source_route",
            },
            f"candidate witness {index}",
        )
        witnessed.append(_text(witness["episode_id"], "candidate episode ID"))
        _finite_float(witness["source_score"], "candidate source score")
        for digest in _string_list(
            witness["representative_identity_sha256s"], "representative identities"
        ):
            _sha(digest, "representative identity")
        _sha(witness["candidate_text_sha256"], "candidate text SHA-256")
    if any(seed["episode_id"] not in set(witnessed) for seed in rep_seeds):
        raise LatentTrainingCorpusError("representative seed lacks a candidate witness")
    for name in (
        "passes", "max_workspace_candidates", "max_workspace_tokens",
        "total_candidate_inspections", "returned_plan_transformer_state_bytes",
    ):
        _integer(representative[name], f"representative {name}")
    if (
        representative["runtime_binding_certified"] is not True
        or representative["returned_plan_transformer_state_bytes"] != 0
        or representative["source_scope_receipt_sha256"] != source["receipt_sha256"]
        or representative["receipt_sha256"] != route.representative_expansion_receipt_sha256
        or representative["artifact_id"] != route.artifact_id
        or representative["policy_sha256"] != route.representative_policy_sha256
        or representative["linker_identity_sha256"]
        != route.representative_linker_identity_sha256
    ):
        raise LatentTrainingCorpusError("representative body does not join route receipt")
    rep_exhaustive = bool(
        representative["source_universe_exhaustive"] is True
        and not _list(representative["truncated_source_ids"], "truncated sources")
        and not _list(representative["truncated_episode_ids"], "truncated episodes")
        and not _list(representative["unavailable_episode_ids"], "unavailable episodes")
        and all(scan["exhaustive"] is True for scan in scans)
    )
    plan_seeds = tuple(_plain(seed.identity_payload()) for seed in payload.plan.seeds)
    if tuple(_plain(seed) for seed in rep_seeds) != plan_seeds:
        raise LatentTrainingCorpusError("closure seeds differ from representative seeds")
    seed_sha = identity_sha256([_plain(seed) for seed in rep_seeds])
    combined = identity_sha256(
        {
            "episodic_route": "episode_primary",
            "direct_expansion_receipt_sha256": direct["receipt_sha256"],
            "representative_expansion_receipt_sha256": representative["receipt_sha256"],
            "seeds": [_plain(seed) for seed in rep_seeds],
            "direct_chunk_ids": [],
        }
    )
    if (
        representative["query_sha256"] != route.retrieval_query_sha256
        or representative["source_universe_exhaustive"]
        is not source["universe_enumerated"]
        or route.representative_scope_exhaustive is not rep_exhaustive
        or route.expansion_exhaustive is not rep_exhaustive
        or route.representative_seed_count != len(rep_seeds)
        or route.representative_seed_projection_sha256 != seed_sha
        or route.closure_seed_projection_sha256 != seed_sha
        or route.combined_expansion_sha256 != combined
        or payload.plan.expansion_receipt_sha256 != combined
        or payload.plan.direct_chunk_ids
    ):
        raise LatentTrainingCorpusError("episode-primary seed route cannot be reconstructed")

    routing = tuple(w for w in payload.plan.scope_witnesses if w.kind == "closure_routing_scope")
    expansion = tuple(w for w in payload.plan.scope_witnesses if w.kind == "episode_expansion")
    expected_routing = ClosureScopeWitness(
        kind="closure_routing_scope",
        subject_id="seeded_graph",
        requested_limit=payload.plan.policy.max_frontier * 2,
        returned_count=len(rep_seeds),
        exhaustive=False,
        detail={"artifact_global_routes_admitted": False, "seed_count": len(rep_seeds), "direct_chunk_count": 0},
    )
    expected_expansion = ClosureScopeWitness(
        kind="episode_expansion",
        subject_id=combined,
        requested_limit=None,
        returned_count=len(rep_seeds),
        exhaustive=rep_exhaustive,
        detail={"seed_count": len(rep_seeds), "direct_chunk_count": 0, "receipt_sha256": combined, "exhaustiveness_attested": True},
    )
    if (
        len(routing) != 1 or routing[0].witness_sha256 != expected_routing.witness_sha256
        or len(expansion) != 1 or expansion[0].witness_sha256 != expected_expansion.witness_sha256
        or any(w.kind == "artifact_unit_scan" for w in payload.plan.scope_witnesses)
        or route.closure_routing_scope_witness_sha256 != expected_routing.witness_sha256
        or route.episode_expansion_witness_sha256 != expected_expansion.witness_sha256
        or route.closure_routing_scope_witness_count != 1
        or route.episode_expansion_witness_count != 1
    ):
        raise LatentTrainingCorpusError("route witness bodies cannot be reconstructed")

    probe_sha = identity_sha256(
        {
            "question_id": payload.question_id,
            "retrieval_query": payload.retrieval_query,
            "prompt_question": payload.prompt_question,
        }
    )
    query_sha = identity_sha256({"query": payload.retrieval_query})
    prompt_question_sha = identity_sha256({"prompt_question": payload.prompt_question})
    prompt_messages_sha = identity_sha256(
        build_qa_prompt(payload.prompt_question, [payload.packet.context])
    )
    if (
        arm["arm_sha256"] != route.analysis_arm_v2_sha256
        or analysis.receipt_sha256 != route.inner_analysis_query_receipt_sha256
        or diffuse.receipt_sha256 != route.inner_diffuse_query_receipt_sha256
        or legacy.receipt_sha256 != route.legacy_input_receipt_sha256
        or analysis.analysis_arm_sha256 != arm["base_arm_sha256"]
        or analysis.question_probe_sha256 != row.question_probe_sha256
        or analysis.legacy_input_receipt_sha256 != legacy.receipt_sha256
        or analysis.diffuse_query_receipt_sha256 != diffuse.receipt_sha256
        or analysis.artifact_id != route.artifact_id
        or analysis.snapshot_sha256 != route.snapshot_sha256
        or analysis.representative_linker_identity_sha256
        != route.representative_linker_identity_sha256
        or analysis.representative_policy_sha256
        != route.representative_policy_sha256
        or legacy.query_sha256 != query_sha
        or legacy.source_candidate_scope_receipt_sha256 != source["receipt_sha256"]
        or legacy.artifact_id != route.artifact_id
        or diffuse.artifact_id != route.artifact_id
        or diffuse.snapshot_sha256 != route.snapshot_sha256
        or diffuse.retrieval_query_sha256 != query_sha
        or diffuse.prompt_question_sha256 != prompt_question_sha
        or diffuse.expansion_receipt_sha256 != direct["receipt_sha256"]
        or diffuse.representative_receipt_sha256 != representative["receipt_sha256"]
        or diffuse.combined_expansion_sha256 != combined
        or diffuse.closure_plan_sha256 != payload.plan.plan_sha256
        or diffuse.packet_receipt_sha256 != payload.packet.receipt.receipt_sha256
        or diffuse.prompt_messages_sha256 != prompt_messages_sha
    ):
        raise LatentTrainingCorpusError(
            "arm, analysis, diffuse, legacy, and route receipt joins disagree"
        )
    if (
        payload.question_id != row.question_id
        or probe_sha != row.question_probe_sha256
        or query_sha != row.retrieval_query_sha256
        or query_sha != route.retrieval_query_sha256
        or prompt_question_sha != row.prompt_question_sha256
        or prompt_messages_sha != route.prompt_messages_sha256
        or prompt_messages_sha != payload.packet.receipt.prompt_messages_sha256
        or payload.plan.plan_sha256 != row.closure_plan_sha256
        or payload.packet.receipt.receipt_sha256 != row.packet_receipt_sha256
        or payload.packet.receipt.plan_sha256 != payload.plan.plan_sha256
        or payload.plan.snapshot.snapshot_sha256 != route.snapshot_sha256
        or payload.plan.policy.policy_sha256 != route.closure_policy_sha256
        or payload.plan.query_program.program_sha256 != route.query_program_sha256
        or payload.packet.receipt.context_sha256 != route.context_sha256
    ):
        raise LatentTrainingCorpusError("query, plan, packet, or prompt join changed")
    expected_target = build_latent_router_structural_targets(
        payload.packet, payload.plan, caps=FusionCaps()
    )
    if _target_body(expected_target) != _target_body(row.structural_target):
        raise LatentTrainingCorpusError("persisted structural target cannot be reconstructed")
    refs_sha = resident_values_sha256(_ATOM_REF_KIND, _atom_refs(payload.packet))
    edges_sha = resident_values_sha256(
        _HYPEREDGE_KIND, _authoritative_hyperedges(payload.packet)
    )
    if refs_sha != row.ordered_atom_refs_sha256 or edges_sha != row.authoritative_hyperedges_sha256:
        raise LatentTrainingCorpusError("packet-order atom refs or hyperedges changed")


def _partition(
    role: Literal["fit", "validation"],
    start: int,
    rows: tuple[LatentTrainingCorpusRowManifest, ...],
) -> LatentTrainingCorpusPartitionManifest:
    return LatentTrainingCorpusPartitionManifest(
        partition=role,
        start_ordinal=start,
        row_count=len(rows),
        ordered_question_ids_sha256=_ids_sha256(tuple(item.question_id for item in rows)),
        row_relative_paths=tuple(f"rows/{item.ordinal:06d}.json" for item in rows),
        row_sha256s=tuple(item.row_sha256 for item in rows),
    )


def publish_structural_corpus(
    population: AnalysisPopulationProjection,
    destination: str | Path,
    *,
    row_mapper: RouteV2PopulationMapper,
    expected: LatentTrainingPopulationExpectation,
    population_status: Literal["locked_projection", "synthetic_projection"],
) -> LatentTrainingCorpusPublicationReceipt:
    rows_to_map = _validate_projection(population, expected)
    if not callable(row_mapper):
        raise TypeError("row_mapper must be callable")
    if type(population_status) is not str or population_status not in {
        "locked_projection", "synthetic_projection"
    }:
        raise TypeError("population_status has the wrong exact literal")
    implementation = latent_training_corpus_implementation_sha256()
    route_implementation = live_route_v2_implementation_sha256()
    target = Path(os.path.abspath(os.fspath(destination)))
    if not target.name or target.name in {".", ".."}:
        raise LatentTrainingCorpusError("destination requires a bounded child name")
    parent = target.parent
    require_plain_parent(parent)
    if os.path.lexists(target):
        raise FileExistsError(target)
    staging = owned_staging(parent, target.name)
    try:
        manifests: list[LatentTrainingCorpusRowManifest] = []
        inventory_by_path: dict[str, LatentTrainingFileIdentity] = {}
        for population_row in rows_to_map:
            manifest, payload = _build_row(population_row, row_mapper(population_row))
            payload_identity = write_new(staging, manifest.payload_relative_path, payload)
            inventory_by_path[payload_identity.relative_path] = payload_identity
            row_path = f"rows/{manifest.ordinal:06d}.json"
            row_identity = write_new(staging, row_path, _canonical_bytes(manifest.identity_payload()))
            inventory_by_path[row_path] = row_identity
            manifests.append(manifest)
        fit_rows = tuple(manifests[: expected.fit_count])
        validation_rows = tuple(manifests[expected.fit_count :])
        fit = _partition("fit", 0, fit_rows)
        validation = _partition("validation", expected.fit_count, validation_rows)
        for name, value in (("fit", fit), ("validation", validation)):
            relative = f"partitions/{name}.json"
            inventory_by_path[relative] = write_new(
                staging, relative, _canonical_bytes(value.identity_payload())
            )
        inventory = tuple(
            inventory_by_path[name] for name in sorted(inventory_by_path)
        )
        inventory_sha = identity_sha256([item.identity_payload() for item in inventory])
        root = LatentTrainingCorpusManifest(
            population_projection_sha256=population.projection_sha256,
            implementation_sha256=implementation,
            treatment_file_sha256=population.treatment_file_sha256,
            sanitized_projection_sha256=population.sanitized_projection_sha256,
            dataset_sha256=population.dataset_sha256,
            split_manifest_sha256=population.split_manifest_sha256,
            analysis_ordered_question_ids_sha256=population.ordered_question_ids_sha256,
            fit_partition_sha256=fit.partition_sha256,
            validation_partition_sha256=validation.partition_sha256,
            excluded_confirmation_count=population.excluded_confirmation_count,
            excluded_confirmation_ordered_question_ids_sha256=(
                population.excluded_confirmation_ordered_question_ids_sha256
            ),
            inventory=inventory,
            inventory_sha256=inventory_sha,
            population_status=population_status,
        )
        root_bytes = _canonical_bytes(root.identity_payload())
        write_new(staging, ROOT_MANIFEST_NAME, root_bytes)
        if latent_training_corpus_implementation_sha256() != implementation:
            raise RuntimeError("corpus implementation changed during construction")
        verify_structural_corpus(staging.path)
        if (
            latent_training_corpus_implementation_sha256() != implementation
            or live_route_v2_implementation_sha256() != route_implementation
        ):
            raise RuntimeError("corpus implementation changed before publication")
        published = publish_staging(staging, target)
        try:
            verified_target = verify_structural_corpus(target)
            if verified_target.manifest.corpus_sha256 != root.corpus_sha256:
                raise LatentTrainingCorpusError(
                    "published corpus differs from the constructed corpus"
                )
            if (
                latent_training_corpus_implementation_sha256() != implementation
                or live_route_v2_implementation_sha256() != route_implementation
            ):
                raise RuntimeError("corpus implementation changed during publication")
        except BaseException:
            remove_owned(published)
            raise
        return LatentTrainingCorpusPublicationReceipt(
            corpus_sha256=root.corpus_sha256,
            implementation_sha256=implementation,
            root_manifest_sha256=hashlib.sha256(root_bytes).hexdigest(),
            root_manifest_bytes=len(root_bytes),
            inventory_sha256=inventory_sha,
        )
    except BaseException:
        if os.path.lexists(staging.path):
            remove_owned(staging)
        raise


def _require_locked_manifest(
    root: LatentTrainingCorpusManifest,
    fit: LatentTrainingCorpusPartitionManifest,
    validation: LatentTrainingCorpusPartitionManifest,
) -> None:
    expected = {
        "dataset_sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "split_manifest_sha256": "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4",
        "treatment_file_sha256": "b4d1d34538fdabbd6127c339bff8167293d290eb732afc18a5d8963d12b15001",
        "sanitized_projection_sha256": "58a1982122d259e046ac5268de8fc3c2857a63d24c859e3bc13e4e6b9aa52ad8",
        "analysis_ordered_question_ids_sha256": "cf5e8648b71634e4e22be872881766e37e0dc24a2931d0c63365e075b2742046",
        "excluded_confirmation_ordered_question_ids_sha256": "6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102",
    }
    if any(getattr(root, name) != value for name, value in expected.items()) or (
        root.excluded_confirmation_count != 200
        or fit.row_count != 200
        or fit.ordered_question_ids_sha256
        != "533aa545efb8032f7b181f39264c6d10a49471bd460414f420e37dc840a19c55"
        or validation.row_count != 100
        or validation.ordered_question_ids_sha256
        != "7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1"
    ):
        raise LatentTrainingCorpusError("locked projection differs from frozen literals")


def _verify_snapshot(
    snapshot_tree: CorpusTreeSnapshot,
    implementation: str,
    route_implementation: str,
) -> VerifiedLatentTrainingFullCorpus:
    snapshots = snapshot_tree.files
    root_snapshot = snapshots.get(ROOT_MANIFEST_NAME)
    if root_snapshot is None:
        raise LatentTrainingCorpusError("corpus publication has no final manifest marker")
    root = _decode_manifest(
        _mapping(
            _loads(
                snapshot_tree.read(ROOT_MANIFEST_NAME),
                "root manifest",
                limit=MAX_METADATA_FILE_BYTES,
            ),
            _ROOT_KEYS,
            "root manifest",
        )
    )
    if root.implementation_sha256 != implementation:
        raise LatentTrainingCorpusError("corpus implementation identity changed")
    inventory = {item.relative_path: item for item in root.inventory}
    if set(snapshots) != {ROOT_MANIFEST_NAME, *inventory}:
        raise LatentTrainingCorpusError("root inventory has missing or extra files")
    for relative, expected_file in inventory.items():
        actual = snapshots[relative]
        if actual.sha256 != expected_file.sha256 or actual.size != expected_file.bytes:
            raise LatentTrainingCorpusError("inventory content hash or byte count changed")
    partition_values = {}
    for role in ("fit", "validation"):
        relative = f"partitions/{role}.json"
        snapshot = snapshots.get(relative)
        if snapshot is None:
            raise LatentTrainingCorpusError("corpus partition manifest is missing")
        partition_values[role] = _decode_partition(
            _mapping(
                _loads(
                    snapshot_tree.read(relative),
                    f"{role} partition",
                    limit=MAX_METADATA_FILE_BYTES,
                ),
                _PARTITION_KEYS,
                f"{role} partition",
            )
        )
    fit_manifest = partition_values["fit"]
    validation_manifest = partition_values["validation"]
    if (
        fit_manifest.partition != "fit"
        or fit_manifest.start_ordinal != 0
        or validation_manifest.partition != "validation"
        or validation_manifest.start_ordinal != fit_manifest.row_count
        or fit_manifest.partition_sha256 != root.fit_partition_sha256
        or validation_manifest.partition_sha256 != root.validation_partition_sha256
    ):
        raise LatentTrainingCorpusError("partition roles or root joins changed")
    if root.population_status == "locked_projection":
        _require_locked_manifest(root, fit_manifest, validation_manifest)

    decoded_by_role: dict[str, tuple[DecodedLatentTrainingCorpusRow, ...]] = {}
    referenced = {"partitions/fit.json", "partitions/validation.json"}
    all_ids: list[str] = []
    ordinal = 0
    for role, partition in (("fit", fit_manifest), ("validation", validation_manifest)):
        values = []
        ids = []
        for index, (row_path, row_sha) in enumerate(
            zip(partition.row_relative_paths, partition.row_sha256s, strict=True)
        ):
            expected_path = f"rows/{ordinal:06d}.json"
            if row_path != expected_path:
                raise LatentTrainingCorpusError("row path changed locked positional order")
            row_snapshot = snapshots.get(row_path)
            if row_snapshot is None:
                raise LatentTrainingCorpusError("partition row file is missing")
            row = _decode_row(
                _mapping(
                    _loads(
                        snapshot_tree.read(row_path),
                        "row manifest",
                        limit=MAX_METADATA_FILE_BYTES,
                    ),
                    _ROW_KEYS,
                    "row manifest",
                )
            )
            if (
                row.row_sha256 != row_sha
                or row.ordinal != ordinal
                or row.partition != role
                or row.partition_ordinal != index
            ):
                raise LatentTrainingCorpusError("row manifest differs from partition order")
            payload_snapshot = snapshots.get(row.payload_relative_path)
            if payload_snapshot is None or payload_snapshot.size != row.payload_bytes or (
                payload_snapshot.sha256 != row.payload_sha256
            ):
                raise LatentTrainingCorpusError("row payload content address changed")
            payload = decode_latent_training_payload(
                snapshot_tree.read(row.payload_relative_path)
            )
            validate_persisted_route(
                row,
                payload,
                expected_route_implementation_sha256=route_implementation,
            )
            values.append(DecodedLatentTrainingCorpusRow(row, payload))
            ids.append(row.question_id)
            all_ids.append(row.question_id)
            referenced.update({row_path, row.payload_relative_path})
            ordinal += 1
        if _ids_sha256(tuple(ids)) != partition.ordered_question_ids_sha256:
            raise LatentTrainingCorpusError("partition ordered question IDs changed")
        decoded_by_role[role] = tuple(values)
    if len(set(all_ids)) != len(all_ids):
        raise LatentTrainingCorpusError("corpus question IDs are not globally unique")
    if referenced != set(inventory) or _ids_sha256(tuple(all_ids)) != (
        root.analysis_ordered_question_ids_sha256
    ):
        raise LatentTrainingCorpusError("corpus file or analysis population order changed")
    projection_unsigned = {
        "format": "memory-condense-latent-training-analysis-population-projection-v1",
        "treatment_file_sha256": root.treatment_file_sha256,
        "sanitized_projection_sha256": root.sanitized_projection_sha256,
        "dataset_sha256": root.dataset_sha256,
        "split_manifest_sha256": root.split_manifest_sha256,
        "ordered_question_count": len(all_ids),
        "ordered_question_ids_sha256": root.analysis_ordered_question_ids_sha256,
        "excluded_confirmation_count": root.excluded_confirmation_count,
        "excluded_confirmation_ordered_question_ids_sha256": (
            root.excluded_confirmation_ordered_question_ids_sha256
        ),
        "source_treatment_exact_type_verified": False,
    }
    if identity_sha256(projection_unsigned) != root.population_projection_sha256:
        raise LatentTrainingCorpusError("population projection seal cannot be reconstructed")
    snapshot_tree.assert_unchanged()
    if latent_training_corpus_implementation_sha256() != implementation:
        raise RuntimeError("corpus implementation changed during verification")
    fit = VerifiedLatentTrainingFitCorpus(root, fit_manifest, decoded_by_role["fit"])
    validation = VerifiedLatentTrainingValidationCorpus(
        root, validation_manifest, decoded_by_role["validation"]
    )
    return VerifiedLatentTrainingFullCorpus(root, fit, validation)


def verify_structural_corpus(path: str | Path) -> VerifiedLatentTrainingFullCorpus:
    implementation = latent_training_corpus_implementation_sha256()
    route_implementation = live_route_v2_implementation_sha256()
    with CorpusTreeSnapshot(path) as snapshot_tree:
        result = _verify_snapshot(
            snapshot_tree, implementation, route_implementation
        )
    if (
        latent_training_corpus_implementation_sha256() != implementation
        or live_route_v2_implementation_sha256() != route_implementation
    ):
        raise RuntimeError("corpus implementation changed during verification")
    return result


__all__ = ["publish_structural_corpus", "verify_structural_corpus"]
