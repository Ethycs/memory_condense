"""Independent reconstruction of persisted route-v2 corpus evidence."""

from __future__ import annotations

import math
from dataclasses import fields, replace
from typing import Any, Mapping

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import truncate_to_tokens_lossless
from memory_condense.domain.discourse import (
    ClosurePolicy,
    ClosureScopeWitness,
    DiscourseArtifact,
    DiscourseSnapshot,
    EpisodeSeed,
)
from memory_condense.eval._diffuse_latent_training_corpus_models import (
    DecodedLatentTrainingCorpusRow,
    LatentTrainingCorpusError,
    LatentTrainingCorpusRowManifest,
    _ATOM_REF_KIND,
    _HYPEREDGE_KIND,
    _plain,
    _target_body,
)
from memory_condense.eval._diffuse_route_v2_validation import (
    assert_exact_value,
    require_exact_scalar,
    seed_projection_sha256,
    validate_closure_stopping_state,
    validate_compilation_receipt,
)
from memory_condense.eval._retrieval_qa_prompt import QA_SYSTEM_PROMPT, build_qa_prompt
from memory_condense.eval.diffuse_compilation import (
    DiffuseCompilationPolicy,
    DiffuseCompilationReceipt,
    DiffuseSourceCompilationReceipt,
)
from memory_condense.eval.diffuse_longmemeval import (
    LongMemEvalDiffuseQueryReceipt,
    qa_packet_framing,
)
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalAnalysisQueryReceipt,
    DiffuseLongMemEvalArm,
)
from memory_condense.eval.diffuse_longmemeval_inputs import LegacyDiffuseInputReceipt
from memory_condense.eval.diffuse_longmemeval_route_v2 import (
    EPISODE_PRIMARY_ANALYSIS_ARM_V2_FORMAT,
    EPISODE_PRIMARY_ANALYSIS_QUERY_V2_FORMAT,
)
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.search.episodes.representative_retrieval import (
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRepresentativeRetrievalPlan,
    EpisodeRepresentativeWitness,
    EpisodeSourceCandidate,
    EpisodeSourceCandidateScope,
    EpisodeSourceScan,
)
from memory_condense.search.episodes.retrieval import (
    DirectChunkSeed,
    EpisodeRetrievalPlan,
    EpisodeRetrievalPolicy,
)
from memory_condense.search.fusion.models import FusionCaps
from memory_condense.search.fusion.planner import _atom_refs, _authoritative_hyperedges
from memory_condense.search.fusion.resident_models import resident_values_sha256
from memory_condense.search.fusion.training_targets import (
    build_latent_router_structural_targets,
)
from memory_condense.search.packing.evidence_packet import pack_evidence_plan


_COMPILATION_POLICY_KEYS = {
    "boundary_mode", "min_episode_size", "max_episode_size", "fixed_interval",
    "surprise_window", "surprise_gamma", "surprise_min_history",
    "refinement_window", "refinement_max_nodes", "refinement_max_degree",
    "lexical_weight", "embedding_weight", "representative_limit",
}
_ARTIFACT_KEYS = {
    "artifact_id", "kind", "implementation_sha256", "policy_sha256",
    "model_id", "model_revision", "checkpoint_sha256", "metadata",
}
_SOURCE_RECEIPT_KEYS = {
    "source_id", "source_stream_sha256", "content_chunks", "metadata_chunks",
    "episode_ids", "unit_ids", "relation_ids", "episode_build_sha256",
    "discourse_link_sha256", "surprise_signal_receipt_sha256",
    "returned_signal_transformer_state_bytes", "receipt_sha256",
}
_LEGACY_ANCHOR_KEYS = {"chunk", "turn", "diagnostics"}
_LEGACY_CHUNK_KEYS = {
    "chunk_id", "turn_id", "start_char", "end_char", "token_count",
    "text_sha256", "embedding_sha256", "lexical_weights_sha256",
}
_LEGACY_TURN_KEYS = {"turn_id", "role", "source_id", "created_at", "text_sha256"}
_LEGACY_DIAGNOSTIC_KEYS = {
    "route", "consolidation_score", "consolidation_anchor",
    "consolidation_support", "score", "dense_score", "lexical_score",
    "association_score", "anchor_chunk_id", "association_hop",
    "edge_source_chunk_id", "association_path", "diffusion_heat",
    "association_support", "memory_source_id", "source_heat",
    "source_token_budget", "transition_distance", "transition_direction",
}
_DIFFUSE_ANCHOR_KEYS = {
    "chunk_id", "turn_id", "source_id", "start_char", "end_char",
    "token_count", "text_sha256", "score", "route", "dense_score",
    "lexical_score", "association_score",
}


def _episode_policy_body(value: EpisodeRetrievalPolicy) -> dict[str, object]:
    body = {
        "artifact_id": value.artifact_id,
        "max_anchor_episodes": value.max_anchor_episodes,
        "previous_episodes": value.previous_episodes,
        "next_episodes": value.next_episodes,
        "max_episode_seeds": value.max_episode_seeds,
        "max_direct_fallbacks": value.max_direct_fallbacks,
        "neighbor_decay": value.neighbor_decay,
    }
    _guard_explicit_fields(value, body, "episode policy")
    return body


def _closure_policy_body(value: ClosurePolicy) -> dict[str, object]:
    body = {
        "max_hops": value.max_hops,
        "max_units": value.max_units,
        "max_relations": value.max_relations,
        "max_degree": value.max_degree,
        "max_episode_neighbors": value.max_episode_neighbors,
        "max_frontier": value.max_frontier,
        "max_bundles": value.max_bundles,
        "beam_width": value.beam_width,
        "min_relation_confidence": value.min_relation_confidence,
    }
    _guard_explicit_fields(value, body, "closure policy")
    return body


def _representative_policy_body(
    value: EpisodeRepresentativeRetrievalPolicy,
) -> dict[str, object]:
    body = {
        "artifact_id": value.artifact_id,
        "max_input_sources": value.max_input_sources,
        "max_source_groups": value.max_source_groups,
        "max_episodes_per_source": value.max_episodes_per_source,
        "max_total_episodes": value.max_total_episodes,
        "max_representatives_per_episode": value.max_representatives_per_episode,
        "group_size": value.group_size,
        "beam_per_group": value.beam_per_group,
        "top_k": value.top_k,
        "representative_tokens": value.representative_tokens,
        "query_tokens": value.query_tokens,
        "score_mode": value.score_mode,
    }
    _guard_explicit_fields(value, body, "representative policy")
    return body


def _guard_explicit_fields(
    value: object,
    body: Mapping[str, object],
    label: str,
) -> None:
    if tuple(item.name for item in fields(value)) != tuple(body):
        raise RuntimeError(f"{label} domain fields changed its explicit projection")


def _body(value: Mapping[str, Any], keys: set[str], label: str) -> dict[str, Any]:
    result = _plain(value)
    if type(result) is not dict or set(result) != keys:
        raise LatentTrainingCorpusError(f"{label} has a non-closed schema")
    return result


def _tuple_strings(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(
        type(item) is not str or not item.strip() for item in value
    ):
        raise TypeError(f"{label} must be an exact non-empty-string sequence")
    return tuple(value)


def _seed(value: object, label: str) -> EpisodeSeed:
    if type(value) is not dict or set(value) != {
        "episode_id", "anchor_chunk_id", "score", "route", "path"
    }:
        raise LatentTrainingCorpusError(f"{label} has a non-closed schema")
    return EpisodeSeed(
        episode_id=value["episode_id"],
        anchor_chunk_id=value["anchor_chunk_id"],
        score=value["score"],
        route=value["route"],
        path=_tuple_strings(value["path"], f"{label}.path"),
    )


def _current_body(value: Any, expected: Mapping[str, Any], label: str) -> None:
    actual = value.identity_payload()
    try:
        assert_exact_value(actual, expected, label)
    except (TypeError, ValueError) as exc:
        raise LatentTrainingCorpusError(
            f"{label} changed during reconstruction"
        ) from exc


def _digest(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise TypeError(f"{label} must be an exact lowercase SHA-256")
    return value


def _optional_scalar(value: object, expected: type, label: str) -> None:
    if value is not None:
        require_exact_scalar(value, expected, label)


def _optional_number(value: object, expected: type, label: str) -> None:
    _optional_scalar(value, expected, label)
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _reconstruct_compilation(
    evidence: Any,
    arm: DiffuseLongMemEvalArm,
    package_implementation_sha256: str,
) -> DiffuseCompilationReceipt:
    body = _plain(evidence.compilation_receipt_body)
    snapshot_body = _plain(evidence.compilation_snapshot_body)
    if type(body) is not dict or type(snapshot_body) is not dict:
        raise TypeError("compilation bodies must be exact objects")
    artifact_body = body.get("artifact")
    if type(artifact_body) is not dict or set(artifact_body) != _ARTIFACT_KEYS:
        raise LatentTrainingCorpusError("compiled artifact has a non-closed schema")
    metadata = artifact_body.get("metadata")
    if type(metadata) is not dict or set(metadata) != {
        "boundary_policy_id", "scorer_id"
    }:
        raise LatentTrainingCorpusError("compiled artifact metadata is not closed")
    _digest(metadata["scorer_id"], "compiled artifact scorer_id")
    unsigned_artifact = {
        key: value for key, value in artifact_body.items() if key != "artifact_id"
    }
    if (
        artifact_body["artifact_id"]
        != f"disc-{identity_sha256(unsigned_artifact)[:24]}"
        or artifact_body["kind"]
        != f"longmemeval-diffuse-{arm.compilation.boundary_mode}"
        or artifact_body["implementation_sha256"]
        != package_implementation_sha256
        or metadata["boundary_policy_id"] != arm.compilation.boundary_mode
    ):
        raise LatentTrainingCorpusError("compiled artifact is not the owned artifact")
    artifact = DiscourseArtifact(**artifact_body)
    sources_value = body.get("source_receipts")
    if type(sources_value) is not list:
        raise TypeError("source receipts must be an exact sequence")
    sources = []
    for index, source_body in enumerate(sources_value):
        if type(source_body) is not dict or set(source_body) != _SOURCE_RECEIPT_KEYS:
            raise LatentTrainingCorpusError(
                f"source receipt[{index}] has a non-closed schema"
            )
        source = dict(source_body)
        for name in ("episode_ids", "unit_ids", "relation_ids"):
            source[name] = _tuple_strings(source[name], f"source receipt {name}")
        reconstructed = DiffuseSourceCompilationReceipt(**source)
        if reconstructed.returned_signal_transformer_state_bytes != 0:
            raise LatentTrainingCorpusError("compilation retained transformer state")
        _current_body(reconstructed, source_body, f"source receipt[{index}]")
        sources.append(reconstructed)
    snapshot = dict(snapshot_body)
    snapshot["artifact_ids"] = _tuple_strings(
        snapshot["artifact_ids"], "snapshot artifact IDs"
    )
    final_snapshot = DiscourseSnapshot(**snapshot)
    _current_body(final_snapshot, snapshot_body, "compilation final snapshot")
    compilation = DiffuseCompilationReceipt(
        artifact=artifact,
        compilation_policy_sha256=body["compilation_policy_sha256"],
        policy_sha256=body["policy_sha256"],
        source_receipts=tuple(sources),
        episode_coverage_receipt_sha256=body["episode_coverage_receipt_sha256"],
        discourse_coverage_receipt_sha256=body["discourse_coverage_receipt_sha256"],
        final_snapshot=final_snapshot,
        persisted_request_token_state_bytes=body[
            "persisted_request_token_state_bytes"
        ],
        format=body["format"],
        receipt_sha256=body["receipt_sha256"],
    )
    validate_compilation_receipt(compilation)
    _current_body(compilation, body, "compilation receipt")
    return compilation


def _reconstruct_arm(evidence: Any) -> tuple[DiffuseLongMemEvalArm, Mapping[str, Any]]:
    base_body = _plain(evidence.analysis_base_arm_body)
    if type(base_body) is not dict:
        raise TypeError("analysis base arm must be an exact object")
    compilation_body = base_body.get("compilation")
    if type(compilation_body) is not dict or set(compilation_body) != _COMPILATION_POLICY_KEYS:
        raise LatentTrainingCorpusError("compilation policy has a non-closed schema")
    episode_body = _plain(evidence.episode_policy_body)
    closure_body = _plain(evidence.closure_policy_body)
    if type(episode_body) is not dict or type(closure_body) is not dict:
        raise TypeError("analysis policy bodies must be exact objects")
    compilation = DiffuseCompilationPolicy(**compilation_body)
    episode = EpisodeRetrievalPolicy(**episode_body)
    closure = ClosurePolicy(**closure_body)
    for value, expected, label in (
        (compilation.identity_payload(), compilation_body, "compilation policy"),
        (
            _episode_policy_body(episode),
            episode_body,
            "episode policy",
        ),
        (
            _closure_policy_body(closure),
            closure_body,
            "closure policy",
        ),
    ):
        try:
            assert_exact_value(value, expected, label)
        except (TypeError, ValueError) as exc:
            raise LatentTrainingCorpusError(f"{label} body changed") from exc
    arm = DiffuseLongMemEvalArm(
        arm_id=base_body["arm_id"],
        compilation=compilation,
        episode=episode,
        closure=closure,
        max_context_tokens=base_body["max_context_tokens"],
        responder_output_token_reserve=base_body["responder_output_token_reserve"],
        require_owned_representative_runtime=base_body[
            "require_owned_representative_runtime"
        ],
    )
    try:
        assert_exact_value(arm.identity_payload(), base_body, "analysis base arm")
    except (TypeError, ValueError) as exc:
        raise LatentTrainingCorpusError(
            "analysis base arm cannot be reconstructed"
        ) from exc
    if arm.require_owned_representative_runtime is not True:
        raise LatentTrainingCorpusError("analysis base arm is not owned")
    arm_v2 = _plain(evidence.analysis_arm_v2_body)
    expected_v2 = {
        "format": EPISODE_PRIMARY_ANALYSIS_ARM_V2_FORMAT,
        "base_arm_sha256": arm.arm_sha256,
        "episodic_route": "episode_primary",
        "closure_routing_scope": "seeded_graph",
    }
    if type(arm_v2) is not dict or arm_v2.get("arm_sha256") != identity_sha256(expected_v2):
        raise LatentTrainingCorpusError("episode-primary analysis arm cannot be rebuilt")
    try:
        assert_exact_value(
            {key: value for key, value in arm_v2.items() if key != "arm_sha256"},
            expected_v2,
            "episode-primary analysis arm",
        )
    except (TypeError, ValueError) as exc:
        raise LatentTrainingCorpusError(
            "episode-primary analysis arm cannot be rebuilt"
        ) from exc
    return arm, arm_v2


def _reconstruct_receipts(evidence: Any) -> tuple[Any, Any, Any]:
    analysis_body = _plain(evidence.inner_analysis_query_receipt_body)
    diffuse_body = _plain(evidence.inner_diffuse_query_receipt_body)
    legacy_body = _plain(evidence.legacy_input_receipt_body)
    if not all(type(value) is dict for value in (analysis_body, diffuse_body, legacy_body)):
        raise TypeError("upstream receipts must be exact objects")
    for name in (
        "prompt_token_proxy", "max_input_prompt_token_proxy",
        "responder_output_token_reserve", "prompt_workspace_token_proxy",
        "max_prompt_workspace_token_proxy",
        "packet_retained_request_token_state_bytes",
    ):
        require_exact_scalar(diffuse_body[name], int, f"diffuse {name}")
    for name in (
        "representative_returned_plan_transformer_state_bytes",
        "store_retained_request_token_state_bytes",
    ):
        _optional_scalar(diffuse_body[name], int, f"diffuse {name}")
    for name in (
        "representative_scope_exhaustive",
        "representative_runtime_binding_certified",
    ):
        _optional_scalar(diffuse_body[name], bool, f"diffuse {name}")
    for name in (
        "expansion_exhaustive", "closure_complete_claimed",
        "closure_scope_exhaustive",
    ):
        require_exact_scalar(diffuse_body[name], bool, f"diffuse {name}")
    require_exact_scalar(
        diffuse_body["closure_stopping_reason"],
        str,
        "diffuse closure_stopping_reason",
    )
    for name in (
        "input_anchor_chunk_ids", "representative_seed_episode_ids",
        "truncated_episode_ids", "truncated_direct_chunk_ids",
        "scope_witness_sha256s",
    ):
        diffuse_body[name] = _tuple_strings(diffuse_body[name], f"diffuse {name}")
    diffuse = LongMemEvalDiffuseQueryReceipt(**diffuse_body)
    legacy_body["anchor_chunk_ids"] = _tuple_strings(
        legacy_body["anchor_chunk_ids"], "legacy anchor IDs"
    )
    legacy_body["source_candidate_ids"] = _tuple_strings(
        legacy_body["source_candidate_ids"], "legacy source candidate IDs"
    )
    legacy = LegacyDiffuseInputReceipt(**legacy_body)
    analysis = DiffuseLongMemEvalAnalysisQueryReceipt(**analysis_body)
    _current_body(diffuse, _plain(evidence.inner_diffuse_query_receipt_body), "diffuse receipt")
    _current_body(legacy, _plain(evidence.legacy_input_receipt_body), "legacy receipt")
    _current_body(analysis, _plain(evidence.inner_analysis_query_receipt_body), "analysis receipt")
    return analysis, diffuse, legacy


def _reconstruct_source(evidence: Any) -> EpisodeSourceCandidateScope:
    body = _plain(evidence.source_scope_body)
    if type(body) is not dict or type(body.get("candidates")) is not list:
        raise TypeError("source scope has malformed collections")
    candidates_list = body["candidates"]
    if any(
        type(item) is not dict
        or set(item) != {"source_id", "score", "route"}
        or type(item["source_id"]) is not str
        or type(item["route"]) is not str
        or type(item["score"]) is not float
        for item in candidates_list
    ):
        raise TypeError("source candidates have non-exact scalar fields")
    candidates = tuple(EpisodeSourceCandidate(**item) for item in candidates_list)
    scope = EpisodeSourceCandidateScope(
        artifact_id=body["artifact_id"],
        snapshot_sha256=body["snapshot_sha256"],
        source_revision=body["source_revision"],
        source_content_sha256=body["source_content_sha256"],
        query_sha256=body["query_sha256"],
        router_policy_sha256=body["router_policy_sha256"],
        universe_source_ids=_tuple_strings(body["universe_source_ids"], "source universe"),
        candidates=candidates,
        truncated_source_ids=_tuple_strings(body["truncated_source_ids"], "truncated sources"),
        universe_enumerated=body["universe_enumerated"],
        receipt_sha256=body["receipt_sha256"],
    )
    _current_body(scope, body, "source candidate scope")
    if not scope.candidates:
        raise LatentTrainingCorpusError("episode-primary source scope is empty")
    if scope.candidates != tuple(
        sorted(scope.candidates, key=lambda item: (-item.score, item.source_id, item.route))
    ):
        raise LatentTrainingCorpusError("source candidates changed canonical rank order")
    return scope


def _reconstruct_direct(evidence: Any) -> EpisodeRetrievalPlan:
    body = _plain(evidence.direct_expansion_body)
    if type(body) is not dict:
        raise TypeError("direct expansion must be an exact object")
    seeds = tuple(_seed(value, "direct seed") for value in body["seeds"])
    fallbacks = []
    for value in body["direct_fallbacks"]:
        item = dict(value)
        item["path"] = _tuple_strings(item["path"], "direct fallback path")
        fallbacks.append(DirectChunkSeed(**item))
    plan = EpisodeRetrievalPlan(
        policy_sha256=body["policy_sha256"],
        seeds=seeds,
        direct_fallbacks=tuple(fallbacks),
        truncated_episode_ids=_tuple_strings(
            body["truncated_episode_ids"], "direct truncated episode IDs"
        ),
        truncated_direct_chunk_ids=_tuple_strings(
            body["truncated_direct_chunk_ids"], "direct truncated chunk IDs"
        ),
        receipt_sha256=body["receipt_sha256"],
    )
    _current_body(plan, body, "direct expansion")
    return plan


def _validate_anchor_projections(evidence: Any, legacy: Any, diffuse: Any) -> None:
    body = _plain(evidence.anchor_projection_body)
    if type(body) is not dict or set(body) != {"legacy", "diffuse"}:
        raise LatentTrainingCorpusError("anchor projection has a non-closed schema")
    legacy_rows = body["legacy"]
    diffuse_rows = body["diffuse"]
    if type(legacy_rows) is not list or type(diffuse_rows) is not list or (
        len(legacy_rows) != len(diffuse_rows)
    ):
        raise TypeError("anchor projections must be parallel exact sequences")
    chunk_ids = []
    for index, (legacy_row, diffuse_row) in enumerate(zip(legacy_rows, diffuse_rows)):
        if type(legacy_row) is not dict or set(legacy_row) != _LEGACY_ANCHOR_KEYS:
            raise LatentTrainingCorpusError(f"legacy anchor[{index}] schema changed")
        chunk = legacy_row["chunk"]
        turn = legacy_row["turn"]
        diagnostics = legacy_row["diagnostics"]
        if type(chunk) is not dict or set(chunk) != _LEGACY_CHUNK_KEYS:
            raise LatentTrainingCorpusError(f"legacy anchor[{index}] chunk changed")
        if turn is not None and (
            type(turn) is not dict or set(turn) != _LEGACY_TURN_KEYS
        ):
            raise LatentTrainingCorpusError(f"legacy anchor[{index}] turn changed")
        if type(diagnostics) is not dict or set(diagnostics) != _LEGACY_DIAGNOSTIC_KEYS:
            raise LatentTrainingCorpusError(
                f"legacy anchor[{index}] diagnostics changed"
            )
        if type(diffuse_row) is not dict or set(diffuse_row) != _DIFFUSE_ANCHOR_KEYS:
            raise LatentTrainingCorpusError(f"diffuse anchor[{index}] schema changed")
        for name in ("chunk_id", "turn_id", "text_sha256"):
            require_exact_scalar(chunk[name], str, f"legacy anchor[{index}].chunk.{name}")
        for name in ("start_char", "end_char", "token_count"):
            require_exact_scalar(chunk[name], int, f"legacy anchor[{index}].chunk.{name}")
        _digest(chunk["text_sha256"], f"legacy anchor[{index}] text identity")
        for name in ("embedding_sha256", "lexical_weights_sha256"):
            _optional_scalar(chunk[name], str, f"legacy anchor[{index}].chunk.{name}")
            if chunk[name] is not None:
                _digest(chunk[name], f"legacy anchor[{index}].chunk.{name}")
        if turn is not None:
            for name in ("turn_id", "role", "created_at", "text_sha256"):
                require_exact_scalar(turn[name], str, f"legacy anchor[{index}].turn.{name}")
            _optional_scalar(turn["source_id"], str, f"legacy anchor[{index}].turn.source_id")
            _digest(turn["text_sha256"], f"legacy anchor[{index}] turn text identity")
        for name in (
            "route", "consolidation_anchor", "anchor_chunk_id",
            "edge_source_chunk_id", "memory_source_id", "transition_direction",
        ):
            _optional_scalar(diagnostics[name], str, f"anchor diagnostics.{name}")
        for name in (
            "consolidation_score", "dense_score", "lexical_score",
            "association_score", "diffusion_heat", "source_heat",
        ):
            _optional_number(diagnostics[name], float, f"anchor diagnostics.{name}")
        require_exact_scalar(diagnostics["score"], float, "anchor diagnostics.score")
        if not math.isfinite(diagnostics["score"]):
            raise ValueError("anchor diagnostics.score must be finite")
        for name in (
            "consolidation_support", "association_hop", "association_support",
            "source_token_budget", "transition_distance",
        ):
            _optional_scalar(diagnostics[name], int, f"anchor diagnostics.{name}")
        association_path = diagnostics["association_path"]
        if association_path is not None:
            _tuple_strings(association_path, "anchor diagnostics.association_path")
        if diagnostics["transition_direction"] not in {None, "previous", "next"}:
            raise ValueError("anchor transition_direction changed")
        for name in ("chunk_id", "turn_id", "source_id", "text_sha256", "route"):
            require_exact_scalar(diffuse_row[name], str, f"diffuse anchor[{index}].{name}")
        for name in ("start_char", "end_char", "token_count"):
            require_exact_scalar(diffuse_row[name], int, f"diffuse anchor[{index}].{name}")
        require_exact_scalar(diffuse_row["score"], float, f"diffuse anchor[{index}].score")
        for name in ("dense_score", "lexical_score", "association_score"):
            _optional_number(diffuse_row[name], float, f"diffuse anchor[{index}].{name}")
        _digest(diffuse_row["text_sha256"], f"diffuse anchor[{index}] text identity")
        durable_source = diagnostics["memory_source_id"]
        if durable_source is None and turn is not None:
            durable_source = turn["source_id"] or turn["turn_id"]
        if durable_source is None:
            durable_source = chunk["turn_id"]
        expected_diffuse = {
            "chunk_id": chunk["chunk_id"],
            "turn_id": chunk["turn_id"],
            "source_id": durable_source,
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
            "token_count": chunk["token_count"],
            "text_sha256": chunk["text_sha256"],
            "score": diagnostics["score"],
            "route": diagnostics["route"] or "unspecified",
            "dense_score": diagnostics["dense_score"],
            "lexical_score": diagnostics["lexical_score"],
            "association_score": diagnostics["association_score"],
        }
        try:
            assert_exact_value(
                diffuse_row, expected_diffuse, f"anchor[{index}] projections"
            )
        except (TypeError, ValueError) as exc:
            raise LatentTrainingCorpusError(
                f"legacy/diffuse anchor[{index}] projections disagree"
            ) from exc
        chunk_ids.append(chunk["chunk_id"])
    if (
        identity_sha256(tuple(legacy_rows)) != legacy.anchor_sequence_sha256
        or identity_sha256(tuple(diffuse_rows)) != diffuse.anchor_sequence_sha256
        or tuple(chunk_ids) != legacy.anchor_chunk_ids
        or tuple(chunk_ids) != diffuse.input_anchor_chunk_ids
    ):
        raise LatentTrainingCorpusError("anchor sequence cannot be reconstructed")


def _reconstruct_representative(
    evidence: Any,
    source: EpisodeSourceCandidateScope,
) -> tuple[EpisodeRepresentativeRetrievalPlan, EpisodeRepresentativeRetrievalPolicy]:
    body = _plain(evidence.representative_expansion_body)
    if type(body) is not dict:
        raise TypeError("representative expansion must be an exact object")
    scan_values = body["source_scans"]
    if type(scan_values) is not list or any(
        type(value) is not dict
        or set(value) != {
            "source_id", "requested_limit", "observed_count",
            "candidate_count", "exhaustive", "status",
        }
        or type(value["source_id"]) is not str
        or type(value["status"]) is not str
        or type(value["exhaustive"]) is not bool
        or any(
            type(value[name]) is not int
            for name in ("requested_limit", "observed_count", "candidate_count")
        )
        for value in scan_values
    ):
        raise TypeError("representative source scans have non-exact fields")
    scans = tuple(EpisodeSourceScan(**value) for value in scan_values)
    witnesses = []
    for value in body["candidate_witnesses"]:
        if type(value) is not dict or set(value) != {
            "episode_id", "source_id", "anchor_chunk_id",
            "representative_chunk_ids", "representative_identity_sha256s",
            "candidate_text_sha256", "source_score", "source_route",
        } or type(value["source_score"]) is not float:
            raise TypeError("representative witness has non-exact fields")
        item = dict(value)
        item["representative_chunk_ids"] = _tuple_strings(
            item["representative_chunk_ids"], "representative chunk IDs"
        )
        item["representative_identity_sha256s"] = _tuple_strings(
            item["representative_identity_sha256s"], "representative identities"
        )
        _digest(item["candidate_text_sha256"], "candidate text identity")
        for digest in item["representative_identity_sha256s"]:
            _digest(digest, "representative identity")
        witnesses.append(EpisodeRepresentativeWitness(**item))
    seeds = tuple(_seed(value, "representative seed") for value in body["seeds"])
    plan = EpisodeRepresentativeRetrievalPlan(
        artifact_id=body["artifact_id"],
        policy_sha256=body["policy_sha256"],
        query_sha256=body["query_sha256"],
        query_input_sha256=body["query_input_sha256"],
        linker_identity_sha256=body["linker_identity_sha256"],
        runtime_binding_certified=body["runtime_binding_certified"],
        source_scope_receipt_sha256=body["source_scope_receipt_sha256"],
        source_universe_exhaustive=body["source_universe_exhaustive"],
        source_scans=scans,
        candidate_witnesses=tuple(witnesses),
        seeds=seeds,
        truncated_source_ids=_tuple_strings(body["truncated_source_ids"], "representative truncated sources"),
        truncated_episode_ids=_tuple_strings(body["truncated_episode_ids"], "representative truncated episodes"),
        unavailable_episode_ids=_tuple_strings(body["unavailable_episode_ids"], "representative unavailable episodes"),
        passes=body["passes"],
        max_workspace_candidates=body["max_workspace_candidates"],
        max_workspace_tokens=body["max_workspace_tokens"],
        total_candidate_inspections=body["total_candidate_inspections"],
        returned_plan_transformer_state_bytes=body[
            "returned_plan_transformer_state_bytes"
        ],
        receipt_sha256=body["receipt_sha256"],
    )
    _current_body(plan, body, "representative expansion")
    policy_body = _plain(evidence.representative_policy_body)
    if type(policy_body) is not dict:
        raise TypeError("representative policy must be an exact object")
    policy = EpisodeRepresentativeRetrievalPolicy(**policy_body)
    try:
        assert_exact_value(
            _representative_policy_body(policy),
            policy_body,
            "representative policy",
        )
    except (TypeError, ValueError) as exc:
        raise LatentTrainingCorpusError(
            "representative policy body changed"
        ) from exc
    if policy.policy_sha256 != plan.policy_sha256 or (
        policy.artifact_id != plan.artifact_id
    ):
        raise LatentTrainingCorpusError(
            "representative policy does not bind its plan"
        )
    candidates = {item.source_id: item for item in source.candidates}
    witness_by_episode = {item.episode_id: item for item in plan.candidate_witnesses}
    if (
        len(source.candidates) > policy.max_input_sources
        or any(scan.status not in {"ok", "lookup_error", "identity_error"} for scan in scans)
        or any(scan.source_id not in candidates for scan in scans)
        or any(
            witness.source_id not in candidates
            or witness.source_score != candidates[witness.source_id].score
            or witness.source_route != candidates[witness.source_id].route
            for witness in plan.candidate_witnesses
        )
        or any(
            seed.route != "episode_representative_qwen"
            or seed.anchor_chunk_id != witness_by_episode[seed.episode_id].anchor_chunk_id
            or seed.path
            != (
                seed.anchor_chunk_id,
                seed.episode_id,
                f"source_route:{witness_by_episode[seed.episode_id].source_route}",
                "qwen_nested_representative",
            )
            for seed in plan.seeds
        )
        or any(
            scan.candidate_count
            != sum(
                witness.source_id == scan.source_id
                for witness in plan.candidate_witnesses
            )
            for scan in scans
        )
    ):
        raise LatentTrainingCorpusError("representative expansion is not an owned plan")
    if (
        tuple(scan.source_id for scan in scans)
        != tuple(item.source_id for item in source.candidates[: policy.max_source_groups])
        or plan.truncated_source_ids
        != tuple(item.source_id for item in source.candidates[policy.max_source_groups :])
        or any(scan.requested_limit != policy.max_episodes_per_source for scan in scans)
        or any(
            len(witness.representative_chunk_ids)
            > policy.max_representatives_per_episode
            for witness in plan.candidate_witnesses
        )
        or len(plan.candidate_witnesses) > policy.max_total_episodes
        or len(plan.seeds) > policy.top_k
    ):
        raise LatentTrainingCorpusError("representative policy caps do not reconstruct")
    return plan, policy


def live_route_v2_implementation_sha256() -> str:
    return identity_sha256(
        {
            "format": (
                "memory-condense-longmemeval-episode-primary-route-"
                "implementation-v2"
            ),
            "package_implementation_sha256": implementation_sha256(),
        }
    )


def validate_persisted_route(
    row: LatentTrainingCorpusRowManifest,
    payload: Any,
    *,
    expected_route_implementation_sha256: str,
) -> None:
    evidence = row.route_evidence
    route = evidence.route_receipt
    package_implementation = implementation_sha256()
    current_route_implementation = identity_sha256(
        {
            "format": (
                "memory-condense-longmemeval-episode-primary-route-"
                "implementation-v2"
            ),
            "package_implementation_sha256": package_implementation,
        }
    )
    if (
        current_route_implementation != expected_route_implementation_sha256
        or route.route_v2_implementation_sha256 != current_route_implementation
    ):
        raise LatentTrainingCorpusError("route-v2 live implementation identity changed")
    arm, arm_v2 = _reconstruct_arm(evidence)
    compilation = _reconstruct_compilation(evidence, arm, package_implementation)
    analysis, diffuse, legacy = _reconstruct_receipts(evidence)
    source = _reconstruct_source(evidence)
    direct = _reconstruct_direct(evidence)
    representative, representative_policy = _reconstruct_representative(
        evidence, source
    )
    _validate_anchor_projections(evidence, legacy, diffuse)
    query_sha = identity_sha256({"query": payload.retrieval_query})
    prompt_question_sha = identity_sha256({"prompt_question": payload.prompt_question})
    probe_sha = identity_sha256(
        {
            "question_id": payload.question_id,
            "retrieval_query": payload.retrieval_query,
            "prompt_question": payload.prompt_question,
        }
    )
    expected_query_input = truncate_to_tokens_lossless(
        payload.retrieval_query, representative_policy.query_tokens
    )
    candidates = source.candidates
    source_sequence_sha = identity_sha256(
        tuple(item.identity_payload() for item in candidates)
    )
    input_checks = {
        "legacy artifact": legacy.artifact_id == route.artifact_id,
        "legacy query": legacy.query_sha256 == query_sha,
        "route query": route.retrieval_query_sha256 == query_sha,
        "source candidate IDs": legacy.source_candidate_ids
        == tuple(item.source_id for item in candidates),
        "source candidate sequence": legacy.source_candidate_sequence_sha256
        == source_sequence_sha,
        "source scope": legacy.source_candidate_scope_receipt_sha256
        == source.receipt_sha256,
        "anchor IDs": legacy.anchor_chunk_ids == diffuse.input_anchor_chunk_ids,
        "diffuse episode policy": direct.policy_sha256
        == diffuse.episode_policy_sha256,
        "arm episode policy": direct.policy_sha256
        == replace(arm.episode, artifact_id=route.artifact_id).policy_sha256,
        "representative query input": representative.query_input_sha256
        == identity_sha256({"query_input": expected_query_input}),
    }
    failed_input = tuple(name for name, passed in input_checks.items() if not passed)
    if failed_input:
        raise LatentTrainingCorpusError(
            "legacy/direct inputs do not join the route: " + ", ".join(failed_input)
        )
    if (
        source.artifact_id != route.artifact_id
        or diffuse.artifact_id != route.artifact_id
        or diffuse.snapshot_sha256 != route.snapshot_sha256
        or source.snapshot_sha256 != route.snapshot_sha256
        or source.source_revision != compilation.final_snapshot.source_revision
        or source.source_content_sha256
        != compilation.final_snapshot.source_content_sha256
        or source.query_sha256 != query_sha
        or source.receipt_sha256 != route.source_candidate_scope_receipt_sha256
        or source.selected_scope_exhaustive is not route.source_scope_exhaustive
        or representative.artifact_id != route.artifact_id
        or representative.query_sha256 != query_sha
        or representative.source_scope_receipt_sha256 != source.receipt_sha256
        or representative.source_universe_exhaustive
        is not source.selected_scope_exhaustive
        or representative.runtime_binding_certified is not True
        or representative.returned_plan_transformer_state_bytes != 0
        or not representative.seeds
    ):
        raise LatentTrainingCorpusError("source/representative route joins changed")
    if direct.seeds or direct.direct_fallbacks or direct.truncated_episode_ids or (
        direct.truncated_direct_chunk_ids
    ):
        raise LatentTrainingCorpusError("episode_primary persisted a direct route")
    rep_seeds = representative.seeds
    if tuple(seed.identity_payload() for seed in payload.plan.seeds) != tuple(
        seed.identity_payload() for seed in rep_seeds
    ) or payload.plan.direct_chunk_ids:
        raise LatentTrainingCorpusError("closure seeds differ from representative seeds")
    seed_sha = seed_projection_sha256(rep_seeds)
    combined = identity_sha256(
        {
            "episodic_route": "episode_primary",
            "direct_expansion_receipt_sha256": direct.receipt_sha256,
            "representative_expansion_receipt_sha256": representative.receipt_sha256,
            "seeds": [seed.identity_payload() for seed in rep_seeds],
            "direct_chunk_ids": [],
        }
    )
    rep_exhaustive = representative.candidate_scope_exhaustive
    if (
        diffuse.representative_receipt_sha256 != representative.receipt_sha256
        or diffuse.representative_scope_exhaustive is not rep_exhaustive
        or diffuse.representative_runtime_binding_certified is not True
        or diffuse.representative_returned_plan_transformer_state_bytes != 0
        or diffuse.representative_seed_episode_ids
        != tuple(seed.episode_id for seed in rep_seeds)
        or diffuse.truncated_episode_ids != direct.truncated_episode_ids
        or diffuse.truncated_direct_chunk_ids != direct.truncated_direct_chunk_ids
        or diffuse.combined_expansion_sha256 != combined
        or diffuse.expansion_exhaustive is not rep_exhaustive
        or payload.plan.expansion_receipt_sha256 != combined
    ):
        raise LatentTrainingCorpusError("diffuse expansion route cannot be reconstructed")
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
    witness_sha256s = tuple(w.witness_sha256 for w in payload.plan.scope_witnesses)
    if (
        len(routing) != 1
        or routing[0].identity_payload() != expected_routing.identity_payload()
        or len(expansion) != 1
        or expansion[0].identity_payload() != expected_expansion.identity_payload()
        or any(w.kind == "artifact_unit_scan" for w in payload.plan.scope_witnesses)
        or diffuse.scope_witness_sha256s != witness_sha256s
        or diffuse.closure_scope_exhaustive is not False
    ):
        raise LatentTrainingCorpusError("route witness bodies cannot be reconstructed")
    prompt_messages = tuple(build_qa_prompt(payload.prompt_question, [payload.packet.context]))
    prompt_messages_sha = identity_sha256(list(prompt_messages))
    coordinates = tuple(
        {"atom_id": atom.atom_id, **atom.span.identity_payload(), "label": atom.label}
        for atom in payload.packet.atoms
    )
    prefix, suffix = qa_packet_framing(payload.prompt_question)
    repacked = pack_evidence_plan(
        payload.plan,
        max_context_tokens=arm.max_context_tokens,
        base_messages=({"role": "system", "content": QA_SYSTEM_PROMPT},),
        evidence_message_role="user",
        evidence_prefix=prefix,
        evidence_suffix=suffix,
        max_prompt_tokens=diffuse.max_prompt_workspace_token_proxy,
        output_token_reserve=arm.responder_output_token_reserve,
    )
    try:
        assert_exact_value(
            {
                "context": repacked.context,
                "atoms": [item.identity_payload() for item in repacked.atoms],
                "bundles": [item.identity_payload() for item in repacked.bundles],
                "receipt": repacked.receipt.identity_payload(),
            },
            {
                "context": payload.packet.context,
                "atoms": [item.identity_payload() for item in payload.packet.atoms],
                "bundles": [item.identity_payload() for item in payload.packet.bundles],
                "receipt": payload.packet.receipt.identity_payload(),
            },
            "provider-free repacked evidence packet",
        )
    except (TypeError, ValueError) as exc:
        raise LatentTrainingCorpusError(
            "evidence packet cannot be independently repacked"
        ) from exc
    validate_closure_stopping_state(diffuse, payload.plan, payload.packet.receipt)
    if (
        diffuse.query_program_sha256 != payload.plan.query_program.program_sha256
        or diffuse.retrieval_query_sha256 != query_sha
        or diffuse.prompt_question_sha256 != prompt_question_sha
        or diffuse.closure_policy_sha256 != payload.plan.policy.policy_sha256
        or diffuse.closure_policy_sha256 != arm.closure.policy_sha256
        or diffuse.closure_plan_sha256 != payload.plan.plan_sha256
        or diffuse.packet_receipt_sha256 != payload.packet.receipt.receipt_sha256
        or diffuse.context_sha256 != payload.packet.receipt.context_sha256
        or diffuse.evidence_coordinates_sha256 != identity_sha256(coordinates)
        or diffuse.prompt_messages_sha256 != prompt_messages_sha
        or payload.packet.receipt.prompt_messages_sha256 != prompt_messages_sha
        or diffuse.packet_retained_request_token_state_bytes != 0
        or diffuse.store_retained_request_token_state_bytes != 0
    ):
        raise LatentTrainingCorpusError("plan/packet/diffuse receipt joins changed")
    packet_receipt = payload.packet.receipt
    for packet_name, diffuse_name in (
        ("prompt_token_proxy", "prompt_token_proxy"),
        ("responder_output_token_reserve", "responder_output_token_reserve"),
        ("prompt_workspace_token_proxy", "prompt_workspace_token_proxy"),
        ("max_prompt_token_proxy", "max_prompt_workspace_token_proxy"),
    ):
        if getattr(packet_receipt, packet_name) != getattr(diffuse, diffuse_name):
            raise LatentTrainingCorpusError("packet prompt accounting changed")
    if (
        arm.episode.artifact_id not in (None, route.artifact_id)
        or payload.plan.artifact_id != route.artifact_id
        or payload.plan.snapshot.snapshot_sha256 != route.snapshot_sha256
        or payload.packet.receipt.plan_sha256 != payload.plan.plan_sha256
        or diffuse.expansion_receipt_sha256 != direct.receipt_sha256
        or analysis.representative_policy_factory_identity_sha256 is None
        or _digest(
            analysis.representative_policy_factory_identity_sha256,
            "representative policy factory identity",
        )
        != analysis.representative_policy_factory_identity_sha256
        or
        compilation.compilation_policy_sha256 != arm.compilation.policy_sha256
        or compilation.artifact.artifact_id != route.artifact_id
        or compilation.final_snapshot.snapshot_sha256 != route.snapshot_sha256
        or analysis.analysis_arm_sha256 != arm.arm_sha256
        or analysis.matched_controls_sha256 != arm.matched_controls_sha256
        or analysis.compilation_receipt_sha256 != compilation.receipt_sha256
        or analysis.question_probe_sha256 != probe_sha
        or analysis.legacy_input_receipt_sha256 != legacy.receipt_sha256
        or analysis.diffuse_query_receipt_sha256 != diffuse.receipt_sha256
        or analysis.artifact_id != route.artifact_id
        or analysis.snapshot_sha256 != route.snapshot_sha256
        or analysis.representative_linker_identity_sha256
        != representative.linker_identity_sha256
        or analysis.representative_policy_sha256 != representative.policy_sha256
        or analysis.representative_policy_controls_sha256
        != replace(
            representative_policy,
            artifact_id="matched-artifact",
        ).policy_sha256
    ):
        raise LatentTrainingCorpusError("analysis/compilation route joins changed")
    expected_record = identity_sha256(
        {
            "format": EPISODE_PRIMARY_ANALYSIS_QUERY_V2_FORMAT,
            "analysis_arm_v2_sha256": arm_v2["arm_sha256"],
            "inner_analysis_query_receipt_sha256": analysis.receipt_sha256,
            "inner_diffuse_query_receipt_sha256": diffuse.receipt_sha256,
            "route_receipt_sha256": route.receipt_sha256,
        }
    )
    route_checks = {
        "analysis_arm_v2_sha256": arm_v2["arm_sha256"],
        "inner_analysis_query_receipt_sha256": analysis.receipt_sha256,
        "inner_diffuse_query_receipt_sha256": diffuse.receipt_sha256,
        "legacy_input_receipt_sha256": legacy.receipt_sha256,
        "direct_expansion_receipt_sha256": direct.receipt_sha256,
        "representative_expansion_receipt_sha256": representative.receipt_sha256,
        "representative_linker_identity_sha256": representative.linker_identity_sha256,
        "representative_policy_sha256": representative.policy_sha256,
        "representative_seed_projection_sha256": seed_sha,
        "closure_seed_projection_sha256": seed_sha,
        "combined_expansion_sha256": combined,
        "closure_policy_sha256": payload.plan.policy.policy_sha256,
        "closure_plan_sha256": payload.plan.plan_sha256,
        "closure_routing_scope_witness_sha256": expected_routing.witness_sha256,
        "episode_expansion_witness_sha256": expected_expansion.witness_sha256,
        "query_program_sha256": payload.plan.query_program.program_sha256,
        "packet_receipt_sha256": payload.packet.receipt.receipt_sha256,
        "context_sha256": payload.packet.receipt.context_sha256,
        "prompt_messages_sha256": prompt_messages_sha,
    }
    if any(getattr(route, name) != value for name, value in route_checks.items()) or (
        route.representative_seed_count != len(rep_seeds)
        or route.representative_scope_exhaustive is not rep_exhaustive
        or route.expansion_exhaustive is not rep_exhaustive
        or route.closure_max_frontier != payload.plan.policy.max_frontier
        or expected_record != row.route_record_sha256
        or payload.question_id != row.question_id
        or probe_sha != row.question_probe_sha256
        or query_sha != row.retrieval_query_sha256
        or prompt_question_sha != row.prompt_question_sha256
    ):
        raise LatentTrainingCorpusError("route-v2 record cannot be independently rebuilt")
    expected_target = build_latent_router_structural_targets(
        payload.packet, payload.plan, caps=FusionCaps()
    )
    try:
        assert_exact_value(
            _target_body(expected_target),
            _target_body(row.structural_target),
            "structural target",
        )
    except (TypeError, ValueError) as exc:
        raise LatentTrainingCorpusError(
            "structural target cannot be reconstructed"
        ) from exc
    refs_sha = resident_values_sha256(_ATOM_REF_KIND, _atom_refs(payload.packet))
    edges_sha = resident_values_sha256(
        _HYPEREDGE_KIND, _authoritative_hyperedges(payload.packet)
    )
    if refs_sha != row.ordered_atom_refs_sha256 or edges_sha != row.authoritative_hyperedges_sha256:
        raise LatentTrainingCorpusError("packet-order atom refs or hyperedges changed")


__all__ = ["live_route_v2_implementation_sha256", "validate_persisted_route"]
