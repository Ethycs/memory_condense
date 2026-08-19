"""Cross-record validation for the shared-base replay receipt."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from memory_condense.domain.discourse import (
    ClosurePolicy,
    DiscourseArtifact,
    DiscourseSnapshot,
    EpisodeSeed,
    identity_sha256,
)
from memory_condense.eval.diffuse_compilation import (
    DIFFUSE_COMPILATION_FORMAT,
    DiffuseCompilationPolicy,
    DiffuseCompilationReceipt,
    DiffuseSourceCompilationReceipt,
)
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DIFFUSE_ANALYSIS_PHASE_FORMAT,
    DiffuseLongMemEvalAnalysisQueryReceipt,
    DiffuseLongMemEvalArm,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    DETERMINISTIC_DIFFUSE_INGEST_FORMAT,
)
from memory_condense.eval.diffuse_longmemeval_matched import (
    DiffuseLongMemEvalMatchedProbeReceipt,
    DiffuseLongMemEvalMatchedSuiteReceipt,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DIFFUSE_RUNTIME_RESULT_FORMAT,
    ResidencyPreflightObservation,
)
from memory_condense.eval.diffuse_longmemeval_runtime_matched import (
    DiffuseLongMemEvalMatchedRuntimeSuiteReceipt,
)
from memory_condense.search.episodes import EpisodeRetrievalPolicy
from memory_condense.search.episodes.representative_retrieval import (
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRepresentativeRetrievalPlan,
    EpisodeRepresentativeWitness,
    EpisodeSourceScan,
)


_MODES = ("fixed_interval", "lexical_embedding", "qwen_head")


def _body(value) -> dict[str, object]:
    body = json.loads(value.canonical_identity_json)
    if not isinstance(body, dict):  # pragma: no cover - guarded by its model
        raise ValueError("identity body must be an object")
    return body


def _keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has an unsupported identity schema")


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _matched_arm_sha256(arm: DiffuseLongMemEvalArm) -> str:
    return identity_sha256(arm.identity_payload(include_boundary=False))


def _cuda_device(value: object) -> str | None:
    normalized = str(value).casefold().strip()
    if normalized == "cuda":
        return "cuda:0"
    prefix = "cuda:"
    ordinal = normalized[len(prefix):] if normalized.startswith(prefix) else ""
    if not ordinal.isdigit() or str(int(ordinal)) != ordinal:
        return None
    return f"cuda:{ordinal}"


def _owned_arm(row) -> DiffuseLongMemEvalArm:
    arm_body = _body(row.arm_identity)
    episode_body = _body(row.episode_policy)
    closure_body = _body(row.closure_policy)
    episode = EpisodeRetrievalPolicy(**episode_body)
    closure = ClosurePolicy(**closure_body)
    if _jsonable(asdict(episode)) != episode_body or _jsonable(
        asdict(closure)
    ) != closure_body:
        raise ValueError("arm policy body is not an exact owned policy")
    arm = DiffuseLongMemEvalArm(
        arm_id=arm_body["arm_id"],
        compilation=DiffuseCompilationPolicy(**arm_body["compilation"]),
        episode=episode,
        closure=closure,
        max_context_tokens=arm_body["max_context_tokens"],
        responder_output_token_reserve=arm_body[
            "responder_output_token_reserve"
        ],
        require_owned_representative_runtime=arm_body[
            "require_owned_representative_runtime"
        ],
    )
    if _jsonable(arm.identity_payload()) != arm_body or arm.arm_sha256 != (
        row.arm_identity.identity_sha256
    ):
        raise ValueError("arm body is not an exact owned arm identity")
    return arm


def _owned_compilation(row, arm: DiffuseLongMemEvalArm) -> DiffuseCompilationReceipt:
    body = _body(row.compilation)
    artifact_body = body.get("artifact")
    source_bodies = body.get("source_receipts")
    if not isinstance(artifact_body, dict) or not isinstance(source_bodies, list):
        raise ValueError("compilation receipt has malformed nested bodies")
    _keys(
        artifact_body,
        {
            "artifact_id", "kind", "implementation_sha256", "policy_sha256",
            "model_id", "model_revision", "checkpoint_sha256", "metadata",
        },
        "compilation artifact",
    )
    metadata = artifact_body.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("compilation artifact metadata is malformed")
    _keys(metadata, {"boundary_policy_id", "scorer_id"}, "artifact metadata")
    unsigned_artifact = {
        key: value for key, value in artifact_body.items() if key != "artifact_id"
    }
    expected_artifact_id = f"disc-{identity_sha256(unsigned_artifact)[:24]}"
    if (
        artifact_body.get("artifact_id") != expected_artifact_id
        or artifact_body.get("kind")
        != f"longmemeval-diffuse-{arm.compilation.boundary_mode}"
        or metadata.get("boundary_policy_id") != arm.compilation.boundary_mode
    ):
        raise ValueError("compilation artifact is not the owned boundary artifact")
    source_keys = {
        "source_id", "source_stream_sha256", "content_chunks",
        "metadata_chunks", "episode_ids", "unit_ids", "relation_ids",
        "episode_build_sha256", "discourse_link_sha256",
        "surprise_signal_receipt_sha256",
        "returned_signal_transformer_state_bytes", "receipt_sha256",
    }
    sources = []
    for source_body in source_bodies:
        if not isinstance(source_body, dict):
            raise ValueError("compilation source receipt is malformed")
        _keys(source_body, source_keys, "compilation source receipt")
        source = DiffuseSourceCompilationReceipt(**source_body)
        if (
            _jsonable(source.identity_payload()) != source_body
            or source.returned_signal_transformer_state_bytes != 0
        ):
            raise ValueError("compilation source identity or zero-state claim changed")
        sources.append(source)
    artifact = DiscourseArtifact(**artifact_body)
    snapshot_body = _body(row.final_snapshot)
    snapshot = DiscourseSnapshot(**snapshot_body)
    compilation = DiffuseCompilationReceipt(
        artifact=artifact,
        compilation_policy_sha256=body["compilation_policy_sha256"],
        policy_sha256=body["policy_sha256"],
        source_receipts=tuple(sources),
        episode_coverage_receipt_sha256=body[
            "episode_coverage_receipt_sha256"
        ],
        discourse_coverage_receipt_sha256=body[
            "discourse_coverage_receipt_sha256"
        ],
        final_snapshot=snapshot,
        persisted_request_token_state_bytes=body[
            "persisted_request_token_state_bytes"
        ],
        format=body["format"],
        receipt_sha256=body["receipt_sha256"],
    )
    if (
        body.get("format") != DIFFUSE_COMPILATION_FORMAT
        or body.get("compilation_policy_sha256") != arm.compilation.policy_sha256
        or body.get("final_snapshot_sha256") != row.final_snapshot.identity_sha256
        or row.finalization.final_snapshot_sha256
        != row.final_snapshot.identity_sha256
        or _jsonable(compilation.identity_payload()) != body
    ):
        raise ValueError("compilation receipt is not authoritative for its arm")
    return compilation


def _owned_representative_plan(query) -> EpisodeRepresentativeRetrievalPlan:
    body = _body(query.representative_expansion)
    scans = tuple(EpisodeSourceScan(**item) for item in body["source_scans"])
    witnesses = tuple(
        EpisodeRepresentativeWitness(**item)
        for item in body["candidate_witnesses"]
    )
    seeds = tuple(EpisodeSeed(**item) for item in body["seeds"])
    plan = EpisodeRepresentativeRetrievalPlan(
        **{
            **body,
            "source_scans": scans,
            "candidate_witnesses": witnesses,
            "seeds": seeds,
        }
    )
    scope = _body(query.source_scope)
    candidates = {item["source_id"]: item for item in scope["candidates"]}
    witness_by_episode = {item.episode_id: item for item in witnesses}
    if (
        _jsonable(plan.identity_payload()) != body
        or any(item.status not in {"ok", "lookup_error", "identity_error"} for item in scans)
        or any(item.source_id not in candidates for item in scans)
        or any(
            item.source_id not in candidates
            or item.source_score != candidates[item.source_id]["score"]
            or item.source_route != candidates[item.source_id]["route"]
            for item in witnesses
        )
        or any(
            item.route != "episode_representative_qwen"
            or item.anchor_chunk_id
            != witness_by_episode[item.episode_id].anchor_chunk_id
            or item.path
            != (
                item.anchor_chunk_id,
                item.episode_id,
                f"source_route:{witness_by_episode[item.episode_id].source_route}",
                "qwen_nested_representative",
            )
            for item in seeds
        )
    ):
        raise ValueError("representative expansion is not an owned plan")
    return plan


def _validate_runtime(receipt, provider: dict[str, object]) -> tuple[dict, dict]:
    runtime = _body(receipt.runtime_binding)
    _keys(
        runtime,
        {
            "format", "runtime_binding_certified", "residency_mode",
            "resident_preflight", "embedding", "qwen", "source_router",
            "representative", "retrieval_policy_sha256", "factories",
        },
        "runtime binding",
    )
    router = runtime.get("source_router")
    preflight = runtime.get("resident_preflight")
    qwen = runtime.get("qwen")
    representative = runtime.get("representative")
    factories = runtime.get("factories")
    if not all(
        isinstance(item, dict)
        for item in (router, preflight, qwen, representative, factories)
    ):
        raise ValueError("runtime binding has malformed controls")
    _keys(router, {"max_sources", "rrf_constant"}, "runtime source router")
    _keys(preflight, {"policy", "required_free_bytes"}, "runtime preflight")
    _keys(
        qwen,
        {
            "model_locator", "model_id", "model_revision",
            "checkpoint_sha256", "prefix_layers", "attention_layer",
            "device", "dtype", "max_candidates", "max_workspace_tokens",
            "surprise",
        },
        "runtime Qwen",
    )
    surprise = qwen.get("surprise")
    if not isinstance(surprise, dict):
        raise ValueError("runtime Qwen surprise controls are malformed")
    _keys(
        surprise,
        {"max_spans", "span_token_cap", "probe_token_cap", "max_transport_dimension"},
        "runtime Qwen surprise",
    )
    _keys(
        representative,
        {
            "max_input_sources", "max_source_groups",
            "max_episodes_per_source", "max_total_episodes",
            "max_representatives_per_episode", "group_size",
            "beam_per_group", "top_k", "representative_tokens",
            "query_tokens", "score_mode",
        },
        "runtime representative controls",
    )
    factory_names = {
        "embedding", "condenser", "qwen_encoder", "qwen_linker",
        "qwen_scorer", "qwen_reranker", "resident_preflight",
    }
    _keys(factories, factory_names, "runtime factories")
    if any(
        not isinstance(value, dict)
        or set(value) != {"callable", "python_code_sha256"}
        for value in factories.values()
    ):
        raise ValueError("runtime factory identity schema changed")
    retrieval = _body(receipt.retrieval_policy)
    embedding = runtime.get("embedding")
    embedding_device = (
        None
        if not isinstance(embedding, dict)
        else _cuda_device(embedding.get("device"))
    )
    qwen_device = _cuda_device(qwen.get("device"))
    if (
        runtime.get("runtime_binding_certified") is not True
        or runtime.get("residency_mode") != "resident_bge_qwen"
        or runtime.get("retrieval_policy_sha256")
        != receipt.retrieval_policy.identity_sha256
        or runtime.get("embedding")
        != receipt.base_manifest.embedding_identity.model_dump(mode="json")
        or preflight.get("policy") != "cuda-mem-get-info-min-free-v1"
        or router.get("max_sources") != provider.get("max_sources")
        or router.get("rrf_constant") != provider.get("rrf_constant")
        or qwen.get("model_locator") != "local-verified-checkpoint"
        or qwen.get("prefix_layers")
        != retrieval.get("qwen_rerank_prefix_layers")
        or qwen.get("attention_layer")
        != retrieval.get("qwen_rerank_attention_layer")
        or embedding_device is None
        or qwen_device is None
        or embedding_device != qwen_device
    ):
        raise ValueError("runtime binding is not the authoritative resident policy")
    matched = _body(receipt.matched_runtime_suite)
    _keys(
        matched,
        {
            "format", "sample_id", "runtime_binding_sha256",
            "runtime_binding_certified", "residency_policy",
            "residency_device", "required_free_bytes",
            "runtime_result_receipt_sha256s", "preflight_observations",
            "matched_suite_receipt_sha256", "receipt_sha256",
        },
        "matched runtime suite",
    )
    if (
        matched.get("runtime_binding_sha256")
        != receipt.runtime_binding.identity_sha256
        or matched.get("runtime_binding_certified") is not True
        or matched.get("matched_suite_receipt_sha256")
        != receipt.matched_phase_suite.identity_sha256
        or matched.get("residency_policy") != preflight.get("policy")
        or matched.get("residency_device") != qwen.get("device")
        or matched.get("required_free_bytes")
        != preflight.get("required_free_bytes")
    ):
        raise ValueError("matched runtime is not bound to resident execution")
    return runtime, matched


def _validate_probe(
    probe: dict,
    rows: tuple,
    provider_sha256: str,
    episode_policy_sha256: str,
) -> None:
    _keys(
        probe,
        {
            "question_id", "question_probe_sha256", "retrieval_query_sha256",
            "retrieval_policy_sha256", "anchor_sequence_sha256",
            "anchor_chunk_ids", "source_candidate_sequence_sha256",
            "source_candidate_ids", "source_scope_identity_sha256",
            "legacy_input_provider_identity_sha256",
            "representative_linker_identity_sha256",
            "representative_policy_factory_identity_sha256",
            "representative_policy_controls_sha256", "episode_policy_sha256",
            "closure_policy_sha256", "format", "receipt_sha256",
        },
        "matched probe",
    )
    if identity_sha256({
        key: value for key, value in probe.items() if key != "receipt_sha256"
    }) != probe.get("receipt_sha256"):
        raise ValueError("matched probe self identity changed")
    first = rows[0]
    legacy = _body(first.legacy_input)
    analysis = _body(first.analysis_query)
    expected = {
        "question_probe_sha256": first.question_probe_sha256,
        "retrieval_query_sha256": first.query_receipt.retrieval_query_sha256,
        "retrieval_policy_sha256": legacy["retrieval_policy_sha256"],
        "anchor_sequence_sha256": legacy["anchor_sequence_sha256"],
        "anchor_chunk_ids": list(first.query_receipt.input_anchor_chunk_ids),
        "source_candidate_sequence_sha256": legacy[
            "source_candidate_sequence_sha256"
        ],
        "source_candidate_ids": legacy["source_candidate_ids"],
        "source_scope_identity_sha256": (
            first.matched_source_scope_identity_sha256
        ),
        "legacy_input_provider_identity_sha256": provider_sha256,
        "representative_linker_identity_sha256": analysis[
            "representative_linker_identity_sha256"
        ],
        "representative_policy_factory_identity_sha256": analysis[
            "representative_policy_factory_identity_sha256"
        ],
        "representative_policy_controls_sha256": analysis[
            "representative_policy_controls_sha256"
        ],
        "episode_policy_sha256": episode_policy_sha256,
        "closure_policy_sha256": first.query_receipt.closure_policy_sha256,
    }
    changed = tuple(
        key for key, value in expected.items() if probe.get(key) != value
    )
    if changed:
        raise ValueError(f"matched probe changed {changed[0]}")
    if identity_sha256({"question_id": probe.get("question_id")}) != (
        first.question_id_sha256
    ):
        raise ValueError("matched probe changed question_id")
    for row in rows[1:]:
        row_legacy = _body(row.legacy_input)
        row_analysis = _body(row.analysis_query)
        if (
            row.question_id_sha256 != first.question_id_sha256
            or row.question_probe_sha256 != first.question_probe_sha256
            or row.matched_source_scope_identity_sha256
            != first.matched_source_scope_identity_sha256
            or probe.get("retrieval_query_sha256")
            != row.query_receipt.retrieval_query_sha256
            or probe.get("retrieval_policy_sha256")
            != row_legacy["retrieval_policy_sha256"]
            or probe.get("anchor_sequence_sha256")
            != row_legacy["anchor_sequence_sha256"]
            or tuple(probe.get("anchor_chunk_ids", ()))
            != row.query_receipt.input_anchor_chunk_ids
            or probe.get("source_candidate_sequence_sha256")
            != row_legacy["source_candidate_sequence_sha256"]
            or tuple(probe.get("source_candidate_ids", ()))
            != tuple(row_legacy["source_candidate_ids"])
            or probe.get("legacy_input_provider_identity_sha256")
            != row_analysis["legacy_input_provider_identity_sha256"]
            or probe.get("representative_linker_identity_sha256")
            != row_analysis["representative_linker_identity_sha256"]
            or probe.get("representative_policy_factory_identity_sha256")
            != row_analysis["representative_policy_factory_identity_sha256"]
            or probe.get("representative_policy_controls_sha256")
            != row_analysis["representative_policy_controls_sha256"]
            or probe.get("closure_policy_sha256")
            != row.query_receipt.closure_policy_sha256
        ):
            raise ValueError("matched arms changed a probe coordinate")


def validate_replay_crosslinks(receipt) -> None:
    """Validate exact runtime, phase, policy, and matched-probe joins."""

    provider_outer = _body(receipt.verified_base_provider_identity)
    provider = provider_outer["declared_identity"]
    first_queries = receipt.arms[0].queries
    frozen_receipts = [item.frozen_input.identity_sha256 for item in first_queries]
    if (
        identity_sha256(frozen_receipts)
        != receipt.query_manifest.frozen_receipts_sha256
        or provider.get("ordered_frozen_receipts_sha256")
        != receipt.query_manifest.frozen_receipts_sha256
        or any(
            _body(item.frozen_input).get("source_streams_sha256")
            != receipt.base_manifest.source_streams_sha256
            for item in first_queries
        )
    ):
        raise ValueError("provider/frozen aggregate differs from query manifest")
    runtime_binding, matched_runtime = _validate_runtime(receipt, provider)
    matched_phase = _body(receipt.matched_phase_suite)
    _keys(
        matched_phase,
        {
            "format", "sample_id", "corpus_sha256",
            "deterministic_turn_ids_sha256", "evaluation_policy_sha256",
            "matched_controls_sha256", "pipeline_modes",
            "pipeline_arm_sha256s", "compilation_receipt_sha256s",
            "retrieval_phase_receipt_sha256s", "probes",
            "qwen_source_signal_receipt_sha256s",
            "qwen_owned_representative_runtime",
            "zero_returned_transformer_state",
            "zero_persisted_transformer_state", "receipt_sha256",
        },
        "matched phase suite",
    )
    typed_probes = tuple(
        DiffuseLongMemEvalMatchedProbeReceipt(**item)
        for item in matched_phase["probes"]
    )
    typed_phase = DiffuseLongMemEvalMatchedSuiteReceipt(
        **{**matched_phase, "probes": typed_probes}
    )
    if _jsonable(typed_phase.identity_payload()) != matched_phase:
        raise ValueError("matched phase body is not authoritative")
    typed_preflights = tuple(
        ResidencyPreflightObservation(
            **{key: value for key, value in item.items() if key != "receipt_sha256"}
        )
        for item in matched_runtime["preflight_observations"]
    )
    typed_runtime = DiffuseLongMemEvalMatchedRuntimeSuiteReceipt(
        **{
            **{
                key: value for key, value in matched_runtime.items()
                if key != "matched_suite_receipt_sha256"
            },
            "preflight_observations": typed_preflights,
            "matched_suite": typed_phase,
        }
    )
    if _jsonable(typed_runtime.identity_payload()) != matched_runtime:
        raise ValueError("matched runtime body is not authoritative")
    arms = tuple(_owned_arm(item) for item in receipt.arms)
    qwen_compilation = _body(receipt.arms[-1].compilation)
    qwen_signals = tuple(
        item["surprise_signal_receipt_sha256"]
        for item in qwen_compilation["source_receipts"]
        if item["content_chunks"] > 0
    )
    if (
        tuple(matched_phase.get("pipeline_modes", ())) != _MODES
        or tuple(matched_phase.get("pipeline_arm_sha256s", ()))
        != tuple(item.arm_sha256 for item in arms)
        or tuple(matched_phase.get("compilation_receipt_sha256s", ()))
        != tuple(item.compilation.identity_sha256 for item in receipt.arms)
        or tuple(matched_phase.get("retrieval_phase_receipt_sha256s", ()))
        != tuple(item.retrieval_phase.identity_sha256 for item in receipt.arms)
        or matched_phase.get("corpus_sha256") != receipt.base_manifest.corpus_sha256
        or matched_phase.get("deterministic_turn_ids_sha256")
        != receipt.base_manifest.deterministic_turn_ids_sha256
        or matched_phase.get("evaluation_policy_sha256")
        != receipt.evaluation_policy.identity_sha256
        or identity_sha256({"sample_id": matched_phase.get("sample_id")})
        != receipt.sample_id_sha256
        or matched_runtime.get("sample_id") != matched_phase.get("sample_id")
        or matched_phase.get("matched_controls_sha256")
        != _matched_arm_sha256(arms[0])
        or any(_matched_arm_sha256(item) != _matched_arm_sha256(arms[0]) for item in arms)
        or tuple(matched_phase.get("qwen_source_signal_receipt_sha256s", ()))
        != qwen_signals
        or any(
            _body(item.compilation).get("persisted_request_token_state_bytes") != 0
            or any(
                query.query_receipt.store_retained_request_token_state_bytes != 0
                for query in item.queries
            )
            for item in receipt.arms
        )
    ):
        raise ValueError("matched phase suite differs from top-level arms/base")
    preflights = tuple(matched_runtime.get("preflight_observations", ()))
    if len(preflights) != len(receipt.arms):
        raise ValueError("matched runtime preflight count changed")
    for index, (row, arm) in enumerate(zip(receipt.arms, arms, strict=True)):
        compilation = _body(row.compilation)
        _keys(
            compilation,
            {
                "format", "artifact", "compilation_policy_sha256",
                "policy_sha256", "source_receipts",
                "episode_coverage_receipt_sha256",
                "discourse_coverage_receipt_sha256", "final_snapshot_sha256",
                "persisted_request_token_state_bytes", "receipt_sha256",
            },
            "compilation receipt",
        )
        _owned_compilation(row, arm)
        phase = _body(row.retrieval_phase)
        runtime = _body(row.runtime_result)
        _keys(
            phase,
            {
                "format", "sample_id", "corpus_sha256",
                "deterministic_ingest_format", "deterministic_turn_ids",
                "analysis_arm_sha256", "matched_controls_sha256",
                "evaluation_policy_sha256", "compilation_receipt_sha256",
                "question_receipt_sha256s", "receipt_sha256",
            },
            "retrieval phase",
        )
        _keys(
            runtime,
            {
                "format", "retrieval_phase_receipt_sha256",
                "runtime_binding_sha256", "runtime_binding_certified",
                "residency_preflight", "residency_preflight_receipt_sha256",
                "receipt_sha256",
            },
            "runtime result",
        )
        observed_preflight = dict(preflights[index])
        preflight_receipt = observed_preflight.pop("receipt_sha256", None)
        artifact_id = compilation["artifact"]["artifact_id"]
        episode_active = asdict(arm.episode)
        episode_active["artifact_id"] = artifact_id
        if (
            phase.get("format") != DIFFUSE_ANALYSIS_PHASE_FORMAT
            or phase.get("sample_id") != matched_phase.get("sample_id")
            or phase.get("deterministic_ingest_format")
            != DETERMINISTIC_DIFFUSE_INGEST_FORMAT
            or identity_sha256(phase.get("deterministic_turn_ids"))
            != receipt.base_manifest.deterministic_turn_ids_sha256
            or phase.get("analysis_arm_sha256") != arm.arm_sha256
            or phase.get("matched_controls_sha256") != _matched_arm_sha256(arm)
            or phase.get("evaluation_policy_sha256")
            != receipt.evaluation_policy.identity_sha256
            or phase.get("compilation_receipt_sha256")
            != row.compilation.identity_sha256
            or tuple(phase.get("question_receipt_sha256s", ()))
            != tuple(item.analysis_query.identity_sha256 for item in row.queries)
            or phase.get("corpus_sha256") != receipt.base_manifest.corpus_sha256
            or runtime.get("format") != DIFFUSE_RUNTIME_RESULT_FORMAT
            or runtime.get("retrieval_phase_receipt_sha256")
            != row.retrieval_phase.identity_sha256
            or runtime.get("runtime_binding_sha256")
            != receipt.runtime_binding.identity_sha256
            or runtime.get("runtime_binding_certified") is not True
            or runtime.get("residency_preflight") != observed_preflight
            or runtime.get("residency_preflight_receipt_sha256")
            != preflight_receipt
            or identity_sha256(observed_preflight) != preflight_receipt
            or observed_preflight.get("embedding_released_before_qwen_load")
            is not False
            or any(
                item.query_receipt.episode_policy_sha256
                != identity_sha256(episode_active)
                or item.query_receipt.closure_policy_sha256
                != arm.closure.policy_sha256
                for item in row.queries
            )
        ):
            raise ValueError("arm phase/runtime/policy lineage changed")
        for query in row.queries:
            analysis = _body(query.analysis_query)
            try:
                authoritative = DiffuseLongMemEvalAnalysisQueryReceipt(**analysis)
            except Exception as exc:
                raise ValueError("analysis query body is not authoritative") from exc
            if authoritative.identity_payload() != analysis:
                raise ValueError("analysis query identity schema changed")
            direct = _body(query.direct_expansion)
            representative = _body(query.representative_expansion)
            representative_plan = _owned_representative_plan(query)
            representative_policy = EpisodeRepresentativeRetrievalPolicy(
                artifact_id=compilation["artifact"]["artifact_id"],
                **runtime_binding["representative"],
            )
            scope = _body(query.source_scope)
            _keys(
                scope,
                {
                    "artifact_id", "snapshot_sha256", "source_revision",
                    "source_content_sha256", "query_sha256",
                    "router_policy_sha256", "universe_source_ids", "candidates",
                    "truncated_source_ids", "universe_enumerated", "receipt_sha256",
                },
                "source scope",
            )
            if any(
                not isinstance(item, dict)
                or set(item) != {"source_id", "score", "route"}
                for item in scope["candidates"]
            ):
                raise ValueError("source scope candidate schema changed")
            representative_exhaustive = (
                representative_plan.candidate_scope_exhaustive
            )
            expansion_exhaustive = bool(
                not direct["truncated_episode_ids"]
                and not direct["truncated_direct_chunk_ids"]
                and representative_exhaustive
            )
            if (
                analysis["corpus_sha256"] != receipt.base_manifest.corpus_sha256
                or analysis["question_probe_sha256"]
                != query.question_probe_sha256
                or analysis["analysis_arm_sha256"] != arm.arm_sha256
                or analysis["matched_controls_sha256"]
                != _matched_arm_sha256(arm)
                or analysis["evaluation_policy_sha256"]
                != receipt.evaluation_policy.identity_sha256
                or analysis["compilation_receipt_sha256"]
                != row.compilation.identity_sha256
                or analysis["artifact_id"] != compilation["artifact"]["artifact_id"]
                or analysis["snapshot_sha256"]
                != row.final_snapshot.identity_sha256
                or representative_plan.policy_sha256
                != representative_policy.policy_sha256
                or analysis["representative_policy_sha256"]
                != representative_policy.policy_sha256
                or analysis["representative_policy_controls_sha256"]
                != replace(
                    representative_policy,
                    artifact_id="matched-artifact",
                ).policy_sha256
                or representative_plan.max_workspace_candidates
                > runtime_binding["qwen"]["max_candidates"]
                or representative_plan.max_workspace_tokens
                > runtime_binding["qwen"]["max_workspace_tokens"]
                or representative_policy.group_size
                > runtime_binding["qwen"]["max_candidates"]
                or representative_policy.beam_per_group
                >= runtime_binding["qwen"]["max_candidates"]
                or query.query_receipt.representative_scope_exhaustive
                != representative_exhaustive
                or query.query_receipt.expansion_exhaustive != expansion_exhaustive
                or query.query_receipt.representative_runtime_binding_certified
                is not True
                or query.query_receipt.representative_returned_plan_transformer_state_bytes
                != 0
                or analysis["legacy_input_receipt_sha256"]
                != query.legacy_input.identity_sha256
                or analysis["diffuse_query_receipt_sha256"]
                != query.query_receipt.receipt_sha256
                or analysis["representative_linker_identity_sha256"]
                != representative["linker_identity_sha256"]
                or analysis["representative_policy_sha256"]
                != representative["policy_sha256"]
                or analysis["legacy_input_provider_identity_sha256"]
                != receipt.verified_base_provider_identity.identity_sha256
            ):
                raise ValueError("analysis/expansion exhaustiveness lineage changed")
    probes = tuple(matched_phase.get("probes", ()))
    if len(probes) != receipt.query_manifest.query_count:
        raise ValueError("matched probe count changed")
    for index, probe in enumerate(probes):
        _validate_probe(
            probe,
            tuple(arm.queries[index] for arm in receipt.arms),
            receipt.verified_base_provider_identity.identity_sha256,
            arms[0].episode.policy_sha256,
        )


__all__ = ["validate_replay_crosslinks"]
