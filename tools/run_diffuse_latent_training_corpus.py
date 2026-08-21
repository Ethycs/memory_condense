"""Closed public launcher and private route-v2 candidate audit plumbing.

The public launcher is disabled before it touches caller arguments because the
writable derived-store open, lease, and finalization lifecycle does not yet
satisfy this launcher's owned-capability boundary. The private plumbing remains
available for static audit only. It does not authenticate an execution or
authorize D1.

Run from the repository root with ``python -m``.  Cold import does not resolve
the evaluator scoring schema, provider SDKs, torch/transformers, or model
weights.  Lightweight modeling identity modules may load transitively.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Callable

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.diffuse_latent_training_corpus import (
    AnalysisPopulationProjection,
    AnalysisPopulationRow,
    StructuralRouteV2MappedRow,
    latent_training_corpus_implementation_sha256,
    publish_structural_latent_training_corpus,
    verify_structural_latent_training_corpus,
)
from tools._diffuse_latent_training_corpus_authority_filesystem import (
    candidate_staging_path,
    capture_candidate_generic,
    cleanup_candidate_staging,
    create_candidate_staging,
    publish_candidate_root,
    recheck_captured_candidate_generic,
    verify_latent_training_corpus_candidate,
)
from tools._diffuse_latent_training_corpus_authority_models import (
    CANDIDATE_EXECUTION_DISABLED_REASON,
    DeclaredProductionExecutionCoordinates,
    ProductionCandidateExecutionStatus,
    ProductionCandidateExecutionUnavailable,
    ProductionCorpusCandidateReceipt,
    ProductionLatentTrainingCorpusError,
    VerifiedLatentTrainingCorpusCandidate,
    locked_production_external_lock,
)
from tools._diffuse_latent_training_corpus_workspace import (
    OwnedCandidateWorkspace,
    candidate_workspace_path,
    capture_candidate_execution_workspace,
    capture_candidate_row_workspace,
    cleanup_candidate_workspace,
    create_candidate_workspace,
)
from tools.v4_population_firebreak import (
    AnalysisTreatmentInput,
    TreatmentSample,
    load_analysis_treatment_input,
)


_LAUNCHER_RELATIVE_PATH = "tools/run_diffuse_latent_training_corpus.py"
_EXPECTED_SAMPLE_COUNT = 300


@dataclass(frozen=True, slots=True)
class _ImplementationProjection:
    package_sha256: str
    corpus_sha256: str
    route_sha256: str


@dataclass(frozen=True, slots=True)
class _CheckpointProjection:
    bge_sha256: str
    qwen_retrieval_sha256: str
    qwen_feature_sha256: str
    qwen_retrieval_contract_sha256: str
    qwen_feature_contract_sha256: str


@dataclass(frozen=True, slots=True)
class _RuntimeProjection:
    binding_sha256: str
    linker_sha256: str
    factory_sha256: str


def _absolute_child(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if not path.name or path.name in {".", ".."}:
        raise ValueError(f"{label} must name one bounded filesystem child")
    return path


def _require_offline_environment() -> None:
    missing = [
        name
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        if os.environ.get(name) != "1"
    ]
    if missing:
        raise RuntimeError(
            "production candidate execution requires exact offline variables: "
            + ", ".join(missing)
        )
    if os.environ.get("KMP_DUPLICATE_LIB_OK") != "TRUE":
        raise RuntimeError(
            "production candidate execution requires KMP_DUPLICATE_LIB_OK=TRUE"
        )


def _campaign_config() -> Any:
    """Reconstruct the locked retrieval config without importing at startup."""

    from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig

    retrieval = RetrievalConfig(
        mode="hybrid_graph",
        k=10,
        ef_search=50,
        candidates=100,
        alpha=0.65,
        neighbor_radius=5,
        neighbor_slots=24,
        neighbor_direction="next",
        source_slots=48,
        source_candidate_pool=750,
        source_activation_k=65,
        role_aware_retrieval=True,
        source_tfisf_activation=True,
        source_tfisf_slots=8,
        source_hsc_activation=True,
        source_hsc_slots=8,
        source_hsc_hops=2,
        source_hsc_chunk_slots=4,
        source_local_search=True,
        source_partition_routing=True,
        source_partition_slots=4,
        qwen_rerank=False,
        qwen_feedback=False,
        qwen_rerank_prefix_layers=2,
        qwen_rerank_attention_layer=1,
        qwen_rerank_max_workspace_tokens=2048,
    )
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=120, max_tokens=250),
        retrieval=retrieval,
        embedding_device="cuda:0",
        max_prompt_tokens=8000,
    )


def _reference_arm() -> Any:
    """Reconstruct the locked fixed-interval episode-primary base arm."""

    from memory_condense.domain.discourse import ClosurePolicy
    from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
    from memory_condense.eval.diffuse_longmemeval_analysis import DiffuseLongMemEvalArm
    from memory_condense.eval.diffuse_longmemeval_route_v2 import (
        EpisodePrimaryAnalysisArmV2,
    )
    from memory_condense.search.episodes import EpisodeRetrievalPolicy

    base = DiffuseLongMemEvalArm(
        arm_id="fixed_interval",
        compilation=DiffuseCompilationPolicy(boundary_mode="fixed_interval"),
        episode=EpisodeRetrievalPolicy(
            max_anchor_episodes=96,
            previous_episodes=1,
            next_episodes=1,
            max_episode_seeds=256,
            max_direct_fallbacks=96,
        ),
        closure=ClosurePolicy(
            max_hops=3,
            max_units=1024,
            max_relations=2048,
            max_degree=32,
            max_episode_neighbors=2,
            max_frontier=1024,
            max_bundles=256,
            beam_width=128,
            min_relation_confidence=0.5,
        ),
        max_context_tokens=7000,
        responder_output_token_reserve=256,
        require_owned_representative_runtime=True,
    )
    return EpisodePrimaryAnalysisArmV2(base_arm=base)


def _runtime_config(qwen_model_dir: Path) -> Any:
    from memory_condense.eval.diffuse_longmemeval_runtime import (
        DiffuseLongMemEvalRuntimeConfig,
    )

    return DiffuseLongMemEvalRuntimeConfig(
        qwen_model_dir=qwen_model_dir,
        residency_mode="resident_bge_qwen",
        embedding_batch_size=32,
        qwen_device="cuda:0",
        qwen_dtype="float16",
        qwen_max_candidates=8,
        qwen_max_workspace_tokens=2048,
        resident_min_free_mib=3072,
        source_router_max_sources=64,
        source_router_rrf_constant=60,
        surprise_max_spans=256,
        surprise_span_tokens=64,
        surprise_probe_tokens=96,
        surprise_max_transport_dimension=8192,
        representative_max_input_sources=64,
        representative_max_source_groups=64,
        representative_max_episodes_per_source=64,
        representative_max_total_episodes=256,
        representative_max_per_episode=2,
        representative_group_size=8,
        representative_beam_per_group=2,
        representative_top_k=8,
        representative_tokens=96,
        representative_query_tokens=96,
        representative_score_mode="qk_ov",
    )


def _new_owned_binding(qwen_model_dir: Path) -> Any:
    from memory_condense.eval.diffuse_longmemeval_runtime import (
        DiffuseLongMemEvalExecutionBinding,
        build_diffuse_longmemeval_execution_binding,
    )

    binding = build_diffuse_longmemeval_execution_binding(
        config=_campaign_config(),
        runtime=_runtime_config(qwen_model_dir),
    )
    if type(binding) is not DiffuseLongMemEvalExecutionBinding:
        raise TypeError("owned runtime factory returned another binding type")
    if binding.runtime_binding_certified is not True:
        raise RuntimeError("owned runtime binding is not certified")
    if binding.binding_sha256 != identity_sha256(
        dict(binding.analysis_identity_payload())
    ):
        raise RuntimeError("owned runtime binding is not content addressed")
    return binding


def _assert_control_lock(config: Any, arm: Any, binding: Any) -> None:
    from memory_condense.eval.diffuse_longmemeval_analysis import (
        _evaluation_policy_sha256,
    )
    from memory_condense.search.fusion.models import FusionCaps
    from memory_condense.search.fusion.qwen_feature_models import QwenAtomFeatureCaps

    lock = locked_production_external_lock()
    base = arm.base_arm
    representative = binding.representative_policy_factory("matched-artifact")
    observed = {
        "compilation policy": base.compilation.policy_sha256,
        "episode policy": base.episode.policy_sha256,
        "closure policy": base.closure.policy_sha256,
        "base arm": base.arm_sha256,
        "episode-primary arm": arm.arm_sha256,
        "matched controls": base.matched_controls_sha256,
        "representative controls": representative.policy_sha256,
        "retrieval config": identity_sha256(
            config.retrieval.model_dump(mode="json")
        ),
        "evaluation policy": _evaluation_policy_sha256(config),
        "fusion caps": FusionCaps().caps_sha256,
        "Qwen atom feature caps": QwenAtomFeatureCaps().caps_sha256,
    }
    expected = {
        "compilation policy": lock.compilation_policy_sha256,
        "episode policy": lock.episode_policy_sha256,
        "closure policy": lock.closure_policy_sha256,
        "base arm": lock.base_arm_sha256,
        "episode-primary arm": lock.episode_primary_arm_sha256,
        "matched controls": lock.matched_controls_sha256,
        "representative controls": lock.representative_policy_controls_sha256,
        "retrieval config": lock.retrieval_config_sha256,
        "evaluation policy": lock.evaluation_policy_sha256,
        "fusion caps": lock.fusion_caps_sha256,
        "Qwen atom feature caps": lock.qwen_atom_feature_caps_sha256,
    }
    changed = tuple(name for name, value in observed.items() if value != expected[name])
    if changed:
        raise RuntimeError("production candidate control lock changed: " + ", ".join(changed))
    payload = dict(binding.analysis_identity_payload())
    expected_runtime = {
        "residency_mode": lock.runtime_mode,
        "resident_preflight": {
            "policy": "cuda-mem-get-info-min-free-v1",
            "required_free_bytes": 3072 * 1024 * 1024,
        },
        "embedding": {
            "backend": "sentence-transformers.encode-v1",
            "model_id": lock.bge_model_id,
            "model_revision": lock.bge_model_revision,
            "checkpoint_sha256": lock.bge_checkpoint_sha256,
            "dimension": 1024,
            "device": lock.device,
            "batch_size": 32,
            "normalize_embeddings": False,
            "output_dtype": "float32",
        },
        "qwen": {
            "model_locator": "local-verified-checkpoint",
            "model_id": lock.qwen_model_id,
            "model_revision": lock.qwen_model_revision,
            "checkpoint_sha256": lock.qwen_checkpoint_sha256,
            "prefix_layers": lock.retrieval_prefix_layers,
            "attention_layer": lock.retrieval_attention_layer,
            "device": lock.device,
            "dtype": "float16",
            "max_candidates": 8,
            "max_workspace_tokens": 2048,
            "surprise": {
                "max_spans": 256,
                "span_token_cap": 64,
                "probe_token_cap": 96,
                "max_transport_dimension": 8192,
            },
        },
        "source_router": {"max_sources": 64, "rrf_constant": 60},
        "representative": {
            "max_input_sources": 64,
            "max_source_groups": 64,
            "max_episodes_per_source": 64,
            "max_total_episodes": 256,
            "max_representatives_per_episode": 2,
            "group_size": 8,
            "beam_per_group": 2,
            "top_k": 8,
            "representative_tokens": 96,
            "query_tokens": 96,
            "score_mode": "qk_ov",
        },
        "retrieval_policy_sha256": lock.retrieval_config_sha256,
    }
    if any(payload.get(name) != value for name, value in expected_runtime.items()):
        raise RuntimeError("production candidate runtime projection changed")


def _implementation_projection() -> _ImplementationProjection:
    from memory_condense.eval._diffuse_latent_training_corpus_route import (
        live_route_v2_implementation_sha256,
    )
    from memory_condense.eval.reproducibility import implementation_sha256

    return _ImplementationProjection(
        package_sha256=implementation_sha256(),
        corpus_sha256=latent_training_corpus_implementation_sha256(),
        route_sha256=live_route_v2_implementation_sha256(),
    )


def _qwen_contract(
    *, retained_layers: int, selected_layer_kind: str, selected_layer: int
) -> str:
    lock = locked_production_external_lock()
    return identity_sha256(
        {
            "format": "qwen-prefix-layer-contract-v1",
            "model_id": lock.qwen_model_id,
            "model_revision": lock.qwen_model_revision,
            "checkpoint_sha256": lock.qwen_checkpoint_sha256,
            "retained_layers": retained_layers,
            "selected_layer_kind": selected_layer_kind,
            "selected_layer": selected_layer,
        }
    )


def _verify_checkpoints(qwen_model_dir: Path) -> _CheckpointProjection:
    from memory_condense.modeling.embedding import verify_bge_m3_checkpoint
    from memory_condense.modeling.qwen_prefix import verify_prefix_checkpoint

    lock = locked_production_external_lock()
    bge = verify_bge_m3_checkpoint()
    retrieval = verify_prefix_checkpoint(
        qwen_model_dir,
        layers=lock.retrieval_prefix_layers,
        model_id=lock.qwen_model_id,
        model_revision=lock.qwen_model_revision,
        expected_checkpoint_sha256=lock.qwen_checkpoint_sha256,
    )
    feature = verify_prefix_checkpoint(
        qwen_model_dir,
        layers=lock.feature_prefix_layers,
        model_id=lock.qwen_model_id,
        model_revision=lock.qwen_model_revision,
        expected_checkpoint_sha256=lock.qwen_checkpoint_sha256,
    )
    if (
        bge != lock.bge_checkpoint_sha256
        or retrieval.checkpoint_sha256 != lock.qwen_checkpoint_sha256
        or feature.checkpoint_sha256 != lock.qwen_checkpoint_sha256
        or retrieval.model_id != lock.qwen_model_id
        or feature.model_id != lock.qwen_model_id
        or retrieval.model_revision != lock.qwen_model_revision
        or feature.model_revision != lock.qwen_model_revision
    ):
        raise RuntimeError("local checkpoint projection changed")
    retrieval_contract = _qwen_contract(
        retained_layers=lock.retrieval_prefix_layers,
        selected_layer_kind="attention",
        selected_layer=lock.retrieval_attention_layer,
    )
    feature_contract = _qwen_contract(
        retained_layers=lock.feature_prefix_layers,
        selected_layer_kind="output",
        selected_layer=lock.feature_output_layer,
    )
    if retrieval_contract == feature_contract:
        raise RuntimeError("retrieval and feature Qwen contracts collapsed")
    return _CheckpointProjection(
        bge,
        retrieval.checkpoint_sha256,
        feature.checkpoint_sha256,
        retrieval_contract,
        feature_contract,
    )


def _load_exact_treatment(
    path: Path,
    *,
    loader: Callable[..., object],
    input_type: type,
) -> AnalysisTreatmentInput:
    lock = locked_production_external_lock()
    value = loader(
        path,
        expected_file_sha256=lock.treatment_file_sha256,
        expected_sanitized_projection_sha256=lock.sanitized_projection_sha256,
        expected_dataset_sha256=lock.dataset_sha256,
        expected_split_manifest_sha256=lock.split_manifest_sha256,
        expected_ordered_question_ids_sha256=(
            lock.analysis_ordered_question_ids_sha256
        ),
        expected_sample_count=lock.fit_count + lock.validation_count,
    )
    if type(value) is not input_type or type(value) is not AnalysisTreatmentInput:
        raise TypeError("treatment loader returned another exact type")
    if type(value.samples) is not tuple or len(value.samples) != _EXPECTED_SAMPLE_COUNT:
        raise ValueError("treatment loader returned another population size")
    ids: list[str] = []
    for sample in value.samples:
        if type(sample) is not TreatmentSample or type(sample.questions) is not tuple:
            raise TypeError("treatment population contains another sample type")
        if len(sample.questions) != 1:
            raise ValueError("each treatment sample must contain exactly one query")
        ids.append(sample.questions[0].question_id)
    if (
        identity_sha256(ids) != lock.analysis_ordered_question_ids_sha256
        or identity_sha256(ids[: lock.fit_count])
        != lock.fit_ordered_question_ids_sha256
        or identity_sha256(ids[lock.fit_count :])
        != lock.validation_ordered_question_ids_sha256
    ):
        raise ValueError("treatment population partition projection changed")
    return value


def _certify_source_execution(
    launcher_path: Path,
    *,
    certifier: Callable[[Path], object],
) -> Any:
    """Certify HEAD and reject untracked code that source hashes can observe."""

    import subprocess

    execution = certifier(launcher_path)
    root_result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=launcher_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        raise RuntimeError("candidate source-root certification failed")
    root = Path(root_result.stdout.strip()).resolve()
    try:
        relative = launcher_path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("candidate launcher is outside its git worktree") from exc
    if relative != _LAUNCHER_RELATIVE_PATH:
        raise ValueError("candidate launcher has another tracked path")
    status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "src/memory_condense",
            "tools",
        ),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError(
            "candidate source closure is not clean, including untracked code"
        )
    return execution


def _population_projection(
    treatment: AnalysisTreatmentInput,
) -> AnalysisPopulationProjection:
    lock = locked_production_external_lock()
    return AnalysisPopulationProjection(
        treatment_file_sha256=treatment.file_sha256,
        sanitized_projection_sha256=treatment.sanitized_projection_sha256,
        dataset_sha256=treatment.dataset_sha256,
        split_manifest_sha256=treatment.split_manifest_sha256,
        ordered_question_ids=tuple(
            sample.questions[0].question_id for sample in treatment.samples
        ),
        excluded_confirmation_count=lock.excluded_confirmation_count,
        excluded_confirmation_ordered_question_ids_sha256=(
            lock.excluded_confirmation_ordered_question_ids_sha256
        ),
        source_treatment_exact_type_verified=False,
    )


def _treatment_identity(treatment: AnalysisTreatmentInput, ordinal: int) -> Any:
    from memory_condense.eval.diffuse_longmemeval_base import (
        DiffuseBaseTreatmentIdentity,
    )

    return DiffuseBaseTreatmentIdentity(
        treatment_file_sha256=treatment.file_sha256,
        sanitized_projection_sha256=treatment.sanitized_projection_sha256,
        dataset_sha256=treatment.dataset_sha256,
        split_manifest_sha256=treatment.split_manifest_sha256,
        ordered_question_ids_sha256=treatment.ordered_question_ids_sha256,
        sample_count=len(treatment.samples),
        sample_ordinal=ordinal,
    )


def _provider_identity(provider: object) -> str:
    from memory_condense.eval.diffuse_longmemeval_analysis import (
        analysis_callable_identity_payload,
    )

    return identity_sha256(
        analysis_callable_identity_payload(provider, "legacy_input_provider")
    )


def _runtime_projection(binding: Any, qwen: Any) -> _RuntimeProjection:
    from memory_condense.eval.diffuse_longmemeval_analysis import (
        _representative_linker_identity_sha256,
        analysis_callable_identity_payload,
    )

    linker = _representative_linker_identity_sha256(qwen.linker)
    if type(linker) is not str:
        raise RuntimeError("resident Qwen runtime has no linker identity")
    factory = identity_sha256(
        analysis_callable_identity_payload(
            binding.representative_policy_factory,
            "representative_policy_factory",
        )
    )
    if binding.runtime_binding_certified is not True:
        raise RuntimeError("owned runtime certification changed")
    return _RuntimeProjection(binding.binding_sha256, linker, factory)


class _RouteV2Mapper:
    """Sequential owned mapper that freshly binds one provider per row."""

    def __init__(
        self,
        *,
        treatment: AnalysisTreatmentInput,
        binding: Any,
        qwen: Any,
        arm: Any,
        cache_root: Path,
        workspace_root: Path,
        runtime: _RuntimeProjection,
    ) -> None:
        self.treatment = treatment
        self.binding = binding
        self.qwen = qwen
        self.arm = arm
        self.cache_root = cache_root
        self.workspace_root = workspace_root
        self.runtime = runtime
        self.provider_identities: list[str] = []

    def __call__(self, row: AnalysisPopulationRow) -> StructuralRouteV2MappedRow:
        from memory_condense.eval.diffuse_longmemeval_base import (
            clone_diffuse_longmemeval_base,
            finalize_diffuse_longmemeval_derived_store,
            open_diffuse_longmemeval_derived_store,
            owned_build_runtime_identity,
            publish_diffuse_longmemeval_base,
        )
        from memory_condense.eval._diffuse_base_derived import (
            _abort_diffuse_longmemeval_derived_store,
        )
        from memory_condense.eval._diffuse_base_publication_guard import (
            freeze_callable_guard,
        )
        from memory_condense.eval.diffuse_longmemeval_replay import (
            VerifiedBaseLegacyDiffuseInputProvider,
        )
        from memory_condense.eval.diffuse_longmemeval_route_v2 import (
            retrieve_episode_primary_analysis_phase_v2,
        )
        from memory_condense.eval.diffuse_longmemeval_runtime import (
            gold_blind_from_treatment_sample,
        )

        abort_clone = _abort_diffuse_longmemeval_derived_store
        assert_abort_clone = freeze_callable_guard(
            abort_clone,
            error_type=RuntimeError,
            label="derived candidate-row abort helper",
        )
        if type(row) is not AnalysisPopulationRow:
            raise TypeError("generic publisher supplied another row type")
        expected_ordinal = len(self.provider_identities)
        sample = self.treatment.samples[row.ordinal]
        question_id = sample.questions[0].question_id
        if row.ordinal != expected_ordinal or row.question_id != question_id:
            raise RuntimeError("generic publisher reordered the treatment population")
        blind = gold_blind_from_treatment_sample(sample)
        base = publish_diffuse_longmemeval_base(
            self.cache_root,
            treatment_identity=_treatment_identity(self.treatment, row.ordinal),
            sample=blind,
            config=self.binding.config,
            embedding_identity=self.binding.embedding_identity,
            build_runtime_identity=owned_build_runtime_identity(
                self.binding.new_condenser
            ),
            embedder=self.binding.embedder,
            condenser_factory=self.binding.new_condenser,
        )
        provider = VerifiedBaseLegacyDiffuseInputProvider.from_verified_base(
            base,
            max_sources=self.binding.runtime.source_router_max_sources,
            rrf_constant=self.binding.runtime.source_router_rrf_constant,
        )
        provider_sha256 = _provider_identity(provider)
        row_owner = create_candidate_workspace(
            self.workspace_root / f"row-{row.ordinal:06d}",
            kind="row",
        )
        row_root = candidate_workspace_path(row_owner)
        clone = None
        try:
            clone = clone_diffuse_longmemeval_base(
                base,
                row_root / "derived",
                arm_id=self.arm.base_arm.arm_id,
                arm_sha256=self.arm.base_arm.arm_sha256,
            )
            condenser = open_diffuse_longmemeval_derived_store(
                clone,
                config=self.binding.config,
                embedder=self.binding.embedder,
            )
            try:
                if self.qwen.reranker is not None:
                    raise RuntimeError(
                        "candidate route loaded a legacy Qwen reranker"
                    )
                phase = retrieve_episode_primary_analysis_phase_v2(
                    condenser,
                    blind,
                    config=self.binding.config,
                    arm=self.arm,
                    legacy_input_provider=provider,
                    representative_linker=self.qwen.linker,
                    representative_policy_factory=(
                        self.binding.representative_policy_factory
                    ),
                    qwen_scorer=None,
                    embedding_identity=self.binding.embedding_identity,
                )
            except BaseException as original:
                try:
                    condenser.close()
                except BaseException as close_error:
                    original.add_note(
                        f"derived condenser close also failed: {close_error!r}"
                    )
                raise
            else:
                condenser.close()
            finalize_diffuse_longmemeval_derived_store(clone, phase=phase)
            inner = phase.questions[0].inner
            receipt = inner.receipt
            if (
                receipt.legacy_input_provider_identity_sha256 != provider_sha256
                or receipt.representative_linker_identity_sha256
                != self.runtime.linker_sha256
                or receipt.representative_policy_factory_identity_sha256
                != self.runtime.factory_sha256
                or _provider_identity(provider) != provider_sha256
            ):
                raise RuntimeError("route-v2 row changed an owned runtime identity")
            representative = inner.retrieval.representative_expansion
            if representative is None:
                raise RuntimeError("route-v2 row omitted representative expansion")
            mapped = StructuralRouteV2MappedRow(
                phase=phase,
                representative_policy=self.binding.representative_policy_factory(
                    representative.artifact_id
                ),
            )
            capture_candidate_row_workspace(row_owner)
            self.provider_identities.append(provider_sha256)
        except BaseException as original:
            if clone is not None:
                try:
                    assert_abort_clone(abort_clone)
                    abort_clone(clone)
                except BaseException as cleanup_error:
                    original.add_note(
                        "derived lifecycle abort also failed: "
                        f"{cleanup_error!r}"
                    )
            try:
                cleanup_candidate_workspace(row_owner)
            except BaseException as cleanup_error:
                original.add_note(
                    f"exact row-workspace cleanup was refused: {cleanup_error!r}"
                )
            raise
        cleanup_candidate_workspace(row_owner)
        return mapped

    def finish(self) -> tuple[str, ...]:
        if len(self.provider_identities) != _EXPECTED_SAMPLE_COUNT:
            raise RuntimeError("generic publisher did not map exactly 300 rows")
        return tuple(self.provider_identities)


def _verify_source_projection(
    generic: Any,
    treatment: AnalysisTreatmentInput,
    *,
    binding: Any,
    qwen: Any,
    cache_root: Path,
    expected_provider_identities: tuple[str, ...],
) -> _RuntimeProjection:
    from memory_condense.eval.diffuse_longmemeval_analysis import (
        analysis_callable_identity_payload,
    )
    from memory_condense.eval.diffuse_longmemeval_base import (
        owned_build_runtime_identity,
        verify_diffuse_longmemeval_base,
    )
    from memory_condense.eval.diffuse_longmemeval_replay import (
        VerifiedBaseLegacyDiffuseInputProvider,
    )
    from memory_condense.eval.diffuse_longmemeval_runtime import (
        gold_blind_from_treatment_sample,
    )

    generic.__post_init__()
    rows = (*generic.fit.rows, *generic.validation.rows)
    if len(rows) != _EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("reopened generic candidate has another row count")
    before = _runtime_projection(binding, qwen)
    providers: list[str] = []
    build_runtime = owned_build_runtime_identity(binding.new_condenser)
    for ordinal, (item, sample) in enumerate(
        zip(rows, treatment.samples, strict=True)
    ):
        blind = gold_blind_from_treatment_sample(sample)
        if len(blind.questions) != 1:
            raise RuntimeError("gold-blind adapter returned another query count")
        probe = blind.questions[0]
        base = verify_diffuse_longmemeval_base(
            cache_root,
            treatment_identity=_treatment_identity(treatment, ordinal),
            sample=blind,
            config=binding.config,
            embedding_identity=binding.embedding_identity,
            build_runtime_identity=build_runtime,
        )
        provider = VerifiedBaseLegacyDiffuseInputProvider.from_verified_base(
            base,
            max_sources=binding.runtime.source_router_max_sources,
            rrf_constant=binding.runtime.source_router_rrf_constant,
        )
        provider_sha256 = identity_sha256(
            analysis_callable_identity_payload(provider, "legacy_input_provider")
        )
        providers.append(provider_sha256)
        payload = item.payload
        receipt = item.manifest.route_evidence.inner_analysis_query_receipt_body
        if (
            item.manifest.ordinal != ordinal
            or item.manifest.question_id != probe.question_id
            or payload.question_id != probe.question_id
            or payload.retrieval_query != probe.retrieval_query
            or payload.prompt_question != probe.prompt_question
            or receipt.get("corpus_sha256") != blind.corpus_sha256
            or receipt.get("legacy_input_provider_identity_sha256")
            != provider_sha256
            or receipt.get("representative_linker_identity_sha256")
            != before.linker_sha256
            or receipt.get("representative_policy_factory_identity_sha256")
            != before.factory_sha256
        ):
            raise RuntimeError(
                f"generic row {ordinal} differs from its exact treatment/runtime source"
            )
    observed = tuple(providers)
    if observed != expected_provider_identities:
        raise RuntimeError("fresh per-row provider identities changed")
    after = _runtime_projection(binding, qwen)
    if after != before:
        raise RuntimeError("owned runtime identities changed during source audit")
    return _RuntimeProjection(
        binding_sha256=after.binding_sha256,
        linker_sha256=after.linker_sha256,
        factory_sha256=after.factory_sha256,
    )


def _candidate_receipt(
    *,
    binding: Any,
    execution: Any,
    implementations: _ImplementationProjection,
    checkpoints: _CheckpointProjection,
    runtime: _RuntimeProjection,
    provider_identities: tuple[str, ...],
    generic_binding: Any,
) -> ProductionCorpusCandidateReceipt:
    coordinates = DeclaredProductionExecutionCoordinates(
        launcher_relative_path=_LAUNCHER_RELATIVE_PATH,
        launcher_sha256=execution.launcher_sha256,
        source_commit=execution.source_commit,
        package_implementation_sha256=implementations.package_sha256,
        corpus_implementation_sha256=implementations.corpus_sha256,
        route_implementation_sha256=implementations.route_sha256,
        runtime_binding_sha256=runtime.binding_sha256,
        ordered_legacy_input_provider_identities_sha256=identity_sha256(
            list(provider_identities)
        ),
        representative_linker_identity_sha256=runtime.linker_sha256,
        representative_policy_factory_identity_sha256=runtime.factory_sha256,
        bge_checkpoint_sha256=checkpoints.bge_sha256,
        qwen_retrieval_checkpoint_sha256=checkpoints.qwen_retrieval_sha256,
        qwen_feature_checkpoint_sha256=checkpoints.qwen_feature_sha256,
        qwen_retrieval_contract_sha256=(
            checkpoints.qwen_retrieval_contract_sha256
        ),
        qwen_feature_contract_sha256=checkpoints.qwen_feature_contract_sha256,
    )
    if coordinates.runtime_binding_sha256 != binding.binding_sha256:
        raise RuntimeError("candidate coordinates bind another resident runtime")
    return ProductionCorpusCandidateReceipt(
        generic_root_manifest_sha256=generic_binding.root_manifest_sha256,
        generic_root_manifest_bytes=generic_binding.root_manifest_bytes,
        generic_corpus_sha256=generic_binding.corpus_sha256,
        generic_inventory_sha256=generic_binding.inventory_sha256,
        generic_population_projection_sha256=(
            generic_binding.population_projection_sha256
        ),
        generic_implementation_sha256=generic_binding.implementation_sha256,
        generic_fit_partition_sha256=generic_binding.fit_partition_sha256,
        generic_fit_manifest_file_sha256=(
            generic_binding.fit_manifest_file_sha256
        ),
        generic_fit_manifest_file_bytes=generic_binding.fit_manifest_file_bytes,
        generic_validation_partition_sha256=(
            generic_binding.validation_partition_sha256
        ),
        generic_validation_manifest_file_sha256=(
            generic_binding.validation_manifest_file_sha256
        ),
        generic_validation_manifest_file_bytes=(
            generic_binding.validation_manifest_file_bytes
        ),
        external_lock=locked_production_external_lock(),
        declared_execution=coordinates,
    )


def _release_binding(binding: Any | None) -> None:
    if binding is None:
        return
    import gc

    qwen = getattr(binding, "_qwen", None)
    object.__setattr__(binding, "_qwen", None)
    if qwen is not None:
        for value in (qwen.reranker, qwen.scorer, qwen.linker, qwen.encoder):
            close = getattr(value, "close", None)
            if callable(close):
                close()
    close = getattr(binding.embedder, "close", None)
    if callable(close):
        close()
    del qwen
    gc.collect()
    torch = sys.modules.get("torch")
    cuda = None if torch is None else getattr(torch, "cuda", None)
    empty_cache = None if cuda is None else getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _run_real_candidate(
    treatment_input: str | os.PathLike[str],
    qwen_model_dir: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    restart: bool = False,
    *,
    loader: Callable[..., object],
    input_type: type,
    launcher_path: Path,
) -> VerifiedLatentTrainingCorpusCandidate:
    if type(restart) is not bool:
        raise TypeError("restart must be an exact boolean")
    target = _absolute_child(output_root, "output root")
    if os.path.lexists(target):
        if restart is not True:
            raise FileExistsError(target)
        return verify_latent_training_corpus_candidate(target)

    treatment_path = _absolute_child(treatment_input, "treatment input")
    qwen_path = _absolute_child(qwen_model_dir, "Qwen model directory")
    if target.resolve(strict=False) == qwen_path.resolve(strict=False) or (
        target.resolve(strict=False).is_relative_to(qwen_path.resolve(strict=False))
    ):
        raise ValueError("output root must not overlap the Qwen checkpoint")
    from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
        require_plain_parent,
    )

    require_plain_parent(target.parent)
    _require_offline_environment()

    from memory_condense.eval.diffuse_longmemeval_replay import (
        certify_replay_launcher,
    )

    execution = _certify_source_execution(
        launcher_path, certifier=certify_replay_launcher
    )
    treatment = _load_exact_treatment(
        treatment_path, loader=loader, input_type=input_type
    )
    population = _population_projection(treatment)
    implementations = _implementation_projection()
    config = _campaign_config()
    arm = _reference_arm()
    binding: Any | None = None
    staging: Any | None = None
    work_owner: OwnedCandidateWorkspace | None = None
    qwen: Any | None = None
    mapper: _RouteV2Mapper | None = None
    try:
        binding = _new_owned_binding(qwen_path)
        _assert_control_lock(config, arm, binding)
        staging = create_candidate_staging(target)
        checkpoints = _verify_checkpoints(qwen_path)
        binding_sha256 = binding.binding_sha256
        _observation, qwen = binding.prepare_resident_replay_runtime()
        if qwen.reranker is not None or binding.binding_sha256 != binding_sha256:
            raise RuntimeError("resident runtime changed during model load")
        runtime = _runtime_projection(binding, qwen)
        work_owner = create_candidate_workspace(
            target,
            kind="execution",
        )
        work_root = candidate_workspace_path(work_owner)
        workspace_root = work_root / "rows"
        workspace_root.mkdir(mode=0o700)
        cache_root = work_root / "cache"
        generic_path = candidate_staging_path(staging) / "generic"
        mapper = _RouteV2Mapper(
            treatment=treatment,
            binding=binding,
            qwen=qwen,
            arm=arm,
            cache_root=cache_root,
            workspace_root=workspace_root,
            runtime=runtime,
        )
        publish_structural_latent_training_corpus(
            population,
            generic_path,
            row_mapper=mapper,
        )
        generic_binding = capture_candidate_generic(staging)
        provider_identities = mapper.finish()
        generic = verify_structural_latent_training_corpus(generic_path)
        source_runtime = _verify_source_projection(
            generic,
            treatment,
            binding=binding,
            qwen=qwen,
            cache_root=cache_root,
            expected_provider_identities=provider_identities,
        )
        if source_runtime != runtime:
            raise RuntimeError("source audit returned another runtime projection")
        capture_candidate_execution_workspace(work_owner)
        candidate = _candidate_receipt(
            binding=binding,
            execution=execution,
            implementations=implementations,
            checkpoints=checkpoints,
            runtime=runtime,
            provider_identities=provider_identities,
            generic_binding=generic_binding,
        )

        def final_guard() -> None:
            current_treatment = _load_exact_treatment(
                treatment_path, loader=loader, input_type=input_type
            )
            if current_treatment != treatment:
                raise RuntimeError("treatment input changed during candidate execution")
            if _certify_source_execution(
                launcher_path, certifier=certify_replay_launcher
            ) != execution:
                raise RuntimeError("tracked launcher identity changed")
            if _implementation_projection() != implementations:
                raise RuntimeError("source implementation changed")
            if _verify_checkpoints(qwen_path) != checkpoints:
                raise RuntimeError("local checkpoint bytes changed")
            fresh_binding = _new_owned_binding(qwen_path)
            try:
                _assert_control_lock(fresh_binding.config, arm, fresh_binding)
                if fresh_binding.binding_sha256 != runtime.binding_sha256:
                    raise RuntimeError("fresh runtime binding identity changed")
            finally:
                _release_binding(fresh_binding)
            current_generic = verify_structural_latent_training_corpus(generic_path)
            if recheck_captured_candidate_generic(staging) != generic_binding:
                raise RuntimeError("generic candidate binding changed")
            if _verify_source_projection(
                current_generic,
                current_treatment,
                binding=binding,
                qwen=qwen,
                cache_root=cache_root,
                expected_provider_identities=provider_identities,
            ) != runtime:
                raise RuntimeError("fresh source/runtime projection changed")
            # The 300-row audit can be long.  Repeat every compact external
            # observation after it so no drift window reaches receipt write or
            # atomic promotion.
            final_treatment = _load_exact_treatment(
                treatment_path, loader=loader, input_type=input_type
            )
            if final_treatment != treatment:
                raise RuntimeError("treatment input changed during final row audit")
            if _certify_source_execution(
                launcher_path, certifier=certify_replay_launcher
            ) != execution:
                raise RuntimeError("tracked source changed during final row audit")
            if _implementation_projection() != implementations:
                raise RuntimeError("implementation changed during final row audit")
            if _verify_checkpoints(qwen_path) != checkpoints:
                raise RuntimeError("checkpoint changed during final row audit")
            if _runtime_projection(binding, qwen) != runtime:
                raise RuntimeError("runtime changed during final row audit")

        return publish_candidate_root(
            staging,
            target,
            candidate=candidate,
            final_guard=final_guard,
        )
    except BaseException as original:
        if staging is not None:
            try:
                cleanup_candidate_staging(staging)
            except BaseException as cleanup_error:
                original.add_note(
                    f"exact candidate staging cleanup was refused: {cleanup_error!r}"
                )
        raise
    finally:
        active = sys.exception()
        cleanup_errors: list[BaseException] = []
        if work_owner is not None:
            try:
                cleanup_candidate_workspace(work_owner)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        # Drop every closure/local reference to the resident Qwen object before
        # clearing the binding and CUDA allocator.
        mapper = None
        qwen = None
        try:
            _release_binding(binding)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            if active is not None:
                for cleanup_error in cleanup_errors:
                    active.add_note(
                        f"owned resource cleanup failed: {cleanup_error!r}"
                    )
            else:
                raise cleanup_errors[0]


def _define_closed_public_surface():
    # Closure-owned literals keep module-global rebinding from opening the
    # public path.  The returned function deliberately does not inspect any
    # argument, including ``restart`` and hostile PathLike implementations.
    disabled_reason = "candidate_execution_activation_not_audited"
    if disabled_reason != CANDIDATE_EXECUTION_DISABLED_REASON:
        raise RuntimeError("candidate execution disabled reason drifted")
    unavailable_type = ProductionCandidateExecutionUnavailable
    status_type = ProductionCandidateExecutionStatus

    def run(
        treatment_input: str | os.PathLike[str],
        qwen_model_dir: str | os.PathLike[str],
        output_root: str | os.PathLike[str],
        restart: bool = False,
    ) -> None:
        del treatment_input, qwen_model_dir, output_root, restart
        raise unavailable_type(
            f"candidate execution is unavailable: {disabled_reason}"
        )

    def status() -> ProductionCandidateExecutionStatus:
        return status_type(reason=disabled_reason)

    return run, status


run, candidate_execution_status = _define_closed_public_surface()


def main(argv: list[str] | None = None) -> int:
    # Match the Python API's zero-coercion boundary: even command-line text is
    # not parsed while the capability-safe upstream publishers are absent.
    del argv
    status = candidate_execution_status()
    print(
        f"latent-training corpus candidate unavailable: {status.reason}",
        file=sys.stderr,
    )
    return 2


__all__ = ["candidate_execution_status", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
