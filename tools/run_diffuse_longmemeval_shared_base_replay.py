"""Run the pinned ordinal-169 shared-base replay with local resident models.

This tracked launcher consumes only the closed-schema sanitized analysis
treatment.  It does not open the benchmark dataset, an exposure audit, gold,
an answerer, or a judge.  The outer receipt certifies population selection and
launcher identity; model-generated Qwen records remain runtime attestations.

Run this module from the repository root with ``python -m`` so the tracked
``__file__`` passed to launcher certification is the file Git records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_condense.domain.discourse import ClosurePolicy, identity_sha256
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.diffuse_longmemeval_analysis import DiffuseLongMemEvalArm
from memory_condense.eval.diffuse_longmemeval_base import (
    DiffuseBaseTreatmentIdentity,
    owned_build_runtime_identity,
    verify_diffuse_longmemeval_base,
)
from memory_condense.eval.diffuse_longmemeval_replay import (
    DiffuseLongMemEvalReplayReceipt,
    ReplayExecutionIdentity,
    certify_replay_launcher,
    run_diffuse_longmemeval_shared_base_replay,
    verify_diffuse_longmemeval_replay_package,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DiffuseLongMemEvalExecutionBinding,
    DiffuseLongMemEvalRuntimeConfig,
    build_diffuse_longmemeval_execution_binding,
    gold_blind_from_treatment_sample,
)
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    verify_bge_m3_checkpoint,
)
from memory_condense.modeling.qwen_prefix import verify_prefix_checkpoint
from memory_condense.search.episodes import EpisodeRetrievalPolicy
from tools.v4_population_firebreak import (
    PRODUCTION_LOCK,
    AnalysisTreatmentInput,
    TreatmentSample,
    load_analysis_treatment_input,
)


CAMPAIGN_FORMAT = "memory-condense-v4-shared-base-real-model-replay-campaign-v1"
CAMPAIGN_RECEIPT_NAME = "campaign-receipt.json"
PINNED_SAMPLE_ORDINAL = 169
PINNED_TREATMENT_FILE_SHA256 = (
    "b4d1d34538fdabbd6127c339bff8167293d290eb732afc18a5d8963d12b15001"
)
PINNED_SANITIZED_PROJECTION_SHA256 = (
    "58a1982122d259e046ac5268de8fc3c2857a63d24c859e3bc13e4e6b9aa52ad8"
)
PINNED_SAMPLE_COUNT = 300
_DIGEST = r"^[0-9a-f]{64}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CampaignPopulationIdentity(_FrozenModel):
    treatment_identity: DiffuseBaseTreatmentIdentity
    treatment_identity_sha256: str = Field(pattern=_DIGEST)
    selected_sample_id_sha256: str = Field(pattern=_DIGEST)
    selected_corpus_sha256: str = Field(pattern=_DIGEST)
    selected_question_id_sha256s: tuple[str, ...]
    selected_question_probe_sha256s: tuple[str, ...]
    turn_count: int = Field(ge=1)
    question_count: Literal[1] = 1

    @model_validator(mode="after")
    def _bind_population(self) -> "CampaignPopulationIdentity":
        if self.treatment_identity.sample_ordinal != PINNED_SAMPLE_ORDINAL:
            raise ValueError("campaign receipt selected another sample ordinal")
        if self.treatment_identity.sample_count != PINNED_SAMPLE_COUNT:
            raise ValueError("campaign receipt selected another population")
        if self.treatment_identity_sha256 != identity_sha256(
            self.treatment_identity.model_dump(mode="json")
        ):
            raise ValueError("campaign treatment identity digest differs")
        if len(self.selected_question_id_sha256s) != self.question_count or len(
            self.selected_question_probe_sha256s
        ) != self.question_count:
            raise ValueError("campaign question identities are incomplete")
        return self


class CampaignCheckpointIdentity(_FrozenModel):
    bge_m3_checkpoint_sha256: str = Field(pattern=_DIGEST)
    qwen_prefix_checkpoint_sha256: str = Field(pattern=_DIGEST)
    qwen_prefix_verified_file_count: Literal[7] = 7
    bge_preflight_resolution_local_only: Literal[True] = True
    qwen_loader_local_files_only: Literal[True] = True

    @model_validator(mode="after")
    def _bind_bge(self) -> "CampaignCheckpointIdentity":
        if self.bge_m3_checkpoint_sha256 != BGE_M3_CHECKPOINT_SHA256:
            raise ValueError("campaign receipt binds another BGE checkpoint")
        return self


class CampaignArtifactIdentity(_FrozenModel):
    base_store_key: str = Field(pattern=_DIGEST)
    base_artifact_sha256: str = Field(pattern=_DIGEST)
    base_manifest_file_sha256: str = Field(pattern=_DIGEST)
    query_input_key: str = Field(pattern=_DIGEST)
    query_artifact_sha256: str = Field(pattern=_DIGEST)
    query_manifest_file_sha256: str = Field(pattern=_DIGEST)
    replay_manifest_file_sha256: str = Field(pattern=_DIGEST)
    replay_receipt_sha256: str = Field(pattern=_DIGEST)


class CampaignClaimBoundary(_FrozenModel):
    pinned_sanitized_population_artifact_verified: Literal[True] = True
    selected_ordinal_membership_in_pinned_artifact_verified: Literal[True] = True
    sample_ordinal_bound_by_treatment_identity: Literal[True] = True
    offline_environment_variables_enforced: Literal[True] = True
    pixi_openmp_activation_variable_enforced: Literal[True] = True
    local_checkpoint_bytes_verified: Literal[True] = True
    runtime_binding_independently_rederived: Literal[True] = True
    base_and_replay_independently_reverified: Literal[True] = True
    retrieval_input_schema_contains_gold_fields: Literal[False] = False
    qa_responder_or_judge_calls: Literal[0] = 0
    network_transport_audit_performed: Literal[False] = False
    network_calls_proven_zero: Literal[False] = False
    model_outputs_independently_reexecuted: Literal[False] = False
    qwen_compilation_records_are_runtime_attestations: Literal[True] = True


class PinnedReplayCampaignReceipt(_FrozenModel):
    format: Literal[
        "memory-condense-v4-shared-base-real-model-replay-campaign-v1"
    ] = CAMPAIGN_FORMAT
    population: CampaignPopulationIdentity
    launcher: ReplayExecutionIdentity
    runtime_binding_sha256: str = Field(pattern=_DIGEST)
    checkpoints: CampaignCheckpointIdentity
    artifacts: CampaignArtifactIdentity
    claims: CampaignClaimBoundary = CampaignClaimBoundary()
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _self_hash(self) -> "PinnedReplayCampaignReceipt":
        expected = identity_sha256(
            self.model_dump(mode="json", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected:
            raise ValueError("campaign receipt digest differs from its body")
        return self


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _require_offline_environment() -> None:
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    missing = [name for name in required if os.environ.get(name) != "1"]
    if missing:
        raise RuntimeError(
            "offline replay requires environment variables set exactly to 1: "
            + ", ".join(missing)
        )
    if os.environ.get("KMP_DUPLICATE_LIB_OK") != "TRUE":
        raise RuntimeError(
            "real-model replay requires Pixi activation or "
            "KMP_DUPLICATE_LIB_OK=TRUE"
        )


def _canonical_cuda_device(value: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized == "cuda":
        return "cuda:0"
    prefix = "cuda:"
    ordinal = normalized[len(prefix) :] if normalized.startswith(prefix) else ""
    if not ordinal.isdigit() or str(int(ordinal)) != ordinal:
        raise ValueError("device must be cuda or cuda:<nonnegative integer>")
    return f"cuda:{ordinal}"


def _load_pinned_treatment(path: Path) -> AnalysisTreatmentInput:
    expected_count = sum(
        PRODUCTION_LOCK.partitions[name].count
        for name in ("development", "validation")
    )
    if expected_count != PINNED_SAMPLE_COUNT:
        raise RuntimeError("production analysis population size changed")
    return load_analysis_treatment_input(
        path,
        expected_file_sha256=PINNED_TREATMENT_FILE_SHA256,
        expected_sanitized_projection_sha256=(
            PINNED_SANITIZED_PROJECTION_SHA256
        ),
        expected_dataset_sha256=PRODUCTION_LOCK.dataset_sha256,
        expected_split_manifest_sha256=PRODUCTION_LOCK.split_manifest_sha256,
        expected_ordered_question_ids_sha256=(
            PRODUCTION_LOCK.analysis_ordered_question_ids_sha256
        ),
        expected_sample_count=PINNED_SAMPLE_COUNT,
    )


def _treatment_identity(
    treatment: AnalysisTreatmentInput,
) -> DiffuseBaseTreatmentIdentity:
    if len(treatment.samples) != PINNED_SAMPLE_COUNT:
        raise RuntimeError("pinned treatment population changed size")
    return DiffuseBaseTreatmentIdentity(
        treatment_file_sha256=treatment.file_sha256,
        sanitized_projection_sha256=treatment.sanitized_projection_sha256,
        dataset_sha256=treatment.dataset_sha256,
        split_manifest_sha256=treatment.split_manifest_sha256,
        ordered_question_ids_sha256=treatment.ordered_question_ids_sha256,
        sample_count=len(treatment.samples),
        sample_ordinal=PINNED_SAMPLE_ORDINAL,
    )


def _campaign_config(device: str) -> EvalConfig:
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
        embedding_device=device,
        max_prompt_tokens=8000,
    )


def _campaign_runtime(
    qwen_model_dir: Path,
    device: str,
) -> DiffuseLongMemEvalRuntimeConfig:
    return DiffuseLongMemEvalRuntimeConfig(
        qwen_model_dir=qwen_model_dir,
        residency_mode="resident_bge_qwen",
        qwen_device=device,
    )


def _reference_arm() -> DiffuseLongMemEvalArm:
    return DiffuseLongMemEvalArm(
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


def _new_owned_binding(
    qwen_model_dir: Path,
    device: str,
) -> DiffuseLongMemEvalExecutionBinding:
    return build_diffuse_longmemeval_execution_binding(
        config=_campaign_config(device),
        runtime=_campaign_runtime(qwen_model_dir, device),
    )


def _require_campaign_binding(binding: object) -> None:
    if type(binding) is not DiffuseLongMemEvalExecutionBinding:
        raise TypeError("campaign requires the exact owned execution binding")
    if binding.runtime_binding_certified is not True:
        raise RuntimeError("campaign runtime binding is not certified")
    payload = dict(binding.analysis_identity_payload())
    if payload.get("residency_mode") != "resident_bge_qwen":
        raise ValueError("campaign requires simultaneous BGE/Qwen residency")
    embedding_device = _canonical_cuda_device(payload["embedding"]["device"])
    qwen_device = _canonical_cuda_device(payload["qwen"]["device"])
    if embedding_device != qwen_device:
        raise ValueError("campaign requires BGE and Qwen on one CUDA device")
    if binding.config.retrieval.qwen_rerank or (
        binding.config.retrieval.qwen_feedback
    ):
        raise ValueError("campaign freezes inputs before Qwen and forbids legacy Qwen")
    if binding.binding_sha256 != identity_sha256(payload):
        raise RuntimeError("campaign runtime binding is not content addressed")


def _verify_local_checkpoints(
    binding: DiffuseLongMemEvalExecutionBinding,
    qwen_model_dir: Path,
) -> CampaignCheckpointIdentity:
    if not qwen_model_dir.is_dir():
        raise FileNotFoundError(f"Qwen checkpoint directory is missing: {qwen_model_dir}")
    runtime_payload = dict(binding.analysis_identity_payload())
    bge_digest = verify_bge_m3_checkpoint()
    if bge_digest != runtime_payload["embedding"]["checkpoint_sha256"]:
        raise RuntimeError("local BGE checkpoint differs from runtime identity")
    qwen_payload = runtime_payload["qwen"]
    qwen_identity = verify_prefix_checkpoint(
        qwen_model_dir,
        layers=int(qwen_payload["prefix_layers"]),
        model_id=str(qwen_payload["model_id"]),
        model_revision=str(qwen_payload["model_revision"]),
        expected_checkpoint_sha256=str(qwen_payload["checkpoint_sha256"]),
    )
    if (
        qwen_identity.checkpoint_sha256
        != qwen_payload["checkpoint_sha256"]
    ):
        raise RuntimeError("local Qwen checkpoint differs from runtime identity")
    return CampaignCheckpointIdentity(
        bge_m3_checkpoint_sha256=bge_digest,
        qwen_prefix_checkpoint_sha256=qwen_identity.checkpoint_sha256,
        qwen_prefix_verified_file_count=len(qwen_identity.verified_files),
    )


def _population_receipt(
    treatment_identity: DiffuseBaseTreatmentIdentity,
    sample: TreatmentSample,
) -> CampaignPopulationIdentity:
    blind = gold_blind_from_treatment_sample(sample)
    return CampaignPopulationIdentity(
        treatment_identity=treatment_identity,
        treatment_identity_sha256=identity_sha256(
            treatment_identity.model_dump(mode="json")
        ),
        selected_sample_id_sha256=identity_sha256(
            {"sample_id": blind.sample_id}
        ),
        selected_corpus_sha256=blind.corpus_sha256,
        selected_question_id_sha256s=tuple(
            identity_sha256({"question_id": item.question_id})
            for item in blind.questions
        ),
        selected_question_probe_sha256s=tuple(
            item.probe_sha256 for item in blind.questions
        ),
        turn_count=len(blind.turns),
        question_count=len(blind.questions),
    )


def _assert_inner_receipt(
    replay,
    *,
    execution: ReplayExecutionIdentity,
    runtime_binding_sha256: str,
    treatment_identity: DiffuseBaseTreatmentIdentity,
) -> None:
    expected_treatment_sha = identity_sha256(
        treatment_identity.model_dump(mode="json")
    )
    if replay.execution_identity != execution or not replay.launcher_binding_certified:
        raise RuntimeError("replay did not preserve tracked launcher certification")
    if replay.runtime_binding.identity_sha256 != runtime_binding_sha256:
        raise RuntimeError("replay did not preserve the owned runtime binding")
    if replay.query_manifest.treatment_identity_sha256 != expected_treatment_sha:
        raise RuntimeError("replay did not preserve the pinned treatment identity")
    if replay.treatment_population_membership_certified:
        raise RuntimeError("low-level replay overclaimed population membership")
    if replay.retrieval_input_schema_contains_gold_fields:
        raise RuntimeError("low-level replay admitted a gold-bearing schema")
    if replay.qa_responder_or_judge_calls != 0:
        raise RuntimeError("low-level replay invoked an answerer or judge")


def _write_receipt(path: Path, receipt: PinnedReplayCampaignReceipt) -> None:
    payload = _canonical_json_bytes(receipt.model_dump(mode="json"))
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular file")


def _require_regular_directory(path: Path, label: str) -> None:
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction() or not path.is_dir():
        raise RuntimeError(f"{label} must be a regular directory")


def _require_exact_output_children(root: Path) -> None:
    expected = {
        "cache",
        "replay",
        ".replay.publish.lock",
        CAMPAIGN_RECEIPT_NAME,
    }
    actual = {item.name for item in root.iterdir()}
    if actual != expected:
        raise RuntimeError(
            "campaign output root has unexpected or missing entries: "
            f"{sorted(actual ^ expected)}"
        )


def _require_exact_children(root: Path, expected: set[str], label: str) -> None:
    actual = {item.name for item in root.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"{label} has unexpected or missing entries: "
            f"{sorted(actual ^ expected)}"
        )


def _load_nested_replay_manifest(path: Path) -> DiffuseLongMemEvalReplayReceipt:
    try:
        return DiffuseLongMemEvalReplayReceipt.model_validate_json(path.read_bytes())
    except Exception as exc:
        raise RuntimeError("invalid nested replay manifest") from exc


def verify_pinned_replay_campaign_receipt(
    output_root: str | Path,
    *,
    expected_population: CampaignPopulationIdentity,
    expected_launcher: ReplayExecutionIdentity,
    expected_runtime_binding_sha256: str,
) -> PinnedReplayCampaignReceipt:
    """Read-only outer verification against externally derived identities.

    The nested base and replay packages still require their dedicated
    verifiers.  This function checks that the already-verified nested package
    bytes and the population/launcher/runtime coordinates are exactly those
    sealed by the outer manifest.
    """

    root = Path(output_root)
    _require_regular_directory(root, "campaign output root")
    _require_exact_output_children(root)
    cache_root = root / "cache"
    replay_root = root / "replay"
    lock_path = root / ".replay.publish.lock"
    receipt_path = root / CAMPAIGN_RECEIPT_NAME
    _require_regular_directory(cache_root, "campaign cache root")
    _require_regular_directory(replay_root, "campaign replay root")
    _require_regular_file(lock_path, "replay publication lock")
    _require_regular_file(receipt_path, "campaign receipt")
    stores_root = cache_root / "stores"
    queries_root = cache_root / "query-inputs"
    _require_exact_children(
        cache_root, {"stores", "query-inputs"}, "campaign cache root"
    )
    _require_regular_directory(stores_root, "campaign base stores root")
    _require_regular_directory(queries_root, "campaign query inputs root")
    before = {
        path: (file_sha256(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in (lock_path, receipt_path)
    }
    raw = receipt_path.read_bytes()
    try:
        receipt = PinnedReplayCampaignReceipt.model_validate_json(raw)
    except Exception as exc:
        raise RuntimeError("invalid campaign receipt") from exc
    if raw != _canonical_json_bytes(receipt.model_dump(mode="json")):
        raise RuntimeError("campaign receipt is not canonical JSON")
    expected_runtime = str(expected_runtime_binding_sha256).strip().casefold()
    if (
        receipt.population != expected_population
        or receipt.launcher != expected_launcher
        or receipt.runtime_binding_sha256 != expected_runtime
    ):
        raise RuntimeError("campaign receipt differs from external identities")
    base_manifest = (
        stores_root
        / receipt.artifacts.base_store_key
        / "base-manifest.json"
    )
    query_manifest = (
        queries_root
        / receipt.artifacts.query_input_key
        / "query-manifest.json"
    )
    replay_manifest = replay_root / "replay-manifest.json"
    base_lock = stores_root / (
        f".{receipt.artifacts.base_store_key}.publish.lock"
    )
    query_lock = queries_root / (
        f".{receipt.artifacts.query_input_key}.publish.lock"
    )
    _require_exact_children(
        stores_root,
        {receipt.artifacts.base_store_key, base_lock.name},
        "campaign base stores root",
    )
    _require_exact_children(
        queries_root,
        {receipt.artifacts.query_input_key, query_lock.name},
        "campaign query inputs root",
    )
    _require_regular_file(base_lock, "base publication lock")
    _require_regular_file(query_lock, "query publication lock")
    before[base_lock] = (
        file_sha256(base_lock),
        base_lock.stat().st_mtime_ns,
        base_lock.stat().st_size,
    )
    before[query_lock] = (
        file_sha256(query_lock),
        query_lock.stat().st_mtime_ns,
        query_lock.stat().st_size,
    )
    for path, label, expected_sha in (
        (
            base_manifest,
            "base manifest",
            receipt.artifacts.base_manifest_file_sha256,
        ),
        (
            query_manifest,
            "query manifest",
            receipt.artifacts.query_manifest_file_sha256,
        ),
        (
            replay_manifest,
            "replay manifest",
            receipt.artifacts.replay_manifest_file_sha256,
        ),
    ):
        _require_regular_file(path, label)
        if file_sha256(path) != expected_sha:
            raise RuntimeError(f"{label} differs from the campaign receipt")
        before[path] = (
            expected_sha,
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
    nested = _load_nested_replay_manifest(replay_manifest)
    runtime_payload = json.loads(
        nested.runtime_binding.canonical_identity_json
    )
    if (
        nested.receipt_sha256 != receipt.artifacts.replay_receipt_sha256
        or nested.execution_identity != receipt.launcher
        or nested.runtime_binding.identity_sha256
        != receipt.runtime_binding_sha256
        or nested.base_manifest_file_sha256
        != receipt.artifacts.base_manifest_file_sha256
        or nested.query_manifest_file_sha256
        != receipt.artifacts.query_manifest_file_sha256
        or nested.base_manifest.base_store_key
        != receipt.artifacts.base_store_key
        or nested.base_manifest.artifact_sha256
        != receipt.artifacts.base_artifact_sha256
        or nested.query_manifest.query_input_key
        != receipt.artifacts.query_input_key
        or nested.query_manifest.artifact_sha256
        != receipt.artifacts.query_artifact_sha256
        or nested.base_manifest.embedding_identity.checkpoint_sha256
        != receipt.checkpoints.bge_m3_checkpoint_sha256
        or runtime_payload["embedding"]["checkpoint_sha256"]
        != receipt.checkpoints.bge_m3_checkpoint_sha256
        or runtime_payload["qwen"]["checkpoint_sha256"]
        != receipt.checkpoints.qwen_prefix_checkpoint_sha256
        or nested.base_manifest.sample_id_sha256
        != receipt.population.selected_sample_id_sha256
        or nested.base_manifest.corpus_sha256
        != receipt.population.selected_corpus_sha256
        or nested.base_manifest.turn_count != receipt.population.turn_count
        or nested.query_manifest.treatment_identity
        != receipt.population.treatment_identity
        or nested.query_manifest.query_count != receipt.population.question_count
        or any(
            tuple(item.question_id_sha256 for item in arm.queries)
            != receipt.population.selected_question_id_sha256s
            or tuple(item.question_probe_sha256 for item in arm.queries)
            != receipt.population.selected_question_probe_sha256s
            for arm in nested.arms
        )
    ):
        raise RuntimeError("nested replay differs from the campaign receipt")
    _require_exact_output_children(root)
    _require_exact_children(
        cache_root, {"stores", "query-inputs"}, "campaign cache root"
    )
    _require_exact_children(
        stores_root,
        {receipt.artifacts.base_store_key, base_lock.name},
        "campaign base stores root",
    )
    _require_exact_children(
        queries_root,
        {receipt.artifacts.query_input_key, query_lock.name},
        "campaign query inputs root",
    )
    after = {
        path: (file_sha256(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in before
    }
    if before != after:
        raise RuntimeError("campaign receipt verification observed mutable files")
    return receipt


def _run_campaign(
    *,
    treatment_input: Path,
    qwen_model_dir: Path,
    output_root: Path,
    device: str,
    launcher_path: Path,
) -> PinnedReplayCampaignReceipt:
    """Execute, independently reload, and verify the pinned campaign."""

    _require_offline_environment()
    normalized_device = _canonical_cuda_device(device)
    treatment_path = treatment_input.resolve()
    qwen_path = qwen_model_dir.resolve()
    target = output_root.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to reuse output root: {target}")
    if target == qwen_path or target.is_relative_to(qwen_path):
        raise ValueError("output root must not be inside the Qwen checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("output parent must be a regular directory")

    execution = certify_replay_launcher(launcher_path)
    treatment = _load_pinned_treatment(treatment_path)
    treatment_identity = _treatment_identity(treatment)
    sample = treatment.samples[PINNED_SAMPLE_ORDINAL]
    population = _population_receipt(treatment_identity, sample)
    binding = _new_owned_binding(qwen_path, normalized_device)
    _require_campaign_binding(binding)
    runtime_binding_sha256 = binding.binding_sha256
    checkpoints = _verify_local_checkpoints(binding, qwen_path)

    target.mkdir(exist_ok=False)
    cache_root = target / "cache"
    replay_root = target / "replay"
    receipt_path = target / CAMPAIGN_RECEIPT_NAME
    if cache_root.exists() or replay_root.exists() or receipt_path.exists():
        raise RuntimeError("fresh campaign root unexpectedly contains outputs")
    replay = run_diffuse_longmemeval_shared_base_replay(
        sample,
        treatment_identity=treatment_identity,
        binding=binding,
        reference_arm=_reference_arm(),
        cache_root=cache_root,
        replay_root=replay_root,
        launcher_path=launcher_path,
    )
    _assert_inner_receipt(
        replay,
        execution=execution,
        runtime_binding_sha256=runtime_binding_sha256,
        treatment_identity=treatment_identity,
    )

    reloaded = _load_pinned_treatment(treatment_path)
    reloaded_identity = _treatment_identity(reloaded)
    reloaded_sample = reloaded.samples[PINNED_SAMPLE_ORDINAL]
    if reloaded_identity != treatment_identity or reloaded_sample != sample:
        raise RuntimeError("pinned treatment changed between execution and audit")
    verification_binding = _new_owned_binding(qwen_path, normalized_device)
    _require_campaign_binding(verification_binding)
    expected_runtime_sha256 = verification_binding.binding_sha256
    if expected_runtime_sha256 != runtime_binding_sha256:
        raise RuntimeError("independently derived runtime identity changed")
    blind = gold_blind_from_treatment_sample(reloaded_sample)
    verified_base = verify_diffuse_longmemeval_base(
        cache_root,
        treatment_identity=reloaded_identity,
        sample=blind,
        config=verification_binding.config,
        embedding_identity=verification_binding.embedding_identity,
        build_runtime_identity=owned_build_runtime_identity(
            verification_binding.new_condenser
        ),
    )
    verified_replay = verify_diffuse_longmemeval_replay_package(
        replay_root,
        base=verified_base,
        expected_runtime_binding_sha256=expected_runtime_sha256,
    )
    if verified_replay != replay:
        raise RuntimeError("independent replay verification returned another receipt")
    final_execution = certify_replay_launcher(launcher_path)
    if final_execution != execution:
        raise RuntimeError("tracked launcher identity changed during execution")
    _assert_inner_receipt(
        verified_replay,
        execution=final_execution,
        runtime_binding_sha256=expected_runtime_sha256,
        treatment_identity=reloaded_identity,
    )

    artifacts = CampaignArtifactIdentity(
        base_store_key=verified_base.store_manifest.base_store_key,
        base_artifact_sha256=verified_base.store_manifest.artifact_sha256,
        base_manifest_file_sha256=verified_base.store_manifest_sha256,
        query_input_key=verified_base.query_manifest.query_input_key,
        query_artifact_sha256=verified_base.query_manifest.artifact_sha256,
        query_manifest_file_sha256=verified_base.query_manifest_sha256,
        replay_manifest_file_sha256=file_sha256(
            replay_root / "replay-manifest.json"
        ),
        replay_receipt_sha256=verified_replay.receipt_sha256,
    )
    values = {
        "population": population,
        "launcher": final_execution,
        "runtime_binding_sha256": expected_runtime_sha256,
        "checkpoints": checkpoints,
        "artifacts": artifacts,
        "claims": CampaignClaimBoundary(),
    }
    unsigned = {
        "format": CAMPAIGN_FORMAT,
        **{
            key: value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else value
            for key, value in values.items()
        },
    }
    receipt = PinnedReplayCampaignReceipt(
        **values,
        receipt_sha256=identity_sha256(unsigned),
    )
    _write_receipt(receipt_path, receipt)
    published = verify_pinned_replay_campaign_receipt(
        target,
        expected_population=population,
        expected_launcher=final_execution,
        expected_runtime_binding_sha256=expected_runtime_sha256,
    )
    if published != receipt or file_sha256(receipt_path) != hashlib.sha256(
        _canonical_json_bytes(receipt.model_dump(mode="json"))
    ).hexdigest():
        raise RuntimeError("published campaign receipt bytes changed")
    return published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "run the pinned sanitized ordinal-169 three-arm shared-base replay"
        )
    )
    parser.add_argument("--treatment-input", type=Path, required=True)
    parser.add_argument("--qwen-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = _run_campaign(
            treatment_input=args.treatment_input,
            qwen_model_dir=args.qwen_model_dir,
            output_root=args.output_root,
            device=args.device,
            launcher_path=Path(__file__),
        )
    except Exception as exc:
        print(f"shared-base replay failed: {exc}", file=sys.stderr)
        return 2
    print(
        "SHARED_BASE_REPLAY_PASS "
        f"receipt={receipt.receipt_sha256} "
        f"runtime={receipt.runtime_binding_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
