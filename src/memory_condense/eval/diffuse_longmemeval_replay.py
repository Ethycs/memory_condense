"""Provider-free three-arm LongMemEval replay from one verified shared base."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import memory_condense.eval._diffuse_replay_provider_history as provider_history
from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval._diffuse_base_contracts import (
    DiffuseDerivedFinalization,
    DiffuseDerivedStore,
    VerifiedDiffuseLongMemEvalBase,
    canonical_json_bytes,
    publish_complete_directory,
    require_exact_children,
    require_regular_directory,
    require_regular_file,
    safe_remove_staging,
    write_new_bytes,
)
from memory_condense.eval._diffuse_replay_contracts import (
    REPLAY_MANIFEST_NAME,
    DiffuseLongMemEvalReplayReceipt,
    ReplayExecutionIdentity,
)
from memory_condense.eval._diffuse_replay_packets import (
    VerifiedDiffuseReplayPackage,
)
from memory_condense.eval._diffuse_replay_reconstruction import (
    REPLAY_MODES,
    build_replay_arm_record,
    build_replay_file_inventory,
    canonical_identity,
    _verify_and_reconstruct_replay_package as _reconstruct_replay_package,
)
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalArm,
    analysis_callable_identity_payload,
    matched_diffuse_boundary_arms,
    retrieve_diffuse_longmemeval_sample,
)
from memory_condense.eval.diffuse_longmemeval_base import (
    FROZEN_QUERY_INPUTS_NAME,
    QUERY_MANIFEST_NAME,
    STORE_MANIFEST_NAME,
    DiffuseBaseTreatmentIdentity,
    clone_diffuse_longmemeval_base,
    finalize_diffuse_longmemeval_derived_store,
    open_diffuse_longmemeval_derived_store,
    owned_build_runtime_identity,
    publish_diffuse_longmemeval_base,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    GoldBlindLongMemEvalSample,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DiffuseLongMemEvalExecutionBinding,
    DiffuseLongMemEvalRuntimeResult,
    FrozenLegacyDiffuseInputProvider,
    TreatmentSampleLike,
    gold_blind_from_treatment_sample,
)
from memory_condense.eval.diffuse_longmemeval_runtime_matched import (
    validate_matched_diffuse_runtime_results,
)
from memory_condense.eval.reproducibility import file_sha256


@dataclass(frozen=True, slots=True)
class VerifiedBaseLegacyDiffuseInputProvider:
    """Route verified frozen pointers without a false residency assertion."""

    _verified: VerifiedDiffuseLongMemEvalBase = field(repr=False)
    _delegate: FrozenLegacyDiffuseInputProvider = field(repr=False)
    base_store_key: str
    base_artifact_sha256: str
    base_manifest_sha256: str
    query_input_key: str
    query_artifact_sha256: str
    query_manifest_sha256: str
    frozen_inputs_sha256: str
    query_set_sha256: str
    ordered_frozen_receipts_sha256: str

    def __post_init__(self) -> None:
        if type(self._verified) is not VerifiedDiffuseLongMemEvalBase or type(
            self._delegate
        ) is not FrozenLegacyDiffuseInputProvider:
            raise TypeError("verified-base provider requires exact owned inputs")
        if self._delegate.inputs != self._verified.frozen_query_inputs:
            raise ValueError("verified-base provider delegate changed frozen inputs")
        receipts = tuple(
            item.receipt_sha256 for item in self._verified.frozen_query_inputs
        )
        expected = (
            self._verified.base_store_key,
            self._verified.store_manifest.artifact_sha256,
            self._verified.store_manifest_sha256,
            self._verified.query_input_key,
            self._verified.query_manifest.artifact_sha256,
            self._verified.query_manifest_sha256,
            self._verified.query_manifest.frozen_inputs_sha256,
            self._verified.query_manifest.query_set_sha256,
            identity_sha256(list(receipts)),
        )
        observed = (
            self.base_store_key,
            self.base_artifact_sha256,
            self.base_manifest_sha256,
            self.query_input_key,
            self.query_artifact_sha256,
            self.query_manifest_sha256,
            self.frozen_inputs_sha256,
            self.query_set_sha256,
            self.ordered_frozen_receipts_sha256,
        )
        if observed != expected:
            raise ValueError("verified-base provider identity changed")

    @classmethod
    def from_verified_base(
        cls,
        verified: VerifiedDiffuseLongMemEvalBase,
        *,
        max_sources: int,
        rrf_constant: int,
    ) -> "VerifiedBaseLegacyDiffuseInputProvider":
        if type(verified) is not VerifiedDiffuseLongMemEvalBase:
            raise TypeError("verified must be an exact shared-base bundle")
        if file_sha256(verified.store_path / STORE_MANIFEST_NAME) != (
            verified.store_manifest_sha256
        ) or file_sha256(verified.query_inputs_path / QUERY_MANIFEST_NAME) != (
            verified.query_manifest_sha256
        ):
            raise RuntimeError("shared-base manifest bytes changed")
        if file_sha256(
            verified.query_inputs_path / FROZEN_QUERY_INPUTS_NAME
        ) != verified.query_manifest.frozen_inputs_sha256:
            raise RuntimeError("shared-base frozen pointers changed")
        receipts = tuple(item.receipt_sha256 for item in verified.frozen_query_inputs)
        return cls(
            _verified=verified,
            _delegate=FrozenLegacyDiffuseInputProvider(
                verified.frozen_query_inputs,
                max_sources=max_sources,
                rrf_constant=rrf_constant,
            ),
            base_store_key=verified.base_store_key,
            base_artifact_sha256=verified.store_manifest.artifact_sha256,
            base_manifest_sha256=verified.store_manifest_sha256,
            query_input_key=verified.query_input_key,
            query_artifact_sha256=verified.query_manifest.artifact_sha256,
            query_manifest_sha256=verified.query_manifest_sha256,
            frozen_inputs_sha256=verified.query_manifest.frozen_inputs_sha256,
            query_set_sha256=verified.query_manifest.query_set_sha256,
            ordered_frozen_receipts_sha256=identity_sha256(list(receipts)),
        )

    def analysis_identity_payload(self) -> dict[str, object]:
        return {
            "provider": "verified-shared-base-pointer-v1",
            "acquisition": "verified_shared_base_pointer_v1",
            "base_store_key": self.base_store_key,
            "base_artifact_sha256": self.base_artifact_sha256,
            "base_manifest_sha256": self.base_manifest_sha256,
            "query_input_key": self.query_input_key,
            "query_artifact_sha256": self.query_artifact_sha256,
            "query_manifest_sha256": self.query_manifest_sha256,
            "frozen_inputs_sha256": self.frozen_inputs_sha256,
            "query_set_sha256": self.query_set_sha256,
            "ordered_frozen_receipts_sha256": (
                self.ordered_frozen_receipts_sha256
            ),
            "max_sources": self._delegate.max_sources,
            "rrf_constant": self._delegate.rrf_constant,
        }

    def __call__(self, condenser, **kwargs):
        if (
            file_sha256(self._verified.store_path / STORE_MANIFEST_NAME)
            != self.base_manifest_sha256
            or file_sha256(self._verified.query_inputs_path / QUERY_MANIFEST_NAME)
            != self.query_manifest_sha256
            or file_sha256(
                self._verified.query_inputs_path / FROZEN_QUERY_INPUTS_NAME
            )
            != self.frozen_inputs_sha256
        ):
            raise RuntimeError("verified-base provider artifacts changed")
        return self._delegate(condenser, **kwargs)


def certify_replay_launcher(path: str | Path) -> ReplayExecutionIdentity:
    """Bind one tracked launcher to a clean checked-out commit."""

    launcher = Path(path).resolve()
    if not launcher.is_file() or launcher.is_symlink():
        raise ValueError("launcher must be a regular non-symlink file")

    root_result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=launcher.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        raise RuntimeError("launcher git certification failed")
    root = Path(root_result.stdout.strip()).resolve()

    def git(*arguments: str, binary: bool = False):
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=not binary,
        )
        if result.returncode != 0:
            raise RuntimeError("launcher git certification failed")
        return result.stdout

    try:
        relative = launcher.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("launcher is outside its git worktree") from exc
    git("ls-files", "--error-unmatch", "--", relative)
    if str(git("status", "--porcelain", "--untracked-files=no")).strip():
        raise RuntimeError("tracked worktree is not clean")
    committed = git("show", f"HEAD:{relative}", binary=True)
    active = launcher.read_bytes()
    if committed != active:
        raise RuntimeError("launcher bytes differ from HEAD")
    return ReplayExecutionIdentity(
        launcher_sha256=hashlib.sha256(active).hexdigest(),
        source_commit=str(git("rev-parse", "HEAD")).strip().casefold(),
        tracked_worktree_clean=True,
    )


def _require_owned_binding(binding: object) -> DiffuseLongMemEvalExecutionBinding:
    if type(binding) is not DiffuseLongMemEvalExecutionBinding:
        raise TypeError("replay requires the exact owned execution binding")
    if not binding.runtime_binding_certified:
        raise RuntimeError("replay runtime binding is not certified")
    if binding.runtime.residency_mode != "resident_bge_qwen":
        raise ValueError("shared-base replay requires resident_bge_qwen")
    if binding.config.retrieval.qwen_rerank or binding.config.retrieval.qwen_feedback:
        raise ValueError(
            "shared-base replay forbids legacy Qwen rerank/feedback before freezing"
        )
    _require_resident_cuda_pair(binding)
    return binding


def _require_resident_cuda_pair(binding: object) -> None:
    """Require the real replay's BGE and Qwen identities on one CUDA device."""

    embedding = str(binding.embedding_identity.get("device", "")).casefold().strip()
    qwen = str(binding.runtime.qwen_device).casefold().strip()

    def canonical(value: str) -> str | None:
        if value == "cuda":
            return "cuda:0"
        prefix = "cuda:"
        ordinal = value[len(prefix):] if value.startswith(prefix) else ""
        if not ordinal.isdigit() or str(int(ordinal)) != ordinal:
            return None
        return f"cuda:{ordinal}"

    embedding_device, qwen_device = canonical(embedding), canonical(qwen)
    if embedding_device is None or qwen_device is None:
        raise ValueError("resident replay requires both BGE and Qwen on CUDA")
    if embedding_device != qwen_device:
        raise ValueError("resident replay requires BGE and Qwen on one CUDA device")


def _blind_sample(sample: TreatmentSampleLike | GoldBlindLongMemEvalSample):
    if isinstance(sample, GoldBlindLongMemEvalSample):
        return sample
    return gold_blind_from_treatment_sample(sample)


@dataclass(frozen=True, slots=True)
class _ExecutedArm:
    clone: DiffuseDerivedStore
    result: DiffuseLongMemEvalRuntimeResult
    finalization: DiffuseDerivedFinalization


def run_diffuse_longmemeval_shared_base_replay(
    sample: TreatmentSampleLike | GoldBlindLongMemEvalSample,
    *,
    treatment_identity: DiffuseBaseTreatmentIdentity,
    binding: DiffuseLongMemEvalExecutionBinding,
    reference_arm: DiffuseLongMemEvalArm,
    cache_root: str | Path,
    replay_root: str | Path,
    launcher_path: str | Path | None = None,
    implementation_digest: str | None = None,
    environment_digest: str | None = None,
) -> DiffuseLongMemEvalReplayReceipt:
    """Run and publish one structurally sanitized matched three-arm replay."""

    binding = _require_owned_binding(binding)
    if type(treatment_identity) is not DiffuseBaseTreatmentIdentity:
        raise TypeError("treatment_identity must be exact")
    if not isinstance(reference_arm, DiffuseLongMemEvalArm):
        raise TypeError("reference_arm must be a diffuse arm")
    blind = _blind_sample(sample)
    arms = matched_diffuse_boundary_arms(reference_arm)
    if tuple(item.arm_id for item in arms) != REPLAY_MODES:
        raise RuntimeError("matched arm factory changed canonical order")
    target = Path(replay_root)
    cache = Path(cache_root)
    target_resolved, cache_resolved = target.resolve(), cache.resolve()
    if target_resolved == cache_resolved or (
        target_resolved.is_relative_to(cache_resolved)
        or cache_resolved.is_relative_to(target_resolved)
    ):
        raise ValueError("replay package and immutable cache must not overlap")
    target.parent.mkdir(parents=True, exist_ok=True)
    require_regular_directory(target.parent, "replay package parent")
    if target.exists():
        raise FileExistsError(target)
    if launcher_path is not None and (
        implementation_digest is not None or environment_digest is not None
    ):
        raise ValueError("certified launcher forbids caller-supplied code digests")
    execution = (
        None if launcher_path is None else certify_replay_launcher(launcher_path)
    )
    runtime_binding_sha256 = binding.binding_sha256
    build_runtime = owned_build_runtime_identity(binding.new_condenser)
    base = publish_diffuse_longmemeval_base(
        cache,
        treatment_identity=treatment_identity,
        sample=blind,
        config=binding.config,
        embedding_identity=binding.embedding_identity,
        build_runtime_identity=build_runtime,
        embedder=binding.embedder,
        condenser_factory=binding.new_condenser,
        implementation_digest=implementation_digest,
        environment_digest=environment_digest,
    )
    provider = VerifiedBaseLegacyDiffuseInputProvider.from_verified_base(
        base,
        max_sources=binding.runtime.source_router_max_sources,
        rrf_constant=binding.runtime.source_router_rrf_constant,
    )
    provider_payload = analysis_callable_identity_payload(
        provider, "verified_base_provider"
    )
    provider_identity = canonical_identity(
        provider_payload,
        identity_sha256(provider_payload),
    )
    observation, qwen = binding.prepare_resident_replay_runtime()
    if binding.binding_sha256 != runtime_binding_sha256:
        raise RuntimeError("runtime binding identity changed after model load")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.replay-", dir=target.parent)
    )
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.replay-work-", dir=target.parent)
    )
    try:
        executed: list[_ExecutedArm] = []
        for arm in arms:
            clone = clone_diffuse_longmemeval_base(
                base,
                workspace / arm.arm_id,
                arm_id=arm.arm_id,
                arm_sha256=arm.arm_sha256,
            )
            condenser = open_diffuse_longmemeval_derived_store(
                clone,
                config=binding.config,
                embedder=binding.embedder,
            )
            try:
                if qwen.reranker is not None:
                    raise RuntimeError("shared-base replay unexpectedly loaded a reranker")
                phase = retrieve_diffuse_longmemeval_sample(
                    condenser,
                    blind,
                    config=binding.config,
                    arm=arm,
                    legacy_input_provider=provider,
                    qwen_scorer=(
                        qwen.scorer
                        if arm.compilation.boundary_mode == "qwen_head"
                        else None
                    ),
                    embedding_identity=binding.embedding_identity,
                    representative_linker=qwen.linker,
                    representative_policy_factory=(
                        binding.representative_policy_factory
                    ),
                )
            finally:
                condenser.close()
            result = DiffuseLongMemEvalRuntimeResult(
                phase=phase,
                runtime_binding_sha256=runtime_binding_sha256,
                runtime_binding_certified=binding.runtime_binding_certified,
                residency_preflight=observation,
            )
            finalization = finalize_diffuse_longmemeval_derived_store(
                clone,
                phase=phase,
            )
            executed.append(_ExecutedArm(clone, result, finalization))
        matched = validate_matched_diffuse_runtime_results(
            tuple(item.result for item in executed)
        )
        if binding.binding_sha256 != runtime_binding_sha256 or not (
            binding.runtime_binding_certified
        ):
            raise RuntimeError("runtime binding changed during replay")
        arm_records = tuple(
            build_replay_arm_record(item, matched.matched_suite.probes) for item in executed
        )
        workspace_children = {
            *(arm.arm_id for arm in arms),
            *(f".{arm.arm_id}.publish.lock" for arm in arms),
        }
        require_exact_children(workspace, workspace_children, "replay workspace")
        for arm in arms:
            require_regular_directory(
                workspace / arm.arm_id, f"completed {arm.arm_id} arm"
            )
            require_regular_file(
                workspace / f".{arm.arm_id}.publish.lock",
                f"{arm.arm_id} publication lock",
            )
            (workspace / arm.arm_id).replace(staging / arm.arm_id)
        require_exact_children(
            staging, {arm.arm_id for arm in arms}, "replay staging"
        )
        safe_remove_staging(workspace, target.parent)
        inventory = build_replay_file_inventory(staging)
        retrieval_payload = binding.config.retrieval.model_dump(mode="json")
        eval_payload = binding.config.model_dump(mode="json")
        evaluation_payload = {
            "chunker": binding.config.chunker.model_dump(mode="json"),
            "retrieval": retrieval_payload,
            "max_prompt_tokens": binding.config.max_prompt_tokens,
        }
        matched_phase_payload = matched.matched_suite.identity_payload()
        matched_runtime_payload = matched.identity_payload()
        values: dict[str, object] = {
            "sample_id_sha256": base.store_manifest.sample_id_sha256,
            "treatment_identity_sha256": (
                base.query_manifest.treatment_identity_sha256
            ),
            "base_manifest_file_sha256": base.store_manifest_sha256,
            "base_manifest": base.store_manifest,
            "query_manifest_file_sha256": base.query_manifest_sha256,
            "query_manifest": base.query_manifest,
            "verified_base_provider_identity": provider_identity,
            "eval_config": canonical_identity(
                eval_payload, identity_sha256(eval_payload)
            ),
            "retrieval_policy": canonical_identity(
                retrieval_payload, identity_sha256(retrieval_payload)
            ),
            "evaluation_policy": canonical_identity(
                evaluation_payload, identity_sha256(evaluation_payload)
            ),
            "runtime_binding": canonical_identity(
                dict(binding.analysis_identity_payload()),
                runtime_binding_sha256,
            ),
            "matched_phase_suite": canonical_identity(
                matched_phase_payload,
                matched.matched_suite.receipt_sha256,
                self_hash_field="receipt_sha256",
            ),
            "matched_runtime_suite": canonical_identity(
                matched_runtime_payload,
                matched.receipt_sha256,
                self_hash_field="receipt_sha256",
            ),
            "execution_identity": execution,
            "launcher_binding_certified": execution is not None,
            "arms": arm_records,
            "files": inventory,
            "qa_responder_or_judge_calls": 0,
            "retrieval_input_schema_contains_gold_fields": False,
            "treatment_population_membership_certified": False,
            "provider_transports_invoked_by_runner": 0,
        }
        unsigned = {
            key: (
                value.model_dump(mode="json")
                if hasattr(value, "model_dump")
                else [item.model_dump(mode="json") for item in value]
                if isinstance(value, tuple)
                else value
            )
            for key, value in values.items()
        }
        unsigned["format"] = "memory-condense-longmemeval-shared-base-replay-v1"
        receipt = DiffuseLongMemEvalReplayReceipt(
            **values,
            receipt_sha256=identity_sha256(unsigned),
        )
        write_new_bytes(
            staging / REPLAY_MANIFEST_NAME,
            canonical_json_bytes(receipt.model_dump(mode="json")),
        )
        publish_complete_directory(
            staging,
            target,
            manifest_name=REPLAY_MANIFEST_NAME,
        )
    except BaseException:
        if staging.exists():
            safe_remove_staging(staging, target.parent)
        if workspace.exists():
            safe_remove_staging(workspace, target.parent)
        raise
    return verify_diffuse_longmemeval_replay_package(
        target,
        base=base,
        expected_runtime_binding_sha256=runtime_binding_sha256,
    )


def _resolve_provider_parameters(
    receipt: DiffuseLongMemEvalReplayReceipt,
    base: VerifiedDiffuseLongMemEvalBase,
    *,
    historical_provider_identity_proof: (
        provider_history.HistoricalProviderIdentityProof | None
    ),
) -> tuple[int, int]:
    """Validate the facade-bound provider and return its routing controls."""

    provider_body = json.loads(
        receipt.verified_base_provider_identity.canonical_identity_json
    )
    provider_declaration = provider_body["declared_identity"]
    max_sources = int(provider_declaration["max_sources"])
    rrf_constant = int(provider_declaration["rrf_constant"])
    expected_provider = VerifiedBaseLegacyDiffuseInputProvider.from_verified_base(
        base,
        max_sources=max_sources,
        rrf_constant=rrf_constant,
    )
    expected_provider_payload = analysis_callable_identity_payload(
        expected_provider,
        "verified_base_provider",
    )
    expected_provider_identity = canonical_identity(
        expected_provider_payload,
        identity_sha256(expected_provider_payload),
    )
    if receipt.verified_base_provider_identity != expected_provider_identity:
        # Reconstruction never invokes this execution-time provider. The
        # optional proof authenticates only the frozen v1 callable identity.
        provider_history.require_historical_provider_compatibility(
            historical_provider_identity_proof,
            execution_identity=receipt.execution_identity,
            recorded_identity=receipt.verified_base_provider_identity,
            current_identity_payload=expected_provider_payload,
        )
    return max_sources, rrf_constant


def verify_and_reconstruct_diffuse_longmemeval_replay_package(
    path: str | Path,
    *,
    base: VerifiedDiffuseLongMemEvalBase,
    expected_runtime_binding_sha256: str,
    historical_provider_identity_proof: (
        provider_history.HistoricalProviderIdentityProof | None
    ) = None,
) -> VerifiedDiffuseReplayPackage:
    """Verify one replay and return the packets reconstructed by that pass."""

    def resolve_provider_parameters(
        receipt: DiffuseLongMemEvalReplayReceipt,
        verified_base: VerifiedDiffuseLongMemEvalBase,
    ) -> tuple[int, int]:
        return _resolve_provider_parameters(
            receipt,
            verified_base,
            historical_provider_identity_proof=(
                historical_provider_identity_proof
            ),
        )

    package = _reconstruct_replay_package(
        path,
        base=base,
        expected_runtime_binding_sha256=expected_runtime_binding_sha256,
        resolve_provider_parameters=resolve_provider_parameters,
    )
    manifest_path = Path(path) / REPLAY_MANIFEST_NAME
    require_regular_file(manifest_path, "replay manifest")
    before = (
        file_sha256(manifest_path),
        manifest_path.stat().st_mtime_ns,
        manifest_path.stat().st_size,
    )
    raw = manifest_path.read_bytes()
    require_regular_file(manifest_path, "replay manifest")
    after = (
        file_sha256(manifest_path),
        manifest_path.stat().st_mtime_ns,
        manifest_path.stat().st_size,
    )
    expected = canonical_json_bytes(package.receipt.model_dump(mode="json"))
    if (
        before != after
        or after[0] != package.manifest_file_sha256
        or hashlib.sha256(raw).hexdigest() != after[0]
        or raw != expected
    ):
        raise RuntimeError("replay manifest changed after packet reconstruction")
    return package


def verify_diffuse_longmemeval_replay_package(
    path: str | Path,
    *,
    base: VerifiedDiffuseLongMemEvalBase,
    expected_runtime_binding_sha256: str,
) -> DiffuseLongMemEvalReplayReceipt:
    """Strictly verify deterministic replay against its base and runtime."""

    return verify_and_reconstruct_diffuse_longmemeval_replay_package(
        path,
        base=base,
        expected_runtime_binding_sha256=expected_runtime_binding_sha256,
    ).receipt


__all__ = [
    "DiffuseLongMemEvalReplayReceipt",
    "ReplayExecutionIdentity",
    "VerifiedBaseLegacyDiffuseInputProvider",
    "certify_replay_launcher",
    "run_diffuse_longmemeval_shared_base_replay",
    "verify_and_reconstruct_diffuse_longmemeval_replay_package",
    "verify_diffuse_longmemeval_replay_package",
]
