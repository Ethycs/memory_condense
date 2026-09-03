#!/usr/bin/env python3
"""Concrete policy-v5-r3 prediction adapters through adaptive tail.

The executor owns phase ordering and provider authorization.  This module owns
the production implementation of the first eleven phases: it composes the
already-tested confirmation APIs, publishes only content-addressed artifacts,
and reconstructs downstream Python objects through their exact replay APIs.

No validation artifact, benchmark answer, scorer, or judge is imported here.
Provider clients are constructed only inside an already-authorized execution
call, and every provider-bearing phase authenticates its existing journals
before declaring the exact remaining call count.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from tools import confirmation_adaptive_source_map as adaptive_map
from tools import confirmation_adaptive_tail as adaptive_tail
from tools import confirmation_contracts as contracts
from tools import confirmation_cumulative_retrieval as cumulative
from tools import confirmation_evidence_map_parent as evidence_map
from tools import confirmation_namespace_store_adapter as namespace_store
from tools import confirmation_production_runtime as production_runtime
from tools import confirmation_protected_s0_plane as protected_s0
from tools import confirmation_query_expansion_adapter as query_expansion
from tools import confirmation_query_payload_parent as query_payload
from tools import confirmation_s0_prompt_preflight as s0_prompt
from tools import confirmation_semantic_planes as semantic_planes
from tools import confirmation_source_streams as source_streams
from tools import confirmation_staged_cumulative_coordinator as staged
from tools import confirmation_terra_completion_lifecycle as terra_lifecycle
from tools.matched_eval import query_evidence_map_solver_v2_live as map_live
from tools.matched_eval import query_expansion as query_live
from tools.matched_eval import query_payload_live
from tools.run_confirmation_policy_v5_r3 import (
    PhaseArtifact,
    PhaseOutcome,
    PhaseRequest,
    PredictionPhaseAdapter,
    ProviderRequirement,
    ensure_owned_run_directory,
)
from tools.confirmation_canonical import canonical_sha256


FORMAT = "memory-condense-confirmation-production-first-half-adapters-v1"
PRODUCTION_DIRECTORY_NAME = "confirmation-production"
TARGET_TOKENS = 1_000_000
API_KEY_ENV = "LITELLM_KEY"

FIRST_HALF_PHASE_IDS = (
    "namespace_ingest",
    "staged_cumulative_s0_s3",
    "s0_terra_answer",
    "protected_s0",
    "query_expansion",
    "query_direct_answer",
    "evidence_map",
    "source_streams",
    "adaptive_source_map",
    "adaptive_evidence_solver",
    "adaptive_tail",
)

PROVIDER_CLASS_BY_PHASE: Mapping[str, str | None] = MappingProxyType(
    {
        "namespace_ingest": None,
        "staged_cumulative_s0_s3": None,
        "s0_terra_answer": "terra",
        "protected_s0": None,
        "query_expansion": "terra",
        "query_direct_answer": "terra",
        "evidence_map": "terra",
        "source_streams": None,
        "adaptive_source_map": "terra",
        "adaptive_evidence_solver": "terra",
        "adaptive_tail": "terra",
    }
)

NAMESPACE_WORKSET_NAME = "confirmation-namespace-workset-v1.json"
NAMESPACE_EXECUTION_NAME = "confirmation-namespace-execution-v1.json"
STAGED_EXECUTION_NAME = "confirmation-staged-execution-v1.json"
CUMULATIVE_MERGE_NAME = "confirmation-cumulative-merge-v1.json"
S0_PROMPT_NAME = "confirmation-s0-prompts-v1.json"
PROTECTED_S0_NAME = "confirmation-protected-s0-v1.json"
PROTECTED_S0_REPLAY_NAME = "confirmation-protected-s0-replay-v1.json"
QUERY_RETRIEVER_AUDIT_NAME = "confirmation-query-retriever-audit-v1.json"


class ConfirmationProductionPhaseAdapterError(ValueError):
    """A production phase or exact replay failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationProductionPhaseAdapterError(message)


def _sha(value: object, label: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and set(value) <= set("0123456789abcdef"),
        f"{label} is not a lowercase SHA-256",
    )
    return str(value)


def _phase_root(output_root: Path, phase_id: str) -> Path:
    _require(phase_id in PROVIDER_CLASS_BY_PHASE, "unknown first-half phase")
    return ensure_owned_run_directory(
        output_root,
        Path(PRODUCTION_DIRECTORY_NAME) / phase_id,
        create=True,
    )


def _sealed(path: Path, label: str) -> namespace_store.SealedPayload:
    return namespace_store.read_sealed_payload(path, label=label)


def _dependency_artifact(
    request: PhaseRequest,
    phase_id: str,
    role: str,
) -> namespace_store.SealedPayload:
    """Resolve one prior artifact solely through its declared checkpoint role."""

    _require(
        phase_id in request.dependency_checkpoints,
        f"required phase checkpoint is undeclared: {phase_id}",
    )
    checkpoint = request.dependency_checkpoints[phase_id]
    matches = [
        row
        for row in checkpoint.payload.get("artifacts", ())
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    _require(len(matches) == 1, f"required phase artifact is absent: {phase_id}/{role}")
    row = matches[0]
    relative = Path(str(row.get("path")))
    _require(not relative.is_absolute(), "dependency artifact path is absolute")
    path = (request.output_root / relative).resolve()
    try:
        path.relative_to(request.output_root.resolve())
    except ValueError as exc:
        raise ConfirmationProductionPhaseAdapterError(
            "dependency artifact escapes the execution root"
        ) from exc
    sealed = _sealed(path, f"{phase_id} {role}")
    _require(
        sealed.sha256 == _sha(row.get("sha256"), f"{phase_id} {role}"),
        f"dependency artifact differs: {phase_id}/{role}",
    )
    return sealed


def _prior_or_owned_artifact(
    request: PhaseRequest,
    *,
    phase_id: str,
    role: str,
    owned_path: Path,
    label: str,
) -> namespace_store.SealedPayload:
    if phase_id in request.dependency_checkpoints:
        return _dependency_artifact(request, phase_id, role)
    return _sealed(owned_path, label)


def _artifact(role: str, value: Any) -> PhaseArtifact:
    path = Path(value.path).resolve()
    return PhaseArtifact(role=role, path=path, sha256=_sha(value.sha256, role))


def _publish_summary(
    path: Path,
    *,
    format_suffix: str,
    body: Mapping[str, Any],
) -> contracts.SealedJson:
    unsigned = {
        "format": f"{FORMAT}-{format_suffix}",
        **dict(body),
    }
    payload = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    artifact, _created = contracts.publish_sealed_json(path, payload)
    return artifact


def _zero_requirement() -> ProviderRequirement:
    return ProviderRequirement(None, 0, 0, 0, retry_limit=0)


def _terra_requirement(total: int, completed: int) -> ProviderRequirement:
    _require(type(total) is int and total >= 0, "provider total is invalid")
    _require(
        type(completed) is int and 0 <= completed <= total,
        "provider checkpoint count is invalid",
    )
    return ProviderRequirement(
        "terra",
        total,
        completed,
        total - completed,
        retry_limit=0,
    )


TerraClientFactory = Callable[[str, str], Any]
QuerySessionFactory = Callable[[Any], Any]


class ConfirmationProductionAdapterEnvironment:
    """Shared inert/local resources and provider construction boundary."""

    def __init__(
        self,
        *,
        policy_manifest_sha256: str,
        qwen_prefix_model_dir: str | Path,
        qwen_choice_model_dir: str | Path,
        terra_client_factory: TerraClientFactory = terra_lifecycle._default_client_factory,  # noqa: SLF001
        query_session_factory: QuerySessionFactory | None = None,
        token_counter: Callable[[str], int] | None = None,
        target_tokens: int = TARGET_TOKENS,
        api_key_env: str = API_KEY_ENV,
        runtime_builder: Callable[..., Any] = production_runtime.build_confirmation_production_runtime,
        runtime_builder_identity_sha256: str | None = None,
    ) -> None:
        self.policy_manifest_sha256 = _sha(
            policy_manifest_sha256, "environment policy manifest"
        )
        self.qwen_prefix_model_dir = Path(qwen_prefix_model_dir)
        self.qwen_choice_model_dir = Path(qwen_choice_model_dir)
        self.terra_client_factory = terra_client_factory
        self.query_session_factory = query_session_factory
        self.token_counter = token_counter
        self.target_tokens = target_tokens
        self.api_key_env = api_key_env
        self._runtime_builder = runtime_builder
        self._runtime: Any | None = None
        self._initial_runtime_consumed = False
        _require(
            type(target_tokens) is int and target_tokens > 0,
            "namespace token target must be positive",
        )
        _require(
            type(api_key_env) is str and bool(api_key_env.strip()),
            "Terra API-key environment name must be nonblank",
        )
        _require(callable(terra_client_factory), "Terra client factory is absent")
        _require(callable(runtime_builder), "production runtime builder is absent")
        _require(
            query_session_factory is None or callable(query_session_factory),
            "query session factory is invalid",
        )
        default_builder_identity = canonical_sha256(
            {
                "format": f"{FORMAT}-runtime-builder-v1",
                "factory": (
                    "tools.confirmation_production_runtime."
                    "build_confirmation_production_runtime"
                ),
                "full_config_sha256": production_runtime.FROZEN_FULL_CONFIG_SHA256,
                "retrieval_policy_sha256": (
                    production_runtime.FROZEN_RETRIEVAL_POLICY_SHA256
                ),
                "source_config_sha256": production_runtime.FROZEN_SOURCE_CONFIG_SHA256,
            }
        )
        self.runtime_builder_identity_sha256 = _sha(
            runtime_builder_identity_sha256 or default_builder_identity,
            "runtime builder identity",
        )

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "format": f"{FORMAT}-environment-v1",
                "policy_manifest_sha256": self.policy_manifest_sha256,
                "qwen_choice_model_dir": str(self.qwen_choice_model_dir.resolve()),
                "qwen_prefix_model_dir": str(self.qwen_prefix_model_dir.resolve()),
                "runtime_builder_identity_sha256": self.runtime_builder_identity_sha256,
                "target_tokens": self.target_tokens,
                # Bind only the selector, never the secret value.
                "api_key_env": self.api_key_env,
                "retry_limit": 0,
            }
        )

    def initial_runtime(self) -> Any:
        """Lazily build the shared ingest/staged runtime and never rebuild it."""

        _require(
            not self._initial_runtime_consumed,
            "initial BGE/Qwen runtime was already released",
        )
        if self._runtime is None:
            runtime = self._runtime_builder(
                policy_manifest_sha256=self.policy_manifest_sha256,
                qwen_prefix_model_dir=self.qwen_prefix_model_dir,
                qwen_choice_model_dir=self.qwen_choice_model_dir,
            )
            _sha(getattr(runtime, "identity_sha256", None), "production runtime")
            _require(
                getattr(runtime, "policy_manifest_sha256", None)
                == self.policy_manifest_sha256,
                "production runtime binds another policy",
            )
            self._runtime = runtime
        return self._runtime

    def mark_initial_runtime_consumed(self) -> None:
        self._initial_runtime_consumed = True
        self._runtime = None

    def open_query_session(self, context: Any) -> Any:
        if self.query_session_factory is not None:
            return self.query_session_factory(context)
        return production_runtime.build_confirmation_query_retriever_session(
            context=context,
            policy_manifest_sha256=self.policy_manifest_sha256,
            qwen_prefix_model_dir=self.qwen_prefix_model_dir,
            qwen_choice_model_dir=self.qwen_choice_model_dir,
        )


@dataclass(frozen=True, slots=True)
class RestoredStagedExecution:
    """Durable subset needed after the resident staged runtime has closed."""

    cumulative: cumulative.ConfirmationCumulativeExecution
    barrier: namespace_store.SealedPayload
    cumulative_merge: namespace_store.SealedPayload
    semantic_facet_preparation_artifact: namespace_store.SealedPayload
    semantic_facet_release_artifact: namespace_store.SealedPayload


@dataclass(frozen=True, slots=True)
class FirstHalfState:
    """Exact replayed handoff consumed by typed and semantic final adapters."""

    query_context: query_expansion.ConfirmationQueryExpansionContext
    query_artifacts: query_payload.VerifiedQueryExpansionArtifacts
    query_payload_plan: query_payload.ConfirmationQueryPayloadPlan
    direct_plane: Any
    evidence_map_plan: evidence_map.ConfirmationEvidenceMapPlan
    evidence_map_plane: Any
    source_streams: source_streams.ConfirmationSourceStreamsResult
    source_map: adaptive_map.VerifiedConfirmationAdaptiveSourceMapPlane
    adaptive_upstream: adaptive_tail.ConfirmationAdaptiveUpstream
    adaptive_evidence: adaptive_tail.VerifiedConfirmationAdaptiveEvidencePlane
    adaptive_tail: adaptive_tail.VerifiedConfirmationAdaptiveTailPlane
    cumulative_inputs: cumulative.ConfirmationCumulativeInput
    staged_execution: RestoredStagedExecution
    staged_output_root: Path
    staged_barrier: namespace_store.SealedPayload
    semantic_facet_preparation_artifact: namespace_store.SealedPayload
    semantic_facet_release_artifact: namespace_store.SealedPayload
    semantic_facet_release: namespace_store.SealedPayload


class FirstHalfOperations(Protocol):
    identity_sha256: str

    def prepare(self, phase_id: str, request: PhaseRequest) -> ProviderRequirement: ...

    def execute(
        self,
        phase_id: str,
        request: PhaseRequest,
        *,
        requirement: ProviderRequirement,
        enable_provider: bool,
        authorized_provider_calls: int,
    ) -> PhaseOutcome: ...

    def replay(
        self,
        phase_id: str,
        request: PhaseRequest,
        checkpoint: contracts.SealedJson,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ConfirmationProductionPhaseAdapter:
    """Thin executor-protocol adapter over one first-half operation."""

    phase_id: str
    provider_class: str | None
    operations: FirstHalfOperations
    identity_sha256: str

    def prepare(self, request: PhaseRequest) -> ProviderRequirement:
        _require(request.spec.phase_id == self.phase_id, "phase request differs")
        return self.operations.prepare(self.phase_id, request)

    def execute(
        self,
        request: PhaseRequest,
        *,
        requirement: ProviderRequirement,
        enable_provider: bool,
        authorized_provider_calls: int,
    ) -> PhaseOutcome:
        _require(request.spec.phase_id == self.phase_id, "phase request differs")
        return self.operations.execute(
            self.phase_id,
            request,
            requirement=requirement,
            enable_provider=enable_provider,
            authorized_provider_calls=authorized_provider_calls,
        )

    def replay(self, request: PhaseRequest, checkpoint: contracts.SealedJson) -> None:
        _require(request.spec.phase_id == self.phase_id, "phase request differs")
        self.operations.replay(self.phase_id, request, checkpoint)


class ProductionFirstHalfOperations:
    """Compose the actual confirmation implementations into executor phases."""

    def __init__(self, environment: ConfirmationProductionAdapterEnvironment) -> None:
        self.environment = environment
        self.identity_sha256 = canonical_sha256(
            {
                "format": f"{FORMAT}-operations-v1",
                "environment_identity_sha256": environment.identity_sha256,
                "phase_ids": list(FIRST_HALF_PHASE_IDS),
                "provider_classes": dict(PROVIDER_CLASS_BY_PHASE),
            }
        )
        self._cache: dict[tuple[str, str], Any] = {}
        self._replayed: set[tuple[str, str]] = set()

    @staticmethod
    def _key(request: PhaseRequest, label: str) -> tuple[str, str]:
        return (str(request.output_root.resolve()), label)

    def _remember(self, request: PhaseRequest, label: str, value: Any) -> Any:
        self._cache[self._key(request, label)] = value
        return value

    def _cached(self, request: PhaseRequest, label: str) -> Any | None:
        return self._cache.get(self._key(request, label))

    @staticmethod
    def _policy_freeze(request: PhaseRequest) -> namespace_store.SealedPayload:
        artifact = request.context.runtime_policy
        return namespace_store.SealedPayload(
            Path(artifact.path), artifact.runtime_policy_sha256, artifact.payload
        )

    def _decode_workset(
        self,
        request: PhaseRequest,
        sealed: namespace_store.SealedPayload,
    ) -> namespace_store.ConfirmationNamespaceWorkset:
        value = sealed.payload
        raw_namespaces = value.get("namespaces")
        _require(isinstance(raw_namespaces, list), "namespace workset rows are absent")
        namespaces: list[namespace_store.ConfirmationNamespaceWork] = []
        for raw in raw_namespaces:
            _require(isinstance(raw, Mapping), "namespace work row changed type")
            raw_probes = raw.get("probes")
            raw_haystack = raw.get("haystack")
            _require(
                isinstance(raw_probes, list) and isinstance(raw_haystack, list),
                "namespace work membership is absent",
            )
            probes = tuple(
                namespace_store.ProbeBinding(
                    question_id=str(row["question_id"]),
                    row_receipt_sha256=_sha(row["row_receipt_sha256"], "probe row"),
                    content_binding_sha256=_sha(
                        row["content_binding_sha256"], "probe content"
                    ),
                )
                for row in raw_probes
                if isinstance(row, Mapping)
            )
            haystack = tuple(
                namespace_store.NamespaceMember(
                    row_receipt_sha256=_sha(row["row_receipt_sha256"], "member row"),
                    content_binding_sha256=_sha(
                        row["content_binding_sha256"], "member content"
                    ),
                    content_occurrence=int(row["content_occurrence"]),
                    transcript_tokens=int(row["transcript_tokens"]),
                )
                for row in raw_haystack
                if isinstance(row, Mapping)
            )
            _require(
                len(probes) == len(raw_probes) and len(haystack) == len(raw_haystack),
                "namespace work membership row changed type",
            )
            work = namespace_store.ConfirmationNamespaceWork(
                namespace_id=_sha(raw.get("namespace_id"), "namespace ID"),
                namespace_receipt_sha256=_sha(
                    raw.get("namespace_receipt_sha256"), "namespace receipt"
                ),
                namespace_store_id=_sha(
                    raw.get("namespace_store_id"), "namespace store ID"
                ),
                target_tokens=int(raw.get("target_tokens")),
                actual_tokens=int(raw.get("actual_tokens")),
                probes=probes,
                haystack=haystack,
                work_receipt_sha256=_sha(
                    raw.get("work_receipt_sha256"), "namespace work receipt"
                ),
            )
            _require(work.projection() == dict(raw), "namespace work receipt differs")
            namespaces.append(work)
        result = namespace_store.ConfirmationNamespaceWorkset(
            treatment_file_sha256=_sha(
                value.get("treatment_file_sha256"), "workset treatment"
            ),
            sanitized_projection_sha256=_sha(
                value.get("sanitized_projection_sha256"), "workset projection"
            ),
            dataset_sha256=_sha(value.get("dataset_sha256"), "workset dataset"),
            split_manifest_sha256=_sha(
                value.get("split_manifest_sha256"), "workset split"
            ),
            preflight_sha256=_sha(value.get("preflight_sha256"), "workset preflight"),
            freeze_sha256=_sha(value.get("freeze_sha256"), "workset freeze"),
            target_tokens=int(value.get("target_tokens")),
            namespaces=tuple(namespaces),
            workset_identity_sha256=_sha(
                value.get("workset_identity_sha256"), "workset identity"
            ),
        )
        treatment = request.context.treatment
        _require(
            result.projection() == dict(value)
            and result.treatment_file_sha256 == treatment.file_sha256
            and result.sanitized_projection_sha256
            == treatment.sanitized_projection_sha256
            and result.dataset_sha256 == treatment.dataset_sha256
            and result.split_manifest_sha256 == treatment.split_manifest_sha256
            and result.preflight_sha256 == request.context.preflight.sha256
            and result.freeze_sha256 == request.context.runtime_policy.sha256
            and result.target_tokens == self.environment.target_tokens,
            "sealed namespace workset escaped confirmation inputs",
        )
        declared = request.context.preflight.payload.get("namespaces")
        _require(
            isinstance(declared, list) and len(declared) == len(result.namespaces),
            "namespace schedule differs from workset",
        )
        for raw, work in zip(declared, result.namespaces, strict=True):
            _require(
                isinstance(raw, Mapping)
                and raw.get("namespace_id") == work.namespace_id
                and raw.get("namespace_receipt_sha256")
                == work.namespace_receipt_sha256
                and raw.get("question_ids")
                == [probe.question_id for probe in work.probes],
                "namespace probe membership differs from preflight",
            )
        return result

    def _workset(
        self, request: PhaseRequest
    ) -> namespace_store.ConfirmationNamespaceWorkset:
        cached = self._cached(request, "workset")
        if cached is not None:
            return cached
        path = _phase_root(request.output_root, "namespace_ingest") / NAMESPACE_WORKSET_NAME
        if "namespace_ingest" in request.dependency_checkpoints:
            sealed = _dependency_artifact(
                request, "namespace_ingest", "namespace_workset"
            )
            result = self._decode_workset(request, sealed)
        else:
            result = namespace_store.compile_confirmation_namespace_workset(
                request.context.treatment,
                preflight=request.context.preflight,
                freeze=self._policy_freeze(request),
                target_tokens=self.environment.target_tokens,
                token_counter=self.environment.token_counter,
            )
        if "namespace_ingest" not in request.dependency_checkpoints and path.exists():
            sealed = _sealed(path, "namespace workset")
            _require(
                sealed.payload == result.projection(),
                "owned namespace workset differs from reconstruction",
            )
        return self._remember(request, "workset", result)

    def _base_execution(
        self, request: PhaseRequest
    ) -> namespace_store.ConfirmationNamespaceExecution:
        cached = self._cached(request, "base_execution")
        if cached is not None:
            return cached
        workset = self._workset(request)
        summary = _dependency_artifact(
            request, "namespace_ingest", "namespace_execution"
        )
        raw_rows = summary.payload.get("checkpoints")
        _require(isinstance(raw_rows, list), "namespace checkpoint registry is absent")
        paths: list[Path] = []
        checkpoints: list[namespace_store.SealedPayload] = []
        for raw in raw_rows:
            _require(isinstance(raw, Mapping), "namespace checkpoint registry changed")
            relative = Path(str(raw.get("path")))
            _require(not relative.is_absolute(), "namespace checkpoint path is absolute")
            path = (request.output_root / relative).resolve()
            _require(
                path.is_relative_to(request.output_root.resolve()),
                "namespace checkpoint escapes output root",
            )
            sealed = _sealed(path, "namespace checkpoint")
            _require(
                sealed.sha256 == _sha(raw.get("sha256"), "namespace checkpoint"),
                "namespace checkpoint differs from registry",
            )
            paths.append(path)
            checkpoints.append(sealed)
        _require(
            summary.payload.get("workset_identity_sha256")
            == workset.workset_identity_sha256
            and summary.payload.get("checkpoint_sha256s")
            == [row.sha256 for row in checkpoints],
            "namespace execution summary differs",
        )
        result = namespace_store.ConfirmationNamespaceExecution(
            checkpoint_paths=tuple(paths),
            checkpoint_sha256s=tuple(row.sha256 for row in checkpoints),
            created_count=0,
            reused_count=len(paths),
            physical_provider_calls=0,
        )
        return self._remember(request, "base_execution", result)

    def _cumulative_inputs(
        self, request: PhaseRequest
    ) -> cumulative.ConfirmationCumulativeInput:
        cached = self._cached(request, "cumulative_inputs")
        if cached is not None:
            return cached
        result = cumulative.ConfirmationCumulativeInput(
            treatment=request.context.treatment,
            preflight=request.context.preflight,
            policy_freeze=self._policy_freeze(request),
            workset=self._workset(request),
            base_execution=self._base_execution(request),
        )
        return self._remember(request, "cumulative_inputs", result)

    def _cumulative_execution(
        self, request: PhaseRequest
    ) -> cumulative.ConfirmationCumulativeExecution:
        cached = self._cached(request, "cumulative_execution")
        if cached is not None:
            return cached
        summary = _dependency_artifact(
            request, "staged_cumulative_s0_s3", "staged_execution"
        )
        raw_rows = summary.payload.get("cumulative_checkpoints")
        _require(isinstance(raw_rows, list), "cumulative checkpoint registry is absent")
        paths: list[Path] = []
        checkpoints: list[namespace_store.SealedPayload] = []
        for raw in raw_rows:
            _require(isinstance(raw, Mapping), "cumulative checkpoint registry changed")
            relative = Path(str(raw.get("path")))
            _require(not relative.is_absolute(), "cumulative checkpoint path is absolute")
            path = (request.output_root / relative).resolve()
            _require(
                path.is_relative_to(request.output_root.resolve()),
                "cumulative checkpoint escapes output root",
            )
            sealed = _sealed(path, "cumulative checkpoint")
            _require(
                sealed.sha256 == _sha(raw.get("sha256"), "cumulative checkpoint"),
                "cumulative checkpoint differs from registry",
            )
            paths.append(path)
            checkpoints.append(sealed)
        identities = {str(row.payload.get("backend_identity_sha256")) for row in checkpoints}
        _require(len(identities) == 1, "cumulative backend identities differ")
        _require(
            summary.payload.get("cumulative_checkpoint_sha256s")
            == [row.sha256 for row in checkpoints]
            and summary.payload.get("namespace_count") == len(checkpoints),
            "cumulative execution summary differs",
        )
        result = cumulative.ConfirmationCumulativeExecution(
            checkpoint_paths=tuple(paths),
            checkpoint_sha256s=tuple(row.sha256 for row in checkpoints),
            backend_identity_sha256=_sha(identities.pop(), "cumulative backend"),
            created_count=0,
            reused_count=len(paths),
            physical_provider_calls=0,
        )
        return self._remember(request, "cumulative_execution", result)

    def _restored_staged(self, request: PhaseRequest) -> RestoredStagedExecution:
        cached = self._cached(request, "restored_staged")
        if cached is not None:
            return cached
        inputs = self._cumulative_inputs(request)
        execution = self._cumulative_execution(request)
        merge = _dependency_artifact(
            request, "staged_cumulative_s0_s3", "cumulative_merge"
        )
        _require(
            merge.payload
            == cumulative.replay_confirmation_cumulative_merge(
                inputs,
                cumulative_execution=execution,
                token_counter=self.environment.token_counter,
            ),
            "cumulative merge replay differs",
        )
        barrier = _dependency_artifact(
            request, "staged_cumulative_s0_s3", "staged_barrier"
        )
        staged._verified_qwen_barrier(  # noqa: SLF001
            barrier,
            qwen_factory_identity_sha256=_sha(
                barrier.payload.get("qwen_factory_identity_sha256"),
                "barrier Qwen factory",
            ),
        )
        facet_preparation = _dependency_artifact(
            request, "staged_cumulative_s0_s3", "semantic_facet_preparation"
        )
        facet_release = _dependency_artifact(
            request, "staged_cumulative_s0_s3", "semantic_facet_release"
        )
        result = RestoredStagedExecution(
            cumulative=execution,
            barrier=barrier,
            cumulative_merge=merge,
            semantic_facet_preparation_artifact=facet_preparation,
            semantic_facet_release_artifact=facet_release,
        )
        return self._remember(request, "restored_staged", result)

    def _s0_kwargs(self, request: PhaseRequest) -> dict[str, Any]:
        restored = self._restored_staged(request)
        return {
            "runtime_policy_path": request.context.runtime_policy.path,
            "expected_runtime_policy_sha256": (
                request.context.runtime_policy.runtime_policy_sha256
            ),
            "treatment_input_path": request.context.treatment_artifact.path,
            "expected_treatment_input_sha256": request.context.treatment_artifact.sha256,
            "treatment_preflight_path": request.context.preflight_artifact.path,
            "expected_treatment_preflight_sha256": request.context.preflight_artifact.sha256,
            "cumulative_retrieval_path": restored.cumulative_merge.path,
            "expected_cumulative_retrieval_sha256": restored.cumulative_merge.sha256,
        }

    def _s0_prompt_and_preflight(self, request: PhaseRequest) -> tuple[Any, Any, Any]:
        root = _phase_root(request.output_root, "s0_terra_answer")
        if "s0_terra_answer" in request.dependency_checkpoints:
            prompt = _dependency_artifact(request, "s0_terra_answer", "s0_prompt")
            preflight = _dependency_artifact(
                request, "s0_terra_answer", "s0_lifecycle_preflight"
            )
        else:
            prompt, _created = s0_prompt.publish_confirmation_s0_preflight(
                root / S0_PROMPT_NAME, **self._s0_kwargs(request)
            )
            preflight, _created = terra_lifecycle.publish_lifecycle_preflight(
                prompt_artifact_path=prompt.path,
                expected_prompt_artifact_sha256=prompt.sha256,
                output_root=root,
            )
        verified = terra_lifecycle.verify_prompt_artifact(
            prompt.path, expected_sha256=prompt.sha256
        )
        return prompt, verified, preflight

    def _protected_inputs(self, request: PhaseRequest) -> dict[str, Any]:
        prompt, _verified, preflight = self._s0_prompt_and_preflight(request)
        release = _dependency_artifact(
            request, "s0_terra_answer", "s0_provider_release"
        )
        completion = _dependency_artifact(request, "s0_terra_answer", "s0_completion")
        return {
            **self._s0_kwargs(request),
            "s0_prompt_path": prompt.path,
            "expected_s0_prompt_sha256": prompt.sha256,
            "s0_completion_path": completion.path,
            "expected_s0_completion_sha256": completion.sha256,
            "expected_s0_lifecycle_preflight_sha256": preflight.sha256,
            "expected_s0_provider_release_sha256": release.sha256,
        }

    def _protected_plane(self, request: PhaseRequest) -> tuple[Any, Any]:
        cached = self._cached(request, "protected_plane")
        if cached is not None:
            return cached
        root = _phase_root(request.output_root, "protected_s0")
        if "protected_s0" in request.dependency_checkpoints:
            artifact = _dependency_artifact(request, "protected_s0", "protected_s0")
            replay = _dependency_artifact(
                request, "protected_s0", "protected_s0_replay"
            )
            plane = protected_s0.build_protected_s0_answer_plane(
                **self._protected_inputs(request)
            )
            _require(
                artifact.payload == plane.payload
                and replay.sha256 == artifact.sha256
                and replay.payload == artifact.payload,
                "protected S0 artifacts differ from reconstruction",
            )
        else:
            artifact, _created, plane = protected_s0.publish_protected_s0_answer_plane(
                root / PROTECTED_S0_NAME,
                **self._protected_inputs(request),
            )
            protected_s0.replay_protected_s0_answer_plane(
                source_plane_path=artifact.path,
                expected_source_plane_sha256=artifact.sha256,
                replay_output_path=root / PROTECTED_S0_REPLAY_NAME,
                **self._protected_inputs(request),
            )
        return self._remember(request, "protected_plane", (artifact, plane))

    def _query_context(
        self, request: PhaseRequest
    ) -> query_expansion.ConfirmationQueryExpansionContext:
        cached = self._cached(request, "query_context")
        if cached is not None:
            return cached
        artifact, _plane = self._protected_plane(request)
        workset = self._workset(request)
        execution = self._cumulative_execution(request)
        checkpoints = {
            row.namespace_store_id: path
            for row, path in zip(
                workset.namespaces, execution.checkpoint_paths, strict=True
            )
        }
        result = query_expansion.load_confirmation_query_expansion_context(
            protected_s0_plane_path=artifact.path,
            expected_protected_s0_plane_sha256=artifact.sha256,
            protected_s0_inputs=self._protected_inputs(request),
            namespace_checkpoint_paths_by_store_id=checkpoints,
            include_s0_evidence=True,
        )
        return self._remember(request, "query_context", result)

    def _query_preflight(self, request: PhaseRequest) -> tuple[Any, Any]:
        context = self._query_context(request)
        root = _phase_root(request.output_root, "query_expansion")
        if "query_expansion" in request.dependency_checkpoints:
            preflight = _dependency_artifact(
                request, "query_expansion", "query_preflight"
            )
            root = preflight.path.parent
        else:
            preflight = query_expansion.preflight_confirmation_query_expansion(
                context, output_root=root
            )
        state = query_expansion._native_state(  # noqa: SLF001
            context,
            output_root=root,
            expected_preflight_sha256=preflight.sha256,
        )
        return preflight, state

    def _protected_parent(self, request: PhaseRequest) -> Any:
        cached = self._cached(request, "protected_parent")
        if cached is not None:
            return cached
        artifact, _plane = self._protected_plane(request)
        if "query_direct_answer" in request.dependency_checkpoints:
            bridge = _dependency_artifact(
                request, "query_direct_answer", "protected_parent_bridge"
            )
            root = bridge.path.parent
        else:
            root = _phase_root(request.output_root, "query_direct_answer") / "protected-parent"
        result = query_payload.materialize_verified_protected_s0_parent(
            protected_s0_plane_path=artifact.path,
            expected_protected_s0_plane_sha256=artifact.sha256,
            output_root=root,
            **self._protected_inputs(request),
        )
        return self._remember(request, "protected_parent", result)

    def _query_artifacts(self, request: PhaseRequest) -> Any:
        cached = self._cached(request, "query_artifacts")
        if cached is not None:
            return cached
        parent = self._protected_parent(request)
        context = self._query_context(request)
        preflight = _dependency_artifact(request, "query_expansion", "query_preflight")
        run = _dependency_artifact(request, "query_expansion", "query_run")
        run_replay = _dependency_artifact(
            request, "query_expansion", "query_run_replay"
        )
        ledger = _dependency_artifact(
            request, "query_expansion", "query_runtime_ledger"
        )
        ledger_replay = _dependency_artifact(
            request, "query_expansion", "query_runtime_ledger_replay"
        )
        result = query_payload.load_verified_query_expansion_artifacts(
            parent,
            query_preflight_path=preflight.path,
            expected_query_preflight_sha256=preflight.sha256,
            query_run_path=run.path,
            query_run_replay_path=run_replay.path,
            expected_query_run_sha256=run.sha256,
            query_runtime_ledger_path=ledger.path,
            query_runtime_ledger_replay_path=ledger_replay.path,
            expected_query_runtime_ledger_sha256=ledger.sha256,
        )
        _require(
            result.population_id == context.population.population_id
            if hasattr(result, "population_id")
            else True,
            "query artifact population changed",
        )
        return self._remember(request, "query_artifacts", result)

    def _query_payload_plan(self, request: PhaseRequest) -> Any:
        cached = self._cached(request, "query_payload_plan")
        if cached is not None:
            return cached
        parent = self._protected_parent(request)
        context = self._query_context(request)
        preflight = _dependency_artifact(request, "query_expansion", "query_preflight")
        run = _dependency_artifact(request, "query_expansion", "query_run")
        run_replay = _dependency_artifact(
            request, "query_expansion", "query_run_replay"
        )
        ledger = _dependency_artifact(
            request, "query_expansion", "query_runtime_ledger"
        )
        ledger_replay = _dependency_artifact(
            request, "query_expansion", "query_runtime_ledger_replay"
        )
        result = query_payload.build_confirmation_query_payload_plan(
            parent,
            query_preflight_path=preflight.path,
            expected_query_preflight_sha256=preflight.sha256,
            query_run_path=run.path,
            query_run_replay_path=run_replay.path,
            expected_query_run_sha256=run.sha256,
            query_runtime_ledger_path=ledger.path,
            query_runtime_ledger_replay_path=ledger_replay.path,
            expected_query_runtime_ledger_sha256=ledger.sha256,
            expected_query_population_id=context.population.population_id,
            expected_query_prompt_population_sha256=(
                context.population.prompt_population.prompt_population_sha256
            ),
        )
        return self._remember(request, "query_payload_plan", result)

    def _direct_preflight(self, request: PhaseRequest) -> Any:
        if "query_direct_answer" in request.dependency_checkpoints:
            prompt = _dependency_artifact(
                request, "query_direct_answer", "query_direct_prompt"
            )
            root = prompt.path.parent
        else:
            root = _phase_root(request.output_root, "query_direct_answer")
        return query_payload.publish_confirmation_query_payload_preflight(
            self._query_payload_plan(request),
            output_root=root,
        )

    def _direct_plane(self, request: PhaseRequest) -> Any:
        cached = self._cached(request, "direct_plane")
        if cached is not None:
            return cached
        preflight = self._direct_preflight(request)
        release = _dependency_artifact(
            request, "query_direct_answer", "query_direct_release"
        )
        run = _dependency_artifact(request, "query_direct_answer", "query_direct_run")
        root = run.path.parent
        result = query_payload.replay_confirmation_query_payload_answers(
            preflight,
            output_root=root,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_answer_preflight_sha256=preflight.answer_preflight_artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_run_sha256=run.sha256,
        )
        return self._remember(request, "direct_plane", result)

    def _map_preflight(self, request: PhaseRequest) -> Any:
        plan = evidence_map.build_confirmation_evidence_map_plan(
            self._query_payload_plan(request), self._direct_plane(request)
        )
        if "evidence_map" in request.dependency_checkpoints:
            prompt = _dependency_artifact(request, "evidence_map", "evidence_map_prompt")
            root = prompt.path.parent
        else:
            root = _phase_root(request.output_root, "evidence_map")
        preflight = evidence_map.publish_confirmation_evidence_map_preflight(
            plan, output_root=root
        )
        return plan, preflight

    def _map_plane(self, request: PhaseRequest) -> tuple[Any, Any]:
        cached = self._cached(request, "map_plane")
        if cached is not None:
            return cached
        plan, preflight = self._map_preflight(request)
        release = _dependency_artifact(request, "evidence_map", "evidence_map_release")
        run = _dependency_artifact(request, "evidence_map", "evidence_map_run")
        root = run.path.parent
        plane = evidence_map.replay_confirmation_evidence_map(
            preflight,
            output_root=root,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_run_sha256=run.sha256,
        )
        return self._remember(request, "map_plane", (plan, plane))

    def _source_streams(self, request: PhaseRequest) -> Any:
        cached = self._cached(request, "source_streams")
        if cached is not None:
            return cached
        plan, plane = self._map_plane(request)
        expected = _dependency_artifact(request, "source_streams", "source_streams")
        root = expected.path.parent
        result = source_streams.replay_confirmation_source_streams(
            self._query_context(request),
            self._query_artifacts(request),
            plan.map_plan,
            plane,
            output_root=root,
            expected_plane_sha256=expected.sha256,
        )
        return self._remember(request, "source_streams", result)

    def _adaptive_map_preflight(self, request: PhaseRequest) -> Any:
        if "adaptive_source_map" in request.dependency_checkpoints:
            artifact = _dependency_artifact(
                request, "adaptive_source_map", "adaptive_source_map_preflight"
            )
            root = artifact.path.parent
        else:
            root = _phase_root(request.output_root, "adaptive_source_map")
        return adaptive_map.publish_confirmation_adaptive_source_map_from_streams(
            self._source_streams(request),
            output_root=root,
        )

    def _adaptive_source_map(self, request: PhaseRequest) -> Any:
        cached = self._cached(request, "adaptive_source_map")
        if cached is not None:
            return cached
        parents = self._source_streams(request)
        preflight = self._adaptive_map_preflight(request)
        release = _dependency_artifact(
            request, "adaptive_source_map", "adaptive_source_map_release"
        )
        materialized = _dependency_artifact(
            request,
            "adaptive_source_map",
            "adaptive_source_map_materialization",
        )
        root = materialized.path.parent
        result = adaptive_map.replay_confirmation_adaptive_source_map(
            parents.base_population,
            parents.query_map_adapter,
            output_root=root,
            expected_preflight_sha256=preflight.preflight_artifact.sha256,
            expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_materialization_sha256=materialized.sha256,
        )
        return self._remember(request, "adaptive_source_map", result)

    def _adaptive_upstream(self, request: PhaseRequest) -> Any:
        cached = self._cached(request, "adaptive_upstream")
        if cached is not None:
            return cached
        plan, plane = self._map_plane(request)
        result = adaptive_tail.confirmation_adaptive_upstream(
            self._source_streams(request),
            self._adaptive_source_map(request),
            plan.map_plan,
            plane,
        )
        return self._remember(request, "adaptive_upstream", result)

    def _adaptive_evidence_preflight(self, request: PhaseRequest) -> Any:
        plan = adaptive_tail.build_confirmation_adaptive_evidence_plan(
            self._adaptive_upstream(request)
        )
        if "adaptive_evidence_solver" in request.dependency_checkpoints:
            artifact = _dependency_artifact(
                request, "adaptive_evidence_solver", "adaptive_evidence_preflight"
            )
            root = artifact.path.parent
        else:
            root = _phase_root(request.output_root, "adaptive_evidence_solver")
        return adaptive_tail.publish_confirmation_adaptive_evidence_preflight(
            plan, output_root=root
        )

    def _adaptive_evidence(self, request: PhaseRequest) -> Any:
        cached = self._cached(request, "adaptive_evidence")
        if cached is not None:
            return cached
        preflight = self._adaptive_evidence_preflight(request)
        release = _dependency_artifact(
            request, "adaptive_evidence_solver", "adaptive_evidence_release"
        )
        run = _dependency_artifact(
            request, "adaptive_evidence_solver", "adaptive_evidence_run"
        )
        replay = _dependency_artifact(
            request, "adaptive_evidence_solver", "adaptive_evidence_replay"
        )
        root = run.path.parent
        result = adaptive_tail.replay_confirmation_adaptive_evidence(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_run_sha256=run.sha256,
            expected_replay_sha256=replay.sha256,
        )
        return self._remember(request, "adaptive_evidence", result)

    def _adaptive_tail_preflight(self, request: PhaseRequest) -> Any:
        plan = adaptive_tail.build_confirmation_adaptive_tail_plan(
            self._adaptive_upstream(request)
        )
        if "adaptive_tail" in request.dependency_checkpoints:
            artifact = _dependency_artifact(
                request, "adaptive_tail", "adaptive_tail_preflight"
            )
            root = artifact.path.parent
        else:
            root = _phase_root(request.output_root, "adaptive_tail")
        return adaptive_tail.publish_confirmation_adaptive_tail_preflight(
            plan, output_root=root
        )

    def _adaptive_tail(self, request: PhaseRequest) -> Any:
        cached = self._cached(request, "adaptive_tail")
        if cached is not None:
            return cached
        preflight = self._adaptive_tail_preflight(request)
        release = _dependency_artifact(request, "adaptive_tail", "adaptive_tail_release")
        run = _dependency_artifact(request, "adaptive_tail", "adaptive_tail_run")
        replay = _dependency_artifact(request, "adaptive_tail", "adaptive_tail_replay")
        root = run.path.parent
        result = adaptive_tail.replay_confirmation_adaptive_tail(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_run_sha256=run.sha256,
            expected_replay_sha256=replay.sha256,
        )
        return self._remember(request, "adaptive_tail", result)

    def prepare(self, phase_id: str, request: PhaseRequest) -> ProviderRequirement:
        if PROVIDER_CLASS_BY_PHASE[phase_id] is None:
            return _zero_requirement()
        if phase_id == "s0_terra_answer":
            _prompt, verified, preflight = self._s0_prompt_and_preflight(request)
            records = terra_lifecycle._authenticated_records(  # noqa: SLF001
                verified, preflight, output_root=_phase_root(request.output_root, phase_id)
            )
            return _terra_requirement(
                verified.prompt_population.unique_prompt_count, len(records)
            )
        if phase_id == "query_expansion":
            _preflight, state = self._query_preflight(request)
            return _terra_requirement(
                self._query_context(request).population.prompt_population.unique_prompt_count,
                len(state.records),
            )
        if phase_id == "query_direct_answer":
            preflight = self._direct_preflight(request)
            records = query_payload._checkpoint_records(  # noqa: SLF001
                preflight,
                output_root=_phase_root(request.output_root, phase_id),
                answer_preflight_sha256=preflight.answer_preflight_artifact.sha256,
            )
            return _terra_requirement(preflight.plan.required_calls, len(records))
        if phase_id == "evidence_map":
            plan, preflight = self._map_preflight(request)
            records = evidence_map._checkpoint_records(  # noqa: SLF001
                preflight,
                output_root=_phase_root(request.output_root, phase_id),
                map_preflight_sha256=preflight.map_preflight_artifact.sha256,
            )
            return _terra_requirement(plan.required_calls, len(records))
        if phase_id == "adaptive_source_map":
            preflight = self._adaptive_map_preflight(request)
            records = adaptive_map._checkpoint_records(  # noqa: SLF001
                preflight,
                preflight.preflight_artifact,
                output_root=_phase_root(request.output_root, phase_id),
            )
            return _terra_requirement(preflight.required_provider_calls, len(records))
        if phase_id == "adaptive_evidence_solver":
            preflight = self._adaptive_evidence_preflight(request)
            records = adaptive_tail._checkpoint_records(  # noqa: SLF001
                preflight,
                output_root=_phase_root(request.output_root, phase_id),
                preflight_artifact=preflight.artifact,
            )
            return _terra_requirement(preflight.plan.required_calls, len(records))
        if phase_id == "adaptive_tail":
            preflight = self._adaptive_tail_preflight(request)
            records = adaptive_tail._checkpoint_records(  # noqa: SLF001
                preflight,
                output_root=_phase_root(request.output_root, phase_id),
                preflight_artifact=preflight.artifact,
            )
            return _terra_requirement(preflight.plan.required_calls, len(records))
        raise ConfirmationProductionPhaseAdapterError("unsupported provider phase")

    @staticmethod
    def _validate_execution_authority(
        requirement: ProviderRequirement,
        *,
        enable_provider: bool,
        authorized_provider_calls: int,
    ) -> None:
        _require(
            authorized_provider_calls == requirement.remaining_calls,
            "adapter authority differs from exact remaining calls",
        )
        _require(
            enable_provider is bool(requirement.remaining_calls),
            "adapter provider opt-in differs from remaining calls",
        )

    @staticmethod
    def _outcome(
        request: PhaseRequest,
        requirement: ProviderRequirement,
        artifacts: Sequence[PhaseArtifact],
        *,
        physical_calls: int,
        metadata: Mapping[str, Any],
    ) -> PhaseOutcome:
        return PhaseOutcome(
            artifacts=tuple(artifacts),
            provider_requirement=requirement,
            authorized_provider_calls=requirement.remaining_calls,
            physical_provider_calls=physical_calls,
            logical_question_count=request.context.question_count,
            metadata=MappingProxyType(dict(metadata)),
        )

    def execute(
        self,
        phase_id: str,
        request: PhaseRequest,
        *,
        requirement: ProviderRequirement,
        enable_provider: bool,
        authorized_provider_calls: int,
    ) -> PhaseOutcome:
        self._validate_execution_authority(
            requirement,
            enable_provider=enable_provider,
            authorized_provider_calls=authorized_provider_calls,
        )
        handler = getattr(self, f"_execute_{phase_id}", None)
        _require(callable(handler), f"production phase handler is missing: {phase_id}")
        outcome = handler(request, requirement)
        _require(
            outcome.physical_provider_calls == requirement.remaining_calls,
            "production phase physical calls differ from authorization",
        )
        self._replayed.add(self._key(request, phase_id))
        return outcome

    def replay(
        self,
        phase_id: str,
        request: PhaseRequest,
        checkpoint: contracts.SealedJson,
    ) -> None:
        _require(
            checkpoint.payload.get("phase_id") == phase_id,
            "adapter replay received another checkpoint",
        )
        # Routine executor replay is intentionally O(number of declared
        # artifacts): the executor already authenticates bytes, and reopening
        # 1M stores or reloading BGE here would duplicate the just-completed
        # work on every advance/resume.  ``restore_first_half`` is the explicit
        # typed reconstruction path and invokes the inexpensive native
        # artifact loaders/replayers needed by downstream phases.
        rows = checkpoint.payload.get("artifacts")
        _require(isinstance(rows, list) and bool(rows), "phase artifact registry is empty")
        seen: set[str] = set()
        for raw in rows:
            _require(isinstance(raw, Mapping), "phase artifact registry changed")
            role = str(raw.get("role"))
            _require(role and role not in seen, "phase artifact role repeats")
            seen.add(role)
            relative = Path(str(raw.get("path")))
            _require(not relative.is_absolute(), "phase artifact path is absolute")
            path = (request.output_root / relative).resolve()
            _require(
                path.is_relative_to(request.output_root.resolve()),
                "phase artifact escapes output root",
            )
            sealed = _sealed(path, f"{phase_id} replay artifact")
            _require(
                sealed.sha256 == _sha(raw.get("sha256"), "phase replay artifact"),
                "phase replay artifact changed",
            )
            if role in {"namespace_execution", "staged_execution"}:
                registry_key = (
                    "checkpoints"
                    if role == "namespace_execution"
                    else "cumulative_checkpoints"
                )
                registry = sealed.payload.get(registry_key)
                _require(
                    isinstance(registry, list),
                    f"{role} subordinate registry is absent",
                )
                for child in registry:
                    _require(
                        isinstance(child, Mapping),
                        f"{role} subordinate registry changed",
                    )
                    child_relative = Path(str(child.get("path")))
                    _require(
                        not child_relative.is_absolute(),
                        f"{role} subordinate path is absolute",
                    )
                    child_path = (request.output_root / child_relative).resolve()
                    _require(
                        child_path.is_relative_to(request.output_root.resolve()),
                        f"{role} subordinate escapes output root",
                    )
                    child_sealed = _sealed(child_path, f"{role} subordinate")
                    _require(
                        child_sealed.sha256
                        == _sha(child.get("sha256"), f"{role} subordinate"),
                        f"{role} subordinate artifact changed",
                    )
            if role == "query_retriever_audit":
                _require(
                    sealed.payload.get("bge_released") is True
                    and sealed.payload.get(
                        "maximum_simultaneous_namespace_indexes"
                    )
                    in {0, 1}
                    and sealed.payload.get("physical_provider_calls") == 0,
                    "query retriever audit changed",
                )
        self._replayed.add(self._key(request, phase_id))

    def _execute_namespace_ingest(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        root = _phase_root(request.output_root, "namespace_ingest")
        runtime = self.environment.initial_runtime()
        workset = self._workset(request)
        workset_artifact, _created = contracts.publish_sealed_json(
            root / NAMESPACE_WORKSET_NAME, workset.projection()
        )
        execution = namespace_store.execute_confirmation_namespaces(
            request.context.treatment,
            preflight=request.context.preflight,
            workset=workset,
            output_root=root / "base",
            backend=runtime.base_backend,
        )
        self._remember(request, "base_execution", execution)
        checkpoint_rows = [
            {
                "path": str(path.resolve().relative_to(request.output_root.resolve())).replace(
                    "\\", "/"
                ),
                "sha256": digest,
            }
            for path, digest in zip(
                execution.checkpoint_paths, execution.checkpoint_sha256s, strict=True
            )
        ]
        summary = _publish_summary(
            root / NAMESPACE_EXECUTION_NAME,
            format_suffix="namespace-execution-v1",
            body={
                "backend_identity_sha256": runtime.base_backend.identity_sha256,
                "checkpoints": checkpoint_rows,
                "checkpoint_sha256s": list(execution.checkpoint_sha256s),
                "namespace_count": len(execution.checkpoint_paths),
                "physical_provider_calls": execution.physical_provider_calls,
                "workset_identity_sha256": workset.workset_identity_sha256,
            },
        )
        return self._outcome(
            request,
            requirement,
            (_artifact("namespace_workset", workset_artifact), _artifact("namespace_execution", summary)),
            physical_calls=0,
            metadata={
                "namespace_count": len(execution.checkpoint_paths),
                "target_tokens_per_namespace": workset.target_tokens,
            },
        )

    def _execute_staged_cumulative_s0_s3(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        root = _phase_root(request.output_root, "staged_cumulative_s0_s3") / "staged"
        runtime = self.environment.initial_runtime()
        inputs = self._cumulative_inputs(request)
        preparations: list[semantic_planes.ConfirmationSemanticFacetPreparation] = []

        def freeze_semantic_facets(
            preparation: staged.StagedPreparationExecution,
            backend: staged.StagedPreparationBackend,
        ) -> None:
            preparations.append(
                semantic_planes.prepare_confirmation_semantic_facet_vectors(
                    inputs,
                    preparation,
                    backend=backend,
                    output_root=root,
                    token_counter=self.environment.token_counter,
                )
            )

        try:
            execution = staged.execute_staged_confirmation_cumulative(
                inputs,
                output_root=root,
                preparation_backend=runtime.preparation_backend,
                qwen_factory=runtime.qwen_factory,
                retrieval_factory=runtime.retrieval_factory,
                token_counter=self.environment.token_counter,
                before_bge_release=freeze_semantic_facets,
            )
        finally:
            # The staged coordinator owns both release boundaries even when it
            # fails.  This object must never resurrect that initial BGE.
            self.environment.mark_initial_runtime_consumed()
        _require(len(preparations) == 1, "semantic facet hook did not complete once")
        facet_preparation = preparations[0]
        facet_release = semantic_planes.publish_confirmation_semantic_facet_release(
            facet_preparation,
            execution.barrier.payload["release_receipt"],
            output_root=root,
        )
        merge, _created = cumulative.publish_confirmation_cumulative_merge(
            inputs,
            cumulative_execution=execution.cumulative,
            output_path=root / CUMULATIVE_MERGE_NAME,
            token_counter=self.environment.token_counter,
        )
        self._remember(request, "cumulative_execution", execution.cumulative)
        restored = RestoredStagedExecution(
            execution.cumulative,
            execution.barrier,
            merge,
            facet_preparation.artifact,
            facet_release,
        )
        self._remember(request, "restored_staged", restored)
        cumulative_checkpoint_rows = [
            {
                "path": str(path.resolve().relative_to(request.output_root.resolve())).replace(
                    "\\", "/"
                ),
                "sha256": digest,
            }
            for path, digest in zip(
                execution.cumulative.checkpoint_paths,
                execution.cumulative.checkpoint_sha256s,
                strict=True,
            )
        ]
        summary = _publish_summary(
            _phase_root(request.output_root, "staged_cumulative_s0_s3") / STAGED_EXECUTION_NAME,
            format_suffix="staged-execution-v1",
            body={
                "barrier_sha256": execution.barrier.sha256,
                "cumulative_checkpoints": cumulative_checkpoint_rows,
                "cumulative_checkpoint_sha256s": list(execution.cumulative.checkpoint_sha256s),
                "cumulative_merge_sha256": merge.sha256,
                "namespace_count": len(execution.cumulative.checkpoint_paths),
                "physical_provider_calls": execution.physical_provider_calls,
                "semantic_facet_preparation_sha256": facet_preparation.artifact.sha256,
                "semantic_facet_release_sha256": facet_release.sha256,
            },
        )
        return self._outcome(
            request,
            requirement,
            (
                _artifact("staged_execution", summary),
                _artifact("cumulative_merge", merge),
                _artifact("staged_barrier", execution.barrier),
                _artifact("semantic_facet_preparation", facet_preparation.artifact),
                _artifact("semantic_facet_release", facet_release),
            ),
            physical_calls=0,
            metadata={"namespace_count": len(execution.cumulative.checkpoint_paths)},
        )

    def _existing_release(self, path: Path, label: str) -> Any | None:
        if not path.exists() and not path.with_name(path.name + ".sha256").exists():
            return None
        return _sealed(path, label)

    def _execute_s0_terra_answer(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        root = _phase_root(request.output_root, "s0_terra_answer")
        prompt, _verified, preflight = self._s0_prompt_and_preflight(request)
        release = self._existing_release(root / terra_lifecycle.RELEASE_NAME, "S0 release")
        if release is None:
            release, _created = terra_lifecycle.approve_provider_release(
                prompt_artifact_path=prompt.path,
                expected_prompt_artifact_sha256=prompt.sha256,
                output_root=root,
                expected_lifecycle_preflight_sha256=preflight.sha256,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        physical = 0
        if requirement.remaining_calls:
            result = terra_lifecycle.run_provider_completion(
                prompt_artifact_path=prompt.path,
                expected_prompt_artifact_sha256=prompt.sha256,
                output_root=root,
                expected_lifecycle_preflight_sha256=preflight.sha256,
                expected_release_sha256=release.sha256,
                enable_provider=True,
                authorized_provider_calls=requirement.remaining_calls,
                api_key_env=self.environment.api_key_env,
                client_factory=self.environment.terra_client_factory,
            )
            physical = int(result["physical_provider_calls"])
        completion, _created = terra_lifecycle.materialize_completions(
            prompt_artifact_path=prompt.path,
            expected_prompt_artifact_sha256=prompt.sha256,
            output_root=root,
            expected_lifecycle_preflight_sha256=preflight.sha256,
            expected_release_sha256=release.sha256,
        )
        replay, _created = terra_lifecycle.replay_completions(
            prompt_artifact_path=prompt.path,
            expected_prompt_artifact_sha256=prompt.sha256,
            output_root=root,
            expected_lifecycle_preflight_sha256=preflight.sha256,
            expected_release_sha256=release.sha256,
            expected_completion_sha256=completion.sha256,
        )
        return self._outcome(
            request,
            requirement,
            tuple(
                _artifact(role, value)
                for role, value in (
                    ("s0_prompt", prompt),
                    ("s0_lifecycle_preflight", preflight),
                    ("s0_provider_release", release),
                    ("s0_completion", completion),
                    ("s0_completion_replay", replay),
                )
            ),
            physical_calls=physical,
            metadata={"completion_count": request.context.question_count},
        )

    def _execute_protected_s0(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        artifact, plane = self._protected_plane(request)
        replay = _sealed(
            _phase_root(request.output_root, "protected_s0") / PROTECTED_S0_REPLAY_NAME,
            "protected S0 replay",
        )
        return self._outcome(
            request,
            requirement,
            (_artifact("protected_s0", artifact), _artifact("protected_s0_replay", replay)),
            physical_calls=0,
            metadata={"prediction_count": len(plane.predictions)},
        )

    def _execute_query_expansion(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        root = _phase_root(request.output_root, "query_expansion")
        context = self._query_context(request)
        preflight, _state = self._query_preflight(request)
        release = self._existing_release(root / query_expansion.RELEASE_NAME, "query release")
        if release is None:
            release, _created = query_expansion.approve_confirmation_query_expansion_provider_release(
                context,
                output_root=root,
                expected_query_preflight_sha256=preflight.sha256,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        physical = 0
        if requirement.remaining_calls:
            client = self.environment.terra_client_factory(
                context.runtime["gateway_url"], self.environment.api_key_env
            )
            try:
                provider = query_expansion.run_confirmation_query_expansion_provider(
                    context,
                    output_root=root,
                    expected_query_preflight_sha256=preflight.sha256,
                    expected_release_sha256=release.sha256,
                    enable_provider=True,
                    authorized_provider_calls=requirement.remaining_calls,
                    client=client,
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            physical = provider.physical_provider_calls
        session = self.environment.open_query_session(context)
        with session:
            run = query_expansion.materialize_confirmation_query_expansion(
                context,
                output_root=root,
                expected_query_preflight_sha256=preflight.sha256,
                expected_release_sha256=release.sha256,
                retrievers_by_namespace=session.retrievers,
            )
            replay = query_expansion.replay_confirmation_query_expansion(
                context,
                output_root=root,
                expected_query_preflight_sha256=preflight.sha256,
                expected_release_sha256=release.sha256,
                retrievers_by_namespace=session.retrievers,
                expected_run_sha256=run.run_artifact.sha256,
                expected_runtime_ledger_sha256=run.runtime_ledger_artifact.sha256,
            )
        audit_payload = dict(session.audit_projection())
        _require(
            audit_payload.get("bge_released") is True
            and audit_payload.get("maximum_simultaneous_namespace_indexes") in {0, 1}
            and audit_payload.get("physical_provider_calls") == 0,
            "query retriever ownership audit differs",
        )
        audit, _created = contracts.publish_sealed_json(
            root / QUERY_RETRIEVER_AUDIT_NAME, audit_payload
        )
        self._remember(request, "query_run", replay)
        run_replay = _sealed(root / query_live.RUN_REPLAY_NAME, "query run replay")
        ledger_replay = _sealed(
            root / query_live.RUNTIME_LEDGER_REPLAY_NAME, "query ledger replay"
        )
        return self._outcome(
            request,
            requirement,
            tuple(
                _artifact(role, value)
                for role, value in (
                    ("query_preflight", preflight),
                    ("query_provider_release", release),
                    ("query_run", run.run_artifact),
                    ("query_run_replay", run_replay),
                    ("query_runtime_ledger", run.runtime_ledger_artifact),
                    ("query_runtime_ledger_replay", ledger_replay),
                    ("query_retriever_audit", audit),
                )
            ),
            physical_calls=physical,
            metadata={"query_population_id": context.population.population_id},
        )

    def _execute_query_direct_answer(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        root = _phase_root(request.output_root, "query_direct_answer")
        parent = self._protected_parent(request)
        preflight = self._direct_preflight(request)
        release = self._existing_release(root / query_payload.RELEASE_NAME, "query direct release")
        if release is None:
            release = query_payload.approve_confirmation_query_payload_release(
                preflight,
                output_root=root,
                expected_prompt_sha256=preflight.prompt_artifact.sha256,
                expected_answer_preflight_sha256=preflight.answer_preflight_artifact.sha256,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        physical = 0
        if requirement.remaining_calls:
            result = query_payload.run_confirmation_query_payload_provider(
                preflight,
                output_root=root,
                expected_prompt_sha256=preflight.prompt_artifact.sha256,
                expected_answer_preflight_sha256=preflight.answer_preflight_artifact.sha256,
                expected_release_sha256=release.sha256,
                enable_provider=True,
                authorized_provider_calls=requirement.remaining_calls,
                api_key_env=self.environment.api_key_env,
                client_factory=self.environment.terra_client_factory,
            )
            physical = result.physical_provider_calls
        materialized = query_payload.materialize_confirmation_query_payload_answers(
            preflight,
            output_root=root,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_answer_preflight_sha256=preflight.answer_preflight_artifact.sha256,
            expected_release_sha256=release.sha256,
        )
        plane = query_payload.replay_confirmation_query_payload_answers(
            preflight,
            output_root=root,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_answer_preflight_sha256=preflight.answer_preflight_artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_run_sha256=materialized.answer_artifact.sha256,
        )
        self._remember(request, "direct_plane", plane)
        replay = _sealed(root / query_payload_live.ANSWER_REPLAY_NAME, "query direct replay")
        return self._outcome(
            request,
            requirement,
            tuple(
                _artifact(role, value)
                for role, value in (
                    ("protected_parent_bridge", parent.bridge_artifact),
                    ("protected_parent_run", parent.run_artifact),
                    ("protected_parent_replay", parent.replay_artifact),
                    ("protected_parent_ledger", parent.runtime_ledger_artifact),
                    ("query_direct_prompt", preflight.prompt_artifact),
                    ("query_direct_preflight", preflight.answer_preflight_artifact),
                    ("query_direct_release", release),
                    ("query_direct_run", materialized.answer_artifact),
                    ("query_direct_replay", replay),
                    ("query_direct_ledger", materialized.runtime_ledger_artifact),
                )
            ),
            physical_calls=physical,
            metadata={"direct_answer_count": len(plane.rows)},
        )

    def _execute_evidence_map(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        root = _phase_root(request.output_root, "evidence_map")
        plan, preflight = self._map_preflight(request)
        release = self._existing_release(root / evidence_map.RELEASE_NAME, "evidence map release")
        if release is None:
            release = evidence_map.approve_confirmation_evidence_map_release(
                preflight,
                output_root=root,
                expected_prompt_sha256=preflight.prompt_artifact.sha256,
                expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        physical = 0
        if requirement.remaining_calls:
            result = evidence_map.run_confirmation_evidence_map_provider(
                preflight,
                output_root=root,
                expected_prompt_sha256=preflight.prompt_artifact.sha256,
                expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
                expected_release_sha256=release.sha256,
                enable_provider=True,
                authorized_provider_calls=requirement.remaining_calls,
                api_key_env=self.environment.api_key_env,
                client_factory=self.environment.terra_client_factory,
            )
            physical = result.physical_provider_calls
        materialized = evidence_map.materialize_confirmation_evidence_map(
            preflight,
            output_root=root,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
            expected_release_sha256=release.sha256,
        )
        plane = evidence_map.replay_confirmation_evidence_map(
            preflight,
            output_root=root,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_run_sha256=materialized.map_artifact.sha256,
        )
        self._remember(request, "map_plane", (plan, plane))
        replay = _sealed(root / map_live.MAP_REPLAY_NAME, "evidence map replay")
        return self._outcome(
            request,
            requirement,
            tuple(
                _artifact(role, value)
                for role, value in (
                    ("evidence_map_prompt", preflight.prompt_artifact),
                    ("evidence_map_preflight", preflight.map_preflight_artifact),
                    ("evidence_map_release", release),
                    ("evidence_map_run", materialized.map_artifact),
                    ("evidence_map_replay", replay),
                    ("evidence_map_ledger", materialized.runtime_ledger_artifact),
                )
            ),
            physical_calls=physical,
            metadata={"map_row_count": len(plane.rows)},
        )

    def _execute_source_streams(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        plan, plane = self._map_plane(request)
        root = _phase_root(request.output_root, "source_streams")
        result = source_streams.materialize_confirmation_source_streams(
            self._query_context(request),
            self._query_artifacts(request),
            plan.map_plan,
            plane,
            output_root=root,
        )
        replayed = source_streams.replay_confirmation_source_streams(
            self._query_context(request),
            self._query_artifacts(request),
            plan.map_plan,
            plane,
            output_root=root,
            expected_plane_sha256=result.plane_artifact.sha256,
        )
        self._remember(request, "source_streams", replayed)
        replay = _sealed(root / source_streams.PLANE_REPLAY_NAME, "source streams replay")
        return self._outcome(
            request,
            requirement,
            (
                _artifact("source_streams", result.plane_artifact),
                _artifact("source_streams_replay", replay),
                _artifact("source_streams_eligibility", result.eligibility_artifact),
                _artifact("source_streams_partition", result.partition_artifact),
            ),
            physical_calls=0,
            metadata={"source_question_count": request.context.question_count},
        )

    def _execute_adaptive_source_map(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        root = _phase_root(request.output_root, "adaptive_source_map")
        parents = self._source_streams(request)
        preflight = self._adaptive_map_preflight(request)
        release = self._existing_release(root / adaptive_map.RELEASE_NAME, "adaptive map release")
        if release is None:
            release = adaptive_map.approve_confirmation_adaptive_source_map_release(
                preflight,
                output_root=root,
                expected_preflight_sha256=preflight.preflight_artifact.sha256,
                expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        physical = 0
        if requirement.remaining_calls:
            batch = adaptive_map.run_confirmation_adaptive_source_map_provider(
                preflight,
                output_root=root,
                expected_preflight_sha256=preflight.preflight_artifact.sha256,
                expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
                expected_release_sha256=release.sha256,
                enable_provider=True,
                authorized_provider_calls=requirement.remaining_calls,
                api_key_env=self.environment.api_key_env,
                client_factory=self.environment.terra_client_factory,
            )
            physical = batch.usage.physical_calls
        materialized = adaptive_map.materialize_confirmation_adaptive_source_map(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.preflight_artifact.sha256,
            expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
            expected_release_sha256=release.sha256,
        )
        replayed = adaptive_map.replay_confirmation_adaptive_source_map(
            parents.base_population,
            parents.query_map_adapter,
            output_root=root,
            expected_preflight_sha256=preflight.preflight_artifact.sha256,
            expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_materialization_sha256=materialized.materialization_artifact.sha256,
        )
        self._remember(request, "adaptive_source_map", replayed)
        return self._outcome(
            request,
            requirement,
            tuple(
                _artifact(role, value)
                for role, value in (
                    ("adaptive_source_map_preflight", preflight.preflight_artifact),
                    ("adaptive_source_map_work", preflight.work_manifest_artifact),
                    ("adaptive_source_map_release", release),
                    ("adaptive_source_map_materialization", materialized.materialization_artifact),
                    ("adaptive_source_map_replay", replayed.replay_artifact),
                )
            ),
            physical_calls=physical,
            metadata={"mapped_question_count": len(replayed.questions)},
        )

    def _execute_adaptive_evidence_solver(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        root = _phase_root(request.output_root, "adaptive_evidence_solver")
        preflight = self._adaptive_evidence_preflight(request)
        release = self._existing_release(root / adaptive_tail.SOLVER_RELEASE_NAME, "adaptive solver release")
        if release is None:
            release = adaptive_tail.approve_confirmation_adaptive_release(
                preflight,
                output_root=root,
                expected_preflight_sha256=preflight.artifact.sha256,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        physical = 0
        if requirement.remaining_calls:
            result = adaptive_tail.run_confirmation_adaptive_provider(
                preflight,
                output_root=root,
                expected_preflight_sha256=preflight.artifact.sha256,
                expected_release_sha256=release.sha256,
                enable_provider=True,
                authorized_provider_calls=requirement.remaining_calls,
                api_key_env=self.environment.api_key_env,
                client_factory=self.environment.terra_client_factory,
            )
            physical = result.physical_provider_calls
        materialized = adaptive_tail.materialize_confirmation_adaptive_evidence(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=release.sha256,
        )
        replayed = adaptive_tail.replay_confirmation_adaptive_evidence(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_run_sha256=materialized.run_artifact.sha256,
            expected_replay_sha256=materialized.replay_artifact.sha256,
        )
        self._remember(request, "adaptive_evidence", replayed)
        return self._outcome(
            request,
            requirement,
            tuple(
                _artifact(role, value)
                for role, value in (
                    ("adaptive_evidence_preflight", preflight.artifact),
                    ("adaptive_evidence_release", release),
                    ("adaptive_evidence_run", materialized.run_artifact),
                    ("adaptive_evidence_replay", materialized.replay_artifact),
                )
            ),
            physical_calls=physical,
            metadata={"solver_question_count": len(replayed.run.rows)},
        )

    def _execute_adaptive_tail(
        self, request: PhaseRequest, requirement: ProviderRequirement
    ) -> PhaseOutcome:
        root = _phase_root(request.output_root, "adaptive_tail")
        preflight = self._adaptive_tail_preflight(request)
        release = self._existing_release(root / adaptive_tail.TAIL_RELEASE_NAME, "adaptive tail release")
        if release is None:
            release = adaptive_tail.approve_confirmation_adaptive_release(
                preflight,
                output_root=root,
                expected_preflight_sha256=preflight.artifact.sha256,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        physical = 0
        if requirement.remaining_calls:
            result = adaptive_tail.run_confirmation_adaptive_provider(
                preflight,
                output_root=root,
                expected_preflight_sha256=preflight.artifact.sha256,
                expected_release_sha256=release.sha256,
                enable_provider=True,
                authorized_provider_calls=requirement.remaining_calls,
                api_key_env=self.environment.api_key_env,
                client_factory=self.environment.terra_client_factory,
            )
            physical = result.physical_provider_calls
        materialized = adaptive_tail.materialize_confirmation_adaptive_tail(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=release.sha256,
        )
        replayed = adaptive_tail.replay_confirmation_adaptive_tail(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_run_sha256=materialized.run_artifact.sha256,
            expected_replay_sha256=materialized.replay_artifact.sha256,
        )
        self._remember(request, "adaptive_tail", replayed)
        return self._outcome(
            request,
            requirement,
            tuple(
                _artifact(role, value)
                for role, value in (
                    ("adaptive_tail_preflight", preflight.artifact),
                    ("adaptive_tail_work", preflight.work_manifest_artifact),
                    ("adaptive_tail_release", release),
                    ("adaptive_tail_run", materialized.run_artifact),
                    ("adaptive_tail_replay", materialized.replay_artifact),
                )
            ),
            physical_calls=physical,
            metadata={"tail_question_count": len(replayed.decisions)},
        )

    def _restore_phase(self, phase_id: str, request: PhaseRequest) -> Any:
        if phase_id == "namespace_ingest":
            return self._base_execution(request)
        if phase_id == "staged_cumulative_s0_s3":
            return self._restored_staged(request)
        if phase_id == "s0_terra_answer":
            prompt, _verified, preflight = self._s0_prompt_and_preflight(request)
            root = _phase_root(request.output_root, phase_id)
            release = _sealed(root / terra_lifecycle.RELEASE_NAME, "S0 release")
            completion = _sealed(root / terra_lifecycle.COMPLETION_NAME, "S0 completion")
            return terra_lifecycle.replay_completions(
                prompt_artifact_path=prompt.path,
                expected_prompt_artifact_sha256=prompt.sha256,
                output_root=root,
                expected_lifecycle_preflight_sha256=preflight.sha256,
                expected_release_sha256=release.sha256,
                expected_completion_sha256=completion.sha256,
            )
        if phase_id == "protected_s0":
            return self._protected_plane(request)
        if phase_id == "query_expansion":
            context = self._query_context(request)
            root = _phase_root(request.output_root, phase_id)
            preflight = _sealed(root / query_live.PREFLIGHT_NAME, "query preflight")
            release = _sealed(root / query_expansion.RELEASE_NAME, "query release")
            run = _sealed(root / query_live.RUN_NAME, "query run")
            ledger = _sealed(root / query_live.RUNTIME_LEDGER_NAME, "query ledger")
            session = self.environment.open_query_session(context)
            with session:
                return query_expansion.replay_confirmation_query_expansion(
                    context,
                    output_root=root,
                    expected_query_preflight_sha256=preflight.sha256,
                    expected_release_sha256=release.sha256,
                    retrievers_by_namespace=session.retrievers,
                    expected_run_sha256=run.sha256,
                    expected_runtime_ledger_sha256=ledger.sha256,
                )
        if phase_id == "query_direct_answer":
            return self._direct_plane(request)
        if phase_id == "evidence_map":
            return self._map_plane(request)
        if phase_id == "source_streams":
            return self._source_streams(request)
        if phase_id == "adaptive_source_map":
            return self._adaptive_source_map(request)
        if phase_id == "adaptive_evidence_solver":
            return self._adaptive_evidence(request)
        if phase_id == "adaptive_tail":
            return self._adaptive_tail(request)
        raise ConfirmationProductionPhaseAdapterError("unknown replay phase")

    def restore_first_half(self, request: PhaseRequest) -> FirstHalfState:
        query_context = self._query_context(request)
        query_artifacts = self._query_artifacts(request)
        query_plan = self._query_payload_plan(request)
        direct = self._direct_plane(request)
        map_plan, map_plane = self._map_plane(request)
        streams = self._source_streams(request)
        source_map = self._adaptive_source_map(request)
        upstream = self._adaptive_upstream(request)
        solver = self._adaptive_evidence(request)
        tail = self._adaptive_tail(request)
        staged_state = self._restored_staged(request)
        return FirstHalfState(
            query_context=query_context,
            query_artifacts=query_artifacts,
            query_payload_plan=query_plan,
            direct_plane=direct,
            evidence_map_plan=map_plan,
            evidence_map_plane=map_plane,
            source_streams=streams,
            source_map=source_map,
            adaptive_upstream=upstream,
            adaptive_evidence=solver,
            adaptive_tail=tail,
            cumulative_inputs=self._cumulative_inputs(request),
            staged_execution=staged_state,
            staged_output_root=staged_state.barrier.path.parent.parent,
            staged_barrier=staged_state.barrier,
            semantic_facet_preparation_artifact=(
                staged_state.semantic_facet_preparation_artifact
            ),
            semantic_facet_release_artifact=staged_state.semantic_facet_release_artifact,
            semantic_facet_release=staged_state.semantic_facet_release_artifact,
        )


def restore_confirmation_first_half(
    request: PhaseRequest,
    environment: ConfirmationProductionAdapterEnvironment,
    *,
    operations: ProductionFirstHalfOperations | None = None,
) -> FirstHalfState:
    """Reconstruct the complete first-half handoff through exact replay APIs."""

    owner = operations or ProductionFirstHalfOperations(environment)
    return owner.restore_first_half(request)


def build_confirmation_first_half_adapters(
    environment: ConfirmationProductionAdapterEnvironment | None = None,
    *,
    operations: FirstHalfOperations | None = None,
) -> tuple[PredictionPhaseAdapter, ...]:
    """Build the ordered first-half adapter population.

    ``operations`` is the provider/model-free injection seam used by the full
    executor rehearsal.  Production must omit it and supply ``environment``.
    """

    if operations is None:
        _require(environment is not None, "production adapter environment is required")
        assert environment is not None
        operations = ProductionFirstHalfOperations(environment)
    base_identity = _sha(operations.identity_sha256, "first-half operations")
    result: list[PredictionPhaseAdapter] = []
    for phase_id in FIRST_HALF_PHASE_IDS:
        provider_class = PROVIDER_CLASS_BY_PHASE[phase_id]
        result.append(
            ConfirmationProductionPhaseAdapter(
                phase_id=phase_id,
                provider_class=provider_class,
                operations=operations,
                identity_sha256=canonical_sha256(
                    {
                        "format": f"{FORMAT}-adapter-v1",
                        "operations_identity_sha256": base_identity,
                        "phase_id": phase_id,
                        "provider_class": provider_class,
                        "retry_limit": 0,
                    }
                ),
            )
        )
    return tuple(result)


__all__ = [
    "FIRST_HALF_PHASE_IDS",
    "FORMAT",
    "PRODUCTION_DIRECTORY_NAME",
    "PROVIDER_CLASS_BY_PHASE",
    "ConfirmationProductionAdapterEnvironment",
    "ConfirmationProductionPhaseAdapter",
    "ConfirmationProductionPhaseAdapterError",
    "FirstHalfOperations",
    "FirstHalfState",
    "ProductionFirstHalfOperations",
    "RestoredStagedExecution",
    "build_confirmation_first_half_adapters",
    "restore_confirmation_first_half",
]
