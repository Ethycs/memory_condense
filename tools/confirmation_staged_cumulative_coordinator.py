#!/usr/bin/env python3
"""Two-phase, provider-free confirmation cumulative retrieval coordinator.

Phase A visits one namespace at a time under BGE, builds or verifies its
combined store, freezes the exact held-out query batch, and closes the store.
Only after every namespace preparation is sealed does the coordinator close
BGE and publish a release barrier.  Phase B may then load Qwen and delegates
S0--S3 execution/checkpointing to ``confirmation_cumulative_retrieval``.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from memory_condense.domain.discourse import (
    identity_sha256 as runtime_identity_sha256,
    quote_sha256,
)
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.consolidation_replay import FrozenQueryEmbedder
from tools.confirmation_combined_store_receipt import CombinedCumulativeStoreReceipt
from tools import confirmation_cumulative_retrieval as cumulative
from tools.confirmation_namespace_store_adapter import SealedPayload, read_sealed_payload
from tools.confirmation_canonical import canonical_sha256


FROZEN_QUERY_FORMAT = "memory-condense-confirmation-frozen-query-batch-v1"
PREPARATION_FORMAT = "memory-condense-confirmation-staged-preparation-v1"
BGE_RELEASE_FORMAT = "memory-condense-confirmation-bge-release-v1"
BARRIER_FORMAT = "memory-condense-confirmation-staged-barrier-v1"
PRODUCTION_QWEN_FACTORY_FORMAT = (
    "memory-condense-confirmation-production-shared-qwen-factory-v1"
)
PRODUCTION_QWEN_RUNTIME_FORMAT = (
    "memory-condense-confirmation-production-shared-qwen-runtime-v1"
)

_VECTOR_KEYS = frozenset(
    {
        "artifact_receipt_sha256",
        "base_checkpoint_sha256",
        "dimension",
        "embedding_identity_sha256",
        "format",
        "namespace_id",
        "namespace_store_id",
        "query_batch_sha256",
        "rows",
        "vector_values_sha256",
        "work_receipt_sha256",
    }
)
_PREPARATION_KEYS = frozenset(
    {
        "backend_identity_sha256",
        "base_checkpoint_sha256",
        "checkpoint_receipt_sha256",
        "combined_store_mode",
        "combined_store_receipt",
        "compilation_receipt_sha256",
        "format",
        "freeze_sha256",
        "gold_loaded",
        "namespace_id",
        "namespace_store_id",
        "physical_provider_calls",
        "preflight_sha256",
        "query_vector_artifact_receipt_sha256",
        "query_vector_artifact_relative_path",
        "query_vector_artifact_sha256",
        "work_receipt_sha256",
        "workset_identity_sha256",
    }
)
_RELEASE_KEYS = frozenset(
    {
        "embedding_identity_sha256",
        "embedding_released_before_qwen_load",
        "format",
        "physical_provider_calls",
        "preparation_backend_identity_sha256",
        "release_policy",
        "release_receipt_sha256",
    }
)
_BARRIER_KEYS = frozenset(
    {
        "barrier_receipt_sha256",
        "format",
        "freeze_sha256",
        "gold_loaded",
        "physical_provider_calls",
        "preflight_sha256",
        "preparation_backend_identity_sha256",
        "preparations",
        "qwen_factory_identity_sha256",
        "release_receipt",
        "retrieval_factory_identity_sha256",
        "workset_identity_sha256",
    }
)


class StagedCoordinatorError(cumulative.ConfirmationCumulativeError):
    """A staged lifecycle invariant or artifact failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise StagedCoordinatorError(message)


def _sha256(value: object, label: str) -> str:
    try:
        return cumulative._sha256(value, label)  # noqa: SLF001
    except cumulative.ConfirmationCumulativeError as exc:
        raise StagedCoordinatorError(str(exc)) from exc


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value  # type: ignore[return-value]


def _exact(value: Mapping[str, Any], keys: frozenset[str], label: str) -> None:
    _require(set(value) == keys, f"{label} has a non-closed schema")


def _read_sealed(path: Path, *, label: str) -> SealedPayload:
    try:
        return read_sealed_payload(path, label=label)
    except ValueError as exc:
        raise StagedCoordinatorError(f"cannot verify {label}") from exc


@dataclass(frozen=True, slots=True)
class StagedPreparationResult:
    namespace_id: str
    namespace_store_id: str
    base_checkpoint_sha256: str
    combined_store_receipt: Mapping[str, Any]
    compilation_receipt_sha256: str
    combined_store_mode: str
    query_batch: tuple[str, ...]
    query_vectors: Mapping[str, Sequence[float]]
    physical_provider_calls: int = 0


@dataclass(frozen=True, slots=True)
class FrozenQueryDescriptor:
    namespace_store_id: str
    artifact_path: Path
    artifact_sha256: str
    artifact_receipt_sha256: str
    query_batch: tuple[str, ...]
    query_batch_sha256: str
    vector_values_sha256: str
    dimension: int


@dataclass(frozen=True, slots=True)
class StagedPreparationExecution:
    checkpoint_paths: tuple[Path, ...]
    checkpoint_sha256s: tuple[str, ...]
    descriptors: tuple[FrozenQueryDescriptor, ...]
    created_count: int
    reused_count: int
    physical_provider_calls: int = 0


@dataclass(frozen=True, slots=True)
class StagedCoordinatorExecution:
    preparation: StagedPreparationExecution
    barrier: SealedPayload
    cumulative: cumulative.ConfirmationCumulativeExecution
    physical_provider_calls: int = 0


class StagedPreparationBackend(Protocol):
    @property
    def identity_sha256(self) -> str: ...

    @property
    def policy_freeze_sha256(self) -> str: ...

    @property
    def embedding_identity(self) -> Mapping[str, Any]: ...

    def prepare(
        self, request: cumulative.CumulativeNamespaceRequest
    ) -> StagedPreparationResult: ...

    def verify(
        self,
        request: cumulative.CumulativeNamespaceRequest,
        expected: Mapping[str, Any],
        query_vectors: Mapping[str, Sequence[float]],
    ) -> None: ...

    def freeze_query_batch(
        self, queries: Sequence[str]
    ) -> Mapping[str, Sequence[float]]: ...

    def release_bge(self) -> Mapping[str, Any]: ...


class StagedBeforeBgeReleaseHook(Protocol):
    """One synchronous, provider-free hook at the resident-BGE safe point."""

    def __call__(
        self,
        preparation: StagedPreparationExecution,
        preparation_backend: StagedPreparationBackend,
        /,
    ) -> None: ...


class StagedQwenRuntime(Protocol):
    @property
    def identity_sha256(self) -> str: ...

    @property
    def coverage_selector(self) -> Any: ...

    @property
    def representative_linker(self) -> Any: ...

    @property
    def physical_provider_calls(self) -> int: ...

    def close(self) -> None: ...


class StagedQwenRuntimeFactory(Protocol):
    @property
    def identity_sha256(self) -> str: ...

    def load_after_barrier(self, barrier: SealedPayload) -> StagedQwenRuntime: ...


class StagedRetrievalBackendFactory(Protocol):
    @property
    def identity_sha256(self) -> str: ...

    def create(
        self,
        *,
        qwen_runtime: StagedQwenRuntime,
        barrier: SealedPayload,
        frozen_queries: tuple[FrozenQueryDescriptor, ...],
    ) -> cumulative.CumulativeNamespaceBackend: ...


def bge_release_receipt(
    *,
    preparation_backend_identity_sha256: str,
    embedding_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Construct the deterministic attestation used at the phase barrier."""

    body = {
        "embedding_identity_sha256": runtime_identity_sha256(
            cumulative._plain_json(embedding_identity)  # noqa: SLF001
        ),
        "embedding_released_before_qwen_load": True,
        "format": BGE_RELEASE_FORMAT,
        "physical_provider_calls": 0,
        "preparation_backend_identity_sha256": _sha256(
            preparation_backend_identity_sha256, "preparation backend identity"
        ),
        "release_policy": "close_bge_before_qwen_factory_call_v1",
    }
    return MappingProxyType(
        {**body, "release_receipt_sha256": canonical_sha256(body)}
    )


def _validated_release(
    value: object,
    *,
    backend: StagedPreparationBackend,
) -> dict[str, Any]:
    release = dict(_mapping(value, "BGE release receipt"))
    _exact(release, _RELEASE_KEYS, "BGE release receipt")
    declared = _sha256(
        release.pop("release_receipt_sha256"), "BGE release receipt"
    )
    _require(
        release.get("format") == BGE_RELEASE_FORMAT
        and release.get("embedding_released_before_qwen_load") is True
        and release.get("physical_provider_calls") == 0
        and release.get("preparation_backend_identity_sha256")
        == backend.identity_sha256
        and release.get("embedding_identity_sha256")
        == runtime_identity_sha256(
            cumulative._plain_json(backend.embedding_identity)  # noqa: SLF001
        )
        and canonical_sha256(release) == declared,
        "BGE release receipt is invalid",
    )
    return {**release, "release_receipt_sha256": declared}


def _normalize_vectors(
    result: StagedPreparationResult,
    request: cumulative.CumulativeNamespaceRequest,
    embedding_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _require(type(result) is StagedPreparationResult, "preparation result changed type")
    _require(
        result.namespace_id == request.work.namespace_id
        and result.namespace_store_id == request.work.namespace_store_id
        and result.base_checkpoint_sha256 == request.base.checkpoint.sha256,
        "preparation result escaped its namespace",
    )
    _require(result.physical_provider_calls == 0, "preparation used a provider")
    batch = tuple(result.query_batch)
    _require(batch and len(batch) == len(set(batch)), "query batch must be unique")
    required = set(cumulative.held_out_queries(request.queries))
    _require(required <= set(batch), "query batch omitted a declared held-out query")
    _require(
        tuple(result.query_vectors) == batch,
        "query vector order differs from the frozen batch",
    )
    rows: list[dict[str, Any]] = []
    dimensions: set[int] = set()
    for query in batch:
        _require(type(query) is str and bool(query.strip()), "frozen query is invalid")
        raw_vector = result.query_vectors[query]
        _require(
            isinstance(raw_vector, Sequence) and not isinstance(raw_vector, (str, bytes)),
            "frozen query vector is invalid",
        )
        vector = [float(item) for item in raw_vector]
        _require(vector and all(math.isfinite(item) for item in vector), "vector is nonfinite")
        dimensions.add(len(vector))
        rows.append(
            {
                "query": query,
                "query_sha256": quote_sha256(query),
                "vector": vector,
                "vector_sha256": canonical_sha256(vector),
            }
        )
    _require(len(dimensions) == 1, "frozen query dimensions changed")
    dimension = dimensions.pop()
    declared_dimension = embedding_identity.get("dimension")
    _require(
        declared_dimension is None or declared_dimension == dimension,
        "frozen query dimension differs from the embedding identity",
    )
    receipt_payload = cumulative._plain_json(  # noqa: SLF001
        result.combined_store_receipt
    )
    try:
        receipt = CombinedCumulativeStoreReceipt(**receipt_payload)
    except (TypeError, ValueError) as exc:
        raise StagedCoordinatorError("combined store receipt is invalid") from exc
    _require(
        asdict(receipt) == receipt_payload
        and receipt.compilation_receipt_sha256
        == _sha256(result.compilation_receipt_sha256, "compilation receipt")
        and receipt.held_out_query_batch_sha256
        == runtime_identity_sha256(
            [{"query_sha256": quote_sha256(query)} for query in batch]
        ),
        "combined store receipt changed",
    )
    row_identity = [
        {
            "query_sha256": row["query_sha256"],
            "vector_sha256": row["vector_sha256"],
        }
        for row in rows
    ]
    body = {
        "base_checkpoint_sha256": request.base.checkpoint.sha256,
        "dimension": dimension,
        "embedding_identity_sha256": runtime_identity_sha256(
            cumulative._plain_json(embedding_identity)  # noqa: SLF001
        ),
        "format": FROZEN_QUERY_FORMAT,
        "namespace_id": request.work.namespace_id,
        "namespace_store_id": request.work.namespace_store_id,
        "query_batch_sha256": runtime_identity_sha256(
            [{"query_sha256": quote_sha256(query)} for query in batch]
        ),
        "rows": rows,
        "vector_values_sha256": canonical_sha256(row_identity),
        "work_receipt_sha256": request.work.work_receipt_sha256,
    }
    normalized = {
        "combined_store_mode": result.combined_store_mode,
        "combined_store_receipt": receipt_payload,
        "compilation_receipt_sha256": receipt.compilation_receipt_sha256,
        "vector_artifact": {
            **body,
            "artifact_receipt_sha256": canonical_sha256(body),
        },
    }
    cumulative._assert_label_free(normalized, "staged_preparation")  # noqa: SLF001
    return normalized


def _validate_vector_artifact(
    sealed: SealedPayload,
    *,
    request: cumulative.CumulativeNamespaceRequest,
    embedding_identity: Mapping[str, Any],
) -> tuple[FrozenQueryDescriptor, Mapping[str, Sequence[float]]]:
    artifact = dict(sealed.payload)
    cumulative._assert_label_free(artifact, "frozen_query_artifact")  # noqa: SLF001
    _exact(artifact, _VECTOR_KEYS, "frozen query artifact")
    declared = _sha256(
        artifact.pop("artifact_receipt_sha256"), "frozen query artifact receipt"
    )
    _require(
        artifact.get("format") == FROZEN_QUERY_FORMAT
        and artifact.get("namespace_id") == request.work.namespace_id
        and artifact.get("namespace_store_id") == request.work.namespace_store_id
        and artifact.get("base_checkpoint_sha256") == request.base.checkpoint.sha256
        and artifact.get("work_receipt_sha256") == request.work.work_receipt_sha256
        and artifact.get("embedding_identity_sha256")
        == runtime_identity_sha256(
            cumulative._plain_json(embedding_identity)  # noqa: SLF001
        )
        and canonical_sha256(artifact) == declared,
        "frozen query artifact binding changed",
    )
    rows = artifact.get("rows")
    _require(type(rows) is list and bool(rows), "frozen query rows are invalid")
    batch: list[str] = []
    vectors: dict[str, Sequence[float]] = {}
    identities: list[dict[str, str]] = []
    dimension = artifact.get("dimension")
    _require(type(dimension) is int and dimension > 0, "vector dimension is invalid")
    for raw in rows:
        row = _mapping(raw, "frozen query row")
        _exact(
            row,
            frozenset({"query", "query_sha256", "vector", "vector_sha256"}),
            "frozen query row",
        )
        query = row.get("query")
        vector = row.get("vector")
        _require(type(query) is str and bool(query.strip()), "frozen query is invalid")
        _require(
            type(vector) is list
            and len(vector) == dimension
            and all(type(item) in {int, float} and math.isfinite(item) for item in vector),
            "frozen query vector changed",
        )
        query_digest = _sha256(row.get("query_sha256"), "frozen query digest")
        vector_digest = _sha256(row.get("vector_sha256"), "frozen vector digest")
        _require(
            query_digest == quote_sha256(query)
            and vector_digest == canonical_sha256(vector)
            and query not in vectors,
            "frozen query row identity changed",
        )
        batch.append(query)
        vectors[query] = vector
        identities.append(
            {"query_sha256": query_digest, "vector_sha256": vector_digest}
        )
    query_batch_sha256 = _sha256(
        artifact.get("query_batch_sha256"), "frozen query batch"
    )
    vector_values_sha256 = _sha256(
        artifact.get("vector_values_sha256"), "frozen vector values"
    )
    _require(
        set(cumulative.held_out_queries(request.queries)) <= set(batch)
        and query_batch_sha256
        == runtime_identity_sha256(
            [{"query_sha256": quote_sha256(query)} for query in batch]
        )
        and vector_values_sha256 == canonical_sha256(identities),
        "frozen query batch changed",
    )
    return (
        FrozenQueryDescriptor(
            namespace_store_id=request.work.namespace_store_id,
            artifact_path=sealed.path,
            artifact_sha256=sealed.sha256,
            artifact_receipt_sha256=declared,
            query_batch=tuple(batch),
            query_batch_sha256=query_batch_sha256,
            vector_values_sha256=vector_values_sha256,
            dimension=dimension,
        ),
        MappingProxyType(vectors),
    )


def _paths(root: Path, store_id: str) -> tuple[Path, Path]:
    stage = root / "staged-preparation"
    return (
        stage / "vectors" / f"{store_id}.json",
        stage / "checkpoints" / f"{store_id}.json",
    )


def _preparation_checkpoint_body(
    *,
    inputs: cumulative.ConfirmationCumulativeInput,
    request: cumulative.CumulativeNamespaceRequest,
    backend_identity: str,
    normalized: Mapping[str, Any],
    vector: SealedPayload,
    vector_path: Path,
    root: Path,
) -> dict[str, Any]:
    vector_artifact = _mapping(normalized["vector_artifact"], "vector artifact")
    return {
        "backend_identity_sha256": backend_identity,
        "base_checkpoint_sha256": request.base.checkpoint.sha256,
        "combined_store_mode": normalized["combined_store_mode"],
        "combined_store_receipt": normalized["combined_store_receipt"],
        "compilation_receipt_sha256": normalized["compilation_receipt_sha256"],
        "format": PREPARATION_FORMAT,
        "freeze_sha256": inputs.policy_freeze.sha256,
        "gold_loaded": False,
        "namespace_id": request.work.namespace_id,
        "namespace_store_id": request.work.namespace_store_id,
        "physical_provider_calls": 0,
        "preflight_sha256": inputs.preflight.sha256,
        "query_vector_artifact_receipt_sha256": vector_artifact[
            "artifact_receipt_sha256"
        ],
        "query_vector_artifact_relative_path": str(
            vector_path.relative_to(root)
        ).replace("\\", "/"),
        "query_vector_artifact_sha256": vector.sha256,
        "work_receipt_sha256": request.work.work_receipt_sha256,
        "workset_identity_sha256": inputs.workset.workset_identity_sha256,
    }


def prepare_staged_namespaces(
    inputs: cumulative.ConfirmationCumulativeInput,
    *,
    output_root: str | Path,
    backend: StagedPreparationBackend,
    token_counter: Callable[[str], int] | None = None,
) -> StagedPreparationExecution:
    """Prepare every namespace under BGE without retaining open stores."""

    backend_identity = _sha256(backend.identity_sha256, "preparation backend identity")
    _require(
        _sha256(backend.policy_freeze_sha256, "preparation policy freeze")
        == inputs.policy_freeze.sha256,
        "preparation backend binds another policy freeze",
    )
    embedding_identity = cumulative._plain_json(backend.embedding_identity)  # noqa: SLF001
    cumulative._assert_label_free(embedding_identity, "embedding_identity")  # noqa: SLF001
    root = Path(output_root)
    requests = cumulative.confirmation_cumulative_requests(
        inputs, output_root=root, token_counter=token_counter
    )
    checkpoints: list[Path] = []
    digests: list[str] = []
    descriptors: list[FrozenQueryDescriptor] = []
    created = reused = 0
    for request in requests:
        vector_path, checkpoint_path = _paths(root, request.work.namespace_store_id)
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            checkpoint = _read_sealed(
                checkpoint_path, label="staged preparation checkpoint"
            )
            payload = dict(checkpoint.payload)
            _exact(payload, _PREPARATION_KEYS, "staged preparation checkpoint")
            declared = _sha256(
                payload.pop("checkpoint_receipt_sha256"),
                "staged preparation checkpoint receipt",
            )
            _require(canonical_sha256(payload) == declared, "preparation seal changed")
            expected_path = Path(
                str(payload.get("query_vector_artifact_relative_path"))
            )
            _require(not expected_path.is_absolute(), "vector artifact path is absolute")
            resolved = (root.resolve() / expected_path).resolve()
            _require(
                resolved == vector_path.resolve() and resolved.is_relative_to(root.resolve()),
                "vector artifact escaped its namespace address",
            )
            vector = _read_sealed(resolved, label="frozen query artifact")
            _require(
                vector.sha256
                == _sha256(payload.get("query_vector_artifact_sha256"), "vector artifact")
                and payload.get("backend_identity_sha256") == backend_identity
                and payload.get("base_checkpoint_sha256")
                == request.base.checkpoint.sha256
                and payload.get("freeze_sha256") == inputs.policy_freeze.sha256
                and payload.get("preflight_sha256") == inputs.preflight.sha256
                and payload.get("workset_identity_sha256")
                == inputs.workset.workset_identity_sha256
                and payload.get("namespace_id") == request.work.namespace_id
                and payload.get("namespace_store_id")
                == request.work.namespace_store_id
                and payload.get("work_receipt_sha256")
                == request.work.work_receipt_sha256
                and payload.get("physical_provider_calls") == 0
                and payload.get("gold_loaded") is False,
                "preparation checkpoint binding changed",
            )
            _require(
                payload.get("format") == PREPARATION_FORMAT
                and type(payload.get("combined_store_mode")) is str
                and bool(str(payload.get("combined_store_mode")).strip()),
                "preparation checkpoint format or mode changed",
            )
            compilation_sha = _sha256(
                payload.get("compilation_receipt_sha256"),
                "preparation compilation receipt",
            )
            _sha256(
                payload.get("query_vector_artifact_receipt_sha256"),
                "query vector artifact receipt",
            )
            combined_payload = _mapping(
                payload.get("combined_store_receipt"), "combined store receipt"
            )
            try:
                combined = CombinedCumulativeStoreReceipt(**combined_payload)
            except (TypeError, ValueError) as exc:
                raise StagedCoordinatorError(
                    "preparation combined store receipt is invalid"
                ) from exc
            _require(
                asdict(combined) == combined_payload
                and combined.compilation_receipt_sha256 == compilation_sha
                and combined.held_out_query_batch_sha256
                == _sha256(
                    vector.payload.get("query_batch_sha256"),
                    "frozen query batch",
                ),
                "preparation combined store receipt changed",
            )
            cumulative._assert_label_free(  # noqa: SLF001
                payload, "staged_preparation_checkpoint"
            )
            descriptor, vectors = _validate_vector_artifact(
                vector, request=request, embedding_identity=embedding_identity
            )
            _require(
                descriptor.artifact_receipt_sha256
                == payload.get("query_vector_artifact_receipt_sha256"),
                "vector artifact receipt changed",
            )
            backend.verify(request, payload, vectors)
            sealed_checkpoint = checkpoint
            reused += 1
        else:
            result = backend.prepare(request)
            normalized = _normalize_vectors(result, request, embedding_identity)
            vector_payload = _mapping(normalized["vector_artifact"], "vector artifact")
            vector, _ = cumulative._publish_sealed(  # noqa: SLF001
                vector_path, vector_payload, label="frozen query artifact"
            )
            descriptor, vectors = _validate_vector_artifact(
                vector, request=request, embedding_identity=embedding_identity
            )
            body = _preparation_checkpoint_body(
                inputs=inputs,
                request=request,
                backend_identity=backend_identity,
                normalized=normalized,
                vector=vector,
                vector_path=vector_path,
                root=root,
            )
            checkpoint_payload = {
                **body,
                "checkpoint_receipt_sha256": canonical_sha256(body),
            }
            sealed_checkpoint, was_created = cumulative._publish_sealed(  # noqa: SLF001
                checkpoint_path,
                checkpoint_payload,
                label="staged preparation checkpoint",
            )
            _require(was_created, "preparation checkpoint unexpectedly existed")
            created += 1
        checkpoints.append(checkpoint_path)
        digests.append(sealed_checkpoint.sha256)
        descriptors.append(descriptor)
        del vectors
    return StagedPreparationExecution(
        checkpoint_paths=tuple(checkpoints),
        checkpoint_sha256s=tuple(digests),
        descriptors=tuple(descriptors),
        created_count=created,
        reused_count=reused,
    )


def _publish_barrier(
    inputs: cumulative.ConfirmationCumulativeInput,
    *,
    root: Path,
    preparation: StagedPreparationExecution,
    backend: StagedPreparationBackend,
    release: Mapping[str, Any],
    qwen_factory_identity_sha256: str,
    retrieval_factory_identity_sha256: str,
) -> SealedPayload:
    refs = sorted(
        (
            {
                "namespace_store_id": descriptor.namespace_store_id,
                "preparation_checkpoint_sha256": checkpoint_sha,
                "query_vector_artifact_sha256": descriptor.artifact_sha256,
            }
            for descriptor, checkpoint_sha in zip(
                preparation.descriptors,
                preparation.checkpoint_sha256s,
                strict=True,
            )
        ),
        key=lambda row: row["namespace_store_id"],
    )
    body = {
        "format": BARRIER_FORMAT,
        "freeze_sha256": inputs.policy_freeze.sha256,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "preflight_sha256": inputs.preflight.sha256,
        "preparation_backend_identity_sha256": backend.identity_sha256,
        "preparations": refs,
        "qwen_factory_identity_sha256": qwen_factory_identity_sha256,
        "release_receipt": dict(release),
        "retrieval_factory_identity_sha256": retrieval_factory_identity_sha256,
        "workset_identity_sha256": inputs.workset.workset_identity_sha256,
    }
    payload = {**body, "barrier_receipt_sha256": canonical_sha256(body)}
    cumulative._assert_label_free(payload, "staged_barrier")  # noqa: SLF001
    barrier, _ = cumulative._publish_sealed(  # noqa: SLF001
        root / "staged-preparation" / "bge-release-barrier.json",
        payload,
        label="BGE release barrier",
    )
    return barrier


def _verified_qwen_barrier(
    barrier: SealedPayload,
    *,
    qwen_factory_identity_sha256: str,
) -> SealedPayload:
    """Re-authenticate the phase barrier before any Qwen construction."""

    _require(type(barrier) is SealedPayload, "Qwen requires a sealed BGE barrier")
    observed = _read_sealed(barrier.path, label="BGE release barrier")
    _require(
        observed.sha256 == barrier.sha256
        and cumulative._plain_json(observed.payload)  # noqa: SLF001
        == cumulative._plain_json(barrier.payload),  # noqa: SLF001
        "BGE release barrier changed before Qwen load",
    )
    payload = dict(observed.payload)
    cumulative._assert_label_free(payload, "staged_barrier")  # noqa: SLF001
    _exact(payload, _BARRIER_KEYS, "BGE release barrier")
    declared = _sha256(
        payload.pop("barrier_receipt_sha256"), "BGE release barrier receipt"
    )
    preparations = payload.get("preparations")
    _require(type(preparations) is list and bool(preparations), "barrier is incomplete")
    stores: set[str] = set()
    for raw in preparations:
        row = _mapping(raw, "barrier preparation")
        _exact(
            row,
            frozenset(
                {
                    "namespace_store_id",
                    "preparation_checkpoint_sha256",
                    "query_vector_artifact_sha256",
                }
            ),
            "barrier preparation",
        )
        store_id = row.get("namespace_store_id")
        _require(
            type(store_id) is str and bool(store_id) and store_id not in stores,
            "barrier preparation identity changed",
        )
        stores.add(store_id)
        _sha256(row.get("preparation_checkpoint_sha256"), "preparation checkpoint")
        _sha256(row.get("query_vector_artifact_sha256"), "query vector artifact")
    release = dict(_mapping(payload.get("release_receipt"), "BGE release receipt"))
    _exact(release, _RELEASE_KEYS, "BGE release receipt")
    release_declared = _sha256(
        release.pop("release_receipt_sha256"), "BGE release receipt"
    )
    _require(
        payload.get("format") == BARRIER_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls") == 0
        and payload.get("qwen_factory_identity_sha256")
        == _sha256(qwen_factory_identity_sha256, "Qwen factory identity")
        and payload.get("preparation_backend_identity_sha256")
        == release.get("preparation_backend_identity_sha256")
        and release.get("format") == BGE_RELEASE_FORMAT
        and release.get("embedding_released_before_qwen_load") is True
        and release.get("physical_provider_calls") == 0
        and release.get("release_policy")
        == "close_bge_before_qwen_factory_call_v1"
        and canonical_sha256(release) == release_declared
        and canonical_sha256(payload) == declared,
        "BGE release barrier is invalid for Qwen construction",
    )
    for key, label in (
        ("freeze_sha256", "barrier freeze"),
        ("preflight_sha256", "barrier preflight"),
        ("preparation_backend_identity_sha256", "preparation backend identity"),
        ("retrieval_factory_identity_sha256", "retrieval factory identity"),
        ("workset_identity_sha256", "barrier workset identity"),
    ):
        _sha256(payload.get(key), label)
    _sha256(release.get("embedding_identity_sha256"), "BGE embedding identity")
    return observed


def _production_qwen_config_binding(config: Any) -> Mapping[str, Any]:
    """Validate the exact pinned identities consumed by ``_load_shared_qwen``."""

    from memory_condense.modeling.qwen_prefix import (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
        expected_prefix_checkpoint_sha256,
    )
    from memory_condense.search.selectors.causal_choice_scorer import (
        QWEN_CHOICE_CHECKPOINT_SHA256,
        QWEN_CHOICE_MODEL_ID,
        QWEN_CHOICE_MODEL_REVISION,
    )

    retrieval = getattr(config, "retrieval", None)
    dump = getattr(retrieval, "model_dump", None)
    _require(callable(dump), "Qwen factory requires an EvalConfig retrieval policy")
    controls = dump(mode="json")
    _require(isinstance(controls, Mapping), "Qwen retrieval policy is invalid")
    layers = controls.get("coverage_selector_prefix_layers")
    _require(
        type(layers) is int and layers > 0,
        "Qwen prefix layer count is invalid",
    )
    expected_prefix = expected_prefix_checkpoint_sha256(layers)
    _require(
        controls.get("coverage_selection") is True
        and controls.get("coverage_selector_backend") == "qwen_prefix_choice"
        and controls.get("coverage_selector_prefix_model_id") == DEFAULT_MODEL_ID
        and controls.get("coverage_selector_prefix_revision")
        == DEFAULT_MODEL_REVISION
        and controls.get("coverage_selector_prefix_checkpoint_sha256")
        == expected_prefix
        and controls.get("coverage_selector_choice_model_id")
        == QWEN_CHOICE_MODEL_ID
        and controls.get("coverage_selector_choice_revision")
        == QWEN_CHOICE_MODEL_REVISION
        and controls.get("coverage_selector_choice_checkpoint_sha256")
        == QWEN_CHOICE_CHECKPOINT_SHA256,
        "Qwen factory is not bound to the pinned shared-Qwen identities",
    )
    binding = {
        "choice": {
            "checkpoint_sha256": QWEN_CHOICE_CHECKPOINT_SHA256,
            "device": controls.get("coverage_selector_choice_device"),
            "dtype": controls.get("coverage_selector_choice_dtype"),
            "model_id": QWEN_CHOICE_MODEL_ID,
            "model_revision": QWEN_CHOICE_MODEL_REVISION,
        },
        "prefix": {
            "attention_layer": controls.get("coverage_selector_attention_layer"),
            "checkpoint_sha256": expected_prefix,
            "device": controls.get("coverage_selector_prefix_device"),
            "dtype": controls.get("coverage_selector_prefix_dtype"),
            "layers": layers,
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
        },
        "retrieval_policy_sha256": runtime_identity_sha256(dict(controls)),
    }
    cumulative._assert_label_free(binding, "production_qwen_binding")  # noqa: SLF001
    return MappingProxyType(binding)


class ProductionStagedQwenRuntime:
    """Owned shared selector/linker pair returned by the historical loader."""

    def __init__(
        self,
        *,
        factory_identity_sha256: str,
        coverage_selector: Any,
        representative_linker: Any,
    ) -> None:
        _require(coverage_selector is not None, "shared Qwen selector is missing")
        _require(representative_linker is not None, "shared Qwen linker is missing")
        self._selector = coverage_selector
        self._linker = representative_linker
        self._closed = False
        self._identity = canonical_sha256(
            {
                "coverage_selector_runtime": (
                    f"{type(coverage_selector).__module__}."
                    f"{type(coverage_selector).__qualname__}"
                ),
                "factory_identity_sha256": _sha256(
                    factory_identity_sha256, "Qwen factory identity"
                ),
                "format": PRODUCTION_QWEN_RUNTIME_FORMAT,
                "physical_provider_calls": 0,
                "representative_linker_runtime": (
                    f"{type(representative_linker).__module__}."
                    f"{type(representative_linker).__qualname__}"
                ),
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @property
    def coverage_selector(self) -> Any:
        _require(not self._closed, "shared Qwen runtime is closed")
        return self._selector

    @property
    def representative_linker(self) -> Any:
        _require(not self._closed, "shared Qwen runtime is closed")
        return self._linker

    @property
    def physical_provider_calls(self) -> int:
        return 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        selector, linker = self._selector, self._linker
        self._selector = None
        self._linker = None
        torch = getattr(getattr(linker, "encoder", None), "_torch", None)
        try:
            close = getattr(selector, "close", None)
            _require(callable(close), "shared Qwen selector has no close()")
            close()
        finally:
            del selector, linker
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


class ProductionStagedQwenRuntimeFactory:
    """Load the historical shared local-Qwen pair only after a sealed barrier."""

    def __init__(
        self,
        *,
        config: Any,
        qwen_prefix_model_dir: str | Path,
        qwen_choice_model_dir: str | Path,
        load_shared_qwen: Callable[[Any, Path, Path], tuple[Any, Any]] | None = None,
        loader_identity_sha256: str | None = None,
    ) -> None:
        binding = _production_qwen_config_binding(config)
        if load_shared_qwen is None:
            from tools.confirmation_qwen_runtime import load_shared_qwen as qwen_loader

            load_shared_qwen = qwen_loader
            source = Path(qwen_loader.__code__.co_filename).resolve()
            loader_identity_sha256 = canonical_sha256(
                {
                    "callable": (
                        "tools.confirmation_qwen_runtime.load_shared_qwen"
                    ),
                    "source_file_sha256": file_sha256(source),
                }
            )
        else:
            loader_identity_sha256 = _sha256(
                loader_identity_sha256, "injected Qwen loader identity"
            )
        self._config = config
        self._prefix_dir = Path(qwen_prefix_model_dir).resolve()
        self._choice_dir = Path(qwen_choice_model_dir).resolve()
        self._loader = load_shared_qwen
        self._loader_identity = loader_identity_sha256
        self._identity = canonical_sha256(
            {
                "format": PRODUCTION_QWEN_FACTORY_FORMAT,
                "loader_identity_sha256": loader_identity_sha256,
                "model_binding": cumulative._plain_json(binding),  # noqa: SLF001
                "physical_provider_calls": 0,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    def load_after_barrier(self, barrier: SealedPayload) -> ProductionStagedQwenRuntime:
        _verified_qwen_barrier(
            barrier, qwen_factory_identity_sha256=self._identity
        )
        selector, linker = self._loader(
            self._config, self._prefix_dir, self._choice_dir
        )
        try:
            return ProductionStagedQwenRuntime(
                factory_identity_sha256=self._identity,
                coverage_selector=selector,
                representative_linker=linker,
            )
        except BaseException:
            close = getattr(selector, "close", None)
            if callable(close):
                close()
            raise


def execute_staged_confirmation_cumulative(
    inputs: cumulative.ConfirmationCumulativeInput,
    *,
    output_root: str | Path,
    preparation_backend: StagedPreparationBackend,
    qwen_factory: StagedQwenRuntimeFactory,
    retrieval_factory: StagedRetrievalBackendFactory,
    token_counter: Callable[[str], int] | None = None,
    before_bge_release: StagedBeforeBgeReleaseHook | None = None,
) -> StagedCoordinatorExecution:
    """Run the population-wide BGE barrier before constructing local Qwen."""

    root = Path(output_root)
    qwen_identity = _sha256(qwen_factory.identity_sha256, "Qwen factory identity")
    retrieval_identity = _sha256(
        retrieval_factory.identity_sha256, "retrieval factory identity"
    )
    try:
        preparation = prepare_staged_namespaces(
            inputs,
            output_root=root,
            backend=preparation_backend,
            token_counter=token_counter,
        )
        if before_bge_release is not None:
            _require(
                before_bge_release(preparation, preparation_backend) is None,
                "before-BGE-release hook must not transfer resident state",
            )
    except BaseException as original:
        # Release model memory on failed preparation, but never publish a phase
        # barrier or construct Qwen for an incomplete population.
        try:
            preparation_backend.release_bge()
        except BaseException as cleanup_error:
            original.add_note(f"BGE release also failed: {cleanup_error!r}")
        raise
    release = _validated_release(
        preparation_backend.release_bge(), backend=preparation_backend
    )
    barrier = _publish_barrier(
        inputs,
        root=root,
        preparation=preparation,
        backend=preparation_backend,
        release=release,
        qwen_factory_identity_sha256=qwen_identity,
        retrieval_factory_identity_sha256=retrieval_identity,
    )
    qwen = qwen_factory.load_after_barrier(barrier)
    backend: cumulative.CumulativeNamespaceBackend | None = None
    try:
        _sha256(qwen.identity_sha256, "Qwen runtime identity")
        _require(
            type(qwen.physical_provider_calls) is int
            and qwen.physical_provider_calls == 0,
            "Qwen runtime used a provider",
        )
        backend = retrieval_factory.create(
            qwen_runtime=qwen,
            barrier=barrier,
            frozen_queries=preparation.descriptors,
        )
        execution = cumulative.execute_confirmation_cumulative_namespaces(
            inputs,
            output_root=root,
            backend=backend,
            token_counter=token_counter,
        )
    finally:
        backend = None
        qwen.close()
    _require(execution.physical_provider_calls == 0, "staged retrieval used providers")
    return StagedCoordinatorExecution(
        preparation=preparation,
        barrier=barrier,
        cumulative=execution,
    )


class _RecordingEmbedder:
    def __init__(self, delegate: Any, expected_batch: tuple[str, ...]) -> None:
        self._delegate = delegate
        self._expected = expected_batch
        self.vectors: Mapping[str, Sequence[float]] | None = None

    @property
    def dim(self) -> int:
        return int(self._delegate.dim)

    def embed_query(self, query: str) -> Any:
        return self._delegate.embed_query(query)

    def embed_queries(self, queries: Sequence[str]) -> Any:
        values = self._delegate.embed_queries(queries)
        if tuple(queries) == self._expected:
            rows = np.asarray(values, dtype=np.float32)
            _require(
                rows.ndim == 2 and rows.shape[0] == len(self._expected),
                "BGE query batch shape changed",
            )
            self.vectors = MappingProxyType(
                {
                    query: tuple(float(item) for item in rows[index].tolist())
                    for index, query in enumerate(self._expected)
                }
            )
        return values

    def embed_chunks(self, chunks: Any) -> Any:
        return self._delegate.embed_chunks(chunks)


class ProductionStagedPreparationBackend:
    """Production combined-store preparation with no Qwen dependency."""

    def __init__(
        self,
        *,
        policy_freeze_sha256: str,
        source_backend_identity_sha256: str,
        source_treatment_contract_sha256: str,
        config: Any,
        embedder: Any,
        embedding_identity: Mapping[str, Any],
        compilation_policy: Any,
        build_store: Callable[..., Any] | None = None,
        open_store: Callable[..., Any] | None = None,
    ) -> None:
        from memory_condense.eval.recall_guarded_cumulative_runtime import (
            build_recall_guarded_cumulative_store,
            open_recall_guarded_cumulative_store,
        )

        self._policy_freeze_sha256 = _sha256(policy_freeze_sha256, "policy freeze")
        self._source_backend_identity = _sha256(
            source_backend_identity_sha256, "source backend identity"
        )
        self._source_contract = _sha256(
            source_treatment_contract_sha256, "source treatment contract"
        )
        self._config = config
        self._embedder = embedder
        self._embedding_identity = cumulative._plain_json(embedding_identity)  # noqa: SLF001
        self._compilation_policy = compilation_policy
        self._build = build_store or build_recall_guarded_cumulative_store
        self._open = open_store or open_recall_guarded_cumulative_store
        self._sentinel = object()
        self._released = False
        self._identity = canonical_sha256(
            {
                "backend": "production-confirmation-staged-preparation-v1",
                "compilation_policy": cumulative._plain_json(compilation_policy),  # noqa: SLF001
                "embedding_identity": self._embedding_identity,
                "policy_freeze_sha256": self._policy_freeze_sha256,
                "retrieval": config.retrieval.model_dump(mode="json"),
                "source_backend_identity_sha256": self._source_backend_identity,
                "source_treatment_contract_sha256": self._source_contract,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @property
    def policy_freeze_sha256(self) -> str:
        return self._policy_freeze_sha256

    @property
    def embedding_identity(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._embedding_identity))

    def _source(self, request: cumulative.CumulativeNamespaceRequest) -> Path:
        return cumulative.validate_production_source_database(
            request.base,
            source_backend_identity_sha256=self._source_backend_identity,
            source_treatment_contract_sha256=self._source_contract,
            embedding_identity=self._embedding_identity,
        )

    def prepare(
        self, request: cumulative.CumulativeNamespaceRequest
    ) -> StagedPreparationResult:
        _require(not self._released, "BGE was already released")
        from memory_condense.eval.recall_guarded_cumulative_runtime import _query_batch

        source = self._source(request)
        held_out = cumulative.held_out_queries(request.queries)
        query_batch = _query_batch(held_out, self._config)
        recording = _RecordingEmbedder(self._embedder, query_batch)
        target = request.namespace_root / "combined-store"
        if target.exists():
            prepared = self._open(
                target,
                config=self._config,
                embedder=recording,
                held_out_queries=held_out,
                coverage_selector=self._sentinel,
            )
            mode = "verified_cache_hit"
        else:
            prepared = self._build(
                source,
                target,
                config=self._config,
                embedder=recording,
                held_out_queries=held_out,
                compilation_policy=self._compilation_policy,
                coverage_selector=self._sentinel,
                embedding_identity=self._embedding_identity,
            )
            mode = "fresh_atomic_build"
        try:
            _require(recording.vectors is not None, "runtime did not freeze query vectors")
            _require(
                prepared.receipt.source_database_sha256 == file_sha256(source),
                "prepared store does not bind the source database",
            )
            return StagedPreparationResult(
                namespace_id=request.work.namespace_id,
                namespace_store_id=request.work.namespace_store_id,
                base_checkpoint_sha256=request.base.checkpoint.sha256,
                combined_store_receipt=MappingProxyType(asdict(prepared.receipt)),
                compilation_receipt_sha256=prepared.compilation.receipt_sha256,
                combined_store_mode=mode,
                query_batch=query_batch,
                query_vectors=recording.vectors,
            )
        finally:
            prepared.close()

    def verify(
        self,
        request: cumulative.CumulativeNamespaceRequest,
        expected: Mapping[str, Any],
        query_vectors: Mapping[str, Sequence[float]],
    ) -> None:
        _require(not self._released, "BGE was already released")
        source = self._source(request)
        target = request.namespace_root / "combined-store"
        _require(target.is_dir(), "prepared combined store is missing")
        prepared = self._open(
            target,
            config=self._config,
            embedder=FrozenQueryEmbedder(query_vectors),
            held_out_queries=cumulative.held_out_queries(request.queries),
            coverage_selector=self._sentinel,
        )
        try:
            _require(
                asdict(prepared.receipt) == expected.get("combined_store_receipt")
                and prepared.compilation.receipt_sha256
                == expected.get("compilation_receipt_sha256")
                and prepared.receipt.source_database_sha256 == file_sha256(source),
                "prepared combined store changed",
            )
        finally:
            prepared.close()

    def freeze_query_batch(
        self, queries: Sequence[str]
    ) -> Mapping[str, Sequence[float]]:
        """Freeze one supplemental question-only batch before BGE release.

        The normal retrieval descriptor remains byte-for-byte unchanged.  The
        confirmation semantic-plane freezer uses this narrow hook to persist
        residual-query facets in a separate artifact while the same pinned
        BGE instance and embedding identity are still resident.
        """

        _require(not self._released, "BGE was already released")
        batch = tuple(queries)
        _require(
            bool(batch)
            and len(batch) == len(set(batch))
            and all(type(query) is str and bool(query.strip()) for query in batch),
            "supplemental BGE query batch must be ordered unique text",
        )
        values = np.asarray(self._embedder.embed_queries(batch), dtype=np.float32)
        _require(
            values.ndim == 2
            and values.shape[0] == len(batch)
            and values.shape[1] > 0
            and bool(np.isfinite(values).all()),
            "supplemental BGE query batch shape changed",
        )
        declared_dimension = self._embedding_identity.get("dimension")
        _require(
            declared_dimension is None or declared_dimension == values.shape[1],
            "supplemental BGE query dimension differs from its identity",
        )
        return MappingProxyType(
            {
                query: tuple(float(item) for item in values[index].tolist())
                for index, query in enumerate(batch)
            }
        )

    def release_bge(self) -> Mapping[str, Any]:
        if not self._released:
            embedder = self._embedder
            close = getattr(embedder, "close", None)
            _require(callable(close), "staged BGE runtime must expose close()")
            close()
            self._embedder = None
            self._released = True
        return bge_release_receipt(
            preparation_backend_identity_sha256=self._identity,
            embedding_identity=self._embedding_identity,
        )


class SealedFrozenQueryEmbedder:
    """Bounded replay embedder loading one authenticated namespace at a time."""

    def __init__(self, descriptors: Sequence[FrozenQueryDescriptor]) -> None:
        _require(bool(descriptors), "frozen query descriptors are empty")
        dimensions = {item.dimension for item in descriptors}
        _require(len(dimensions) == 1, "frozen query descriptor dimensions differ")
        self._dim = dimensions.pop()
        self._by_batch: dict[tuple[str, ...], FrozenQueryDescriptor] = {}
        vector_identities: dict[tuple[str, ...], str] = {}
        for descriptor in descriptors:
            previous = vector_identities.setdefault(
                descriptor.query_batch, descriptor.vector_values_sha256
            )
            _require(
                previous == descriptor.vector_values_sha256,
                "duplicate query batches have different frozen vectors",
            )
            self._by_batch.setdefault(descriptor.query_batch, descriptor)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        batch = tuple(queries)
        descriptor = self._by_batch.get(batch)
        _require(descriptor is not None, "query batch was not frozen before Qwen load")
        sealed = _read_sealed(
            descriptor.artifact_path, label="frozen query replay artifact"
        )
        _require(
            sealed.sha256 == descriptor.artifact_sha256,
            "frozen query replay artifact changed",
        )
        rows = sealed.payload.get("rows")
        _require(type(rows) is list, "frozen query replay rows changed")
        vectors = {
            str(row["query"]): row["vector"]
            for row in rows
            if type(row) is dict
        }
        _require(tuple(vectors) == batch, "frozen query replay batch changed")
        return FrozenQueryEmbedder(vectors).embed_queries(batch)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_queries((query,))[0]

    def embed_chunks(self, chunks: Any) -> list[Any]:
        _require(not chunks, "staged replay must not embed chunks")
        return []


class ProductionStagedRetrievalBackendFactory:
    """Create the existing production S0--S3 backend after the BGE barrier."""

    def __init__(
        self,
        *,
        policy_freeze_sha256: str,
        runtime_policy_binding: Mapping[str, Any],
        source_backend_identity_sha256: str,
        source_treatment_contract_sha256: str,
        config: Any,
        compilation_policy: Any,
        episode_policy_factory: Callable[[str], Any],
        representative_policy_factory: Callable[[str], Any],
        closure_policy: Any,
        max_context_tokens: int,
        max_prompt_tokens: int,
        responder_output_token_reserve: int,
        source_router_max_sources: int,
        source_router_rrf_constant: int,
        embedding_identity: Mapping[str, Any],
        open_store: Callable[..., Any] | None = None,
        retrieve: Callable[..., Any] | None = None,
    ) -> None:
        self._values = dict(locals())
        self._values.pop("self")
        self._identity = canonical_sha256(
            {
                "factory": "production-confirmation-staged-retrieval-v1",
                "policy_freeze_sha256": policy_freeze_sha256,
                "runtime_policy_binding": cumulative._plain_json(  # noqa: SLF001
                    runtime_policy_binding
                ),
                "source_backend_identity_sha256": source_backend_identity_sha256,
                "source_treatment_contract_sha256": source_treatment_contract_sha256,
                "retrieval": config.retrieval.model_dump(mode="json"),
                "embedding_identity": cumulative._plain_json(embedding_identity),  # noqa: SLF001
                "budgets": {
                    "max_context_tokens": max_context_tokens,
                    "max_prompt_tokens": max_prompt_tokens,
                    "responder_output_token_reserve": responder_output_token_reserve,
                    "source_router_max_sources": source_router_max_sources,
                    "source_router_rrf_constant": source_router_rrf_constant,
                },
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    def create(
        self,
        *,
        qwen_runtime: StagedQwenRuntime,
        barrier: SealedPayload,
        frozen_queries: tuple[FrozenQueryDescriptor, ...],
    ) -> cumulative.CumulativeNamespaceBackend:
        values = dict(self._values)
        binding = dict(values.pop("runtime_policy_binding"))
        _require(
            binding.get("model_residency_mode")
            == cumulative.STAGED_PRODUCTION_MODE,
            "retrieval factory is not bound to staged residency",
        )
        binding["qwen_runtime_identity_sha256"] = qwen_runtime.identity_sha256
        binding["staged_barrier_receipt_sha256"] = barrier.payload[
            "barrier_receipt_sha256"
        ]
        for optional in ("open_store", "retrieve"):
            if values.get(optional) is None:
                values.pop(optional)
        return cumulative.ProductionCumulativeNamespaceBackend(
            runtime_policy_binding=binding,
            model_residency_mode=cumulative.STAGED_PRODUCTION_MODE,
            embedding_runtime_kind="sealed_frozen_queries",
            staged_barrier_receipt_sha256=barrier.payload[
                "barrier_receipt_sha256"
            ],
            embedder=SealedFrozenQueryEmbedder(frozen_queries),
            coverage_selector=qwen_runtime.coverage_selector,
            representative_linker=qwen_runtime.representative_linker,
            **values,
        )


__all__ = [
    "BARRIER_FORMAT",
    "BGE_RELEASE_FORMAT",
    "FROZEN_QUERY_FORMAT",
    "PREPARATION_FORMAT",
    "FrozenQueryDescriptor",
    "ProductionStagedQwenRuntime",
    "ProductionStagedQwenRuntimeFactory",
    "ProductionStagedPreparationBackend",
    "ProductionStagedRetrievalBackendFactory",
    "SealedFrozenQueryEmbedder",
    "StagedCoordinatorError",
    "StagedCoordinatorExecution",
    "StagedBeforeBgeReleaseHook",
    "StagedPreparationBackend",
    "StagedPreparationExecution",
    "StagedPreparationResult",
    "StagedQwenRuntime",
    "StagedQwenRuntimeFactory",
    "StagedRetrievalBackendFactory",
    "bge_release_receipt",
    "execute_staged_confirmation_cumulative",
    "prepare_staged_namespaces",
]
