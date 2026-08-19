"""Closed, path-free contracts shared by diffuse base artifact layers."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.diffuse_longmemeval_inputs import (
    GoldBlindLongMemEvalSample,
)
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    implementation_sha256,
)
from memory_condense.eval.schemas import EvalConfig
from memory_condense.persistence.db import CURRENT_SCHEMA_VERSION

if TYPE_CHECKING:
    from memory_condense.eval.diffuse_longmemeval_runtime import (
        FrozenLegacyQueryInputs,
    )


BASE_STORE_FORMAT = "memory-condense-longmemeval-diffuse-base-store-v1"
BASE_STORE_REVISION = 1
QUERY_INPUT_FORMAT = "memory-condense-longmemeval-frozen-query-pointers-v1"
QUERY_INPUT_REVISION = 1
QUERY_MANIFEST_FORMAT = "memory-condense-longmemeval-query-manifest-v1"
DERIVED_STORE_FORMAT = "memory-condense-longmemeval-derived-store-v1"

STORE_MANIFEST_NAME = "base-manifest.json"
QUERY_MANIFEST_NAME = "query-manifest.json"
FROZEN_QUERY_INPUTS_NAME = "frozen-legacy-inputs.json"
DERIVED_ORIGIN_NAME = "derived-origin.json"
DERIVED_LEASE_NAME = "derived-open.claim"
STORE_DIRECTORY_NAME = "store"
DATABASE_NAME = "memory.db"
INDEX_NAME = "hnsw_index.bin"
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"


class DiffuseBaseArtifactError(RuntimeError):
    """A shared diffuse artifact violates its closed integrity contract."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DiffuseBaseTreatmentIdentity(_FrozenModel):
    """Gold-free coordinates of one sanitized treatment population."""

    treatment_file_sha256: str = Field(pattern=_DIGEST_PATTERN)
    sanitized_projection_sha256: str = Field(pattern=_DIGEST_PATTERN)
    dataset_sha256: str = Field(pattern=_DIGEST_PATTERN)
    split_manifest_sha256: str = Field(pattern=_DIGEST_PATTERN)
    ordered_question_ids_sha256: str = Field(pattern=_DIGEST_PATTERN)
    sample_count: int = Field(ge=1)
    sample_ordinal: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordinal_within_population(self) -> "DiffuseBaseTreatmentIdentity":
        if self.sample_ordinal >= self.sample_count:
            raise ValueError("sample_ordinal must be within sample_count")
        return self


class DiffuseBaseEmbeddingIdentity(_FrozenModel):
    """Exact vector-producing execution identity supplied by the runtime."""

    backend: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=_DIGEST_PATTERN)
    dimension: int = Field(ge=1)
    device: str = Field(min_length=1)
    batch_size: int = Field(ge=1)
    normalize_embeddings: bool
    output_dtype: str = Field(min_length=1)


class DiffuseBaseBuildRuntimeIdentity(_FrozenModel):
    """Declared store-builder identity, checked against the live facade."""

    runtime_id: str = Field(min_length=1)
    factory_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    condenser_class: str = Field(min_length=1)
    index_space: Literal["cosine"] = "cosine"
    index_dimension: int = Field(ge=1)
    index_ef_construction: int = Field(ge=1)
    index_m: int = Field(ge=1)
    index_max_elements: int = Field(ge=1)
    certification: Literal[
        "owned_runtime_v1",
        "deterministic_test_v1",
        "declaration_only_v1",
    ] = "declaration_only_v1"


class DiffuseChunkerIdentity(_FrozenModel):
    min_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "DiffuseChunkerIdentity":
        if self.max_tokens < self.min_tokens:
            raise ValueError("chunker maximum must be at least its minimum")
        return self


class DiffuseQueryConfigIdentity(_FrozenModel):
    retrieval_policy_sha256: str = Field(pattern=_DIGEST_PATTERN)


class DiffuseBaseStoreManifest(_FrozenModel):
    format: str = BASE_STORE_FORMAT
    revision: int = BASE_STORE_REVISION
    address_kind: Literal["input_recipe_sha256"] = "input_recipe_sha256"
    base_store_key: str = Field(pattern=_DIGEST_PATTERN)
    sample_id_sha256: str = Field(pattern=_DIGEST_PATTERN)
    corpus_sha256: str = Field(pattern=_DIGEST_PATTERN)
    store_sample_sha256: str = Field(pattern=_DIGEST_PATTERN)
    chunker_identity: DiffuseChunkerIdentity
    chunker_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    embedding_identity: DiffuseBaseEmbeddingIdentity
    embedding_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    build_runtime_identity: DiffuseBaseBuildRuntimeIdentity
    build_runtime_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    implementation_sha256: str = Field(pattern=_DIGEST_PATTERN)
    environment_lock_sha256: str = Field(pattern=_DIGEST_PATTERN)
    schema_version: int = Field(ge=1)
    database_schema_sha256: str = Field(pattern=_DIGEST_PATTERN)
    deterministic_turn_ids_sha256: str = Field(pattern=_DIGEST_PATTERN)
    turn_sequence_sha256: str = Field(pattern=_DIGEST_PATTERN)
    turn_count: int = Field(ge=1)
    chunk_sequence_sha256: str = Field(pattern=_DIGEST_PATTERN)
    chunk_count: int = Field(ge=1)
    source_ids: tuple[str, ...]
    source_streams_sha256: str = Field(pattern=_DIGEST_PATTERN)
    source_count: int = Field(ge=1)
    # Exact empty derived-table counts plus persisted discourse revision row;
    # this does not claim anything about transient process/GPU memory.
    base_state_audit_sha256: str = Field(pattern=_DIGEST_PATTERN)
    database_sha256: str = Field(pattern=_DIGEST_PATTERN)
    database_bytes: int = Field(ge=1)
    index_sha256: str = Field(pattern=_DIGEST_PATTERN)
    index_bytes: int = Field(ge=1)
    artifact_sha256: str = Field(pattern=_DIGEST_PATTERN)


class FrozenAnchorDiagnostics(_FrozenModel):
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    route: str | None = None
    association_score: float | None = None
    anchor_chunk_id: str | None = None
    association_hop: int | None = Field(default=None, ge=1)
    edge_source_chunk_id: str | None = None
    association_path: tuple[str, ...] | None = None
    diffusion_heat: float | None = Field(default=None, ge=0.0)
    association_support: int | None = Field(default=None, ge=0)
    memory_source_id: str | None = None
    source_heat: float | None = Field(default=None, ge=0.0)
    source_token_budget: int | None = Field(default=None, ge=0)
    transition_distance: int | None = Field(default=None, ge=1)
    transition_direction: Literal["previous", "next"] | None = None
    consolidation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    consolidation_anchor: str | None = None
    consolidation_support: int | None = Field(default=None, ge=0)


class FrozenAnchorPointer(_FrozenModel):
    chunk_id: str = Field(min_length=1)
    diagnostics: FrozenAnchorDiagnostics


class FrozenLexicalSourcePointer(_FrozenModel):
    source_id: str = Field(min_length=1)
    score: float


class FrozenQueryPointer(_FrozenModel):
    question_id: str = Field(min_length=1)
    question_probe_sha256: str = Field(pattern=_DIGEST_PATTERN)
    query_sha256: str = Field(pattern=_DIGEST_PATTERN)
    retrieval_policy_sha256: str = Field(pattern=_DIGEST_PATTERN)
    anchors: tuple[FrozenAnchorPointer, ...]
    lexical_sources: tuple[FrozenLexicalSourcePointer, ...]
    universe_source_ids: tuple[str, ...]
    source_streams_sha256: str = Field(pattern=_DIGEST_PATTERN)
    frozen_receipt_sha256: str = Field(pattern=_DIGEST_PATTERN)


class FrozenQueryPointerArtifact(_FrozenModel):
    format: str = QUERY_INPUT_FORMAT
    revision: int = QUERY_INPUT_REVISION
    base_store_key: str = Field(pattern=_DIGEST_PATTERN)
    rows: tuple[FrozenQueryPointer, ...]
    query_set_sha256: str = Field(pattern=_DIGEST_PATTERN)


class DiffuseQueryInputManifest(_FrozenModel):
    format: str = QUERY_MANIFEST_FORMAT
    revision: int = QUERY_INPUT_REVISION
    address_kind: Literal["input_recipe_sha256"] = "input_recipe_sha256"
    query_input_key: str = Field(pattern=_DIGEST_PATTERN)
    base_store_key: str = Field(pattern=_DIGEST_PATTERN)
    base_artifact_sha256: str = Field(pattern=_DIGEST_PATTERN)
    base_manifest_sha256: str = Field(pattern=_DIGEST_PATTERN)
    database_sha256: str = Field(pattern=_DIGEST_PATTERN)
    index_sha256: str = Field(pattern=_DIGEST_PATTERN)
    source_streams_sha256: str = Field(pattern=_DIGEST_PATTERN)
    turn_sequence_sha256: str = Field(pattern=_DIGEST_PATTERN)
    chunk_sequence_sha256: str = Field(pattern=_DIGEST_PATTERN)
    treatment_identity: DiffuseBaseTreatmentIdentity
    treatment_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    query_sample_sha256: str = Field(pattern=_DIGEST_PATTERN)
    question_ids_sha256: str = Field(pattern=_DIGEST_PATTERN)
    question_probes_sha256: str = Field(pattern=_DIGEST_PATTERN)
    config_identity: DiffuseQueryConfigIdentity
    config_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    embedding_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    frozen_inputs_sha256: str = Field(pattern=_DIGEST_PATTERN)
    frozen_inputs_bytes: int = Field(ge=1)
    query_set_sha256: str = Field(pattern=_DIGEST_PATTERN)
    frozen_receipts_sha256: str = Field(pattern=_DIGEST_PATTERN)
    query_count: int = Field(ge=1)
    artifact_sha256: str = Field(pattern=_DIGEST_PATTERN)


class DiffuseDerivedOrigin(_FrozenModel):
    format: str = DERIVED_STORE_FORMAT
    base_store_key: str = Field(pattern=_DIGEST_PATTERN)
    base_artifact_sha256: str = Field(pattern=_DIGEST_PATTERN)
    base_manifest_sha256: str = Field(pattern=_DIGEST_PATTERN)
    query_input_key: str = Field(pattern=_DIGEST_PATTERN)
    query_artifact_sha256: str = Field(pattern=_DIGEST_PATTERN)
    treatment_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    config_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    embedding_identity_sha256: str = Field(pattern=_DIGEST_PATTERN)
    source_streams_sha256: str = Field(pattern=_DIGEST_PATTERN)
    turn_sequence_sha256: str = Field(pattern=_DIGEST_PATTERN)
    chunk_sequence_sha256: str = Field(pattern=_DIGEST_PATTERN)
    query_set_sha256: str = Field(pattern=_DIGEST_PATTERN)
    arm_id: str = Field(min_length=1)
    arm_sha256: str = Field(pattern=_DIGEST_PATTERN)
    initial_database_sha256: str = Field(pattern=_DIGEST_PATTERN)
    initial_index_sha256: str = Field(pattern=_DIGEST_PATTERN)
    receipt_sha256: str = Field(pattern=_DIGEST_PATTERN)


@dataclass(frozen=True, slots=True)
class VerifiedDiffuseLongMemEvalBase:
    """Path-independent receipt plus rehydrated, gold-free query rows."""

    store_path: Path
    query_inputs_path: Path
    store_manifest: DiffuseBaseStoreManifest
    query_manifest: DiffuseQueryInputManifest
    frozen_query_inputs: tuple["FrozenLegacyQueryInputs", ...]
    store_manifest_sha256: str
    query_manifest_sha256: str
    _sample: GoldBlindLongMemEvalSample = field(repr=False)
    _treatment_identity: DiffuseBaseTreatmentIdentity = field(repr=False)
    _config: EvalConfig = field(repr=False)
    _embedding_identity: DiffuseBaseEmbeddingIdentity = field(repr=False)

    @property
    def base_store_key(self) -> str:
        return self.store_manifest.base_store_key

    @property
    def query_input_key(self) -> str:
        return self.query_manifest.query_input_key


@dataclass(frozen=True, slots=True)
class DiffuseDerivedStore:
    path: Path
    origin: DiffuseDerivedOrigin
    base: VerifiedDiffuseLongMemEvalBase


def canonical_json_bytes(value: object) -> bytes:
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


def model_bytes(value: BaseModel) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json"))


def self_sha256(value: BaseModel, field_name: str) -> str:
    return canonical_sha256(
        value.model_dump(mode="json", exclude={field_name})
    )


def require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def identity(value: BaseModel | Mapping[str, object] | object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return canonical_sha256(value)


def sample_store_payload(
    sample: GoldBlindLongMemEvalSample,
) -> dict[str, object]:
    return {
        "format": "memory-condense-longmemeval-store-sample-v1",
        "sample_id_sha256": canonical_sha256({"sample_id": sample.sample_id}),
        "corpus_sha256": sample.corpus_sha256,
        "deterministic_turn_ids": list(sample.deterministic_turn_ids),
    }


def query_sample_payload(
    sample: GoldBlindLongMemEvalSample,
) -> dict[str, object]:
    return {
        "format": "memory-condense-longmemeval-query-sample-v1",
        "store_sample_sha256": canonical_sha256(sample_store_payload(sample)),
        "questions": [
            {
                "question_id": question.question_id,
                "probe_sha256": question.probe_sha256,
            }
            for question in sample.questions
        ],
    }


def config_identity(config: EvalConfig) -> DiffuseQueryConfigIdentity:
    # Prompt/provider controls are downstream arm identities, not acquisition.
    return DiffuseQueryConfigIdentity(
        retrieval_policy_sha256=canonical_sha256(
            config.retrieval.model_dump(mode="json")
        )
    )


def chunker_identity(config: EvalConfig) -> DiffuseChunkerIdentity:
    return DiffuseChunkerIdentity(
        min_tokens=config.chunker.min_tokens,
        max_tokens=config.chunker.max_tokens,
    )


def diffuse_base_store_key(
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    *,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    build_runtime_identity: DiffuseBaseBuildRuntimeIdentity,
    implementation_digest: str,
    environment_digest: str,
) -> str:
    """Address only corpus-ingest inputs; probes/retrieval are excluded."""

    return canonical_sha256(
        {
            "format": BASE_STORE_FORMAT,
            "revision": BASE_STORE_REVISION,
            "store_sample_sha256": canonical_sha256(sample_store_payload(sample)),
            "chunker": chunker_identity(config).model_dump(mode="json"),
            "embedding_identity": embedding_identity.model_dump(mode="json"),
            "build_runtime_identity": build_runtime_identity.model_dump(mode="json"),
            "implementation_sha256": implementation_digest,
            "environment_lock_sha256": environment_digest,
            "schema_version": CURRENT_SCHEMA_VERSION,
        }
    )


def diffuse_query_input_key(
    *,
    base_store_key: str,
    treatment_identity: DiffuseBaseTreatmentIdentity,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
) -> str:
    return canonical_sha256(
        {
            "format": QUERY_MANIFEST_FORMAT,
            "revision": QUERY_INPUT_REVISION,
            "base_store_key": base_store_key,
            "treatment_identity": treatment_identity.model_dump(mode="json"),
            "query_sample_sha256": canonical_sha256(query_sample_payload(sample)),
            "config_identity": config_identity(config).model_dump(mode="json"),
            "embedding_identity": embedding_identity.model_dump(mode="json"),
        }
    )


def coerce_treatment_identity(
    value: DiffuseBaseTreatmentIdentity | Mapping[str, object],
) -> DiffuseBaseTreatmentIdentity:
    identity_value = (
        value
        if isinstance(value, DiffuseBaseTreatmentIdentity)
        else DiffuseBaseTreatmentIdentity.model_validate(value)
    )
    if identity_value.sample_ordinal >= identity_value.sample_count:
        raise ValueError("sample_ordinal must be within treatment sample_count")
    return identity_value


def coerce_embedding_identity(
    value: DiffuseBaseEmbeddingIdentity | Mapping[str, object],
) -> DiffuseBaseEmbeddingIdentity:
    identity_value = (
        value
        if isinstance(value, DiffuseBaseEmbeddingIdentity)
        else DiffuseBaseEmbeddingIdentity.model_validate(value)
    )
    normalized = identity_value.model_copy(
        update={
            "backend": identity_value.backend.strip(),
            "model_id": identity_value.model_id.strip(),
            "model_revision": identity_value.model_revision.strip(),
            "checkpoint_sha256": identity_value.checkpoint_sha256.casefold(),
            "device": identity_value.device.strip().casefold(),
            "output_dtype": identity_value.output_dtype.strip().casefold(),
        }
    )
    if normalized.output_dtype != "float32":
        raise ValueError("diffuse base embeddings must use float32 output")
    return normalized


def coerce_build_runtime_identity(
    value: DiffuseBaseBuildRuntimeIdentity | Mapping[str, object],
) -> DiffuseBaseBuildRuntimeIdentity:
    return (
        value
        if isinstance(value, DiffuseBaseBuildRuntimeIdentity)
        else DiffuseBaseBuildRuntimeIdentity.model_validate(value)
    )


def validate_live_embedder(
    embedder: object,
    expected: DiffuseBaseEmbeddingIdentity,
) -> None:
    execution = getattr(embedder, "execution_identity", None)
    if callable(execution):
        execution = execution()
    if not isinstance(execution, Mapping):
        raise TypeError("diffuse base embedder needs execution_identity")
    actual = {
        "backend": str(execution.get("backend", "")).strip(),
        "model_id": str(getattr(embedder, "model_name", "")).strip(),
        "model_revision": str(
            getattr(embedder, "model_revision", "") or ""
        ).strip(),
        "checkpoint_sha256": str(
            getattr(embedder, "checkpoint_sha256", "") or ""
        ).strip().casefold(),
        "dimension": int(getattr(embedder, "dim")),
        "device": str(execution.get("device", "")).strip().casefold(),
        "batch_size": int(execution.get("batch_size", 0)),
        "normalize_embeddings": execution.get("normalize_embeddings"),
        "output_dtype": str(execution.get("output_dtype", ""))
        .strip()
        .casefold(),
    }
    if actual != expected.model_dump(mode="json"):
        raise ValueError("live embedder does not match its claimed identity")


def require_exact_children(path: Path, expected: set[str], label: str) -> None:
    try:
        actual = {item.name for item in path.iterdir()}
    except OSError as exc:
        raise DiffuseBaseArtifactError(f"cannot inspect {label}: {path}") from exc
    if actual != expected:
        raise DiffuseBaseArtifactError(
            f"{label} has unexpected or missing files: {sorted(actual ^ expected)}"
        )


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DiffuseBaseArtifactError(f"{label} must be a regular file: {path}")


def require_regular_directory(path: Path, label: str) -> None:
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction() or not path.is_dir():
        raise DiffuseBaseArtifactError(f"{label} must be a regular directory: {path}")


def require_no_sqlite_sidecars(store_path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = store_path / f"{DATABASE_NAME}{suffix}"
        if sidecar.exists():
            raise DiffuseBaseArtifactError(
                f"immutable SQLite store has a sidecar: {sidecar}"
            )


def write_new_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def safe_remove_staging(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    resolved_parent = parent.resolve()
    if resolved.parent != resolved_parent or not resolved.name.startswith("."):
        raise RuntimeError("refusing to remove an unexpected staging directory")
    shutil.rmtree(resolved, ignore_errors=True)


@contextmanager
def _publication_lock(target: Path):
    """Serialize conforming publishers with an OS-released crash-safe lock."""

    lock_path = target.parent / f".{target.name}.publish.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def publish_complete_directory(
    temporary: Path,
    target: Path,
    *,
    manifest_name: str,
) -> None:
    """Exclusively publish children with the manifest as validity marker.

    A process exception rolls back its owned target. An uncatchable crash can
    leave a partial target, which is deliberately fail-closed and requires
    operator cleanup or a fresh cache root; it is never inferred to be a hit.
    """

    expected_names = {child.name for child in temporary.iterdir()}
    if manifest_name not in expected_names:
        raise DiffuseBaseArtifactError("publication staging has no manifest")
    with _publication_lock(target):
        if target.exists():
            raise FileExistsError(target)
        target.mkdir()
        try:
            for child in temporary.iterdir():
                if child.name != manifest_name:
                    child.replace(target / child.name)
            (temporary / manifest_name).replace(target / manifest_name)
            temporary.rmdir()
        except BaseException:
            # This process exclusively created the exact target while holding
            # the publication lock, so rollback cannot remove another writer.
            shutil.rmtree(target, ignore_errors=True)
            raise


def finite_json_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DiffuseBaseArtifactError(f"{label} must be a JSON mapping")
    normalized = dict(value)

    def inspect(item: object) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise DiffuseBaseArtifactError(f"{label} contains a non-finite float")
            return
        if isinstance(item, list):
            for child in item:
                inspect(child)
            return
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise DiffuseBaseArtifactError(f"{label} has a non-string key")
            for child in item.values():
                inspect(child)
            return
        raise DiffuseBaseArtifactError(f"{label} contains a non-JSON scalar")

    inspect(normalized)
    return normalized


def active_digests(
    implementation_digest: str | None,
    environment_digest: str | None,
) -> tuple[str, str]:
    return (
        require_sha256(
            implementation_digest or implementation_sha256(),
            "implementation_digest",
        ),
        require_sha256(
            environment_digest or environment_lock_sha256(),
            "environment_digest",
        ),
    )
