"""Public facade for shared, gold-blind diffuse LongMemEval artifacts.

Corpus stores and frozen query pointers use separate recipe addresses: probe
or retrieval-policy changes never force another BGE ingest. Derived treatment
arms receive exact byte copies and can claim each copy for writable use once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.eval._diffuse_base_contracts import (
    BASE_STORE_FORMAT,
    DERIVED_LEASE_NAME,
    DERIVED_ORIGIN_NAME,
    FROZEN_QUERY_INPUTS_NAME,
    QUERY_INPUT_FORMAT,
    QUERY_MANIFEST_NAME,
    STORE_DIRECTORY_NAME,
    STORE_MANIFEST_NAME,
    DiffuseBaseArtifactError,
    DiffuseBaseBuildRuntimeIdentity,
    DiffuseBaseEmbeddingIdentity,
    DiffuseBaseStoreManifest,
    DiffuseBaseTreatmentIdentity,
    DiffuseDerivedOrigin,
    DiffuseDerivedStore,
    DiffuseQueryInputManifest,
    VerifiedDiffuseLongMemEvalBase,
    active_digests,
    coerce_build_runtime_identity,
    coerce_embedding_identity,
    coerce_treatment_identity,
    diffuse_base_store_key,
    diffuse_query_input_key,
    require_regular_directory,
    validate_live_embedder,
)
from memory_condense.eval._diffuse_base_derived import (
    clone_diffuse_longmemeval_base,
    open_diffuse_longmemeval_derived_store,
)
from memory_condense.eval._diffuse_base_queries import (
    publish_query_entry,
    verify_query_entry,
)
from memory_condense.eval._diffuse_base_store import (
    callable_build_factory_sha256,
    declared_factory_identity,
    owned_build_runtime_identity,
    publish_store_entry,
    validate_embedder_certification,
    verify_store_entry,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    GoldBlindLongMemEvalSample,
)
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.eval.schemas import EvalConfig


def _verified_bundle(
    *,
    store_path: Path,
    query_path: Path,
    store_manifest: DiffuseBaseStoreManifest,
    query_manifest: DiffuseQueryInputManifest,
    rows: tuple,
    sample: GoldBlindLongMemEvalSample,
    treatment: DiffuseBaseTreatmentIdentity,
    config: EvalConfig,
    embedding: DiffuseBaseEmbeddingIdentity,
) -> VerifiedDiffuseLongMemEvalBase:
    return VerifiedDiffuseLongMemEvalBase(
        store_path=store_path,
        query_inputs_path=query_path,
        store_manifest=store_manifest,
        query_manifest=query_manifest,
        frozen_query_inputs=rows,
        store_manifest_sha256=file_sha256(store_path / STORE_MANIFEST_NAME),
        query_manifest_sha256=file_sha256(query_path / QUERY_MANIFEST_NAME),
        _sample=sample,
        _treatment_identity=treatment,
        _config=config,
        _embedding_identity=embedding,
    )


def publish_diffuse_longmemeval_base(
    cache_root: str | Path,
    *,
    treatment_identity: DiffuseBaseTreatmentIdentity | Mapping[str, object],
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedding_identity: DiffuseBaseEmbeddingIdentity | Mapping[str, object],
    build_runtime_identity: (
        DiffuseBaseBuildRuntimeIdentity | Mapping[str, object]
    ),
    embedder: object,
    condenser_factory: Callable[[Path], MemoryCondenser],
    implementation_digest: str | None = None,
    environment_digest: str | None = None,
) -> VerifiedDiffuseLongMemEvalBase:
    """Publish/reuse one store and one independently addressed query bundle."""

    if not isinstance(sample, GoldBlindLongMemEvalSample):
        raise TypeError("sample must be a GoldBlindLongMemEvalSample")
    if not isinstance(config, EvalConfig):
        raise TypeError("config must be an EvalConfig")
    treatment = coerce_treatment_identity(treatment_identity)
    embedding = coerce_embedding_identity(embedding_identity)
    build_runtime = coerce_build_runtime_identity(build_runtime_identity)
    if build_runtime != declared_factory_identity(condenser_factory):
        raise ValueError("factory declaration differs from build-runtime identity")
    if build_runtime.index_dimension != embedding.dimension:
        raise ValueError("build-runtime and embedding dimensions disagree")
    validate_live_embedder(embedder, embedding)
    validate_embedder_certification(
        embedder,
        build_runtime,
    )
    active_implementation, active_environment = active_digests(
        implementation_digest, environment_digest
    )
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    require_regular_directory(root, "cache root")
    stores_root, queries_root = root / "stores", root / "query-inputs"
    stores_root.mkdir(exist_ok=True)
    queries_root.mkdir(exist_ok=True)
    require_regular_directory(stores_root, "base stores root")
    require_regular_directory(queries_root, "query-inputs root")
    store_path, store_manifest = publish_store_entry(
        stores_root,
        sample=sample,
        config=config,
        embedding_identity=embedding,
        build_runtime_identity=build_runtime,
        embedder=embedder,
        condenser_factory=condenser_factory,
        implementation_digest=active_implementation,
        environment_digest=active_environment,
    )
    query_path, query_manifest, rows = publish_query_entry(
        queries_root,
        store_artifact_path=store_path,
        store_manifest=store_manifest,
        treatment_identity=treatment,
        sample=sample,
        config=config,
        embedder=embedder,
        embedding_identity=embedding,
    )
    return _verified_bundle(
        store_path=store_path,
        query_path=query_path,
        store_manifest=store_manifest,
        query_manifest=query_manifest,
        rows=rows,
        sample=sample,
        treatment=treatment,
        config=config,
        embedding=embedding,
    )


def verify_diffuse_longmemeval_base(
    cache_root: str | Path,
    *,
    treatment_identity: DiffuseBaseTreatmentIdentity | Mapping[str, object],
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedding_identity: DiffuseBaseEmbeddingIdentity | Mapping[str, object],
    build_runtime_identity: (
        DiffuseBaseBuildRuntimeIdentity | Mapping[str, object]
    ),
    implementation_digest: str | None = None,
    environment_digest: str | None = None,
) -> VerifiedDiffuseLongMemEvalBase:
    """Load both recipe addresses through read-only, byte-exact verification."""

    if not isinstance(sample, GoldBlindLongMemEvalSample):
        raise TypeError("sample must be a GoldBlindLongMemEvalSample")
    if not isinstance(config, EvalConfig):
        raise TypeError("config must be an EvalConfig")
    treatment = coerce_treatment_identity(treatment_identity)
    embedding = coerce_embedding_identity(embedding_identity)
    build_runtime = coerce_build_runtime_identity(build_runtime_identity)
    active_implementation, active_environment = active_digests(
        implementation_digest, environment_digest
    )
    root = Path(cache_root)
    require_regular_directory(root, "cache root")
    stores_root, queries_root = root / "stores", root / "query-inputs"
    require_regular_directory(stores_root, "base stores root")
    require_regular_directory(queries_root, "query-inputs root")
    store_key = diffuse_base_store_key(
        sample,
        config,
        embedding_identity=embedding,
        build_runtime_identity=build_runtime,
        implementation_digest=active_implementation,
        environment_digest=active_environment,
    )
    store_path = stores_root / store_key
    store_manifest = verify_store_entry(
        store_path,
        sample=sample,
        config=config,
        embedding_identity=embedding,
        build_runtime_identity=build_runtime,
        implementation_digest=active_implementation,
        environment_digest=active_environment,
    )
    query_key = diffuse_query_input_key(
        base_store_key=store_key,
        treatment_identity=treatment,
        sample=sample,
        config=config,
        embedding_identity=embedding,
    )
    query_path = queries_root / query_key
    query_manifest, rows = verify_query_entry(
        query_path,
        store_artifact_path=store_path,
        store_manifest=store_manifest,
        treatment_identity=treatment,
        sample=sample,
        config=config,
        embedding_identity=embedding,
    )
    return _verified_bundle(
        store_path=store_path,
        query_path=query_path,
        store_manifest=store_manifest,
        query_manifest=query_manifest,
        rows=rows,
        sample=sample,
        treatment=treatment,
        config=config,
        embedding=embedding,
    )


__all__ = [
    "BASE_STORE_FORMAT",
    "DERIVED_LEASE_NAME",
    "DERIVED_ORIGIN_NAME",
    "FROZEN_QUERY_INPUTS_NAME",
    "QUERY_INPUT_FORMAT",
    "QUERY_MANIFEST_NAME",
    "STORE_DIRECTORY_NAME",
    "STORE_MANIFEST_NAME",
    "DiffuseBaseArtifactError",
    "DiffuseBaseBuildRuntimeIdentity",
    "DiffuseBaseEmbeddingIdentity",
    "DiffuseBaseStoreManifest",
    "DiffuseBaseTreatmentIdentity",
    "DiffuseDerivedOrigin",
    "DiffuseDerivedStore",
    "DiffuseQueryInputManifest",
    "VerifiedDiffuseLongMemEvalBase",
    "clone_diffuse_longmemeval_base",
    "callable_build_factory_sha256",
    "diffuse_base_store_key",
    "diffuse_query_input_key",
    "open_diffuse_longmemeval_derived_store",
    "owned_build_runtime_identity",
    "publish_diffuse_longmemeval_base",
    "verify_diffuse_longmemeval_base",
]
