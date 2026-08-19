"""Exact byte-copy clones and one-shot writable diffuse arm stores."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.eval._diffuse_base_contracts import (
    DATABASE_NAME,
    DERIVED_LEASE_NAME,
    DERIVED_ORIGIN_NAME,
    DERIVED_STORE_FORMAT,
    FROZEN_QUERY_INPUTS_NAME,
    INDEX_NAME,
    QUERY_MANIFEST_NAME,
    STORE_DIRECTORY_NAME,
    STORE_MANIFEST_NAME,
    DiffuseBaseArtifactError,
    DiffuseDerivedOrigin,
    DiffuseDerivedStore,
    VerifiedDiffuseLongMemEvalBase,
    canonical_json_bytes,
    chunker_identity,
    config_identity,
    model_bytes,
    publish_complete_directory,
    require_exact_children,
    require_no_sqlite_sidecars,
    require_regular_directory,
    require_regular_file,
    require_sha256,
    safe_remove_staging,
    self_sha256,
    validate_live_embedder,
    write_new_bytes,
)
from memory_condense.eval._diffuse_base_queries import (
    load_query_manifest,
    verify_query_entry,
)
from memory_condense.eval._diffuse_base_store import (
    load_store_manifest,
    validate_embedder_certification,
)
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.eval.schemas import EvalConfig


def _assert_verified_bundle_current(
    base: VerifiedDiffuseLongMemEvalBase,
) -> None:
    require_regular_directory(base.store_path, "base artifact")
    require_exact_children(
        base.store_path,
        {STORE_MANIFEST_NAME, STORE_DIRECTORY_NAME},
        "base artifact",
    )
    active_store_manifest = load_store_manifest(base.store_path)
    if active_store_manifest != base.store_manifest or file_sha256(
        base.store_path / STORE_MANIFEST_NAME
    ) != base.store_manifest_sha256:
        raise DiffuseBaseArtifactError("verified base manifest changed")
    store = base.store_path / STORE_DIRECTORY_NAME
    require_regular_directory(store, "base store")
    require_exact_children(store, {DATABASE_NAME, INDEX_NAME}, "base store")
    require_no_sqlite_sidecars(store)
    if (
        file_sha256(store / DATABASE_NAME)
        != base.store_manifest.database_sha256
        or file_sha256(store / INDEX_NAME) != base.store_manifest.index_sha256
    ):
        raise DiffuseBaseArtifactError("verified base bytes changed")
    require_regular_directory(base.query_inputs_path, "frozen-query artifact")
    require_exact_children(
        base.query_inputs_path,
        {QUERY_MANIFEST_NAME, FROZEN_QUERY_INPUTS_NAME},
        "frozen-query artifact",
    )
    active_query_manifest = load_query_manifest(base.query_inputs_path)
    if active_query_manifest != base.query_manifest or file_sha256(
        base.query_inputs_path / QUERY_MANIFEST_NAME
    ) != base.query_manifest_sha256:
        raise DiffuseBaseArtifactError("verified query manifest changed")
    pointer_path = base.query_inputs_path / FROZEN_QUERY_INPUTS_NAME
    if (
        file_sha256(pointer_path) != base.query_manifest.frozen_inputs_sha256
        or pointer_path.stat().st_size
        != base.query_manifest.frozen_inputs_bytes
    ):
        raise DiffuseBaseArtifactError("verified frozen pointer bytes changed")


def _derived_origin(
    base: VerifiedDiffuseLongMemEvalBase,
    *,
    arm_id: str,
    arm_sha256: str,
) -> DiffuseDerivedOrigin:
    normalized_arm = str(arm_id).strip()
    if not normalized_arm:
        raise ValueError("arm_id must be non-empty")
    store_manifest, query_manifest = base.store_manifest, base.query_manifest
    origin = DiffuseDerivedOrigin(
        base_store_key=store_manifest.base_store_key,
        base_artifact_sha256=store_manifest.artifact_sha256,
        base_manifest_sha256=base.store_manifest_sha256,
        query_input_key=query_manifest.query_input_key,
        query_artifact_sha256=query_manifest.artifact_sha256,
        treatment_identity_sha256=query_manifest.treatment_identity_sha256,
        config_identity_sha256=query_manifest.config_identity_sha256,
        embedding_identity_sha256=store_manifest.embedding_identity_sha256,
        source_streams_sha256=store_manifest.source_streams_sha256,
        turn_sequence_sha256=store_manifest.turn_sequence_sha256,
        chunk_sequence_sha256=store_manifest.chunk_sequence_sha256,
        query_set_sha256=query_manifest.query_set_sha256,
        arm_id=normalized_arm,
        arm_sha256=require_sha256(arm_sha256, "arm_sha256"),
        initial_database_sha256=store_manifest.database_sha256,
        initial_index_sha256=store_manifest.index_sha256,
        receipt_sha256="0" * 64,
    )
    return origin.model_copy(
        update={"receipt_sha256": self_sha256(origin, "receipt_sha256")}
    )


def _load_derived_origin(path: Path) -> DiffuseDerivedOrigin:
    origin_path = path / DERIVED_ORIGIN_NAME
    require_regular_file(origin_path, "derived origin")
    try:
        payload = origin_path.read_bytes()
        origin = DiffuseDerivedOrigin.model_validate_json(payload)
    except (OSError, ValueError) as exc:
        raise DiffuseBaseArtifactError("invalid derived-store origin") from exc
    if payload != model_bytes(origin):
        raise DiffuseBaseArtifactError("derived origin is not canonical JSON")
    if origin.format != DERIVED_STORE_FORMAT or (
        origin.receipt_sha256 != self_sha256(origin, "receipt_sha256")
    ):
        raise DiffuseBaseArtifactError("derived origin receipt changed")
    return origin


def _assert_not_hardlinked(source: Path, destination: Path) -> None:
    try:
        aliased = os.path.samefile(source, destination)
    except OSError as exc:
        raise DiffuseBaseArtifactError("cannot compare cloned file identities") from exc
    if aliased:
        raise DiffuseBaseArtifactError("derived store must not hardlink base files")


def clone_diffuse_longmemeval_base(
    verified: VerifiedDiffuseLongMemEvalBase,
    destination: str | Path,
    *,
    arm_id: str,
    arm_sha256: str,
) -> DiffuseDerivedStore:
    """Byte-copy a verified base into one no-clobber writable arm store."""

    if not isinstance(verified, VerifiedDiffuseLongMemEvalBase):
        raise TypeError("verified must be a VerifiedDiffuseLongMemEvalBase")
    _assert_verified_bundle_current(verified)
    target = Path(destination)
    resolved_target = target.resolve()
    immutable_trees = (
        verified.store_path.resolve(),
        verified.query_inputs_path.resolve(),
    )
    if any(
        resolved_target == tree or resolved_target.is_relative_to(tree)
        for tree in immutable_trees
    ):
        raise ValueError("derived destination overlaps an immutable artifact tree")
    if target.exists():
        raise FileExistsError(f"derived store destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    require_regular_directory(target.parent, "derived store parent")
    base_store = verified.store_path / STORE_DIRECTORY_NAME
    tracked = (base_store / DATABASE_NAME, base_store / INDEX_NAME)
    before = {
        path: (file_sha256(path), path.stat().st_mtime_ns) for path in tracked
    }
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name or 'derived'}.clone-",
            dir=target.parent,
        )
    )
    try:
        for name in (DATABASE_NAME, INDEX_NAME):
            source, copied = base_store / name, staging / name
            shutil.copyfile(source, copied)
            copied.chmod(stat.S_IMODE(copied.stat().st_mode) | stat.S_IWUSR)
            _assert_not_hardlinked(source, copied)
        if (
            file_sha256(staging / DATABASE_NAME)
            != verified.store_manifest.database_sha256
            or file_sha256(staging / INDEX_NAME)
            != verified.store_manifest.index_sha256
        ):
            raise DiffuseBaseArtifactError("derived byte copy differs from base")
        origin = _derived_origin(
            verified,
            arm_id=arm_id,
            arm_sha256=arm_sha256,
        )
        write_new_bytes(staging / DERIVED_ORIGIN_NAME, model_bytes(origin))
        require_exact_children(
            staging,
            {DATABASE_NAME, INDEX_NAME, DERIVED_ORIGIN_NAME},
            "derived staging store",
        )
        publish_complete_directory(
            staging, target, manifest_name=DERIVED_ORIGIN_NAME
        )
    except BaseException:
        if staging.exists():
            safe_remove_staging(staging, target.parent)
        raise
    after = {
        path: (file_sha256(path), path.stat().st_mtime_ns) for path in tracked
    }
    if before != after:
        raise DiffuseBaseArtifactError("cloning mutated the shared base")
    active_origin = _load_derived_origin(target)
    if active_origin != origin:
        raise DiffuseBaseArtifactError("published derived origin changed")
    for name in (DATABASE_NAME, INDEX_NAME):
        _assert_not_hardlinked(base_store / name, target / name)
    return DiffuseDerivedStore(path=target, origin=origin, base=verified)


def _verify_derived_store(
    clone: DiffuseDerivedStore,
    *,
    config: EvalConfig,
) -> tuple["FrozenLegacyQueryInputs", ...]:
    _assert_verified_bundle_current(clone.base)
    require_regular_directory(clone.path, "derived store")
    require_exact_children(
        clone.path,
        {DATABASE_NAME, INDEX_NAME, DERIVED_ORIGIN_NAME},
        "derived store",
    )
    require_no_sqlite_sidecars(clone.path)
    origin = _load_derived_origin(clone.path)
    if origin != clone.origin:
        raise DiffuseBaseArtifactError("derived store origin changed")
    base = clone.base
    if origin != _derived_origin(
        base, arm_id=origin.arm_id, arm_sha256=origin.arm_sha256
    ):
        raise DiffuseBaseArtifactError("derived origin does not bind this base")
    if (
        chunker_identity(config) != base.store_manifest.chunker_identity
        or config_identity(config) != base.query_manifest.config_identity
    ):
        raise ValueError("derived open config differs from frozen base/query inputs")
    if (
        file_sha256(clone.path / DATABASE_NAME)
        != origin.initial_database_sha256
        or file_sha256(clone.path / INDEX_NAME) != origin.initial_index_sha256
    ):
        raise DiffuseBaseArtifactError("derived store changed before its first open")
    base_store = base.store_path / STORE_DIRECTORY_NAME
    for name in (DATABASE_NAME, INDEX_NAME):
        _assert_not_hardlinked(base_store / name, clone.path / name)
    _query_manifest, rows = verify_query_entry(
        base.query_inputs_path,
        store_artifact_path=base.store_path,
        store_manifest=base.store_manifest,
        treatment_identity=base._treatment_identity,
        sample=base._sample,
        config=config,
        embedding_identity=base._embedding_identity,
        database_path=clone.path / DATABASE_NAME,
    )
    return rows


def open_diffuse_longmemeval_derived_store(
    clone: DiffuseDerivedStore,
    *,
    config: EvalConfig,
    embedder: object,
) -> MemoryCondenser:
    """Claim and open an exact clone once, permanently disabling index saves."""

    if not isinstance(clone, DiffuseDerivedStore):
        raise TypeError("clone must be a DiffuseDerivedStore")
    if not isinstance(config, EvalConfig):
        raise TypeError("config must be an EvalConfig")
    validate_live_embedder(embedder, clone.base._embedding_identity)
    validate_embedder_certification(
        embedder,
        clone.base.store_manifest.build_runtime_identity,
    )
    lease_payload = canonical_json_bytes(
        {
            "format": "memory-condense-longmemeval-derived-open-claim-v1",
            "origin_receipt_sha256": clone.origin.receipt_sha256,
        }
    )
    lease_path = clone.path / DERIVED_LEASE_NAME
    if lease_path.exists():
        require_regular_file(lease_path, "derived open claim")
        if lease_path.read_bytes() != lease_payload:
            raise DiffuseBaseArtifactError("derived open claim is invalid")
        raise DiffuseBaseArtifactError(
            "derived store has already been claimed for writable use"
        )
    rows = _verify_derived_store(clone, config=config)
    try:
        write_new_bytes(lease_path, lease_payload)
    except FileExistsError as exc:
        raise DiffuseBaseArtifactError(
            "derived store has already been claimed for writable use"
        ) from exc
    index_path = clone.path / INDEX_NAME
    index_before = (file_sha256(index_path), index_path.stat().st_mtime_ns)
    condenser: MemoryCondenser | None = None
    try:
        condenser = MemoryCondenser(
            data_dir=clone.path,
            model_name=clone.base._embedding_identity.model_id,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            device=config.embedding_device,
            auto_extract=False,
            embedder=embedder,
            persist_index_on_close=False,
            retriever_max_elements=(
                clone.base.store_manifest.build_runtime_identity.index_max_elements
            ),
            read_only=False,
        )
        runtime = clone.base.store_manifest.build_runtime_identity
        retriever = condenser._retriever  # noqa: SLF001
        # The clone is a one-shot graph-compilation workspace. Loading needs
        # the path, but no public retriever.save() call may rewrite its frozen
        # base index after that load succeeds.
        retriever._index_path = None  # noqa: SLF001
        observable = (
            int(retriever._dim),  # noqa: SLF001
            int(retriever._ef_construction),  # noqa: SLF001
            int(retriever._M),  # noqa: SLF001
            int(retriever._max_elements),  # noqa: SLF001
        )
        expected = (
            runtime.index_dimension,
            runtime.index_ef_construction,
            runtime.index_m,
            runtime.index_max_elements,
        )
        if observable != expected:
            raise DiffuseBaseArtifactError(
                "derived HNSW runtime differs from immutable base"
            )
        if condenser._embedder is not embedder:  # noqa: SLF001
            raise DiffuseBaseArtifactError("derived store replaced the embedder")
        if condenser._persist_index_on_close is not False:  # noqa: SLF001
            raise DiffuseBaseArtifactError("derived index persistence is enabled")
        if retriever._index_path is not None:  # noqa: SLF001
            raise DiffuseBaseArtifactError("derived retriever can persist HNSW")
        if (file_sha256(index_path), index_path.stat().st_mtime_ns) != index_before:
            raise DiffuseBaseArtifactError("opening the derived store rewrote HNSW")
        origin_payload = clone.origin.model_dump(mode="json")
        condenser.diffuse_base_origin_receipt = origin_payload  # type: ignore[attr-defined]
        condenser.diffuse_base_origin_receipt_sha256 = canonical_sha256(  # type: ignore[attr-defined]
            origin_payload
        )
        condenser.frozen_legacy_query_inputs = rows  # type: ignore[attr-defined]
        return condenser
    except BaseException:
        if condenser is not None:
            try:
                condenser.close()
            except Exception:
                pass
        raise
