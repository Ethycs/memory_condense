"""Exact byte-copy clones and one-shot writable diffuse arm stores."""

from __future__ import annotations

import os
from pathlib import Path

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.eval._diffuse_base_contracts import (
    DATABASE_NAME,
    DERIVED_FINALIZATION_NAME,
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
    DiffuseDerivedFinalization,
    DiffuseDerivedStore,
    VerifiedDiffuseLongMemEvalBase,
    canonical_json_bytes,
    chunker_identity,
    config_identity,
    model_bytes,
    require_exact_children,
    require_no_sqlite_sidecars,
    require_regular_directory,
    require_regular_file,
    require_sha256,
    self_sha256,
    validate_live_embedder,
    write_new_bytes,
)
from memory_condense.eval._diffuse_base_publication_filesystem import (
    abandon_publication,
    capture_publication,
    commit_publication,
    copy_publication_file,
    create_publication,
    promote_publication,
    publication_operation_guard,
    publication_path,
    rollback_publication,
    validate_publication_lock_marker,
    write_publication_bytes,
)
from memory_condense.eval._diffuse_base_publication_guard import (
    freeze_callable_guard,
    freeze_namespace_guard,
)
from memory_condense.eval._diffuse_base_derived_lifecycle_filesystem import (
    _abort_derived_lifecycle_for_clone,
    derived_lifecycle_operation_guard,
    discard_derived_registration,
    register_derived_publication,
)
from memory_condense.eval._diffuse_base_derived_finalization import (
    _finalize_owned_derived_store,
)
from memory_condense.eval._diffuse_base_derived_phase import (
    validated_finalization_phase as _validated_finalization_phase,
)
from memory_condense.eval._diffuse_base_derived_runtime import (
    _open_owned_derived_store,
)
from memory_condense.eval._diffuse_base_derived_verified import (
    held_verified_finalized_store,
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
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore


_FORBIDDEN_DERIVED_TABLES = (
    "association_artifacts",
    "chunk_cav_signatures",
    "chunk_head_edges",
    "consolidation_access_events",
    "consolidation_edges",
    "consolidation_nodes",
    "hebbian_access_events",
    "hebbian_chunk_edges",
    "hebbian_chunk_nodes",
    "memory_items",
    "memory_provenance",
)


def _assert_verified_bundle_current(
    base: VerifiedDiffuseLongMemEvalBase,
) -> None:
    validate_publication_lock_marker(base.store_path)
    validate_publication_lock_marker(base.query_inputs_path)
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


def _clone_diffuse_longmemeval_base(
    verified: VerifiedDiffuseLongMemEvalBase,
    destination: str | Path,
    *,
    arm_id: str,
    arm_sha256: str,
    _sealed_import_guard=None,
) -> DiffuseDerivedStore:
    """Byte-copy a verified base into one no-clobber writable arm store."""

    create_owned = create_publication
    owned_path = publication_path
    copy_owned = copy_publication_file
    write_owned = write_publication_bytes
    capture_owned = capture_publication
    promote_owned = promote_publication
    commit_owned = commit_publication
    rollback_owned = rollback_publication
    abandon_owned = abandon_publication
    register_lifecycle = register_derived_publication
    discard_lifecycle = discard_derived_registration
    assert_import_seam = _sealed_import_guard
    assert_capability_intact, emergency_abandon = publication_operation_guard()
    (
        assert_lifecycle_intact,
        _emergency_lifecycle,
        emergency_registration,
    ) = derived_lifecycle_operation_guard()
    module_namespace = globals()
    guarded_globals = {
        name: value
        for name, value in module_namespace.items()
        if not name.startswith("__")
    }
    error_type = DiffuseBaseArtifactError

    def assert_operations_intact() -> None:
        assert_import_seam()
        changed = [
            name
            for name, value in guarded_globals.items()
            if module_namespace.get(name) is not value
        ]
        if changed:
            raise error_type(
                "derived publication module was rebound: " + ", ".join(changed)
            )
        assert_capability_intact()
        assert_lifecycle_intact()

    def quarantine(owner: object, original: BaseException) -> None:
        try:
            emergency_abandon(owner)  # type: ignore[arg-type]
        except BaseException as abandon_error:
            original.add_note(f"publication abandon also failed: {abandon_error!r}")

    assert_operations_intact()

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
    owner = create_owned(target, role="derived")
    staging = owned_path(owner)
    owner_live = True
    lifecycle_clone: DiffuseDerivedStore | None = None
    try:
        for name in (DATABASE_NAME, INDEX_NAME):
            source = base_store / name
            copy_owned(owner, source, name)
            assert_operations_intact()
            _assert_not_hardlinked(source, staging / name)
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
        write_owned(owner, DERIVED_ORIGIN_NAME, model_bytes(origin))
        require_exact_children(
            staging,
            {DATABASE_NAME, INDEX_NAME, DERIVED_ORIGIN_NAME},
            "derived staging store",
        )
        capture_owned(owner)
        assert_operations_intact()
        promote_owned(owner)
        assert_operations_intact()
        active_origin = _load_derived_origin(target)
        if active_origin != origin:
            raise DiffuseBaseArtifactError("published derived origin changed")
        for name in (DATABASE_NAME, INDEX_NAME):
            _assert_not_hardlinked(base_store / name, target / name)
        after = {
            path: (file_sha256(path), path.stat().st_mtime_ns)
            for path in tracked
        }
        if before != after:
            raise DiffuseBaseArtifactError("cloning mutated the shared base")
        lifecycle_clone = DiffuseDerivedStore(
            path=target,
            origin=origin,
            base=verified,
        )
        register_lifecycle(lifecycle_clone, owner)
        assert_operations_intact()
        commit_owned(owner)
        owner_live = False
    except BaseException as original:
        if lifecycle_clone is not None:
            try:
                assert_operations_intact()
            except BaseException as guard_error:
                original.add_note(
                    "derived lifecycle operations changed; registration "
                    f"quarantined: {guard_error!r}"
                )
                try:
                    emergency_registration(lifecycle_clone)
                except BaseException as lifecycle_error:
                    original.add_note(
                        "derived lifecycle emergency registration release failed: "
                        f"{lifecycle_error!r}"
                    )
            else:
                try:
                    discard_lifecycle(lifecycle_clone)
                except BaseException as lifecycle_error:
                    original.add_note(
                        "derived lifecycle registration could not be released: "
                        f"{lifecycle_error!r}"
                    )
            lifecycle_clone = None
        if owner_live:
            try:
                assert_operations_intact()
            except BaseException as guard_error:
                original.add_note(
                    f"publication operations changed; bytes quarantined: {guard_error!r}"
                )
                quarantine(owner, original)
            else:
                try:
                    rollback_owned(owner)
                except BaseException as cleanup_error:
                    original.add_note(
                        f"exact rollback refused; bytes quarantined: {cleanup_error!r}"
                    )
                    try:
                        abandon_owned(owner)
                    except BaseException:
                        quarantine(owner, original)
            owner_live = False
        raise
    assert lifecycle_clone is not None
    return lifecycle_clone


def _seal_clone_entrypoint(implementation, import_guard):
    assert_implementation = freeze_callable_guard(
        implementation,
        error_type=DiffuseBaseArtifactError,
        label="derived publication implementation",
    )
    assert_import_guard = freeze_callable_guard(
        import_guard,
        error_type=DiffuseBaseArtifactError,
        label="derived publication guard",
    )

    def clone_diffuse_longmemeval_base(
        verified: VerifiedDiffuseLongMemEvalBase,
        destination: str | Path,
        *,
        arm_id: str,
        arm_sha256: str,
    ) -> DiffuseDerivedStore:
        """Byte-copy a verified base into one no-clobber writable arm store."""

        assert_implementation(implementation)
        assert_import_guard(import_guard)
        import_guard()
        return implementation(
            verified,
            destination,
            arm_id=arm_id,
            arm_sha256=arm_sha256,
            _sealed_import_guard=import_guard,
        )

    return clone_diffuse_longmemeval_base


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


def _open_diffuse_longmemeval_derived_store(
    clone: DiffuseDerivedStore,
    *,
    config: EvalConfig,
    embedder: object,
    _sealed_import_guard=None,
) -> MemoryCondenser:
    """Claim a current-process clone and open its held SQLite image once."""

    if not isinstance(clone, DiffuseDerivedStore):
        raise TypeError("clone must be a DiffuseDerivedStore")
    if not isinstance(config, EvalConfig):
        raise TypeError("config must be an EvalConfig")
    _sealed_import_guard()
    _assert_verified_bundle_current(clone.base)
    if (
        chunker_identity(config) != clone.base.store_manifest.chunker_identity
        or config_identity(config) != clone.base.query_manifest.config_identity
    ):
        raise ValueError("derived open config differs from frozen base/query inputs")
    validate_live_embedder(embedder, clone.base._embedding_identity)
    _sealed_import_guard()
    validate_embedder_certification(
        embedder,
        clone.base.store_manifest.build_runtime_identity,
    )
    _sealed_import_guard()
    return _open_owned_derived_store(
        clone,
        config=config,
        embedder=embedder,
        assert_base_current=_assert_verified_bundle_current,
        expected_origin=_derived_origin,
        assert_outer_intact=_sealed_import_guard,
    )


def _seal_open_entrypoint(implementation, import_guard):
    assert_implementation = freeze_callable_guard(
        implementation,
        error_type=DiffuseBaseArtifactError,
        label="derived open implementation",
    )
    assert_import_guard = freeze_callable_guard(
        import_guard,
        error_type=DiffuseBaseArtifactError,
        label="derived open guard",
    )

    def open_diffuse_longmemeval_derived_store(
        clone: DiffuseDerivedStore,
        *,
        config: EvalConfig,
        embedder: object,
    ) -> MemoryCondenser:
        """Claim a current-process clone and open its held SQLite image once.

        The original image stays visible until close publication begins; a
        successful ``close()`` proves the sealed replacement. A failed close
        quarantines an unfinalizable clone whose database may already be
        partially or fully changed. A crash before publication loses in-memory
        work and leaves the claim fail-closed; reconstructed unfinalized clones
        cannot be opened.
        """

        assert_implementation(implementation)
        assert_import_guard(import_guard)
        import_guard()
        return implementation(
            clone,
            config=config,
            embedder=embedder,
            _sealed_import_guard=import_guard,
        )

    return open_diffuse_longmemeval_derived_store


def _load_derived_finalization(path: Path) -> DiffuseDerivedFinalization:
    marker = path / DERIVED_FINALIZATION_NAME
    require_regular_file(marker, "derived finalization")
    raw = marker.read_bytes()
    try:
        value = DiffuseDerivedFinalization.model_validate_json(raw)
    except Exception as exc:
        raise DiffuseBaseArtifactError("invalid derived finalization") from exc
    if raw != model_bytes(value):
        raise DiffuseBaseArtifactError("derived finalization is not canonical JSON")
    if value.receipt_sha256 != self_sha256(value, "receipt_sha256"):
        raise DiffuseBaseArtifactError("derived finalization receipt changed")
    return value


def _immutable_source_tables_sha256(database: Database) -> str:
    """Bind exact transcript, chunk vectors, lexical rows, and ANN labels."""

    turns = database.execute(
        "SELECT turn_id, role, text, source_id, created_at, ordinal "
        "FROM turns ORDER BY ordinal, turn_id"
    ).fetchall()
    chunks = database.execute(
        "SELECT chunk_id, turn_id, text, start_char, end_char, token_count, "
        "hex(embedding), lexical_weights, hnsw_label, term_count "
        "FROM chunks ORDER BY rowid"
    ).fetchall()
    terms = database.execute(
        "SELECT chunk_id, term, tf FROM chunk_terms ORDER BY chunk_id, term"
    ).fetchall()
    schema = database.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    meta = database.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
    return canonical_sha256(
        {
            "turns": [list(row) for row in turns],
            "chunks": [list(row) for row in chunks],
            "chunk_terms": [list(row) for row in terms],
            "schema": [list(row) for row in schema],
            "meta": [list(row) for row in meta],
        }
    )


def _audit_derived_finalization(
    clone: DiffuseDerivedStore,
    *,
    phase: object,
    finalized: bool,
) -> DiffuseDerivedFinalization:
    phase = _validated_finalization_phase(clone, phase)
    _assert_verified_bundle_current(clone.base)

    require_regular_directory(clone.path, "derived store")
    expected_children = {
        DATABASE_NAME,
        INDEX_NAME,
        DERIVED_ORIGIN_NAME,
        DERIVED_LEASE_NAME,
    }
    if finalized:
        expected_children.add(DERIVED_FINALIZATION_NAME)
    require_exact_children(
        clone.path,
        expected_children,
        "finalized derived store" if finalized else "consumed derived store",
    )
    for name in expected_children:
        require_regular_file(clone.path / name, f"derived store {name}")
    require_no_sqlite_sidecars(clone.path)
    origin = _load_derived_origin(clone.path)
    if origin != clone.origin:
        raise DiffuseBaseArtifactError("derived origin changed before finalization")
    lease_payload = canonical_json_bytes(
        {
            "format": "memory-condense-longmemeval-derived-open-claim-v1",
            "origin_receipt_sha256": clone.origin.receipt_sha256,
        }
    )
    lease_path = clone.path / DERIVED_LEASE_NAME
    require_regular_file(lease_path, "derived open claim")
    if lease_path.read_bytes() != lease_payload:
        raise DiffuseBaseArtifactError("derived open claim changed")

    database_path = clone.path / DATABASE_NAME
    index_path = clone.path / INDEX_NAME
    base_store = clone.base.store_path / STORE_DIRECTORY_NAME
    for name in (DATABASE_NAME, INDEX_NAME):
        _assert_not_hardlinked(base_store / name, clone.path / name)
    if file_sha256(index_path) != clone.origin.initial_index_sha256:
        raise DiffuseBaseArtifactError("derived HNSW changed during compilation")
    base_database_path = base_store / DATABASE_NAME
    with Database(base_database_path, read_only=True) as base_database:
        base_source_tables = _immutable_source_tables_sha256(base_database)
        base_snapshot = DiscourseStore(base_database).snapshot()
    with Database(database_path, read_only=True) as database:
        integrity = database.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise DiffuseBaseArtifactError("derived database integrity check failed")
        if database.execute("PRAGMA foreign_key_check").fetchall():
            raise DiffuseBaseArtifactError("derived database foreign keys failed")
        derived_source_tables = _immutable_source_tables_sha256(database)
        forbidden_counts = {
            table: int(
                database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in _FORBIDDEN_DERIVED_TABLES
        }
        snapshot = DiscourseStore(database).snapshot()
    if any(forbidden_counts.values()):
        raise DiffuseBaseArtifactError(
            "derived store persisted unauthorized memory or learned state"
        )
    if derived_source_tables != base_source_tables:
        raise DiffuseBaseArtifactError(
            "derived transcript, chunk vectors, or ANN labels changed"
        )
    base_source_identity = (
        base_snapshot.max_turn_ordinal,
        base_snapshot.chunk_count,
        base_snapshot.schema_version,
        base_snapshot.source_revision,
        base_snapshot.source_content_sha256,
    )
    derived_source_identity = (
        snapshot.max_turn_ordinal,
        snapshot.chunk_count,
        snapshot.schema_version,
        snapshot.source_revision,
        snapshot.source_content_sha256,
    )
    if derived_source_identity != base_source_identity:
        raise DiffuseBaseArtifactError("derived immutable source snapshot changed")
    if snapshot != phase.compilation.final_snapshot:
        raise DiffuseBaseArtifactError("derived database snapshot differs from phase")
    require_no_sqlite_sidecars(clone.path)

    unsigned: dict[str, object] = {
        "format": "memory-condense-longmemeval-derived-finalization-v1",
        "origin_receipt_sha256": clone.origin.receipt_sha256,
        "arm_id": clone.origin.arm_id,
        "arm_sha256": clone.origin.arm_sha256,
        "compilation_receipt_sha256": phase.compilation.receipt_sha256,
        "retrieval_phase_receipt_sha256": phase.receipt_sha256,
        "final_snapshot_sha256": snapshot.snapshot_sha256,
        "database_sha256": file_sha256(database_path),
        "database_bytes": database_path.stat().st_size,
        "index_sha256": file_sha256(index_path),
        "index_bytes": index_path.stat().st_size,
    }
    expected = DiffuseDerivedFinalization(
        **unsigned,
        receipt_sha256=canonical_sha256(unsigned),
    )
    if finalized:
        observed = _load_derived_finalization(clone.path)
        if observed != expected:
            raise DiffuseBaseArtifactError("derived finalization no longer matches")
    for name in (DATABASE_NAME, INDEX_NAME):
        _assert_not_hardlinked(base_store / name, clone.path / name)
    return expected


def _finalize_diffuse_longmemeval_derived_store(
    clone: DiffuseDerivedStore,
    *,
    phase: object,
    _sealed_import_guard=None,
) -> DiffuseDerivedFinalization:
    """Seal one closed clone after retrieval and verify its final database."""

    if not isinstance(clone, DiffuseDerivedStore):
        raise TypeError("clone must be a DiffuseDerivedStore")
    _sealed_import_guard()
    return _finalize_owned_derived_store(
        clone,
        phase=phase,
        validate_phase=_validated_finalization_phase,
        assert_base_current=_assert_verified_bundle_current,
        assert_outer_intact=_sealed_import_guard,
    )


def _seal_finalize_entrypoint(implementation, import_guard):
    assert_implementation = freeze_callable_guard(
        implementation,
        error_type=DiffuseBaseArtifactError,
        label="derived finalization implementation",
    )
    assert_import_guard = freeze_callable_guard(
        import_guard,
        error_type=DiffuseBaseArtifactError,
        label="derived finalization guard",
    )

    def finalize_diffuse_longmemeval_derived_store(
        clone: DiffuseDerivedStore,
        *,
        phase: object,
    ) -> DiffuseDerivedFinalization:
        """Seal one closed clone after held final self-verification."""

        assert_implementation(implementation)
        assert_import_guard(import_guard)
        import_guard()
        return implementation(
            clone,
            phase=phase,
            _sealed_import_guard=import_guard,
        )

    return finalize_diffuse_longmemeval_derived_store


def _seal_verify_entrypoints(held_verifier, assert_base_current, import_guard):
    assert_held_verifier = freeze_callable_guard(
        held_verifier,
        error_type=DiffuseBaseArtifactError,
        label="held finalized verifier",
    )
    assert_import_guard = freeze_callable_guard(
        import_guard,
        error_type=DiffuseBaseArtifactError,
        label="derived verification guard",
    )
    assert_base_guard = freeze_callable_guard(
        assert_base_current,
        error_type=DiffuseBaseArtifactError,
        label="verified base currentness check",
    )

    def verify_diffuse_longmemeval_derived_finalization(
        clone: DiffuseDerivedStore,
        *,
        phase: object,
    ) -> DiffuseDerivedFinalization:
        """Read and re-verify a sealed derived store without mutating it."""

        assert_held_verifier(held_verifier)
        assert_import_guard(import_guard)
        assert_base_guard(assert_base_current)
        import_guard()
        with held_verifier(
            clone,
            phase=phase,
            assert_base_current=assert_base_current,
            assert_outer_intact=import_guard,
        ) as held:
            return held.finalization

    def verify_diffuse_longmemeval_finalized_store(
        clone: DiffuseDerivedStore,
        *,
        expected_finalization: DiffuseDerivedFinalization,
        expected_snapshot: object,
    ) -> DiffuseDerivedFinalization:
        """Verify a replay clone from persisted receipts, without a live phase."""

        assert_held_verifier(held_verifier)
        assert_import_guard(import_guard)
        assert_base_guard(assert_base_current)
        import_guard()
        with held_verifier(
            clone,
            expected_finalization=expected_finalization,
            expected_snapshot=expected_snapshot,
            assert_base_current=assert_base_current,
            assert_outer_intact=import_guard,
        ) as held:
            return held.finalization

    def _held_verified_diffuse_longmemeval_finalized_store(
        clone: DiffuseDerivedStore,
        *,
        expected_finalization: DiffuseDerivedFinalization,
        expected_snapshot: object,
    ):
        assert_held_verifier(held_verifier)
        assert_import_guard(import_guard)
        assert_base_guard(assert_base_current)
        import_guard()
        return held_verifier(
            clone,
            expected_finalization=expected_finalization,
            expected_snapshot=expected_snapshot,
            assert_base_current=assert_base_current,
            assert_outer_intact=import_guard,
        )

    return (
        verify_diffuse_longmemeval_derived_finalization,
        verify_diffuse_longmemeval_finalized_store,
        _held_verified_diffuse_longmemeval_finalized_store,
    )


def _seal_abort_entrypoint(
    abort_for_clone,
    operation_guard,
):
    assert_lifecycle, _emergency_owner, emergency_clone = operation_guard()

    def _abort_diffuse_longmemeval_derived_store(clone: DiffuseDerivedStore) -> None:
        """Best-effort idempotent cleanup for an unpublished derived clone."""

        try:
            assert_lifecycle()
        except BaseException as guard_error:
            try:
                emergency_clone(clone)
            except BaseException as cleanup_error:
                guard_error.add_note(
                    "derived registration emergency cleanup also failed: "
                    f"{cleanup_error!r}"
                )
            raise guard_error
        abort_for_clone(clone)

    return _abort_diffuse_longmemeval_derived_store


_DERIVED_GUARD_EXCLUDES = (
    "_seal_clone_entrypoint",
    "_seal_open_entrypoint",
    "_seal_finalize_entrypoint",
    "_seal_verify_entrypoints",
    "_seal_abort_entrypoint",
    "derived_publication_import_guard",
    "clone_diffuse_longmemeval_base",
    "open_diffuse_longmemeval_derived_store",
    "finalize_diffuse_longmemeval_derived_store",
    "verify_diffuse_longmemeval_derived_finalization",
    "verify_diffuse_longmemeval_finalized_store",
    "_held_verified_diffuse_longmemeval_finalized_store",
    "_abort_diffuse_longmemeval_derived_store",
    "_DERIVED_GUARD_EXCLUDES",
    "_sealed_derived_guard",
)
_sealed_derived_guard = freeze_namespace_guard(
    globals(),
    error_type=DiffuseBaseArtifactError,
    label="derived publication module",
    exclude=_DERIVED_GUARD_EXCLUDES,
)
derived_publication_import_guard = freeze_namespace_guard(
    globals(),
    error_type=DiffuseBaseArtifactError,
    label="derived publication module",
    exclude=_DERIVED_GUARD_EXCLUDES,
)
clone_diffuse_longmemeval_base = _seal_clone_entrypoint(
    _clone_diffuse_longmemeval_base, _sealed_derived_guard
)
open_diffuse_longmemeval_derived_store = _seal_open_entrypoint(
    _open_diffuse_longmemeval_derived_store, _sealed_derived_guard
)
finalize_diffuse_longmemeval_derived_store = _seal_finalize_entrypoint(
    _finalize_diffuse_longmemeval_derived_store, _sealed_derived_guard
)
(
    verify_diffuse_longmemeval_derived_finalization,
    verify_diffuse_longmemeval_finalized_store,
    _held_verified_diffuse_longmemeval_finalized_store,
) = _seal_verify_entrypoints(
    held_verified_finalized_store,
    _assert_verified_bundle_current,
    _sealed_derived_guard,
)
_abort_diffuse_longmemeval_derived_store = _seal_abort_entrypoint(
    _abort_derived_lifecycle_for_clone,
    derived_lifecycle_operation_guard,
)
for _entrypoint, _entrypoint_name in (
    (clone_diffuse_longmemeval_base, "clone_diffuse_longmemeval_base"),
    (
        open_diffuse_longmemeval_derived_store,
        "open_diffuse_longmemeval_derived_store",
    ),
    (
        finalize_diffuse_longmemeval_derived_store,
        "finalize_diffuse_longmemeval_derived_store",
    ),
    (
        verify_diffuse_longmemeval_derived_finalization,
        "verify_diffuse_longmemeval_derived_finalization",
    ),
    (
        verify_diffuse_longmemeval_finalized_store,
        "verify_diffuse_longmemeval_finalized_store",
    ),
):
    _entrypoint.__name__ = _entrypoint_name
    _entrypoint.__qualname__ = _entrypoint_name
    _entrypoint.__module__ = __name__
del _entrypoint, _entrypoint_name
del (
    _seal_clone_entrypoint,
    _seal_open_entrypoint,
    _seal_finalize_entrypoint,
    _seal_verify_entrypoints,
    _seal_abort_entrypoint,
    _sealed_derived_guard,
)
