"""Held final audit and atomic completion for one derived store."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Callable

import memory_condense.persistence.discourse_store as _discourse_store_module

from memory_condense.eval._diffuse_base_contracts import (
    DATABASE_NAME,
    DERIVED_FINALIZATION_NAME,
    STORE_DIRECTORY_NAME,
    DiffuseBaseArtifactError,
    DiffuseDerivedFinalization,
    DiffuseDerivedStore,
    model_bytes,
)
from memory_condense.eval._diffuse_base_derived_lifecycle_filesystem import (
    _acquire_derived_lifecycle_operation,
    OwnedDerivedLifecycle,
    abort_derived_lifecycle,
    commit_derived_lifecycle,
    derived_lifecycle_files,
    derived_lifecycle_operation_guard,
    owned_derived_lifecycle_for_clone,
    write_derived_finalization,
)
from memory_condense.eval._diffuse_base_derived_runtime import (
    _MAX_DATABASE_BYTES,
    _audit_live_database,
    _database_from_wal_bytes,
    _lease_bytes,
    _origin_from_bytes,
    _schema_rows,
    _source_tables_sha256,
    derived_runtime_operation_guard,
)
from memory_condense.eval._diffuse_base_derived_snapshot import (
    held_regular_file_snapshot,
)
from memory_condense.eval._diffuse_base_publication_guard import (
    freeze_callable_guard,
    freeze_namespace_guard,
)
from memory_condense.eval._diffuse_base_sqlite_wal_image import (
    sqlite_wal_image_operation_guard,
)
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.persistence.discourse_store import DiscourseStore


def _audit_owned_finalization(
    clone: DiffuseDerivedStore,
    owner: OwnedDerivedLifecycle,
    *,
    phase: object,
    finalized: bool,
    validate_phase: Callable[[DiffuseDerivedStore, object], object],
    assert_base_current: Callable[[object], None],
    assert_intact: Callable[[], None],
) -> DiffuseDerivedFinalization:
    phase = validate_phase(clone, phase)
    assert_intact()
    assert_base_current(clone.base)
    assert_intact()
    files = derived_lifecycle_files(owner)
    if _origin_from_bytes(files.origin_bytes) != clone.origin:
        raise DiffuseBaseArtifactError("derived origin changed before finalization")
    if files.lease_bytes != _lease_bytes(clone):
        raise DiffuseBaseArtifactError("derived open claim changed")
    if files.index_sha256 != clone.origin.initial_index_sha256:
        raise DiffuseBaseArtifactError("derived HNSW changed during compilation")

    base_path = clone.base.store_path / STORE_DIRECTORY_NAME / DATABASE_NAME
    if clone.base.store_manifest.database_bytes > _MAX_DATABASE_BYTES:
        raise DiffuseBaseArtifactError("base database exceeds the derived audit bound")
    with held_regular_file_snapshot(
        base_path,
        maximum_bytes=_MAX_DATABASE_BYTES,
    ) as base_snapshot_file:
        base_bytes = base_snapshot_file.payload
    if len(base_bytes) != clone.base.store_manifest.database_bytes:
        raise DiffuseBaseArtifactError("base database size changed during final audit")
    if hashlib.sha256(base_bytes).hexdigest() != (
        clone.base.store_manifest.database_sha256
    ):
        raise DiffuseBaseArtifactError("base database changed during final audit")
    base_database = _database_from_wal_bytes(base_bytes, path=base_path)
    derived_database = _database_from_wal_bytes(
        files.database_bytes,
        path=clone.path / DATABASE_NAME,
    )
    try:
        _audit_live_database(
            derived_database,
            initial_schema=_schema_rows(base_database),
            initial_source_sha256=_source_tables_sha256(base_database),
            expected_page_size=int(
                sqlite3.Connection.execute(
                    derived_database.connection, "PRAGMA page_size"
                ).fetchone()[0]
            ),
            expected_maximum_pages=int(
                sqlite3.Connection.execute(
                    derived_database.connection, "PRAGMA max_page_count"
                ).fetchone()[0]
            ),
        )
        assert_intact()
        base_snapshot = DiscourseStore(base_database).snapshot()
        assert_intact()
        snapshot = DiscourseStore(derived_database).snapshot()
        assert_intact()
    except BaseException as original:
        for label, database in (("derived", derived_database), ("base", base_database)):
            try:
                sqlite3.Connection.close(database.connection)
            except BaseException as cleanup_error:
                original.add_note(
                    f"{label} final-audit database close also failed: "
                    f"{cleanup_error!r}"
                )
        raise
    else:
        close_failure: BaseException | None = None
        for label, database in (("derived", derived_database), ("base", base_database)):
            try:
                sqlite3.Connection.close(database.connection)
            except BaseException as cleanup_error:
                if close_failure is None:
                    close_failure = cleanup_error
                else:
                    close_failure.add_note(
                        f"{label} final-audit database close also failed: "
                        f"{cleanup_error!r}"
                    )
        if close_failure is not None:
            raise close_failure
    source_fields = (
        "max_turn_ordinal",
        "chunk_count",
        "schema_version",
        "source_revision",
        "source_content_sha256",
    )
    if tuple(getattr(snapshot, name) for name in source_fields) != tuple(
        getattr(base_snapshot, name) for name in source_fields
    ):
        raise DiffuseBaseArtifactError("derived immutable source snapshot changed")
    if snapshot != phase.compilation.final_snapshot:
        raise DiffuseBaseArtifactError("derived database snapshot differs from phase")
    unsigned: dict[str, object] = {
        "format": "memory-condense-longmemeval-derived-finalization-v1",
        "origin_receipt_sha256": clone.origin.receipt_sha256,
        "arm_id": clone.origin.arm_id,
        "arm_sha256": clone.origin.arm_sha256,
        "compilation_receipt_sha256": phase.compilation.receipt_sha256,
        "retrieval_phase_receipt_sha256": phase.receipt_sha256,
        "final_snapshot_sha256": snapshot.snapshot_sha256,
        "database_sha256": files.database_sha256,
        "database_bytes": files.database_size,
        "index_sha256": files.index_sha256,
        "index_bytes": files.index_size,
    }
    expected = DiffuseDerivedFinalization(
        **unsigned,
        receipt_sha256=canonical_sha256(unsigned),
    )
    if finalized:
        assert_intact()
        raw = files.finalization_bytes
        if raw is None:
            raise DiffuseBaseArtifactError("derived finalization is missing")
        try:
            observed = DiffuseDerivedFinalization.model_validate_json(raw)
        except ValueError as exc:
            raise DiffuseBaseArtifactError("invalid held derived finalization") from exc
        if raw != model_bytes(observed) or observed != expected:
            raise DiffuseBaseArtifactError("held derived finalization changed")
    return expected


def _finalize_owned_derived_store_impl(
    clone: DiffuseDerivedStore,
    *,
    phase: object,
    validate_phase: Callable[[DiffuseDerivedStore, object], object],
    assert_base_current: Callable[[object], None],
    assert_outer_intact: Callable[[], None],
    _sealed_guard: Callable[[], None],
) -> DiffuseDerivedFinalization:
    assert_runtime_intact, _runtime_emergency, _resources = (
        derived_runtime_operation_guard()
    )
    assert_lifecycle_intact, lifecycle_emergency, _registration = (
        derived_lifecycle_operation_guard()
    )
    assert_codec_intact = sqlite_wal_image_operation_guard()

    def assert_operations_intact() -> None:
        assert_outer_intact()
        _sealed_guard()
        assert_runtime_intact()
        assert_lifecycle_intact()
        assert_codec_intact()

    assert_operations_intact()
    final_path = clone.path / DERIVED_FINALIZATION_NAME
    if final_path.exists():
        raise FileExistsError(final_path)
    owner = owned_derived_lifecycle_for_clone(clone)
    release_operation = _acquire_derived_lifecycle_operation(owner)
    try:
        expected = _audit_owned_finalization(
            clone,
            owner,
            phase=phase,
            finalized=False,
            validate_phase=validate_phase,
            assert_base_current=assert_base_current,
            assert_intact=assert_operations_intact,
        )
        assert_operations_intact()
        write_derived_finalization(owner, model_bytes(expected))
        observed = _audit_owned_finalization(
            clone,
            owner,
            phase=phase,
            finalized=True,
            validate_phase=validate_phase,
            assert_base_current=assert_base_current,
            assert_intact=assert_operations_intact,
        )
        if observed != expected:
            raise DiffuseBaseArtifactError("derived final self-verification changed")
        assert_operations_intact()
        commit_derived_lifecycle(owner)
        return expected
    except BaseException as original:
        try:
            assert_lifecycle_intact()
        except BaseException:
            try:
                lifecycle_emergency(owner)
            except BaseException as cleanup_error:
                original.add_note(
                    f"derived final emergency abandon also failed: {cleanup_error!r}"
                )
        else:
            try:
                abort_derived_lifecycle(owner)
            except BaseException as cleanup_error:
                original.add_note(
                    f"derived final abort also failed: {cleanup_error!r}"
                )
        raise
    finally:
        release_operation()


def _seal_finalizer(implementation, guard):
    assert_implementation = freeze_callable_guard(
        implementation,
        error_type=DiffuseBaseArtifactError,
        label="owned derived finalizer implementation",
    )
    def classmethod_function(owner, name):
        value = getattr(owner, name, None)
        return getattr(value, "__func__", None)

    final_init = DiffuseDerivedFinalization.__dict__.get("__init__")
    final_validate = classmethod_function(
        DiffuseDerivedFinalization, "model_validate_json"
    )
    final_dump = getattr(DiffuseDerivedFinalization, "model_dump", None)
    discourse_init = DiscourseStore.__dict__.get("__init__")
    discourse_snapshot = DiscourseStore.__dict__.get("snapshot")
    callable_guards = tuple(
        (
            expected,
            freeze_callable_guard(
                expected,
                error_type=DiffuseBaseArtifactError,
                label=f"derived finalization model {label}",
            ),
        )
        for label, expected in (
            ("constructor", final_init),
            ("JSON validator", final_validate),
            ("dumper", final_dump),
            ("discourse constructor", discourse_init),
            ("discourse snapshot", discourse_snapshot),
        )
    )
    discourse_guard = freeze_namespace_guard(
        _discourse_store_module.__dict__,
        error_type=DiffuseBaseArtifactError,
        label="derived final discourse-store module",
    )

    def assert_all() -> None:
        guard()
        discourse_guard()
        current = (
            DiffuseDerivedFinalization.__dict__.get("__init__"),
            classmethod_function(
                DiffuseDerivedFinalization, "model_validate_json"
            ),
            getattr(DiffuseDerivedFinalization, "model_dump", None),
            DiscourseStore.__dict__.get("__init__"),
            DiscourseStore.__dict__.get("snapshot"),
        )
        expected_values = (
            final_init,
            final_validate,
            final_dump,
            discourse_init,
            discourse_snapshot,
        )
        if any(value is not expected for value, expected in zip(current, expected_values)):
            raise DiffuseBaseArtifactError(
                "derived finalization model methods were rebound"
            )
        for expected, callable_guard in callable_guards:
            callable_guard(expected)

    def finalized(
        clone: DiffuseDerivedStore,
        *,
        phase: object,
        validate_phase: Callable[[DiffuseDerivedStore, object], object],
        assert_base_current: Callable[[object], None],
        assert_outer_intact: Callable[[], None],
    ) -> DiffuseDerivedFinalization:
        assert_all()
        assert_implementation(implementation)
        return implementation(
            clone,
            phase=phase,
            validate_phase=validate_phase,
            assert_base_current=assert_base_current,
            assert_outer_intact=assert_outer_intact,
            _sealed_guard=assert_all,
        )

    return finalized


_FINAL_GUARD_EXCLUDES = (
    "_seal_finalizer",
    "_finalize_owned_derived_store",
    "_FINAL_GUARD_EXCLUDES",
    "_sealed_final_guard",
)
_sealed_final_guard = freeze_namespace_guard(
    globals(),
    error_type=DiffuseBaseArtifactError,
    label="owned derived finalization module",
    exclude=_FINAL_GUARD_EXCLUDES,
)
_finalize_owned_derived_store = _seal_finalizer(
    _finalize_owned_derived_store_impl,
    _sealed_final_guard,
)
del _seal_finalizer, _sealed_final_guard


__all__ = ["_finalize_owned_derived_store"]
