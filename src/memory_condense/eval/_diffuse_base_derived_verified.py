"""Held, one-snapshot verification of finalized derived stores."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import sqlite3
from typing import Callable, Iterator

from memory_condense.domain.discourse import DiscourseSnapshot
from memory_condense.eval._diffuse_base_contracts import (
    DATABASE_NAME,
    STORE_DIRECTORY_NAME,
    DiffuseBaseArtifactError,
    DiffuseDerivedFinalization,
    DiffuseDerivedStore,
    model_bytes,
    self_sha256,
)
from memory_condense.eval._diffuse_base_derived_lifecycle_filesystem import (
    derived_lifecycle_operation_guard,
)
from memory_condense.eval._diffuse_base_derived_phase import (
    validated_finalization_phase,
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
    HeldFinalizedDerivedSnapshot,
    held_finalized_derived_snapshot,
    held_regular_file_snapshot,
)
from memory_condense.eval._diffuse_base_publication_guard import (
    freeze_callable_guard,
    freeze_namespace_guard,
)
from memory_condense.eval._diffuse_base_sqlite_wal_image import (
    sqlite_wal_image_operation_guard,
)
from memory_condense.eval._diffuse_route_v2_validation import (
    assert_current_identity as _assert_current_identity,
)
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore


@dataclass(frozen=True, slots=True)
class HeldVerifiedDerivedStore:
    finalization: DiffuseDerivedFinalization
    database: Database
    store: DiscourseStore


def _assert_database_graph(
    database: Database,
    connection: sqlite3.Connection,
    path,
) -> None:
    if (
        type(database) is not Database
        or object.__getattribute__(database, "_conn") is not connection
        or object.__getattribute__(database, "_path") != path
        or object.__getattribute__(database, "_read_only") is not True
    ):
        raise DiffuseBaseArtifactError("held replay database graph changed")


def _assert_result_graph(
    result: HeldVerifiedDerivedStore,
    result_type: type[HeldVerifiedDerivedStore],
    finalization: DiffuseDerivedFinalization,
    database: Database,
    store: DiscourseStore,
) -> None:
    if (
        type(result) is not result_type
        or object.__getattribute__(result, "finalization") is not finalization
        or object.__getattribute__(result, "database") is not database
        or object.__getattribute__(result, "store") is not store
        or object.__getattribute__(store, "_db") is not database
    ):
        raise DiffuseBaseArtifactError("held replay result graph changed")


def _assert_origin_matches_clone(clone: DiffuseDerivedStore, origin) -> None:
    base = clone.base
    store = base.store_manifest
    query = base.query_manifest
    if (
        origin != clone.origin
        or origin.base_store_key != store.base_store_key
        or origin.base_artifact_sha256 != store.artifact_sha256
        or origin.base_manifest_sha256 != base.store_manifest_sha256
        or origin.query_input_key != query.query_input_key
        or origin.query_artifact_sha256 != query.artifact_sha256
        or origin.treatment_identity_sha256 != query.treatment_identity_sha256
        or origin.config_identity_sha256 != query.config_identity_sha256
        or origin.embedding_identity_sha256 != store.embedding_identity_sha256
        or origin.source_streams_sha256 != store.source_streams_sha256
        or origin.turn_sequence_sha256 != store.turn_sequence_sha256
        or origin.chunk_sequence_sha256 != store.chunk_sequence_sha256
        or origin.query_set_sha256 != query.query_set_sha256
        or origin.initial_database_sha256 != store.database_sha256
        or origin.initial_index_sha256 != store.index_sha256
    ):
        raise DiffuseBaseArtifactError("finalized derived origin changed")


def _expected_finalization(
    clone: DiffuseDerivedStore,
    files: HeldFinalizedDerivedSnapshot,
    snapshot: DiscourseSnapshot,
    *,
    phase: object | None,
    expected: DiffuseDerivedFinalization | None,
) -> DiffuseDerivedFinalization:
    if phase is not None:
        phase = validated_finalization_phase(clone, phase)
        if snapshot != phase.compilation.final_snapshot:
            raise DiffuseBaseArtifactError(
                "finalized database snapshot differs from phase"
            )
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
        return DiffuseDerivedFinalization(
            **unsigned,
            receipt_sha256=canonical_sha256(unsigned),
        )
    if type(expected) is not DiffuseDerivedFinalization:
        raise TypeError("expected_finalization must be exact")
    if expected.receipt_sha256 != self_sha256(expected, "receipt_sha256"):
        raise DiffuseBaseArtifactError("expected finalization receipt changed")
    if expected.final_snapshot_sha256 != snapshot.snapshot_sha256:
        raise DiffuseBaseArtifactError("finalization snapshot differs from expected")
    return expected


def _assert_persisted_finalization(
    clone: DiffuseDerivedStore,
    files: HeldFinalizedDerivedSnapshot,
    observed: DiffuseDerivedFinalization,
    expected: DiffuseDerivedFinalization,
) -> None:
    if (
        files.finalization_bytes != model_bytes(observed)
        or observed.receipt_sha256
        != self_sha256(observed, "receipt_sha256")
        or observed != expected
        or expected.origin_receipt_sha256 != clone.origin.receipt_sha256
        or expected.arm_id != clone.origin.arm_id
        or expected.arm_sha256 != clone.origin.arm_sha256
        or expected.database_sha256 != files.database_sha256
        or expected.database_bytes != files.database_size
        or expected.index_sha256 != files.index_sha256
        or expected.index_bytes != files.index_size
    ):
        raise DiffuseBaseArtifactError(
            "persisted finalization differs from held store"
        )


def _audit_databases(
    base_database: Database,
    derived_database: Database,
    *,
    expected_snapshot: DiscourseSnapshot,
) -> tuple[DiscourseSnapshot, DiscourseStore]:
    initial_schema = _schema_rows(base_database)
    initial_source_sha256 = _source_tables_sha256(base_database)
    connection = derived_database.connection
    execute = sqlite3.Connection.execute
    page_size = int(execute(connection, "PRAGMA page_size").fetchone()[0])
    maximum_pages = int(
        execute(connection, "PRAGMA max_page_count").fetchone()[0]
    )
    _audit_live_database(
        derived_database,
        initial_schema=initial_schema,
        initial_source_sha256=initial_source_sha256,
        expected_page_size=page_size,
        expected_maximum_pages=maximum_pages,
    )
    base_snapshot = DiscourseStore(base_database).snapshot()
    store = DiscourseStore(derived_database)
    observed = store.snapshot()
    source_fields = (
        "max_turn_ordinal",
        "chunk_count",
        "schema_version",
        "source_revision",
        "source_content_sha256",
    )
    if (
        tuple(getattr(observed, name) for name in source_fields)
        != tuple(getattr(base_snapshot, name) for name in source_fields)
        or observed != expected_snapshot
    ):
        raise DiffuseBaseArtifactError("finalized database snapshot changed")
    return observed, store


@contextmanager
def _held_verified_finalized_store(
    clone: DiffuseDerivedStore,
    *,
    expected_finalization: DiffuseDerivedFinalization | None,
    expected_snapshot: object | None,
    phase: object | None,
    assert_base_current: Callable[[object], None] | None,
    assert_outer_intact: Callable[[], None] | None,
    _sealed_guard: Callable[[], None],
    _result_type: type[HeldVerifiedDerivedStore],
) -> Iterator[HeldVerifiedDerivedStore]:
    assert_runtime, _runtime_emergency, _resources = derived_runtime_operation_guard()
    assert_lifecycle, _lifecycle_emergency, _registration = (
        derived_lifecycle_operation_guard()
    )
    assert_codec = sqlite_wal_image_operation_guard()
    raw_close = sqlite3.Connection.close

    def assert_intact() -> None:
        _sealed_guard()
        assert_runtime()
        assert_lifecycle()
        assert_codec()
        if assert_outer_intact is not None:
            assert_outer_intact()

    assert_intact()
    if type(clone) is not DiffuseDerivedStore:
        raise TypeError("clone must be an exact DiffuseDerivedStore")
    if phase is not None:
        phase = validated_finalization_phase(clone, phase)
        snapshot = phase.compilation.final_snapshot
    else:
        if type(expected_snapshot) is not DiscourseSnapshot:
            raise TypeError("expected_snapshot must be exact")
        snapshot = expected_snapshot
        _assert_current_identity(snapshot, "snapshot_sha256", "expected snapshot")
    if assert_base_current is not None:
        assert_base_current(clone.base)
        assert_intact()

    base_path = clone.base.store_path / STORE_DIRECTORY_NAME / DATABASE_NAME
    with held_regular_file_snapshot(
        base_path,
        maximum_bytes=_MAX_DATABASE_BYTES,
    ) as base_file, held_finalized_derived_snapshot(clone.path) as files:
        assert_intact()
        if (
            base_file.size != clone.base.store_manifest.database_bytes
            or hashlib.sha256(base_file.payload).hexdigest()
            != clone.base.store_manifest.database_sha256
        ):
            raise DiffuseBaseArtifactError("verified base database changed")
        origin = _origin_from_bytes(files.origin_bytes)
        _assert_origin_matches_clone(clone, origin)
        if files.lease_bytes != _lease_bytes(clone):
            raise DiffuseBaseArtifactError("finalized derived lease changed")
        if (
            files.index_sha256 != clone.origin.initial_index_sha256
            or files.database_size < 1
        ):
            raise DiffuseBaseArtifactError("finalized derived file identity changed")

        base_database: Database | None = None
        derived_database: Database | None = None
        base_connection: sqlite3.Connection | None = None
        derived_connection: sqlite3.Connection | None = None
        primary: BaseException | None = None
        try:
            base_database = _database_from_wal_bytes(
                base_file.payload,
                path=base_path,
            )
            base_connection = object.__getattribute__(base_database, "_conn")
            derived_path = clone.path / DATABASE_NAME
            derived_database = _database_from_wal_bytes(
                files.database_bytes,
                path=derived_path,
            )
            derived_connection = object.__getattribute__(derived_database, "_conn")
            _assert_database_graph(base_database, base_connection, base_path)
            _assert_database_graph(
                derived_database,
                derived_connection,
                derived_path,
            )
            observed_snapshot, store = _audit_databases(
                base_database,
                derived_database,
                expected_snapshot=snapshot,
            )
            expected = _expected_finalization(
                clone,
                files,
                observed_snapshot,
                phase=phase,
                expected=expected_finalization,
            )
            try:
                observed = DiffuseDerivedFinalization.model_validate_json(
                    files.finalization_bytes
                )
            except ValueError as exc:
                raise DiffuseBaseArtifactError(
                    "invalid held derived finalization"
                ) from exc
            _assert_persisted_finalization(clone, files, observed, expected)
            assert_intact()
            result = _result_type(expected, derived_database, store)
            _assert_result_graph(
                result,
                _result_type,
                expected,
                derived_database,
                store,
            )
            yield result
            assert_intact()
            _assert_result_graph(
                result,
                _result_type,
                expected,
                derived_database,
                store,
            )
            _assert_database_graph(base_database, base_connection, base_path)
            _assert_database_graph(
                derived_database,
                derived_connection,
                derived_path,
            )
            refreshed_snapshot, _refreshed_store = _audit_databases(
                base_database,
                derived_database,
                expected_snapshot=snapshot,
            )
            assert_intact()
            if assert_base_current is not None:
                assert_base_current(clone.base)
                assert_intact()
            _assert_origin_matches_clone(clone, origin)
            if files.lease_bytes != _lease_bytes(clone):
                raise DiffuseBaseArtifactError(
                    "finalized derived lease changed during replay"
                )
            if phase is None:
                _assert_current_identity(
                    snapshot,
                    "snapshot_sha256",
                    "expected snapshot",
                )
            refreshed_expected = _expected_finalization(
                clone,
                files,
                refreshed_snapshot,
                phase=phase,
                expected=expected_finalization,
            )
            _assert_persisted_finalization(
                clone,
                files,
                observed,
                refreshed_expected,
            )
            if refreshed_expected != expected:
                raise DiffuseBaseArtifactError(
                    "held finalization changed during replay"
                )
            if derived_connection.in_transaction:
                raise DiffuseBaseArtifactError(
                    "held replay verification opened a transaction"
                )
        except BaseException as exc:
            primary = exc
            raise
        finally:
            failure: BaseException | None = None
            for label, connection in (
                ("derived", derived_connection),
                ("base", base_connection),
            ):
                if connection is None:
                    continue
                try:
                    raw_close(connection)
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note(
                            f"{label} held database close also failed: {exc!r}"
                        )
            if failure is not None:
                if primary is not None:
                    primary.add_note(
                        f"held database cleanup also failed: {failure!r}"
                    )
                else:
                    raise failure


def _seal_verified_entrypoint(implementation, namespace_guard):
    assert_implementation = freeze_callable_guard(
        implementation,
        error_type=DiffuseBaseArtifactError,
        label="held finalized verifier implementation",
    )
    final_init = DiffuseDerivedFinalization.__dict__.get("__init__")
    final_validate = getattr(
        DiffuseDerivedFinalization.model_validate_json,
        "__func__",
        None,
    )
    final_dump = getattr(DiffuseDerivedFinalization, "model_dump", None)
    snapshot_identity = getattr(DiscourseSnapshot, "identity_payload", None)
    discourse_init = DiscourseStore.__dict__.get("__init__")
    discourse_snapshot = DiscourseStore.__dict__.get("snapshot")
    result_type = HeldVerifiedDerivedStore
    result_init = result_type.__dict__.get("__init__")
    result_fields = tuple(
        result_type.__dict__.get(name)
        for name in ("finalization", "database", "store")
    )
    expected_callables = (
        final_init,
        final_validate,
        final_dump,
        snapshot_identity,
        discourse_init,
        discourse_snapshot,
        result_init,
    )
    callable_guards = tuple(
        freeze_callable_guard(
            value,
            error_type=DiffuseBaseArtifactError,
            label="held finalized verifier callable",
        )
        for value in expected_callables
    )
    expected_sha256 = hashlib.sha256
    expected_execute = sqlite3.Connection.execute
    expected_close = sqlite3.Connection.close

    def assert_all() -> None:
        namespace_guard()
        validator = getattr(
            getattr(DiffuseDerivedFinalization, "model_validate_json", None),
            "__func__",
            None,
        )
        current = (
            DiffuseDerivedFinalization.__dict__.get("__init__"),
            validator,
            getattr(DiffuseDerivedFinalization, "model_dump", None),
            getattr(DiscourseSnapshot, "identity_payload", None),
            DiscourseStore.__dict__.get("__init__"),
            DiscourseStore.__dict__.get("snapshot"),
            result_type.__dict__.get("__init__"),
        )
        if (
            any(value is not expected for value, expected in zip(current, expected_callables))
            or HeldVerifiedDerivedStore is not result_type
            or tuple(
                result_type.__dict__.get(name)
                for name in ("finalization", "database", "store")
            )
            != result_fields
            or hashlib.sha256 is not expected_sha256
            or sqlite3.Connection.execute is not expected_execute
            or sqlite3.Connection.close is not expected_close
        ):
            raise DiffuseBaseArtifactError("held finalized verifier was rebound")
        for guard, value in zip(callable_guards, current):
            guard(value)

    def held_verified_finalized_store(
        clone: DiffuseDerivedStore,
        *,
        expected_finalization: DiffuseDerivedFinalization | None = None,
        expected_snapshot: object | None = None,
        phase: object | None = None,
        assert_base_current: Callable[[object], None] | None = None,
        assert_outer_intact: Callable[[], None] | None = None,
    ):
        assert_all()
        assert_implementation(implementation)
        return implementation(
            clone,
            expected_finalization=expected_finalization,
            expected_snapshot=expected_snapshot,
            phase=phase,
            assert_base_current=assert_base_current,
            assert_outer_intact=assert_outer_intact,
            _sealed_guard=assert_all,
            _result_type=result_type,
        )

    return held_verified_finalized_store


_VERIFIED_GUARD_EXCLUDES = (
    "_seal_verified_entrypoint",
    "held_verified_finalized_store",
    "_VERIFIED_GUARD_EXCLUDES",
    "_sealed_verified_guard",
)
_sealed_verified_guard = freeze_namespace_guard(
    globals(),
    error_type=DiffuseBaseArtifactError,
    label="held finalized verifier module",
    exclude=_VERIFIED_GUARD_EXCLUDES,
)
held_verified_finalized_store = _seal_verified_entrypoint(
    _held_verified_finalized_store,
    _sealed_verified_guard,
)
del _seal_verified_entrypoint, _sealed_verified_guard


__all__ = ["HeldVerifiedDerivedStore", "held_verified_finalized_store"]
