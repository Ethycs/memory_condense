"""Owned in-memory runtime for one writable diffuse derived store."""

from __future__ import annotations

import sqlite3
import os
import json
from pathlib import Path
from typing import Callable
import weakref

import hnswlib

import memory_condense.application.condenser as _condenser_module
import memory_condense.application.discourse_workflow as _discourse_module
import memory_condense.eval._diffuse_base_contracts as _contracts_module
import memory_condense.eval._diffuse_base_queries as _query_module
import memory_condense.eval._diffuse_base_sqlite_wal_image as _codec_module
import memory_condense.eval.cache_receipts as _receipt_module
import memory_condense.persistence.db as _database_module
import memory_condense.search.indexes.index_lifecycle as _index_module
from memory_condense.application.condenser import (
    MemoryCondenser,
    _bind_owned_close,
)
from memory_condense.eval._diffuse_base_contracts import (
    DATABASE_NAME,
    DiffuseBaseArtifactError,
    DiffuseDerivedOrigin,
    DiffuseDerivedStore,
    canonical_json_bytes,
    chunker_identity,
    config_identity,
    model_bytes,
    self_sha256,
    validate_live_embedder,
)
from memory_condense.eval._diffuse_base_derived_lifecycle_filesystem import (
    _acquire_derived_lifecycle_operation,
    OwnedDerivedLifecycle,
    abort_derived_lifecycle,
    claim_derived_lifecycle,
    close_derived_lifecycle,
    derived_lifecycle_files,
    derived_lifecycle_operation_guard,
    mark_derived_open,
    owned_database_image,
    owned_derived_lifecycle_for_clone,
    owned_index_load_path,
)
from memory_condense.eval._diffuse_base_publication_guard import (
    freeze_callable_guard,
    freeze_namespace_guard,
)
from memory_condense.eval._diffuse_base_queries import (
    verify_query_entry_with_database,
)
from memory_condense.eval._diffuse_base_sqlite_wal_image import (
    deserialize_wal_image,
    serialize_wal_image,
    sqlite_wal_image_operation_guard,
)
from memory_condense.eval._diffuse_base_store import (
    validate_embedder_certification,
)
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.schemas import EvalConfig
from memory_condense.persistence.db import Database
from memory_condense.search.indexes.retrieval import SimilarityRetriever


_MAX_DATABASE_BYTES = 512 * 1024 * 1024
_LEASE_FORMAT = "memory-condense-longmemeval-derived-open-claim-v1"
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


def _lease_bytes(clone: DiffuseDerivedStore) -> bytes:
    return canonical_json_bytes(
        {
            "format": _LEASE_FORMAT,
            "origin_receipt_sha256": clone.origin.receipt_sha256,
        }
    )


def _origin_from_bytes(payload: bytes) -> DiffuseDerivedOrigin:
    try:
        origin = DiffuseDerivedOrigin.model_validate_json(payload)
    except ValueError as exc:
        raise DiffuseBaseArtifactError("invalid held derived-store origin") from exc
    if payload != model_bytes(origin) or (
        origin.receipt_sha256 != self_sha256(origin, "receipt_sha256")
    ):
        raise DiffuseBaseArtifactError("held derived origin receipt changed")
    return origin


def _schema_rows(database: Database) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in sqlite3.Connection.execute(
            database.connection,
            "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_schema "
            "ORDER BY type, name, tbl_name, rootpage",
        ).fetchall()
    )


def _source_tables_sha256(database: Database) -> str:
    connection = database.connection

    def rows(sql: str) -> list[list[object]]:
        return [
            list(row)
            for row in sqlite3.Connection.execute(connection, sql).fetchall()
        ]

    return canonical_sha256(
        {
            "turns": rows(
                "SELECT turn_id, role, text, source_id, created_at, ordinal "
                "FROM turns ORDER BY ordinal, turn_id"
            ),
            "chunks": rows(
                "SELECT chunk_id, turn_id, text, start_char, end_char, "
                "token_count, hex(embedding), lexical_weights, hnsw_label, "
                "term_count FROM chunks ORDER BY rowid"
            ),
            "chunk_terms": rows(
                "SELECT chunk_id, term, tf FROM chunk_terms "
                "ORDER BY chunk_id, term"
            ),
            "schema": [list(row) for row in _schema_rows(database)],
            "meta": rows("SELECT key, value FROM meta ORDER BY key"),
        }
    )


def _audit_live_database(
    database: Database,
    *,
    initial_schema: tuple[tuple[object, ...], ...],
    initial_source_sha256: str,
    expected_page_size: int,
    expected_maximum_pages: int,
) -> None:
    connection = database.connection
    execute = sqlite3.Connection.execute
    if execute(connection, "PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise DiffuseBaseArtifactError("derived database integrity check failed")
    if execute(connection, "PRAGMA foreign_key_check").fetchall():
        raise DiffuseBaseArtifactError("derived database foreign keys failed")
    if int(execute(connection, "PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise DiffuseBaseArtifactError("derived database foreign keys were disabled")
    observed_page_size = int(execute(connection, "PRAGMA page_size").fetchone()[0])
    observed_maximum = int(
        execute(connection, "PRAGMA max_page_count").fetchone()[0]
    )
    observed_pages = int(execute(connection, "PRAGMA page_count").fetchone()[0])
    if (
        observed_page_size != expected_page_size
        or observed_maximum != expected_maximum_pages
        or observed_pages > expected_maximum_pages
    ):
        raise DiffuseBaseArtifactError("derived SQLite resource boundary changed")
    if _schema_rows(database) != initial_schema:
        raise DiffuseBaseArtifactError("derived compilation changed sqlite_schema")
    if _source_tables_sha256(database) != initial_source_sha256:
        raise DiffuseBaseArtifactError("derived immutable source rows changed")
    forbidden = {
        table: int(
            execute(connection, f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        )
        for table in _FORBIDDEN_DERIVED_TABLES
    }
    if any(forbidden.values()):
        raise DiffuseBaseArtifactError(
            "derived store persisted unauthorized memory or learned state"
        )


def _record_failure(
    current: BaseException | None,
    candidate: BaseException,
    label: str,
) -> BaseException:
    if current is None:
        return candidate
    current.add_note(f"{label} also failed: {candidate!r}")
    return current


def _component_graph(
    condenser: MemoryCondenser,
    database: Database,
    retriever: SimilarityRetriever,
    embedder: object,
    chunker_min_tokens: int,
    chunker_max_tokens: int,
    expected_database_path: Path,
) -> tuple[
    weakref.ReferenceType[MemoryCondenser],
    tuple[tuple[str, object], ...],
    tuple[tuple[object, str, object], ...],
]:
    facade_bindings = tuple(
        (name, getattr(condenser, name))
        for name in (
            "_db",
            "_discourse",
            "_transcript",
            "_chunker",
            "_embedder",
            "_associations",
            "_retriever",
            "_memory",
            "_consolidation",
            "_validator",
            "_auto_extract",
        )
    ) + (("_persist_index_on_close", False),)
    if (
        condenser._db is not database  # noqa: SLF001
        or condenser._retriever is not retriever  # noqa: SLF001
        or condenser._embedder is not embedder  # noqa: SLF001
        or type(retriever) is not SimilarityRetriever
        or retriever._index_path is not None  # noqa: SLF001
        or retriever._index is None  # noqa: SLF001
        or condenser._auto_extract is not False  # noqa: SLF001
        or condenser._chunker.min_tokens != chunker_min_tokens  # noqa: SLF001
        or condenser._chunker.max_tokens != chunker_max_tokens  # noqa: SLF001
        or condenser._memory._embedder is not embedder  # noqa: SLF001
        or object.__getattribute__(database, "_path") != expected_database_path
        or object.__getattribute__(database, "_read_only") is not False
    ):
        raise DiffuseBaseArtifactError("derived component graph is malformed")
    component_bindings: list[tuple[object, str, object]] = []
    associations = condenser._associations  # noqa: SLF001
    lexical = retriever._lexical  # noqa: SLF001
    source_hierarchy = retriever._source_hierarchy  # noqa: SLF001
    database_components = (
        condenser._discourse,  # noqa: SLF001
        condenser._transcript,  # noqa: SLF001
        associations,
        retriever,
        condenser._memory,  # noqa: SLF001
        condenser._consolidation,  # noqa: SLF001
        condenser._validator,  # noqa: SLF001
        lexical,
        source_hierarchy,
    )
    if any(getattr(component, "_db", None) is not database for component in database_components):
        raise DiffuseBaseArtifactError("derived database graph is malformed")
    component_bindings.extend(
        (component, "_db", database) for component in database_components
    )
    component_bindings.extend(
        (
            (
                database,
                "_conn",
                object.__getattribute__(database, "_conn"),
            ),
            (
                database,
                "_path",
                object.__getattribute__(database, "_path"),
            ),
            (database, "_read_only", False),
            (retriever, "_associations", associations),
            (retriever, "_lexical", lexical),
            (retriever, "_source_hierarchy", source_hierarchy),
            (retriever, "_index", retriever._index),  # noqa: SLF001
            (retriever, "_index_path", None),
            (retriever, "_dim", retriever._dim),  # noqa: SLF001
            (
                retriever,
                "_ef_construction",
                retriever._ef_construction,  # noqa: SLF001
            ),
            (retriever, "_M", retriever._M),  # noqa: SLF001
            (
                retriever,
                "_max_elements",
                retriever._max_elements,  # noqa: SLF001
            ),
            (condenser._memory, "_embedder", embedder),  # noqa: SLF001
            (
                condenser._chunker,  # noqa: SLF001
                "min_tokens",
                condenser._chunker.min_tokens,  # noqa: SLF001
            ),
            (
                condenser._chunker,  # noqa: SLF001
                "max_tokens",
                condenser._chunker.max_tokens,  # noqa: SLF001
            ),
        )
    )
    for name in _RETRIEVER_CACHE_NAMES:
        value = object.__getattribute__(retriever, name)
        if type(value) is not dict:
            raise DiffuseBaseArtifactError("derived retriever cache is malformed")
        component_bindings.append((retriever, name, value))
    if object.__getattribute__(source_hierarchy, "_built") is not False:
        raise DiffuseBaseArtifactError("derived source hierarchy was prebuilt")
    component_bindings.append((source_hierarchy, "_built", False))
    for name in _SOURCE_CACHE_NAMES:
        value = object.__getattribute__(source_hierarchy, name)
        if type(value) is not dict:
            raise DiffuseBaseArtifactError(
                "derived source hierarchy cache is malformed"
            )
        component_bindings.append((source_hierarchy, name, value))
    return weakref.ref(condenser), facade_bindings, tuple(component_bindings)


def _assert_component_graph(
    graph: tuple[
        weakref.ReferenceType[MemoryCondenser],
        tuple[tuple[str, object], ...],
        tuple[tuple[object, str, object], ...],
    ],
) -> None:
    reference, facade_bindings, component_bindings = graph
    condenser = reference()
    if condenser is None or type(condenser) is not MemoryCondenser:
        raise DiffuseBaseArtifactError("owned condenser was abandoned")
    if any(getattr(condenser, name, None) is not expected for name, expected in facade_bindings):
        raise DiffuseBaseArtifactError("owned condenser component graph changed")
    if any(
        getattr(component, name, object()) is not expected
        for component, name, expected in component_bindings
    ):
        raise DiffuseBaseArtifactError("owned condenser nested graph changed")
    database = condenser._db  # noqa: SLF001
    if any(
        name in database.__dict__
        for name in ("execute", "executemany", "commit", "close", "connection")
    ):
        raise DiffuseBaseArtifactError("owned database methods were shadowed")


_RETRIEVER_CACHE_NAMES = (
    "_span_cache",
    "_span_vector_buffers",
    "_span_tail_sums",
    "_span_tail_tokens",
    "_span_cached_through_rowid",
)
_SOURCE_CACHE_NAMES = ("_nodes", "_vectors", "_parents", "_leaf_keys")


def _release_owned_retriever(retriever: SimilarityRetriever) -> None:
    """Release owned transient state without invoking replaceable callbacks."""

    hierarchy = object.__getattribute__(retriever, "_source_hierarchy")
    caches = tuple(
        object.__getattribute__(retriever, name) for name in _RETRIEVER_CACHE_NAMES
    )
    source_caches = tuple(
        object.__getattribute__(hierarchy, name) for name in _SOURCE_CACHE_NAMES
    )
    if any(type(value) is not dict for value in (*caches, *source_caches)):
        raise DiffuseBaseArtifactError("owned retriever cache graph changed")
    object.__setattr__(retriever, "_index", None)
    for value in caches:
        dict.clear(value)
    object.__setattr__(hierarchy, "_built", False)
    for value in source_caches:
        dict.clear(value)


def _abort_owner(
    owner: OwnedDerivedLifecycle,
    *,
    assert_operations_intact: Callable[[], None],
    emergency_abandon: Callable[[object], Path],
    original: BaseException | None = None,
) -> None:
    try:
        assert_operations_intact()
    except BaseException as guard_error:
        if original is not None:
            original.add_note(
                f"derived runtime operations changed; quarantining: {guard_error!r}"
            )
        emergency_abandon(owner)
    else:
        abort_derived_lifecycle(owner)


def _owned_close_callbacks(
    *,
    clone: DiffuseDerivedStore,
    owner: OwnedDerivedLifecycle,
    database: Database,
    retriever: SimilarityRetriever,
    component_graph: tuple[
        weakref.ReferenceType[MemoryCondenser],
        tuple[tuple[str, object], ...],
        tuple[tuple[object, str, object], ...],
    ],
    original_database: bytes,
    initial_schema: tuple[tuple[object, ...], ...],
    initial_source_sha256: str,
    expected_page_size: int,
    expected_maximum_pages: int,
    assert_operations_intact: Callable[[], None],
    emergency_abandon: Callable[[object], Path],
    emergency_resources: Callable[[object, sqlite3.Connection], None],
) -> tuple[Callable[[], None], Callable[[], None]]:
    # Retaining the exact clone here prevents its lifecycle weak-finalizer from
    # revoking file handles while the condenser is live or closing.
    retained_clone = clone
    connection = database.connection

    def abandon_locked() -> None:
        _ = retained_clone
        try:
            assert_operations_intact()
        except BaseException:
            emergency_resources(retriever, connection)
            try:
                emergency_abandon(owner)
            except BaseException:
                pass
            return
        try:
            _release_owned_retriever(retriever)
        except BaseException:
            pass
        try:
            if connection.in_transaction:
                connection.rollback()
        except BaseException:
            pass
        try:
            sqlite3.Connection.close(connection)
        except BaseException:
            pass
        try:
            abort_derived_lifecycle(owner)
        except BaseException:
            pass

    def close_locked() -> None:
        _ = retained_clone
        try:
            assert_operations_intact()
            _assert_component_graph(component_graph)
        except BaseException as original:
            emergency_resources(retriever, connection)
            try:
                emergency_abandon(owner)
            except BaseException as cleanup_error:
                original.add_note(
                    f"derived emergency abandon also failed: {cleanup_error!r}"
                )
            raise

        failure: BaseException | None = None
        encoded: bytes | None = None
        serialized: bytes | None = None
        try:
            if connection.in_transaction:
                sqlite3.Connection.rollback(connection)
            if connection.in_transaction:
                raise DiffuseBaseArtifactError(
                    "derived database transaction did not roll back"
                )
            _audit_live_database(
                database,
                initial_schema=initial_schema,
                initial_source_sha256=initial_source_sha256,
                expected_page_size=expected_page_size,
                expected_maximum_pages=expected_maximum_pages,
            )
            serialized = sqlite3.Connection.serialize(connection)
            if len(serialized) > _MAX_DATABASE_BYTES:
                raise DiffuseBaseArtifactError(
                    "derived database output exceeds the 512 MiB boundary"
                )
            assert_operations_intact()
            _assert_component_graph(component_graph)
            encoded = serialize_wal_image(original_database, serialized)
            if len(encoded) > _MAX_DATABASE_BYTES:
                raise DiffuseBaseArtifactError(
                    "derived database output exceeds the 512 MiB boundary"
                )
        except BaseException as exc:
            failure = _record_failure(failure, exc, "database serialization")
        try:
            _release_owned_retriever(retriever)
        except BaseException as exc:
            failure = _record_failure(failure, exc, "retriever release")
        try:
            assert_operations_intact()
        except BaseException as exc:
            failure = _record_failure(failure, exc, "post-release guard")
        try:
            sqlite3.Connection.close(connection)
        except BaseException as exc:
            failure = _record_failure(failure, exc, "database close")

        if failure is None:
            assert serialized is not None and encoded is not None
            try:
                assert_operations_intact()
                close_derived_lifecycle(owner, encoded)
                return
            except BaseException as exc:
                failure = exc
        assert failure is not None
        try:
            _abort_owner(
                owner,
                assert_operations_intact=assert_operations_intact,
                emergency_abandon=emergency_abandon,
                original=failure,
            )
        except BaseException as cleanup_error:
            failure.add_note(
                f"derived lifecycle abort also failed: {cleanup_error!r}"
            )
        raise failure

    def close() -> None:
        release = _acquire_derived_lifecycle_operation(owner)
        try:
            close_locked()
        finally:
            release()

    def abandon() -> None:
        try:
            release = _acquire_derived_lifecycle_operation(owner)
        except BaseException:
            return
        try:
            abandon_locked()
        finally:
            release()

    return close, abandon


def _open_owned_derived_store(
    clone: DiffuseDerivedStore,
    *,
    config: EvalConfig,
    embedder: object,
    assert_base_current: Callable[[object], None],
    expected_origin: Callable[..., DiffuseDerivedOrigin],
    assert_outer_intact: Callable[[], None],
) -> MemoryCondenser:
    (
        assert_runtime_intact,
        emergency_abandon,
        emergency_resources,
    ) = derived_runtime_operation_guard()
    assert_lifecycle_intact, lifecycle_emergency, _ = (
        derived_lifecycle_operation_guard()
    )
    assert_codec_intact = sqlite_wal_image_operation_guard()

    def assert_operations_intact() -> None:
        assert_outer_intact()
        assert_runtime_intact()
        assert_lifecycle_intact()
        assert_codec_intact()

    assert_operations_intact()
    lease_payload = _lease_bytes(clone)
    try:
        owned_derived_lifecycle_for_clone(clone)
    except DiffuseBaseArtifactError as lookup_error:
        if str(lookup_error) != "derived clone has no live ownership":
            raise
    else:
        raise DiffuseBaseArtifactError(
            "derived store has already been claimed for writable use"
        )
    try:
        owner = claim_derived_lifecycle(clone, lease_payload)
    except (TypeError, DiffuseBaseArtifactError) as exc:
        if str(exc) != "derived clone has no live ownership":
            raise
        raise DiffuseBaseArtifactError(
            "unfinalized derived store has no current-process ownership"
        ) from exc
    database: Database | None = None
    condenser: MemoryCondenser | None = None
    owned_database: Database | None = None
    owned_retriever: SimilarityRetriever | None = None
    release_operation = _acquire_derived_lifecycle_operation(owner)
    try:
        files = derived_lifecycle_files(owner)
        origin = _origin_from_bytes(files.origin_bytes)
        if origin != clone.origin or origin != expected_origin(
            clone.base,
            arm_id=origin.arm_id,
            arm_sha256=origin.arm_sha256,
        ):
            raise DiffuseBaseArtifactError("held derived origin changed")
        assert_base_current(clone.base)
        if (
            chunker_identity(config) != clone.base.store_manifest.chunker_identity
            or config_identity(config) != clone.base.query_manifest.config_identity
        ):
            raise ValueError("derived open config differs from frozen base/query inputs")
        if (
            files.database_sha256 != origin.initial_database_sha256
            or files.index_sha256 != origin.initial_index_sha256
        ):
            raise DiffuseBaseArtifactError("derived store changed before its first open")

        original_database = owned_database_image(owner)
        rollback_image = deserialize_wal_image(original_database)
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            connection.deserialize(rollback_image)
            connection.execute("PRAGMA foreign_keys=ON")
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            maximum_pages = _MAX_DATABASE_BYTES // page_size
            observed_limit = int(
                connection.execute(
                    f"PRAGMA max_page_count={maximum_pages}"
                ).fetchone()[0]
            )
            if observed_limit != maximum_pages:
                raise DiffuseBaseArtifactError(
                    "derived SQLite max_page_count was not enforced"
                )
            database = Database._from_connection(
                connection,
                path=clone.path / "memory.db",
                read_only=False,
            )
        except BaseException:
            connection.close()
            raise
        initial_schema = _schema_rows(database)
        initial_source_sha256 = _source_tables_sha256(database)
        _query_manifest, rows = verify_query_entry_with_database(
            clone.base.query_inputs_path,
            store_artifact_path=clone.base.store_path,
            store_manifest=clone.base.store_manifest,
            treatment_identity=clone.base._treatment_identity,
            sample=clone.base._sample,
            config=config,
            embedding_identity=clone.base._embedding_identity,
            database=database,
        )
        if connection.in_transaction:
            raise DiffuseBaseArtifactError(
                "derived query verification opened a transaction"
            )
        validate_live_embedder(embedder, clone.base._embedding_identity)
        assert_operations_intact()
        validate_embedder_certification(
            embedder,
            clone.base.store_manifest.build_runtime_identity,
        )
        assert_operations_intact()
        embedding_dimension = int(embedder.dim)  # type: ignore[attr-defined]
        assert_operations_intact()
        if embedding_dimension != (
            clone.base.store_manifest.build_runtime_identity.index_dimension
        ):
            raise DiffuseBaseArtifactError(
                "derived embedder dimension differs from immutable base"
            )
        index_path = owned_index_load_path(owner)
        index_mtime_ns = os.stat(index_path).st_mtime_ns
        condenser = MemoryCondenser._from_owned_database(
            database,
            index_path=index_path,
            model_name=clone.base._embedding_identity.model_id,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            device=config.embedding_device,
            auto_extract=False,
            embedder=embedder,  # type: ignore[arg-type]
            retriever_max_elements=(
                clone.base.store_manifest.build_runtime_identity.index_max_elements
            ),
            owned_embedding_dimension=embedding_dimension,
        )
        owned_database = database
        candidate_retriever = object.__getattribute__(condenser, "_retriever")
        if type(candidate_retriever) is not SimilarityRetriever:
            raise DiffuseBaseArtifactError("derived retriever type changed")
        owned_retriever = candidate_retriever
        database = None
        assert_operations_intact()
        runtime = clone.base.store_manifest.build_runtime_identity
        retriever = owned_retriever
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
        retriever._index_path = None  # noqa: SLF001
        if os.stat(index_path).st_mtime_ns != index_mtime_ns:
            raise DiffuseBaseArtifactError("opening the derived store rewrote HNSW")
        if derived_lifecycle_files(owner).index_sha256 != files.index_sha256:
            raise DiffuseBaseArtifactError("opening the derived store changed HNSW")
        mark_derived_open(owner)
        graph = _component_graph(
            condenser,
            condenser._db,  # noqa: SLF001
            retriever,
            embedder,
            config.chunker.min_tokens,
            config.chunker.max_tokens,
            clone.path / DATABASE_NAME,
        )
        close_op, abandon_op = _owned_close_callbacks(
            clone=clone,
            owner=owner,
            database=condenser._db,  # noqa: SLF001
            retriever=retriever,
            component_graph=graph,
            original_database=original_database,
            initial_schema=initial_schema,
            initial_source_sha256=initial_source_sha256,
            expected_page_size=page_size,
            expected_maximum_pages=maximum_pages,
            assert_operations_intact=assert_operations_intact,
            emergency_abandon=lifecycle_emergency,
            emergency_resources=emergency_resources,
        )
        _bind_owned_close(condenser, close_op, abandon_op)
        assert_operations_intact()
        origin_payload = clone.origin.model_dump(mode="json")
        condenser.diffuse_base_origin_receipt = origin_payload  # type: ignore[attr-defined]
        condenser.diffuse_base_origin_receipt_sha256 = canonical_sha256(  # type: ignore[attr-defined]
            origin_payload
        )
        condenser.frozen_legacy_query_inputs = rows  # type: ignore[attr-defined]
        return condenser
    except BaseException as original:
        if owned_database is not None and owned_retriever is not None:
            emergency_resources(
                owned_retriever,
                object.__getattribute__(owned_database, "_conn"),
            )
        elif database is not None:
            try:
                sqlite3.Connection.close(
                    object.__getattribute__(database, "_conn")
                )
            except BaseException as cleanup_error:
                original.add_note(
                    f"derived database cleanup also failed: {cleanup_error!r}"
                )
        try:
            _abort_owner(
                owner,
                assert_operations_intact=assert_operations_intact,
                emergency_abandon=emergency_abandon,
                original=original,
            )
        except BaseException as cleanup_error:
            original.add_note(
                f"derived owner cleanup also failed: {cleanup_error!r}"
            )
        raise
    finally:
        release_operation()


def _database_from_wal_bytes(payload: bytes, *, path: Path) -> Database:
    if len(payload) > _MAX_DATABASE_BYTES:
        raise DiffuseBaseArtifactError(
            "derived database audit exceeds the 512 MiB boundary"
        )
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        connection.deserialize(deserialize_wal_image(payload))
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA query_only=ON")
        return Database._from_connection(connection, path=path, read_only=True)
    except BaseException:
        sqlite3.Connection.close(connection)
        raise


def _freeze_runtime_guard(namespace):
    module_guards = (
        freeze_namespace_guard(
            _condenser_module.__dict__,
            error_type=DiffuseBaseArtifactError,
            label="owned condenser module",
        ),
        freeze_namespace_guard(
            _contracts_module.__dict__,
            error_type=DiffuseBaseArtifactError,
            label="owned derived contracts module",
        ),
        freeze_namespace_guard(
            _database_module.__dict__,
            error_type=DiffuseBaseArtifactError,
            label="owned database module",
        ),
        freeze_namespace_guard(
            _discourse_module.__dict__,
            error_type=DiffuseBaseArtifactError,
            label="owned discourse workflow module",
        ),
        freeze_namespace_guard(
            _query_module.__dict__,
            error_type=DiffuseBaseArtifactError,
            label="owned query module",
        ),
        freeze_namespace_guard(
            _codec_module.__dict__,
            error_type=DiffuseBaseArtifactError,
            label="owned SQLite codec module",
        ),
        freeze_namespace_guard(
            _receipt_module.__dict__,
            error_type=DiffuseBaseArtifactError,
            label="owned receipt module",
        ),
        freeze_namespace_guard(
            _index_module.__dict__,
            error_type=DiffuseBaseArtifactError,
            label="owned HNSW lifecycle module",
        ),
    )
    def classmethod_function(owner, name):
        return owner.__dict__[name].__func__

    resolvers = (
        (
            "Database._from_connection",
            lambda: classmethod_function(Database, "_from_connection"),
        ),
        (
            "DiffuseDerivedOrigin.__init__",
            lambda: getattr(DiffuseDerivedOrigin, "__init__", None),
        ),
        (
            "DiffuseDerivedOrigin.model_validate_json",
            lambda: getattr(
                DiffuseDerivedOrigin, "model_validate_json"
            ).__func__,
        ),
        (
            "DiffuseDerivedOrigin.model_dump",
            lambda: getattr(DiffuseDerivedOrigin, "model_dump", None),
        ),
        (
            "MemoryCondenser._from_owned_database",
            lambda: classmethod_function(MemoryCondenser, "_from_owned_database"),
        ),
        (
            "MemoryCondenser._initialize_components",
            lambda: MemoryCondenser.__dict__.get("_initialize_components"),
        ),
        (
            "MemoryCondenser._close_unowned",
            lambda: MemoryCondenser.__dict__.get("_close_unowned"),
        ),
        (
            "MemoryCondenser.close",
            lambda: MemoryCondenser.__dict__.get("close"),
        ),
        (
            "DiscourseWorkflowMixin._init_discourse_workflow",
            lambda: _discourse_module.DiscourseWorkflowMixin.__dict__.get(
                "_init_discourse_workflow"
            ),
        ),
        (
            "MemoryCondenser._init_discourse_workflow",
            lambda: getattr(MemoryCondenser, "_init_discourse_workflow", None),
        ),
        (
            "owned close binder",
            lambda: _condenser_module.__dict__.get("_bind_owned_close"),
        ),
        (
            "Database.execute",
            lambda: Database.__dict__.get("execute"),
        ),
        (
            "Database.executemany",
            lambda: Database.__dict__.get("executemany"),
        ),
        ("Database.commit", lambda: Database.__dict__.get("commit")),
        ("Database.close", lambda: Database.__dict__.get("close")),
        (
            "Database.connection",
            lambda: Database.__dict__["connection"].fget,
        ),
        (
            "HNSW constructor",
            lambda: _index_module.IndexLifecycleMixin.__dict__.get("__init__"),
        ),
        (
            "HNSW loader",
            lambda: _index_module.IndexLifecycleMixin.__dict__.get(
                "_load_or_create_index"
            ),
        ),
        (
            "HNSW label loader",
            lambda: _index_module.IndexLifecycleMixin.__dict__.get(
                "_load_label_mapping"
            ),
        ),
        (
            "HNSW release",
            lambda: _index_module.IndexLifecycleMixin.__dict__.get("release"),
        ),
        (
            "lexical index constructor",
            lambda: _index_module.LexicalIndex.__dict__.get("__init__"),
        ),
        (
            "source contraction constructor",
            lambda: _index_module.SourceContractionIndex.__dict__.get("__init__"),
        ),
        (
            "source contraction invalidator",
            lambda: _index_module.SourceContractionIndex.__dict__.get("invalidate"),
        ),
        (
            "SimilarityRetriever.__init__",
            lambda: getattr(SimilarityRetriever, "__init__", None),
        ),
        (
            "SimilarityRetriever.release",
            lambda: getattr(SimilarityRetriever, "release", None),
        ),
        (
            "owned query verifier",
            lambda: _query_module.__dict__.get(
                "verify_query_entry_with_database"
            ),
        ),
        (
            "WAL deserialize",
            lambda: _codec_module.__dict__.get("deserialize_wal_image"),
        ),
        (
            "WAL serialize",
            lambda: _codec_module.__dict__.get("serialize_wal_image"),
        ),
        ("Path.exists", lambda: Path.__dict__.get("exists")),
        ("Path.read_bytes", lambda: Path.__dict__.get("read_bytes")),
        ("Path.stat", lambda: Path.__dict__.get("stat")),
        ("Path.__truediv__", lambda: Path.__dict__.get("__truediv__")),
    )
    callable_guards = tuple(
        (
            resolver,
            value,
            freeze_callable_guard(
                value,
                error_type=DiffuseBaseArtifactError,
                label=label,
            ),
        )
        for label, resolver in resolvers
        for value in (resolver(),)
    )
    component_guards = tuple(
        (
            component,
            getattr(component, "__init__", None),
            freeze_callable_guard(
                getattr(component, "__init__", None),
                error_type=DiffuseBaseArtifactError,
                label=f"{component.__name__} constructor",
            ),
        )
        for component in (
            _condenser_module.AssociationStore,
            _condenser_module.TranscriptStore,
            _condenser_module.Chunker,
            _condenser_module.SimilarityRetriever,
            _condenser_module.MemoryStore,
            _condenser_module.LiveConsolidationStore,
            _condenser_module.Validator,
            _condenser_module.RuleBasedExtractor,
            _condenser_module.ContextPacker,
            _discourse_module.DiscourseStore,
        )
    )
    expected_database_descriptor = Database.__dict__["_from_connection"]
    expected_condenser_descriptor = MemoryCondenser.__dict__["_from_owned_database"]
    expected_hnsw_index = hnswlib.Index
    sqlite_attributes = tuple(
        (owner, name, getattr(owner, name))
        for owner, name in (
            (sqlite3, "connect"),
            (sqlite3, "sqlite_version_info"),
            (sqlite3.Connection, "deserialize"),
            (sqlite3.Connection, "serialize"),
            (sqlite3.Connection, "execute"),
            (sqlite3.Connection, "rollback"),
            (sqlite3.Connection, "close"),
            (sqlite3.Connection, "in_transaction"),
            (hnswlib.Index, "load_index"),
            (hnswlib.Index, "init_index"),
            (weakref, "ref"),
            (weakref, "finalize"),
            (json, "dumps"),
        )
    )
    assert_namespace = freeze_namespace_guard(
        namespace,
        error_type=DiffuseBaseArtifactError,
        label="owned derived runtime",
        exclude=("_freeze_runtime_guard", "derived_runtime_operation_guard"),
    )

    def acquire():
        def assert_intact() -> None:
            assert_namespace()
            for guard in module_guards:
                guard()
            for resolver, expected, guard in callable_guards:
                current = resolver()
                if current is not expected:
                    raise DiffuseBaseArtifactError(
                        "owned derived runtime callable was rebound"
                    )
                guard(current)
            for component, expected, guard in component_guards:
                current = getattr(component, "__init__", None)
                if current is not expected:
                    raise DiffuseBaseArtifactError(
                        "owned derived component constructor was rebound"
                    )
                guard(current)
            if (
                Database.__dict__.get("_from_connection")
                is not expected_database_descriptor
                or MemoryCondenser.__dict__.get("_from_owned_database")
                is not expected_condenser_descriptor
                or hnswlib.Index is not expected_hnsw_index
                or "__init__" in SimilarityRetriever.__dict__
                or "release" in SimilarityRetriever.__dict__
                or "_init_discourse_workflow" in MemoryCondenser.__dict__
                or any(getattr(owner, name, None) != value for owner, name, value in sqlite_attributes)
                or os.stat is not expected_os_stat
            ):
                raise DiffuseBaseArtifactError("owned derived runtime was rebound")

        assert_intact()
        return assert_intact, emergency_abandon, emergency_resources

    _, emergency_abandon, _ = derived_lifecycle_operation_guard()
    expected_os_stat = os.stat
    expected_rollback = sqlite3.Connection.rollback
    expected_close = sqlite3.Connection.close
    expected_in_transaction = sqlite3.Connection.in_transaction

    def emergency_resources(retriever, connection) -> None:
        try:
            object.__setattr__(retriever, "_index", None)
        except BaseException:
            pass
        try:
            if expected_in_transaction.__get__(connection, sqlite3.Connection):
                expected_rollback(connection)
        except BaseException:
            pass
        try:
            expected_close(connection)
        except BaseException:
            pass

    return acquire


derived_runtime_operation_guard = _freeze_runtime_guard(globals())
del _freeze_runtime_guard


__all__ = ["_open_owned_derived_store", "derived_runtime_operation_guard"]
