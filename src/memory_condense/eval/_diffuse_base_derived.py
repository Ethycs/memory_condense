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


def _validated_finalization_phase(clone: DiffuseDerivedStore, phase: object):
    from memory_condense.eval.diffuse_longmemeval_analysis import (
        DiffuseLongMemEvalRetrievalPhase,
    )

    if not isinstance(clone, DiffuseDerivedStore):
        raise TypeError("clone must be a DiffuseDerivedStore")
    if type(phase) is not DiffuseLongMemEvalRetrievalPhase:
        raise TypeError("phase must be an exact diffuse retrieval phase")
    if (
        phase.arm.arm_id != clone.origin.arm_id
        or phase.arm.arm_sha256 != clone.origin.arm_sha256
    ):
        raise DiffuseBaseArtifactError("final phase belongs to another clone arm")
    if (
        phase.corpus_sha256 != clone.base.store_manifest.corpus_sha256
        or any(
            item.receipt.snapshot_sha256
            != phase.compilation.final_snapshot.snapshot_sha256
            for item in phase.questions
        )
    ):
        raise DiffuseBaseArtifactError("final phase does not bind this base snapshot")
    if canonical_sha256(list(phase.deterministic_turn_ids)) != (
        clone.base.store_manifest.deterministic_turn_ids_sha256
    ):
        raise DiffuseBaseArtifactError("final phase changed deterministic ingest")
    expected_probes = tuple(clone.base._sample.questions)
    observed_probes = tuple(item.probe for item in phase.questions)
    if observed_probes != expected_probes or len(observed_probes) != (
        clone.base.query_manifest.query_count
    ):
        raise DiffuseBaseArtifactError("final phase changed the frozen query set")
    if len(clone.base.frozen_query_inputs) != len(phase.questions):
        raise DiffuseBaseArtifactError("frozen query rows are incomplete")
    for frozen, question in zip(
        clone.base.frozen_query_inputs,
        phase.questions,
        strict=True,
    ):
        frozen_identity = frozen.identity_payload(include_receipt=False)
        receipt = question.legacy_inputs.receipt
        if (
            receipt.query_sha256 != frozen_identity["query_sha256"]
            or receipt.retrieval_policy_sha256
            != frozen.retrieval_policy_sha256
            or receipt.anchor_chunk_ids
            != tuple(item.chunk.chunk_id for item in frozen.anchors)
            or question.legacy_inputs.candidates.anchors != frozen.anchors
        ):
            raise DiffuseBaseArtifactError(
                "final phase changed a frozen query or anchor row"
            )
    return phase


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


def finalize_diffuse_longmemeval_derived_store(
    clone: DiffuseDerivedStore,
    *,
    phase: object,
) -> DiffuseDerivedFinalization:
    """Seal one closed clone after retrieval and verify its final database."""

    if (clone.path / DERIVED_FINALIZATION_NAME).exists():
        raise FileExistsError(clone.path / DERIVED_FINALIZATION_NAME)
    finalization = _audit_derived_finalization(
        clone,
        phase=phase,
        finalized=False,
    )
    write_new_bytes(
        clone.path / DERIVED_FINALIZATION_NAME,
        model_bytes(finalization),
    )
    return verify_diffuse_longmemeval_derived_finalization(
        clone,
        phase=phase,
    )


def verify_diffuse_longmemeval_derived_finalization(
    clone: DiffuseDerivedStore,
    *,
    phase: object,
) -> DiffuseDerivedFinalization:
    """Read and re-verify a sealed derived store without mutating it."""

    tracked = tuple(clone.path / name for name in (
        DATABASE_NAME,
        INDEX_NAME,
        DERIVED_ORIGIN_NAME,
        DERIVED_LEASE_NAME,
        DERIVED_FINALIZATION_NAME,
    ))
    before = {path: (file_sha256(path), path.stat().st_mtime_ns) for path in tracked}
    value = _audit_derived_finalization(clone, phase=phase, finalized=True)
    after = {path: (file_sha256(path), path.stat().st_mtime_ns) for path in tracked}
    if before != after:
        raise DiffuseBaseArtifactError("read-only finalization verification mutated files")
    return value


def verify_diffuse_longmemeval_finalized_store(
    clone: DiffuseDerivedStore,
    *,
    expected_finalization: DiffuseDerivedFinalization,
    expected_snapshot: object,
) -> DiffuseDerivedFinalization:
    """Verify a replay clone from persisted receipts, without a live phase."""

    from memory_condense.domain.discourse import DiscourseSnapshot

    if not isinstance(clone, DiffuseDerivedStore):
        raise TypeError("clone must be a DiffuseDerivedStore")
    if type(expected_finalization) is not DiffuseDerivedFinalization:
        raise TypeError("expected_finalization must be exact")
    if type(expected_snapshot) is not DiscourseSnapshot:
        raise TypeError("expected_snapshot must be exact")
    _assert_verified_bundle_current(clone.base)
    require_regular_directory(clone.path, "finalized derived store")
    require_exact_children(
        clone.path,
        {
            DATABASE_NAME,
            INDEX_NAME,
            DERIVED_ORIGIN_NAME,
            DERIVED_LEASE_NAME,
            DERIVED_FINALIZATION_NAME,
        },
        "finalized derived store",
    )
    for name in (
        DATABASE_NAME,
        INDEX_NAME,
        DERIVED_ORIGIN_NAME,
        DERIVED_LEASE_NAME,
        DERIVED_FINALIZATION_NAME,
    ):
        require_regular_file(clone.path / name, f"finalized derived {name}")
    require_no_sqlite_sidecars(clone.path)
    if _load_derived_origin(clone.path) != clone.origin:
        raise DiffuseBaseArtifactError("finalized derived origin changed")
    if _load_derived_finalization(clone.path) != expected_finalization:
        raise DiffuseBaseArtifactError("persisted finalization differs from expected")
    if (
        expected_finalization.origin_receipt_sha256
        != clone.origin.receipt_sha256
        or expected_finalization.arm_id != clone.origin.arm_id
        or expected_finalization.arm_sha256 != clone.origin.arm_sha256
        or expected_finalization.final_snapshot_sha256
        != expected_snapshot.snapshot_sha256
    ):
        raise DiffuseBaseArtifactError("finalization identity differs from clone")
    lease_expected = canonical_json_bytes(
        {
            "format": "memory-condense-longmemeval-derived-open-claim-v1",
            "origin_receipt_sha256": clone.origin.receipt_sha256,
        }
    )
    if (clone.path / DERIVED_LEASE_NAME).read_bytes() != lease_expected:
        raise DiffuseBaseArtifactError("finalized derived lease changed")

    database_path = clone.path / DATABASE_NAME
    index_path = clone.path / INDEX_NAME
    tracked = tuple(clone.path / name for name in (
        DATABASE_NAME,
        INDEX_NAME,
        DERIVED_ORIGIN_NAME,
        DERIVED_LEASE_NAME,
        DERIVED_FINALIZATION_NAME,
    ))
    before = {path: (file_sha256(path), path.stat().st_mtime_ns) for path in tracked}
    if (
        before[database_path][0] != expected_finalization.database_sha256
        or database_path.stat().st_size != expected_finalization.database_bytes
        or before[index_path][0] != expected_finalization.index_sha256
        or index_path.stat().st_size != expected_finalization.index_bytes
        or expected_finalization.index_sha256
        != clone.origin.initial_index_sha256
    ):
        raise DiffuseBaseArtifactError("finalized derived file identity changed")
    base_store = clone.base.store_path / STORE_DIRECTORY_NAME
    for name in (DATABASE_NAME, INDEX_NAME):
        _assert_not_hardlinked(base_store / name, clone.path / name)
    with Database(base_store / DATABASE_NAME, read_only=True) as base_database:
        base_source_tables = _immutable_source_tables_sha256(base_database)
        base_snapshot = DiscourseStore(base_database).snapshot()
    with Database(database_path, read_only=True) as database:
        if database.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise DiffuseBaseArtifactError("finalized database integrity failed")
        if database.execute("PRAGMA foreign_key_check").fetchall():
            raise DiffuseBaseArtifactError("finalized database foreign keys failed")
        derived_source_tables = _immutable_source_tables_sha256(database)
        if any(
            int(database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in _FORBIDDEN_DERIVED_TABLES
        ):
            raise DiffuseBaseArtifactError("finalized store has forbidden state")
        observed_snapshot = DiscourseStore(database).snapshot()
    source_fields = (
        "max_turn_ordinal",
        "chunk_count",
        "schema_version",
        "source_revision",
        "source_content_sha256",
    )
    if (
        derived_source_tables != base_source_tables
        or tuple(getattr(observed_snapshot, name) for name in source_fields)
        != tuple(getattr(base_snapshot, name) for name in source_fields)
        or observed_snapshot != expected_snapshot
    ):
        raise DiffuseBaseArtifactError("finalized database snapshot changed")
    require_no_sqlite_sidecars(clone.path)
    after = {path: (file_sha256(path), path.stat().st_mtime_ns) for path in tracked}
    if before != after:
        raise DiffuseBaseArtifactError("standalone verification mutated files")
    for name in (DATABASE_NAME, INDEX_NAME):
        _assert_not_hardlinked(base_store / name, clone.path / name)
    return expected_finalization
