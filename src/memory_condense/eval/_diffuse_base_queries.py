"""Pointer-only frozen legacy query acquisition and DB rehydration."""

from __future__ import annotations

from contextlib import nullcontext
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.discourse import identity_sha256
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval._diffuse_base_contracts import (
    DATABASE_NAME,
    FROZEN_QUERY_INPUTS_NAME,
    QUERY_INPUT_FORMAT,
    QUERY_INPUT_REVISION,
    QUERY_MANIFEST_FORMAT,
    QUERY_MANIFEST_NAME,
    STORE_DIRECTORY_NAME,
    STORE_MANIFEST_NAME,
    DiffuseBaseArtifactError,
    DiffuseBaseEmbeddingIdentity,
    DiffuseBaseStoreManifest,
    DiffuseBaseTreatmentIdentity,
    DiffuseQueryInputManifest,
    FrozenAnchorDiagnostics,
    FrozenAnchorPointer,
    FrozenLexicalSourcePointer,
    FrozenQueryPointer,
    FrozenQueryPointerArtifact,
    config_identity,
    diffuse_query_input_key,
    identity,
    model_bytes,
    query_sample_payload,
    require_exact_children,
    require_no_sqlite_sidecars,
    require_regular_directory,
    require_regular_file,
    self_sha256,
)
from memory_condense.eval._diffuse_base_publication_filesystem import (
    abandon_publication,
    capture_publication,
    commit_publication,
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
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.diffuse_longmemeval_inputs import (
    GoldBlindLongMemEvalSample,
)
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.eval.schemas import EvalConfig
from memory_condense.persistence.db import Database
from memory_condense.search.indexes.retrieval_models import (
    load_chunk_payload,
    load_turn_payload,
)

if TYPE_CHECKING:
    from memory_condense.eval.diffuse_longmemeval_runtime import (
        FrozenLegacyQueryInputs,
    )


def _anchor_pointer(result: RetrievalResult) -> FrozenAnchorPointer:
    if result.turn is None or result.turn.turn_id != result.chunk.turn_id:
        raise DiffuseBaseArtifactError(
            "frozen legacy anchors require an exact hydrated parent turn"
        )
    if result.chunk.embedding is not None:
        raise DiffuseBaseArtifactError(
            "frozen legacy anchors must not retain embedding arrays"
        )
    try:
        diagnostics = FrozenAnchorDiagnostics.model_validate(
            result.model_dump(mode="json", exclude={"chunk", "turn"})
        )
    except ValueError as exc:
        raise DiffuseBaseArtifactError(
            "retrieval diagnostic schema changed"
        ) from exc
    return FrozenAnchorPointer(
        chunk_id=result.chunk.chunk_id,
        diagnostics=diagnostics,
    )


def _pointer_artifact(
    *,
    base_store_key: str,
    sample: GoldBlindLongMemEvalSample,
    frozen: Sequence["FrozenLegacyQueryInputs"],
) -> FrozenQueryPointerArtifact:
    if len(frozen) != len(sample.questions):
        raise DiffuseBaseArtifactError("frozen query rows are not sample-parallel")
    rows: list[FrozenQueryPointer] = []
    for question, item in zip(sample.questions, frozen, strict=True):
        if identity_sha256({"query": item.query}) != identity_sha256(
            {"query": question.retrieval_query}
        ):
            raise DiffuseBaseArtifactError("frozen row belongs to another query")
        anchors = tuple(_anchor_pointer(result) for result in item.anchors)
        if len({anchor.chunk_id for anchor in anchors}) != len(anchors):
            raise DiffuseBaseArtifactError("frozen anchors contain duplicate IDs")
        lexical = tuple(
            FrozenLexicalSourcePointer(source_id=source_id, score=score)
            for source_id, score in item.lexical_sources
        )
        rows.append(
            FrozenQueryPointer(
                question_id=question.question_id,
                question_probe_sha256=question.probe_sha256,
                query_sha256=identity_sha256(
                    {"query": question.retrieval_query}
                ),
                retrieval_policy_sha256=item.retrieval_policy_sha256,
                anchors=anchors,
                lexical_sources=lexical,
                universe_source_ids=item.universe_source_ids,
                source_streams_sha256=item.source_streams_sha256,
                frozen_receipt_sha256=item.receipt_sha256,
            )
        )
    query_set_sha256 = canonical_sha256(
        [
            {
                "question_id": row.question_id,
                "question_probe_sha256": row.question_probe_sha256,
                "frozen_receipt_sha256": row.frozen_receipt_sha256,
            }
            for row in rows
        ]
    )
    return FrozenQueryPointerArtifact(
        base_store_key=base_store_key,
        rows=tuple(rows),
        query_set_sha256=query_set_sha256,
    )


def _manifest_for_query_inputs(
    artifact_path: Path,
    *,
    key: str,
    store_path: Path,
    store_manifest: DiffuseBaseStoreManifest,
    treatment_identity: DiffuseBaseTreatmentIdentity,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    pointer_artifact: FrozenQueryPointerArtifact,
) -> DiffuseQueryInputManifest:
    pointer_path = artifact_path / FROZEN_QUERY_INPUTS_NAME
    config_value = config_identity(config)
    manifest = DiffuseQueryInputManifest(
        query_input_key=key,
        base_store_key=store_manifest.base_store_key,
        base_artifact_sha256=store_manifest.artifact_sha256,
        base_manifest_sha256=file_sha256(store_path / STORE_MANIFEST_NAME),
        database_sha256=store_manifest.database_sha256,
        index_sha256=store_manifest.index_sha256,
        source_streams_sha256=store_manifest.source_streams_sha256,
        turn_sequence_sha256=store_manifest.turn_sequence_sha256,
        chunk_sequence_sha256=store_manifest.chunk_sequence_sha256,
        treatment_identity=treatment_identity,
        treatment_identity_sha256=identity(treatment_identity),
        query_sample_sha256=canonical_sha256(query_sample_payload(sample)),
        question_ids_sha256=canonical_sha256(
            [question.question_id for question in sample.questions]
        ),
        question_probes_sha256=canonical_sha256(
            [question.probe_sha256 for question in sample.questions]
        ),
        config_identity=config_value,
        config_identity_sha256=identity(config_value),
        embedding_identity_sha256=identity(embedding_identity),
        frozen_inputs_sha256=file_sha256(pointer_path),
        frozen_inputs_bytes=pointer_path.stat().st_size,
        query_set_sha256=pointer_artifact.query_set_sha256,
        frozen_receipts_sha256=canonical_sha256(
            [row.frozen_receipt_sha256 for row in pointer_artifact.rows]
        ),
        query_count=len(pointer_artifact.rows),
        artifact_sha256="0" * 64,
    )
    return manifest.model_copy(
        update={"artifact_sha256": self_sha256(manifest, "artifact_sha256")}
    )


def load_query_manifest(path: Path) -> DiffuseQueryInputManifest:
    require_regular_directory(path, "frozen-query artifact")
    manifest_path = path / QUERY_MANIFEST_NAME
    require_regular_file(manifest_path, "query manifest")
    try:
        payload = manifest_path.read_bytes()
        manifest = DiffuseQueryInputManifest.model_validate_json(payload)
    except (OSError, ValueError) as exc:
        raise DiffuseBaseArtifactError(
            f"invalid frozen-query manifest: {manifest_path}"
        ) from exc
    if payload != model_bytes(manifest):
        raise DiffuseBaseArtifactError("query manifest is not canonical JSON")
    if (
        manifest.format != QUERY_MANIFEST_FORMAT
        or manifest.revision != QUERY_INPUT_REVISION
    ):
        raise DiffuseBaseArtifactError("unsupported frozen-query manifest")
    if manifest.artifact_sha256 != self_sha256(manifest, "artifact_sha256"):
        raise DiffuseBaseArtifactError("query manifest self-receipt changed")
    return manifest


def _assert_pointer_structure(pointer_payload: object) -> None:
    forbidden = {
        "query",
        "prompt_question",
        "text",
        "chunk",
        "turn",
        "embedding",
        "lexical_weights",
    }

    def inspect(value: object) -> None:
        if isinstance(value, Mapping):
            if set(value) & forbidden:
                raise DiffuseBaseArtifactError(
                    "frozen pointer schema contains a forbidden payload field"
                )
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(pointer_payload)


def _rehydrate_pointer_rows(
    *,
    pointer: FrozenQueryPointerArtifact,
    database_path: Path | None,
    database: Database | None,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    store_manifest: DiffuseBaseStoreManifest,
) -> tuple["FrozenLegacyQueryInputs", ...]:
    from memory_condense.eval.diffuse_longmemeval_runtime import (
        FrozenLegacyQueryInputs,
    )

    if len(pointer.rows) != len(sample.questions):
        raise DiffuseBaseArtifactError("frozen query count differs from sample")
    policy_sha256 = identity_sha256(config.retrieval.model_dump(mode="json"))
    rehydrated: list[FrozenLegacyQueryInputs] = []
    duplicate_queries: dict[str, str] = {}
    if (database_path is None) == (database is None):
        raise TypeError("query rehydration requires exactly one database authority")
    manager = (
        nullcontext(database)
        if database is not None
        else Database(database_path, read_only=True)
    )
    with manager as db:
        for row, question in zip(pointer.rows, sample.questions, strict=True):
            if (
                row.question_id != question.question_id
                or row.question_probe_sha256 != question.probe_sha256
                or row.query_sha256
                != identity_sha256({"query": question.retrieval_query})
                or row.retrieval_policy_sha256 != policy_sha256
            ):
                raise DiffuseBaseArtifactError(
                    "frozen query coordinates differ from gold-blind sample"
                )
            if (
                row.universe_source_ids != store_manifest.source_ids
                or row.source_streams_sha256
                != store_manifest.source_streams_sha256
            ):
                raise DiffuseBaseArtifactError(
                    "frozen source universe differs from verified base"
                )
            lexical_ids = [item.source_id for item in row.lexical_sources]
            if len(lexical_ids) != len(set(lexical_ids)) or not set(
                lexical_ids
            ).issubset(row.universe_source_ids):
                raise DiffuseBaseArtifactError(
                    "lexical source rows are duplicated or outside the universe"
                )
            anchors: list[RetrievalResult] = []
            anchor_ids: set[str] = set()
            for anchor in row.anchors:
                if anchor.chunk_id in anchor_ids:
                    raise DiffuseBaseArtifactError("frozen anchor ID is duplicated")
                anchor_ids.add(anchor.chunk_id)
                chunk = load_chunk_payload(db, anchor.chunk_id)
                if chunk is None or chunk.embedding is not None:
                    raise DiffuseBaseArtifactError(
                        "frozen anchor cannot be hydrated from the base DB"
                    )
                turn = load_turn_payload(db, chunk.turn_id)
                if turn is None or turn.turn_id != chunk.turn_id:
                    raise DiffuseBaseArtifactError(
                        "frozen anchor parent turn is missing or wrong"
                    )
                try:
                    anchors.append(
                        RetrievalResult(
                            chunk=chunk,
                            turn=turn,
                            **anchor.diagnostics.model_dump(mode="json"),
                        )
                    )
                except ValueError as exc:
                    raise DiffuseBaseArtifactError(
                        "frozen anchor diagnostics are invalid"
                    ) from exc
            try:
                frozen = FrozenLegacyQueryInputs(
                    query=question.retrieval_query,
                    retrieval_policy_sha256=row.retrieval_policy_sha256,
                    anchors=tuple(anchors),
                    lexical_sources=tuple(
                        (item.source_id, item.score)
                        for item in row.lexical_sources
                    ),
                    universe_source_ids=row.universe_source_ids,
                    source_streams_sha256=row.source_streams_sha256,
                    receipt_sha256=row.frozen_receipt_sha256,
                )
            except (TypeError, ValueError) as exc:
                raise DiffuseBaseArtifactError(
                    "frozen query receipt does not bind rehydrated DB rows"
                ) from exc
            previous = duplicate_queries.setdefault(
                row.query_sha256, frozen.receipt_sha256
            )
            if previous != frozen.receipt_sha256:
                raise DiffuseBaseArtifactError(
                    "duplicate query identity has conflicting frozen rows"
                )
            rehydrated.append(frozen)
    return tuple(rehydrated)


def _verify_query_entry(
    path: Path,
    *,
    store_artifact_path: Path,
    store_manifest: DiffuseBaseStoreManifest,
    treatment_identity: DiffuseBaseTreatmentIdentity,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    database_path: Path | None = None,
    database: Database | None = None,
) -> tuple[
    DiffuseQueryInputManifest,
    tuple["FrozenLegacyQueryInputs", ...],
]:
    require_regular_directory(path, "frozen-query artifact")
    require_exact_children(
        path,
        {QUERY_MANIFEST_NAME, FROZEN_QUERY_INPUTS_NAME},
        "frozen-query artifact",
    )
    manifest = load_query_manifest(path)
    expected_key = diffuse_query_input_key(
        base_store_key=store_manifest.base_store_key,
        treatment_identity=treatment_identity,
        sample=sample,
        config=config,
        embedding_identity=embedding_identity,
    )
    if manifest.query_input_key != expected_key or path.name != expected_key:
        raise DiffuseBaseArtifactError("frozen-query key or directory changed")
    config_value = config_identity(config)
    expected_fields: dict[str, object] = {
        "base_store_key": store_manifest.base_store_key,
        "base_artifact_sha256": store_manifest.artifact_sha256,
        "base_manifest_sha256": file_sha256(
            store_artifact_path / STORE_MANIFEST_NAME
        ),
        "database_sha256": store_manifest.database_sha256,
        "index_sha256": store_manifest.index_sha256,
        "source_streams_sha256": store_manifest.source_streams_sha256,
        "turn_sequence_sha256": store_manifest.turn_sequence_sha256,
        "chunk_sequence_sha256": store_manifest.chunk_sequence_sha256,
        "treatment_identity": treatment_identity,
        "treatment_identity_sha256": identity(treatment_identity),
        "query_sample_sha256": canonical_sha256(query_sample_payload(sample)),
        "question_ids_sha256": canonical_sha256(
            [question.question_id for question in sample.questions]
        ),
        "question_probes_sha256": canonical_sha256(
            [question.probe_sha256 for question in sample.questions]
        ),
        "config_identity": config_value,
        "config_identity_sha256": identity(config_value),
        "embedding_identity_sha256": identity(embedding_identity),
        "query_count": len(sample.questions),
    }
    for name, expected in expected_fields.items():
        if getattr(manifest, name) != expected:
            raise DiffuseBaseArtifactError(f"query manifest changed {name}")
    pointer_path = path / FROZEN_QUERY_INPUTS_NAME
    require_regular_file(pointer_path, "frozen pointer file")
    try:
        pointer_bytes = pointer_path.read_bytes()
        pointer = FrozenQueryPointerArtifact.model_validate_json(pointer_bytes)
    except (OSError, ValueError) as exc:
        raise DiffuseBaseArtifactError("invalid frozen pointer file") from exc
    if pointer_bytes != model_bytes(pointer):
        raise DiffuseBaseArtifactError("frozen pointer file is not canonical JSON")
    if pointer.format != QUERY_INPUT_FORMAT or pointer.revision != QUERY_INPUT_REVISION:
        raise DiffuseBaseArtifactError("unsupported frozen pointer format")
    if pointer.base_store_key != store_manifest.base_store_key:
        raise DiffuseBaseArtifactError("frozen pointers bind another base store")
    if (
        file_sha256(pointer_path) != manifest.frozen_inputs_sha256
        or pointer_path.stat().st_size != manifest.frozen_inputs_bytes
    ):
        raise DiffuseBaseArtifactError("frozen pointer bytes changed")
    _assert_pointer_structure(pointer.model_dump(mode="json"))
    expected_query_set = canonical_sha256(
        [
            {
                "question_id": row.question_id,
                "question_probe_sha256": row.question_probe_sha256,
                "frozen_receipt_sha256": row.frozen_receipt_sha256,
            }
            for row in pointer.rows
        ]
    )
    if (
        pointer.query_set_sha256 != expected_query_set
        or manifest.query_set_sha256 != expected_query_set
        or manifest.frozen_receipts_sha256
        != canonical_sha256(
            [row.frozen_receipt_sha256 for row in pointer.rows]
        )
    ):
        raise DiffuseBaseArtifactError("frozen query-set receipt changed")
    if database is not None and database_path is not None:
        raise TypeError("query verification received two database authorities")
    active_database = (
        database_path
        if database_path is not None
        else store_artifact_path / STORE_DIRECTORY_NAME / DATABASE_NAME
    )
    rows = _rehydrate_pointer_rows(
        pointer=pointer,
        database_path=None if database is not None else active_database,
        database=database,
        sample=sample,
        config=config,
        store_manifest=store_manifest,
    )
    return manifest, rows


def verify_query_entry(
    path: Path,
    *,
    store_artifact_path: Path,
    store_manifest: DiffuseBaseStoreManifest,
    treatment_identity: DiffuseBaseTreatmentIdentity,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    database_path: Path | None = None,
) -> tuple[
    DiffuseQueryInputManifest,
    tuple["FrozenLegacyQueryInputs", ...],
]:
    return _verify_query_entry(
        path,
        store_artifact_path=store_artifact_path,
        store_manifest=store_manifest,
        treatment_identity=treatment_identity,
        sample=sample,
        config=config,
        embedding_identity=embedding_identity,
        database_path=database_path,
    )


def verify_query_entry_with_database(
    path: Path,
    *,
    store_artifact_path: Path,
    store_manifest: DiffuseBaseStoreManifest,
    treatment_identity: DiffuseBaseTreatmentIdentity,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    database: Database,
) -> tuple[
    DiffuseQueryInputManifest,
    tuple["FrozenLegacyQueryInputs", ...],
]:
    """Rehydrate frozen rows only through an explicitly owned database."""

    if type(database) is not Database:
        raise TypeError("owned query verification requires an exact Database")
    return _verify_query_entry(
        path,
        store_artifact_path=store_artifact_path,
        store_manifest=store_manifest,
        treatment_identity=treatment_identity,
        sample=sample,
        config=config,
        embedding_identity=embedding_identity,
        database=database,
    )


def _freeze_against_base(
    store_artifact_path: Path,
    *,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedder: object,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    store_manifest: DiffuseBaseStoreManifest,
) -> tuple["FrozenLegacyQueryInputs", ...]:
    from memory_condense.eval.diffuse_longmemeval_runtime import (
        freeze_legacy_query_inputs,
    )

    store_path = store_artifact_path / STORE_DIRECTORY_NAME
    tracked = (store_path / DATABASE_NAME, store_path / "hnsw_index.bin")
    before = {
        path: (file_sha256(path), path.stat().st_mtime_ns) for path in tracked
    }
    condenser: MemoryCondenser | None = None
    try:
        condenser = MemoryCondenser(
            data_dir=store_path,
            model_name=embedding_identity.model_id,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            device=config.embedding_device,
            auto_extract=False,
            embedder=embedder,
            persist_index_on_close=False,
            retriever_max_elements=(
                store_manifest.build_runtime_identity.index_max_elements
            ),
            read_only=True,
        )
        frozen = freeze_legacy_query_inputs(
            condenser,
            [question.retrieval_query for question in sample.questions],
            config.retrieval,
        )
    finally:
        if condenser is not None:
            condenser.close()
    after = {
        path: (file_sha256(path), path.stat().st_mtime_ns) for path in tracked
    }
    require_no_sqlite_sidecars(store_path)
    if before != after:
        raise DiffuseBaseArtifactError(
            "read-only query acquisition mutated the shared base"
        )
    return tuple(frozen)


def _publish_query_entry(
    queries_root: Path,
    *,
    store_artifact_path: Path,
    store_manifest: DiffuseBaseStoreManifest,
    treatment_identity: DiffuseBaseTreatmentIdentity,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedder: object,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    _sealed_import_guard=None,
) -> tuple[
    Path,
    DiffuseQueryInputManifest,
    tuple["FrozenLegacyQueryInputs", ...],
]:
    create_owned = create_publication
    owned_path = publication_path
    write_owned = write_publication_bytes
    capture_owned = capture_publication
    promote_owned = promote_publication
    commit_owned = commit_publication
    rollback_owned = rollback_publication
    validate_marker = validate_publication_lock_marker
    abandon_owned = abandon_publication
    verify_active = verify_query_entry
    freeze_active = _freeze_against_base
    assert_capability_intact, emergency_abandon = publication_operation_guard()
    assert_import_seam = _sealed_import_guard
    module_namespace = globals()
    guarded_globals = {
        name: value
        for name, value in module_namespace.items()
        if not name.startswith("__")
    }
    artifact_error = DiffuseBaseArtifactError

    def assert_operations_intact() -> None:
        assert_import_seam()
        changed = [
            name
            for name, value in guarded_globals.items()
            if module_namespace.get(name) is not value
        ]
        if changed:
            raise artifact_error(
                "query publication module was rebound: " + ", ".join(changed)
            )
        assert_capability_intact()

    def quarantine(owner: object, original: BaseException) -> None:
        try:
            emergency_abandon(owner)  # type: ignore[arg-type]
        except BaseException as abandon_error:
            original.add_note(f"publication abandon also failed: {abandon_error!r}")

    assert_operations_intact()
    require_regular_directory(queries_root, "query-inputs root")
    key = diffuse_query_input_key(
        base_store_key=store_manifest.base_store_key,
        treatment_identity=treatment_identity,
        sample=sample,
        config=config,
        embedding_identity=embedding_identity,
    )
    target = queries_root / key
    if target.exists():
        validate_marker(target)
        manifest, rows = verify_active(
            target,
            store_artifact_path=store_artifact_path,
            store_manifest=store_manifest,
            treatment_identity=treatment_identity,
            sample=sample,
            config=config,
            embedding_identity=embedding_identity,
        )
        return target, manifest, rows
    frozen = freeze_active(
        store_artifact_path,
        sample=sample,
        config=config,
        embedder=embedder,
        embedding_identity=embedding_identity,
        store_manifest=store_manifest,
    )
    assert_operations_intact()
    pointer = _pointer_artifact(
        base_store_key=store_manifest.base_store_key,
        sample=sample,
        frozen=frozen,
    )
    try:
        owner = create_owned(target, role="query")
    except FileExistsError:
        validate_marker(target)
        manifest, rows = verify_active(
            target,
            store_artifact_path=store_artifact_path,
            store_manifest=store_manifest,
            treatment_identity=treatment_identity,
            sample=sample,
            config=config,
            embedding_identity=embedding_identity,
        )
        return target, manifest, rows
    temporary = owned_path(owner)
    owner_live = True
    try:
        write_owned(
            owner,
            FROZEN_QUERY_INPUTS_NAME,
            model_bytes(pointer),
        )
        manifest = _manifest_for_query_inputs(
            temporary,
            key=key,
            store_path=store_artifact_path,
            store_manifest=store_manifest,
            treatment_identity=treatment_identity,
            sample=sample,
            config=config,
            embedding_identity=embedding_identity,
            pointer_artifact=pointer,
        )
        assert_operations_intact()
        write_owned(owner, QUERY_MANIFEST_NAME, model_bytes(manifest))
        assert_operations_intact()
        capture_owned(owner)
        assert_operations_intact()
        try:
            promote_owned(owner)
        except FileExistsError:
            rollback_owned(owner)
            owner_live = False
            validate_marker(target)
        assert_operations_intact()
        active_manifest, active_rows = verify_active(
            target,
            store_artifact_path=store_artifact_path,
            store_manifest=store_manifest,
            treatment_identity=treatment_identity,
            sample=sample,
            config=config,
            embedding_identity=embedding_identity,
        )
        assert_operations_intact()
        if owner_live:
            commit_owned(owner)
            owner_live = False
        return target, active_manifest, active_rows
    except BaseException as original:
        if owner_live:
            try:
                assert_operations_intact()
            except BaseException as guard_error:
                original.add_note(
                    f"publication operations changed; bytes quarantined: {guard_error!r}"
                )
                quarantine(owner, original)
                owner_live = False
            else:
                try:
                    rollback_owned(owner)
                    owner_live = False
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


def _seal_query_entrypoint(implementation, import_guard):
    assert_implementation = freeze_callable_guard(
        implementation,
        error_type=DiffuseBaseArtifactError,
        label="query publication implementation",
    )
    assert_import_guard = freeze_callable_guard(
        import_guard,
        error_type=DiffuseBaseArtifactError,
        label="query publication guard",
    )

    def publish_query_entry(
        queries_root: Path,
        *,
        store_artifact_path: Path,
        store_manifest: DiffuseBaseStoreManifest,
        treatment_identity: DiffuseBaseTreatmentIdentity,
        sample: GoldBlindLongMemEvalSample,
        config: EvalConfig,
        embedder: object,
        embedding_identity: DiffuseBaseEmbeddingIdentity,
    ) -> tuple[
        Path,
        DiffuseQueryInputManifest,
        tuple["FrozenLegacyQueryInputs", ...],
    ]:
        assert_implementation(implementation)
        assert_import_guard(import_guard)
        import_guard()
        return implementation(
            queries_root,
            store_artifact_path=store_artifact_path,
            store_manifest=store_manifest,
            treatment_identity=treatment_identity,
            sample=sample,
            config=config,
            embedder=embedder,
            embedding_identity=embedding_identity,
            _sealed_import_guard=import_guard,
        )

    return publish_query_entry


_QUERY_GUARD_EXCLUDES = (
    "_seal_query_entrypoint",
    "query_publication_import_guard",
    "publish_query_entry",
    "_QUERY_GUARD_EXCLUDES",
    "_sealed_query_guard",
)
_sealed_query_guard = freeze_namespace_guard(
    globals(),
    error_type=DiffuseBaseArtifactError,
    label="query publication module",
    exclude=_QUERY_GUARD_EXCLUDES,
)
query_publication_import_guard = freeze_namespace_guard(
    globals(),
    error_type=DiffuseBaseArtifactError,
    label="query publication module",
    exclude=_QUERY_GUARD_EXCLUDES,
)
publish_query_entry = _seal_query_entrypoint(
    _publish_query_entry, _sealed_query_guard
)
del _seal_query_entrypoint, _sealed_query_guard
