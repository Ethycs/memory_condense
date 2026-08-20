"""Immutable SQLite/HNSW publication and verification for diffuse bases."""

from __future__ import annotations

import hashlib
import json
import tempfile
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import hnswlib
import numpy as np

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._diffuse_base_contracts import (
    BASE_STORE_FORMAT,
    BASE_STORE_REVISION,
    DATABASE_NAME,
    INDEX_NAME,
    STORE_DIRECTORY_NAME,
    STORE_MANIFEST_NAME,
    DiffuseBaseArtifactError,
    DiffuseBaseBuildRuntimeIdentity,
    DiffuseBaseEmbeddingIdentity,
    DiffuseBaseStoreManifest,
    chunker_identity,
    diffuse_base_store_key,
    finite_json_mapping,
    identity,
    model_bytes,
    publish_complete_directory,
    require_exact_children,
    require_no_sqlite_sidecars,
    require_regular_directory,
    require_regular_file,
    safe_remove_staging,
    sample_store_payload,
    self_sha256,
    write_new_bytes,
)
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.diffuse_longmemeval_inputs import (
    GoldBlindLongMemEvalSample,
    ingest_gold_blind_sample_deterministically,
)
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.eval.schemas import EvalConfig
from memory_condense.ingest.chunker import Chunker
from memory_condense.persistence.db import CURRENT_SCHEMA_VERSION, Database
from memory_condense.search.indexes.lexical import term_frequencies


_BASE_DERIVED_TABLES = (
    "association_artifacts",
    "chunk_cav_signatures",
    "chunk_head_edges",
    "consolidation_access_events",
    "consolidation_edges",
    "consolidation_nodes",
    "discourse_artifact_coverage",
    "discourse_artifact_coverage_receipts",
    "discourse_artifacts",
    "discourse_graph_revisions",
    "discourse_relation_evidence",
    "discourse_relation_members",
    "discourse_relations",
    "discourse_unit_evidence",
    "discourse_units",
    "episode_evidence",
    "episode_representatives",
    "episodes",
    "hebbian_access_events",
    "hebbian_chunk_edges",
    "hebbian_chunk_nodes",
    "memory_items",
    "memory_provenance",
)


@dataclass(frozen=True, slots=True)
class _StoreAudit:
    schema_version: int
    database_schema_sha256: str
    deterministic_turn_ids_sha256: str
    turn_sequence_sha256: str
    turn_count: int
    chunk_sequence_sha256: str
    chunk_count: int
    source_ids: tuple[str, ...]
    source_streams_sha256: str
    source_count: int
    base_state_audit_sha256: str


def declared_factory_identity(
    factory: Callable[[Path], MemoryCondenser],
) -> DiffuseBaseBuildRuntimeIdentity:
    raw = getattr(factory, "diffuse_build_runtime_identity", None)
    if callable(raw):
        raw = raw()
    if raw is None:
        return owned_build_runtime_identity(factory)
    if not isinstance(raw, (DiffuseBaseBuildRuntimeIdentity, Mapping)):
        raise TypeError("factory build-runtime identity must be a mapping")
    declared = (
        raw
        if isinstance(raw, DiffuseBaseBuildRuntimeIdentity)
        else DiffuseBaseBuildRuntimeIdentity.model_validate(raw)
    )
    if declared.certification == "owned_runtime_v1":
        raise ValueError("injected factories cannot self-certify as owned runtime")
    actual_digest = callable_build_factory_sha256(factory)
    if declared.factory_identity_sha256 != actual_digest:
        raise ValueError("factory code does not match its declared identity")
    return declared


def _code_payload(code: types.CodeType) -> dict[str, object]:
    constants: list[object] = []
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            constants.append({"code": _code_payload(value)})
        elif value is None or isinstance(value, (str, bool, int, float)):
            constants.append(value)
        elif isinstance(value, tuple):
            constants.append([str(item) for item in value])
        else:
            constants.append(
                {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
            )
    return {
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "constants": constants,
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
    }


def callable_build_factory_sha256(factory: object) -> str:
    """Path-independent code identity for a declared injected factory."""

    function = getattr(factory, "__func__", factory)
    code = getattr(function, "__code__", None)
    if code is None:
        call = getattr(factory, "__call__", None)
        function = getattr(call, "__func__", call)
        code = getattr(function, "__code__", None)
    if not isinstance(code, types.CodeType):
        raise TypeError("declared factory must expose Python callable code")
    return canonical_sha256(
        {
            "format": "memory-condense-diffuse-build-callable-v1",
            "module": str(getattr(function, "__module__", "")),
            "qualname": str(getattr(function, "__qualname__", "")),
            "code": _code_payload(code),
        }
    )


def owned_build_runtime_identity(
    factory: Callable[[Path], MemoryCondenser],
) -> DiffuseBaseBuildRuntimeIdentity:
    """Derive identity only for the repository-owned certified factory seam."""

    from memory_condense.eval.diffuse_longmemeval_runtime import (
        DiffuseLongMemEvalExecutionBinding,
    )
    from memory_condense.modeling.embedding import DEFAULT_MODEL_DIM

    owner = getattr(factory, "__self__", None)
    function = getattr(factory, "__func__", None)
    if (
        type(owner) is not DiffuseLongMemEvalExecutionBinding
        or function is not DiffuseLongMemEvalExecutionBinding.new_condenser
        or not owner.runtime_binding_certified
    ):
        raise TypeError(
            "factory needs a declaration or the exact certified owned runtime seam"
        )
    validate_embedder_certification(
        owner.embedder,
        DiffuseBaseBuildRuntimeIdentity(
            runtime_id="owned-diffuse-longmemeval-runtime-v1",
            factory_identity_sha256="0" * 64,
            condenser_class=(
                "memory_condense.application.condenser.MemoryCondenser"
            ),
            index_space="cosine",
            index_dimension=DEFAULT_MODEL_DIM,
            index_ef_construction=200,
            index_m=16,
            index_max_elements=100_000,
            certification="owned_runtime_v1",
        ),
    )
    callable_name = (
        "memory_condense.eval.diffuse_longmemeval_runtime."
        "DiffuseLongMemEvalExecutionBinding.new_condenser"
    )
    factory_digest = canonical_sha256(
        {
            "format": "memory-condense-owned-diffuse-store-factory-v1",
            "callable": callable_name,
        }
    )
    return DiffuseBaseBuildRuntimeIdentity(
        runtime_id="owned-diffuse-longmemeval-runtime-v1",
        factory_identity_sha256=factory_digest,
        condenser_class=(
            "memory_condense.application.condenser.MemoryCondenser"
        ),
        index_space="cosine",
        index_dimension=DEFAULT_MODEL_DIM,
        index_ef_construction=200,
        index_m=16,
        index_max_elements=100_000,
        certification="owned_runtime_v1",
    )


def validate_embedder_certification(
    embedder: object,
    runtime: DiffuseBaseBuildRuntimeIdentity,
) -> None:
    """Keep the owned label exclusive to the unshadowed BGE implementation."""

    if runtime.certification != "owned_runtime_v1":
        return
    from memory_condense.modeling.embedding import (
        BGE_M3_CHECKPOINT_SHA256,
        DEFAULT_MODEL_DIM,
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
        EmbeddingService,
    )

    shadowable = {
        "embed_chunks",
        "embed_query",
        "embed_queries",
        "_load_model",
        "execution_identity",
        "dim",
        "model_name",
        "model_revision",
        "checkpoint_sha256",
    }
    if type(embedder) is not EmbeddingService or shadowable & set(vars(embedder)):
        raise TypeError(
            "owned diffuse runtime requires exact unshadowed EmbeddingService"
        )
    internal_identity = (
        embedder._model_name,  # noqa: SLF001 - owned certification boundary
        embedder._model_revision,  # noqa: SLF001
        embedder._checkpoint_sha256,  # noqa: SLF001
        runtime.index_dimension,
    )
    if internal_identity != (
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
        BGE_M3_CHECKPOINT_SHA256,
        DEFAULT_MODEL_DIM,
    ):
        raise TypeError("owned diffuse runtime does not contain pinned BGE-M3")
    if embedder._verify_checkpoint is not True:  # noqa: SLF001
        raise TypeError("owned BGE checkpoint verification is disabled")
    model = embedder._model  # noqa: SLF001
    if model is not None:
        from sentence_transformers import SentenceTransformer

        if type(model) is not SentenceTransformer or {
            "encode",
            "forward",
            "tokenize",
            "get_embedding_dimension",
            "get_sentence_embedding_dimension",
        } & set(vars(model)):
            raise TypeError(
                "loaded owned BGE must be an unshadowed SentenceTransformer"
            )
        if (
            embedder._verified_checkpoint_sha256  # noqa: SLF001
            != BGE_M3_CHECKPOINT_SHA256
        ):
            raise TypeError(
                "loaded owned BGE has no verified-checkpoint receipt"
            )
        if int(embedder.dim) != DEFAULT_MODEL_DIM:
            raise TypeError("loaded owned BGE has an unexpected vector dimension")
    elif embedder._verified_checkpoint_sha256 is not None:  # noqa: SLF001
        raise TypeError("unloaded owned BGE carries a stale checkpoint receipt")


def _validate_live_condenser(
    condenser: MemoryCondenser,
    *,
    config: EvalConfig,
    embedder: object,
    expected: DiffuseBaseBuildRuntimeIdentity,
) -> None:
    if type(condenser) is not MemoryCondenser:
        raise TypeError("diffuse base builder must return exact MemoryCondenser")
    shadowed = {
        name
        for name in ("ingest", "ingest_many", "search", "search_hybrid", "close")
        if name in vars(condenser)
    }
    if shadowed:
        raise ValueError(f"base condenser shadows owned methods: {sorted(shadowed)}")
    retriever = condenser._retriever  # noqa: SLF001 - certification boundary
    chunker = condenser._chunker  # noqa: SLF001 - certification boundary
    retriever_shadowed = {
        name
        for name in ("add_chunks", "query", "hybrid_query", "save")
        if name in vars(retriever)
    }
    if retriever_shadowed or "chunk_turn" in vars(chunker):
        raise ValueError("base builder shadows owned ingest/index methods")
    actual = {
        "condenser_class": (
            f"{type(condenser).__module__}.{type(condenser).__qualname__}"
        ),
        "index_dimension": int(retriever._dim),  # noqa: SLF001
        "index_ef_construction": int(retriever._ef_construction),  # noqa: SLF001
        "index_m": int(retriever._M),  # noqa: SLF001
        "index_max_elements": int(retriever._max_elements),  # noqa: SLF001
    }
    expected_observable = {
        "condenser_class": expected.condenser_class,
        "index_dimension": expected.index_dimension,
        "index_ef_construction": expected.index_ef_construction,
        "index_m": expected.index_m,
        "index_max_elements": expected.index_max_elements,
    }
    if actual != expected_observable or expected.index_space != "cosine":
        raise ValueError("live condenser does not match build-runtime identity")
    if condenser._embedder is not embedder:  # noqa: SLF001
        raise ValueError("builder did not use the verified shared embedder")
    if condenser._auto_extract is not False:  # noqa: SLF001
        raise ValueError("diffuse base builder must disable memory extraction")
    if condenser._persist_index_on_close is not True:  # noqa: SLF001
        raise ValueError("diffuse base builder must persist its initial index")
    if (
        chunker.min_tokens != config.chunker.min_tokens
        or chunker.max_tokens != config.chunker.max_tokens
    ):
        raise ValueError("live chunker does not match base config")


def _explicit_chunk_id(
    *, turn_id: str, start_char: int, end_char: int, text: str
) -> str:
    payload = json.dumps(
        {
            "format": "memory-condense-explicit-chunk-id-v1",
            "turn_id": turn_id,
            "start_char": start_char,
            "end_char": end_char,
            "text": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_stream_identity(db: Database) -> tuple[tuple[str, ...], str]:
    streams = scan_discourse_source_chunks(db)
    payload = [
        {
            "source_id": stream.source_id,
            "content_chunk_ids": list(stream.content_chunk_ids),
            "metadata_chunk_ids": list(stream.metadata_chunk_ids),
            "first_ordinal": stream.first_ordinal,
            "last_ordinal": stream.last_ordinal,
            "stream_sha256": stream.stream_sha256,
        }
        for stream in streams
    ]
    return tuple(stream.source_id for stream in streams), identity_sha256(payload)


def _database_schema_sha256(db: Database) -> str:
    rows = db.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type, name, tbl_name"
    ).fetchall()
    return canonical_sha256(
        [
            {
                "type": str(row[0]),
                "name": str(row[1]),
                "table": str(row[2]),
                "sql": None if row[3] is None else str(row[3]),
            }
            for row in rows
        ]
    )


@lru_cache(maxsize=1)
def _expected_database_schema_sha256() -> str:
    with Database(":memory:") as expected:
        return _database_schema_sha256(expected)


def _audit_store(
    store_path: Path,
    sample: GoldBlindLongMemEvalSample,
    *,
    embedding_dimension: int,
    build_runtime_identity: DiffuseBaseBuildRuntimeIdentity,
    chunker_min_tokens: int,
    chunker_max_tokens: int,
) -> _StoreAudit:
    require_regular_directory(store_path, "base store")
    require_exact_children(store_path, {DATABASE_NAME, INDEX_NAME}, "base store")
    database_path = store_path / DATABASE_NAME
    index_path = store_path / INDEX_NAME
    require_regular_file(database_path, "base database")
    require_regular_file(index_path, "base index")
    require_no_sqlite_sidecars(store_path)
    before = {
        path: (file_sha256(path), path.stat().st_mtime_ns)
        for path in (database_path, index_path)
    }
    expected_records = sample.deterministic_ingest_records()
    turn_payload: list[dict[str, object]] = []
    chunk_payload: list[dict[str, object]] = []
    labels: list[int] = []
    with Database(database_path, read_only=True) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise DiffuseBaseArtifactError("base SQLite integrity_check failed")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise DiffuseBaseArtifactError("base SQLite foreign keys are invalid")
        if db.schema_version != CURRENT_SCHEMA_VERSION:
            raise DiffuseBaseArtifactError("base SQLite schema is not current")
        database_schema_sha256 = _database_schema_sha256(db)
        if database_schema_sha256 != _expected_database_schema_sha256():
            raise DiffuseBaseArtifactError(
                "base SQLite schema differs from the current canonical schema"
            )
        turn_rows = db.execute(
            "SELECT turn_id, role, text, source_id, created_at, ordinal "
            "FROM turns ORDER BY ordinal"
        ).fetchall()
        if len(turn_rows) != len(expected_records):
            raise DiffuseBaseArtifactError("base turn count differs from sample")
        turn_text: dict[str, str] = {}
        for ordinal, (row, expected) in enumerate(
            zip(turn_rows, expected_records, strict=True), start=1
        ):
            role, text, source_id, created_at, turn_id = expected
            if tuple(row) != (
                turn_id,
                role,
                text,
                source_id,
                created_at.isoformat(),
                ordinal,
            ):
                raise DiffuseBaseArtifactError(
                    "base turn differs from deterministic gold-blind sample"
                )
            turn_text[str(row[0])] = str(row[2])
            turn_payload.append(
                {
                    "turn_id": row[0],
                    "role": row[1],
                    "text_sha256": quote_sha256(str(row[2])),
                    "source_id": row[3],
                    "created_at": row[4],
                    "ordinal": int(row[5]),
                }
            )
        chunk_count = int(
            db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        )
        if chunk_count == 0:
            raise DiffuseBaseArtifactError("diffuse base produced no chunks")
        chunk_cursor = db.execute(
            "SELECT chunk_id, turn_id, text, start_char, end_char, token_count, "
            "embedding, lexical_weights, hnsw_label, term_count "
            "FROM chunks ORDER BY rowid"
        )
        expected_chunks: list[tuple[str, str, str, int, int, int]] = []
        deterministic_chunker = Chunker(
            min_tokens=chunker_min_tokens,
            max_tokens=chunker_max_tokens,
        )
        for _role, text, _source_id, _created_at, turn_id in expected_records:
            for chunk in deterministic_chunker.chunk_turn(turn_id, text):
                expected_chunks.append(
                    (
                        _explicit_chunk_id(
                            turn_id=turn_id,
                            start_char=chunk.start_char,
                            end_char=chunk.end_char,
                            text=chunk.text,
                        ),
                        turn_id,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.token_count,
                    )
                )
        if chunk_count != len(expected_chunks):
            raise DiffuseBaseArtifactError(
                "base chunks differ from deterministic chunker output"
            )
        for row, expected_chunk in zip(
            chunk_cursor, expected_chunks, strict=True
        ):
            chunk_id, turn_id, text = str(row[0]), str(row[1]), str(row[2])
            start_char, end_char = int(row[3]), int(row[4])
            if (
                chunk_id,
                turn_id,
                text,
                start_char,
                end_char,
                int(row[5]),
            ) != expected_chunk or int(row[5]) != count_tokens(text):
                raise DiffuseBaseArtifactError(
                    "base chunks differ from deterministic chunker output"
                )
            parent_text = turn_text.get(turn_id)
            if (
                parent_text is None
                or start_char < 0
                or end_char <= start_char
                or parent_text[start_char:end_char] != text
            ):
                raise DiffuseBaseArtifactError(
                    "base chunk does not bind an exact parent-turn span"
                )
            if chunk_id != _explicit_chunk_id(
                turn_id=turn_id,
                start_char=start_char,
                end_char=end_char,
                text=text,
            ):
                raise DiffuseBaseArtifactError("base chunk ID is not deterministic")
            embedding_blob = row[6]
            if not isinstance(embedding_blob, bytes):
                raise DiffuseBaseArtifactError("base chunk has no dense embedding")
            vector = np.frombuffer(embedding_blob, dtype=np.float32)
            if len(vector) != embedding_dimension or not np.isfinite(vector).all():
                raise DiffuseBaseArtifactError(
                    "base chunk embedding dimension or finiteness changed"
                )
            lexical = (
                None
                if row[7] is None
                else finite_json_mapping(
                    json.loads(str(row[7])), "chunk lexical weights"
                )
            )
            expected_terms = term_frequencies(text)
            if lexical is None:
                lexical_terms: dict[str, int] = {}
            else:
                if any(
                    not isinstance(term, str)
                    or type(frequency) is not int
                    or frequency < 1
                    for term, frequency in lexical.items()
                ):
                    raise DiffuseBaseArtifactError(
                        "base chunk lexical weights are not term frequencies"
                    )
                lexical_terms = {
                    str(term): int(frequency)
                    for term, frequency in lexical.items()
                }
            postings = {
                str(term): int(frequency)
                for term, frequency in db.execute(
                    "SELECT term, tf FROM chunk_terms "
                    "WHERE chunk_id = ? ORDER BY term",
                    (chunk_id,),
                ).fetchall()
            }
            if lexical_terms != expected_terms or postings != expected_terms:
                raise DiffuseBaseArtifactError(
                    "base lexical index differs from deterministic chunk text"
                )
            if row[8] is None or row[9] is None:
                raise DiffuseBaseArtifactError(
                    "base chunk is missing ANN or lexical coordinates"
                )
            label = int(row[8])
            if int(row[9]) != sum(expected_terms.values()):
                raise DiffuseBaseArtifactError(
                    "base chunk lexical term count differs from postings"
                )
            labels.append(label)
            chunk_payload.append(
                {
                    "chunk_id": chunk_id,
                    "turn_id": turn_id,
                    "text_sha256": quote_sha256(text),
                    "start_char": start_char,
                    "end_char": end_char,
                    "token_count": int(row[5]),
                    "embedding_sha256": hashlib.sha256(
                        embedding_blob
                    ).hexdigest(),
                    "lexical_weights_sha256": (
                        None if lexical is None else canonical_sha256(lexical)
                    ),
                    "hnsw_label": label,
                    "term_count": int(row[9]),
                }
            )
        if len(labels) != len(set(labels)):
            raise DiffuseBaseArtifactError("base ANN labels are not unique")
        meta_rows = [
            (str(key), str(value))
            for key, value in db.execute(
                "SELECT key, value FROM meta ORDER BY key"
            ).fetchall()
        ]
        if meta_rows != [
            ("next_hnsw_label", str(len(labels))),
            ("schema_version", str(CURRENT_SCHEMA_VERSION)),
        ]:
            raise DiffuseBaseArtifactError("base SQLite metadata is not pristine")
        source_ids, streams_sha256 = _source_stream_identity(db)
        derived_counts = {
            table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in _BASE_DERIVED_TABLES
        }
        if any(derived_counts.values()):
            raise DiffuseBaseArtifactError(
                "base store contains derived memory or discourse state"
            )
        revision = db.execute(
            "SELECT singleton, source_revision, graph_content_revision "
            "FROM discourse_revision_state"
        ).fetchall()
        if len(revision) != 1 or tuple(revision[0])[0] != 1:
            raise DiffuseBaseArtifactError("base discourse revision state is invalid")
        if int(revision[0][1]) != len(turn_rows) + chunk_count:
            raise DiffuseBaseArtifactError("base source revision is not pristine")
        if int(revision[0][2]) != 0:
            raise DiffuseBaseArtifactError("base graph revision is not zero")
        base_state_audit_sha256 = canonical_sha256(
            {
                "derived_row_counts": derived_counts,
                "discourse_revision_state": list(revision[0]),
                "meta": [list(row) for row in meta_rows],
            }
        )
    index = hnswlib.Index(
        space=build_runtime_identity.index_space,
        dim=embedding_dimension,
    )
    try:
        index.load_index(str(index_path))
        controls = (
            str(index.space),
            int(index.dim),
            int(index.ef_construction),
            int(index.M),
            int(index.get_max_elements()),
        )
        expected_controls = (
            build_runtime_identity.index_space,
            build_runtime_identity.index_dimension,
            build_runtime_identity.index_ef_construction,
            build_runtime_identity.index_m,
            build_runtime_identity.index_max_elements,
        )
        if controls != expected_controls:
            raise DiffuseBaseArtifactError(
                "base HNSW controls differ from build-runtime identity"
            )
        index_labels = tuple(sorted(int(item) for item in index.get_ids_list()))
        if index.get_current_count() != len(labels):
            raise DiffuseBaseArtifactError("base HNSW count differs from SQLite")
        if index_labels != tuple(sorted(labels)):
            raise DiffuseBaseArtifactError("base HNSW labels differ from SQLite")
        with Database(database_path, read_only=True) as db:
            vector_cursor = db.execute(
                "SELECT hnsw_label, embedding FROM chunks "
                "WHERE hnsw_label IS NOT NULL ORDER BY hnsw_label"
            )
            offset = 0
            while batch := vector_cursor.fetchmany(256):
                batch_labels = tuple(int(row[0]) for row in batch)
                if batch_labels != index_labels[offset : offset + len(batch)]:
                    raise DiffuseBaseArtifactError(
                        "base HNSW label order differs from SQLite"
                    )
                actual_vectors = np.asarray(
                    index.get_items(list(batch_labels)), dtype=np.float32
                )
                sqlite_vectors = np.stack(
                    [
                        np.frombuffer(row[1], dtype=np.float32)
                        for row in batch
                    ]
                )
                norms = np.linalg.norm(sqlite_vectors, axis=1)
                if not np.isfinite(norms).all() or not (norms > 0.0).all():
                    raise DiffuseBaseArtifactError(
                        "base SQLite embeddings are invalid for cosine indexing"
                    )
                expected_vectors = sqlite_vectors / norms[:, None]
                # hnswlib's cosine space stores normalized float32 vectors.
                # Its C++ accumulation differs slightly from NumPy at 1024d.
                if not np.allclose(
                    actual_vectors,
                    expected_vectors,
                    rtol=1e-6,
                    atol=1e-7,
                    equal_nan=False,
                ):
                    raise DiffuseBaseArtifactError(
                        "base HNSW vectors differ from SQLite embeddings"
                    )
                offset += len(batch)
            if offset != len(index_labels):
                raise DiffuseBaseArtifactError(
                    "base HNSW vector count differs from SQLite"
                )
    except DiffuseBaseArtifactError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise DiffuseBaseArtifactError("base HNSW index cannot be verified") from exc
    finally:
        index = None
    require_no_sqlite_sidecars(store_path)
    after = {
        path: (file_sha256(path), path.stat().st_mtime_ns)
        for path in (database_path, index_path)
    }
    if before != after:
        raise DiffuseBaseArtifactError("read-only base verification mutated files")
    return _StoreAudit(
        schema_version=CURRENT_SCHEMA_VERSION,
        database_schema_sha256=database_schema_sha256,
        deterministic_turn_ids_sha256=canonical_sha256(
            list(sample.deterministic_turn_ids)
        ),
        turn_sequence_sha256=canonical_sha256(turn_payload),
        turn_count=len(turn_payload),
        chunk_sequence_sha256=canonical_sha256(chunk_payload),
        chunk_count=len(chunk_payload),
        source_ids=source_ids,
        source_streams_sha256=streams_sha256,
        source_count=len(source_ids),
        base_state_audit_sha256=base_state_audit_sha256,
    )


def _manifest_for_store(
    artifact_path: Path,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    *,
    key: str,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    build_runtime_identity: DiffuseBaseBuildRuntimeIdentity,
    implementation_digest: str,
    environment_digest: str,
) -> DiffuseBaseStoreManifest:
    store_path = artifact_path / STORE_DIRECTORY_NAME
    audit = _audit_store(
        store_path,
        sample,
        embedding_dimension=embedding_identity.dimension,
        build_runtime_identity=build_runtime_identity,
        chunker_min_tokens=config.chunker.min_tokens,
        chunker_max_tokens=config.chunker.max_tokens,
    )
    database_path, index_path = store_path / DATABASE_NAME, store_path / INDEX_NAME
    chunker = chunker_identity(config)
    manifest = DiffuseBaseStoreManifest(
        base_store_key=key,
        sample_id_sha256=identity_sha256({"sample_id": sample.sample_id}),
        corpus_sha256=sample.corpus_sha256,
        store_sample_sha256=canonical_sha256(sample_store_payload(sample)),
        chunker_identity=chunker,
        chunker_identity_sha256=identity(chunker),
        embedding_identity=embedding_identity,
        embedding_identity_sha256=identity(embedding_identity),
        build_runtime_identity=build_runtime_identity,
        build_runtime_identity_sha256=identity(build_runtime_identity),
        implementation_sha256=implementation_digest,
        environment_lock_sha256=environment_digest,
        schema_version=audit.schema_version,
        database_schema_sha256=audit.database_schema_sha256,
        deterministic_turn_ids_sha256=audit.deterministic_turn_ids_sha256,
        turn_sequence_sha256=audit.turn_sequence_sha256,
        turn_count=audit.turn_count,
        chunk_sequence_sha256=audit.chunk_sequence_sha256,
        chunk_count=audit.chunk_count,
        source_ids=audit.source_ids,
        source_streams_sha256=audit.source_streams_sha256,
        source_count=audit.source_count,
        base_state_audit_sha256=audit.base_state_audit_sha256,
        database_sha256=file_sha256(database_path),
        database_bytes=database_path.stat().st_size,
        index_sha256=file_sha256(index_path),
        index_bytes=index_path.stat().st_size,
        artifact_sha256="0" * 64,
    )
    return manifest.model_copy(
        update={"artifact_sha256": self_sha256(manifest, "artifact_sha256")}
    )


def load_store_manifest(path: Path) -> DiffuseBaseStoreManifest:
    require_regular_directory(path, "base artifact")
    manifest_path = path / STORE_MANIFEST_NAME
    require_regular_file(manifest_path, "base manifest")
    try:
        payload = manifest_path.read_bytes()
        manifest = DiffuseBaseStoreManifest.model_validate_json(payload)
    except (OSError, ValueError) as exc:
        raise DiffuseBaseArtifactError(
            f"invalid base-store manifest: {manifest_path}"
        ) from exc
    if payload != model_bytes(manifest):
        raise DiffuseBaseArtifactError("base manifest is not canonical JSON")
    if manifest.format != BASE_STORE_FORMAT or manifest.revision != BASE_STORE_REVISION:
        raise DiffuseBaseArtifactError("unsupported base-store manifest")
    if manifest.artifact_sha256 != self_sha256(manifest, "artifact_sha256"):
        raise DiffuseBaseArtifactError("base manifest self-receipt changed")
    return manifest


def verify_store_entry(
    path: Path,
    *,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    build_runtime_identity: DiffuseBaseBuildRuntimeIdentity,
    implementation_digest: str,
    environment_digest: str,
) -> DiffuseBaseStoreManifest:
    require_regular_directory(path, "base artifact")
    require_exact_children(
        path, {STORE_MANIFEST_NAME, STORE_DIRECTORY_NAME}, "base artifact"
    )
    manifest = load_store_manifest(path)
    expected_key = diffuse_base_store_key(
        sample,
        config,
        embedding_identity=embedding_identity,
        build_runtime_identity=build_runtime_identity,
        implementation_digest=implementation_digest,
        environment_digest=environment_digest,
    )
    if manifest.base_store_key != expected_key or path.name != expected_key:
        raise DiffuseBaseArtifactError("base store key or directory changed")
    chunker = chunker_identity(config)
    expected_fields: dict[str, object] = {
        "sample_id_sha256": identity_sha256({"sample_id": sample.sample_id}),
        "corpus_sha256": sample.corpus_sha256,
        "store_sample_sha256": canonical_sha256(sample_store_payload(sample)),
        "chunker_identity": chunker,
        "chunker_identity_sha256": identity(chunker),
        "embedding_identity": embedding_identity,
        "embedding_identity_sha256": identity(embedding_identity),
        "build_runtime_identity": build_runtime_identity,
        "build_runtime_identity_sha256": identity(build_runtime_identity),
        "implementation_sha256": implementation_digest,
        "environment_lock_sha256": environment_digest,
        "schema_version": CURRENT_SCHEMA_VERSION,
    }
    for name, expected in expected_fields.items():
        if getattr(manifest, name) != expected:
            raise DiffuseBaseArtifactError(f"base manifest changed {name}")
    store_path = path / STORE_DIRECTORY_NAME
    database_path, index_path = store_path / DATABASE_NAME, store_path / INDEX_NAME
    if (
        file_sha256(database_path) != manifest.database_sha256
        or database_path.stat().st_size != manifest.database_bytes
    ):
        raise DiffuseBaseArtifactError("base SQLite bytes changed")
    if (
        file_sha256(index_path) != manifest.index_sha256
        or index_path.stat().st_size != manifest.index_bytes
    ):
        raise DiffuseBaseArtifactError("base HNSW bytes changed")
    audit = _audit_store(
        store_path,
        sample,
        embedding_dimension=embedding_identity.dimension,
        build_runtime_identity=build_runtime_identity,
        chunker_min_tokens=config.chunker.min_tokens,
        chunker_max_tokens=config.chunker.max_tokens,
    )
    for name in _StoreAudit.__dataclass_fields__:
        if getattr(manifest, name) != getattr(audit, name):
            raise DiffuseBaseArtifactError(f"base logical identity changed {name}")
    return manifest


def publish_store_entry(
    stores_root: Path,
    *,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    embedding_identity: DiffuseBaseEmbeddingIdentity,
    build_runtime_identity: DiffuseBaseBuildRuntimeIdentity,
    embedder: object,
    condenser_factory: Callable[[Path], MemoryCondenser],
    implementation_digest: str,
    environment_digest: str,
) -> tuple[Path, DiffuseBaseStoreManifest]:
    require_regular_directory(stores_root, "base stores root")
    key = diffuse_base_store_key(
        sample,
        config,
        embedding_identity=embedding_identity,
        build_runtime_identity=build_runtime_identity,
        implementation_digest=implementation_digest,
        environment_digest=environment_digest,
    )
    target = stores_root / key
    if target.exists():
        return target, verify_store_entry(
            target,
            sample=sample,
            config=config,
            embedding_identity=embedding_identity,
            build_runtime_identity=build_runtime_identity,
            implementation_digest=implementation_digest,
            environment_digest=environment_digest,
        )
    temporary = Path(tempfile.mkdtemp(prefix=".building-store-", dir=stores_root))
    built: MemoryCondenser | None = None
    try:
        built = condenser_factory(temporary / STORE_DIRECTORY_NAME)
        _validate_live_condenser(
            built,
            config=config,
            embedder=embedder,
            expected=build_runtime_identity,
        )
        ingest_gold_blind_sample_deterministically(built, sample)
        built.close()
        built = None
        manifest = _manifest_for_store(
            temporary,
            sample,
            config,
            key=key,
            embedding_identity=embedding_identity,
            build_runtime_identity=build_runtime_identity,
            implementation_digest=implementation_digest,
            environment_digest=environment_digest,
        )
        write_new_bytes(temporary / STORE_MANIFEST_NAME, model_bytes(manifest))
        try:
            publish_complete_directory(
                temporary, target, manifest_name=STORE_MANIFEST_NAME
            )
        except FileExistsError:
            safe_remove_staging(temporary, stores_root)
        return target, verify_store_entry(
            target,
            sample=sample,
            config=config,
            embedding_identity=embedding_identity,
            build_runtime_identity=build_runtime_identity,
            implementation_digest=implementation_digest,
            environment_digest=environment_digest,
        )
    except BaseException:
        if built is not None:
            try:
                built.close()
            except Exception:
                pass
        if temporary.exists():
            safe_remove_staging(temporary, stores_root)
        raise
