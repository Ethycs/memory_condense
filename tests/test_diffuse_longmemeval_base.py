from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import hnswlib
import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.schemas import Chunk
from memory_condense.eval.diffuse_longmemeval_base import (
    DERIVED_ORIGIN_NAME,
    FROZEN_QUERY_INPUTS_NAME,
    QUERY_MANIFEST_NAME,
    STORE_DIRECTORY_NAME,
    STORE_MANIFEST_NAME,
    DiffuseBaseArtifactError,
    DiffuseBaseBuildRuntimeIdentity,
    DiffuseBaseEmbeddingIdentity,
    DiffuseBaseTreatmentIdentity,
    VerifiedDiffuseLongMemEvalBase,
    callable_build_factory_sha256,
    clone_diffuse_longmemeval_base,
    open_diffuse_longmemeval_derived_store,
    owned_build_runtime_identity,
    publish_diffuse_longmemeval_base,
    verify_diffuse_longmemeval_base,
)
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.diffuse_compilation import (
    DiffuseCompilationPolicy,
    compile_diffuse_artifact,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    GoldBlindLongMemEvalSample,
    gold_blind_longmemeval_sample,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DiffuseLongMemEvalRuntimeConfig,
    FrozenLegacyDiffuseInputProvider,
    build_diffuse_longmemeval_execution_binding,
)
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


_DIMENSION = 16
_MAX_ELEMENTS = 128
_DATABASE_NAME = "memory.db"
_INDEX_NAME = "hnsw_index.bin"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _DeterministicEmbedder:
    """Small, provider-free embedding implementation with a closed identity."""

    model_name = "test/deterministic-bow"
    model_revision = "fixture-v1"
    checkpoint_sha256 = _sha("deterministic-bow-checkpoint")
    batch_size = 4

    def __init__(self) -> None:
        self.chunk_batches = 0
        self.query_calls = 0

    @property
    def dim(self) -> int:
        return _DIMENSION

    @property
    def execution_identity(self) -> dict[str, object]:
        return {
            "backend": "test-deterministic-bow-v1",
            "device": "cpu",
            "batch_size": self.batch_size,
            "normalize_embeddings": False,
            "output_dtype": "float32",
        }

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(_DIMENSION, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            vector[zlib.crc32(token.encode("utf-8")) % _DIMENSION] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        return self._vector(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        self.chunk_batches += 1
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]


def _embedding_identity() -> DiffuseBaseEmbeddingIdentity:
    return DiffuseBaseEmbeddingIdentity(
        backend="test-deterministic-bow-v1",
        model_id=_DeterministicEmbedder.model_name,
        model_revision=_DeterministicEmbedder.model_revision,
        checkpoint_sha256=_DeterministicEmbedder.checkpoint_sha256,
        dimension=_DIMENSION,
        device="cpu",
        batch_size=_DeterministicEmbedder.batch_size,
        normalize_embeddings=False,
        output_dtype="float32",
    )


def _build_runtime_identity(
    *, factory_digest: str
) -> DiffuseBaseBuildRuntimeIdentity:
    return DiffuseBaseBuildRuntimeIdentity(
        runtime_id="deterministic-test-runtime-v1",
        factory_identity_sha256=factory_digest,
        condenser_class=(
            "memory_condense.application.condenser.MemoryCondenser"
        ),
        index_dimension=_DIMENSION,
        index_ef_construction=200,
        index_m=16,
        index_max_elements=_MAX_ELEMENTS,
        certification="deterministic_test_v1",
    )


def _config() -> EvalConfig:
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=1, max_tokens=24),
        retrieval=RetrievalConfig(mode="dense", k=2, ef_search=16),
        embedding_device="cpu",
        max_prompt_tokens=256,
    )


def _sample(
    *,
    question: str = "Which color was assigned to the launch badge?",
    question_id: str = "q-1",
    turns: tuple[tuple[str, str], ...] | None = None,
) -> GoldBlindLongMemEvalSample:
    haystack = turns or (
        ("user", "The ultraviolet launch badge was amber."),
        ("assistant", "The private relay route was north."),
    )
    benchmark = BenchmarkSample(
        sample_id="fixture-sample",
        turns=list(haystack),
        turn_source_ids=[f"source-{index}" for index in range(len(haystack))],
        turn_created_at=[
            datetime(2025, 1, index + 1, tzinfo=timezone.utc)
            for index in range(len(haystack))
        ],
        questions=[
            BenchmarkQuestion(
                question_id=question_id,
                question=question,
                answer="gold-is-never-projected-into-the-base",
            )
        ],
    )
    return gold_blind_longmemeval_sample(benchmark)


def _treatment(label: str = "primary") -> DiffuseBaseTreatmentIdentity:
    return DiffuseBaseTreatmentIdentity(
        treatment_file_sha256=_sha(f"{label}:treatment-file"),
        sanitized_projection_sha256=_sha(f"{label}:projection"),
        dataset_sha256=_sha("dataset"),
        split_manifest_sha256=_sha("split"),
        ordered_question_ids_sha256=_sha(f"{label}:questions"),
        sample_count=1,
        sample_ordinal=0,
    )


def _factory(
    *,
    config: EvalConfig,
    embedder: _DeterministicEmbedder,
    calls: list[Path],
) -> Callable[[Path], MemoryCondenser]:
    def create(data_dir: Path) -> MemoryCondenser:
        calls.append(data_dir)
        return MemoryCondenser(
            data_dir=data_dir,
            model_name=embedder.model_name,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            device=config.embedding_device,
            auto_extract=False,
            embedder=embedder,
            persist_index_on_close=True,
            retriever_max_elements=_MAX_ELEMENTS,
        )

    create.diffuse_build_runtime_identity = _build_runtime_identity(  # type: ignore[attr-defined]
        factory_digest=callable_build_factory_sha256(create)
    )
    return create


@contextmanager
def _published(
    root: Path,
    *,
    sample: GoldBlindLongMemEvalSample | None = None,
    treatment: DiffuseBaseTreatmentIdentity | None = None,
) -> Iterator[
    tuple[
        VerifiedDiffuseLongMemEvalBase,
        GoldBlindLongMemEvalSample,
        EvalConfig,
        DiffuseBaseTreatmentIdentity,
        DiffuseBaseEmbeddingIdentity,
        DiffuseBaseBuildRuntimeIdentity,
        _DeterministicEmbedder,
        list[Path],
        Callable[[Path], MemoryCondenser],
    ]
]:
    active_sample = sample or _sample()
    active_treatment = treatment or _treatment()
    config = _config()
    embedder = _DeterministicEmbedder()
    embedding = _embedding_identity()
    calls: list[Path] = []
    factory = _factory(config=config, embedder=embedder, calls=calls)
    runtime = factory.diffuse_build_runtime_identity  # type: ignore[attr-defined]
    base = publish_diffuse_longmemeval_base(
        root,
        treatment_identity=active_treatment,
        sample=active_sample,
        config=config,
        embedding_identity=embedding,
        build_runtime_identity=runtime,
        embedder=embedder,
        condenser_factory=factory,
        implementation_digest=_sha("implementation-under-test"),
        environment_digest=_sha("environment-under-test"),
    )
    yield (
        base,
        active_sample,
        config,
        active_treatment,
        embedding,
        runtime,
        embedder,
        calls,
        factory,
    )


def _verify(
    root: Path,
    *,
    sample: GoldBlindLongMemEvalSample,
    config: EvalConfig,
    treatment: DiffuseBaseTreatmentIdentity,
    embedding: DiffuseBaseEmbeddingIdentity,
    runtime: DiffuseBaseBuildRuntimeIdentity,
) -> VerifiedDiffuseLongMemEvalBase:
    return verify_diffuse_longmemeval_base(
        root,
        treatment_identity=treatment,
        sample=sample,
        config=config,
        embedding_identity=embedding,
        build_runtime_identity=runtime,
        implementation_digest=_sha("implementation-under-test"),
        environment_digest=_sha("environment-under-test"),
    )


def _tracked_files(base: VerifiedDiffuseLongMemEvalBase) -> tuple[Path, ...]:
    return (
        base.store_path / STORE_MANIFEST_NAME,
        base.store_path / STORE_DIRECTORY_NAME / _DATABASE_NAME,
        base.store_path / STORE_DIRECTORY_NAME / _INDEX_NAME,
        base.query_inputs_path / QUERY_MANIFEST_NAME,
        base.query_inputs_path / FROZEN_QUERY_INPUTS_NAME,
    )


def _bytes_and_mtime(paths: tuple[Path, ...]) -> dict[Path, tuple[str, int]]:
    return {
        path: (file_sha256(path), path.stat().st_mtime_ns) for path in paths
    }


def _canonical_bytes(value: object) -> bytes:
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


def _rewrite_self_hashed_manifest(path: Path, value: dict[str, object]) -> None:
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = canonical_sha256(unsigned)
    path.write_bytes(_canonical_bytes(value))


def test_store_and_query_addresses_are_separate_and_hits_do_not_reingest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    first_sample = _sample()
    second_sample = _sample(
        question="Where was the private relay routed?",
    )
    config = _config()
    embedder = _DeterministicEmbedder()
    embedding = _embedding_identity()
    calls: list[Path] = []
    factory = _factory(config=config, embedder=embedder, calls=calls)
    runtime = factory.diffuse_build_runtime_identity  # type: ignore[attr-defined]

    def publish(
        sample: GoldBlindLongMemEvalSample,
        treatment: DiffuseBaseTreatmentIdentity,
        active_config: EvalConfig = config,
    ) -> VerifiedDiffuseLongMemEvalBase:
        return publish_diffuse_longmemeval_base(
            root,
            treatment_identity=treatment,
            sample=sample,
            config=active_config,
            embedding_identity=embedding,
            build_runtime_identity=runtime,
            embedder=embedder,
            condenser_factory=factory,
            implementation_digest=_sha("implementation-under-test"),
            environment_digest=_sha("environment-under-test"),
        )

    first = publish(first_sample, _treatment("first"))
    calls_after_first = len(calls)
    chunk_batches_after_first = embedder.chunk_batches
    query_calls_after_first = embedder.query_calls
    cache_hit = publish(first_sample, _treatment("first"))

    assert calls_after_first == 1
    assert len(calls) == calls_after_first
    assert embedder.chunk_batches == chunk_batches_after_first == 1
    assert embedder.query_calls == query_calls_after_first
    assert cache_hit.base_store_key == first.base_store_key
    assert cache_hit.query_input_key == first.query_input_key

    changed_query = publish(second_sample, _treatment("first"))

    assert changed_query.base_store_key == first.base_store_key
    assert changed_query.query_input_key != first.query_input_key
    assert changed_query.store_path == first.store_path
    assert changed_query.query_inputs_path != first.query_inputs_path
    assert len(calls) == 1
    assert embedder.chunk_batches == 1
    assert embedder.query_calls > query_calls_after_first

    changed_treatment = publish(first_sample, _treatment("second"))
    changed_retrieval = publish(
        first_sample,
        _treatment("first"),
        config.model_copy(
            update={"retrieval": config.retrieval.model_copy(update={"k": 1})}
        ),
    )
    assert changed_treatment.base_store_key == first.base_store_key
    assert changed_treatment.query_input_key != first.query_input_key
    assert changed_retrieval.base_store_key == first.base_store_key
    assert changed_retrieval.query_input_key != first.query_input_key
    assert len(calls) == 1
    assert embedder.chunk_batches == 1


def test_injected_factory_cannot_self_claim_owned_runtime(
    tmp_path: Path,
) -> None:
    config = _config()
    embedder = _DeterministicEmbedder()
    calls: list[Path] = []
    factory = _factory(config=config, embedder=embedder, calls=calls)
    declared = factory.diffuse_build_runtime_identity  # type: ignore[attr-defined]
    forged = declared.model_copy(update={"certification": "owned_runtime_v1"})
    factory.diffuse_build_runtime_identity = forged  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="cannot self-certify"):
        publish_diffuse_longmemeval_base(
            tmp_path / "cache",
            treatment_identity=_treatment(),
            sample=_sample(),
            config=config,
            embedding_identity=_embedding_identity(),
            build_runtime_identity=forged,
            embedder=embedder,
            condenser_factory=factory,
            implementation_digest=_sha("implementation-under-test"),
            environment_digest=_sha("environment-under-test"),
        )

    assert calls == []
    assert embedder.chunk_batches == 0
    assert embedder.query_calls == 0


def test_owned_factory_rejects_instance_shadowed_bge(tmp_path: Path) -> None:
    binding = build_diffuse_longmemeval_execution_binding(
        config=_config(),
        runtime=DiffuseLongMemEvalRuntimeConfig(
            qwen_model_dir=tmp_path / "unused-qwen"
        ),
    )
    assert binding.runtime_binding_certified
    binding.embedder.embed_query = (  # type: ignore[method-assign]
        lambda _query: np.zeros(_DIMENSION, dtype=np.float32)
    )

    with pytest.raises(TypeError, match="unshadowed EmbeddingService"):
        owned_build_runtime_identity(binding.new_condenser)

    checkpoint_disabled = build_diffuse_longmemeval_execution_binding(
        config=_config(),
        runtime=DiffuseLongMemEvalRuntimeConfig(
            qwen_model_dir=tmp_path / "unused-qwen-disabled"
        ),
    )
    checkpoint_disabled.embedder._verify_checkpoint = False  # noqa: SLF001
    with pytest.raises(TypeError, match="verification is disabled"):
        owned_build_runtime_identity(checkpoint_disabled.new_condenser)

    preloaded_fake = build_diffuse_longmemeval_execution_binding(
        config=_config(),
        runtime=DiffuseLongMemEvalRuntimeConfig(
            qwen_model_dir=tmp_path / "unused-qwen-preloaded"
        ),
    )
    from sentence_transformers import SentenceTransformer

    preloaded_fake.embedder._model = SentenceTransformer.__new__(  # noqa: SLF001
        SentenceTransformer
    )
    preloaded_fake.embedder._dim = 1024  # noqa: SLF001
    with pytest.raises(TypeError, match="no verified-checkpoint receipt"):
        owned_build_runtime_identity(preloaded_fake.new_condenser)


def test_pointer_artifact_is_canonical_closed_and_text_free(tmp_path: Path) -> None:
    with _published(tmp_path / "cache") as (base, *_rest):
        pointer_path = base.query_inputs_path / FROZEN_QUERY_INPUTS_NAME
        raw = pointer_path.read_bytes()
        payload = json.loads(raw)

    assert raw == (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert set(path.name for path in base.query_inputs_path.iterdir()) == {
        QUERY_MANIFEST_NAME,
        FROZEN_QUERY_INPUTS_NAME,
    }
    assert payload["base_store_key"] == base.base_store_key
    assert payload["rows"][0]["anchors"]
    assert payload["rows"][0]["anchors"][0]["chunk_id"]
    assert payload["rows"][0]["universe_source_ids"] == [
        "source-0",
        "source-1",
    ]

    keys: set[str] = set()

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        else:
            assert value is None or isinstance(value, (str, bool, int, float))

    inspect(payload)
    assert not {
        "query",
        "prompt_question",
        "text",
        "chunk",
        "turn",
        "embedding",
        "lexical_weights",
    } & keys
    lowered = raw.decode("utf-8").casefold()
    assert "ultraviolet launch badge" not in lowered
    assert "private relay route" not in lowered
    assert "which color was assigned" not in lowered
    assert "gold-is-never-projected" not in lowered


def test_verification_is_read_only_down_to_hashes_and_mtimes(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        paths = _tracked_files(base)
        before = _bytes_and_mtime(paths)
        verified = _verify(
            root,
            sample=sample,
            config=config,
            treatment=treatment,
            embedding=embedding,
            runtime=runtime,
        )
        after = _bytes_and_mtime(paths)

    assert after == before
    assert verified.base_store_key == base.base_store_key
    assert verified.query_input_key == base.query_input_key
    assert tuple(row.receipt_sha256 for row in verified.frozen_query_inputs) == (
        tuple(row.receipt_sha256 for row in base.frozen_query_inputs)
    )
    store = base.store_path / STORE_DIRECTORY_NAME
    assert not (store / f"{_DATABASE_NAME}-wal").exists()
    assert not (store / f"{_DATABASE_NAME}-shm").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "database",
        "index",
        "store_manifest",
        "pointer",
        "query_manifest",
        "extra_store_file",
        "extra_query_file",
        "sqlite_sidecar",
    ),
)
def test_verification_rejects_every_closed_artifact_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / mutation / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        store = base.store_path / STORE_DIRECTORY_NAME
        targets = {
            "database": store / _DATABASE_NAME,
            "index": store / _INDEX_NAME,
            "store_manifest": base.store_path / STORE_MANIFEST_NAME,
            "pointer": base.query_inputs_path / FROZEN_QUERY_INPUTS_NAME,
            "query_manifest": base.query_inputs_path / QUERY_MANIFEST_NAME,
        }
        if mutation in targets:
            with targets[mutation].open("ab") as handle:
                handle.write(b"tamper")
        elif mutation == "extra_store_file":
            (store / "unexpected.bin").write_bytes(b"unexpected")
        elif mutation == "extra_query_file":
            (base.query_inputs_path / "unexpected.json").write_text(
                "{}", encoding="utf-8"
            )
        elif mutation == "sqlite_sidecar":
            (store / f"{_DATABASE_NAME}-wal").write_bytes(b"not-a-wal")
        else:  # pragma: no cover - parameter exhaustiveness guard
            raise AssertionError(mutation)

        with pytest.raises(DiffuseBaseArtifactError):
            _verify(
                root,
                sample=sample,
                config=config,
                treatment=treatment,
                embedding=embedding,
                runtime=runtime,
            )


@pytest.mark.parametrize("semantic_mutation", ("score", "unknown_chunk"))
def test_rehydration_rejects_rehashed_semantic_pointer_tamper(
    tmp_path: Path,
    semantic_mutation: str,
) -> None:
    root = tmp_path / semantic_mutation / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        pointer_path = base.query_inputs_path / FROZEN_QUERY_INPUTS_NAME
        pointer = json.loads(pointer_path.read_bytes())
        anchor = pointer["rows"][0]["anchors"][0]
        if semantic_mutation == "score":
            anchor["diagnostics"]["score"] += 0.125
        else:
            anchor["chunk_id"] = "missing-chunk-id"
        pointer_path.write_bytes(_canonical_bytes(pointer))

        query_manifest_path = base.query_inputs_path / QUERY_MANIFEST_NAME
        query_manifest = json.loads(query_manifest_path.read_bytes())
        query_manifest["frozen_inputs_sha256"] = file_sha256(pointer_path)
        query_manifest["frozen_inputs_bytes"] = pointer_path.stat().st_size
        _rewrite_self_hashed_manifest(query_manifest_path, query_manifest)
        before = _bytes_and_mtime((pointer_path, query_manifest_path))

        with pytest.raises(DiffuseBaseArtifactError):
            _verify(
                root,
                sample=sample,
                config=config,
                treatment=treatment,
                embedding=embedding,
                runtime=runtime,
            )

        assert _bytes_and_mtime((pointer_path, query_manifest_path)) == before


def test_logical_store_audit_rejects_rehashed_sqlite_tamper(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        database_path = base.store_path / STORE_DIRECTORY_NAME / _DATABASE_NAME
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute(
                "UPDATE turns SET text = text || ' tampered' WHERE ordinal = 1"
            )
            connection.commit()

        store_manifest_path = base.store_path / STORE_MANIFEST_NAME
        store_manifest = json.loads(store_manifest_path.read_bytes())
        store_manifest["database_sha256"] = file_sha256(database_path)
        store_manifest["database_bytes"] = database_path.stat().st_size
        _rewrite_self_hashed_manifest(store_manifest_path, store_manifest)

        query_manifest_path = base.query_inputs_path / QUERY_MANIFEST_NAME
        query_manifest = json.loads(query_manifest_path.read_bytes())
        query_manifest["base_artifact_sha256"] = store_manifest["artifact_sha256"]
        query_manifest["base_manifest_sha256"] = file_sha256(store_manifest_path)
        query_manifest["database_sha256"] = store_manifest["database_sha256"]
        _rewrite_self_hashed_manifest(query_manifest_path, query_manifest)
        tracked = (database_path, store_manifest_path, query_manifest_path)
        before = _bytes_and_mtime(tracked)

        with pytest.raises(DiffuseBaseArtifactError):
            _verify(
                root,
                sample=sample,
                config=config,
                treatment=treatment,
                embedding=embedding,
                runtime=runtime,
            )

        assert _bytes_and_mtime(tracked) == before


def test_store_audit_rejects_rehashed_wrong_hnsw_vector(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        store = base.store_path / STORE_DIRECTORY_NAME
        database_path = store / _DATABASE_NAME
        index_path = store / _INDEX_NAME
        database_uri = f"file:{database_path.as_posix()}?mode=ro&immutable=1"
        with sqlite3.connect(database_uri, uri=True) as connection:
            label, embedding_blob = connection.execute(
                "SELECT hnsw_label, embedding FROM chunks "
                "WHERE hnsw_label IS NOT NULL ORDER BY rowid LIMIT 1"
            ).fetchone()
        original = np.frombuffer(embedding_blob, dtype=np.float32).copy()
        assert np.linalg.norm(original) > 0.0
        index = hnswlib.Index(space="cosine", dim=_DIMENSION)
        index.load_index(str(index_path))
        index.add_items(
            (-original)[None, :],
            np.asarray([int(label)], dtype=np.int64),
        )
        index.save_index(str(index_path))
        index = None

        store_manifest_path = base.store_path / STORE_MANIFEST_NAME
        store_manifest = json.loads(store_manifest_path.read_bytes())
        store_manifest["index_sha256"] = file_sha256(index_path)
        store_manifest["index_bytes"] = index_path.stat().st_size
        _rewrite_self_hashed_manifest(store_manifest_path, store_manifest)

        query_manifest_path = base.query_inputs_path / QUERY_MANIFEST_NAME
        query_manifest = json.loads(query_manifest_path.read_bytes())
        query_manifest["base_artifact_sha256"] = store_manifest["artifact_sha256"]
        query_manifest["base_manifest_sha256"] = file_sha256(store_manifest_path)
        query_manifest["index_sha256"] = store_manifest["index_sha256"]
        _rewrite_self_hashed_manifest(query_manifest_path, query_manifest)
        tracked = (index_path, store_manifest_path, query_manifest_path)
        before = _bytes_and_mtime(tracked)

        with pytest.raises(
            DiffuseBaseArtifactError,
            match="HNSW vectors differ from SQLite",
        ):
            _verify(
                root,
                sample=sample,
                config=config,
                treatment=treatment,
                embedding=embedding,
                runtime=runtime,
            )

        assert _bytes_and_mtime(tracked) == before


def test_store_audit_rejects_rehashed_wrong_hnsw_controls(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        store = base.store_path / STORE_DIRECTORY_NAME
        database_path = store / _DATABASE_NAME
        index_path = store / _INDEX_NAME
        database_uri = f"file:{database_path.as_posix()}?mode=ro&immutable=1"
        with sqlite3.connect(database_uri, uri=True) as connection:
            rows = connection.execute(
                "SELECT hnsw_label, embedding FROM chunks "
                "WHERE hnsw_label IS NOT NULL ORDER BY hnsw_label"
            ).fetchall()
        labels = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
        vectors = np.stack(
            [np.frombuffer(row[1], dtype=np.float32) for row in rows]
        )
        index = hnswlib.Index(space="cosine", dim=_DIMENSION)
        index.init_index(
            max_elements=_MAX_ELEMENTS,
            ef_construction=2,
            M=2,
        )
        index.add_items(vectors, labels)
        index.save_index(str(index_path))
        index = None

        store_manifest_path = base.store_path / STORE_MANIFEST_NAME
        store_manifest = json.loads(store_manifest_path.read_bytes())
        store_manifest["index_sha256"] = file_sha256(index_path)
        store_manifest["index_bytes"] = index_path.stat().st_size
        _rewrite_self_hashed_manifest(store_manifest_path, store_manifest)

        query_manifest_path = base.query_inputs_path / QUERY_MANIFEST_NAME
        query_manifest = json.loads(query_manifest_path.read_bytes())
        query_manifest["base_artifact_sha256"] = store_manifest["artifact_sha256"]
        query_manifest["base_manifest_sha256"] = file_sha256(store_manifest_path)
        query_manifest["index_sha256"] = store_manifest["index_sha256"]
        _rewrite_self_hashed_manifest(query_manifest_path, query_manifest)
        tracked = (index_path, store_manifest_path, query_manifest_path)
        before = _bytes_and_mtime(tracked)

        with pytest.raises(
            DiffuseBaseArtifactError,
            match="HNSW controls differ from build-runtime identity",
        ):
            _verify(
                root,
                sample=sample,
                config=config,
                treatment=treatment,
                embedding=embedding,
                runtime=runtime,
            )

        assert _bytes_and_mtime(tracked) == before


def test_store_audit_rejects_rehashed_lexical_posting_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        database_path = base.store_path / STORE_DIRECTORY_NAME / _DATABASE_NAME
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA journal_mode=DELETE")
            term, chunk_id = connection.execute(
                "SELECT term, chunk_id FROM chunk_terms "
                "ORDER BY chunk_id, term LIMIT 1"
            ).fetchone()
            connection.execute(
                "UPDATE chunk_terms SET tf = tf + 37 "
                "WHERE term = ? AND chunk_id = ?",
                (term, chunk_id),
            )
            connection.commit()

        store_manifest_path = base.store_path / STORE_MANIFEST_NAME
        store_manifest = json.loads(store_manifest_path.read_bytes())
        store_manifest["database_sha256"] = file_sha256(database_path)
        store_manifest["database_bytes"] = database_path.stat().st_size
        _rewrite_self_hashed_manifest(store_manifest_path, store_manifest)

        query_manifest_path = base.query_inputs_path / QUERY_MANIFEST_NAME
        query_manifest = json.loads(query_manifest_path.read_bytes())
        query_manifest["base_artifact_sha256"] = store_manifest["artifact_sha256"]
        query_manifest["base_manifest_sha256"] = file_sha256(store_manifest_path)
        query_manifest["database_sha256"] = store_manifest["database_sha256"]
        _rewrite_self_hashed_manifest(query_manifest_path, query_manifest)
        tracked = (database_path, store_manifest_path, query_manifest_path)
        before = _bytes_and_mtime(tracked)

        with pytest.raises(
            DiffuseBaseArtifactError,
            match="lexical index differs from deterministic chunk text",
        ):
            _verify(
                root,
                sample=sample,
                config=config,
                treatment=treatment,
                embedding=embedding,
                runtime=runtime,
            )

        assert _bytes_and_mtime(tracked) == before


def test_store_audit_rejects_rehashed_sqlite_schema_tamper(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        database_path = base.store_path / STORE_DIRECTORY_NAME / _DATABASE_NAME
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("DROP TRIGGER trg_turns_source_insert")
            connection.commit()

        store_manifest_path = base.store_path / STORE_MANIFEST_NAME
        store_manifest = json.loads(store_manifest_path.read_bytes())
        store_manifest["database_sha256"] = file_sha256(database_path)
        store_manifest["database_bytes"] = database_path.stat().st_size
        _rewrite_self_hashed_manifest(store_manifest_path, store_manifest)

        query_manifest_path = base.query_inputs_path / QUERY_MANIFEST_NAME
        query_manifest = json.loads(query_manifest_path.read_bytes())
        query_manifest["base_artifact_sha256"] = store_manifest["artifact_sha256"]
        query_manifest["base_manifest_sha256"] = file_sha256(store_manifest_path)
        query_manifest["database_sha256"] = store_manifest["database_sha256"]
        _rewrite_self_hashed_manifest(query_manifest_path, query_manifest)
        tracked = (database_path, store_manifest_path, query_manifest_path)
        before = _bytes_and_mtime(tracked)

        with pytest.raises(
            DiffuseBaseArtifactError,
            match="SQLite schema differs from the current canonical schema",
        ):
            _verify(
                root,
                sample=sample,
                config=config,
                treatment=treatment,
                embedding=embedding,
                runtime=runtime,
            )

        assert _bytes_and_mtime(tracked) == before


def test_verification_rejects_pointer_symlink_when_platform_allows_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        pointer = base.query_inputs_path / FROZEN_QUERY_INPUTS_NAME
        target = tmp_path / "external-pointer.json"
        target.write_bytes(pointer.read_bytes())
        pointer.unlink()
        try:
            pointer.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"file symlinks are unavailable: {exc}")

        with pytest.raises(DiffuseBaseArtifactError, match="regular file"):
            _verify(
                root,
                sample=sample,
                config=config,
                treatment=treatment,
                embedding=embedding,
                runtime=runtime,
            )


def test_clone_is_a_byte_copy_without_hardlinks_and_never_clobbers(
    tmp_path: Path,
) -> None:
    with _published(tmp_path / "cache") as (base, *_rest):
        destination = tmp_path / "arms" / "fixed"
        clone = clone_diffuse_longmemeval_base(
            base,
            destination,
            arm_id="fixed_interval",
            arm_sha256=_sha("fixed-interval-arm"),
        )
        base_store = base.store_path / STORE_DIRECTORY_NAME

        assert set(path.name for path in destination.iterdir()) == {
            _DATABASE_NAME,
            _INDEX_NAME,
            DERIVED_ORIGIN_NAME,
        }
        for name in (_DATABASE_NAME, _INDEX_NAME):
            source = base_store / name
            copied = destination / name
            assert copied.read_bytes() == source.read_bytes()
            assert not os.path.samefile(source, copied)
        before = _bytes_and_mtime(
            (destination / _DATABASE_NAME, destination / _INDEX_NAME)
        )

        with pytest.raises(FileExistsError):
            clone_diffuse_longmemeval_base(
                base,
                destination,
                arm_id="replacement",
                arm_sha256=_sha("replacement-arm"),
            )

        assert _bytes_and_mtime(
            (destination / _DATABASE_NAME, destination / _INDEX_NAME)
        ) == before
        assert clone.path == destination
        assert clone.origin.initial_database_sha256 == (
            base.store_manifest.database_sha256
        )
        assert clone.origin.initial_index_sha256 == base.store_manifest.index_sha256


def test_clone_rejects_destinations_inside_immutable_artifacts(tmp_path: Path) -> None:
    with _published(tmp_path / "cache") as (base, *_rest):
        store_children = {path.name for path in base.store_path.iterdir()}
        query_children = {path.name for path in base.query_inputs_path.iterdir()}

        for destination in (
            base.store_path / "nested-arm",
            base.query_inputs_path / "nested-arm",
        ):
            with pytest.raises(ValueError, match="overlaps an immutable"):
                clone_diffuse_longmemeval_base(
                    base,
                    destination,
                    arm_id="invalid-overlap",
                    arm_sha256=_sha("invalid-overlap-arm"),
                )
            assert not destination.exists()

        assert {path.name for path in base.store_path.iterdir()} == store_children
        assert {
            path.name for path in base.query_inputs_path.iterdir()
        } == query_children


def test_clone_fails_closed_on_interrupted_or_unowned_destination(
    tmp_path: Path,
) -> None:
    with _published(tmp_path / "cache") as (base, *_rest):
        destination = tmp_path / "arms" / "interrupted"
        destination.mkdir(parents=True)
        partial = destination / _DATABASE_NAME
        partial.write_bytes(b"partial-owned-by-unknown-process")
        before = partial.read_bytes()
        with pytest.raises(FileExistsError):
            clone_diffuse_longmemeval_base(
                base,
                destination,
                arm_id="must-not-clobber",
                arm_sha256=_sha("must-not-clobber"),
            )
        assert {path.name for path in destination.iterdir()} == {_DATABASE_NAME}
        assert partial.read_bytes() == before


def test_derived_open_is_one_shot_and_preserves_hnsw_bytes_after_close(
    tmp_path: Path,
) -> None:
    with _published(tmp_path / "cache") as (base, _sample_value, config, *_rest):
        clone = clone_diffuse_longmemeval_base(
            base,
            tmp_path / "arm",
            arm_id="lexical_embedding",
            arm_sha256=_sha("lexical-embedding-arm"),
        )
        index_path = clone.path / _INDEX_NAME
        index_before = (file_sha256(index_path), index_path.stat().st_mtime_ns)
        database_path = clone.path / _DATABASE_NAME
        database_before = (
            file_sha256(database_path),
            database_path.stat().st_mtime_ns,
        )
        embedder = _DeterministicEmbedder()

        condenser = open_diffuse_longmemeval_derived_store(
            clone,
            config=config,
            embedder=embedder,
        )
        assert condenser._persist_index_on_close is False  # noqa: SLF001
        assert tuple(
            row.receipt_sha256
            for row in condenser.frozen_legacy_query_inputs  # type: ignore[attr-defined]
        ) == tuple(row.receipt_sha256 for row in base.frozen_query_inputs)
        condenser.retriever.save()
        assert (file_sha256(index_path), index_path.stat().st_mtime_ns) == (
            index_before
        )
        condenser.close()

        assert (file_sha256(index_path), index_path.stat().st_mtime_ns) == (
            index_before
        )
        assert (
            file_sha256(database_path),
            database_path.stat().st_mtime_ns,
        ) == database_before
        with pytest.raises(
            DiffuseBaseArtifactError,
            match="already.*claimed",
        ):
            open_diffuse_longmemeval_derived_store(
                clone,
                config=config,
                embedder=embedder,
            )


def test_compiled_clone_mutates_only_sqlite_and_reuses_frozen_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    with _published(root) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        clone = clone_diffuse_longmemeval_base(
            base,
            tmp_path / "compiled-arm",
            arm_id="fixed_interval",
            arm_sha256=_sha("fixed-interval-compiled-arm"),
        )
        base_paths = (
            base.store_path / STORE_DIRECTORY_NAME / _DATABASE_NAME,
            base.store_path / STORE_DIRECTORY_NAME / _INDEX_NAME,
        )
        base_before = _bytes_and_mtime(base_paths)
        derived_database = clone.path / _DATABASE_NAME
        derived_index = clone.path / _INDEX_NAME
        database_before = file_sha256(derived_database)
        index_before = _bytes_and_mtime((derived_index,))
        embedder = _DeterministicEmbedder()

        condenser = open_diffuse_longmemeval_derived_store(
            clone,
            config=config,
            embedder=embedder,
        )
        receipt = compile_diffuse_artifact(
            condenser,
            policy=DiffuseCompilationPolicy(
                boundary_mode="fixed_interval",
                min_episode_size=1,
                max_episode_size=4,
                fixed_interval=2,
            ),
        )
        provider = FrozenLegacyDiffuseInputProvider(
            condenser.frozen_legacy_query_inputs  # type: ignore[attr-defined]
        )
        candidates = provider(
            condenser,
            query=sample.questions[0].retrieval_query,
            retrieval=config.retrieval,
            artifact_id=receipt.artifact.artifact_id,
        )
        condenser.close()

        assert tuple(item.chunk.chunk_id for item in candidates.anchors) == tuple(
            item.chunk.chunk_id for item in base.frozen_query_inputs[0].anchors
        )
        assert file_sha256(derived_database) != database_before
        assert _bytes_and_mtime((derived_index,)) == index_before
        assert _bytes_and_mtime(base_paths) == base_before
        assert not (clone.path / f"{_DATABASE_NAME}-wal").exists()
        assert not (clone.path / f"{_DATABASE_NAME}-shm").exists()
        verified = _verify(
            root,
            sample=sample,
            config=config,
            treatment=treatment,
            embedding=embedding,
            runtime=runtime,
        )
        assert verified.base_store_key == base.base_store_key


def test_short_text_equal_to_route_labels_is_not_mistaken_for_payload(
    tmp_path: Path,
) -> None:
    collision_sample = _sample(
        question="dense",
        turns=(("user", "dense"), ("assistant", "source_tfisf")),
    )
    root = tmp_path / "cache"
    with _published(root, sample=collision_sample) as (
        base,
        sample,
        config,
        treatment,
        embedding,
        runtime,
        *_rest,
    ):
        pointer = json.loads(
            (base.query_inputs_path / FROZEN_QUERY_INPUTS_NAME).read_bytes()
        )
        verified = _verify(
            root,
            sample=sample,
            config=config,
            treatment=treatment,
            embedding=embedding,
            runtime=runtime,
        )

    assert pointer["rows"][0]["anchors"][0]["diagnostics"]["route"] == "dense"
    assert verified.query_input_key == base.query_input_key
    assert verified.frozen_query_inputs[0].query == "dense"
