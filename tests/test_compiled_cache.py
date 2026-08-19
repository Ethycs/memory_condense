from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingService,
)
from memory_condense.eval.compiled_cache import (
    MANIFEST_NAME,
    CompiledStoreCacheError,
    CompiledStoreManifest,
    _embedding_execution_identity,
    _embedding_identity,
    cache_key,
    compiled_store_ingest_fn,
)
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


class CountingEmbedder:
    model_name = "test/counting-v1"
    dim = 8

    def __init__(self) -> None:
        self.chunk_calls = 0

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.frombuffer(digest[:8], dtype=np.uint8).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_chunks(self, chunks):
        self.chunk_calls += 1
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


SAMPLE = BenchmarkSample(
    sample_id="sample/one",
    turns=[
        ("user", "Alpha uses SQLite."),
        ("assistant", "WAL remains enabled."),
    ],
    turn_source_ids=["session-a", "session-a"],
    questions=[
        BenchmarkQuestion(
            question_id="q1",
            question="What does Alpha use?",
            answer="SQLite",
        )
    ],
)


def _manifests(cache: Path) -> list[Path]:
    return list(cache.glob(f"*/{MANIFEST_NAME}"))


def test_cache_identity_binds_embedding_revision_and_checkpoint_manifest():
    class PinnedEmbedder(CountingEmbedder):
        model_revision = "revision-a"
        checkpoint_sha256 = "a" * 64

    embedder = PinnedEmbedder()
    model_identity, dimension = _embedding_identity(embedder)
    config = EvalConfig(retrieval=RetrievalConfig(mode="hybrid"))
    first = cache_key(
        SAMPLE,
        config,
        embedding_model=model_identity,
        embedding_dim=dimension,
    )
    changed_identity = model_identity.replace("a" * 64, "b" * 64)
    second = cache_key(
        SAMPLE,
        config,
        embedding_model=changed_identity,
        embedding_dim=dimension,
    )

    assert model_identity == f"test/counting-v1@revision-a#{'a' * 64}"
    assert first != second


def test_cache_identity_binds_execution_code_and_environment():
    embedder = CountingEmbedder()
    model_identity, dimension = _embedding_identity(embedder)
    execution = _embedding_execution_identity(embedder)
    config = EvalConfig(retrieval=RetrievalConfig(mode="hybrid"))
    baseline = cache_key(
        SAMPLE,
        config,
        embedding_model=model_identity,
        embedding_dim=dimension,
        embedding_execution=execution,
        implementation_digest="a" * 64,
        environment_digest="b" * 64,
    )

    assert baseline != cache_key(
        SAMPLE,
        config,
        embedding_model=model_identity,
        embedding_dim=dimension,
        embedding_execution={**execution, "device": "cuda"},
        implementation_digest="a" * 64,
        environment_digest="b" * 64,
    )
    assert baseline != cache_key(
        SAMPLE,
        config,
        embedding_model=model_identity,
        embedding_dim=dimension,
        embedding_execution=execution,
        implementation_digest="c" * 64,
        environment_digest="b" * 64,
    )
    assert baseline != cache_key(
        SAMPLE,
        config,
        embedding_model=model_identity,
        embedding_dim=dimension,
        embedding_execution=execution,
        implementation_digest="a" * 64,
        environment_digest="d" * 64,
    )


def test_default_cache_identity_uses_exact_pinned_bge_manifest():
    model_identity, dimension = _embedding_identity(EmbeddingService())

    assert model_identity == (
        f"{DEFAULT_MODEL_NAME}@{DEFAULT_MODEL_REVISION}"
        f"#{BGE_M3_CHECKPOINT_SHA256}"
    )
    assert dimension == 1024


def test_compiled_factory_closes_owned_embedder_when_identity_resolution_fails(
    tmp_path: Path,
    monkeypatch,
):
    events: list[str] = []

    class BrokenIdentityEmbedder:
        model_name = "test/broken-identity"

        @property
        def dim(self):
            raise RuntimeError("simulated identity failure")

        def close(self):
            events.append("close")

    monkeypatch.setattr(
        "memory_condense.eval.compiled_cache.EmbeddingService",
        lambda **_kwargs: BrokenIdentityEmbedder(),
    )

    with pytest.raises(RuntimeError, match="identity failure"):
        compiled_store_ingest_fn(tmp_path / "cache")

    assert events == ["close"]


def test_compiled_store_builds_once_then_reopens_without_embedding(tmp_path: Path):
    cache = tmp_path / "cache"
    embedder = CountingEmbedder()
    ingest = compiled_store_ingest_fn(cache, embedder=embedder)
    config = EvalConfig(retrieval=RetrievalConfig(mode="hybrid"))

    first = ingest(SAMPLE, config, tmp_path / "ignored-1")
    assert first.transcript.count() == 2
    database_path = first.database_path
    receipt = first.compiled_cache_receipt
    with pytest.raises(sqlite3.OperationalError):
        first.ingest("user", "A compiled cache reader must reject writes.")
    first.close()
    assert embedder.chunk_calls == 1
    manifests = _manifests(cache)
    assert len(manifests) == 1
    manifest = CompiledStoreManifest.model_validate_json(
        manifests[0].read_text(encoding="utf-8")
    )
    assert manifest.turn_count == 2
    assert manifest.chunk_count == 2
    assert receipt == {
        "manifest_sha256": hashlib.sha256(
            manifests[0].read_bytes()
        ).hexdigest(),
        "cache_key": manifest.cache_key,
        "sample_sha256": manifest.sample_sha256,
        "database_sha256": manifest.database_sha256,
        "index_sha256": manifest.index_sha256,
        "embedding_execution_sha256": canonical_sha256(
            manifest.embedding_execution
        ),
        "implementation_sha256": manifest.implementation_sha256,
        "environment_lock_sha256": manifest.environment_lock_sha256,
        "turn_count": 2,
        "chunk_count": 2,
    }
    database_before = database_path.read_bytes()

    index_path = manifests[0].parent / "hnsw_index.bin"
    before = hashlib.sha256(index_path.read_bytes()).hexdigest()
    second = ingest(
        SAMPLE,
        config.model_copy(update={"retrieval": RetrievalConfig(mode="dense")}),
        tmp_path / "ignored-2",
    )
    assert second.transcript.count() == 2
    second.close()

    assert embedder.chunk_calls == 1
    assert len(_manifests(cache)) == 1
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == before
    assert database_path.read_bytes() == database_before
    assert not database_path.with_name(f"{database_path.name}-wal").exists()
    assert not database_path.with_name(f"{database_path.name}-shm").exists()


def test_required_compiled_cache_hit_never_creates_or_builds(tmp_path: Path):
    root = tmp_path / "missing-cache"
    with pytest.raises(CompiledStoreCacheError, match="root does not exist"):
        compiled_store_ingest_fn(
            root,
            embedder=CountingEmbedder(),
            require_cache_hit=True,
        )
    assert not root.exists()

    root.mkdir()
    embedder = CountingEmbedder()
    ingest = compiled_store_ingest_fn(
        root,
        embedder=embedder,
        require_cache_hit=True,
    )
    with pytest.raises(CompiledStoreCacheError, match="entry is missing"):
        ingest(
            SAMPLE,
            EvalConfig(retrieval=RetrievalConfig(mode="hybrid")),
            tmp_path / "ignored",
        )
    assert embedder.chunk_calls == 0
    assert list(root.iterdir()) == []


def test_compiled_store_key_changes_with_sample_or_chunker(tmp_path: Path):
    cache = tmp_path / "cache"
    embedder = CountingEmbedder()
    ingest = compiled_store_ingest_fn(cache, embedder=embedder)
    config = EvalConfig(retrieval=RetrievalConfig(mode="hybrid"))
    changed = SAMPLE.model_copy(
        update={
            "turns": [("user", "Alpha uses Postgres.")],
            "turn_source_ids": ["session-a"],
        }
    )

    ingest(SAMPLE, config, tmp_path / "a").close()
    ingest(changed, config, tmp_path / "b").close()
    ingest(
        SAMPLE,
        config.model_copy(
            update={"chunker": config.chunker.model_copy(update={"max_tokens": 249})}
        ),
        tmp_path / "c",
    ).close()

    assert embedder.chunk_calls == 3
    assert len(_manifests(cache)) == 3


def test_compiled_store_rejects_hash_mismatch(tmp_path: Path):
    cache = tmp_path / "cache"
    ingest = compiled_store_ingest_fn(cache, embedder=CountingEmbedder())
    config = EvalConfig(retrieval=RetrievalConfig(mode="hybrid"))
    ingest(SAMPLE, config, tmp_path / "a").close()
    database_path = _manifests(cache)[0].parent / "memory.db"
    with database_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(CompiledStoreCacheError, match="SQLite hash mismatch"):
        ingest(SAMPLE, config, tmp_path / "b")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("cache_revision", 1, "revision mismatch"),
        ("turn_count", 999, "count metadata mismatch"),
        ("schema_version", 1, "schema metadata mismatch"),
    ),
)
def test_compiled_store_rejects_false_manifest_metadata(
    tmp_path: Path,
    field,
    value,
    error,
):
    cache = tmp_path / "cache"
    ingest = compiled_store_ingest_fn(cache, embedder=CountingEmbedder())
    config = EvalConfig(retrieval=RetrievalConfig(mode="hybrid"))
    ingest(SAMPLE, config, tmp_path / "a").close()
    manifest_path = _manifests(cache)[0]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CompiledStoreCacheError, match=error):
        ingest(SAMPLE, config, tmp_path / "b")


def test_compiled_store_rejects_extracting_memory_mode(tmp_path: Path):
    ingest = compiled_store_ingest_fn(
        tmp_path / "cache", embedder=CountingEmbedder()
    )
    config = EvalConfig(retrieval=RetrievalConfig(mode="memory"))

    with pytest.raises(ValueError, match="non-extracting"):
        ingest(SAMPLE, config, tmp_path / "ignored")


def test_compiled_store_uses_manifest_last_fallback_when_rename_is_denied(
    tmp_path: Path, monkeypatch
):
    cache = tmp_path / "cache"
    ingest = compiled_store_ingest_fn(cache, embedder=CountingEmbedder())
    config = EvalConfig(retrieval=RetrievalConfig(mode="hybrid"))

    def denied(_self, _target):
        raise PermissionError("simulated Windows directory lock")

    monkeypatch.setattr(Path, "rename", denied)
    ingest(SAMPLE, config, tmp_path / "ignored").close()

    manifests = _manifests(cache)
    assert len(manifests) == 1
    assert not list(cache.glob(".building-*"))
    assert CompiledStoreManifest.model_validate_json(
        manifests[0].read_text(encoding="utf-8")
    ).sample_id == SAMPLE.sample_id


def test_failed_ingest_closes_store_before_windows_cleanup(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache"
    ingest = compiled_store_ingest_fn(cache, embedder=CountingEmbedder())
    config = EvalConfig(retrieval=RetrievalConfig(mode="hybrid"))

    def fail_ingest(_self, _records):
        raise ValueError("simulated corpus failure")

    monkeypatch.setattr(
        "memory_condense.eval.benchmark.MemoryCondenser.ingest_many",
        fail_ingest,
    )
    with pytest.raises(ValueError, match="simulated corpus failure"):
        ingest(SAMPLE, config, tmp_path / "ignored")

    assert not list(cache.glob(".building-*"))
    assert _manifests(cache) == []
