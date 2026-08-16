from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from memory_condense.eval.compiled_cache import (
    MANIFEST_NAME,
    CompiledStoreCacheError,
    CompiledStoreManifest,
    compiled_store_ingest_fn,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample


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


def test_compiled_store_builds_once_then_reopens_without_embedding(tmp_path: Path):
    cache = tmp_path / "cache"
    embedder = CountingEmbedder()
    ingest = compiled_store_ingest_fn(cache, embedder=embedder)
    config = EvalConfig(retrieval=RetrievalConfig(mode="hybrid"))

    first = ingest(SAMPLE, config, tmp_path / "ignored-1")
    assert first.transcript.count() == 2
    first.close()
    assert embedder.chunk_calls == 1
    manifests = _manifests(cache)
    assert len(manifests) == 1
    manifest = CompiledStoreManifest.model_validate_json(
        manifests[0].read_text(encoding="utf-8")
    )
    assert manifest.turn_count == 2
    assert manifest.chunk_count == 2

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
