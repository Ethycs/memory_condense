from __future__ import annotations

import re
import zlib
from datetime import datetime, timezone

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.schemas import Chunk
from memory_condense.eval.diffuse_compilation import (
    DiffuseCompilationPolicy,
    compile_diffuse_artifact,
)


class _Embedder:
    def __init__(self, dimension: int = 32) -> None:
        self._dimension = dimension

    @property
    def dim(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            vector[zlib.crc32(token.encode()) % self._dimension] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]


def _condenser(path) -> MemoryCondenser:
    return MemoryCondenser(
        data_dir=path,
        embedder=_Embedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
        persist_index_on_close=False,
    )


def _ingest(condenser: MemoryCondenser) -> None:
    condenser.ingest(
        "system",
        "[source-a took place at 2024/05/01 (Wed) 12:00]",
        source_id="source-a",
    )
    condenser.ingest(
        "user",
        "The launch badge was amber.",
        source_id="source-a",
    )
    condenser.ingest(
        "assistant",
        "The current badge state is amber.",
        source_id="source-a",
    )
    condenser.ingest(
        "user",
        "The deployment target is ninety five percent accuracy.",
        source_id="source-b",
    )


def _ingest_deterministic(condenser: MemoryCondenser) -> None:
    timestamp = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    condenser.ingest_many(
        [
            (
                "system",
                "[source-a took place at 2024/05/01 (Wed) 12:00]",
                "source-a",
                timestamp,
                "stable-turn-0000",
            ),
            (
                "user",
                "The launch badge was amber.",
                "source-a",
                timestamp,
                "stable-turn-0001",
            ),
            (
                "assistant",
                "The current badge state is amber.",
                "source-a",
                timestamp,
                "stable-turn-0002",
            ),
            (
                "user",
                "The deployment target is ninety five percent accuracy.",
                "source-b",
                timestamp,
                "stable-turn-0003",
            ),
        ]
    )


def test_fixed_interval_compiles_every_chunk_and_replays_identically(tmp_path):
    with _condenser(tmp_path / "fixed") as condenser:
        _ingest(condenser)
        policy = DiffuseCompilationPolicy(
            boundary_mode="fixed_interval",
            min_episode_size=1,
            max_episode_size=4,
            fixed_interval=2,
        )

        first = compile_diffuse_artifact(condenser, policy=policy)
        replay = compile_diffuse_artifact(condenser, policy=policy)

        assert first.receipt_sha256 == replay.receipt_sha256
        assert first.compilation_policy_sha256 == policy.policy_sha256
        assert first.policy_sha256 == first.artifact.policy_sha256
        assert first.persisted_request_token_state_bytes == 0
        assert [item.source_id for item in first.source_receipts] == [
            "source-a",
            "source-b",
        ]
        assert sum(item.content_chunks for item in first.source_receipts) == 3
        assert sum(item.metadata_chunks for item in first.source_receipts) == 1
        assert all(item.episode_ids for item in first.source_receipts)
        assert all(item.unit_ids for item in first.source_receipts)
        assert condenser.discourse.validate_snapshot(first.final_snapshot)
        assert condenser.discourse.artifact_coverage(
            first.artifact.artifact_id,
            "episode",
        ) is not None
        assert condenser.discourse.artifact_coverage(
            first.artifact.artifact_id,
            "discourse",
        ) is not None


def test_fresh_deterministic_ingest_replays_the_full_compilation_receipt(tmp_path):
    policy = DiffuseCompilationPolicy(
        boundary_mode="fixed_interval",
        min_episode_size=1,
        max_episode_size=4,
        fixed_interval=2,
    )

    def compile_at(path):
        with _condenser(path) as condenser:
            _ingest_deterministic(condenser)
            return compile_diffuse_artifact(condenser, policy=policy)

    first = compile_at(tmp_path / "fresh-a")
    replay = compile_at(tmp_path / "fresh-b")

    assert first.receipt_sha256 == replay.receipt_sha256
    assert first.final_snapshot.snapshot_sha256 == (
        replay.final_snapshot.snapshot_sha256
    )


def test_embedding_change_arm_binds_embedding_identity(tmp_path):
    with _condenser(tmp_path / "embedding") as condenser:
        _ingest(condenser)
        receipt = compile_diffuse_artifact(
            condenser,
            policy=DiffuseCompilationPolicy(
                boundary_mode="lexical_embedding",
                min_episode_size=1,
                max_episode_size=4,
                surprise_min_history=1,
            ),
            embedding_identity={
                "model_id": "fixture-embedding",
                "revision": "1",
                "checkpoint_sha256": "e" * 64,
                "dimension": 32,
            },
        )

        assert receipt.artifact.metadata["boundary_policy_id"] == (
            "lexical_embedding"
        )
        assert receipt.artifact.model_id is None
        assert receipt.persisted_request_token_state_bytes == 0


def test_embedding_change_requires_frozen_embedding_identity(tmp_path):
    with _condenser(tmp_path / "missing-identity") as condenser:
        _ingest(condenser)
        with pytest.raises(ValueError, match="frozen embedding identity"):
            compile_diffuse_artifact(
                condenser,
                policy=DiffuseCompilationPolicy(
                    boundary_mode="lexical_embedding",
                    surprise_min_history=1,
                ),
            )
