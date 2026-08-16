"""Local runtime probes for the post-B0 hot paths. No model or network."""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from memory_condense.db import Database
from memory_condense.memory_store import MemoryStore
from memory_condense.retrieval import SimilarityRetriever
from memory_condense.schemas import CreateOp, MemoryType, Provenance
from memory_condense.transcript_store import TranscriptStore


DIM = 64


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def memory_probe(items: int = 200, repeats: int = 11) -> None:
    """Measure ranked-memory reads and count SQLite commits per call."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "memory.db")
        turn = TranscriptStore(db).append("user", "performance probe")
        store = MemoryStore(db)
        for i in range(items):
            vector = np.zeros(DIM, dtype=np.float32)
            vector[i % DIM] = 1.0
            store.create(
                CreateOp(
                    type=MemoryType.DECISION,
                    content=f"memory item {i}",
                    provenance=[
                        Provenance(turn_id=turn.turn_id, quote="performance probe")
                    ],
                    importance=(i % 10) / 10,
                ),
                embedding=vector.tolist(),
            )

        query = np.zeros(DIM, dtype=np.float32)
        query[0] = 1.0
        for k in (8, 50):
            timings: list[float] = []
            commit_counts: list[int] = []
            for repeat in range(repeats):
                commits = 0

                def trace(sql: str) -> None:
                    nonlocal commits
                    if sql.strip().upper() == "COMMIT":
                        commits += 1

                db.connection.set_trace_callback(trace)
                started = time.perf_counter()
                store.retrieve(query, k=k, now_turn=100 + repeat)
                timings.append(time.perf_counter() - started)
                commit_counts.append(commits)
            db.connection.set_trace_callback(None)
            print(
                f"memory retrieve k={k}: median={_median_ms(timings):.3f} ms, "
                f"commits/call={statistics.median(commit_counts):g}"
            )
        db.close()


def span_probe(chunks: int = 20_000, repeats: int = 7) -> None:
    """Compare incremental append refresh with the former full rebuild."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "spans.db")
        turn = TranscriptStore(db).append("user", "span performance probe")
        vectors = []
        for i in range(DIM):
            vector = np.zeros(DIM, dtype=np.float32)
            vector[i] = 1.0
            vectors.append(vector.tobytes())

        db.executemany(
            "INSERT INTO chunks "
            "(chunk_id, turn_id, text, start_char, end_char, token_count, "
            " embedding, hnsw_label) VALUES (?, ?, ?, 0, 1, 10, ?, ?)",
            [
                (f"chunk-{i}", turn.turn_id, f"chunk {i}", vectors[i % DIM], i)
                for i in range(chunks)
            ],
        )
        db.commit()
        retriever = SimilarityRetriever(
            db=db, dim=DIM, index_path=Path(tmp) / "spans.bin"
        )
        level = 220
        retriever._span_vectors(level)

        incremental: list[float] = []
        rebuild: list[float] = []
        next_label = chunks
        for repeat in range(repeats):
            batch = []
            for offset in range(8):
                label = next_label + offset
                batch.append(
                    (
                        f"chunk-{label}",
                        turn.turn_id,
                        f"chunk {label}",
                        vectors[label % DIM],
                        label,
                    )
                )
            db.executemany(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, token_count, "
                " embedding, hnsw_label) VALUES (?, ?, ?, 0, 1, 10, ?, ?)",
                batch,
            )
            db.commit()
            next_label += len(batch)

            started = time.perf_counter()
            retriever._span_vectors(level)
            incremental.append(time.perf_counter() - started)

            retriever._clear_span_cache()
            started = time.perf_counter()
            retriever._span_vectors(level)
            rebuild.append(time.perf_counter() - started)

        inc_ms = _median_ms(incremental)
        rebuild_ms = _median_ms(rebuild)
        print(
            f"span append refresh ({chunks:,} chunks): incremental={inc_ms:.3f} ms, "
            f"full-rebuild={rebuild_ms:.3f} ms, speedup={rebuild_ms / inc_ms:.1f}x"
        )
        db.close()


if __name__ == "__main__":
    memory_probe()
    span_probe()
