from __future__ import annotations

import re
import sqlite3
import zlib

import numpy as np
import pytest

from memory_condense.condenser import MemoryCondenser
from memory_condense.eval.consolidation_replay import (
    FrozenQueryEmbedder,
    _source_rows,
    _comparison,
    stage_causal_store,
)
from memory_condense.schemas import Chunk


class ReplayEmbedder:
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % self._dim] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_queries(self, texts):
        return np.stack([self.embed_query(text) for text in texts])

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embedding": self.embed_query(chunk.text).tolist()})
            for chunk in chunks
        ]


def test_frozen_query_embedder_refuses_unbatched_queries():
    embedder = FrozenQueryEmbedder({"known": [1.0, 0.0]})
    assert embedder.dim == 2
    assert embedder.embed_query("known").tolist() == [1.0, 0.0]
    with pytest.raises(KeyError, match="frozen batch"):
        embedder.embed_query("unknown")


def test_source_reader_accepts_pre_source_id_historical_store(tmp_path):
    database = tmp_path / "historical.db"
    embedding = np.asarray([1.0, 0.0], dtype=np.float32).tobytes()
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE turns (
                turn_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                text TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                embedding BLOB,
                lexical_weights TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO turns VALUES ('t1', 'user', 'old prompt', 'then')"
        )
        connection.execute(
            "INSERT INTO chunks VALUES ('c1', 't1', 'old prompt', 0, 10, 2, ?, NULL)",
            (embedding,),
        )

    rows = _source_rows(database)
    assert rows[0][:4] == (1, "user", "old prompt", None)
    assert rows[0][4][0].chunk_id == "c1"


def test_causal_staging_never_retrieves_the_current_or_future_turn(tmp_path):
    source = tmp_path / "source"
    live = ReplayEmbedder()
    turns = [
        ("assistant", "alpha evidence establishes the first durable fact"),
        ("user", "What does alpha establish?"),
        ("assistant", "beta evidence is revealed only after that question"),
        ("user", "How do alpha and beta relate?"),
    ]
    with MemoryCondenser(
        data_dir=source,
        embedder=live,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=40,
    ) as condenser:
        for role, text in turns:
            condenser.ingest(role, text)

    queries = [text for role, text in turns if role == "user"]
    frozen = FrozenQueryEmbedder(
        {query: live.embed_query(query) for query in queries}
    )
    events, stats = stage_causal_store(
        source / "memory.db",
        tmp_path / "staged",
        frozen,
        retrieval_k=3,
        max_event_nodes=3,
        new_event_nodes=1,
        max_prompt_tokens=128,
    )

    # The completed first user/assistant episode binds its one prior anchor to
    # the newly revealed beta response. The second user has no response, so it
    # cannot manufacture a future member or teach from its own query text.
    assert stats["events"] == 1
    assert events[0].event_id == "causal-user:2:part:0"
    assert len(events[0].causal_chunk_ids) == 1
    with sqlite3.connect(tmp_path / "staged" / "memory.db") as connection:
        selected_text = [
            connection.execute(
                "SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()[0]
            for chunk_id in events[0].chunk_ids
        ]
    assert any("alpha evidence" in text for text in selected_text)
    assert any("beta evidence" in text for text in selected_text)
    assert all("How do alpha and beta relate?" not in text for text in selected_text)


def test_comparison_reports_gains_and_losses_without_question_text():
    baseline = {
        "literal_recall": 0.5,
        "mean_context_tokens": 100.0,
        "rows": [
            {"question_id": "q0", "hit": True},
            {"question_id": "q1", "hit": False},
        ],
    }
    treatment = {
        "literal_recall": 0.5,
        "mean_context_tokens": 90.0,
        "rows": [
            {"question_id": "q0", "hit": False},
            {"question_id": "q1", "hit": True},
        ],
    }
    result = _comparison(treatment, baseline)
    assert result["gained_question_ids"] == ["q1"]
    assert result["lost_question_ids"] == ["q0"]
    assert result["mean_context_token_delta"] == -10.0
