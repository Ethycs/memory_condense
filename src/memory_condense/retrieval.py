from __future__ import annotations

import json
import struct
from pathlib import Path

import hnswlib
import numpy as np

from memory_condense.db import Database
from memory_condense.schemas import Chunk, RetrievalResult, Turn


class SimilarityRetriever:
    """Dense cosine similarity retrieval using hnswlib."""

    def __init__(
        self,
        db: Database,
        dim: int = 1024,
        index_path: str | Path | None = None,
        ef_construction: int = 200,
        M: int = 16,
        max_elements: int = 100_000,
    ) -> None:
        self._db = db
        self._dim = dim
        self._index_path = Path(index_path) if index_path else None
        self._ef_construction = ef_construction
        self._M = M
        self._max_elements = max_elements

        # label <-> chunk_id mapping
        self._label_to_chunk_id: dict[int, str] = {}
        self._chunk_id_to_label: dict[str, int] = {}
        self._next_label = 0

        self._index: hnswlib.Index | None = None
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load index from file if it exists, otherwise create empty."""
        self._index = hnswlib.Index(space="cosine", dim=self._dim)

        if self._index_path and self._index_path.exists():
            self._index.load_index(str(self._index_path))
            self._load_label_mapping()
        else:
            self._index.init_index(
                max_elements=self._max_elements,
                ef_construction=self._ef_construction,
                M=self._M,
            )
            # Load mapping from DB if available
            self._load_label_mapping()

    def _load_label_mapping(self) -> None:
        """Load label<->chunk_id mapping from the chunks table."""
        cur = self._db.execute(
            "SELECT chunk_id, hnsw_label FROM chunks WHERE hnsw_label IS NOT NULL"
        )
        for chunk_id, label in cur.fetchall():
            self._label_to_chunk_id[label] = chunk_id
            self._chunk_id_to_label[chunk_id] = label
            if label >= self._next_label:
                self._next_label = label + 1

    def _assign_label(self, chunk_id: str) -> int:
        """Assign a new integer label for a chunk."""
        label = self._next_label
        self._next_label += 1
        self._label_to_chunk_id[label] = chunk_id
        self._chunk_id_to_label[chunk_id] = label
        return label

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add embedded chunks to the ANN index and persist to SQLite.

        Chunks must have non-None embedding fields. Idempotent:
        chunks already in the index are skipped.
        """
        if not chunks:
            return

        new_chunks = [
            c for c in chunks
            if c.chunk_id not in self._chunk_id_to_label and c.embedding is not None
        ]

        if not new_chunks:
            return

        # Resize index if needed
        current_count = self._index.get_current_count()
        needed = current_count + len(new_chunks)
        if needed > self._index.get_max_elements():
            self._index.resize_index(max(needed * 2, self._max_elements))

        labels: list[int] = []
        vectors: list[np.ndarray] = []

        for chunk in new_chunks:
            label = self._assign_label(chunk.chunk_id)
            labels.append(label)
            vectors.append(np.array(chunk.embedding, dtype=np.float32))

            # Persist chunk + embedding to SQLite
            embedding_blob = np.array(chunk.embedding, dtype=np.float32).tobytes()
            lexical_json = (
                json.dumps(chunk.lexical_weights) if chunk.lexical_weights else None
            )
            self._db.execute(
                "INSERT OR IGNORE INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, "
                "token_count, embedding, lexical_weights, hnsw_label) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.chunk_id,
                    chunk.turn_id,
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.token_count,
                    embedding_blob,
                    lexical_json,
                    label,
                ),
            )

        self._db.commit()

        data = np.stack(vectors)
        self._index.add_items(data, np.array(labels, dtype=np.int64))

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        ef_search: int = 50,
    ) -> list[RetrievalResult]:
        """Find the k most similar chunks to the query embedding."""
        if self._index.get_current_count() == 0:
            return []

        k = min(k, self._index.get_current_count())
        self._index.set_ef(max(ef_search, k))

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        labels_arr, distances_arr = self._index.knn_query(query_vec, k=k)

        results: list[RetrievalResult] = []
        for label, distance in zip(labels_arr[0], distances_arr[0]):
            label = int(label)
            chunk_id = self._label_to_chunk_id.get(label)
            if chunk_id is None:
                continue

            # Hydrate chunk from SQLite
            chunk = self._load_chunk(chunk_id)
            if chunk is None:
                continue

            # Hydrate turn
            turn = self._load_turn(chunk.turn_id)

            # hnswlib cosine distance = 1 - cosine_similarity
            score = 1.0 - float(distance)

            results.append(RetrievalResult(chunk=chunk, score=score, turn=turn))

        return results

    def rebuild_index(self) -> None:
        """Rebuild the hnswlib index from all embeddings in SQLite."""
        cur = self._db.execute(
            "SELECT chunk_id, embedding, hnsw_label FROM chunks "
            "WHERE embedding IS NOT NULL"
        )
        rows = cur.fetchall()

        self._index = hnswlib.Index(space="cosine", dim=self._dim)
        max_el = max(len(rows), self._max_elements)
        self._index.init_index(
            max_elements=max_el,
            ef_construction=self._ef_construction,
            M=self._M,
        )

        self._label_to_chunk_id.clear()
        self._chunk_id_to_label.clear()
        self._next_label = 0

        if not rows:
            return

        labels: list[int] = []
        vectors: list[np.ndarray] = []

        for chunk_id, emb_blob, hnsw_label in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            label = hnsw_label if hnsw_label is not None else self._assign_label(chunk_id)

            if hnsw_label is not None:
                self._label_to_chunk_id[label] = chunk_id
                self._chunk_id_to_label[chunk_id] = label
                if label >= self._next_label:
                    self._next_label = label + 1
            else:
                self._assign_label(chunk_id)

            labels.append(label)
            vectors.append(vec)

        data = np.stack(vectors)
        self._index.add_items(data, np.array(labels, dtype=np.int64))

    def save(self) -> None:
        """Persist the hnswlib index to disk."""
        if self._index_path and self._index is not None:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save_index(str(self._index_path))

    def _load_chunk(self, chunk_id: str) -> Chunk | None:
        """Load a chunk from SQLite by ID."""
        cur = self._db.execute(
            "SELECT chunk_id, turn_id, text, start_char, end_char, "
            "token_count, embedding, lexical_weights "
            "FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        embedding = None
        if row[6] is not None:
            embedding = np.frombuffer(row[6], dtype=np.float32).tolist()

        lexical_weights = None
        if row[7] is not None:
            lexical_weights = json.loads(row[7])

        return Chunk(
            chunk_id=row[0],
            turn_id=row[1],
            text=row[2],
            start_char=row[3],
            end_char=row[4],
            token_count=row[5],
            embedding=embedding,
            lexical_weights=lexical_weights,
        )

    def _load_turn(self, turn_id: str) -> Turn | None:
        """Load a turn from SQLite by ID."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, created_at FROM turns WHERE turn_id = ?",
            (turn_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return Turn(turn_id=row[0], role=row[1], text=row[2], created_at=row[3])
