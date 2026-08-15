from __future__ import annotations

import json
import struct
from pathlib import Path

import hnswlib
import numpy as np

from memory_condense import lexical, ranking
from memory_condense.db import Database
from memory_condense.lexical import LexicalIndex
from memory_condense.schemas import Chunk, RetrievalResult, Turn


class SimilarityRetriever:
    """Chunk retrieval over hnswlib (dense) and BM25 (lexical).

    ``query`` is the pure-dense baseline path and is kept deliberately
    untouched — the eval ablations compare against it. ``hybrid_query``
    unions dense and lexical candidates and reranks them with
    ``ranking.blend_hybrid``.
    """

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
        #: Labels marked deleted in this process; hnswlib still counts them.
        self._deleted_labels: set[int] = set()

        self._lexical = LexicalIndex(db)

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

        The same chunks are also written to the BM25 inverted index, and
        ``chunks.lexical_weights`` is filled with the chunk's term-frequency
        map (computed from the text when the Chunk does not carry one).
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
        indexed: list[Chunk] = []

        for chunk in new_chunks:
            label = self._assign_label(chunk.chunk_id)
            labels.append(label)
            vectors.append(np.array(chunk.embedding, dtype=np.float32))

            # The lexical_weights column stores the chunk's term frequencies;
            # derive them from the text when the caller did not supply any.
            weights: dict = (
                dict(chunk.lexical_weights)
                if chunk.lexical_weights
                else lexical.term_frequencies(chunk.text)
            )
            indexed.append(chunk.model_copy(update={"lexical_weights": weights}))

            # Persist chunk + embedding to SQLite. The row may already exist
            # (e.g. written by the lexical index first), so fill in the dense
            # columns on conflict instead of silently dropping the embedding.
            embedding_blob = np.array(chunk.embedding, dtype=np.float32).tobytes()
            lexical_json = json.dumps(weights) if weights else None
            self._db.execute(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, "
                "token_count, embedding, lexical_weights, hnsw_label) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET "
                "embedding = excluded.embedding, "
                "lexical_weights = excluded.lexical_weights, "
                "hnsw_label = excluded.hnsw_label",
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

        self._lexical.add_chunks(indexed)

    def _live_count(self) -> int:
        """Elements hnswlib will actually return (excludes deleted labels)."""
        if self._index is None:
            return 0
        return max(0, self._index.get_current_count() - len(self._deleted_labels))

    def _dense_candidates(
        self,
        query_embedding: np.ndarray,
        k: int,
        ef_search: int,
    ) -> list[tuple[str, float]]:
        """``(chunk_id, cosine_similarity)`` for the k nearest neighbours."""
        live = self._live_count()
        if live == 0 or k <= 0:
            return []

        k = min(k, live)
        self._index.set_ef(max(ef_search, k))

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        labels_arr, distances_arr = self._index.knn_query(query_vec, k=k)

        out: list[tuple[str, float]] = []
        for label, distance in zip(labels_arr[0], distances_arr[0]):
            chunk_id = self._label_to_chunk_id.get(int(label))
            if chunk_id is None:
                continue
            # hnswlib cosine distance = 1 - cosine_similarity
            out.append((chunk_id, 1.0 - float(distance)))
        return out

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        ef_search: int = 50,
    ) -> list[RetrievalResult]:
        """Find the k most similar chunks to the query embedding (dense only).

        This is the baseline retrieval path; ``hybrid_query`` layers BM25 on
        top of it without changing this behaviour.
        """
        results: list[RetrievalResult] = []
        for chunk_id, score in self._dense_candidates(query_embedding, k, ef_search):
            hydrated = self._hydrate(chunk_id, score=score)
            if hydrated is not None:
                results.append(hydrated)
        return results

    def hybrid_query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 10,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Dense + lexical retrieval, blended into a single ranking.

        ``candidates`` chunks are pulled from each side (hnswlib for dense,
        BM25 for lexical) and unioned. Each side's raw scores are min-max
        normalised over the candidates it produced; a chunk missing from one
        side scores 0 there. The two are combined with
        ``ranking.blend_hybrid(dense, lexical, alpha)``, where ``alpha`` is the
        dense weight — ``alpha=1.0`` reproduces the dense ordering and
        ``alpha=0.0`` gives pure BM25.

        The returned results carry the blended value in ``score`` and the
        normalised components in ``dense_score`` / ``lexical_score``, so
        ``score == blend_hybrid(dense_score, lexical_score, alpha)`` exactly.
        """
        if k <= 0:
            return []

        dense_hits = self._dense_candidates(query_embedding, candidates, ef_search)
        lexical_hits = self._lexical.search(query_text, limit=candidates)

        dense_norm = ranking.min_max_normalize([s for _, s in dense_hits])
        lexical_norm = ranking.min_max_normalize([s for _, s in lexical_hits])

        dense_scores = {cid: n for (cid, _), n in zip(dense_hits, dense_norm)}
        lexical_scores = {cid: n for (cid, _), n in zip(lexical_hits, lexical_norm)}

        # Deterministic union order: dense candidates in rank order, then the
        # lexical-only ones in rank order. Ties in the blended score therefore
        # resolve the same way on every run.
        ordered_ids: list[str] = [cid for cid, _ in dense_hits]
        seen = set(ordered_ids)
        for cid, _ in lexical_hits:
            if cid not in seen:
                seen.add(cid)
                ordered_ids.append(cid)

        scored: list[tuple[float, str]] = []
        for chunk_id in ordered_ids:
            d = dense_scores.get(chunk_id, 0.0)
            lx = lexical_scores.get(chunk_id, 0.0)
            scored.append((ranking.blend_hybrid(d, lx, alpha), chunk_id))

        results: list[RetrievalResult] = []
        for blended, candidate_id in ranking.top_k(scored, k):
            chunk_id = str(candidate_id)
            hydrated = self._hydrate(
                chunk_id,
                score=blended,
                dense_score=dense_scores.get(chunk_id, 0.0),
                lexical_score=lexical_scores.get(chunk_id, 0.0),
            )
            if hydrated is not None:
                results.append(hydrated)
        return results

    def delete_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from both indexes. Returns False if it was unknown.

        SQLite is authoritative: the chunk's postings and its embedding /
        hnsw label are cleared, so a later ``rebuild_index`` (or a fresh
        process) never sees it again. The in-memory hnswlib graph is updated
        best-effort with ``mark_deleted``; the chunk row itself is kept so
        memory provenance pointing at it does not dangle.
        """
        label = self._chunk_id_to_label.pop(chunk_id, None)
        if label is not None:
            self._label_to_chunk_id.pop(label, None)
            try:
                self._index.mark_deleted(label)
                self._deleted_labels.add(label)
            except (RuntimeError, ValueError):
                # Already deleted or absent from this index instance.
                pass

        cur = self._db.execute(
            "UPDATE chunks SET embedding = NULL, hnsw_label = NULL "
            "WHERE chunk_id = ?",
            (chunk_id,),
        )
        touched = cur.rowcount > 0
        self._db.commit()
        self._lexical.delete_chunk(chunk_id)
        return touched or label is not None

    @property
    def lexical(self) -> LexicalIndex:
        """The BM25 index backing the lexical half of ``hybrid_query``."""
        return self._lexical

    def _hydrate(
        self,
        chunk_id: str,
        score: float,
        dense_score: float | None = None,
        lexical_score: float | None = None,
    ) -> RetrievalResult | None:
        """Build a RetrievalResult from SQLite, or None if the chunk is gone."""
        chunk = self._load_chunk(chunk_id)
        if chunk is None:
            return None
        return RetrievalResult(
            chunk=chunk,
            score=score,
            turn=self._load_turn(chunk.turn_id),
            dense_score=dense_score,
            lexical_score=lexical_score,
        )

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
        self._deleted_labels.clear()
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
