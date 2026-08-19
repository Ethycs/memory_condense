"""Stateful ANN/lexical index allocation, synchronization, and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import hnswlib
import numpy as np

from memory_condense.associations.association_store import AssociationStore
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.search.indexes import lexical
from memory_condense.search.indexes.lexical import LexicalIndex
from memory_condense.search.indexes.source_hierarchy import SourceContractionIndex


# Labels are shared by every process open on a store and therefore allocated
# through SQLite instead of a per-process counter.
_LABEL_KEY = "next_hnsw_label"


class IndexLifecycleMixin:
    """Internal stateful methods composed by ``SimilarityRetriever``."""

    def __init__(
        self,
        db: Database,
        dim: int = 1024,
        index_path: str | Path | None = None,
        ef_construction: int = 200,
        M: int = 16,
        max_elements: int = 100_000,
        association_store: AssociationStore | None = None,
    ) -> None:
        self._db = db
        self._dim = dim
        self._index_path = Path(index_path) if index_path else None
        self._ef_construction = ef_construction
        self._M = M
        self._max_elements = max_elements
        self._associations = association_store

        # label <-> chunk_id mapping
        self._label_to_chunk_id: dict[int, str] = {}
        self._chunk_id_to_label: dict[str, int] = {}
        #: Labels marked deleted in this process; hnswlib still counts them.
        self._deleted_labels: set[int] = set()
        #: Derived span vectors, per level.  The first query builds them; later
        #: appends update only the open tail span.  Rebuilding every level over
        #: the full transcript after every turn made the span path O(N^2) over
        #: a session despite its bounded output.
        self._span_cache: dict[int, tuple[np.ndarray, list[list[str]]]] = {}
        # Capacity-backed arrays make starting a new span amortized O(1).  A
        # naive ``vstack`` would copy every old vector on each append and put
        # the same quadratic schedule back under a different name.
        self._span_vector_buffers: dict[int, np.ndarray] = {}
        self._span_tail_sums: dict[int, np.ndarray] = {}
        self._span_tail_tokens: dict[int, int] = {}
        self._span_cached_through_rowid: dict[int, int] = {}

        self._lexical = LexicalIndex(db)
        self._source_hierarchy = SourceContractionIndex(db, dim=dim)

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

    def release(self) -> None:
        """Release native ANN buffers without mutating the persisted index.

        ``hnswlib.Index`` owns capacity-sized native allocations. Closing only
        SQLite leaves those buffers live as long as the facade remains in a
        Python frame, which is especially costly for sequential benchmark
        samples. Dropping the final reference is deterministic and idempotent.
        """

        self._index = None
        self._span_cache.clear()
        self._span_vector_buffers.clear()
        self._span_tail_sums.clear()
        self._span_tail_tokens.clear()
        self._span_cached_through_rowid.clear()
        self._source_hierarchy.invalidate()

    def _load_label_mapping(self) -> None:
        """Load label<->chunk_id mapping from the chunks table."""
        cur = self._db.execute(
            "SELECT chunk_id, hnsw_label FROM chunks WHERE hnsw_label IS NOT NULL"
        )
        for chunk_id, label in cur.fetchall():
            self._label_to_chunk_id[label] = chunk_id
            self._chunk_id_to_label[chunk_id] = label

    def _allocate_labels(self, count: int) -> list[int]:
        """Reserve `count` labels that no other process can also hand out.

        hnswlib labels are persisted in ``chunks.hnsw_label``, which is UNIQUE,
        so they are shared state across every process open on the store. A
        per-process counter (the previous approach) gave two concurrent
        sessions the same labels and crashed the second writer with an
        IntegrityError on its first ingest.

        The counter therefore lives in ``meta`` and is bumped inside the
        implicit write transaction, which holds SQLite's write lock until the
        commit below. The ``MAX(...)`` guard also repairs stores written
        before this counter existed, where the counter would otherwise start
        behind the labels already on disk.
        """
        if count <= 0:
            return []

        conn = self._db.connection
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) "
            "SELECT ?, CAST(COALESCE(MAX(hnsw_label), -1) + 1 AS TEXT) FROM chunks",
            (_LABEL_KEY,),
        )
        conn.execute(
            "UPDATE meta SET value = CAST("
            "  MAX(CAST(value AS INTEGER),"
            "      (SELECT COALESCE(MAX(hnsw_label), -1) + 1 FROM chunks)) + ?"
            " AS TEXT) WHERE key = ?",
            (count, _LABEL_KEY),
        )
        end = int(
            conn.execute(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key = ?", (_LABEL_KEY,)
            ).fetchone()[0]
        )
        conn.commit()

        start = end - count
        return list(range(start, end))

    def _register(self, chunk_id: str, label: int) -> None:
        self._label_to_chunk_id[label] = chunk_id
        self._chunk_id_to_label[chunk_id] = label

    def _sync_from_db(self) -> int:
        """Adopt chunks another process indexed since we last looked.

        Each process keeps its own in-memory hnswlib graph, so without this a
        second session's writes stay invisible until restart. SQLite is the
        source of truth (MC-STD-DATA clause 1), so we reconcile against it.
        Returns the number of vectors adopted.
        """
        if self._index is None:
            return 0

        on_disk = self._db.execute(
            "SELECT COUNT(*) FROM chunks "
            "WHERE hnsw_label IS NOT NULL AND embedding IS NOT NULL"
        ).fetchone()[0]
        if on_disk <= len(self._chunk_id_to_label):
            return 0

        rows = self._db.execute(
            "SELECT chunk_id, hnsw_label, embedding FROM chunks "
            "WHERE hnsw_label IS NOT NULL AND embedding IS NOT NULL"
        ).fetchall()

        labels: list[int] = []
        vectors: list[np.ndarray] = []
        for chunk_id, label, blob in rows:
            if chunk_id in self._chunk_id_to_label or label in self._deleted_labels:
                continue
            self._register(chunk_id, int(label))
            labels.append(int(label))
            vectors.append(np.frombuffer(blob, dtype=np.float32))

        if not labels:
            return 0

        needed = self._index.get_current_count() + len(labels)
        if needed > self._index.get_max_elements():
            self._index.resize_index(max(needed * 2, self._max_elements))
        self._index.add_items(np.stack(vectors), np.array(labels, dtype=np.int64))
        return len(labels)

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

        allocated = self._allocate_labels(len(new_chunks))

        for chunk, label in zip(new_chunks, allocated):
            self._register(chunk.chunk_id, label)
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
        # Do not invalidate spans here. `_span_vectors` consumes rows after its
        # per-level high-water mark and changes only the last open span, so an
        # append costs O(new chunks) instead of rebuilding O(all chunks).

        data = np.stack(vectors)
        self._index.add_items(data, np.array(labels, dtype=np.int64))

        self._lexical.add_chunks(indexed)
        self._source_hierarchy.invalidate()

    def _live_count(self) -> int:
        """Elements hnswlib will actually return (excludes deleted labels)."""
        if self._index is None:
            return 0
        return max(0, self._index.get_current_count() - len(self._deleted_labels))


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
        self._clear_span_cache()
        self._lexical.delete_chunk(chunk_id)
        self._source_hierarchy.invalidate()
        artifacts_removed = 0
        if self._associations is not None:
            artifacts_removed = self._associations.remove_chunk_artifacts(chunk_id)
        return touched or label is not None or artifacts_removed > 0


    def rebuild_index(self) -> None:
        """Rebuild the hnswlib index from all embeddings in SQLite."""
        self._clear_span_cache()
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

        if not rows:
            return

        # Rows that never got a label (or lost it) need fresh ones, allocated
        # through the database like any other write.
        unlabelled = [r[0] for r in rows if r[2] is None]
        fresh = iter(self._allocate_labels(len(unlabelled)))

        labels: list[int] = []
        vectors: list[np.ndarray] = []

        for chunk_id, emb_blob, hnsw_label in rows:
            label = int(hnsw_label) if hnsw_label is not None else next(fresh)
            if hnsw_label is None:
                self._db.execute(
                    "UPDATE chunks SET hnsw_label = ? WHERE chunk_id = ?",
                    (label, chunk_id),
                )
            self._register(chunk_id, label)
            labels.append(label)
            vectors.append(np.frombuffer(emb_blob, dtype=np.float32))

        self._db.commit()

        data = np.stack(vectors)
        self._index.add_items(data, np.array(labels, dtype=np.int64))

    def save(self) -> None:
        """Persist the hnswlib index to disk."""
        if self._index_path and self._index is not None:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save_index(str(self._index_path))


__all__ = ["IndexLifecycleMixin"]
