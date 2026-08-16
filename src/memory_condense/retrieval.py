from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Sequence

import hnswlib
import numpy as np

from memory_condense import lexical, ranking
from memory_condense.association_store import AssociationStore
from memory_condense.db import Database
from memory_condense.lexical import LexicalIndex
from memory_condense.schemas import Chunk, RetrievalResult, Turn

#: `meta` key holding the next free hnswlib label. Labels are shared state
#: across every process open on a store, so they are allocated through the
#: database rather than from a per-process counter.
_LABEL_KEY = "next_hnsw_label"

#: Span sizes that :meth:`SimilarityRetriever.span_query` pools over, in
#: **tokens of the pooled span**, not in chunks.
#:
#: Measured in chunks first, which was a bug waiting for a different corpus:
#: 4 and 8 chunks is ~110/~220 tokens on short-turn dialogue but ~900/~1800 on
#: long-form prose, where chunks already run 227 tokens. The same setting would
#: have helped dialogue and quietly wrecked monologue. A token target is
#: corpus-independent — it asks for a span of a given size regardless of how
#: many chunks that takes.
#:
#: 110 and 220 are the measured optima, translated. They are not round numbers:
#: at ~440 tokens (16 chunks) recall *falls* — 20.6% against 21.6% — while
#: costing 2.2x, because a mean vector washes out once a span straddles several
#: topics. Two levels beat one: stratified scored 22.1% at 660 tokens where the
#: single coarse level gave 21.6% at 673.
DEFAULT_SPAN_TOKENS: tuple[int, ...] = (110, 220)


def load_chunk_payload(db: Database, chunk_id: str) -> Chunk | None:
    """Load a chunk without decoding its unused dense embedding."""
    row = db.execute(
        "SELECT chunk_id, turn_id, text, start_char, end_char, token_count, "
        "lexical_weights FROM chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    lexical_weights = None if row[6] is None else json.loads(row[6])
    return Chunk(
        chunk_id=row[0],
        turn_id=row[1],
        text=row[2],
        start_char=row[3],
        end_char=row[4],
        token_count=row[5],
        embedding=None,
        lexical_weights=lexical_weights,
    )


def load_turn_payload(db: Database, turn_id: str) -> Turn | None:
    row = db.execute(
        "SELECT turn_id, role, text, source_id, created_at FROM turns WHERE turn_id = ?",
        (turn_id,),
    ).fetchone()
    if row is None:
        return None
    return Turn(
        turn_id=row[0],
        role=row[1],
        text=row[2],
        source_id=row[3],
        created_at=row[4],
    )


def hydrate_chunk_result(
    db: Database,
    chunk_id: str,
    *,
    score: float,
    dense_score: float | None = None,
    lexical_score: float | None = None,
    route: str | None = None,
    association_score: float | None = None,
    anchor_chunk_id: str | None = None,
    transition_distance: int | None = None,
    transition_direction: str | None = None,
) -> RetrievalResult | None:
    """Hydrate an external-memory ID without constructing an ANN retriever."""
    chunk = load_chunk_payload(db, chunk_id)
    if chunk is None:
        return None
    return RetrievalResult(
        chunk=chunk,
        score=score,
        turn=load_turn_payload(db, chunk.turn_id),
        dense_score=dense_score,
        lexical_score=lexical_score,
        route=route,
        association_score=association_score,
        anchor_chunk_id=anchor_chunk_id,
        transition_distance=transition_distance,
        transition_direction=transition_direction,
    )


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
        self._sync_from_db()
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
            hydrated = self._hydrate(chunk_id, score=score, route="dense")
            if hydrated is not None:
                results.append(hydrated)
        return results

    # ------------------------------------------------------------------
    # Span retrieval
    # ------------------------------------------------------------------

    def _span_vectors(self, target_tokens: int) -> tuple[np.ndarray, list[list[str]]]:
        """Pooled vectors for contiguous chunk runs of about ``target_tokens``.

        A span vector is the L2-normalised mean of its members' normalised
        vectors — the same operation the design specifies for cold-tier
        centroids, applied to chunk runs instead of memory clusters.

        Grouping by token budget rather than by chunk count is what makes one
        setting work on both short-turn dialogue (27-token chunks) and
        long-form prose (227-token chunks). A single chunk already at or over
        the target becomes its own span rather than being merged.

        Derived, never stored: SQLite already holds every member vector.  The
        per-process cache records a rowid high-water mark and the unnormalised
        sum of its last span.  Because the transcript is append-only, new rows
        can only extend that last span or start later ones; earlier spans never
        need to be revisited.  Deletion and index rebuild remain rare full
        invalidations.
        """
        cached = self._span_cache.get(target_tokens)
        if cached is not None:
            rows = self._load_span_rows(
                after_rowid=self._span_cached_through_rowid[target_tokens]
            )
            if rows:
                self._extend_span_cache(target_tokens, rows)
            return self._span_cache[target_tokens]

        rows = self._load_span_rows()
        if not rows:
            empty = (np.zeros((0, self._dim), dtype=np.float32), [])
            self._span_cache[target_tokens] = empty
            self._span_vector_buffers[target_tokens] = empty[0]
            self._span_tail_sums[target_tokens] = np.zeros(
                self._dim, dtype=np.float32
            )
            self._span_tail_tokens[target_tokens] = 0
            self._span_cached_through_rowid[target_tokens] = 0
            return empty

        self._span_cache[target_tokens] = (
            np.zeros((0, self._dim), dtype=np.float32),
            [],
        )
        self._span_vector_buffers[target_tokens] = self._span_cache[target_tokens][0]
        self._span_tail_sums[target_tokens] = np.zeros(self._dim, dtype=np.float32)
        self._span_tail_tokens[target_tokens] = 0
        self._span_cached_through_rowid[target_tokens] = 0
        self._extend_span_cache(target_tokens, rows)
        return self._span_cache[target_tokens]

    def _load_span_rows(
        self, after_rowid: int | None = None
    ) -> list[tuple[int, str, bytes, int]]:
        """Load live span inputs, optionally only after a cache high-water mark."""
        # `delete_chunk` nulls both indexed columns, so deleted chunks are
        # excluded without another bookkeeping set.  rowid is append order,
        # hence conversation order for the append-only transcript.
        sql = (
            "SELECT rowid, chunk_id, embedding, token_count FROM chunks "
            "WHERE embedding IS NOT NULL AND hnsw_label IS NOT NULL"
        )
        params: tuple[int, ...] = ()
        if after_rowid is not None:
            sql += " AND rowid > ?"
            params = (after_rowid,)
        return self._db.execute(sql + " ORDER BY rowid", params).fetchall()

    def _extend_span_cache(
        self,
        target_tokens: int,
        rows: list[tuple[int, str, bytes, int]],
    ) -> None:
        """Append rows to one cached level, touching only its open tail span."""
        _, members = self._span_cache[target_tokens]
        buffer = self._span_vector_buffers[target_tokens]
        tail_sum = self._span_tail_sums[target_tokens].copy()
        tail_tokens = self._span_tail_tokens[target_tokens]
        target = max(target_tokens, 1)

        for rowid, chunk_id, blob, token_count in rows:
            vector = np.frombuffer(blob, dtype=np.float32).copy()
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9:
                vector /= norm
            size = int(token_count or 0)

            if members and tail_tokens + size <= target:
                tail_sum += vector
                tail_tokens += size
                members[-1].append(chunk_id)
                tail_norm = float(np.linalg.norm(tail_sum))
                buffer[len(members) - 1] = (
                    tail_sum / tail_norm if tail_norm > 1e-9 else tail_sum.copy()
                )
            else:
                buffer = self._ensure_span_capacity(
                    target_tokens, len(members) + 1
                )
                tail_sum = vector.copy()
                tail_tokens = size
                members.append([chunk_id])
                buffer[len(members) - 1] = vector

            self._span_cached_through_rowid[target_tokens] = int(rowid)

        self._span_cache[target_tokens] = (
            buffer[: len(members)],
            members,
        )
        self._span_tail_sums[target_tokens] = tail_sum
        self._span_tail_tokens[target_tokens] = tail_tokens

    def _ensure_span_capacity(
        self, target_tokens: int, required: int
    ) -> np.ndarray:
        """Grow a level's vector buffer geometrically, amortizing append cost."""
        buffer = self._span_vector_buffers[target_tokens]
        if required <= len(buffer):
            return buffer
        capacity = max(required, max(8, len(buffer) * 2))
        grown = np.zeros((capacity, self._dim), dtype=np.float32)
        count = len(self._span_cache[target_tokens][1])
        if count:
            grown[:count] = buffer[:count]
        self._span_vector_buffers[target_tokens] = grown
        return grown

    def _clear_span_cache(self) -> None:
        """Invalidate every piece of derived span state after a non-append write."""
        self._span_cache.clear()
        self._span_vector_buffers.clear()
        self._span_tail_sums.clear()
        self._span_tail_tokens.clear()
        self._span_cached_through_rowid.clear()

    def span_query(
        self,
        query_embedding: np.ndarray,
        levels: Sequence[int] = DEFAULT_SPAN_TOKENS,
        k_per_level: int = 2,
    ) -> list[RetrievalResult]:
        """Retrieve by matching pooled spans, returning their member chunks.

        Short conversational turns produce chunks too small to carry topical
        signal: on LoCoMo the median chunk is 27 tokens, and ``k=10`` buys
        roughly 220 tokens of context against 2,270 on long-form prose. Pooling
        contiguous chunks up to a token target gives the matcher a unit with
        enough content to be found. ``levels`` are **token targets**, so one
        setting holds across corpora with very different turn lengths.

        Replicated across four LoCoMo samples (n=757): dense ``k=10`` reaches
        10.3% answer containment, stratified spans 23.4%, higher on every
        sample individually. Read that against cost — at a matched token budget
        the gap is roughly 2.2x, while raw recall-per-token flatters ``k=10``
        purely because it operates at a 200-token budget where marginal returns
        are highest.

        **Retrieval is stratified — top-k *within* each level, merged after —
        and that is load-bearing, not tidiness.** Cosine similarity is not
        length-invariant: measured on this corpus, per-turn chunks average
        0.678 top-10 cosine against 0.602 for 8-chunk spans, because short text
        has fewer competing topical directions to dilute the match. Searching a
        single mixed-granularity pool therefore lets small chunks crowd out
        every span — measured, that collapsed recall from 21.6% to 6.0%.

        Scoring happens at span granularity but **real member chunks are
        returned**, so provenance, ``ContextPacker`` and every downstream
        consumer keep working on chunks exactly as before. Nothing here invents
        a synthetic chunk.
        """
        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query_vec))
        if norm > 1e-9:
            query_vec = query_vec / norm

        picked: dict[str, float] = {}
        for level in levels:
            pooled, members = self._span_vectors(level)
            if not len(pooled):
                continue
            scores = pooled @ query_vec
            for idx in np.argsort(-scores)[: max(k_per_level, 0)]:
                score = float(scores[idx])
                for chunk_id in members[idx]:
                    # A chunk reachable from several levels keeps its best score.
                    if score > picked.get(chunk_id, float("-inf")):
                        picked[chunk_id] = score

        results: list[RetrievalResult] = []
        for chunk_id, score in sorted(picked.items(), key=lambda kv: -kv[1]):
            hydrated = self._hydrate(chunk_id, score=score, route="span")
            if hydrated is not None:
                results.append(hydrated)
        return results

    def source_query(
        self,
        query_embedding: np.ndarray,
        *,
        k_sources: int = 4,
    ) -> list[RetrievalResult]:
        """Retrieve whole provenance sources using pooled chunk embeddings.

        A source is a conversation session, document, or file recorded on its
        turns. Legacy turns fall back to one source per turn. Source vectors
        are exact normalized means over normalized member chunks; no additional
        ANN or persisted activation state is created. Winning sources return
        their real chunks in conversation order so prompt caps and provenance
        continue to operate normally.
        """

        if k_sources <= 0:
            return []
        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm > 1e-9:
            query_vec = query_vec / query_norm

        rows = self._db.execute(
            "SELECT c.chunk_id, c.embedding, COALESCE(t.source_id, t.turn_id), "
            "t.ordinal, c.rowid FROM chunks AS c "
            "JOIN turns AS t ON t.turn_id = c.turn_id "
            "WHERE c.embedding IS NOT NULL AND c.hnsw_label IS NOT NULL "
            "ORDER BY t.ordinal, c.rowid"
        ).fetchall()
        if not rows:
            return []

        members: dict[str, list[str]] = {}
        sums: dict[str, np.ndarray] = {}
        for chunk_id, blob, source_id, _ordinal, _rowid in rows:
            vector = np.frombuffer(blob, dtype=np.float32).copy()
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9:
                vector /= norm
            key = str(source_id)
            members.setdefault(key, []).append(str(chunk_id))
            if key in sums:
                sums[key] += vector
            else:
                sums[key] = vector

        scored: list[tuple[float, str]] = []
        for source_id, vector_sum in sums.items():
            norm = float(np.linalg.norm(vector_sum))
            pooled = vector_sum / norm if norm > 1e-9 else vector_sum
            scored.append((float(pooled @ query_vec), source_id))
        selected = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[
            :k_sources
        ]

        results: list[RetrievalResult] = []
        for score, source_id in selected:
            for chunk_id in members[source_id]:
                hydrated = self._hydrate(chunk_id, score=score, route="source")
                if hydrated is not None:
                    results.append(hydrated)
        return results

    def hydrate_sources(
        self,
        source_ids: Sequence[str],
        *,
        source_scores: dict[str, float] | None = None,
        interleave: bool = True,
        route: str = "anchored_source",
    ) -> list[RetrievalResult]:
        """Hydrate selected sources without reranking their member chunks.

        Sources retain caller order. With ``interleave=True`` one chunk per
        source is emitted per round, so a final hard prompt cap cannot let one
        long session crowd every other selected evidence source out.
        """
        selected = list(dict.fromkeys(str(source_id) for source_id in source_ids))
        if not selected:
            return []
        placeholders = ",".join("?" for _ in selected)
        rows = self._db.execute(
            "SELECT c.chunk_id, COALESCE(t.source_id, t.turn_id) "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE COALESCE(t.source_id, t.turn_id) IN ({placeholders}) "
            "AND c.embedding IS NOT NULL AND c.hnsw_label IS NOT NULL "
            "ORDER BY t.ordinal, c.rowid",
            tuple(selected),
        ).fetchall()
        members: dict[str, list[str]] = {source_id: [] for source_id in selected}
        for chunk_id, source_id in rows:
            key = str(source_id)
            if key in members:
                members[key].append(str(chunk_id))

        ordered_ids: list[tuple[str, str]] = []
        if interleave:
            depth = 0
            while any(depth < len(members[source_id]) for source_id in selected):
                for source_id in selected:
                    if depth < len(members[source_id]):
                        ordered_ids.append((source_id, members[source_id][depth]))
                depth += 1
        else:
            ordered_ids = [
                (source_id, chunk_id)
                for source_id in selected
                for chunk_id in members[source_id]
            ]

        scores = source_scores or {}
        results: list[RetrievalResult] = []
        for source_id, chunk_id in ordered_ids:
            hydrated = self._hydrate(
                chunk_id,
                score=float(scores.get(source_id, 0.0)),
                route=route,
            )
            if hydrated is not None:
                results.append(hydrated)
        return results

    def hydrate_source_neighbors(
        self,
        anchors: Sequence[RetrievalResult],
        *,
        radius: int = 1,
        max_neighbors: int | None = None,
        route: str = "hybrid_neighbor",
    ) -> list[RetrievalResult]:
        """Expand ranked anchors by bounded distance inside their sources.

        Direct anchors remain first and retain their hybrid score components.
        Neighbor shells then follow stepwise: distance one around every anchor,
        then distance two, and so on. This keeps the useful retrieval ranking
        intact while exposing local conversational transitions without loading
        an entire source or persisting any token/attention state.
        """
        if radius < 0:
            raise ValueError("radius must be non-negative")
        if max_neighbors is not None and max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative or None")
        if not anchors:
            return []

        unique_anchors: list[RetrievalResult] = []
        seen: set[str] = set()
        for anchor in anchors:
            chunk_id = anchor.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique_anchors.append(anchor)
        if radius == 0 or max_neighbors == 0:
            return unique_anchors

        source_ids = list(
            dict.fromkeys(
                str(anchor.turn.source_id or anchor.turn.turn_id)
                for anchor in unique_anchors
                if anchor.turn is not None
            )
        )
        if not source_ids:
            return unique_anchors

        placeholders = ",".join("?" for _ in source_ids)
        rows = self._db.execute(
            "SELECT c.chunk_id, COALESCE(t.source_id, t.turn_id) "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE COALESCE(t.source_id, t.turn_id) IN ({placeholders}) "
            "AND c.embedding IS NOT NULL AND c.hnsw_label IS NOT NULL "
            "ORDER BY t.ordinal, c.rowid",
            tuple(source_ids),
        ).fetchall()
        members: dict[str, list[str]] = {source_id: [] for source_id in source_ids}
        for chunk_id, source_id in rows:
            key = str(source_id)
            if key in members:
                members[key].append(str(chunk_id))
        positions = {
            (source_id, chunk_id): index
            for source_id, chunk_ids in members.items()
            for index, chunk_id in enumerate(chunk_ids)
        }

        results = list(unique_anchors)
        for distance in range(1, radius + 1):
            for anchor in unique_anchors:
                if anchor.turn is None:
                    continue
                source_id = str(anchor.turn.source_id or anchor.turn.turn_id)
                position = positions.get((source_id, anchor.chunk.chunk_id))
                if position is None:
                    continue
                chunk_ids = members[source_id]
                # Earlier context precedes the following response at each
                # shell, matching ordinary conversation reading order.
                for neighbor_position in (position - distance, position + distance):
                    if not 0 <= neighbor_position < len(chunk_ids):
                        continue
                    chunk_id = chunk_ids[neighbor_position]
                    if chunk_id in seen:
                        continue
                    hydrated = self._hydrate(
                        chunk_id,
                        score=float(anchor.score),
                        route=route,
                        anchor_chunk_id=anchor.chunk.chunk_id,
                        transition_distance=distance,
                        transition_direction=(
                            "previous"
                            if neighbor_position < position
                            else "next"
                        ),
                    )
                    if hydrated is not None:
                        seen.add(chunk_id)
                        results.append(hydrated)
                        if (
                            max_neighbors is not None
                            and len(results) - len(unique_anchors) >= max_neighbors
                        ):
                            return results
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
                route="hybrid",
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
        self._clear_span_cache()
        self._lexical.delete_chunk(chunk_id)
        artifacts_removed = 0
        if self._associations is not None:
            artifacts_removed = self._associations.remove_chunk_artifacts(chunk_id)
        return touched or label is not None or artifacts_removed > 0

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
        route: str | None = None,
        association_score: float | None = None,
        anchor_chunk_id: str | None = None,
        transition_distance: int | None = None,
        transition_direction: str | None = None,
    ) -> RetrievalResult | None:
        """Build a RetrievalResult from SQLite, or None if the chunk is gone."""
        return hydrate_chunk_result(
            self._db,
            chunk_id,
            score=score,
            dense_score=dense_score,
            lexical_score=lexical_score,
            route=route,
            association_score=association_score,
            anchor_chunk_id=anchor_chunk_id,
            transition_distance=transition_distance,
            transition_direction=transition_direction,
        )

    def hydrate_chunk(
        self,
        chunk_id: str,
        *,
        score: float,
        route: str | None = None,
        association_score: float | None = None,
        anchor_chunk_id: str | None = None,
    ) -> RetrievalResult | None:
        """Hydrate a known external-memory ID without searching the ANN index."""
        return self._hydrate(
            chunk_id,
            score=score,
            route=route,
            association_score=association_score,
            anchor_chunk_id=anchor_chunk_id,
        )

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

    def _load_chunk(self, chunk_id: str) -> Chunk | None:
        """Load retrieval payload only; the 1024-float source vector stays put.

        Returning every stored embedding made each candidate hydration decode
        ~4 KiB of data that ranking and context assembly never read.  The ANN
        index already consumed the vector before this point, so retrieval
        results intentionally carry ``embedding=None``. Lexical weights remain
        because they are part of the hydrated chunk's public contract.
        """
        return load_chunk_payload(self._db, chunk_id)

    def _load_turn(self, turn_id: str) -> Turn | None:
        """Load a turn from SQLite by ID."""
        return load_turn_payload(self._db, turn_id)
