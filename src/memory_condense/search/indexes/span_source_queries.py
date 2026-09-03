"""Stateful span, source, partition, and source-neighbor query workflows."""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from memory_condense.domain.ranking import round_robin_unique
from memory_condense.persistence.db import INDEXED_CHUNK_SQL, TURN_SOURCE_ID_SQL
from memory_condense.persistence.transcript_store import parse_source_metadata
from memory_condense.search.indexes.retrieval_models import (
    DEFAULT_SPAN_TOKENS,
    PartitionContentRow,
)
from memory_condense.domain.schemas import RetrievalResult


class SpanSourceQueryMixin:
    """Internal stateful methods composed by ``SimilarityRetriever``."""

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
        self._sync_from_db()
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

        self._sync_from_db()
        if k_sources <= 0:
            return []
        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm > 1e-9:
            query_vec = query_vec / query_norm

        rows = self._db.execute(
            f"SELECT c.chunk_id, c.embedding, {TURN_SOURCE_ID_SQL}, "
            "t.ordinal, c.rowid FROM chunks AS c "
            "JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {INDEXED_CHUNK_SQL} "
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
            f"SELECT c.chunk_id, {TURN_SOURCE_ID_SQL} "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {TURN_SOURCE_ID_SQL} IN ({placeholders}) "
            f"AND {INDEXED_CHUNK_SQL} "
            "ORDER BY t.ordinal, c.rowid",
            tuple(selected),
        ).fetchall()
        members: dict[str, list[str]] = {source_id: [] for source_id in selected}
        for chunk_id, source_id in rows:
            key = str(source_id)
            if key in members:
                members[key].append(str(chunk_id))

        ordered_ids: list[tuple[str, str]]
        if interleave:
            ordered_ids = round_robin_unique(
                [
                    [(source_id, chunk_id) for chunk_id in members[source_id]]
                    for source_id in selected
                ]
            )
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

    def source_tfisf_query(
        self,
        query_text: str,
        *,
        k_sources: int = 8,
    ) -> list[tuple[str, float]]:
        """Return live source-level lexical activations without hydration."""

        self._sync_from_db()
        return self._lexical.search_source_tfisf(query_text, limit=k_sources)

    def source_hsc_expand(
        self,
        query_embedding: np.ndarray,
        seed_source_ids: Sequence[str],
        *,
        slots: int = 8,
        hops: int = 2,
    ) -> list[tuple[str, float]]:
        """Expand source seeds through the transient contraction hierarchy."""

        self._sync_from_db()
        return self._source_hierarchy.expand(
            query_embedding,
            seed_source_ids,
            slots=slots,
            hops=hops,
        )

    def source_ids_in_partitions(
        self,
        partition_ids: Sequence[str],
        *,
        separator: str = "::",
    ) -> list[str]:
        """Return source IDs belonging to hierarchical top-level partitions.

        Partition identity is encoded in the durable source ID as
        ``partition::source``.  Keeping it in the existing provenance field
        avoids a second copy of transcript or embedding state.  Sources that
        do not contain the separator are their own top-level partition.
        """

        selected = {str(value) for value in partition_ids if str(value)}
        if not selected:
            return []
        if not separator:
            raise ValueError("partition separator must be non-empty")

        rows = self._db.execute(
            "SELECT source_id, MIN(ordinal) FROM turns "
            "WHERE source_id IS NOT NULL GROUP BY source_id "
            "ORDER BY MIN(ordinal), source_id"
        ).fetchall()
        results: list[str] = []
        for source_id, _ordinal in rows:
            source = str(source_id)
            partition = source.split(separator, 1)[0]
            if partition in selected:
                results.append(source)
        return results

    def source_partition_ids(self, *, separator: str = "::") -> list[str]:
        """Return the complete durable top-level source-partition inventory.

        The order follows first transcript occurrence and is therefore stable
        for one immutable store.  Coverage diagnostics use this inventory to
        distinguish an exhaustive partition scope from approximate top-k
        routing; it contains IDs only and materializes no transcript text.
        """

        if not separator:
            raise ValueError("partition separator must be non-empty")
        rows = self._db.execute(
            "SELECT source_id, MIN(ordinal) FROM turns "
            "WHERE source_id IS NOT NULL GROUP BY source_id "
            "ORDER BY MIN(ordinal), source_id"
        ).fetchall()
        partitions: list[str] = []
        seen: set[str] = set()
        for source_id, _ordinal in rows:
            partition = str(source_id).split(separator, 1)[0]
            if partition and partition not in seen:
                seen.add(partition)
                partitions.append(partition)
        return partitions

    def iter_partition_content_rows(
        self,
        partition_ids: Sequence[str],
        *,
        separator: str = "::",
        source_batch_size: int = 400,
    ) -> Iterator[PartitionContentRow]:
        """Stream every raw content chunk in the selected partitions.

        Partition membership is resolved through :meth:`source_ids_in_partitions`
        rather than a prefix ``LIKE`` expression, so ``alpha`` cannot silently
        include ``alpha-extra`` and separators containing SQL wildcard
        characters remain literal.  Source IDs are queried in bounded groups
        to stay below SQLite's bind-variable limit.  Text is yielded one row at
        a time and no embedding, activation, or query state is retained.
        """

        if source_batch_size < 1:
            raise ValueError("source_batch_size must be positive")
        yield from self.iter_source_content_rows(
            self.source_ids_in_partitions(
                partition_ids,
                separator=separator,
            ),
            source_batch_size=source_batch_size,
        )

    def iter_source_content_rows(
        self,
        source_ids: Sequence[str],
        *,
        source_batch_size: int = 400,
    ) -> Iterator[PartitionContentRow]:
        """Stream raw chunks from one immutable caller-supplied source set."""

        if source_batch_size < 1:
            raise ValueError("source_batch_size must be positive")
        selected_sources = list(
            dict.fromkeys(str(source_id) for source_id in source_ids if source_id)
        )
        source_expr = TURN_SOURCE_ID_SQL
        for start in range(0, len(selected_sources), source_batch_size):
            batch = selected_sources[start : start + source_batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT c.chunk_id, "
                + source_expr
                + ", t.role, t.ordinal, c.text "
                "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
                f"WHERE {source_expr} IN ({placeholders}) "
                "ORDER BY t.ordinal, c.rowid",
                tuple(batch),
            )
            for chunk_id, source_id, role, ordinal, text in rows:
                raw_text = str(text)
                if parse_source_metadata(raw_text) is not None:
                    continue
                yield PartitionContentRow(
                    chunk_id=str(chunk_id),
                    source_id=str(source_id),
                    role=str(role),
                    ordinal=int(ordinal),
                    text=raw_text,
                )

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
            f"SELECT c.chunk_id, {TURN_SOURCE_ID_SQL} "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {TURN_SOURCE_ID_SQL} IN ({placeholders}) "
            f"AND {INDEXED_CHUNK_SQL} "
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


__all__ = ["SpanSourceQueryMixin"]
