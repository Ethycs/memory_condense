"""Stateful dense, hybrid, source-companion, and hydration queries."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from memory_condense.domain import ranking
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.persistence.db import INDEXED_CHUNK_SQL, TURN_SOURCE_ID_SQL
from memory_condense.search.indexes.lexical import LexicalIndex
from memory_condense.search.indexes.retrieval_models import hydrate_chunk_result


class HybridQueryMixin:
    """Internal stateful methods composed by ``SimilarityRetriever``."""

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

    def hybrid_query_sources(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        source_ids: Sequence[str],
        *,
        k: int = 24,
        candidates_per_source: int = 200,
        alpha: float = 0.65,
        source_scores: dict[str, float] | None = None,
        anchor_chunk_ids: dict[str, str] | None = None,
        exclude_chunk_ids: Sequence[str] = (),
    ) -> list[RetrievalResult]:
        """Search every chunk in activated sources, then return a bounded rank.

        This is deliberately different from filtering a global ANN pool. Dense
        vectors are streamed from SQLite and only bounded per-source buffers are
        retained. BM25 is also computed in each source's local corpus. Text is
        hydrated only for the final ``k`` rows, keeping workspace proportional
        to ``sources * candidates_per_source`` rather than source length.

        Source activation is an eligibility gate, not a second relevance
        multiplier. Dense and lexical scores are normalized once across the
        complete activated-source union so scores remain comparable between
        partitions while chunks absent from the global pool can still enter.
        """

        selected = list(dict.fromkeys(str(value) for value in source_ids if str(value)))
        if k <= 0 or not selected:
            return []
        if candidates_per_source <= 0:
            raise ValueError("candidates_per_source must be positive")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query))
        if query.shape != (self._dim,) or query_norm <= 0.0:
            raise ValueError("query_embedding must be a non-zero vector of index dimension")
        query /= query_norm

        source_placeholders = ",".join("?" for _ in selected)
        source_expr = TURN_SOURCE_ID_SQL
        dense_buffers: dict[str, list[tuple[float, str]]] = {
            source_id: [] for source_id in selected
        }
        rows = self._db.execute(
            "SELECT c.chunk_id, c.embedding, " + source_expr + " "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {source_expr} IN ({source_placeholders}) "
            f"AND {INDEXED_CHUNK_SQL} "
            "ORDER BY c.chunk_id",
            tuple(selected),
        )
        for chunk_id, blob, source_id in rows:
            source_key = str(source_id)
            vector = np.frombuffer(blob, dtype=np.float32)
            denominator = float(np.linalg.norm(vector))
            if vector.shape != query.shape or denominator <= 0.0:
                continue
            score = float(np.dot(query, vector) / denominator)
            buffer = dense_buffers[source_key]
            buffer.append((score, str(chunk_id)))
            if len(buffer) >= candidates_per_source * 2:
                buffer.sort(key=lambda item: (-item[0], item[1]))
                del buffer[candidates_per_source:]

        dense_by_source = {
            source_id: sorted(buffer, key=lambda item: (-item[0], item[1]))[
                :candidates_per_source
            ]
            for source_id, buffer in dense_buffers.items()
        }
        lexical_by_source = self._lexical.search_sources(
            query_text,
            selected,
            limit_per_source=candidates_per_source,
        )
        priors = source_scores or {}
        anchors = anchor_chunk_ids or {}
        excluded = {str(value) for value in exclude_chunk_ids}

        dense_flat = [
            (source_id, chunk_id, score)
            for source_id in selected
            for score, chunk_id in dense_by_source.get(source_id, [])
        ]
        lexical_flat = [
            (source_id, chunk_id, score)
            for source_id in selected
            for chunk_id, score in lexical_by_source.get(source_id, [])
        ]
        dense_normalized = ranking.min_max_normalize(
            [score for _source_id, _chunk_id, score in dense_flat]
        )
        lexical_normalized = ranking.min_max_normalize(
            [score for _source_id, _chunk_id, score in lexical_flat]
        )
        dense_scores = {
            (source_id, chunk_id): score
            for (source_id, chunk_id, _raw), score in zip(
                dense_flat, dense_normalized
            )
        }
        lexical_scores = {
            (source_id, chunk_id): score
            for (source_id, chunk_id, _raw), score in zip(
                lexical_flat, lexical_normalized
            )
        }

        scored: list[tuple[float, int, str, float, float, str]] = []
        for source_order, source_id in enumerate(selected):
            dense_hits = dense_by_source.get(source_id, [])
            lexical_hits = lexical_by_source.get(source_id, [])
            ordered_ids = [chunk_id for _score, chunk_id in dense_hits]
            seen = set(ordered_ids)
            for chunk_id, _score in lexical_hits:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    ordered_ids.append(chunk_id)

            for chunk_id in ordered_ids:
                if chunk_id in excluded:
                    continue
                dense_score = dense_scores.get((source_id, chunk_id), 0.0)
                lexical_score = lexical_scores.get((source_id, chunk_id), 0.0)
                local_score = ranking.blend_hybrid(
                    dense_score, lexical_score, alpha
                )
                scored.append(
                    (
                        local_score,
                        source_order,
                        chunk_id,
                        dense_score,
                        lexical_score,
                        source_id,
                    )
                )

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        results: list[RetrievalResult] = []
        for score, _source_order, chunk_id, dense, lexical_score, source_id in scored:
            hydrated = self._hydrate(
                chunk_id,
                score=score,
                dense_score=dense,
                lexical_score=lexical_score,
                route="hybrid_source_local",
                anchor_chunk_id=anchors.get(source_id),
            )
            if hydrated is not None:
                results.append(
                    hydrated.model_copy(
                        update={
                            "memory_source_id": source_id,
                            "source_heat": max(
                                0.0, float(priors.get(source_id, 1.0))
                            ),
                        }
                    )
                )
            if len(results) >= k:
                break
        return results

    def hybrid_query_source_companions(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        source_ids: Sequence[str],
        *,
        metadata_chunk_ids: Sequence[str],
        max_sources: int,
        max_per_source: int = 1,
        candidates_per_source: int = 64,
        alpha: float = 0.65,
        source_scores: dict[str, float] | None = None,
        anchor_chunk_ids: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Return a bounded real-content shortlist per routed source.

        ``metadata_chunk_ids`` must contain only synthetic source-boundary
        timestamp rows identified by the caller. They are excluded by exact
        durable chunk ID; other system-authored evidence remains eligible.
        Sources, per-source search, and hydrated rows are all hard bounded.
        The default preserves the historical one-best behavior; callers with
        an injected semantic chooser may request a small ``max_per_source``
        shortlist and still select exactly one payload before packing. No gold
        labels or model state enter this operation.
        """

        if max_sources < 0:
            raise ValueError("max_sources must be non-negative")
        if max_per_source < 1:
            raise ValueError("max_per_source must be positive")
        if candidates_per_source < 1:
            raise ValueError("candidates_per_source must be positive")
        selected = list(
            dict.fromkeys(str(source_id) for source_id in source_ids if str(source_id))
        )[:max_sources]
        if not selected:
            return []
        excluded = tuple(dict.fromkeys(str(value) for value in metadata_chunk_ids))
        scores = source_scores or {}
        anchors = anchor_chunk_ids or {}
        companions: list[RetrievalResult] = []
        for source_id in selected:
            hits = self.hybrid_query_sources(
                query_text,
                query_embedding,
                [source_id],
                k=max_per_source,
                candidates_per_source=candidates_per_source,
                alpha=alpha,
                source_scores={source_id: float(scores.get(source_id, 0.0))},
                anchor_chunk_ids=(
                    {source_id: anchors[source_id]}
                    if source_id in anchors
                    else None
                ),
                exclude_chunk_ids=excluded,
            )
            if not hits:
                continue
            companions.extend(
                hit.model_copy(update={"route": "source_metadata_companion"})
                for hit in hits[:max_per_source]
            )
        return companions


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

    def cosine_scores(
        self,
        query_embedding: np.ndarray,
        chunk_ids: Sequence[str],
    ) -> dict[str, float]:
        """Score known chunk IDs without widening ANN retrieval.

        This is used to rerank a small graph frontier against the live query.
        It reads only durable chunk embeddings and returns scalar cosines; no
        transformer activation or query state is retained.
        """

        ids = list(dict.fromkeys(str(value) for value in chunk_ids if str(value)))
        if not ids:
            return {}
        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query.ndim != 1 or query_norm <= 0.0:
            raise ValueError("query_embedding must be a non-zero vector")
        placeholders = ",".join("?" for _ in ids)
        rows = self._db.execute(
            "SELECT chunk_id, embedding FROM chunks "
            f"WHERE embedding IS NOT NULL AND chunk_id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        scores: dict[str, float] = {}
        for chunk_id, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            denominator = query_norm * float(np.linalg.norm(vector))
            if vector.shape == query.shape and denominator > 0.0:
                scores[str(chunk_id)] = float(np.dot(query, vector) / denominator)
        return scores

    def stored_embeddings(
        self,
        chunk_ids: Sequence[str],
    ) -> dict[str, tuple[float, ...]]:
        """Load exact durable chunk vectors for an explicit bounded ID set.

        Ordinary result hydration deliberately omits the dense vector.  Episode
        compilation is the narrow exception: its cohesion control must use the
        same vectors that were indexed, without embedding the source a second
        time.  This method therefore reads only the requested IDs, validates
        their fixed-width finite float32 payloads, and preserves input order.
        """

        ids = list(dict.fromkeys(str(value).strip() for value in chunk_ids))
        if any(not value for value in ids):
            raise ValueError("chunk_ids must contain only non-empty IDs")
        if not ids:
            return {}

        found: dict[str, tuple[float, ...]] = {}
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT chunk_id, embedding FROM chunks "
                f"WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            for chunk_id, blob in rows:
                if blob is None:
                    raise ValueError(
                        f"chunk {chunk_id!r} has no durable embedding"
                    )
                vector = np.frombuffer(blob, dtype=np.float32)
                if vector.ndim != 1 or vector.size != self._dim:
                    raise ValueError(
                        f"chunk {chunk_id!r} embedding has the wrong width"
                    )
                if not np.isfinite(vector).all():
                    raise ValueError(
                        f"chunk {chunk_id!r} embedding contains non-finite values"
                    )
                found[str(chunk_id)] = tuple(float(value) for value in vector)

        missing = [chunk_id for chunk_id in ids if chunk_id not in found]
        if missing:
            raise ValueError(
                "durable embeddings are missing for requested chunks: "
                + ", ".join(missing[:3])
            )
        return {chunk_id: found[chunk_id] for chunk_id in ids}


__all__ = ["HybridQueryMixin"]
