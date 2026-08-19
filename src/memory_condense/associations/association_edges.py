"""Sparse persistent QK/OV edge operations."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from memory_condense.associations.association_models import (
    StoredHeadEdge,
    evidence_weighted_mean,
)


class AssociationEdgeStoreMixin:
    """Sparse head-edge writes, reads, and traversal accounting."""

    def upsert_edge(
        self,
        source_chunk_id: str,
        destination_chunk_id: str,
        artifact_id: str,
        head_weights: Sequence[float],
        *,
        qk_score: float,
        ov_transport: float = 0.0,
        evidence_count: int = 1,
        temporal_forward: bool | None = None,
        reverse: bool = False,
    ) -> None:
        artifact = self._require_artifact(artifact_id)
        if source_chunk_id == destination_chunk_id:
            raise ValueError("self edges are not stored")
        weights = self._pack_f32(
            head_weights,
            width=artifact.head_count,
            field_name="head_weights",
        )
        qk = float(qk_score)
        ov = float(ov_transport)
        count = int(evidence_count)
        if not math.isfinite(qk) or qk < 0.0:
            raise ValueError("qk_score must be finite and non-negative")
        if not math.isfinite(ov) or ov < 0.0:
            raise ValueError("ov_transport must be finite and non-negative")
        if count < 1:
            raise ValueError("evidence_count must be positive")
        self._merge_edge(
            source_chunk_id,
            destination_chunk_id,
            artifact_id,
            weights,
            qk,
            ov,
            count,
            temporal_forward,
        )
        if reverse:
            reverse_direction = (
                None if temporal_forward is None else not temporal_forward
            )
            self._merge_edge(
                destination_chunk_id,
                source_chunk_id,
                artifact_id,
                weights,
                qk,
                ov,
                count,
                reverse_direction,
            )
        self._db.commit()
        self._edge_neighbor_cache.clear()

    def _merge_edge(
        self,
        source_chunk_id: str,
        destination_chunk_id: str,
        artifact_id: str,
        weights_blob: bytes,
        qk_score: float,
        ov_transport: float,
        evidence_count: int,
        temporal_forward: bool | None,
    ) -> None:
        existing = self._db.execute(
            "SELECT head_weights, qk_score, ov_transport, evidence_count, "
            "temporal_forward FROM chunk_head_edges "
            "WHERE source_chunk_id = ? AND destination_chunk_id = ? "
            "AND artifact_id = ?",
            (source_chunk_id, destination_chunk_id, artifact_id),
        ).fetchone()
        if existing is None:
            merged_weights = weights_blob
            merged_qk = qk_score
            merged_ov = ov_transport
            merged_count = evidence_count
            merged_direction = temporal_forward
        else:
            old_count = int(existing[3])
            merged_count = old_count + evidence_count
            merged_weights = evidence_weighted_mean(
                np.frombuffer(existing[0], dtype="<f4"),
                np.frombuffer(weights_blob, dtype="<f4"),
                old_count,
                evidence_count,
            ).astype("<f4").tobytes()
            merged_qk = evidence_weighted_mean(
                float(existing[1]), qk_score, old_count, evidence_count
            )
            merged_ov = evidence_weighted_mean(
                float(existing[2]), ov_transport, old_count, evidence_count
            )
            old_direction = (
                None if existing[4] is None else bool(existing[4])
            )
            merged_direction = (
                old_direction if old_direction == temporal_forward else None
            )
        self._db.execute(
            "INSERT INTO chunk_head_edges "
            "(source_chunk_id, destination_chunk_id, artifact_id, head_weights, "
            "qk_score, ov_transport, evidence_count, temporal_forward) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_chunk_id, destination_chunk_id, artifact_id) "
            "DO UPDATE SET head_weights = excluded.head_weights, "
            "qk_score = excluded.qk_score, ov_transport = excluded.ov_transport, "
            "evidence_count = excluded.evidence_count, "
            "temporal_forward = excluded.temporal_forward",
            (
                source_chunk_id,
                destination_chunk_id,
                artifact_id,
                merged_weights,
                merged_qk,
                merged_ov,
                merged_count,
                None if merged_direction is None else int(merged_direction),
            ),
        )

    def neighbors(
        self,
        source_chunk_id: str,
        artifact_id: str,
        *,
        top_k: int,
        exclude: Sequence[str] = (),
        now_turn: int | None = None,
        usage_half_life: float = 100.0,
        usage_weight: float = 0.05,
    ) -> tuple[StoredHeadEdge, ...]:
        return self.neighbors_many(
            [source_chunk_id],
            artifact_id,
            top_k_per_source=top_k,
            exclude=exclude,
            now_turn=now_turn,
            usage_half_life=usage_half_life,
            usage_weight=usage_weight,
        )[source_chunk_id]

    def neighbors_many(
        self,
        source_chunk_ids: Sequence[str],
        artifact_id: str,
        *,
        top_k_per_source: int,
        exclude: Sequence[str] = (),
        exclude_frontier: bool = True,
        now_turn: int | None = None,
        usage_half_life: float = 100.0,
        usage_weight: float = 0.05,
    ) -> dict[str, tuple[StoredHeadEdge, ...]]:
        """Fetch bounded adjacency for many anchors in one SQLite query."""
        if top_k_per_source < 0:
            raise ValueError("top_k_per_source must be non-negative")
        sources = list(dict.fromkeys(source_chunk_ids))
        if not sources:
            return {}
        if top_k_per_source == 0:
            return {source_id: () for source_id in sources}
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        cache_key = (
            artifact_id,
            tuple(sources),
            top_k_per_source,
            tuple(sorted(set(exclude))),
            bool(exclude_frontier),
            turn,
            float(usage_half_life),
            float(usage_weight),
        )
        cached = (
            self._edge_neighbor_cache.get(cache_key)
            if self._cache_limit
            else None
        )
        if cached is not None:
            self._edge_neighbor_cache.move_to_end(cache_key)
            return dict(cached)
        artifact = self._require_artifact(artifact_id)
        # Ordinary bounded expansion does not need edges among nodes already in
        # the current frontier. Heat diffusion does: recurrent transitions are
        # safe because its scalar mass is normalized and hop-capped, and
        # excluding them would make the transition matrix depend on batching.
        excluded = set(exclude)
        if exclude_frontier:
            excluded.update(sources)
        placeholders = ",".join("?" for _ in sources)
        rows = self._db.execute(
            "SELECT e.source_chunk_id, e.destination_chunk_id, e.head_weights, "
            "e.qk_score, "
            "e.ov_transport, e.evidence_count, e.traversal_count, "
            "e.last_access_turn, e.temporal_forward "
            "FROM chunk_head_edges AS e "
            "JOIN chunks AS c ON c.chunk_id = e.destination_chunk_id "
            f"WHERE e.source_chunk_id IN ({placeholders}) AND e.artifact_id = ? "
            "AND c.embedding IS NOT NULL AND c.hnsw_label IS NOT NULL",
            (*sources, artifact_id),
        ).fetchall()
        edges_by_source: dict[str, list[StoredHeadEdge]] = {
            source_id: [] for source_id in sources
        }
        for row in rows:
            source_id, destination_id = row[0], row[1]
            if destination_id in excluded:
                continue
            weights = self._unpack_f32(row[2])
            if len(weights) != artifact.head_count:
                raise ValueError("stored head weight width does not match its artifact")
            edges_by_source[source_id].append(
                StoredHeadEdge(
                    source_chunk_id=source_id,
                    destination_chunk_id=destination_id,
                    artifact_id=artifact_id,
                    head_weights=weights,
                    qk_score=float(row[3]),
                    ov_transport=float(row[4]),
                    evidence_count=int(row[5]),
                    traversal_count=int(row[6]),
                    last_access_turn=int(row[7]),
                    temporal_forward=None if row[8] is None else bool(row[8]),
                )
            )
        result: dict[str, tuple[StoredHeadEdge, ...]] = {}
        for source_id, edges in edges_by_source.items():
            edges.sort(
                key=lambda edge: (
                    edge.utility(
                        now_turn=turn,
                        usage_half_life=usage_half_life,
                        usage_weight=usage_weight,
                    ),
                    edge.destination_chunk_id,
                ),
                reverse=True,
            )
            result[source_id] = tuple(edges[:top_k_per_source])
        self._cache_put(self._edge_neighbor_cache, cache_key, result)
        return result

    def touch_edges(
        self,
        artifact_id: str,
        pairs: Sequence[tuple[str, str]],
        *,
        now_turn: int | None = None,
    ) -> int:
        self._require_artifact(artifact_id)
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        if turn < 0:
            raise ValueError("now_turn must be non-negative")
        touched = 0
        for source_id, destination_id in dict.fromkeys(pairs):
            cur = self._db.execute(
                "UPDATE chunk_head_edges SET traversal_count = traversal_count + 1, "
                "last_access_turn = ? WHERE source_chunk_id = ? "
                "AND destination_chunk_id = ? AND artifact_id = ?",
                (turn, source_id, destination_id, artifact_id),
            )
            touched += cur.rowcount
        self._db.commit()
        if touched:
            self._edge_neighbor_cache.clear()
        return touched
