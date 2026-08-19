"""Composed SQLite repository for compact association state."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any, Sequence

from memory_condense.associations.association_artifacts import (
    AssociationArtifactStoreMixin,
)
from memory_condense.associations.association_edges import AssociationEdgeStoreMixin
from memory_condense.associations.association_models import (
    AssociationArtifact,
    StoredCAVNeighbor,
    StoredHeadEdge,
)
from memory_condense.associations.hebbian_store import HebbianAssociationStoreMixin
from memory_condense.persistence.db import Database


class AssociationStore(
    AssociationArtifactStoreMixin,
    AssociationEdgeStoreMixin,
    HebbianAssociationStoreMixin,
):
    """SQLite implementation of the bounded external association layer."""

    def __init__(
        self,
        db: Database,
        *,
        cache_neighbors: bool = False,
        cache_limit: int = 512,
    ) -> None:
        self._db = db
        self._artifact_cache: dict[str, AssociationArtifact] = {}
        self._cav_neighbor_cache: OrderedDict[tuple, tuple[StoredCAVNeighbor, ...]] = (
            OrderedDict()
        )
        self._edge_neighbor_cache: OrderedDict[
            tuple, dict[str, tuple[StoredHeadEdge, ...]]
        ] = OrderedDict()
        self._cache_limit = max(0, int(cache_limit)) if cache_neighbors else 0

    def _cache_put(self, cache: OrderedDict, key: tuple, value: Any) -> None:
        if self._cache_limit == 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._cache_limit:
            cache.popitem(last=False)

    def prune_edges(
        self,
        artifact_id: str,
        max_neighbors: int,
        *,
        source_chunk_ids: Sequence[str] | None = None,
        now_turn: int | None = None,
        usage_half_life: float = 100.0,
        usage_weight: float = 0.05,
    ) -> int:
        """Enforce a hard per-source degree cap using live edge utility."""
        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        self._require_artifact(artifact_id)
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        select_columns = (
            "SELECT source_chunk_id, destination_chunk_id, head_weights, qk_score, "
            "ov_transport, evidence_count, traversal_count, last_access_turn, "
            "temporal_forward FROM chunk_head_edges "
        )
        rows: list = []
        if source_chunk_ids is None:
            rows = self._db.execute(
                select_columns + "WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchall()
        else:
            # Keep below SQLite's conservative host-parameter limit while
            # avoiding one SELECT per source.
            sources = list(dict.fromkeys(source_chunk_ids))
            for offset in range(0, len(sources), 500):
                batch = sources[offset : offset + 500]
                if not batch:
                    continue
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    self._db.execute(
                        select_columns
                        + f"WHERE artifact_id = ? AND source_chunk_id IN ({placeholders})",
                        (artifact_id, *batch),
                    ).fetchall()
                )

        rows_by_source: dict[str, list] = defaultdict(list)
        for row in rows:
            rows_by_source[row[0]].append(row)
        deletions: list[tuple[str, str, str]] = []
        for source_id, source_rows in rows_by_source.items():
            ranked: list[tuple[float, str]] = []
            for row in source_rows:
                edge = StoredHeadEdge(
                    source_chunk_id=source_id,
                    destination_chunk_id=row[1],
                    artifact_id=artifact_id,
                    head_weights=self._unpack_f32(row[2]),
                    qk_score=float(row[3]),
                    ov_transport=float(row[4]),
                    evidence_count=int(row[5]),
                    traversal_count=int(row[6]),
                    last_access_turn=int(row[7]),
                    temporal_forward=None if row[8] is None else bool(row[8]),
                )
                ranked.append(
                    (
                        edge.utility(
                            now_turn=turn,
                            usage_half_life=usage_half_life,
                            usage_weight=usage_weight,
                        ),
                        edge.destination_chunk_id,
                    )
                )
            ranked.sort(reverse=True)
            for _, destination_id in ranked[max_neighbors:]:
                deletions.append((source_id, destination_id, artifact_id))
        removed = 0
        if deletions:
            cur = self._db.executemany(
                "DELETE FROM chunk_head_edges WHERE source_chunk_id = ? "
                "AND destination_chunk_id = ? AND artifact_id = ?",
                deletions,
            )
            removed = (
                cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0
                else len(deletions)
            )
        self._db.commit()
        if removed:
            self._edge_neighbor_cache.clear()
        return removed

    def remove_chunk_artifacts(self, chunk_id: str) -> int:
        """Remove a chunk from the live graph while preserving its source row."""
        hebbian_edge_cur = self._db.execute(
            "DELETE FROM hebbian_chunk_edges WHERE chunk_low = ? OR chunk_high = ?",
            (chunk_id, chunk_id),
        )
        hebbian_node_cur = self._db.execute(
            "DELETE FROM hebbian_chunk_nodes WHERE chunk_id = ?",
            (chunk_id,),
        )
        edge_cur = self._db.execute(
            "DELETE FROM chunk_head_edges WHERE source_chunk_id = ? "
            "OR destination_chunk_id = ?",
            (chunk_id, chunk_id),
        )
        signature_cur = self._db.execute(
            "DELETE FROM chunk_cav_signatures WHERE chunk_id = ?",
            (chunk_id,),
        )
        self._db.commit()
        if edge_cur.rowcount:
            self._edge_neighbor_cache.clear()
        if signature_cur.rowcount:
            self._cav_neighbor_cache.clear()
        return (
            edge_cur.rowcount
            + signature_cur.rowcount
            + hebbian_edge_cur.rowcount
            + hebbian_node_cur.rowcount
        )

    def stats(self, artifact_id: str) -> dict[str, int]:
        self._require_artifact(artifact_id)
        signatures = int(
            self._db.execute(
                "SELECT COUNT(*) FROM chunk_cav_signatures WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()[0]
        )
        edges = int(
            self._db.execute(
                "SELECT COUNT(*) FROM chunk_head_edges WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()[0]
        )
        cav_payload_bytes = int(
            self._db.execute(
                "SELECT COALESCE(SUM(LENGTH(signature)), 0) "
                "FROM chunk_cav_signatures WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()[0]
        )
        head_payload_bytes = int(
            self._db.execute(
                "SELECT COALESCE(SUM(LENGTH(head_weights)), 0) "
                "FROM chunk_head_edges WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()[0]
        )
        return {
            "signatures": signatures,
            "edges": edges,
            "cav_payload_bytes": cav_payload_bytes,
            "head_payload_bytes": head_payload_bytes,
            # Request-derived token IDs, Q/K/V, attention maps, residuals, and
            # generation K/V are never retained between linker invocations.
            # Static checkpoint/tokenizer assets are outside this metric.
            "retained_request_token_state_bytes": 0,
            # Compatibility alias for historical artifacts.
            "retained_token_state_bytes": 0,
        }
