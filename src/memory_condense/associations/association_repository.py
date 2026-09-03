"""Composed SQLite repository for compact association state."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any, Sequence

from memory_condense.associations.association_artifacts import (
    AssociationArtifactStoreMixin,
)
from memory_condense.associations.association_edges import (
    AssociationEdgeStoreMixin,
    head_edge_columns,
)
from memory_condense.associations.association_models import (
    AssociationArtifact,
    StoredCAVNeighbor,
    StoredHeadEdge,
)
from memory_condense.associations.hebbian_store import HebbianAssociationStoreMixin
from memory_condense.persistence.db import Database

#: Bound on each opt-in neighbor cache. No caller has ever varied it.
_NEIGHBOR_CACHE_LIMIT = 512


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
    ) -> None:
        self._db = db
        self._artifact_cache: dict[str, AssociationArtifact] = {}
        self._cav_neighbor_cache: OrderedDict[tuple, tuple[StoredCAVNeighbor, ...]] = (
            OrderedDict()
        )
        self._edge_neighbor_cache: OrderedDict[
            tuple, dict[str, tuple[StoredHeadEdge, ...]]
        ] = OrderedDict()
        self._cache_limit = _NEIGHBOR_CACHE_LIMIT if cache_neighbors else 0

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
    ) -> int:
        """Enforce a hard per-source degree cap using live edge utility."""
        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        self._require_artifact(artifact_id)
        connection = self._db.connection
        removed = 0
        try:
            connection.execute("BEGIN IMMEDIATE")
            turn = self._db.current_turn() if now_turn is None else int(now_turn)
            select_columns = f"SELECT {head_edge_columns()} FROM chunk_head_edges "
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
                            + "WHERE artifact_id = ? AND source_chunk_id IN "
                            + f"({placeholders})",
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
                    edge = self._stored_edge(row, artifact_id)
                    ranked.append(
                        (edge.utility(now_turn=turn), edge.destination_chunk_id)
                    )
                ranked.sort(reverse=True)
                for _, destination_id in ranked[max_neighbors:]:
                    deletions.append((source_id, destination_id, artifact_id))
            if deletions:
                cur = self._db.executemany(
                    "DELETE FROM chunk_head_edges WHERE source_chunk_id = ? "
                    "AND destination_chunk_id = ? AND artifact_id = ?",
                    deletions,
                )
                removed = (
                    cur.rowcount
                    if cur.rowcount is not None and cur.rowcount >= 0
                    else len(deletions)
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        if removed:
            self._edge_neighbor_cache.clear()
        return removed

    def remove_chunk_artifacts(
        self, chunk_id: str, *, commit: bool = True
    ) -> int:
        """Remove a chunk from the live graph while preserving its source row.

        ``commit=False`` is the transaction-composition seam used by chunk
        retirement; caches are invalidated immediately and are safe even if
        the caller subsequently rolls the transaction back.
        """
        connection = self._db.connection
        try:
            if commit and not connection.in_transaction:
                connection.execute("BEGIN IMMEDIATE")
            if not commit and not connection.in_transaction:
                raise RuntimeError(
                    "commit=False requires an owning SQLite transaction"
                )
            hebbian_edge_cur = self._db.execute(
                "DELETE FROM hebbian_chunk_edges "
                "WHERE chunk_low = ? OR chunk_high = ?",
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
            if commit:
                connection.commit()
        except BaseException:
            if commit:
                connection.rollback()
            raise
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
