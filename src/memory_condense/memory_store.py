"""Persistence and retrieval for typed long-term memory items.

Backed by the ``memory_items`` + ``memory_provenance`` tables (schema v2).
Three invariants the rest of the system relies on:

* **Nothing is ever hard-deleted.** ``delete`` flips status to ``deleted`` and
  ``supersede`` flips the old item to ``superseded``; both rows survive so the
  audit trail back to the transcript stays intact.
* **Decay is lazy.** No timer, no background job. Energy is decayed forward
  from ``last_access_at`` on read via :mod:`memory_condense.decay`, and
  retrieval reheats what it returns.
* **Provenance travels with the item.** Loading an item always loads its
  provenance rows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Optional, Protocol

import numpy as np

from memory_condense import decay, ranking
from memory_condense.db import Database
from memory_condense.ranking import DEFAULT_WEIGHTS, RankWeights
from memory_condense.schemas import (
    DEFAULT_HALF_LIFE_S,
    CreateOp,
    DeleteOp,
    Heat,
    MemoryItem,
    MemoryOps,
    MemoryResult,
    MemoryStatus,
    MemoryType,
    PinOp,
    PinState,
    Provenance,
    SupersedeOp,
    UpdateOp,
    ValidationReport,
)

_ITEM_COLUMNS = (
    "mem_id, type, content, details, status, supersedes, pin, energy, "
    "half_life_s, importance, created_at, last_access_at, embedding"
)


class Embedder(Protocol):
    """Duck-typed embedding provider (see ``embedding.BGEM3Embedder``)."""

    def embed_query(self, query: str) -> np.ndarray:  # pragma: no cover - protocol
        ...


class MemoryStore:
    """CRUD + ranked retrieval over memory items.

    ``embedder`` is optional and only used to vectorize new items when the
    caller does not supply an embedding. Anything exposing
    ``embed_query(str) -> np.ndarray`` works.
    """

    def __init__(self, db: Database, embedder: Optional[Embedder] = None) -> None:
        self._db = db
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Create / read
    # ------------------------------------------------------------------

    def create(
        self,
        op: CreateOp,
        embedding: Any = None,
        half_life_s: float = DEFAULT_HALF_LIFE_S,
        supersedes: str | None = None,
    ) -> MemoryItem:
        """Insert a new item plus its provenance rows.

        Starting energy comes from ``decay.seed_energy(op.importance)`` —
        important items enter HOT, everything else WARM.
        """
        vector = self._resolve_embedding(op.content, embedding)
        item = MemoryItem(
            type=op.type,
            content=op.content,
            details=op.details,
            provenance=list(op.provenance),
            status=MemoryStatus.ACTIVE,
            supersedes=supersedes,
            pin=PinState.NONE,
            energy=decay.seed_energy(op.importance),
            half_life_s=half_life_s,
            importance=op.importance,
            embedding=vector,
        )
        self._insert(item)
        return item

    def get(self, mem_id: str) -> MemoryItem | None:
        cur = self._db.execute(
            f"SELECT {_ITEM_COLUMNS} FROM memory_items WHERE mem_id = ?", (mem_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_item(row)

    def list_items(
        self,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        """Items with the given status (``None`` for every status), newest first."""
        sql = f"SELECT {_ITEM_COLUMNS} FROM memory_items"
        params: tuple = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status.value,)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params = params + (int(limit),)

        cur = self._db.execute(sql, params)
        return [self._row_to_item(row) for row in cur.fetchall()]

    def count(self, status: MemoryStatus | None = None) -> int:
        if status is None:
            cur = self._db.execute("SELECT COUNT(*) FROM memory_items")
        else:
            cur = self._db.execute(
                "SELECT COUNT(*) FROM memory_items WHERE status = ?", (status.value,)
            )
        return int(cur.fetchone()[0])

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update(self, op: UpdateOp) -> MemoryItem | None:
        """Amend content/details in place and append any new provenance.

        Timestamps and energy are left alone — an amendment is not an access.
        Use ``supersede`` for semantic reversals.
        """
        item = self.get(op.mem_id)
        if item is None:
            return None

        content = op.content if op.content is not None else item.content
        details = op.details if op.details is not None else item.details

        embedding_blob = _to_blob(item.embedding)
        if op.content is not None and op.content != item.content:
            new_vector = self._resolve_embedding(content, None)
            if new_vector is not None:
                embedding_blob = _to_blob(new_vector)

        self._db.execute(
            "UPDATE memory_items SET content = ?, details = ?, embedding = ? "
            "WHERE mem_id = ?",
            (content, details, embedding_blob, op.mem_id),
        )
        self._insert_provenance(op.mem_id, op.provenance)
        self._db.commit()
        return self.get(op.mem_id)

    def supersede(self, op: SupersedeOp) -> MemoryItem | None:
        """Create the replacement and mark the old item ``superseded``.

        The old row is never removed: ``replacement.supersedes`` points back at
        it so the correction chain stays walkable.
        """
        old = self.get(op.mem_id)
        if old is None:
            return None

        replacement = self.create(op.replacement, supersedes=op.mem_id)
        self._set_status(op.mem_id, MemoryStatus.SUPERSEDED)
        return replacement

    def delete(self, op: DeleteOp) -> bool:
        """Soft-delete: status becomes ``deleted``, the row survives."""
        if self.get(op.mem_id) is None:
            return False
        self._set_status(op.mem_id, MemoryStatus.DELETED)
        return True

    def pin(self, op: PinOp) -> MemoryItem | None:
        """Pin or unpin an item. Pinned items are exempt from decay."""
        if self.get(op.mem_id) is None:
            return None
        self._db.execute(
            "UPDATE memory_items SET pin = ? WHERE mem_id = ?",
            (op.pin.value, op.mem_id),
        )
        self._db.commit()
        return self.get(op.mem_id)

    def apply(self, report_or_ops: ValidationReport | MemoryOps) -> dict[str, int]:
        """Apply a validated batch in create → update → supersede → delete → pin order.

        Accepts either a ``ValidationReport`` (its ``accepted`` ops are used) or
        a raw ``MemoryOps``. Returns a count summary; ops that reference a
        missing ``mem_id`` are counted under ``"skipped"`` rather than raising.
        """
        ops = (
            report_or_ops.accepted
            if isinstance(report_or_ops, ValidationReport)
            else report_or_ops
        )

        summary = {
            "created": 0,
            "updated": 0,
            "superseded": 0,
            "deleted": 0,
            "pinned": 0,
            "skipped": 0,
        }

        for create_op in ops.create:
            self.create(create_op)
            summary["created"] += 1

        for update_op in ops.update:
            if self.update(update_op) is not None:
                summary["updated"] += 1
            else:
                summary["skipped"] += 1

        for sup_op in ops.supersede:
            if self.supersede(sup_op) is not None:
                summary["superseded"] += 1
            else:
                summary["skipped"] += 1

        for del_op in ops.delete:
            if self.delete(del_op):
                summary["deleted"] += 1
            else:
                summary["skipped"] += 1

        for pin_op in ops.pin:
            if self.pin(pin_op) is not None:
                summary["pinned"] += 1
            else:
                summary["skipped"] += 1

        return summary

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def touch(self, mem_id: str, now: datetime | None = None) -> MemoryItem | None:
        """Access reheating: decay to *now*, add the reheat boost, restamp.

        Pinned items keep their stored energy (pins override decay) but their
        ``last_access_at`` is still refreshed so the timestamp stays honest.

        Within :data:`decay.REHEAT_REFRACTORY_S` of the previous access the
        item is decayed and restamped but **not** boosted: a burst of recalls
        in one working session is one access, not ten. Restamping is still
        correct — the access really happened — and the withheld decay over a
        five-minute window is negligible against a seven-day half-life.
        """
        item = self.get(mem_id)
        if item is None:
            return None

        stamp = now or decay.now_utc()
        if item.is_pinned:
            energy = item.energy
        else:
            energy = decay.item_energy(item, now=stamp)
            if decay.should_reheat(item.last_access_at, now=stamp):
                energy = decay.reheat(energy)

        self._db.execute(
            "UPDATE memory_items SET energy = ?, last_access_at = ? WHERE mem_id = ?",
            (float(energy), stamp.isoformat(), mem_id),
        )
        self._db.commit()
        return self.get(mem_id)

    def items_by_heat(
        self,
        now: datetime | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        hot_cap: int = decay.HOT_CAP,
    ) -> dict[Heat, list[MemoryItem]]:
        """Items bucketed by decayed heat, with the HOT cap applied.

        A pure read — nothing is touched, restamped, or reheated. Contrast
        :meth:`retrieve`, which reheats everything it returns.
        """
        items = self.list_items(status=status)
        tiers = decay.heat_map(items, now=now, hot_cap=hot_cap)
        buckets: dict[Heat, list[MemoryItem]] = {h: [] for h in Heat}
        for item in items:
            buckets[tiers[item.mem_id]].append(item)
        return buckets

    def heat_counts(self, now: datetime | None = None) -> dict[str, int]:
        """How many active items sit in each HOT/WARM/COLD tier right now."""
        buckets = self.items_by_heat(now=now)
        return {heat.value: len(buckets[heat]) for heat in Heat}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: Any = None,
        k: int = 10,
        weights: RankWeights = DEFAULT_WEIGHTS,
        now: datetime | None = None,
        include_superseded: bool = False,
        min_energy: float = 0.0,
        reheat: bool = True,
    ) -> list[MemoryResult]:
        """Rank memory items by the deterministic rerank scalar.

        Similarity is **brute-force exact cosine with numpy** over the stored
        embeddings. There is deliberately no second ANN index here: memory
        items are few (tens to low hundreds) so an exact scan is both faster
        and simpler than maintaining another hnswlib index alongside the chunk
        one, and it never returns stale neighbours after a supersede.

        Cosine is mapped from [-1, 1] into [0, 1] via ``(cos + 1) / 2`` so it
        composes with the other [0, 1] rank components. Items without an
        embedding — or any item when ``query_embedding`` is None — score
        ``relevance = 0`` and fall back to importance/energy/pin.

        ``min_energy`` drops items whose decayed energy is below the given
        floor. It defaults to 0.0 — **off** — deliberately: a rarely-touched
        Constraint that exactly answers the query is precisely what must not be
        dropped, which is what ``relevance`` at weight 1.0 is for. The
        parameter exists so the cost of a COLD cutoff can be *measured* rather
        than assumed.

        ``reheat=False`` scores without recording an access, for callers that
        inspect rankings rather than put items in front of a model.

        Every returned item is otherwise ``touch``ed (access reheating), so the
        items in the results reflect their post-reheat energy while the scores
        reflect the state at query time.
        """
        if k <= 0:
            return []

        statuses = [MemoryStatus.ACTIVE]
        if include_superseded:
            statuses.append(MemoryStatus.SUPERSEDED)

        items: list[MemoryItem] = []
        for status in statuses:
            items.extend(self.list_items(status=status))
        if not items:
            return []

        stamp = now or decay.now_utc()
        query_vec = _to_vector(query_embedding)

        scored: list[tuple[float, MemoryResult]] = []
        for item in items:
            energy = decay.item_energy(item, now=stamp)
            if energy < min_energy:
                continue
            relevance = self._relevance(query_vec, item)
            superseded = item.status is MemoryStatus.SUPERSEDED
            score = ranking.rank_score(
                relevance=relevance,
                importance=item.importance,
                pin=item.pin,
                energy=energy,
                superseded=superseded,
                weights=weights,
            )
            scored.append(
                (
                    score,
                    MemoryResult(
                        item=item,
                        score=score,
                        relevance=relevance,
                        importance=item.importance,
                        energy=energy,
                        # The time factor alone, without the stored amplitude.
                        # Reporting both is what makes it obvious when energy
                        # is high only because the item was recently read.
                        recency=decay.decay_factor(
                            item.last_access_at,
                            now=stamp,
                            half_life_s=item.half_life_s,
                        ),
                        pin_boost=ranking.pin_boost(item.pin),
                    ),
                )
            )

        best = ranking.top_k(scored, k)

        results: list[MemoryResult] = []
        for _, result in best:
            if reheat:
                refreshed = self.touch(result.item.mem_id, now=stamp)
                if refreshed is not None:
                    result = result.model_copy(update={"item": refreshed})
            results.append(result)
        return results

    @staticmethod
    def _relevance(query_vec: np.ndarray | None, item: MemoryItem) -> float:
        if query_vec is None or not item.embedding:
            return 0.0
        item_vec = np.asarray(item.embedding, dtype=np.float32)
        if item_vec.shape != query_vec.shape:
            return 0.0
        return _cosine_to_unit(_cosine(query_vec, item_vec))

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _insert(self, item: MemoryItem) -> None:
        self._db.execute(
            "INSERT INTO memory_items "
            "(mem_id, type, content, details, status, supersedes, pin, energy, "
            "half_life_s, importance, created_at, last_access_at, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.mem_id,
                item.type.value,
                item.content,
                item.details,
                item.status.value,
                item.supersedes,
                item.pin.value,
                float(item.energy),
                float(item.half_life_s),
                float(item.importance),
                item.created_at.isoformat(),
                item.last_access_at.isoformat(),
                _to_blob(item.embedding),
            ),
        )
        self._insert_provenance(item.mem_id, item.provenance)
        self._db.commit()

    def _insert_provenance(
        self, mem_id: str, provenance: Iterable[Provenance]
    ) -> None:
        rows = [(mem_id, p.turn_id, p.chunk_id, p.quote) for p in provenance]
        if not rows:
            return
        self._db.executemany(
            "INSERT OR IGNORE INTO memory_provenance "
            "(mem_id, turn_id, chunk_id, quote) VALUES (?, ?, ?, ?)",
            rows,
        )

    def _set_status(self, mem_id: str, status: MemoryStatus) -> None:
        self._db.execute(
            "UPDATE memory_items SET status = ? WHERE mem_id = ?",
            (status.value, mem_id),
        )
        self._db.commit()

    def _load_provenance(self, mem_id: str) -> list[Provenance]:
        cur = self._db.execute(
            "SELECT turn_id, chunk_id, quote FROM memory_provenance WHERE mem_id = ?",
            (mem_id,),
        )
        return [
            Provenance(turn_id=row[0], chunk_id=row[1], quote=row[2])
            for row in cur.fetchall()
        ]

    def _row_to_item(self, row: tuple) -> MemoryItem:
        embedding = None
        if row[12] is not None:
            embedding = np.frombuffer(row[12], dtype=np.float32).tolist()

        return MemoryItem(
            mem_id=row[0],
            type=MemoryType(row[1]),
            content=row[2],
            details=row[3],
            provenance=self._load_provenance(row[0]),
            status=MemoryStatus(row[4]),
            supersedes=row[5],
            pin=PinState(row[6]),
            energy=row[7],
            half_life_s=row[8],
            importance=row[9],
            created_at=datetime.fromisoformat(row[10]),
            last_access_at=datetime.fromisoformat(row[11]),
            embedding=embedding,
        )

    def _resolve_embedding(self, text: str, embedding: Any) -> list[float] | None:
        if embedding is not None:
            return [float(x) for x in np.asarray(embedding, dtype=np.float32).ravel()]
        if self._embedder is None:
            return None
        vector = self._embedder.embed_query(text)
        return [float(x) for x in np.asarray(vector, dtype=np.float32).ravel()]


# ----------------------------------------------------------------------
# Vector helpers (float32 blobs, matching retrieval.py's chunk convention)
# ----------------------------------------------------------------------


def _to_blob(embedding: Any) -> bytes | None:
    if embedding is None:
        return None
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _to_vector(embedding: Any) -> np.ndarray | None:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).ravel()
    return vector if vector.size else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cosine_to_unit(cosine: float) -> float:
    """Map cosine similarity from [-1, 1] into [0, 1]."""
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
