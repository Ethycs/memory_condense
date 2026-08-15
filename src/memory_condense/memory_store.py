"""Persistence and retrieval for typed long-term memory items.

Backed by the ``memory_items`` + ``memory_provenance`` tables (schema v2).
Three invariants the rest of the system relies on:

* **Nothing is ever hard-deleted.** ``delete`` flips status to ``deleted`` and
  ``supersede`` flips the old item to ``superseded``; both rows survive so the
  audit trail back to the transcript stays intact.
* **Decay is lazy, and counted in turns.** No timer, no background job. Energy
  is decayed forward from ``last_access_turn`` on read via
  :mod:`memory_condense.decay`, and retrieval reheats what it returns. Those
  two halves *are* the mechanism: appending a turn cools everything the
  conversation did not reach for, and ``touch`` exempts everything it did.
  ``last_access_at`` still exists but is an audit timestamp only.
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
    DEFAULT_HALF_LIFE_TURNS,
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
    content_key,
)

_ITEM_COLUMNS = (
    "mem_id, type, content, details, status, supersedes, pin, energy, "
    "half_life_turns, importance, created_at, last_access_at, embedding, "
    "last_access_turn"
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
        half_life_turns: float = DEFAULT_HALF_LIFE_TURNS,
        supersedes: str | None = None,
        dedupe: bool = True,
    ) -> MemoryItem:
        """Insert a new item plus its provenance rows.

        Starting energy comes from ``decay.seed_energy(op.importance)`` —
        important items enter HOT, everything else WARM.

        **Exact duplicates are merged, not inserted.** If an active item
        already exists with the same ``(type, content)`` under
        ``schemas.content_key`` normalisation, its provenance gains the new
        citations, it is ``touch``ed — re-asserting a fact is a genuine
        salience signal — and the existing item is returned. Without this,
        re-ingesting the same text grows the store without bound, and every
        duplicate is then scanned by every :meth:`retrieve`.

        Only *exact* duplicates. Near-duplicate detection by embedding
        similarity is deliberately not done: "the beta ships on Friday" and
        "the beta ships on Monday" are highly similar and mean opposite
        things, and collapsing them would destroy the distinction
        :meth:`supersede` exists to record. Semantic conflict already has a
        mechanism, and it is not dedup.
        """
        if dedupe:
            existing = self.find_by_content(op.type, op.content)
            if existing is not None:
                self._insert_provenance(existing.mem_id, op.provenance)
                self._db.commit()
                return self.touch(existing.mem_id) or existing

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
            half_life_turns=half_life_turns,
            importance=op.importance,
            # Creation is an access: an item enters the store at the current
            # turn, not at turn 0. Without this every new memory would be born
            # already `current_turn` turns behind and go COLD immediately.
            last_access_turn=self._db.current_turn(),
            embedding=vector,
        )
        self._insert(item)
        return item

    def find_by_content(
        self, mem_type: MemoryType, content: str
    ) -> MemoryItem | None:
        """The active item with this exact content, if one exists.

        Scoped to ``status = 'active'`` on purpose: forgetting a fact and then
        remembering it again should recreate it, not silently resurrect the
        retired row.
        """
        cur = self._db.execute(
            f"SELECT {_ITEM_COLUMNS} FROM memory_items "
            "WHERE content_hash = ? AND status = ? LIMIT 1",
            (content_key(mem_type, content), MemoryStatus.ACTIVE.value),
        )
        row = cur.fetchone()
        return self._row_to_item(row) if row is not None else None

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
            "UPDATE memory_items SET content = ?, details = ?, embedding = ?, "
            "content_hash = ? WHERE mem_id = ?",
            (
                content,
                details,
                embedding_blob,
                # Must move with the content, or an amended item keeps the old
                # identity and stops deduplicating against its own new text.
                content_key(item.type, content),
                op.mem_id,
            ),
        )
        self._insert_provenance(op.mem_id, op.provenance)
        self._db.commit()
        return self.get(op.mem_id)

    def supersede(self, op: SupersedeOp) -> MemoryItem | None:
        """Mark the old item ``superseded`` and create its replacement.

        The old row is never removed: ``replacement.supersedes`` points back at
        it so the correction chain stays walkable.

        The old row is retired **first**, deliberately. In the other order a
        replacement with identical content (a details-only correction) would
        find its own still-active predecessor as an exact duplicate, merge into
        it, and return the old row — no replacement, no chain.
        """
        old = self.get(op.mem_id)
        if old is None:
            return None

        self._set_status(op.mem_id, MemoryStatus.SUPERSEDED)
        return self.create(op.replacement, supersedes=op.mem_id)

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
            "duplicate": 0,
            "updated": 0,
            "superseded": 0,
            "deleted": 0,
            "pinned": 0,
            "skipped": 0,
        }

        for create_op in ops.create:
            before = self.count()
            self.create(create_op)
            # A create that added no row was merged into an existing item.
            # Counted rather than hidden: silent merging looks like extraction
            # producing less than it did.
            if self.count() > before:
                summary["created"] += 1
            else:
                summary["duplicate"] += 1

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

    def touch(
        self,
        mem_id: str,
        now_turn: int | None = None,
        now: datetime | None = None,
    ) -> MemoryItem | None:
        """Access reheating: decay to *now_turn*, add the boost, restamp.

        **This is half the decay mechanism.** Advancing the turn is what makes
        items cool; this is what exempts the ones the conversation reached for.
        Nothing sweeps the store — an item is warm precisely because its
        ``last_access_turn`` keeps being pushed forward, and cold precisely
        because it stopped being.

        Pinned items keep their stored energy (pins override decay) but are
        still restamped so the record of the access stays honest.

        Within the same turn the item is restamped but **not** boosted: ten
        recalls while answering one turn is one access, not ten. ``now`` sets
        the audit timestamp only; it has no effect on energy.
        """
        item = self.get(mem_id)
        if item is None:
            return None

        turn = self._db.current_turn() if now_turn is None else now_turn
        stamp = now or decay.now_utc()
        if item.is_pinned:
            energy = item.energy
        else:
            energy = decay.item_energy(item, now_turn=turn)
            if decay.should_reheat(item.last_access_turn, now_turn=turn):
                energy = decay.reheat(energy)

        self._db.execute(
            "UPDATE memory_items SET energy = ?, last_access_at = ?,"
            " last_access_turn = ? WHERE mem_id = ?",
            (float(energy), stamp.isoformat(), int(turn), mem_id),
        )
        self._db.commit()
        return self.get(mem_id)

    def items_by_heat(
        self,
        now_turn: int | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        hot_cap: int = decay.HOT_CAP,
    ) -> dict[Heat, list[MemoryItem]]:
        """Items bucketed by decayed heat, with the HOT cap applied.

        A pure read — nothing is touched, restamped, or reheated. Contrast
        :meth:`retrieve`, which reheats everything it returns.
        """
        turn = self._db.current_turn() if now_turn is None else now_turn
        items = self.list_items(status=status)
        tiers = decay.heat_map(items, now_turn=turn, hot_cap=hot_cap)
        buckets: dict[Heat, list[MemoryItem]] = {h: [] for h in Heat}
        for item in items:
            buckets[tiers[item.mem_id]].append(item)
        return buckets

    def heat_counts(self, now_turn: int | None = None) -> dict[str, int]:
        """How many active items sit in each HOT/WARM/COLD tier right now.

        Before v4 this was near-vacuous — with a wall-clock coordinate and a
        run lasting minutes, every item reported HOT. It only became a real
        distribution once decay started counting turns.
        """
        buckets = self.items_by_heat(now_turn=now_turn)
        return {heat.value: len(buckets[heat]) for heat in Heat}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: Any = None,
        k: int = 10,
        weights: RankWeights = DEFAULT_WEIGHTS,
        now_turn: int | None = None,
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

        turn = self._db.current_turn() if now_turn is None else now_turn
        stamp = now or decay.now_utc()
        query_vec = _to_vector(query_embedding)

        scored: list[tuple[float, MemoryResult]] = []
        for item in items:
            energy = decay.item_energy(item, now_turn=turn)
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
                        # The turn factor alone, without the stored amplitude.
                        # Reporting both is what makes it obvious when energy
                        # is high only because the item was recently read.
                        recency=decay.decay_factor(
                            item.last_access_turn,
                            now_turn=turn,
                            half_life_turns=item.half_life_turns,
                        ),
                        pin_boost=ranking.pin_boost(item.pin),
                    ),
                )
            )

        best = ranking.top_k(scored, k)

        results: list[MemoryResult] = []
        for _, result in best:
            if reheat:
                refreshed = self.touch(result.item.mem_id, now_turn=turn, now=stamp)
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
            "half_life_turns, importance, created_at, last_access_at, "
            "last_access_turn, embedding, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.mem_id,
                item.type.value,
                item.content,
                item.details,
                item.status.value,
                item.supersedes,
                item.pin.value,
                float(item.energy),
                float(item.half_life_turns),
                float(item.importance),
                item.created_at.isoformat(),
                item.last_access_at.isoformat(),
                int(item.last_access_turn),
                _to_blob(item.embedding),
                content_key(item.type, item.content),
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

    def dedupe_existing(self) -> int:
        """Collapse pre-existing exact duplicates. Returns how many were retired.

        Opt-in maintenance for stores written before duplicate detection
        existed. Deliberately **not** run from the migration: it changes data,
        and opening a database should not silently rewrite it.

        Nothing is destroyed — losers become ``superseded`` with ``supersedes``
        pointing at the survivor, so clause 9 holds and the chain stays
        walkable. The newest row survives, on the grounds that it carries the
        most recent provenance.
        """
        rows = self._db.execute(
            "SELECT content_hash FROM memory_items "
            "WHERE status = ? AND content_hash IS NOT NULL "
            "GROUP BY content_hash HAVING COUNT(*) > 1",
            (MemoryStatus.ACTIVE.value,),
        ).fetchall()

        retired = 0
        for (digest,) in rows:
            dupes = self._db.execute(
                "SELECT mem_id FROM memory_items "
                "WHERE content_hash = ? AND status = ? ORDER BY created_at DESC",
                (digest, MemoryStatus.ACTIVE.value),
            ).fetchall()
            survivor = dupes[0][0]
            for (loser,) in dupes[1:]:
                self._db.execute(
                    "UPDATE memory_items SET status = ?, supersedes = ? "
                    "WHERE mem_id = ?",
                    (MemoryStatus.SUPERSEDED.value, survivor, loser),
                )
                retired += 1
        self._db.commit()
        return retired

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
            half_life_turns=row[8],
            importance=row[9],
            created_at=datetime.fromisoformat(row[10]),
            last_access_at=datetime.fromisoformat(row[11]),
            last_access_turn=row[13],
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
