"""Persistence and retrieval for typed long-term memory items.

Backed by the ``memory_items`` + ``memory_provenance`` tables (schema v2).
Three invariants the rest of the system relies on:

* **Nothing is ever hard-deleted.** ``delete`` flips status to ``deleted`` and
  ``supersede`` flips the old item to ``superseded``; both rows survive so the
  audit trail back to the transcript stays intact.
* **Decay is lazy, and counted in turns.** No timer, no background job. Energy
  is decayed forward from ``last_access_turn`` on read via
  :mod:`memory_condense.domain.decay`, and retrieval reheats what it returns. Those
  two halves *are* the mechanism: appending a turn cools everything the
  conversation did not reach for, and ``touch`` exempts everything it did.
  ``last_access_at`` still exists but is an audit timestamp only.
* **Provenance travels with the item.** Loading an item always loads its
  provenance rows.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Iterable, Optional, Protocol

import numpy as np

from memory_condense.domain import decay, ranking
from memory_condense.persistence.db import Database
from memory_condense.domain.ranking import DEFAULT_WEIGHTS, RankWeights
from memory_condense.domain.schemas import (
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

# One canonical column list for ``memory_items`` rows. The SELECT list, the
# INSERT statement, and the row reader are all derived from it, so the three
# can never drift out of positional sync. ``content_hash`` is write-only
# (recomputed from type + content on read) and appended at INSERT time.
_COLUMNS = (
    "mem_id",
    "type",
    "content",
    "details",
    "status",
    "supersedes",
    "pin",
    "energy",
    "half_life_turns",
    "importance",
    "created_at",
    "last_access_at",
    "last_access_turn",
    "embedding",
)
_ITEM_COLUMNS = ", ".join(_COLUMNS)

# Per-column coercions between MemoryItem attributes and stored values.
# Columns absent from a map pass through unchanged (plain TEXT / numerics).
_TO_DB = {
    "type": lambda v: v.value,
    "status": lambda v: v.value,
    "pin": lambda v: v.value,
    "energy": float,
    "half_life_turns": float,
    "importance": float,
    "created_at": lambda v: v.isoformat(),
    "last_access_at": lambda v: v.isoformat(),
    "last_access_turn": int,
    "embedding": lambda v: _to_blob(v),
}
_FROM_DB = {
    "type": MemoryType,
    "status": MemoryStatus,
    "pin": PinState,
    "created_at": datetime.fromisoformat,
    "last_access_at": datetime.fromisoformat,
    "embedding": lambda v: (
        None if v is None else np.frombuffer(v, dtype=np.float32).tolist()
    ),
}

# Same single-source treatment for the ``memory_provenance`` round-trip
# (``mem_id`` is the join key, supplied separately on both paths).
_PROVENANCE_COLUMNS = ("turn_id", "chunk_id", "quote")


def _acquire_write_transaction(connection: sqlite3.Connection) -> None:
    """Start an immediate transaction or upgrade an existing caller one.

    Public memory mutations historically commit the connection, including
    writes a caller staged immediately beforehand. Preserve that ownership
    while still acquiring SQLite's write lock before a duplicate/status
    precondition is read.
    """
    if connection.in_transaction:
        connection.execute(
            "UPDATE meta SET value = value WHERE key = 'schema_version'"
        )
    else:
        connection.execute("BEGIN IMMEDIATE")


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

        Supplying ``supersedes`` is retained for API compatibility, but is no
        longer a way to write a bare pointer. It performs the same atomic
        status transition as :meth:`supersede` and rejects a missing or
        already-retired predecessor. That keeps every public creation path
        from manufacturing a dangling correction chain.
        """
        self._require_writable()
        if supersedes is not None:
            replacement = self._supersede_create(
                supersedes,
                op,
                embedding=embedding,
                half_life_turns=half_life_turns,
            )
            if replacement is None:
                raise ValueError(
                    "supersedes must name an active memory item; "
                    f"got {supersedes!r}"
                )
            return replacement

        if dedupe:
            existing = self.find_by_content(op.type, op.content)
            if existing is not None:
                merged = self._merge_active_duplicate(op)
                if merged is not None:
                    return merged

        item = self._build_item(
            op,
            embedding=embedding,
            half_life_turns=half_life_turns,
        )
        if dedupe:
            return self._insert_or_merge_active(item)
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

    def successors(self, mem_id: str) -> list[MemoryItem]:
        """Return direct successors without losing duplicate-merge edges.

        Ordinary revisions use the backwards-compatible scalar
        ``replacement.supersedes`` pointer. Exact-content coalescence can have
        more than one predecessor, so its additional forward redirects live in
        ``memory_successor_redirects``. Reading the union here keeps both kinds
        of history walkable through one public API. Pre-v12
        :meth:`dedupe_existing` wrote its exact-duplicate pointer in the
        opposite direction; a newer identical referenced row is recognized as
        that legacy layout without rewriting the historical row.
        """
        predecessor = self.get(mem_id)
        if predecessor is None:
            return []

        candidates = [
            self.get(row[0])
            for row in self._db.execute(
                "SELECT mem_id FROM memory_items WHERE supersedes = ? "
                "UNION SELECT successor_mem_id FROM memory_successor_redirects "
                "WHERE predecessor_mem_id = ?",
                (mem_id, mem_id),
            ).fetchall()
        ]
        predecessor_key = content_key(predecessor.type, predecessor.content)
        successors = {
            item.mem_id: item
            for item in candidates
            if item is not None
            and not (
                item.status is MemoryStatus.SUPERSEDED
                and content_key(item.type, item.content) == predecessor_key
                and (
                    predecessor.status is MemoryStatus.ACTIVE
                    or item.created_at < predecessor.created_at
                )
            )
        }

        if predecessor.status is MemoryStatus.SUPERSEDED and predecessor.supersedes:
            legacy_target = self.get(predecessor.supersedes)
            if (
                legacy_target is not None
                and content_key(legacy_target.type, legacy_target.content)
                == predecessor_key
                and (
                    legacy_target.status is MemoryStatus.ACTIVE
                    or legacy_target.created_at > predecessor.created_at
                )
            ):
                successors[legacy_target.mem_id] = legacy_target

        return [successors[key] for key in sorted(successors)]

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
        rows = cur.fetchall()
        provenance = self._load_provenance_many(row[0] for row in rows)
        return [self._row_to_item(row, provenance[row[0]]) for row in rows]

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
        self._require_writable()
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

        connection = self._db.connection
        try:
            _acquire_write_transaction(connection)
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
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return self.get(op.mem_id)

    def supersede(self, op: SupersedeOp) -> MemoryItem | None:
        """Mark the old item ``superseded`` and create its replacement.

        The old row is never removed: ``replacement.supersedes`` points back at
        it so the correction chain stays walkable.

        Embedding, retirement, replacement insertion, provenance merging, and
        exact-duplicate redirects form one transaction. A failure therefore
        leaves the predecessor active. A correction always gets a fresh row;
        it never silently resolves to an unrelated active duplicate.
        """
        self._require_writable()
        return self._supersede_create(op.mem_id, op.replacement)

    def delete(self, op: DeleteOp) -> bool:
        """Soft-delete: status becomes ``deleted``, the row survives."""
        self._require_writable()
        if self.get(op.mem_id) is None:
            return False
        self._set_status(op.mem_id, MemoryStatus.DELETED)
        return True

    def pin(self, op: PinOp) -> MemoryItem | None:
        """Pin or unpin an item. Pinned items are exempt from decay."""
        self._require_writable()
        if self.get(op.mem_id) is None:
            return None
        try:
            self._db.execute(
                "UPDATE memory_items SET pin = ? WHERE mem_id = ?",
                (op.pin.value, op.mem_id),
            )
            self._db.commit()
        except BaseException:
            self._db.connection.rollback()
            raise
        return self.get(op.mem_id)

    def apply(self, report_or_ops: ValidationReport | MemoryOps) -> dict[str, int]:
        """Apply a validated batch in create → update → supersede → delete → pin order.

        Accepts either a ``ValidationReport`` (its ``accepted`` ops are used) or
        a raw ``MemoryOps``. Returns a count summary; ops that reference a
        missing ``mem_id`` are counted under ``"skipped"`` rather than raising.
        """
        self._require_writable()
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
        touched = self.touch_many([mem_id], now_turn=now_turn, now=now)
        return touched[0] if touched else None

    def touch_many(
        self,
        memories: Iterable[str | MemoryItem],
        now_turn: int | None = None,
        now: datetime | None = None,
    ) -> list[MemoryItem]:
        """Reheat several memory rows in one SQLite transaction.

        Context assembly commonly exposes 8–50 items at once.  Calling
        :meth:`touch` in a loop previously paid one durable commit per item;
        the access is one logical event and should be one transaction.  Input
        order is preserved and duplicate ids are touched once. Callers that
        already loaded ranked items may pass those objects and avoid another
        SELECT per row; ids remain accepted for the ordinary public API.
        """
        self._require_writable()
        seen: set[str] = set()
        items: list[MemoryItem] = []
        for memory in memories:
            item = memory if isinstance(memory, MemoryItem) else self.get(memory)
            if item is None:
                continue
            mem_id = item.mem_id
            if mem_id in seen:
                continue
            seen.add(mem_id)
            items.append(item)
        return self._touch_items(items, now_turn=now_turn, now=now)

    def _touch_items(
        self,
        items: list[MemoryItem],
        now_turn: int | None = None,
        now: datetime | None = None,
    ) -> list[MemoryItem]:
        """Batch-update already-loaded items, avoiding reads and commits per row."""
        if not items:
            return []
        self._require_writable()
        turn = self._db.current_turn() if now_turn is None else now_turn
        stamp = now or decay.now_utc()
        updates: list[tuple[float, str, int, str]] = []
        refreshed: list[MemoryItem] = []
        for item in items:
            if item.is_pinned:
                energy = item.energy
            else:
                energy = decay.item_energy(item, now_turn=turn)
                if decay.should_reheat(item.last_access_turn, now_turn=turn):
                    energy = decay.reheat(energy)
            updates.append((float(energy), stamp.isoformat(), int(turn), item.mem_id))
            refreshed.append(
                item.model_copy(
                    update={
                        "energy": float(energy),
                        "last_access_at": stamp,
                        "last_access_turn": int(turn),
                    }
                )
            )

        try:
            self._db.executemany(
                "UPDATE memory_items SET energy = ?, last_access_at = ?,"
                " last_access_turn = ? WHERE mem_id = ?",
                updates,
            )
            self._db.commit()
        except BaseException:
            self._db.connection.rollback()
            raise
        return refreshed

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
            rows = self._db.execute(
                f"SELECT {_ITEM_COLUMNS} FROM memory_items "
                "WHERE status = ? ORDER BY created_at DESC",
                (status.value,),
            ).fetchall()
            # Provenance is irrelevant to scoring. Hydrate it for the top-k
            # only, after ranking, instead of issuing one query per candidate.
            items.extend(self._row_to_item(row, []) for row in rows)
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

        results = [result for _, result in best]
        provenance = self._load_provenance_many(
            result.item.mem_id for result in results
        )
        results = [
            result.model_copy(
                update={
                    "item": result.item.model_copy(
                        update={"provenance": provenance[result.item.mem_id]}
                    )
                }
            )
            for result in results
        ]
        if not reheat or self._db.read_only:
            return results

        refreshed = self._touch_items(
            [result.item for result in results], now_turn=turn, now=stamp
        )
        by_id = {item.mem_id: item for item in refreshed}
        return [
            result.model_copy(update={"item": by_id[result.item.mem_id]})
            for result in results
        ]

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

    def _build_item(
        self,
        op: CreateOp,
        *,
        embedding: Any = None,
        half_life_turns: float = DEFAULT_HALF_LIFE_TURNS,
        supersedes: str | None = None,
    ) -> MemoryItem:
        """Resolve fallible inputs before a mutation transaction begins."""
        return MemoryItem.from_create(
            op,
            embedding=self._resolve_embedding(op.content, embedding),
            half_life_turns=half_life_turns,
            supersedes=supersedes,
            # Creation is an access: an item enters the store at the current
            # turn, not at turn 0. Without this every new memory would be born
            # already `current_turn` turns behind and go COLD immediately.
            last_access_turn=self._db.current_turn(),
        )

    def _supersede_create(
        self,
        predecessor_id: str,
        replacement: CreateOp,
        *,
        embedding: Any = None,
        half_life_turns: float = DEFAULT_HALF_LIFE_TURNS,
    ) -> MemoryItem | None:
        """Atomically retire one active row and publish its fresh successor.

        Exact-content deduplication cannot return an already-active row here:
        that would retire ``predecessor_id`` without creating the canonical
        backwards link. Instead a fresh successor is inserted, citations from
        any other active exact duplicates are merged into it, and those rows
        are retired with explicit forward redirects. Their own ``supersedes``
        pointers are deliberately left untouched, preserving earlier chains.
        """
        self._require_writable()
        predecessor = self.get(predecessor_id)
        if predecessor is None or predecessor.status is not MemoryStatus.ACTIVE:
            return None

        # Embedding may call an external/model provider. Resolve it before any
        # status mutation so provider failure cannot orphan the predecessor.
        item = self._build_item(
            replacement,
            embedding=embedding,
            half_life_turns=half_life_turns,
            supersedes=predecessor_id,
        )
        content_hash = content_key(replacement.type, replacement.content)
        connection = self._db.connection

        try:
            _acquire_write_transaction(connection)
            retired = self._db.execute(
                "UPDATE memory_items SET status = ? "
                "WHERE mem_id = ? AND status = ?",
                (
                    MemoryStatus.SUPERSEDED.value,
                    predecessor_id,
                    MemoryStatus.ACTIVE.value,
                ),
            )
            # Another writer may have retired it while the embedding was being
            # produced. Publish no replacement unless this transaction owns
            # the active -> superseded transition.
            if retired.rowcount != 1:
                connection.rollback()
                return None

            duplicate_ids = [
                row[0]
                for row in self._db.execute(
                    "SELECT mem_id FROM memory_items "
                    "WHERE content_hash = ? AND status = ? AND mem_id <> ? "
                    "ORDER BY created_at, mem_id",
                    (
                        content_hash,
                        MemoryStatus.ACTIVE.value,
                        predecessor_id,
                    ),
                ).fetchall()
            ]

            self._insert(item, commit=False)
            for duplicate_id in duplicate_ids:
                self._db.execute(
                    "INSERT OR IGNORE INTO memory_provenance "
                    "(mem_id, turn_id, chunk_id, quote) "
                    "SELECT ?, turn_id, chunk_id, quote "
                    "FROM memory_provenance WHERE mem_id = ?",
                    (item.mem_id, duplicate_id),
                )
                self._db.execute(
                    "UPDATE memory_items SET status = ? "
                    "WHERE mem_id = ? AND status = ?",
                    (
                        MemoryStatus.SUPERSEDED.value,
                        duplicate_id,
                        MemoryStatus.ACTIVE.value,
                    ),
                )
                self._db.execute(
                    "INSERT INTO memory_successor_redirects "
                    "(predecessor_mem_id, successor_mem_id, reason, created_at) "
                    "VALUES (?, ?, 'exact_duplicate_merge', ?)",
                    (duplicate_id, item.mem_id, item.created_at.isoformat()),
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

        # Rehydrate after commit so merged provenance is part of the returned
        # object as well as the durable row.
        return self.get(item.mem_id)

    def _insert(self, item: MemoryItem, *, commit: bool = True) -> None:
        values = tuple(
            _TO_DB[column](getattr(item, column))
            if column in _TO_DB
            else getattr(item, column)
            for column in _COLUMNS
        )
        try:
            self._db.execute(
                f"INSERT INTO memory_items ({_ITEM_COLUMNS}, content_hash) "
                f"VALUES ({', '.join('?' * (len(_COLUMNS) + 1))})",
                values + (content_key(item.type, item.content),),
            )
            self._insert_provenance(item.mem_id, item.provenance)
            if commit:
                self._db.commit()
        except BaseException:
            if commit:
                self._db.connection.rollback()
            raise

    def _merge_active_duplicate(self, op: CreateOp) -> MemoryItem | None:
        """Serialize duplicate adoption and its provenance/heat mutation."""
        connection = self._db.connection
        try:
            _acquire_write_transaction(connection)
            existing = self.find_by_content(op.type, op.content)
            if existing is None:
                connection.rollback()
                return None
            self._insert_provenance(existing.mem_id, op.provenance)
            # ``_touch_items`` commits the transaction containing both the
            # citations and heat update. Reload so the caller sees the newly
            # merged provenance rather than the pre-transaction snapshot.
            self._touch_items([existing])
        except BaseException:
            connection.rollback()
            raise
        return self.get(existing.mem_id)

    def _insert_or_merge_active(self, item: MemoryItem) -> MemoryItem:
        """Atomically recheck exact identity before publishing a new row."""
        connection = self._db.connection
        try:
            _acquire_write_transaction(connection)
            existing = self.find_by_content(item.type, item.content)
            if existing is not None:
                self._insert_provenance(existing.mem_id, item.provenance)
                self._touch_items([existing])
                return self.get(existing.mem_id) or existing
            self._insert(item, commit=False)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return item

    def _insert_provenance(
        self, mem_id: str, provenance: Iterable[Provenance]
    ) -> None:
        rows = [
            (mem_id,) + tuple(getattr(p, column) for column in _PROVENANCE_COLUMNS)
            for p in provenance
        ]
        if not rows:
            return
        self._db.executemany(
            "INSERT OR IGNORE INTO memory_provenance "
            f"(mem_id, {', '.join(_PROVENANCE_COLUMNS)}) "
            f"VALUES ({', '.join('?' * (len(_PROVENANCE_COLUMNS) + 1))})",
            rows,
        )

    def dedupe_existing(self) -> int:
        """Collapse pre-existing exact duplicates. Returns how many were retired.

        Opt-in maintenance for stores written before duplicate detection
        existed. Deliberately **not** run from the migration: it changes data,
        and opening a database should not silently rewrite it.

        Nothing is destroyed — losers become ``superseded`` and gain explicit
        forward redirects to the survivor, so clause 9 holds and the chain
        stays walkable. A loser's existing backwards ``supersedes`` pointer is
        preserved rather than overwritten. The newest row survives, on the
        grounds that it carries the most recent provenance.
        """
        self._require_writable()
        connection = self._db.connection
        retired = 0
        try:
            _acquire_write_transaction(connection)
            rows = self._db.execute(
                "SELECT content_hash FROM memory_items "
                "WHERE status = ? AND content_hash IS NOT NULL "
                "GROUP BY content_hash HAVING COUNT(*) > 1",
                (MemoryStatus.ACTIVE.value,),
            ).fetchall()

            for (digest,) in rows:
                dupes = self._db.execute(
                    "SELECT mem_id FROM memory_items "
                    "WHERE content_hash = ? AND status = ? "
                    "ORDER BY created_at DESC, mem_id DESC",
                    (digest, MemoryStatus.ACTIVE.value),
                ).fetchall()
                survivor = dupes[0][0]
                for (loser,) in dupes[1:]:
                    self._db.execute(
                        "INSERT OR IGNORE INTO memory_provenance "
                        "(mem_id, turn_id, chunk_id, quote) "
                        "SELECT ?, turn_id, chunk_id, quote "
                        "FROM memory_provenance WHERE mem_id = ?",
                        (survivor, loser),
                    )
                    changed = self._db.execute(
                        "UPDATE memory_items SET status = ? "
                        "WHERE mem_id = ? AND status = ?",
                        (
                            MemoryStatus.SUPERSEDED.value,
                            loser,
                            MemoryStatus.ACTIVE.value,
                        ),
                    )
                    if changed.rowcount != 1:
                        continue
                    self._db.execute(
                        "INSERT INTO memory_successor_redirects "
                        "(predecessor_mem_id, successor_mem_id, reason, created_at) "
                        "VALUES (?, ?, 'exact_duplicate_merge', ?)",
                        (loser, survivor, decay.now_utc().isoformat()),
                    )
                    retired += 1
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return retired

    def _require_writable(self) -> None:
        """Reject mutation before embedding or any other fallible side effect."""
        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")

    def _set_status(self, mem_id: str, status: MemoryStatus) -> None:
        try:
            self._db.execute(
                "UPDATE memory_items SET status = ? WHERE mem_id = ?",
                (status.value, mem_id),
            )
            self._db.commit()
        except BaseException:
            self._db.connection.rollback()
            raise

    def _load_provenance(self, mem_id: str) -> list[Provenance]:
        cur = self._db.execute(
            f"SELECT {', '.join(_PROVENANCE_COLUMNS)} "
            "FROM memory_provenance WHERE mem_id = ?",
            (mem_id,),
        )
        return [
            Provenance(**dict(zip(_PROVENANCE_COLUMNS, row)))
            for row in cur.fetchall()
        ]

    def _load_provenance_many(
        self, mem_ids: Iterable[str]
    ) -> dict[str, list[Provenance]]:
        """Load provenance for many items without an N+1 query sequence."""
        ids = list(dict.fromkeys(mem_ids))
        out: dict[str, list[Provenance]] = {mem_id: [] for mem_id in ids}
        # Stay below SQLite builds whose variable limit is the traditional 999.
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                f"SELECT mem_id, {', '.join(_PROVENANCE_COLUMNS)} "
                f"FROM memory_provenance WHERE mem_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            for mem_id, *values in rows:
                out[mem_id].append(
                    Provenance(**dict(zip(_PROVENANCE_COLUMNS, values)))
                )
        return out

    def _row_to_item(
        self, row: tuple, provenance: list[Provenance] | None = None
    ) -> MemoryItem:
        record = {
            column: _FROM_DB[column](value) if column in _FROM_DB else value
            for column, value in zip(_COLUMNS, row)
        }
        record["provenance"] = (
            self._load_provenance(record["mem_id"])
            if provenance is None
            else provenance
        )
        return MemoryItem(**record)

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
