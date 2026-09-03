"""Stateful ANN/lexical index allocation, synchronization, and lifecycle."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable

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
_INDEX_REVISION_KEY = "chunk_index_revision"


def _save_hnsw_index(index: hnswlib.Index, path: Path) -> None:
    """Small publication seam: hnswlib writes only a private image path."""

    index.save_index(str(path))


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
        self._observed_index_revision = 0
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
            try:
                self._index.load_index(str(self._index_path))
            except (OSError, RuntimeError, ValueError):
                # The ANN image is disposable.  A torn/corrupt publication
                # must not prevent startup when SQLite still holds every
                # authoritative embedding and durable label.
                self.rebuild_index()
            else:
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

    def _discard_local_index(self, *, max_elements: int | None = None) -> None:
        """Replace a possibly partial ANN graph with an empty recoverable one."""
        self._index = hnswlib.Index(space="cosine", dim=self._dim)
        self._index.init_index(
            max_elements=max_elements or self._max_elements,
            ef_construction=self._ef_construction,
            M=self._M,
        )
        self._label_to_chunk_id.clear()
        self._chunk_id_to_label.clear()
        self._deleted_labels.clear()
        # The next query reconstructs the disposable graph from authoritative
        # SQLite, even when the durable revision did not change.
        self._observed_index_revision = -1

    def _load_label_mapping(self) -> None:
        """Reconcile a loaded ANN image with authoritative durable labels."""
        cur = self._db.execute(
            "SELECT chunk_id, hnsw_label FROM chunks "
            "WHERE hnsw_label IS NOT NULL AND embedding IS NOT NULL"
        )
        durable = {str(chunk_id): int(label) for chunk_id, label in cur.fetchall()}
        present = {int(label) for label in self._index.get_ids_list()}
        for chunk_id, label in durable.items():
            if label in present:
                self._register(chunk_id, label)
        orphan_labels = present - set(durable.values())
        if orphan_labels:
            # A persisted deleted bit and a failed native retirement are not
            # distinguishable through hnswlib's exception surface. Do not
            # guess: discard any image containing labels SQLite no longer owns.
            self._discard_local_index(
                max_elements=max(self._index.get_max_elements(), self._max_elements)
            )
            return
        # A stale/missing index image must force one exact DB reconciliation
        # even when the durable revision itself has not changed since startup.
        self._observed_index_revision = (
            self._read_index_revision()
            if len(self._chunk_id_to_label) == len(durable)
            else -1
        )

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
        try:
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) "
                "SELECT ?, CAST(COALESCE(MAX(hnsw_label), -1) + 1 AS TEXT) "
                "FROM chunks",
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
                    "SELECT CAST(value AS INTEGER) FROM meta WHERE key = ?",
                    (_LABEL_KEY,),
                ).fetchone()[0]
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

        start = end - count
        return list(range(start, end))

    def _register(self, chunk_id: str, label: int) -> None:
        self._label_to_chunk_id[label] = chunk_id
        self._chunk_id_to_label[chunk_id] = label

    def _read_index_revision(self) -> int:
        row = self._db.execute(
            "SELECT value FROM meta WHERE key = ?", (_INDEX_REVISION_KEY,)
        ).fetchone()
        return 0 if row is None else int(row[0])

    def _bump_index_revision(self) -> int:
        """Advance the durable cross-process index visibility coordinate."""
        self._db.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, '0')",
            (_INDEX_REVISION_KEY,),
        )
        self._db.execute(
            "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
            "WHERE key = ?",
            (_INDEX_REVISION_KEY,),
        )
        return self._read_index_revision()

    def _retire_local_label(self, chunk_id: str, label: int) -> bool:
        """Drop one local mapping; report whether ANN retirement was proven."""
        self._chunk_id_to_label.pop(chunk_id, None)
        self._label_to_chunk_id.pop(label, None)
        try:
            self._index.mark_deleted(label)
            self._deleted_labels.add(label)
        except (RuntimeError, ValueError):
            return False
        except BaseException:
            # A native call may mutate its deleted bit before propagating an
            # interruption.  With the Python mapping already removed there is
            # no safe way to distinguish accepted from rejected retirement.
            # Drop the entire disposable graph before preserving the signal.
            self._discard_local_index()
            raise
        return True

    def _sync_from_db(self) -> int:
        """Adopt chunks another process indexed since we last looked.

        Each process keeps its own in-memory hnswlib graph, so without this a
        second session's writes stay invisible until restart. SQLite is the
        source of truth (MC-STD-DATA clause 1), so we reconcile against it.
        Returns the number of vectors adopted.
        """
        if self._index is None:
            return 0

        revision = self._read_index_revision()
        if revision == self._observed_index_revision:
            return 0

        self._clear_span_cache()
        self._source_hierarchy.invalidate()
        self._lexical.invalidate_cache()

        rows = self._db.execute(
            "SELECT chunk_id, hnsw_label, embedding FROM chunks "
            "WHERE hnsw_label IS NOT NULL AND embedding IS NOT NULL"
        ).fetchall()

        durable = {
            str(chunk_id): (int(label), blob)
            for chunk_id, label, blob in rows
        }
        for chunk_id, label in list(self._chunk_id_to_label.items()):
            current = durable.get(chunk_id)
            if current is None or current[0] != label:
                if not self._retire_local_label(chunk_id, label):
                    self._discard_local_index()
                    break

        labels: list[int] = []
        vectors: list[np.ndarray] = []
        chunk_ids: list[str] = []
        for chunk_id, (label, blob) in durable.items():
            if chunk_id in self._chunk_id_to_label:
                continue
            labels.append(label)
            vectors.append(np.frombuffer(blob, dtype=np.float32))
            chunk_ids.append(chunk_id)

        if not labels:
            self._observed_index_revision = revision
            return 0

        try:
            needed = self._index.get_current_count() + len(labels)
            if needed > self._index.get_max_elements():
                self._index.resize_index(max(needed * 2, self._max_elements))
            self._index.add_items(
                np.stack(vectors), np.array(labels, dtype=np.int64)
            )
            for chunk_id, label in zip(chunk_ids, labels):
                self._register(chunk_id, label)
            self._observed_index_revision = revision
        except BaseException:
            # Both resize and bulk add are native, non-transactional
            # mutations.  Either may have accepted a prefix before raising;
            # retaining that graph would leave labels with no Python owner.
            self._discard_local_index()
            raise
        return len(labels)

    def _lexically_incomplete_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return dense-live chunks left without a completed BM25 document."""
        incomplete: set[str] = set()
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start : start + 500]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT chunk_id FROM chunks "
                f"WHERE chunk_id IN ({placeholders}) "
                "AND embedding IS NOT NULL AND hnsw_label IS NOT NULL "
                "AND term_count IS NULL",
                tuple(batch),
            ).fetchall()
            incomplete.update(str(row[0]) for row in rows)
        return incomplete

    def add_chunks(
        self,
        chunks: list[Chunk],
        *,
        finalize: Callable[[], None] | None = None,
    ) -> None:
        """Add embedded chunks to the ANN index and persist to SQLite.

        Chunks must have non-None embedding fields. Idempotent:
        chunks already in the index are skipped.

        The same chunks are also written to the BM25 inverted index, and
        ``chunks.lexical_weights`` is filled with the chunk's term-frequency
        map (computed from the text when the Chunk does not carry one).
        """
        if not chunks:
            return

        unique_chunks: dict[str, Chunk] = {}
        for chunk in chunks:
            previous = unique_chunks.setdefault(chunk.chunk_id, chunk)
            if previous != chunk:
                raise ValueError("duplicate chunk_id has different content")
        chunks = list(unique_chunks.values())
        preexisting_ids = self._lexical.validate_chunk_identities(chunks)

        # Reconcile another writer before deciding which IDs are new locally.
        # A later interleaving is detected by the revision gap at commit.
        self._sync_from_db()
        base_revision = self._observed_index_revision

        new_chunks = [
            c for c in chunks
            if c.chunk_id not in self._chunk_id_to_label and c.embedding is not None
        ]
        mapped = [c for c in chunks if c.chunk_id in self._chunk_id_to_label]
        repair_ids = self._lexically_incomplete_chunk_ids(
            [chunk.chunk_id for chunk in mapped]
        )
        repair_chunks = [c for c in mapped if c.chunk_id in repair_ids]

        if not new_chunks and not repair_chunks and finalize is None:
            return

        if not new_chunks and not repair_chunks:
            try:
                self._db.connection.execute("BEGIN IMMEDIATE")
                self._lexical.validate_chunk_identities(chunks)
                finalize()
                self._db.commit()
            except BaseException:
                self._db.connection.rollback()
                raise
            return

        # Resize index if needed
        current_count = self._index.get_current_count()
        needed = current_count + len(new_chunks)
        if needed > self._index.get_max_elements():
            try:
                self._index.resize_index(max(needed * 2, self._max_elements))
            except BaseException:
                # Capacity growth is a native mutation too.  Its exception
                # contract does not prove that the old graph stayed intact.
                self._discard_local_index()
                raise

        allocated = self._allocate_labels(len(new_chunks))
        labels = list(allocated)
        vectors = [
            np.array(chunk.embedding, dtype=np.float32) for chunk in new_chunks
        ]
        indexed: list[Chunk] = []
        for chunk in [*new_chunks, *repair_chunks]:
            weights: dict = (
                dict(chunk.lexical_weights)
                if chunk.lexical_weights
                else lexical.term_frequencies(chunk.text)
            )
            indexed.append(chunk.model_copy(update={"lexical_weights": weights}))

        ann_attempted = False
        try:
            # The fast validation above improves error latency, but only this
            # recheck owns the SQLite write lock. It closes the cross-process
            # window in which a conflicting durable ID could otherwise arrive
            # before the unconditional dense/lexical upsert.
            self._db.connection.execute("BEGIN IMMEDIATE")
            preexisting_ids.update(
                self._lexical.validate_chunk_identities(chunks)
            )
            for chunk, label, indexed_chunk in zip(new_chunks, labels, indexed):
                # The row may already exist from a lexical-only publication;
                # fill its dense columns without replacing source ownership.
                embedding_blob = np.array(
                    chunk.embedding, dtype=np.float32
                ).tobytes()
                lexical_json = (
                    json.dumps(indexed_chunk.lexical_weights)
                    if indexed_chunk.lexical_weights
                    else None
                )
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
            self._lexical.add_chunks(
                indexed,
                commit=False,
                validate_identity=False,
            )
            if labels:
                data = np.stack(vectors)
                ann_attempted = True
                self._index.add_items(data, np.array(labels, dtype=np.int64))
            revision = self._bump_index_revision()
            if finalize is not None:
                finalize()
            self._db.commit()
        except BaseException:
            self._db.connection.rollback()
            if ann_attempted:
                # hnswlib can accept a prefix before raising, and deletion of
                # that prefix is not itself transactional. Discarding the
                # complete disposable image prevents unmapped labels from
                # consuming top-k slots until the next durable reconciliation.
                self._discard_local_index()
            raise

        for chunk, label in zip(new_chunks, labels):
            self._register(chunk.chunk_id, label)
        if revision == base_revision + 1:
            self._observed_index_revision = revision
        if any(chunk.chunk_id in preexisting_ids for chunk in new_chunks):
            # A lexical-only row can predate the span cache high-water mark.
            # Promoting it to dense state is not an append, so incremental
            # tail loading cannot see it and the complete cache must retire.
            self._clear_span_cache()
        # Ordinary appends remain incremental: `_span_vectors` consumes rows
        # after its per-level high-water mark and changes only the open tail.
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
        self._sync_from_db()
        base_revision = self._observed_index_revision
        label = self._chunk_id_to_label.get(chunk_id)
        artifacts_removed = 0
        try:
            cur = self._db.execute(
                "UPDATE chunks SET embedding = NULL, hnsw_label = NULL "
                "WHERE chunk_id = ?",
                (chunk_id,),
            )
            touched = cur.rowcount > 0
            self._lexical.delete_chunk(chunk_id, commit=False)
            if self._associations is not None:
                artifacts_removed = self._associations.remove_chunk_artifacts(
                    chunk_id, commit=False
                )
            revision = self._bump_index_revision()
            self._db.commit()
        except BaseException:
            self._db.connection.rollback()
            raise

        # SQLite is authoritative. Reconcile disposable in-memory state only
        # after the complete durable retirement commits.
        retirement_proven = (
            label is None or self._retire_local_label(chunk_id, label)
        )
        if not retirement_proven:
            self._discard_local_index()
        if retirement_proven and revision == base_revision + 1:
            self._observed_index_revision = revision
        self._clear_span_cache()
        self._source_hierarchy.invalidate()
        return touched or label is not None or artifacts_removed > 0


    def rebuild_index(self) -> None:
        """Rebuild the hnswlib index from all embeddings in SQLite."""
        self._clear_span_cache()
        # Bind the local graph to a revision observed *before* its row
        # snapshot. A concurrent writer may commit after either read; keeping
        # this older coordinate guarantees the next query reconciles instead
        # of acknowledging unseen rows or retirements as already represented.
        snapshot_revision = self._read_index_revision()
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
        # The fresh graph is not a faithful image until every durable vector
        # has been accepted.  Keep synchronization armed throughout rebuild
        # so a failed stack/add cannot publish empty local state as current.
        self._observed_index_revision = -1

        if not rows:
            self._observed_index_revision = snapshot_revision
            return

        # Decode and validate every durable vector before allocating or
        # publishing any labels. A corrupt row must not leave an implicit
        # transaction (or a labelled-but-unusable chunk) behind.
        vectors: list[np.ndarray] = []
        for chunk_id, emb_blob, _hnsw_label in rows:
            vector = np.frombuffer(emb_blob, dtype=np.float32)
            if vector.shape != (self._dim,):
                raise ValueError(
                    f"stored chunk embedding {chunk_id!r} has shape "
                    f"{vector.shape}, expected ({self._dim},)"
                )
            vectors.append(vector)

        # Rows that never got a label (or lost it) need fresh ones, allocated
        # through the database like any other write. Burning a counter value
        # after a later failure is harmless; publishing a partial mapping is
        # not.
        unlabelled = [str(row[0]) for row in rows if row[2] is None]
        fresh = iter(self._allocate_labels(len(unlabelled)))
        chunk_labels: list[tuple[str, int]] = []
        labels: list[int] = []
        for chunk_id, _emb_blob, hnsw_label in rows:
            label = int(hnsw_label) if hnsw_label is not None else next(fresh)
            chunk_labels.append((str(chunk_id), label))
            labels.append(label)

        try:
            data = np.stack(vectors)
            self._index.add_items(data, np.array(labels, dtype=np.int64))

            if unlabelled:
                self._db.connection.execute("BEGIN IMMEDIATE")
                for chunk_id, label in chunk_labels:
                    if chunk_id not in unlabelled:
                        continue
                    updated = self._db.execute(
                        "UPDATE chunks SET hnsw_label = ? "
                        "WHERE chunk_id = ? AND hnsw_label IS NULL "
                        "AND embedding IS NOT NULL",
                        (label, chunk_id),
                    )
                    if updated.rowcount != 1:
                        raise RuntimeError(
                            "chunk changed while its index was being rebuilt"
                        )
                self._bump_index_revision()
                self._db.commit()
        except BaseException:
            self._db.connection.rollback()
            # ``add_items`` is not transactional and may have accepted only a
            # prefix.  Discard the whole disposable graph; the next query can
            # now rebuild it exactly from authoritative SQLite via revision
            # reconciliation.
            self._discard_local_index(max_elements=max_el)
            raise
        for chunk_id, label in chunk_labels:
            self._register(chunk_id, label)
        # This coordinate describes the earliest state the fetched rows can
        # represent. It intentionally remains behind our own unlabelled-row
        # update too; one cheap reconciliation is safer than skipping a
        # concurrent add/delete that landed during rebuild.
        self._observed_index_revision = snapshot_revision

    def save(self) -> None:
        """Persist the hnswlib index to disk."""
        if self._db.read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        if self._index_path and self._index is not None:
            # A stale live session must never overwrite a newer shared graph.
            self._sync_from_db()
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._index_path.name}.",
                suffix=".tmp",
                dir=self._index_path.parent,
            )
            temporary_path = Path(temporary_name)
            try:
                os.close(descriptor)
                descriptor = -1
                _save_hnsw_index(self._index, temporary_path)
                # hnswlib closes its stream before returning. Flush the image
                # itself before atomically publishing the directory entry.
                with temporary_path.open("rb+") as image:
                    os.fsync(image.fileno())
                os.replace(temporary_path, self._index_path)
            finally:
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    # Preserve the save/replace error. A uniquely named
                    # private file is never treated as an authoritative image.
                    pass


__all__ = ["IndexLifecycleMixin"]
