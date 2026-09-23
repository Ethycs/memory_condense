"""Immutable in-memory Okapi BM25 over a sealed lexical index.

``LexicalIndex`` is the durable authority and remains the implementation used
for writes.  :class:`ResidentBM25Index` compiles that authority once, while it
is open read-only, into compact NumPy posting arrays.  Searches thereafter do
no SQLite work and retain the durable index's scoring and deterministic
``chunk_id`` tie break.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from types import MappingProxyType

import numpy as np

from memory_condense.persistence.db import Database
from memory_condense.search.indexes.lexical import (
    BM25_B,
    BM25_K1,
    LexicalIndex,
    tokenize,
)


@dataclass(frozen=True, slots=True)
class _PostingList:
    """One term's sealed active postings plus its durable document frequency."""

    rows: np.ndarray
    term_frequencies: np.ndarray
    document_frequency: int


class ResidentBM25Index:
    """A query-independent, immutable snapshot of :class:`LexicalIndex`.

    ``source`` may be either a read-only :class:`Database` (using the standard
    BM25 constants) or a :class:`LexicalIndex` backed by one (inheriting its
    configured ``k1`` and ``b``).  Requiring a read-only source prevents the
    compile from racing a writer and makes the resident object a precise
    snapshot.  The source may be closed immediately after construction.

    Numeric storage is four bytes per document length and eight bytes per
    posting (a ``uint32`` row plus ``uint32`` term frequency).  Chunk/term
    strings and Python container overhead are intentionally excluded from
    :attr:`resident_bytes`, matching the numeric-payload convention used by
    the other hot address indexes.
    """

    __slots__ = (
        "_avgdl",
        "_b",
        "_chunk_ids",
        "_document_lengths",
        "_k1",
        "_locked",
        "_posting_count",
        "_postings",
        "_resident_bytes",
    )

    def __init__(self, source: Database | LexicalIndex) -> None:
        object.__setattr__(self, "_locked", False)
        if isinstance(source, LexicalIndex):
            database = source._db
            k1, b = float(source._k1), float(source._b)
        elif isinstance(source, Database):
            database = source
            k1, b = BM25_K1, BM25_B
        else:
            raise TypeError("source must be a Database or LexicalIndex")
        if not database.read_only:
            raise ValueError(
                "resident BM25 compilation requires a read-only database"
            )

        document_rows = database.execute(
            "SELECT chunk_id, term_count FROM chunks "
            "WHERE term_count IS NOT NULL ORDER BY chunk_id"
        ).fetchall()
        if len(document_rows) > int(np.iinfo(np.uint32).max):
            raise OverflowError("resident BM25 supports at most 2^32 - 1 documents")

        chunk_ids = tuple(str(row[0]) for row in document_rows)
        if any(not value for value in chunk_ids) or any(
            left >= right for left, right in zip(chunk_ids, chunk_ids[1:])
        ):
            raise ValueError("indexed chunk IDs must be non-empty and unique")
        raw_lengths = [int(row[1]) for row in document_rows]
        if any(
            value < 0 or value > int(np.iinfo(np.uint32).max)
            for value in raw_lengths
        ):
            raise ValueError("document lengths must fit in uint32")
        document_lengths = np.asarray(raw_lengths, dtype=np.uint32)
        document_lengths.setflags(write=False)
        avgdl = float(sum(raw_lengths) / len(raw_lengths)) if raw_lengths else 0.0
        row_by_chunk_id = {
            chunk_id: row for row, chunk_id in enumerate(chunk_ids)
        }
        postings: dict[str, _PostingList] = {}
        current_term: str | None = None
        current_rows: list[int] = []
        current_frequencies: list[int] = []
        current_document_frequency = 0
        posting_count = 0

        def seal_current() -> None:
            if current_term is None:
                return
            rows = np.asarray(current_rows, dtype=np.uint32)
            frequencies = np.asarray(current_frequencies, dtype=np.uint32)
            rows.setflags(write=False)
            frequencies.setflags(write=False)
            postings[current_term] = _PostingList(
                rows=rows,
                term_frequencies=frequencies,
                document_frequency=current_document_frequency,
            )

        cursor = database.execute(
            "SELECT ct.term, ct.chunk_id, ct.tf, c.term_count "
            "FROM chunk_terms AS ct JOIN chunks AS c ON c.chunk_id = ct.chunk_id "
            "ORDER BY ct.term, ct.chunk_id"
        )
        for raw_term, raw_chunk_id, raw_tf, term_count in cursor:
            term = str(raw_term)
            if current_term != term:
                seal_current()
                current_term = term
                current_rows = []
                current_frequencies = []
                current_document_frequency = 0
            posting_count += 1
            current_document_frequency += 1
            if term_count is None:
                # LexicalIndex.document_frequencies deliberately counts stale
                # postings, while search excludes their retired documents.
                continue
            tf = int(raw_tf)
            if tf < 0 or tf > int(np.iinfo(np.uint32).max):
                raise ValueError("term frequencies must fit in uint32")
            chunk_id = str(raw_chunk_id)
            try:
                row = row_by_chunk_id[chunk_id]
            except KeyError as exc:  # Defensive consistency check for a bad snapshot.
                raise ValueError("active posting has no indexed document") from exc
            current_rows.append(row)
            current_frequencies.append(tf)
        seal_current()

        resident_bytes = int(document_lengths.nbytes) + sum(
            int(posting.rows.nbytes + posting.term_frequencies.nbytes)
            for posting in postings.values()
        )
        object.__setattr__(self, "_chunk_ids", chunk_ids)
        object.__setattr__(self, "_document_lengths", document_lengths)
        object.__setattr__(self, "_postings", MappingProxyType(postings))
        object.__setattr__(self, "_posting_count", posting_count)
        object.__setattr__(self, "_avgdl", avgdl)
        object.__setattr__(self, "_k1", k1)
        object.__setattr__(self, "_b", b)
        object.__setattr__(self, "_resident_bytes", resident_bytes)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError(f"{type(self).__name__} is immutable")
        object.__setattr__(self, name, value)

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids)

    @property
    def resident_bytes(self) -> int:
        """Bytes in document-length and posting NumPy payloads."""

        return self._resident_bytes

    def stats(self) -> dict[str, float]:
        """Return the same corpus counters as :meth:`LexicalIndex.stats`."""

        return {
            "chunks": float(self.chunk_count),
            "avg_term_count": self._avgdl,
            "postings": float(self._posting_count),
            "distinct_terms": float(len(self._postings)),
        }

    def search(self, query: str, limit: int = 100) -> list[tuple[str, float]]:
        """Return exact BM25 hits without consulting SQLite.

        Query tokenization, corpus-wide document frequencies, raw Okapi BM25
        scores, omission of zero-overlap documents, and the ``chunk_id`` tie
        break match :meth:`LexicalIndex.search`.
        """

        if limit <= 0:
            return []
        scores = np.zeros(self.chunk_count, dtype=np.float64)
        matched = np.zeros(self.chunk_count, dtype=np.bool_)
        terms = sorted(set(tokenize(query)))
        if not terms or self.chunk_count == 0 or self._avgdl <= 0.0:
            return []

        for term in (term for term in terms if term in self._postings):
            posting = self._postings[term]
            rows = posting.rows
            if rows.size == 0:
                continue
            term_frequencies = posting.term_frequencies
            denominator = term_frequencies + self._k1 * (
                1.0
                - self._b
                + self._b * (self._document_lengths[rows] / self._avgdl)
            )
            valid = denominator > 0.0
            if not np.any(valid):
                continue
            active_rows = rows[valid]
            active_tf = term_frequencies[valid]
            idf = log(
                1.0
                + (self.chunk_count - posting.document_frequency + 0.5)
                / (posting.document_frequency + 0.5)
            )
            scores[active_rows] += idf * (
                active_tf * (self._k1 + 1.0) / denominator[valid]
            )
            matched[active_rows] = True
        candidate_rows = np.flatnonzero(matched)
        if candidate_rows.size == 0:
            return []
        # IDs were compiled ascending. Stable descending score order therefore
        # implements LexicalIndex's secondary ascending chunk-ID key.
        order = np.argsort(-scores[candidate_rows], kind="stable")
        ranked_rows = candidate_rows[order]
        return [
            (self._chunk_ids[int(row)], float(scores[int(row)]))
            for row in ranked_rows[:limit]
        ]


__all__ = ["ResidentBM25Index"]
