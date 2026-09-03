"""Lexical (sparse) retrieval: Okapi BM25 over a SQLite inverted index.

The design allows lexical candidates to come either from bge-m3's learned
lexical weights or from classic BM25 over the chunk text. This module
implements BM25 because it needs no model and no extra dependency: the
inverted index lives in the ``chunk_terms`` table and the document length
in ``chunks.term_count`` (schema v2, see ``db.py``).

Everything here is pure Python + SQL and fully deterministic, so lexical
retrieval is reproducible across machines and runs.
"""

from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Iterable, Sequence

from memory_condense.persistence.db import TURN_SOURCE_ID_SQL, Database
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.pending_ingest_store import PendingIngestStore

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

#: A token is a maximal run of alphanumeric characters. ``[^\W_]`` is
#: "word character but not underscore", so ``snake_case`` splits into
#: ``snake`` + ``case`` and ``bge-m3`` into ``bge`` + ``m3``.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Minimum token length kept. Single characters carry almost no signal and
#: blow up the posting lists, so they are dropped.
MIN_TOKEN_LEN = 2

#: A deliberately small, closed English stopword list. Kept short on purpose:
#: BM25's IDF already discounts common words, and an aggressive list would
#: silently delete meaningful query terms ("no", "not" matter in preferences,
#: so they are *not* listed here).
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "as", "at", "be", "because", "been", "before",
        "being", "below", "between", "both", "but", "by", "can", "did", "do",
        "does", "doing", "during", "each", "for", "from", "further", "had",
        "has", "have", "having", "he", "her", "here", "hers", "him", "his",
        "how", "i", "if", "in", "into", "is", "it", "its", "just", "me",
        "more", "most", "my", "of", "off", "on", "once", "only", "or",
        "other", "our", "ours", "out", "over", "own", "same", "she",
        "should", "so", "some", "such", "than", "that", "the", "their",
        "theirs", "them", "then", "there", "these", "they", "this", "those",
        "through", "to", "too", "under", "until", "up", "us", "very", "was",
        "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "will", "with", "would", "you", "your", "yours",
    }
)

#: Okapi BM25 term-frequency saturation parameter.
BM25_K1 = 1.5
#: Okapi BM25 length-normalisation parameter.
BM25_B = 0.75

#: Shared durability coordinate for every mutation of the chunk retrieval
#: indexes.  ``IndexLifecycleMixin`` observes this value before answering a
#: query so process-local source aggregates cannot outlive another writer.
_INDEX_REVISION_KEY = "chunk_index_revision"


def tokenize(text: str) -> list[str]:
    """Split ``text`` into BM25 terms.

    Rules, in order:

    1. lowercase the whole string;
    2. take every maximal run of alphanumeric characters (underscore and all
       punctuation act as separators);
    3. drop tokens shorter than :data:`MIN_TOKEN_LEN`;
    4. drop members of :data:`STOPWORDS`.

    Order is preserved and duplicates are kept, so the result can be counted
    directly into term frequencies.
    """
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= MIN_TOKEN_LEN and token not in STOPWORDS
    ]


def term_frequencies(text: str) -> dict[str, int]:
    """Term -> raw count for ``text``, using :func:`tokenize`."""
    return dict(Counter(tokenize(text)))


def _batched(items: Sequence[str], size: int = 500) -> Iterable[Sequence[str]]:
    """Yield slices small enough to stay under SQLite's bound-parameter cap."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class LexicalIndex:
    """BM25 index backed by the ``chunk_terms`` table.

    Chunk statistics are read from SQLite. Source-level routing keeps only a
    small derived ``source_id -> term_count`` cache, invalidated by every
    write through this index; no term postings or text are duplicated.
    """

    def __init__(self, db: Database, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self._db = db
        self._pending_ingests = PendingIngestStore(db)
        self._k1 = k1
        self._b = b
        self._source_length_cache: dict[str, int] | None = None

    def invalidate_cache(self) -> None:
        """Forget process-local aggregates after another writer changes chunks."""
        self._source_length_cache = None

    def _bump_index_revision(self) -> None:
        """Publish a direct lexical mutation to other live retrievers."""
        self._db.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, '0')",
            (_INDEX_REVISION_KEY,),
        )
        self._db.execute(
            "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
            "WHERE key = ?",
            (_INDEX_REVISION_KEY,),
        )

    # -- writing ------------------------------------------------------------

    def validate_chunk_identities(
        self,
        chunks: list[Chunk],
        *,
        allow_indexed_rebuild: bool = False,
    ) -> set[str]:
        """Fail closed on a durable ID whose source-owned fields disagree."""
        self._pending_ingests.validate_chunk_membership(
            chunks,
            allow_indexed_rebuild=allow_indexed_rebuild,
        )
        incoming: dict[str, Chunk] = {}
        for chunk in chunks:
            previous = incoming.setdefault(chunk.chunk_id, chunk)
            if previous != chunk:
                raise ValueError("duplicate chunk_id has different content")

        existing_ids: set[str] = set()
        ids = list(incoming)
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = self._db.execute(
                "SELECT chunk_id, turn_id, text, start_char, end_char, token_count "
                f"FROM chunks WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            for row in rows:
                chunk_id = str(row[0])
                existing_ids.add(chunk_id)
                chunk = incoming[chunk_id]
                if (
                    str(row[1]) != chunk.turn_id
                    or str(row[2]) != chunk.text
                    or int(row[3]) != chunk.start_char
                    or int(row[4]) != chunk.end_char
                    or int(row[5]) != chunk.token_count
                ):
                    raise ValueError(
                        "chunk_id already exists with different content"
                    )
        return existing_ids

    def add_chunks(
        self,
        chunks: list[Chunk],
        *,
        commit: bool = True,
        validate_identity: bool = True,
        _trusted_indexed_rebuild: bool = False,
    ) -> None:
        """Index ``chunks`` lexically.

        Writes one ``chunk_terms`` row per distinct term and stores the chunk's
        document length in ``chunks.term_count``. Idempotent: re-adding the same
        chunk replaces its postings rather than duplicating them (the
        ``(term, chunk_id)`` primary key plus ``INSERT OR REPLACE``).

        The chunk row itself is created with ``INSERT OR IGNORE`` if it is not
        there yet, so the lexical index can be populated independently of the
        dense index. Existing rows (with their embedding and hnsw label) are
        never overwritten.
        """
        if not chunks:
            return
        connection = self._db.connection
        try:
            if commit and not connection.in_transaction:
                connection.execute("BEGIN IMMEDIATE")
            if not commit and not connection.in_transaction:
                raise RuntimeError(
                    "commit=False requires an owning SQLite transaction"
                )
            if validate_identity:
                self.validate_chunk_identities(
                    chunks,
                    allow_indexed_rebuild=_trusted_indexed_rebuild,
                )
            else:
                # The caller may have validated global chunk-ID ownership,
                # but manifest membership is a separate topology invariant and
                # must hold under this insertion's owning transaction.
                self._pending_ingests.validate_chunk_membership(
                    chunks,
                    allow_indexed_rebuild=_trusted_indexed_rebuild,
                )
            self._source_length_cache = None

            for chunk in chunks:
                tf = (
                    {t: int(v) for t, v in chunk.lexical_weights.items()}
                    if chunk.lexical_weights
                    else term_frequencies(chunk.text)
                )

                self._db.execute(
                    "INSERT OR IGNORE INTO chunks "
                    "(chunk_id, turn_id, text, start_char, end_char, token_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.turn_id,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.token_count,
                    ),
                )

                # Drop stale postings first so a re-index after an edit cannot
                # leave orphaned terms behind.
                self._db.execute(
                    "DELETE FROM chunk_terms WHERE chunk_id = ?", (chunk.chunk_id,)
                )
                if tf:
                    self._db.executemany(
                        "INSERT OR REPLACE INTO chunk_terms (term, chunk_id, tf) "
                        "VALUES (?, ?, ?)",
                        [(term, chunk.chunk_id, count) for term, count in tf.items()],
                    )
                self._db.execute(
                    "UPDATE chunks SET term_count = ? WHERE chunk_id = ?",
                    (sum(tf.values()), chunk.chunk_id),
                )

            if commit:
                self._bump_index_revision()
                connection.commit()
        except BaseException:
            if commit:
                connection.rollback()
            raise

    def rebuild(self, chunks_iterable: Iterable[Chunk] | None = None) -> int:
        """Drop and rebuild the whole inverted index. Returns the chunk count.

        With ``chunks_iterable`` the given chunks become the index. Without it
        the index is rebuilt from the text already stored in ``chunks`` — the
        transcript remains the single source of truth either way.
        """
        supplied = chunks_iterable is not None
        batch = list(chunks_iterable) if supplied else []
        batch = [
            chunk.model_copy(update={"lexical_weights": None}) for chunk in batch
        ]
        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not supplied:
                rows = self._db.execute(
                    "SELECT chunk_id, turn_id, text, start_char, end_char, "
                    "token_count FROM chunks "
                    "WHERE (embedding IS NOT NULL AND hnsw_label IS NOT NULL) "
                    "OR term_count IS NOT NULL"
                ).fetchall()
                batch = [
                    Chunk(
                        chunk_id=row[0],
                        turn_id=row[1],
                        text=row[2],
                        start_char=row[3],
                        end_char=row[4],
                        token_count=row[5],
                    )
                    for row in rows
                ]
            self.validate_chunk_identities(
                batch,
                allow_indexed_rebuild=not supplied,
            )
            self._source_length_cache = None
            self._db.execute("DELETE FROM chunk_terms")
            self._db.execute("UPDATE chunks SET term_count = NULL")
            # Re-derive term frequencies from text inside the same transaction
            # that retired the old postings.
            self.add_chunks(
                batch,
                commit=False,
                validate_identity=False,
                # The exact batch was checked before term_count was cleared.
                # Both default and supplied rebuilds need this continuation to
                # avoid treating their own in-transaction NULL as retirement.
                _trusted_indexed_rebuild=True,
            )
            self._bump_index_revision()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return len(batch)

    def delete_chunk(self, chunk_id: str, *, commit: bool = True) -> None:
        """Remove a chunk's postings and clear its document length.

        ``commit=False`` lets the owning index lifecycle combine this cleanup
        atomically with dense and association retirement.
        """
        connection = self._db.connection
        try:
            if commit and not connection.in_transaction:
                connection.execute("BEGIN IMMEDIATE")
            if not commit and not connection.in_transaction:
                raise RuntimeError(
                    "commit=False requires an owning SQLite transaction"
                )
            self._source_length_cache = None
            self._db.execute(
                "DELETE FROM chunk_terms WHERE chunk_id = ?", (chunk_id,)
            )
            self._db.execute(
                "UPDATE chunks SET term_count = NULL WHERE chunk_id = ?", (chunk_id,)
            )
            if commit:
                self._bump_index_revision()
                connection.commit()
        except BaseException:
            if commit:
                connection.rollback()
            raise

    # -- reading ------------------------------------------------------------

    def corpus_stats(self) -> tuple[int, float]:
        """``(N, avgdl)`` over lexically indexed chunks only."""
        cur = self._db.execute(
            "SELECT COUNT(*), AVG(term_count) FROM chunks WHERE term_count IS NOT NULL"
        )
        count, avg = cur.fetchone()
        n = int(count or 0)
        avgdl = float(avg) if avg else 0.0
        return n, avgdl

    def stats(self) -> dict[str, float]:
        """Index size summary, handy for debugging and eval reports."""
        n, avgdl = self.corpus_stats()
        cur = self._db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT term) FROM chunk_terms"
        )
        postings, distinct_terms = cur.fetchone()
        return {
            "chunks": float(n),
            "avg_term_count": avgdl,
            "postings": float(postings or 0),
            "distinct_terms": float(distinct_terms or 0),
        }

    def document_frequencies(self, terms: Sequence[str]) -> dict[str, int]:
        """How many indexed chunks contain each term (one query per batch)."""
        df: dict[str, int] = {}
        for group in _batched(list(terms)):
            placeholders = ",".join("?" * len(group))
            cur = self._db.execute(
                f"SELECT term, COUNT(*) FROM chunk_terms WHERE term IN ({placeholders}) "
                "GROUP BY term",
                tuple(group),
            )
            for term, count in cur.fetchall():
                df[term] = int(count)
        return df

    def search(self, query: str, limit: int = 100) -> list[tuple[str, float]]:
        """Top ``limit`` ``(chunk_id, bm25_score)`` pairs, best first.

        Standard Okapi BM25::

            idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
            score(d) = sum_t idf(t) * tf(t,d) * (k1 + 1)
                       / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

        with ``k1=1.5``, ``b=0.75``. Only chunks containing at least one query
        term are considered — everything else scores exactly 0 and is omitted.
        Scores are raw BM25; normalise them before blending with dense scores.
        Ties are broken by ``chunk_id`` so the ordering is deterministic.
        """
        if limit <= 0:
            return []

        terms = sorted(set(tokenize(query)))
        if not terms:
            return []

        n_docs, avgdl = self.corpus_stats()
        if n_docs == 0 or avgdl <= 0:
            return []

        df = self.document_frequencies(terms)
        present = [t for t in terms if df.get(t)]
        if not present:
            return []

        idf = {
            t: log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5)) for t in present
        }

        scores: dict[str, float] = {}
        for group in _batched(present):
            placeholders = ",".join("?" * len(group))
            cur = self._db.execute(
                "SELECT ct.term, ct.chunk_id, ct.tf, c.term_count "
                "FROM chunk_terms ct JOIN chunks c ON c.chunk_id = ct.chunk_id "
                f"WHERE ct.term IN ({placeholders}) AND c.term_count IS NOT NULL",
                tuple(group),
            )
            for term, chunk_id, tf, doc_len in cur.fetchall():
                tf = float(tf)
                doc_len = float(doc_len or 0)
                denom = tf + self._k1 * (
                    1.0 - self._b + self._b * (doc_len / avgdl)
                )
                if denom <= 0:
                    continue
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf[term] * (
                    tf * (self._k1 + 1.0) / denom
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]

    def search_sources(
        self,
        query: str,
        source_ids: Sequence[str],
        *,
        limit_per_source: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Run globally calibrated BM25 inside selected provenance sources.

        Unlike filtering :meth:`search`, this examines every lexical posting
        in the activated sources. BM25 corpus statistics remain global so
        scores from different partitions stay comparable; independently
        normalizing every source would manufacture one top-scoring chunk per
        partition and let weak sources crowd out real temporal evidence.
        Returned rows are bounded per source and deterministic.
        """

        selected = list(dict.fromkeys(str(value) for value in source_ids if str(value)))
        results: dict[str, list[tuple[str, float]]] = {
            source_id: [] for source_id in selected
        }
        if not selected or limit_per_source <= 0:
            return results

        terms = sorted(set(tokenize(query)))
        if not terms:
            return results

        source_placeholders = ",".join("?" for _ in selected)
        source_expr = TURN_SOURCE_ID_SQL
        n_docs, avgdl = self.corpus_stats()
        if n_docs <= 0 or avgdl <= 0.0:
            return results

        term_placeholders = ",".join("?" for _ in terms)
        params = (*terms, *selected)
        document_frequency = self.document_frequencies(terms)

        # Each buffer is trimmed whenever it reaches twice its final bound.
        # This keeps lexical workspace independent of source length while
        # preserving exact deterministic top-k results.
        buffers: dict[str, list[tuple[str, float]]] = {
            source_id: [] for source_id in selected
        }
        def admit(source_key: str, chunk_id: str, score: float) -> None:
            if score <= 0.0:
                return
            buffer = buffers[source_key]
            buffer.append((chunk_id, score))
            if len(buffer) >= limit_per_source * 2:
                buffer.sort(key=lambda item: (-item[1], item[0]))
                del buffer[limit_per_source:]

        postings = self._db.execute(
            "SELECT " + source_expr + ", ct.chunk_id, c.term_count, "
            "ct.term, ct.tf FROM chunk_terms AS ct "
            "JOIN chunks AS c ON c.chunk_id = ct.chunk_id "
            "JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE ct.term IN ({term_placeholders}) "
            f"AND {source_expr} IN ({source_placeholders}) "
            "AND c.term_count IS NOT NULL "
            "ORDER BY " + source_expr + ", ct.chunk_id, ct.term",
            params,
        )
        current_key: tuple[str, str] | None = None
        current_score = 0.0
        for source_id, chunk_id, doc_len, term, tf in postings:
            source_key = str(source_id)
            chunk_key = str(chunk_id)
            key = (source_key, chunk_key)
            if current_key is not None and key != current_key:
                admit(current_key[0], current_key[1], current_score)
                current_score = 0.0
            current_key = key
            df = document_frequency.get(str(term), 0)
            frequency = float(tf)
            if df <= 0 or frequency <= 0.0:
                continue
            idf = log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = frequency + self._k1 * (
                1.0 - self._b + self._b * (float(doc_len or 0) / avgdl)
            )
            if denom > 0.0:
                current_score += idf * (
                    frequency * (self._k1 + 1.0) / denom
                )
        if current_key is not None:
            admit(current_key[0], current_key[1], current_score)

        for source_id, buffer in buffers.items():
            buffer.sort(key=lambda item: (-item[1], item[0]))
            results[source_id] = buffer[:limit_per_source]
        return results

    def search_source_tfisf(
        self,
        query: str,
        *,
        limit: int = 8,
    ) -> list[tuple[str, float]]:
        """Rank provenance sources with BM25-style TF–ISF.

        Chunk BM25 answers "which excerpt uses these terms?" This companion
        view answers "which conversation or document owns these terms?" by
        treating every durable source as one aggregate lexical document.
        Statistics are derived from the live index on each call, so appended
        turns immediately affect source frequency without another stored
        summary or rebuild.
        """

        if limit <= 0:
            return []
        terms = sorted(set(tokenize(query)))
        if not terms:
            return []

        source_expr = TURN_SOURCE_ID_SQL
        if self._source_length_cache is None:
            source_rows = self._db.execute(
                "SELECT " + source_expr + ", SUM(c.term_count) "
                "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
                "WHERE c.term_count IS NOT NULL "
                "GROUP BY " + source_expr
            ).fetchall()
            self._source_length_cache = {
                str(source_id): int(length or 0)
                for source_id, length in source_rows
            }
        source_lengths = self._source_length_cache
        if not source_lengths:
            return []
        n_sources = len(source_lengths)
        avg_length = sum(source_lengths.values()) / n_sources
        if avg_length <= 0.0:
            return []

        term_placeholders = ",".join("?" for _ in terms)
        rows = self._db.execute(
            "SELECT " + source_expr + ", ct.term, SUM(ct.tf) "
            "FROM chunk_terms AS ct "
            "JOIN chunks AS c ON c.chunk_id = ct.chunk_id "
            "JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE ct.term IN ({term_placeholders}) "
            "GROUP BY " + source_expr + ", ct.term "
            "ORDER BY " + source_expr + ", ct.term",
            tuple(terms),
        ).fetchall()
        if not rows:
            return []

        source_frequency = Counter(str(term) for _source, term, _tf in rows)
        scores: dict[str, float] = {}
        for source_id, term, raw_tf in rows:
            source_key = str(source_id)
            term_key = str(term)
            df = source_frequency[term_key]
            tf = float(raw_tf)
            source_length = float(source_lengths[source_key])
            isf = log(
                1.0 + (n_sources - df + 0.5) / (df + 0.5)
            )
            denominator = tf + self._k1 * (
                1.0 - self._b + self._b * source_length / avg_length
            )
            if denominator > 0.0:
                scores[source_key] = scores.get(source_key, 0.0) + isf * (
                    tf * (self._k1 + 1.0) / denominator
                )

        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
