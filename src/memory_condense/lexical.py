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

from memory_condense.db import Database
from memory_condense.schemas import Chunk

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

    The index owns no state of its own: every read recomputes N, avgdl and
    the document frequencies from SQLite, so two processes pointed at the
    same database always agree.
    """

    def __init__(self, db: Database, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self._db = db
        self._k1 = k1
        self._b = b

    # -- writing ------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> None:
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

        self._db.commit()

    def rebuild(self, chunks_iterable: Iterable[Chunk] | None = None) -> int:
        """Drop and rebuild the whole inverted index. Returns the chunk count.

        With ``chunks_iterable`` the given chunks become the index. Without it
        the index is rebuilt from the text already stored in ``chunks`` — the
        transcript remains the single source of truth either way.
        """
        self._db.execute("DELETE FROM chunk_terms")
        self._db.execute("UPDATE chunks SET term_count = NULL")
        self._db.commit()

        if chunks_iterable is None:
            cur = self._db.execute(
                "SELECT chunk_id, turn_id, text, start_char, end_char, token_count "
                "FROM chunks"
            )
            chunks_iterable = [
                Chunk(
                    chunk_id=row[0],
                    turn_id=row[1],
                    text=row[2],
                    start_char=row[3],
                    end_char=row[4],
                    token_count=row[5],
                )
                for row in cur.fetchall()
            ]

        batch = list(chunks_iterable)
        # Re-derive term frequencies from text, ignoring any stored weights.
        self.add_chunks(
            [c.model_copy(update={"lexical_weights": None}) for c in batch]
        )
        return len(batch)

    def delete_chunk(self, chunk_id: str) -> None:
        """Remove a chunk's postings and clear its document length."""
        self._db.execute("DELETE FROM chunk_terms WHERE chunk_id = ?", (chunk_id,))
        self._db.execute(
            "UPDATE chunks SET term_count = NULL WHERE chunk_id = ?", (chunk_id,)
        )
        self._db.commit()

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
