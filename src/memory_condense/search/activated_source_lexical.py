"""Provider-free lexical search inside already activated memory sources.

The sealed hot-retrieval parent exposes a corpus-wide
:class:`~memory_condense.search.hot_lexical.ResidentBM25Index`.  This module
adds the complementary local read without changing that versioned parent:
score the query once against the resident corpus, then partition the exact
ranked hits by an ingest-time ``chunk_id -> source_id`` map.

Because the parent BM25 scores use global document frequencies and lengths,
filtering its complete ranked result is exactly equivalent to the durable
source-local search.  No SQLite query, embedding, model call, or retained
query-token state is required once both resident structures are warm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from memory_condense.search.hot_lexical import ResidentBM25Index


class ActivatedSourceLexicalIndex:
    """Immutable source partition over one resident BM25 snapshot."""

    __slots__ = (
        "_chunk_source_ids",
        "_resident",
        "_source_ids",
    )

    def __init__(
        self,
        resident: ResidentBM25Index,
        chunk_source_ids: Mapping[str, str],
    ) -> None:
        if not isinstance(resident, ResidentBM25Index):
            raise TypeError("resident must be a ResidentBM25Index")
        normalized: dict[str, str] = {}
        for raw_chunk_id, raw_source_id in chunk_source_ids.items():
            chunk_id = str(raw_chunk_id)
            source_id = str(raw_source_id)
            if not chunk_id or not source_id:
                raise ValueError("chunk/source IDs must be non-empty")
            normalized[chunk_id] = source_id
        if len(normalized) != resident.chunk_count:
            raise ValueError(
                "chunk/source map must cover the complete resident snapshot"
            )
        self._resident = resident
        self._chunk_source_ids = MappingProxyType(normalized)
        self._source_ids = frozenset(normalized.values())

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_source_ids)

    @property
    def source_count(self) -> int:
        return len(self._source_ids)

    def search_sources(
        self,
        query: str,
        source_ids: Sequence[str],
        *,
        limit_per_source: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Return exact BM25 ranks independently inside activated sources.

        Source order and duplicate handling match the durable lexical API.
        The complete resident result is requested deliberately: filtering a
        bounded global top-k would orphan locally relevant rows.
        """

        selected = list(
            dict.fromkeys(str(value) for value in source_ids if str(value))
        )
        results: dict[str, list[tuple[str, float]]] = {
            source_id: [] for source_id in selected
        }
        if not selected or limit_per_source <= 0:
            return results

        eligible = set(selected) & self._source_ids
        if not eligible:
            return results
        completed: set[str] = set()
        for chunk_id, score in self._resident.search(
            query,
            limit=self._resident.chunk_count,
        ):
            source_id = self._chunk_source_ids.get(chunk_id)
            if source_id not in eligible or source_id in completed:
                continue
            rows = results[source_id]
            rows.append((chunk_id, score))
            if len(rows) >= limit_per_source:
                completed.add(source_id)
                if completed == eligible:
                    break
        return results


__all__ = ["ActivatedSourceLexicalIndex"]
