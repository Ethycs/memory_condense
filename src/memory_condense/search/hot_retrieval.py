"""Compact exact-dense addresses for provider-free hot retrieval.

The durable memory database remains the authority for raw evidence.  This
module owns only query-independent, row-normalized chunk addresses and compact
source-centroid descriptors.  A query returns discrete IDs that callers must
hydrate from the source store before exposing evidence to a model.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class RankedChunkAddress:
    """One scalar-ranked pointer into the authoritative raw chunk store."""

    chunk_id: str
    score: float
    route: str

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.route, str) or not self.route:
            raise ValueError("route must be a non-empty string")
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True, slots=True)
class RankedSourceAddress:
    """A source-centroid hit and its best query-local raw-chunk address."""

    source_id: str
    score: float
    representative_chunk_id: str
    representative_score: float
    member_count: int
    route: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string")
        if (
            not isinstance(self.representative_chunk_id, str)
            or not self.representative_chunk_id
        ):
            raise ValueError("representative_chunk_id must be a non-empty string")
        if not np.isfinite(self.score) or not np.isfinite(self.representative_score):
            raise ValueError("source and representative scores must be finite")
        if isinstance(self.member_count, bool) or not isinstance(self.member_count, int):
            raise TypeError("member_count must be an integer")
        if self.member_count < 1:
            raise ValueError("member_count must be positive")
        if not isinstance(self.route, str) or not self.route:
            raise ValueError("route must be a non-empty string")


class ExactDenseAddressIndex:
    """Read-only exact cosine search over normalized chunk addresses.

    Rows must already be L2-normalized during the query-independent compile
    phase.  IDs must be strictly increasing, which makes stable score sorting
    resolve ties by exact chunk ID without another query-time sort key.
    """

    def __init__(
        self,
        chunk_ids: Sequence[str],
        matrix: np.ndarray,
        *,
        validate_rows: bool = True,
        _copy_matrix: bool = True,
    ) -> None:
        ids = tuple(str(value) for value in chunk_ids)
        if not ids or any(not value for value in ids):
            raise ValueError("chunk_ids must be non-empty strings")
        if any(left >= right for left, right in zip(ids, ids[1:])):
            raise ValueError("chunk_ids must be unique and strictly increasing")

        values = np.asarray(matrix)
        if values.dtype != np.float32:
            raise TypeError("dense address matrix must use float32")
        if values.ndim != 2 or values.shape[0] != len(ids):
            raise ValueError("dense address matrix rows must align with chunk_ids")
        if values.shape[1] < 1:
            raise ValueError("dense address matrix must have positive width")
        if not np.isfinite(values).all():
            raise ValueError("dense address matrix contains non-finite values")
        if validate_rows:
            norms = np.linalg.norm(values, axis=1)
            if not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-5):
                raise ValueError("dense address matrix rows must be L2-normalized")

        if _copy_matrix:
            resident = np.array(values, dtype=np.float32, order="C", copy=True)
        else:
            if values.flags.writeable:
                raise ValueError("borrowed dense address matrix must be read-only")
            resident = values
        resident.setflags(write=False)

        self._chunk_ids = ids
        self._matrix = resident
        self._closed = False

    @classmethod
    def open(
        cls,
        chunk_ids: Sequence[str],
        matrix_path: str | Path,
        *,
        validate_rows: bool = True,
    ) -> "ExactDenseAddressIndex":
        """Memory-map a compiled NumPy matrix without copying it to the GPU."""

        matrix = np.load(Path(matrix_path), mmap_mode="r", allow_pickle=False)
        return cls(
            chunk_ids,
            matrix,
            validate_rows=validate_rows,
            _copy_matrix=False,
        )

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self._matrix.nbytes)

    def score_all(self, query_embedding: np.ndarray) -> np.ndarray:
        """Return immutable chunk-query cosine scores in ``chunk_ids`` order."""

        if self._closed:
            raise RuntimeError("dense address index is closed")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape != (self.dimension,) or not np.isfinite(query).all():
            raise ValueError("query embedding has the wrong shape or is non-finite")
        norm = float(np.linalg.norm(query))
        if norm <= 0.0:
            raise ValueError("query embedding must be non-zero")
        scores = np.asarray(self._matrix @ (query / norm), dtype=np.float32)
        scores.setflags(write=False)
        return scores

    def close(self) -> None:
        """Release an underlying memory map, if any.

        This is primarily useful when a process streams several independent
        hot namespaces on Windows, where relying on garbage collection can
        leave the compiled matrix file mapped longer than intended.
        """

        if self._closed:
            return
        candidate: object | None = self._matrix
        seen: set[int] = set()
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            mapping = getattr(candidate, "_mmap", None)
            if mapping is not None:
                mapping.close()
                break
            candidate = getattr(candidate, "base", None)
        self._closed = True

    def __enter__(self) -> "ExactDenseAddressIndex":
        if self._closed:
            raise RuntimeError("dense address index is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        limit: int,
        route: str = "exact_dense",
    ) -> tuple[RankedChunkAddress, ...]:
        """Return exact top cosine addresses with deterministic ID tie-breaks."""

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit <= 0:
            return ()
        scores = self.score_all(query_embedding)
        # Compilation sorts IDs ascending. Stable descending score order thus
        # gives an exact, deterministic ID tie-break without argpartition's
        # ambiguous boundary behavior.
        order = np.argsort(-scores, kind="stable")[: min(limit, self.count)]
        return tuple(
            RankedChunkAddress(
                chunk_id=self._chunk_ids[int(index)],
                score=float(scores[int(index)]),
                route=route,
            )
            for index in order
        )


class ExactSourceDescriptorIndex:
    """Immutable exact search over query-independent source centroids.

    Compilation groups the sorted chunk address space by source, sums each
    source's normalized chunk rows in deterministic chunk-ID order, and stores
    only a normalized centroid plus compact membership row indices.  Querying
    accepts the chunk-query cosine vector already produced by
    :meth:`ExactDenseAddressIndex.score_all`; that vector elects one raw-chunk
    representative per selected source without another chunk-matrix scan.
    """

    __slots__ = (
        "_centroids",
        "_chunk_ids",
        "_chunk_source_ids",
        "_locked",
        "_member_offsets",
        "_member_rows",
        "_source_ids",
    )

    def __init__(
        self,
        chunk_ids: Sequence[str],
        matrix: np.ndarray,
        chunk_source_ids: Sequence[str],
    ) -> None:
        object.__setattr__(self, "_locked", False)

        ids = tuple(chunk_ids)
        if not ids or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("chunk_ids must be non-empty strings")
        if any(left >= right for left, right in zip(ids, ids[1:])):
            raise ValueError("chunk_ids must be unique and strictly increasing")
        if len(ids) - 1 > np.iinfo(np.uint32).max:
            raise ValueError("source membership exceeds the uint32 row-address space")

        source_by_chunk = tuple(chunk_source_ids)
        if len(source_by_chunk) != len(ids):
            raise ValueError("chunk_source_ids must align with chunk_ids")
        if any(
            not isinstance(value, str) or not value for value in source_by_chunk
        ):
            raise ValueError("chunk_source_ids must be non-empty strings")

        values = np.asarray(matrix)
        if values.dtype != np.float32:
            raise TypeError("source compile matrix must use float32")
        if values.ndim != 2 or values.shape[0] != len(ids):
            raise ValueError("source compile matrix rows must align with chunk_ids")
        if values.shape[1] < 1:
            raise ValueError("source compile matrix must have positive width")
        if not np.isfinite(values).all():
            raise ValueError("source compile matrix contains non-finite values")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-5):
            raise ValueError("source compile matrix rows must be L2-normalized")

        source_ids = tuple(sorted(set(source_by_chunk)))
        source_position = {
            source_id: position for position, source_id in enumerate(source_ids)
        }
        grouped_rows: list[list[int]] = [[] for _ in source_ids]
        sums = np.zeros((len(source_ids), values.shape[1]), dtype=np.float64)
        for row, source_id in enumerate(source_by_chunk):
            position = source_position[source_id]
            grouped_rows[position].append(row)
            sums[position] += values[row]

        centroid_norms = np.linalg.norm(sums, axis=1)
        zero_positions = np.flatnonzero(centroid_norms <= 1e-12)
        if zero_positions.size:
            source_id = source_ids[int(zero_positions[0])]
            raise ValueError(f"source {source_id!r} has a zero centroid")
        centroids = np.asarray(sums / centroid_norms[:, None], dtype=np.float32)
        # Normalize once more in float32 so the persisted query-time artifact
        # itself, rather than only its float64 precursor, satisfies the norm.
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

        member_offsets = np.zeros(len(source_ids) + 1, dtype=np.int64)
        member_offsets[1:] = np.cumsum(
            np.asarray([len(rows) for rows in grouped_rows], dtype=np.int64)
        )
        member_rows = np.asarray(
            [row for rows in grouped_rows for row in rows],
            dtype=np.uint32,
        )
        centroids.setflags(write=False)
        member_offsets.setflags(write=False)
        member_rows.setflags(write=False)

        object.__setattr__(self, "_chunk_ids", ids)
        object.__setattr__(self, "_chunk_source_ids", source_by_chunk)
        object.__setattr__(self, "_source_ids", source_ids)
        object.__setattr__(self, "_centroids", centroids)
        object.__setattr__(self, "_member_offsets", member_offsets)
        object.__setattr__(self, "_member_rows", member_rows)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError(f"{type(self).__name__} is immutable")
        object.__setattr__(self, name, value)

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids)

    @property
    def source_count(self) -> int:
        return len(self._source_ids)

    @property
    def dimension(self) -> int:
        return int(self._centroids.shape[1])

    @property
    def nbytes(self) -> int:
        """Bytes held by numeric centroid and membership artifacts."""

        return int(
            self._centroids.nbytes
            + self._member_offsets.nbytes
            + self._member_rows.nbytes
        )

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    @property
    def chunk_source_ids(self) -> tuple[str, ...]:
        return self._chunk_source_ids

    @property
    def source_ids(self) -> tuple[str, ...]:
        return self._source_ids

    @property
    def centroid_matrix(self) -> np.ndarray:
        """Read-only normalized centroids in ``source_ids`` order."""

        view = self._centroids.view()
        view.setflags(write=False)
        return view

    def members(self, source_id: str) -> tuple[str, ...]:
        """Return a source's member chunk IDs in deterministic ID order."""

        position = bisect_left(self._source_ids, source_id)
        if position == self.source_count or self._source_ids[position] != source_id:
            raise KeyError(source_id)
        start = int(self._member_offsets[position])
        end = int(self._member_offsets[position + 1])
        return tuple(
            self._chunk_ids[int(row)] for row in self._member_rows[start:end]
        )

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        chunk_scores: np.ndarray,
        limit: int,
        route: str = "exact_source_centroid",
    ) -> tuple[RankedSourceAddress, ...]:
        """Return exact top sources and their best pre-scored member chunks."""

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit <= 0:
            return ()
        if not isinstance(route, str) or not route:
            raise ValueError("route must be a non-empty string")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape != (self.dimension,) or not np.isfinite(query).all():
            raise ValueError("query embedding has the wrong shape or is non-finite")
        norm = float(np.linalg.norm(query))
        if norm <= 0.0:
            raise ValueError("query embedding must be non-zero")
        query = query / norm

        member_scores = np.asarray(chunk_scores)
        if member_scores.shape != (self.chunk_count,):
            raise ValueError("chunk_scores must align with chunk_ids")
        if not np.issubdtype(member_scores.dtype, np.floating):
            raise TypeError("chunk_scores must use a floating dtype")
        if not np.isfinite(member_scores).all():
            raise ValueError("chunk_scores contains non-finite values")
        if np.any(member_scores < -1.0001) or np.any(member_scores > 1.0001):
            raise ValueError("chunk_scores must contain cosine similarities")

        source_scores = np.asarray(self._centroids @ query, dtype=np.float32)
        # Source IDs were compiled ascending. Stable descending score order
        # therefore resolves exact centroid ties by source ID.
        order = np.argsort(-source_scores, kind="stable")[
            : min(limit, self.source_count)
        ]
        results: list[RankedSourceAddress] = []
        for raw_position in order:
            position = int(raw_position)
            start = int(self._member_offsets[position])
            end = int(self._member_offsets[position + 1])
            rows = self._member_rows[start:end]
            scores = member_scores[rows]
            # Membership rows preserve ascending chunk IDs, and argmax returns
            # the first maximum, so member ties resolve by exact chunk ID.
            local_position = int(np.argmax(scores))
            row = int(rows[local_position])
            results.append(
                RankedSourceAddress(
                    source_id=self._source_ids[position],
                    score=float(source_scores[position]),
                    representative_chunk_id=self._chunk_ids[row],
                    representative_score=float(member_scores[row]),
                    member_count=end - start,
                    route=route,
                )
            )
        return tuple(results)


__all__ = [
    "ExactDenseAddressIndex",
    "ExactSourceDescriptorIndex",
    "RankedChunkAddress",
    "RankedSourceAddress",
]
