"""Transient exact index over persisted episode representative embeddings.

The index moves query-independent validation and vector normalization out of
the retrieval hot path.  SQLite remains authoritative: this module adds no
schema and writes no sidecar.  A catalog is reconstructed with one ordered
joined query, bound to the caller's discourse snapshot and the two durable
revision clocks that can change its inputs.

Catalog failures are deliberately typed.  Callers should catch
``EpisodeDescriptorIndexUnavailable`` and preserve the complete candidate
population for the slower retrieval path; a malformed or stale cache must
never become an exclusion decision.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Literal

import numpy as np

from memory_condense.domain.discourse import identity_sha256
from memory_condense.persistence.db import Database


CatalogFailureReason = Literal[
    "released",
    "not_built",
    "unknown_artifact",
    "invalid_snapshot",
    "invalid_embedding_identity",
    "embedding_identity_mismatch",
    "missing_revision_state",
    "invalid_revision_state",
    "catalog_changed_during_build",
    "stale_catalog",
    "invalid_episode_identity",
    "invalid_episode_order",
    "missing_representative",
    "invalid_representative_identity",
    "invalid_representative_rank",
    "duplicate_representative",
    "representative_outside_episode",
    "missing_representative_chunk",
    "missing_representative_vector",
    "invalid_representative_vector",
    "dimension_mismatch",
    "zero_representative_vector",
    "vector_identity_mismatch",
    "invalid_query_vector",
    "zero_query_vector",
    "invalid_episode_ids",
    "missing_episode",
    "invalid_max_representatives",
]


_CATALOG_FORMAT = "memory-condense-episode-descriptor-catalog-v1"
_SCORE_FORMAT = "memory-condense-episode-descriptor-score-v1"
_ALGORITHM = "episode-representative-max-cosine-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


_CATALOG_SQL = """
SELECT
    a.artifact_id,
    e.episode_id,
    e.source_id,
    e.sequence_no,
    r.episode_id,
    r.chunk_id,
    r.rank,
    r.vector_identity_sha256,
    c.chunk_id,
    c.embedding,
    c.text,
    CASE
        WHEN r.chunk_id IS NULL THEN NULL
        ELSE EXISTS (
            SELECT 1
            FROM episode_evidence AS ee
            WHERE ee.episode_id = e.episode_id
              AND ee.chunk_id = r.chunk_id
        )
    END AS representative_is_evidence
FROM discourse_artifacts AS a
LEFT JOIN episodes AS e
  ON e.artifact_id = a.artifact_id
LEFT JOIN episode_representatives AS r
  ON r.episode_id = e.episode_id
LEFT JOIN chunks AS c
  ON c.chunk_id = r.chunk_id
WHERE a.artifact_id = ?
ORDER BY
    e.source_id,
    e.sequence_no,
    e.episode_id,
    r.rank,
    r.chunk_id
"""


class EpisodeDescriptorIndexUnavailable(RuntimeError):
    """A derived catalog cannot safely narrow the retrieval population."""

    def __init__(
        self,
        reason: CatalogFailureReason,
        detail: str | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail
        message = f"episode descriptor index unavailable: {reason}"
        if detail:
            message += f" ({detail})"
        super().__init__(message)


def _unavailable(
    reason: CatalogFailureReason,
    detail: str | None = None,
) -> EpisodeDescriptorIndexUnavailable:
    return EpisodeDescriptorIndexUnavailable(reason, detail)


def _digest(value: object, *, reason: CatalogFailureReason) -> str:
    normalized = str(value)
    if not _SHA256_RE.fullmatch(normalized):
        raise _unavailable(reason)
    return normalized


def _nonempty(value: object, *, reason: CatalogFailureReason) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise _unavailable(reason)
    return normalized


def _embedding_identity(
    value: Mapping[str, object],
    *,
    dimension: int,
) -> str:
    if not isinstance(value, Mapping) or not value:
        raise _unavailable("invalid_embedding_identity")
    body: dict[str, object] = {}
    for key, child in value.items():
        if type(key) is not str or not key.strip():
            raise _unavailable("invalid_embedding_identity")
        body[key] = child
    declared_dimension = body.get("dimension")
    if declared_dimension is not None and (
        isinstance(declared_dimension, bool)
        or not isinstance(declared_dimension, int)
        or declared_dimension != dimension
    ):
        raise _unavailable("invalid_embedding_identity")
    try:
        return identity_sha256(body)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _unavailable("invalid_embedding_identity") from exc


@dataclass(frozen=True, slots=True)
class EpisodeDescriptorCatalogReceipt:
    """Text- and vector-free identity of one resident descriptor catalog."""

    artifact_id: str
    snapshot_sha256: str
    embedding_identity_sha256: str
    algorithm: str
    dimension: int
    graph_content_revision: int
    chunk_index_revision: int
    episode_count: int
    representative_count: int
    population_identity_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("episode descriptor catalog receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": _CATALOG_FORMAT,
            "artifact_id": self.artifact_id,
            "snapshot_sha256": self.snapshot_sha256,
            "embedding_identity_sha256": self.embedding_identity_sha256,
            "algorithm": self.algorithm,
            "dimension": self.dimension,
            "graph_content_revision": self.graph_content_revision,
            "chunk_index_revision": self.chunk_index_revision,
            "episode_count": self.episode_count,
            "representative_count": self.representative_count,
            "population_identity_sha256": self.population_identity_sha256,
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class EpisodeDescriptorEnsureTiming:
    """Nondeterministic build/cache timing kept outside catalog identity."""

    elapsed_ms: float
    rebuilt: bool
    episode_count: int
    representative_count: int
    semantic_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class EpisodeDescriptorEnsureResult:
    receipt: EpisodeDescriptorCatalogReceipt
    rebuilt: bool
    timing: EpisodeDescriptorEnsureTiming


@dataclass(frozen=True, slots=True)
class EpisodeDescriptorScore:
    episode_id: str
    score: float

    def __post_init__(self) -> None:
        if not str(self.episode_id).strip() or not math.isfinite(float(self.score)):
            raise ValueError("episode descriptor scores require finite values and IDs")


@dataclass(frozen=True, slots=True)
class EpisodeDescriptorScoreReceipt:
    """Vector-free binding of one query feature to its ordered score result."""

    catalog_receipt_sha256: str
    query_feature_sha256: str
    max_representatives: int
    offered_episode_ids: tuple[str, ...]
    ranked_scores: tuple[EpisodeDescriptorScore, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("episode descriptor score receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": _SCORE_FORMAT,
            "catalog_receipt_sha256": self.catalog_receipt_sha256,
            "query_feature_sha256": self.query_feature_sha256,
            "max_representatives": self.max_representatives,
            "offered_episode_ids": list(self.offered_episode_ids),
            "ranked_scores": [
                {"episode_id": item.episode_id, "score": item.score}
                for item in self.ranked_scores
            ],
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class EpisodeDescriptorScoreTiming:
    """Nondeterministic score timing kept outside score identity."""

    elapsed_ms: float
    offered_count: int
    representative_count: int
    semantic_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class EpisodeDescriptorScoreResult:
    scores: tuple[EpisodeDescriptorScore, ...]
    receipt: EpisodeDescriptorScoreReceipt
    timing: EpisodeDescriptorScoreTiming


@dataclass(slots=True)
class _ResidentCatalog:
    receipt: EpisodeDescriptorCatalogReceipt
    coordinate: tuple[int, int]
    episode_rows: dict[str, tuple[int, ...]]
    matrix: np.ndarray


class EpisodeDescriptorIndex:
    """Lazy exact cosine index over immutable episode representative rows."""

    def __init__(self, db: Database, *, dimension: int) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self._db = db
        self._dimension = dimension
        self._catalog: _ResidentCatalog | None = None
        self._released = False
        self._lock = RLock()

    @property
    def built(self) -> bool:
        return self._catalog is not None

    @property
    def resident_bytes(self) -> int:
        catalog = self._catalog
        return 0 if catalog is None else int(catalog.matrix.nbytes)

    @property
    def receipt(self) -> EpisodeDescriptorCatalogReceipt | None:
        catalog = self._catalog
        return None if catalog is None else catalog.receipt

    def invalidate(self) -> None:
        """Drop all derived rows; the next ``ensure`` may rebuild them."""

        with self._lock:
            self._catalog = None

    def release(self) -> None:
        """Permanently shed resident buffers for a closing facade."""

        with self._lock:
            self._catalog = None
            self._released = True

    def build(
        self,
        artifact_id: str,
        *,
        snapshot_sha256: str,
        embedding_identity: Mapping[str, object],
    ) -> EpisodeDescriptorEnsureResult:
        """Force one fresh verified catalog build."""

        return self._ensure(
            artifact_id,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=embedding_identity,
            force=True,
        )

    def ensure(
        self,
        artifact_id: str,
        *,
        snapshot_sha256: str,
        embedding_identity: Mapping[str, object],
    ) -> EpisodeDescriptorEnsureResult:
        """Return a current catalog, rebuilding after any durable change."""

        return self._ensure(
            artifact_id,
            snapshot_sha256=snapshot_sha256,
            embedding_identity=embedding_identity,
            force=False,
        )

    def _ensure(
        self,
        artifact_id: str,
        *,
        snapshot_sha256: str,
        embedding_identity: Mapping[str, object],
        force: bool,
    ) -> EpisodeDescriptorEnsureResult:
        started_ns = time.perf_counter_ns()
        selected_artifact = _nonempty(
            artifact_id,
            reason="invalid_episode_identity",
        )
        selected_snapshot = _digest(
            snapshot_sha256,
            reason="invalid_snapshot",
        )
        embedding_sha256 = _embedding_identity(
            embedding_identity,
            dimension=self._dimension,
        )
        with self._lock:
            if self._released:
                raise _unavailable("released")
            coordinate = self._read_coordinate()
            catalog = self._catalog
            reusable = bool(
                not force
                and catalog is not None
                and catalog.coordinate == coordinate
                and catalog.receipt.artifact_id == selected_artifact
                and catalog.receipt.snapshot_sha256 == selected_snapshot
                and catalog.receipt.embedding_identity_sha256 == embedding_sha256
            )
            if not reusable:
                catalog = self._build_catalog(
                    selected_artifact,
                    snapshot_sha256=selected_snapshot,
                    embedding_identity_sha256=embedding_sha256,
                    coordinate=coordinate,
                )
                self._catalog = catalog
            assert catalog is not None
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            timing = EpisodeDescriptorEnsureTiming(
                elapsed_ms=elapsed_ms,
                rebuilt=not reusable,
                episode_count=catalog.receipt.episode_count,
                representative_count=catalog.receipt.representative_count,
                semantic_receipt_sha256=catalog.receipt.receipt_sha256,
            )
            return EpisodeDescriptorEnsureResult(
                receipt=catalog.receipt,
                rebuilt=not reusable,
                timing=timing,
            )

    def _read_coordinate(self) -> tuple[int, int]:
        try:
            row = self._db.execute(
                "SELECT s.graph_content_revision, "
                "COALESCE((SELECT CAST(value AS INTEGER) FROM meta "
                "WHERE key = 'chunk_index_revision'), 0) "
                "FROM discourse_revision_state AS s WHERE s.singleton = 1"
            ).fetchone()
        except Exception as exc:
            raise _unavailable("missing_revision_state") from exc
        if row is None:
            raise _unavailable("missing_revision_state")
        try:
            graph_revision = int(row[0])
            chunk_revision = int(row[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise _unavailable("invalid_revision_state") from exc
        if graph_revision < 0 or chunk_revision < 0:
            raise _unavailable("invalid_revision_state")
        return graph_revision, chunk_revision

    def _build_catalog(
        self,
        artifact_id: str,
        *,
        snapshot_sha256: str,
        embedding_identity_sha256: str,
        coordinate: tuple[int, int],
    ) -> _ResidentCatalog:
        try:
            rows = self._db.execute(_CATALOG_SQL, (artifact_id,)).fetchall()
        except Exception as exc:
            raise _unavailable("unknown_artifact", artifact_id) from exc
        if self._read_coordinate() != coordinate:
            raise _unavailable("catalog_changed_during_build")
        if not rows:
            raise _unavailable("unknown_artifact", artifact_id)

        episode_metadata: dict[str, tuple[str, int]] = {}
        episode_representatives: dict[str, list[dict[str, object]]] = {}
        episode_matrix_rows: dict[str, list[int]] = {}
        normalized_vectors: list[np.ndarray] = []
        seen_rep_ranks: set[tuple[str, int]] = set()
        seen_rep_chunks: set[tuple[str, str]] = set()

        for row in rows:
            if str(row[0]) != artifact_id:
                raise _unavailable("invalid_episode_identity")
            if row[1] is None:
                if len(rows) != 1 or any(value is not None for value in row[2:]):
                    raise _unavailable("invalid_episode_identity")
                continue

            episode_id = _nonempty(
                row[1],
                reason="invalid_episode_identity",
            )
            source_id = _nonempty(
                row[2],
                reason="invalid_episode_identity",
            )
            try:
                sequence_no = int(row[3])
            except (TypeError, ValueError, OverflowError) as exc:
                raise _unavailable("invalid_episode_order", episode_id) from exc
            if isinstance(row[3], bool) or sequence_no < 0 or row[3] != sequence_no:
                raise _unavailable("invalid_episode_order", episode_id)
            metadata = (source_id, sequence_no)
            prior_metadata = episode_metadata.setdefault(episode_id, metadata)
            if prior_metadata != metadata:
                raise _unavailable("invalid_episode_identity", episode_id)

            if row[4] is None:
                raise _unavailable("missing_representative", episode_id)
            if str(row[4]) != episode_id:
                raise _unavailable("invalid_representative_identity", episode_id)
            chunk_id = _nonempty(
                row[5],
                reason="invalid_representative_identity",
            )
            try:
                rank = int(row[6])
            except (TypeError, ValueError, OverflowError) as exc:
                raise _unavailable("invalid_representative_rank", episode_id) from exc
            if isinstance(row[6], bool) or rank < 0 or row[6] != rank:
                raise _unavailable("invalid_representative_rank", episode_id)
            representative_identity = _digest(
                row[7],
                reason="invalid_representative_identity",
            )
            if row[8] is None or str(row[8]) != chunk_id:
                raise _unavailable("missing_representative_chunk", chunk_id)
            if row[11] != 1:
                raise _unavailable("representative_outside_episode", episode_id)
            if (episode_id, rank) in seen_rep_ranks or (
                episode_id,
                chunk_id,
            ) in seen_rep_chunks:
                raise _unavailable("duplicate_representative", episode_id)
            seen_rep_ranks.add((episode_id, rank))
            seen_rep_chunks.add((episode_id, chunk_id))

            blob = row[9]
            if blob is None:
                raise _unavailable("missing_representative_vector", chunk_id)
            try:
                raw = np.frombuffer(blob, dtype=np.float32)
            except (TypeError, ValueError, BufferError) as exc:
                raise _unavailable("invalid_representative_vector", chunk_id) from exc
            if raw.ndim != 1 or raw.size != self._dimension:
                raise _unavailable("dimension_mismatch", chunk_id)
            if not np.isfinite(raw).all():
                raise _unavailable("invalid_representative_vector", chunk_id)
            norm = float(np.linalg.norm(raw))
            if not math.isfinite(norm):
                raise _unavailable("invalid_representative_vector", chunk_id)
            if norm <= 0.0:
                raise _unavailable("zero_representative_vector", chunk_id)

            raw_values = [
                0.0 if float(value) == 0.0 else float(value)
                for value in raw
            ]
            ordinary_identity = identity_sha256(
                {
                    "method": "ordinary_embedding",
                    "chunk_id": chunk_id,
                    "vector": raw_values,
                }
            )
            lexical_identity = identity_sha256(
                {
                    "method": "lexical_control",
                    "chunk_id": chunk_id,
                    "text_sha256": identity_sha256(str(row[10])),
                }
            )
            if representative_identity not in {
                ordinary_identity,
                lexical_identity,
            }:
                raise _unavailable("vector_identity_mismatch", chunk_id)
            identity_method = (
                "ordinary_embedding"
                if representative_identity == ordinary_identity
                else "lexical_control"
            )

            matrix_index = len(normalized_vectors)
            unit = raw.astype(np.float32, copy=True)
            unit /= norm
            normalized_vectors.append(unit)
            episode_matrix_rows.setdefault(episode_id, []).append(matrix_index)
            episode_representatives.setdefault(episode_id, []).append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "identity_method": identity_method,
                    "identity_sha256": representative_identity,
                }
            )

        self._validate_episode_order(episode_metadata)
        population: list[dict[str, object]] = []
        frozen_rows: dict[str, tuple[int, ...]] = {}
        ordered_episode_ids = sorted(
            episode_metadata,
            key=lambda episode_id: (
                episode_metadata[episode_id][0],
                episode_metadata[episode_id][1],
                episode_id,
            ),
        )
        for episode_id in ordered_episode_ids:
            representatives = episode_representatives.get(episode_id)
            if not representatives:
                raise _unavailable("missing_representative", episode_id)
            ranks = tuple(int(item["rank"]) for item in representatives)
            if ranks != tuple(range(len(ranks))):
                raise _unavailable("invalid_representative_rank", episode_id)
            frozen_rows[episode_id] = tuple(episode_matrix_rows[episode_id])
            source_id, sequence_no = episode_metadata[episode_id]
            population.append(
                {
                    "episode_id": episode_id,
                    "source_id": source_id,
                    "sequence_no": sequence_no,
                    "representatives": representatives,
                }
            )

        matrix = (
            np.stack(normalized_vectors).astype(np.float32, copy=False)
            if normalized_vectors
            else np.zeros((0, self._dimension), dtype=np.float32)
        )
        matrix = np.ascontiguousarray(matrix)
        matrix.setflags(write=False)
        receipt = EpisodeDescriptorCatalogReceipt(
            artifact_id=artifact_id,
            snapshot_sha256=snapshot_sha256,
            embedding_identity_sha256=embedding_identity_sha256,
            algorithm=_ALGORITHM,
            dimension=self._dimension,
            graph_content_revision=coordinate[0],
            chunk_index_revision=coordinate[1],
            episode_count=len(frozen_rows),
            representative_count=len(normalized_vectors),
            population_identity_sha256=identity_sha256(population),
        )
        return _ResidentCatalog(
            receipt=receipt,
            coordinate=coordinate,
            episode_rows=frozen_rows,
            matrix=matrix,
        )

    @staticmethod
    def _validate_episode_order(
        metadata: Mapping[str, tuple[str, int]],
    ) -> None:
        by_source: dict[str, list[tuple[int, str]]] = {}
        for episode_id, (source_id, sequence_no) in metadata.items():
            by_source.setdefault(source_id, []).append((sequence_no, episode_id))
        for rows in by_source.values():
            rows.sort()
            for left, right in zip(rows, rows[1:]):
                if right[0] != left[0] + 1:
                    raise _unavailable("invalid_episode_order", right[1])

    def score(
        self,
        query_vector: Sequence[float],
        episode_ids: Sequence[str],
        *,
        max_representatives: int,
        embedding_identity: Mapping[str, object],
    ) -> EpisodeDescriptorScoreResult:
        """Score explicit episodes by max cosine, preserving offered-order ties."""

        started_ns = time.perf_counter_ns()
        with self._lock:
            if self._released:
                raise _unavailable("released")
            catalog = self._catalog
            if catalog is None:
                raise _unavailable("not_built")
            if self._read_coordinate() != catalog.coordinate:
                raise _unavailable("stale_catalog")
            embedding_sha256 = _embedding_identity(
                embedding_identity,
                dimension=self._dimension,
            )
            if embedding_sha256 != catalog.receipt.embedding_identity_sha256:
                raise _unavailable("embedding_identity_mismatch")
            if (
                isinstance(max_representatives, bool)
                or not isinstance(max_representatives, int)
                or max_representatives < 1
            ):
                raise _unavailable("invalid_max_representatives")

            if isinstance(episode_ids, (str, bytes)):
                raise _unavailable("invalid_episode_ids")
            offered = tuple(str(value).strip() for value in episode_ids)
            if any(not value for value in offered) or len(set(offered)) != len(offered):
                raise _unavailable("invalid_episode_ids")
            missing = [
                episode_id
                for episode_id in offered
                if episode_id not in catalog.episode_rows
            ]
            if missing:
                raise _unavailable("missing_episode", missing[0])

            try:
                raw_query_values = tuple(float(value) for value in query_vector)
            except (TypeError, ValueError, OverflowError) as exc:
                raise _unavailable("invalid_query_vector") from exc
            query = np.asarray(raw_query_values, dtype=np.float32)
            if query.ndim != 1 or query.size != self._dimension:
                raise _unavailable("dimension_mismatch")
            if not np.isfinite(query).all():
                raise _unavailable("invalid_query_vector")
            query_norm = float(np.linalg.norm(query))
            if not math.isfinite(query_norm):
                raise _unavailable("invalid_query_vector")
            if query_norm <= 0.0:
                raise _unavailable("zero_query_vector")
            query = query / query_norm

            selected_rows: list[int] = []
            row_counts: list[int] = []
            for episode_id in offered:
                rows = catalog.episode_rows[episode_id][:max_representatives]
                if not rows:
                    raise _unavailable("missing_representative", episode_id)
                selected_rows.extend(rows)
                row_counts.append(len(rows))

            representative_scores = (
                catalog.matrix[np.asarray(selected_rows, dtype=np.intp)] @ query
            )
            offered_scores: list[EpisodeDescriptorScore] = []
            cursor = 0
            for episode_id, count in zip(offered, row_counts, strict=True):
                score = float(np.max(representative_scores[cursor : cursor + count]))
                cursor += count
                if not math.isfinite(score):
                    raise _unavailable("invalid_representative_vector", episode_id)
                offered_scores.append(EpisodeDescriptorScore(episode_id, score))
            ranked = tuple(
                row
                for _offered_index, row in sorted(
                    enumerate(offered_scores),
                    key=lambda item: (-item[1].score, item[0]),
                )
            )
            raw_query = [
                0.0 if value == 0.0 else value for value in raw_query_values
            ]
            query_feature_sha256 = identity_sha256(
                {
                    "embedding_identity_sha256": embedding_sha256,
                    "vector": raw_query,
                }
            )
            receipt = EpisodeDescriptorScoreReceipt(
                catalog_receipt_sha256=catalog.receipt.receipt_sha256,
                query_feature_sha256=query_feature_sha256,
                max_representatives=max_representatives,
                offered_episode_ids=offered,
                ranked_scores=ranked,
            )
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            timing = EpisodeDescriptorScoreTiming(
                elapsed_ms=elapsed_ms,
                offered_count=len(offered),
                representative_count=len(selected_rows),
                semantic_receipt_sha256=receipt.receipt_sha256,
            )
            return EpisodeDescriptorScoreResult(
                scores=ranked,
                receipt=receipt,
                timing=timing,
            )


__all__ = [
    "CatalogFailureReason",
    "EpisodeDescriptorCatalogReceipt",
    "EpisodeDescriptorEnsureResult",
    "EpisodeDescriptorEnsureTiming",
    "EpisodeDescriptorIndex",
    "EpisodeDescriptorIndexUnavailable",
    "EpisodeDescriptorScore",
    "EpisodeDescriptorScoreReceipt",
    "EpisodeDescriptorScoreResult",
    "EpisodeDescriptorScoreTiming",
]
