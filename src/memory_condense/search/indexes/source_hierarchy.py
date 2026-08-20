"""Transient, provenance-preserving hierarchical source contraction.

The hierarchy stores no text and creates no synthetic memory payload. Leaves
are durable source IDs, internal nodes hold only child pointers and a derived
centroid. It is rebuilt lazily from authoritative chunk embeddings and is
invalidated on live writes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from memory_condense.persistence.db import (
    INDEXED_CHUNK_SQL,
    TURN_SOURCE_ID_SQL,
    Database,
)


@dataclass(frozen=True, slots=True)
class SourceContractionNode:
    key: str
    level: int
    children: tuple[str, ...]
    member_count: int


class SourceContractionIndex:
    """Deterministic bounded-depth pairwise contraction over source centroids."""

    def __init__(
        self,
        db: Database,
        *,
        dim: int,
        partition_separator: str = "::",
        max_levels: int = 5,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        if max_levels < 1:
            raise ValueError("max_levels must be positive")
        self._db = db
        self._dim = dim
        self._separator = partition_separator
        self._max_levels = max_levels
        self._built = False
        self._nodes: dict[str, SourceContractionNode] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._parents: dict[str, str] = {}
        self._leaf_keys: dict[str, str] = {}

    def invalidate(self) -> None:
        self._built = False
        self._nodes.clear()
        self._vectors.clear()
        self._parents.clear()
        self._leaf_keys.clear()

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 1e-9 else value

    def _partition(self, source_id: str) -> str:
        if self._separator and self._separator in source_id:
            return source_id.split(self._separator, 1)[0]
        return "__root__"

    def _load_source_vectors(self) -> dict[str, tuple[np.ndarray, int]]:
        rows = self._db.execute(
            f"SELECT {TURN_SOURCE_ID_SQL}, c.embedding "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {INDEXED_CHUNK_SQL} "
            f"ORDER BY {TURN_SOURCE_ID_SQL}, c.rowid"
        )
        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for source_id, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.shape != (self._dim,):
                continue
            key = str(source_id)
            normalized = self._normalized(vector)
            if key in sums:
                sums[key] += normalized
                counts[key] += 1
            else:
                sums[key] = normalized.copy()
                counts[key] = 1
        return {
            source_id: (self._normalized(vector), counts[source_id])
            for source_id, vector in sums.items()
        }

    @staticmethod
    def _parent_key(level: int, children: Sequence[str]) -> str:
        digest = hashlib.sha256("\0".join(children).encode("utf-8")).hexdigest()[:20]
        return f"h:{level}:{digest}"

    def _build(self) -> None:
        self.invalidate()
        sources = self._load_source_vectors()
        partitions: dict[str, list[str]] = {}
        for source_id, (vector, count) in sources.items():
            leaf_key = f"s:{source_id}"
            self._leaf_keys[source_id] = leaf_key
            self._nodes[leaf_key] = SourceContractionNode(
                key=leaf_key,
                level=0,
                children=(),
                member_count=count,
            )
            self._vectors[leaf_key] = vector
            partitions.setdefault(self._partition(source_id), []).append(leaf_key)

        for partition in sorted(partitions):
            current = sorted(partitions[partition])
            for level in range(1, self._max_levels + 1):
                if len(current) <= 1:
                    break
                matrix = np.stack([self._vectors[key] for key in current])
                similarity = matrix @ matrix.T
                used: set[int] = set()
                next_level: list[str] = []
                for left_index, left_key in enumerate(current):
                    if left_index in used:
                        continue
                    available = [
                        index
                        for index in range(left_index + 1, len(current))
                        if index not in used
                    ]
                    if not available:
                        next_level.append(left_key)
                        used.add(left_index)
                        continue
                    right_index = max(
                        available,
                        key=lambda index: (
                            float(similarity[left_index, index]),
                            current[index],
                        ),
                    )
                    right_key = current[right_index]
                    used.update((left_index, right_index))
                    children = tuple(sorted((left_key, right_key)))
                    parent_key = self._parent_key(level, children)
                    left_count = self._nodes[left_key].member_count
                    right_count = self._nodes[right_key].member_count
                    vector = self._normalized(
                        self._vectors[left_key] * left_count
                        + self._vectors[right_key] * right_count
                    )
                    self._nodes[parent_key] = SourceContractionNode(
                        key=parent_key,
                        level=level,
                        children=children,
                        member_count=left_count + right_count,
                    )
                    self._vectors[parent_key] = vector
                    self._parents[left_key] = parent_key
                    self._parents[right_key] = parent_key
                    next_level.append(parent_key)
                current = sorted(next_level)
        self._built = True

    def _descendant_sources(self, node_key: str) -> tuple[str, ...]:
        node = self._nodes[node_key]
        if not node.children:
            return (node_key[2:],)
        descendants: list[str] = []
        for child in node.children:
            descendants.extend(self._descendant_sources(child))
        return tuple(descendants)

    def expand(
        self,
        query_embedding: np.ndarray,
        seed_source_ids: Sequence[str],
        *,
        slots: int = 8,
        hops: int = 2,
    ) -> list[tuple[str, float]]:
        """Return related leaf sources reached through contraction parents."""

        if slots < 0 or hops < 1:
            raise ValueError("slots must be non-negative and hops positive")
        if slots == 0:
            return []
        if not self._built:
            self._build()
        query = self._normalized(np.asarray(query_embedding, dtype=np.float32))
        if query.shape != (self._dim,):
            raise ValueError("query embedding dimension does not match hierarchy")

        seeds = {str(value) for value in seed_source_ids if str(value)}
        candidates: set[str] = set()
        for source_id in sorted(seeds):
            node_key = self._leaf_keys.get(source_id)
            if node_key is None:
                continue
            ancestor = node_key
            for _ in range(hops):
                parent = self._parents.get(ancestor)
                if parent is None:
                    break
                ancestor = parent
            candidates.update(self._descendant_sources(ancestor))
        candidates.difference_update(seeds)

        ranked = [
            (
                source_id,
                float(self._vectors[self._leaf_keys[source_id]] @ query),
            )
            for source_id in candidates
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:slots]

    def stats(self) -> dict[str, int]:
        if not self._built:
            self._build()
        return {
            "sources": len(self._leaf_keys),
            "nodes": len(self._nodes),
            "internal_nodes": len(self._nodes) - len(self._leaf_keys),
            "max_level": max((node.level for node in self._nodes.values()), default=0),
        }
