"""Durable, compact association artifacts for transient head inspection.

The transformer is a bounded linker, never the memory store.  This module
persists only fixed-width CAV coordinates, sparse per-head episode edges, and
small lifecycle counters.  It has deliberately no representation for token
activations, attention matrices, residual streams, or K/V caches.

The API is storage-oriented rather than model-oriented so a Redis backend can
implement the same operations for a concurrent live runtime.  SQLite remains
the deterministic local implementation and restart oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from memory_condense.db import Database
from memory_condense.decay import decay_factor


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class AssociationArtifact:
    """Version identity for one compatible CAV/QK/OV artifact family."""

    artifact_id: str
    model_id: str
    checkpoint_id: str
    prefix_layers: int
    head_layer: int
    cav_layer: int | None
    concept_names: tuple[str, ...]
    head_count: int
    created_at: str = field(default_factory=_now_iso)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.concept_names)
        metadata = dict(self.metadata)
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be non-empty")
        if self.prefix_layers < 1:
            raise ValueError("prefix_layers must be positive")
        if not 0 <= self.head_layer < self.prefix_layers:
            raise ValueError("head_layer must be inside the loaded prefix")
        if self.cav_layer is not None and not 0 <= self.cav_layer < self.prefix_layers:
            raise ValueError("cav_layer must be inside the loaded prefix")
        if self.head_count < 1:
            raise ValueError("head_count must be positive")
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("concept_names must be non-empty strings and unique")
        # Fail at registration time rather than halfway through an experiment.
        _canonical_json(metadata)
        object.__setattr__(self, "concept_names", names)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        checkpoint_id: str,
        prefix_layers: int,
        head_layer: int,
        cav_layer: int | None,
        concept_names: Sequence[str],
        head_count: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> AssociationArtifact:
        """Create a stable ID from every field that changes interpretation."""
        spec = {
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
            "prefix_layers": int(prefix_layers),
            "head_layer": int(head_layer),
            "cav_layer": None if cav_layer is None else int(cav_layer),
            "concept_names": list(concept_names),
            "head_count": int(head_count),
            "metadata": dict(metadata or {}),
        }
        digest = hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()
        return cls(artifact_id=f"assoc-{digest[:24]}", **spec)

    def identity(self) -> tuple[Any, ...]:
        """Fields that must match when an artifact ID already exists."""
        return (
            self.model_id,
            self.checkpoint_id,
            self.prefix_layers,
            self.head_layer,
            self.cav_layer,
            self.concept_names,
            self.head_count,
            _canonical_json(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class StoredCAVSignature:
    chunk_id: str
    artifact_id: str
    values: tuple[float, ...]
    created_turn: int
    access_count: int
    last_access_turn: int


@dataclass(frozen=True, slots=True)
class StoredCAVNeighbor:
    chunk_id: str
    score: float
    shared_concepts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StoredHeadEdge:
    source_chunk_id: str
    destination_chunk_id: str
    artifact_id: str
    head_weights: tuple[float, ...]
    qk_score: float
    ov_transport: float
    evidence_count: int
    traversal_count: int
    last_access_turn: int
    temporal_forward: bool | None

    def utility(
        self,
        *,
        now_turn: int,
        usage_half_life: float = 100.0,
        usage_weight: float = 0.05,
    ) -> float:
        """QK evidence + transported value + decayed live traversal evidence."""
        live_usage = math.log1p(self.traversal_count) * decay_factor(
            self.last_access_turn,
            now_turn,
            usage_half_life,
        )
        return self.qk_score + math.log1p(self.ov_transport) + usage_weight * live_usage


@dataclass(frozen=True, slots=True)
class HebbianUpdate:
    """Result of one idempotent same-turn co-access observation."""

    event_id: str
    created: bool
    concepts_observed: int
    edges_reinforced: int
    edges_pruned: int


@dataclass(frozen=True, slots=True)
class StoredHebbianNeighbor:
    """A conceptual chunk recalled through decayed co-access evidence."""

    chunk_id: str
    score: float
    support: int
    anchor_chunk_id: str
    coaccess_count: int
    last_reinforced_turn: int


class AssociationStore:
    """SQLite implementation of the bounded external association layer."""

    def __init__(
        self,
        db: Database,
        *,
        cache_neighbors: bool = False,
        cache_limit: int = 512,
    ) -> None:
        self._db = db
        self._artifact_cache: dict[str, AssociationArtifact] = {}
        self._cav_neighbor_cache: OrderedDict[tuple, tuple[StoredCAVNeighbor, ...]] = (
            OrderedDict()
        )
        self._edge_neighbor_cache: OrderedDict[
            tuple, dict[str, tuple[StoredHeadEdge, ...]]
        ] = OrderedDict()
        self._cache_limit = max(0, int(cache_limit)) if cache_neighbors else 0

    def _cache_put(self, cache: OrderedDict, key: tuple, value: Any) -> None:
        if self._cache_limit == 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._cache_limit:
            cache.popitem(last=False)

    # -- artifact identity -------------------------------------------------

    def register_artifact(self, artifact: AssociationArtifact) -> AssociationArtifact:
        """Register an artifact or prove an existing ID has the same meaning."""
        self._db.execute(
            "INSERT OR IGNORE INTO association_artifacts "
            "(artifact_id, model_id, checkpoint_id, prefix_layers, head_layer, "
            "cav_layer, concept_names, head_count, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact.artifact_id,
                artifact.model_id,
                artifact.checkpoint_id,
                artifact.prefix_layers,
                artifact.head_layer,
                artifact.cav_layer,
                _canonical_json(list(artifact.concept_names)),
                artifact.head_count,
                artifact.created_at,
                _canonical_json(dict(artifact.metadata)),
            ),
        )
        self._db.commit()
        self._cav_neighbor_cache.clear()
        stored = self.get_artifact(artifact.artifact_id)
        if stored is None:  # pragma: no cover - SQLite insert/read invariant
            raise RuntimeError("artifact registration disappeared")
        if stored.identity() != artifact.identity():
            raise ValueError(
                f"artifact_id {artifact.artifact_id!r} is already registered "
                "with a different model or interpretation"
            )
        self._artifact_cache[stored.artifact_id] = stored
        return stored

    def get_artifact(self, artifact_id: str) -> AssociationArtifact | None:
        cached = self._artifact_cache.get(artifact_id)
        if cached is not None:
            return cached
        row = self._db.execute(
            "SELECT artifact_id, model_id, checkpoint_id, prefix_layers, "
            "head_layer, cav_layer, concept_names, head_count, created_at, metadata "
            "FROM association_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        artifact = AssociationArtifact(
            artifact_id=row[0],
            model_id=row[1],
            checkpoint_id=row[2],
            prefix_layers=int(row[3]),
            head_layer=int(row[4]),
            cav_layer=None if row[5] is None else int(row[5]),
            concept_names=tuple(json.loads(row[6])),
            head_count=int(row[7]),
            created_at=row[8],
            metadata=json.loads(row[9]),
        )
        self._artifact_cache[artifact_id] = artifact
        return artifact

    def _require_artifact(self, artifact_id: str) -> AssociationArtifact:
        artifact = self.get_artifact(artifact_id)
        if artifact is None:
            raise KeyError(f"unknown association artifact: {artifact_id}")
        return artifact

    @staticmethod
    def _pack_f32(
        values: Sequence[float], *, width: int, field_name: str
    ) -> bytes:
        array = np.asarray(tuple(values), dtype="<f4")
        if array.ndim != 1 or len(array) != width:
            raise ValueError(f"{field_name} must contain exactly {width} values")
        if not np.isfinite(array).all():
            raise ValueError(f"{field_name} must contain only finite values")
        return array.tobytes(order="C")

    @staticmethod
    def _unpack_f32(blob: bytes) -> tuple[float, ...]:
        return tuple(float(value) for value in np.frombuffer(blob, dtype="<f4"))

    # -- compact CAV coordinates ------------------------------------------

    def put_signature(
        self,
        chunk_id: str,
        artifact_id: str,
        values: Sequence[float],
        *,
        created_turn: int | None = None,
    ) -> None:
        artifact = self._require_artifact(artifact_id)
        if artifact.cav_layer is None or not artifact.concept_names:
            raise ValueError("artifact does not define a CAV coordinate system")
        blob = self._pack_f32(
            values,
            width=len(artifact.concept_names),
            field_name="CAV signature",
        )
        turn = self._db.current_turn() if created_turn is None else int(created_turn)
        if turn < 0:
            raise ValueError("created_turn must be non-negative")
        self._db.execute(
            "INSERT INTO chunk_cav_signatures "
            "(chunk_id, artifact_id, signature, created_turn) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(chunk_id, artifact_id) DO UPDATE SET "
            "signature = excluded.signature, created_turn = excluded.created_turn",
            (chunk_id, artifact_id, blob, turn),
        )
        self._db.commit()
        self._cav_neighbor_cache.clear()

    def get_signature(
        self, chunk_id: str, artifact_id: str
    ) -> StoredCAVSignature | None:
        artifact = self._require_artifact(artifact_id)
        row = self._db.execute(
            "SELECT signature, created_turn, access_count, last_access_turn "
            "FROM chunk_cav_signatures WHERE chunk_id = ? AND artifact_id = ?",
            (chunk_id, artifact_id),
        ).fetchone()
        if row is None:
            return None
        values = self._unpack_f32(row[0])
        if len(values) != len(artifact.concept_names):
            raise ValueError("stored CAV signature width does not match its artifact")
        return StoredCAVSignature(
            chunk_id=chunk_id,
            artifact_id=artifact_id,
            values=values,
            created_turn=int(row[1]),
            access_count=int(row[2]),
            last_access_turn=int(row[3]),
        )

    def cav_neighbors(
        self,
        seed_chunk_ids: Sequence[str],
        artifact_id: str,
        *,
        top_k: int,
        exclude: Sequence[str] = (),
    ) -> tuple[StoredCAVNeighbor, ...]:
        """Exact search in the small concept space, not transformer context."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return ()
        unique_seeds = tuple(dict.fromkeys(seed_chunk_ids))
        cache_key = (
            artifact_id,
            unique_seeds,
            top_k,
            tuple(sorted(set(exclude))),
        )
        cached = (
            self._cav_neighbor_cache.get(cache_key)
            if self._cache_limit
            else None
        )
        if cached is not None:
            self._cav_neighbor_cache.move_to_end(cache_key)
            return cached
        artifact = self._require_artifact(artifact_id)
        if not artifact.concept_names:
            return ()
        seeds: list[tuple[float, ...]] = []
        for chunk_id in unique_seeds:
            signature = self.get_signature(chunk_id, artifact_id)
            if signature is not None:
                seeds.append(signature.values)
        if not seeds:
            return ()

        rows = self._db.execute(
            "SELECT s.chunk_id, s.signature FROM chunk_cav_signatures AS s "
            "JOIN chunks AS c ON c.chunk_id = s.chunk_id "
            "WHERE s.artifact_id = ? AND c.embedding IS NOT NULL "
            "AND c.hnsw_label IS NOT NULL",
            (artifact_id,),
        ).fetchall()
        excluded = set(exclude) | set(unique_seeds)
        ranked: list[StoredCAVNeighbor] = []
        for chunk_id, blob in rows:
            if chunk_id in excluded:
                continue
            candidate = self._unpack_f32(blob)
            if len(candidate) != len(artifact.concept_names):
                raise ValueError("stored CAV signature width does not match its artifact")
            best_score = -math.inf
            best_shared: tuple[str, ...] = ()
            for seed in seeds:
                shared = tuple(
                    index
                    for index, (left, right) in enumerate(
                        zip(seed, candidate, strict=True)
                    )
                    if left > 0.0 and right > 0.0
                )
                if not shared:
                    continue
                seed_positive = np.maximum(np.asarray(seed, dtype=np.float32), 0.0)
                candidate_positive = np.maximum(
                    np.asarray(candidate, dtype=np.float32), 0.0
                )
                denominator = float(
                    np.linalg.norm(seed_positive) * np.linalg.norm(candidate_positive)
                )
                cosine = float(np.dot(seed_positive, candidate_positive)) / max(
                    denominator, 1e-12
                )
                union_count = sum(
                    left > 0.0 or right > 0.0
                    for left, right in zip(seed, candidate, strict=True)
                )
                score = cosine + 0.1 * len(shared) / max(union_count, 1)
                if score > best_score:
                    best_score = score
                    best_shared = tuple(artifact.concept_names[index] for index in shared)
            if best_shared:
                ranked.append(
                    StoredCAVNeighbor(
                        chunk_id=chunk_id,
                        score=best_score,
                        shared_concepts=best_shared,
                    )
                )
        ranked.sort(key=lambda item: (item.score, item.chunk_id), reverse=True)
        result = tuple(ranked[:top_k])
        self._cache_put(self._cav_neighbor_cache, cache_key, result)
        return result

    # -- sparse QK/OV edge graph ------------------------------------------

    def upsert_edge(
        self,
        source_chunk_id: str,
        destination_chunk_id: str,
        artifact_id: str,
        head_weights: Sequence[float],
        *,
        qk_score: float,
        ov_transport: float = 0.0,
        evidence_count: int = 1,
        temporal_forward: bool | None = None,
        reverse: bool = False,
    ) -> None:
        artifact = self._require_artifact(artifact_id)
        if source_chunk_id == destination_chunk_id:
            raise ValueError("self edges are not stored")
        weights = self._pack_f32(
            head_weights,
            width=artifact.head_count,
            field_name="head_weights",
        )
        qk = float(qk_score)
        ov = float(ov_transport)
        count = int(evidence_count)
        if not math.isfinite(qk) or qk < 0.0:
            raise ValueError("qk_score must be finite and non-negative")
        if not math.isfinite(ov) or ov < 0.0:
            raise ValueError("ov_transport must be finite and non-negative")
        if count < 1:
            raise ValueError("evidence_count must be positive")
        self._merge_edge(
            source_chunk_id,
            destination_chunk_id,
            artifact_id,
            weights,
            qk,
            ov,
            count,
            temporal_forward,
        )
        if reverse:
            reverse_direction = (
                None if temporal_forward is None else not temporal_forward
            )
            self._merge_edge(
                destination_chunk_id,
                source_chunk_id,
                artifact_id,
                weights,
                qk,
                ov,
                count,
                reverse_direction,
            )
        self._db.commit()
        self._edge_neighbor_cache.clear()

    def _merge_edge(
        self,
        source_chunk_id: str,
        destination_chunk_id: str,
        artifact_id: str,
        weights_blob: bytes,
        qk_score: float,
        ov_transport: float,
        evidence_count: int,
        temporal_forward: bool | None,
    ) -> None:
        existing = self._db.execute(
            "SELECT head_weights, qk_score, ov_transport, evidence_count, "
            "temporal_forward FROM chunk_head_edges "
            "WHERE source_chunk_id = ? AND destination_chunk_id = ? "
            "AND artifact_id = ?",
            (source_chunk_id, destination_chunk_id, artifact_id),
        ).fetchone()
        if existing is None:
            merged_weights = weights_blob
            merged_qk = qk_score
            merged_ov = ov_transport
            merged_count = evidence_count
            merged_direction = temporal_forward
        else:
            old_count = int(existing[3])
            merged_count = old_count + evidence_count
            old_weights = np.frombuffer(existing[0], dtype="<f4")
            new_weights = np.frombuffer(weights_blob, dtype="<f4")
            merged_weights = (
                (old_weights * old_count + new_weights * evidence_count)
                / merged_count
            ).astype("<f4").tobytes()
            merged_qk = (
                float(existing[1]) * old_count + qk_score * evidence_count
            ) / merged_count
            merged_ov = (
                float(existing[2]) * old_count + ov_transport * evidence_count
            ) / merged_count
            old_direction = (
                None if existing[4] is None else bool(existing[4])
            )
            merged_direction = (
                old_direction if old_direction == temporal_forward else None
            )
        self._db.execute(
            "INSERT INTO chunk_head_edges "
            "(source_chunk_id, destination_chunk_id, artifact_id, head_weights, "
            "qk_score, ov_transport, evidence_count, temporal_forward) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_chunk_id, destination_chunk_id, artifact_id) "
            "DO UPDATE SET head_weights = excluded.head_weights, "
            "qk_score = excluded.qk_score, ov_transport = excluded.ov_transport, "
            "evidence_count = excluded.evidence_count, "
            "temporal_forward = excluded.temporal_forward",
            (
                source_chunk_id,
                destination_chunk_id,
                artifact_id,
                merged_weights,
                merged_qk,
                merged_ov,
                merged_count,
                None if merged_direction is None else int(merged_direction),
            ),
        )

    def neighbors(
        self,
        source_chunk_id: str,
        artifact_id: str,
        *,
        top_k: int,
        exclude: Sequence[str] = (),
        now_turn: int | None = None,
        usage_half_life: float = 100.0,
        usage_weight: float = 0.05,
    ) -> tuple[StoredHeadEdge, ...]:
        return self.neighbors_many(
            [source_chunk_id],
            artifact_id,
            top_k_per_source=top_k,
            exclude=exclude,
            now_turn=now_turn,
            usage_half_life=usage_half_life,
            usage_weight=usage_weight,
        )[source_chunk_id]

    def neighbors_many(
        self,
        source_chunk_ids: Sequence[str],
        artifact_id: str,
        *,
        top_k_per_source: int,
        exclude: Sequence[str] = (),
        exclude_frontier: bool = True,
        now_turn: int | None = None,
        usage_half_life: float = 100.0,
        usage_weight: float = 0.05,
    ) -> dict[str, tuple[StoredHeadEdge, ...]]:
        """Fetch bounded adjacency for many anchors in one SQLite query."""
        if top_k_per_source < 0:
            raise ValueError("top_k_per_source must be non-negative")
        sources = list(dict.fromkeys(source_chunk_ids))
        if not sources:
            return {}
        if top_k_per_source == 0:
            return {source_id: () for source_id in sources}
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        cache_key = (
            artifact_id,
            tuple(sources),
            top_k_per_source,
            tuple(sorted(set(exclude))),
            bool(exclude_frontier),
            turn,
            float(usage_half_life),
            float(usage_weight),
        )
        cached = (
            self._edge_neighbor_cache.get(cache_key)
            if self._cache_limit
            else None
        )
        if cached is not None:
            self._edge_neighbor_cache.move_to_end(cache_key)
            return dict(cached)
        artifact = self._require_artifact(artifact_id)
        # Ordinary bounded expansion does not need edges among nodes already in
        # the current frontier. Heat diffusion does: recurrent transitions are
        # safe because its scalar mass is normalized and hop-capped, and
        # excluding them would make the transition matrix depend on batching.
        excluded = set(exclude)
        if exclude_frontier:
            excluded.update(sources)
        placeholders = ",".join("?" for _ in sources)
        rows = self._db.execute(
            "SELECT e.source_chunk_id, e.destination_chunk_id, e.head_weights, "
            "e.qk_score, "
            "e.ov_transport, e.evidence_count, e.traversal_count, "
            "e.last_access_turn, e.temporal_forward "
            "FROM chunk_head_edges AS e "
            "JOIN chunks AS c ON c.chunk_id = e.destination_chunk_id "
            f"WHERE e.source_chunk_id IN ({placeholders}) AND e.artifact_id = ? "
            "AND c.embedding IS NOT NULL AND c.hnsw_label IS NOT NULL",
            (*sources, artifact_id),
        ).fetchall()
        edges_by_source: dict[str, list[StoredHeadEdge]] = {
            source_id: [] for source_id in sources
        }
        for row in rows:
            source_id, destination_id = row[0], row[1]
            if destination_id in excluded:
                continue
            weights = self._unpack_f32(row[2])
            if len(weights) != artifact.head_count:
                raise ValueError("stored head weight width does not match its artifact")
            edges_by_source[source_id].append(
                StoredHeadEdge(
                    source_chunk_id=source_id,
                    destination_chunk_id=destination_id,
                    artifact_id=artifact_id,
                    head_weights=weights,
                    qk_score=float(row[3]),
                    ov_transport=float(row[4]),
                    evidence_count=int(row[5]),
                    traversal_count=int(row[6]),
                    last_access_turn=int(row[7]),
                    temporal_forward=None if row[8] is None else bool(row[8]),
                )
            )
        result: dict[str, tuple[StoredHeadEdge, ...]] = {}
        for source_id, edges in edges_by_source.items():
            edges.sort(
                key=lambda edge: (
                    edge.utility(
                        now_turn=turn,
                        usage_half_life=usage_half_life,
                        usage_weight=usage_weight,
                    ),
                    edge.destination_chunk_id,
                ),
                reverse=True,
            )
            result[source_id] = tuple(edges[:top_k_per_source])
        self._cache_put(self._edge_neighbor_cache, cache_key, result)
        return result

    def touch_edges(
        self,
        artifact_id: str,
        pairs: Sequence[tuple[str, str]],
        *,
        now_turn: int | None = None,
    ) -> int:
        self._require_artifact(artifact_id)
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        if turn < 0:
            raise ValueError("now_turn must be non-negative")
        touched = 0
        for source_id, destination_id in dict.fromkeys(pairs):
            cur = self._db.execute(
                "UPDATE chunk_head_edges SET traversal_count = traversal_count + 1, "
                "last_access_turn = ? WHERE source_chunk_id = ? "
                "AND destination_chunk_id = ? AND artifact_id = ?",
                (turn, source_id, destination_id, artifact_id),
            )
            touched += cur.rowcount
        self._db.commit()
        if touched:
            self._edge_neighbor_cache.clear()
        return touched

    def touch_signatures(
        self,
        artifact_id: str,
        chunk_ids: Sequence[str],
        *,
        now_turn: int | None = None,
    ) -> int:
        self._require_artifact(artifact_id)
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        touched = 0
        for chunk_id in dict.fromkeys(chunk_ids):
            cur = self._db.execute(
                "UPDATE chunk_cav_signatures SET access_count = access_count + 1, "
                "last_access_turn = ? WHERE chunk_id = ? AND artifact_id = ?",
                (turn, chunk_id, artifact_id),
            )
            touched += cur.rowcount
        self._db.commit()
        return touched

    # -- live same-turn Hebbian co-access graph ---------------------------

    @staticmethod
    def _decayed_mass(
        mass: float,
        last_turn: int,
        now_turn: int,
        half_life_turns: float,
    ) -> float:
        return max(
            0.0,
            float(mass)
            * decay_factor(last_turn, now_turn, half_life_turns),
        )

    @classmethod
    def _hebbian_edge_score(
        cls,
        *,
        coaccess_mass: float,
        last_reinforced_turn: int,
        left_mass: float,
        left_turn: int,
        right_mass: float,
        right_turn: int,
        now_turn: int,
        half_life_turns: float,
    ) -> float:
        """Time-decayed cosine association, which discounts frequent hubs."""
        edge = cls._decayed_mass(
            coaccess_mass,
            last_reinforced_turn,
            now_turn,
            half_life_turns,
        )
        left = cls._decayed_mass(
            left_mass,
            left_turn,
            now_turn,
            half_life_turns,
        )
        right = cls._decayed_mass(
            right_mass,
            right_turn,
            now_turn,
            half_life_turns,
        )
        denominator = math.sqrt(left * right)
        if denominator <= 0.0:
            return 0.0
        # With matching exponential updates the ratio is a cosine and
        # therefore at most one. A separate freshness term is intentional:
        # otherwise an isolated pair's node and edge masses decay in lockstep
        # and its normalized score never cools.
        normalized = min(1.0, max(0.0, edge / denominator))
        freshness = decay_factor(
            last_reinforced_turn,
            now_turn,
            half_life_turns,
        )
        return normalized * freshness

    def reinforce_retrieval_coaccess(
        self,
        artifact_id: str,
        access_event_id: str,
        concept_activations: Mapping[str, float],
        *,
        now_turn: int | None = None,
        learning_rate: float = 1.0,
        half_life_turns: float = 200.0,
        max_concepts_per_event: int = 12,
        max_degree: int = 32,
        min_edge_score: float = 0.0,
        max_event_history: int = 4096,
    ) -> HebbianUpdate:
        """Learn which conceptual chunks were exposed in one retrieval turn.

        The update is a bounded, external Hebbian projection. Node mass stores
        decayed ``activation**2`` and edge mass stores decayed
        ``activation_i * activation_j``. Their normalized read score is a
        cosine-like association that suppresses ubiquitous hubs. The event
        receipt contains only a caller ID and SHA-256 fingerprint, making an
        exact retry idempotent without retaining query text or result payloads.
        """
        self._require_artifact(artifact_id)
        event_id = str(access_event_id).strip()
        if not event_id:
            raise ValueError("access_event_id must be non-empty")
        if len(event_id) > 256:
            raise ValueError("access_event_id must be at most 256 characters")
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        if turn < 0:
            raise ValueError("now_turn must be non-negative")
        rate = float(learning_rate)
        half_life = float(half_life_turns)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life_turns must be finite and positive")
        if max_concepts_per_event < 1:
            raise ValueError("max_concepts_per_event must be positive")
        if max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        if not 0.0 <= min_edge_score <= 1.0:
            raise ValueError("min_edge_score must lie in [0, 1]")
        if max_event_history < 1:
            raise ValueError("max_event_history must be positive")

        ranked: list[tuple[str, float]] = []
        for raw_chunk_id, raw_activation in concept_activations.items():
            chunk_id = str(raw_chunk_id).strip()
            activation = float(raw_activation)
            if not chunk_id:
                raise ValueError("concept chunk IDs must be non-empty")
            if not math.isfinite(activation) or not 0.0 <= activation <= 1.0:
                raise ValueError("concept activations must be finite and in [0, 1]")
            if activation > 0.0:
                ranked.append((chunk_id, activation))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        selected = ranked[:max_concepts_per_event]
        selected.sort(key=lambda item: item[0])
        fingerprint_payload = [
            [chunk_id, format(activation, ".17g")]
            for chunk_id, activation in selected
        ]
        fingerprint = hashlib.sha256(
            _canonical_json(fingerprint_payload).encode("utf-8")
        ).hexdigest()

        existing_event = self._db.execute(
            "SELECT event_fingerprint, member_count FROM hebbian_access_events "
            "WHERE artifact_id = ? AND event_id = ?",
            (artifact_id, event_id),
        ).fetchone()
        if existing_event is not None:
            if existing_event[0] != fingerprint:
                raise ValueError(
                    "access_event_id was already used with a different retrieval set"
                )
            return HebbianUpdate(
                event_id=event_id,
                created=False,
                concepts_observed=int(existing_event[1]),
                edges_reinforced=0,
                edges_pruned=0,
            )

        chunk_ids = [chunk_id for chunk_id, _ in selected]
        placeholders = ",".join("?" for _ in chunk_ids)
        existing_nodes: dict[str, tuple[float, int, int]] = {}
        if chunk_ids:
            rows = self._db.execute(
                "SELECT chunk_id, access_mass, access_count, last_access_turn "
                "FROM hebbian_chunk_nodes WHERE artifact_id = ? "
                f"AND chunk_id IN ({placeholders})",
                (artifact_id, *chunk_ids),
            ).fetchall()
            existing_nodes = {
                row[0]: (float(row[1]), int(row[2]), int(row[3])) for row in rows
            }

        pairs = [
            (left[0], right[0], left[1], right[1])
            for left, right in combinations(selected, 2)
        ]
        existing_edges: dict[tuple[str, str], tuple[float, int, int]] = {}
        if pairs:
            rows = self._db.execute(
                "SELECT chunk_low, chunk_high, coaccess_mass, coaccess_count, "
                "last_reinforced_turn FROM hebbian_chunk_edges "
                f"WHERE artifact_id = ? AND chunk_low IN ({placeholders}) "
                f"AND chunk_high IN ({placeholders})",
                (artifact_id, *chunk_ids, *chunk_ids),
            ).fetchall()
            existing_edges = {
                (row[0], row[1]): (float(row[2]), int(row[3]), int(row[4]))
                for row in rows
            }

        node_rows: list[tuple[Any, ...]] = []
        for chunk_id, activation in selected:
            old_mass, old_count, old_turn = existing_nodes.get(
                chunk_id, (0.0, 0, turn)
            )
            mass = self._decayed_mass(old_mass, old_turn, turn, half_life)
            mass += rate * activation * activation
            node_rows.append(
                (artifact_id, chunk_id, mass, old_count + 1, turn)
            )

        edge_rows: list[tuple[Any, ...]] = []
        for low, high, low_activation, high_activation in pairs:
            old_mass, old_count, old_turn = existing_edges.get(
                (low, high), (0.0, 0, turn)
            )
            mass = self._decayed_mass(old_mass, old_turn, turn, half_life)
            mass += rate * low_activation * high_activation
            edge_rows.append(
                (artifact_id, low, high, mass, old_count + 1, turn)
            )

        connection = self._db.connection
        with connection:
            connection.execute(
                "INSERT INTO hebbian_access_events "
                "(artifact_id, event_id, observed_turn, event_fingerprint, member_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (artifact_id, event_id, turn, fingerprint, len(selected)),
            )
            if node_rows:
                connection.executemany(
                    "INSERT INTO hebbian_chunk_nodes "
                    "(artifact_id, chunk_id, access_mass, access_count, "
                    "last_access_turn) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(artifact_id, chunk_id) DO UPDATE SET "
                    "access_mass = excluded.access_mass, "
                    "access_count = excluded.access_count, "
                    "last_access_turn = excluded.last_access_turn",
                    node_rows,
                )
            if edge_rows:
                connection.executemany(
                    "INSERT INTO hebbian_chunk_edges "
                    "(artifact_id, chunk_low, chunk_high, coaccess_mass, "
                    "coaccess_count, last_reinforced_turn) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(artifact_id, chunk_low, chunk_high) DO UPDATE SET "
                    "coaccess_mass = excluded.coaccess_mass, "
                    "coaccess_count = excluded.coaccess_count, "
                    "last_reinforced_turn = excluded.last_reinforced_turn",
                    edge_rows,
                )

            old_receipts = connection.execute(
                "SELECT event_id FROM hebbian_access_events "
                "WHERE artifact_id = ? ORDER BY observed_turn DESC, rowid DESC "
                "LIMIT -1 OFFSET ?",
                (artifact_id, max_event_history),
            ).fetchall()
            if old_receipts:
                connection.executemany(
                    "DELETE FROM hebbian_access_events "
                    "WHERE artifact_id = ? AND event_id = ?",
                    [(artifact_id, row[0]) for row in old_receipts],
                )

        edges_pruned = self.prune_hebbian_edges(
            artifact_id,
            max_degree=max_degree,
            min_score=min_edge_score,
            chunk_ids=chunk_ids,
            now_turn=turn,
            half_life_turns=half_life,
        )
        return HebbianUpdate(
            event_id=event_id,
            created=True,
            concepts_observed=len(selected),
            edges_reinforced=len(edge_rows),
            edges_pruned=edges_pruned,
        )

    def hebbian_neighbors(
        self,
        concept_activations: Mapping[str, float],
        artifact_id: str,
        *,
        top_k: int,
        exclude: Sequence[str] = (),
        now_turn: int | None = None,
        half_life_turns: float = 200.0,
        min_score: float = 0.0,
    ) -> tuple[StoredHebbianNeighbor, ...]:
        """Recall conceptual chunks associated by prior same-turn exposure."""
        self._require_artifact(artifact_id)
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return ()
        half_life = float(half_life_turns)
        if not math.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life_turns must be finite and positive")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must lie in [0, 1]")
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        if turn < 0:
            raise ValueError("now_turn must be non-negative")

        seeds: dict[str, float] = {}
        for raw_chunk_id, raw_activation in concept_activations.items():
            chunk_id = str(raw_chunk_id).strip()
            activation = float(raw_activation)
            if not chunk_id:
                raise ValueError("concept chunk IDs must be non-empty")
            if not math.isfinite(activation) or not 0.0 <= activation <= 1.0:
                raise ValueError("concept activations must be finite and in [0, 1]")
            if activation > 0.0:
                seeds[chunk_id] = max(seeds.get(chunk_id, 0.0), activation)
        if not seeds:
            return ()

        seed_ids = list(seeds)
        placeholders = ",".join("?" for _ in seed_ids)
        edge_rows = self._db.execute(
            "SELECT chunk_low, chunk_high, coaccess_mass, coaccess_count, "
            "last_reinforced_turn FROM hebbian_chunk_edges "
            f"WHERE artifact_id = ? AND (chunk_low IN ({placeholders}) "
            f"OR chunk_high IN ({placeholders}))",
            (artifact_id, *seed_ids, *seed_ids),
        ).fetchall()
        if not edge_rows:
            return ()

        endpoint_ids = sorted(
            {row[0] for row in edge_rows} | {row[1] for row in edge_rows}
        )
        endpoint_placeholders = ",".join("?" for _ in endpoint_ids)
        node_rows = self._db.execute(
            "SELECT chunk_id, access_mass, last_access_turn "
            "FROM hebbian_chunk_nodes WHERE artifact_id = ? "
            f"AND chunk_id IN ({endpoint_placeholders})",
            (artifact_id, *endpoint_ids),
        ).fetchall()
        nodes = {
            row[0]: (float(row[1]), int(row[2])) for row in node_rows
        }
        excluded = set(exclude) | set(seed_ids)
        candidates: dict[str, dict[str, Any]] = {}
        for low, high, mass, count, edge_turn in edge_rows:
            if low in seeds and high not in seeds:
                anchor_id, candidate_id = low, high
            elif high in seeds and low not in seeds:
                anchor_id, candidate_id = high, low
            else:
                continue
            if candidate_id in excluded:
                continue
            left = nodes.get(low, (0.0, turn))
            right = nodes.get(high, (0.0, turn))
            edge_score = self._hebbian_edge_score(
                coaccess_mass=float(mass),
                last_reinforced_turn=int(edge_turn),
                left_mass=left[0],
                left_turn=left[1],
                right_mass=right[0],
                right_turn=right[1],
                now_turn=turn,
                half_life_turns=half_life,
            )
            evidence = min(1.0, edge_score * seeds[anchor_id])
            if evidence < min_score:
                continue
            current = candidates.setdefault(
                candidate_id,
                {
                    "score": 0.0,
                    "anchors": set(),
                    "anchor_chunk_id": anchor_id,
                    "best_evidence": -1.0,
                    "coaccess_count": 0,
                    "last_reinforced_turn": 0,
                },
            )
            # Noisy-OR combines support from several anchors without allowing
            # a high-degree candidate to gain an unbounded additive score.
            current["score"] = 1.0 - (1.0 - current["score"]) * (1.0 - evidence)
            current["anchors"].add(anchor_id)
            current["coaccess_count"] += int(count)
            current["last_reinforced_turn"] = max(
                current["last_reinforced_turn"], int(edge_turn)
            )
            if evidence > current["best_evidence"]:
                current["best_evidence"] = evidence
                current["anchor_chunk_id"] = anchor_id

        neighbors = [
            StoredHebbianNeighbor(
                chunk_id=chunk_id,
                score=float(state["score"]),
                support=len(state["anchors"]),
                anchor_chunk_id=str(state["anchor_chunk_id"]),
                coaccess_count=int(state["coaccess_count"]),
                last_reinforced_turn=int(state["last_reinforced_turn"]),
            )
            for chunk_id, state in candidates.items()
        ]
        neighbors.sort(key=lambda item: (-item.score, -item.support, item.chunk_id))
        return tuple(neighbors[:top_k])

    def prune_hebbian_edges(
        self,
        artifact_id: str,
        max_degree: int,
        *,
        min_score: float = 0.0,
        chunk_ids: Sequence[str] | None = None,
        now_turn: int | None = None,
        half_life_turns: float = 200.0,
    ) -> int:
        """Enforce an undirected degree cap and remove weak co-access links."""
        self._require_artifact(artifact_id)
        if max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must lie in [0, 1]")
        half_life = float(half_life_turns)
        if not math.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life_turns must be finite and positive")
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        if turn < 0:
            raise ValueError("now_turn must be non-negative")

        scoped_ids = list(dict.fromkeys(chunk_ids or ()))
        if chunk_ids is not None and not scoped_ids:
            return 0
        if chunk_ids is None:
            edge_rows = self._db.execute(
                "SELECT chunk_low, chunk_high, coaccess_mass, coaccess_count, "
                "last_reinforced_turn FROM hebbian_chunk_edges "
                "WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchall()
        else:
            placeholders = ",".join("?" for _ in scoped_ids)
            edge_rows = self._db.execute(
                "SELECT chunk_low, chunk_high, coaccess_mass, coaccess_count, "
                "last_reinforced_turn FROM hebbian_chunk_edges "
                f"WHERE artifact_id = ? AND (chunk_low IN ({placeholders}) "
                f"OR chunk_high IN ({placeholders}))",
                (artifact_id, *scoped_ids, *scoped_ids),
            ).fetchall()
        if not edge_rows:
            return 0

        endpoint_ids = sorted(
            {row[0] for row in edge_rows} | {row[1] for row in edge_rows}
        )
        placeholders = ",".join("?" for _ in endpoint_ids)
        node_rows = self._db.execute(
            "SELECT chunk_id, access_mass, last_access_turn "
            "FROM hebbian_chunk_nodes WHERE artifact_id = ? "
            f"AND chunk_id IN ({placeholders})",
            (artifact_id, *endpoint_ids),
        ).fetchall()
        nodes = {row[0]: (float(row[1]), int(row[2])) for row in node_rows}
        scored: dict[tuple[str, str], float] = {}
        for low, high, mass, _count, edge_turn in edge_rows:
            left = nodes.get(low, (0.0, turn))
            right = nodes.get(high, (0.0, turn))
            scored[(low, high)] = self._hebbian_edge_score(
                coaccess_mass=float(mass),
                last_reinforced_turn=int(edge_turn),
                left_mass=left[0],
                left_turn=left[1],
                right_mass=right[0],
                right_turn=right[1],
                now_turn=turn,
                half_life_turns=half_life,
            )

        deletions = {
            edge for edge, score in scored.items() if score < min_score
        }
        scoped = set(endpoint_ids) if chunk_ids is None else set(scoped_ids)
        for chunk_id in scoped:
            incident = [
                (score, edge)
                for edge, score in scored.items()
                if chunk_id in edge and edge not in deletions
            ]
            incident.sort(key=lambda item: (-item[0], item[1]))
            deletions.update(edge for _score, edge in incident[max_degree:])
        if not deletions:
            return 0
        cur = self._db.executemany(
            "DELETE FROM hebbian_chunk_edges WHERE artifact_id = ? "
            "AND chunk_low = ? AND chunk_high = ?",
            [(artifact_id, low, high) for low, high in sorted(deletions)],
        )
        self._db.commit()
        return (
            cur.rowcount
            if cur.rowcount is not None and cur.rowcount >= 0
            else len(deletions)
        )

    def hebbian_stats(self, artifact_id: str) -> dict[str, int]:
        """Compact live-graph counts, including the zero-token-state invariant."""
        self._require_artifact(artifact_id)
        counts = {}
        for name, table in (
            ("nodes", "hebbian_chunk_nodes"),
            ("edges", "hebbian_chunk_edges"),
            ("event_receipts", "hebbian_access_events"),
        ):
            counts[name] = int(
                self._db.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()[0]
            )
        counts["retained_token_state_bytes"] = 0
        return counts

    def prune_edges(
        self,
        artifact_id: str,
        max_neighbors: int,
        *,
        source_chunk_ids: Sequence[str] | None = None,
        now_turn: int | None = None,
        usage_half_life: float = 100.0,
        usage_weight: float = 0.05,
    ) -> int:
        """Enforce a hard per-source degree cap using live edge utility."""
        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        self._require_artifact(artifact_id)
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        select_columns = (
            "SELECT source_chunk_id, destination_chunk_id, head_weights, qk_score, "
            "ov_transport, evidence_count, traversal_count, last_access_turn, "
            "temporal_forward FROM chunk_head_edges "
        )
        rows: list = []
        if source_chunk_ids is None:
            rows = self._db.execute(
                select_columns + "WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchall()
        else:
            # Keep below SQLite's conservative host-parameter limit while
            # avoiding one SELECT per source.
            sources = list(dict.fromkeys(source_chunk_ids))
            for offset in range(0, len(sources), 500):
                batch = sources[offset : offset + 500]
                if not batch:
                    continue
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    self._db.execute(
                        select_columns
                        + f"WHERE artifact_id = ? AND source_chunk_id IN ({placeholders})",
                        (artifact_id, *batch),
                    ).fetchall()
                )

        rows_by_source: dict[str, list] = defaultdict(list)
        for row in rows:
            rows_by_source[row[0]].append(row)
        deletions: list[tuple[str, str, str]] = []
        for source_id, source_rows in rows_by_source.items():
            ranked: list[tuple[float, str]] = []
            for row in source_rows:
                edge = StoredHeadEdge(
                    source_chunk_id=source_id,
                    destination_chunk_id=row[1],
                    artifact_id=artifact_id,
                    head_weights=self._unpack_f32(row[2]),
                    qk_score=float(row[3]),
                    ov_transport=float(row[4]),
                    evidence_count=int(row[5]),
                    traversal_count=int(row[6]),
                    last_access_turn=int(row[7]),
                    temporal_forward=None if row[8] is None else bool(row[8]),
                )
                ranked.append(
                    (
                        edge.utility(
                            now_turn=turn,
                            usage_half_life=usage_half_life,
                            usage_weight=usage_weight,
                        ),
                        edge.destination_chunk_id,
                    )
                )
            ranked.sort(reverse=True)
            for _, destination_id in ranked[max_neighbors:]:
                deletions.append((source_id, destination_id, artifact_id))
        removed = 0
        if deletions:
            cur = self._db.executemany(
                "DELETE FROM chunk_head_edges WHERE source_chunk_id = ? "
                "AND destination_chunk_id = ? AND artifact_id = ?",
                deletions,
            )
            removed = (
                cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0
                else len(deletions)
            )
        self._db.commit()
        if removed:
            self._edge_neighbor_cache.clear()
        return removed

    def remove_chunk_artifacts(self, chunk_id: str) -> int:
        """Remove a chunk from the live graph while preserving its source row."""
        hebbian_edge_cur = self._db.execute(
            "DELETE FROM hebbian_chunk_edges WHERE chunk_low = ? OR chunk_high = ?",
            (chunk_id, chunk_id),
        )
        hebbian_node_cur = self._db.execute(
            "DELETE FROM hebbian_chunk_nodes WHERE chunk_id = ?",
            (chunk_id,),
        )
        edge_cur = self._db.execute(
            "DELETE FROM chunk_head_edges WHERE source_chunk_id = ? "
            "OR destination_chunk_id = ?",
            (chunk_id, chunk_id),
        )
        signature_cur = self._db.execute(
            "DELETE FROM chunk_cav_signatures WHERE chunk_id = ?",
            (chunk_id,),
        )
        self._db.commit()
        if edge_cur.rowcount:
            self._edge_neighbor_cache.clear()
        if signature_cur.rowcount:
            self._cav_neighbor_cache.clear()
        return (
            edge_cur.rowcount
            + signature_cur.rowcount
            + hebbian_edge_cur.rowcount
            + hebbian_node_cur.rowcount
        )

    def stats(self, artifact_id: str) -> dict[str, int]:
        self._require_artifact(artifact_id)
        signatures = int(
            self._db.execute(
                "SELECT COUNT(*) FROM chunk_cav_signatures WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()[0]
        )
        edges = int(
            self._db.execute(
                "SELECT COUNT(*) FROM chunk_head_edges WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()[0]
        )
        cav_payload_bytes = int(
            self._db.execute(
                "SELECT COALESCE(SUM(LENGTH(signature)), 0) "
                "FROM chunk_cav_signatures WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()[0]
        )
        head_payload_bytes = int(
            self._db.execute(
                "SELECT COALESCE(SUM(LENGTH(head_weights)), 0) "
                "FROM chunk_head_edges WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()[0]
        )
        return {
            "signatures": signatures,
            "edges": edges,
            "cav_payload_bytes": cav_payload_bytes,
            "head_payload_bytes": head_payload_bytes,
            # An explicit invariant useful in reports and backend parity tests.
            "retained_token_state_bytes": 0,
        }
