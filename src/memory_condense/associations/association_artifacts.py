"""Artifact identity and compact CAV-signature persistence."""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np

from memory_condense.associations.association_models import (
    AssociationArtifact,
    StoredCAVNeighbor,
    StoredCAVSignature,
    _canonical_json,
)


class AssociationArtifactStoreMixin:
    """Artifact registration, signatures, and concept-neighbor queries."""

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

    def put_signatures(
        self,
        artifact_id: str,
        signatures: Sequence[tuple[str, Sequence[float]]],
        *,
        created_turn: int | None = None,
    ) -> int:
        """Store a batch of fixed-width signatures in one transaction."""

        artifact = self._require_artifact(artifact_id)
        if artifact.cav_layer is None or not artifact.concept_names:
            raise ValueError("artifact does not define a CAV coordinate system")
        turn = self._db.current_turn() if created_turn is None else int(created_turn)
        if turn < 0:
            raise ValueError("created_turn must be non-negative")
        rows = [
            (
                str(chunk_id),
                artifact_id,
                self._pack_f32(
                    values,
                    width=len(artifact.concept_names),
                    field_name="CAV signature",
                ),
                turn,
            )
            for chunk_id, values in signatures
        ]
        if not rows:
            return 0
        self._db.executemany(
            "INSERT INTO chunk_cav_signatures "
            "(chunk_id, artifact_id, signature, created_turn) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(chunk_id, artifact_id) DO UPDATE SET "
            "signature = excluded.signature, created_turn = excluded.created_turn",
            rows,
        )
        self._db.commit()
        self._cav_neighbor_cache.clear()
        return len(rows)

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

    def concept_members(
        self,
        artifact_id: str,
        concept_name: str,
        *,
        top_k: int,
        min_margin: float = 0.0,
        source_ids: Sequence[str] = (),
        exclude: Sequence[str] = (),
        unique_sources: bool = False,
    ) -> tuple[StoredCAVNeighbor, ...]:
        """Return strongest positive members of one named concept.

        This is an inverted concept lookup over compact float coordinates.
        Optional durable-source filtering makes it suitable after coarse
        partition routing without scanning or hydrating source text.
        """

        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return ()
        artifact = self._require_artifact(artifact_id)
        try:
            concept_index = artifact.concept_names.index(concept_name)
        except ValueError as exc:
            raise KeyError(f"unknown CAV concept: {concept_name}") from exc
        params: list[Any] = [artifact_id]
        where = [
            "s.artifact_id = ?",
            "c.embedding IS NOT NULL",
            "c.hnsw_label IS NOT NULL",
        ]
        filtered_source_ids = tuple(
            dict.fromkeys(str(value) for value in source_ids)
        )
        if filtered_source_ids:
            placeholders = ",".join("?" for _ in filtered_source_ids)
            where.append(f"COALESCE(t.source_id, t.turn_id) IN ({placeholders})")
            params.extend(filtered_source_ids)
        rows = self._db.execute(
            "SELECT s.chunk_id, s.signature, COALESCE(t.source_id, t.turn_id) "
            "FROM chunk_cav_signatures AS s "
            "JOIN chunks AS c ON c.chunk_id = s.chunk_id "
            "JOIN turns AS t ON t.turn_id = c.turn_id WHERE "
            + " AND ".join(where),
            tuple(params),
        ).fetchall()
        excluded = set(exclude)
        ranked: list[StoredCAVNeighbor] = []
        for chunk_id, blob, source_id in rows:
            if chunk_id in excluded:
                continue
            signature = self._unpack_f32(blob)
            if len(signature) != len(artifact.concept_names):
                raise ValueError(
                    "stored CAV signature width does not match its artifact"
                )
            margin = float(signature[concept_index])
            if margin <= min_margin:
                continue
            ranked.append(
                StoredCAVNeighbor(
                    chunk_id=chunk_id,
                    score=margin,
                    shared_concepts=(concept_name,),
                    source_id=str(source_id),
                )
            )
        ranked.sort(key=lambda hit: (hit.score, hit.chunk_id), reverse=True)
        if unique_sources:
            seen_sources: set[str] = set()
            source_unique: list[StoredCAVNeighbor] = []
            for hit in ranked:
                source_id = str(hit.source_id)
                if source_id in seen_sources:
                    continue
                seen_sources.add(source_id)
                source_unique.append(hit)
            ranked = source_unique
        return tuple(ranked[:top_k])

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
