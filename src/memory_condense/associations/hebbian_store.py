"""Bounded same-turn Hebbian co-access persistence and retrieval."""

from __future__ import annotations

import hashlib
import math
from itertools import combinations
from typing import Any, Mapping, Sequence

from memory_condense.associations.association_models import (
    HebbianUpdate,
    StoredHebbianNeighbor,
    _canonical_json,
)
from memory_condense.domain.decay import decay_factor


class HebbianAssociationStoreMixin:
    """Decay-aware co-access reinforcement, traversal, and pruning."""

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
        """Compact graph counts and the zero request-token-state invariant.

        The metric covers request-derived token IDs, Q/K/V, attention maps,
        residual streams, and generation K/V caches retained across requests.
        It intentionally does not count reusable static model weights or a
        tokenizer, neither of which is memory derived from a request.
        """
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
        counts["retained_request_token_state_bytes"] = 0
        # Backward-compatible spelling used by historical reports.  Its scope
        # is identical to retained_request_token_state_bytes; it never meant
        # static checkpoint weights or tokenizer assets.
        counts["retained_token_state_bytes"] = 0
        return counts
