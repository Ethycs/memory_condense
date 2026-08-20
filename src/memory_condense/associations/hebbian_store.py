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
from memory_condense.associations.coaccess_graph import (
    accumulate_neighbor_evidence,
    decayed_mass,
    edge_endpoint_keys,
    positive_seed_activations,
    ranked_neighbor_states,
    score_coaccess_edges,
    select_prune_victims,
    validate_observation_params,
    validated_recall_params,
)


class HebbianAssociationStoreMixin:
    """Decay-aware co-access reinforcement, traversal, and pruning."""

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
        event_id, rate, half_life = validate_observation_params(
            access_event_id=access_event_id,
            learning_rate=learning_rate,
            half_life_turns=half_life_turns,
            max_members_per_event=max_concepts_per_event,
            max_degree=max_degree,
            min_edge_score=min_edge_score,
            max_event_history=max_event_history,
            member_limit_name="max_concepts_per_event",
        )
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        if turn < 0:
            raise ValueError("now_turn must be non-negative")

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
                members_observed=int(existing_event[1]),
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
            mass = decayed_mass(old_mass, old_turn, turn, half_life)
            mass += rate * activation * activation
            node_rows.append(
                (artifact_id, chunk_id, mass, old_count + 1, turn)
            )

        edge_rows: list[tuple[Any, ...]] = []
        for low, high, low_activation, high_activation in pairs:
            old_mass, old_count, old_turn = existing_edges.get(
                (low, high), (0.0, 0, turn)
            )
            mass = decayed_mass(old_mass, old_turn, turn, half_life)
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
            members_observed=len(selected),
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
        window = validated_recall_params(
            top_k=top_k,
            min_score=min_score,
            half_life_turns=half_life_turns,
            now_turn=now_turn,
            current_turn=self._db.current_turn,
        )
        if window is None:
            return ()
        half_life, turn = window

        def seed_key(raw_chunk_id: str) -> str:
            chunk_id = str(raw_chunk_id).strip()
            if not chunk_id:
                raise ValueError("concept chunk IDs must be non-empty")
            return chunk_id

        seeds = positive_seed_activations(
            (
                (seed_key(chunk_id), activation)
                for chunk_id, activation in concept_activations.items()
            ),
            what="concept activations",
        )
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

        endpoint_ids = edge_endpoint_keys(edge_rows)
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
        candidates = accumulate_neighbor_evidence(
            ((low, high, mass, count, 0, edge_turn)
             for low, high, mass, count, edge_turn in edge_rows),
            seeds=seeds,
            excluded=set(exclude) | set(seed_ids),
            nodes=nodes,
            default_node=(0.0, turn),
            now_turn=turn,
            half_life_turns=half_life,
            min_score=min_score,
        )
        neighbors = [
            StoredHebbianNeighbor(
                chunk_id=chunk_id,
                score=float(state.score),
                support=state.support,
                anchor_chunk_id=str(state.anchor_key),
                coaccess_count=int(state.coaccess_count),
                last_reinforced_turn=int(state.last_reinforced_turn),
            )
            for chunk_id, state in ranked_neighbor_states(candidates)
        ]
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
        scored = score_coaccess_edges(
            ((low, high, mass, edge_turn)
             for low, high, mass, _count, edge_turn in edge_rows),
            nodes,
            now_turn=turn,
            half_life_turns=half_life,
        )
        deletions = select_prune_victims(
            scored,
            set(endpoint_ids) if chunk_ids is None else set(scoped_ids),
            max_degree=max_degree,
            min_score=min_score,
        )
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
