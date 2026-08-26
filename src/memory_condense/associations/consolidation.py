"""Prompt-driven, bounded consolidation across durable memory partitions.

Consolidation here is not an offline summary pass.  A later prompt activates
already durable semantic memories and source chunks.  When several *directly
retrieved* nodes actually survive context packing together, their scalar
co-activation statistics are strengthened.  Associations that stop recurring
decay in conversation-turn space and weak/high-degree edges are pruned.

The graph deliberately stores no text, query, token, embedding, attention,
residual, or K/V state.  It is an index over the authoritative ``memory_items``
and ``chunks`` partitions, never another memory payload.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

from memory_condense.associations.association_models import (
    CoaccessUpdate,
    _canonical_json,
)
from memory_condense.associations.coaccess_graph import (
    accumulate_neighbor_evidence,
    decayed_mass,
    edge_endpoint_keys,
    positive_seed_activations,
    rank_discount,
    ranked_neighbor_states,
    score_coaccess_edges,
    select_prune_victims,
    validate_observation_params,
    validated_recall_params,
)
from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
    MemoryLinkHit,
    MemoryLinkResult,
)
from memory_condense.domain import decay, ranking
from memory_condense.persistence.db import Database
from memory_condense.domain.schemas import MemoryItem, MemoryResult, MemoryStatus, RetrievalResult


class ConsolidationNodeKind(str, Enum):
    """Durable partitions that may participate in one memory assembly."""

    MEMORY = "memory"
    CHUNK = "chunk"


@dataclass(frozen=True, slots=True)
class ConsolidationNode:
    """A compact typed pointer; its payload remains in the authoritative store."""

    kind: ConsolidationNodeKind
    item_id: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("consolidation node item_id must be non-empty")

    @property
    def key(self) -> str:
        prefix = "m" if self.kind is ConsolidationNodeKind.MEMORY else "c"
        return f"{prefix}:{self.item_id}"

    @classmethod
    def memory(cls, mem_id: str) -> "ConsolidationNode":
        return cls(ConsolidationNodeKind.MEMORY, mem_id)

    @classmethod
    def chunk(cls, chunk_id: str) -> "ConsolidationNode":
        return cls(ConsolidationNodeKind.CHUNK, chunk_id)

    @classmethod
    def from_key(cls, node_key: str) -> "ConsolidationNode":
        prefix, separator, item_id = str(node_key).partition(":")
        if not separator or not item_id:
            raise ValueError("invalid consolidation node key")
        if prefix == "m":
            return cls.memory(item_id)
        if prefix == "c":
            return cls.chunk(item_id)
        raise ValueError("invalid consolidation node kind prefix")


#: Backward-compatible name for the consolidation store's update result.
ConsolidationUpdate = CoaccessUpdate


@dataclass(frozen=True, slots=True)
class ConsolidationNeighbor:
    """One candidate reached through decayed cross-partition co-activation."""

    node: ConsolidationNode
    score: float
    support: int
    anchor: ConsolidationNode
    coactivation_count: int
    causal_count: int
    last_reinforced_turn: int


def context_activations(
    memory_ids: Sequence[str],
    chunk_ids: Sequence[str],
    *,
    max_nodes: int = 16,
) -> dict[ConsolidationNode, float]:
    """Turn packed result ranks into bounded, partition-balanced activity.

    Ranks restart inside each partition.  Otherwise a long memory header would
    make every evidence chunk artificially weaker merely because it is rendered
    later in the prompt.  The final global cap is deterministic.
    """

    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    ranked: dict[ConsolidationNode, float] = {}
    for ids, constructor in (
        (memory_ids, ConsolidationNode.memory),
        (chunk_ids, ConsolidationNode.chunk),
    ):
        rank = 0
        for item_id in dict.fromkeys(str(value) for value in ids):
            if not item_id.strip():
                continue
            rank += 1
            ranked[constructor(item_id)] = rank_discount(rank)
    selected = sorted(ranked.items(), key=lambda item: (-item[1], item[0].key))[
        :max_nodes
    ]
    return dict(selected)


def qwen_head_activations(
    hits: Sequence[MemoryLinkHit],
    *,
    qk_weight: float = 0.7,
    ov_weight: float = 0.3,
) -> dict[ConsolidationNode, float]:
    """Convert one transient Qwen turn inspection into bounded node activity.

    ``QwenMemoryLinker`` should receive consolidation node keys as candidate
    episode IDs.  QK and ``log1p(OV)`` are normalized within that bounded turn
    workspace before blending, because their raw scales are checkpoint- and
    layer-dependent.  The returned mapping is suitable for
    :meth:`LiveConsolidationStore.observe`; neither hits nor activations need to
    survive after that update.
    """

    qk_mix = float(qk_weight)
    ov_mix = float(ov_weight)
    if not math.isfinite(qk_mix) or not math.isfinite(ov_mix):
        raise ValueError("QK/OV weights must be finite")
    if qk_mix < 0.0 or ov_mix < 0.0 or qk_mix + ov_mix <= 0.0:
        raise ValueError("QK/OV weights must be non-negative with a positive sum")
    parsed: list[tuple[ConsolidationNode, float, float]] = []
    for hit in hits:
        qk = max(0.0, float(hit.qk_score))
        ov = math.log1p(max(0.0, float(hit.ov_transport)))
        if not math.isfinite(qk) or not math.isfinite(ov):
            raise ValueError("QK/OV hit values must be finite")
        parsed.append((ConsolidationNode.from_key(hit.episode_id), qk, ov))
    if not parsed:
        return {}
    max_qk = max((item[1] for item in parsed), default=0.0)
    max_ov = max((item[2] for item in parsed), default=0.0)
    denominator = qk_mix + ov_mix
    activations: dict[ConsolidationNode, float] = {}
    for node, qk, ov in parsed:
        qk_signal = qk / max_qk if max_qk > 0.0 else 0.0
        ov_signal = ov / max_ov if max_ov > 0.0 else 0.0
        activation = (qk_mix * qk_signal + ov_mix * ov_signal) / denominator
        if activation > 0.0:
            activations[node] = max(activations.get(node, 0.0), activation)
    return activations


def inspect_qwen_context_hyperplane(
    # Duck-typed: anything exposing ``QwenMemoryLinker.link(source_text,
    # candidates, top_k=)`` and, optionally, a ``max_candidates`` workspace cap.
    linker: Any,
    user_text: str,
    memories: Sequence[MemoryItem],
    chunks: Sequence[RetrievalResult],
    *,
    max_nodes: int = 16,
) -> tuple[object, dict[ConsolidationNode, float]]:
    """Run one bounded Qwen inspection over a packed context's direct members.

    Candidate IDs are consolidation node keys, so the returned head hits map
    back to durable partitions without retaining their text.  The linker's own
    candidate and workspace caps remain authoritative.
    """

    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    candidates = [
        AssociativeMemoryCandidate(
            episode_id=ConsolidationNode.memory(item.mem_id).key,
            text=(
                item.content
                if not item.details
                else f"{item.content} ({item.details})"
            ),
            metadata={"partition": "memory"},
        )
        for item in memories
    ]
    candidates.extend(
        AssociativeMemoryCandidate(
            episode_id=ConsolidationNode.chunk(result.chunk.chunk_id).key,
            text=result.chunk.text,
            metadata={"partition": "chunk"},
        )
        for result in chunks
    )
    candidates = candidates[:max_nodes]
    if not candidates:
        raise ValueError("Qwen consolidation needs at least one packed direct member")
    # A resident Qwen linker deliberately caps each token workspace. Cover a
    # larger packed turn through fresh groups and merge only scalar hit data;
    # no activation, residual, or K/V state crosses between passes.
    group_size = max(1, int(getattr(linker, "max_candidates", len(candidates))))
    results = [
        linker.link(
            user_text,
            candidates[start : start + group_size],
            top_k=min(group_size, len(candidates) - start),
        )
        for start in range(0, len(candidates), group_size)
    ]
    hits = [hit for result in results for hit in getattr(result, "hits", ())]
    if any(getattr(result, "hits", None) is None for result in results):
        raise TypeError("Qwen consolidation linker result must expose hits")
    result = results[0] if len(results) == 1 else MemoryLinkResult.merge(results)
    return result, qwen_head_activations(hits)


#: Optional per-pair gates supplied by a transient head/CAV inspection.
PairAffinities = Mapping[tuple[ConsolidationNode, ConsolidationNode], float]


def _selected_activations(
    activations: Mapping[ConsolidationNode, float],
    *,
    max_nodes_per_event: int,
) -> list[tuple[ConsolidationNode, float]]:
    """Decide which activated nodes one event is allowed to reinforce.

    Only strictly positive activations participate.  Candidates are ranked by
    descending activation with the node key breaking ties, so the cap keeps a
    deterministic top slice; the survivors are then re-sorted by key so pair
    enumeration, the fingerprint, and the written rows never depend on the
    caller's mapping order.
    """

    ranked: list[tuple[ConsolidationNode, float]] = []
    for node, raw_activation in activations.items():
        if not isinstance(node, ConsolidationNode):
            raise TypeError("activation keys must be ConsolidationNode values")
        activation = float(raw_activation)
        if not math.isfinite(activation) or not 0.0 <= activation <= 1.0:
            raise ValueError("node activations must be finite and in [0, 1]")
        if activation > 0.0:
            ranked.append((node, activation))
    ranked.sort(key=lambda item: (-item[1], item[0].key))
    selected = ranked[:max_nodes_per_event]
    selected.sort(key=lambda item: item[0].key)
    return selected


def _gated_pair_affinities(
    pair_affinities: PairAffinities | None,
    selected_keys: set[str],
) -> dict[tuple[str, str], float]:
    """Normalize caller-supplied per-pair gates onto the selected node set.

    Each gate is re-keyed by its sorted endpoint keys, so ``(a, b)`` and
    ``(b, a)`` name one edge and the stronger of the two wins.  Self pairs and
    pairs touching a node that did not survive selection are dropped.  A gate
    of exactly ``0.0`` is kept here and suppresses its edge during row
    arithmetic, which is what lets an inspection veto a co-access.
    """

    affinities: dict[tuple[str, str], float] = {}
    for raw_pair, raw_affinity in (pair_affinities or {}).items():
        if len(raw_pair) != 2:
            raise ValueError("pair affinity keys must contain exactly two nodes")
        left, right = raw_pair
        if left == right:
            continue
        affinity = float(raw_affinity)
        if not math.isfinite(affinity) or not 0.0 <= affinity <= 1.0:
            raise ValueError("pair affinities must be finite and in [0, 1]")
        low, high = sorted((left.key, right.key))
        if low in selected_keys and high in selected_keys:
            affinities[(low, high)] = max(affinities.get((low, high), 0.0), affinity)
    return affinities


def _observation_fingerprint(
    selected: Sequence[tuple[ConsolidationNode, float]],
    affinities: Mapping[tuple[str, str], float],
    causal_keys: set[str],
) -> str:
    """Digest exactly what one event observed, so retries can be recognized.

    The payload is canonical JSON over the selected node keys with their
    activations, the sorted pair gates, and the sorted causal targets — never
    the query or the rendered context.  Every float is rendered with
    ``format(value, ".17g")``, the shortest form that round-trips a double, so
    an identical observation reproduces the digest bit for bit rather than
    drifting with repr formatting.  A reused ``access_event_id`` whose digest
    matches is a retry; one whose digest differs is a caller bug.
    """

    payload = {
        "nodes": [
            [node.key, format(activation, ".17g")]
            for node, activation in selected
        ],
        "pair_affinities": [
            [low, high, format(affinity, ".17g")]
            for (low, high), affinity in sorted(affinities.items())
        ],
        "causal_targets": sorted(causal_keys),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _node_rows(
    selected: Sequence[tuple[ConsolidationNode, float]],
    existing_nodes: Mapping[str, tuple[float, int, int]],
    *,
    turn: int,
    half_life_turns: float,
    learning_rate: float,
) -> list[tuple[object, ...]]:
    """Compute the upsert rows for the nodes this event touched.

    Each node's stored mass first decays from its own last access turn to
    ``turn``, then gains ``rate * activation**2``.  Squaring keeps a weakly
    activated node from accruing the durability of a strongly activated one.
    A node with no prior row starts at zero mass dated to ``turn``, so its
    first observation decays by nothing.
    """

    rows: list[tuple[object, ...]] = []
    for node, activation in selected:
        old_mass, old_count, old_turn = existing_nodes.get(node.key, (0.0, 0, turn))
        mass = decayed_mass(old_mass, old_turn, turn, half_life_turns)
        mass += learning_rate * activation * activation
        rows.append(
            (
                node.key,
                node.kind.value,
                node.item_id if node.kind is ConsolidationNodeKind.MEMORY else None,
                node.item_id if node.kind is ConsolidationNodeKind.CHUNK else None,
                mass,
                old_count + 1,
                turn,
            )
        )
    return rows


def _edge_rows(
    selected: Sequence[tuple[ConsolidationNode, float]],
    existing_edges: Mapping[tuple[str, str], tuple[float, int, int, int]],
    *,
    affinities: Mapping[tuple[str, str], float],
    causal_keys: set[str],
    turn: int,
    half_life_turns: float,
    learning_rate: float,
) -> list[tuple[object, ...]]:
    """Compute the upsert rows for every surviving pair of selected nodes.

    Each edge's stored mass decays to ``turn`` and then gains
    ``rate * left_activation * right_activation * affinity``.  Rank-only
    operation uses affinity ``1``.  A transient CAV/QK/OV inspection can
    instead gate each pair, turning the Qwen heads into the association
    teacher without persisting their workspace in this graph; a gate of
    ``0.0`` drops the pair outright.  ``causal_count`` advances only when an
    endpoint was newly produced by the completed interaction, which is what
    distinguishes it from incidental co-access.
    """

    rows: list[tuple[object, ...]] = []
    for (left, left_activation), (right, right_activation) in combinations(
        selected, 2
    ):
        affinity = affinities.get((left.key, right.key), 1.0)
        if affinity <= 0.0:
            continue
        old_mass, old_count, old_causal_count, old_turn = existing_edges.get(
            (left.key, right.key), (0.0, 0, 0, turn)
        )
        mass = decayed_mass(old_mass, old_turn, turn, half_life_turns)
        mass += learning_rate * left_activation * right_activation * affinity
        rows.append(
            (
                left.key,
                right.key,
                mass,
                old_count + 1,
                old_causal_count
                + int(left.key in causal_keys or right.key in causal_keys),
                turn,
            )
        )
    return rows


class LiveConsolidationStore:
    """SQLite-backed decaying association graph over memories and chunks."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def _existing_item_ids(self, query: str, item_ids: Sequence[str]) -> set[str]:
        """Return the subset of ``item_ids`` that ``query`` still admits.

        ``query`` carries a ``{placeholders}`` slot for its ``IN`` list.  An
        empty id list issues no statement, keeping a single-partition
        observation down to one lookup.
        """

        if not item_ids:
            return set()
        placeholders = ",".join("?" for _ in item_ids)
        return {
            str(row[0])
            for row in self._db.execute(
                query.format(placeholders=placeholders), tuple(item_ids)
            ).fetchall()
        }

    def _validate_nodes(self, nodes: Sequence[ConsolidationNode]) -> None:
        """Reject nodes that no longer point at active, retrievable state.

        One lookup per partition: a memory must still be ``active``, and a
        chunk must still carry both an embedding and an HNSW label, or the
        association would index something retrieval can never hand back.
        """

        item_ids: dict[ConsolidationNodeKind, list[str]] = {
            ConsolidationNodeKind.MEMORY: [],
            ConsolidationNodeKind.CHUNK: [],
        }
        for node in nodes:
            item_ids[node.kind].append(node.item_id)
        retrievable = {
            ConsolidationNodeKind.MEMORY: self._existing_item_ids(
                "SELECT mem_id FROM memory_items "
                "WHERE status = 'active' AND mem_id IN ({placeholders})",
                item_ids[ConsolidationNodeKind.MEMORY],
            ),
            ConsolidationNodeKind.CHUNK: self._existing_item_ids(
                "SELECT chunk_id FROM chunks WHERE embedding IS NOT NULL "
                "AND hnsw_label IS NOT NULL AND chunk_id IN ({placeholders})",
                item_ids[ConsolidationNodeKind.CHUNK],
            ),
        }
        missing = [
            node.key
            for node in nodes
            if node.item_id not in retrievable[node.kind]
        ]
        if missing:
            raise ValueError(
                "consolidation nodes must reference active retrievable state: "
                + ", ".join(missing)
            )

    def _fingerprint_idempotency(
        self, event_id: str, fingerprint: str
    ) -> ConsolidationUpdate | None:
        """Resolve a reused ``event_id`` against the fingerprint on record.

        Returns a ``created=False`` update when this exact context was already
        observed — the caller then writes nothing — and ``None`` when the event
        is new.  A known id carrying a different fingerprint is an id
        collision, not a retry, and applying it would double-count the earlier
        observation, so it raises instead.
        """

        recorded = self._db.execute(
            "SELECT event_fingerprint, member_count "
            "FROM consolidation_access_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if recorded is None:
            return None
        if recorded[0] != fingerprint:
            raise ValueError(
                "access_event_id was already used with a different context set"
            )
        return ConsolidationUpdate(
            event_id=event_id,
            created=False,
            members_observed=int(recorded[1]),
            edges_reinforced=0,
            edges_pruned=0,
        )

    def _existing_node_state(
        self, keys: Sequence[str]
    ) -> dict[str, tuple[float, int, int]]:
        """Read stored ``(mass, count, last turn)`` for the nodes being touched."""

        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        return {
            str(row[0]): (float(row[1]), int(row[2]), int(row[3]))
            for row in self._db.execute(
                "SELECT node_key, access_mass, access_count, last_access_turn "
                f"FROM consolidation_nodes WHERE node_key IN ({placeholders})",
                tuple(keys),
            ).fetchall()
        }

    def _existing_edge_state(
        self, keys: Sequence[str]
    ) -> dict[tuple[str, str], tuple[float, int, int, int]]:
        """Read stored state for edges already joining two of these nodes.

        Below two nodes there are no pairs to reinforce, so the lookup is
        skipped entirely rather than issued and discarded.
        """

        if len(keys) < 2:
            return {}
        placeholders = ",".join("?" for _ in keys)
        return {
            (str(row[0]), str(row[1])): (
                float(row[2]),
                int(row[3]),
                int(row[4]),
                int(row[5]),
            )
            for row in self._db.execute(
                "SELECT node_low, node_high, coactivation_mass, "
                "coactivation_count, causal_count, last_reinforced_turn "
                "FROM consolidation_edges "
                f"WHERE node_low IN ({placeholders}) "
                f"AND node_high IN ({placeholders})",
                (*keys, *keys),
            ).fetchall()
        }

    def _write_observation(
        self,
        *,
        event_id: str,
        turn: int,
        fingerprint: str,
        member_count: int,
        node_rows: Sequence[tuple[object, ...]],
        edge_rows: Sequence[tuple[object, ...]],
        max_event_history: int,
    ) -> None:
        """Commit the receipt, the node and edge upserts, and the receipt trim.

        All four land in one transaction: a receipt written without its rows
        would make the retry short-circuit and lose the update permanently.
        The trim keeps the newest ``max_event_history`` receipts — the graph
        an evicted receipt built survives, it just can no longer deduplicate a
        very late retry of that event.
        """

        connection = self._db.connection
        with connection:
            connection.execute(
                "INSERT INTO consolidation_access_events "
                "(event_id, observed_turn, event_fingerprint, member_count) "
                "VALUES (?, ?, ?, ?)",
                (event_id, turn, fingerprint, member_count),
            )
            if node_rows:
                connection.executemany(
                    "INSERT INTO consolidation_nodes "
                    "(node_key, node_kind, memory_id, chunk_id, access_mass, "
                    "access_count, last_access_turn) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(node_key) DO UPDATE SET "
                    "access_mass = excluded.access_mass, "
                    "access_count = excluded.access_count, "
                    "last_access_turn = excluded.last_access_turn",
                    node_rows,
                )
            if edge_rows:
                connection.executemany(
                    "INSERT INTO consolidation_edges "
                    "(node_low, node_high, coactivation_mass, "
                    "coactivation_count, causal_count, last_reinforced_turn) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(node_low, node_high) DO UPDATE SET "
                    "coactivation_mass = excluded.coactivation_mass, "
                    "coactivation_count = excluded.coactivation_count, "
                    "causal_count = excluded.causal_count, "
                    "last_reinforced_turn = excluded.last_reinforced_turn",
                    edge_rows,
                )
            old_receipts = connection.execute(
                "SELECT event_id FROM consolidation_access_events "
                "ORDER BY observed_turn DESC, rowid DESC LIMIT -1 OFFSET ?",
                (max_event_history,),
            ).fetchall()
            if old_receipts:
                connection.executemany(
                    "DELETE FROM consolidation_access_events WHERE event_id = ?",
                    old_receipts,
                )

    def observe(
        self,
        access_event_id: str,
        activations: Mapping[ConsolidationNode, float],
        *,
        pair_affinities: PairAffinities | None = None,
        causal_targets: Sequence[ConsolidationNode] = (),
        now_turn: int | None = None,
        learning_rate: float = 1.0,
        half_life_turns: float = 200.0,
        max_nodes_per_event: int = 16,
        max_degree: int = 32,
        min_edge_score: float = 0.0,
        max_event_history: int = 4096,
    ) -> ConsolidationUpdate:
        """Strengthen nodes exposed together in one bounded interaction.

        ``access_event_id`` makes exact retries idempotent.  It and a SHA-256
        fingerprint are the only event data retained; the query and rendered
        context are never written to this graph. Edges touching a
        ``causal_target`` record that the node was newly produced by the
        completed interaction, distinct from incidental co-access.

        The pass runs validate, select, fingerprint, write, prune; the
        arithmetic lives in module-level pure functions and this method owns
        only the database edges between them.
        """

        event_id, rate, half_life = validate_observation_params(
            access_event_id=access_event_id,
            learning_rate=learning_rate,
            half_life_turns=half_life_turns,
            max_members_per_event=max_nodes_per_event,
            max_degree=max_degree,
            min_edge_score=min_edge_score,
            max_event_history=max_event_history,
            member_limit_name="max_nodes_per_event",
        )
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        if turn < 0:
            raise ValueError("now_turn must be non-negative")

        selected = _selected_activations(
            activations, max_nodes_per_event=max_nodes_per_event
        )
        self._validate_nodes([node for node, _ in selected])
        selected_keys = {node.key for node, _ in selected}
        causal_keys = {
            node.key for node in causal_targets if node.key in selected_keys
        }
        affinities = _gated_pair_affinities(pair_affinities, selected_keys)

        fingerprint = _observation_fingerprint(selected, affinities, causal_keys)
        replay = self._fingerprint_idempotency(event_id, fingerprint)
        if replay is not None:
            return replay

        keys = [node.key for node, _ in selected]
        node_rows = _node_rows(
            selected,
            self._existing_node_state(keys),
            turn=turn,
            half_life_turns=half_life,
            learning_rate=rate,
        )
        edge_rows = _edge_rows(
            selected,
            self._existing_edge_state(keys),
            affinities=affinities,
            causal_keys=causal_keys,
            turn=turn,
            half_life_turns=half_life,
            learning_rate=rate,
        )
        self._write_observation(
            event_id=event_id,
            turn=turn,
            fingerprint=fingerprint,
            member_count=len(selected),
            node_rows=node_rows,
            edge_rows=edge_rows,
            max_event_history=max_event_history,
        )

        edges_pruned = self.prune_edges(
            max_degree=max_degree,
            min_score=min_edge_score,
            node_keys=keys,
            now_turn=turn,
            half_life_turns=half_life,
        )
        return ConsolidationUpdate(
            event_id=event_id,
            created=True,
            members_observed=len(selected),
            edges_reinforced=len(edge_rows),
            edges_pruned=edges_pruned,
        )

    @staticmethod
    def _node_from_row(row: Sequence[object]) -> ConsolidationNode:
        kind = ConsolidationNodeKind(str(row[1]))
        item_id = row[2] if kind is ConsolidationNodeKind.MEMORY else row[3]
        return ConsolidationNode(kind, str(item_id))

    def neighbors(
        self,
        activations: Mapping[ConsolidationNode, float],
        *,
        top_k: int,
        target_kind: ConsolidationNodeKind | None = None,
        exclude: Sequence[ConsolidationNode] = (),
        now_turn: int | None = None,
        half_life_turns: float = 200.0,
        min_score: float = 0.05,
        min_coactivation_count: int = 2,
        min_causal_count: int = 1,
    ) -> tuple[ConsolidationNeighbor, ...]:
        """Recall nodes supported by repetition or a completed interaction."""

        window = validated_recall_params(
            top_k=top_k,
            min_score=min_score,
            half_life_turns=half_life_turns,
            now_turn=now_turn,
            current_turn=self._db.current_turn,
        )
        if window is None:
            return ()
        if min_coactivation_count < 1:
            raise ValueError("min_coactivation_count must be positive")
        if min_causal_count < 1:
            raise ValueError("min_causal_count must be positive")
        half_life, turn = window

        seeds = positive_seed_activations(
            (
                (node.key, activation)
                for node, activation in activations.items()
            ),
            what="node activations",
        )
        if not seeds:
            return ()

        seed_keys = list(seeds)
        placeholders = ",".join("?" for _ in seed_keys)
        edge_rows = self._db.execute(
            "SELECT node_low, node_high, coactivation_mass, "
            "coactivation_count, causal_count, last_reinforced_turn "
            "FROM consolidation_edges "
            f"WHERE node_low IN ({placeholders}) OR node_high IN ({placeholders})",
            (*seed_keys, *seed_keys),
        ).fetchall()
        if not edge_rows:
            return ()

        endpoint_keys = edge_endpoint_keys(edge_rows)
        endpoint_placeholders = ",".join("?" for _ in endpoint_keys)
        node_rows = self._db.execute(
            "SELECT node_key, node_kind, memory_id, chunk_id, access_mass, "
            "last_access_turn FROM consolidation_nodes "
            f"WHERE node_key IN ({endpoint_placeholders})",
            tuple(endpoint_keys),
        ).fetchall()
        nodes = {str(row[0]): self._node_from_row(row) for row in node_rows}
        node_masses = {
            str(row[0]): (float(row[4]), int(row[5])) for row in node_rows
        }
        candidates = accumulate_neighbor_evidence(
            edge_rows,
            seeds=seeds,
            excluded={node.key for node in exclude} | set(seed_keys),
            nodes=node_masses,
            default_node=None,
            now_turn=turn,
            half_life_turns=half_life,
            min_score=min_score,
            min_coaccess_count=min_coactivation_count,
            min_causal_count=min_causal_count,
            candidate_allowed=lambda key: key in nodes
            and (target_kind is None or nodes[key].kind is target_kind),
        )
        neighbors = [
            ConsolidationNeighbor(
                node=nodes[key],
                score=float(state.score),
                support=state.support,
                anchor=nodes[str(state.anchor_key)],
                coactivation_count=int(state.coaccess_count),
                causal_count=int(state.causal_count),
                last_reinforced_turn=int(state.last_reinforced_turn),
            )
            for key, state in ranked_neighbor_states(candidates)
        ]
        return tuple(neighbors[:top_k])

    def prune_edges(
        self,
        max_degree: int,
        *,
        min_score: float = 0.0,
        node_keys: Sequence[str],
        now_turn: int | None = None,
        half_life_turns: float = 200.0,
    ) -> int:
        """Enforce a hard degree bound and remove associations that cooled.

        Pruning is always scoped to the nodes one observation touched, so the
        pass never walks the whole edge table.
        """

        if max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must lie in [0, 1]")
        half_life = float(half_life_turns)
        if not math.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life_turns must be finite and positive")
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        scoped_keys = list(dict.fromkeys(node_keys))
        if not scoped_keys:
            return 0
        placeholders = ",".join("?" for _ in scoped_keys)
        edge_rows = self._db.execute(
            "SELECT node_low, node_high, coactivation_mass, "
            "coactivation_count, last_reinforced_turn "
            "FROM consolidation_edges "
            f"WHERE node_low IN ({placeholders}) "
            f"OR node_high IN ({placeholders})",
            (*scoped_keys, *scoped_keys),
        ).fetchall()
        if not edge_rows:
            return 0

        endpoint_keys = sorted(
            {str(row[0]) for row in edge_rows}
            | {str(row[1]) for row in edge_rows}
        )
        placeholders = ",".join("?" for _ in endpoint_keys)
        node_rows = self._db.execute(
            "SELECT node_key, access_mass, last_access_turn "
            f"FROM consolidation_nodes WHERE node_key IN ({placeholders})",
            tuple(endpoint_keys),
        ).fetchall()
        nodes = {
            str(row[0]): (float(row[1]), int(row[2])) for row in node_rows
        }
        scored = score_coaccess_edges(
            ((low, high, mass, edge_turn)
             for low, high, mass, _count, edge_turn in edge_rows),
            nodes,
            now_turn=turn,
            half_life_turns=half_life,
        )
        # Visit order decides which edges survive the degree cap, so pass the
        # deduplicated sequence rather than a set: set-of-str iteration varies
        # with PYTHONHASHSEED and would make pruning irreproducible.
        deletions = select_prune_victims(
            scored,
            scoped_keys,
            max_degree=max_degree,
            min_score=min_score,
        )
        if not deletions:
            return 0
        cursor = self._db.executemany(
            "DELETE FROM consolidation_edges WHERE node_low = ? AND node_high = ?",
            sorted(deletions),
        )
        self._db.commit()
        return (
            cursor.rowcount
            if cursor.rowcount is not None and cursor.rowcount >= 0
            else len(deletions)
        )

    def stats(self) -> dict[str, int]:
        counts = {
            name: int(self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for name, table in (
                ("nodes", "consolidation_nodes"),
                ("edges", "consolidation_edges"),
                ("event_receipts", "consolidation_access_events"),
            )
        }
        counts["retained_prompt_state_bytes"] = 0
        return counts


GetMemory = Callable[[str], MemoryItem | None]
HydrateChunk = Callable[..., RetrievalResult | None]
ScoreChunks = Callable[[Sequence[str]], Mapping[str, float]]
CandidateKey = Callable[[ConsolidationNeighbor], tuple[float, float, str]]


def _candidate_ordering(relevance: Mapping[str, float]) -> CandidateKey:
    """Build the sort key that ranks graph candidates against each other.

    Query relevance leads when a scorer supplied it, because a chunk the graph
    likes but the query does not is a worse read than one both like; the raw
    association score stands in when it did not, and is also the tiebreak.
    The node key breaks the remaining ties so the ordering is total.
    """

    def key(neighbor: ConsolidationNeighbor) -> tuple[float, float, str]:
        return (
            -relevance.get(neighbor.node.item_id, neighbor.score),
            -neighbor.score,
            neighbor.node.key,
        )

    return key


def _allocate_hop_quotas(hops: Sequence[int], slots: int) -> dict[int, int]:
    """Split ``slots`` read slots across the hop depths that produced candidates.

    Every depth receives the same floor share ``slots // len(hops)``.  The
    ``slots % len(hops)`` slots that do not divide evenly go to the *deepest*
    hops — ``hops[-extra:]`` — rather than the nearest ones.  That direction is
    the point of the whole scheme: one-hop neighbours are the most numerous and
    often near-duplicates of what direct retrieval already returned, so left
    alone they would consume every slot and the iteratively reached candidates
    would never be read.  With three slots over two hops this yields one
    immediate association and two candidates found by traversal.

    ``hops`` is expected in ascending order.  A depth may still end up with a
    quota of zero when slots are scarcer than depths; the caller's remainder
    backfill then decides who actually gets read.
    """

    quotas = {hop: slots // len(hops) for hop in hops}
    extra_slots = slots % len(hops)
    for hop in reversed(hops[-extra_slots:] if extra_slots else []):
        quotas[hop] += 1
    return quotas


def _select_hop_balanced(
    by_hop: Mapping[int, Sequence[ConsolidationNeighbor]],
    *,
    chunk_slots: int,
    candidate_key: CandidateKey,
) -> list[tuple[ConsolidationNeighbor, int]]:
    """Pick the candidates to hydrate, honouring the per-hop quota first.

    Each depth contributes the best of its own ranked candidates up to its
    quota.  When the quotas cannot fill ``chunk_slots`` — a depth ran out of
    candidates, or there were more depths than slots — the shortfall is
    backfilled from everything unselected in one global ranking, so a scarce
    frontier never wastes a slot.  The result is returned in reading order.
    """

    hops = sorted(by_hop)
    quotas = _allocate_hop_quotas(hops, chunk_slots)
    selected: list[tuple[ConsolidationNeighbor, int]] = []
    selected_keys: set[str] = set()
    for hop in hops:
        for neighbor in by_hop[hop][: quotas[hop]]:
            selected.append((neighbor, hop))
            selected_keys.add(neighbor.node.key)
    if len(selected) < chunk_slots:
        remainder = sorted(
            (
                (neighbor, hop)
                for hop, values in by_hop.items()
                for neighbor in values
                if neighbor.node.key not in selected_keys
            ),
            key=lambda item: candidate_key(item[0]),
        )
        selected.extend(remainder[: chunk_slots - len(selected)])
    selected.sort(key=lambda item: candidate_key(item[0]))
    return selected


def _expand_memory_arm(
    direct_memories: Sequence[MemoryResult],
    activations: Mapping[ConsolidationNode, float],
    excluded: Sequence[ConsolidationNode],
    *,
    store: LiveConsolidationStore,
    get_memory: GetMemory,
    now_turn: int,
    memory_slots: int,
    max_candidates: int,
    min_coactivation_count: int,
) -> list[MemoryResult]:
    """Swap the weakest direct memories for learned one-hop associations.

    At most ``memory_slots`` slots are taken, and never more than the number of
    direct results, so the graph can displace the tail of the ranking but can
    never take over the whole memory section.  Superseded or deleted items are
    skipped: the graph indexes ``memory_items`` rather than mirroring it, so an
    edge can outlive the row it points at.
    """

    neighbors = store.neighbors(
        activations,
        top_k=max_candidates,
        target_kind=ConsolidationNodeKind.MEMORY,
        exclude=excluded,
        now_turn=now_turn,
        min_coactivation_count=min_coactivation_count,
    )
    learned: list[MemoryResult] = []
    for neighbor in neighbors:
        item = get_memory(neighbor.node.item_id)
        if item is None or item.status is not MemoryStatus.ACTIVE:
            continue
        learned.append(
            MemoryResult(
                item=item,
                score=neighbor.score,
                relevance=0.0,
                importance=item.importance,
                energy=decay.item_energy(item, now_turn=now_turn),
                recency=decay.decay_factor(
                    item.last_access_turn,
                    now_turn,
                    item.half_life_turns,
                ),
                pin_boost=ranking.pin_boost(item.pin),
                route="live_consolidation",
                consolidation_score=neighbor.score,
                consolidation_anchor=neighbor.anchor.key,
                consolidation_support=neighbor.support,
            )
        )
        if len(learned) >= min(memory_slots, len(direct_memories)):
            break
    if not learned:
        return list(direct_memories)
    return [*direct_memories[: -len(learned)], *learned]


def _diffuse_chunk_candidates(
    activations: Mapping[ConsolidationNode, float],
    excluded: Sequence[ConsolidationNode],
    *,
    store: LiveConsolidationStore,
    now_turn: int,
    max_candidates: int,
    min_coactivation_count: int,
    diffusion_hops: int,
    diffusion_width: int,
) -> dict[str, tuple[ConsolidationNeighbor, int]]:
    """Walk the association graph outward and record each node's best reach.

    Hop one is the direct frontier of the packed context, bounded by
    ``diffusion_width``.  Each further hop re-seeds from the previous frontier's
    scores and excludes everything already traversed, so the walk never doubles
    back.  A node reached at several depths keeps its best-scoring arrival and
    the hop that produced it, which is what the quota split later balances.
    """

    frontier = store.neighbors(
        activations,
        top_k=min(diffusion_width, max_candidates),
        target_kind=ConsolidationNodeKind.CHUNK,
        exclude=excluded,
        now_turn=now_turn,
        min_coactivation_count=min_coactivation_count,
    )
    candidates = {neighbor.node.key: (neighbor, 1) for neighbor in frontier}
    for hop in range(2, diffusion_hops + 1):
        if not frontier:
            break
        frontier_activations = {
            neighbor.node: neighbor.score for neighbor in frontier
        }
        traversed = tuple(
            [
                *excluded,
                *(neighbor.node for neighbor, _depth in candidates.values()),
            ]
        )
        frontier = store.neighbors(
            frontier_activations,
            top_k=max_candidates,
            target_kind=ConsolidationNodeKind.CHUNK,
            exclude=traversed,
            now_turn=now_turn,
            # Deeper hops keep every surviving edge: the score floor was
            # already applied when the frontier they grow from was built.
            min_score=0.0,
            min_coactivation_count=min_coactivation_count,
        )
        for neighbor in frontier:
            current = candidates.get(neighbor.node.key)
            if current is None or neighbor.score > current[0].score:
                candidates[neighbor.node.key] = (neighbor, hop)
    return candidates


def _expand_chunk_arm(
    direct_chunks: Sequence[RetrievalResult],
    activations: Mapping[ConsolidationNode, float],
    excluded: Sequence[ConsolidationNode],
    *,
    store: LiveConsolidationStore,
    hydrate_chunk: HydrateChunk,
    now_turn: int,
    chunk_slots: int,
    max_candidates: int,
    min_coactivation_count: int,
    diffusion_hops: int,
    diffusion_width: int,
    chunk_relevance: ScoreChunks | None,
) -> list[RetrievalResult]:
    """Append up to ``chunk_slots`` graph-reached chunks after direct evidence.

    Unlike the memory arm this one only adds: evidence chunks are appended, so
    nothing dense retrieval found is displaced.  The pass is diffuse, score,
    allocate slots across hop depths, then hydrate — hydration is the only step
    that touches the chunk store, and a chunk that fails to hydrate simply
    yields its slot to the next candidate.
    """

    candidates = _diffuse_chunk_candidates(
        activations,
        excluded,
        store=store,
        now_turn=now_turn,
        max_candidates=max_candidates,
        min_coactivation_count=min_coactivation_count,
        diffusion_hops=diffusion_hops,
        diffusion_width=diffusion_width,
    )
    relevance = (
        dict(
            chunk_relevance(
                [item.node.item_id for item, _depth in candidates.values()]
            )
        )
        if chunk_relevance is not None and candidates
        else {}
    )
    by_hop: dict[int, list[ConsolidationNeighbor]] = {}
    for neighbor, hop in candidates.values():
        by_hop.setdefault(hop, []).append(neighbor)
    if not by_hop:
        return list(direct_chunks)
    candidate_key = _candidate_ordering(relevance)
    for values in by_hop.values():
        values.sort(key=candidate_key)
    selected_neighbors = _select_hop_balanced(
        by_hop, chunk_slots=chunk_slots, candidate_key=candidate_key
    )

    learned_chunks: list[RetrievalResult] = []
    for neighbor, hop in selected_neighbors:
        result = hydrate_chunk(
            neighbor.node.item_id,
            score=relevance.get(neighbor.node.item_id, neighbor.score),
            route="live_consolidation",
            association_score=neighbor.score,
            anchor_chunk_id=(
                neighbor.anchor.item_id
                if neighbor.anchor.kind is ConsolidationNodeKind.CHUNK
                else None
            ),
        )
        if result is None:
            continue
        learned_chunks.append(
            result.model_copy(
                update={
                    "consolidation_score": neighbor.score,
                    "consolidation_anchor": neighbor.anchor.key,
                    "consolidation_support": neighbor.support,
                    "dense_score": relevance.get(neighbor.node.item_id),
                    "association_hop": hop,
                }
            )
        )
        if len(learned_chunks) >= chunk_slots:
            break
    if not learned_chunks:
        return list(direct_chunks)
    return [*direct_chunks, *learned_chunks]


def expand_context_associations(
    memories: Sequence[MemoryResult],
    chunks: Sequence[RetrievalResult],
    *,
    store: LiveConsolidationStore,
    get_memory: GetMemory,
    hydrate_chunk: HydrateChunk,
    now_turn: int,
    memory_slots: int = 1,
    chunk_slots: int = 1,
    max_candidates: int = 32,
    min_coactivation_count: int = 2,
    diffusion_hops: int = 1,
    diffusion_width: int = 32,
    chunk_relevance: ScoreChunks | None = None,
) -> tuple[list[MemoryResult], list[RetrievalResult]]:
    """Add bounded graph candidates without evicting direct retrieval.

    The two arms are independent.  They share only the packed context's
    activations and the exclusion set built from it; neither reads the other's
    result, so each is free to decline to expand.
    """

    if memory_slots < 0 or chunk_slots < 0:
        raise ValueError("consolidation slots must be non-negative")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if diffusion_hops < 1:
        raise ValueError("diffusion_hops must be positive")
    if diffusion_width < 1:
        raise ValueError("diffusion_width must be positive")

    direct_memories = list(memories)
    direct_chunks = list(chunks)
    activations = context_activations(
        [result.item.mem_id for result in direct_memories],
        [result.chunk.chunk_id for result in direct_chunks],
    )
    if not activations:
        return direct_memories, direct_chunks
    excluded = tuple(activations)

    expanded_memories = direct_memories
    if direct_memories and memory_slots:
        expanded_memories = _expand_memory_arm(
            direct_memories,
            activations,
            excluded,
            store=store,
            get_memory=get_memory,
            now_turn=now_turn,
            memory_slots=memory_slots,
            max_candidates=max_candidates,
            min_coactivation_count=min_coactivation_count,
        )

    expanded_chunks = direct_chunks
    if direct_chunks and chunk_slots:
        expanded_chunks = _expand_chunk_arm(
            direct_chunks,
            activations,
            excluded,
            store=store,
            hydrate_chunk=hydrate_chunk,
            now_turn=now_turn,
            chunk_slots=chunk_slots,
            max_candidates=max_candidates,
            min_coactivation_count=min_coactivation_count,
            diffusion_hops=diffusion_hops,
            diffusion_width=diffusion_width,
            chunk_relevance=chunk_relevance,
        )

    return expanded_memories, expanded_chunks
