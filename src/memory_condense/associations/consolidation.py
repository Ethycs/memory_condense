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
from typing import Callable, Mapping, Protocol, Sequence

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


class QwenHeadHit(Protocol):
    """The fields emitted by :class:`head_memory.MemoryLinkHit`."""

    episode_id: str
    qk_score: float
    ov_transport: float


class QwenTurnLinker(Protocol):
    """Bounded transient linker interface implemented by ``QwenMemoryLinker``."""

    def link(
        self,
        source_text: str,
        candidates: Sequence[object],
        *,
        top_k: int | None = None,
    ) -> object: ...


def qwen_head_activations(
    hits: Sequence[QwenHeadHit],
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
    linker: QwenTurnLinker,
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
    if len(results) == 1:
        result = results[0]
    else:
        signatures = [
            tuple(float(value) for value in result.source_cav_signature)
            for result in results
            if getattr(result, "source_cav_signature", ())
        ]
        signature: tuple[float, ...] = ()
        if signatures and len({len(value) for value in signatures}) == 1:
            signature = tuple(
                sum(values) / len(values) for values in zip(*signatures, strict=True)
            )
        result = MemoryLinkResult(
            hits=tuple(
                sorted(
                    hits,
                    key=lambda hit: (
                        float(hit.qk_score),
                        float(hit.ov_transport),
                        hit.episode_id,
                    ),
                    reverse=True,
                )
            ),
            source_cav_signature=signature,
            workspace_candidates=max(
                int(result.workspace_candidates) for result in results
            ),
            workspace_tokens=max(int(result.workspace_tokens) for result in results),
            passes=sum(int(getattr(result, "passes", 1)) for result in results),
            total_candidate_inspections=sum(
                int(
                    getattr(result, "total_candidate_inspections", 0)
                    or result.workspace_candidates
                )
                for result in results
            ),
        )
    return result, qwen_head_activations(hits)


class LiveConsolidationStore:
    """SQLite-backed decaying association graph over memories and chunks."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def _validate_nodes(self, nodes: Sequence[ConsolidationNode]) -> None:
        memory_ids = [
            node.item_id
            for node in nodes
            if node.kind is ConsolidationNodeKind.MEMORY
        ]
        chunk_ids = [
            node.item_id
            for node in nodes
            if node.kind is ConsolidationNodeKind.CHUNK
        ]
        existing_memories: set[str] = set()
        if memory_ids:
            placeholders = ",".join("?" for _ in memory_ids)
            existing_memories = {
                str(row[0])
                for row in self._db.execute(
                    "SELECT mem_id FROM memory_items "
                    f"WHERE status = 'active' AND mem_id IN ({placeholders})",
                    tuple(memory_ids),
                ).fetchall()
            }
        existing_chunks: set[str] = set()
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            existing_chunks = {
                str(row[0])
                for row in self._db.execute(
                    "SELECT chunk_id FROM chunks WHERE embedding IS NOT NULL "
                    f"AND hnsw_label IS NOT NULL AND chunk_id IN ({placeholders})",
                    tuple(chunk_ids),
                ).fetchall()
            }
        missing = [
            node.key
            for node in nodes
            if (
                node.kind is ConsolidationNodeKind.MEMORY
                and node.item_id not in existing_memories
            )
            or (
                node.kind is ConsolidationNodeKind.CHUNK
                and node.item_id not in existing_chunks
            )
        ]
        if missing:
            raise ValueError(
                "consolidation nodes must reference active retrievable state: "
                + ", ".join(missing)
            )

    def observe(
        self,
        access_event_id: str,
        activations: Mapping[ConsolidationNode, float],
        *,
        pair_affinities: Mapping[
            tuple[ConsolidationNode, ConsolidationNode], float
        ]
        | None = None,
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
        self._validate_nodes([node for node, _ in selected])

        selected_keys = {node.key for node, _ in selected}
        causal_keys = {
            node.key for node in causal_targets if node.key in selected_keys
        }
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
                affinities[(low, high)] = max(
                    affinities.get((low, high), 0.0), affinity
                )
        fingerprint_payload = {
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
        fingerprint = hashlib.sha256(
            _canonical_json(fingerprint_payload).encode("utf-8")
        ).hexdigest()
        existing_event = self._db.execute(
            "SELECT event_fingerprint, member_count "
            "FROM consolidation_access_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if existing_event is not None:
            if existing_event[0] != fingerprint:
                raise ValueError(
                    "access_event_id was already used with a different context set"
                )
            return ConsolidationUpdate(
                event_id=event_id,
                created=False,
                members_observed=int(existing_event[1]),
                edges_reinforced=0,
                edges_pruned=0,
            )

        keys = [node.key for node, _ in selected]
        placeholders = ",".join("?" for _ in keys)
        existing_nodes: dict[str, tuple[float, int, int]] = {}
        if keys:
            existing_nodes = {
                str(row[0]): (float(row[1]), int(row[2]), int(row[3]))
                for row in self._db.execute(
                    "SELECT node_key, access_mass, access_count, last_access_turn "
                    f"FROM consolidation_nodes WHERE node_key IN ({placeholders})",
                    tuple(keys),
                ).fetchall()
            }

        pairs = [
            (left[0], right[0], left[1], right[1])
            for left, right in combinations(selected, 2)
        ]
        existing_edges: dict[tuple[str, str], tuple[float, int, int, int]] = {}
        if pairs:
            existing_edges = {
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

        node_rows: list[tuple[object, ...]] = []
        for node, activation in selected:
            old_mass, old_count, old_turn = existing_nodes.get(
                node.key, (0.0, 0, turn)
            )
            mass = decayed_mass(old_mass, old_turn, turn, half_life)
            mass += rate * activation * activation
            node_rows.append(
                (
                    node.key,
                    node.kind.value,
                    node.item_id
                    if node.kind is ConsolidationNodeKind.MEMORY
                    else None,
                    node.item_id
                    if node.kind is ConsolidationNodeKind.CHUNK
                    else None,
                    mass,
                    old_count + 1,
                    turn,
                )
            )

        edge_rows: list[tuple[object, ...]] = []
        for left, right, left_activation, right_activation in pairs:
            affinity = affinities.get((left.key, right.key), 1.0)
            if affinity <= 0.0:
                continue
            old_mass, old_count, old_causal_count, old_turn = existing_edges.get(
                (left.key, right.key), (0.0, 0, 0, turn)
            )
            mass = decayed_mass(old_mass, old_turn, turn, half_life)
            # Rank-only operation uses affinity=1.  A transient CAV/QK/OV
            # inspection can provide a bounded gate per pair, turning the Qwen
            # heads into the association teacher without persisting their
            # workspace in this graph.
            mass += rate * left_activation * right_activation * affinity
            edge_rows.append(
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

        connection = self._db.connection
        with connection:
            connection.execute(
                "INSERT INTO consolidation_access_events "
                "(event_id, observed_turn, event_fingerprint, member_count) "
                "VALUES (?, ?, ?, ?)",
                (event_id, turn, fingerprint, len(selected)),
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
        node_keys: Sequence[str] | None = None,
        now_turn: int | None = None,
        half_life_turns: float = 200.0,
    ) -> int:
        """Enforce a hard degree bound and remove associations that cooled."""

        if max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must lie in [0, 1]")
        half_life = float(half_life_turns)
        if not math.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life_turns must be finite and positive")
        turn = self._db.current_turn() if now_turn is None else int(now_turn)
        scoped_keys = list(dict.fromkeys(node_keys or ()))
        if node_keys is not None and not scoped_keys:
            return 0
        if node_keys is None:
            edge_rows = self._db.execute(
                "SELECT node_low, node_high, coactivation_mass, "
                "coactivation_count, last_reinforced_turn "
                "FROM consolidation_edges"
            ).fetchall()
        else:
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
        deletions = select_prune_victims(
            scored,
            set(endpoint_keys) if node_keys is None else set(scoped_keys),
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
    half_life_turns: float = 200.0,
    min_score: float = 0.05,
    min_coactivation_count: int = 2,
    min_causal_count: int = 1,
    diffusion_hops: int = 1,
    diffusion_width: int = 32,
    chunk_relevance: ScoreChunks | None = None,
) -> tuple[list[MemoryResult], list[RetrievalResult]]:
    """Add bounded graph candidates without evicting direct retrieval."""

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
        neighbors = store.neighbors(
            activations,
            top_k=max_candidates,
            target_kind=ConsolidationNodeKind.MEMORY,
            exclude=excluded,
            now_turn=now_turn,
            half_life_turns=half_life_turns,
            min_score=min_score,
            min_coactivation_count=min_coactivation_count,
            min_causal_count=min_causal_count,
        )
        learned: list[MemoryResult] = []
        for neighbor in neighbors:
            item = get_memory(neighbor.node.item_id)
            if item is None or item.status is not MemoryStatus.ACTIVE:
                continue
            energy = decay.item_energy(item, now_turn=now_turn)
            learned.append(
                MemoryResult(
                    item=item,
                    score=neighbor.score,
                    relevance=0.0,
                    importance=item.importance,
                    energy=energy,
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
        if learned:
            expanded_memories = direct_memories[: -len(learned)] + learned

    expanded_chunks = direct_chunks
    if direct_chunks and chunk_slots:
        frontier = store.neighbors(
            activations,
            top_k=min(diffusion_width, max_candidates),
            target_kind=ConsolidationNodeKind.CHUNK,
            exclude=excluded,
            now_turn=now_turn,
            half_life_turns=half_life_turns,
            min_score=min_score,
            min_coactivation_count=min_coactivation_count,
            min_causal_count=min_causal_count,
        )
        candidates = {
            neighbor.node.key: (neighbor, 1) for neighbor in frontier
        }
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
                half_life_turns=half_life_turns,
                min_score=0.0,
                min_coactivation_count=min_coactivation_count,
                min_causal_count=min_causal_count,
            )
            for neighbor in frontier:
                current = candidates.get(neighbor.node.key)
                if current is None or neighbor.score > current[0].score:
                    candidates[neighbor.node.key] = (neighbor, hop)
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
            return expanded_memories, expanded_chunks
        candidate_key = lambda item: (
            -relevance.get(item.node.item_id, item.score),
            -item.score,
            item.node.key,
        )
        for values in by_hop.values():
            values.sort(key=candidate_key)
        # Keep an explicit share for the deeper frontier instead of allowing
        # near-duplicate one-hop nodes to consume every read slot. With three
        # slots and two hops this yields one immediate association and two
        # iteratively reached candidates.
        hops = sorted(by_hop)
        quotas = {hop: chunk_slots // len(hops) for hop in hops}
        extra_slots = chunk_slots % len(hops)
        for hop in reversed(hops[-extra_slots:] if extra_slots else []):
            quotas[hop] += 1
        selected_neighbors: list[tuple[ConsolidationNeighbor, int]] = []
        selected_keys: set[str] = set()
        for hop in hops:
            for neighbor in by_hop[hop][: quotas[hop]]:
                selected_neighbors.append((neighbor, hop))
                selected_keys.add(neighbor.node.key)
        if len(selected_neighbors) < chunk_slots:
            remainder = sorted(
                (
                    (neighbor, hop)
                    for hop, values in by_hop.items()
                    for neighbor in values
                    if neighbor.node.key not in selected_keys
                ),
                key=lambda item: candidate_key(item[0]),
            )
            selected_neighbors.extend(remainder[: chunk_slots - len(selected_neighbors)])
        selected_neighbors.sort(key=lambda item: candidate_key(item[0]))
        learned_chunks: list[RetrievalResult] = []
        for neighbor, hop in selected_neighbors:
            query_score = relevance.get(neighbor.node.item_id, neighbor.score)
            result = hydrate_chunk(
                neighbor.node.item_id,
                score=query_score,
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
        if learned_chunks:
            expanded_chunks = [*direct_chunks, *learned_chunks]

    return expanded_memories, expanded_chunks
