from __future__ import annotations

import numpy as np
import pytest

from memory_condense.associations.consolidation import (
    ConsolidationNode,
    ConsolidationNodeKind,
    LiveConsolidationStore,
    context_activations,
    expand_context_associations,
    inspect_qwen_context_hyperplane,
    qwen_head_activations,
)
from memory_condense.associations.head_memory import MemoryLinkHit, MemoryLinkResult
from memory_condense.persistence.memory_store import MemoryStore
from memory_condense.domain.schemas import (
    Chunk,
    CreateOp,
    MemoryResult,
    MemoryStatus,
    MemoryType,
    Provenance,
    RetrievalResult,
)
from memory_condense.persistence.transcript_store import TranscriptStore


def _populate(db, *, memory_count: int = 4, chunk_count: int = 4):
    transcript = TranscriptStore(db)
    memory = MemoryStore(db)
    memories = []
    chunks = {}
    count = max(memory_count, chunk_count)
    for index in range(count):
        text = f"We decided durable fact {index}."
        turn = transcript.append("user", text)
        chunk_id = f"chunk-{index}"
        if index < chunk_count:
            db.execute(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, token_count, "
                "embedding, hnsw_label) VALUES (?, ?, ?, 0, ?, 5, ?, ?)",
                (
                    chunk_id,
                    turn.turn_id,
                    text,
                    len(text),
                    np.asarray([float(index + 1)], dtype=np.float32).tobytes(),
                    index,
                ),
            )
            chunks[chunk_id] = Chunk(
                chunk_id=chunk_id,
                turn_id=turn.turn_id,
                text=text,
                start_char=0,
                end_char=len(text),
                token_count=5,
            )
        if index < memory_count:
            memories.append(
                memory.create(
                    CreateOp(
                        type=MemoryType.DECISION,
                        content=f"durable fact {index}",
                        provenance=[
                            Provenance(
                                turn_id=turn.turn_id,
                                quote=text,
                                chunk_id=chunk_id if index < chunk_count else None,
                            )
                        ],
                    ),
                    embedding=[float(index + 1)],
                )
            )
    db.commit()
    return memory, memories, chunks


def test_context_activations_balance_memory_and_evidence_ranks():
    activations = context_activations(["m0", "m1"], ["c0", "c1"])
    assert activations[ConsolidationNode.memory("m0")] == pytest.approx(1.0)
    assert activations[ConsolidationNode.chunk("c0")] == pytest.approx(1.0)
    assert activations[ConsolidationNode.memory("m1")] == pytest.approx(2**-0.5)
    assert activations[ConsolidationNode.chunk("c1")] == pytest.approx(2**-0.5)


def test_qwen_head_hits_define_the_transient_turn_hyperplane():
    memory_node = ConsolidationNode.memory("m0")
    chunk_node = ConsolidationNode.chunk("c0")
    activations = qwen_head_activations(
        [
            MemoryLinkHit(
                episode_id=memory_node.key,
                qk_score=2.0,
                ov_transport=0.5,
                head_weights=(0.1, 0.9),
            ),
            MemoryLinkHit(
                episode_id=chunk_node.key,
                qk_score=1.0,
                ov_transport=0.0,
                head_weights=(0.8, 0.2),
            ),
        ]
    )
    assert activations[memory_node] == pytest.approx(1.0)
    assert activations[chunk_node] == pytest.approx(0.35)


def test_qwen_hyperplane_recursively_covers_bounded_candidate_groups(db):
    memory, memories, chunks = _populate(db, memory_count=2, chunk_count=3)

    class GroupedLinker:
        max_candidates = 2

        def __init__(self):
            self.group_sizes = []

        def link(self, source_text, candidates, *, top_k=None):
            del source_text
            bounded = list(candidates[:top_k])
            self.group_sizes.append(len(bounded))
            return MemoryLinkResult(
                hits=tuple(
                    MemoryLinkHit(
                        episode_id=candidate.episode_id,
                        qk_score=1.0,
                        ov_transport=1.0,
                        head_weights=(1.0,),
                    )
                    for candidate in bounded
                ),
                source_cav_signature=(0.2, 0.8),
                workspace_candidates=len(bounded),
                workspace_tokens=32,
                total_candidate_inspections=len(bounded),
            )

    linker = GroupedLinker()
    results = [
        RetrievalResult(chunk=chunks[f"chunk-{index}"], score=1.0)
        for index in range(3)
    ]
    inspection, activations = inspect_qwen_context_hyperplane(
        linker,
        "later prompt",
        memories,
        results,
    )

    assert linker.group_sizes == [2, 2, 1]
    assert inspection.passes == 3
    assert inspection.workspace_candidates == 2
    assert inspection.total_candidate_inspections == 5
    assert len(activations) == 5


def test_repeated_later_contexts_create_cross_partition_recall(db):
    _memory, memories, chunks = _populate(db, memory_count=1, chunk_count=1)
    store = LiveConsolidationStore(db)
    memory_node = ConsolidationNode.memory(memories[0].mem_id)
    chunk_node = ConsolidationNode.chunk(chunks["chunk-0"].chunk_id)
    activations = {memory_node: 1.0, chunk_node: 0.8}

    first = store.observe("prompt-1", activations, now_turn=10)
    assert first.edges_reinforced == 1
    # One accidental co-occurrence is not yet a consolidated association.
    assert store.neighbors(
        {memory_node: 1.0},
        target_kind=ConsolidationNodeKind.CHUNK,
        top_k=1,
        now_turn=10,
    ) == ()

    store.observe("prompt-2", activations, now_turn=11)
    neighbors = store.neighbors(
        {memory_node: 1.0},
        target_kind=ConsolidationNodeKind.CHUNK,
        top_k=1,
        now_turn=11,
    )
    assert [neighbor.node for neighbor in neighbors] == [chunk_node]
    assert neighbors[0].coactivation_count == 2


def test_one_completed_interaction_can_recall_its_unique_outcome(db):
    _memory, memories, chunks = _populate(db, memory_count=1, chunk_count=1)
    store = LiveConsolidationStore(db)
    prompt = ConsolidationNode.memory(memories[0].mem_id)
    outcome = ConsolidationNode.chunk(chunks["chunk-0"].chunk_id)

    store.observe(
        "completed-interaction",
        {prompt: 1.0, outcome: 0.8},
        causal_targets=(outcome,),
        now_turn=10,
    )

    neighbor = store.neighbors(
        {prompt: 1.0},
        target_kind=ConsolidationNodeKind.CHUNK,
        top_k=1,
        now_turn=10,
    )[0]
    assert neighbor.node == outcome
    assert neighbor.coactivation_count == 1
    assert neighbor.causal_count == 1


def test_same_context_event_is_idempotent(db):
    _memory, memories, chunks = _populate(db, memory_count=1, chunk_count=1)
    store = LiveConsolidationStore(db)
    activations = {
        ConsolidationNode.memory(memories[0].mem_id): 1.0,
        ConsolidationNode.chunk(chunks["chunk-0"].chunk_id): 1.0,
    }
    created = store.observe("same-prompt", activations)
    repeated = store.observe("same-prompt", activations)
    assert created.created is True
    assert repeated.created is False
    assert store.stats()["event_receipts"] == 1


def test_qwen_pair_affinity_weights_the_persistent_update(db):
    _memory, memories, chunks = _populate(db, memory_count=1, chunk_count=1)
    store = LiveConsolidationStore(db)
    left = ConsolidationNode.memory(memories[0].mem_id)
    right = ConsolidationNode.chunk(chunks["chunk-0"].chunk_id)
    activations = {left: 1.0, right: 1.0}
    store.observe(
        "qwen-weighted",
        activations,
        pair_affinities={(left, right): 0.25},
        now_turn=10,
    )
    neighbor = store.neighbors(
        {left: 1.0},
        target_kind=ConsolidationNodeKind.CHUNK,
        top_k=1,
        min_coactivation_count=1,
        now_turn=10,
    )[0]
    assert neighbor.score == pytest.approx(0.25)


def test_zero_head_affinity_does_not_create_an_edge(db):
    _memory, memories, chunks = _populate(db, memory_count=1, chunk_count=1)
    store = LiveConsolidationStore(db)
    left = ConsolidationNode.memory(memories[0].mem_id)
    right = ConsolidationNode.chunk(chunks["chunk-0"].chunk_id)
    store.observe(
        "qwen-reject",
        {left: 1.0, right: 1.0},
        pair_affinities={(left, right): 0.0},
    )
    assert store.stats()["edges"] == 0


def test_link_strength_decays_in_turn_space(db):
    _memory, memories, chunks = _populate(db, memory_count=1, chunk_count=1)
    store = LiveConsolidationStore(db)
    left = ConsolidationNode.memory(memories[0].mem_id)
    right = ConsolidationNode.chunk(chunks["chunk-0"].chunk_id)
    for index in range(2):
        store.observe(
            f"prompt-{index}",
            {left: 1.0, right: 1.0},
            now_turn=10,
            half_life_turns=20.0,
        )
    fresh = store.neighbors(
        {left: 1.0}, top_k=1, now_turn=10, half_life_turns=20.0
    )[0]
    stale = store.neighbors(
        {left: 1.0}, top_k=1, now_turn=30, half_life_turns=20.0
    )[0]
    assert fresh.score == pytest.approx(1.0)
    assert stale.score == pytest.approx(0.5)


def test_degree_pruning_bounds_live_consolidation_graph(db):
    _memory, memories, _chunks = _populate(db, memory_count=5, chunk_count=0)
    store = LiveConsolidationStore(db)
    store.observe(
        "crowded",
        {ConsolidationNode.memory(item.mem_id): 1.0 for item in memories},
        max_degree=2,
    )
    rows = db.execute("SELECT node_low, node_high FROM consolidation_edges").fetchall()
    degrees: dict[str, int] = {}
    for low, high in rows:
        degrees[low] = degrees.get(low, 0) + 1
        degrees[high] = degrees.get(high, 0) + 1
    assert rows
    assert max(degrees.values()) <= 2


def test_retiring_authoritative_state_removes_only_derived_graph_state(db):
    memory, memories, chunks = _populate(db, memory_count=1, chunk_count=1)
    store = LiveConsolidationStore(db)
    left = ConsolidationNode.memory(memories[0].mem_id)
    right = ConsolidationNode.chunk(chunks["chunk-0"].chunk_id)
    store.observe("prompt", {left: 1.0, right: 1.0})
    memory._set_status(memories[0].mem_id, MemoryStatus.SUPERSEDED)

    assert store.stats()["nodes"] == 1
    assert store.stats()["edges"] == 0
    # The source memory and provenance were retired, not hard-deleted.
    assert memory.get(memories[0].mem_id) is not None


def test_association_expansion_uses_one_reserved_memory_slot(db):
    memory, memories, _chunks = _populate(db, memory_count=3, chunk_count=0)
    store = LiveConsolidationStore(db)
    first, displaced, linked = memories
    for index in range(2):
        store.observe(
            f"learn-{index}",
            {
                ConsolidationNode.memory(first.mem_id): 1.0,
                ConsolidationNode.memory(linked.mem_id): 1.0,
            },
        )
    direct = [
        MemoryResult(item=first, score=0.9),
        MemoryResult(item=displaced, score=0.8),
    ]
    expanded, chunks = expand_context_associations(
        direct,
        [],
        store=store,
        get_memory=memory.get,
        hydrate_chunk=lambda *_args, **_kwargs: None,
        now_turn=db.current_turn(),
        memory_slots=1,
        chunk_slots=0,
    )
    assert chunks == []
    assert [result.item.mem_id for result in expanded] == [first.mem_id, linked.mem_id]
    assert expanded[-1].route == "live_consolidation"


def test_chunk_expansion_appends_without_evicting_direct_evidence(db):
    _memory, _memories, chunks = _populate(db, memory_count=0, chunk_count=2)
    store = LiveConsolidationStore(db)
    direct_node = ConsolidationNode.chunk("chunk-0")
    learned_node = ConsolidationNode.chunk("chunk-1")
    for index in range(2):
        store.observe(
            f"learn-chunks-{index}",
            {direct_node: 1.0, learned_node: 0.8},
        )
    direct = RetrievalResult(chunk=chunks["chunk-0"], score=0.9)

    _memories_out, expanded = expand_context_associations(
        [],
        [direct],
        store=store,
        get_memory=lambda _mem_id: None,
        hydrate_chunk=lambda chunk_id, **kwargs: RetrievalResult(
            chunk=chunks[chunk_id],
            score=kwargs["score"],
            route=kwargs["route"],
        ),
        now_turn=db.current_turn(),
        memory_slots=0,
        chunk_slots=1,
    )

    assert [result.chunk.chunk_id for result in expanded] == ["chunk-0", "chunk-1"]
    assert expanded[-1].route == "live_consolidation"


def test_two_hop_read_balances_slots_across_frontiers(db):
    _memory, _memories, chunks = _populate(db, memory_count=0, chunk_count=5)
    store = LiveConsolidationStore(db)
    nodes = [ConsolidationNode.chunk(f"chunk-{index}") for index in range(5)]
    for event, left, right in (
        ("first-a", nodes[0], nodes[1]),
        ("first-b", nodes[0], nodes[2]),
        ("second-a", nodes[1], nodes[3]),
        ("second-b", nodes[1], nodes[4]),
    ):
        for repeat in range(2):
            store.observe(
                f"{event}-{repeat}",
                {left: 1.0, right: 0.8},
            )
    relevance = {
        "chunk-1": 0.9,
        "chunk-2": 0.8,
        "chunk-3": 0.7,
        "chunk-4": 0.6,
    }

    _memories_out, expanded = expand_context_associations(
        [],
        [RetrievalResult(chunk=chunks["chunk-0"], score=1.0)],
        store=store,
        get_memory=lambda _mem_id: None,
        hydrate_chunk=lambda chunk_id, **kwargs: RetrievalResult(
            chunk=chunks[chunk_id],
            score=kwargs["score"],
            route=kwargs["route"],
        ),
        now_turn=db.current_turn(),
        memory_slots=0,
        chunk_slots=3,
        diffusion_hops=2,
        max_candidates=16,
        diffusion_width=8,
        chunk_relevance=lambda ids: {
            chunk_id: relevance[chunk_id] for chunk_id in ids
        },
    )

    assert [result.chunk.chunk_id for result in expanded] == [
        "chunk-0",
        "chunk-1",
        "chunk-3",
        "chunk-4",
    ]
    assert [result.association_hop for result in expanded[1:]] == [1, 2, 2]


def test_schema_cannot_retain_prompt_or_transformer_state(db):
    columns = {
        row[1]
        for table in (
            "consolidation_access_events",
            "consolidation_nodes",
            "consolidation_edges",
        )
        for row in db.execute(f"PRAGMA table_info({table})").fetchall()
    }
    forbidden = {
        "text",
        "query",
        "prompt",
        "keys",
        "values",
        "kv_cache",
        "token_ids",
        "attention",
        "residual",
        "hidden_states",
    }
    assert not columns & forbidden
    assert LiveConsolidationStore(db).stats()["retained_prompt_state_bytes"] == 0
