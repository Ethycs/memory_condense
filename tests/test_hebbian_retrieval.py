from __future__ import annotations

import numpy as np
import pytest

from memory_condense.associations.association_store import AssociationArtifact, AssociationStore
from memory_condense.persistence.db import Database
from memory_condense.associations.hebbian_retrieval import expand_hebbian_results
from memory_condense.domain.schemas import Chunk, RetrievalResult


def _artifact() -> AssociationArtifact:
    return AssociationArtifact.create(
        model_id="Qwen/Qwen3-8B",
        checkpoint_id="bf16:test",
        prefix_layers=7,
        head_layer=1,
        cav_layer=5,
        concept_names=("context", "binding"),
        head_count=32,
    )


def _insert_chunks(db: Database, count: int = 6) -> dict[str, Chunk]:
    chunks: dict[str, Chunk] = {}
    for index in range(count):
        turn_id = f"turn-{index}"
        chunk_id = f"chunk-{index}"
        text = f"conceptual chunk {index}"
        db.execute(
            "INSERT INTO turns (turn_id, role, text, created_at, ordinal) "
            "VALUES (?, 'user', ?, '2026-08-16T00:00:00+00:00', ?)",
            (turn_id, text, index + 1),
        )
        db.execute(
            "INSERT INTO chunks "
            "(chunk_id, turn_id, text, start_char, end_char, token_count, "
            "embedding, hnsw_label) VALUES (?, ?, ?, 0, ?, ?, ?, ?)",
            (
                chunk_id,
                turn_id,
                text,
                len(text),
                4 + index,
                np.asarray([float(index + 1)], dtype=np.float32).tobytes(),
                index,
            ),
        )
        chunks[chunk_id] = Chunk(
            chunk_id=chunk_id,
            turn_id=turn_id,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=4 + index,
            embedding=[float(index + 1)],
        )
    db.commit()
    return chunks


def _result(chunk: Chunk, score: float, *, lexical: float = 0.0) -> RetrievalResult:
    return RetrievalResult(
        chunk=chunk,
        score=score,
        lexical_score=lexical,
    )


def test_same_event_is_idempotent_and_persists_across_restart(tmp_path):
    path = tmp_path / "hebbian.db"
    artifact = _artifact()
    with Database(path) as db:
        chunks = _insert_chunks(db, 3)
        store = AssociationStore(db)
        store.register_artifact(artifact)
        first = store.reinforce_retrieval_coaccess(
            artifact.artifact_id,
            "turn:7",
            {"chunk-0": 1.0, "chunk-1": 0.5},
            now_turn=7,
        )
        repeated = store.reinforce_retrieval_coaccess(
            artifact.artifact_id,
            "turn:7",
            {"chunk-0": 1.0, "chunk-1": 0.5},
            now_turn=7,
        )
        assert first.created is True
        assert first.edges_reinforced == 1
        assert repeated.created is False
        assert store.hebbian_stats(artifact.artifact_id) == {
            "nodes": 2,
            "edges": 1,
            "event_receipts": 1,
            "retained_request_token_state_bytes": 0,
            "retained_token_state_bytes": 0,
        }

    with Database(path) as db:
        store = AssociationStore(db)
        neighbors = store.hebbian_neighbors(
            {"chunk-0": 1.0},
            artifact.artifact_id,
            top_k=2,
            now_turn=7,
        )
        assert [item.chunk_id for item in neighbors] == ["chunk-1"]
        assert neighbors[0].score == pytest.approx(1.0)
        assert chunks["chunk-1"].text == "conceptual chunk 1"


def test_reusing_event_id_with_different_concepts_is_rejected(db):
    _insert_chunks(db, 3)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "same-turn",
        {"chunk-0": 1.0, "chunk-1": 1.0},
    )
    with pytest.raises(ValueError, match="different retrieval set"):
        store.reinforce_retrieval_coaccess(
            artifact.artifact_id,
            "same-turn",
            {"chunk-0": 1.0, "chunk-2": 1.0},
        )


def test_different_turns_do_not_create_false_coaccess_edge(db):
    _insert_chunks(db, 2)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "event-a",
        {"chunk-0": 1.0},
        now_turn=1,
    )
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "event-b",
        {"chunk-1": 1.0},
        now_turn=2,
    )
    assert store.hebbian_stats(artifact.artifact_id)["edges"] == 0
    assert store.hebbian_neighbors(
        {"chunk-0": 1.0}, artifact.artifact_id, top_k=3
    ) == ()


def test_link_freshness_cools_in_turn_space(db):
    _insert_chunks(db, 2)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "event-1",
        {"chunk-0": 1.0, "chunk-1": 1.0},
        now_turn=10,
        half_life_turns=20.0,
    )
    fresh = store.hebbian_neighbors(
        {"chunk-0": 1.0},
        artifact.artifact_id,
        top_k=1,
        now_turn=10,
        half_life_turns=20.0,
    )[0]
    stale = store.hebbian_neighbors(
        {"chunk-0": 1.0},
        artifact.artifact_id,
        top_k=1,
        now_turn=30,
        half_life_turns=20.0,
    )[0]
    assert fresh.score == pytest.approx(1.0)
    assert stale.score == pytest.approx(0.5)


def test_degree_cap_bounds_graph_growth(db):
    _insert_chunks(db, 5)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "crowded-turn",
        {f"chunk-{index}": 1.0 for index in range(5)},
        max_degree=2,
    )
    rows = db.execute(
        "SELECT chunk_low, chunk_high FROM hebbian_chunk_edges "
        "WHERE artifact_id = ?",
        (artifact.artifact_id,),
    ).fetchall()
    degrees = {f"chunk-{index}": 0 for index in range(5)}
    for low, high in rows:
        degrees[low] += 1
        degrees[high] += 1
    assert rows
    assert max(degrees.values()) <= 2


def test_event_receipt_history_is_bounded(db):
    _insert_chunks(db, 2)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    for index in range(5):
        store.reinforce_retrieval_coaccess(
            artifact.artifact_id,
            f"event-{index}",
            {"chunk-0": 1.0, "chunk-1": 1.0},
            now_turn=index,
            max_event_history=3,
        )
    assert store.hebbian_stats(artifact.artifact_id)["event_receipts"] == 3


def test_expansion_uses_reserved_slot_and_never_grows_prompt_tokens(db):
    chunks = _insert_chunks(db, 4)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "learn-a-c",
        {"chunk-0": 1.0, "chunk-2": 1.0},
        now_turn=4,
    )
    anchors = [
        _result(chunks["chunk-0"], 0.9),
        _result(chunks["chunk-1"], 0.8),
    ]
    by_id = {
        chunk_id: _result(chunk, 0.0) for chunk_id, chunk in chunks.items()
    }

    def hydrate(chunk_id: str, **_kwargs) -> RetrievalResult | None:
        return by_id.get(chunk_id)

    expanded = expand_hebbian_results(
        anchors,
        artifact.artifact_id,
        store=store,
        hydrate=hydrate,
        now_turn=4,
        k=2,
        hebbian_slots=1,
        max_prompt_token_increase=1,
    )
    assert [result.chunk.chunk_id for result in expanded] == ["chunk-0", "chunk-2"]
    assert expanded[-1].route == "hebbian_coaccess"
    assert expanded[-1].association_path == ("chunk-0", "chunk-2")

    rolled_back = expand_hebbian_results(
        anchors,
        artifact.artifact_id,
        store=store,
        hydrate=hydrate,
        now_turn=4,
        k=2,
        hebbian_slots=1,
        max_prompt_token_increase=0,
    )
    assert [result.chunk.chunk_id for result in rolled_back] == [
        "chunk-0",
        "chunk-1",
    ]


def test_hebbian_schema_cannot_store_transformer_token_state(db):
    tables = (
        "hebbian_access_events",
        "hebbian_chunk_nodes",
        "hebbian_chunk_edges",
    )
    columns = {
        row[1]
        for table in tables
        for row in db.execute(f"PRAGMA table_info({table})").fetchall()
    }
    forbidden = {
        "text",
        "query",
        "keys",
        "values",
        "kv_cache",
        "token_ids",
        "attention",
        "residual",
        "hidden_states",
    }
    assert not columns & forbidden
