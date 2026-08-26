from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from memory_condense.application.retrieval_workflow import RetrievalWorkflowMixin
from memory_condense.associations.association_store import (
    AssociationArtifact,
    AssociationStore,
)
from memory_condense.associations.hebbian_retrieval import (
    HebbianExpansionReceipt,
    HebbianNeighborCandidateReceipt,
    expand_hebbian_results,
    expand_hebbian_results_with_receipt,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult
from memory_condense.persistence.db import Database


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

    expanded, receipt = expand_hebbian_results_with_receipt(
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
    assert receipt.status == "replaced"
    assert receipt.base_chunk_ids == ("chunk-0", "chunk-1")
    assert receipt.base_activations == (
        ("chunk-0", 1.0),
        ("chunk-1", pytest.approx(2**-0.5)),
    )
    assert receipt.protected_chunk_ids == ()
    assert receipt.replaceable_chunk_ids == ("chunk-1", "chunk-0")
    assert receipt.proposed_removed_chunk_ids == ("chunk-1",)
    assert receipt.proposed_added_chunk_ids == ("chunk-2",)
    assert receipt.removed_chunk_ids == ("chunk-1",)
    assert receipt.added_chunk_ids == ("chunk-2",)
    assert receipt.final_chunk_ids == ("chunk-0", "chunk-2")
    assert receipt.base_chunk_token_total == 9
    assert receipt.proposed_chunk_token_total == 10
    assert receipt.final_chunk_token_total == 10
    assert receipt.retained_request_token_state_bytes == 0
    assert len(receipt.receipt_sha256) == 64
    assert receipt.neighbor_candidates == (
        HebbianNeighborCandidateReceipt(
            rank=1,
            chunk_id="chunk-2",
            score=1.0,
            support=1,
            anchor_chunk_id="chunk-0",
            coaccess_count=1,
            last_reinforced_turn=4,
        ),
    )

    compatible = expand_hebbian_results(
        anchors,
        artifact.artifact_id,
        store=store,
        hydrate=hydrate,
        now_turn=4,
        k=2,
        hebbian_slots=1,
        max_prompt_token_increase=1,
    )
    assert compatible == expanded

    replayed, replay_receipt = expand_hebbian_results_with_receipt(
        anchors,
        artifact.artifact_id,
        store=store,
        hydrate=hydrate,
        now_turn=4,
        k=2,
        hebbian_slots=1,
        max_prompt_token_increase=1,
    )
    assert replayed == expanded
    assert replay_receipt.receipt_sha256 == receipt.receipt_sha256
    with pytest.raises(FrozenInstanceError):
        receipt.status = "no_neighbor"  # type: ignore[misc]
    with pytest.raises(ValueError, match="does not match"):
        replace(receipt, min_score=0.06)

    rolled_back, rollback_receipt = expand_hebbian_results_with_receipt(
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
    assert rollback_receipt.status == "token_budget_rollback"
    assert rollback_receipt.proposed_removed_chunk_ids == ("chunk-1",)
    assert rollback_receipt.proposed_added_chunk_ids == ("chunk-2",)
    assert rollback_receipt.removed_chunk_ids == ()
    assert rollback_receipt.added_chunk_ids == ()
    assert rollback_receipt.final_chunk_ids == rollback_receipt.base_chunk_ids
    assert rollback_receipt.proposed_chunk_token_total == 10
    assert rollback_receipt.final_chunk_token_total == 9


def test_expansion_receipt_covers_bounded_noop_outcomes(db):
    chunks = _insert_chunks(db, 4)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "learn-a-c",
        {"chunk-0": 1.0, "chunk-2": 1.0},
        now_turn=4,
    )
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "learn-a-d",
        {"chunk-0": 1.0, "chunk-3": 1.0},
        now_turn=4,
    )
    by_id = {
        chunk_id: _result(chunk, 0.0) for chunk_id, chunk in chunks.items()
    }

    _, no_slot = expand_hebbian_results_with_receipt(
        [_result(chunks["chunk-0"], 0.9)],
        artifact.artifact_id,
        store=store,
        hydrate=lambda chunk_id, **_kwargs: by_id.get(chunk_id),
        now_turn=4,
        hebbian_slots=0,
    )
    assert no_slot.status == "no_slot"
    assert no_slot.neighbor_candidates == ()
    assert no_slot.base_activations == ()
    assert no_slot.protected_chunk_ids == ()
    assert no_slot.replaceable_chunk_ids == ()
    with pytest.raises(ValueError, match="no_slot receipt"):
        replace(
            no_slot,
            base_activations=(("chunk-0", 1.0),),
            protected_chunk_ids=("chunk-0",),
            receipt_sha256="",
        )

    _, no_neighbor = expand_hebbian_results_with_receipt(
        [_result(chunks["chunk-1"], 0.9)],
        artifact.artifact_id,
        store=store,
        hydrate=lambda chunk_id, **_kwargs: by_id.get(chunk_id),
        now_turn=4,
    )
    assert no_neighbor.status == "no_neighbor"
    assert no_neighbor.base_activations == (("chunk-1", 1.0),)
    assert no_neighbor.protected_chunk_ids == ()
    assert no_neighbor.replaceable_chunk_ids == ()
    assert no_neighbor.hydration_failed_chunk_ids == ()
    with pytest.raises(ValueError, match="no_neighbor receipt"):
        replace(
            no_neighbor,
            protected_chunk_ids=("chunk-1",),
            receipt_sha256="",
        )
    with pytest.raises(ValueError, match="no_neighbor receipt"):
        replace(
            no_neighbor,
            base_activations=(),
            receipt_sha256="",
        )

    protected = [
        _result(chunks["chunk-0"], 0.9, lexical=0.95),
        _result(chunks["chunk-1"], 0.8, lexical=0.99),
    ]
    _, all_protected = expand_hebbian_results_with_receipt(
        protected,
        artifact.artifact_id,
        store=store,
        hydrate=lambda chunk_id, **_kwargs: by_id.get(chunk_id),
        now_turn=4,
        lexical_protection_threshold=0.9,
    )
    assert all_protected.status == "all_protected"
    assert all_protected.protected_chunk_ids == ("chunk-0", "chunk-1")
    assert all_protected.replaceable_chunk_ids == ()

    _, hydration_failed = expand_hebbian_results_with_receipt(
        [_result(chunks["chunk-0"], 0.9)],
        artifact.artifact_id,
        store=store,
        hydrate=lambda _chunk_id, **_kwargs: None,
        now_turn=4,
    )
    assert hydration_failed.status == "hydration_failed"
    candidate_ids = tuple(
        candidate.chunk_id for candidate in hydration_failed.neighbor_candidates
    )
    assert len(candidate_ids) == 2
    assert hydration_failed.hydration_failed_chunk_ids == candidate_ids
    assert hydration_failed.final_chunk_ids == hydration_failed.base_chunk_ids
    with pytest.raises(ValueError, match="hydration_failed receipt"):
        replace(
            hydration_failed,
            hydration_failed_chunk_ids=candidate_ids[:1],
            receipt_sha256="",
        )
    with pytest.raises(ValueError, match="hydration_failed receipt"):
        replace(
            hydration_failed,
            protected_chunk_ids=hydration_failed.base_chunk_ids,
            replaceable_chunk_ids=(),
            receipt_sha256="",
        )


def test_zero_cap_receipt_validates_artifact_but_legacy_wrapper_stays_noop(db):
    store = AssociationStore(db)
    hydrate = lambda _chunk_id, **_kwargs: None

    assert expand_hebbian_results(
        [],
        "assoc-does-not-exist",
        store=store,
        hydrate=hydrate,
        now_turn=0,
        k=0,
    ) == []
    assert expand_hebbian_results(
        [],
        "assoc-does-not-exist",
        store=store,
        hydrate=hydrate,
        now_turn=0,
        k=0.5,
    ) == []
    with pytest.raises(KeyError, match="unknown association artifact"):
        expand_hebbian_results_with_receipt(
            [],
            "assoc-does-not-exist",
            store=store,
            hydrate=hydrate,
            now_turn=0,
            k=0,
        )

    artifact = store.register_artifact(_artifact())
    assert expand_hebbian_results(
        [],
        artifact.artifact_id,
        store=store,
        hydrate=hydrate,
        now_turn="0",  # type: ignore[arg-type]
        k="1",  # type: ignore[arg-type]
        hebbian_slots="0",  # type: ignore[arg-type]
        max_seed_concepts="1",  # type: ignore[arg-type]
        max_candidates="1",  # type: ignore[arg-type]
        half_life_turns="200",  # type: ignore[arg-type]
        min_score="0.05",  # type: ignore[arg-type]
        max_prompt_token_increase="0",  # type: ignore[arg-type]
    ) == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("now_turn", True),
        ("k", 1.0),
        ("hebbian_slots", 1.0),
        ("max_seed_concepts", True),
        ("max_candidates", "2"),
        ("half_life_turns", "200.0"),
        ("min_score", False),
        ("lexical_protection_threshold", "0.5"),
        ("max_prompt_token_increase", 0.0),
    ],
)
def test_receipt_api_rejects_nonexact_request_types(
    db,
    field_name,
    value,
):
    chunks = _insert_chunks(db, 1)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    request = {
        "store": store,
        "hydrate": lambda _chunk_id, **_kwargs: None,
        "now_turn": 1,
        "k": 1,
        "hebbian_slots": 1,
        "max_seed_concepts": 1,
        "max_candidates": 1,
        "half_life_turns": 200.0,
        "min_score": 0.05,
        "lexical_protection_threshold": None,
        "max_prompt_token_increase": 0,
    }
    request[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        expand_hebbian_results_with_receipt(
            [_result(chunks["chunk-0"], 1.0)],
            artifact.artifact_id,
            **request,
        )


def test_expansion_never_uses_more_than_the_reserved_hebbian_slots(db):
    chunks = _insert_chunks(db, 4)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "learn-two-candidates",
        {"chunk-0": 1.0, "chunk-2": 1.0, "chunk-3": 1.0},
        now_turn=4,
    )
    anchors = [
        _result(chunks["chunk-0"], 0.9),
        _result(chunks["chunk-1"], 0.8),
    ]
    by_id = {
        chunk_id: _result(chunk, 0.0) for chunk_id, chunk in chunks.items()
    }

    expanded, receipt = expand_hebbian_results_with_receipt(
        anchors,
        artifact.artifact_id,
        store=store,
        hydrate=lambda chunk_id, **_kwargs: by_id.get(chunk_id),
        now_turn=4,
        hebbian_slots=1,
    )

    assert receipt.status == "replaced"
    assert len(receipt.neighbor_candidates) == 2
    assert len(receipt.removed_chunk_ids) == 1
    assert len(receipt.added_chunk_ids) == 1
    assert len(expanded) == len(anchors)


def test_expansion_rejects_mismatched_hydration_and_inconsistent_receipts(db):
    chunks = _insert_chunks(db, 3)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "learn-a-c",
        {"chunk-0": 1.0, "chunk-2": 1.0},
        now_turn=3,
    )
    anchors = [
        _result(chunks["chunk-0"], 0.9),
        _result(chunks["chunk-1"], 0.8),
    ]

    expanded, receipt = expand_hebbian_results_with_receipt(
        anchors,
        artifact.artifact_id,
        store=store,
        hydrate=lambda _chunk_id, **_kwargs: _result(chunks["chunk-0"], 0.0),
        now_turn=3,
    )

    assert expanded == anchors
    assert receipt.status == "hydration_failed"
    assert receipt.hydration_failed_chunk_ids == ("chunk-2",)
    assert receipt.final_chunk_ids == ("chunk-0", "chunk-1")
    with pytest.raises(ValueError, match="no_neighbor receipt"):
        replace(receipt, status="no_neighbor", receipt_sha256="")


def test_read_seed_cap_does_not_shrink_tail_replacement_window(db):
    chunks = _insert_chunks(db, 15)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "learn-first-to-new",
        {"chunk-0": 1.0, "chunk-14": 1.0},
        now_turn=15,
    )
    anchors = [
        _result(chunks[f"chunk-{index}"], 1.0 - index / 100.0)
        for index in range(14)
    ]
    by_id = {
        chunk_id: _result(chunk, 0.0) for chunk_id, chunk in chunks.items()
    }
    expanded, receipt = expand_hebbian_results_with_receipt(
        anchors,
        artifact.artifact_id,
        store=store,
        hydrate=lambda chunk_id, **_kwargs: by_id.get(chunk_id),
        now_turn=15,
        k=14,
    )

    assert receipt.max_seed_concepts == 12
    assert tuple(chunk_id for chunk_id, _ in receipt.base_activations) == tuple(
        f"chunk-{index}" for index in range(12)
    )
    assert receipt.replaceable_chunk_ids[0] == "chunk-13"
    assert receipt.removed_chunk_ids == ("chunk-13",)
    assert receipt.added_chunk_ids == ("chunk-14",)
    assert [item.chunk.chunk_id for item in expanded][-1] == "chunk-14"

    with pytest.raises(ValueError, match="max_seed_concepts must be positive"):
        expand_hebbian_results_with_receipt(
            anchors,
            artifact.artifact_id,
            store=store,
            hydrate=lambda chunk_id, **_kwargs: by_id.get(chunk_id),
            now_turn=15,
            max_seed_concepts=0,
        )


def test_read_seed_cap_still_excludes_every_existing_anchor(db):
    chunks = _insert_chunks(db, 14)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "learn-first-to-lower-anchor",
        {"chunk-0": 1.0, "chunk-13": 1.0},
        now_turn=14,
    )
    anchors = [
        _result(chunks[f"chunk-{index}"], 1.0 - index / 100.0)
        for index in range(14)
    ]

    expanded, receipt = expand_hebbian_results_with_receipt(
        anchors,
        artifact.artifact_id,
        store=store,
        hydrate=lambda chunk_id, **_kwargs: _result(chunks[chunk_id], 0.0),
        now_turn=14,
        k=14,
    )

    assert receipt.status == "no_neighbor"
    assert receipt.neighbor_candidates == ()
    assert receipt.final_chunk_ids == receipt.base_chunk_ids
    assert [item.chunk.chunk_id for item in expanded] == list(
        receipt.base_chunk_ids
    )


def test_workflow_exposes_receipt_returning_hebbian_seam(db):
    chunks = _insert_chunks(db, 3)
    store = AssociationStore(db)
    artifact = store.register_artifact(_artifact())
    store.reinforce_retrieval_coaccess(
        artifact.artifact_id,
        "learn-a-c",
        {"chunk-0": 1.0, "chunk-2": 1.0},
        now_turn=3,
    )
    by_id = {
        chunk_id: _result(chunk, 0.0) for chunk_id, chunk in chunks.items()
    }
    workflow = RetrievalWorkflowMixin()
    workflow._associations = store
    workflow._retriever = SimpleNamespace(
        hydrate_chunk=lambda chunk_id, **_kwargs: by_id.get(chunk_id)
    )
    workflow._db = db

    expanded, receipt = workflow.expand_hebbian_with_receipt(
        [
            _result(chunks["chunk-0"], 0.9),
            _result(chunks["chunk-1"], 0.8),
        ],
        artifact.artifact_id,
        k=2,
        max_prompt_token_increase=1,
    )

    assert [item.chunk.chunk_id for item in expanded] == ["chunk-0", "chunk-2"]
    assert receipt.status == "replaced"
    assert receipt.now_turn == db.current_turn()


def test_expansion_receipt_schema_has_no_request_or_transformer_payload_fields():
    names = {
        item.name
        for cls in (HebbianExpansionReceipt, HebbianNeighborCandidateReceipt)
        for item in fields(cls)
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
        "tensor",
    }
    assert not names & forbidden


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
