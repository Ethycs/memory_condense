from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from memory_condense.associations.association_store import AssociationArtifact, AssociationStore
from memory_condense.persistence.db import Database


def _artifact(**updates) -> AssociationArtifact:
    fields = {
        "model_id": "Qwen/Qwen3-8B",
        "checkpoint_id": "bf16:first-shard:sha256-demo",
        "prefix_layers": 7,
        "head_layer": 1,
        "cav_layer": 5,
        "concept_names": ("context_dependency", "binding_constraint"),
        "head_count": 32,
        "metadata": {"selected_heads": [3, 7]},
    }
    fields.update(updates)
    return AssociationArtifact.create(**fields)


def _insert_chunks(db: Database, count: int = 5) -> list[str]:
    ids: list[str] = []
    for index in range(count):
        turn_id = f"turn-{index}"
        chunk_id = f"chunk-{index}"
        db.execute(
            "INSERT INTO turns (turn_id, role, text, created_at, ordinal) "
            "VALUES (?, 'user', ?, '2026-08-15T00:00:00+00:00', ?)",
            (turn_id, f"turn text {index}", index + 1),
        )
        db.execute(
            "INSERT INTO chunks "
            "(chunk_id, turn_id, text, start_char, end_char, token_count, "
            "embedding, hnsw_label) VALUES (?, ?, ?, 0, 12, 3, ?, ?)",
            (
                chunk_id,
                turn_id,
                f"chunk text {index}",
                np.asarray([index + 1], dtype=np.float32).tobytes(),
                index,
            ),
        )
        ids.append(chunk_id)
    db.commit()
    return ids


class TestArtifactIdentity:
    def test_id_is_stable_and_interpretation_is_versioned(self, db):
        store = AssociationStore(db)
        artifact = _artifact()
        assert artifact.artifact_id == _artifact().artifact_id
        assert artifact.artifact_id != _artifact(head_layer=2).artifact_id
        assert store.register_artifact(artifact).identity() == artifact.identity()

    def test_reusing_an_id_for_a_different_model_is_rejected(self, db):
        store = AssociationStore(db)
        artifact = _artifact()
        store.register_artifact(artifact)
        collision = AssociationArtifact(
            artifact_id=artifact.artifact_id,
            model_id="different/model",
            checkpoint_id=artifact.checkpoint_id,
            prefix_layers=artifact.prefix_layers,
            head_layer=artifact.head_layer,
            cav_layer=artifact.cav_layer,
            concept_names=artifact.concept_names,
            head_count=artifact.head_count,
            metadata=artifact.metadata,
        )
        with pytest.raises(ValueError, match="different model or interpretation"):
            store.register_artifact(collision)


class TestCompactSignatures:
    def test_float32_width_is_enforced_and_payload_is_reported(self, db):
        chunks = _insert_chunks(db, 2)
        store = AssociationStore(db)
        artifact = store.register_artifact(_artifact())
        with pytest.raises(ValueError, match="exactly 2"):
            store.put_signature(chunks[0], artifact.artifact_id, [0.5])

        store.put_signature(chunks[0], artifact.artifact_id, [1.0, -0.5])
        stored = store.get_signature(chunks[0], artifact.artifact_id)
        assert stored is not None
        assert stored.values == pytest.approx((1.0, -0.5))
        assert store.stats(artifact.artifact_id) == {
            "signatures": 1,
            "edges": 0,
            "cav_payload_bytes": 8,
            "head_payload_bytes": 0,
            "retained_request_token_state_bytes": 0,
            "retained_token_state_bytes": 0,
        }

    def test_neighbors_require_shared_positive_concepts(self, db):
        chunks = _insert_chunks(db, 4)
        store = AssociationStore(db)
        artifact = store.register_artifact(_artifact())
        signatures = ([1.0, -1.0], [0.9, -0.2], [-1.0, 1.0], [0.7, 0.7])
        for chunk_id, values in zip(chunks, signatures, strict=True):
            store.put_signature(chunk_id, artifact.artifact_id, values)

        hits = store.cav_neighbors(
            [chunks[0]], artifact.artifact_id, top_k=3
        )
        assert [hit.chunk_id for hit in hits] == [chunks[1], chunks[3]]
        assert all("context_dependency" in hit.shared_concepts for hit in hits)

    def test_batch_write_and_source_filtered_concept_members(self, db):
        chunks = _insert_chunks(db, 4)
        db.execute("UPDATE turns SET source_id = 'source-a' WHERE ordinal <= 2")
        db.execute("UPDATE turns SET source_id = 'source-b' WHERE ordinal > 2")
        db.commit()
        store = AssociationStore(db)
        artifact = store.register_artifact(
            _artifact(concept_names=("completed_event",))
        )

        assert store.put_signatures(
            artifact.artifact_id,
            list(zip(chunks, ([0.2], [0.9], [1.2], [-0.1]), strict=True)),
        ) == 4

        hits = store.concept_members(
            artifact.artifact_id,
            "completed_event",
            top_k=4,
            source_ids=("source-a",),
        )
        assert [hit.chunk_id for hit in hits] == [chunks[1], chunks[0]]
        assert [hit.score for hit in hits] == pytest.approx([0.9, 0.2])
        assert store.stats(artifact.artifact_id)["cav_payload_bytes"] == 16


class TestSparseEdges:
    def test_weighted_merge_and_restart_keep_only_compact_state(self, tmp_path):
        path = tmp_path / "restart.db"
        artifact = _artifact()
        with Database(path) as db:
            chunks = _insert_chunks(db, 2)
            store = AssociationStore(db)
            store.register_artifact(artifact)
            store.upsert_edge(
                chunks[0],
                chunks[1],
                artifact.artifact_id,
                [0.25] * 32,
                qk_score=0.2,
                ov_transport=1.0,
            )
            store.upsert_edge(
                chunks[0],
                chunks[1],
                artifact.artifact_id,
                [0.75] * 32,
                qk_score=0.6,
                ov_transport=3.0,
                evidence_count=3,
            )

        with Database(path) as reopened:
            store = AssociationStore(reopened)
            edge = store.neighbors(
                chunks[0], artifact.artifact_id, top_k=1
            )[0]
            assert edge.evidence_count == 4
            assert edge.qk_score == pytest.approx(0.5)
            assert edge.ov_transport == pytest.approx(2.5)
            assert edge.head_weights == pytest.approx((0.625,) * 32)
            stats = store.stats(artifact.artifact_id)
            assert stats["retained_request_token_state_bytes"] == 0
            assert stats["retained_token_state_bytes"] == 0

    def test_pruning_enforces_degree_and_uses_live_traversal_evidence(self, db):
        chunks = _insert_chunks(db, 4)
        store = AssociationStore(db)
        artifact = store.register_artifact(_artifact())
        for destination, score in zip(chunks[1:], (0.30, 0.20, 0.10), strict=True):
            store.upsert_edge(
                chunks[0],
                destination,
                artifact.artifact_id,
                [score] * 32,
                qk_score=score,
            )
        for _ in range(20):
            store.touch_edges(
                artifact.artifact_id,
                [(chunks[0], chunks[3])],
                now_turn=4,
            )

        assert store.prune_edges(
            artifact.artifact_id,
            2,
            now_turn=4,
            usage_weight=0.1,
        ) == 1
        kept = {
            edge.destination_chunk_id
            for edge in store.neighbors(
                chunks[0], artifact.artifact_id, top_k=10, now_turn=4
            )
        }
        assert kept == {chunks[0 + 1], chunks[3]}

    def test_many_anchor_lookup_matches_individual_neighbors(self, db):
        chunks = _insert_chunks(db, 4)
        store = AssociationStore(db)
        artifact = store.register_artifact(_artifact())
        store.upsert_edge(
            chunks[0], chunks[2], artifact.artifact_id, [0.5] * 32, qk_score=0.5
        )
        store.upsert_edge(
            chunks[1], chunks[3], artifact.artifact_id, [0.7] * 32, qk_score=0.7
        )
        many = store.neighbors_many(
            chunks[:2], artifact.artifact_id, top_k_per_source=2
        )
        assert many[chunks[0]] == store.neighbors(
            chunks[0], artifact.artifact_id, top_k=2
        )
        assert many[chunks[1]] == store.neighbors(
            chunks[1], artifact.artifact_id, top_k=2
        )

    def test_opt_in_neighbor_cache_is_invalidated_by_live_writes(self, db):
        chunks = _insert_chunks(db, 4)
        store = AssociationStore(db, cache_neighbors=True)
        artifact = store.register_artifact(_artifact())
        store.put_signature(chunks[0], artifact.artifact_id, [1.0, 0.0])
        store.put_signature(chunks[1], artifact.artifact_id, [0.9, 0.0])
        assert len(
            store.cav_neighbors([chunks[0]], artifact.artifact_id, top_k=5)
        ) == 1
        store.put_signature(chunks[2], artifact.artifact_id, [0.8, 0.0])
        assert len(
            store.cav_neighbors([chunks[0]], artifact.artifact_id, top_k=5)
        ) == 2

        assert store.neighbors(chunks[0], artifact.artifact_id, top_k=5) == ()
        store.upsert_edge(
            chunks[0], chunks[3], artifact.artifact_id, [0.6] * 32, qk_score=0.6
        )
        assert len(store.neighbors(chunks[0], artifact.artifact_id, top_k=5)) == 1

    def test_remove_chunk_clears_both_edge_directions_and_signature(self, db):
        chunks = _insert_chunks(db, 3)
        store = AssociationStore(db)
        artifact = store.register_artifact(_artifact())
        store.put_signature(chunks[1], artifact.artifact_id, [1.0, 0.0])
        store.upsert_edge(
            chunks[0],
            chunks[1],
            artifact.artifact_id,
            [0.5] * 32,
            qk_score=0.5,
            reverse=True,
        )
        store.upsert_edge(
            chunks[2],
            chunks[1],
            artifact.artifact_id,
            [0.4] * 32,
            qk_score=0.4,
        )

        assert store.remove_chunk_artifacts(chunks[1]) == 4
        assert store.stats(artifact.artifact_id)["signatures"] == 0
        assert store.stats(artifact.artifact_id)["edges"] == 0

    def test_foreign_keys_prevent_dangling_artifacts(self, db):
        store = AssociationStore(db)
        artifact = store.register_artifact(_artifact())
        with pytest.raises(sqlite3.IntegrityError):
            store.put_signature("missing", artifact.artifact_id, [1.0, 0.0])


def test_schema_has_no_token_state_columns(db):
    columns = {
        table: {
            row[1]
            for row in db.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for table in ("chunk_cav_signatures", "chunk_head_edges")
    }
    forbidden = {
        "keys",
        "values",
        "kv_cache",
        "token_ids",
        "attention",
        "residual",
        "hidden_states",
    }
    assert not any(names & forbidden for names in columns.values())
