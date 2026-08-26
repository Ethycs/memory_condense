from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import memory_condense.eval.hebbian_derived_store as derived_module
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.discourse import identity_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.eval.consolidation_replay import (
    FrozenQueryEmbedder,
    RetrievalAccessCapture,
    RetrievalAccessCaptureSink,
    RetrievalAccessEvent,
    _mint_retrieval_access_capture,
    stage_causal_store,
)
from memory_condense.eval.hebbian_derived_store import (
    DETERMINISTIC_CREATED_AT,
    MANIFEST_NAME,
    HebbianDerivedStoreError,
    HebbianLearningPolicy,
    apply_hebbian_history_to_staged_store,
    load_hebbian_derived_store_receipt,
    verify_hebbian_derived_store,
)
from memory_condense.eval.hebbian_history import (
    HebbianHistoryValidationError,
    seal_hebbian_history_artifact,
)
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)
from memory_condense.persistence.db import Database


SOURCE_RECEIPT_SHA256 = "a" * 64
IMPLEMENTATION_SHA256 = "b" * 64
ENVIRONMENT_SHA256 = "c" * 64


class _ReplayEmbedder:
    @property
    def dim(self) -> int:
        return 16

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(
                update={"embedding": self.embed_query(chunk.text).tolist()}
            )
            for chunk in chunks
        ]


def _build_store(
    root: Path,
    *,
    text_suffix: str = "",
    chunk_count: int = 14,
    user_ordinals: frozenset[int] = frozenset({3, 5, 14}),
) -> Path:
    root.mkdir()
    database_path = root / "memory.db"
    with Database(database_path) as database:
        for index in range(chunk_count):
            ordinal = index + 1
            turn_id = f"turn-{index}"
            chunk_id = f"chunk-{index}"
            text = f"SECRET source text {index}{text_suffix}"
            role = "user" if ordinal in user_ordinals else "assistant"
            database.execute(
                "INSERT INTO turns "
                "(turn_id, role, text, source_id, created_at, ordinal) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    turn_id,
                    role,
                    text,
                    f"session-{index // 2}",
                    f"2026-08-{ordinal + 1:02d}T00:00:00+00:00",
                    ordinal,
                ),
            )
            database.execute(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, token_count, "
                "embedding, lexical_weights, hnsw_label, term_count) "
                "VALUES (?, ?, ?, 0, ?, 4, ?, ?, ?, 1)",
                (
                    chunk_id,
                    turn_id,
                    text,
                    len(text),
                    bytes((index, 0, 0, 0)),
                    json.dumps({"source": 1}),
                    index,
                ),
            )
            database.execute(
                "INSERT INTO chunk_terms (term, chunk_id, tf) VALUES (?, ?, 1)",
                ("source", chunk_id),
            )
        database.commit()
        checkpoint = database.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert checkpoint is not None and int(checkpoint[0]) == 0
    (root / "hnsw_index.bin").write_bytes(b"deterministic-test-index-v1")
    return root


def _copy_stage(source: Path, destination: Path) -> Path:
    destination.mkdir()
    shutil.copy2(source / "memory.db", destination / "memory.db")
    shutil.copy2(source / "hnsw_index.bin", destination / "hnsw_index.bin")
    return destination


def _remove_zero_wal_sidecars(database_path: Path) -> None:
    wal_path = database_path.with_name(database_path.name + "-wal")
    if wal_path.exists():
        assert wal_path.stat().st_size == 0
        wal_path.unlink()
    shm_path = database_path.with_name(database_path.name + "-shm")
    if shm_path.exists():
        shm_path.unlink()


def _events() -> list[RetrievalAccessEvent]:
    return [
        RetrievalAccessEvent(
            event_id="causal-user:3",
            now_turn=2,
            chunk_ids=("chunk-0", "chunk-1"),
        ),
        RetrievalAccessEvent(
            event_id="causal-user:5",
            now_turn=4,
            chunk_ids=(),
        ),
        RetrievalAccessEvent(
            event_id="causal-user:14",
            now_turn=13,
            chunk_ids=tuple(f"chunk-{index}" for index in range(13)),
        ),
    ]


def _history_policy(retrieval_k: int) -> dict[str, object]:
    return {
        "format": "memory-condense.hebbian-capture-policy.v1",
        "retrieval_k": retrieval_k,
        "expansion_tokens": 1600,
        "max_prompt_tokens": 128,
        "direct_expansion_only": True,
        "event_id_scheme": "causal-user:{ordinal}",
        "capture_point": (
            "after_direct_context_pack_before_current_user_append"
        ),
        "exclude_current_and_future_turns": True,
        "query_embedding_model_id": DEFAULT_MODEL_NAME,
        "query_embedding_model_revision": DEFAULT_MODEL_REVISION,
        "query_embedding_checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "query_embedding_execution_sha256": "d" * 64,
    }


def _history(
    source: Path,
    events=None,
    *,
    retrieval_k: int = 14,
    capture: RetrievalAccessCapture | None = None,
):
    policy = _history_policy(retrieval_k)
    if capture is None:
        capture = _mint_retrieval_access_capture(
            source_database_sha256=file_sha256(source / "memory.db"),
            capture_policy_sha256=identity_sha256(policy),
            retrieval_k=retrieval_k,
            expansion_tokens=1600,
            max_prompt_tokens=128,
            events=_events() if events is None else events,
        )
    return seal_hebbian_history_artifact(
        capture,
        source_database_path=source / "memory.db",
        source_store_receipt_sha256=SOURCE_RECEIPT_SHA256,
        implementation_sha256=IMPLEMENTATION_SHA256,
        environment_lock_sha256=ENVIRONMENT_SHA256,
        capture_policy_payload=policy,
    )


def _apply(
    source: Path,
    stage: Path,
    *,
    events=None,
    capture: RetrievalAccessCapture | None = None,
    policy=None,
    retrieval_k: int = 14,
):
    return apply_hebbian_history_to_staged_store(
        stage,
        source_database_path=source / "memory.db",
        source_index_path=source / "hnsw_index.bin",
        history=_history(
            source,
            events=events,
            retrieval_k=retrieval_k,
            capture=capture,
        ),
        policy=policy,
    )


def test_applies_bounded_rank_history_and_publishes_text_free_store(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _copy_stage(source, tmp_path / "derived")
    source_database_before = file_sha256(source / "memory.db")
    source_index_before = file_sha256(source / "hnsw_index.bin")
    policy = HebbianLearningPolicy(max_concepts_per_event=2)

    receipt = _apply(source, stage, policy=policy)

    assert receipt.events_offered == 3
    assert receipt.events_applied == 3
    assert receipt.graph_event_receipts == 3
    assert receipt.graph_nodes == 2
    assert receipt.graph_edges == 1
    assert receipt.retained_request_token_state_bytes == 0
    assert receipt.learning_policy == policy
    assert receipt.source_turn_sequence_sha256 == (
        receipt.derived_turn_sequence_sha256
    )
    assert receipt.source_chunk_sequence_sha256 == (
        receipt.derived_chunk_sequence_sha256
    )
    assert file_sha256(source / "memory.db") == source_database_before
    assert file_sha256(source / "hnsw_index.bin") == source_index_before
    assert not (stage / "memory.db-wal").exists()
    assert not (stage / "memory.db-shm").exists()
    assert verify_hebbian_derived_store(stage, expected=receipt) is receipt

    manifest_text = (stage / MANIFEST_NAME).read_text(encoding="utf-8")
    assert "SECRET source text" not in manifest_text
    assert '"query"' not in manifest_text
    assert '"text"' not in manifest_text

    connection = sqlite3.connect(
        f"{(stage / 'memory.db').resolve().as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        artifact = connection.execute(
            "SELECT created_at, metadata FROM association_artifacts "
            "WHERE artifact_id = ?",
            (receipt.association_artifact_id,),
        ).fetchone()
        event_members = connection.execute(
            "SELECT event_id, member_count FROM hebbian_access_events "
            "WHERE artifact_id = ? ORDER BY observed_turn, rowid",
            (receipt.association_artifact_id,),
        ).fetchall()
        node_ids = connection.execute(
            "SELECT chunk_id FROM hebbian_chunk_nodes WHERE artifact_id = ? "
            "ORDER BY chunk_id",
            (receipt.association_artifact_id,),
        ).fetchall()
    finally:
        connection.close()
    assert artifact is not None
    assert artifact[0] == DETERMINISTIC_CREATED_AT
    assert json.loads(artifact[1])["history_artifact_sha256"] == (
        receipt.history_artifact_sha256
    )
    assert event_members == [
        ("causal-user:3", 2),
        ("causal-user:5", 0),
        ("causal-user:14", 2),
    ]
    assert node_ids == [("chunk-0",), ("chunk-1",)]


def test_accepts_the_exact_output_of_stage_causal_store(tmp_path):
    source = tmp_path / "source"
    stage = tmp_path / "derived"
    live = _ReplayEmbedder()
    turns = [
        ("assistant", "alpha evidence existed before either question"),
        ("user", "What does alpha establish?"),
        ("assistant", "beta evidence appeared after the first question"),
        ("user", "How do alpha and beta relate?"),
    ]
    with MemoryCondenser(
        data_dir=source,
        embedder=live,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=40,
    ) as condenser:
        for role, text in turns:
            condenser.ingest(role, text)
    queries = [text for role, text in turns if role == "user"]
    frozen = FrozenQueryEmbedder(
        {query: live.embed_query(query) for query in queries}
    )
    events: list[RetrievalAccessEvent] = []
    capture_sink = RetrievalAccessCaptureSink()
    stage_causal_store(
        source / "memory.db",
        stage,
        frozen,
        retrieval_k=3,
        max_event_nodes=3,
        new_event_nodes=1,
        max_prompt_tokens=128,
        retrieval_access_events=events,
        retrieval_access_capture_sink=capture_sink,
        retrieval_access_capture_policy_sha256=identity_sha256(
            _history_policy(3)
        ),
    )

    receipt = _apply(
        source,
        stage,
        events=events,
        capture=capture_sink.capture,
        retrieval_k=3,
    )

    assert receipt.events_offered == len(events) == 2
    assert receipt.events_applied == len(events)
    assert receipt.source_chunk_sequence_sha256 == (
        receipt.derived_chunk_sequence_sha256
    )
    assert verify_hebbian_derived_store(stage, expected=receipt) is receipt


def test_empty_history_still_seals_an_isolated_zero_state_namespace(tmp_path):
    source = _build_store(
        tmp_path / "source",
        user_ordinals=frozenset(),
    )
    stage = _copy_stage(source, tmp_path / "derived")

    receipt = _apply(source, stage, events=[])

    assert receipt.events_offered == 0
    assert receipt.events_applied == 0
    assert receipt.graph_event_receipts == 0
    assert receipt.graph_nodes == 0
    assert receipt.graph_edges == 0
    assert verify_hebbian_derived_store(stage, expected=receipt) is receipt


def test_receipt_json_loader_round_trips_the_published_phase_boundary(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _copy_stage(source, tmp_path / "derived")
    receipt = _apply(source, stage)
    payload = json.loads((stage / MANIFEST_NAME).read_bytes())

    loaded = load_hebbian_derived_store_receipt(payload)

    assert loaded == receipt
    assert loaded.payload() == payload
    assert verify_hebbian_derived_store(stage, expected=loaded) is loaded


def test_receipt_json_loader_rejects_shape_type_and_seal_tampering(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _copy_stage(source, tmp_path / "derived")
    receipt = _apply(source, stage)

    extra = receipt.payload()
    extra["unexpected"] = True
    with pytest.raises(HebbianDerivedStoreError, match="noncanonical shape"):
        load_hebbian_derived_store_receipt(extra)

    integer_float = receipt.payload()
    integer_float["learning_policy"]["learning_rate"] = 1
    with pytest.raises(HebbianDerivedStoreError, match="exact JSON float"):
        load_hebbian_derived_store_receipt(integer_float)

    changed_count = receipt.payload()
    changed_count["events_applied"] = receipt.events_applied + 1
    with pytest.raises(HebbianDerivedStoreError, match="every frozen history"):
        load_hebbian_derived_store_receipt(changed_count)

    resealed_wrong_artifact = receipt.payload()
    resealed_wrong_artifact["association_artifact_id"] = "assoc-wrong"
    resealed_wrong_artifact["receipt_sha256"] = identity_sha256(
        {
            key: value
            for key, value in resealed_wrong_artifact.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(HebbianDerivedStoreError, match="artifact ID mismatch"):
        load_hebbian_derived_store_receipt(resealed_wrong_artifact)


def test_verify_rejects_json_numeric_type_drift_in_manifest(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _copy_stage(source, tmp_path / "derived")
    receipt = _apply(source, stage)
    changed = receipt.payload()
    changed["learning_policy"]["learning_rate"] = 1
    (stage / MANIFEST_NAME).write_bytes(derived_module._canonical_bytes(changed))

    with pytest.raises(HebbianDerivedStoreError, match="differs from its receipt"):
        verify_hebbian_derived_store(stage, expected=receipt)


def test_rejects_hard_link_alias_before_source_can_be_mutated(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = tmp_path / "derived"
    stage.mkdir()
    try:
        os.link(source / "memory.db", stage / "memory.db")
    except OSError as exc:  # pragma: no cover - platform permission fallback
        pytest.skip(f"hard links unavailable: {exc}")
    shutil.copy2(source / "hnsw_index.bin", stage / "hnsw_index.bin")
    before = file_sha256(source / "memory.db")

    with pytest.raises(HebbianDerivedStoreError, match="must not alias"):
        _apply(source, stage)

    assert file_sha256(source / "memory.db") == before
    assert not (stage / MANIFEST_NAME).exists()
    connection = sqlite3.connect(source / "memory.db")
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM association_artifacts"
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_rejects_staged_database_index_alias_before_learning(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = tmp_path / "derived"
    stage.mkdir()
    shutil.copy2(source / "memory.db", stage / "memory.db")
    try:
        os.link(stage / "memory.db", stage / "hnsw_index.bin")
    except OSError as exc:  # pragma: no cover - platform permission fallback
        pytest.skip(f"hard links unavailable: {exc}")
    before = file_sha256(stage / "memory.db")

    with pytest.raises(HebbianDerivedStoreError, match="database and index"):
        _apply(source, stage)

    assert file_sha256(stage / "memory.db") == before
    assert not (stage / MANIFEST_NAME).exists()


def test_rejects_source_sqlite_sidecars_before_learning(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _copy_stage(source, tmp_path / "derived")
    (source / "memory.db-wal").write_bytes(b"")
    before = file_sha256(stage / "memory.db")

    with pytest.raises(
        (HebbianDerivedStoreError, HebbianHistoryValidationError),
        match="source database retained",
    ):
        _apply(source, stage)

    assert file_sha256(stage / "memory.db") == before
    assert not (stage / MANIFEST_NAME).exists()


def test_rejects_changed_staged_source_sequence_before_learning(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _build_store(tmp_path / "derived", text_suffix=" CHANGED")
    before = file_sha256(stage / "memory.db")

    with pytest.raises(HebbianDerivedStoreError, match="turn or chunk sequence"):
        _apply(source, stage)

    assert file_sha256(stage / "memory.db") == before
    assert not (stage / MANIFEST_NAME).exists()


def test_rejects_changed_staged_retrieval_material_before_learning(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _copy_stage(source, tmp_path / "derived")
    database_path = stage / "memory.db"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "UPDATE chunks SET embedding = ? WHERE chunk_id = 'chunk-0'",
            (b"changed-embedding",),
        )
        connection.commit()
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert checkpoint is not None and int(checkpoint[0]) == 0
    finally:
        connection.close()
    _remove_zero_wal_sidecars(database_path)
    before = file_sha256(database_path)

    with pytest.raises(HebbianDerivedStoreError, match="turn or chunk sequence"):
        _apply(source, stage)

    assert file_sha256(database_path) == before
    assert not (stage / MANIFEST_NAME).exists()


def test_rejects_preexisting_graph_state_before_learning(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _copy_stage(source, tmp_path / "derived")
    database_path = stage / "memory.db"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "INSERT INTO consolidation_access_events "
            "(event_id, observed_turn, event_fingerprint, member_count) "
            "VALUES ('contaminant', 1, ?, 0)",
            ("d" * 64,),
        )
        connection.commit()
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert checkpoint is not None and int(checkpoint[0]) == 0
    finally:
        connection.close()
    _remove_zero_wal_sidecars(database_path)
    before = file_sha256(database_path)

    with pytest.raises(HebbianDerivedStoreError, match="pre-existing graph state"):
        _apply(source, stage)

    assert file_sha256(database_path) == before
    assert not (stage / MANIFEST_NAME).exists()


def test_rejects_unexpected_staged_children_without_writing_graph(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _copy_stage(source, tmp_path / "derived")
    (stage / "debug.txt").write_text("unexpected", encoding="utf-8")
    before = file_sha256(stage / "memory.db")

    with pytest.raises(HebbianDerivedStoreError, match="unexpected or missing"):
        _apply(source, stage)

    assert file_sha256(stage / "memory.db") == before
    assert not (stage / MANIFEST_NAME).exists()


def test_verify_reproves_exact_association_artifact_row(tmp_path):
    source = _build_store(tmp_path / "source")
    stage = _copy_stage(source, tmp_path / "derived")
    receipt = _apply(source, stage)
    database_path = stage / "memory.db"

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "UPDATE association_artifacts SET created_at = ? "
            "WHERE artifact_id = ?",
            ("2030-01-01T00:00:00+00:00", receipt.association_artifact_id),
        )
        connection.commit()
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert checkpoint is not None and int(checkpoint[0]) == 0
    finally:
        connection.close()
    _remove_zero_wal_sidecars(database_path)

    unsigned = replace(
        receipt,
        derived_database_sha256=file_sha256(database_path),
        receipt_sha256="0" * 64,
    )
    tampered_receipt = replace(
        unsigned,
        receipt_sha256=identity_sha256(unsigned.payload(include_seal=False)),
    )
    (stage / MANIFEST_NAME).write_bytes(
        derived_module._canonical_bytes(tampered_receipt.payload())
    )

    with pytest.raises(
        HebbianDerivedStoreError,
        match="association artifact changed",
    ):
        verify_hebbian_derived_store(stage, expected=tampered_receipt)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": True}, "learning_rate must be numeric"),
        (
            {"max_concepts_per_event": True},
            "max_concepts_per_event must be an exact integer",
        ),
        ({"min_edge_score": 1.1}, "must not exceed one"),
        ({"retain_all_event_receipts": False}, "must retain every"),
    ],
)
def test_learning_policy_fails_closed_on_noncanonical_values(kwargs, message):
    with pytest.raises(HebbianDerivedStoreError, match=message):
        HebbianLearningPolicy(**kwargs)
