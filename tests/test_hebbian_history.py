from __future__ import annotations

import json
import sqlite3
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import pytest

import memory_condense.eval.consolidation_replay as replay_module
import memory_condense.eval.hebbian_history as history_module
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.consolidation_replay import RetrievalAccessEvent
from memory_condense.eval.hebbian_history import (
    CAPTURE_POLICY_FORMAT,
    EVENT_POPULATION_FORMAT,
    HISTORY_ARTIFACT_FORMAT,
    HISTORY_RECEIPT_FORMAT,
    HebbianHistoryValidationError,
    load_hebbian_history_artifact,
    seal_hebbian_history_artifact,
    verify_hebbian_history_artifact,
)
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)


SOURCE_RECEIPT_SHA256 = "a" * 64
IMPLEMENTATION_SHA256 = "b" * 64
ENVIRONMENT_SHA256 = "c" * 64


def _source_database(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE turns (
                turn_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                text TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO turns (turn_id, ordinal, role, text) VALUES (?, ?, ?, ?)",
            (
                ("turn-1", 1, "system", "SECRET TRANSCRIPT ALPHA"),
                ("turn-2", 2, "user", "SECRET TRANSCRIPT BETA"),
                ("turn-3", 3, "assistant", "SECRET TRANSCRIPT GAMMA"),
                ("turn-4", 4, "user", "SECRET TRANSCRIPT DELTA"),
                ("turn-5", 5, "assistant", "SECRET TRANSCRIPT EPSILON"),
                ("turn-6", 6, "user", "SECRET TRANSCRIPT ZETA"),
            ),
        )
        connection.executemany(
            "INSERT INTO chunks (chunk_id, turn_id, text) VALUES (?, ?, ?)",
            (
                ("chunk-0", "turn-1", "secret chunk zero"),
                ("chunk-1", "turn-2", "secret chunk one"),
                ("chunk-2", "turn-3", "secret chunk two"),
            ),
        )
    return path


def _events() -> list[RetrievalAccessEvent]:
    return [
        RetrievalAccessEvent(
            event_id="causal-user:2",
            now_turn=1,
            chunk_ids=("chunk-0",),
        ),
        RetrievalAccessEvent(
            event_id="causal-user:4",
            now_turn=3,
            chunk_ids=(),
        ),
        RetrievalAccessEvent(
            event_id="causal-user:6",
            now_turn=5,
            chunk_ids=("chunk-0", "chunk-2"),
        ),
    ]


def _policy() -> dict[str, object]:
    return {
        "format": CAPTURE_POLICY_FORMAT,
        "retrieval_k": 12,
        "expansion_tokens": 1600,
        "max_prompt_tokens": 128,
        "direct_expansion_only": True,
        "event_id_scheme": "causal-user:{ordinal}",
        "capture_point": "after_direct_context_pack_before_current_user_append",
        "exclude_current_and_future_turns": True,
        "query_embedding_model_id": DEFAULT_MODEL_NAME,
        "query_embedding_model_revision": DEFAULT_MODEL_REVISION,
        "query_embedding_checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "query_embedding_execution_sha256": "d" * 64,
    }


def _seal(database: Path, events=None, policy=None):
    try:
        capture = replay_module._mint_retrieval_access_capture(
            source_database_sha256=file_sha256(database),
            capture_policy_sha256=identity_sha256(_policy()),
            retrieval_k=12,
            expansion_tokens=1600,
            max_prompt_tokens=128,
            events=_events() if events is None else events,
        )
    except replay_module.RetrievalAccessCaptureValidationError as exc:
        raise HebbianHistoryValidationError(str(exc)) from exc
    return seal_hebbian_history_artifact(
        capture,
        source_database_path=database,
        source_store_receipt_sha256=SOURCE_RECEIPT_SHA256,
        implementation_sha256=IMPLEMENTATION_SHA256,
        environment_lock_sha256=ENVIRONMENT_SHA256,
        capture_policy_payload=_policy() if policy is None else policy,
    )


def test_seal_is_immutable_text_free_and_binds_all_provenance(tmp_path):
    database = _source_database(tmp_path / "memory.db")
    policy = _policy()
    artifact = _seal(database, policy=policy)

    assert artifact.format == HISTORY_ARTIFACT_FORMAT
    assert artifact.receipt.format == HISTORY_RECEIPT_FORMAT
    assert artifact.receipt.source_database_sha256 == file_sha256(database)
    assert (
        artifact.receipt.source_store_receipt_sha256 == SOURCE_RECEIPT_SHA256
    )
    assert artifact.receipt.implementation_sha256 == IMPLEMENTATION_SHA256
    assert artifact.receipt.environment_lock_sha256 == ENVIRONMENT_SHA256
    assert artifact.receipt.capture_policy_sha256 == identity_sha256(policy)
    capture_sha256 = replay_module.retrieval_access_capture_sha256(
        source_database_sha256=file_sha256(database),
        capture_policy_sha256=identity_sha256(policy),
        retrieval_k=12,
        expansion_tokens=1600,
        max_prompt_tokens=128,
        events=artifact.events,
    )
    assert artifact.receipt.direct_capture_sha256 == capture_sha256
    assert len(capture_sha256) == 64
    assert artifact.receipt.ordered_event_sha256s == tuple(
        event.event_sha256 for event in artifact.events
    )
    assert artifact.receipt.event_population_sha256 == identity_sha256(
        {
            "format": EVENT_POPULATION_FORMAT,
            "events": [
                {
                    "event_id": event.event_id,
                    "now_turn": event.now_turn,
                    "event_sha256": event.event_sha256,
                }
                for event in artifact.events
            ],
        }
    )
    assert artifact.receipt.event_count == 3
    assert artifact.receipt.empty_event_count == 1
    assert artifact.receipt.retained_request_token_state_bytes == 0
    assert verify_hebbian_history_artifact(artifact) is artifact
    assert (
        verify_hebbian_history_artifact(
            artifact,
            source_database_path=database,
        )
        is artifact
    )

    serialized = json.dumps(artifact.payload(), sort_keys=True)
    assert "SECRET TRANSCRIPT" not in serialized
    assert "secret chunk" not in serialized
    assert '"query"' not in serialized
    assert '"text"' not in serialized
    assert set(artifact.payload()) == {
        "format",
        "capture_policy_payload",
        "events",
        "receipt",
        "artifact_sha256",
    }

    policy["retrieval_k"] = 999
    assert artifact.capture_policy_payload["retrieval_k"] == 12
    with pytest.raises(TypeError):
        artifact.capture_policy_payload["retrieval_k"] = 2
    with pytest.raises(FrozenInstanceError):
        artifact.format = "changed"


def test_seals_are_canonical_across_input_container_and_policy_key_order(tmp_path):
    database = _source_database(tmp_path / "memory.db")
    first = _seal(database)
    reversed_policy = dict(reversed(list(_policy().items())))
    second = _seal(database, events=tuple(_events()), policy=reversed_policy)

    assert first.events == second.events
    assert first.receipt == second.receipt
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.payload() == second.payload()


def test_json_phase_boundary_roundtrip_reconstructs_frozen_verified_types(
    tmp_path,
):
    database = _source_database(tmp_path / "memory.db")
    original = _seal(database)
    transported = json.loads(json.dumps(original.payload()))

    loaded = load_hebbian_history_artifact(transported)

    assert loaded == original
    assert loaded.payload() == original.payload()
    assert type(loaded.events) is tuple
    assert all(type(event.chunk_ids) is tuple for event in loaded.events)
    assert type(loaded.receipt.ordered_event_sha256s) is tuple
    with pytest.raises(TypeError):
        loaded.capture_policy_payload["retrieval_k"] = 999


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda payload: payload["events"][0]["chunk_ids"].append("chunk-1"),
            "event_sha256",
        ),
        (
            lambda payload: payload["receipt"].update({"event_count": 99}),
            "event_count",
        ),
        (
            lambda payload: payload.update({"artifact_sha256": "f" * 64}),
            "artifact seal",
        ),
        (
            lambda payload: payload.update({"unknown": True}),
            "noncanonical shape",
        ),
    ],
)
def test_json_phase_boundary_rejects_nested_and_outer_tampering(
    tmp_path,
    mutate,
    message,
):
    database = _source_database(tmp_path / "memory.db")
    payload = json.loads(json.dumps(_seal(database).payload()))
    mutate(payload)

    with pytest.raises(HebbianHistoryValidationError, match=message):
        load_hebbian_history_artifact(payload)


def test_database_validation_uses_immutable_read_only_uri_without_sidecars(
    tmp_path,
    monkeypatch,
):
    database = _source_database(tmp_path / "memory.db")
    before = {path.name for path in tmp_path.iterdir()}
    observed: list[tuple[object, object]] = []
    real_connect = history_module.sqlite3.connect

    def tracking_connect(database_target, *args, **kwargs):
        observed.append((database_target, kwargs.get("uri")))
        return real_connect(database_target, *args, **kwargs)

    monkeypatch.setattr(history_module.sqlite3, "connect", tracking_connect)
    artifact = _seal(database)

    assert verify_hebbian_history_artifact(artifact) is artifact
    assert observed
    assert all("mode=ro&immutable=1" in str(target) for target, _ in observed)
    assert all(uri is True for _, uri in observed)
    assert {path.name for path in tmp_path.iterdir()} == before
    assert not (tmp_path / "memory.db-wal").exists()
    assert not (tmp_path / "memory.db-shm").exists()
    assert not (tmp_path / "memory.db-journal").exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_source_database_sidecars_fail_closed_for_seal_and_verify(tmp_path, suffix):
    database = _source_database(tmp_path / "memory.db")
    artifact = _seal(database)
    sidecar = database.with_name(database.name + suffix)
    sidecar.write_bytes(b"")

    with pytest.raises(HebbianHistoryValidationError, match="WAL/SHM sidecars"):
        _seal(database)
    with pytest.raises(HebbianHistoryValidationError, match="WAL/SHM sidecars"):
        verify_hebbian_history_artifact(
            artifact,
            source_database_path=database,
        )


@pytest.mark.parametrize(
    "events, message",
    [
        (
            [
                RetrievalAccessEvent("duplicate", 0, ("chunk-0",)),
                RetrievalAccessEvent("duplicate", 1, ("chunk-1",)),
            ],
            "event IDs must be unique",
        ),
        (
            [
                RetrievalAccessEvent("later", 2, ("chunk-2",)),
                RetrievalAccessEvent("earlier", 1, ("chunk-1",)),
            ],
            "nondecreasing",
        ),
        (
            [RetrievalAccessEvent("duplicates", 1, ("chunk-0", "chunk-0"))],
            "unique within the event",
        ),
    ],
)
def test_event_sequence_validation_fails_closed(tmp_path, events, message):
    database = _source_database(tmp_path / "memory.db")
    with pytest.raises(HebbianHistoryValidationError, match=message):
        _seal(database, events=events)


@pytest.mark.parametrize(
    "event, message",
    [
        (
            RetrievalAccessEvent("missing", 2, ("does-not-exist",)),
            "missing an event chunk ID",
        ),
        (
            RetrievalAccessEvent("future", 0, ("chunk-2",)),
            "future turn",
        ),
    ],
)
def test_source_database_must_prove_chunk_existence_and_causality(
    tmp_path,
    event,
    message,
):
    database = _source_database(tmp_path / "memory.db")
    events = _events()
    events[0] = RetrievalAccessEvent(
        events[0].event_id,
        events[0].now_turn,
        event.chunk_ids,
    )
    with pytest.raises(HebbianHistoryValidationError, match=message):
        _seal(database, events=events)


def test_source_database_rejects_omitted_and_post_corpus_event_coordinates(tmp_path):
    database = _source_database(tmp_path / "memory.db")
    with pytest.raises(HebbianHistoryValidationError, match="event population"):
        _seal(database, events=_events()[:-1])
    with pytest.raises(HebbianHistoryValidationError, match="event population"):
        _seal(
            database,
            events=[
                *_events(),
                RetrievalAccessEvent("causal-user:7", 999, ("chunk-0",)),
            ],
        )


def test_event_exact_types_and_event_seal_are_revalidated(tmp_path):
    database = _source_database(tmp_path / "memory.db")
    wrong_type = RetrievalAccessEvent("causal-user:2", 1, ("chunk-0",))
    object.__setattr__(wrong_type, "now_turn", True)
    with pytest.raises(HebbianHistoryValidationError, match="exact non-negative"):
        _seal(database, events=[wrong_type, *_events()[1:]])

    wrong_seal = RetrievalAccessEvent("causal-user:2", 1, ("chunk-0",))
    object.__setattr__(wrong_seal, "event_sha256", "d" * 64)
    with pytest.raises(HebbianHistoryValidationError, match="does not match"):
        _seal(database, events=[wrong_seal, *_events()[1:]])


@pytest.mark.parametrize(
    "policy",
    [
        {"query": "What is the secret?"},
        {"nested": {"text": "transcript content"}},
        MappingProxyType({"retrieval_k": 3}),
        {"threshold": float("nan")},
    ],
)
def test_capture_policy_rejects_text_fields_and_nonexact_json(policy, tmp_path):
    database = _source_database(tmp_path / "memory.db")
    with pytest.raises(HebbianHistoryValidationError):
        _seal(database, policy=policy)


def test_capture_policy_whitelist_rejects_hidden_text_and_model_field_abuse(tmp_path):
    database = _source_database(tmp_path / "memory.db")
    hidden_text = {**_policy(), "notes": "SECRET TRANSCRIPT"}
    with pytest.raises(HebbianHistoryValidationError, match="noncanonical shape"):
        _seal(database, policy=hidden_text)
    abused_model = {**_policy(), "query_embedding_model_id": "SECRET TRANSCRIPT"}
    with pytest.raises(HebbianHistoryValidationError, match="locked value"):
        _seal(database, policy=abused_model)


def test_digest_inputs_require_exact_lowercase_strings(tmp_path):
    database = _source_database(tmp_path / "memory.db")
    capture = replay_module._mint_retrieval_access_capture(
        source_database_sha256=file_sha256(database),
        capture_policy_sha256=identity_sha256(_policy()),
        retrieval_k=12,
        expansion_tokens=1600,
        max_prompt_tokens=128,
        events=_events(),
    )
    with pytest.raises(HebbianHistoryValidationError, match="exact lowercase"):
        seal_hebbian_history_artifact(
            capture,
            source_database_path=database,
            source_store_receipt_sha256=123,
            implementation_sha256=IMPLEMENTATION_SHA256,
            environment_lock_sha256=ENVIRONMENT_SHA256,
            capture_policy_payload=_policy(),
        )
    with pytest.raises(HebbianHistoryValidationError, match="exact lowercase"):
        seal_hebbian_history_artifact(
            capture,
            source_database_path=database,
            source_store_receipt_sha256=SOURCE_RECEIPT_SHA256.upper(),
            implementation_sha256=IMPLEMENTATION_SHA256,
            environment_lock_sha256=ENVIRONMENT_SHA256,
            capture_policy_payload=_policy(),
        )


def test_history_sealer_rejects_raw_and_independently_resealed_membership(tmp_path):
    database = _source_database(tmp_path / "memory.db")
    policy = _policy()
    issued = replay_module._mint_retrieval_access_capture(
        source_database_sha256=file_sha256(database),
        capture_policy_sha256=identity_sha256(policy),
        retrieval_k=12,
        expansion_tokens=1600,
        max_prompt_tokens=128,
        events=_events(),
    )

    with pytest.raises(HebbianHistoryValidationError, match="staging-issued"):
        seal_hebbian_history_artifact(
            _events(),
            source_database_path=database,
            source_store_receipt_sha256=SOURCE_RECEIPT_SHA256,
            implementation_sha256=IMPLEMENTATION_SHA256,
            environment_lock_sha256=ENVIRONMENT_SHA256,
            capture_policy_payload=policy,
        )

    substituted_events = list(issued.events)
    substituted_events[0] = RetrievalAccessEvent(
        event_id=substituted_events[0].event_id,
        now_turn=substituted_events[0].now_turn,
        chunk_ids=("chunk-0", "chunk-1"),
    )
    substituted_tuple = tuple(substituted_events)
    resealed_digest = replay_module.retrieval_access_capture_sha256(
        source_database_sha256=issued.source_database_sha256,
        capture_policy_sha256=issued.capture_policy_sha256,
        retrieval_k=issued.retrieval_k,
        expansion_tokens=issued.expansion_tokens,
        max_prompt_tokens=issued.max_prompt_tokens,
        events=substituted_tuple,
    )
    forged = replace(
        issued,
        events=substituted_tuple,
        capture_sha256=resealed_digest,
    )
    with pytest.raises(HebbianHistoryValidationError, match="staging process"):
        seal_hebbian_history_artifact(
            forged,
            source_database_path=database,
            source_store_receipt_sha256=SOURCE_RECEIPT_SHA256,
            implementation_sha256=IMPLEMENTATION_SHA256,
            environment_lock_sha256=ENVIRONMENT_SHA256,
            capture_policy_payload=policy,
        )


def test_receipt_artifact_and_database_tampering_are_rejected(tmp_path):
    database = _source_database(tmp_path / "memory.db")
    artifact = _seal(database)

    with pytest.raises(HebbianHistoryValidationError, match="artifact seal"):
        verify_hebbian_history_artifact(
            replace(artifact, artifact_sha256="f" * 64)
        )

    bad_receipt = replace(
        artifact.receipt,
        empty_event_count=0,
    )
    with pytest.raises(HebbianHistoryValidationError, match="empty_event_count"):
        verify_hebbian_history_artifact(replace(artifact, receipt=bad_receipt))

    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO turns (turn_id, ordinal, role, text) VALUES (?, ?, ?, ?)",
            ("turn-7", 7, "system", "new source content"),
        )
    with pytest.raises(HebbianHistoryValidationError, match="database digest"):
        verify_hebbian_history_artifact(
            artifact,
            source_database_path=database,
        )


def test_empty_history_is_explicitly_sealed(tmp_path):
    database = tmp_path / "memory.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE turns (
                turn_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                text TEXT NOT NULL
            );
            INSERT INTO turns (turn_id, ordinal, role, text)
            VALUES ('turn-1', 1, 'user', 'first prompt has no prior chunks');
            """
        )
    artifact = _seal(database, events=[])

    assert artifact.events == ()
    assert artifact.receipt.ordered_event_sha256s == ()
    assert artifact.receipt.event_count == 0
    assert artifact.receipt.empty_event_count == 0
    assert verify_hebbian_history_artifact(
        artifact,
        source_database_path=database,
    ) is artifact
