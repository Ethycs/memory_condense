import pytest

from memory_condense.persistence.transcript_store import TranscriptStore


def test_append_and_get(db):
    store = TranscriptStore(db)
    turn = store.append("user", "Hello world")
    assert turn.role == "user"
    assert turn.text == "Hello world"

    fetched = store.get_turn(turn.turn_id)
    assert fetched is not None
    assert fetched.text == "Hello world"


def test_source_id_round_trips(db):
    store = TranscriptStore(db)
    turn = store.append("user", "session fact", source_id="session-2")
    assert turn.source_id == "session-2"
    assert store.get_turn(turn.turn_id).source_id == "session-2"


def test_explicit_turn_id_round_trips_and_empty_id_is_rejected(db):
    store = TranscriptStore(db)
    turn = store.append(
        "user",
        "stable source turn",
        source_id="session-2",
        turn_id="stable-turn-0001",
    )
    assert turn.turn_id == "stable-turn-0001"
    assert store.get_turn("stable-turn-0001") == turn
    with pytest.raises(ValueError, match="turn_id must be non-empty"):
        store.append("user", "invalid", turn_id="   ")


def test_exact_explicit_turn_retry_is_idempotent_and_conflicts_fail_closed(db):
    store = TranscriptStore(db)
    first = store.append(
        "user", "stable source turn", source_id="session-2", turn_id="stable-turn"
    )

    assert store.append(
        "user", "stable source turn", source_id="session-2", turn_id="stable-turn"
    ) == first
    assert store.count() == 1
    with pytest.raises(ValueError, match="different content"):
        store.append("user", "changed", source_id="session-2", turn_id="stable-turn")


def test_direct_publish_commit_failure_rolls_back_owned_transaction(
    db, monkeypatch
):
    store = TranscriptStore(db)

    def fail_commit() -> None:
        raise RuntimeError("synthetic turn commit failure")

    monkeypatch.setattr(db, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="synthetic turn commit failure"):
        store.append("user", "must not remain uncommitted", turn_id="failed-turn")

    assert store.get_turn("failed-turn") is None


def test_publish_without_commit_requires_an_outer_transaction(db):
    store = TranscriptStore(db)
    turn = store.stage("user", "caller must own publication", turn_id="outer-turn")

    with pytest.raises(RuntimeError, match="requires an active caller transaction"):
        store.publish_turn(turn, commit=False)

    assert store.get_turn("outer-turn") is None


def test_source_metadata_returns_first_system_turn_per_source(db):
    store = TranscriptStore(db)
    store.append(
        "system",
        "[session-2 took place at 2023/06/28 (Wed) 20:26]",
        source_id="session-2",
    )
    store.append("user", "real evidence", source_id="session-2")
    store.append("system", "later metadata", source_id="session-2")
    store.append("system", "[session-3 took place at Friday]", source_id="session-3")

    assert store.source_metadata(["session-2", "missing", "session-3"]) == {
        "session-2": "[session-2 took place at 2023/06/28 (Wed) 20:26]",
        "session-3": "[session-3 took place at Friday]",
    }


def test_get_turn_not_found(db):
    store = TranscriptStore(db)
    assert store.get_turn("nonexistent") is None


def test_count(db):
    store = TranscriptStore(db)
    assert store.count() == 0
    store.append("user", "one")
    store.append("assistant", "two")
    assert store.count() == 2


def test_get_recent(db):
    store = TranscriptStore(db)
    store.append("user", "first")
    store.append("assistant", "second")
    store.append("user", "third")

    recent = store.get_recent(2)
    assert len(recent) == 2
    # oldest first
    assert recent[0].text == "second"
    assert recent[1].text == "third"


def test_get_all(db):
    store = TranscriptStore(db)
    store.append("user", "a")
    store.append("assistant", "b")

    all_turns = store.get_all()
    assert len(all_turns) == 2
    assert all_turns[0].text == "a"
    assert all_turns[1].text == "b"
