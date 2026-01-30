from memory_condense.transcript_store import TranscriptStore


def test_append_and_get(db):
    store = TranscriptStore(db)
    turn = store.append("user", "Hello world")
    assert turn.role == "user"
    assert turn.text == "Hello world"

    fetched = store.get_turn(turn.turn_id)
    assert fetched is not None
    assert fetched.text == "Hello world"


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
