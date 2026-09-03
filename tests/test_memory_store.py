import sqlite3
from datetime import timedelta

import numpy as np
import pytest

from memory_condense.domain import decay
from memory_condense.persistence.memory_store import MemoryStore
from memory_condense.persistence.db import Database
from memory_condense.domain.schemas import (
    CreateOp,
    DeleteOp,
    Heat,
    MemoryOps,
    MemoryStatus,
    MemoryType,
    PinOp,
    PinState,
    Provenance,
    SupersedeOp,
    UpdateOp,
    ValidationReport,
)
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.ingest.validator import Validator

TEXT = "I prefer dark mode. We decided to use Postgres for storage."


@pytest.fixture
def turn(db):
    return TranscriptStore(db).append("user", TEXT)


@pytest.fixture
def store(db):
    return MemoryStore(db)


def make_create(turn_id, quote="I prefer dark mode", content=None, **kwargs):
    return CreateOp(
        type=kwargs.pop("type", MemoryType.PREFERENCE),
        content=content or quote,
        provenance=[Provenance(turn_id=turn_id, quote=quote)],
        **kwargs,
    )


class StubEmbedder:
    """Deterministic 4-dim embedder: no model, no network."""

    def __init__(self, table=None):
        self.table = table or {}
        self.calls = []

    def embed_query(self, query):
        self.calls.append(query)
        return np.asarray(self.table.get(query, [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)


class FailingEmbedder:
    def embed_query(self, _query):
        raise RuntimeError("embedding failed")


def test_read_only_supersede_rejects_before_embedding(tmp_path):
    path = tmp_path / "read-only-memory.db"
    with Database(path) as writable:
        source = TranscriptStore(writable).append("user", TEXT)
        old = MemoryStore(writable).create(make_create(source.turn_id))

    embedder = StubEmbedder()
    with Database(path, read_only=True) as readonly:
        read_store = MemoryStore(readonly, embedder=embedder)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            read_store.supersede(
                SupersedeOp(
                    mem_id=old.mem_id,
                    replacement=make_create(
                        source.turn_id, content="A replacement must not embed."
                    ),
                )
            )

    assert embedder.calls == []


# ----------------------------------------------------------------------
# create / get / list
# ----------------------------------------------------------------------


def test_create_round_trips(store, turn):
    item = store.create(make_create(turn.turn_id))
    fetched = store.get(item.mem_id)

    assert fetched is not None
    assert fetched.mem_id == item.mem_id
    assert fetched.type is MemoryType.PREFERENCE
    assert fetched.content == "I prefer dark mode"
    assert fetched.status is MemoryStatus.ACTIVE
    assert fetched.pin is PinState.NONE


def test_create_persists_provenance(store, turn):
    op = CreateOp(
        type=MemoryType.DECISION,
        content="use postgres",
        provenance=[
            Provenance(turn_id=turn.turn_id, quote="We decided to use Postgres"),
            Provenance(turn_id=turn.turn_id, quote="dark mode", chunk_id="c1"),
        ],
    )
    item = store.create(op)
    loaded = store.get(item.mem_id)

    assert len(loaded.provenance) == 2
    assert {p.quote for p in loaded.provenance} == {
        "We decided to use Postgres",
        "dark mode",
    }
    assert any(p.chunk_id == "c1" for p in loaded.provenance)


def test_create_provenance_failure_cannot_publish_item(store, db, turn):
    db.execute(
        "CREATE TEMP TRIGGER fail_memory_provenance "
        "BEFORE INSERT ON memory_provenance BEGIN "
        "SELECT RAISE(ABORT, 'synthetic provenance failure'); END"
    )
    db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="provenance failure"):
        store.create(make_create(turn.turn_id), dedupe=False)

    db.execute("DROP TRIGGER fail_memory_provenance")
    db.commit()
    assert store.count(status=None) == 0


def test_concurrent_exact_create_rechecks_under_write_lock(tmp_path, monkeypatch):
    path = tmp_path / "concurrent-memory.db"
    db_a, db_b = Database(path), Database(path)
    first_turn = TranscriptStore(db_a).append("user", TEXT)
    second_turn = TranscriptStore(db_b).append("user", TEXT)
    first, second = MemoryStore(db_a), MemoryStore(db_b)
    original_build = first._build_item
    interleaved = False

    def build_after_other_writer(
        op, *, embedding=None, half_life_turns=30.0, supersedes=None
    ):
        nonlocal interleaved
        if not interleaved:
            interleaved = True
            second.create(
                make_create(second_turn.turn_id, content="one concurrent fact")
            )
        return original_build(
            op,
            embedding=embedding,
            half_life_turns=half_life_turns,
            supersedes=supersedes,
        )

    monkeypatch.setattr(first, "_build_item", build_after_other_writer)
    merged = first.create(
        make_create(first_turn.turn_id, content="one concurrent fact")
    )

    active = first.list_items(status=MemoryStatus.ACTIVE)
    assert [item.mem_id for item in active] == [merged.mem_id]
    assert {citation.turn_id for citation in merged.provenance} == {
        first_turn.turn_id,
        second_turn.turn_id,
    }
    db_a.close()
    db_b.close()


def test_seed_energy_follows_importance(store, turn):
    hot = store.create(make_create(turn.turn_id, importance=0.9))
    warm = store.create(make_create(turn.turn_id, "dark mode", importance=0.4))

    assert hot.energy == decay.HOT_SEED_ENERGY
    assert warm.energy == decay.WARM_SEED_ENERGY
    assert store.get(hot.mem_id).heat is Heat.HOT


def test_get_missing_returns_none(store):
    assert store.get("nope") is None


def test_list_items_filters_by_status(store, turn):
    a = store.create(make_create(turn.turn_id))
    b = store.create(make_create(turn.turn_id, "dark mode"))
    store.delete(DeleteOp(mem_id=b.mem_id))

    active = store.list_items()
    assert [i.mem_id for i in active] == [a.mem_id]
    assert len(store.list_items(status=MemoryStatus.DELETED)) == 1
    assert len(store.list_items(status=None)) == 2


def test_list_items_limit(store, turn):
    for i in range(5):
        store.create(make_create(turn.turn_id, content=f"item {i}"))
    assert len(store.list_items(limit=2)) == 2


# ----------------------------------------------------------------------
# embeddings
# ----------------------------------------------------------------------


def test_explicit_embedding_round_trips_as_float32(store, turn):
    vec = [0.25, -0.5, 0.75, 1.0]
    item = store.create(make_create(turn.turn_id), embedding=np.asarray(vec))
    loaded = store.get(item.mem_id)

    assert loaded.embedding == pytest.approx(vec)


def test_embedder_is_used_when_no_embedding_given(db, turn):
    embedder = StubEmbedder({"I prefer dark mode": [1.0, 0.0, 0.0, 0.0]})
    store = MemoryStore(db, embedder=embedder)
    item = store.create(make_create(turn.turn_id))

    assert embedder.calls == ["I prefer dark mode"]
    assert store.get(item.mem_id).embedding == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_no_embedder_means_no_embedding(store, turn):
    item = store.create(make_create(turn.turn_id))
    assert store.get(item.mem_id).embedding is None


# ----------------------------------------------------------------------
# update / supersede / delete / pin
# ----------------------------------------------------------------------


def test_update_amends_fields(store, turn):
    item = store.create(make_create(turn.turn_id))
    updated = store.update(
        UpdateOp(mem_id=item.mem_id, content="prefers dark mode", details="editors too")
    )

    assert updated.content == "prefers dark mode"
    assert updated.details == "editors too"
    assert updated.mem_id == item.mem_id


def test_update_appends_provenance(store, turn):
    item = store.create(make_create(turn.turn_id))
    updated = store.update(
        UpdateOp(
            mem_id=item.mem_id,
            provenance=[Provenance(turn_id=turn.turn_id, quote="Postgres")],
        )
    )
    assert len(updated.provenance) == 2


def test_update_provenance_failure_rolls_back_content(store, db, turn):
    item = store.create(make_create(turn.turn_id, content="before"))
    db.execute(
        "CREATE TEMP TRIGGER fail_memory_update_provenance "
        "BEFORE INSERT ON memory_provenance BEGIN "
        "SELECT RAISE(ABORT, 'synthetic update provenance failure'); END"
    )
    db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="update provenance failure"):
        store.update(
            UpdateOp(
                mem_id=item.mem_id,
                content="after",
                details="must roll back",
                provenance=[Provenance(turn_id=turn.turn_id, quote="Postgres")],
            )
        )

    db.execute("DROP TRIGGER fail_memory_update_provenance")
    db.commit()
    unchanged = store.get(item.mem_id)
    assert unchanged.content == "before"
    assert unchanged.details is None
    assert unchanged.provenance == item.provenance


def test_update_missing_returns_none(store):
    assert store.update(UpdateOp(mem_id="ghost", content="x")) is None


def test_supersede_keeps_old_row_with_superseded_status(store, turn):
    old = store.create(make_create(turn.turn_id))
    new = store.supersede(
        SupersedeOp(
            mem_id=old.mem_id,
            replacement=make_create(turn.turn_id, "dark mode", content="light mode now"),
        )
    )

    old_reloaded = store.get(old.mem_id)
    assert old_reloaded is not None, "supersede must never delete the old row"
    assert old_reloaded.status is MemoryStatus.SUPERSEDED
    assert new.supersedes == old.mem_id
    assert new.status is MemoryStatus.ACTIVE
    assert store.count() == 2


def test_supersede_missing_returns_none(store, turn):
    op = SupersedeOp(mem_id="ghost", replacement=make_create(turn.turn_id))
    assert store.supersede(op) is None


def test_supersede_embedding_failure_leaves_predecessor_active(db, turn):
    store = MemoryStore(db, embedder=FailingEmbedder())
    old = store.create(make_create(turn.turn_id, content="old"), embedding=[1.0])

    with pytest.raises(RuntimeError, match="embedding failed"):
        store.supersede(
            SupersedeOp(
                mem_id=old.mem_id,
                replacement=make_create(turn.turn_id, content="replacement"),
            )
        )

    assert store.get(old.mem_id).status is MemoryStatus.ACTIVE
    assert store.count() == 1


def test_supersede_insert_failure_rolls_back_retirement(store, turn, monkeypatch):
    old = store.create(make_create(turn.turn_id, content="old"))
    original_insert = store._insert

    def fail_after_insert(item, *, commit=True):
        original_insert(item, commit=False)
        raise RuntimeError("insert publication failed")

    monkeypatch.setattr(store, "_insert", fail_after_insert)
    with pytest.raises(RuntimeError, match="publication failed"):
        store.supersede(
            SupersedeOp(
                mem_id=old.mem_id,
                replacement=make_create(turn.turn_id, content="replacement"),
            )
        )

    assert store.get(old.mem_id).status is MemoryStatus.ACTIVE
    assert store.count() == 1


def test_create_supersedes_uses_atomic_lifecycle_and_rejects_ghost(store, turn):
    old = store.create(make_create(turn.turn_id, content="old"))
    new = store.create(
        make_create(turn.turn_id, content="replacement"),
        supersedes=old.mem_id,
    )

    assert new.supersedes == old.mem_id
    assert store.get(old.mem_id).status is MemoryStatus.SUPERSEDED
    with pytest.raises(ValueError, match="must name an active memory item"):
        store.create(
            make_create(turn.turn_id, content="ghost replacement"),
            supersedes="ghost",
        )
    assert store.count() == 2


def test_supersede_rejects_an_already_retired_predecessor(store, turn):
    old = store.create(make_create(turn.turn_id, content="old"))
    first = store.supersede(
        SupersedeOp(
            mem_id=old.mem_id,
            replacement=make_create(turn.turn_id, content="first replacement"),
        )
    )

    assert first is not None
    assert (
        store.supersede(
            SupersedeOp(
                mem_id=old.mem_id,
                replacement=make_create(turn.turn_id, content="second replacement"),
            )
        )
        is None
    )
    assert store.count() == 2


def test_delete_is_soft(store, turn):
    item = store.create(make_create(turn.turn_id))
    assert store.delete(DeleteOp(mem_id=item.mem_id, reason="obsolete")) is True

    row = store.get(item.mem_id)
    assert row is not None, "delete must be soft — the row survives"
    assert row.status is MemoryStatus.DELETED
    assert store.list_items() == []


def test_delete_missing_returns_false(store):
    assert store.delete(DeleteOp(mem_id="ghost")) is False


def test_pin_and_unpin(store, turn):
    item = store.create(make_create(turn.turn_id))
    pinned = store.pin(PinOp(mem_id=item.mem_id, pin=PinState.USER))
    assert pinned.pin is PinState.USER
    assert pinned.is_pinned

    unpinned = store.pin(PinOp(mem_id=item.mem_id, pin=PinState.NONE))
    assert unpinned.is_pinned is False


def test_pin_missing_returns_none(store):
    assert store.pin(PinOp(mem_id="ghost")) is None


# ----------------------------------------------------------------------
# decay / pins / reheating
# ----------------------------------------------------------------------


def test_pin_exempts_item_from_decay(store, turn):
    plain = store.create(make_create(turn.turn_id, content="plain"))
    kept = store.create(make_create(turn.turn_id, content="kept"))
    store.pin(PinOp(mem_id=kept.mem_id, pin=PinState.USER))

    far_future = 120  # four 30-turn half-lives
    plain_item = store.get(plain.mem_id)
    kept_item = store.get(kept.mem_id)

    assert decay.item_energy(plain_item, now_turn=far_future) < 0.05
    assert decay.item_energy(kept_item, now_turn=far_future) == pytest.approx(
        kept_item.energy
    )
    assert decay.item_heat(plain_item, now_turn=far_future) is Heat.COLD
    assert decay.item_heat(kept_item, now_turn=far_future) is Heat.WARM


def test_touch_reheats_a_decayed_item(store, turn):
    item = store.create(make_create(turn.turn_id, importance=0.1))
    stale = item.last_access_turn + 30  # exactly one half-life, in turns

    decayed = decay.item_energy(store.get(item.mem_id), now_turn=stale)
    assert decayed == pytest.approx(decay.WARM_SEED_ENERGY / 2, abs=1e-3)

    touched = store.touch(item.mem_id, now_turn=stale)
    assert touched.energy > decayed
    assert touched.energy == pytest.approx(decay.reheat(decayed), abs=1e-6)
    assert touched.last_access_turn == stale


def test_touch_is_refractory_within_one_turn(store, turn):
    """Ten recalls while answering one turn are one access, not ten.

    This replaces a test that asserted ten same-instant touches drove energy
    to exactly 1.0 — which was the bug, not the contract: it let any item read
    a few times in a row pin itself at maximum energy permanently.
    """
    item = store.create(make_create(turn.turn_id, importance=0.9))
    # A turn past the one creation opened, so the first touch counts.
    now = item.last_access_turn + 1
    expected = decay.reheat(decay.item_energy(item, now_turn=now))

    for _ in range(10):
        item = store.touch(item.mem_id, now_turn=now)

    # Exactly one boost applied (the first), and it saturates rather than adds.
    assert item.energy == pytest.approx(expected)
    assert item.energy < 1.0


def test_touch_boosts_again_on_the_next_turn(store, turn):
    item = store.create(make_create(turn.turn_id, importance=0.9))

    first = store.touch(item.mem_id, now_turn=item.last_access_turn + 1)
    later = first.last_access_turn + 1
    second = store.touch(item.mem_id, now_turn=later)

    # Strictly above what plain decay alone would have left it at.
    assert second.energy > decay.item_energy(first, now_turn=later)


def test_a_fresh_item_is_not_reheated_by_being_read_immediately(store, turn):
    """Creating then recalling in the same breath is one event, not two."""
    item = store.create(make_create(turn.turn_id, importance=0.9))
    touched = store.touch(item.mem_id, now_turn=item.last_access_turn)
    assert touched.energy == pytest.approx(item.energy)


def test_touch_keeps_pinned_energy_but_restamps(store, turn):
    item = store.create(make_create(turn.turn_id))
    store.pin(PinOp(mem_id=item.mem_id, pin=PinState.USER))
    later = item.last_access_turn + 60

    touched = store.touch(item.mem_id, now_turn=later)
    assert touched.energy == pytest.approx(item.energy)
    assert touched.last_access_turn == later


def test_touch_missing_returns_none(store):
    assert store.touch("ghost") is None


def test_touch_many_uses_one_transaction_and_preserves_order(store, turn):
    first = store.create(make_create(turn.turn_id, content="first"))
    second = store.create(make_create(turn.turn_id, content="second"))
    commits = 0

    def trace(sql):
        nonlocal commits
        if sql.strip().upper() == "COMMIT":
            commits += 1

    store._db.connection.set_trace_callback(trace)
    touched = store.touch_many(
        [second.mem_id, "ghost", first.mem_id, second.mem_id],
        now_turn=first.last_access_turn + 5,
    )
    store._db.connection.set_trace_callback(None)

    assert [item.mem_id for item in touched] == [second.mem_id, first.mem_id]
    assert all(item.last_access_turn == first.last_access_turn + 5 for item in touched)
    assert commits == 1


def test_heat_counts(store, turn):
    store.create(make_create(turn.turn_id, content="hot", importance=0.9))
    store.create(make_create(turn.turn_id, content="warm", importance=0.2))
    counts = store.heat_counts()

    assert counts == {"HOT": 1, "WARM": 1, "COLD": 0}

    cold = store.heat_counts(now_turn=store._db.current_turn() + 120)
    assert cold["COLD"] == 2


# ----------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------


def test_apply_memory_ops(store, turn):
    summary = store.apply(MemoryOps(create=[make_create(turn.turn_id)]))
    assert summary["created"] == 1
    assert store.count(MemoryStatus.ACTIVE) == 1


def test_apply_validation_report_uses_accepted_only(db, turn):
    store = MemoryStore(db)
    ops = MemoryOps(
        create=[
            make_create(turn.turn_id, "I prefer dark mode"),
            make_create(turn.turn_id, "quote that was never said"),
        ]
    )
    report = Validator(db).validate(ops)
    summary = store.apply(report)

    assert summary["created"] == 1
    assert len(report.rejected) == 1
    assert store.count() == 1


def test_apply_runs_ops_in_order(store, turn):
    existing = store.create(make_create(turn.turn_id, content="old"))
    other = store.create(make_create(turn.turn_id, content="other"))
    doomed = store.create(make_create(turn.turn_id, content="doomed"))

    summary = store.apply(
        MemoryOps(
            create=[make_create(turn.turn_id, content="fresh")],
            update=[UpdateOp(mem_id=other.mem_id, content="amended")],
            supersede=[
                SupersedeOp(
                    mem_id=existing.mem_id,
                    replacement=make_create(turn.turn_id, content="replacement"),
                )
            ],
            delete=[DeleteOp(mem_id=doomed.mem_id)],
            pin=[PinOp(mem_id=other.mem_id, pin=PinState.SYSTEM)],
        )
    )

    assert summary == {
        "created": 1,
        "duplicate": 0,
        "updated": 1,
        "superseded": 1,
        "deleted": 1,
        "pinned": 1,
        "skipped": 0,
    }
    assert store.get(existing.mem_id).status is MemoryStatus.SUPERSEDED
    assert store.get(doomed.mem_id).status is MemoryStatus.DELETED
    assert store.get(other.mem_id).content == "amended"
    assert store.get(other.mem_id).pin is PinState.SYSTEM


def test_apply_counts_skipped_ops(store):
    summary = store.apply(MemoryOps(delete=[DeleteOp(mem_id="ghost")]))
    assert summary["skipped"] == 1
    assert summary["deleted"] == 0


def test_apply_empty_report(store):
    assert store.apply(ValidationReport())["created"] == 0


# ----------------------------------------------------------------------
# retrieve
# ----------------------------------------------------------------------


def test_retrieve_ranks_by_cosine_relevance(store, turn):
    near = store.create(make_create(turn.turn_id, content="near"), embedding=[1.0, 0.0])
    far = store.create(make_create(turn.turn_id, content="far"), embedding=[-1.0, 0.0])

    results = store.retrieve(np.asarray([1.0, 0.0], dtype=np.float32), k=2)

    assert [r.item.mem_id for r in results] == [near.mem_id, far.mem_id]
    assert results[0].relevance == pytest.approx(1.0)
    assert results[1].relevance == pytest.approx(0.0)
    assert results[0].score > results[1].score


def test_retrieve_batches_provenance_and_returns_it(store, turn):
    for i in range(5):
        store.create(make_create(turn.turn_id, content=f"fact {i}"))
    statements = []
    store._db.connection.set_trace_callback(statements.append)

    results = store.retrieve(None, k=3, reheat=False)

    store._db.connection.set_trace_callback(None)
    provenance_reads = [
        sql for sql in statements if "FROM memory_provenance" in sql
    ]
    assert len(provenance_reads) == 1
    assert all(result.item.provenance for result in results)


def test_retrieve_populates_score_components(store, turn):
    item = store.create(make_create(turn.turn_id, importance=0.9), embedding=[1.0, 0.0])
    store.pin(PinOp(mem_id=item.mem_id, pin=PinState.USER))

    result = store.retrieve(np.asarray([1.0, 0.0]), k=1)[0]
    assert result.relevance == pytest.approx(1.0)
    assert result.importance == pytest.approx(0.9)
    assert result.recency == pytest.approx(1.0, abs=1e-3)
    assert result.pin_boost == pytest.approx(1.0)
    # The scored term is decayed *energy*, not the bare time factor. importance
    # 0.9 seeds energy at HOT_SEED_ENERGY (0.8), and the item is pinned so no
    # decay applies — hence 0.2 * 0.8, where this once asserted 0.2 * 1.0.
    assert result.energy == pytest.approx(decay.HOT_SEED_ENERGY)
    assert result.score == pytest.approx(1.0 + 0.3 * 0.9 + 0.5 * 1.0 + 0.2 * 0.8, abs=1e-3)


def test_retrieve_without_query_scores_relevance_zero(store, turn):
    important = store.create(make_create(turn.turn_id, content="a", importance=0.9))
    store.create(make_create(turn.turn_id, content="b", importance=0.1))

    results = store.retrieve(None, k=2)
    assert all(r.relevance == 0.0 for r in results)
    assert results[0].item.mem_id == important.mem_id


def test_retrieve_item_without_embedding_scores_zero_relevance(store, turn):
    store.create(make_create(turn.turn_id, content="no vector"))
    result = store.retrieve(np.asarray([1.0, 0.0]), k=1)[0]
    assert result.relevance == 0.0


def test_retrieve_skips_dimension_mismatch(store, turn):
    store.create(make_create(turn.turn_id, content="wrong dim"), embedding=[1.0, 0.0, 0.0])
    result = store.retrieve(np.asarray([1.0, 0.0]), k=1)[0]
    assert result.relevance == 0.0


def test_pin_boost_lifts_a_less_relevant_item(store, turn):
    store.create(make_create(turn.turn_id, content="relevant"), embedding=[1.0, 0.0])
    # Not fully orthogonal: at cosine 0 the relevance gap is exactly 0.5, which
    # is exactly the pin weight, so the two scores tied and this test passed on
    # sort stability alone — it would have passed with the pin weight at zero.
    pinned = store.create(
        make_create(turn.turn_id, content="pinned"), embedding=[0.2, 1.0]
    )
    store.pin(PinOp(mem_id=pinned.mem_id, pin=PinState.USER))

    results = store.retrieve(np.asarray([1.0, 0.0]), k=2)
    assert results[0].item.mem_id == pinned.mem_id
    assert results[0].score > results[1].score


def test_hotter_item_outranks_colder_at_equal_relevance(store, turn):
    """Decay influences what comes back — the claim that had no test.

    Before the two decay kernels were collapsed, `rank_score` took a `recency`
    term computed from `last_access_at`, which `touch` restamps on every
    retrieve. Energy never entered the scalar at all, so this assertion could
    not have held.
    """
    now = store._db.current_turn()
    hot = store.create(make_create(turn.turn_id, content="hot"), embedding=[1.0, 0.0])
    cold = store.create(make_create(turn.turn_id, content="cold"), embedding=[1.0, 0.0])
    # Leave one of them 60 turns behind without touching the other.
    store._db.execute(
        "UPDATE memory_items SET last_access_turn = ? WHERE mem_id = ?",
        (now - 60, cold.mem_id),
    )
    store._db.commit()

    results = store.retrieve(np.asarray([1.0, 0.0]), k=2, now_turn=now)
    assert [r.item.mem_id for r in results] == [hot.mem_id, cold.mem_id]
    assert results[0].energy > results[1].energy


def test_relevance_still_beats_heat(store, turn):
    """Decay is a tiebreaker, not a gate.

    A rarely-touched item that exactly answers the query must still win — the
    reason `min_energy` defaults to off.
    """
    now = decay.now_utc()
    stale = store.create(make_create(turn.turn_id, content="stale"), embedding=[1.0, 0.0])
    store.create(make_create(turn.turn_id, content="fresh"), embedding=[0.0, 1.0])
    store._db.execute(
        "UPDATE memory_items SET last_access_at = ? WHERE mem_id = ?",
        ((now - timedelta(days=180)).isoformat(), stale.mem_id),
    )
    store._db.commit()

    results = store.retrieve(np.asarray([1.0, 0.0]), k=2, now=now)
    assert results[0].item.mem_id == stale.mem_id


def test_min_energy_filters_cold_items_and_is_off_by_default(store, turn):
    now = store._db.current_turn()
    cold = store.create(make_create(turn.turn_id, content="cold"), embedding=[1.0, 0.0])
    store._db.execute(
        "UPDATE memory_items SET last_access_turn = ? WHERE mem_id = ?",
        (now - 90, cold.mem_id),
    )
    store._db.commit()

    # reheat=False on the probe, or the first call restamps the item and the
    # second one sees a freshly hot row — retrieval is a write path.
    assert store.retrieve(np.asarray([1.0, 0.0]), k=5, now_turn=now, reheat=False)
    assert not store.retrieve(
        np.asarray([1.0, 0.0]), k=5, now_turn=now, min_energy=0.25, reheat=False
    )


def test_retrieve_with_reheat_false_does_not_record_an_access(store, turn):
    item = store.create(make_create(turn.turn_id), embedding=[1.0, 0.0])
    later = item.last_access_turn + 5

    store.retrieve(np.asarray([1.0, 0.0]), k=1, now_turn=later, reheat=False)

    unchanged = store.get(item.mem_id)
    assert unchanged.energy == pytest.approx(item.energy)
    assert unchanged.last_access_at == item.last_access_at


def test_retrieve_excludes_deleted_and_superseded_by_default(store, turn):
    old = store.create(make_create(turn.turn_id, content="old"))
    store.supersede(
        SupersedeOp(
            mem_id=old.mem_id, replacement=make_create(turn.turn_id, content="new")
        )
    )
    doomed = store.create(make_create(turn.turn_id, content="doomed"))
    store.delete(DeleteOp(mem_id=doomed.mem_id))

    ids = {r.item.mem_id for r in store.retrieve(None, k=10)}
    assert old.mem_id not in ids
    assert doomed.mem_id not in ids


def test_include_superseded_applies_the_penalty(store, turn):
    old = store.create(make_create(turn.turn_id, content="old"))
    store.supersede(
        SupersedeOp(
            mem_id=old.mem_id, replacement=make_create(turn.turn_id, content="new")
        )
    )

    results = store.retrieve(None, k=10, include_superseded=True)
    by_id = {r.item.mem_id: r for r in results}
    assert old.mem_id in by_id
    assert by_id[old.mem_id].score < 0
    assert results[-1].item.mem_id == old.mem_id


def test_retrieve_touches_returned_items(store, turn):
    item = store.create(make_create(turn.turn_id, importance=0.1))
    later = item.last_access_turn + 30  # exactly one half-life, in turns
    seeded = item.energy

    result = store.retrieve(None, k=1, now_turn=later)[0]
    expected = decay.reheat(seeded / 2)

    # The reheat is persisted...
    stored = store.get(item.mem_id)
    assert stored.energy == pytest.approx(expected, abs=1e-3)
    assert stored.energy > seeded / 2
    assert stored.last_access_turn == later
    # ...and the returned MemoryResult carries the refreshed item.
    assert result.item.energy == pytest.approx(expected, abs=1e-3)


def test_retrieve_does_not_touch_unreturned_items(store, turn):
    kept = store.create(make_create(turn.turn_id, content="kept", importance=0.9))
    ignored = store.create(make_create(turn.turn_id, content="ignored", importance=0.1))
    before = store.get(ignored.mem_id).last_access_at

    results = store.retrieve(None, k=1)
    assert results[0].item.mem_id == kept.mem_id
    assert store.get(ignored.mem_id).last_access_at == before


def test_retrieve_reheats_all_results_in_one_transaction(store, turn):
    for i in range(5):
        store.create(make_create(turn.turn_id, content=f"item {i}"))
    commits = 0

    def trace(sql):
        nonlocal commits
        if sql.strip().upper() == "COMMIT":
            commits += 1

    store._db.connection.set_trace_callback(trace)
    results = store.retrieve(None, k=5, now_turn=10)
    store._db.connection.set_trace_callback(None)

    assert len(results) == 5
    assert commits == 1


def test_retrieve_honours_k(store, turn):
    for i in range(5):
        store.create(make_create(turn.turn_id, content=f"item {i}"))
    assert len(store.retrieve(None, k=3)) == 3
    assert store.retrieve(None, k=0) == []


def test_retrieve_on_empty_store(store):
    assert store.retrieve(np.asarray([1.0, 0.0]), k=5) == []


# ----------------------------------------------------------------------
# Duplicate detection
# ----------------------------------------------------------------------


class TestDedup:
    def test_creating_the_same_fact_twice_merges(self, store, turn):
        first = store.create(make_create(turn.turn_id, content="ship on Friday"))
        again = store.create(make_create(turn.turn_id, content="ship on Friday"))

        assert store.count() == 1
        assert again.mem_id == first.mem_id

    def test_whitespace_and_case_variants_collapse(self, store, turn):
        store.create(make_create(turn.turn_id, content="Ship on Friday"))
        store.create(make_create(turn.turn_id, content="ship   on\n  FRIDAY"))
        assert store.count() == 1

    def test_the_same_text_under_a_different_type_is_a_different_memory(
        self, store, turn
    ):
        store.create(
            make_create(turn.turn_id, content="ship on Friday", type=MemoryType.DECISION)
        )
        store.create(
            make_create(
                turn.turn_id, content="ship on Friday", type=MemoryType.CONSTRAINT
            )
        )
        assert store.count() == 2

    def test_merging_adds_provenance_without_duplicating_it(self, store, db):
        transcript = TranscriptStore(db)
        t1 = transcript.append("user", "I prefer dark mode. It is easier.")
        t2 = transcript.append("user", "I prefer dark mode. Still true.")

        store.create(make_create(t1.turn_id, content="dark mode"))
        merged = store.create(make_create(t2.turn_id, content="dark mode"))

        assert {p.turn_id for p in merged.provenance} == {t1.turn_id, t2.turn_id}
        # Re-asserting the identical citation must not add a third row.
        again = store.create(make_create(t2.turn_id, content="dark mode"))
        assert len(again.provenance) == 2

    def test_re_asserting_a_fact_reheats_it(self, store, turn):
        item = store.create(make_create(turn.turn_id, content="ship on Friday"))
        # Past the refractory window opened by creation itself.
        store._db.execute(
            "UPDATE memory_items SET last_access_at = ? WHERE mem_id = ?",
            (
                (item.last_access_at - timedelta(days=3)).isoformat(),
                item.mem_id,
            ),
        )
        store._db.commit()
        before = store.get(item.mem_id)

        after = store.create(make_create(turn.turn_id, content="ship on Friday"))
        assert after.last_access_at > before.last_access_at

    def test_forgetting_then_remembering_recreates(self, store, turn):
        first = store.create(make_create(turn.turn_id, content="ship on Friday"))
        store.delete(DeleteOp(mem_id=first.mem_id))

        second = store.create(make_create(turn.turn_id, content="ship on Friday"))
        assert second.mem_id != first.mem_id
        assert store.count(status=MemoryStatus.ACTIVE) == 1

    def test_supersede_with_identical_content_still_replaces(self, store, turn):
        """The reason the old row is retired before the replacement is made.

        In the other order the replacement would find its own still-active
        predecessor as a duplicate and merge into it — no chain, no new row.
        """
        old = store.create(make_create(turn.turn_id, content="same text"))
        new = store.supersede(
            SupersedeOp(
                mem_id=old.mem_id,
                replacement=make_create(turn.turn_id, content="same text"),
            )
        )

        assert new is not None
        assert new.mem_id != old.mem_id
        assert new.supersedes == old.mem_id
        assert store.get(old.mem_id).status is MemoryStatus.SUPERSEDED

    def test_supersede_coalesces_unrelated_active_duplicate_without_losing_chains(
        self, store, db, turn
    ):
        prior = store.create(make_create(turn.turn_id, content="prior value"))
        duplicate_turn = TranscriptStore(db).append(
            "user", "The canonical replacement is independently supported."
        )
        duplicate = store.supersede(
            SupersedeOp(
                mem_id=prior.mem_id,
                replacement=make_create(
                    duplicate_turn.turn_id,
                    quote="canonical replacement",
                    content="canonical value",
                ),
            )
        )
        old = store.create(make_create(turn.turn_id, content="old value"))

        fresh = store.supersede(
            SupersedeOp(
                mem_id=old.mem_id,
                replacement=make_create(
                    turn.turn_id,
                    quote="dark mode",
                    content="canonical value",
                ),
            )
        )

        assert fresh is not None
        assert fresh.mem_id not in {old.mem_id, duplicate.mem_id}
        assert fresh.supersedes == old.mem_id
        assert store.get(old.mem_id).status is MemoryStatus.SUPERSEDED
        retired_duplicate = store.get(duplicate.mem_id)
        assert retired_duplicate.status is MemoryStatus.SUPERSEDED
        assert retired_duplicate.supersedes == prior.mem_id
        assert [item.mem_id for item in store.successors(old.mem_id)] == [fresh.mem_id]
        assert [item.mem_id for item in store.successors(duplicate.mem_id)] == [
            fresh.mem_id
        ]
        assert {
            item.mem_id
            for item in store.list_items()
            if item.content == "canonical value"
        } == {fresh.mem_id}
        assert {citation.turn_id for citation in fresh.provenance} == {
            turn.turn_id,
            duplicate_turn.turn_id,
        }

    def test_apply_reports_duplicates(self, store, turn):
        ops = MemoryOps(
            create=[
                make_create(turn.turn_id, content="a fact"),
                make_create(turn.turn_id, content="a fact"),
            ]
        )
        summary = store.apply(ops)
        assert summary["created"] == 1
        assert summary["duplicate"] == 1

    def test_dedupe_existing_collapses_legacy_duplicates(self, store, turn):
        a = store.create(make_create(turn.turn_id, content="legacy"))
        b = store.create(make_create(turn.turn_id, content="legacy"), dedupe=False)
        c = store.create(make_create(turn.turn_id, content="legacy"), dedupe=False)
        assert store.count(status=MemoryStatus.ACTIVE) == 3

        retired = store.dedupe_existing()

        assert retired == 2
        active = store.list_items(status=MemoryStatus.ACTIVE)
        assert len(active) == 1
        survivor = active[0].mem_id
        assert survivor in {a.mem_id, b.mem_id, c.mem_id}
        # Nothing destroyed; supplemental forward redirects point at the
        # survivor without reversing the scalar revision-link direction.
        for item in store.list_items(status=MemoryStatus.SUPERSEDED):
            assert item.supersedes is None
            assert [successor.mem_id for successor in store.successors(item.mem_id)] == [
                survivor
            ]

    def test_dedupe_existing_is_a_no_op_on_a_clean_store(self, store, turn):
        store.create(make_create(turn.turn_id, content="one"))
        store.create(make_create(turn.turn_id, content="two"))
        assert store.dedupe_existing() == 0

    @pytest.mark.parametrize(
        ("loser_created", "survivor_created"),
        [
            ("2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"),
            ("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        ],
        ids=["ordered", "equal-time"],
    )
    def test_successors_reads_pre_v12_reversed_dedupe_links(
        self, store, db, turn, loser_created, survivor_created
    ):
        loser = store.create(
            make_create(turn.turn_id, content="legacy duplicate"), dedupe=False
        )
        survivor = store.create(
            make_create(turn.turn_id, content="legacy duplicate"), dedupe=False
        )
        db.execute(
            "UPDATE memory_items SET created_at = ? WHERE mem_id = ?",
            (loser_created, loser.mem_id),
        )
        db.execute(
            "UPDATE memory_items SET created_at = ? WHERE mem_id = ?",
            (survivor_created, survivor.mem_id),
        )
        # Layout written by the old maintenance path: loser.supersedes points
        # forward at the active survivor instead of backward at a predecessor.
        db.execute(
            "UPDATE memory_items SET status = ?, supersedes = ? WHERE mem_id = ?",
            (MemoryStatus.SUPERSEDED.value, survivor.mem_id, loser.mem_id),
        )
        db.commit()

        assert [item.mem_id for item in store.successors(loser.mem_id)] == [
            survivor.mem_id
        ]
        assert store.successors(survivor.mem_id) == []

    def test_updating_content_moves_the_identity(self, store, turn):
        item = store.create(make_create(turn.turn_id, content="before"))
        store.update(UpdateOp(mem_id=item.mem_id, content="after"))

        # The new text now dedupes; the old text no longer does.
        store.create(make_create(turn.turn_id, content="after"))
        assert store.count(status=MemoryStatus.ACTIVE) == 1
        store.create(make_create(turn.turn_id, content="before"))
        assert store.count(status=MemoryStatus.ACTIVE) == 2
