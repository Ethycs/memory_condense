from datetime import timedelta

import numpy as np
import pytest

from memory_condense import decay
from memory_condense.memory_store import MemoryStore
from memory_condense.schemas import (
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
from memory_condense.transcript_store import TranscriptStore
from memory_condense.validator import Validator

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

    far_future = decay.now_utc() + timedelta(days=28)  # four half-lives
    plain_item = store.get(plain.mem_id)
    kept_item = store.get(kept.mem_id)

    assert decay.item_energy(plain_item, now=far_future) < 0.05
    assert decay.item_energy(kept_item, now=far_future) == pytest.approx(kept_item.energy)
    assert decay.item_heat(plain_item, now=far_future) is Heat.COLD
    assert decay.item_heat(kept_item, now=far_future) is Heat.WARM


def test_touch_reheats_a_decayed_item(store, turn):
    item = store.create(make_create(turn.turn_id, importance=0.1))
    stale = decay.now_utc() + timedelta(days=7)  # exactly one half-life

    decayed = decay.item_energy(store.get(item.mem_id), now=stale)
    assert decayed == pytest.approx(decay.WARM_SEED_ENERGY / 2, abs=1e-3)

    touched = store.touch(item.mem_id, now=stale)
    assert touched.energy > decayed
    assert touched.energy == pytest.approx(decay.reheat(decayed), abs=1e-6)
    assert touched.last_access_at.replace(tzinfo=None) == stale.replace(tzinfo=None)


def test_touch_caps_energy_at_one(store, turn):
    item = store.create(make_create(turn.turn_id, importance=0.9))
    now = decay.now_utc()
    for _ in range(10):
        item = store.touch(item.mem_id, now=now)
    assert item.energy == pytest.approx(1.0)


def test_touch_keeps_pinned_energy_but_restamps(store, turn):
    item = store.create(make_create(turn.turn_id))
    store.pin(PinOp(mem_id=item.mem_id, pin=PinState.USER))
    later = decay.now_utc() + timedelta(days=14)

    touched = store.touch(item.mem_id, now=later)
    assert touched.energy == pytest.approx(item.energy)
    assert touched.last_access_at.replace(tzinfo=None) == later.replace(tzinfo=None)


def test_touch_missing_returns_none(store):
    assert store.touch("ghost") is None


def test_heat_counts(store, turn):
    store.create(make_create(turn.turn_id, content="hot", importance=0.9))
    store.create(make_create(turn.turn_id, content="warm", importance=0.2))
    counts = store.heat_counts()

    assert counts == {"HOT": 1, "WARM": 1, "COLD": 0}

    cold = store.heat_counts(now=decay.now_utc() + timedelta(days=30))
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


def test_retrieve_populates_score_components(store, turn):
    item = store.create(make_create(turn.turn_id, importance=0.9), embedding=[1.0, 0.0])
    store.pin(PinOp(mem_id=item.mem_id, pin=PinState.USER))

    result = store.retrieve(np.asarray([1.0, 0.0]), k=1)[0]
    assert result.relevance == pytest.approx(1.0)
    assert result.importance == pytest.approx(0.9)
    assert result.recency == pytest.approx(1.0, abs=1e-3)
    assert result.pin_boost == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0 + 0.3 * 0.9 + 0.5 * 1.0 + 0.2 * 1.0, abs=1e-3)


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
    pinned = store.create(make_create(turn.turn_id, content="pinned"), embedding=[0.0, 1.0])
    store.pin(PinOp(mem_id=pinned.mem_id, pin=PinState.USER))

    results = store.retrieve(np.asarray([1.0, 0.0]), k=2)
    assert results[0].item.mem_id == pinned.mem_id


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
    later = decay.now_utc() + timedelta(days=7)  # exactly one half-life
    seeded = item.energy

    result = store.retrieve(None, k=1, now=later)[0]
    expected = decay.reheat(seeded / 2)

    # The reheat is persisted...
    stored = store.get(item.mem_id)
    assert stored.energy == pytest.approx(expected, abs=1e-3)
    assert stored.energy > seeded / 2
    assert stored.last_access_at.replace(tzinfo=None) == later.replace(tzinfo=None)
    # ...and the returned MemoryResult carries the refreshed item.
    assert result.item.energy == pytest.approx(expected, abs=1e-3)


def test_retrieve_does_not_touch_unreturned_items(store, turn):
    kept = store.create(make_create(turn.turn_id, content="kept", importance=0.9))
    ignored = store.create(make_create(turn.turn_id, content="ignored", importance=0.1))
    before = store.get(ignored.mem_id).last_access_at

    results = store.retrieve(None, k=1)
    assert results[0].item.mem_id == kept.mem_id
    assert store.get(ignored.mem_id).last_access_at == before


def test_retrieve_honours_k(store, turn):
    for i in range(5):
        store.create(make_create(turn.turn_id, content=f"item {i}"))
    assert len(store.retrieve(None, k=3)) == 3
    assert store.retrieve(None, k=0) == []


def test_retrieve_on_empty_store(store):
    assert store.retrieve(np.asarray([1.0, 0.0]), k=5) == []
