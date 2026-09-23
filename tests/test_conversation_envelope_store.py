"""Focused lifecycle tests for append-only user-led conversation envelopes."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.conversation_envelope_store import (
    CONVERSATION_ENVELOPE_DEFAULT_MAX_TURNS,
    CONVERSATION_ENVELOPE_HARD_MAX_TURNS,
    ConversationEnvelopeStore,
)
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore


class _Embedder:
    dim = 8

    def embed_query(self, _query: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        vector = self.embed_query("").tolist()
        return [chunk.model_copy(update={"embedding": vector}) for chunk in chunks]


def _condenser(path) -> MemoryCondenser:
    return MemoryCondenser(
        data_dir=path,
        embedder=_Embedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    )


def _publish_for_envelope(
    db: Database,
    transcript: TranscriptStore,
    store: ConversationEnvelopeStore,
    *,
    turn_id: str,
    role: str,
    source_id: str | None,
    parent_turn_id: str | None = None,
) -> None:
    connection = db.connection
    try:
        connection.execute("BEGIN IMMEDIATE")
        transcript.publish_turn(
            transcript.stage(
                role,
                f"text for {turn_id}",
                source_id=source_id,
                turn_id=turn_id,
            ),
            compare_created_at=False,
            commit=False,
        )
        store.claim_many(
            (turn_id,),
            parent_turn_ids={turn_id: parent_turn_id},
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def test_t0_claim_contains_no_raw_text_and_capture_does_not_assign(tmp_path) -> None:
    with _condenser(tmp_path / "claim") as condenser:
        turn, _chunks = condenser.capture(
            "user",
            "The raw amber envelope text must remain only in turns.",
            source_id="session-a",
            turn_id="claim-user",
        )

        assert condenser.pending_conversation_envelope_count() == 1
        assert condenser.conversation_envelope_for_turn(turn.turn_id) is None
        row = condenser._db.execute(
            "SELECT source_id, role, actor_kind, authority_kind, status "
            "FROM pending_conversation_envelope_assignments WHERE turn_id = ?",
            (turn.turn_id,),
        ).fetchone()
        assert row == (
            "session-a",
            "user",
            "user",
            "user_assertion",
            "pending",
        )
        columns = {
            str(row[1])
            for row in condenser._db.execute(
                "PRAGMA table_info(conversation_envelope_events)"
            ).fetchall()
        }
        assert "text" not in columns
        assert "quote" not in columns


def test_ingest_many_groups_exchange_and_next_user_closes_immutably(tmp_path) -> None:
    with _condenser(tmp_path / "group") as condenser:
        condenser.ingest_many(
            [
                ("user", "first request", "session-a", None, "u1"),
                ("assistant", "first answer", "session-a", None, "a1"),
                ("system", "machine status", "session-a", None, "s1"),
            ]
        )
        first = condenser.conversation_envelope_for_turn("u1")
        assistant = condenser.conversation_envelope_for_turn("a1")
        system = condenser.conversation_envelope_for_turn("s1")
        assert first is not None and first.event_kind == "open"
        assert first.predecessor_envelope_id is None
        assert assistant is not None and assistant.envelope_id == first.envelope_id
        assert assistant.authority_kind == "machine_generated"
        assert system is not None and system.envelope_id == first.envelope_id
        assert system.authority_kind == "system_instruction"
        sealed_first = first

        condenser.ingest_many(
            [
                ("user", "second request", "session-a", None, "u2"),
                ("assistant", "second answer", "session-a", None, "a2"),
            ]
        )
        second = condenser.conversation_envelope_for_turn("u2")
        second_answer = condenser.conversation_envelope_for_turn("a2")
        assert second is not None and second.envelope_id != first.envelope_id
        assert second.predecessor_envelope_id == first.envelope_id
        assert second_answer is not None
        assert second_answer.envelope_id == second.envelope_id
        assert condenser.conversation_envelope_for_turn("u1") == sealed_first
        assert tuple(
            event.turn_id
            for event in condenser.conversation_envelope_members(first.envelope_id)
        ) == ("u1", "a1", "s1")


def test_proxy_like_capture_batch_becomes_ready_on_pending_ingest_drain(
    tmp_path,
) -> None:
    with _condenser(tmp_path / "capture-drain") as condenser:
        condenser.capture_many(
            [
                ("user", "captured request", "proxy-session", None, "proxy-u"),
                (
                    "assistant",
                    "captured response",
                    "proxy-session",
                    None,
                    "proxy-a",
                ),
            ]
        )
        assert condenser.pending_conversation_envelope_count() == 2

        completed = condenser.drain_pending_ingests()

        assert len(completed) == 2
        user = condenser.conversation_envelope_for_turn("proxy-u")
        assistant = condenser.conversation_envelope_for_turn("proxy-a")
        assert user is not None and assistant is not None
        assert assistant.envelope_id == user.envelope_id
        assert condenser.pending_conversation_envelope_count() == 0


def test_no_source_and_pre_user_turns_terminalize_with_diagnostics(tmp_path) -> None:
    with _condenser(tmp_path / "no-anchor") as condenser:
        condenser.ingest_many(
            [
                ("user", "unscoped", None, None, "unscoped-user"),
                (
                    "assistant",
                    "arrived before a user",
                    "session-pre",
                    None,
                    "pre-assistant",
                ),
            ]
        )
        rows = condenser._db.execute(
            "SELECT turn_id, status, terminal_reason FROM "
            "pending_conversation_envelope_assignments ORDER BY turn_ordinal"
        ).fetchall()
        assert rows == [
            ("unscoped-user", "no_anchor", "missing_source"),
            ("pre-assistant", "no_anchor", "no_prior_user"),
        ]
        assert condenser.conversation_envelope_for_turn("unscoped-user") is None
        assert condenser.conversation_envelope_for_turn("pre-assistant") is None


def test_sources_are_isolated_and_same_session_pending_cannot_be_overtaken(
    tmp_path,
) -> None:
    db = Database(tmp_path / "isolation.db")
    try:
        transcript = TranscriptStore(db)
        store = ConversationEnvelopeStore(db)
        _publish_for_envelope(
            db,
            transcript,
            store,
            turn_id="a-user",
            role="user",
            source_id="source-a",
        )
        _publish_for_envelope(
            db,
            transcript,
            store,
            turn_id="a-answer",
            role="assistant",
            source_id="source-a",
        )
        _publish_for_envelope(
            db,
            transcript,
            store,
            turn_id="b-user",
            role="user",
            source_id="source-b",
        )

        # The later source can progress, but source-a's answer cannot overtake
        # its still-pending opener.
        assert store.drain_pending(max_turns=1, turn_ids=("a-answer",)) == []
        assert store.status("a-answer") == "pending"
        b_result = store.drain_pending(max_turns=1, turn_ids=("b-user",))
        assert len(b_result) == 1 and b_result[0].status == "ready"

        store.drain_pending(max_turns=3)
        a_user = store.event_for_turn("a-user")
        a_answer = store.event_for_turn("a-answer")
        b_user = store.event_for_turn("b-user")
        assert a_user is not None and a_answer is not None and b_user is not None
        assert a_answer.envelope_id == a_user.envelope_id
        assert b_user.envelope_id != a_user.envelope_id
    finally:
        db.close()


def test_explicit_parent_routes_to_older_envelope_and_rejects_future_parent(
    tmp_path,
) -> None:
    db = Database(tmp_path / "parent.db")
    try:
        transcript = TranscriptStore(db)
        store = ConversationEnvelopeStore(db)
        _publish_for_envelope(
            db,
            transcript,
            store,
            turn_id="old-user",
            role="user",
            source_id="session-parent",
        )
        assert store.drain_pending(max_turns=1)[0].status == "ready"
        _publish_for_envelope(
            db,
            transcript,
            store,
            turn_id="new-user",
            role="user",
            source_id="session-parent",
        )
        assert store.drain_pending(max_turns=1)[0].status == "ready"
        _publish_for_envelope(
            db,
            transcript,
            store,
            turn_id="late-tool-result",
            role="assistant",
            source_id="session-parent",
            parent_turn_id="old-user",
        )
        assert store.drain_pending(max_turns=1)[0].status == "ready"
        old = store.event_for_turn("old-user")
        routed = store.event_for_turn("late-tool-result")
        assert old is not None and routed is not None
        assert routed.envelope_id == old.envelope_id
        assert routed.parent_turn_id == "old-user"

        # Create an older unclaimed child, then publish its future parent. The
        # explicit chronology check must reject the backward attachment.
        connection = db.connection
        connection.execute("BEGIN IMMEDIATE")
        older = transcript.stage(
            "assistant",
            "older child",
            source_id="future-parent-session",
            turn_id="older-child",
        )
        transcript.publish_turn(older, compare_created_at=False, commit=False)
        future = transcript.stage(
            "user",
            "future parent",
            source_id="future-parent-session",
            turn_id="future-parent",
        )
        transcript.publish_turn(future, compare_created_at=False, commit=False)
        store.claim_many(("future-parent",))
        connection.commit()
        assert store.drain_pending(max_turns=1)[0].status == "ready"

        connection.execute("BEGIN IMMEDIATE")
        store.claim_many(
            ("older-child",),
            parent_turn_ids={"older-child": "future-parent"},
        )
        connection.commit()
        rejected = store.drain_pending(max_turns=1)[0]
        assert rejected.status == "no_anchor"
        assert rejected.terminal_reason == "invalid_parent"
    finally:
        db.close()


def test_exact_replay_and_restart_do_not_duplicate_events(tmp_path) -> None:
    data_dir = tmp_path / "replay"
    with _condenser(data_dir) as condenser:
        condenser.ingest_many(
            [
                ("user", "stable request", "session-replay", None, "stable-u"),
                (
                    "assistant",
                    "stable response",
                    "session-replay",
                    None,
                    "stable-a",
                ),
            ]
        )
        expected = condenser.conversation_envelope_for_turn("stable-a")

    with _condenser(data_dir) as restarted:
        restarted.ingest_many(
            [
                ("user", "stable request", "session-replay", None, "stable-u"),
                (
                    "assistant",
                    "stable response",
                    "session-replay",
                    None,
                    "stable-a",
                ),
            ]
        )
        assert restarted.conversation_envelope_for_turn("stable-a") == expected
        assert restarted._db.execute(
            "SELECT COUNT(*) FROM conversation_envelope_events"
        ).fetchone() == (2,)
        assert restarted.pending_conversation_envelope_count() == 0


def test_events_are_immutable_and_durable(tmp_path) -> None:
    with _condenser(tmp_path / "immutable") as condenser:
        condenser.ingest(
            "user", "immutable opener", source_id="session-i", turn_id="i-user"
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            condenser._db.execute(
                "UPDATE conversation_envelope_events SET actor_kind = 'assistant'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="durable"):
            condenser._db.execute("DELETE FROM conversation_envelope_events")


def test_envelope_failure_does_not_roll_back_t1_or_t1g(tmp_path, monkeypatch) -> None:
    with _condenser(tmp_path / "fail-open") as condenser:
        def fail_drain(*, max_turns, turn_ids=None):
            raise RuntimeError("synthetic envelope outage")

        monkeypatch.setattr(condenser._conversation_envelopes, "drain_pending", fail_drain)
        turn, chunks = condenser.ingest(
            "user", "searchable despite envelope failure", source_id="session-f"
        )

        assert chunks and all(chunk.embedding is not None for chunk in chunks)
        assert condenser._db.execute(
            "SELECT status FROM pending_ingests WHERE turn_id = ?",
            (turn.turn_id,),
        ).fetchone() == ("indexed",)
        assert condenser._conversation_graphs.status(turn.turn_id) == "ready"
        assert condenser._conversation_envelopes.status(turn.turn_id) == "pending"
        assert condenser.last_conversation_envelope_error.endswith("RuntimeError")


def test_bootstrap_is_bounded_idempotent_and_policy_scoped(tmp_path) -> None:
    with _condenser(tmp_path / "bootstrap") as condenser:
        condenser.capture_many(
            [
                ("user", "one", "session-b", None, "b-u1"),
                ("assistant", "two", "session-b", None, "b-a1"),
                ("user", "three", "session-b", None, "b-u2"),
            ]
        )
        # The T0 path claimed the active policy. A different policy models a
        # policy-version bootstrap without relabeling old events.
        changed = ConversationEnvelopeStore(condenser._db, policy_sha256="a" * 64)
        first = changed.bootstrap(max_turns=2)
        second = changed.bootstrap(max_turns=2)
        replay = changed.bootstrap(max_turns=2)

        assert len(first.selected_turn_ids) == 2
        assert first.claimed_turn_ids == first.selected_turn_ids
        assert first.remaining_turn_count == 1
        assert second.selected_turn_ids == ("b-u2",)
        assert second.remaining_turn_count == 0
        assert replay.selected_turn_ids == ()
        assert changed.pending_count() == 0
        assert condenser.pending_conversation_envelope_count() == 3


def test_bootstrap_rejects_missing_history_before_immutable_policy_tail(
    tmp_path,
) -> None:
    with _condenser(tmp_path / "unsafe-backfill") as condenser:
        condenser.capture_many(
            [
                ("user", "older opener", "session-tail", None, "tail-old"),
                ("user", "newer opener", "session-tail", None, "tail-new"),
            ]
        )
        changed = ConversationEnvelopeStore(condenser._db, policy_sha256="b" * 64)
        connection = condenser._db.connection
        connection.execute("BEGIN IMMEDIATE")
        changed.claim_many(("tail-new",))
        connection.commit()
        assert changed.drain_pending(max_turns=1)[0].status == "ready"

        with pytest.raises(ValueError, match="fresh policy"):
            changed.bootstrap(max_turns=1)

        assert changed.status("tail-old") is None
        assert changed.event_for_turn("tail-new") is not None


@pytest.mark.parametrize(
    "value",
    [0, -1, True, CONVERSATION_ENVELOPE_HARD_MAX_TURNS + 1],
)
def test_bounded_apis_reject_invalid_limits(tmp_path, value) -> None:
    with _condenser(tmp_path / f"bound-{value}") as condenser:
        with pytest.raises(ValueError, match="max_turns"):
            condenser.drain_pending_conversation_envelopes(max_turns=value)
        with pytest.raises(ValueError, match="max_turns"):
            condenser.bootstrap_conversation_envelopes(max_turns=value)

    assert CONVERSATION_ENVELOPE_DEFAULT_MAX_TURNS == 32
