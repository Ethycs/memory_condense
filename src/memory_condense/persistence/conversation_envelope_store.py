"""Durable user-led exchange envelopes over the append-only transcript."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.persistence.conversation_envelope_expansion import (
    DEFAULT_MAX_EXPANSION_COMPANION_CHUNKS,
    DEFAULT_MAX_EXPANSION_COMPANION_TOKENS,
    DEFAULT_MAX_EXPANSION_ENVELOPES,
    DEFAULT_MAX_EXPANSION_TURNS,
    ConversationEnvelopeExpansionPlan,
    plan_retrieval_expansion,
)
from memory_condense.persistence.db import Database, INDEXED_CHUNK_SQL
from memory_condense.persistence.pending_ingest_store import PendingIngestManifest


CONVERSATION_ENVELOPE_FORMAT = "memory-condense-conversation-envelope-v1"
CONVERSATION_ENVELOPE_DEFAULT_MAX_TURNS = 32
CONVERSATION_ENVELOPE_HARD_MAX_TURNS = 512
CONVERSATION_ENVELOPE_POLICY_SHA256 = identity_sha256(
    {
        "schema": "conversation-envelope-policy-v1",
        "boundary": "user-opens-next-user-closes-by-predecessor-link",
        "membership": "assistant-system-to-latest-prior-user-in-source",
        "ordering": "durable-global-turn-ordinal-with-source-pending-barrier",
        "authority": {
            "user": "user_assertion",
            "assistant": "machine_generated",
            "system": "system_instruction",
        },
        "raw_text": "authoritative-turn-table-only",
    }
)

_AUTHORITY_BY_ROLE = {
    "user": "user_assertion",
    "assistant": "machine_generated",
    "system": "system_instruction",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure_kind(error: BaseException) -> str:
    return f"{type(error).__module__}.{type(error).__qualname__}"[:255]


def _validate_limit(max_turns: int) -> None:
    if (
        isinstance(max_turns, bool)
        or not isinstance(max_turns, int)
        or max_turns < 1
        or max_turns > CONVERSATION_ENVELOPE_HARD_MAX_TURNS
    ):
        raise ValueError(
            "max_turns must be an integer from 1 through "
            f"{CONVERSATION_ENVELOPE_HARD_MAX_TURNS}"
        )


@dataclass(frozen=True, slots=True)
class ConversationEnvelopeEvent:
    """One immutable turn-to-envelope assignment; it contains no raw text."""

    policy_sha256: str
    turn_id: str
    envelope_id: str
    opener_turn_id: str
    predecessor_envelope_id: str | None
    event_kind: str
    source_id: str
    turn_ordinal: int
    actor_kind: str
    authority_kind: str
    parent_turn_id: str | None
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ConversationEnvelopeAssignment:
    """Terminal result for one assignment attempt."""

    created: bool
    turn_id: str
    status: str
    terminal_reason: str | None
    receipt_sha256: str
    event: ConversationEnvelopeEvent | None


@dataclass(frozen=True, slots=True)
class ConversationEnvelopeBootstrapResult:
    """One finite, resumable current-policy bootstrap page."""

    policy_sha256: str
    selected_turn_ids: tuple[str, ...]
    claimed_turn_ids: tuple[str, ...]
    completed_turn_ids: tuple[str, ...]
    pending_turn_ids: tuple[str, ...]
    remaining_turn_count: int
    unsupported_turn_count: int


class ConversationEnvelopeStore:
    """Journal and publish deterministic, append-only user-led envelopes."""

    def __init__(
        self,
        db: Database,
        *,
        policy_sha256: str = CONVERSATION_ENVELOPE_POLICY_SHA256,
    ) -> None:
        if len(policy_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in policy_sha256
        ):
            raise ValueError("policy_sha256 must be a lowercase SHA-256 digest")
        self._db = db
        self._policy_sha256 = policy_sha256
        self.last_failures: dict[str, str] = {}

    @property
    def policy_sha256(self) -> str:
        return self._policy_sha256

    def _ensure_policy(self) -> None:
        if not self._db.connection.in_transaction:
            raise RuntimeError("envelope policy creation requires caller transaction")
        self._db.execute(
            "INSERT INTO conversation_envelope_policies "
            "(policy_sha256, format, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(policy_sha256) DO NOTHING",
            (self._policy_sha256, CONVERSATION_ENVELOPE_FORMAT, _utc_now()),
        )
        row = self._db.execute(
            "SELECT format FROM conversation_envelope_policies "
            "WHERE policy_sha256 = ?",
            (self._policy_sha256,),
        ).fetchone()
        if row is None or str(row[0]) != CONVERSATION_ENVELOPE_FORMAT:
            raise ValueError("conversation envelope policy identity collision")

    @staticmethod
    def _input_sha256(
        *,
        policy_sha256: str,
        turn_id: str,
        source_id: str | None,
        turn_ordinal: int,
        role: str,
        actor_kind: str,
        authority_kind: str,
        parent_turn_id: str | None,
    ) -> str:
        return identity_sha256(
            {
                "schema": "conversation-envelope-assignment-input-v1",
                "policy_sha256": policy_sha256,
                "turn_id": turn_id,
                "source_id": source_id,
                "turn_ordinal": turn_ordinal,
                "role": role,
                "actor_kind": actor_kind,
                "authority_kind": authority_kind,
                "parent_turn_id": parent_turn_id,
            }
        )

    def claim_many(
        self,
        turn_ids: Sequence[str],
        *,
        parent_turn_ids: Mapping[str, str | None] | None = None,
    ) -> dict[str, str]:
        """Claim assignment inputs inside the caller's T0 transaction."""

        normalized = tuple(dict.fromkeys(str(value).strip() for value in turn_ids))
        if any(not value for value in normalized):
            raise ValueError("turn IDs must be non-empty")
        if not normalized:
            return {}
        if not self._db.connection.in_transaction:
            raise RuntimeError("claim_many requires an active caller transaction")
        parents = dict(parent_turn_ids or {})
        if set(parents) - set(normalized):
            raise ValueError("parent mapping contains an unclaimed turn")
        self._ensure_policy()
        placeholders = ",".join("?" for _ in normalized)
        rows = self._db.execute(
            "SELECT turn_id, source_id, ordinal, role FROM turns "
            f"WHERE turn_id IN ({placeholders})",
            normalized,
        ).fetchall()
        by_turn = {str(row[0]): row for row in rows}
        if set(by_turn) != set(normalized):
            raise ValueError("conversation envelope claim references an unknown turn")
        now = _utc_now()
        claims: list[tuple[object, ...]] = []
        expected: dict[str, tuple[object, ...]] = {}
        for turn_id in normalized:
            _stored_id, raw_source, raw_ordinal, raw_role = by_turn[turn_id]
            source_id = None if raw_source is None else str(raw_source)
            role = str(raw_role)
            if role not in _AUTHORITY_BY_ROLE:
                raise ValueError("conversation envelope claim has an unsupported role")
            actor_kind = role
            authority_kind = _AUTHORITY_BY_ROLE[role]
            parent_turn_id = parents.get(turn_id)
            if parent_turn_id is not None:
                parent_turn_id = str(parent_turn_id).strip()
                if not parent_turn_id:
                    raise ValueError("parent turn ID must be non-empty")
            ordinal = int(raw_ordinal)
            input_sha256 = self._input_sha256(
                policy_sha256=self._policy_sha256,
                turn_id=turn_id,
                source_id=source_id,
                turn_ordinal=ordinal,
                role=role,
                actor_kind=actor_kind,
                authority_kind=authority_kind,
                parent_turn_id=parent_turn_id,
            )
            values = (
                self._policy_sha256,
                turn_id,
                source_id,
                ordinal,
                role,
                actor_kind,
                authority_kind,
                parent_turn_id,
                input_sha256,
                now,
            )
            claims.append(values)
            expected[turn_id] = values[2:9]
        self._db.connection.executemany(
            "INSERT INTO pending_conversation_envelope_assignments "
            "(policy_sha256, turn_id, source_id, turn_ordinal, role, actor_kind, "
            " authority_kind, parent_turn_id, input_sha256, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?) "
            "ON CONFLICT(policy_sha256, turn_id) DO NOTHING",
            claims,
        )
        stored_rows = self._db.execute(
            "SELECT turn_id, source_id, turn_ordinal, role, actor_kind, "
            "authority_kind, parent_turn_id, input_sha256, status "
            "FROM pending_conversation_envelope_assignments "
            "WHERE policy_sha256 = ? "
            f"AND turn_id IN ({placeholders})",
            (self._policy_sha256, *normalized),
        ).fetchall()
        stored = {str(row[0]): row[1:] for row in stored_rows}
        output: dict[str, str] = {}
        for turn_id in normalized:
            row = stored.get(turn_id)
            if row is None or tuple(row[:7]) != expected[turn_id]:
                raise ValueError("turn already has a different envelope assignment")
            output[turn_id] = str(row[7])
        return output

    def status(self, turn_id: str) -> str | None:
        row = self._db.execute(
            "SELECT status FROM pending_conversation_envelope_assignments "
            "WHERE policy_sha256 = ? AND turn_id = ?",
            (self._policy_sha256, turn_id),
        ).fetchone()
        return None if row is None else str(row[0])

    def assignment(
        self, turn_id: str
    ) -> ConversationEnvelopeAssignment | None:
        """Return a terminal assignment receipt, including no-anchor reason."""

        return self._terminal_assignment(turn_id)

    def pending_count(self) -> int:
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM pending_conversation_envelope_assignments "
                "WHERE policy_sha256 = ? AND status = 'pending'",
                (self._policy_sha256,),
            ).fetchone()[0]
        )

    def _event_from_row(self, row: tuple[object, ...]) -> ConversationEnvelopeEvent:
        return ConversationEnvelopeEvent(
            policy_sha256=str(row[0]),
            turn_id=str(row[1]),
            envelope_id=str(row[2]),
            opener_turn_id=str(row[3]),
            predecessor_envelope_id=(None if row[4] is None else str(row[4])),
            event_kind=str(row[5]),
            source_id=str(row[6]),
            turn_ordinal=int(row[7]),
            actor_kind=str(row[8]),
            authority_kind=str(row[9]),
            parent_turn_id=(None if row[10] is None else str(row[10])),
            receipt_sha256=str(row[11]),
        )

    def event_for_turn(self, turn_id: str) -> ConversationEnvelopeEvent | None:
        row = self._db.execute(
            "SELECT policy_sha256, turn_id, envelope_id, opener_turn_id, "
            "predecessor_envelope_id, event_kind, source_id, turn_ordinal, "
            "actor_kind, authority_kind, parent_turn_id, receipt_sha256 "
            "FROM conversation_envelope_events "
            "WHERE policy_sha256 = ? AND turn_id = ?",
            (self._policy_sha256, turn_id),
        ).fetchone()
        return None if row is None else self._event_from_row(row)

    def events_for_envelope(
        self, envelope_id: str
    ) -> tuple[ConversationEnvelopeEvent, ...]:
        rows = self._db.execute(
            "SELECT policy_sha256, turn_id, envelope_id, opener_turn_id, "
            "predecessor_envelope_id, event_kind, source_id, turn_ordinal, "
            "actor_kind, authority_kind, parent_turn_id, receipt_sha256 "
            "FROM conversation_envelope_events "
            "WHERE policy_sha256 = ? AND envelope_id = ? "
            "ORDER BY turn_ordinal, turn_id",
            (self._policy_sha256, envelope_id),
        ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def plan_retrieval_expansion(
        self,
        anchor_turn_ids: Sequence[str],
        *,
        original_chunk_token_counts: Mapping[str, int] | None = None,
        max_envelopes: int = DEFAULT_MAX_EXPANSION_ENVELOPES,
        max_turns_per_envelope: int = DEFAULT_MAX_EXPANSION_TURNS,
        max_companion_chunks: int = DEFAULT_MAX_EXPANSION_COMPANION_CHUNKS,
        max_companion_tokens: int = DEFAULT_MAX_EXPANSION_COMPANION_TOKENS,
    ) -> ConversationEnvelopeExpansionPlan:
        """Return a bounded, text-free envelope plan for retrieval anchors."""

        return plan_retrieval_expansion(
            self,
            anchor_turn_ids,
            original_chunk_token_counts=original_chunk_token_counts,
            max_envelopes=max_envelopes,
            max_turns_per_envelope=max_turns_per_envelope,
            max_companion_chunks=max_companion_chunks,
            max_companion_tokens=max_companion_tokens,
        )

    def live_chunk_ids(self, chunk_ids: Sequence[str]) -> frozenset[str]:
        """Return IDs satisfying the same durable predicate as live search."""

        normalized = tuple(
            dict.fromkeys(str(chunk_id).strip() for chunk_id in chunk_ids)
        )
        if any(not chunk_id for chunk_id in normalized):
            raise ValueError("chunk IDs must be non-empty")
        if not normalized:
            return frozenset()
        placeholders = ",".join("?" for _ in normalized)
        rows = self._db.execute(
            "SELECT c.chunk_id FROM chunks AS c "
            f"WHERE c.chunk_id IN ({placeholders}) AND {INDEXED_CHUNK_SQL}",
            normalized,
        ).fetchall()
        return frozenset(str(row[0]) for row in rows)

    def current_envelope(self, source_id: str) -> ConversationEnvelopeEvent | None:
        normalized = str(source_id).strip()
        if not normalized:
            raise ValueError("source_id must be non-empty")
        row = self._db.execute(
            "SELECT policy_sha256, turn_id, envelope_id, opener_turn_id, "
            "predecessor_envelope_id, event_kind, source_id, turn_ordinal, "
            "actor_kind, authority_kind, parent_turn_id, receipt_sha256 "
            "FROM conversation_envelope_events "
            "WHERE policy_sha256 = ? AND source_id = ? AND event_kind = 'open' "
            "ORDER BY turn_ordinal DESC, turn_id DESC LIMIT 1",
            (self._policy_sha256, normalized),
        ).fetchone()
        return None if row is None else self._event_from_row(row)

    def _terminal_assignment(self, turn_id: str) -> ConversationEnvelopeAssignment | None:
        row = self._db.execute(
            "SELECT status, terminal_reason, receipt_sha256 FROM "
            "pending_conversation_envelope_assignments "
            "WHERE policy_sha256 = ? AND turn_id = ?",
            (self._policy_sha256, turn_id),
        ).fetchone()
        if row is None or str(row[0]) == "pending":
            return None
        status = str(row[0])
        terminal_reason = None if row[1] is None else str(row[1])
        receipt = str(row[2])
        event = self.event_for_turn(turn_id)
        if (status == "ready") != (event is not None):
            raise ValueError("envelope terminal receipt disagrees with its event")
        return ConversationEnvelopeAssignment(
            False, turn_id, status, terminal_reason, receipt, event
        )

    def _latest_open(
        self, *, source_id: str, before_ordinal: int
    ) -> ConversationEnvelopeEvent | None:
        row = self._db.execute(
            "SELECT policy_sha256, turn_id, envelope_id, opener_turn_id, "
            "predecessor_envelope_id, event_kind, source_id, turn_ordinal, "
            "actor_kind, authority_kind, parent_turn_id, receipt_sha256 "
            "FROM conversation_envelope_events "
            "WHERE policy_sha256 = ? AND source_id = ? AND event_kind = 'open' "
            "AND turn_ordinal < ? ORDER BY turn_ordinal DESC, turn_id DESC LIMIT 1",
            (self._policy_sha256, source_id, before_ordinal),
        ).fetchone()
        return None if row is None else self._event_from_row(row)

    def _complete_no_anchor(
        self, *, turn_id: str, input_sha256: str, reason: str
    ) -> ConversationEnvelopeAssignment:
        receipt = identity_sha256(
            {
                "schema": "conversation-envelope-terminal-v1",
                "policy_sha256": self._policy_sha256,
                "turn_id": turn_id,
                "input_sha256": input_sha256,
                "status": "no_anchor",
                "terminal_reason": reason,
            }
        )
        changed = self._db.execute(
            "UPDATE pending_conversation_envelope_assignments "
            "SET status = 'no_anchor', completed_at = ?, terminal_reason = ?, "
            "receipt_sha256 = ? "
            "WHERE policy_sha256 = ? AND turn_id = ? AND status = 'pending'",
            (_utc_now(), reason, receipt, self._policy_sha256, turn_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError("envelope no-anchor CAS did not advance")
        return ConversationEnvelopeAssignment(
            True, turn_id, "no_anchor", reason, receipt, None
        )

    def _publish_one(self, turn_id: str) -> ConversationEnvelopeAssignment | None:
        existing = self._terminal_assignment(turn_id)
        if existing is not None:
            return existing
        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT source_id, turn_ordinal, role, actor_kind, authority_kind, "
                "parent_turn_id, input_sha256, status "
                "FROM pending_conversation_envelope_assignments "
                "WHERE policy_sha256 = ? AND turn_id = ?",
                (self._policy_sha256, turn_id),
            ).fetchone()
            if row is None:
                raise ValueError("unknown conversation envelope assignment")
            if str(row[7]) != "pending":
                connection.commit()
                terminal = self._terminal_assignment(turn_id)
                if terminal is None:
                    raise RuntimeError("terminal envelope assignment disappeared")
                return terminal
            source_id = None if row[0] is None else str(row[0])
            ordinal = int(row[1])
            role = str(row[2])
            actor_kind = str(row[3])
            authority_kind = str(row[4])
            parent_turn_id = None if row[5] is None else str(row[5])
            input_sha256 = str(row[6])

            # A later worker may not overtake unfinished same-session history.
            if source_id is not None and self._db.execute(
                "SELECT 1 FROM pending_conversation_envelope_assignments "
                "WHERE policy_sha256 = ? AND source_id = ? AND status = 'pending' "
                "AND turn_ordinal < ? LIMIT 1",
                (self._policy_sha256, source_id, ordinal),
            ).fetchone() is not None:
                connection.commit()
                return None

            if source_id is None or not source_id.strip():
                result = self._complete_no_anchor(
                    turn_id=turn_id,
                    input_sha256=input_sha256,
                    reason="missing_source",
                )
                connection.commit()
                return result

            prior = self._latest_open(source_id=source_id, before_ordinal=ordinal)
            if role == "user":
                event_kind = "open"
                opener_turn_id = turn_id
                envelope_id = identity_sha256(
                    {
                        "schema": "conversation-envelope-id-v1",
                        "policy_sha256": self._policy_sha256,
                        "source_id": source_id,
                        "opener_turn_id": turn_id,
                    }
                )
                predecessor_envelope_id = (
                    None if prior is None else prior.envelope_id
                )
            else:
                if parent_turn_id is not None:
                    parent = self.event_for_turn(parent_turn_id)
                    if (
                        parent is None
                        or parent.source_id != source_id
                        or parent.turn_ordinal >= ordinal
                    ):
                        result = self._complete_no_anchor(
                            turn_id=turn_id,
                            input_sha256=input_sha256,
                            reason="invalid_parent",
                        )
                        connection.commit()
                        return result
                    prior = self.event_for_turn(parent.opener_turn_id)
                if prior is None:
                    result = self._complete_no_anchor(
                        turn_id=turn_id,
                        input_sha256=input_sha256,
                        reason="no_prior_user",
                    )
                    connection.commit()
                    return result
                event_kind = "member"
                opener_turn_id = prior.opener_turn_id
                envelope_id = prior.envelope_id
                predecessor_envelope_id = None

            receipt = identity_sha256(
                {
                    "schema": "conversation-envelope-event-v1",
                    "policy_sha256": self._policy_sha256,
                    "turn_id": turn_id,
                    "input_sha256": input_sha256,
                    "envelope_id": envelope_id,
                    "opener_turn_id": opener_turn_id,
                    "predecessor_envelope_id": predecessor_envelope_id,
                    "event_kind": event_kind,
                    "source_id": source_id,
                    "turn_ordinal": ordinal,
                    "actor_kind": actor_kind,
                    "authority_kind": authority_kind,
                    "parent_turn_id": parent_turn_id,
                }
            )
            self._db.execute(
                "INSERT INTO conversation_envelope_events "
                "(policy_sha256, turn_id, envelope_id, opener_turn_id, "
                " predecessor_envelope_id, event_kind, source_id, turn_ordinal, "
                " actor_kind, authority_kind, parent_turn_id, receipt_sha256, "
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self._policy_sha256,
                    turn_id,
                    envelope_id,
                    opener_turn_id,
                    predecessor_envelope_id,
                    event_kind,
                    source_id,
                    ordinal,
                    actor_kind,
                    authority_kind,
                    parent_turn_id,
                    receipt,
                    _utc_now(),
                ),
            )
            changed = self._db.execute(
                "UPDATE pending_conversation_envelope_assignments "
                "SET status = 'ready', completed_at = ?, receipt_sha256 = ? "
                "WHERE policy_sha256 = ? AND turn_id = ? AND status = 'pending'",
                (_utc_now(), receipt, self._policy_sha256, turn_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("envelope ready CAS did not advance")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        event = self.event_for_turn(turn_id)
        if event is None or event.receipt_sha256 != receipt:
            raise RuntimeError("committed conversation envelope event is missing")
        return ConversationEnvelopeAssignment(
            True, turn_id, "ready", None, receipt, event
        )

    def _record_failure(self, turn_id: str, error: BaseException) -> None:
        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._db.execute(
                "UPDATE pending_conversation_envelope_assignments "
                "SET attempt_count = attempt_count + 1, last_attempt_at = ?, "
                "last_error_kind = ? WHERE policy_sha256 = ? AND turn_id = ? "
                "AND status = 'pending'",
                (_utc_now(), _failure_kind(error), self._policy_sha256, turn_id),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def _pending_turn_ids(
        self,
        *,
        max_turns: int,
        turn_ids: Sequence[str] | None,
    ) -> tuple[str, ...]:
        params: list[object] = [self._policy_sha256]
        sql = (
            "SELECT turn_id FROM pending_conversation_envelope_assignments "
            "WHERE policy_sha256 = ? AND status = 'pending' "
        )
        if turn_ids is not None:
            normalized = tuple(dict.fromkeys(str(value) for value in turn_ids))
            if not normalized:
                return ()
            sql += "AND turn_id IN (" + ",".join("?" for _ in normalized) + ") "
            params.extend(normalized)
        sql += "ORDER BY turn_ordinal, turn_id LIMIT ?"
        params.append(max_turns)
        return tuple(
            str(row[0]) for row in self._db.execute(sql, tuple(params)).fetchall()
        )

    def drain_pending(
        self,
        *,
        max_turns: int = CONVERSATION_ENVELOPE_DEFAULT_MAX_TURNS,
        turn_ids: Sequence[str] | None = None,
    ) -> list[ConversationEnvelopeAssignment]:
        """Publish at most one finite page, isolating failures from raw T0/T1."""

        _validate_limit(max_turns)
        selected = self._pending_turn_ids(max_turns=max_turns, turn_ids=turn_ids)
        output: list[ConversationEnvelopeAssignment] = []
        failures: dict[str, str] = {}
        for turn_id in selected:
            try:
                result = self._publish_one(turn_id)
                if result is not None:
                    output.append(result)
            except Exception as error:
                failures[turn_id] = _failure_kind(error)
                try:
                    self._record_failure(turn_id, error)
                except Exception as receipt_error:
                    failures[turn_id] += f";receipt={_failure_kind(receipt_error)}"
        self.last_failures = failures
        return output

    def _claim_bootstrap_page(
        self, *, max_turns: int
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        connection = self._db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = self._db.execute(
                "SELECT t.turn_id, p.manifest_sha256, p.manifest_json, e.status, "
                "t.ordinal "
                "FROM turns AS t "
                "JOIN pending_ingests AS p ON p.turn_id = t.turn_id "
                "LEFT JOIN pending_conversation_envelope_assignments AS e "
                "ON e.policy_sha256 = ? AND e.turn_id = t.turn_id "
                "WHERE e.turn_id IS NULL OR e.status = 'pending' "
                "ORDER BY t.ordinal, t.turn_id LIMIT ?",
                (self._policy_sha256, max_turns),
            ).fetchall()
            selected_values: list[str] = []
            missing_values: list[str] = []
            missing_ordinals: list[int] = []
            for turn_id, manifest_sha256, manifest_json, status, ordinal in rows:
                normalized_turn_id = str(turn_id)
                manifest = PendingIngestManifest.from_json(str(manifest_json))
                if (
                    manifest.turn_id != normalized_turn_id
                    or manifest.sha256 != str(manifest_sha256)
                ):
                    raise ValueError(
                        "envelope bootstrap ingest manifest receipt is inconsistent"
                    )
                selected_values.append(normalized_turn_id)
                if status is None:
                    missing_values.append(normalized_turn_id)
                    missing_ordinals.append(int(ordinal))
            selected = tuple(selected_values)
            missing = tuple(missing_values)
            published_max = self._db.execute(
                "SELECT MAX(turn_ordinal) FROM conversation_envelope_events "
                "WHERE policy_sha256 = ?",
                (self._policy_sha256,),
            ).fetchone()[0]
            if (
                missing_ordinals
                and published_max is not None
                and min(missing_ordinals) < int(published_max)
            ):
                raise ValueError(
                    "conversation envelope backfill would precede immutable "
                    "published events; use a fresh policy"
                )
            if missing:
                statuses = self.claim_many(missing)
                if any(statuses[turn_id] != "pending" for turn_id in missing):
                    raise RuntimeError("envelope bootstrap claim was not pending")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return selected, missing

    def bootstrap(
        self,
        *,
        max_turns: int = CONVERSATION_ENVELOPE_DEFAULT_MAX_TURNS,
    ) -> ConversationEnvelopeBootstrapResult:
        """Claim and drain one current-policy page from durable ingest receipts."""

        _validate_limit(max_turns)
        selected, claimed = self._claim_bootstrap_page(max_turns=max_turns)
        if selected:
            self.drain_pending(max_turns=len(selected), turn_ids=selected)
        statuses = {turn_id: self.status(turn_id) for turn_id in selected}
        completed = tuple(
            turn_id
            for turn_id in selected
            if statuses[turn_id] in {"ready", "no_anchor"}
        )
        pending = tuple(
            turn_id for turn_id in selected if statuses[turn_id] == "pending"
        )
        remaining = int(
            self._db.execute(
                "SELECT COUNT(*) FROM turns AS t "
                "JOIN pending_ingests AS p ON p.turn_id = t.turn_id "
                "LEFT JOIN pending_conversation_envelope_assignments AS e "
                "ON e.policy_sha256 = ? AND e.turn_id = t.turn_id "
                "WHERE e.turn_id IS NULL OR e.status = 'pending'",
                (self._policy_sha256,),
            ).fetchone()[0]
        )
        unsupported = int(
            self._db.execute(
                "SELECT COUNT(*) FROM turns AS t WHERE NOT EXISTS ("
                "SELECT 1 FROM pending_ingests AS p WHERE p.turn_id = t.turn_id) "
                "AND NOT EXISTS (SELECT 1 FROM "
                "pending_conversation_envelope_assignments AS e "
                "WHERE e.policy_sha256 = ? AND e.turn_id = t.turn_id)",
                (self._policy_sha256,),
            ).fetchone()[0]
        )
        return ConversationEnvelopeBootstrapResult(
            policy_sha256=self._policy_sha256,
            selected_turn_ids=selected,
            claimed_turn_ids=claimed,
            completed_turn_ids=completed,
            pending_turn_ids=pending,
            remaining_turn_count=remaining,
            unsupported_turn_count=unsupported,
        )


__all__ = [
    "CONVERSATION_ENVELOPE_DEFAULT_MAX_TURNS",
    "CONVERSATION_ENVELOPE_FORMAT",
    "CONVERSATION_ENVELOPE_HARD_MAX_TURNS",
    "CONVERSATION_ENVELOPE_POLICY_SHA256",
    "ConversationEnvelopeExpansionPlan",
    "ConversationEnvelopeAssignment",
    "ConversationEnvelopeBootstrapResult",
    "ConversationEnvelopeEvent",
    "ConversationEnvelopeStore",
]
