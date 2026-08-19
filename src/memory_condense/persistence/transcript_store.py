from __future__ import annotations

import re
from datetime import datetime, timezone

from memory_condense.persistence.db import Database
from memory_condense.domain.schemas import Turn


_SOURCE_METADATA_RE = re.compile(
    r"^\[(?P<source>.+?) took place at (?P<timestamp>.+?)\]\s*$"
)


def parse_source_metadata(text: str) -> tuple[str, str] | None:
    """Parse the synthetic timestamp turn emitted at a source boundary.

    This intentionally recognizes only the ingest format, not arbitrary
    system evidence containing a date. Callers may therefore exclude these
    provenance rows while leaving ordinary system-authored facts eligible.
    """

    match = _SOURCE_METADATA_RE.fullmatch(text.strip())
    if match is None:
        return None
    return match.group("source").strip(), match.group("timestamp").strip()


class TranscriptStore:
    """Append-only store for conversation transcript turns."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def current_turn(self) -> int:
        """The conversation's position — **the decay coordinate**.

        Energy decays in turns, not seconds (see :mod:`memory_condense.domain.decay`),
        so this is the clock the whole memory layer reads. Delegates to
        :meth:`Database.current_turn`, which ``MemoryStore`` shares.
        """
        return self._db.current_turn()

    def append(
        self,
        role: str,
        text: str,
        *,
        source_id: str | None = None,
        created_at: datetime | None = None,
        turn_id: str | None = None,
    ) -> Turn:
        """Create and persist a turn with a generated or explicit stable ID.

        Advancing ``ordinal`` here is what makes decay happen: every item the
        conversation did *not* reach for this turn falls one turn further
        behind. Nothing else has to run — no sweep, no timer.
        """
        normalized_source = (
            source_id.strip() if source_id and source_id.strip() else None
        )
        normalized_turn_id = None
        if turn_id is not None:
            normalized_turn_id = str(turn_id).strip()
            if not normalized_turn_id:
                raise ValueError("turn_id must be non-empty when supplied")
        normalized_created_at = created_at
        if normalized_created_at is not None:
            normalized_created_at = (
                normalized_created_at.replace(tzinfo=timezone.utc)
                if normalized_created_at.tzinfo is None
                else normalized_created_at.astimezone(timezone.utc)
            )
        turn = Turn(
            role=role,
            text=text,
            source_id=normalized_source,
            **(
                {"turn_id": normalized_turn_id}
                if normalized_turn_id is not None
                else {}
            ),
            **(
                {"created_at": normalized_created_at}
                if normalized_created_at is not None
                else {}
            ),
        )
        ordinal = self.current_turn() + 1
        self._db.execute(
            "INSERT INTO turns (turn_id, role, text, source_id, created_at, ordinal)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                turn.turn_id,
                turn.role,
                turn.text,
                turn.source_id,
                turn.created_at.isoformat(),
                ordinal,
            ),
        )
        self._db.commit()
        return turn

    def get_turn(self, turn_id: str) -> Turn | None:
        """Retrieve a single turn by ID."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, source_id, created_at FROM turns WHERE turn_id = ?",
            (turn_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_turn(row)

    def get_recent(self, n: int = 20) -> list[Turn]:
        """Return the N most recent turns, ordered oldest-first."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, source_id, created_at FROM turns "
            "ORDER BY ordinal DESC LIMIT ?",
            (n,),
        )
        rows = cur.fetchall()
        return [self._row_to_turn(r) for r in reversed(rows)]

    def get_all(self) -> list[Turn]:
        """Return all turns, ordered by created_at."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, source_id, created_at FROM turns ORDER BY ordinal"
        )
        return [self._row_to_turn(r) for r in cur.fetchall()]

    def source_metadata(self, source_ids: list[str]) -> dict[str, str]:
        """Return the first system metadata turn for each requested source."""

        selected = list(dict.fromkeys(source_id for source_id in source_ids if source_id))
        if not selected:
            return {}
        placeholders = ",".join("?" for _ in selected)
        cur = self._db.execute(
            "SELECT source_id, text FROM turns "
            f"WHERE role = 'system' AND source_id IN ({placeholders}) "
            "ORDER BY ordinal",
            tuple(selected),
        )
        metadata: dict[str, str] = {}
        for source_id, text in cur.fetchall():
            metadata.setdefault(str(source_id), str(text))
        return metadata

    def find_containing(self, text: str) -> Turn | None:
        """The most recent turn containing ``text`` verbatim, if any.

        Used to give a memory *real* provenance: if the fact was actually said
        in the conversation, cite that turn rather than manufacturing a new one
        the memory quotes from itself.
        """
        needle = text.strip()
        if not needle:
            return None
        # LIKE's wildcards have to be escaped or a fact containing % or _ would
        # match turns that never said it.
        escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        cur = self._db.execute(
            "SELECT turn_id, role, text, source_id, created_at FROM turns "
            "WHERE text LIKE ? ESCAPE '\\' ORDER BY created_at DESC LIMIT 1",
            (f"%{escaped}%",),
        )
        row = cur.fetchone()
        return self._row_to_turn(row) if row is not None else None

    def count(self) -> int:
        """Return total number of stored turns."""
        cur = self._db.execute("SELECT COUNT(*) FROM turns")
        return cur.fetchone()[0]

    @staticmethod
    def _row_to_turn(row: tuple) -> Turn:
        return Turn(
            turn_id=row[0],
            role=row[1],
            text=row[2],
            source_id=row[3],
            created_at=row[4],
        )
