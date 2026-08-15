from __future__ import annotations

from memory_condense.db import Database
from memory_condense.schemas import Turn


class TranscriptStore:
    """Append-only store for conversation transcript turns."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def current_turn(self) -> int:
        """The conversation's position — **the decay coordinate**.

        Energy decays in turns, not seconds (see :mod:`memory_condense.decay`),
        so this is the clock the whole memory layer reads. Delegates to
        :meth:`Database.current_turn`, which ``MemoryStore`` shares.
        """
        return self._db.current_turn()

    def append(self, role: str, text: str) -> Turn:
        """Create and persist a new turn. Returns the Turn with generated ID.

        Advancing ``ordinal`` here is what makes decay happen: every item the
        conversation did *not* reach for this turn falls one turn further
        behind. Nothing else has to run — no sweep, no timer.
        """
        turn = Turn(role=role, text=text)
        ordinal = self.current_turn() + 1
        self._db.execute(
            "INSERT INTO turns (turn_id, role, text, created_at, ordinal)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                turn.turn_id,
                turn.role,
                turn.text,
                turn.created_at.isoformat(),
                ordinal,
            ),
        )
        self._db.commit()
        return turn

    def get_turn(self, turn_id: str) -> Turn | None:
        """Retrieve a single turn by ID."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, created_at FROM turns WHERE turn_id = ?",
            (turn_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_turn(row)

    def get_recent(self, n: int = 20) -> list[Turn]:
        """Return the N most recent turns, ordered oldest-first."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, created_at FROM turns "
            "ORDER BY created_at DESC LIMIT ?",
            (n,),
        )
        rows = cur.fetchall()
        return [self._row_to_turn(r) for r in reversed(rows)]

    def get_all(self) -> list[Turn]:
        """Return all turns, ordered by created_at."""
        cur = self._db.execute(
            "SELECT turn_id, role, text, created_at FROM turns ORDER BY created_at"
        )
        return [self._row_to_turn(r) for r in cur.fetchall()]

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
            "SELECT turn_id, role, text, created_at FROM turns "
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
            created_at=row[3],
        )
