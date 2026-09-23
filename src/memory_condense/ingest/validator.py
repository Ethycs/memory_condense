"""Provenance enforcement for proposed memory operations.

The design's single load-bearing rule:

    "Every op must include provenance (turn/chunk refs + quote).
     This one rule is what keeps a pure-LLM approach from drifting."

``Validator`` is the gate between an extractor (rules or LLM) and the
``MemoryStore``. It never raises and never mutates state: it splits a
``MemoryOps`` batch into the ops that are backed by real transcript text and a
list of ``ValidationError`` records explaining every rejection.

Quote matching is **whitespace-insensitive**: both the turn text and the quote
are passed through :func:`_normalize`, which collapses every run of whitespace
(spaces, tabs, newlines) to a single space and strips the ends. A quote is
accepted when the normalized quote is a substring of the normalized turn text.
Nothing else is relaxed — no case folding, no punctuation stripping, no fuzzy
matching. An LLM that paraphrases gets rejected, which is the point.
"""

from __future__ import annotations

import re
from typing import Sequence

from memory_condense.persistence.db import Database
from memory_condense.domain.schemas import (
    Chunk,
    CreateOp,
    MemoryOps,
    MemoryType,
    Provenance,
    SupersedeOp,
    Turn,
    UpdateOp,
    ValidationError,
    ValidationReport,
)
from memory_condense.persistence.transcript_store import TranscriptStore

#: Rejection reason slugs (stable strings — callers may switch on these).
REASON_MISSING_PROVENANCE = "missing_provenance"
REASON_UNKNOWN_TURN = "unknown_turn"
REASON_UNKNOWN_CHUNK = "unknown_chunk"
REASON_CHUNK_TURN_MISMATCH = "chunk_turn_mismatch"
REASON_CHUNK_SPAN_MISMATCH = "chunk_span_mismatch"
REASON_CHUNK_QUOTE_NOT_FOUND = "chunk_quote_not_found"
REASON_QUOTE_NOT_FOUND = "quote_not_found"
REASON_UNKNOWN_MEM_ID = "unknown_mem_id"
REASON_INVALID_MEM_STATUS = "invalid_mem_status"
REASON_EMPTY_CONTENT = "empty_content"
REASON_OUTSIDE_EVIDENCE_VIEW = "outside_evidence_view"
REASON_REQUIRES_TARGET_BINDING = "requires_target_binding"

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse whitespace runs to single spaces and strip the ends.

    Used on both sides of a quote comparison so that a quote copied out of a
    wrapped or re-indented turn still matches the stored transcript text.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


class Validator:
    """Validates ``MemoryOps`` against the transcript and the memory table."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._transcripts = TranscriptStore(db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, ops: MemoryOps) -> ValidationReport:
        """Split ``ops`` into accepted operations and explained rejections.

        Never raises: malformed or unbacked operations are reported, not
        thrown. The accepted ``MemoryOps`` is safe to hand to
        ``MemoryStore.apply``.
        """
        accepted = MemoryOps()
        rejected: list[ValidationError] = []
        # Per-call cache so a batch of ops over one turn hits SQLite once.
        turn_cache: dict[str, str | None] = {}
        chunk_cache: dict[str, tuple[str, str, int, int] | None] = {}

        def admit(op, error: ValidationError | None, bucket: list) -> None:
            if error is None:
                bucket.append(op)
            else:
                rejected.append(error)

        for op in ops.create:
            admit(
                op,
                self._check_create(op, "create", turn_cache, chunk_cache),
                accepted.create,
            )

        for update in ops.update:
            error = self._check_mem_id(
                update.mem_id, "update", require_active=True
            )
            if error is None:
                error = self._check_provenance(
                    update.provenance,
                    "update",
                    turn_cache,
                    chunk_cache,
                    required=False,
                )
            admit(update, error, accepted.update)

        for sup in ops.supersede:
            error = self._check_mem_id(
                sup.mem_id, "supersede", require_active=True
            )
            if error is None:
                error = self._check_create(
                    sup.replacement,
                    "supersede",
                    turn_cache,
                    chunk_cache,
                )
            admit(sup, error, accepted.supersede)

        for dele in ops.delete:
            admit(
                dele,
                self._check_mem_id(dele.mem_id, "delete", require_active=True),
                accepted.delete,
            )

        for pin in ops.pin:
            admit(
                pin,
                self._check_mem_id(pin.mem_id, "pin", require_active=True),
                accepted.pin,
            )

        return ValidationReport(accepted=accepted, rejected=rejected)

    def quote_matches(self, turn_id: str, quote: str) -> bool:
        """True when ``quote`` appears verbatim (modulo whitespace) in the turn."""
        text = self._turn_text(turn_id, {})
        if text is None:
            return False
        needle = _normalize(quote)
        return bool(needle) and needle in _normalize(text)

    def validate_for_enrichment(
        self,
        ops: MemoryOps,
        turn: Turn,
        chunks: Sequence[Chunk],
        *,
        allow_unbound_corrections: bool = False,
    ) -> ValidationReport:
        """Validate globally, then confine provenance to the supplied live view.

        The transcript is append-only even after a chunk is retired, so the
        ordinary validator intentionally still recognizes its historical
        text. Deferred T2 extraction has a stricter boundary: an operation may
        cite only evidence that was actually present in its current live chunk
        window. This rejects both explicit retired chunk IDs and a chunk-less
        quote that exists only in a retired span.
        """
        report = self.validate(ops)
        accepted = MemoryOps()
        rejected = list(report.rejected)
        live_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        def visible_provenance(provenance: list[Provenance]):
            visible_entries: list[Provenance] = []
            for entry in provenance:
                if entry.turn_id != turn.turn_id:
                    continue
                needle = _normalize(entry.quote)
                if entry.chunk_id is not None:
                    candidate = live_by_id.get(entry.chunk_id)
                    visible = (
                        candidate is not None
                        and needle
                        and needle in _normalize(candidate.text)
                    )
                else:
                    visible = bool(needle) and any(
                        needle in _normalize(chunk.text) for chunk in chunks
                    )
                if visible:
                    visible_entries.append(entry)
            return visible_entries

        def admit(op, provenance: list[Provenance], kind: str, bucket: list):
            visible = visible_provenance(provenance)
            if visible:
                bucket.append(
                    op
                    if len(visible) == len(provenance)
                    else op.model_copy(update={"provenance": visible})
                )
            else:
                rejected.append(
                    ValidationError(
                        op_kind=kind,
                        reason=REASON_OUTSIDE_EVIDENCE_VIEW,
                        detail="no provenance remains in the live chunk view",
                    )
                )

        for op in report.accepted.create:
            if op.type is MemoryType.CORRECTION:
                if allow_unbound_corrections:
                    admit(op, op.provenance, "create", accepted.create)
                else:
                    rejected.append(
                        ValidationError(
                            op_kind="create",
                            reason=REASON_REQUIRES_TARGET_BINDING,
                            detail=(
                                "deferred corrections require an explicit "
                                "versioned supersede target"
                            ),
                        )
                    )
            else:
                admit(op, op.provenance, "create", accepted.create)
        # Deferred T2 sees one local evidence window, not a versioned snapshot
        # of the whole memory store. It may publish grounded facts, but it may
        # not mutate existing rows whose status/content could change between
        # staging and replay. Those operations remain available through the
        # explicit/public validation path, where the caller controls context.
        for _op in report.accepted.update:
            rejected.append(
                ValidationError(
                    op_kind="update",
                    reason=REASON_OUTSIDE_EVIDENCE_VIEW,
                    detail="deferred updates require a versioned target binding",
                )
            )
        for _op in report.accepted.supersede:
            rejected.append(
                ValidationError(
                    op_kind="supersede",
                    reason=REASON_OUTSIDE_EVIDENCE_VIEW,
                    detail="deferred supersedes require a versioned target binding",
                )
            )
        for _op in report.accepted.delete:
            rejected.append(
                ValidationError(
                    op_kind="delete",
                    reason=REASON_OUTSIDE_EVIDENCE_VIEW,
                    detail="deferred deletes have no evidence binding",
                )
            )
        for _op in report.accepted.pin:
            rejected.append(
                ValidationError(
                    op_kind="pin",
                    reason=REASON_OUTSIDE_EVIDENCE_VIEW,
                    detail="deferred pins have no evidence binding",
                )
            )
        return ValidationReport(accepted=accepted, rejected=rejected)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_create(
        self,
        op: CreateOp,
        op_kind: str,
        turn_cache: dict[str, str | None],
        chunk_cache: dict[str, tuple[str, str, int, int] | None],
    ) -> ValidationError | None:
        """A create is only as trustworthy as its provenance."""
        if not op.content or not op.content.strip():
            return ValidationError(
                op_kind=op_kind,
                reason=REASON_EMPTY_CONTENT,
                detail="content is empty or whitespace-only",
            )
        return self._check_provenance(
            op.provenance,
            op_kind,
            turn_cache,
            chunk_cache,
            required=True,
        )

    def _check_provenance(
        self,
        provenance: list[Provenance],
        op_kind: str,
        turn_cache: dict[str, str | None],
        chunk_cache: dict[str, tuple[str, str, int, int] | None],
        required: bool,
    ) -> ValidationError | None:
        """Every entry must name a real turn and quote it verbatim.

        ``required=False`` allows an empty list (used for updates, which amend
        an item that already carries provenance) but still checks any entry
        that *is* supplied.
        """
        if not provenance:
            if required:
                return ValidationError(
                    op_kind=op_kind,
                    reason=REASON_MISSING_PROVENANCE,
                    detail="op has no provenance entries; at least one is required",
                )
            return None

        for entry in provenance:
            text = self._turn_text(entry.turn_id, turn_cache)
            if text is None:
                return ValidationError(
                    op_kind=op_kind,
                    reason=REASON_UNKNOWN_TURN,
                    detail=f"turn_id {entry.turn_id!r} is not in the transcript",
                )

            needle = _normalize(entry.quote)
            if not needle:
                return ValidationError(
                    op_kind=op_kind,
                    reason=REASON_QUOTE_NOT_FOUND,
                    detail=f"empty quote for turn_id {entry.turn_id!r}",
                )
            if needle not in _normalize(text):
                return ValidationError(
                    op_kind=op_kind,
                    reason=REASON_QUOTE_NOT_FOUND,
                    detail=(
                        f"quote {_truncate(entry.quote)!r} does not appear in "
                        f"turn {entry.turn_id!r}"
                    ),
                )

            if entry.chunk_id is None:
                continue
            chunk = self._chunk(entry.chunk_id, chunk_cache)
            if chunk is None:
                return ValidationError(
                    op_kind=op_kind,
                    reason=REASON_UNKNOWN_CHUNK,
                    detail=f"chunk_id {entry.chunk_id!r} is not in chunks",
                )
            chunk_turn_id, chunk_text, start_char, end_char = chunk
            if chunk_turn_id != entry.turn_id:
                return ValidationError(
                    op_kind=op_kind,
                    reason=REASON_CHUNK_TURN_MISMATCH,
                    detail=(
                        f"chunk_id {entry.chunk_id!r} belongs to turn "
                        f"{chunk_turn_id!r}, not cited turn {entry.turn_id!r}"
                    ),
                )
            if (
                start_char < 0
                or end_char <= start_char
                or end_char > len(text)
                or text[start_char:end_char] != chunk_text
            ):
                return ValidationError(
                    op_kind=op_kind,
                    reason=REASON_CHUNK_SPAN_MISMATCH,
                    detail=(
                        f"chunk_id {entry.chunk_id!r} does not match its "
                        f"stored [{start_char}:{end_char}] span in turn "
                        f"{entry.turn_id!r}"
                    ),
                )
            if needle not in _normalize(chunk_text):
                return ValidationError(
                    op_kind=op_kind,
                    reason=REASON_CHUNK_QUOTE_NOT_FOUND,
                    detail=(
                        f"quote {_truncate(entry.quote)!r} appears in turn "
                        f"{entry.turn_id!r} but not in cited chunk "
                        f"{entry.chunk_id!r}"
                    ),
                )

        return None

    def _check_mem_id(
        self,
        mem_id: str,
        op_kind: str,
        *,
        require_active: bool = False,
    ) -> ValidationError | None:
        status = self._mem_status(mem_id)
        if status is None:
            return ValidationError(
                op_kind=op_kind,
                reason=REASON_UNKNOWN_MEM_ID,
                detail=f"mem_id {mem_id!r} is not in memory_items",
            )
        if require_active and status != "active":
            return ValidationError(
                op_kind=op_kind,
                reason=REASON_INVALID_MEM_STATUS,
                detail=(
                    f"mem_id {mem_id!r} has status {status!r}; "
                    "supersede requires an active predecessor"
                ),
            )
        return None

    # ------------------------------------------------------------------
    # Storage lookups
    # ------------------------------------------------------------------

    def _turn_text(
        self, turn_id: str, turn_cache: dict[str, str | None]
    ) -> str | None:
        if turn_id in turn_cache:
            return turn_cache[turn_id]
        turn = self._transcripts.get_turn(turn_id)
        text = turn.text if turn is not None else None
        turn_cache[turn_id] = text
        return text

    def _chunk(
        self,
        chunk_id: str,
        chunk_cache: dict[str, tuple[str, str, int, int] | None],
    ) -> tuple[str, str, int, int] | None:
        if chunk_id in chunk_cache:
            return chunk_cache[chunk_id]
        row = self._db.execute(
            "SELECT turn_id, text, start_char, end_char "
            "FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        chunk = (
            None
            if row is None
            else (str(row[0]), str(row[1]), int(row[2]), int(row[3]))
        )
        chunk_cache[chunk_id] = chunk
        return chunk

    def _mem_status(self, mem_id: str) -> str | None:
        cur = self._db.execute(
            "SELECT status FROM memory_items WHERE mem_id = ?", (mem_id,)
        )
        row = cur.fetchone()
        return None if row is None else str(row[0])


def _truncate(text: str, limit: int = 60) -> str:
    text = _normalize(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."
