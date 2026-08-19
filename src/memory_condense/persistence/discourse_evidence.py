"""Exact-source evidence and bounded metadata support for discourse storage."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence, TypeVar

from memory_condense.domain.discourse import (
    EvidenceSpan,
    canonical_json,
    identity_sha256,
    quote_sha256,
)


class DiscourseIdentityError(ValueError):
    """A stable discourse ID was reused for different immutable contents."""


class SourceEvidenceError(ValueError):
    """Stored evidence does not exactly match its authoritative chunk/turn."""


class _Executor(Protocol):
    def execute(self, sql: str, params: tuple = ()) -> Any: ...

    def executemany(self, sql: str, params_seq: Any) -> Any: ...


_T = TypeVar("_T")
_EVIDENCE_COLUMNS = (
    "chunk_id, start_char, end_char, quote_sha256, ordinal, source_id, "
    "turn_start_char, turn_id, role, created_at"
)
_METADATA_KEYS = {
    "artifact": frozenset(
        {"tokenizer_id", "window", "scorer_id", "boundary_policy_id"}
    ),
    "unit": frozenset({"episode_id", "linker", "route"}),
    "relation": frozenset({"linker", "resolved_relation_id"}),
}
_ROUTING_SCALAR_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,160}$")


@dataclass(frozen=True, slots=True)
class _SourceRow:
    chunk_id: str
    chunk_text: str
    chunk_start: int
    chunk_end: int
    token_count: int
    turn_id: str
    turn_text: str
    ordinal: int
    source_id: str
    source_id_raw: str | None
    role: str
    created_at: str

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(
            {
                "chunk_id": self.chunk_id,
                "chunk_text": self.chunk_text,
                "chunk_start": self.chunk_start,
                "chunk_end": self.chunk_end,
                "token_count": self.token_count,
                "turn_id": self.turn_id,
                "turn_text": self.turn_text,
                "ordinal": self.ordinal,
                "source_id": self.source_id_raw,
                "role": self.role,
                "created_at": self.created_at,
            }
        )


def _unique(values: Iterable[_T]) -> tuple[_T, ...]:
    return tuple(dict.fromkeys(values))


def _strict_json_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DiscourseIdentityError(f"stored {label} is not strict JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise DiscourseIdentityError(f"stored {label} is not canonical JSON")
    return value


def _safe_metadata(
    value: Mapping[str, Any],
    *,
    label: str,
    owner: str,
) -> str:
    """Persist only bounded scalar routing metadata from a closed schema."""

    body = dict(value)
    unknown = set(body) - _METADATA_KEYS[owner]
    if unknown:
        raise ValueError(
            f"{label} cannot persist request-derived or unsupported keys: "
            f"{sorted(unknown)}"
        )
    for key, child in body.items():
        if child is None or isinstance(child, bool):
            continue
        if isinstance(child, int):
            if abs(child) > 1_000_000:
                raise ValueError(f"{label} integer {key!r} exceeds its bound")
            continue
        if isinstance(child, float):
            if not math.isfinite(child) or abs(child) > 1_000_000.0:
                raise ValueError(f"{label} float {key!r} exceeds its bound")
            continue
        if isinstance(child, str) and _ROUTING_SCALAR_RE.fullmatch(child):
            continue
        raise ValueError(
            f"{label} field {key!r} must be one bounded scalar routing value"
        )
    return canonical_json(body)


class EvidenceStoreMixin:
    """Mixin for exact source-span hydration and persistence."""

    _db: _Executor

    def _source_row(self, chunk_id: str) -> _SourceRow:
        row = self._db.execute(
            "SELECT c.text, c.start_char, c.end_char, c.token_count, c.turn_id, "
            "t.text, t.ordinal, t.source_id, t.role, t.created_at "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            "WHERE c.chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise SourceEvidenceError(f"unknown evidence chunk: {chunk_id}")
        start = int(row[1])
        end = int(row[2])
        if start < 0 or end <= start or end > len(row[5]):
            raise SourceEvidenceError(
                f"chunk {chunk_id!r} has invalid authoritative turn coordinates"
            )
        if row[5][start:end] != row[0]:
            raise SourceEvidenceError(
                f"chunk {chunk_id!r} no longer exactly matches its turn span"
            )
        return _SourceRow(
            chunk_id=str(chunk_id),
            chunk_text=str(row[0]),
            chunk_start=start,
            chunk_end=end,
            token_count=int(row[3]),
            turn_id=str(row[4]),
            turn_text=str(row[5]),
            ordinal=int(row[6]),
            source_id=str(row[7] or row[4]),
            source_id_raw=row[7],
            role=str(row[8]),
            created_at=str(row[9]),
        )

    def _validate_span(
        self,
        span: EvidenceSpan,
        *,
        required_source: str | None = None,
        require_complete_provenance: bool = False,
    ) -> str:
        source = self._source_row(span.chunk_id)
        if span.end_char > len(source.chunk_text):
            raise SourceEvidenceError(
                f"evidence span exceeds chunk {span.chunk_id!r}"
            )
        text = source.chunk_text[span.start_char : span.end_char]
        if source.turn_text[
            source.chunk_start + span.start_char : source.chunk_start + span.end_char
        ] != text:
            raise SourceEvidenceError(
                f"evidence span in {span.chunk_id!r} does not match its turn"
            )
        if quote_sha256(text) != span.quote_sha256:
            raise SourceEvidenceError(
                f"evidence quote hash does not match chunk {span.chunk_id!r}"
            )
        if span.ordinal != source.ordinal:
            raise SourceEvidenceError(
                f"evidence ordinal for {span.chunk_id!r} is not authoritative"
            )
        if span.turn_start_char != source.chunk_start:
            raise SourceEvidenceError(
                f"evidence turn start for {span.chunk_id!r} is not authoritative"
            )
        if span.source_id is not None and span.source_id != source.source_id:
            raise SourceEvidenceError(
                f"evidence source for {span.chunk_id!r} is not authoritative"
            )
        for name in ("turn_id", "role", "created_at"):
            value = getattr(span, name)
            if value is not None and value != getattr(source, name):
                raise SourceEvidenceError(
                    f"evidence {name} for {span.chunk_id!r} is not authoritative"
                )
        if require_complete_provenance and any(
            getattr(span, name) is None
            for name in ("source_id", "turn_id", "role", "created_at")
        ):
            raise SourceEvidenceError(
                f"evidence provenance for {span.chunk_id!r} is incomplete"
            )
        if required_source is not None and source.source_id != required_source:
            raise SourceEvidenceError(
                f"evidence chunk {span.chunk_id!r} is outside source "
                f"{required_source!r}"
            )
        return text

    def hydrate_span(self, span: EvidenceSpan) -> str:
        """Return verified raw evidence transiently; never persist the text."""

        return self._validate_span(span)

    def evidence_for_chunks(
        self, chunk_ids: Sequence[str]
    ) -> tuple[EvidenceSpan, ...]:
        """Create verified full-chunk evidence in first-input order."""

        spans: list[EvidenceSpan] = []
        for chunk_id in _unique(str(item) for item in chunk_ids):
            source = self._source_row(chunk_id)
            if not source.chunk_text:
                raise SourceEvidenceError(
                    f"empty chunk {chunk_id!r} cannot form an evidence span"
                )
            span = EvidenceSpan(
                chunk_id=chunk_id,
                start_char=0,
                end_char=len(source.chunk_text),
                quote_sha256=quote_sha256(source.chunk_text),
                ordinal=source.ordinal,
                source_id=source.source_id,
                turn_start_char=source.chunk_start,
                turn_id=source.turn_id,
                role=source.role,
                created_at=source.created_at,
            )
            self._validate_span(span)
            spans.append(span)
        return tuple(spans)

    def _read_evidence(
        self,
        table: str,
        owner_column: str,
        owner_id: str,
        *,
        required_source: str | None = None,
    ) -> tuple[EvidenceSpan, ...]:
        rows = self._db.execute(
            f"SELECT evidence_order, {_EVIDENCE_COLUMNS} FROM {table} "
            f"WHERE {owner_column} = ? ORDER BY evidence_order",
            (owner_id,),
        ).fetchall()
        if [int(row[0]) for row in rows] != list(range(len(rows))):
            raise DiscourseIdentityError(
                f"stored evidence order for {owner_id!r} is not contiguous"
            )
        spans = tuple(
            EvidenceSpan(
                chunk_id=row[1],
                start_char=int(row[2]),
                end_char=int(row[3]),
                quote_sha256=row[4],
                ordinal=int(row[5]),
                source_id=row[6],
                turn_start_char=int(row[7]),
                turn_id=row[8],
                role=row[9],
                created_at=row[10],
            )
            for row in rows
        )
        for span in spans:
            self._validate_span(span, required_source=required_source)
        return spans

    def _insert_evidence(
        self,
        table: str,
        owner_column: str,
        owner_id: str,
        evidence: Sequence[EvidenceSpan],
        *,
        required_source: str | None = None,
    ) -> None:
        for span in evidence:
            self._validate_span(
                span,
                required_source=required_source,
                require_complete_provenance=True,
            )
        self._db.executemany(
            f"INSERT INTO {table} "
            f"({owner_column}, evidence_order, {_EVIDENCE_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    owner_id,
                    position,
                    span.chunk_id,
                    span.start_char,
                    span.end_char,
                    span.quote_sha256,
                    span.ordinal,
                    span.source_id,
                    span.turn_start_char,
                    span.turn_id,
                    span.role,
                    span.created_at,
                )
                for position, span in enumerate(evidence)
            ],
        )

    def authoritative_turn_coverage_sha256(self) -> str:
        """Prove exact chunks cover every non-whitespace authoritative byte."""

        payload: list[dict[str, Any]] = []
        turns = self._db.execute(
            "SELECT turn_id, role, text, source_id, created_at, ordinal "
            "FROM turns ORDER BY ordinal, turn_id"
        ).fetchall()
        for turn_id, role, text, source_id, created_at, ordinal in turns:
            text = str(text)
            rows = self._db.execute(
                "SELECT chunk_id, start_char, end_char FROM chunks "
                "WHERE turn_id = ? ORDER BY start_char, end_char, chunk_id",
                (turn_id,),
            ).fetchall()
            cursor = 0
            chunk_identities: list[str] = []
            for chunk_id, start_char, end_char in rows:
                start = int(start_char)
                end = int(end_char)
                source = self._source_row(str(chunk_id))
                if source.turn_id != turn_id:
                    raise SourceEvidenceError(
                        f"chunk {chunk_id!r} has mismatched turn ownership"
                    )
                if start < cursor:
                    raise SourceEvidenceError(
                        f"turn {turn_id!r} has overlapping authoritative chunks"
                    )
                if text[cursor:start].strip():
                    raise SourceEvidenceError(
                        f"turn {turn_id!r} has uncovered non-whitespace evidence"
                    )
                cursor = end
                chunk_identities.append(source.identity_sha256)
            if text[cursor:].strip():
                raise SourceEvidenceError(
                    f"turn {turn_id!r} has uncovered non-whitespace evidence"
                )
            payload.append(
                {
                    "turn_id": str(turn_id),
                    "role": str(role),
                    "text_sha256": quote_sha256(text),
                    "source_id": source_id,
                    "created_at": str(created_at),
                    "ordinal": int(ordinal),
                    "chunk_identity_sha256": chunk_identities,
                }
            )
        return identity_sha256(payload)

    def verified_artifact_coverage_rows(
        self,
        artifact_id: str,
        chunk_ids: Sequence[str],
        coverage_kind: str,
    ) -> tuple[dict[str, str], ...]:
        """Bind every current chunk identity to its immutable stage status."""

        covered = self.coverage_for_chunks(  # type: ignore[attr-defined]
            artifact_id,
            chunk_ids,
            coverage_kind=coverage_kind,
        )
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in covered]
        if missing:
            raise ValueError(
                "cannot finalize incomplete artifact coverage; missing "
                + ", ".join(missing[:8])
            )
        return tuple(
            {
                "chunk_id": chunk_id,
                "chunk_identity_sha256": self._source_row(
                    chunk_id
                ).identity_sha256,
                "status": covered[chunk_id],
            }
            for chunk_id in chunk_ids
        )


__all__ = [
    "DiscourseIdentityError",
    "EvidenceStoreMixin",
    "SourceEvidenceError",
]
