"""Fail-closed resolution of rendered excerpts to exact cache coordinates."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .cache_artifacts import ChunkRecord
from .canonical import AuditError, bytes_sha256, canonical_sha256
from .prompt import FrozenPromptRuntime, TextVariant


_ENTRY_RE = re.compile(
    r"^\[(?P<ordinal>[1-9][0-9]*)"
    r"(?: @ (?P<timestamp>.*?))?"
    r"(?: \| (?P<role>[^\]]+))?\] (?P<body>[\s\S]+)$"
)
_NONSPACE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class _Resolved:
    chunk: ChunkRecord
    transforms: tuple[str, ...]
    body_tokens: int
    source_spans: tuple[dict[str, Any], ...]
    synthetic_segments: tuple[dict[str, Any], ...]

    def identity(self) -> str:
        return canonical_sha256(
            {
                "chunk_id": self.chunk.chunk_id,
                "source_spans": self.source_spans,
                "synthetic_segments": self.synthetic_segments,
            }
        )


class ExcerptResolver:
    def __init__(
        self,
        connection: sqlite3.Connection,
        chunks: list[ChunkRecord],
        runtime: FrozenPromptRuntime,
        *,
        source_metadata: bool,
        query_aware: bool,
        max_sentences: int,
    ) -> None:
        self.connection = connection
        self.runtime = runtime
        self.source_metadata = source_metadata
        self.query_aware = query_aware
        self.max_sentences = max_sentences
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self.by_id) != len(chunks):
            raise AuditError("cache contains duplicate chunk IDs")
        self._turn_maps: dict[str, tuple[int | None, ...]] = {}
        for chunk in chunks:
            if runtime.count_tokens(chunk.text) != chunk.token_count:
                raise AuditError(f"chunk token count mismatch: {chunk.chunk_id}")
            self._turn_maps[chunk.chunk_id] = _chunk_to_turn_map(chunk)
        self._validate_lexical_index()

    def _validate_lexical_index(self) -> None:
        """Prove candidate pruning cannot hide a duplicate provenance match."""

        observed: set[str] = set()
        cursor = self.connection.execute(
            "SELECT chunk_id, term, tf FROM chunk_terms ORDER BY chunk_id, term"
        )
        active_id: str | None = None
        active_terms: dict[str, int] = {}

        def finish() -> None:
            if active_id is None:
                return
            chunk = self.by_id.get(active_id)
            if chunk is None:
                raise AuditError("chunk_terms refers to an absent chunk")
            expected = dict(Counter(self.runtime.lexical_tokens(chunk.text)))
            if active_terms != expected:
                raise AuditError(f"chunk_terms differs from chunk text: {active_id}")
            observed.add(active_id)

        for raw_chunk_id, raw_term, raw_tf in cursor:
            chunk_id = str(raw_chunk_id)
            if active_id is not None and chunk_id != active_id:
                finish()
                active_terms = {}
            active_id = chunk_id
            term = str(raw_term)
            if term in active_terms:
                raise AuditError(f"chunk_terms contains a duplicate term: {chunk_id}")
            try:
                tf = int(raw_tf)
            except (TypeError, ValueError) as exc:
                raise AuditError(f"chunk_terms has a non-integer tf: {chunk_id}") from exc
            if isinstance(raw_tf, float) and raw_tf != tf:
                raise AuditError(f"chunk_terms has a non-integral tf: {chunk_id}")
            active_terms[term] = tf
        finish()
        for chunk_id, chunk in self.by_id.items():
            if chunk_id not in observed and self.runtime.lexical_tokens(chunk.text):
                raise AuditError(f"chunk_terms omits indexed text: {chunk_id}")

    def resolve_question(
        self,
        dated_question: str,
        excerpts: list[str],
    ) -> list[dict[str, Any]]:
        return [
            self.resolve_excerpt(dated_question, excerpt, ordinal)
            for ordinal, excerpt in enumerate(excerpts, start=1)
        ]

    def resolve_excerpt(
        self,
        dated_question: str,
        excerpt: str,
        ordinal: int,
    ) -> dict[str, Any]:
        match = _ENTRY_RE.fullmatch(excerpt)
        if match is None:
            raise AuditError(f"retrieved excerpt {ordinal} has no frozen provenance label")
        if int(match.group("ordinal")) != ordinal:
            raise AuditError("retrieved excerpt ordinal does not match packet order")
        body = match.group("body")
        timestamp = match.group("timestamp")
        role = match.group("role")
        candidate_ids = self._candidate_ids(body)
        resolved: dict[str, _Resolved] = {}
        for chunk_id in candidate_ids:
            chunk = self.by_id.get(chunk_id)
            if chunk is None:
                raise AuditError("chunk_terms refers to an absent chunk")
            expected_timestamp = chunk.source_timestamp if self.source_metadata else None
            if timestamp != expected_timestamp:
                continue
            expected_role = chunk.role.strip().lower() if chunk.role.strip() else None
            if role != expected_role:
                continue
            prefix = f"[{ordinal}"
            if expected_timestamp:
                prefix += f" @ {expected_timestamp}"
            if expected_role:
                prefix += f" | {expected_role}"
            if excerpt != prefix + "] " + body:
                continue
            variants = self.runtime.text_variants(
                chunk.text,
                dated_question,
                query_aware=self.query_aware,
                max_sentences=self.max_sentences,
            )
            for variant, body_characters, body_tokens in self.runtime.matching_prefixes(
                body,
                variants,
                max_tokens=self.runtime.source.max_expansion_tokens,
            ):
                candidate = self._coordinates(
                    chunk,
                    variant,
                    body_characters,
                    body_tokens,
                    body,
                )
                identity = candidate.identity()
                previous = resolved.get(identity)
                if previous is None:
                    resolved[identity] = candidate
                else:
                    resolved[identity] = _Resolved(
                        chunk=previous.chunk,
                        transforms=tuple(sorted(set(previous.transforms + candidate.transforms))),
                        body_tokens=previous.body_tokens,
                        source_spans=previous.source_spans,
                        synthetic_segments=previous.synthetic_segments,
                    )
        if not resolved:
            raise AuditError(
                f"excerpt {ordinal} cannot be reproduced from an exact cache chunk"
            )
        if len(resolved) != 1:
            chunk_ids = sorted({row.chunk.chunk_id for row in resolved.values()})
            raise AuditError(
                f"excerpt {ordinal} has ambiguous exact provenance: {chunk_ids}"
            )
        row = next(iter(resolved.values()))
        return {
            "packet_index": ordinal,
            "rendered_excerpt_sha256": bytes_sha256(excerpt.encode("utf-8")),
            "body_sha256": bytes_sha256(body.encode("utf-8")),
            "body_characters": len(body),
            "body_tokens": row.body_tokens,
            "source_id": row.chunk.source_id,
            "turn_id": row.chunk.turn_id,
            "chunk_id": row.chunk.chunk_id,
            "role": row.chunk.role,
            "source_timestamp": row.chunk.source_timestamp,
            "chunk_source_start_char": row.chunk.start_char,
            "chunk_source_end_char": row.chunk.end_char,
            "chunk_sha256": bytes_sha256(row.chunk.text.encode("utf-8")),
            "turn_sha256": bytes_sha256(row.chunk.turn_text.encode("utf-8")),
            "packing_transform_candidates": list(row.transforms),
            "source_spans": list(row.source_spans),
            "synthetic_segments": list(row.synthetic_segments),
        }

    def _candidate_ids(self, body: str) -> list[str]:
        terms = sorted(set(self.runtime.lexical_tokens(body)))
        if not terms:
            return list(self.by_id)
        if len(terms) > 900:
            terms = terms[:900]
        placeholders = ",".join("?" for _ in terms)
        counts = self.connection.execute(
            "SELECT term, COUNT(*) FROM chunk_terms "
            f"WHERE term IN ({placeholders}) GROUP BY term",
            tuple(terms),
        ).fetchall()
        if not counts:
            return []
        rarest = min(((int(count), str(term)) for term, count in counts))[1]
        return [
            str(row[0])
            for row in self.connection.execute(
                "SELECT chunk_id FROM chunk_terms WHERE term = ? ORDER BY chunk_id",
                (rarest,),
            ).fetchall()
        ]

    def _coordinates(
        self,
        chunk: ChunkRecord,
        variant: TextVariant,
        body_characters: int,
        body_tokens: int,
        body: str,
    ) -> _Resolved:
        prepared_to_chunk: list[int | None] = [None] * body_characters
        for prepared_start, prepared_end, chunk_start, chunk_end in variant.mappings:
            upper = min(prepared_end, body_characters)
            for prepared_index in range(prepared_start, upper):
                chunk_index = chunk_start + (prepared_index - prepared_start)
                if chunk_index >= chunk_end:
                    raise AuditError("prepared-to-chunk provenance mapping overran a span")
                prepared_to_chunk[prepared_index] = chunk_index
        chunk_to_turn = self._turn_maps[chunk.chunk_id]
        mapped: list[tuple[int | None, int | None]] = []
        for rendered_index, chunk_index in enumerate(prepared_to_chunk):
            if chunk_index is None:
                mapped.append((None, None))
                continue
            turn_index = chunk_to_turn[chunk_index]
            if turn_index is None and not body[rendered_index].isspace():
                raise AuditError("non-whitespace excerpt content lacks a turn coordinate")
            if turn_index is not None:
                rendered_character = body[rendered_index]
                if (
                    chunk.text[chunk_index] != rendered_character
                    or chunk.turn_text[turn_index] != rendered_character
                ):
                    raise AuditError(
                        "rendered excerpt character differs from its claimed source coordinate"
                    )
            mapped.append((chunk_index, turn_index))

        source_spans: list[dict[str, Any]] = []
        synthetic: list[dict[str, Any]] = []
        index = 0
        while index < len(mapped):
            chunk_index, turn_index = mapped[index]
            if chunk_index is not None and turn_index is not None:
                end = index + 1
                while end < len(mapped):
                    next_chunk, next_turn = mapped[end]
                    if next_chunk != chunk_index + (end - index) or next_turn != turn_index + (end - index):
                        break
                    end += 1
                text = body[index:end]
                source_spans.append(
                    {
                        "rendered_start_char": index,
                        "rendered_end_char": end,
                        "chunk_start_char": chunk_index,
                        "chunk_end_char": chunk_index + (end - index),
                        "turn_start_char": turn_index,
                        "turn_end_char": turn_index + (end - index),
                        "text_sha256": bytes_sha256(text.encode("utf-8")),
                    }
                )
                index = end
                continue
            end = index + 1
            while end < len(mapped) and (
                mapped[end][0] is None or mapped[end][1] is None
            ):
                end += 1
            text = body[index:end]
            if any(not character.isspace() for character in text):
                raise AuditError("a non-whitespace excerpt segment is synthetic")
            synthetic.append(
                {
                    "rendered_start_char": index,
                    "rendered_end_char": end,
                    "text_sha256": bytes_sha256(text.encode("utf-8")),
                    "reason": "sentence_join_or_whitespace_normalization",
                }
            )
            index = end
        return _Resolved(
            chunk=chunk,
            transforms=(variant.kind,),
            body_tokens=body_tokens,
            source_spans=tuple(source_spans),
            synthetic_segments=tuple(synthetic),
        )


def _chunk_to_turn_map(chunk: ChunkRecord) -> tuple[int | None, ...]:
    """Map normalized chunk characters to its exact turn slice."""

    region = chunk.turn_text[chunk.start_char : chunk.end_char]
    chunk_tokens = list(_NONSPACE.finditer(chunk.text))
    turn_tokens = list(_NONSPACE.finditer(region))
    if [match.group(0) for match in chunk_tokens] != [
        match.group(0) for match in turn_tokens
    ]:
        raise AuditError(
            f"chunk text is not a whitespace-normalized source span: {chunk.chunk_id}"
        )
    mapping: list[int | None] = [None] * len(chunk.text)
    for chunk_match, turn_match in zip(chunk_tokens, turn_tokens, strict=True):
        token = chunk_match.group(0)
        for offset in range(len(token)):
            mapping[chunk_match.start() + offset] = (
                chunk.start_char + turn_match.start() + offset
            )
    # Preserve whitespace coordinates only where the normalized and original
    # gaps are byte-for-character identical. Other whitespace is explicitly
    # recorded as a synthetic normalization segment in the receipt.
    boundaries = [
        (0, 0),
        *[
            (chunk_match.end(), turn_match.end())
            for chunk_match, turn_match in zip(chunk_tokens, turn_tokens, strict=True)
        ],
    ]
    next_boundaries = [
        *[
            (chunk_match.start(), turn_match.start())
            for chunk_match, turn_match in zip(chunk_tokens, turn_tokens, strict=True)
        ],
        (len(chunk.text), len(region)),
    ]
    for (chunk_left, turn_left), (chunk_right, turn_right) in zip(
        boundaries,
        next_boundaries,
        strict=True,
    ):
        chunk_gap = chunk.text[chunk_left:chunk_right]
        turn_gap = region[turn_left:turn_right]
        if chunk_gap == turn_gap:
            for offset in range(len(chunk_gap)):
                mapping[chunk_left + offset] = chunk.start_char + turn_left + offset
    return tuple(mapping)
