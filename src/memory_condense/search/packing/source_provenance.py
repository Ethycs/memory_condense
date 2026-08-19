"""Source timestamp recognition and evidence/provenance binding."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable

from memory_condense.domain.schemas import RetrievalResult
from memory_condense.persistence.transcript_store import parse_source_metadata


_PROVENANCE_TIMESTAMP_RE = re.compile(
    r"\b(?P<year>(?:19|20)\d{2})[/-](?P<month>\d{1,2})"
    r"[/-](?P<day>\d{1,2})(?:\D+(?P<hour>\d{1,2})"
    r":(?P<minute>\d{2}))?"
)


def provenance_timestamp_key(value: str | None) -> float | None:
    """Parse a full source date conservatively for closure-order validation."""

    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned).timestamp()
    except ValueError:
        match = _PROVENANCE_TIMESTAMP_RE.search(cleaned)
        if match is None:
            return None
        try:
            return datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour") or 0),
                int(match.group("minute") or 0),
            ).timestamp()
        except (OverflowError, ValueError):
            return None


def is_source_metadata_text(text: str) -> bool:
    """Whether ``text`` is a synthetic source timestamp, not evidence."""

    return parse_source_metadata(text) is not None


def bind_source_metadata(
    selected: list[RetrievalResult],
    *,
    candidate_pool: list[RetrievalResult] | None = None,
    source_metadata: dict[str, str] | None = None,
    result_source_id: Callable[[RetrievalResult], str],
    metadata_predicate: Callable[[str], bool] = is_source_metadata_text,
) -> tuple[dict[str, str], list[RetrievalResult]]:
    """Bind timestamps and replace routed metadata rows with source content."""

    pool = candidate_pool or selected
    timestamps: dict[str, str] = {}
    persisted_metadata_sources: set[str] = set()
    for source_id, text in (source_metadata or {}).items():
        parsed = parse_source_metadata(text)
        if parsed is not None:
            timestamps[source_id] = parsed[1]
            persisted_metadata_sources.add(source_id)
    companions: dict[str, RetrievalResult] = {}
    for result in pool:
        source_id = result_source_id(result)
        parsed = parse_source_metadata(result.chunk.text)
        if parsed is None:
            companions.setdefault(source_id, result)
            continue
        timestamps.setdefault(source_id, parsed[1])

    evidence: list[RetrievalResult] = []
    seen_chunks: set[str] = set()
    resolved_metadata_sources: set[str] = set()
    for result in selected:
        source_id = result_source_id(result)
        is_metadata = metadata_predicate(result.chunk.text)
        candidate = result
        if is_metadata and source_id not in resolved_metadata_sources:
            companion = companions.get(source_id)
            resolved_metadata_sources.add(source_id)
            if companion is not None:
                candidate = companion
            elif source_id in persisted_metadata_sources:
                # The durable store already supplied this timestamp. With no
                # content companion, the anonymous metadata row adds no fact
                # that the responder can associate with the live question.
                continue
        elif is_metadata:
            continue
        if candidate.chunk.chunk_id in seen_chunks:
            continue
        seen_chunks.add(candidate.chunk.chunk_id)
        evidence.append(candidate)
    return timestamps, evidence
