"""Authoritative, text-free source projections for discourse compilation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from memory_condense.domain.discourse import (
    DiscourseSnapshot,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import parse_source_metadata
from memory_condense.search.episodes.representative_retrieval import (
    EpisodeSourceCandidate,
    EpisodeSourceCandidateScope,
)


@dataclass(frozen=True, slots=True)
class SourceChunkStream:
    """One ordered source stream split into evidence and metadata chunks."""

    source_id: str
    content_chunk_ids: tuple[str, ...]
    metadata_chunk_ids: tuple[str, ...]
    first_ordinal: int
    last_ordinal: int
    stream_sha256: str

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        content = tuple(self.content_chunk_ids)
        metadata = tuple(self.metadata_chunk_ids)
        if not content and not metadata:
            raise ValueError("a source stream must contain at least one chunk")
        if len(set((*content, *metadata))) != len(content) + len(metadata):
            raise ValueError("source stream chunk IDs must be unique")
        if self.first_ordinal < 0 or self.last_ordinal < self.first_ordinal:
            raise ValueError("source stream ordinal range is invalid")
        if len(self.stream_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.stream_sha256
        ):
            raise ValueError("stream_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "content_chunk_ids", content)
        object.__setattr__(self, "metadata_chunk_ids", metadata)

    @property
    def all_chunk_ids(self) -> tuple[str, ...]:
        return (*self.content_chunk_ids, *self.metadata_chunk_ids)


def scan_discourse_source_chunks(db: Database) -> tuple[SourceChunkStream, ...]:
    """Project every stored chunk into an exact, deterministic source stream."""

    rows = db.execute(
        "SELECT c.chunk_id, c.turn_id, c.text, c.start_char, c.end_char, "
        "t.source_id, t.role, t.created_at, t.ordinal "
        "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
        "ORDER BY t.ordinal, c.rowid"
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    seen_chunks: set[str] = set()
    for row in rows:
        chunk_id = str(row[0])
        if chunk_id in seen_chunks:
            raise ValueError(f"duplicate source chunk identity: {chunk_id}")
        seen_chunks.add(chunk_id)
        turn_id = str(row[1])
        text = str(row[2])
        source_id = str(row[5] or turn_id).strip()
        if not source_id:
            raise ValueError("source projection found an empty source identity")
        grouped.setdefault(source_id, []).append(
            {
                "chunk_id": chunk_id,
                "turn_id": turn_id,
                "text_sha256": quote_sha256(text),
                "start_char": int(row[3]),
                "end_char": int(row[4]),
                "role": str(row[6]),
                "created_at": str(row[7]),
                "ordinal": int(row[8]),
                "source_metadata": parse_source_metadata(text) is not None,
            }
        )

    streams: list[SourceChunkStream] = []
    for source_id, source_rows in grouped.items():
        content = tuple(
            str(row["chunk_id"])
            for row in source_rows
            if not bool(row["source_metadata"])
        )
        metadata = tuple(
            str(row["chunk_id"])
            for row in source_rows
            if bool(row["source_metadata"])
        )
        streams.append(
            SourceChunkStream(
                source_id=source_id,
                content_chunk_ids=content,
                metadata_chunk_ids=metadata,
                first_ordinal=int(source_rows[0]["ordinal"]),
                last_ordinal=int(source_rows[-1]["ordinal"]),
                stream_sha256=identity_sha256(
                    {"source_id": source_id, "chunks": source_rows}
                ),
            )
        )
    return tuple(
        sorted(
            streams,
            key=lambda stream: (
                stream.first_ordinal,
                stream.source_id,
            ),
        )
    )


def rank_episode_source_candidates(
    anchors: Sequence[RetrievalResult],
    lexical_sources: Sequence[tuple[str, float]],
    *,
    universe_source_ids: Sequence[str],
    max_sources: int,
    rrf_constant: int = 60,
) -> tuple[EpisodeSourceCandidate, ...]:
    """Fuse chunk and source-lexical ranks without comparing raw scores."""

    limit = _positive_integer(max_sources, "max_sources")
    constant = _positive_integer(rrf_constant, "rrf_constant")
    universe = tuple(str(value).strip() for value in universe_source_ids)
    if any(not value for value in universe) or len(set(universe)) != len(universe):
        raise ValueError("source universe must contain unique non-empty IDs")
    universe_set = set(universe)
    direct_ids: list[str] = []
    for result in anchors:
        if not isinstance(result, RetrievalResult):
            raise TypeError("anchors must contain RetrievalResult values")
        memory_source = (
            None
            if result.memory_source_id is None
            else str(result.memory_source_id).strip()
        )
        turn_source = (
            None
            if result.turn is None or result.turn.source_id is None
            else str(result.turn.source_id).strip()
        )
        if memory_source and turn_source and memory_source != turn_source:
            raise ValueError("anchor source identities disagree")
        source_id = (
            memory_source
            or turn_source
            or result.chunk.turn_id
        )
        normalized = str(source_id).strip()
        if normalized not in universe_set:
            raise ValueError("anchor source is absent from the source universe")
        if normalized and normalized not in direct_ids:
            direct_ids.append(normalized)

    lexical_ids: list[str] = []
    for raw_source_id, raw_score in lexical_sources:
        source_id = str(raw_source_id).strip()
        score = float(raw_score)
        if not source_id or not math.isfinite(score):
            raise ValueError("source TF-ISF rows require finite scores and IDs")
        if source_id not in universe_set:
            raise ValueError("TF-ISF source is absent from the source universe")
        if source_id not in lexical_ids:
            lexical_ids.append(source_id)

    scores: dict[str, float] = {source_id: 0.0 for source_id in universe}
    routes: dict[str, set[str]] = {
        source_id: {"unscored_source"} for source_id in universe
    }
    for route, source_ids in (
        ("direct", direct_ids),
        ("source_tfisf", lexical_ids),
    ):
        for rank, source_id in enumerate(source_ids, start=1):
            scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (
                constant + rank
            )
            routes[source_id].discard("unscored_source")
            routes.setdefault(source_id, set()).add(route)
    if not universe:
        return ()
    scale = max(scores.values()) or 1.0
    candidates = [
        EpisodeSourceCandidate(
            source_id=source_id,
            score=score / scale,
            route="+".join(sorted(routes[source_id])),
        )
        for source_id, score in scores.items()
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda item: (-item.score, item.source_id, item.route),
        )[:limit]
    )


def build_episode_source_candidate_scope(
    *,
    artifact_id: str,
    snapshot: DiscourseSnapshot,
    query: str,
    anchors: Sequence[RetrievalResult],
    lexical_sources: Sequence[tuple[str, float]],
    universe_source_ids: Sequence[str],
    max_sources: int,
    rrf_constant: int = 60,
) -> EpisodeSourceCandidateScope:
    """Build a source-universe receipt for one independent episode route."""

    normalized_artifact = str(artifact_id).strip()
    normalized_query = str(query).strip()
    if not normalized_artifact or normalized_artifact not in snapshot.artifact_ids:
        raise ValueError("source routing requires an artifact in the snapshot")
    if not normalized_query:
        raise ValueError("query must be non-empty")
    limit = _positive_integer(max_sources, "max_sources")
    constant = _positive_integer(rrf_constant, "rrf_constant")
    universe = tuple(str(value).strip() for value in universe_source_ids)
    ranked = rank_episode_source_candidates(
        anchors,
        lexical_sources,
        universe_source_ids=universe,
        max_sources=max(len(universe), 1),
        rrf_constant=constant,
    )
    selected = ranked[:limit]
    truncated = tuple(item.source_id for item in ranked[limit:])
    return EpisodeSourceCandidateScope(
        artifact_id=normalized_artifact,
        snapshot_sha256=snapshot.snapshot_sha256,
        source_revision=snapshot.source_revision,
        source_content_sha256=snapshot.source_content_sha256,
        query_sha256=identity_sha256({"query": normalized_query}),
        router_policy_sha256=identity_sha256(
            {
                "router": "direct-source-tfisf-rrf-v1",
                "max_sources": limit,
                "rrf_constant": constant,
                "source_tfisf_scope": "complete_source_universe",
            }
        ),
        universe_source_ids=universe,
        candidates=selected,
        truncated_source_ids=truncated,
        universe_enumerated=True,
    )


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        normalized = int(value)  # type: ignore[arg-type]
        exact = math.isfinite(float(value)) and float(value) == normalized  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if not exact or normalized < 1:
        raise ValueError(f"{label} must be a positive integer")
    return normalized


__all__ = [
    "SourceChunkStream",
    "build_episode_source_candidate_scope",
    "rank_episode_source_candidates",
    "scan_discourse_source_chunks",
]
