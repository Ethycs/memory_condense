"""Deterministic, source-local construction of grounded episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable, Sequence

from memory_condense.domain.discourse import (
    Episode,
    EvidenceSpan,
    evidence_span_sort_key,
    identity_sha256,
    make_episode_id,
    quote_sha256,
)

from .boundaries import (
    AdaptiveBoundaryDetector,
    BoundaryDetector,
    BoundaryProposal,
    BoundaryRefinement,
    CohesionBoundaryRefiner,
)
from .surprise import LexicalEmbeddingChangeScorer, SurpriseScorer, score_surprise_sequence


@dataclass(frozen=True, slots=True)
class EpisodeBuildResult:
    """Text-free result of one episode-formation call."""

    source_id: str
    artifact_id: str
    episodes: tuple[Episode, ...]
    initial_boundaries: tuple[BoundaryProposal, ...]
    refined_boundaries: tuple[BoundaryRefinement, ...]
    forced_boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.artifact_id.strip():
            raise ValueError("source_id and artifact_id must be non-empty")
        episodes = tuple(self.episodes)
        if any(item.source_id != self.source_id for item in episodes):
            raise ValueError("build results cannot contain another source")
        if any(item.artifact_id != self.artifact_id for item in episodes):
            raise ValueError("build results cannot mix annotation artifacts")
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "initial_boundaries", tuple(self.initial_boundaries))
        object.__setattr__(self, "refined_boundaries", tuple(self.refined_boundaries))
        object.__setattr__(self, "forced_boundaries", tuple(self.forced_boundaries))


class EpisodeBuilder:
    """Turn ordered source spans into size-bounded episodes.

    Data-dependent text, embeddings, surprise scores, and graph similarities
    exist only as local variables during :meth:`build`.  The builder retains
    four immutable configuration values and returns domain episodes containing
    exact evidence references plus scalar boundary receipts.
    """

    __slots__ = ("min_size", "max_size", "detector", "refiner")

    def __init__(
        self,
        *,
        min_size: int = 2,
        max_size: int = 16,
        detector: BoundaryDetector | None = None,
        refiner: CohesionBoundaryRefiner | None = None,
    ) -> None:
        if int(min_size) < 1:
            raise ValueError("min_size must be positive")
        if int(max_size) < int(min_size):
            raise ValueError("max_size cannot be smaller than min_size")
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.detector = detector or AdaptiveBoundaryDetector()
        self.refiner = refiner

    def build(
        self,
        *,
        source_id: str,
        artifact_id: str,
        spans: Sequence[EvidenceSpan],
        texts: Sequence[str] | None = None,
        embeddings: Sequence[Sequence[float] | None] | None = None,
        surprise_scores: Sequence[float] | None = None,
        surprise_scorer: SurpriseScorer | None = None,
        sequence_start: int = 0,
    ) -> EpisodeBuildResult:
        normalized_source = str(source_id).strip()
        normalized_artifact = str(artifact_id).strip()
        if not normalized_source or not normalized_artifact:
            raise ValueError("source_id and artifact_id must be non-empty")
        if int(sequence_start) < 0:
            raise ValueError("sequence_start must be non-negative")

        evidence = tuple(spans)
        if not evidence:
            return EpisodeBuildResult(
                source_id=normalized_source,
                artifact_id=normalized_artifact,
                episodes=(),
                initial_boundaries=(),
                refined_boundaries=(),
                forced_boundaries=(),
            )
        expected_order = tuple(sorted(evidence, key=evidence_span_sort_key))
        if evidence != expected_order:
            raise ValueError("spans must be supplied in deterministic source order")
        if len({identity_sha256(item.identity_payload()) for item in evidence}) != len(evidence):
            raise ValueError("duplicate evidence spans are not allowed")
        if any(item.source_id not in (None, normalized_source) for item in evidence):
            raise ValueError("episode formation cannot cross source histories")

        text_rows = _validate_texts(texts, evidence)
        vector_rows = _validate_embeddings(embeddings, len(evidence))
        scores = _resolve_surprises(
            item_count=len(evidence),
            texts=text_rows,
            embeddings=vector_rows,
            surprise_scores=surprise_scores,
            surprise_scorer=surprise_scorer,
            require_scores=getattr(self.detector, "requires_surprise_scores", True),
        )
        initial = self.detector.detect(scores)
        if self.refiner is None:
            refined = tuple(
                BoundaryRefinement(
                    initial_position=item.position,
                    position=item.position,
                    score=item.score,
                    threshold=item.threshold,
                )
                for item in initial
            )
        else:
            refined = self.refiner.refine(
                initial,
                item_count=len(evidence),
                similarities=_similarity_lookup(text_rows, vector_rows),
            )

        boundaries, forced = _select_size_bounded_boundaries(
            len(evidence),
            refined,
            min_size=self.min_size,
            max_size=self.max_size,
        )
        refined_by_position = {item.position: item for item in refined}
        starts = (0, *boundaries)
        ends = (*boundaries, len(evidence))
        episodes: list[Episode] = []
        forced_set = set(forced)
        for offset, (start, end) in enumerate(zip(starts, ends, strict=True)):
            episode_evidence = evidence[start:end]
            if start == 0:
                method = "stream_start"
                initial_position = None
                refined_position = None
                boundary_score = None
                boundary_threshold = None
            elif start in forced_set:
                method = "forced_max_size"
                initial_position = start
                refined_position = start
                boundary_score = None
                boundary_threshold = None
            else:
                receipt = refined_by_position[start]
                detector_method = str(
                    getattr(self.detector, "method", "injected_boundary")
                )
                method = (
                    f"{detector_method}_graph_refined"
                    if receipt.cohesion is not None
                    else detector_method
                )
                initial_position = receipt.initial_position
                refined_position = receipt.position
                boundary_score = receipt.score
                boundary_threshold = receipt.threshold

            sequence_no = int(sequence_start) + offset
            episode_id = make_episode_id(
                artifact_id=normalized_artifact,
                source_id=normalized_source,
                sequence_no=sequence_no,
                evidence=episode_evidence,
            )
            episodes.append(
                Episode(
                    episode_id=episode_id,
                    artifact_id=normalized_artifact,
                    source_id=normalized_source,
                    sequence_no=sequence_no,
                    first_ordinal=episode_evidence[0].ordinal,
                    last_ordinal=episode_evidence[-1].ordinal,
                    evidence=episode_evidence,
                    boundary_method=method,
                    initial_boundary=initial_position,
                    refined_boundary=refined_position,
                    boundary_score=boundary_score,
                    boundary_threshold=boundary_threshold,
                )
            )

        return EpisodeBuildResult(
            source_id=normalized_source,
            artifact_id=normalized_artifact,
            episodes=tuple(episodes),
            initial_boundaries=initial,
            refined_boundaries=refined,
            forced_boundaries=forced,
        )


def _validate_texts(
    texts: Sequence[str] | None,
    spans: Sequence[EvidenceSpan],
) -> tuple[str, ...] | None:
    if texts is None:
        return None
    rows = tuple(str(text) for text in texts)
    if len(rows) != len(spans):
        raise ValueError("texts must align one-for-one with evidence spans")
    for index, (text, span) in enumerate(zip(rows, spans, strict=True)):
        if quote_sha256(text) != span.quote_sha256:
            raise ValueError(f"text at index {index} does not match its evidence hash")
    return rows


def _validate_embeddings(
    embeddings: Sequence[Sequence[float] | None] | None,
    item_count: int,
) -> tuple[tuple[float, ...] | None, ...] | None:
    if embeddings is None:
        return None
    rows = tuple(embeddings)
    if len(rows) != item_count:
        raise ValueError("embeddings must align one-for-one with evidence spans")
    normalized: list[tuple[float, ...] | None] = []
    dimension: int | None = None
    for row in rows:
        if row is None:
            normalized.append(None)
            continue
        vector = tuple(float(value) for value in row)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embeddings must have finite positive dimension")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("all supplied embeddings must share one dimension")
        normalized.append(vector)
    return tuple(normalized)


def _resolve_surprises(
    *,
    item_count: int,
    texts: tuple[str, ...] | None,
    embeddings: tuple[tuple[float, ...] | None, ...] | None,
    surprise_scores: Sequence[float] | None,
    surprise_scorer: SurpriseScorer | None,
    require_scores: bool,
) -> tuple[float, ...]:
    if surprise_scores is not None:
        values = tuple(float(value) for value in surprise_scores)
        if len(values) != item_count:
            raise ValueError("surprise_scores must align one-for-one with spans")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("surprise_scores must all be finite")
        return values
    if texts is None:
        if not require_scores:
            return (0.0,) * item_count
        raise ValueError("texts or injected surprise_scores are required")
    scorer = surprise_scorer or LexicalEmbeddingChangeScorer()
    return score_surprise_sequence(
        scorer,
        texts,
        embeddings=embeddings,
    )


def _similarity_lookup(
    texts: tuple[str, ...] | None,
    embeddings: tuple[tuple[float, ...] | None, ...] | None,
) -> Callable[[int, int], float] | None:
    if texts is None and embeddings is None:
        return None
    item_count = len(texts) if texts is not None else len(embeddings or ())
    if texts is None:
        if any(row is None for row in embeddings or ()):
            return None
        text_rows = ("",) * item_count
        scorer = LexicalEmbeddingChangeScorer(
            lexical_weight=0.0,
            embedding_weight=1.0,
        )
    else:
        text_rows = texts
        scorer = LexicalEmbeddingChangeScorer()
    vector_rows = embeddings or ((None,) * item_count)
    def similarity(left: int, right: int) -> float:
        if left == right:
            return 1.0
        change = scorer.score(
            text_rows[left],
            text_rows[right],
            previous_embedding=vector_rows[left],
            current_embedding=vector_rows[right],
        )
        return max(0.0, min(1.0, 1.0 - change))

    return similarity


def _select_size_bounded_boundaries(
    item_count: int,
    refinements: Sequence[BoundaryRefinement],
    *,
    min_size: int,
    max_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if item_count == 0:
        return (), ()
    if item_count >= min_size and not _partition_feasible(item_count, min_size, max_size):
        raise ValueError(
            "source length cannot satisfy both min_size and max_size; "
            "choose compatible episode limits"
        )
    if item_count < min_size:
        return (), ()

    desired = tuple(sorted({item.position for item in refinements}))
    selected: list[int] = []
    forced: list[int] = []
    start = 0
    while start < item_count:
        remaining = item_count - start
        candidates = [
            position
            for position in desired
            if start + min_size <= position <= min(start + max_size, item_count - 1)
            and _partition_feasible(item_count - position, min_size, max_size)
        ]
        if candidates:
            position = candidates[0]
            selected.append(position)
            start = position
            continue
        if remaining <= max_size:
            break
        possible = [
            position
            for position in range(start + max_size, start + min_size - 1, -1)
            if _partition_feasible(item_count - position, min_size, max_size)
        ]
        if not possible:  # guarded by the whole-stream feasibility check
            raise RuntimeError("failed to construct a feasible episode partition")
        position = possible[0]
        selected.append(position)
        forced.append(position)
        start = position
    return tuple(selected), tuple(forced)


def _partition_feasible(item_count: int, min_size: int, max_size: int) -> bool:
    if item_count == 0:
        return True
    if item_count < min_size:
        return False
    minimum_parts = math.ceil(item_count / max_size)
    maximum_parts = item_count // min_size
    return minimum_parts <= maximum_parts


__all__ = ["EpisodeBuildResult", "EpisodeBuilder"]
