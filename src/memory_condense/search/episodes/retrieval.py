"""Source-bounded episode seeding and temporal-contiguity expansion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

from memory_condense.domain.discourse import Episode, EpisodeSeed, identity_sha256
from memory_condense.domain.schemas import RetrievalResult


@runtime_checkable
class EpisodeLookup(Protocol):
    """Read-only store seam used by episode retrieval.

    The concrete persistence layer may be SQLite or an in-memory fixture.  The
    retrieval algorithm depends only on these bounded, source-aware reads.
    """

    def episode_ids_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        artifact_id: str | None = None,
    ) -> dict[str, str]: ...

    def get_episode(self, episode_id: str) -> Episode | None: ...

    def adjacent_episodes(
        self,
        episode_id: str,
        *,
        radius: int = 1,
        include_self: bool = False,
    ) -> tuple[Episode, ...]: ...

    def episodes_for_source(
        self,
        artifact_id: str,
        source_id: str,
        *,
        start_sequence: int | None = None,
        end_sequence: int | None = None,
        limit: int | None = None,
    ) -> tuple[Episode, ...]: ...


@dataclass(frozen=True, slots=True)
class EpisodeRetrievalPolicy:
    """All independent caps for chunk-to-episode expansion."""

    artifact_id: str | None = None
    max_anchor_episodes: int = 8
    previous_episodes: int = 1
    next_episodes: int = 1
    max_episode_seeds: int = 24
    max_direct_fallbacks: int = 16
    neighbor_decay: float = 0.85

    def __post_init__(self) -> None:
        if self.artifact_id is not None:
            normalized_artifact = str(self.artifact_id).strip()
            if not normalized_artifact:
                raise ValueError("artifact_id must be non-empty when supplied")
            object.__setattr__(self, "artifact_id", normalized_artifact)
        for name in ("max_anchor_episodes", "max_episode_seeds", "max_direct_fallbacks"):
            value = _exact_int(getattr(self, name), name)
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("previous_episodes", "next_episodes"):
            value = _exact_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        decay = float(self.neighbor_decay)
        if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError("neighbor_decay must be finite and inside [0, 1]")
        object.__setattr__(self, "neighbor_decay", decay)

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(
            {
                "artifact_id": self.artifact_id,
                "max_anchor_episodes": self.max_anchor_episodes,
                "previous_episodes": self.previous_episodes,
                "next_episodes": self.next_episodes,
                "max_episode_seeds": self.max_episode_seeds,
                "max_direct_fallbacks": self.max_direct_fallbacks,
                "neighbor_decay": self.neighbor_decay,
            }
        )


@dataclass(frozen=True, slots=True)
class DirectChunkSeed:
    """One original raw retrieval hit retained independently of annotation."""

    chunk_id: str
    score: float
    route: str = "direct_chunk"
    failure_code: str = "not_annotated"
    path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        chunk_id = str(self.chunk_id).strip()
        route = str(self.route).strip()
        failure_code = str(self.failure_code).strip()
        if not chunk_id or not route or not failure_code:
            raise ValueError("direct chunk seed IDs and routes must be non-empty")
        if failure_code not in {
            "artifact_not_selected",
            "episode_mapped",
            "identity_mismatch",
            "lookup_error",
            "not_annotated",
        }:
            raise ValueError("unsupported direct fallback failure code")
        if not math.isfinite(float(self.score)):
            raise ValueError("direct chunk seed score must be finite")
        path = tuple(self.path) or (chunk_id,)
        if any(not str(item).strip() for item in path):
            raise ValueError("direct chunk seed paths must contain non-empty IDs")
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class EpisodeRetrievalPlan:
    """Text-free, self-hashed result of bounded episode expansion."""

    policy_sha256: str
    seeds: tuple[EpisodeSeed, ...]
    direct_fallbacks: tuple[DirectChunkSeed, ...]
    truncated_episode_ids: tuple[str, ...] = ()
    truncated_direct_chunk_ids: tuple[str, ...] = ()
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        seeds = tuple(self.seeds)
        direct = tuple(self.direct_fallbacks)
        if len({item.episode_id for item in seeds}) != len(seeds):
            raise ValueError("episode seeds must be unique")
        if len({item.chunk_id for item in direct}) != len(direct):
            raise ValueError("direct fallback chunks must be unique")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "direct_fallbacks", direct)
        object.__setattr__(self, "truncated_episode_ids", tuple(self.truncated_episode_ids))
        object.__setattr__(
            self,
            "truncated_direct_chunk_ids",
            tuple(self.truncated_direct_chunk_ids),
        )
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("episode retrieval receipt does not match its contents")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def direct_chunk_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.direct_fallbacks)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "policy_sha256": self.policy_sha256,
            "seeds": [
                {
                    "episode_id": item.episode_id,
                    "anchor_chunk_id": item.anchor_chunk_id,
                    "score": item.score,
                    "route": item.route,
                    "path": list(item.path),
                }
                for item in self.seeds
            ],
            "direct_fallbacks": [
                {
                    "chunk_id": item.chunk_id,
                    "score": item.score,
                    "route": item.route,
                    "failure_code": item.failure_code,
                    "path": list(item.path),
                }
                for item in self.direct_fallbacks
            ],
            "truncated_episode_ids": list(self.truncated_episode_ids),
            "truncated_direct_chunk_ids": list(self.truncated_direct_chunk_ids),
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class _DirectInput:
    chunk_id: str
    score: float
    route: str
    source_hint: str | None
    source_conflicted: bool


@dataclass(frozen=True, slots=True)
class _Anchor:
    episode: Episode
    chunk_id: str
    score: float
    source_route: str


@dataclass(frozen=True, slots=True)
class _Neighbor:
    episode: Episode
    anchor: _Anchor
    direction: str
    distance: int
    score: float
    anchor_rank: int


def expand_episode_seeds(
    results: Sequence[RetrievalResult],
    store: EpisodeLookup,
    *,
    policy: EpisodeRetrievalPolicy | None = None,
) -> EpisodeRetrievalPlan:
    """Map direct chunk hits to episodes and bounded source-local neighbors.

    Annotation failure is fail-open: the original chunk ID and score survive
    as a :class:`DirectChunkSeed`.  Store errors never fabricate an episode.
    Every episode is revalidated against its anchor chunk, artifact, and source
    before admission; adjacency rows are independently source checked.
    """
    active_policy = policy or EpisodeRetrievalPolicy()
    direct_inputs = _deduplicate_inputs(results)
    chunk_ids = tuple(item.chunk_id for item in direct_inputs)
    lookup_failed = False
    if active_policy.artifact_id is None:
        episode_ids: Mapping[str, str] = {}
    else:
        try:
            looked_up = store.episode_ids_for_chunks(
                chunk_ids,
                artifact_id=active_policy.artifact_id,
            )
            episode_ids = looked_up if isinstance(looked_up, Mapping) else {}
            lookup_failed = not isinstance(looked_up, Mapping)
        except Exception:
            episode_ids = {}
            lookup_failed = True

    fallback_by_chunk: dict[str, DirectChunkSeed] = {}
    anchor_by_episode: dict[str, _Anchor] = {}
    for direct in direct_inputs:
        episode_id = episode_ids.get(direct.chunk_id)
        episode, episode_failure = _get_episode(store, episode_id)
        valid_episode = _valid_anchor_episode(
            episode,
            direct,
            artifact_id=active_policy.artifact_id,
        )
        if not valid_episode:
            if active_policy.artifact_id is None:
                failure_code = "artifact_not_selected"
            elif lookup_failed or episode_failure == "lookup_error":
                failure_code = "lookup_error"
            elif episode_id is None:
                failure_code = "not_annotated"
            else:
                failure_code = "identity_mismatch"
            disposition = f"episode_failure:{failure_code}"
        else:
            failure_code = "episode_mapped"
            disposition = f"episode_mapped:{episode_id}"
        fallback_by_chunk[direct.chunk_id] = DirectChunkSeed(
            chunk_id=direct.chunk_id,
            score=direct.score,
            route=direct.route,
            failure_code=failure_code,
            path=(
                direct.chunk_id,
                f"retrieval_route:{direct.route}",
                disposition,
            ),
        )
        if not valid_episode:
            continue
        assert episode is not None
        anchor = _Anchor(
            episode=episode,
            chunk_id=direct.chunk_id,
            score=direct.score,
            source_route=direct.route,
        )
        existing = anchor_by_episode.get(episode.episode_id)
        if existing is None or _anchor_choice_key(anchor) < _anchor_choice_key(existing):
            anchor_by_episode[episode.episode_id] = anchor

    ranked_anchors = sorted(anchor_by_episode.values(), key=_anchor_rank_key)
    anchor_cap = min(
        active_policy.max_anchor_episodes,
        active_policy.max_episode_seeds,
    )
    admitted_anchors = ranked_anchors[:anchor_cap]
    truncated_episode_ids = [
        item.episode.episode_id for item in ranked_anchors[anchor_cap:]
    ]

    seeds: list[EpisodeSeed] = [
        EpisodeSeed(
            episode_id=anchor.episode.episode_id,
            anchor_chunk_id=anchor.chunk_id,
            score=anchor.score,
            route="episode_direct",
            path=(
                anchor.chunk_id,
                anchor.episode.episode_id,
                f"retrieval_route:{anchor.source_route}",
            ),
        )
        for anchor in admitted_anchors
    ]
    admitted_episode_ids = {item.episode_id for item in seeds}

    neighbor_candidates: list[_Neighbor] = []
    radius = max(active_policy.previous_episodes, active_policy.next_episodes)
    if radius:
        for anchor_rank, anchor in enumerate(admitted_anchors):
            neighbors = _safe_adjacent(store, anchor.episode.episode_id, radius)
            previous = sorted(
                (
                    item
                    for item in neighbors
                    if _valid_neighbor(item, anchor.episode, active_policy.artifact_id)
                    and item.sequence_no < anchor.episode.sequence_no
                ),
                key=lambda item: (-item.sequence_no, item.episode_id),
            )[: active_policy.previous_episodes]
            following = sorted(
                (
                    item
                    for item in neighbors
                    if _valid_neighbor(item, anchor.episode, active_policy.artifact_id)
                    and item.sequence_no > anchor.episode.sequence_no
                ),
                key=lambda item: (item.sequence_no, item.episode_id),
            )[: active_policy.next_episodes]
            for direction, rows in (("previous", previous), ("next", following)):
                for distance, episode in enumerate(rows, start=1):
                    neighbor_candidates.append(
                        _Neighbor(
                            episode=episode,
                            anchor=anchor,
                            direction=direction,
                            distance=distance,
                            score=_neighbor_score(
                                anchor.score,
                                distance,
                                active_policy.neighbor_decay,
                            ),
                            anchor_rank=anchor_rank,
                        )
                    )

    neighbor_candidates.sort(key=_neighbor_rank_key)
    for candidate in neighbor_candidates:
        episode_id = candidate.episode.episode_id
        if episode_id in admitted_episode_ids:
            continue
        if len(seeds) >= active_policy.max_episode_seeds:
            if episode_id not in truncated_episode_ids:
                truncated_episode_ids.append(episode_id)
            continue
        seeds.append(
            EpisodeSeed(
                episode_id=episode_id,
                # ``anchor_chunk_id`` is the in-episode evidence coordinate
                # that makes this seed independently verifiable.  The
                # originating direct hit remains explicit in ``path`` below.
                anchor_chunk_id=candidate.episode.evidence[0].chunk_id,
                score=candidate.score,
                route=f"episode_{candidate.direction}",
                path=(
                    candidate.anchor.chunk_id,
                    candidate.anchor.episode.episode_id,
                    episode_id,
                    f"retrieval_route:{candidate.anchor.source_route}",
                ),
            )
        )
        admitted_episode_ids.add(episode_id)

    ranked_fallbacks = sorted(
        fallback_by_chunk.values(),
        key=lambda item: (-item.score, item.chunk_id, item.route),
    )
    fallback_cap = active_policy.max_direct_fallbacks
    admitted_fallbacks = tuple(ranked_fallbacks[:fallback_cap])
    truncated_direct = tuple(
        item.chunk_id for item in ranked_fallbacks[fallback_cap:]
    )
    return EpisodeRetrievalPlan(
        policy_sha256=active_policy.policy_sha256,
        seeds=tuple(seeds),
        direct_fallbacks=admitted_fallbacks,
        truncated_episode_ids=tuple(truncated_episode_ids),
        truncated_direct_chunk_ids=truncated_direct,
    )


def _deduplicate_inputs(results: Sequence[RetrievalResult]) -> tuple[_DirectInput, ...]:
    by_chunk: dict[str, _DirectInput] = {}
    for result in results:
        score = float(result.score)
        if not math.isfinite(score):
            raise ValueError("retrieval result scores must be finite")
        chunk_id = str(result.chunk.chunk_id).strip()
        if not chunk_id:
            raise ValueError("retrieval result chunk IDs must be non-empty")
        hints = {
            str(value).strip()
            for value in (
                result.memory_source_id,
                None if result.turn is None else result.turn.source_id,
            )
            if value is not None and str(value).strip()
        }
        direct = _DirectInput(
            chunk_id=chunk_id,
            score=score,
            route=str(result.route or "direct_chunk"),
            source_hint=next(iter(hints)) if len(hints) == 1 else None,
            source_conflicted=len(hints) > 1,
        )
        existing = by_chunk.get(chunk_id)
        if existing is None or _direct_choice_key(direct) < _direct_choice_key(existing):
            by_chunk[chunk_id] = direct
    return tuple(sorted(by_chunk.values(), key=_direct_rank_key))


def _get_episode(
    store: EpisodeLookup,
    episode_id: str | None,
) -> tuple[Episode | None, str]:
    if episode_id is None or not str(episode_id).strip():
        return None, "not_annotated"
    try:
        episode = store.get_episode(str(episode_id))
    except Exception:
        return None, "lookup_error"
    if not isinstance(episode, Episode):
        return None, "identity_mismatch"
    return episode, "ok"


def _safe_adjacent(
    store: EpisodeLookup,
    episode_id: str,
    radius: int,
) -> tuple[Episode, ...]:
    try:
        rows = tuple(
            store.adjacent_episodes(
                episode_id,
                radius=radius,
                include_self=False,
            )
        )
    except Exception:
        return ()
    unique: dict[str, Episode] = {}
    for row in rows:
        if isinstance(row, Episode):
            unique.setdefault(row.episode_id, row)
    return tuple(unique.values())


def _valid_anchor_episode(
    episode: Episode | None,
    direct: _DirectInput,
    *,
    artifact_id: str | None,
) -> bool:
    if episode is None or direct.source_conflicted:
        return False
    if artifact_id is not None and episode.artifact_id != artifact_id:
        return False
    if direct.source_hint is not None and episode.source_id != direct.source_hint:
        return False
    return any(item.chunk_id == direct.chunk_id for item in episode.evidence)


def _valid_neighbor(
    episode: Episode,
    anchor: Episode,
    artifact_id: str | None,
) -> bool:
    return (
        episode.episode_id != anchor.episode_id
        and episode.source_id == anchor.source_id
        and episode.artifact_id == anchor.artifact_id
        and (artifact_id is None or episode.artifact_id == artifact_id)
    )


def _neighbor_score(anchor_score: float, distance: int, decay: float) -> float:
    penalty = distance * (1.0 - decay) * max(1.0, abs(anchor_score))
    value = anchor_score - penalty
    if math.isfinite(value):
        return value
    return -float.fromhex("0x1.fffffffffffffp+1023")


def _exact_int(value: object, label: str) -> int:
    try:
        normalized = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    try:
        exact = float(value) == float(normalized)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact:
        raise ValueError(f"{label} must be an integer")
    return normalized


def _direct_choice_key(item: _DirectInput) -> tuple[float, str, str, str]:
    return (-item.score, item.route, item.source_hint or "", item.chunk_id)


def _direct_rank_key(item: _DirectInput) -> tuple[float, str, str]:
    return (-item.score, item.chunk_id, item.route)


def _anchor_choice_key(item: _Anchor) -> tuple[float, str, str]:
    return (-item.score, item.chunk_id, item.source_route)


def _anchor_rank_key(item: _Anchor) -> tuple[float, str, int, str, str]:
    return (
        -item.score,
        item.episode.source_id,
        item.episode.sequence_no,
        item.episode.episode_id,
        item.chunk_id,
    )


def _neighbor_rank_key(
    item: _Neighbor,
) -> tuple[float, int, int, int, str, int, str]:
    return (
        -item.score,
        item.distance,
        0 if item.direction == "previous" else 1,
        item.anchor_rank,
        item.episode.source_id,
        item.episode.sequence_no,
        item.episode.episode_id,
    )


__all__ = [
    "DirectChunkSeed",
    "EpisodeLookup",
    "EpisodeRetrievalPlan",
    "EpisodeRetrievalPolicy",
    "expand_episode_seeds",
]
