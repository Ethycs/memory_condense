"""Bounded query-to-episode discovery over persisted representatives.

Representative text is hydrated only for one query-local Qwen tournament.  The
returned plan retains source/chunk/episode identities, scalar scores, hashes,
and workspace diagnostics; candidate text and transformer state do not escape
the call.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Literal, Protocol, runtime_checkable

from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
    NestedMemoryInspection,
)
from memory_condense.domain._tokenizer import (
    tokenizer_proxy_identity,
    truncate_to_tokens_lossless,
)
from memory_condense.domain.discourse import (
    Episode,
    EpisodeRepresentative,
    EpisodeSeed,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.schemas import RetrievalResult


@runtime_checkable
class EpisodeRepresentativeLookup(Protocol):
    """Read-only episode/representative seam required by discovery."""

    def episodes_for_source(
        self,
        artifact_id: str,
        source_id: str,
        *,
        start_sequence: int | None = None,
        end_sequence: int | None = None,
        limit: int | None = None,
    ) -> tuple[Episode, ...]: ...

    def get_representatives(
        self,
        episode_id: str,
    ) -> tuple[EpisodeRepresentative, ...]: ...


@runtime_checkable
class NestedEpisodeLinker(Protocol):
    """Structural seam implemented by :class:`QwenMemoryLinker`."""

    max_candidates: int

    def inspect_nested(
        self,
        source_text: str,
        candidate_groups: Sequence[Sequence[AssociativeMemoryCandidate]],
        *,
        beam_per_group: int = 2,
        top_k: int = 4,
        score_mode: Literal["qk", "qk_ov"] = "qk",
    ) -> NestedMemoryInspection: ...


@runtime_checkable
class RepresentativeHydrator(Protocol):
    """Hydrate one known chunk without initiating another search."""

    def __call__(
        self,
        chunk_id: str,
        *,
        score: float,
        route: str,
    ) -> RetrievalResult | None: ...


@dataclass(frozen=True, slots=True)
class EpisodeSourceCandidate:
    """One gold-blind source-router result offered to episode discovery."""

    source_id: str
    score: float
    route: str = "source_router"

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        route = str(self.route).strip()
        if not source_id or not route:
            raise ValueError("source candidate IDs and routes must be non-empty")
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("source candidate score must be finite")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "route", route)

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "score": self.score,
            "route": self.route,
        }


@dataclass(frozen=True, slots=True)
class EpisodeSourceCandidateScope:
    """Content-bound account of the source universe offered to discovery."""

    artifact_id: str
    snapshot_sha256: str
    source_revision: int
    source_content_sha256: str
    query_sha256: str
    router_policy_sha256: str
    universe_source_ids: tuple[str, ...]
    candidates: tuple[EpisodeSourceCandidate, ...]
    truncated_source_ids: tuple[str, ...]
    universe_enumerated: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        artifact_id = str(self.artifact_id).strip()
        if not artifact_id:
            raise ValueError("artifact_id must be non-empty")
        object.__setattr__(self, "artifact_id", artifact_id)
        for name in (
            "snapshot_sha256",
            "source_content_sha256",
            "query_sha256",
            "router_policy_sha256",
        ):
            _digest(getattr(self, name), name)
        source_revision = _exact_int(self.source_revision, "source_revision")
        if source_revision < 0:
            raise ValueError("source_revision must be non-negative")
        object.__setattr__(self, "source_revision", source_revision)
        if type(self.universe_enumerated) is not bool:
            raise ValueError("universe_enumerated must be boolean")
        universe = tuple(str(value).strip() for value in self.universe_source_ids)
        if any(not value for value in universe) or len(set(universe)) != len(universe):
            raise ValueError("source universe must contain unique non-empty IDs")
        candidates = tuple(self.candidates)
        if any(not isinstance(item, EpisodeSourceCandidate) for item in candidates):
            raise TypeError("candidates must contain EpisodeSourceCandidate values")
        candidate_ids = tuple(item.source_id for item in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("source candidates must be unique")
        truncated = tuple(str(value).strip() for value in self.truncated_source_ids)
        if any(not value for value in truncated) or len(set(truncated)) != len(truncated):
            raise ValueError("truncated source IDs must be unique and non-empty")
        if set(candidate_ids) & set(truncated):
            raise ValueError("selected and truncated sources must be disjoint")
        if not set((*candidate_ids, *truncated)) <= set(universe):
            raise ValueError("source routing rows must belong to the universe")
        if self.universe_enumerated and set((*candidate_ids, *truncated)) != set(
            universe
        ):
            raise ValueError("an exhaustive source route must account for its universe")
        object.__setattr__(self, "universe_source_ids", universe)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "truncated_source_ids", truncated)
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("source candidate scope receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def selected_scope_exhaustive(self) -> bool:
        return self.universe_enumerated and not self.truncated_source_ids

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "snapshot_sha256": self.snapshot_sha256,
            "source_revision": self.source_revision,
            "source_content_sha256": self.source_content_sha256,
            "query_sha256": self.query_sha256,
            "router_policy_sha256": self.router_policy_sha256,
            "universe_source_ids": list(self.universe_source_ids),
            "candidates": [item.identity_payload() for item in self.candidates],
            "truncated_source_ids": list(self.truncated_source_ids),
            "universe_enumerated": self.universe_enumerated,
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class EpisodeRepresentativeRetrievalPolicy:
    """Independent hard caps for one representative discovery pass."""

    artifact_id: str
    max_input_sources: int = 256
    max_source_groups: int = 16
    max_episodes_per_source: int = 64
    max_total_episodes: int = 256
    max_representatives_per_episode: int = 2
    group_size: int = 8
    beam_per_group: int = 2
    top_k: int = 8
    representative_tokens: int = 96
    query_tokens: int = 96
    score_mode: Literal["qk", "qk_ov"] = "qk_ov"

    def __post_init__(self) -> None:
        artifact_id = str(self.artifact_id).strip()
        if not artifact_id:
            raise ValueError("artifact_id is required")
        object.__setattr__(self, "artifact_id", artifact_id)
        for name in (
            "max_input_sources",
            "max_source_groups",
            "max_episodes_per_source",
            "max_total_episodes",
            "max_representatives_per_episode",
            "group_size",
            "beam_per_group",
            "top_k",
            "representative_tokens",
            "query_tokens",
        ):
            value = _exact_int(getattr(self, name), name)
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.max_source_groups > self.max_input_sources:
            raise ValueError("max_source_groups cannot exceed max_input_sources")
        if self.top_k > self.max_total_episodes:
            raise ValueError("top_k cannot exceed max_total_episodes")
        if self.score_mode not in {"qk", "qk_ov"}:
            raise ValueError("score_mode must be 'qk' or 'qk_ov'")

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(
            {
                "policy": {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                },
                "tokenizer_proxy": tokenizer_proxy_identity(),
            }
        )


@dataclass(frozen=True, slots=True)
class EpisodeSourceScan:
    """Text-free account of one bounded source-local episode scan."""

    source_id: str
    requested_limit: int
    observed_count: int
    candidate_count: int
    exhaustive: bool
    status: str = "ok"

    def __post_init__(self) -> None:
        for name in ("source_id", "status"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        for name in ("requested_limit", "observed_count", "candidate_count"):
            value = _exact_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.candidate_count > self.observed_count:
            raise ValueError("candidate_count cannot exceed observed_count")


@dataclass(frozen=True, slots=True)
class EpisodeRepresentativeWitness:
    """Identity of the exact representative text inspected for one episode."""

    episode_id: str
    source_id: str
    anchor_chunk_id: str
    representative_chunk_ids: tuple[str, ...]
    representative_identity_sha256s: tuple[str, ...]
    candidate_text_sha256: str
    source_score: float
    source_route: str

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "source_id",
            "anchor_chunk_id",
            "candidate_text_sha256",
            "source_route",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        chunk_ids = tuple(str(value).strip() for value in self.representative_chunk_ids)
        identities = tuple(
            str(value).strip() for value in self.representative_identity_sha256s
        )
        if not chunk_ids or any(not value for value in chunk_ids):
            raise ValueError("representative chunk IDs must be non-empty")
        if len(chunk_ids) != len(identities) or any(not value for value in identities):
            raise ValueError("each representative chunk requires an identity hash")
        if self.anchor_chunk_id != chunk_ids[0]:
            raise ValueError("anchor_chunk_id must be the first representative")
        if not math.isfinite(float(self.source_score)):
            raise ValueError("source_score must be finite")
        object.__setattr__(self, "representative_chunk_ids", chunk_ids)
        object.__setattr__(self, "representative_identity_sha256s", identities)
        object.__setattr__(self, "source_score", float(self.source_score))

    def identity_payload(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "source_id": self.source_id,
            "anchor_chunk_id": self.anchor_chunk_id,
            "representative_chunk_ids": list(self.representative_chunk_ids),
            "representative_identity_sha256s": list(
                self.representative_identity_sha256s
            ),
            "candidate_text_sha256": self.candidate_text_sha256,
            "source_score": self.source_score,
            "source_route": self.source_route,
        }


@dataclass(frozen=True, slots=True)
class EpisodeRepresentativeRetrievalPlan:
    """Self-hashed, text-free output of query-to-episode discovery."""

    artifact_id: str
    policy_sha256: str
    query_sha256: str
    query_input_sha256: str
    linker_identity_sha256: str
    runtime_binding_certified: bool
    source_scope_receipt_sha256: str | None
    source_universe_exhaustive: bool
    source_scans: tuple[EpisodeSourceScan, ...]
    candidate_witnesses: tuple[EpisodeRepresentativeWitness, ...]
    seeds: tuple[EpisodeSeed, ...]
    truncated_source_ids: tuple[str, ...] = ()
    truncated_episode_ids: tuple[str, ...] = ()
    unavailable_episode_ids: tuple[str, ...] = ()
    passes: int = 0
    max_workspace_candidates: int = 0
    max_workspace_tokens: int = 0
    total_candidate_inspections: int = 0
    returned_plan_transformer_state_bytes: int = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        artifact_id = str(self.artifact_id).strip()
        if not artifact_id:
            raise ValueError("artifact_id must be non-empty")
        object.__setattr__(self, "artifact_id", artifact_id)
        for name in (
            "policy_sha256",
            "query_sha256",
            "query_input_sha256",
            "linker_identity_sha256",
        ):
            _digest(getattr(self, name), name)
        if type(self.runtime_binding_certified) is not bool:
            raise ValueError("runtime_binding_certified must be boolean")
        if self.source_scope_receipt_sha256 is not None:
            _digest(
                self.source_scope_receipt_sha256,
                "source_scope_receipt_sha256",
            )
        if type(self.source_universe_exhaustive) is not bool:
            raise ValueError("source_universe_exhaustive must be boolean")
        if self.source_scope_receipt_sha256 is None and self.source_universe_exhaustive:
            raise ValueError("source exhaustiveness requires a scope receipt")
        scans = tuple(self.source_scans)
        witnesses = tuple(self.candidate_witnesses)
        seeds = tuple(self.seeds)
        if len({item.source_id for item in scans}) != len(scans):
            raise ValueError("source scans must be unique")
        if len({item.episode_id for item in witnesses}) != len(witnesses):
            raise ValueError("candidate episode witnesses must be unique")
        if len({item.episode_id for item in seeds}) != len(seeds):
            raise ValueError("representative episode seeds must be unique")
        witnessed = {item.episode_id for item in witnesses}
        if any(item.episode_id not in witnessed for item in seeds):
            raise ValueError("every seed must have an inspected candidate witness")
        for name in (
            "passes",
            "max_workspace_candidates",
            "max_workspace_tokens",
            "total_candidate_inspections",
            "returned_plan_transformer_state_bytes",
        ):
            value = _exact_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.returned_plan_transformer_state_bytes != 0:
            raise ValueError("representative plan cannot retain transformer state")
        object.__setattr__(self, "source_scans", scans)
        object.__setattr__(self, "candidate_witnesses", witnesses)
        object.__setattr__(self, "seeds", seeds)
        for name in (
            "truncated_source_ids",
            "truncated_episode_ids",
            "unavailable_episode_ids",
        ):
            values = tuple(dict.fromkeys(str(value).strip() for value in getattr(self, name)))
            if any(not value for value in values):
                raise ValueError(f"{name} must contain non-empty IDs")
            object.__setattr__(self, name, values)
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("representative retrieval receipt does not match its contents")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def selected_source_episode_scope_exhaustive(self) -> bool:
        return (
            not self.truncated_source_ids
            and not self.truncated_episode_ids
            and not self.unavailable_episode_ids
            and all(item.exhaustive for item in self.source_scans)
        )

    @property
    def candidate_scope_exhaustive(self) -> bool:
        """Whether both the source universe and its episode scans are complete."""

        return bool(
            self.source_universe_exhaustive
            and self.selected_source_episode_scope_exhaustive
        )

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "policy_sha256": self.policy_sha256,
            "query_sha256": self.query_sha256,
            "query_input_sha256": self.query_input_sha256,
            "linker_identity_sha256": self.linker_identity_sha256,
            "runtime_binding_certified": self.runtime_binding_certified,
            "source_scope_receipt_sha256": self.source_scope_receipt_sha256,
            "source_universe_exhaustive": self.source_universe_exhaustive,
            "source_scans": [
                {
                    "source_id": item.source_id,
                    "requested_limit": item.requested_limit,
                    "observed_count": item.observed_count,
                    "candidate_count": item.candidate_count,
                    "exhaustive": item.exhaustive,
                    "status": item.status,
                }
                for item in self.source_scans
            ],
            "candidate_witnesses": [
                item.identity_payload() for item in self.candidate_witnesses
            ],
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
            "truncated_source_ids": list(self.truncated_source_ids),
            "truncated_episode_ids": list(self.truncated_episode_ids),
            "unavailable_episode_ids": list(self.unavailable_episode_ids),
            "passes": self.passes,
            "max_workspace_candidates": self.max_workspace_candidates,
            "max_workspace_tokens": self.max_workspace_tokens,
            "total_candidate_inspections": self.total_candidate_inspections,
            "returned_plan_transformer_state_bytes": (
                self.returned_plan_transformer_state_bytes
            ),
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def retrieve_episode_representatives(
    query: str,
    source_candidates: Sequence[EpisodeSourceCandidate],
    store: EpisodeRepresentativeLookup,
    hydrate: RepresentativeHydrator,
    linker: NestedEpisodeLinker,
    *,
    policy: EpisodeRepresentativeRetrievalPolicy,
    source_scope: EpisodeSourceCandidateScope | None = None,
) -> EpisodeRepresentativeRetrievalPlan:
    """Discover episodes independently of direct chunk hits.

    Sources are supplied by a separate gold-blind router. Every episode read is
    revalidated against the selected artifact/source, and every representative
    is checked against exact episode evidence before its text enters the
    transient nested-attention workspace.
    """

    normalized_query = str(query).strip()
    if not normalized_query:
        raise ValueError("query must be non-empty")
    query_sha256 = identity_sha256({"query": normalized_query})
    exact_source_candidates = tuple(source_candidates)
    if source_scope is not None:
        if not isinstance(source_scope, EpisodeSourceCandidateScope):
            raise TypeError("source_scope must be an EpisodeSourceCandidateScope")
        if source_scope.artifact_id != policy.artifact_id:
            raise ValueError("source scope belongs to another artifact")
        if source_scope.query_sha256 != query_sha256:
            raise ValueError("source scope belongs to another query")
        if source_scope.candidates != exact_source_candidates:
            raise ValueError("source scope does not bind the supplied candidates")
    if len(source_candidates) > policy.max_input_sources:
        raise ValueError("source candidates exceed max_input_sources")
    linker_max = _exact_int(getattr(linker, "max_candidates", None), "linker.max_candidates")
    if linker_max < 2:
        raise ValueError("linker.max_candidates must be at least two")
    if policy.group_size > linker_max:
        raise ValueError("group_size cannot exceed linker.max_candidates")
    if policy.beam_per_group >= linker_max:
        raise ValueError("beam_per_group must be smaller than linker.max_candidates")

    ranked_sources = _deduplicate_sources(exact_source_candidates)
    admitted_sources = ranked_sources[: policy.max_source_groups]
    truncated_source_ids = tuple(
        item.source_id for item in ranked_sources[policy.max_source_groups :]
    )
    source_scans: list[EpisodeSourceScan] = []
    witnesses: list[EpisodeRepresentativeWitness] = []
    inspection_candidates: dict[str, AssociativeMemoryCandidate] = {}
    candidates_by_source: dict[str, list[AssociativeMemoryCandidate]] = {}
    truncated_episode_ids: list[str] = []
    unavailable_episode_ids: list[str] = []
    query_input = truncate_to_tokens_lossless(
        normalized_query,
        policy.query_tokens,
    )
    query_input_sha256 = identity_sha256({"query_input": query_input})
    linker_identity = _linker_identity(linker)
    linker_identity_sha256 = identity_sha256(linker_identity)
    runtime_binding_certified = bool(
        linker_identity.get("owned_runtime_binding", False)
    )

    for source in admitted_sources:
        requested = policy.max_episodes_per_source
        try:
            raw_rows = store.episodes_for_source(
                policy.artifact_id,
                source.source_id,
                limit=requested + 1,
            )
            rows = tuple(islice(iter(raw_rows), requested + 1))
        except Exception:
            source_scans.append(
                EpisodeSourceScan(
                    source_id=source.source_id,
                    requested_limit=requested,
                    observed_count=0,
                    candidate_count=0,
                    exhaustive=False,
                    status="lookup_error",
                )
            )
            continue

        unique: dict[str, Episode] = {}
        invalid_row = False
        for row in rows:
            if not isinstance(row, Episode):
                invalid_row = True
                continue
            if row.artifact_id != policy.artifact_id or row.source_id != source.source_id:
                unavailable_episode_ids.append(row.episode_id)
                invalid_row = True
                continue
            unique.setdefault(row.episode_id, row)
        ordered = sorted(
            unique.values(),
            key=lambda item: (item.sequence_no, item.episode_id),
        )
        source_overflow = len(ordered) > requested
        if source_overflow:
            truncated_episode_ids.extend(
                item.episode_id for item in ordered[requested:]
            )
        ordered = ordered[:requested]
        source_candidate_count = 0
        for episode in ordered:
            if len(witnesses) >= policy.max_total_episodes:
                truncated_episode_ids.append(episode.episode_id)
                continue
            prepared = _prepare_candidate(
                episode,
                source,
                store=store,
                hydrate=hydrate,
                policy=policy,
            )
            if prepared is None:
                unavailable_episode_ids.append(episode.episode_id)
                continue
            candidate, witness = prepared
            if episode.episode_id in inspection_candidates:
                continue
            inspection_candidates[episode.episode_id] = candidate
            candidates_by_source.setdefault(source.source_id, []).append(candidate)
            witnesses.append(witness)
            source_candidate_count += 1
        source_scans.append(
            EpisodeSourceScan(
                source_id=source.source_id,
                requested_limit=requested,
                observed_count=len(rows),
                candidate_count=source_candidate_count,
                exhaustive=(
                    not source_overflow
                    and not invalid_row
                    and source_candidate_count == len(ordered)
                ),
                status="identity_error" if invalid_row else "ok",
            )
        )

    groups = [
        tuple(rows[start : start + policy.group_size])
        for source in admitted_sources
        for rows in (candidates_by_source.get(source.source_id, []),)
        for start in range(0, len(rows), policy.group_size)
    ]
    if not groups:
        return EpisodeRepresentativeRetrievalPlan(
            artifact_id=policy.artifact_id,
            policy_sha256=policy.policy_sha256,
            query_sha256=query_sha256,
            query_input_sha256=query_input_sha256,
            linker_identity_sha256=linker_identity_sha256,
            runtime_binding_certified=runtime_binding_certified,
            source_scope_receipt_sha256=(
                None if source_scope is None else source_scope.receipt_sha256
            ),
            source_universe_exhaustive=bool(
                source_scope is not None and source_scope.selected_scope_exhaustive
            ),
            source_scans=tuple(source_scans),
            candidate_witnesses=tuple(witnesses),
            seeds=(),
            truncated_source_ids=truncated_source_ids,
            truncated_episode_ids=tuple(truncated_episode_ids),
            unavailable_episode_ids=tuple(unavailable_episode_ids),
        )

    inspection = linker.inspect_nested(
        query_input,
        groups,
        beam_per_group=policy.beam_per_group,
        top_k=min(policy.top_k, linker_max, len(inspection_candidates)),
        score_mode=policy.score_mode,
    )
    if not isinstance(inspection, NestedMemoryInspection):
        raise TypeError("linker returned an invalid nested inspection")
    seeds: list[EpisodeSeed] = []
    seen_hits: set[str] = set()
    witness_by_episode = {item.episode_id: item for item in witnesses}
    for hit in inspection.hits:
        if hit.episode_id in seen_hits:
            continue
        witness = witness_by_episode.get(hit.episode_id)
        if witness is None:
            raise ValueError("linker returned an episode outside the candidate set")
        seen_hits.add(hit.episode_id)
        score = _hit_utility(hit, score_mode=policy.score_mode)
        seeds.append(
            EpisodeSeed(
                episode_id=hit.episode_id,
                anchor_chunk_id=witness.anchor_chunk_id,
                score=score,
                route="episode_representative_qwen",
                path=(
                    witness.anchor_chunk_id,
                    hit.episode_id,
                    f"source_route:{witness.source_route}",
                    "qwen_nested_representative",
                ),
            )
        )
        if len(seeds) >= policy.top_k:
            break

    return EpisodeRepresentativeRetrievalPlan(
        artifact_id=policy.artifact_id,
        policy_sha256=policy.policy_sha256,
        query_sha256=query_sha256,
        query_input_sha256=query_input_sha256,
        linker_identity_sha256=linker_identity_sha256,
        runtime_binding_certified=runtime_binding_certified,
        source_scope_receipt_sha256=(
            None if source_scope is None else source_scope.receipt_sha256
        ),
        source_universe_exhaustive=bool(
            source_scope is not None and source_scope.selected_scope_exhaustive
        ),
        source_scans=tuple(source_scans),
        candidate_witnesses=tuple(witnesses),
        seeds=tuple(seeds),
        truncated_source_ids=truncated_source_ids,
        truncated_episode_ids=tuple(truncated_episode_ids),
        unavailable_episode_ids=tuple(unavailable_episode_ids),
        passes=inspection.passes,
        max_workspace_candidates=inspection.max_workspace_candidates,
        max_workspace_tokens=inspection.max_workspace_tokens,
        total_candidate_inspections=inspection.total_candidate_inspections,
    )


def episode_source_candidates_from_results(
    results: Sequence[RetrievalResult],
    *,
    max_sources: int,
) -> tuple[EpisodeSourceCandidate, ...]:
    """Reduce a gold-blind ranked source route to one scalar per source."""

    limit = _exact_int(max_sources, "max_sources")
    if limit < 1:
        raise ValueError("max_sources must be positive")
    selected: dict[str, EpisodeSourceCandidate] = {}
    for result in results:
        if not isinstance(result, RetrievalResult):
            raise TypeError("results must contain RetrievalResult values")
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
            raise ValueError("retrieval source identities disagree")
        source_id = (
            memory_source
            or turn_source
        )
        if source_id is None or not str(source_id).strip():
            continue
        candidate = EpisodeSourceCandidate(
            source_id=str(source_id),
            score=float(result.score),
            route=str(result.route or "source_router"),
        )
        prior = selected.get(candidate.source_id)
        if prior is None or _source_rank_key(candidate) < _source_rank_key(prior):
            selected[candidate.source_id] = candidate
    return tuple(sorted(selected.values(), key=_source_rank_key)[:limit])


def _prepare_candidate(
    episode: Episode,
    source: EpisodeSourceCandidate,
    *,
    store: EpisodeRepresentativeLookup,
    hydrate: RepresentativeHydrator,
    policy: EpisodeRepresentativeRetrievalPolicy,
) -> tuple[AssociativeMemoryCandidate, EpisodeRepresentativeWitness] | None:
    try:
        raw_representatives = store.get_representatives(episode.episode_id)
        bounded_representatives = tuple(
            islice(
                iter(raw_representatives),
                policy.max_representatives_per_episode + 1,
            )
        )
        representatives = tuple(
            islice(
                sorted(
                    (
                        item
                        for item in bounded_representatives
                        if isinstance(item, EpisodeRepresentative)
                    ),
                    key=lambda item: (item.rank, item.chunk_id),
                ),
                policy.max_representatives_per_episode,
            )
        )
    except Exception:
        return None
    evidence_chunk_ids = {item.chunk_id for item in episode.evidence}
    hydrated_parts: list[tuple[EpisodeRepresentative, str]] = []
    seen_chunks: set[str] = set()
    for representative in representatives:
        if (
            representative.episode_id != episode.episode_id
            or representative.chunk_id not in evidence_chunk_ids
            or representative.chunk_id in seen_chunks
        ):
            continue
        seen_chunks.add(representative.chunk_id)
        try:
            result = hydrate(
                representative.chunk_id,
                score=source.score,
                route="episode_representative_candidate",
            )
        except Exception:
            continue
        if not isinstance(result, RetrievalResult):
            continue
        if result.chunk.chunk_id != representative.chunk_id:
            continue
        source_hints = {
            str(value).strip()
            for value in (
                result.memory_source_id,
                None if result.turn is None else result.turn.source_id,
            )
            if value is not None and str(value).strip()
        }
        if source_hints and source_hints != {episode.source_id}:
            continue
        spans = tuple(
            item
            for item in episode.evidence
            if item.chunk_id == representative.chunk_id
        )
        if not spans or any(
            item.end_char > len(result.chunk.text)
            or quote_sha256(result.chunk.text[item.start_char : item.end_char])
            != item.quote_sha256
            for item in spans
        ):
            continue
        hydrated_parts.append((representative, result.chunk.text))
    if not hydrated_parts:
        return None

    candidate_text = truncate_to_tokens_lossless(
        "\n\n".join(
            f"[Representative {representative.rank}]\n{text}"
            for representative, text in hydrated_parts
        ),
        policy.representative_tokens,
    )
    if not candidate_text.strip():
        return None
    representative_chunk_ids = tuple(
        representative.chunk_id for representative, _ in hydrated_parts
    )
    representative_hashes = tuple(
        representative.vector_identity_sha256
        for representative, _ in hydrated_parts
    )
    witness = EpisodeRepresentativeWitness(
        episode_id=episode.episode_id,
        source_id=episode.source_id,
        anchor_chunk_id=representative_chunk_ids[0],
        representative_chunk_ids=representative_chunk_ids,
        representative_identity_sha256s=representative_hashes,
        candidate_text_sha256=identity_sha256({"text": candidate_text}),
        source_score=source.score,
        source_route=source.route,
    )
    return (
        AssociativeMemoryCandidate(
            episode_id=episode.episode_id,
            text=candidate_text,
            score=source.score,
            route="episode_representative",
            metadata={
                "artifact_id": episode.artifact_id,
                "source_id": episode.source_id,
                "anchor_chunk_id": witness.anchor_chunk_id,
                "representative_chunk_ids": witness.representative_chunk_ids,
                "representative_identity_sha256s": (
                    witness.representative_identity_sha256s
                ),
            },
        ),
        witness,
    )


def _deduplicate_sources(
    candidates: Sequence[EpisodeSourceCandidate],
) -> tuple[EpisodeSourceCandidate, ...]:
    by_source: dict[str, EpisodeSourceCandidate] = {}
    for candidate in candidates:
        if not isinstance(candidate, EpisodeSourceCandidate):
            raise TypeError("source candidates must be EpisodeSourceCandidate values")
        existing = by_source.get(candidate.source_id)
        if existing is None or _source_rank_key(candidate) < _source_rank_key(existing):
            by_source[candidate.source_id] = candidate
    return tuple(sorted(by_source.values(), key=_source_rank_key))


def _source_rank_key(item: EpisodeSourceCandidate) -> tuple[float, str, str]:
    return (-item.score, item.source_id, item.route)


def _hit_utility(hit: object, *, score_mode: str) -> float:
    qk_score = float(getattr(hit, "qk_score"))
    ov_transport = float(getattr(hit, "ov_transport"))
    if not math.isfinite(qk_score) or not math.isfinite(ov_transport):
        raise ValueError("linker hit scores must be finite")
    utility = max(0.0, qk_score)
    if score_mode == "qk_ov":
        utility += math.log1p(max(0.0, ov_transport))
        metadata = getattr(hit, "metadata", {})
        signature = metadata.get("cav_signature", ()) if isinstance(metadata, dict) else ()
        if signature:
            values = tuple(float(value) for value in signature)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("linker CAV signature must be finite")
            utility += math.log1p(
                sum(max(0.0, value) for value in values) / len(values)
            )
    return utility


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


def _digest(value: object, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _linker_identity(linker: object) -> dict[str, object]:
    """Bind the observable query-time linker without trusting injected types."""

    encoder = getattr(linker, "encoder", None)
    checkpoint = getattr(encoder, "checkpoint_identity", None)
    raw_device = getattr(encoder, "device", None)
    try:
        from memory_condense.associations.qwen_memory_linker import (
            QwenMemoryLinker,
        )
        from memory_condense.search.episodes.qwen_episode_signal import (
            _attention_head_implementation_sha256,
            _owned_qwen_runtime_binding,
        )

        inspection = getattr(linker, "inspect_nested", None)
        owned = bool(
            _owned_qwen_runtime_binding(linker)
            and getattr(inspection, "__self__", None) is linker
            and getattr(inspection, "__func__", None)
            is QwenMemoryLinker.inspect_nested
        )
        implementation_sha256 = (
            _attention_head_implementation_sha256(linker) if owned else None
        )
    except Exception:
        owned = False
        implementation_sha256 = None
    return {
        "implementation": f"{type(linker).__module__}.{type(linker).__qualname__}",
        "owned_runtime_binding": bool(owned),
        "implementation_sha256": implementation_sha256,
        "max_candidates": _exact_int(
            getattr(linker, "max_candidates", None),
            "linker.max_candidates",
        ),
        "max_workspace_tokens": getattr(linker, "max_workspace_tokens", None),
        "attention_layer": getattr(linker, "layer", None),
        "head_vote_k": getattr(linker, "head_vote_k", None),
        "model_id": getattr(checkpoint, "model_id", None),
        "model_revision": getattr(checkpoint, "model_revision", None),
        "checkpoint_sha256": getattr(checkpoint, "checkpoint_sha256", None),
        "encoder_layers": getattr(encoder, "layers", None),
        # Qwen3PrefixEncoder stores a torch.device, which is not JSON-native.
        # Its canonical string preserves the exact device/index while making
        # the runtime identity strict-JSON serializable.
        "device": None if raw_device is None else str(raw_device).strip(),
        "dtype": getattr(encoder, "dtype_name", None),
    }


__all__ = [
    "EpisodeRepresentativeLookup",
    "EpisodeRepresentativeRetrievalPlan",
    "EpisodeRepresentativeRetrievalPolicy",
    "EpisodeRepresentativeWitness",
    "EpisodeSourceCandidate",
    "EpisodeSourceCandidateScope",
    "EpisodeSourceScan",
    "NestedEpisodeLinker",
    "RepresentativeHydrator",
    "episode_source_candidates_from_results",
    "retrieve_episode_representatives",
]
