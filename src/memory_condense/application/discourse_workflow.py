"""Opt-in application composition for episodic discourse closure.

The ordinary retrieval and context-building paths do not call this mixin.
Every method is an explicit publication, expansion, closure, or packing action;
semantic/model-assisted strategies may be injected but are never selected
implicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ArtifactCoverageReceipt,
    DiscourseArtifact,
    DiscourseRelation,
    DiscourseSnapshot,
    DiscourseUnit,
    EpisodeRepresentative,
    EpisodeSeed,
    EvidenceAtom,
    EvidencePacket,
    EvidenceSpan,
    QueryProgram,
    evidence_span_sort_key,
    identity_sha256,
    make_atom_id,
)
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.ingest.discourse_linker import (
    LinkerInput,
    LinkerOutput,
    RuleBasedDiscourseLinker,
)
from memory_condense.persistence.discourse_store import (
    ArtifactCoverageMark,
    DiscourseStore,
)
from memory_condense.search.closure import close_evidence
from memory_condense.search.episodes import (
    EpisodeBuildResult,
    EpisodeBuilder,
    EpisodeRetrievalPlan,
    EpisodeRetrievalPolicy,
    SurpriseScorer,
    expand_episode_seeds,
    select_episode_representatives,
)
from memory_condense.search.packing.evidence_packet import pack_evidence_plan


@runtime_checkable
class DiscourseLinker(Protocol):
    """Injected linker seam; implementations receive only verified atoms."""

    def link(
        self,
        artifact_id: str,
        inputs: Sequence[LinkerInput],
    ) -> LinkerOutput: ...


class _ArtifactScopedDiscourseStore:
    """Fail-closed adapter preventing cross-artifact closure reads."""

    __slots__ = ("artifact_id", "store")

    def __init__(self, store: DiscourseStore, artifact_id: str) -> None:
        self.store = store
        self.artifact_id = artifact_id

    def _scope(self, requested: str | None) -> str:
        if requested is not None and requested != self.artifact_id:
            raise ValueError("closure attempted to cross discourse artifacts")
        return self.artifact_id

    def snapshot(self, graph_revision: int | None = None):
        return self.store.snapshot(graph_revision)

    def evidence_for_chunks(self, chunk_ids):
        return self.store.evidence_for_chunks(chunk_ids)

    def hydrate_span(self, span):
        return self.store.hydrate_span(span)

    def episode_ids_for_chunks(self, chunk_ids, *, artifact_id=None):
        return self.store.episode_ids_for_chunks(
            chunk_ids,
            artifact_id=self._scope(artifact_id),
        )

    def get_episode(self, episode_id):
        value = self.store.get_episode(episode_id)
        return value if value is not None and value.artifact_id == self.artifact_id else None

    def adjacent_episodes(self, episode_id, *, radius=1, include_self=False):
        seed = self.get_episode(episode_id)
        if seed is None:
            return ()
        return tuple(
            item
            for item in self.store.adjacent_episodes(
                episode_id,
                radius=radius,
                include_self=include_self,
            )
            if item.artifact_id == self.artifact_id
        )

    def episodes_for_source(
        self,
        artifact_id,
        source_id,
        *,
        start_sequence=None,
        end_sequence=None,
        limit=None,
    ):
        return self.store.episodes_for_source(
            self._scope(artifact_id),
            source_id,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            limit=limit,
        )

    def units_for_chunks(self, chunk_ids, *, artifact_id=None, limit=None):
        return self.store.units_for_chunks(
            chunk_ids,
            artifact_id=self._scope(artifact_id),
            limit=limit,
        )

    def units_for_artifact(self, artifact_id, *, limit=None):
        return self.store.units_for_artifact(
            self._scope(artifact_id),
            limit=limit,
        )

    def relations_for_chunks(self, chunk_ids, *, artifact_id=None, limit=None):
        return self.store.relations_for_chunks(
            chunk_ids,
            artifact_id=self._scope(artifact_id),
            limit=limit,
        )

    def get_unit(self, unit_id):
        value = self.store.get_unit(unit_id)
        return value if value is not None and value.artifact_id == self.artifact_id else None

    def get_relation(self, relation_id):
        value = self.store.get_relation(relation_id)
        return value if value is not None and value.artifact_id == self.artifact_id else None

    def incident_relations(
        self,
        unit_ids,
        *,
        artifact_id=None,
        max_degree,
    ):
        return self.store.incident_relations(
            unit_ids,
            artifact_id=self._scope(artifact_id),
            max_degree=max_degree,
        )

    def coverage_for_chunks(
        self,
        artifact_id,
        chunk_ids,
        *,
        coverage_kind="discourse",
    ):
        return self.store.coverage_for_chunks(
            self._scope(artifact_id),
            chunk_ids,
            coverage_kind=coverage_kind,
        )

    def artifact_coverage(self, artifact_id, coverage_kind="discourse"):
        return self.store.artifact_coverage(
            self._scope(artifact_id),
            coverage_kind,
        )


@dataclass(frozen=True, slots=True)
class EpisodePublication:
    """One atomic episode publication plus its immutable graph receipt."""

    build: EpisodeBuildResult
    representatives: tuple[EpisodeRepresentative, ...]
    snapshot: DiscourseSnapshot
    retained_request_token_state_bytes: int = 0

    def __post_init__(self) -> None:
        if self.retained_request_token_state_bytes != 0:
            raise ValueError("episode publication cannot retain request token state")


@dataclass(frozen=True, slots=True)
class LinkPublication:
    """One atomic discourse-link publication and its graph receipt."""

    output: LinkerOutput
    snapshot: DiscourseSnapshot


def _span_identity(span: EvidenceSpan) -> str:
    return identity_sha256(span.identity_payload())


def _aligned_scores(
    scores: Mapping[str, float] | Sequence[float] | None,
    *,
    input_spans: Sequence[EvidenceSpan],
    ordered_spans: Sequence[EvidenceSpan],
) -> tuple[float, ...] | None:
    if scores is None:
        return None
    if isinstance(scores, Mapping):
        missing = [span.chunk_id for span in ordered_spans if span.chunk_id not in scores]
        if missing:
            raise ValueError(
                "surprise_scores is missing chunk IDs: " + ", ".join(missing)
            )
        return tuple(float(scores[span.chunk_id]) for span in ordered_spans)
    rows = tuple(float(value) for value in scores)
    if len(rows) != len(input_spans):
        raise ValueError(
            "sequence surprise_scores must align with the deduplicated input chunks"
        )
    by_span = {
        _span_identity(span): value
        for span, value in zip(input_spans, rows, strict=True)
    }
    return tuple(by_span[_span_identity(span)] for span in ordered_spans)


def _validate_episode_build(
    result: EpisodeBuildResult,
    *,
    artifact: DiscourseArtifact,
    source_id: str,
    spans: Sequence[EvidenceSpan],
) -> None:
    if not isinstance(result, EpisodeBuildResult):
        raise TypeError("episode builder must return EpisodeBuildResult")
    if result.artifact_id != artifact.artifact_id or result.source_id != source_id:
        raise ValueError("episode builder returned another source or artifact")
    emitted = tuple(span for episode in result.episodes for span in episode.evidence)
    if tuple(map(_span_identity, emitted)) != tuple(map(_span_identity, spans)):
        raise ValueError("episode builder must partition every supplied span exactly once")


class DiscourseWorkflowMixin:
    """Explicit, provider-free discourse workflows for ``MemoryCondenser``."""

    def _init_discourse_workflow(self) -> None:
        self._discourse = DiscourseStore(self._db)

    @property
    def discourse(self) -> DiscourseStore:
        """Access the opt-in, source-grounded discourse repository."""

        return self._discourse

    def build_and_publish_discourse_episodes(
        self,
        artifact: DiscourseArtifact,
        chunk_ids: Sequence[str],
        *,
        source_id: str | None = None,
        builder: EpisodeBuilder | None = None,
        sequence_start: int = 0,
        surprise_scores: Mapping[str, float] | Sequence[float] | None = None,
        surprise_scorer: SurpriseScorer | None = None,
        embeddings: Mapping[str, Sequence[float]] | None = None,
        representative_limit: int = 2,
    ) -> EpisodePublication:
        """Build and atomically publish episodes from exact stored chunks."""

        input_spans = self._discourse.evidence_for_chunks(chunk_ids)
        if not input_spans:
            raise ValueError("at least one chunk_id is required")
        spans = tuple(sorted(input_spans, key=evidence_span_sort_key))
        sources = {span.source_id for span in spans}
        if None in sources or len(sources) != 1:
            raise ValueError("episode formation requires one authoritative source")
        authoritative_source = str(next(iter(sources)))
        if source_id is not None:
            requested_source = str(source_id).strip()
            if not requested_source:
                raise ValueError("source_id must be non-empty when supplied")
            if requested_source != authoritative_source:
                raise ValueError("source_id does not match the authoritative chunks")

        texts_by_chunk = {
            span.chunk_id: self._discourse.hydrate_span(span) for span in spans
        }
        ordered_embeddings = (
            None
            if embeddings is None
            else tuple(embeddings.get(span.chunk_id) for span in spans)
        )
        build = (builder or EpisodeBuilder()).build(
            source_id=authoritative_source,
            artifact_id=artifact.artifact_id,
            spans=spans,
            texts=tuple(texts_by_chunk[span.chunk_id] for span in spans),
            embeddings=ordered_embeddings,
            surprise_scores=_aligned_scores(
                surprise_scores,
                input_spans=input_spans,
                ordered_spans=spans,
            ),
            surprise_scorer=surprise_scorer,
            sequence_start=sequence_start,
        )
        _validate_episode_build(
            build,
            artifact=artifact,
            source_id=authoritative_source,
            spans=spans,
        )
        representatives = tuple(
            representative
            for episode in build.episodes
            for representative in select_episode_representatives(
                episode,
                limit=representative_limit,
                texts=texts_by_chunk,
                embeddings=embeddings,
            )
        )
        snapshot = self._discourse.publish(
            artifact,
            episodes=build.episodes,
            representatives=representatives,
            coverage=tuple(
                ArtifactCoverageMark(span.chunk_id, "episode", "annotated")
                for span in spans
            ),
        )
        return EpisodePublication(
            build=build,
            representatives=representatives,
            snapshot=snapshot,
        )

    def link_and_publish_discourse(
        self,
        artifact: DiscourseArtifact,
        *,
        chunk_ids: Sequence[str] = (),
        inputs: Sequence[LinkerInput] = (),
        linker: DiscourseLinker | None = None,
    ) -> LinkPublication:
        """Link verified evidence and atomically publish the resulting graph."""

        supplied_inputs = tuple(inputs)
        if bool(chunk_ids) == bool(supplied_inputs):
            raise ValueError("supply exactly one of chunk_ids or inputs")
        if chunk_ids:
            spans = tuple(
                sorted(
                    self._discourse.evidence_for_chunks(chunk_ids),
                    key=evidence_span_sort_key,
                )
            )
            episode_ids = self._discourse.episode_ids_for_chunks(
                tuple(span.chunk_id for span in spans),
                artifact_id=artifact.artifact_id,
            )
            verified_inputs = tuple(
                LinkerInput(
                    atom=EvidenceAtom(
                        atom_id=make_atom_id(span),
                        span=span,
                        text=self._discourse.hydrate_span(span),
                        label="direct_chunk",
                        role=span.role,
                        created_at=span.created_at,
                    ),
                    episode_id=episode_ids.get(span.chunk_id),
                )
                for span in spans
            )
        else:
            verified_inputs = self._verify_linker_inputs(
                artifact.artifact_id,
                supplied_inputs,
            )

        output = (linker or RuleBasedDiscourseLinker()).link(
            artifact.artifact_id,
            verified_inputs,
        )
        if not isinstance(output, LinkerOutput):
            raise TypeError("discourse linker must return LinkerOutput")
        self._validate_linker_output(artifact, verified_inputs, output)
        annotated_chunks = {
            span.chunk_id
            for owner in (*output.units, *output.relations)
            for span in owner.evidence
        }
        snapshot = self._discourse.publish(
            artifact,
            units=output.units,
            relations=output.relations,
            coverage=tuple(
                ArtifactCoverageMark(
                    item.atom.span.chunk_id,
                    "discourse",
                    (
                        "annotated"
                        if item.atom.span.chunk_id in annotated_chunks
                        else "no_output"
                    ),
                )
                for item in verified_inputs
            ),
        )
        return LinkPublication(output=output, snapshot=snapshot)

    def _verify_linker_inputs(
        self,
        artifact_id: str,
        inputs: Sequence[LinkerInput],
    ) -> tuple[LinkerInput, ...]:
        seen: set[str] = set()
        verified: list[LinkerInput] = []
        for item in inputs:
            if not isinstance(item, LinkerInput):
                raise TypeError("inputs must contain LinkerInput values")
            span_identity = _span_identity(item.atom.span)
            if span_identity in seen:
                raise ValueError("linker inputs cannot repeat an evidence span")
            seen.add(span_identity)
            if any(
                getattr(item.atom.span, name) is None
                for name in ("source_id", "turn_id", "role", "created_at")
            ):
                raise ValueError(
                    "linker inputs require complete authoritative provenance"
                )
            if self._discourse.hydrate_span(item.atom.span) != item.atom.text:
                raise ValueError("linker atom text is not the exact stored evidence")
            if item.episode_id is not None:
                episode = self._discourse.get_episode(item.episode_id)
                if episode is None:
                    raise KeyError(f"unknown episode: {item.episode_id}")
                if episode.artifact_id != artifact_id:
                    raise ValueError("linker input episode belongs to another artifact")
                if span_identity not in {
                    _span_identity(span) for span in episode.evidence
                }:
                    raise ValueError("linker input episode does not contain its evidence")
            verified.append(item)
        return tuple(
            sorted(
                verified,
                key=lambda item: (
                    *evidence_span_sort_key(item.atom.span),
                    item.atom.atom_id,
                    item.episode_id or "",
                ),
            )
        )

    @staticmethod
    def _validate_linker_output(
        artifact: DiscourseArtifact,
        inputs: Sequence[LinkerInput],
        output: LinkerOutput,
    ) -> None:
        allowed_evidence = {_span_identity(item.atom.span) for item in inputs}
        if any(unit.artifact_id != artifact.artifact_id for unit in output.units):
            raise ValueError("linker returned a unit for another artifact")
        if any(
            relation.artifact_id != artifact.artifact_id
            for relation in output.relations
        ):
            raise ValueError("linker returned a relation for another artifact")
        cited = (
            span
            for owner in (*output.units, *output.relations)
            for span in owner.evidence
        )
        if any(_span_identity(span) not in allowed_evidence for span in cited):
            raise ValueError("linker output cites evidence outside its verified inputs")
        emitted_unit_ids = {unit.unit_id for unit in output.units}
        if any(
            member.unit_id not in emitted_unit_ids
            for relation in output.relations
            for member in relation.members
        ):
            raise ValueError("linker relations must reference units in the same output")

    def publish_discourse_graph(
        self,
        artifact: DiscourseArtifact,
        *,
        units: Sequence[DiscourseUnit] = (),
        relations: Sequence[DiscourseRelation] = (),
    ) -> DiscourseSnapshot:
        """Publish caller-constructed, source-validated graph objects."""

        return self._discourse.publish(
            artifact,
            units=units,
            relations=relations,
        )

    def finalize_discourse_coverage(
        self,
        artifact_id: str,
    ) -> ArtifactCoverageReceipt:
        """Assert full-corpus discourse processing after the store verifies it."""

        return self._discourse.finalize_artifact_coverage(
            artifact_id,
            coverage_kind="discourse",
        )

    def expand_discourse_episode_seeds(
        self,
        results: Sequence[RetrievalResult],
        *,
        policy: EpisodeRetrievalPolicy | None = None,
    ) -> EpisodeRetrievalPlan:
        """Bridge direct retrieval hits to episode seeds with raw fail-open IDs."""

        return expand_episode_seeds(results, self._discourse, policy=policy)

    def close_discourse_evidence(
        self,
        query: str | QueryProgram | None = None,
        *,
        query_program: QueryProgram | None = None,
        seeds: Sequence[EpisodeSeed] = (),
        direct_chunk_ids: Sequence[str] = (),
        policy: ClosurePolicy | None = None,
        artifact_id: str | None = None,
        expansion_receipt_sha256: str | None = None,
        expansion_exhaustive: bool | None = None,
    ) -> ClosurePlan:
        """Retrieve bounded evidence closure only from caller-provided seeds."""

        selected_artifact = self._resolve_discourse_artifact(artifact_id)
        store = (
            self._discourse
            if selected_artifact is None
            else _ArtifactScopedDiscourseStore(
                self._discourse,
                selected_artifact,
            )
        )
        return close_evidence(
            store,
            query,
            query_program=query_program,
            seeds=seeds,
            direct_chunk_ids=direct_chunk_ids,
            policy=policy,
            artifact_id=selected_artifact,
            expansion_receipt_sha256=expansion_receipt_sha256,
            expansion_exhaustive=expansion_exhaustive,
        )

    def _resolve_discourse_artifact(self, artifact_id: str | None) -> str | None:
        available = self._discourse.snapshot().artifact_ids
        if artifact_id is not None:
            selected = str(artifact_id).strip()
            if not selected:
                raise ValueError("artifact_id must be non-empty when supplied")
            if selected not in available:
                raise KeyError(f"unknown discourse artifact: {selected}")
            return selected
        if len(available) == 1:
            return available[0]
        if len(available) > 1:
            raise ValueError(
                "artifact_id is required when multiple discourse artifacts exist"
            )
        return None

    def retrieve_close_discourse_evidence(
        self,
        results: Sequence[RetrievalResult],
        *,
        artifact_id: str,
        query: str | QueryProgram | None = None,
        query_program: QueryProgram | None = None,
        episode_policy: EpisodeRetrievalPolicy | None = None,
        closure_policy: ClosurePolicy | None = None,
    ) -> ClosurePlan:
        """Safely expand retrieval rows and preserve every raw fallback."""

        selected = str(artifact_id).strip()
        if not selected:
            raise ValueError("artifact_id is required")
        if episode_policy is not None and episode_policy.artifact_id not in (
            None,
            selected,
        ):
            raise ValueError("episode policy belongs to another artifact")
        active_episode_policy = episode_policy or EpisodeRetrievalPolicy(
            artifact_id=selected
        )
        if active_episode_policy.artifact_id is None:
            active_episode_policy = EpisodeRetrievalPolicy(
                artifact_id=selected,
                max_anchor_episodes=active_episode_policy.max_anchor_episodes,
                previous_episodes=active_episode_policy.previous_episodes,
                next_episodes=active_episode_policy.next_episodes,
                max_episode_seeds=active_episode_policy.max_episode_seeds,
                max_direct_fallbacks=active_episode_policy.max_direct_fallbacks,
                neighbor_decay=active_episode_policy.neighbor_decay,
            )
        expansion = self.expand_discourse_episode_seeds(
            results,
            policy=active_episode_policy,
        )
        return self.close_discourse_evidence(
            query,
            query_program=query_program,
            seeds=expansion.seeds,
            direct_chunk_ids=expansion.direct_chunk_ids,
            policy=closure_policy,
            artifact_id=selected,
            expansion_receipt_sha256=expansion.receipt_sha256,
            expansion_exhaustive=not (
                expansion.truncated_episode_ids
                or expansion.truncated_direct_chunk_ids
            ),
        )

    def pack_discourse_evidence(
        self,
        plan: ClosurePlan,
        *,
        max_context_tokens: int,
        encoding: str = "cl100k_base",
        base_messages: Sequence[Mapping[str, str]] | None = None,
        evidence_message_role: str = "user",
        evidence_prefix: str = "",
        evidence_suffix: str = "",
        max_prompt_tokens: int | None = None,
        output_token_reserve: int = 0,
    ) -> EvidencePacket:
        """Atomically pack whole evidence bundles under the caller's budgets."""

        return pack_evidence_plan(
            plan,
            max_context_tokens=max_context_tokens,
            encoding=encoding,
            base_messages=base_messages,
            evidence_message_role=evidence_message_role,
            evidence_prefix=evidence_prefix,
            evidence_suffix=evidence_suffix,
            max_prompt_tokens=max_prompt_tokens,
            output_token_reserve=output_token_reserve,
        )


__all__ = [
    "DiscourseLinker",
    "DiscourseWorkflowMixin",
    "EpisodePublication",
    "LinkPublication",
]
