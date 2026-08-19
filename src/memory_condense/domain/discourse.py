"""Immutable contracts for episodic discourse closure.

These values deliberately contain only source references, scalar routing
metadata, and transient hydrated evidence text.  They cannot hold token IDs,
K/V caches, activations, or another request-shaped transformer state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._discourse_identity import (
    _confidence,
    _json_mapping,
    _nonempty,
    _plain_json,
    _sha256,
    canonical_json,
    identity_sha256,
    make_atom_id,
    make_bundle_id,
    make_episode_id,
    quote_sha256,
)


_TEMPORAL_STANCES = frozenset(
    {"any", "latest", "terminal", "ordered", "ascending", "descending"}
)
ClosureStatus = Literal[
    "satisfied",
    "not_found",
    "conflicted",
    "budget_impossible",
]
ClosureStopReason = Literal[
    "complete",
    "frontier_exhausted",
    "budget_exhausted",
    "budget_impossible",
    "workspace_cap",
    "conflicted",
    "not_found",
]


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """An exact, independently verifiable substring of one authoritative chunk."""

    chunk_id: str
    start_char: int
    end_char: int
    quote_sha256: str
    ordinal: int
    source_id: str | None = None
    # Absolute start of the owning chunk inside its authoritative turn.  This
    # disambiguates multiple chunks with the same turn ordinal; ``start_char``
    # remains relative to the chunk itself.  Zero preserves compatibility for
    # the common full-turn/first-chunk case.
    turn_start_char: int = 0
    turn_id: str | None = None
    role: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _nonempty(self.chunk_id, "chunk_id"))
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError("evidence span must have 0 <= start_char < end_char")
        if self.ordinal < 0:
            raise ValueError("evidence span ordinal must be non-negative")
        if self.turn_start_char < 0:
            raise ValueError("turn_start_char must be non-negative")
        object.__setattr__(
            self,
            "quote_sha256",
            _sha256(self.quote_sha256, "quote_sha256"),
        )
        if self.source_id is not None:
            object.__setattr__(
                self,
                "source_id",
                _nonempty(self.source_id, "source_id"),
            )
        if self.turn_id is not None:
            object.__setattr__(self, "turn_id", _nonempty(self.turn_id, "turn_id"))
        if self.role is not None:
            role = _nonempty(self.role, "role")
            if role not in {"user", "assistant", "system"}:
                raise ValueError("evidence role must be user, assistant, or system")
            object.__setattr__(self, "role", role)
        if self.created_at is not None:
            object.__setattr__(
                self,
                "created_at",
                _nonempty(self.created_at, "created_at"),
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote_sha256": self.quote_sha256,
            "ordinal": self.ordinal,
            "source_id": self.source_id,
            "turn_start_char": self.turn_start_char,
            "turn_id": self.turn_id,
            "role": self.role,
            "created_at": self.created_at,
        }


def evidence_span_sort_key(
    span: EvidenceSpan,
) -> tuple[int, str, int, int, int, str, str]:
    """Return the authoritative, total source order for exact evidence.

    Turn ordinals are global.  ``source_id`` is therefore only a deterministic
    tie-breaker, while ``turn_start_char`` orders chunks cut from the same turn
    and ``start_char`` orders sub-spans inside one chunk.
    """
    return (
        span.ordinal,
        span.source_id or "",
        span.turn_start_char,
        span.start_char,
        span.end_char,
        span.chunk_id,
        span.quote_sha256,
    )


@dataclass(frozen=True, slots=True)
class DiscourseArtifact:
    """Identity of one deterministic or model-assisted annotation procedure."""

    artifact_id: str
    kind: str
    implementation_sha256: str
    policy_sha256: str
    model_id: str | None = None
    model_revision: str | None = None
    checkpoint_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "kind", _nonempty(self.kind, "artifact kind"))
        object.__setattr__(
            self,
            "implementation_sha256",
            _sha256(self.implementation_sha256, "implementation_sha256"),
        )
        object.__setattr__(
            self,
            "policy_sha256",
            _sha256(self.policy_sha256, "policy_sha256"),
        )
        if self.checkpoint_sha256 is not None:
            object.__setattr__(
                self,
                "checkpoint_sha256",
                _sha256(self.checkpoint_sha256, "checkpoint_sha256"),
            )
        for name in ("model_id", "model_revision"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        implementation_sha256: str,
        policy: Mapping[str, Any],
        model_id: str | None = None,
        model_revision: str | None = None,
        checkpoint_sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DiscourseArtifact:
        normalized_kind = _nonempty(kind, "artifact kind")
        normalized_implementation = _sha256(
            implementation_sha256,
            "implementation_sha256",
        )
        normalized_model_id = (
            None if model_id is None else _nonempty(model_id, "model_id")
        )
        normalized_model_revision = (
            None
            if model_revision is None
            else _nonempty(model_revision, "model_revision")
        )
        normalized_checkpoint = (
            None
            if checkpoint_sha256 is None
            else _sha256(checkpoint_sha256, "checkpoint_sha256")
        )
        policy_body = _json_mapping(policy, "policy")
        metadata_body = _json_mapping(metadata or {}, "metadata")
        policy_digest = identity_sha256(policy_body)
        body = {
            "kind": normalized_kind,
            "implementation_sha256": normalized_implementation,
            "policy_sha256": policy_digest,
            "model_id": normalized_model_id,
            "model_revision": normalized_model_revision,
            "checkpoint_sha256": normalized_checkpoint,
            "metadata": _plain_json(metadata_body),
        }
        return cls(artifact_id=f"disc-{identity_sha256(body)[:24]}", **body)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "implementation_sha256": self.implementation_sha256,
            "policy_sha256": self.policy_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "metadata": _plain_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Episode:
    """A source-local, ordered event backed only by exact source spans."""

    episode_id: str
    artifact_id: str
    source_id: str
    sequence_no: int
    first_ordinal: int
    last_ordinal: int
    evidence: tuple[EvidenceSpan, ...]
    boundary_method: str
    initial_boundary: int | None = None
    refined_boundary: int | None = None
    boundary_score: float | None = None
    boundary_threshold: float | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _nonempty(self.episode_id, "episode_id"))
        object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "source_id", _nonempty(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "boundary_method",
            _nonempty(self.boundary_method, "boundary_method"),
        )
        evidence = tuple(self.evidence)
        if self.sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
        if self.first_ordinal < 0 or self.last_ordinal < self.first_ordinal:
            raise ValueError("episode ordinal range is invalid")
        if not evidence:
            raise ValueError("episode evidence must be non-empty")
        if tuple(sorted(evidence, key=evidence_span_sort_key)) != evidence:
            raise ValueError("episode evidence must be in deterministic source order")
        if evidence[0].ordinal != self.first_ordinal or evidence[-1].ordinal != self.last_ordinal:
            raise ValueError("episode ordinal bounds must match its evidence")
        if any(item.source_id not in (None, self.source_id) for item in evidence):
            raise ValueError("an episode cannot cross source histories")
        for name in ("boundary_score", "boundary_threshold"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "evidence", evidence)
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256:
            if _sha256(self.receipt_sha256, "receipt_sha256") != expected:
                raise ValueError("episode receipt does not match its identity payload")
        else:
            object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "episode_id": self.episode_id,
            "artifact_id": self.artifact_id,
            "source_id": self.source_id,
            "sequence_no": self.sequence_no,
            "first_ordinal": self.first_ordinal,
            "last_ordinal": self.last_ordinal,
            "evidence": [item.identity_payload() for item in self.evidence],
            "boundary_method": self.boundary_method,
            "initial_boundary": self.initial_boundary,
            "refined_boundary": self.refined_boundary,
            "boundary_score": self.boundary_score,
            "boundary_threshold": self.boundary_threshold,
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class EpisodeRepresentative:
    episode_id: str
    chunk_id: str
    rank: int
    vector_identity_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _nonempty(self.episode_id, "episode_id"))
        object.__setattr__(self, "chunk_id", _nonempty(self.chunk_id, "chunk_id"))
        if self.rank < 0:
            raise ValueError("representative rank must be non-negative")
        object.__setattr__(
            self,
            "vector_identity_sha256",
            _sha256(self.vector_identity_sha256, "vector_identity_sha256"),
        )


@dataclass(frozen=True, slots=True)
class DiscourseUnit:
    """A typed routing unit whose factual content remains in raw evidence."""

    unit_id: str
    artifact_id: str
    kind: str
    canonical_key: str
    asserted_ordinal: int
    confidence: float
    evidence: tuple[EvidenceSpan, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("unit_id", "artifact_id", "kind", "canonical_key"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.asserted_ordinal < 0:
            raise ValueError("asserted_ordinal must be non-negative")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        evidence = tuple(sorted(self.evidence, key=evidence_span_sort_key))
        if not evidence:
            raise ValueError("a discourse unit requires source evidence")
        if self.asserted_ordinal != max(item.ordinal for item in evidence):
            raise ValueError(
                "asserted_ordinal must equal the latest cited evidence ordinal"
            )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class RelationMember:
    unit_id: str
    role: str
    ordinal: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_id", _nonempty(self.unit_id, "unit_id"))
        object.__setattr__(self, "role", _nonempty(self.role, "relation member role"))
        if self.ordinal < 0:
            raise ValueError("relation member ordinal must be non-negative")
        if not math.isfinite(float(self.weight)) or self.weight < 0:
            raise ValueError("relation member weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DiscourseRelation:
    """An evidenced, possibly n-ary relationship among discourse units."""

    relation_id: str
    artifact_id: str
    relation_type: str
    members: tuple[RelationMember, ...]
    evidence: tuple[EvidenceSpan, ...]
    confidence: float
    created_ordinal: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("relation_id", "artifact_id", "relation_type"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        members = tuple(
            sorted(
                self.members,
                key=lambda item: (item.ordinal, item.unit_id, item.role),
            )
        )
        if len(members) < 2:
            raise ValueError("a discourse relation requires at least two members")
        if len({(item.unit_id, item.role, item.ordinal) for item in members}) != len(members):
            raise ValueError("relation members must be unique")
        if tuple(item.ordinal for item in members) != tuple(range(len(members))):
            raise ValueError("relation member ordinals must be contiguous from zero")
        evidence = tuple(sorted(self.evidence, key=evidence_span_sort_key))
        if not evidence:
            raise ValueError("a discourse relation requires exact source evidence")
        if self.created_ordinal < 0:
            raise ValueError("created_ordinal must be non-negative")
        if self.created_ordinal != max(item.ordinal for item in evidence):
            raise ValueError(
                "created_ordinal must equal the latest cited evidence ordinal"
            )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class DiscourseSnapshot:
    max_turn_ordinal: int
    chunk_count: int
    graph_revision: int
    schema_version: int
    artifact_ids: tuple[str, ...]
    source_revision: int = 0
    graph_content_revision: int = 0
    source_content_sha256: str = "0" * 64
    graph_content_sha256: str = "0" * 64
    snapshot_sha256: str = ""

    def __post_init__(self) -> None:
        if min(
            self.max_turn_ordinal,
            self.chunk_count,
            self.graph_revision,
            self.schema_version,
            self.source_revision,
            self.graph_content_revision,
        ) < 0:
            raise ValueError("snapshot counters must be non-negative")
        artifacts = tuple(sorted({_nonempty(item, "artifact_id") for item in self.artifact_ids}))
        object.__setattr__(self, "artifact_ids", artifacts)
        object.__setattr__(
            self,
            "source_content_sha256",
            _sha256(self.source_content_sha256, "source_content_sha256"),
        )
        object.__setattr__(
            self,
            "graph_content_sha256",
            _sha256(self.graph_content_sha256, "graph_content_sha256"),
        )
        zero = "0" * 64
        if (
            self.max_turn_ordinal or self.chunk_count or self.source_revision
        ) and self.source_content_sha256 == zero:
            raise ValueError("non-empty source snapshot requires a content root")
        if (
            artifacts or self.graph_content_revision
        ) and self.graph_content_sha256 == zero:
            raise ValueError("non-empty graph snapshot requires a content root")
        body = {
            "max_turn_ordinal": self.max_turn_ordinal,
            "chunk_count": self.chunk_count,
            "graph_revision": self.graph_revision,
            "schema_version": self.schema_version,
            "artifact_ids": list(artifacts),
            "source_revision": self.source_revision,
            "graph_content_revision": self.graph_content_revision,
            "source_content_sha256": self.source_content_sha256,
            "graph_content_sha256": self.graph_content_sha256,
        }
        expected = identity_sha256(body)
        if self.snapshot_sha256:
            if _sha256(self.snapshot_sha256, "snapshot_sha256") != expected:
                raise ValueError("snapshot SHA-256 does not match its counters")
        else:
            object.__setattr__(self, "snapshot_sha256", expected)


@dataclass(frozen=True, slots=True)
class ArtifactCoverageReceipt:
    artifact_id: str
    coverage_kind: str
    source_revision: int
    chunk_count: int
    coverage_sha256: str
    turn_coverage_sha256: str = "0" * 64
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        if self.coverage_kind not in {"episode", "discourse"}:
            raise ValueError("coverage_kind must be episode or discourse")
        if min(self.source_revision, self.chunk_count) < 0:
            raise ValueError("coverage receipt counters must be non-negative")
        object.__setattr__(
            self,
            "coverage_sha256",
            _sha256(self.coverage_sha256, "coverage_sha256"),
        )
        object.__setattr__(
            self,
            "turn_coverage_sha256",
            _sha256(self.turn_coverage_sha256, "turn_coverage_sha256"),
        )
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256:
            if _sha256(self.receipt_sha256, "receipt_sha256") != expected:
                raise ValueError("coverage receipt SHA-256 does not match its contents")
        else:
            object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "artifact_id": self.artifact_id,
            "coverage_kind": self.coverage_kind,
            "source_revision": self.source_revision,
            "chunk_count": self.chunk_count,
            "coverage_sha256": self.coverage_sha256,
            "turn_coverage_sha256": self.turn_coverage_sha256,
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceObligation:
    obligation_id: str
    kind: str
    required: bool
    weight: float
    unit_kinds: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    subject_terms: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    min_count: int = 1
    max_count: int | None = None
    temporal_stance: str = "any"

    def __post_init__(self) -> None:
        for name in ("obligation_id", "kind", "temporal_stance"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.temporal_stance not in _TEMPORAL_STANCES:
            raise ValueError("temporal_stance is not a supported closed value")
        if not math.isfinite(float(self.weight)) or self.weight <= 0:
            raise ValueError("obligation weight must be finite and positive")
        if self.min_count < 1:
            raise ValueError("obligation min_count must be positive")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("obligation max_count cannot be below min_count")
        for name in ("unit_kinds", "relation_types", "subject_terms", "dependencies"):
            values = tuple(dict.fromkeys(_nonempty(value, name) for value in getattr(self, name)))
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class QueryProgram:
    query: str
    intent: str
    subject_terms: tuple[str, ...]
    obligations: tuple[EvidenceObligation, ...]
    as_of_ordinal: int | None = None
    ordering: str = "none"
    cardinality: int | None = None
    program_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _nonempty(self.query, "query"))
        object.__setattr__(self, "intent", _nonempty(self.intent, "intent"))
        object.__setattr__(self, "ordering", _nonempty(self.ordering, "ordering"))
        subjects = tuple(dict.fromkeys(_nonempty(item, "subject term") for item in self.subject_terms))
        obligations = tuple(self.obligations)
        if not obligations:
            raise ValueError("a query program requires at least one obligation")
        if len({item.obligation_id for item in obligations}) != len(obligations):
            raise ValueError("obligation IDs must be unique")
        known = {item.obligation_id for item in obligations}
        if any(dep not in known for item in obligations for dep in item.dependencies):
            raise ValueError("obligation dependency references an unknown obligation")
        dependencies = {
            item.obligation_id: item.dependencies for item in obligations
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(obligation_id: str) -> None:
            if obligation_id in visiting:
                raise ValueError("obligation dependencies must form an acyclic graph")
            if obligation_id in visited:
                return
            visiting.add(obligation_id)
            for dependency in dependencies[obligation_id]:
                visit(dependency)
            visiting.remove(obligation_id)
            visited.add(obligation_id)

        for obligation_id in dependencies:
            visit(obligation_id)
        if self.as_of_ordinal is not None and self.as_of_ordinal < 0:
            raise ValueError("as_of_ordinal must be non-negative")
        if self.cardinality is not None and self.cardinality < 1:
            raise ValueError("cardinality must be positive")
        object.__setattr__(self, "subject_terms", subjects)
        object.__setattr__(self, "obligations", obligations)
        body = self.identity_payload(include_sha=False)
        expected = identity_sha256(body)
        if self.program_sha256:
            if _sha256(self.program_sha256, "program_sha256") != expected:
                raise ValueError("query program SHA-256 does not match its contents")
        else:
            object.__setattr__(self, "program_sha256", expected)

    def identity_payload(self, *, include_sha: bool = True) -> dict[str, Any]:
        payload = {
            "query": self.query,
            "intent": self.intent,
            "subject_terms": list(self.subject_terms),
            "obligations": [
                {
                    "obligation_id": item.obligation_id,
                    "kind": item.kind,
                    "required": item.required,
                    "weight": item.weight,
                    "unit_kinds": list(item.unit_kinds),
                    "relation_types": list(item.relation_types),
                    "subject_terms": list(item.subject_terms),
                    "dependencies": list(item.dependencies),
                    "min_count": item.min_count,
                    "max_count": item.max_count,
                    "temporal_stance": item.temporal_stance,
                }
                for item in self.obligations
            ],
            "as_of_ordinal": self.as_of_ordinal,
            "ordering": self.ordering,
            "cardinality": self.cardinality,
        }
        if include_sha:
            payload["program_sha256"] = self.program_sha256
        return payload


@dataclass(frozen=True, slots=True)
class ClosurePolicy:
    max_hops: int = 3
    max_units: int = 96
    max_relations: int = 192
    max_degree: int = 16
    max_episode_neighbors: int = 2
    max_frontier: int = 256
    max_bundles: int = 64
    beam_width: int = 128
    min_relation_confidence: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "max_hops",
            "max_units",
            "max_relations",
            "max_degree",
            "max_frontier",
            "max_bundles",
            "beam_width",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_episode_neighbors < 0:
            raise ValueError("max_episode_neighbors must be non-negative")
        object.__setattr__(
            self,
            "min_relation_confidence",
            _confidence(self.min_relation_confidence, "min_relation_confidence"),
        )

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class EpisodeSeed:
    episode_id: str
    anchor_chunk_id: str
    score: float
    route: str
    path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("episode_id", "anchor_chunk_id", "route"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if not math.isfinite(float(self.score)):
            raise ValueError("seed score must be finite")
        object.__setattr__(self, "path", tuple(self.path))


@dataclass(frozen=True, slots=True)
class EvidenceAtom:
    """One verified source span hydrated transiently for final packing."""

    atom_id: str
    span: EvidenceSpan
    text: str
    label: str
    role: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_id", _nonempty(self.atom_id, "atom_id"))
        # Evidence bytes are authoritative.  Validate with ``strip`` but never
        # normalize the stored value before comparing its source-span digest.
        text = str(self.text)
        if not text.strip():
            raise ValueError("atom text must be non-empty")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "label", _nonempty(self.label, "atom label"))
        if quote_sha256(self.text) != self.span.quote_sha256:
            raise ValueError("hydrated atom text does not match its source-span hash")
        role = self.role
        if self.span.role is not None:
            if role is not None and role != self.span.role:
                raise ValueError("evidence atom role contradicts its source span")
            role = self.span.role
        if role is not None:
            role = _nonempty(role, "atom role")
            if role not in {"user", "assistant", "system"}:
                raise ValueError("atom role must be user, assistant, or system")
        created_at = self.created_at
        if self.span.created_at is not None:
            if created_at is not None and created_at != self.span.created_at:
                raise ValueError("evidence atom created_at contradicts its source span")
            created_at = self.span.created_at
        if created_at is not None:
            created_at = _nonempty(created_at, "atom created_at")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "created_at", created_at)


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    bundle_id: str
    atom_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    unit_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    required: bool = False
    utility: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _nonempty(self.bundle_id, "bundle_id"))
        for name in ("atom_ids", "obligation_ids", "unit_ids", "relation_ids"):
            values = tuple(dict.fromkeys(_nonempty(item, name) for item in getattr(self, name)))
            object.__setattr__(self, name, values)
        if not self.atom_ids:
            raise ValueError("an evidence bundle must contain at least one atom")
        if not math.isfinite(float(self.utility)):
            raise ValueError("bundle utility must be finite")


@dataclass(frozen=True, slots=True)
class ObligationResult:
    obligation_id: str
    status: ClosureStatus
    unit_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    bundle_ids: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _nonempty(self.obligation_id, "obligation_id"))


@dataclass(frozen=True, slots=True)
class ClosureScopeWitness:
    """Receipt for one bounded graph read used by a closure plan."""

    kind: str
    subject_id: str
    requested_limit: int | None
    returned_count: int
    exhaustive: bool
    detail: Mapping[str, Any] = field(default_factory=dict)
    witness_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _nonempty(self.kind, "scope witness kind"))
        object.__setattr__(
            self,
            "subject_id",
            _nonempty(self.subject_id, "scope witness subject_id"),
        )
        if self.requested_limit is not None and self.requested_limit < 0:
            raise ValueError("scope witness requested_limit must be non-negative")
        if self.returned_count < 0:
            raise ValueError("scope witness returned_count must be non-negative")
        if (
            self.requested_limit is not None
            and self.returned_count > self.requested_limit
        ):
            raise ValueError("scope witness returned_count exceeds its requested limit")
        object.__setattr__(
            self,
            "detail",
            _json_mapping(self.detail, "scope witness detail"),
        )
        expected = identity_sha256(self.identity_payload(include_sha=False))
        if self.witness_sha256:
            if _sha256(self.witness_sha256, "witness_sha256") != expected:
                raise ValueError("scope witness SHA-256 does not match its contents")
        else:
            object.__setattr__(self, "witness_sha256", expected)

    def identity_payload(self, *, include_sha: bool = True) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "requested_limit": self.requested_limit,
            "returned_count": self.returned_count,
            "exhaustive": self.exhaustive,
            "detail": _plain_json(self.detail),
        }
        if include_sha:
            payload["witness_sha256"] = self.witness_sha256
        return payload


@dataclass(frozen=True, slots=True)
class ClosurePlan:
    query_program: QueryProgram
    policy: ClosurePolicy
    snapshot: DiscourseSnapshot
    seeds: tuple[EpisodeSeed, ...]
    atoms: tuple[EvidenceAtom, ...]
    bundles: tuple[EvidenceBundle, ...]
    obligation_results: tuple[ObligationResult, ...]
    visited_episode_ids: tuple[str, ...]
    visited_unit_ids: tuple[str, ...]
    visited_relation_ids: tuple[str, ...]
    stopping_reason: ClosureStopReason
    complete_claimed: bool
    scope_witnesses: tuple[ClosureScopeWitness, ...] = ()
    direct_chunk_ids: tuple[str, ...] = ()
    expansion_receipt_sha256: str | None = None
    artifact_id: str | None = None
    plan_sha256: str = ""

    def __post_init__(self) -> None:
        atoms = tuple(
            sorted(
                self.atoms,
                key=lambda item: evidence_span_sort_key(item.span) + (item.atom_id,),
            )
        )
        bundles = tuple(sorted(self.bundles, key=lambda item: item.bundle_id))
        witnesses = tuple(
            sorted(
                self.scope_witnesses,
                key=lambda item: (
                    item.kind,
                    item.subject_id,
                    -1 if item.requested_limit is None else item.requested_limit,
                    item.returned_count,
                    item.witness_sha256,
                ),
            )
        )
        direct_chunk_ids = tuple(
            sorted({_nonempty(item, "direct_chunk_id") for item in self.direct_chunk_ids})
        )
        expansion_receipt = self.expansion_receipt_sha256
        if expansion_receipt is not None:
            expansion_receipt = _sha256(
                expansion_receipt,
                "expansion_receipt_sha256",
            )
        artifact_id = self.artifact_id
        if artifact_id is not None:
            artifact_id = _nonempty(artifact_id, "artifact_id")
            if artifact_id not in self.snapshot.artifact_ids:
                raise ValueError("closure artifact_id is absent from its snapshot")
        results = tuple(self.obligation_results)
        known_obligations = {
            item.obligation_id: item for item in self.query_program.obligations
        }
        if len({item.atom_id for item in atoms}) != len(atoms):
            raise ValueError("closure atom IDs must be unique")
        if len({item.bundle_id for item in bundles}) != len(bundles):
            raise ValueError("closure bundle IDs must be unique")
        if len({item.witness_sha256 for item in witnesses}) != len(witnesses):
            raise ValueError("closure scope witnesses must be unique")
        if (
            len(results) != len(known_obligations)
            or len({item.obligation_id for item in results}) != len(results)
            or {item.obligation_id for item in results} != set(known_obligations)
        ):
            raise ValueError("closure results must cover every query obligation exactly")
        if any(
            item.status
            not in {"satisfied", "not_found", "conflicted", "budget_impossible"}
            for item in results
        ):
            raise ValueError("closure result has an unsupported status")
        known_atoms = {item.atom_id for item in atoms}
        if any(atom_id not in known_atoms for item in bundles for atom_id in item.atom_ids):
            raise ValueError("closure bundle references an unknown atom")
        known_bundle_ids = {item.bundle_id for item in bundles}
        visited_units = set(self.visited_unit_ids)
        visited_relations = set(self.visited_relation_ids)
        if any(
            obligation_id not in known_obligations
            for bundle in bundles
            for obligation_id in bundle.obligation_ids
        ):
            raise ValueError("closure bundle references an unknown obligation")
        if any(
            unit_id not in visited_units
            for owner in (*bundles, *results)
            for unit_id in owner.unit_ids
        ) or any(
            relation_id not in visited_relations
            for owner in (*bundles, *results)
            for relation_id in owner.relation_ids
        ):
            raise ValueError("closure evidence IDs must belong to the visited graph")
        if any(
            bundle_id not in known_bundle_ids
            for result in results
            for bundle_id in result.bundle_ids
        ):
            raise ValueError("closure result references an unknown bundle")
        result_by_id = {item.obligation_id: item for item in results}
        if any(
            bundle.bundle_id not in result_by_id[obligation_id].bundle_ids
            for bundle in bundles
            for obligation_id in bundle.obligation_ids
        ) or any(
            result.obligation_id
            not in next(
                bundle.obligation_ids
                for bundle in bundles
                if bundle.bundle_id == bundle_id
            )
            for result in results
            for bundle_id in result.bundle_ids
        ):
            raise ValueError("closure result and bundle obligation links disagree")
        required_ids = {
            item.obligation_id for item in known_obligations.values() if item.required
        }
        if any(
            bundle.required != bool(required_ids & set(bundle.obligation_ids))
            for bundle in bundles
        ):
            raise ValueError("closure bundle required flag is inconsistent")
        if any(
            result.status == "satisfied"
            and max(
                len(set(result.unit_ids)),
                len(set(result.relation_ids)),
                len(set(result.bundle_ids)),
            )
            < known_obligations[result.obligation_id].min_count
            for result in results
        ):
            raise ValueError("satisfied closure result does not meet min_count")
        required = {
            item.obligation_id
            for item in self.query_program.obligations
            if item.required
        }
        satisfied = {
            item.obligation_id for item in results if item.status == "satisfied"
        }
        should_complete = (
            required <= satisfied
            and self.stopping_reason == "complete"
            and bool(witnesses)
            and all(item.exhaustive for item in witnesses)
        )
        if self.complete_claimed != should_complete:
            raise ValueError("complete_claimed is inconsistent with required obligations")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "bundles", bundles)
        object.__setattr__(self, "scope_witnesses", witnesses)
        object.__setattr__(self, "direct_chunk_ids", direct_chunk_ids)
        object.__setattr__(self, "expansion_receipt_sha256", expansion_receipt)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(
            self,
            "visited_episode_ids",
            tuple(sorted(dict.fromkeys(self.visited_episode_ids))),
        )
        object.__setattr__(
            self,
            "visited_unit_ids",
            tuple(sorted(dict.fromkeys(self.visited_unit_ids))),
        )
        object.__setattr__(
            self,
            "visited_relation_ids",
            tuple(sorted(dict.fromkeys(self.visited_relation_ids))),
        )
        body = self.identity_payload(include_sha=False)
        expected = identity_sha256(body)
        if self.plan_sha256:
            if _sha256(self.plan_sha256, "plan_sha256") != expected:
                raise ValueError("closure plan SHA-256 does not match its contents")
        else:
            object.__setattr__(self, "plan_sha256", expected)

    def identity_payload(self, *, include_sha: bool = True) -> dict[str, Any]:
        payload = {
            "query_program_sha256": self.query_program.program_sha256,
            "policy_sha256": self.policy.policy_sha256,
            "snapshot_sha256": self.snapshot.snapshot_sha256,
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
            "atoms": [
                {
                    "atom_id": item.atom_id,
                    "span": item.span.identity_payload(),
                    "text_sha256": quote_sha256(item.text),
                    "label": item.label,
                    "role": item.role,
                    "created_at": item.created_at,
                }
                for item in self.atoms
            ],
            "bundles": [
                {
                    "bundle_id": item.bundle_id,
                    "atom_ids": list(item.atom_ids),
                    "obligation_ids": list(item.obligation_ids),
                    "unit_ids": list(item.unit_ids),
                    "relation_ids": list(item.relation_ids),
                    "required": item.required,
                    "utility": item.utility,
                }
                for item in self.bundles
            ],
            "obligation_results": [
                {
                    "obligation_id": item.obligation_id,
                    "status": item.status,
                    "unit_ids": list(item.unit_ids),
                    "relation_ids": list(item.relation_ids),
                    "bundle_ids": list(item.bundle_ids),
                    "reason": item.reason,
                }
                for item in self.obligation_results
            ],
            "visited_episode_ids": list(self.visited_episode_ids),
            "visited_unit_ids": list(self.visited_unit_ids),
            "visited_relation_ids": list(self.visited_relation_ids),
            "scope_witnesses": [
                item.identity_payload() for item in self.scope_witnesses
            ],
            "direct_chunk_ids": list(self.direct_chunk_ids),
            "expansion_receipt_sha256": self.expansion_receipt_sha256,
            "artifact_id": self.artifact_id,
            "stopping_reason": self.stopping_reason,
            "complete_claimed": self.complete_claimed,
        }
        if include_sha:
            payload["plan_sha256"] = self.plan_sha256
        return payload


@dataclass(frozen=True, slots=True)
class ClosureReceipt:
    plan_sha256: str
    context_sha256: str
    selected_bundle_ids: tuple[str, ...]
    selected_atom_ids: tuple[str, ...]
    dropped_bundle_reasons: Mapping[str, str]
    context_token_proxy: int
    max_context_token_proxy: int
    tokenizer_identity: str
    stopping_reason: ClosureStopReason
    complete_claimed: bool
    retained_request_token_state_bytes: int = 0
    prompt_token_proxy: int | None = None
    max_prompt_token_proxy: int | None = None
    responder_output_token_reserve: int = 0
    prompt_workspace_token_proxy: int | None = None
    base_messages_sha256: str | None = None
    evidence_message_role: str | None = None
    evidence_prefix_sha256: str | None = None
    evidence_suffix_sha256: str | None = None
    prompt_messages_sha256: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_sha256", _sha256(self.plan_sha256, "plan_sha256"))
        object.__setattr__(self, "context_sha256", _sha256(self.context_sha256, "context_sha256"))
        for name in (
            "context_token_proxy",
            "max_context_token_proxy",
            "retained_request_token_state_bytes",
            "responder_output_token_reserve",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.context_token_proxy < 0 or self.max_context_token_proxy < 0:
            raise ValueError("token counts must be non-negative")
        if self.context_token_proxy > self.max_context_token_proxy:
            raise ValueError("closure packet exceeds its hard token budget")
        if self.retained_request_token_state_bytes != 0:
            raise ValueError("closure receipts require zero retained request token state")
        if self.complete_claimed and self.stopping_reason != "complete":
            raise ValueError(
                "complete_claimed must be false unless stopping_reason is complete"
            )
        if self.responder_output_token_reserve < 0:
            raise ValueError("responder output token reserve must be non-negative")
        prompt_fields = (
            self.prompt_token_proxy,
            self.max_prompt_token_proxy,
            self.prompt_workspace_token_proxy,
            self.base_messages_sha256,
            self.evidence_message_role,
            self.evidence_prefix_sha256,
            self.evidence_suffix_sha256,
            self.prompt_messages_sha256,
        )
        prompt_budget_enabled = any(value is not None for value in prompt_fields)
        if prompt_budget_enabled:
            if any(value is None for value in prompt_fields):
                raise ValueError(
                    "prompt-budget receipt fields must be supplied together"
                )
            assert self.prompt_token_proxy is not None
            assert self.max_prompt_token_proxy is not None
            assert self.prompt_workspace_token_proxy is not None
            for name in (
                "prompt_token_proxy",
                "max_prompt_token_proxy",
                "prompt_workspace_token_proxy",
            ):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{name} must be an integer")
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
            expected_request = (
                self.prompt_token_proxy + self.responder_output_token_reserve
            )
            if self.prompt_workspace_token_proxy != expected_request:
                raise ValueError(
                    "prompt workspace token proxy must equal prompt proxy plus "
                    "output reserve"
                )
            if self.prompt_workspace_token_proxy > self.max_prompt_token_proxy:
                raise ValueError(
                    "closure packet prompt proxy plus output reserve exceeds "
                    "its hard prompt budget"
                )
            object.__setattr__(
                self,
                "base_messages_sha256",
                _sha256(str(self.base_messages_sha256), "base_messages_sha256"),
            )
            object.__setattr__(
                self,
                "evidence_prefix_sha256",
                _sha256(str(self.evidence_prefix_sha256), "evidence_prefix_sha256"),
            )
            object.__setattr__(
                self,
                "evidence_suffix_sha256",
                _sha256(str(self.evidence_suffix_sha256), "evidence_suffix_sha256"),
            )
            object.__setattr__(
                self,
                "prompt_messages_sha256",
                _sha256(str(self.prompt_messages_sha256), "prompt_messages_sha256"),
            )
            object.__setattr__(
                self,
                "evidence_message_role",
                _nonempty(str(self.evidence_message_role), "evidence_message_role"),
            )
        elif self.responder_output_token_reserve != 0:
            raise ValueError(
                "an output reserve requires an enabled prompt-token proxy budget"
            )
        dropped = _json_mapping(
            self.dropped_bundle_reasons,
            "dropped_bundle_reasons",
        )
        if any(not isinstance(value, str) for value in dropped.values()):
            raise ValueError("dropped_bundle_reasons values must be strings")
        object.__setattr__(self, "dropped_bundle_reasons", dropped)
        object.__setattr__(
            self,
            "tokenizer_identity",
            _nonempty(self.tokenizer_identity, "tokenizer_identity"),
        )
        body = {
            "plan_sha256": self.plan_sha256,
            "context_sha256": self.context_sha256,
            "selected_bundle_ids": list(self.selected_bundle_ids),
            "selected_atom_ids": list(self.selected_atom_ids),
            "dropped_bundle_reasons": dict(self.dropped_bundle_reasons),
            "context_token_proxy": self.context_token_proxy,
            "max_context_token_proxy": self.max_context_token_proxy,
            "tokenizer_identity": self.tokenizer_identity,
            "stopping_reason": self.stopping_reason,
            "complete_claimed": self.complete_claimed,
            "retained_request_token_state_bytes": 0,
        }
        if prompt_budget_enabled:
            body.update(
                {
                    "prompt_token_proxy": self.prompt_token_proxy,
                    "max_prompt_token_proxy": self.max_prompt_token_proxy,
                    "responder_output_token_reserve": (
                        self.responder_output_token_reserve
                    ),
                    "prompt_workspace_token_proxy": (
                        self.prompt_workspace_token_proxy
                    ),
                    "base_messages_sha256": self.base_messages_sha256,
                    "evidence_message_role": self.evidence_message_role,
                    "evidence_prefix_sha256": self.evidence_prefix_sha256,
                    "evidence_suffix_sha256": self.evidence_suffix_sha256,
                    "prompt_messages_sha256": self.prompt_messages_sha256,
                }
            )
        expected = identity_sha256(body)
        if self.receipt_sha256:
            if _sha256(self.receipt_sha256, "receipt_sha256") != expected:
                raise ValueError("closure receipt SHA-256 does not match its contents")
        else:
            object.__setattr__(self, "receipt_sha256", expected)


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """The exact source-grounded context presented to an answerer."""

    context: str
    atoms: tuple[EvidenceAtom, ...]
    bundles: tuple[EvidenceBundle, ...]
    receipt: ClosureReceipt

    def __post_init__(self) -> None:
        if quote_sha256(self.context) != self.receipt.context_sha256:
            raise ValueError("packet context does not match the closure receipt")
        if self.receipt.selected_atom_ids != tuple(item.atom_id for item in self.atoms):
            raise ValueError("packet atoms do not match the closure receipt")
        if self.receipt.selected_bundle_ids != tuple(item.bundle_id for item in self.bundles):
            raise ValueError("packet bundles do not match the closure receipt")


__all__ = [
    "ArtifactCoverageReceipt",
    "ClosurePlan",
    "ClosurePolicy",
    "ClosureReceipt",
    "ClosureScopeWitness",
    "DiscourseArtifact",
    "DiscourseRelation",
    "DiscourseSnapshot",
    "DiscourseUnit",
    "Episode",
    "EpisodeRepresentative",
    "EpisodeSeed",
    "EvidenceAtom",
    "EvidenceBundle",
    "EvidenceObligation",
    "EvidencePacket",
    "EvidenceSpan",
    "ObligationResult",
    "QueryProgram",
    "RelationMember",
    "canonical_json",
    "evidence_span_sort_key",
    "identity_sha256",
    "make_atom_id",
    "make_bundle_id",
    "make_episode_id",
    "quote_sha256",
]
