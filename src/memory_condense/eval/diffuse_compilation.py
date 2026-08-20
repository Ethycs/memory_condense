"""Gold-blind compilation of source chunks into a diffuse retrieval artifact."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from memory_condense.application.discourse_workflow import DiscourseLinker
from memory_condense.domain._discourse_identity import _as_tuple, normalize_fields
from memory_condense.domain.discourse import (
    DiscourseArtifact,
    DiscourseSnapshot,
    identity_sha256,
)
from memory_condense.domain.sealed import SealedIdentity, reflect_payload
from memory_condense.eval._identity import exact_int, sha256_digest
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.ingest.discourse_linker import RuleBasedDiscourseLinker
from memory_condense.persistence.discourse_store import ArtifactCoverageMark
from memory_condense.search.episodes import (
    AdaptiveBoundaryDetector,
    CohesionBoundaryRefiner,
    EpisodeBuilder,
    FixedIntervalBoundaryDetector,
    LexicalEmbeddingChangeScorer,
    QwenAttentionHeadSurpriseScorer,
)


DIFFUSE_COMPILATION_FORMAT = "memory-condense-diffuse-compilation-v1"
BoundaryMode = Literal["fixed_interval", "lexical_embedding", "qwen_head"]


@dataclass(frozen=True, slots=True)
class DiffuseCompilationPolicy:
    """Frozen matched-arm controls for episode and discourse compilation."""

    boundary_mode: BoundaryMode
    min_episode_size: int = 2
    max_episode_size: int = 16
    fixed_interval: int = 8
    surprise_window: int = 32
    surprise_gamma: float = 1.0
    surprise_min_history: int = 2
    refinement_window: int = 4
    refinement_max_nodes: int = 32
    refinement_max_degree: int = 4
    lexical_weight: float = 1.0
    embedding_weight: float = 1.0
    representative_limit: int = 2

    def __post_init__(self) -> None:
        if self.boundary_mode not in {
            "fixed_interval",
            "lexical_embedding",
            "qwen_head",
        }:
            raise ValueError("unsupported diffuse boundary mode")
        for name in (
            "min_episode_size",
            "max_episode_size",
            "fixed_interval",
            "surprise_window",
            "surprise_min_history",
            "refinement_max_nodes",
            "refinement_max_degree",
            "representative_limit",
        ):
            value = exact_int(getattr(self, name), name, minimum=1)
            object.__setattr__(self, name, value)
        if self.max_episode_size < self.min_episode_size:
            raise ValueError("max_episode_size cannot be smaller than min")
        if self.surprise_min_history > self.surprise_window:
            raise ValueError("surprise_min_history exceeds its window")
        refinement_window = exact_int(
            self.refinement_window,
            "refinement_window",
            minimum=0,
        )
        object.__setattr__(self, "refinement_window", refinement_window)
        for name in (
            "surprise_gamma",
            "lexical_weight",
            "embedding_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.lexical_weight + self.embedding_weight <= 0.0:
            raise ValueError("lexical/embedding control weights cannot both be zero")

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(self.identity_payload())

    def identity_payload(self) -> dict[str, object]:
        return reflect_payload(self)


@dataclass(frozen=True, slots=True)
class DiffuseSourceCompilationReceipt(SealedIdentity):
    _SEAL_MISMATCH = "source compilation receipt does not match"

    source_id: str
    source_stream_sha256: str
    content_chunks: int
    metadata_chunks: int
    episode_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    episode_build_sha256: str | None
    discourse_link_sha256: str | None
    surprise_signal_receipt_sha256: str | None
    returned_signal_transformer_state_bytes: int | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        sha256_digest(self.source_stream_sha256, "source_stream_sha256")
        for name in (
            "episode_build_sha256",
            "discourse_link_sha256",
            "surprise_signal_receipt_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                sha256_digest(value, name)
        for name in ("content_chunks", "metadata_chunks"):
            if exact_int(getattr(self, name), name, minimum=0) != getattr(
                self,
                name,
            ):
                raise ValueError(f"{name} must be an exact integer")
        retained = self.returned_signal_transformer_state_bytes
        if retained is not None and (type(retained) is not int or retained != 0):
            raise ValueError("returned signal retention must be zero or unattested")
        self._seal()


@dataclass(frozen=True, slots=True)
class DiffuseCompilationReceipt(SealedIdentity):
    _SEAL_MISMATCH = "diffuse compilation receipt does not match"

    artifact: DiscourseArtifact
    # ``policy_sha256`` binds the artifact's complete composition policy;
    # this field separately binds the exact segmentation policy requested by
    # the matched-arm runner.
    compilation_policy_sha256: str
    policy_sha256: str
    source_receipts: tuple[DiffuseSourceCompilationReceipt, ...]
    episode_coverage_receipt_sha256: str
    discourse_coverage_receipt_sha256: str
    final_snapshot: DiscourseSnapshot
    persisted_request_token_state_bytes: int
    format: str = DIFFUSE_COMPILATION_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != DIFFUSE_COMPILATION_FORMAT:
            raise ValueError("unsupported diffuse compilation format")
        for name in (
            "compilation_policy_sha256",
            "policy_sha256",
            "episode_coverage_receipt_sha256",
            "discourse_coverage_receipt_sha256",
        ):
            sha256_digest(getattr(self, name), name)
        if self.policy_sha256 != self.artifact.policy_sha256:
            raise ValueError("artifact and compilation policy disagree")
        if (
            type(self.persisted_request_token_state_bytes) is not int
            or self.persisted_request_token_state_bytes != 0
        ):
            raise ValueError("diffuse persistence must retain zero request state")
        normalize_fields(self, source_receipts=_as_tuple)
        receipts = self.source_receipts
        if len({item.source_id for item in receipts}) != len(receipts):
            raise ValueError("source compilation receipts must be unique")
        if self.artifact.artifact_id not in self.final_snapshot.artifact_ids:
            raise ValueError("final snapshot does not contain the artifact")
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "artifact": self.artifact.identity_payload(),
            "compilation_policy_sha256": self.compilation_policy_sha256,
            "policy_sha256": self.policy_sha256,
            "source_receipts": [
                item.identity_payload() for item in self.source_receipts
            ],
            "episode_coverage_receipt_sha256": (
                self.episode_coverage_receipt_sha256
            ),
            "discourse_coverage_receipt_sha256": (
                self.discourse_coverage_receipt_sha256
            ),
            "final_snapshot_sha256": self.final_snapshot.snapshot_sha256,
            "persisted_request_token_state_bytes": (
                self.persisted_request_token_state_bytes
            ),
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def compile_diffuse_artifact(
    condenser: object,
    *,
    policy: DiffuseCompilationPolicy,
    qwen_scorer: QwenAttentionHeadSurpriseScorer | None = None,
    linker: DiscourseLinker | None = None,
    embedding_identity: Mapping[str, object] | None = None,
) -> DiffuseCompilationReceipt:
    """Compile every current source without accepting questions or gold labels."""

    streams = tuple(condenser.discourse_source_streams())
    if not streams:
        raise ValueError("cannot compile an empty transcript")
    builder, scorer, use_embeddings = _build_controls(policy, qwen_scorer)
    scorer_identity = _scorer_identity(
        policy,
        scorer,
        embedding_identity=embedding_identity,
    )
    artifact = DiscourseArtifact.create(
        kind=f"longmemeval-diffuse-{policy.boundary_mode}",
        implementation_sha256=implementation_sha256(),
        policy={
            "format": DIFFUSE_COMPILATION_FORMAT,
            "compilation": policy.identity_payload(),
            "scorer": scorer_identity,
            "linker": "rule_based_discourse_linker_v1",
            "source_scope": "all_current_chunks",
        },
        model_id=_optional_text(scorer_identity.get("model_id")),
        model_revision=_optional_text(scorer_identity.get("model_revision")),
        checkpoint_sha256=_optional_digest(
            scorer_identity.get("checkpoint_sha256")
        ),
        metadata={
            "boundary_policy_id": policy.boundary_mode,
            "scorer_id": identity_sha256(scorer_identity),
        },
    )
    active_linker = linker or RuleBasedDiscourseLinker()
    if type(active_linker) is not RuleBasedDiscourseLinker:
        raise ValueError(
            "certified diffuse compilation requires the exact rule linker"
        )
    source_receipts: list[DiffuseSourceCompilationReceipt] = []
    metadata_marks: list[ArtifactCoverageMark] = []

    for stream in streams:
        metadata_marks.extend(
            ArtifactCoverageMark(chunk_id, kind, "no_output")
            for chunk_id in stream.metadata_chunk_ids
            for kind in ("episode", "discourse")
        )
        if not stream.content_chunk_ids:
            source_receipts.append(
                DiffuseSourceCompilationReceipt(
                    source_id=stream.source_id,
                    source_stream_sha256=stream.stream_sha256,
                    content_chunks=0,
                    metadata_chunks=len(stream.metadata_chunk_ids),
                    episode_ids=(),
                    unit_ids=(),
                    relation_ids=(),
                    episode_build_sha256=None,
                    discourse_link_sha256=None,
                    surprise_signal_receipt_sha256=None,
                    returned_signal_transformer_state_bytes=0,
                )
            )
            continue

        embeddings = (
            condenser.discourse_chunk_embeddings(stream.content_chunk_ids)
            if use_embeddings
            else None
        )
        episode_publication = condenser.build_and_publish_discourse_episodes(
            artifact,
            stream.content_chunk_ids,
            source_id=stream.source_id,
            builder=builder,
            surprise_scorer=scorer,
            embeddings=embeddings,
            representative_limit=policy.representative_limit,
        )
        link_publication = condenser.link_and_publish_discourse(
            artifact,
            chunk_ids=stream.content_chunk_ids,
            linker=active_linker,
        )
        signal_receipt = episode_publication.build.surprise_signal_receipt
        source_receipts.append(
            DiffuseSourceCompilationReceipt(
                source_id=stream.source_id,
                source_stream_sha256=stream.stream_sha256,
                content_chunks=len(stream.content_chunk_ids),
                metadata_chunks=len(stream.metadata_chunk_ids),
                episode_ids=tuple(
                    item.episode_id for item in episode_publication.build.episodes
                ),
                unit_ids=tuple(
                    item.unit_id for item in link_publication.output.units
                ),
                relation_ids=tuple(
                    item.relation_id for item in link_publication.output.relations
                ),
                episode_build_sha256=_episode_build_sha256(
                    episode_publication.build
                ),
                    discourse_link_sha256=identity_sha256(
                        {
                            "units": [
                                _unit_identity_payload(item)
                                for item in link_publication.output.units
                            ],
                            "relations": [
                                _relation_identity_payload(item)
                                for item in link_publication.output.relations
                            ],
                        }
                ),
                surprise_signal_receipt_sha256=(
                    None if signal_receipt is None else signal_receipt.receipt_sha256
                ),
                returned_signal_transformer_state_bytes=(
                    episode_publication.returned_signal_transformer_state_bytes
                ),
            )
        )

    if metadata_marks:
        condenser.discourse.publish(artifact, coverage=tuple(metadata_marks))
    episode_coverage = condenser.finalize_episode_coverage(artifact.artifact_id)
    discourse_coverage = condenser.finalize_discourse_coverage(
        artifact.artifact_id
    )
    final_snapshot = condenser.discourse.snapshot()
    persisted = condenser.discourse.stats()[
        "retained_request_token_state_bytes"
    ]
    return DiffuseCompilationReceipt(
        artifact=artifact,
        compilation_policy_sha256=policy.policy_sha256,
        policy_sha256=artifact.policy_sha256,
        source_receipts=tuple(source_receipts),
        episode_coverage_receipt_sha256=episode_coverage.receipt_sha256,
        discourse_coverage_receipt_sha256=discourse_coverage.receipt_sha256,
        final_snapshot=final_snapshot,
        persisted_request_token_state_bytes=persisted,
    )


def _build_controls(policy, qwen_scorer):
    if policy.boundary_mode == "fixed_interval":
        if qwen_scorer is not None:
            raise ValueError("fixed interval arm cannot accept a Qwen scorer")
        return (
            EpisodeBuilder(
                min_size=policy.min_episode_size,
                max_size=policy.max_episode_size,
                detector=FixedIntervalBoundaryDetector(policy.fixed_interval),
            ),
            None,
            False,
        )
    detector = AdaptiveBoundaryDetector(
        window_size=policy.surprise_window,
        gamma=policy.surprise_gamma,
        min_history=policy.surprise_min_history,
    )
    refiner = CohesionBoundaryRefiner(
        window=policy.refinement_window,
        max_nodes=policy.refinement_max_nodes,
        max_degree=policy.refinement_max_degree,
    )
    builder = EpisodeBuilder(
        min_size=policy.min_episode_size,
        max_size=policy.max_episode_size,
        detector=detector,
        refiner=refiner,
    )
    if policy.boundary_mode == "qwen_head":
        if type(qwen_scorer) is not QwenAttentionHeadSurpriseScorer:
            raise ValueError("qwen_head arm requires the exact Qwen scorer")
        return builder, qwen_scorer, False
    if qwen_scorer is not None:
        raise ValueError("lexical/embedding arm cannot accept a Qwen scorer")
    return (
        builder,
        LexicalEmbeddingChangeScorer(
            lexical_weight=policy.lexical_weight,
            embedding_weight=policy.embedding_weight,
        ),
        policy.embedding_weight > 0.0,
    )


def _scorer_identity(policy, scorer, *, embedding_identity):
    if scorer is None:
        return {"kind": "none", "model_id": None}
    if type(scorer) is LexicalEmbeddingChangeScorer:
        if policy.embedding_weight > 0.0 and not embedding_identity:
            raise ValueError("embedding arm requires a frozen embedding identity")
        return {
            "kind": "lexical_embedding_change",
            "lexical_weight": scorer.lexical_weight,
            "embedding_weight": scorer.embedding_weight,
            "embedding_identity": dict(embedding_identity or {}),
            "model_id": None,
        }
    if type(scorer) is QwenAttentionHeadSurpriseScorer:
        from memory_condense.search.episodes.qwen_episode_signal import (
            _attention_head_implementation_sha256,
            _qwen_linker_identity,
        )

        identity = _qwen_linker_identity(scorer.linker)
        return {
            "kind": "qwen_attention_head_surprise",
            **identity,
            "implementation_sha256": _attention_head_implementation_sha256(
                scorer.linker
            ),
            "max_spans": scorer.max_spans,
            "span_token_cap": scorer.span_token_cap,
            "probe_token_cap": scorer.probe_token_cap,
            "max_transport_dimension": scorer.max_transport_dimension,
        }
    raise TypeError("unsupported surprise scorer")


def _episode_build_sha256(build) -> str:
    return identity_sha256(
        {
            "source_id": build.source_id,
            "artifact_id": build.artifact_id,
            "episode_receipts": [item.receipt_sha256 for item in build.episodes],
            "initial_boundaries": [
                asdict(item) for item in build.initial_boundaries
            ],
            "refined_boundaries": [
                asdict(item) for item in build.refined_boundaries
            ],
            "forced_boundaries": list(build.forced_boundaries),
            "surprise_signal_receipt_sha256": (
                None
                if build.surprise_signal_receipt is None
                else build.surprise_signal_receipt.receipt_sha256
            ),
        }
    )


def _unit_identity_payload(unit) -> dict[str, object]:
    return {
        "unit_id": unit.unit_id,
        "artifact_id": unit.artifact_id,
        "kind": unit.kind,
        "canonical_key": unit.canonical_key,
        "asserted_ordinal": unit.asserted_ordinal,
        "confidence": unit.confidence,
        "evidence": [item.identity_payload() for item in unit.evidence],
        "metadata": dict(unit.metadata),
    }


def _relation_identity_payload(relation) -> dict[str, object]:
    return {
        "relation_id": relation.relation_id,
        "artifact_id": relation.artifact_id,
        "relation_type": relation.relation_type,
        "members": [
            {
                "unit_id": item.unit_id,
                "role": item.role,
                "ordinal": item.ordinal,
                "weight": item.weight,
            }
            for item in relation.members
        ],
        "evidence": [item.identity_payload() for item in relation.evidence],
        "confidence": relation.confidence,
        "created_ordinal": relation.created_ordinal,
        "metadata": dict(relation.metadata),
    }


def _optional_text(value):
    normalized = "" if value is None else str(value).strip()
    return normalized or None


def _optional_digest(value):
    if value is None:
        return None
    return sha256_digest(value, "checkpoint_sha256")


__all__ = [
    "DIFFUSE_COMPILATION_FORMAT",
    "DiffuseCompilationPolicy",
    "DiffuseCompilationReceipt",
    "DiffuseSourceCompilationReceipt",
    "compile_diffuse_artifact",
]
