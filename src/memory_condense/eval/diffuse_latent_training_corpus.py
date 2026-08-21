"""Provider-free route-v2 corpus facade and population mapper.

The generic core verifies structural projections only.  It cannot verify the
external evaluator ``AnalysisTreatmentInput`` type, authorize production, or
make a D1 checkpoint eligible; those remain future tracked-launcher duties.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval._diffuse_latent_training_corpus_codec import (
    DecodedLatentTrainingPayload,
    encode_latent_training_payload,
)
from memory_condense.eval._diffuse_latent_training_corpus_models import (
    ANALYSIS_POPULATION_PROJECTION_FORMAT,
    LATENT_TRAINING_CORPUS_FORMAT,
    LATENT_TRAINING_PARTITION_FORMAT,
    LATENT_TRAINING_PUBLICATION_FORMAT,
    LATENT_TRAINING_ROW_FORMAT,
    LOCKED_LATENT_TRAINING_POPULATION,
    MAX_PAYLOAD_SHARD_BYTES,
    AnalysisPopulationProjection,
    AnalysisPopulationRow,
    DecodedLatentTrainingCorpusRow,
    LatentTrainingCorpusError,
    LatentTrainingCorpusManifest,
    LatentTrainingCorpusPartitionManifest,
    LatentTrainingCorpusPublicationReceipt,
    LatentTrainingCorpusRowManifest,
    LatentTrainingFileIdentity,
    LatentTrainingPopulationExpectation,
    LatentTrainingRouteEvidence,
    VerifiedLatentTrainingFitCorpus,
    VerifiedLatentTrainingFullCorpus,
    VerifiedLatentTrainingValidationCorpus,
    _ATOM_REF_KIND,
    _HYPEREDGE_KIND,
    _ids_sha256,
)
from memory_condense.eval.diffuse_longmemeval_route_v2 import (
    EpisodePrimaryRetrievalPhaseV2,
    verify_episode_primary_analysis_phase_v2,
)
from memory_condense.eval.diffuse_longmemeval import _anchor_payload
from memory_condense.eval.diffuse_longmemeval_inputs import _anchor_identity
from memory_condense.eval.reproducibility import (
    implementation_sha256 as _package_implementation_sha256,
)
from memory_condense.search.fusion.models import FusionCaps
from memory_condense.search.fusion.planner import (
    _atom_refs,
    _authoritative_hyperedges,
)
from memory_condense.search.fusion.resident_models import resident_values_sha256
from memory_condense.search.fusion.training_targets import (
    build_latent_router_structural_targets,
)
from memory_condense.search.episodes.representative_retrieval import (
    EpisodeRepresentativeRetrievalPolicy,
)


@dataclass(frozen=True, slots=True)
class StructuralRouteV2MappedRow:
    """Exact non-authoritative mapper result, including omitted route policy."""

    phase: EpisodePrimaryRetrievalPhaseV2
    representative_policy: EpisodeRepresentativeRetrievalPolicy

    def __post_init__(self) -> None:
        if type(self.phase) is not EpisodePrimaryRetrievalPhaseV2:
            raise TypeError("mapped phase has the wrong exact route-v2 type")
        if type(self.representative_policy) is not EpisodeRepresentativeRetrievalPolicy:
            raise TypeError("mapped representative policy has the wrong exact type")
        verify_episode_primary_analysis_phase_v2(self.phase)
        if len(self.phase.questions) != 1:
            raise LatentTrainingCorpusError("mapped phase must contain exactly one query")
        representative = self.phase.questions[0].inner.retrieval.representative_expansion
        if representative is None or (
            self.representative_policy.policy_sha256 != representative.policy_sha256
            or self.representative_policy.artifact_id != representative.artifact_id
        ):
            raise LatentTrainingCorpusError(
                "mapped representative policy does not bind the route phase"
            )


RouteV2PopulationMapper = Callable[
    [AnalysisPopulationRow], StructuralRouteV2MappedRow
]


def _validate_projection(
    projection: AnalysisPopulationProjection,
    expected: LatentTrainingPopulationExpectation,
) -> tuple[AnalysisPopulationRow, ...]:
    """Complete the population/count/order firebreak before any mapper call."""

    if type(projection) is not AnalysisPopulationProjection:
        raise TypeError("population must be an exact AnalysisPopulationProjection")
    if type(expected) is not LatentTrainingPopulationExpectation:
        raise TypeError("population expectation has the wrong exact type")
    projection._seal()
    if projection.source_treatment_exact_type_verified is not False:
        raise LatentTrainingCorpusError(
            "generic projection cannot verify the source treatment type"
        )
    ids = projection.ordered_question_ids
    fit_end = expected.fit_count
    total = fit_end + expected.validation_count
    if total > 300 or fit_end > 300 or expected.validation_count > 300:
        raise LatentTrainingCorpusError(
            "analysis population exceeds the closed 300-row corpus cap"
        )
    checks = {
        "dataset": projection.dataset_sha256 == expected.dataset_sha256,
        "split": projection.split_manifest_sha256 == expected.split_manifest_sha256,
        "treatment": projection.treatment_file_sha256 == expected.treatment_file_sha256,
        "projection": projection.sanitized_projection_sha256
        == expected.sanitized_projection_sha256,
        "analysis count": len(ids) == total,
        "analysis order": _ids_sha256(ids)
        == expected.analysis_ordered_question_ids_sha256,
        "fit order": _ids_sha256(ids[:fit_end])
        == expected.fit_ordered_question_ids_sha256,
        "validation order": _ids_sha256(ids[fit_end:])
        == expected.validation_ordered_question_ids_sha256,
        "confirmation count": projection.excluded_confirmation_count
        == expected.excluded_confirmation_count,
        "confirmation order": (
            projection.excluded_confirmation_ordered_question_ids_sha256
            == expected.excluded_confirmation_ordered_question_ids_sha256
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise LatentTrainingCorpusError(
            "analysis population projection failed before row mapping: "
            + ", ".join(failed)
        )
    return tuple(
        AnalysisPopulationRow(
            ordinal=index,
            partition="fit" if index < fit_end else "validation",
            partition_ordinal=index if index < fit_end else index - fit_end,
            question_id=question_id,
        )
        for index, question_id in enumerate(ids)
    )


def _route_evidence(
    record: object,
    phase: EpisodePrimaryRetrievalPhaseV2,
    representative_policy: EpisodeRepresentativeRetrievalPolicy,
) -> LatentTrainingRouteEvidence:
    inner = record.inner
    source_scope = inner.legacy_inputs.candidates.source_candidate_scope
    representative = inner.retrieval.representative_expansion
    if source_scope is None or representative is None:
        raise LatentTrainingCorpusError(
            "route-v2 row lacks required intermediate bodies"
        )
    base_arm = record.analysis_arm.base_arm
    return LatentTrainingRouteEvidence(
        analysis_arm_v2_body=record.analysis_arm.identity_payload(),
        analysis_base_arm_body=base_arm.identity_payload(),
        episode_policy_body={
            "artifact_id": base_arm.episode.artifact_id,
            "max_anchor_episodes": base_arm.episode.max_anchor_episodes,
            "previous_episodes": base_arm.episode.previous_episodes,
            "next_episodes": base_arm.episode.next_episodes,
            "max_episode_seeds": base_arm.episode.max_episode_seeds,
            "max_direct_fallbacks": base_arm.episode.max_direct_fallbacks,
            "neighbor_decay": base_arm.episode.neighbor_decay,
        },
        closure_policy_body={
            "max_hops": base_arm.closure.max_hops,
            "max_units": base_arm.closure.max_units,
            "max_relations": base_arm.closure.max_relations,
            "max_degree": base_arm.closure.max_degree,
            "max_episode_neighbors": base_arm.closure.max_episode_neighbors,
            "max_frontier": base_arm.closure.max_frontier,
            "max_bundles": base_arm.closure.max_bundles,
            "beam_width": base_arm.closure.beam_width,
            "min_relation_confidence": base_arm.closure.min_relation_confidence,
        },
        compilation_receipt_body=phase.inner.compilation.identity_payload(),
        compilation_snapshot_body=(
            phase.inner.compilation.final_snapshot.identity_payload()
        ),
        representative_policy_body={
            "artifact_id": representative_policy.artifact_id,
            "max_input_sources": representative_policy.max_input_sources,
            "max_source_groups": representative_policy.max_source_groups,
            "max_episodes_per_source": representative_policy.max_episodes_per_source,
            "max_total_episodes": representative_policy.max_total_episodes,
            "max_representatives_per_episode": (
                representative_policy.max_representatives_per_episode
            ),
            "group_size": representative_policy.group_size,
            "beam_per_group": representative_policy.beam_per_group,
            "top_k": representative_policy.top_k,
            "representative_tokens": representative_policy.representative_tokens,
            "query_tokens": representative_policy.query_tokens,
            "score_mode": representative_policy.score_mode,
        },
        anchor_projection_body={
            "legacy": [
                _anchor_identity(item)
                for item in inner.legacy_inputs.candidates.anchors
            ],
            "diffuse": [
                _anchor_payload(item)
                for item in inner.legacy_inputs.candidates.anchors
            ],
        },
        inner_analysis_query_receipt_body=inner.receipt.identity_payload(),
        inner_diffuse_query_receipt_body=inner.retrieval.receipt.identity_payload(),
        legacy_input_receipt_body=inner.legacy_inputs.receipt.identity_payload(),
        source_scope_body=source_scope.identity_payload(),
        direct_expansion_body=inner.retrieval.expansion.identity_payload(),
        representative_expansion_body=representative.identity_payload(),
        route_receipt=record.route_receipt,
    )


def _build_row(
    population_row: AnalysisPopulationRow,
    mapped: StructuralRouteV2MappedRow,
) -> tuple[LatentTrainingCorpusRowManifest, bytes]:
    if type(mapped) is not StructuralRouteV2MappedRow:
        raise TypeError(
            "row mapper must return exact StructuralRouteV2MappedRow"
        )
    mapped.__post_init__()
    phase = mapped.phase
    verify_episode_primary_analysis_phase_v2(phase)
    if len(phase.questions) != 1:
        raise LatentTrainingCorpusError(
            "each mapped phase must contain exactly one query"
        )
    record = phase.questions[0]
    if record.inner.probe.question_id != population_row.question_id:
        raise LatentTrainingCorpusError(
            "mapped route row belongs to another question"
        )
    retrieval = record.inner.retrieval
    packet, plan = retrieval.packet, retrieval.plan
    target = build_latent_router_structural_targets(
        packet,
        plan,
        caps=FusionCaps(),
    )
    atom_refs = _atom_refs(packet)
    hyperedges = _authoritative_hyperedges(packet)
    probe = record.inner.probe
    payload = encode_latent_training_payload(
        probe.retrieval_query,
        plan,
        packet,
        question_id=probe.question_id,
        prompt_question=probe.prompt_question,
    )
    if len(payload) > MAX_PAYLOAD_SHARD_BYTES:
        raise MemoryError("latent-training payload exceeds its shard byte cap")
    payload_sha = hashlib.sha256(payload).hexdigest()
    return (
        LatentTrainingCorpusRowManifest(
            ordinal=population_row.ordinal,
            partition=population_row.partition,
            partition_ordinal=population_row.partition_ordinal,
            question_id=population_row.question_id,
            question_id_sha256=identity_sha256(
                {"question_id": population_row.question_id}
            ),
            question_probe_sha256=probe.probe_sha256,
            retrieval_query_sha256=identity_sha256(
                {"query": probe.retrieval_query}
            ),
            prompt_question_sha256=identity_sha256(
                {"prompt_question": probe.prompt_question}
            ),
            route_record_sha256=record.record_sha256,
            route_evidence=_route_evidence(
                record, phase, mapped.representative_policy
            ),
            payload_relative_path=f"payloads/{payload_sha}.json",
            payload_sha256=payload_sha,
            payload_bytes=len(payload),
            packet_receipt_sha256=packet.receipt.receipt_sha256,
            closure_plan_sha256=plan.plan_sha256,
            ordered_atom_refs_sha256=resident_values_sha256(
                _ATOM_REF_KIND,
                atom_refs,
            ),
            authoritative_hyperedges_sha256=resident_values_sha256(
                _HYPEREDGE_KIND,
                hyperedges,
            ),
            structural_target=target,
        ),
        payload,
    )


def _bind_implementation_observer(observer: Callable[[], str]) -> Callable[[], str]:
    def latent_training_corpus_implementation_sha256() -> str:
        """Bind the live package-wide source closure used by build and verify."""

        return identity_sha256(
            {
                "format": "memory-condense-latent-training-corpus-implementation-v2",
                "memory_condense_package_sha256": observer(),
            }
        )

    return latent_training_corpus_implementation_sha256


latent_training_corpus_implementation_sha256 = _bind_implementation_observer(
    _package_implementation_sha256
)
del _bind_implementation_observer, _package_implementation_sha256


def _bind_locked_publisher() -> Callable[..., LatentTrainingCorpusPublicationReceipt]:
    public_lock = LOCKED_LATENT_TRAINING_POPULATION
    expected = LatentTrainingPopulationExpectation(
        dataset_sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        split_manifest_sha256="8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4",
        treatment_file_sha256="b4d1d34538fdabbd6127c339bff8167293d290eb732afc18a5d8963d12b15001",
        sanitized_projection_sha256="58a1982122d259e046ac5268de8fc3c2857a63d24c859e3bc13e4e6b9aa52ad8",
        fit_count=200,
        fit_ordered_question_ids_sha256="533aa545efb8032f7b181f39264c6d10a49471bd460414f420e37dc840a19c55",
        validation_count=100,
        validation_ordered_question_ids_sha256="7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1",
        analysis_ordered_question_ids_sha256="cf5e8648b71634e4e22be872881766e37e0dc24a2931d0c63365e075b2742046",
        excluded_confirmation_count=200,
        excluded_confirmation_ordered_question_ids_sha256="6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102",
    )

    def assert_public_lock() -> None:
        current = globals().get("LOCKED_LATENT_TRAINING_POPULATION")
        if current is not public_lock:
            raise RuntimeError("public latent-training population lock binding changed")
        try:
            current.__post_init__()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "public latent-training population lock value changed"
            ) from exc
        if current != expected:
            raise RuntimeError("public latent-training population lock value changed")

    def publish_structural_latent_training_corpus(
        population: AnalysisPopulationProjection,
        destination: str | Path,
        *,
        row_mapper: RouteV2PopulationMapper,
    ) -> LatentTrainingCorpusPublicationReceipt:
        from memory_condense.eval._diffuse_latent_training_corpus_io import (
            publish_structural_corpus,
        )

        assert_public_lock()
        result = publish_structural_corpus(
            population,
            destination,
            row_mapper=row_mapper,
            expected=expected,
            population_status="locked_projection",
        )
        assert_public_lock()
        return result

    return publish_structural_latent_training_corpus


publish_structural_latent_training_corpus = _bind_locked_publisher()
del _bind_locked_publisher


def _publish_synthetic_structural_latent_training_corpus(
    population: AnalysisPopulationProjection,
    destination: str | Path,
    *,
    row_mapper: RouteV2PopulationMapper,
    expected: LatentTrainingPopulationExpectation,
) -> LatentTrainingCorpusPublicationReceipt:
    from memory_condense.eval._diffuse_latent_training_corpus_io import (
        publish_structural_corpus,
    )

    return publish_structural_corpus(
        population,
        destination,
        row_mapper=row_mapper,
        expected=expected,
        population_status="synthetic_projection",
    )


def verify_structural_latent_training_corpus(
    path: str | Path,
) -> VerifiedLatentTrainingFullCorpus:
    from memory_condense.eval._diffuse_latent_training_corpus_io import (
        verify_structural_corpus,
    )

    return verify_structural_corpus(path)


def verify_structural_latent_training_fit_corpus(
    path: str | Path,
) -> VerifiedLatentTrainingFitCorpus:
    return verify_structural_latent_training_corpus(path).fit


def verify_structural_latent_training_validation_corpus(
    path: str | Path,
) -> VerifiedLatentTrainingValidationCorpus:
    return verify_structural_latent_training_corpus(path).validation


__all__ = [
    "ANALYSIS_POPULATION_PROJECTION_FORMAT",
    "LATENT_TRAINING_CORPUS_FORMAT",
    "LATENT_TRAINING_PARTITION_FORMAT",
    "LATENT_TRAINING_PUBLICATION_FORMAT",
    "LATENT_TRAINING_ROW_FORMAT",
    "LOCKED_LATENT_TRAINING_POPULATION",
    "AnalysisPopulationProjection",
    "AnalysisPopulationRow",
    "DecodedLatentTrainingCorpusRow",
    "DecodedLatentTrainingPayload",
    "LatentTrainingCorpusError",
    "LatentTrainingCorpusManifest",
    "LatentTrainingCorpusPartitionManifest",
    "LatentTrainingCorpusPublicationReceipt",
    "LatentTrainingCorpusRowManifest",
    "LatentTrainingFileIdentity",
    "LatentTrainingPopulationExpectation",
    "LatentTrainingRouteEvidence",
    "RouteV2PopulationMapper",
    "StructuralRouteV2MappedRow",
    "VerifiedLatentTrainingFitCorpus",
    "VerifiedLatentTrainingFullCorpus",
    "VerifiedLatentTrainingValidationCorpus",
    "latent_training_corpus_implementation_sha256",
    "publish_structural_latent_training_corpus",
    "verify_structural_latent_training_corpus",
    "verify_structural_latent_training_fit_corpus",
    "verify_structural_latent_training_validation_corpus",
]
