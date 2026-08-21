"""Route-bearing v2 wrapper for gold-blind diffuse LongMemEval retrieval.

The existing diffuse query, analysis, and replay receipts remain v1 and are
deliberately route-blind.  This module adds a separate, text-free authority for
the already implemented ``episode_primary`` path.  It independently rebuilds
the route hash and both closure witnesses from the returned immutable objects;
it never upgrades a bare v1 receipt by assertion.

No responder, judge, scorer-label input, provider loader, Torch, Transformers,
corpus serializer, or training implementation is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import memory_condense.eval.diffuse_longmemeval as diffuse_module
import memory_condense.eval.diffuse_longmemeval_analysis as analysis_module
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    ClosureScopeWitness,
    DiscourseSnapshot,
    EvidenceObligation,
    EpisodeSeed,
    EvidenceAtom,
    EvidenceBundle,
    EvidencePacket,
    ObligationResult,
    QueryProgram,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval._diffuse_route_v2_validation import (
    assert_current_identity as _assert_current_identity,
    assert_exact_value as _assert_exact_value,
    assert_same_identity as _assert_same_identity,
    assert_same_identity_sequence as _assert_same_identity_sequence,
    bind_route_v2_dependency_guard as _bind_route_v2_dependency_guard,
    episode_seed_payload as _seed_payload,
    freeze_loaded_callable as _freeze_loaded_callable,
    require_exact as _require_exact,
    require_exact_scalar as _require_exact_scalar,
    require_exact_string_tuple as _require_exact_string_tuple,
    require_exact_tuple as _require_exact_tuple,
    route_v2_implementation_sha256,
    seed_projection_sha256 as _seed_projection_sha256,
    validate_closure_stopping_state as _validate_closure_stopping_state,
    validate_compilation_receipt as _validate_compilation_receipt,
    validate_evidence_atom as _validate_evidence_atom,
    validate_evidence_bundle as _validate_evidence_bundle,
    validate_episode_seed as _validate_episode_seed,
    validate_legacy_source_scope as _validate_legacy_source_scope,
)
from memory_condense.eval._identity import exact_int, sha256_digest
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.diffuse_longmemeval import (
    LongMemEvalDiffuseQueryReceipt,
    LongMemEvalDiffuseRetrieval,
)
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalAnalysisQueryReceipt,
    DiffuseLongMemEvalArm,
    DiffuseLongMemEvalGoldBlindQuery,
    DiffuseLongMemEvalRetrievalPhase,
    LegacyDiffuseInputProvider,
    RepresentativePolicyFactory,
    _retrieve_diffuse_longmemeval_sample_with_route,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    ExactLegacyDiffuseInputs,
    GoldBlindLongMemEvalQuestion,
    GoldBlindLongMemEvalSample,
    LegacyDiffuseCandidates,
    LegacyDiffuseInputReceipt,
)
from memory_condense.eval.schemas import EvalConfig
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPlan,
    EpisodeRepresentativeWitness,
    EpisodeRetrievalPlan,
    EpisodeRetrievalPolicy,
    EpisodeSourceScan,
    NestedEpisodeLinker,
    QwenAttentionHeadSurpriseScorer,
)


EPISODE_PRIMARY_ANALYSIS_ARM_V2_FORMAT = (
    "memory-condense-longmemeval-episode-primary-analysis-arm-v2"
)
EPISODE_PRIMARY_ROUTE_RECEIPT_V2_FORMAT = (
    "memory-condense-longmemeval-episode-primary-route-receipt-v2"
)
EPISODE_PRIMARY_ANALYSIS_QUERY_V2_FORMAT = (
    "memory-condense-longmemeval-episode-primary-analysis-query-v2"
)
EPISODE_PRIMARY_ANALYSIS_PHASE_V2_FORMAT = (
    "memory-condense-longmemeval-episode-primary-analysis-phase-v2"
)


@dataclass(frozen=True, slots=True)
class EpisodePrimaryAnalysisArmV2(SealedIdentity):
    """One exact v1 analysis arm plus the non-legacy route authority."""

    _SEAL_FIELD = "arm_sha256"
    _SEAL_MISMATCH = "episode-primary analysis arm does not match its contents"

    base_arm: DiffuseLongMemEvalArm
    episodic_route: Literal["episode_primary"] = "episode_primary"
    closure_routing_scope: Literal["seeded_graph"] = "seeded_graph"
    format: str = EPISODE_PRIMARY_ANALYSIS_ARM_V2_FORMAT
    arm_sha256: str = ""

    def __post_init__(self) -> None:
        _require_exact(self.base_arm, DiffuseLongMemEvalArm, "base analysis arm")
        _require_exact(
            self.base_arm.compilation,
            DiffuseCompilationPolicy,
            "base compilation policy",
        )
        _require_exact(
            self.base_arm.episode,
            EpisodeRetrievalPolicy,
            "base episode policy",
        )
        _require_exact(
            self.base_arm.closure,
            ClosurePolicy,
            "base closure policy",
        )
        if self.format != EPISODE_PRIMARY_ANALYSIS_ARM_V2_FORMAT:
            raise ValueError("unsupported episode-primary analysis arm format")
        if self.episodic_route != "episode_primary":
            raise ValueError("v2 analysis arm requires episode_primary")
        if self.closure_routing_scope != "seeded_graph":
            raise ValueError("episode_primary requires seeded_graph closure")
        if self.base_arm.require_owned_representative_runtime is not True:
            raise ValueError(
                "episode-primary v2 requires an owned representative runtime"
            )
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": self.format,
            "base_arm_sha256": self.base_arm.arm_sha256,
            "episodic_route": self.episodic_route,
            "closure_routing_scope": self.closure_routing_scope,
        }
        if include_receipt:
            payload["arm_sha256"] = self.arm_sha256
        return payload


@dataclass(frozen=True, slots=True)
class EpisodePrimaryRouteReceiptV2(SealedIdentity):
    """Text-free structural proof that one v1 query followed episode_primary.

    The receipt binds returned plans and witnesses, not runtime invocation
    counts.  Exact call-count behavior belongs to tests of the owned retrieval
    entry point and cannot be inferred by certifying an already-built phase.
    """

    _SEAL_MISMATCH = "episode-primary route receipt does not match its contents"

    analysis_arm_v2_sha256: str
    route_v2_implementation_sha256: str
    inner_analysis_query_receipt_sha256: str
    inner_diffuse_query_receipt_sha256: str
    legacy_input_receipt_sha256: str
    artifact_id: str
    snapshot_sha256: str
    retrieval_query_sha256: str
    source_candidate_scope_receipt_sha256: str
    source_scope_exhaustive: bool
    episodic_route: Literal["episode_primary"]
    closure_routing_scope: Literal["seeded_graph"]
    direct_expansion_receipt_sha256: str
    direct_seed_count: int
    direct_fallback_count: int
    direct_truncated_episode_count: int
    direct_truncated_chunk_count: int
    representative_expansion_receipt_sha256: str
    representative_linker_identity_sha256: str
    representative_policy_sha256: str
    representative_runtime_binding_certified: bool
    representative_returned_plan_transformer_state_bytes: int
    representative_scope_exhaustive: bool
    representative_seed_count: int
    representative_seed_projection_sha256: str
    closure_seed_projection_sha256: str
    closure_direct_chunk_count: int
    combined_expansion_sha256: str
    expansion_exhaustive: bool
    closure_policy_sha256: str
    closure_max_frontier: int
    closure_plan_sha256: str
    closure_routing_scope_witness_sha256: str
    closure_routing_scope_witness_count: int
    episode_expansion_witness_sha256: str
    episode_expansion_witness_count: int
    artifact_global_routes_admitted: bool
    artifact_unit_scan_witness_count: int
    closure_scope_exhaustive: bool
    query_program_sha256: str
    packet_receipt_sha256: str
    context_sha256: str
    prompt_messages_sha256: str
    packet_retained_request_token_state_bytes: int
    store_retained_request_token_state_bytes: int
    format: str = EPISODE_PRIMARY_ROUTE_RECEIPT_V2_FORMAT
    receipt_sha256: str = ""

    def __post_init__(
        self,
        _implementation_observer: Any = route_v2_implementation_sha256,
    ) -> None:
        if self.format != EPISODE_PRIMARY_ROUTE_RECEIPT_V2_FORMAT:
            raise ValueError("unsupported episode-primary route receipt format")
        if type(self.artifact_id) is not str or not self.artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        for name in (
            "analysis_arm_v2_sha256",
            "route_v2_implementation_sha256",
            "inner_analysis_query_receipt_sha256",
            "inner_diffuse_query_receipt_sha256",
            "legacy_input_receipt_sha256",
            "snapshot_sha256",
            "retrieval_query_sha256",
            "source_candidate_scope_receipt_sha256",
            "direct_expansion_receipt_sha256",
            "representative_expansion_receipt_sha256",
            "representative_linker_identity_sha256",
            "representative_policy_sha256",
            "representative_seed_projection_sha256",
            "closure_seed_projection_sha256",
            "combined_expansion_sha256",
            "closure_policy_sha256",
            "closure_plan_sha256",
            "closure_routing_scope_witness_sha256",
            "episode_expansion_witness_sha256",
            "query_program_sha256",
            "packet_receipt_sha256",
            "context_sha256",
            "prompt_messages_sha256",
        ):
            sha256_digest(getattr(self, name), name)
        if self.route_v2_implementation_sha256 != _implementation_observer():
            raise ValueError("route-v2 implementation identity changed")
        for name in (
            "source_scope_exhaustive",
            "representative_runtime_binding_certified",
            "representative_scope_exhaustive",
            "expansion_exhaustive",
            "artifact_global_routes_admitted",
            "closure_scope_exhaustive",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.episodic_route != "episode_primary":
            raise ValueError("route receipt requires episode_primary")
        if self.closure_routing_scope != "seeded_graph":
            raise ValueError("route receipt requires seeded_graph closure")
        exact_counts = {
            "direct_seed_count": 0,
            "direct_fallback_count": 0,
            "direct_truncated_episode_count": 0,
            "direct_truncated_chunk_count": 0,
            "representative_returned_plan_transformer_state_bytes": 0,
            "closure_direct_chunk_count": 0,
            "closure_routing_scope_witness_count": 1,
            "episode_expansion_witness_count": 1,
            "artifact_unit_scan_witness_count": 0,
            "packet_retained_request_token_state_bytes": 0,
            "store_retained_request_token_state_bytes": 0,
        }
        for name, expected in exact_counts.items():
            value = exact_int(getattr(self, name), name, minimum=0)
            if value != expected:
                raise ValueError(f"{name} must equal {expected}")
        if exact_int(
            self.representative_seed_count,
            "representative_seed_count",
            minimum=1,
        ) < 1:
            raise ValueError("representative_seed_count must be positive")
        exact_int(self.closure_max_frontier, "closure_max_frontier", minimum=1)
        if self.representative_runtime_binding_certified is not True:
            raise ValueError("representative runtime must be owned")
        if self.artifact_global_routes_admitted is not False:
            raise ValueError("episode_primary cannot admit artifact-global routes")
        if self.closure_scope_exhaustive is not False:
            raise ValueError("seeded_graph closure cannot claim exhaustive scope")
        self._seal()


def _expected_route_receipt(
    query: DiffuseLongMemEvalGoldBlindQuery,
    arm: EpisodePrimaryAnalysisArmV2,
    *,
    implementation_sha256: str,
    _receipt_type: Any = EpisodePrimaryRouteReceiptV2,
) -> EpisodePrimaryRouteReceiptV2:
    _require_exact(query, DiffuseLongMemEvalGoldBlindQuery, "analysis query")
    _require_exact(arm, EpisodePrimaryAnalysisArmV2, "episode-primary arm")
    _require_exact(query.probe, GoldBlindLongMemEvalQuestion, "gold-blind probe")
    _require_exact(query.legacy_inputs, ExactLegacyDiffuseInputs, "legacy inputs")
    _require_exact(query.legacy_inputs.candidates, LegacyDiffuseCandidates, "candidates")
    _require_exact(query.legacy_inputs.receipt, LegacyDiffuseInputReceipt, "legacy receipt")
    _require_exact(query.retrieval, LongMemEvalDiffuseRetrieval, "diffuse retrieval")
    _require_exact(
        query.receipt,
        DiffuseLongMemEvalAnalysisQueryReceipt,
        "analysis query receipt",
    )
    _assert_current_identity(arm, "arm_sha256", "episode-primary arm")
    _assert_current_identity(
        query.legacy_inputs.receipt,
        "receipt_sha256",
        "legacy input receipt",
    )
    _assert_current_identity(
        query.receipt,
        "receipt_sha256",
        "analysis query receipt",
    )

    candidates = query.legacy_inputs.candidates
    scope = _validate_legacy_source_scope(
        candidates,
        query.legacy_inputs.receipt,
    )

    retrieval = query.retrieval
    expansion = _require_exact(
        retrieval.expansion,
        EpisodeRetrievalPlan,
        "direct expansion",
    )
    representative = retrieval.representative_expansion
    if representative is None:
        raise ValueError("episode-primary query lacks representative retrieval")
    _require_exact(
        representative,
        EpisodeRepresentativeRetrievalPlan,
        "representative expansion",
    )
    plan = _require_exact(retrieval.plan, ClosurePlan, "closure plan")
    packet = _require_exact(retrieval.packet, EvidencePacket, "evidence packet")
    packet_receipt = _require_exact(
        packet.receipt,
        ClosureReceipt,
        "closure receipt",
    )
    diffuse_receipt = _require_exact(
        retrieval.receipt,
        LongMemEvalDiffuseQueryReceipt,
        "diffuse query receipt",
    )
    analysis_receipt = query.receipt
    _assert_current_identity(expansion, "receipt_sha256", "direct expansion")
    _assert_current_identity(
        representative,
        "receipt_sha256",
        "representative expansion",
    )
    _assert_current_identity(plan, "plan_sha256", "closure plan")
    _assert_current_identity(packet_receipt, "receipt_sha256", "closure receipt")
    _assert_current_identity(
        diffuse_receipt,
        "receipt_sha256",
        "diffuse query receipt",
    )
    if analysis_receipt.question_probe_sha256 != query.probe.probe_sha256:
        raise ValueError("analysis query changed its gold-blind probe")

    _require_exact(plan.query_program, QueryProgram, "query program")
    _require_exact_tuple(
        plan.query_program.obligations,
        EvidenceObligation,
        "query obligations",
    )
    _require_exact(plan.policy, ClosurePolicy, "closure policy")
    _require_exact(plan.snapshot, DiscourseSnapshot, "closure snapshot")
    _require_exact_tuple(plan.seeds, EpisodeSeed, "closure seeds")
    _require_exact_tuple(plan.atoms, EvidenceAtom, "closure atoms")
    _require_exact_tuple(plan.bundles, EvidenceBundle, "closure bundles")
    _require_exact_tuple(
        plan.obligation_results,
        ObligationResult,
        "closure obligation results",
    )
    _require_exact_tuple(
        plan.scope_witnesses,
        ClosureScopeWitness,
        "closure scope witnesses",
    )
    _assert_current_identity(
        plan.query_program,
        "program_sha256",
        "query program",
    )
    _assert_current_identity(
        plan.snapshot,
        "snapshot_sha256",
        "closure snapshot",
    )
    for witness in plan.scope_witnesses:
        _assert_current_identity(
            witness,
            "witness_sha256",
            "closure scope witness",
        )
    _require_exact_tuple(packet.atoms, EvidenceAtom, "packet atoms")
    _require_exact_tuple(packet.bundles, EvidenceBundle, "packet bundles")
    for index, seed in enumerate(plan.seeds):
        _validate_episode_seed(seed, f"closure seeds[{index}]")
    for prefix, atoms in (("closure atoms", plan.atoms), ("packet atoms", packet.atoms)):
        for index, atom in enumerate(atoms):
            _validate_evidence_atom(atom, f"{prefix}[{index}]")
    for prefix, bundles in (
        ("closure bundles", plan.bundles),
        ("packet bundles", packet.bundles),
    ):
        for index, bundle in enumerate(bundles):
            _validate_evidence_bundle(bundle, f"{prefix}[{index}]")
    _require_exact_tuple(representative.seeds, EpisodeSeed, "representative seeds")
    for index, seed in enumerate(representative.seeds):
        _validate_episode_seed(seed, f"representative seeds[{index}]")
    _require_exact_tuple(
        representative.source_scans,
        EpisodeSourceScan,
        "representative source scans",
    )
    _require_exact_tuple(
        representative.candidate_witnesses,
        EpisodeRepresentativeWitness,
        "representative candidate witnesses",
    )

    if analysis_receipt.analysis_arm_sha256 != arm.base_arm.arm_sha256:
        raise ValueError("v2 arm does not bind the nested v1 analysis arm")
    if analysis_receipt.matched_controls_sha256 != (
        arm.base_arm.matched_controls_sha256
    ):
        raise ValueError("v2 arm changed a nested matched control")
    if analysis_receipt.legacy_input_receipt_sha256 != (
        query.legacy_inputs.receipt.receipt_sha256
    ) or analysis_receipt.diffuse_query_receipt_sha256 != (
        diffuse_receipt.receipt_sha256
    ):
        raise ValueError("analysis query does not bind its nested receipts")
    if (
        analysis_receipt.artifact_id != diffuse_receipt.artifact_id
        or analysis_receipt.snapshot_sha256 != diffuse_receipt.snapshot_sha256
    ):
        raise ValueError("analysis query changed artifact or snapshot")
    if analysis_receipt.representative_linker_identity_sha256 != (
        representative.linker_identity_sha256
    ):
        raise ValueError("analysis query changed the representative linker")
    if analysis_receipt.representative_policy_sha256 != representative.policy_sha256:
        raise ValueError("analysis query changed the representative policy")
    if (
        analysis_receipt.representative_policy_factory_identity_sha256 is None
        or analysis_receipt.representative_policy_controls_sha256 is None
    ):
        raise ValueError("analysis query lacks representative call-time identity")

    query_sha256 = identity_sha256({"query": query.probe.retrieval_query})
    prompt_question_sha256 = identity_sha256(
        {"prompt_question": query.probe.prompt_question}
    )
    if query.legacy_inputs.receipt.query_sha256 != query_sha256 or (
        diffuse_receipt.retrieval_query_sha256 != query_sha256
    ):
        raise ValueError("episode-primary inputs belong to another query")
    if diffuse_receipt.prompt_question_sha256 != prompt_question_sha256:
        raise ValueError("episode-primary prompt question identity changed")
    if (
        scope.artifact_id != diffuse_receipt.artifact_id
        or scope.snapshot_sha256 != diffuse_receipt.snapshot_sha256
        or scope.query_sha256 != query_sha256
        or query.legacy_inputs.receipt.artifact_id != diffuse_receipt.artifact_id
        or query.legacy_inputs.receipt.source_candidate_scope_receipt_sha256
        != scope.receipt_sha256
    ):
        raise ValueError("source candidate scope changed artifact/query/snapshot")
    if representative.source_scope_receipt_sha256 != scope.receipt_sha256:
        raise ValueError("representative retrieval changed the source scope")
    _require_exact_scalar(
        representative.source_universe_exhaustive,
        bool,
        "representative source_universe_exhaustive",
    )
    _assert_exact_value(
        representative.source_universe_exhaustive,
        scope.selected_scope_exhaustive,
        "representative source exhaustiveness",
    )
    if (
        representative.artifact_id != diffuse_receipt.artifact_id
        or representative.query_sha256 != query_sha256
        or representative.policy_sha256
        != analysis_receipt.representative_policy_sha256
        or representative.linker_identity_sha256
        != analysis_receipt.representative_linker_identity_sha256
    ):
        raise ValueError("representative retrieval identity changed")
    if representative.runtime_binding_certified is not True:
        raise ValueError("representative retrieval runtime is not owned")
    if exact_int(
        representative.returned_plan_transformer_state_bytes,
        "representative returned_plan_transformer_state_bytes",
        minimum=0,
    ) != 0:
        raise ValueError("representative plan retained transformer state")
    if not representative.seeds:
        raise ValueError("episode-primary representative route is empty")

    if (
        expansion.seeds
        or expansion.direct_fallbacks
        or expansion.truncated_episode_ids
        or expansion.truncated_direct_chunk_ids
    ):
        raise ValueError("episode-primary executed or populated the direct route")
    if expansion.policy_sha256 != diffuse_receipt.episode_policy_sha256 or (
        expansion.receipt_sha256 != diffuse_receipt.expansion_receipt_sha256
    ):
        raise ValueError("empty direct expansion identity changed")
    if diffuse_receipt.truncated_episode_ids or (
        diffuse_receipt.truncated_direct_chunk_ids
    ):
        raise ValueError("episode-primary v1 direct truncation fields must be empty")
    if diffuse_receipt.representative_receipt_sha256 != representative.receipt_sha256:
        raise ValueError("diffuse receipt changed representative expansion")
    if diffuse_receipt.representative_runtime_binding_certified is not True:
        raise ValueError("diffuse receipt lacks owned zero-state retrieval")
    if _require_exact_scalar(
        diffuse_receipt.representative_returned_plan_transformer_state_bytes,
        int,
        "diffuse receipt representative returned transformer state bytes",
    ) != 0:
        raise ValueError("diffuse receipt lacks owned zero-state retrieval")
    _assert_exact_value(
        diffuse_receipt.representative_scope_exhaustive,
        representative.candidate_scope_exhaustive,
        "representative scope exhaustiveness",
    )
    expected_seed_episode_ids = tuple(
        seed.episode_id for seed in representative.seeds
    )
    _require_exact_string_tuple(
        diffuse_receipt.representative_seed_episode_ids,
        "diffuse receipt representative_seed_episode_ids",
    )
    _assert_exact_value(
        diffuse_receipt.representative_seed_episode_ids,
        expected_seed_episode_ids,
        "representative seed IDs",
    )

    representative_seeds = tuple(representative.seeds)
    closure_seeds = tuple(plan.seeds)
    _assert_same_identity_sequence(
        closure_seeds,
        representative_seeds,
        EpisodeSeed,
        "closure representative seeds",
    )
    if plan.direct_chunk_ids:
        raise ValueError("episode-primary closure admitted direct chunks")
    representative_seed_sha256 = _seed_projection_sha256(representative_seeds)
    closure_seed_sha256 = _seed_projection_sha256(closure_seeds)
    combined_expansion_sha256 = identity_sha256(
        {
            "episodic_route": "episode_primary",
            "direct_expansion_receipt_sha256": expansion.receipt_sha256,
            "representative_expansion_receipt_sha256": (
                representative.receipt_sha256
            ),
            "seeds": [_seed_payload(seed) for seed in representative_seeds],
            "direct_chunk_ids": [],
        }
    )
    expansion_exhaustive = bool(representative.candidate_scope_exhaustive)
    if (
        combined_expansion_sha256 != diffuse_receipt.combined_expansion_sha256
        or plan.expansion_receipt_sha256 != combined_expansion_sha256
    ):
        raise ValueError("episode-primary combined expansion identity changed")
    _assert_exact_value(
        diffuse_receipt.expansion_exhaustive,
        expansion_exhaustive,
        "episode-primary expansion exhaustiveness",
    )

    if (
        plan.artifact_id != diffuse_receipt.artifact_id
        or plan.snapshot.snapshot_sha256 != diffuse_receipt.snapshot_sha256
        or plan.policy.policy_sha256 != diffuse_receipt.closure_policy_sha256
        or plan.policy.policy_sha256 != arm.base_arm.closure.policy_sha256
        or plan.query_program.program_sha256 != diffuse_receipt.query_program_sha256
        or plan.plan_sha256 != diffuse_receipt.closure_plan_sha256
    ):
        raise ValueError("closure plan changed the route-bound query identity")
    witness_sha256s = tuple(
        witness.witness_sha256 for witness in plan.scope_witnesses
    )
    if witness_sha256s != diffuse_receipt.scope_witness_sha256s:
        raise ValueError("diffuse receipt changed the closure witness sequence")

    routing_witnesses = tuple(
        witness
        for witness in plan.scope_witnesses
        if witness.kind == "closure_routing_scope"
    )
    if len(routing_witnesses) != 1:
        raise ValueError("episode-primary requires exactly one routing witness")
    expected_routing_witness = ClosureScopeWitness(
        kind="closure_routing_scope",
        subject_id="seeded_graph",
        requested_limit=plan.policy.max_frontier * 2,
        returned_count=len(representative_seeds),
        exhaustive=False,
        detail={
            "artifact_global_routes_admitted": False,
            "seed_count": len(representative_seeds),
            "direct_chunk_count": 0,
        },
    )
    routing_witness = routing_witnesses[0]
    _assert_same_identity(
        routing_witness,
        expected_routing_witness,
        "seeded_graph routing witness",
    )

    expansion_witnesses = tuple(
        witness
        for witness in plan.scope_witnesses
        if witness.kind == "episode_expansion"
    )
    if len(expansion_witnesses) != 1:
        raise ValueError("episode-primary requires exactly one expansion witness")
    expected_expansion_witness = ClosureScopeWitness(
        kind="episode_expansion",
        subject_id=combined_expansion_sha256,
        requested_limit=None,
        returned_count=len(representative_seeds),
        exhaustive=expansion_exhaustive,
        detail={
            "seed_count": len(representative_seeds),
            "direct_chunk_count": 0,
            "receipt_sha256": combined_expansion_sha256,
            "exhaustiveness_attested": True,
        },
    )
    expansion_witness = expansion_witnesses[0]
    _assert_same_identity(
        expansion_witness,
        expected_expansion_witness,
        "episode expansion witness",
    )
    artifact_scan_count = sum(
        witness.kind == "artifact_unit_scan" for witness in plan.scope_witnesses
    )
    if artifact_scan_count:
        raise ValueError("episode-primary closure admitted an artifact-global scan")
    closure_scope_exhaustive = bool(
        plan.scope_witnesses
        and all(witness.exhaustive for witness in plan.scope_witnesses)
    )
    _assert_exact_value(
        diffuse_receipt.closure_scope_exhaustive,
        False,
        "diffuse closure scope exhaustiveness",
    )
    if closure_scope_exhaustive:
        raise ValueError("seeded_graph route cannot claim exhaustive closure scope")

    plan_atoms = {atom.atom_id: atom for atom in plan.atoms}
    plan_bundles = {bundle.bundle_id: bundle for bundle in plan.bundles}
    try:
        expected_packet_atoms = tuple(
            plan_atoms[atom_id] for atom_id in packet_receipt.selected_atom_ids
        )
        expected_packet_bundles = tuple(
            plan_bundles[bundle_id]
            for bundle_id in packet_receipt.selected_bundle_ids
        )
    except KeyError as exc:
        raise ValueError("packet selected an object absent from its closure plan") from exc
    _assert_same_identity_sequence(
        packet.atoms,
        expected_packet_atoms,
        EvidenceAtom,
        "packet atoms",
    )
    _assert_same_identity_sequence(
        packet.bundles,
        expected_packet_bundles,
        EvidenceBundle,
        "packet bundles",
    )
    if (
        packet_receipt.plan_sha256 != plan.plan_sha256
        or packet_receipt.receipt_sha256 != diffuse_receipt.packet_receipt_sha256
        or packet_receipt.context_sha256 != diffuse_receipt.context_sha256
        or quote_sha256(packet.context) != diffuse_receipt.context_sha256
    ):
        raise ValueError("packet receipt does not bind the route closure plan")
    _validate_closure_stopping_state(diffuse_receipt, plan, packet_receipt)

    if type(retrieval.messages) is not tuple or any(
        type(message) is not dict
        or set(message) != {"role", "content"}
        or any(type(value) is not str for value in message.values())
        for message in retrieval.messages
    ):
        raise TypeError("provider messages must be exact role/content dictionaries")
    prompt_messages_sha256 = identity_sha256(list(retrieval.messages))
    if (
        prompt_messages_sha256 != diffuse_receipt.prompt_messages_sha256
        or packet_receipt.prompt_messages_sha256 != prompt_messages_sha256
    ):
        raise ValueError("packet prompt identity or accounting changed")
    for packet_name, diffuse_name in (
        ("prompt_token_proxy", "prompt_token_proxy"),
        ("responder_output_token_reserve", "responder_output_token_reserve"),
        ("prompt_workspace_token_proxy", "prompt_workspace_token_proxy"),
        ("max_prompt_token_proxy", "max_prompt_workspace_token_proxy"),
    ):
        packet_value = exact_int(
            getattr(packet_receipt, packet_name),
            f"packet receipt {packet_name}",
            minimum=0,
        )
        diffuse_value = exact_int(
            getattr(diffuse_receipt, diffuse_name),
            f"diffuse receipt {diffuse_name}",
            minimum=0,
        )
        if packet_value != diffuse_value:
            raise ValueError("packet prompt identity or accounting changed")

    coordinates = tuple(
        {
            "atom_id": atom.atom_id,
            **atom.span.identity_payload(),
            "label": atom.label,
        }
        for atom in packet.atoms
    )
    if type(retrieval.evidence_coordinates) is not tuple or any(
        type(coordinate) is not dict
        for coordinate in retrieval.evidence_coordinates
    ):
        raise TypeError("evidence coordinates must be exact dictionaries")
    _assert_exact_value(
        retrieval.evidence_coordinates,
        coordinates,
        "packet evidence coordinates",
    )
    actual_coordinates_sha256 = identity_sha256(retrieval.evidence_coordinates)
    if (
        actual_coordinates_sha256 != identity_sha256(coordinates)
        or actual_coordinates_sha256
        != diffuse_receipt.evidence_coordinates_sha256
    ):
        raise ValueError("packet evidence-coordinate identity changed")
    if exact_int(
        diffuse_receipt.packet_retained_request_token_state_bytes,
        "packet retained request token state bytes",
        minimum=0,
    ) != 0 or exact_int(
        diffuse_receipt.store_retained_request_token_state_bytes,
        "store retained request token state bytes",
        minimum=0,
    ) != 0:
        raise ValueError("route-bound query retained request token state")

    return _receipt_type(
        analysis_arm_v2_sha256=arm.arm_sha256,
        route_v2_implementation_sha256=implementation_sha256,
        inner_analysis_query_receipt_sha256=analysis_receipt.receipt_sha256,
        inner_diffuse_query_receipt_sha256=diffuse_receipt.receipt_sha256,
        legacy_input_receipt_sha256=query.legacy_inputs.receipt.receipt_sha256,
        artifact_id=diffuse_receipt.artifact_id,
        snapshot_sha256=diffuse_receipt.snapshot_sha256,
        retrieval_query_sha256=query_sha256,
        source_candidate_scope_receipt_sha256=scope.receipt_sha256,
        source_scope_exhaustive=scope.selected_scope_exhaustive,
        episodic_route="episode_primary",
        closure_routing_scope="seeded_graph",
        direct_expansion_receipt_sha256=expansion.receipt_sha256,
        direct_seed_count=len(expansion.seeds),
        direct_fallback_count=len(expansion.direct_fallbacks),
        direct_truncated_episode_count=len(expansion.truncated_episode_ids),
        direct_truncated_chunk_count=len(expansion.truncated_direct_chunk_ids),
        representative_expansion_receipt_sha256=representative.receipt_sha256,
        representative_linker_identity_sha256=representative.linker_identity_sha256,
        representative_policy_sha256=representative.policy_sha256,
        representative_runtime_binding_certified=(
            representative.runtime_binding_certified
        ),
        representative_returned_plan_transformer_state_bytes=(
            representative.returned_plan_transformer_state_bytes
        ),
        representative_scope_exhaustive=(
            representative.candidate_scope_exhaustive
        ),
        representative_seed_count=len(representative_seeds),
        representative_seed_projection_sha256=representative_seed_sha256,
        closure_seed_projection_sha256=closure_seed_sha256,
        closure_direct_chunk_count=len(plan.direct_chunk_ids),
        combined_expansion_sha256=combined_expansion_sha256,
        expansion_exhaustive=expansion_exhaustive,
        closure_policy_sha256=plan.policy.policy_sha256,
        closure_max_frontier=plan.policy.max_frontier,
        closure_plan_sha256=plan.plan_sha256,
        closure_routing_scope_witness_sha256=routing_witness.witness_sha256,
        closure_routing_scope_witness_count=len(routing_witnesses),
        episode_expansion_witness_sha256=expansion_witness.witness_sha256,
        episode_expansion_witness_count=len(expansion_witnesses),
        artifact_global_routes_admitted=False,
        artifact_unit_scan_witness_count=artifact_scan_count,
        closure_scope_exhaustive=closure_scope_exhaustive,
        query_program_sha256=plan.query_program.program_sha256,
        packet_receipt_sha256=packet_receipt.receipt_sha256,
        context_sha256=packet_receipt.context_sha256,
        prompt_messages_sha256=prompt_messages_sha256,
        packet_retained_request_token_state_bytes=(
            diffuse_receipt.packet_retained_request_token_state_bytes
        ),
        store_retained_request_token_state_bytes=(
            diffuse_receipt.store_retained_request_token_state_bytes
        ),
    )


_expected_route_receipt = _freeze_loaded_callable(
    _expected_route_receipt,
    "route-v2 expected receipt builder",
)


@dataclass(frozen=True, slots=True)
class EpisodePrimaryAnalysisQueryV2(SealedIdentity):
    """One genuine v1 query joined to its independently rebuilt route receipt."""

    _SEAL_FIELD = "record_sha256"
    _SEAL_MISMATCH = "episode-primary analysis query does not match its contents"

    inner: DiffuseLongMemEvalGoldBlindQuery
    analysis_arm: EpisodePrimaryAnalysisArmV2
    route_receipt: EpisodePrimaryRouteReceiptV2
    format: str = EPISODE_PRIMARY_ANALYSIS_QUERY_V2_FORMAT
    record_sha256: str = ""

    def __post_init__(
        self,
        _expected_receipt: Any = _expected_route_receipt,
        _implementation_observer: Any = route_v2_implementation_sha256,
        _route_receipt_type: Any = EpisodePrimaryRouteReceiptV2,
    ) -> None:
        _require_exact(self.inner, DiffuseLongMemEvalGoldBlindQuery, "inner query")
        _require_exact(
            self.analysis_arm,
            EpisodePrimaryAnalysisArmV2,
            "episode-primary arm",
        )
        _require_exact(
            self.route_receipt,
            _route_receipt_type,
            "route receipt",
        )
        _assert_current_identity(
            self.route_receipt,
            "receipt_sha256",
            "route receipt",
        )
        if self.format != EPISODE_PRIMARY_ANALYSIS_QUERY_V2_FORMAT:
            raise ValueError("unsupported episode-primary analysis query format")
        expected = _expected_receipt(
            self.inner,
            self.analysis_arm,
            implementation_sha256=_implementation_observer(),
        )
        _assert_same_identity(
            self.route_receipt,
            expected,
            "route receipt for nested query",
        )
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": self.format,
            "analysis_arm_v2_sha256": self.analysis_arm.arm_sha256,
            "inner_analysis_query_receipt_sha256": self.inner.receipt.receipt_sha256,
            "inner_diffuse_query_receipt_sha256": (
                self.inner.retrieval.receipt.receipt_sha256
            ),
            "route_receipt_sha256": self.route_receipt.receipt_sha256,
        }
        if include_receipt:
            payload["record_sha256"] = self.record_sha256
        return payload


@dataclass(frozen=True, slots=True)
class EpisodePrimaryRetrievalPhaseV2(SealedIdentity):
    """Text-free phase wrapper; the nested v1 phase keeps authoritative bodies."""

    _SEAL_MISMATCH = "episode-primary retrieval phase does not match its contents"

    inner: DiffuseLongMemEvalRetrievalPhase
    analysis_arm: EpisodePrimaryAnalysisArmV2
    questions: tuple[EpisodePrimaryAnalysisQueryV2, ...]
    format: str = EPISODE_PRIMARY_ANALYSIS_PHASE_V2_FORMAT
    receipt_sha256: str = ""

    def __post_init__(
        self,
        _expected_receipt: Any = _expected_route_receipt,
        _implementation_observer: Any = route_v2_implementation_sha256,
        _query_type: Any = EpisodePrimaryAnalysisQueryV2,
    ) -> None:
        _require_exact(
            self.inner,
            DiffuseLongMemEvalRetrievalPhase,
            "inner retrieval phase",
        )
        _require_exact(
            self.analysis_arm,
            EpisodePrimaryAnalysisArmV2,
            "episode-primary arm",
        )
        _validate_compilation_receipt(self.inner.compilation)
        _assert_current_identity(
            self.inner,
            "receipt_sha256",
            "inner retrieval phase",
        )
        _require_exact_tuple(
            self.inner.questions,
            DiffuseLongMemEvalGoldBlindQuery,
            "inner analysis queries",
        )
        rows = _require_exact_tuple(
            self.questions,
            _query_type,
            "episode-primary query records",
        )
        if self.format != EPISODE_PRIMARY_ANALYSIS_PHASE_V2_FORMAT:
            raise ValueError("unsupported episode-primary analysis phase format")
        _assert_same_identity(
            self.inner.arm,
            self.analysis_arm.base_arm,
            "nested analysis arm",
        )
        if len(rows) != len(self.inner.questions):
            raise ValueError("v2 phase changed nested query order or membership")
        for index, (row, inner_query) in enumerate(
            zip(rows, self.inner.questions)
        ):
            _assert_same_identity(
                row.analysis_arm,
                self.analysis_arm,
                f"v2 query[{index}] analysis arm",
            )
            if row.inner is not inner_query:
                raise ValueError("v2 phase changed nested query order or membership")
            inner_receipt = row.inner.receipt
            if (
                inner_receipt.corpus_sha256 != self.inner.corpus_sha256
                or inner_receipt.evaluation_policy_sha256
                != self.inner.evaluation_policy_sha256
                or inner_receipt.compilation_receipt_sha256
                != self.inner.compilation.receipt_sha256
            ):
                raise ValueError("v2 query changed its parent phase identity")
            if (
                inner_receipt.artifact_id
                != self.inner.compilation.artifact.artifact_id
                or inner_receipt.snapshot_sha256
                != self.inner.compilation.final_snapshot.snapshot_sha256
            ):
                raise ValueError("v2 query changed its compilation snapshot")
            expected = _expected_receipt(
                row.inner,
                self.analysis_arm,
                implementation_sha256=_implementation_observer(),
            )
            _assert_current_identity(
                row.route_receipt,
                "receipt_sha256",
                "v2 phase route receipt",
            )
            _assert_same_identity(
                row.route_receipt,
                expected,
                "v2 phase route receipt",
            )
            expected_record_sha256 = identity_sha256(
                row.identity_payload(include_receipt=False)
            )
            if row.record_sha256 != expected_record_sha256:
                raise ValueError("v2 phase contains a changed query record")
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": self.format,
            "inner_phase_receipt_sha256": self.inner.receipt_sha256,
            "analysis_arm_v2_sha256": self.analysis_arm.arm_sha256,
            "query_record_sha256s": [row.record_sha256 for row in self.questions],
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def _bind_certifier(
    expected_receipt: Any,
    implementation_observer: Any,
    query_type: Any,
    phase_type: Any,
) -> Any:
    """Bind certification builders and implementation observation immutably."""

    def certify_episode_primary_analysis_phase_v2(
        phase: DiffuseLongMemEvalRetrievalPhase,
        *,
        arm: EpisodePrimaryAnalysisArmV2,
    ) -> EpisodePrimaryRetrievalPhaseV2:
        """Structurally validate and wrap one frozen episode-primary phase."""

        _require_exact(phase, DiffuseLongMemEvalRetrievalPhase, "retrieval phase")
        _require_exact(arm, EpisodePrimaryAnalysisArmV2, "episode-primary arm")
        _validate_compilation_receipt(phase.compilation)
        _require_exact_tuple(
            phase.questions,
            DiffuseLongMemEvalGoldBlindQuery,
            "analysis queries",
        )
        _assert_current_identity(phase, "receipt_sha256", "retrieval phase")
        implementation = implementation_observer()
        rows = tuple(
            query_type(
                inner=query,
                analysis_arm=arm,
                route_receipt=expected_receipt(
                    query,
                    arm,
                    implementation_sha256=implementation,
                ),
            )
            for query in phase.questions
        )
        if implementation_observer() != implementation:
            raise RuntimeError("route-v2 implementation changed during certification")
        result = phase_type(inner=phase, analysis_arm=arm, questions=rows)
        if implementation_observer() != implementation:
            raise RuntimeError("route-v2 implementation changed during certification")
        return result

    return certify_episode_primary_analysis_phase_v2


certify_episode_primary_analysis_phase_v2 = _freeze_loaded_callable(
    _bind_certifier(
        _expected_route_receipt,
        route_v2_implementation_sha256,
        EpisodePrimaryAnalysisQueryV2,
        EpisodePrimaryRetrievalPhaseV2,
    ),
    "route-v2 certifier",
)
del _bind_certifier


def _bind_verifier(certifier: Any, phase_type: Any) -> Any:
    def verify_episode_primary_analysis_phase_v2(
        value: EpisodePrimaryRetrievalPhaseV2,
    ) -> EpisodePrimaryRetrievalPhaseV2:
        """Rebuild every receipt and return the exact verified wrapper."""

        _require_exact(value, phase_type, "route-v2 phase")
        _assert_current_identity(value, "receipt_sha256", "route-v2 phase")
        expected = certifier(value.inner, arm=value.analysis_arm)
        _assert_same_identity(value, expected, "episode-primary phase")
        return value

    return verify_episode_primary_analysis_phase_v2


verify_episode_primary_analysis_phase_v2 = _bind_verifier(
    certify_episode_primary_analysis_phase_v2,
    EpisodePrimaryRetrievalPhaseV2,
)
del _bind_verifier


def _bind_owned_retriever(
    analysis_retriever: Any,
    packet_retriever: Any,
    certifier: Any,
    implementation_observer: Any,
    expected_receipt: Any,
    freezer: Any,
) -> Any:
    """Close over immutable builder references instead of mutable pin globals."""

    analysis_executor = freezer(analysis_retriever, "route-v2 analysis core")
    packet_executor = freezer(packet_retriever, "route-v2 packet retriever")
    assert_owned_builder_dependencies, assert_packet_callable_unchanged = (
        _bind_route_v2_dependency_guard(
            analysis_module=analysis_module,
            diffuse_module=diffuse_module,
            route_globals=globals(),
            analysis_retriever=analysis_retriever,
            packet_retriever=packet_retriever,
            certifier=certifier,
            implementation_observer=implementation_observer,
            expected_receipt=expected_receipt,
        )
    )

    def retrieve_episode_primary_analysis_phase_v2(
        condenser: MemoryCondenser,
        sample: GoldBlindLongMemEvalSample,
        *,
        config: EvalConfig,
        arm: EpisodePrimaryAnalysisArmV2,
        legacy_input_provider: LegacyDiffuseInputProvider,
        representative_linker: NestedEpisodeLinker,
        representative_policy_factory: RepresentativePolicyFactory,
        qwen_scorer: QwenAttentionHeadSurpriseScorer | None = None,
        embedding_identity: Mapping[str, object] | None = None,
    ) -> EpisodePrimaryRetrievalPhaseV2:
        """Run the owned retrieval core once and structurally bind its route."""

        if not isinstance(condenser, MemoryCondenser):
            raise TypeError("episode-primary retrieval requires a MemoryCondenser")
        _require_exact(sample, GoldBlindLongMemEvalSample, "gold-blind sample")
        _require_exact(config, EvalConfig, "evaluation config")
        _require_exact(arm, EpisodePrimaryAnalysisArmV2, "episode-primary arm")
        if not callable(legacy_input_provider):
            raise TypeError("legacy_input_provider must be callable")
        if representative_linker is None:
            raise TypeError("representative_linker is required")
        if not callable(representative_policy_factory):
            raise TypeError("representative_policy_factory must be callable")

        assert_owned_builder_dependencies()
        implementation = implementation_observer()
        phase = analysis_executor(
            condenser,
            sample,
            config=config,
            arm=arm.base_arm,
            legacy_input_provider=legacy_input_provider,
            qwen_scorer=qwen_scorer,
            embedding_identity=embedding_identity,
            representative_linker=representative_linker,
            representative_policy_factory=representative_policy_factory,
            episodic_route="episode_primary",
            _packet_retriever=packet_executor,
            _packet_retriever_guard=assert_packet_callable_unchanged,
        )
        assert_owned_builder_dependencies()
        if implementation_observer() != implementation:
            raise RuntimeError("route-v2 implementation changed during retrieval")
        result = certifier(phase, arm=arm)
        assert_owned_builder_dependencies()
        if implementation_observer() != implementation:
            raise RuntimeError("route-v2 implementation changed during retrieval")
        return result

    return retrieve_episode_primary_analysis_phase_v2


retrieve_episode_primary_analysis_phase_v2 = _freeze_loaded_callable(
    _bind_owned_retriever(
        _retrieve_diffuse_longmemeval_sample_with_route,
        diffuse_module.retrieve_longmemeval_diffuse_packet,
        certify_episode_primary_analysis_phase_v2,
        route_v2_implementation_sha256,
        _expected_route_receipt,
        _freeze_loaded_callable,
    ),
    "owned route-v2 retriever",
)
del _bind_owned_retriever


__all__ = [
    "EPISODE_PRIMARY_ANALYSIS_ARM_V2_FORMAT",
    "EPISODE_PRIMARY_ANALYSIS_PHASE_V2_FORMAT",
    "EPISODE_PRIMARY_ANALYSIS_QUERY_V2_FORMAT",
    "EPISODE_PRIMARY_ROUTE_RECEIPT_V2_FORMAT",
    "EpisodePrimaryAnalysisArmV2",
    "EpisodePrimaryAnalysisQueryV2",
    "EpisodePrimaryRetrievalPhaseV2",
    "EpisodePrimaryRouteReceiptV2",
    "certify_episode_primary_analysis_phase_v2",
    "retrieve_episode_primary_analysis_phase_v2",
    "route_v2_implementation_sha256",
    "verify_episode_primary_analysis_phase_v2",
]
