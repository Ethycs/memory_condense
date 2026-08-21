"""Deep, guarded currentness validation for one derived retrieval phase."""

from __future__ import annotations

import memory_condense.eval._diffuse_route_v2_validation as _validation_module
import memory_condense.eval.diffuse_longmemeval_analysis as _analysis_module

from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    ClosureScopeWitness,
    DiscourseArtifact,
    DiscourseSnapshot,
    EvidencePacket,
    QueryProgram,
    identity_sha256,
    quote_sha256,
)
from memory_condense.eval._diffuse_base_contracts import (
    DiffuseBaseArtifactError,
    DiffuseDerivedStore,
)
from memory_condense.eval._diffuse_base_publication_guard import (
    freeze_callable_guard,
    freeze_namespace_guard,
)
from memory_condense.eval._diffuse_route_v2_validation import (
    assert_current_identity as _assert_current_identity,
    validate_compilation_receipt as _validate_compilation_receipt,
)
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.diffuse_compilation import (
    DiffuseCompilationReceipt,
    DiffuseSourceCompilationReceipt,
)
from memory_condense.eval.diffuse_longmemeval import (
    LongMemEvalDiffuseQueryReceipt,
    LongMemEvalDiffuseRetrieval,
)
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalAnalysisQueryReceipt,
    DiffuseLongMemEvalArm,
    DiffuseLongMemEvalGoldBlindQuery,
    DiffuseLongMemEvalRetrievalPhase,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    ExactLegacyDiffuseInputs,
    GoldBlindLongMemEvalQuestion,
    LegacyDiffuseInputReceipt,
)
from memory_condense.eval.diffuse_longmemeval_runtime import FrozenLegacyQueryInputs
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPlan,
    EpisodeRetrievalPlan,
    EpisodeSourceCandidateScope,
)


def _validate_nested_query(phase: DiffuseLongMemEvalRetrievalPhase, question) -> None:
    if (
        type(question) is not DiffuseLongMemEvalGoldBlindQuery
        or type(question.legacy_inputs) is not ExactLegacyDiffuseInputs
        or type(question.retrieval) is not LongMemEvalDiffuseRetrieval
        or type(question.receipt) is not DiffuseLongMemEvalAnalysisQueryReceipt
        or type(question.legacy_inputs.receipt) is not LegacyDiffuseInputReceipt
        or type(question.retrieval.receipt) is not LongMemEvalDiffuseQueryReceipt
    ):
        raise TypeError("final phase contains a replaced query receipt")
    for value, field, label in (
        (question.receipt, "receipt_sha256", "analysis query receipt"),
        (question.legacy_inputs.receipt, "receipt_sha256", "legacy input receipt"),
        (question.retrieval.receipt, "receipt_sha256", "diffuse query receipt"),
    ):
        _assert_current_identity(value, field, label)

    retrieval = question.retrieval
    expansion = retrieval.expansion
    representative = retrieval.representative_expansion
    plan = retrieval.plan
    packet = retrieval.packet
    scope = question.legacy_inputs.candidates.source_candidate_scope
    if (
        type(expansion) is not EpisodeRetrievalPlan
        or (
            representative is not None
            and type(representative) is not EpisodeRepresentativeRetrievalPlan
        )
        or type(plan) is not ClosurePlan
        or type(plan.query_program) is not QueryProgram
        or type(plan.policy) is not ClosurePolicy
        or type(plan.snapshot) is not DiscourseSnapshot
        or type(plan.scope_witnesses) is not tuple
        or any(type(item) is not ClosureScopeWitness for item in plan.scope_witnesses)
        or type(packet) is not EvidencePacket
        or type(packet.receipt) is not ClosureReceipt
        or type(retrieval.messages) is not tuple
        or type(retrieval.evidence_coordinates) is not tuple
        or (scope is not None and type(scope) is not EpisodeSourceCandidateScope)
    ):
        raise TypeError("final phase contains a replaced nested receipt")

    _assert_current_identity(expansion, "receipt_sha256", "episode expansion receipt")
    if representative is not None:
        _assert_current_identity(
            representative,
            "receipt_sha256",
            "representative expansion receipt",
        )
    if scope is not None:
        _assert_current_identity(
            scope,
            "receipt_sha256",
            "source candidate scope receipt",
        )
    _assert_current_identity(
        plan.query_program,
        "program_sha256",
        "closure query program",
    )
    _assert_current_identity(plan.snapshot, "snapshot_sha256", "closure snapshot")
    for witness in plan.scope_witnesses:
        _assert_current_identity(
            witness,
            "witness_sha256",
            "closure scope witness",
        )
    _assert_current_identity(plan, "plan_sha256", "closure plan")
    _assert_current_identity(
        packet.receipt,
        "receipt_sha256",
        "closure packet receipt",
    )

    query_receipt = retrieval.receipt
    representative_receipt = (
        None if representative is None else representative.receipt_sha256
    )
    representative_scope = (
        None if representative is None else representative.source_scope_receipt_sha256
    )
    scope_receipt = None if scope is None else scope.receipt_sha256
    if (
        question.receipt.question_probe_sha256 != question.probe.probe_sha256
        or question.receipt.legacy_input_receipt_sha256
        != question.legacy_inputs.receipt.receipt_sha256
        or question.receipt.diffuse_query_receipt_sha256
        != retrieval.receipt.receipt_sha256
        or question.receipt.compilation_receipt_sha256
        != phase.compilation.receipt_sha256
        or question.receipt.artifact_id != phase.compilation.artifact.artifact_id
        or question.receipt.snapshot_sha256
        != phase.compilation.final_snapshot.snapshot_sha256
        or query_receipt.expansion_receipt_sha256 != expansion.receipt_sha256
        or query_receipt.episode_policy_sha256 != expansion.policy_sha256
        or query_receipt.representative_receipt_sha256 != representative_receipt
        or scope_receipt != representative_scope
        or plan.expansion_receipt_sha256 != query_receipt.combined_expansion_sha256
        or plan.query_program.program_sha256 != query_receipt.query_program_sha256
        or plan.snapshot.snapshot_sha256 != query_receipt.snapshot_sha256
        or plan.snapshot != phase.compilation.final_snapshot
        or plan.policy.policy_sha256 != query_receipt.closure_policy_sha256
        or plan.plan_sha256 != query_receipt.closure_plan_sha256
        or plan.stopping_reason != query_receipt.closure_stopping_reason
        or plan.complete_claimed != query_receipt.closure_complete_claimed
        or tuple(item.witness_sha256 for item in plan.scope_witnesses)
        != query_receipt.scope_witness_sha256s
        or packet.receipt.plan_sha256 != plan.plan_sha256
        or packet.receipt.receipt_sha256 != query_receipt.packet_receipt_sha256
        or packet.receipt.context_sha256 != query_receipt.context_sha256
        or packet.receipt.prompt_messages_sha256
        != query_receipt.prompt_messages_sha256
        or quote_sha256(packet.context) != packet.receipt.context_sha256
        or packet.receipt.selected_atom_ids
        != tuple(item.atom_id for item in packet.atoms)
        or packet.receipt.selected_bundle_ids
        != tuple(item.bundle_id for item in packet.bundles)
        or identity_sha256(list(retrieval.messages))
        != query_receipt.prompt_messages_sha256
        or identity_sha256(retrieval.evidence_coordinates)
        != query_receipt.evidence_coordinates_sha256
    ):
        raise ValueError("final phase query receipt links changed")


def _validate_phase(clone: DiffuseDerivedStore, phase: object):
    if not isinstance(clone, DiffuseDerivedStore):
        raise TypeError("clone must be a DiffuseDerivedStore")
    if type(phase) is not DiffuseLongMemEvalRetrievalPhase:
        raise TypeError("phase must be an exact diffuse retrieval phase")
    if type(phase.arm) is not DiffuseLongMemEvalArm:
        raise DiffuseBaseArtifactError("final phase arm type changed")
    if type(phase.questions) is not tuple:
        raise DiffuseBaseArtifactError("final phase questions changed container type")
    try:
        _validate_compilation_receipt(phase.compilation)
        _assert_current_identity(phase, "receipt_sha256", "retrieval phase")
        for question in phase.questions:
            _validate_nested_query(phase, question)
    except (TypeError, ValueError) as exc:
        raise DiffuseBaseArtifactError(
            "final phase identity changed after retrieval"
        ) from exc
    if (
        phase.arm.arm_id != clone.origin.arm_id
        or phase.arm.arm_sha256 != clone.origin.arm_sha256
    ):
        raise DiffuseBaseArtifactError("final phase belongs to another clone arm")
    if (
        phase.corpus_sha256 != clone.base.store_manifest.corpus_sha256
        or any(
            item.receipt.snapshot_sha256
            != phase.compilation.final_snapshot.snapshot_sha256
            for item in phase.questions
        )
    ):
        raise DiffuseBaseArtifactError("final phase does not bind this base snapshot")
    if canonical_sha256(list(phase.deterministic_turn_ids)) != (
        clone.base.store_manifest.deterministic_turn_ids_sha256
    ):
        raise DiffuseBaseArtifactError("final phase changed deterministic ingest")
    rows = clone.base.frozen_query_inputs
    if type(rows) is not tuple:
        raise DiffuseBaseArtifactError("frozen query rows changed container type")
    try:
        for frozen in rows:
            if type(frozen) is not FrozenLegacyQueryInputs:
                raise TypeError("frozen query row type changed")
            _assert_current_identity(frozen, "receipt_sha256", "frozen query receipt")
    except (TypeError, ValueError) as exc:
        raise DiffuseBaseArtifactError("frozen query receipt changed") from exc
    if canonical_sha256([item.receipt_sha256 for item in rows]) != (
        clone.base.query_manifest.frozen_receipts_sha256
    ):
        raise DiffuseBaseArtifactError("frozen query receipt set changed")
    expected_probes = tuple(clone.base._sample.questions)
    observed_probes = tuple(item.probe for item in phase.questions)
    if observed_probes != expected_probes or len(observed_probes) != (
        clone.base.query_manifest.query_count
    ):
        raise DiffuseBaseArtifactError("final phase changed the frozen query set")
    if len(rows) != len(phase.questions):
        raise DiffuseBaseArtifactError("frozen query rows are incomplete")
    for frozen, question in zip(rows, phase.questions, strict=True):
        frozen_identity = frozen.identity_payload(include_receipt=False)
        receipt = question.legacy_inputs.receipt
        if (
            receipt.query_sha256 != frozen_identity["query_sha256"]
            or receipt.retrieval_policy_sha256 != frozen.retrieval_policy_sha256
            or receipt.anchor_chunk_ids
            != tuple(item.chunk.chunk_id for item in frozen.anchors)
            or question.legacy_inputs.candidates.anchors != frozen.anchors
        ):
            raise DiffuseBaseArtifactError(
                "final phase changed a frozen query or anchor row"
            )
    return phase


def _seal_phase_validator(implementation, namespace_guard):
    def descriptor_function(owner, name):
        raw = owner.__dict__.get(name)
        if isinstance(raw, property):
            return raw.fget
        value = getattr(owner, name, None)
        return getattr(value, "__func__", value)

    helper_values = (
        (_validation_module, "assert_current_identity", _assert_current_identity),
        (
            _validation_module,
            "validate_compilation_receipt",
            _validate_compilation_receipt,
        ),
    )
    descriptor_specs = (
        (DiffuseLongMemEvalRetrievalPhase, "identity_payload"),
        (DiffuseLongMemEvalArm, "identity_payload"),
        (DiffuseLongMemEvalArm, "arm_sha256"),
        (GoldBlindLongMemEvalQuestion, "probe_sha256"),
        (DiffuseCompilationReceipt, "identity_payload"),
        (DiffuseSourceCompilationReceipt, "identity_payload"),
        (DiscourseArtifact, "identity_payload"),
        (DiscourseSnapshot, "identity_payload"),
        (DiffuseLongMemEvalAnalysisQueryReceipt, "identity_payload"),
        (LegacyDiffuseInputReceipt, "identity_payload"),
        (LongMemEvalDiffuseQueryReceipt, "identity_payload"),
        (FrozenLegacyQueryInputs, "identity_payload"),
        (EpisodeRetrievalPlan, "identity_payload"),
        (EpisodeRepresentativeRetrievalPlan, "identity_payload"),
        (EpisodeSourceCandidateScope, "identity_payload"),
        (QueryProgram, "identity_payload"),
        (ClosurePolicy, "policy_sha256"),
        (ClosureScopeWitness, "identity_payload"),
        (ClosurePlan, "identity_payload"),
        (ClosureReceipt, "identity_payload"),
    )
    descriptor_values = tuple(
        (owner, name, descriptor_function(owner, name))
        for owner, name in descriptor_specs
    )
    guarded_values = tuple(value for _owner, _name, value in helper_values) + tuple(
        value for _owner, _name, value in descriptor_values
    )
    callable_guards = tuple(
        freeze_callable_guard(
            value,
            error_type=DiffuseBaseArtifactError,
            label="final phase callable",
        )
        for value in guarded_values
    )
    validation_dependencies = tuple(
        (name, _validation_module.__dict__.get(name))
        for name in (
            "identity_sha256",
            "require_exact",
            "require_exact_tuple",
            "assert_current_identity",
            "DiffuseCompilationReceipt",
            "DiffuseSourceCompilationReceipt",
            "DiscourseArtifact",
            "DiscourseSnapshot",
        )
    )
    dependency_guards = tuple(
        (name, freeze_callable_guard(
            expected,
            error_type=DiffuseBaseArtifactError,
            label=f"final phase helper {name}",
        ))
        for name, expected in validation_dependencies
        if getattr(expected, "__code__", None) is not None
    )
    analysis_format = _analysis_module.DETERMINISTIC_DIFFUSE_INGEST_FORMAT
    assert_implementation = freeze_callable_guard(
        implementation,
        error_type=DiffuseBaseArtifactError,
        label="final phase validator implementation",
    )

    def assert_intact() -> None:
        namespace_guard()
        if any(
            _validation_module.__dict__.get(name) is not expected
            for name, expected in validation_dependencies
        ) or _analysis_module.DETERMINISTIC_DIFFUSE_INGEST_FORMAT != analysis_format:
            raise DiffuseBaseArtifactError("final phase helpers were rebound")
        for name, guard in dependency_guards:
            guard(_validation_module.__dict__.get(name))
        current = tuple(
            getattr(owner, name, None) for owner, name, _expected in helper_values
        ) + tuple(
            descriptor_function(owner, name)
            for owner, name, _expected in descriptor_values
        )
        if any(value is not expected for value, expected in zip(current, guarded_values)):
            raise DiffuseBaseArtifactError("final phase validation was rebound")
        for guard, value in zip(callable_guards, current):
            guard(value)

    def validated(clone: DiffuseDerivedStore, phase: object):
        assert_intact()
        assert_implementation(implementation)
        result = implementation(clone, phase)
        assert_intact()
        return result

    return validated


_PHASE_GUARD_EXCLUDES = (
    "_seal_phase_validator",
    "validated_finalization_phase",
    "_PHASE_GUARD_EXCLUDES",
    "_sealed_phase_guard",
)
_sealed_phase_guard = freeze_namespace_guard(
    globals(),
    error_type=DiffuseBaseArtifactError,
    label="derived final phase module",
    exclude=_PHASE_GUARD_EXCLUDES,
)
validated_finalization_phase = _seal_phase_validator(
    _validate_phase,
    _sealed_phase_guard,
)
del _seal_phase_validator, _sealed_phase_guard


__all__ = ["validated_finalization_phase"]
