"""Matched bridge/global membership adapters over sealed closure generation.

The expensive closure retriever runs outside this module.  This boundary only
verifies its gold-blind, sealed output, projects one isolated membership delta
per question, and exposes a separate structural target ledger.  Structural
discovery is deliberately measured before exact-S0 deduplication; admission is
measured only from the final repacked packet.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, make_atom_id, quote_sha256

from .artifacts import SealedArtifact, read_sealed_json
from .contracts import (
    ArmPlan,
    ArtifactRef,
    EvaluationMemorySnapshot,
    EvidenceItem,
    MatchedEvalContractError,
    MembershipDelta,
    MemoryPacket,
    PlanMode,
    StageBudget,
    StageDisposition,
    StagePlan,
    StageTrace,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .population import MatchedS0Population, SOURCE_STAGE_ID


REPRESENTATIVE_ARM = "S0_PLUS_REPRESENTATIVE_BRIDGE"
GLOBAL_ARM = "S0_PLUS_ARTIFACT_GLOBAL"
ARM_LABELS = (REPRESENTATIVE_ARM, GLOBAL_ARM)

ELIGIBILITY_FORMAT = (
    "memory-condense-independent-closure-eligibility-manifest-v9"
)
QUESTION_FORMAT = "memory-condense-independent-closure-question-v9"
GENERATION_FORMAT = "memory-condense-independent-closure-retrieval-v9"
STRUCTURAL_PROJECTION_FORMAT = (
    "memory-condense-independent-closure-structural-projection-v1"
)
STRUCTURAL_LEDGER_FORMAT = "memory-condense-structural-target-ledger-v1"
CLOSURE_GENERATION_ARTIFACT_ROLE = "independent_closure_generation_v9"

EXPECTED_QUESTION_COUNT = 100
EXPECTED_ELIGIBLE_COUNT = 79
ADDITION_TOKEN_CAP = 2_048
MAX_FINAL_PROMPT_TOKENS = 8_000

_ADMISSION_STATUSES = frozenset(
    {
        "added",
        "no_candidates",
        "selection_budget_noop",
        "overflow_noop",
        "no_novel_evidence",
        "admission_budget_noop",
    }
)
_SELECTED_TERMINAL_DISPOSITIONS = frozenset(
    {
        "exact_s0_overlap_after_selection",
        "selection_overflow_noop",
        "projection_drop_after_s0_dedup",
        "admitted_after_dedup",
        "final_repack_budget_drop",
        "admission_budget_noop",
        "admission_overflow_noop",
    }
)
_IDENTITY_ONLY_SCORE_PROVIDER_FIELDS = frozenset(
    {
        "model_id",
        "model_revision",
        "checkpoint_sha256",
        "device",
        "dtype",
        "runtime",
        "retained_transformer_state_bytes",
    }
)


class IndependentClosureError(MatchedEvalContractError):
    """Raised when sealed closure generation cannot enter the matched spine."""


def _fail(message: str) -> None:
    raise IndependentClosureError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _require_json_native(value: object, label: str) -> None:
    if value is None or type(value) in {str, int, float, bool}:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_json_native(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{label} contains a non-string JSON key")
            _require_json_native(item, f"{label}.{key}")
        return
    _fail(f"{label} must contain only JSON maps, lists, and scalar values")


def _rows(value: object, label: str) -> list[Mapping[str, Any]]:
    if type(value) is not list or any(type(row) is not dict for row in value):
        _fail(f"{label} must be an array of exact objects")
    return value  # type: ignore[return-value]


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be exact text")
    try:
        return require_text(value, label)
    except MatchedEvalContractError as exc:
        raise IndependentClosureError(str(exc)) from exc


def _sha(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be an exact SHA-256 string")
    try:
        return require_sha256(value, label)
    except MatchedEvalContractError as exc:
        raise IndependentClosureError(str(exc)) from exc


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an exact integer >= {minimum}")
    return value


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    if any(type(value) is not str or not value for value in values):
        _fail(f"{label} must contain non-empty exact strings")
    result = tuple(values)
    if len(set(result)) != len(result):
        _fail(f"{label} must be ordered and unique")
    return result


def _ordered_subsequence(values: Sequence[str], parent: Sequence[str]) -> bool:
    iterator = iter(parent)
    return all(any(candidate == value for candidate in iterator) for value in values)


def _file_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _require_file_sha(
    value: Mapping[str, Any], declared: str, label: str
) -> str:
    digest = _sha(declared, f"{label} SHA-256")
    if _file_sha256(value) != digest:
        _fail(f"{label} file SHA-256 changed")
    return digest


def _require_self_seal(value: Mapping[str, Any], field: str, label: str) -> str:
    declared = _sha(value.get(field), f"{label} {field}")
    body = dict(value)
    body.pop(field, None)
    if identity_sha256(body) != declared:
        _fail(f"{label} identity seal changed")
    return declared


def _receipt_sha(value: object, label: str) -> str:
    receipt = _mapping(value, f"{label} receipt")
    return _require_self_seal(receipt, "receipt_sha256", f"{label} receipt")


def _normalize_fresh_coverage_report(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(report)
    elapsed = normalized.pop("elapsed_s", None)
    if type(elapsed) not in {int, float}:
        _fail("S0 fresh report is missing top-level elapsed_s")
    removed = ["elapsed_s"]
    score_provider = _mapping(
        normalized.get("score_provider_report"),
        "S0 score-provider report",
    )
    normalized_provider = dict(score_provider)
    missing = object()
    score_elapsed = normalized_provider.pop("elapsed_s", missing)
    if score_elapsed is not missing:
        if report.get("selection_status") == "bypassed":
            _fail("S0 bypass report unexpectedly invoked its score provider")
        if type(score_elapsed) not in {int, float}:
            _fail("S0 invoked score-provider elapsed_s changed type")
        removed.append("score_provider_report.elapsed_s")
    else:
        if report.get("selection_status") != "bypassed":
            _fail("S0 invoked score-provider report is missing elapsed_s")
        if (
            report.get("bypass_reason") != "not a set query"
            or report.get("requires_completeness") is not False
            or report.get("score_provider_fallback") != ""
            or report.get("fallback_reason") != ""
        ):
            _fail(
                "S0 identity-only score-provider report lacks authoritative "
                "bypass status"
            )
        if set(score_provider) != _IDENTITY_ONLY_SCORE_PROVIDER_FIELDS:
            _fail("S0 identity-only score-provider fields changed")
        for key in ("model_id", "device", "dtype", "runtime"):
            _text(score_provider.get(key), f"S0 identity-only {key}")
        if type(score_provider.get("model_revision")) is not str:
            _fail("S0 identity-only model_revision must be exact text")
        _sha(
            score_provider.get("checkpoint_sha256"),
            "S0 identity-only checkpoint",
        )
        if (
            type(score_provider.get("retained_transformer_state_bytes")) is not int
            or score_provider.get("retained_transformer_state_bytes") != 0
        ):
            _fail("S0 identity-only provider retained transformer state")
    normalized["score_provider_report"] = normalized_provider
    return normalized, removed


def _validate_s0_fresh_validation(question: Mapping[str, Any]) -> tuple[str, str]:
    """Validate v9's JSON-native, timing-stable S0 replay attestation."""

    s0 = _mapping(question.get("s0"), "closure question S0")
    _require_json_native(s0, "closure question S0")
    if s0.get("stage_id") != SOURCE_STAGE_ID:
        _fail("closure question S0 stage changed")
    predecessor = _mapping(s0.get("predecessor_receipt"), "S0 predecessor")
    stage = _mapping(s0.get("stage_receipt"), "S0 root stage")
    predecessor_sha = _receipt_sha(predecessor, "S0 predecessor")
    stage_sha = _receipt_sha(stage, "S0 root stage")
    messages = _rows(s0.get("provider_messages"), "S0 provider messages")
    evidence = _rows(s0.get("evidence"), "S0 evidence")
    message_sha = identity_sha256([dict(row) for row in messages])
    if (
        s0.get("provider_messages_sha256") != message_sha
        or predecessor.get("prompt_messages_sha256") != message_sha
        or stage.get("prompt_messages_sha256") != message_sha
        or stage.get("stage_id") != SOURCE_STAGE_ID
        or stage.get("method_evidence_sha256") != predecessor_sha
        or stage.get("selected_evidence_ids")
        != [row.get("evidence_id") for row in evidence]
    ):
        _fail("closure question exact-S0 binding changed")

    fresh = _mapping(s0.get("fresh_validation"), "S0 fresh validation")
    predecessor_projection = dict(predecessor)
    predecessor_projection.pop("coverage_selector_report_sha256", None)
    predecessor_projection.pop("receipt_sha256", None)
    stage_projection = dict(stage)
    stage_projection.pop("method_evidence_sha256", None)
    stage_projection.pop("receipt_sha256", None)
    predecessor_projection_sha = identity_sha256(predecessor_projection)
    stage_projection_sha = identity_sha256(stage_projection)

    for key in (
        "expected_stable_predecessor_projection_sha256",
        "observed_stable_predecessor_projection_sha256",
        "expected_stable_root_stage_projection_sha256",
        "observed_stable_root_stage_projection_sha256",
        "expected_predecessor_receipt_sha256",
        "observed_predecessor_receipt_sha256",
        "expected_root_method_evidence_sha256",
        "observed_root_method_evidence_sha256",
        "expected_root_stage_receipt_sha256",
        "observed_root_stage_receipt_sha256",
        "expected_coverage_selector_report_sha256",
        "observed_coverage_selector_report_sha256",
        "observed_normalized_coverage_selector_report_sha256",
    ):
        _sha(fresh.get(key), f"S0 fresh validation {key}")
    report = _mapping(
        fresh.get("observed_coverage_selector_report"),
        "S0 observed coverage report",
    )
    report_sha = identity_sha256(dict(report))
    normalized_report, removed_fields = _normalize_fresh_coverage_report(report)
    observed_report_sha = fresh.get("observed_coverage_selector_report_sha256")
    expected_report_sha = fresh.get("expected_coverage_selector_report_sha256")
    if (
        fresh.get("expected_stable_predecessor_projection_sha256")
        != predecessor_projection_sha
        or fresh.get("observed_stable_predecessor_projection_sha256")
        != predecessor_projection_sha
        or fresh.get("expected_stable_root_stage_projection_sha256")
        != stage_projection_sha
        or fresh.get("observed_stable_root_stage_projection_sha256")
        != stage_projection_sha
        or fresh.get("expected_predecessor_receipt_sha256") != predecessor_sha
        or fresh.get("expected_root_method_evidence_sha256") != predecessor_sha
        or fresh.get("expected_root_stage_receipt_sha256") != stage_sha
        or fresh.get("expected_coverage_selector_report_sha256")
        != predecessor.get("coverage_selector_report_sha256")
        or fresh.get("observed_root_method_evidence_sha256")
        != fresh.get("observed_predecessor_receipt_sha256")
        or report_sha != observed_report_sha
        or identity_sha256(normalized_report)
        != fresh.get("observed_normalized_coverage_selector_report_sha256")
        or fresh.get("coverage_report_hash_exact_match")
        is not (expected_report_sha == observed_report_sha)
        or fresh.get("stable_predecessor_fields_exact") is not True
        or fresh.get("stable_root_stage_fields_exact") is not True
        or fresh.get("evidence_order_and_prompt_exact") is not True
        or fresh.get("fresh_report_normalization_removed_fields")
        != removed_fields
    ):
        _fail("closure question fresh-S0 attestation changed")
    return stage_sha, identity_sha256(dict(fresh))


def _wrapper_evidence_id(identity: Mapping[str, Any]) -> str:
    return identity_sha256({"kind": "addition_atom", "atom": dict(identity)})


def _structural_source_identity(
    identity: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    expected_fields = {
        "atom_id",
        "span",
        "text_sha256",
        "label",
        "role",
        "created_at",
    }
    if set(identity) != expected_fields:
        _fail(f"{label} atom identity fields changed")
    _text(identity.get("label"), f"{label} plan-local label")
    raw_span = _mapping(identity.get("span"), f"{label} span identity")
    try:
        span = EvidenceSpan(**dict(raw_span))
    except (TypeError, ValueError) as exc:
        raise IndependentClosureError(f"{label} span identity changed") from exc
    if asdict(span) != dict(raw_span):
        _fail(f"{label} span identity is noncanonical")
    if identity.get("atom_id") != make_atom_id(span):
        _fail(f"{label} atom ID does not bind its exact span")
    if identity.get("text_sha256") != span.quote_sha256:
        _fail(f"{label} text digest does not bind its span")
    if (
        identity.get("role") != span.role
        or identity.get("created_at") != span.created_at
    ):
        _fail(f"{label} metadata does not bind its span")
    result = dict(identity)
    del result["label"]
    return result


def _route_structural_projection(
    arm: Mapping[str, Any], *, label: str
) -> tuple[
    tuple[str, ...],
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, Any]],
    tuple[Mapping[str, Any], ...],
]:
    pool = _mapping(arm.get("candidate_pool"), f"{label} candidate pool")
    identities = _rows(pool.get("atom_identities"), f"{label} candidate atoms")
    atom_ids: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for index, identity in enumerate(identities):
        atom_id = _text(identity.get("atom_id"), f"{label} candidate atom ID")
        if atom_id in by_id:
            _fail(f"{label} contains duplicate candidate atom IDs")
        atom_ids.append(atom_id)
        by_id[atom_id] = identity
        sources[atom_id] = _structural_source_identity(
            identity, label=f"{label} candidate {index}"
        )
    if (
        pool.get("atom_count") != len(identities)
        or pool.get("atom_identities_sha256")
        != identity_sha256([dict(row) for row in identities])
    ):
        _fail(f"{label} structural candidate pool seal changed")
    dispositions = tuple(
        _rows(arm.get("route_target_dispositions"), f"{label} target dispositions")
    )
    if (
        tuple(row.get("evidence_atom_id") for row in dispositions)
        != tuple(atom_ids)
        or arm.get("route_target_dispositions_sha256")
        != identity_sha256([dict(row) for row in dispositions])
        or any(
            row.get("atom_identity_sha256")
            != identity_sha256(dict(by_id[atom_id]))
            for atom_id, row in zip(atom_ids, dispositions, strict=True)
        )
    ):
        _fail(f"{label} structural route projection changed")
    return tuple(atom_ids), by_id, sources, dispositions


def _expected_question_structural_attribution(
    question: Mapping[str, Any], raw_arms: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    population_sha = _sha(
        question.get("population_identity_sha256"), "structural population"
    )
    question_id = _text(question.get("question_id"), "structural question ID")
    question_sha = _sha(
        question.get("retrieval_question_part_sha256"),
        "structural question identity",
    )
    if tuple(arm.get("arm_label") for arm in raw_arms) != ARM_LABELS:
        _fail("structural attribution requires exact ordered arms")
    route = {
        label: _route_structural_projection(arm, label=label)
        for label, arm in zip(ARM_LABELS, raw_arms, strict=True)
    }
    atom_order = tuple(
        dict.fromkeys(
            atom_id
            for label in ARM_LABELS
            for atom_id in route[label][0]
        )
    )
    attribution_sets: dict[str, list[str]] = {label: [] for label in ARM_LABELS}
    targets: list[dict[str, Any]] = []
    for atom_id in atom_order:
        routes = [label for label in ARM_LABELS if atom_id in route[label][1]]
        primary = routes[0]
        primary_identity = route[primary][1][atom_id]
        source_identity = route[primary][2][atom_id]
        if any(route[label][2][atom_id] != source_identity for label in routes[1:]):
            _fail("shared structural atom ID has different source identities")
        source_identity_sha = identity_sha256(source_identity)
        target_id = identity_sha256(
            {
                "scope": "question_local_structural_candidate",
                "population_identity_sha256": population_sha,
                "question_id": question_id,
                "question_identity_sha256": question_sha,
                "kind": "evidence_atom",
                "structural_source_identity_sha256": source_identity_sha,
            }
        )
        attribution_sets[primary].append(target_id)
        reachability = [
            {
                "method": label,
                **dict(
                    next(
                        row
                        for row in route[label][3]
                        if row.get("evidence_atom_id") == atom_id
                    )
                ),
            }
            for label in routes
        ]
        primary_route = reachability[0]
        selected_by = [
            row["method"]
            for row in reachability
            if row.get("selection_disposition") != "not_selected"
        ]
        admitted_by = [
            row["method"]
            for row in reachability
            if row.get("admission_disposition") == "admitted"
        ]
        overlap_by = [
            row["method"]
            for row in reachability
            if row.get("dedup_disposition") == "excluded_exact_s0_overlap"
        ]
        targets.append(
            {
                "target_id": target_id,
                "kind": "evidence_atom",
                "evidence_atom_id": atom_id,
                "primary_route_atom_identity_sha256": identity_sha256(
                    dict(primary_identity)
                ),
                "primary_route_atom_identity": dict(primary_identity),
                "structural_source_identity_sha256": source_identity_sha,
                "structural_source_identity": source_identity,
                "route_atom_identity_sha256s": {
                    label: identity_sha256(dict(route[label][1][atom_id]))
                    for label in routes
                },
                "primary_attribution": primary,
                "discovering_methods": routes,
                "secondary_reachability": routes[1:],
                "reachability": reachability,
                "selected_before_dedup_by": selected_by,
                "discovery_credit_preserved_by": selected_by,
                "admitted_after_dedup_by": admitted_by,
                "exact_s0_overlap_discovered_by": overlap_by,
                "primary_attribution_outcome": {
                    "discovery_credit_preserved": primary_route[
                        "discovery_credit_preserved"
                    ],
                    "mechanism_admission_credit": (
                        primary_route["admission_disposition"] == "admitted"
                    ),
                    "exact_s0_overlap_discovered": (
                        primary_route["dedup_disposition"]
                        == "excluded_exact_s0_overlap"
                    ),
                    "secondary_route_only_admission": bool(admitted_by)
                    and primary not in admitted_by,
                },
            }
        )
    universe = [row["target_id"] for row in targets]
    attributed = [
        target_id for label in ARM_LABELS for target_id in attribution_sets[label]
    ]
    intersections = [
        target_id
        for index, left in enumerate(ARM_LABELS)
        for right in ARM_LABELS[index + 1 :]
        for target_id in set(attribution_sets[left]) & set(attribution_sets[right])
    ]
    duplicate_count = len(attributed) - len(set(attributed))
    union_matches = set(attributed) == set(universe) and len(attributed) == len(universe)
    body: dict[str, Any] = {
        "registry_role": "runtime_structural_candidate_attribution_only",
        "target_scope": "question_local_fresh_closure_candidate_union",
        "population_identity_sha256": population_sha,
        "question_id": question_id,
        "question_identity_sha256": question_sha,
        "benchmark_target_tags_loaded": False,
        "desired_target_union_completeness_claimed": False,
        "desired_target_registry_format": (
            "memory-condense-retrieval-target-owner-registry-v1"
        ),
        "declared_structural_candidate_count": len(universe),
        "declared_structural_candidate_universe_sha256": identity_sha256(universe),
        "declared_structural_candidate_ids": universe,
        "primary_attribution_sets": attribution_sets,
        "targets": targets,
        "invariants": {
            "unattributed_structural_candidate_count": 0,
            "duplicate_primary_attribution_count": duplicate_count,
            "pairwise_primary_attribution_intersection_count": len(intersections),
            "primary_attribution_union_equals_declared_structural_candidate_universe": (
                union_matches
            ),
            "shared_structural_source_identity_mismatch_count": 0,
            "selected_terminal_disposition_missing_count": 0,
            "selected_discovery_credit_loss_count": 0,
        },
    }
    if duplicate_count or intersections or not union_matches:
        _fail("structural candidate attribution is incomplete")
    body["manifest_identity_sha256"] = identity_sha256(body)
    return body


def _validate_question_structural_attribution(
    question: Mapping[str, Any], raw_arms: Sequence[Mapping[str, Any]]
) -> None:
    observed = _mapping(
        question.get("structural_candidate_attribution"),
        "question structural candidate attribution",
    )
    _require_self_seal(
        observed,
        "manifest_identity_sha256",
        "question structural candidate attribution",
    )
    if dict(observed) != _expected_question_structural_attribution(question, raw_arms):
        _fail("question structural candidate attribution changed")


def _expected_aggregate_structural_attribution(
    questions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not questions:
        _fail("cannot aggregate an empty structural campaign")
    population_sha = _sha(
        questions[0].get("population_identity_sha256"), "aggregate population"
    )
    targets: list[Mapping[str, Any]] = []
    for question in questions:
        if question.get("population_identity_sha256") != population_sha:
            _fail("question structural populations differ")
        raw_arms = _rows(question.get("arms"), "aggregate question arms")
        _validate_question_structural_attribution(question, raw_arms)
        manifest = _mapping(
            question.get("structural_candidate_attribution"),
            "question structural attribution",
        )
        targets.extend(_rows(manifest.get("targets"), "question structural targets"))
    universe = [_text(row.get("target_id"), "structural target ID") for row in targets]
    if len(universe) != len(set(universe)):
        _fail("merged structural target IDs are not globally unique")
    attribution_sets = {
        label: [
            str(row["target_id"])
            for row in targets
            if row.get("primary_attribution") == label
        ]
        for label in ARM_LABELS
    }
    attributed = [
        target_id for label in ARM_LABELS for target_id in attribution_sets[label]
    ]
    intersections = [
        target_id
        for index, left in enumerate(ARM_LABELS)
        for right in ARM_LABELS[index + 1 :]
        for target_id in set(attribution_sets[left]) & set(attribution_sets[right])
    ]
    duplicate_count = len(attributed) - len(set(attributed))
    union_matches = set(attributed) == set(universe) and len(attributed) == len(universe)
    body: dict[str, Any] = {
        "registry_role": "runtime_structural_candidate_attribution_only",
        "target_scope": "merged_question_scoped_fresh_closure_candidate_union",
        "population_identity_sha256": population_sha,
        "benchmark_target_tags_loaded": False,
        "desired_target_union_completeness_claimed": False,
        "desired_target_registry_format": (
            "memory-condense-retrieval-target-owner-registry-v1"
        ),
        "declared_structural_candidate_count": len(universe),
        "declared_structural_candidate_universe_sha256": identity_sha256(universe),
        "declared_structural_candidate_ids": universe,
        "primary_attribution_sets": attribution_sets,
        "targets": [dict(row) for row in targets],
        "invariants": {
            "unattributed_structural_candidate_count": 0,
            "duplicate_primary_attribution_count": duplicate_count,
            "pairwise_primary_attribution_intersection_count": len(intersections),
            "primary_attribution_union_equals_declared_structural_candidate_universe": (
                union_matches
            ),
        },
    }
    if duplicate_count or intersections or not union_matches:
        _fail("merged structural candidate attribution is incomplete")
    body["manifest_identity_sha256"] = identity_sha256(body)
    return body


def _validate_generation_structural_attribution(
    generation: Mapping[str, Any], questions: Sequence[Mapping[str, Any]]
) -> None:
    observed = _mapping(
        generation.get("structural_candidate_attribution"),
        "merged structural candidate attribution",
    )
    _require_self_seal(
        observed,
        "manifest_identity_sha256",
        "merged structural candidate attribution",
    )
    if dict(observed) != _expected_aggregate_structural_attribution(questions):
        _fail("merged structural candidate attribution changed")


@dataclass(frozen=True, slots=True)
class ClosureAtom:
    atom_id: str
    evidence_id: str
    chunk_id: str
    source_id: str
    text: str
    token_count: int
    atom_identity_sha256: str

    def __post_init__(self) -> None:
        require_text(self.atom_id, "closure atom ID")
        require_sha256(self.evidence_id, "closure wrapper evidence ID")
        require_text(self.chunk_id, "closure atom chunk ID")
        require_text(self.source_id, "closure source ID")
        if type(self.text) is not str or not self.text:
            raise IndependentClosureError("closure atom text must be non-empty exact text")
        if type(self.token_count) is not int or self.token_count != count_tokens(self.text):
            raise IndependentClosureError("closure atom token count changed")
        require_sha256(self.atom_identity_sha256, "closure atom identity SHA-256")

    def evidence_item(self) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=self.evidence_id,
            source_id=self.source_id,
            text=self.text,
            token_count=self.token_count,
        )


@dataclass(frozen=True, slots=True)
class ClosureTargetLifecycle:
    atom_id: str
    evidence_id: str
    source_id: str
    atom_identity_sha256: str
    selection_disposition: str
    selection_packet_receipt_sha256: str | None
    dedup_disposition: str
    dedup_projection_receipt_sha256: str | None
    admission_disposition: str
    admission_packet_receipt_sha256: str | None
    terminal_disposition: str
    discovery_credit_preserved: bool

    def __post_init__(self) -> None:
        require_text(self.atom_id, "target atom ID")
        require_sha256(self.evidence_id, "target evidence ID")
        require_text(self.source_id, "target source ID")
        require_sha256(self.atom_identity_sha256, "target atom identity SHA-256")
        for value, label in (
            (self.selection_disposition, "selection disposition"),
            (self.dedup_disposition, "dedup disposition"),
            (self.admission_disposition, "admission disposition"),
            (self.terminal_disposition, "terminal disposition"),
        ):
            require_text(value, label)
        for value, label in (
            (self.selection_packet_receipt_sha256, "selection packet receipt"),
            (self.dedup_projection_receipt_sha256, "dedup projection receipt"),
            (self.admission_packet_receipt_sha256, "admission packet receipt"),
        ):
            if value is not None:
                require_sha256(value, label)
        if type(self.discovery_credit_preserved) is not bool:
            raise IndependentClosureError("discovery-credit flag must be an exact bool")
        selected = self.selection_disposition != "not_selected"
        if selected != self.discovery_credit_preserved:
            raise IndependentClosureError("selected target lost discovery credit")
        if selected:
            if self.selection_packet_receipt_sha256 is None:
                raise IndependentClosureError("selected target lacks its selection receipt")
            if self.terminal_disposition not in _SELECTED_TERMINAL_DISPOSITIONS:
                raise IndependentClosureError("selected target terminal disposition changed")
        elif (
            self.terminal_disposition != "not_selected"
            or self.selection_packet_receipt_sha256 is not None
        ):
            raise IndependentClosureError("unselected target has a selected lifecycle")

    @property
    def selected(self) -> bool:
        return self.selection_disposition != "not_selected"

    @property
    def admitted(self) -> bool:
        return self.admission_disposition == "admitted"


@dataclass(frozen=True, slots=True)
class IndependentClosureArmProjection:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    retrieval_question_part_sha256: str
    population_identity_sha256: str
    eligibility_manifest_sha256: str
    eligibility_row_identity_sha256: str
    preflight_sha256: str
    policy_receipt_sha256: str
    source_question_artifact_sha256: str
    source_s0_stage_receipt_sha256: str
    source_s0_fresh_validation_sha256: str
    arm_label: str
    candidate_atom_ids: tuple[str, ...]
    candidate_evidence_ids: tuple[str, ...]
    selected_atoms: tuple[ClosureAtom, ...]
    dedup_excluded_atom_ids: tuple[str, ...]
    post_dedup_atom_ids: tuple[str, ...]
    admitted_atoms: tuple[ClosureAtom, ...]
    targets: tuple[ClosureTargetLifecycle, ...]
    admission_status: str
    overflow_reason: str | None

    def __post_init__(self) -> None:
        _integer(self.ordinal, "closure ordinal")
        require_text(self.question_id, "closure question ID")
        for value, label in (
            (self.question_sha256, "closure question SHA-256"),
            (self.dated_question_sha256, "closure dated-question SHA-256"),
            (self.retrieval_question_part_sha256, "closure question-part SHA-256"),
            (self.population_identity_sha256, "closure population identity"),
            (self.eligibility_manifest_sha256, "closure eligibility manifest"),
            (self.eligibility_row_identity_sha256, "closure eligibility row"),
            (self.preflight_sha256, "closure preflight"),
            (self.policy_receipt_sha256, "closure policy receipt"),
            (self.source_question_artifact_sha256, "closure question artifact"),
            (self.source_s0_stage_receipt_sha256, "closure S0 stage receipt"),
            (
                self.source_s0_fresh_validation_sha256,
                "closure S0 fresh validation",
            ),
        ):
            require_sha256(value, label)
        if self.arm_label not in ARM_LABELS:
            raise IndependentClosureError("unknown independent closure arm")
        candidate_atoms = _ordered_unique(self.candidate_atom_ids, "candidate atom IDs")
        candidate_evidence = _ordered_unique(
            self.candidate_evidence_ids, "candidate evidence IDs"
        )
        if len(candidate_atoms) != len(candidate_evidence):
            raise IndependentClosureError("candidate atom/evidence namespaces diverged")
        if type(self.selected_atoms) is not tuple or any(
            type(row) is not ClosureAtom for row in self.selected_atoms
        ):
            raise IndependentClosureError("selected atoms must be immutable typed values")
        if type(self.admitted_atoms) is not tuple or any(
            type(row) is not ClosureAtom for row in self.admitted_atoms
        ):
            raise IndependentClosureError("admitted atoms must be immutable typed values")
        selected_ids = tuple(row.atom_id for row in self.selected_atoms)
        selected_evidence = tuple(row.evidence_id for row in self.selected_atoms)
        excluded = _ordered_unique(
            self.dedup_excluded_atom_ids, "excluded atom IDs"
        )
        projected = _ordered_unique(self.post_dedup_atom_ids, "post-dedup atom IDs")
        admitted_ids = tuple(row.atom_id for row in self.admitted_atoms)
        if (
            not _ordered_subsequence(selected_ids, candidate_atoms)
            or not _ordered_subsequence(selected_evidence, candidate_evidence)
            or not _ordered_subsequence(excluded, selected_ids)
            or not _ordered_subsequence(projected, selected_ids)
            or not _ordered_subsequence(admitted_ids, projected)
            or set(excluded) & set(projected)
        ):
            raise IndependentClosureError("closure atom lifecycle escaped its source order")
        if type(self.targets) is not tuple or any(
            type(row) is not ClosureTargetLifecycle for row in self.targets
        ):
            raise IndependentClosureError("target lifecycles must be immutable typed values")
        if tuple(row.atom_id for row in self.targets) != candidate_atoms:
            raise IndependentClosureError("target lifecycles changed candidate order")
        if tuple(row.evidence_id for row in self.targets) != candidate_evidence:
            raise IndependentClosureError("target lifecycles changed wrapper identities")
        selected_targets = tuple(row.atom_id for row in self.targets if row.selected)
        admitted_targets = tuple(row.atom_id for row in self.targets if row.admitted)
        if selected_targets != selected_ids or admitted_targets != admitted_ids:
            raise IndependentClosureError("target lifecycle disagrees with arm packets")
        if self.admission_status not in _ADMISSION_STATUSES:
            raise IndependentClosureError("closure admission status changed")
        if (self.admission_status == "added") != bool(admitted_ids):
            raise IndependentClosureError("closure admission status/evidence disagree")
        if self.overflow_reason is not None:
            require_text(self.overflow_reason, "closure overflow reason")

    @property
    def selected_evidence_ids(self) -> tuple[str, ...]:
        return tuple(row.evidence_id for row in self.selected_atoms)

    @property
    def admitted_evidence_ids(self) -> tuple[str, ...]:
        return tuple(row.evidence_id for row in self.admitted_atoms)

    @property
    def dedup_excluded_evidence_ids(self) -> tuple[str, ...]:
        by_atom = {row.atom_id: row.evidence_id for row in self.selected_atoms}
        return tuple(by_atom[atom_id] for atom_id in self.dedup_excluded_atom_ids)


def _identity_projection(
    raw: Mapping[str, Any], label: str
) -> tuple[str, str, str, str, Mapping[str, Any]]:
    _structural_source_identity(raw, label=label)
    atom_id = _text(raw.get("atom_id"), f"{label} atom ID")
    span = _mapping(raw.get("span"), f"{label} span")
    source_id = _text(span.get("source_id"), f"{label} source ID")
    text_sha = _sha(raw.get("text_sha256"), f"{label} text SHA-256")
    identity_sha = identity_sha256(dict(raw))
    return atom_id, _wrapper_evidence_id(raw), source_id, text_sha, raw


def _wrapped_atom(
    row: Mapping[str, Any],
    *,
    label: str,
    candidate: tuple[str, str, str, str, Mapping[str, Any]],
) -> ClosureAtom:
    identity = _mapping(row.get("identity"), f"{label} identity")
    atom_id, evidence_id, source_id, text_sha, candidate_identity = candidate
    if (
        dict(identity) != dict(candidate_identity)
        or row.get("atom_id") != atom_id
        or row.get("evidence_id") != evidence_id
        or row.get("source_id") != source_id
    ):
        _fail(f"{label} wrapper changed atom identity")
    span = _mapping(identity.get("span"), f"{label} span")
    if row.get("chunk_id") != span.get("chunk_id"):
        _fail(f"{label} wrapper changed chunk identity")
    text = row.get("text")
    if type(text) is not str or not text or quote_sha256(text) != text_sha:
        _fail(f"{label} evidence text changed")
    if row.get("text_sha256") != text_sha:
        _fail(f"{label} text digest changed")
    return ClosureAtom(
        atom_id=atom_id,
        evidence_id=evidence_id,
        chunk_id=_text(span.get("chunk_id"), f"{label} chunk ID"),
        source_id=source_id,
        text=text,
        token_count=count_tokens(text),
        atom_identity_sha256=identity_sha256(dict(identity)),
    )


def _candidate_index(
    arm: Mapping[str, Any], label: str
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[str, str, str, str, Mapping[str, Any]]],
    str,
    str,
]:
    pool = _mapping(arm.get("candidate_pool"), f"{label} candidate pool")
    identities = _rows(pool.get("atom_identities"), f"{label} candidate atoms")
    parsed = tuple(
        _identity_projection(row, f"{label} candidate {index}")
        for index, row in enumerate(identities)
    )
    atom_ids = _ordered_unique(
        tuple(row[0] for row in parsed), f"{label} candidate atom IDs"
    )
    evidence_ids = _ordered_unique(
        tuple(row[1] for row in parsed), f"{label} candidate evidence IDs"
    )
    bundles = pool.get("bundle_identities")
    if (
        type(bundles) is not list
        or pool.get("atom_count") != len(identities)
        or pool.get("atom_identities_sha256")
        != identity_sha256([dict(row) for row in identities])
        or pool.get("bundle_count") != len(bundles)
        or pool.get("bundle_identities_sha256") != identity_sha256(bundles)
    ):
        _fail(f"{label} candidate-pool seal changed")
    source_plan = _sha(pool.get("source_plan_sha256"), f"{label} source plan")
    scope = _sha(pool.get("scope_witnesses_sha256"), f"{label} scope witnesses")
    return (
        atom_ids,
        evidence_ids,
        dict(zip(atom_ids, parsed, strict=True)),
        source_plan,
        scope,
    )


def _packet_atoms(
    value: object,
    *,
    label: str,
    candidates: Mapping[str, tuple[str, str, str, str, Mapping[str, Any]]],
) -> tuple[tuple[ClosureAtom, ...], str]:
    packet = _mapping(value, label)
    raw_atoms = _rows(packet.get("atoms"), f"{label} atoms")
    atoms: list[ClosureAtom] = []
    for index, row in enumerate(raw_atoms):
        atom_id = _text(row.get("atom_id"), f"{label} atom {index} ID")
        candidate = candidates.get(atom_id)
        if candidate is None:
            _fail(f"{label} atom escaped the candidate pool")
        atoms.append(
            _wrapped_atom(
                row,
                label=f"{label} atom {index}",
                candidate=candidate,
            )
        )
    identities = [dict(_mapping(row.get("identity"), f"{label} identity")) for row in raw_atoms]
    bundles = packet.get("bundles")
    context = packet.get("context")
    if (
        type(bundles) is not list
        or type(context) is not str
        or packet.get("context_sha256") != quote_sha256(context)
        or packet.get("atom_count") != len(atoms)
        or packet.get("atom_identities_sha256") != identity_sha256(identities)
        or packet.get("bundle_count") != len(bundles)
        or packet.get("bundle_identities_sha256") != identity_sha256(bundles)
    ):
        _fail(f"{label} packet projection changed")
    receipt_sha = _receipt_sha(packet.get("packet_receipt"), f"{label} packet")
    return tuple(atoms), receipt_sha


def project_independent_closure_question(
    question: Mapping[str, Any],
    *,
    source_question_artifact_sha256: str,
    arm_label: str,
) -> IndependentClosureArmProjection:
    """Validate and project one arm from one sealed v9 question artifact."""

    if arm_label not in ARM_LABELS:
        _fail("unknown independent closure arm")
    if type(question) is not dict:
        _fail("closure question artifact must be an exact object")
    assert_gold_blind(question, path="independent_closure_question")
    _require_file_sha(question, source_question_artifact_sha256, "question artifact")
    _require_self_seal(question, "artifact_identity_sha256", "question artifact")
    if (
        question.get("format") != QUESTION_FORMAT
        or question.get("provider_calls") != 0
        or question.get("gold_loaded") is not False
        or question.get("retained_request_token_state_bytes") != 0
    ):
        _fail("closure question runtime boundary changed")
    ordinal = _integer(question.get("ordinal"), "closure question ordinal")
    question_id = _text(question.get("question_id"), "closure question ID")
    question_sha = _sha(question.get("question_sha256"), "closure question SHA-256")
    dated_sha = _sha(
        question.get("dated_question_sha256"), "closure dated-question SHA-256"
    )
    question_part_sha = _sha(
        question.get("retrieval_question_part_sha256"),
        "closure retrieval-question-part SHA-256",
    )
    population_sha = _sha(
        question.get("population_identity_sha256"), "closure population identity"
    )
    eligibility_sha = _sha(
        question.get("eligibility_manifest_sha256"), "closure eligibility manifest"
    )
    eligibility_row_sha = _sha(
        question.get("eligibility_row_identity_sha256"), "closure eligibility row"
    )
    preflight_sha = _sha(question.get("preflight_sha256"), "closure preflight")
    policy_sha = _sha(
        question.get("policy_receipt_sha256"), "closure policy receipt"
    )
    s0_stage_sha, s0_fresh_sha = _validate_s0_fresh_validation(question)

    raw_arms = _rows(question.get("arms"), "closure question arms")
    if tuple(row.get("arm_label") for row in raw_arms) != ARM_LABELS:
        _fail("closure question arm order changed")
    _validate_question_structural_attribution(question, raw_arms)
    arm = next(row for row in raw_arms if row.get("arm_label") == arm_label)
    if arm.get("parent_stage") != "exact_sealed_s0":
        _fail(f"{arm_label} changed its exact-S0 parent")
    candidate_ids, candidate_evidence, candidates, source_plan, scope_sha = (
        _candidate_index(arm, arm_label)
    )

    raw_selected = arm.get("selected_before_dedup")
    selected: tuple[ClosureAtom, ...] = ()
    selection_receipt: str | None = None
    if raw_selected is not None:
        selected, selection_receipt = _packet_atoms(
            raw_selected,
            label=f"{arm_label} selected-before-dedup",
            candidates=candidates,
        )

    raw_dedup = arm.get("dedup")
    excluded_ids: tuple[str, ...] = ()
    projected_ids: tuple[str, ...] = ()
    dedup_receipt: str | None = None
    if raw_dedup is not None:
        dedup = _mapping(raw_dedup, f"{arm_label} dedup")
        raw_excluded = dedup.get("excluded_atom_ids")
        raw_projected = _rows(
            dedup.get("post_dedup_atom_identities"),
            f"{arm_label} post-dedup identities",
        )
        if type(raw_excluded) is not list:
            _fail(f"{arm_label} excluded atom IDs must be a list")
        excluded_ids = _ordered_unique(raw_excluded, f"{arm_label} excluded atom IDs")
        projected_parsed = tuple(
            _identity_projection(row, f"{arm_label} projected {index}")
            for index, row in enumerate(raw_projected)
        )
        projected_ids = _ordered_unique(
            tuple(row[0] for row in projected_parsed),
            f"{arm_label} post-dedup atom IDs",
        )
        if any(
            atom_id not in candidates
            or dict(identity) != dict(candidates[atom_id][4])
            for atom_id, _evidence, _source, _text_sha, identity in projected_parsed
        ):
            _fail(f"{arm_label} post-dedup identity changed")
        projection_receipt = _mapping(
            dedup.get("projection_receipt"), f"{arm_label} projection receipt"
        )
        dedup_receipt = _receipt_sha(
            projection_receipt, f"{arm_label} dedup projection"
        )
        projected_bundles = dedup.get("post_dedup_bundle_identities")
        if (
            type(projected_bundles) is not list
            or dedup.get("excluded_atom_count") != len(excluded_ids)
            or projection_receipt.get("excluded_atom_ids") != list(excluded_ids)
            or projection_receipt.get("source_plan_sha256")
            != dedup.get("selected_plan_sha256")
            or dedup.get("post_dedup_atom_count") != len(projected_ids)
            or dedup.get("post_dedup_atom_identities_sha256")
            != identity_sha256([dict(row) for row in raw_projected])
            or dedup.get("post_dedup_bundle_count") != len(projected_bundles)
            or dedup.get("post_dedup_bundle_identities_sha256")
            != identity_sha256(projected_bundles)
        ):
            _fail(f"{arm_label} dedup projection seal changed")

    admission = _mapping(arm.get("admission"), f"{arm_label} admission")
    status = _text(admission.get("status"), f"{arm_label} admission status")
    addition_tokens = _integer(
        admission.get("addition_token_proxy"),
        f"{arm_label} addition token proxy",
    )
    prompt_tokens = _integer(
        admission.get("prompt_token_proxy"),
        f"{arm_label} prompt token proxy",
    )
    if (
        admission.get("addition_token_cap") != ADDITION_TOKEN_CAP
        or addition_tokens > ADDITION_TOKEN_CAP
        or prompt_tokens > MAX_FINAL_PROMPT_TOKENS
    ):
        _fail(f"{arm_label} source budget changed")
    overflow_reason = admission.get("overflow_reason")
    if overflow_reason is not None:
        overflow_reason = _text(overflow_reason, f"{arm_label} overflow reason")
    raw_added = _rows(admission.get("added_evidence"), f"{arm_label} added evidence")
    admitted: tuple[ClosureAtom, ...] = ()
    admission_receipt: str | None = None
    if status == "added":
        admitted, admission_receipt = _packet_atoms(
            admission.get("packet"),
            label=f"{arm_label} admitted",
            candidates=candidates,
        )
        added_atoms = tuple(
            _wrapped_atom(
                row,
                label=f"{arm_label} added evidence {index}",
                candidate=candidates[_text(row.get("atom_id"), "added atom ID")],
            )
            for index, row in enumerate(raw_added)
        )
        if added_atoms != admitted:
            _fail(f"{arm_label} admitted packet and added evidence differ")
    elif raw_added or admission.get("packet") is not None:
        _fail(f"{arm_label} no-op contains admitted evidence")

    selection_overflow = (
        status == "overflow_noop"
        and isinstance(overflow_reason, str)
        and overflow_reason.startswith("selected_before_dedup:")
    )
    if selected and raw_dedup is None and not selection_overflow:
        _fail(f"{arm_label} selected atoms bypassed exact-S0 dedup")
    if raw_dedup is not None and not selected:
        _fail(f"{arm_label} ran exact-S0 dedup without a selected packet")
    if selection_overflow and (excluded_ids or projected_ids):
        _fail(f"{arm_label} selection overflow contains a dedup projection")

    raw_targets = _rows(
        arm.get("route_target_dispositions"), f"{arm_label} target dispositions"
    )
    if (
        arm.get("reachable_structural_candidate_ids") != list(candidate_ids)
        or arm.get("selected_target_ids_before_dedup")
        != [row.atom_id for row in selected]
        or arm.get("preserved_discovery_credit_target_ids")
        != [row.atom_id for row in selected]
        or arm.get("exact_s0_overlap_target_ids_after_selection")
        != list(excluded_ids)
        or arm.get("post_dedup_candidate_target_ids") != list(projected_ids)
        or arm.get("admitted_target_ids_after_dedup")
        != [row.atom_id for row in admitted]
        or arm.get("route_target_dispositions_sha256")
        != identity_sha256([dict(row) for row in raw_targets])
    ):
        _fail(f"{arm_label} target projection seal changed")

    selected_set = {row.atom_id for row in selected}
    excluded_set = set(excluded_ids)
    projected_set = set(projected_ids)
    admitted_set = {row.atom_id for row in admitted}
    lifecycles: list[ClosureTargetLifecycle] = []
    if tuple(row.get("evidence_atom_id") for row in raw_targets) != candidate_ids:
        _fail(f"{arm_label} target disposition order changed")
    for raw in raw_targets:
        atom_id = _text(raw.get("evidence_atom_id"), "target atom ID")
        _atom_id, evidence_id, source_id, _text_sha, identity = candidates[atom_id]
        selected_here = atom_id in selected_set
        admitted_here = atom_id in admitted_set
        if (
            raw.get("atom_identity_sha256") != identity_sha256(dict(identity))
            or raw.get("source_plan_sha256") != source_plan
            or raw.get("source_scope_witnesses_sha256") != scope_sha
            or raw.get("candidate_pool_atom_identities_sha256")
            != arm["candidate_pool"]["atom_identities_sha256"]
            or raw.get("selection_packet_receipt_sha256")
            != (selection_receipt if selected_here else None)
            or raw.get("dedup_projection_receipt_sha256") != dedup_receipt
            or raw.get("admission_packet_receipt_sha256") != admission_receipt
            or raw.get("discovery_credit_preserved") is not selected_here
        ):
            _fail(f"{arm_label} target receipt or identity changed")
        selection_disposition = _text(
            raw.get("selection_disposition"), "target selection disposition"
        )
        dedup_disposition = _text(
            raw.get("dedup_disposition"), "target dedup disposition"
        )
        admission_disposition = _text(
            raw.get("admission_disposition"), "target admission disposition"
        )
        terminal = _text(raw.get("terminal_disposition"), "target terminal disposition")
        if selected_here != (selection_disposition != "not_selected"):
            _fail(f"{arm_label} selection disposition changed")
        if admitted_here != (admission_disposition == "admitted"):
            _fail(f"{arm_label} admission disposition changed")
        if atom_id in excluded_set:
            if (
                dedup_disposition != "excluded_exact_s0_overlap"
                or terminal != "exact_s0_overlap_after_selection"
                or raw.get("final_packet_covered") is not True
                or raw.get("final_coverage_source") != "S0_CONTROL"
            ):
                _fail(f"{arm_label} exact-S0 disposition changed")
        elif admitted_here:
            if (
                terminal != "admitted_after_dedup"
                or raw.get("final_packet_covered") is not True
                or raw.get("final_coverage_source") != arm_label
            ):
                _fail(f"{arm_label} admitted disposition changed")
        elif not selected_here:
            if (
                terminal != "not_selected"
                or raw.get("final_packet_covered") is not False
                or raw.get("final_coverage_source") is not None
            ):
                _fail(f"{arm_label} unselected disposition changed")
        elif (
            terminal not in _SELECTED_TERMINAL_DISPOSITIONS
            or raw.get("final_packet_covered") is not False
            or raw.get("final_coverage_source") is not None
        ):
            _fail(f"{arm_label} terminal non-admission disposition changed")
        if atom_id in projected_set and atom_id not in admitted_set and (
            terminal
            not in {
                "final_repack_budget_drop",
                "admission_budget_noop",
                "admission_overflow_noop",
            }
        ):
            _fail(f"{arm_label} retained atom lacks an admission disposition")
        lifecycles.append(
            ClosureTargetLifecycle(
                atom_id=atom_id,
                evidence_id=evidence_id,
                source_id=source_id,
                atom_identity_sha256=identity_sha256(dict(identity)),
                selection_disposition=selection_disposition,
                selection_packet_receipt_sha256=raw.get(
                    "selection_packet_receipt_sha256"
                ),
                dedup_disposition=dedup_disposition,
                dedup_projection_receipt_sha256=raw.get(
                    "dedup_projection_receipt_sha256"
                ),
                admission_disposition=admission_disposition,
                admission_packet_receipt_sha256=raw.get(
                    "admission_packet_receipt_sha256"
                ),
                terminal_disposition=terminal,
                discovery_credit_preserved=selected_here,
            )
        )

    return IndependentClosureArmProjection(
        ordinal=ordinal,
        question_id=question_id,
        question_sha256=question_sha,
        dated_question_sha256=dated_sha,
        retrieval_question_part_sha256=question_part_sha,
        population_identity_sha256=population_sha,
        eligibility_manifest_sha256=eligibility_sha,
        eligibility_row_identity_sha256=eligibility_row_sha,
        preflight_sha256=preflight_sha,
        policy_receipt_sha256=policy_sha,
        source_question_artifact_sha256=source_question_artifact_sha256,
        source_s0_stage_receipt_sha256=s0_stage_sha,
        source_s0_fresh_validation_sha256=s0_fresh_sha,
        arm_label=arm_label,
        candidate_atom_ids=candidate_ids,
        candidate_evidence_ids=candidate_evidence,
        selected_atoms=selected,
        dedup_excluded_atom_ids=excluded_ids,
        post_dedup_atom_ids=projected_ids,
        admitted_atoms=admitted,
        targets=tuple(lifecycles),
        admission_status=status,
        overflow_reason=overflow_reason,
    )


@dataclass(frozen=True, slots=True)
class IndependentClosureQuestion:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question: str
    dated_question_sha256: str
    question_part_sha256: str
    eligibility_row_identity_sha256: str
    root_packet_id: str
    root_protected_evidence: tuple[EvidenceItem, ...]
    root_applied_stage_ids: tuple[str, ...]
    eligible: bool
    source_question_artifact_sha256: str | None
    source_s0_stage_receipt_sha256: str | None
    source_s0_fresh_validation_sha256: str | None
    arms: tuple[IndependentClosureArmProjection, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.ordinal, "independent closure question ordinal")
        require_text(self.question_id, "independent closure question ID")
        require_text(self.dated_question, "independent closure dated question")
        for value, label in (
            (self.question_sha256, "independent closure question SHA-256"),
            (self.dated_question_sha256, "independent closure dated-question SHA-256"),
            (self.question_part_sha256, "independent closure question-part SHA-256"),
            (self.eligibility_row_identity_sha256, "independent closure eligibility row"),
            (self.root_packet_id, "independent closure root packet"),
        ):
            require_sha256(value, label)
        if quote_sha256(self.dated_question) != self.dated_question_sha256:
            raise IndependentClosureError("closure dated question changed")
        if type(self.root_protected_evidence) is not tuple or any(
            type(row) is not EvidenceItem for row in self.root_protected_evidence
        ):
            raise IndependentClosureError("closure root evidence must be immutable")
        if self.root_applied_stage_ids != ():
            raise IndependentClosureError("closure root already has applied stages")
        root = MemoryPacket(
            question_id=self.question_id,
            question_sha256=self.question_sha256,
            dated_question=self.dated_question,
            dated_question_sha256=self.dated_question_sha256,
            stage_id=SOURCE_STAGE_ID,
            protected_evidence=self.root_protected_evidence,
            applied_stage_ids=self.root_applied_stage_ids,
        )
        if root.packet_id != self.root_packet_id:
            raise IndependentClosureError("closure root packet binding changed")
        if type(self.eligible) is not bool:
            raise IndependentClosureError("closure eligibility must be an exact bool")
        if self.source_question_artifact_sha256 is not None:
            require_sha256(
                self.source_question_artifact_sha256,
                "independent closure source question artifact",
            )
        for value, label in (
            (self.source_s0_stage_receipt_sha256, "independent closure S0 stage"),
            (
                self.source_s0_fresh_validation_sha256,
                "independent closure S0 fresh validation",
            ),
        ):
            if value is not None:
                require_sha256(value, label)
        if type(self.arms) is not tuple or any(
            type(row) is not IndependentClosureArmProjection for row in self.arms
        ):
            raise IndependentClosureError("closure question arms must be immutable")
        if self.eligible:
            if (
                self.source_question_artifact_sha256 is None
                or self.source_s0_stage_receipt_sha256 is None
                or self.source_s0_fresh_validation_sha256 is None
                or tuple(row.arm_label for row in self.arms) != ARM_LABELS
                or any(
                    row.source_s0_stage_receipt_sha256
                    != self.source_s0_stage_receipt_sha256
                    or row.source_s0_fresh_validation_sha256
                    != self.source_s0_fresh_validation_sha256
                    for row in self.arms
                )
            ):
                raise IndependentClosureError("eligible question lacks exact closure arms")
        elif (
            self.source_question_artifact_sha256 is not None
            or self.source_s0_stage_receipt_sha256 is not None
            or self.source_s0_fresh_validation_sha256 is not None
            or self.arms
        ):
            raise IndependentClosureError("ineligible question acquired closure output")

    def arm(self, arm_label: str) -> IndependentClosureArmProjection | None:
        for row in self.arms:
            if row.arm_label == arm_label:
                return row
        return None


@dataclass(frozen=True, slots=True)
class IndependentClosureGeneration:
    source_retrieval_generation_sha256: str
    source_eligibility_manifest_sha256: str
    preflight_sha256: str
    policy_receipt_sha256: str
    retrieval_sha256: str
    population_identity_sha256: str
    questions: tuple[IndependentClosureQuestion, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_retrieval_generation_sha256, "closure generation"),
            (self.source_eligibility_manifest_sha256, "closure eligibility manifest"),
            (self.preflight_sha256, "closure preflight"),
            (self.policy_receipt_sha256, "closure policy receipt"),
            (self.retrieval_sha256, "closure source retrieval"),
            (self.population_identity_sha256, "closure population identity"),
        ):
            require_sha256(value, label)
        if type(self.questions) is not tuple or any(
            type(row) is not IndependentClosureQuestion for row in self.questions
        ):
            raise IndependentClosureError("closure generation questions must be immutable")
        if (
            len(self.questions) != EXPECTED_QUESTION_COUNT
            or tuple(row.ordinal for row in self.questions)
            != tuple(range(EXPECTED_QUESTION_COUNT))
            or len({row.question_id for row in self.questions})
            != EXPECTED_QUESTION_COUNT
            or sum(row.eligible for row in self.questions) != EXPECTED_ELIGIBLE_COUNT
        ):
            raise IndependentClosureError("closure generation population changed")

    @property
    def artifact_ref(self) -> ArtifactRef:
        return ArtifactRef(
            CLOSURE_GENERATION_ARTIFACT_ROLE,
            self.source_retrieval_generation_sha256,
        )

    def question(self, question_id: str) -> IndependentClosureQuestion:
        for row in self.questions:
            if row.question_id == question_id:
                return row
        raise KeyError(question_id)


def project_independent_closure_generation(
    generation: Mapping[str, Any],
    *,
    generation_sha256: str,
    eligibility_manifest: Mapping[str, Any],
    eligibility_manifest_sha256: str,
    population: MatchedS0Population,
) -> IndependentClosureGeneration:
    """Verify sealed v9 generation against the exact 100-row matched S0."""

    if type(population) is not MatchedS0Population:
        _fail("closure generation requires an exact matched S0 population")
    source_retrieval_refs = tuple(
        row
        for row in population.snapshot.source_artifacts
        if row.role == "sealed_retrieval"
    )
    if (
        len(source_retrieval_refs) != 1
        or source_retrieval_refs[0].sha256 != population.retrieval_sha256
    ):
        _fail("matched S0 snapshot changed its sealed retrieval binding")
    if type(generation) is not dict or type(eligibility_manifest) is not dict:
        _fail("closure generation inputs must be exact objects")
    assert_gold_blind(eligibility_manifest, path="closure_eligibility")
    assert_gold_blind(generation, path="closure_generation")
    generation_sha = _require_file_sha(generation, generation_sha256, "closure generation")
    eligibility_sha = _require_file_sha(
        eligibility_manifest,
        eligibility_manifest_sha256,
        "closure eligibility manifest",
    )
    _require_self_seal(
        eligibility_manifest,
        "manifest_identity_sha256",
        "closure eligibility manifest",
    )
    _require_self_seal(generation, "artifact_identity_sha256", "closure generation")
    if (
        eligibility_manifest.get("format") != ELIGIBILITY_FORMAT
        or eligibility_manifest.get("question_count") != EXPECTED_QUESTION_COUNT
        or eligibility_manifest.get("eligible_question_count")
        != EXPECTED_ELIGIBLE_COUNT
        or eligibility_manifest.get("provider_calls") != 0
        or eligibility_manifest.get("gold_loaded") is not False
        or eligibility_manifest.get("retrieval_sha256") != population.retrieval_sha256
        or eligibility_manifest.get("population_identity_sha256")
        != population.snapshot.population_identity_sha256
    ):
        _fail("closure eligibility population changed")
    eligibility_rows = _rows(
        eligibility_manifest.get("questions"), "closure eligibility questions"
    )
    if len(eligibility_rows) != EXPECTED_QUESTION_COUNT:
        _fail("closure eligibility question count changed")

    raw_generation_questions = _rows(
        generation.get("questions"), "closure generated questions"
    )
    raw_hashes = generation.get("question_artifact_sha256s")
    raw_ordinals = generation.get("question_ordinals")
    if (
        generation.get("format") != GENERATION_FORMAT
        or generation.get("arm_labels") != list(ARM_LABELS)
        or generation.get("eligibility_manifest_sha256") != eligibility_sha
        or generation.get("question_count") != EXPECTED_ELIGIBLE_COUNT
        or type(raw_hashes) is not list
        or type(raw_ordinals) is not list
        or len(raw_generation_questions) != EXPECTED_ELIGIBLE_COUNT
        or len(raw_hashes) != EXPECTED_ELIGIBLE_COUNT
        or len(raw_ordinals) != EXPECTED_ELIGIBLE_COUNT
        or generation.get("retrieval_invocation_count") != EXPECTED_ELIGIBLE_COUNT
        or generation.get("provider_calls") != 0
        or generation.get("gold_loaded") is not False
    ):
        _fail("closure generation campaign boundary changed")
    _validate_generation_structural_attribution(
        generation, raw_generation_questions
    )
    preflight_sha = _sha(generation.get("preflight_sha256"), "closure preflight")
    policy_sha = _sha(
        generation.get("policy_receipt_sha256"), "closure policy receipt"
    )
    generated_by_ordinal: dict[int, tuple[Mapping[str, Any], str]] = {}
    for raw, declared_hash, declared_ordinal in zip(
        raw_generation_questions, raw_hashes, raw_ordinals, strict=True
    ):
        ordinal = _integer(declared_ordinal, "generated question ordinal")
        question_hash = _require_file_sha(raw, declared_hash, "generated question")
        if raw.get("ordinal") != ordinal or ordinal in generated_by_ordinal:
            _fail("generated question order or uniqueness changed")
        generated_by_ordinal[ordinal] = (raw, question_hash)
    if tuple(raw_ordinals) != tuple(sorted(raw_ordinals)):
        _fail("generated question ordinals are not in locked order")

    questions: list[IndependentClosureQuestion] = []
    for ordinal, (eligibility, source) in enumerate(
        zip(eligibility_rows, population.rows, strict=True)
    ):
        if eligibility.get("ordinal") != ordinal:
            _fail(f"eligibility ordinal changed at {ordinal}")
        row_identity = _require_self_seal(
            eligibility, "row_identity_sha256", f"eligibility row {ordinal}"
        )
        eligible = eligibility.get("eligible")
        if type(eligible) is not bool:
            _fail(f"eligibility flag changed at {ordinal}")
        packet = source.packet
        if (
            packet.stage_id != SOURCE_STAGE_ID
            or packet.applied_stage_ids != ()
            or packet.admitted_evidence
            or packet.facts
            or packet.links
            or packet.answer_operators
            or eligibility.get("question_id") != packet.question_id
            or eligibility.get("question_sha256") != packet.question_sha256
            or eligibility.get("dated_question_sha256")
            != packet.dated_question_sha256
            or eligibility.get("dated_question") != packet.dated_question
        ):
            _fail(f"eligibility/matched question binding changed at {ordinal}")
        generated = generated_by_ordinal.get(ordinal)
        if eligible != (generated is not None):
            _fail(f"eligible/generated question partition changed at {ordinal}")
        arms: tuple[IndependentClosureArmProjection, ...] = ()
        question_artifact_sha: str | None = None
        s0_stage_sha: str | None = None
        s0_fresh_sha: str | None = None
        if generated is not None:
            raw_question, question_artifact_sha = generated
            raw_s0 = _mapping(raw_question.get("s0"), f"generated S0 {ordinal}")
            raw_s0_evidence = _rows(
                raw_s0.get("evidence"), f"generated S0 evidence {ordinal}"
            )
            expected_s0_evidence = [
                {
                    "evidence_id": row.evidence_id,
                    "source_id": row.source_id,
                    "text": row.text,
                }
                for row in packet.protected_evidence
            ]
            if [dict(row) for row in raw_s0_evidence] != expected_s0_evidence:
                _fail(f"generated/matched protected evidence changed at {ordinal}")
            arms = tuple(
                project_independent_closure_question(
                    raw_question,
                    source_question_artifact_sha256=question_artifact_sha,
                    arm_label=arm_label,
                )
                for arm_label in ARM_LABELS
            )
            if any(
                arm.ordinal != ordinal
                or arm.question_id != packet.question_id
                or arm.question_sha256 != packet.question_sha256
                or arm.dated_question_sha256 != packet.dated_question_sha256
                or arm.retrieval_question_part_sha256
                != source.question_part_sha256
                or arm.population_identity_sha256
                != population.snapshot.population_identity_sha256
                or arm.eligibility_manifest_sha256 != eligibility_sha
                or arm.eligibility_row_identity_sha256 != row_identity
                or arm.preflight_sha256 != preflight_sha
                or arm.policy_receipt_sha256 != policy_sha
                or arm.source_s0_stage_receipt_sha256
                != source.source_stage_receipt_sha256
                for arm in arms
            ):
                _fail(f"generated/matched question binding changed at {ordinal}")
            s0_stage_sha = arms[0].source_s0_stage_receipt_sha256
            s0_fresh_sha = arms[0].source_s0_fresh_validation_sha256
        questions.append(
            IndependentClosureQuestion(
                ordinal=ordinal,
                question_id=packet.question_id,
                question_sha256=packet.question_sha256,
                dated_question=packet.dated_question,
                dated_question_sha256=packet.dated_question_sha256,
                question_part_sha256=source.question_part_sha256,
                eligibility_row_identity_sha256=row_identity,
                root_packet_id=packet.packet_id,
                root_protected_evidence=packet.protected_evidence,
                root_applied_stage_ids=packet.applied_stage_ids,
                eligible=eligible,
                source_question_artifact_sha256=question_artifact_sha,
                source_s0_stage_receipt_sha256=s0_stage_sha,
                source_s0_fresh_validation_sha256=s0_fresh_sha,
                arms=arms,
            )
        )
    expected_eligible_ordinals = tuple(
        row.ordinal for row in questions if row.eligible
    )
    if tuple(raw_ordinals) != expected_eligible_ordinals:
        _fail("closure generation contains an out-of-population question")
    return IndependentClosureGeneration(
        source_retrieval_generation_sha256=generation_sha,
        source_eligibility_manifest_sha256=eligibility_sha,
        preflight_sha256=preflight_sha,
        policy_receipt_sha256=policy_sha,
        retrieval_sha256=population.retrieval_sha256,
        population_identity_sha256=population.snapshot.population_identity_sha256,
        questions=tuple(questions),
    )


def load_independent_closure_generation(
    generation_path: str | Path,
    *,
    expected_generation_sha256: str,
    eligibility_manifest_path: str | Path,
    expected_eligibility_manifest_sha256: str,
    population: MatchedS0Population,
) -> IndependentClosureGeneration:
    generation = read_sealed_json(generation_path)
    eligibility = read_sealed_json(eligibility_manifest_path)
    if generation.sha256 != _sha(
        expected_generation_sha256, "expected closure generation"
    ):
        _fail("sealed closure generation differs from its pinned digest")
    if eligibility.sha256 != _sha(
        expected_eligibility_manifest_sha256,
        "expected closure eligibility manifest",
    ):
        _fail("sealed closure eligibility differs from its pinned digest")
    return project_independent_closure_generation(
        generation.payload,
        generation_sha256=generation.sha256,
        eligibility_manifest=eligibility.payload,
        eligibility_manifest_sha256=eligibility.sha256,
        population=population,
    )


class IndependentClosureMembershipAdapter:
    """Project one independently budgeted closure arm onto exact sealed S0."""

    delta_kind = "membership"

    def __init__(
        self,
        generation: IndependentClosureGeneration,
        arm_label: str,
    ) -> None:
        if type(generation) is not IndependentClosureGeneration:
            _fail("closure adapter requires an exact generation projection")
        if arm_label not in ARM_LABELS:
            _fail("unknown independent closure arm")
        self.generation = generation
        self.arm_label = arm_label
        self.mechanism_id = arm_label

    def propose(
        self,
        *,
        snapshot: EvaluationMemorySnapshot,
        packet: MemoryPacket,
        stage: StagePlan,
    ) -> MembershipDelta:
        if (
            type(snapshot) is not EvaluationMemorySnapshot
            or type(packet) is not MemoryPacket
            or type(stage) is not StagePlan
        ):
            _fail("closure adapter received a noncanonical runner value")
        if (
            stage.mechanism_id != self.mechanism_id
            or stage.delta_kind != self.delta_kind
            or stage.budget.token_cap != ADDITION_TOKEN_CAP
            or stage.budget.provider_prompt_cap != 0
        ):
            _fail("closure adapter stage contract changed")
        if (
            snapshot.population_identity_sha256
            != self.generation.population_identity_sha256
        ):
            _fail("closure adapter snapshot belongs to another population")
        retrieval_refs = tuple(
            row
            for row in snapshot.source_artifacts
            if row.role == "sealed_retrieval"
        )
        if (
            len(retrieval_refs) != 1
            or retrieval_refs[0].sha256 != self.generation.retrieval_sha256
        ):
            _fail("closure adapter snapshot changed its sealed retrieval binding")
        refs = snapshot.source_artifacts + snapshot.overlay_revisions
        if not any(
            row.role == CLOSURE_GENERATION_ARTIFACT_ROLE
            and row.sha256
            == self.generation.source_retrieval_generation_sha256
            for row in refs
        ):
            _fail("closure adapter snapshot does not bind the generation seal")
        try:
            question = self.generation.question(packet.question_id)
        except KeyError as exc:
            raise IndependentClosureError(
                "closure adapter question is outside the sealed population"
            ) from exc
        if (
            packet.question_sha256 != question.question_sha256
            or packet.dated_question != question.dated_question
            or packet.dated_question_sha256 != question.dated_question_sha256
            or packet.packet_id != question.root_packet_id
            or packet.protected_evidence != question.root_protected_evidence
            or packet.applied_stage_ids != question.root_applied_stage_ids
            or packet.stage_id != SOURCE_STAGE_ID
            or stage.parent_stage_id != SOURCE_STAGE_ID
            or packet.admitted_evidence
            or packet.facts
            or packet.links
            or packet.answer_operators
        ):
            _fail("closure adapter requires the exact isolated S0 packet")
        arm = question.arm(self.arm_label)
        if arm is None:
            trace = StageTrace(
                token_cap=stage.budget.token_cap,
                disposition=StageDisposition.NO_OP,
                reason="question_not_eligible_for_independent_closure",
            )
            return MembershipDelta(
                stage_id=stage.stage_id,
                parent_stage_id=stage.parent_stage_id,
                trace=trace,
            )

        protected_ids: set[str] = set()
        for row in packet.protected_evidence:
            if row.evidence_id in protected_ids:
                _fail("closure S0 packet repeats a protected evidence ID")
            protected_ids.add(row.evidence_id)
        excluded_atoms = set(arm.dedup_excluded_atom_ids)
        aliases: list[tuple[str, str]] = []
        for atom in arm.selected_atoms:
            matches = tuple(
                row
                for row in packet.protected_evidence
                if row.source_id == atom.source_id
                and atom.text in row.text
                and row.evidence_id
                == identity_sha256(
                    {
                        "kind": "protected_excerpt",
                        "chunk_id": atom.chunk_id,
                        "source_id": row.source_id,
                        "text_sha256": quote_sha256(row.text),
                    }
                )
            )
            if atom.atom_id in excluded_atoms:
                if len(matches) != 1:
                    _fail(
                        "closure exact-S0 exclusion lacks one unique enclosing "
                        "protected coordinate"
                    )
                aliases.append((atom.evidence_id, matches[0].evidence_id))
            elif matches:
                _fail("closure retained a protected S0-covered atom")

        admitted = tuple(row.evidence_item() for row in arm.admitted_atoms)
        parent_ids = {row.evidence_id for row in packet.protected_evidence}
        if any(row.evidence_id in parent_ids for row in admitted):
            _fail("closure wrapper evidence ID collides with protected S0")
        excluded_evidence = set(arm.dedup_excluded_evidence_ids)
        admitted_evidence = set(arm.admitted_evidence_ids)
        not_admitted = tuple(
            evidence_id
            for evidence_id in arm.selected_evidence_ids
            if evidence_id not in excluded_evidence
            and evidence_id not in admitted_evidence
        )
        disposition = (
            StageDisposition.ADDED
            if admitted
            else (
                StageDisposition.OVERFLOW
                if arm.admission_status == "overflow_noop"
                else StageDisposition.NO_OP
            )
        )
        reason = None if disposition is StageDisposition.ADDED else (
            arm.overflow_reason or arm.admission_status
        )
        trace = StageTrace(
            candidate_ids=arm.candidate_evidence_ids,
            selected_before_dedup_ids=arm.selected_evidence_ids,
            dedup_excluded_ids=arm.dedup_excluded_evidence_ids,
            not_admitted_ids=not_admitted,
            admitted_ids=arm.admitted_evidence_ids,
            token_cap=stage.budget.token_cap,
            tokens_used=sum(row.token_count for row in admitted),
            provider_prompt_count=0,
            disposition=disposition,
            reason=reason,
        )
        return MembershipDelta(
            stage_id=stage.stage_id,
            parent_stage_id=stage.parent_stage_id,
            trace=trace,
            dedup_alias_bindings=tuple(aliases),
            additions=admitted,
        )


def independent_closure_arm_plan(
    arm_label: str,
    *,
    root_stage_id: str = SOURCE_STAGE_ID,
) -> ArmPlan:
    """Return the fixed isolated, non-borrowing 2,048-token arm plan."""

    if arm_label not in ARM_LABELS:
        _fail("unknown independent closure arm")
    require_text(root_stage_id, "independent closure root stage ID")
    stage_id = {
        REPRESENTATIVE_ARM: "representative_bridge_closure",
        GLOBAL_ARM: "artifact_global_closure",
    }[arm_label]
    return ArmPlan(
        plan_id=f"matched_{stage_id}_v1",
        mode=PlanMode.ISOLATED,
        root_stage_id=root_stage_id,
        stages=(
            StagePlan(
                stage_id=stage_id,
                parent_stage_id=root_stage_id,
                mechanism_id=arm_label,
                delta_kind="membership",
                budget=StageBudget(
                    token_cap=ADDITION_TOKEN_CAP,
                    provider_prompt_cap=0,
                ),
            ),
        ),
        global_provider_prompt_cap=0,
        max_final_prompt_tokens=MAX_FINAL_PROMPT_TOKENS,
    )


@dataclass(frozen=True, slots=True)
class ClosureTargetEvent:
    target_id: str
    target_kind: Literal["evidence_atom"]
    discovering_method: str
    disposition: str
    route_local_receipt_sha256: str
    source_target_ids: tuple[str, ...]
    atom_identity_sha256: str

    def __post_init__(self) -> None:
        require_text(self.target_id, "closure target ID")
        if self.target_kind != "evidence_atom":
            raise IndependentClosureError("closure target kind changed")
        if self.discovering_method not in ARM_LABELS:
            raise IndependentClosureError("closure target route changed")
        require_text(self.disposition, "closure target disposition")
        require_sha256(
            self.route_local_receipt_sha256, "closure route-local receipt"
        )
        if type(self.source_target_ids) is not tuple:
            raise IndependentClosureError("closure source aliases must be immutable")
        _ordered_unique(self.source_target_ids, "closure source aliases")
        if len(self.source_target_ids) != 1:
            raise IndependentClosureError(
                "closure ledger persists only the exact span source alias"
            )
        require_sha256(self.atom_identity_sha256, "closure target atom identity")

    def projection(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_target_ids"] = list(self.source_target_ids)
        assert_gold_blind(result, path="closure_target_event")
        return result


@dataclass(frozen=True, slots=True)
class ClosureStructuralQuestionProjection:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    eligibility_row_identity_sha256: str
    source_root_packet_id: str
    eligible: bool
    source_question_artifact_sha256: str | None
    source_s0_stage_receipt_sha256: str | None
    source_s0_fresh_validation_sha256: str | None
    selected_targets_before_dedup: tuple[ClosureTargetEvent, ...]
    admitted_targets_after_dedup: tuple[ClosureTargetEvent, ...]

    def __post_init__(self) -> None:
        _integer(self.ordinal, "structural question ordinal")
        require_text(self.question_id, "structural question ID")
        for value, label in (
            (self.question_sha256, "structural question SHA-256"),
            (self.dated_question_sha256, "structural dated-question SHA-256"),
            (self.eligibility_row_identity_sha256, "structural eligibility row"),
            (self.source_root_packet_id, "structural source root packet"),
        ):
            require_sha256(value, label)
        if type(self.eligible) is not bool:
            raise IndependentClosureError("structural eligibility must be exact")
        if self.source_question_artifact_sha256 is not None:
            require_sha256(
                self.source_question_artifact_sha256,
                "structural source question artifact",
            )
        for value, label in (
            (self.source_s0_stage_receipt_sha256, "structural source S0 stage"),
            (
                self.source_s0_fresh_validation_sha256,
                "structural source S0 fresh validation",
            ),
        ):
            if value is not None:
                require_sha256(value, label)
        for values, label in (
            (self.selected_targets_before_dedup, "structural discovery events"),
            (self.admitted_targets_after_dedup, "structural admission events"),
        ):
            if type(values) is not tuple or any(
                type(row) is not ClosureTargetEvent for row in values
            ):
                raise IndependentClosureError(f"{label} must be immutable typed values")
        before = {
            (row.target_kind, row.target_id, row.discovering_method): row
            for row in self.selected_targets_before_dedup
        }
        if len(before) != len(self.selected_targets_before_dedup):
            raise IndependentClosureError("structural discovery events must be unique")
        for row in self.admitted_targets_after_dedup:
            key = (row.target_kind, row.target_id, row.discovering_method)
            source = before.get(key)
            if source is None:
                raise IndependentClosureError("structural admission was not discovered")
            if (
                source.source_target_ids != row.source_target_ids
                or source.atom_identity_sha256 != row.atom_identity_sha256
            ):
                raise IndependentClosureError("structural admission changed target identity")
        before_keys = tuple(before)
        after_keys = tuple(
            (row.target_kind, row.target_id, row.discovering_method)
            for row in self.admitted_targets_after_dedup
        )
        if not _ordered_subsequence(after_keys, before_keys):
            raise IndependentClosureError("structural admission changed discovery order")
        source_bound = all(
            value is not None
            for value in (
                self.source_question_artifact_sha256,
                self.source_s0_stage_receipt_sha256,
                self.source_s0_fresh_validation_sha256,
            )
        )
        if self.eligible != source_bound:
            raise IndependentClosureError("structural eligibility/source binding changed")
        if not self.eligible and any(
            value is not None
            for value in (
                self.source_question_artifact_sha256,
                self.source_s0_stage_receipt_sha256,
                self.source_s0_fresh_validation_sha256,
            )
        ):
            raise IndependentClosureError("ineligible question acquired source seals")
        if not self.eligible and (
            self.selected_targets_before_dedup
            or self.admitted_targets_after_dedup
        ):
            raise IndependentClosureError("ineligible question acquired target events")

    def projection(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "admitted_targets_after_dedup": [
                row.projection() for row in self.admitted_targets_after_dedup
            ],
            "dated_question_sha256": self.dated_question_sha256,
            "eligibility_row_identity_sha256": self.eligibility_row_identity_sha256,
            "eligible": self.eligible,
            "ordinal": self.ordinal,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "source_root_packet_id": self.source_root_packet_id,
            "selected_targets_before_dedup": [
                row.projection() for row in self.selected_targets_before_dedup
            ],
            "source_question_artifact_sha256": (
                self.source_question_artifact_sha256
            ),
            "source_s0_fresh_validation_sha256": (
                self.source_s0_fresh_validation_sha256
            ),
            "source_s0_stage_receipt_sha256": self.source_s0_stage_receipt_sha256,
        }
        body["ledger_row_sha256"] = identity_sha256(body)
        assert_gold_blind(body, path="closure_structural_question")
        return body


@dataclass(frozen=True, slots=True)
class IndependentClosureStructuralProjection:
    arm_label: str
    source_retrieval_generation_sha256: str
    source_eligibility_manifest_sha256: str
    source_preflight_sha256: str
    population_identity_sha256: str
    questions: tuple[ClosureStructuralQuestionProjection, ...]

    def __post_init__(self) -> None:
        if self.arm_label not in ARM_LABELS:
            raise IndependentClosureError("unknown structural closure arm")
        for value, label in (
            (self.source_retrieval_generation_sha256, "structural generation"),
            (self.source_eligibility_manifest_sha256, "structural eligibility"),
            (self.source_preflight_sha256, "structural preflight"),
            (self.population_identity_sha256, "structural population"),
        ):
            require_sha256(value, label)
        if type(self.questions) is not tuple or any(
            type(row) is not ClosureStructuralQuestionProjection
            for row in self.questions
        ):
            raise IndependentClosureError("structural questions must be immutable")
        if (
            len(self.questions) != EXPECTED_QUESTION_COUNT
            or tuple(row.ordinal for row in self.questions)
            != tuple(range(EXPECTED_QUESTION_COUNT))
        ):
            raise IndependentClosureError("structural projection is not the locked 100")

    def projection(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "arm_label": self.arm_label,
            "format": STRUCTURAL_PROJECTION_FORMAT,
            "gold_loaded": False,
            "population_identity_sha256": self.population_identity_sha256,
            "provider_calls": 0,
            "question_count": len(self.questions),
            "questions": [row.projection() for row in self.questions],
            "source_eligibility_manifest_sha256": (
                self.source_eligibility_manifest_sha256
            ),
            "source_preflight_sha256": self.source_preflight_sha256,
            "source_retrieval_generation_sha256": (
                self.source_retrieval_generation_sha256
            ),
        }
        result["projection_sha256"] = identity_sha256(result)
        assert_gold_blind(result, path="closure_structural_projection")
        return result

    @property
    def projection_sha256(self) -> str:
        return self.projection()["projection_sha256"]


def build_structural_target_projection(
    generation: IndependentClosureGeneration,
    arm_label: str,
) -> IndependentClosureStructuralProjection:
    if type(generation) is not IndependentClosureGeneration:
        _fail("structural projection requires an exact closure generation")
    if arm_label not in ARM_LABELS:
        _fail("unknown structural closure arm")
    questions: list[ClosureStructuralQuestionProjection] = []
    for question in generation.questions:
        arm = question.arm(arm_label)
        selected_events: tuple[ClosureTargetEvent, ...] = ()
        admitted_events: tuple[ClosureTargetEvent, ...] = ()
        if arm is not None:
            selected_events = tuple(
                ClosureTargetEvent(
                    target_id=row.atom_id,
                    target_kind="evidence_atom",
                    discovering_method=arm_label,
                    disposition=row.terminal_disposition,
                    route_local_receipt_sha256=(
                        row.selection_packet_receipt_sha256 or ""
                    ),
                    source_target_ids=(row.source_id,),
                    atom_identity_sha256=row.atom_identity_sha256,
                )
                for row in arm.targets
                if row.selected
            )
            admitted_events = tuple(
                ClosureTargetEvent(
                    target_id=row.atom_id,
                    target_kind="evidence_atom",
                    discovering_method=arm_label,
                    disposition="admitted_after_dedup",
                    route_local_receipt_sha256=(
                        row.admission_packet_receipt_sha256 or ""
                    ),
                    source_target_ids=(row.source_id,),
                    atom_identity_sha256=row.atom_identity_sha256,
                )
                for row in arm.targets
                if row.admitted
            )
        questions.append(
            ClosureStructuralQuestionProjection(
                ordinal=question.ordinal,
                question_id=question.question_id,
                question_sha256=question.question_sha256,
                dated_question_sha256=question.dated_question_sha256,
                eligibility_row_identity_sha256=(
                    question.eligibility_row_identity_sha256
                ),
                source_root_packet_id=question.root_packet_id,
                eligible=question.eligible,
                source_question_artifact_sha256=(
                    question.source_question_artifact_sha256
                ),
                source_s0_stage_receipt_sha256=(
                    question.source_s0_stage_receipt_sha256
                ),
                source_s0_fresh_validation_sha256=(
                    question.source_s0_fresh_validation_sha256
                ),
                selected_targets_before_dedup=selected_events,
                admitted_targets_after_dedup=admitted_events,
            )
        )
    return IndependentClosureStructuralProjection(
        arm_label=arm_label,
        source_retrieval_generation_sha256=(
            generation.source_retrieval_generation_sha256
        ),
        source_eligibility_manifest_sha256=(
            generation.source_eligibility_manifest_sha256
        ),
        source_preflight_sha256=generation.preflight_sha256,
        population_identity_sha256=generation.population_identity_sha256,
        questions=tuple(questions),
    )


@dataclass(frozen=True, slots=True)
class FinalizedClosureStructuralTargetLedger:
    source: IndependentClosureStructuralProjection
    source_run_sha256: str

    def __post_init__(self) -> None:
        if type(self.source) is not IndependentClosureStructuralProjection:
            raise IndependentClosureError("final ledger requires an exact projection")
        require_sha256(self.source_run_sha256, "structural source run")

    def projection(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "admission_projection": "admitted_targets_after_dedup",
            "arm_label": self.source.arm_label,
            "discovery_projection": "selected_targets_before_dedup",
            "format": STRUCTURAL_LEDGER_FORMAT,
            "gold_loaded": False,
            "ownership_policy": (
                "join-primary-owner-from-posthoc-desired-target-registry"
            ),
            "population_identity_sha256": self.source.population_identity_sha256,
            "provider_calls": 0,
            "question_count": len(self.source.questions),
            "questions": [row.projection() for row in self.source.questions],
            "source_eligibility_manifest_sha256": (
                self.source.source_eligibility_manifest_sha256
            ),
            "source_preflight_sha256": self.source.source_preflight_sha256,
            "source_projection_sha256": self.source.projection_sha256,
            "source_retrieval_generation_sha256": (
                self.source.source_retrieval_generation_sha256
            ),
            "source_run_sha256": self.source_run_sha256,
            "target_id_policy": {
                "source_target_ids": "exact_atom_span_source_id",
                "targets": "sealed_closure_atom_id",
            },
        }
        result["ledger_sha256"] = identity_sha256(result)
        assert_gold_blind(result, path="closure_structural_target_ledger")
        return result


def _verify_answer_run(
    artifact: SealedArtifact,
    *,
    source: IndependentClosureStructuralProjection,
) -> None:
    run = artifact.payload
    assert_gold_blind(run, path="closure_source_answer_run")
    rows = _rows(run.get("questions"), "closure source answer questions")
    if (
        run.get("arm_label") != source.arm_label
        or run.get("population_identity_sha256")
        != source.population_identity_sha256
        or run.get("question_count") != len(source.questions)
        or len(rows) != len(source.questions)
        or run.get("gold_loaded") is not False
    ):
        _fail("closure structural source run binding changed")
    for expected, raw in zip(source.questions, rows, strict=True):
        if (
            raw.get("ordinal") != expected.ordinal
            or raw.get("question_id") != expected.question_id
            or raw.get("question_sha256") != expected.question_sha256
            or raw.get("dated_question_sha256")
            != expected.dated_question_sha256
        ):
            _fail(f"closure source run order changed at {expected.ordinal}")


def finalize_structural_target_ledger(
    source: IndependentClosureStructuralProjection,
    *,
    source_run_path: str | Path,
    source_run_replay_path: str | Path,
    expected_source_run_sha256: str,
) -> FinalizedClosureStructuralTargetLedger:
    """Bind retrieval-only projection only after byte-identical run/replay seals."""

    if type(source) is not IndependentClosureStructuralProjection:
        _fail("finalization requires an exact structural projection")
    expected = _sha(expected_source_run_sha256, "expected closure source run")
    run = read_sealed_json(source_run_path)
    replay = read_sealed_json(source_run_replay_path)
    if (
        run.sha256 != expected
        or replay.sha256 != expected
        or canonical_json_bytes(run.payload) != canonical_json_bytes(replay.payload)
    ):
        _fail("closure answer run/replay seals differ")
    _verify_answer_run(run, source=source)
    _verify_answer_run(replay, source=source)
    return FinalizedClosureStructuralTargetLedger(
        source=source,
        source_run_sha256=expected,
    )


__all__ = [
    "ADDITION_TOKEN_CAP",
    "ARM_LABELS",
    "CLOSURE_GENERATION_ARTIFACT_ROLE",
    "ClosureAtom",
    "ClosureStructuralQuestionProjection",
    "ClosureTargetEvent",
    "ClosureTargetLifecycle",
    "FinalizedClosureStructuralTargetLedger",
    "GLOBAL_ARM",
    "IndependentClosureArmProjection",
    "IndependentClosureError",
    "IndependentClosureGeneration",
    "IndependentClosureMembershipAdapter",
    "IndependentClosureQuestion",
    "IndependentClosureStructuralProjection",
    "MAX_FINAL_PROMPT_TOKENS",
    "REPRESENTATIVE_ARM",
    "build_structural_target_projection",
    "finalize_structural_target_ledger",
    "independent_closure_arm_plan",
    "load_independent_closure_generation",
    "project_independent_closure_generation",
    "project_independent_closure_question",
]
