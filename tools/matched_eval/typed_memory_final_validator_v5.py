"""Provider-free asymmetric replacement policy for typed-final answers.

Validator v4 remains the authority for typed operator execution, aggregate
scope, and question-derived temporal targets.  This successor adds the
missing asymmetric decision rule: a supported candidate is not automatically
allowed to overwrite a useful parent.

There are only three replacement authorities:

* an executable deterministic/scalar result already authenticated by v4;
* a direct, fully supported fill for a canonical parent abstention; or
* a non-abstaining correction carrying a sealed parent-defect certificate.

All semantic replacements additionally close any touched conflict
neighbourhood, support every material output term, and use explicit
user-preference evidence for recommendation questions.  Every failed check is
fail-closed: callers retain the byte-exact parent prediction.

The module does not change the v3/v4 validators, call a provider, or read
benchmark gold.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_action_semantics import canonical_action_concepts
from .typed_memory_final_arm import (
    ParsedTypedFinalDecision,
    _validated_semantic_rows,
)
from .typed_memory_final_validator_v4 import (
    VALIDATION_CONTRACT_FORMAT as V4_VALIDATION_CONTRACT_FORMAT,
    dated_question_from_plan_row,
    parse_typed_final_completion_v4,
    upgrade_completion_validation_contract_v4,
)
from .typed_operator_spec import TemporalMode, normalized_terms


FORMAT = "memory-condense-typed-memory-final-validator-v5"
VALIDATION_CONTRACT_FORMAT = (
    "memory-condense-typed-memory-final-arm-v1-"
    "completion-validation-contract-v5"
)
VALIDATOR_POLICY_FORMAT = (
    "memory-condense-typed-memory-final-arm-v1-validator-policy-v5"
)
DECISION_FORMAT = "memory-condense-typed-memory-final-arm-v1-decision-v3"
PARENT_DEFECT_CERTIFICATE_FORMAT = f"{FORMAT}-parent-defect-certificate-v1"

_ABSTENTION_RE = re.compile(
    r"^(?:"
    r"i\s+(?:do\s+not|don['’]t)\s+know|"
    r"i\s+(?:can(?:not|['’]t)|could\s+not)\s+(?:determine|tell)|"
    r"unknown|unclear|"
    r"(?:there\s+is\s+)?(?:not\s+enough|insufficient)\s+(?:information|evidence)|"
    r"no\s+(?:relevant\s+)?(?:information|evidence)"
    r")$",
    re.IGNORECASE,
)
_RECOMMENDATION_RE = re.compile(
    r"\b(?:recommend|suggest|what\s+should|which\s+should)\b",
    re.IGNORECASE,
)
_PREFERENCE_PREFIXES = (
    "avoid",
    "dislik",
    "favorit",
    "hate",
    "hope",
    "intend",
    "interest",
    "lik",
    "lov",
    "need",
    "plan",
    "prefer",
    "want",
)
_CLAIM_GLUE_TERMS = frozenset(
    {
        "also",
        "another",
        "answer",
        "based",
        "because",
        "consider",
        "instead",
        "recommend",
        "recommendation",
        "since",
        "suggest",
        "suggestion",
        "tonight",
        "try",
    }
)
_ACTION_SURFACE_TERMS = frozenset(
    {
        "acquire",
        "acquir",
        "bought",
        "buy",
        "get",
        "got",
        "obtain",
        "purchase",
        "purchas",
    }
)
_CONFLICT_RELATIONS = frozenset({"contradicts", "revises"})
_CORRECTION_RELATIONS = frozenset({"resolves", "revises"})
_DETERMINISTIC_BASES = frozenset(
    {
        "bounded_positive_scalar_agreement",
        "deterministic_execution_agreement",
    }
)


class TypedMemoryFinalValidatorV5Error(MatchedEvalContractError):
    """A v4 contract, v5 derivation, or policy input changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedMemoryFinalValidatorV5Error(message)


def parent_is_abstention(value: str) -> bool:
    """Return whether ``value`` is a deliberately narrow canonical abstention."""

    require_text(value, "v5 parent prediction")
    normalized = " ".join(value.strip().rstrip(".!?").split())
    return _ABSTENTION_RE.fullmatch(normalized) is not None


def _ordered_terms(values: Sequence[str], label: str) -> tuple[str, ...]:
    _require(
        all(type(value) is str and bool(value) for value in values),
        f"{label} changed type",
    )
    normalized = tuple(dict.fromkeys(normalized_terms(" ".join(values))))
    _require(bool(normalized), f"{label} is empty")
    return normalized


def _parent_material_terms(parent: str, dated_question: str) -> tuple[str, ...]:
    question = set(normalized_terms(dated_question))
    return tuple(
        term
        for term in normalized_terms(parent)
        if term not in question
        and term not in _CLAIM_GLUE_TERMS
        and term not in _ACTION_SURFACE_TERMS
    )


def build_parent_defect_certificate(
    *,
    parent_prediction: str,
    dated_question: str,
    challenged_parent_terms: Sequence[str],
    supporting_link_ids: Sequence[str],
    used_handle_ids: Sequence[str],
) -> dict[str, Any]:
    """Build the deterministic certificate required to alter a useful parent.

    The link IDs must later resolve to complete ``revises`` or ``resolves``
    edges in the authenticated story-coherence plane.  All unchallenged
    material parent terms become mandatory preservation terms.
    """

    require_text(parent_prediction, "v5 certificate parent")
    require_text(dated_question, "v5 certificate question")
    challenged = _ordered_terms(challenged_parent_terms, "challenged parent terms")
    parent_terms = _parent_material_terms(parent_prediction, dated_question)
    _require(
        set(challenged) <= set(parent_terms),
        "challenged terms are not material parent terms",
    )
    _require(
        all(type(value) is str and bool(value) for value in supporting_link_ids)
        and len(set(supporting_link_ids)) == len(supporting_link_ids)
        and bool(supporting_link_ids),
        "certificate supporting links changed",
    )
    _require(
        all(type(value) is str and bool(value) for value in used_handle_ids)
        and len(set(used_handle_ids)) == len(used_handle_ids)
        and bool(used_handle_ids),
        "certificate used handles changed",
    )
    body = {
        "challenged_parent_terms": list(challenged),
        "format": PARENT_DEFECT_CERTIFICATE_FORMAT,
        "parent_prediction_sha256": quote_sha256(parent_prediction),
        "required_preserved_parent_terms": [
            term for term in parent_terms if term not in set(challenged)
        ],
        "supporting_link_ids": list(supporting_link_ids),
        "used_handle_ids": list(used_handle_ids),
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def upgrade_completion_validation_contract_v5(
    legacy_contract: Mapping[str, Any],
    *,
    dated_question: str,
    parent_prediction: str,
    parent_defect_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive one v5 contract without changing its authenticated v3 source."""

    require_text(dated_question, "v5 dated question")
    require_text(parent_prediction, "v5 parent prediction")
    v4 = upgrade_completion_validation_contract_v4(
        legacy_contract,
        dated_question=dated_question,
    )
    certificate = (
        None
        if parent_defect_certificate is None
        else dict(parent_defect_certificate)
    )
    body = {
        **v4,
        "format": VALIDATION_CONTRACT_FORMAT,
        "parent_defect_certificate": certificate,
        "parent_prediction_sha256": quote_sha256(parent_prediction),
        "parent_replacement_mode": (
            "abstention_fill"
            if parent_is_abstention(parent_prediction)
            else "certified_correction_only"
        ),
        "v4_validation_contract_sha256": identity_sha256(v4),
    }
    assert_gold_blind(body, path="typed_final_validator_v5_contract")
    return body


def _v4_contract_from_v5(
    contract: Mapping[str, Any],
    *,
    parent_prediction: str,
) -> dict[str, Any]:
    _require(type(contract) is dict, "v5 validation contract changed type")
    value = dict(contract)
    _require(
        value.get("format") == VALIDATION_CONTRACT_FORMAT
        and value.get("parent_prediction_sha256")
        == quote_sha256(parent_prediction)
        and value.get("parent_replacement_mode")
        == (
            "abstention_fill"
            if parent_is_abstention(parent_prediction)
            else "certified_correction_only"
        )
        and (
            value.get("parent_defect_certificate") is None
            or type(value.get("parent_defect_certificate")) is dict
        ),
        "v5 parent policy binding changed",
    )
    expected_v4_sha = require_sha256(
        value.pop("v4_validation_contract_sha256", None),
        "v5 inherited v4 contract",
    )
    for key in (
        "parent_defect_certificate",
        "parent_prediction_sha256",
        "parent_replacement_mode",
    ):
        value.pop(key)
    value["format"] = V4_VALIDATION_CONTRACT_FORMAT
    _require(
        identity_sha256(value) == expected_v4_sha,
        "v5 inherited v4 contract changed",
    )
    return value


def _v5_decision(
    parsed: ParsedTypedFinalDecision,
    *,
    contract: Mapping[str, Any],
    error_code: str | None = None,
    validation_basis: str | None = None,
) -> ParsedTypedFinalDecision:
    error = parsed.error_code if error_code is None else error_code
    valid = parsed.valid and error == "none"
    decision = parsed.decision if valid else "invalid"
    prediction = parsed.prediction if valid else ""
    used = parsed.used_handle_ids if valid else ()
    basis = (validation_basis or parsed.validation_basis) if valid else "invalid"
    receipt = identity_sha256(
        {
            "decision": decision,
            "error_code": error,
            "format": DECISION_FORMAT,
            "prediction_sha256": quote_sha256(prediction),
            "used_handle_ids": list(used),
            "validation_basis": basis,
            "validation_contract_sha256": identity_sha256(contract),
            "validator_policy_format": VALIDATOR_POLICY_FORMAT,
        }
    )
    return ParsedTypedFinalDecision(
        valid,
        decision,
        prediction,
        used,
        basis,
        error,
        receipt,
    )


def _story_links(story: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = story.get("typed_links", [])
    _require(type(raw) is list, "v5 typed story links changed type")
    links: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in raw:
        _require(type(source) is dict, "v5 typed story link changed type")
        row = dict(source)
        link_id = row.get("link_id")
        relation = row.get("relation")
        members = row.get("members")
        _require(
            type(link_id) is str
            and bool(link_id)
            and link_id not in seen
            and type(relation) is str
            and bool(relation)
            and type(members) is list
            and len(members) >= 2
            and all(
                type(member) is dict
                and type(member.get("handle_id")) is str
                and bool(member["handle_id"])
                for member in members
            ),
            "v5 typed story link changed schema",
        )
        seen.add(link_id)
        links.append(row)
    return tuple(links)


def _conflict_neighbourhood_error(
    story: Mapping[str, Any],
    used_handle_ids: Sequence[str],
    handle_group_by_id: Mapping[str, str],
) -> str | None:
    used = set(used_handle_ids)
    for link in _story_links(story):
        if link["relation"] not in _CONFLICT_RELATIONS:
            continue
        members = {member["handle_id"] for member in link["members"]}
        if used & members and not members <= used:
            return "conflict_neighborhood_incomplete"
        if link["relation"] == "contradicts" and members <= used:
            return "conflict_neighborhood_unresolved"

    raw_pairs = story.get("incompatible_group_pairs", [])
    _require(type(raw_pairs) is list, "v5 incompatible groups changed type")
    used_groups = {handle_group_by_id[handle] for handle in used}
    for row in raw_pairs:
        _require(
            type(row) is dict
            and type(row.get("left_group")) is str
            and type(row.get("right_group")) is str,
            "v5 incompatible group changed schema",
        )
        pair = {row["left_group"], row["right_group"]}
        if used_groups & pair and not pair <= used_groups:
            return "conflict_neighborhood_incomplete"
    return None


def _row_terms(row: Mapping[str, Any]) -> frozenset[str]:
    values: list[str] = []
    for key in ("summary_terms", "entity_terms", "group_terms"):
        terms = row.get(key)
        _require(
            type(terms) is list
            and all(type(term) is str and bool(term) for term in terms),
            "v5 semantic claim terms changed",
        )
        values.extend(terms)
    return frozenset(values)


def _unsupported_material_terms(
    prediction: str,
    dated_question: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    question_terms = set(normalized_terms(dated_question))
    material = {
        term
        for term in normalized_terms(prediction)
        if term not in question_terms and term not in _CLAIM_GLUE_TERMS
    }
    evidence_terms = set().union(*(_row_terms(row) for row in rows))
    candidate_actions = set(canonical_action_concepts(prediction))
    evidence_actions = {
        action for row in rows for action in row.get("action_concepts", ())
    }
    if candidate_actions and candidate_actions <= evidence_actions:
        material.difference_update(_ACTION_SURFACE_TERMS)
    return tuple(sorted(material - evidence_terms))


def _row_is_user_grounded(row: Mapping[str, Any]) -> bool:
    relation = row.get("relation_terms")
    _require(
        type(relation) is list
        and all(type(term) is str and bool(term) for term in relation),
        "v5 semantic relation terms changed",
    )
    return "user" in relation


def _has_explicit_user_preference(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if not _row_is_user_grounded(row):
            continue
        if any(
            term.startswith(prefix)
            for term in _row_terms(row)
            for prefix in _PREFERENCE_PREFIXES
        ):
            return True
    return False


def _parent_defect_error(
    certificate: object,
    *,
    parent_prediction: str,
    dated_question: str,
    candidate_prediction: str,
    used_handle_ids: Sequence[str],
    story: Mapping[str, Any],
) -> str | None:
    if certificate is None:
        return "parent_defect_certificate_missing"
    _require(type(certificate) is dict, "v5 parent defect certificate changed type")
    raw = dict(certificate)
    body = {key: value for key, value in raw.items() if key != "receipt_sha256"}
    challenged = body.get("challenged_parent_terms")
    links = body.get("supporting_link_ids")
    used = body.get("used_handle_ids")
    _require(
        set(raw)
        == {
            "challenged_parent_terms",
            "format",
            "parent_prediction_sha256",
            "receipt_sha256",
            "required_preserved_parent_terms",
            "supporting_link_ids",
            "used_handle_ids",
        }
        and body.get("format") == PARENT_DEFECT_CERTIFICATE_FORMAT
        and raw.get("receipt_sha256") == identity_sha256(body)
        and body.get("parent_prediction_sha256")
        == quote_sha256(parent_prediction)
        and type(challenged) is list
        and type(links) is list
        and type(used) is list,
        "v5 parent defect certificate changed schema",
    )
    expected = build_parent_defect_certificate(
        parent_prediction=parent_prediction,
        dated_question=dated_question,
        challenged_parent_terms=challenged,
        supporting_link_ids=links,
        used_handle_ids=used,
    )
    _require(raw == expected, "v5 parent defect certificate derivation changed")
    if tuple(used) != tuple(used_handle_ids):
        return "parent_defect_handle_disagreement"
    candidate_terms = set(normalized_terms(candidate_prediction))
    if not set(body["required_preserved_parent_terms"]) <= candidate_terms:
        return "parent_uncontested_claim_loss"
    if set(challenged) <= candidate_terms:
        return "parent_challenge_not_applied"

    by_id = {row["link_id"]: row for row in _story_links(story)}
    selected = [by_id.get(link_id) for link_id in links]
    if any(row is None for row in selected):
        return "parent_defect_link_missing"
    for row in selected:
        assert row is not None
        members = {member["handle_id"] for member in row["members"]}
        if row["relation"] not in _CORRECTION_RELATIONS:
            return "parent_defect_link_not_corrective"
        if not members <= set(used_handle_ids):
            return "parent_defect_link_incomplete"
    return None


def parse_typed_final_completion_v5(
    completion: str,
    *,
    dated_question: str,
    parent_prediction: str,
    allowed_handle_ids: Sequence[str],
    handle_group_by_id: Mapping[str, str],
    story_coherence: Mapping[str, Any],
    preservation_requirements: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
) -> ParsedTypedFinalDecision:
    """Revalidate one completion with asymmetric overwrite authority."""

    require_text(dated_question, "v5 completion dated question")
    require_text(parent_prediction, "v5 completion parent")
    contract = dict(validation_contract)
    v4_contract = _v4_contract_from_v5(
        contract,
        parent_prediction=parent_prediction,
    )
    parsed = parse_typed_final_completion_v4(
        completion,
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        allowed_handle_ids=allowed_handle_ids,
        handle_group_by_id=handle_group_by_id,
        story_coherence=story_coherence,
        preservation_requirements=preservation_requirements,
        validation_contract=v4_contract,
    )
    if not parsed.valid or parsed.decision != "replace":
        return _v5_decision(parsed, contract=contract)

    by_handle = v4_contract.get("by_handle")
    _require(type(by_handle) is dict, "v5 by-handle contract changed")
    rows = _validated_semantic_rows(by_handle, parsed.used_handle_ids)

    conflict_error = _conflict_neighbourhood_error(
        story_coherence,
        parsed.used_handle_ids,
        handle_group_by_id,
    )
    if conflict_error is not None:
        return _v5_decision(
            parsed,
            contract=contract,
            error_code=conflict_error,
        )

    if parsed.validation_basis in _DETERMINISTIC_BASES:
        return _v5_decision(
            parsed,
            contract=contract,
            validation_basis=f"{parsed.validation_basis}_v5",
        )

    unsupported = _unsupported_material_terms(
        parsed.prediction,
        dated_question,
        rows,
    )
    if unsupported:
        return _v5_decision(
            parsed,
            contract=contract,
            error_code="unsupported_material_claim",
        )

    if (
        v4_contract.get("operation") == "preference_or_causal_synthesis"
        and _RECOMMENDATION_RE.search(dated_question) is not None
        and not _has_explicit_user_preference(rows)
    ):
        return _v5_decision(
            parsed,
            contract=contract,
            error_code="preference_evidence_missing",
        )

    if parent_is_abstention(parent_prediction):
        if (
            v4_contract.get("question_memory_role") == "user"
            and not all(_row_is_user_grounded(row) for row in rows)
        ):
            return _v5_decision(
                parsed,
                contract=contract,
                error_code="abstention_fill_user_scope_missing",
            )
        basis = "abstention_fill_direct_v5"
        if (
            v4_contract.get("temporal_mode") == TemporalMode.RELATIVE_SELECT.value
        ):
            basis = "abstention_fill_relative_exact_day_v5"
        return _v5_decision(
            parsed,
            contract=contract,
            validation_basis=basis,
        )

    defect_error = _parent_defect_error(
        contract.get("parent_defect_certificate"),
        parent_prediction=parent_prediction,
        dated_question=dated_question,
        candidate_prediction=parsed.prediction,
        used_handle_ids=parsed.used_handle_ids,
        story=story_coherence,
    )
    if defect_error is not None:
        return _v5_decision(
            parsed,
            contract=contract,
            error_code=defect_error,
        )
    return _v5_decision(
        parsed,
        contract=contract,
        validation_basis="certified_parent_correction_v5",
    )


def evaluate_typed_final_replacement_policy_v5(
    plan_row: Mapping[str, Any],
    completion: str,
    *,
    parent_defect_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a real sealed preflight row and return a gold-blind proof.

    ``plan_row`` is the physical prompt-row shape emitted by the full100
    preflight.  Its dated question is recovered from the authenticated user
    message rather than supplied separately.  The returned prediction is
    always byte-identical to either the protected parent or the candidate
    accepted by the strict completion parser.
    """

    _require(type(plan_row) is dict, "v5 plan row changed type")
    plan = dict(plan_row)
    prompt_receipt = require_sha256(
        plan.get("prompt_row_receipt_sha256"),
        "v5 prompt row",
    )
    unsigned = {
        key: value
        for key, value in plan.items()
        if key != "prompt_row_receipt_sha256"
    }
    _require(
        identity_sha256(unsigned) == prompt_receipt,
        "v5 prompt row receipt changed",
    )
    parent = require_text(plan.get("parent_prediction"), "v5 plan parent")
    dated_question = dated_question_from_plan_row(plan)
    legacy_contract = plan.get("validation_contract")
    _require(type(legacy_contract) is dict, "v5 legacy plan contract changed type")
    contract = upgrade_completion_validation_contract_v5(
        legacy_contract,
        dated_question=dated_question,
        parent_prediction=parent,
        parent_defect_certificate=parent_defect_certificate,
    )
    parsed = parse_typed_final_completion_v5(
        completion,
        dated_question=dated_question,
        parent_prediction=parent,
        allowed_handle_ids=tuple(plan.get("allowed_handle_ids", ())),
        handle_group_by_id=dict(plan.get("handle_group_by_id", {})),
        story_coherence=dict(plan.get("story_coherence", {})),
        preservation_requirements=dict(
            plan.get("preservation_requirements", {})
        ),
        validation_contract=contract,
    )
    accepted = parsed.valid and parsed.decision == "replace"
    prediction = parsed.prediction if accepted else parent
    body = {
        "accepted_replacement": accepted,
        "completion_sha256": quote_sha256(completion),
        "decision": "replace" if accepted else "keep_parent",
        "error_code": parsed.error_code,
        "final_prediction": prediction,
        "final_prediction_sha256": quote_sha256(prediction),
        "format": f"{FORMAT}-policy-proof-v1",
        "gold_loaded": False,
        "parent_prediction_sha256": quote_sha256(parent),
        "parent_replacement_mode": contract["parent_replacement_mode"],
        "physical_provider_calls": 0,
        "policy_contract_sha256": identity_sha256(contract),
        "prompt_row_receipt_sha256": prompt_receipt,
        "retained_transformer_token_state_bytes": 0,
        "used_handle_ids": list(parsed.used_handle_ids) if accepted else [],
        "validation_basis": parsed.validation_basis,
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
        "validator_receipt_sha256": parsed.receipt_sha256,
    }
    assert_gold_blind(body, path="typed_final_validator_v5_policy_proof")
    return {**body, "policy_proof_receipt_sha256": identity_sha256(body)}


__all__ = [
    "DECISION_FORMAT",
    "FORMAT",
    "PARENT_DEFECT_CERTIFICATE_FORMAT",
    "TypedMemoryFinalValidatorV5Error",
    "VALIDATION_CONTRACT_FORMAT",
    "VALIDATOR_POLICY_FORMAT",
    "build_parent_defect_certificate",
    "evaluate_typed_final_replacement_policy_v5",
    "parent_is_abstention",
    "parse_typed_final_completion_v5",
    "upgrade_completion_validation_contract_v5",
]
